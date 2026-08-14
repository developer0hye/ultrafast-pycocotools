//! Sparse (image, category) grouping of annotations.
//!
//! The obvious structure — a dense `images x categories` table of vectors — is
//! what pycocotools effectively builds, and it is fine for COCO (5k x 80). It
//! falls over on Objects365 val: 80k images x 365 categories is 29 million
//! buckets, and at 24 bytes per empty `Vec` header that is 700 MB of nothing
//! before a single annotation is stored.
//!
//! So we store the pairs that actually exist. Annotation indices are sorted
//! once into (group, image) runs, giving a CSR-style layout: O(annotations)
//! memory, O(1) lookup per run, and iteration in exactly the order
//! pycocotools produces.
//!
//! Ordering is not an implementation detail here. Within one (image,
//! category) the greedy matcher sees annotations in the order they appear in
//! the JSON, and ties in detection score are broken by that order, so a sort
//! that is not stable silently changes which ground truth a detection claims.
//! Every sort below is stable for that reason.

/// A contiguous run of annotations belonging to one image within one group.
#[derive(Clone, Copy, Debug)]
pub struct Run {
    pub img_slot: u32,
    /// Range into [`Grouping::order`].
    pub start: u32,
    pub len: u32,
}

#[derive(Debug, Default)]
pub struct Grouping {
    /// Annotation indices, sorted by (group slot, image slot, category slot).
    pub order: Vec<u32>,
    /// Runs, sorted the same way.
    runs: Vec<Run>,
    /// `[start, end)` into `runs` for each group slot.
    group_runs: Vec<(u32, u32)>,
}

impl Grouping {
    /// `img_slot` / `cat_slot` are per-annotation slot indices; `u32::MAX`
    /// means "not part of this evaluation" and is dropped.
    ///
    /// With `use_cats == false` every annotation lands in group 0, but the
    /// sort keeps the category as a secondary key so that reading a group
    /// yields `[c for c in catIds for ann in gts[img, c]]` — category-major
    /// within an image, which is the order pycocotools concatenates in.
    pub fn build(img_slot: &[u32], cat_slot: &[u32], n_groups: usize, use_cats: bool) -> Grouping {
        let n = img_slot.len();
        let mut order: Vec<u32> = Vec::with_capacity(n);
        for i in 0..n {
            if img_slot[i] != u32::MAX && cat_slot[i] != u32::MAX {
                order.push(i as u32);
            }
        }
        let group_of = |i: u32| -> u32 {
            if use_cats {
                cat_slot[i as usize]
            } else {
                0
            }
        };
        // Stable: equal keys keep annotation order, which is the JSON order.
        order.sort_by_key(|&i| {
            (
                group_of(i),
                img_slot[i as usize],
                if use_cats { 0 } else { cat_slot[i as usize] },
            )
        });

        // `order` is group-major, so runs come out grouped and each group's
        // slice is one contiguous range. Groups with no annotations keep
        // (0, 0) and read back as an empty slice.
        let mut runs: Vec<Run> = Vec::new();
        let mut group_runs = vec![(0u32, 0u32); n_groups];
        let mut i = 0usize;
        while i < order.len() {
            let g = group_of(order[i]) as usize;
            let g_start = runs.len() as u32;
            while i < order.len() && group_of(order[i]) as usize == g {
                let im = img_slot[order[i] as usize];
                let start = i;
                while i < order.len()
                    && group_of(order[i]) as usize == g
                    && img_slot[order[i] as usize] == im
                {
                    i += 1;
                }
                runs.push(Run {
                    img_slot: im,
                    start: start as u32,
                    len: (i - start) as u32,
                });
            }
            group_runs[g] = (g_start, runs.len() as u32);
        }

        Grouping {
            order,
            runs,
            group_runs,
        }
    }

    /// Runs of one group, ascending by image slot.
    #[inline]
    pub fn group(&self, group_slot: usize) -> &[Run] {
        let (a, b) = self.group_runs[group_slot];
        &self.runs[a as usize..b as usize]
    }

    #[inline]
    pub fn indices(&self, run: &Run) -> &[u32] {
        &self.order[run.start as usize..(run.start + run.len) as usize]
    }
}

/// Walk two groupings together, yielding every image slot present in either.
///
/// pycocotools loops over *all* images for every category; images with neither
/// ground truth nor detections produce a `None` entry that contributes nothing
/// to any metric, so skipping them is observationally identical and turns an
/// `images x categories` scan into one proportional to the data.
pub struct RunJoin<'a> {
    a: &'a [Run],
    b: &'a [Run],
    i: usize,
    j: usize,
}

impl<'a> RunJoin<'a> {
    pub fn new(a: &'a [Run], b: &'a [Run]) -> RunJoin<'a> {
        RunJoin { a, b, i: 0, j: 0 }
    }
}

impl<'a> Iterator for RunJoin<'a> {
    /// `(image slot, ground-truth run, detection run)`
    type Item = (u32, Option<Run>, Option<Run>);

    fn next(&mut self) -> Option<Self::Item> {
        match (self.a.get(self.i), self.b.get(self.j)) {
            (None, None) => None,
            (Some(ra), None) => {
                self.i += 1;
                Some((ra.img_slot, Some(*ra), None))
            }
            (None, Some(rb)) => {
                self.j += 1;
                Some((rb.img_slot, None, Some(*rb)))
            }
            (Some(ra), Some(rb)) => {
                if ra.img_slot == rb.img_slot {
                    self.i += 1;
                    self.j += 1;
                    Some((ra.img_slot, Some(*ra), Some(*rb)))
                } else if ra.img_slot < rb.img_slot {
                    self.i += 1;
                    Some((ra.img_slot, Some(*ra), None))
                } else {
                    self.j += 1;
                    Some((rb.img_slot, None, Some(*rb)))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_preserve_annotation_order() {
        // 6 annotations over images {0, 1} and categories {0, 1}.
        let img = [0u32, 0, 1, 0, 1, 1];
        let cat = [0u32, 1, 0, 0, 1, 0];
        let g = Grouping::build(&img, &cat, 2, true);
        // Category 0: image 0 has annotations 0 and 3 (in that order),
        // image 1 has 2 and 5.
        let runs = g.group(0);
        assert_eq!(runs.len(), 2);
        assert_eq!(g.indices(&runs[0]), &[0, 3]);
        assert_eq!(g.indices(&runs[1]), &[2, 5]);
        let runs = g.group(1);
        assert_eq!(g.indices(&runs[0]), &[1]);
        assert_eq!(g.indices(&runs[1]), &[4]);
    }

    #[test]
    fn merged_groups_are_category_major() {
        let img = [0u32, 0, 0];
        let cat = [1u32, 0, 1];
        let g = Grouping::build(&img, &cat, 1, false);
        let runs = g.group(0);
        assert_eq!(runs.len(), 1);
        // Category 0 first (annotation 1), then category 1 (annotations 0, 2).
        assert_eq!(g.indices(&runs[0]), &[1, 0, 2]);
    }

    #[test]
    fn annotations_outside_the_evaluation_are_dropped() {
        // u32::MAX marks an annotation whose image or category is not being
        // evaluated. Keeping it would put it in some other group's bucket.
        let img = [0u32, u32::MAX, 1];
        let cat = [0u32, 0, u32::MAX];
        let g = Grouping::build(&img, &cat, 1, true);
        assert_eq!(g.order, vec![0]);
        assert_eq!(g.group(0).len(), 1);
    }

    #[test]
    fn a_group_with_no_annotations_reads_back_empty() {
        // Categories that appear in params but not in the data are normal;
        // they must not index out of bounds or borrow another group's runs.
        let g = Grouping::build(&[0u32], &[0u32], 3, true);
        assert_eq!(g.group(0).len(), 1);
        assert_eq!(g.group(1).len(), 0);
        assert_eq!(g.group(2).len(), 0);
    }

    #[test]
    fn building_from_nothing_is_empty_everywhere() {
        let g = Grouping::build(&[], &[], 2, true);
        assert!(g.order.is_empty());
        assert_eq!(g.group(0).len(), 0);
        assert_eq!(g.group(1).len(), 0);
    }

    #[test]
    fn runs_are_ascending_by_image_within_a_group() {
        // RunJoin merge-walks two groupings and relies on this order.
        let img = [5u32, 1, 3, 1];
        let cat = [0u32, 0, 0, 0];
        let g = Grouping::build(&img, &cat, 1, true);
        let slots: Vec<u32> = g.group(0).iter().map(|r| r.img_slot).collect();
        assert_eq!(slots, vec![1, 3, 5]);
        // Image 1 keeps both of its annotations, in annotation order.
        assert_eq!(g.indices(&g.group(0)[0]), &[1, 3]);
    }

    #[test]
    fn join_with_one_empty_side() {
        let a = [Run {
            img_slot: 3,
            start: 0,
            len: 2,
        }];
        let got: Vec<(u32, bool, bool)> = RunJoin::new(&a, &[])
            .map(|(i, g, d)| (i, g.is_some(), d.is_some()))
            .collect();
        assert_eq!(got, vec![(3, true, false)]);

        let got: Vec<(u32, bool, bool)> = RunJoin::new(&[], &a)
            .map(|(i, g, d)| (i, g.is_some(), d.is_some()))
            .collect();
        assert_eq!(got, vec![(3, false, true)]);

        assert_eq!(RunJoin::new(&[], &[]).count(), 0);
    }

    #[test]
    fn join_yields_union_of_images() {
        let a = [
            Run {
                img_slot: 0,
                start: 0,
                len: 1,
            },
            Run {
                img_slot: 2,
                start: 1,
                len: 1,
            },
        ];
        let b = [
            Run {
                img_slot: 1,
                start: 0,
                len: 1,
            },
            Run {
                img_slot: 2,
                start: 1,
                len: 1,
            },
        ];
        let got: Vec<u32> = RunJoin::new(&a, &b).map(|(i, _, _)| i).collect();
        assert_eq!(got, vec![0, 1, 2]);
    }
}
