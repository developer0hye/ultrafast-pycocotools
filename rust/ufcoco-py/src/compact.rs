//! Immutable file snapshots with native bbox columns and lazy Python annotations.
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use serde::de::{DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;
use ufcoco_core::eval::{GeomStore, Instances, IouType, MaskRef};
use ufcoco_core::rle;

use super::{ExtractTimings, GroupSet, IdMap, RawSegm, CHUNK};

#[derive(Deserialize)]
#[serde(untagged)]
enum Crowd {
    Bool(bool),
    Int(i64),
}

// Decode directly to f64 without retaining a tagged Number for every coordinate.
// Integer bounds preserve Python's exact multiplication before conversion to f64.
struct BboxNumber(f64);
impl<'de> Deserialize<'de> for BboxNumber {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct NumberVisitor;
        impl<'de> Visitor<'de> for NumberVisitor {
            type Value = BboxNumber;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a bbox coordinate with an exactly representable integer part")
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<BboxNumber, E> {
                if v.unsigned_abs() > (1u64 << 53) {
                    return Err(E::custom("bbox integer exceeds exact float64 range"));
                }
                Ok(BboxNumber(v as f64))
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<BboxNumber, E> {
                if v > (1u64 << 53) {
                    return Err(E::custom("bbox integer exceeds exact float64 range"));
                }
                Ok(BboxNumber(v as f64))
            }
            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<BboxNumber, E> {
                Ok(BboxNumber(v))
            }
        }
        d.deserialize_any(NumberVisitor)
    }
}

#[derive(Deserialize)]
struct Record<'a> {
    #[serde(default, deserialize_with = "field_present")]
    caption: bool,
    #[serde(default)]
    id: Option<i64>,
    image_id: i64,
    category_id: i64,
    bbox: [BboxNumber; 4],
    #[serde(default)]
    area: Option<f64>,
    #[serde(default)]
    score: Option<f64>,
    #[serde(default)]
    iscrowd: Option<Crowd>,
    // Presence tracking makes duplicate geometry fields take the ordinary JSON
    // loader's last-value path, just like duplicate scalar fields already do.
    #[serde(default, borrow)]
    segmentation: Option<&'a serde_json::value::RawValue>,
    #[serde(default, borrow, rename = "keypoints")]
    keypoints: Option<&'a serde_json::value::RawValue>,
    #[serde(default, borrow)]
    num_keypoints: Option<&'a serde_json::value::RawValue>,
}

fn field_present<'de, D: serde::Deserializer<'de>>(d: D) -> Result<bool, D::Error> {
    serde::de::IgnoredAny::deserialize(d)?;
    Ok(true)
}

// Detection IDs and areas are derived by loadRes; retain only independent values.
struct Row {
    image: i64,
    category: i64,
    score: f64,
}

struct GroundTruth {
    id: i64,
    area: f64,
    crowd: bool,
}

// Byte ranges refer to the immutable owned snapshot, never borrowed Python data.
// Allocate this column only for files containing pose fields.
#[derive(Clone, Default)]
struct PoseRanges {
    points: Option<std::ops::Range<usize>>,
    count: Option<std::ops::Range<usize>>,
}

/// `(offset, length)` of a row's `segmentation` value in the snapshot, or
/// [`NO_SEGMENTATION`] when the field is absent or null.
type SegmSpan = (u32, u32);
const NO_SEGMENTATION: SegmSpan = (u32::MAX, 0);

type Columns = (
    Vec<Row>,
    Option<Vec<GroundTruth>>,
    Vec<[f64; 4]>,
    Vec<PoseRanges>,
    Vec<SegmSpan>,
);

/// `(results file, snapshot base address, record segmentation spans)`.
/// Spans use 32-bit offsets and are skipped for snapshots of 4 GiB or more.
struct Rows(bool, usize, bool);
impl<'de> DeserializeSeed<'de> for Rows {
    type Value = Columns;
    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        d.deserialize_seq(self)
    }
}
impl<'de> Visitor<'de> for Rows {
    type Value = Columns;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("bbox annotations")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut columns = ColumnBuilder::new(self.0, self.1, self.2);
        while let Some(record) = seq.next_element::<Record>()? {
            columns.push(record).map_err(serde::de::Error::custom)?;
        }
        Ok(columns.finish())
    }
}

/// Column storage for one annotation array, or one chunk of a result array.
struct ColumnBuilder {
    results: bool,
    /// Snapshot base address for pose and segmentation spans.
    base: usize,
    segm_spans: bool,
    rows: Vec<Row>,
    ground_truth: Option<Vec<GroundTruth>>,
    boxes: Vec<[f64; 4]>,
    pose: Vec<PoseRanges>,
    segm: Vec<SegmSpan>,
    ids: HashSet<i64>,
}

impl ColumnBuilder {
    fn new(results: bool, base: usize, segm_spans: bool) -> Self {
        ColumnBuilder {
            results,
            base,
            segm_spans,
            rows: Vec::new(),
            ground_truth: (!results).then(Vec::new),
            boxes: Vec::new(),
            pose: Vec::new(),
            segm: Vec::new(),
            ids: HashSet::new(),
        }
    }

    fn push(&mut self, record: Record<'_>) -> Result<(), &'static str> {
        if self.results && record.caption {
            return Err("caption results require caption loading");
        }
        let id = if self.results {
            self.rows.len() as i64 + 1
        } else {
            record.id.ok_or("missing GT id")?
        };
        if !self.results && !self.ids.insert(id) {
            return Err("duplicate GT ids require dictionary indexing");
        }
        if record.keypoints.is_some() || record.num_keypoints.is_some() {
            self.pose.resize_with(self.rows.len(), PoseRanges::default);
            let base = self.base;
            let range = |value: &serde_json::value::RawValue| {
                let text = value.get();
                let start = text.as_ptr() as usize - base;
                start..start + text.len()
            };
            self.pose.push(PoseRanges {
                points: record.keypoints.map(range),
                count: record.num_keypoints.map(range),
            });
        } else if !self.pose.is_empty() {
            self.pose.push(PoseRanges::default());
        }
        if let (true, Some(value)) = (self.segm_spans, record.segmentation) {
            // Offsets fit: the caller enables spans only below 4 GiB.
            let text = value.get();
            self.segm.resize(self.rows.len(), NO_SEGMENTATION);
            self.segm.push((
                (text.as_ptr() as usize - self.base) as u32,
                text.len() as u32,
            ));
        } else if !self.segm.is_empty() {
            self.segm.push(NO_SEGMENTATION);
        }
        let bbox = record.bbox.map(|x| x.0);
        let area = if self.results {
            bbox[2] * bbox[3]
        } else {
            record.area.unwrap_or(bbox[2] * bbox[3])
        };
        let crowd = !self.results
            && match record.iscrowd {
                Some(Crowd::Bool(x)) => x,
                Some(Crowd::Int(x)) => x != 0,
                None => false,
            };
        self.boxes.push(bbox);
        self.rows.push(Row {
            image: record.image_id,
            category: record.category_id,
            score: record.score.unwrap_or(0.0),
        });
        if let Some(fields) = &mut self.ground_truth {
            fields.push(GroundTruth { id, area, crowd });
        }
        Ok(())
    }

    fn finish(mut self) -> Columns {
        self.rows.shrink_to_fit();
        self.boxes.shrink_to_fit();
        if let Some(fields) = &mut self.ground_truth {
            fields.shrink_to_fit();
        }
        self.pose.shrink_to_fit();
        self.segm.shrink_to_fit();
        (
            self.rows,
            self.ground_truth,
            self.boxes,
            self.pose,
            self.segm,
        )
    }
}

fn is_json_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\n' | b'\r' | b'\t')
}

fn skip_whitespace(raw: &[u8], mut position: usize) -> usize {
    while raw.get(position).copied().is_some_and(is_json_whitespace) {
        position += 1;
    }
    position
}

/// Result files at least this large are parsed in parallel chunks.
const PARALLEL_RESULTS_BYTES: usize = 1 << 22;

/// Parse a top-level result array in parallel, or `None` to use the
/// sequential parser.
///
/// The array is split at candidate record starts: `{` after `}`, `,` and
/// whitespace. A candidate may lie inside a string, so each chunk is parsed
/// record by record with the ordinary `Record` deserializer and accepted only
/// if its last record ends exactly at the next chunk's start. The first chunk
/// starts at the real first record, so by induction every accepted start is a
/// real record boundary and the records are exactly those of the sequential
/// parse, in order. Anything else (a misaligned or failing chunk, trailing
/// data, a non-object element) returns `None`, and the sequential parser
/// decides, including its error.
fn parallel_results(raw: &[u8], segm_spans: bool) -> Option<Columns> {
    let threads = rayon::current_num_threads();
    if raw.len() < PARALLEL_RESULTS_BYTES || threads < 2 {
        return None;
    }
    let mut first = skip_whitespace(raw, 0);
    if raw.get(first) != Some(&b'[') {
        return None;
    }
    first = skip_whitespace(raw, first + 1);
    if raw.get(first) != Some(&b'{') {
        return None;
    }
    let chunks = threads * 4;
    let mut starts = vec![first];
    for chunk in 1..chunks {
        let target = raw.len() / chunks * chunk;
        if let Some(start) = candidate_record_start(raw, target.max(first + 1)) {
            if start > *starts.last().unwrap() {
                starts.push(start);
            }
        }
    }
    let base = raw.as_ptr() as usize;
    let parts: Vec<Option<ColumnBuilder>> = (0..starts.len())
        .into_par_iter()
        .map(|i| parse_result_chunk(raw, starts[i], starts.get(i + 1).copied(), base, segm_spans))
        .collect();
    let parts: Vec<ColumnBuilder> = parts.into_iter().collect::<Option<_>>()?;
    Some(concatenate(parts, base, segm_spans))
}

/// The first `{` at or after `from` that follows `}`, `,` and JSON whitespace.
fn candidate_record_start(raw: &[u8], from: usize) -> Option<usize> {
    let mut position = from;
    loop {
        position += raw.get(position..)?.iter().position(|&b| b == b'}')? + 1;
        let comma = skip_whitespace(raw, position);
        if raw.get(comma) == Some(&b',') {
            let next = skip_whitespace(raw, comma + 1);
            if raw.get(next) == Some(&b'{') {
                return Some(next);
            }
        }
    }
}

/// Records from `start` up to exactly `end` (the next chunk's start), or to
/// the closing bracket and end of input for the last chunk.
fn parse_result_chunk(
    raw: &[u8],
    start: usize,
    end: Option<usize>,
    base: usize,
    segm_spans: bool,
) -> Option<ColumnBuilder> {
    let mut columns = ColumnBuilder::new(true, base, segm_spans);
    let mut position = start;
    loop {
        let mut stream =
            serde_json::Deserializer::from_slice(&raw[position..]).into_iter::<Record>();
        let record = stream.next()?.ok()?;
        position += stream.byte_offset();
        columns.push(record).ok()?;
        position = skip_whitespace(raw, position);
        match raw.get(position) {
            Some(b',') => {
                position = skip_whitespace(raw, position + 1);
                if let Some(end) = end {
                    if position >= end {
                        return (position == end).then_some(columns);
                    }
                }
            }
            Some(b']') if end.is_none() => {
                return (skip_whitespace(raw, position + 1) == raw.len()).then_some(columns);
            }
            _ => return None,
        }
    }
}

/// Join chunk columns in order, as one sequential parse would have built them.
fn concatenate(parts: Vec<ColumnBuilder>, base: usize, segm_spans: bool) -> Columns {
    let total: usize = parts.iter().map(|part| part.rows.len()).sum();
    let any_pose = parts.iter().any(|part| !part.pose.is_empty());
    let any_segm = parts.iter().any(|part| !part.segm.is_empty());
    let mut out = ColumnBuilder::new(true, base, segm_spans);
    out.rows.reserve_exact(total);
    out.boxes.reserve_exact(total);
    if any_pose {
        out.pose.reserve_exact(total);
    }
    if any_segm {
        out.segm.reserve_exact(total);
    }
    for part in parts {
        let n = part.rows.len();
        out.rows.extend(part.rows);
        out.boxes.extend(part.boxes);
        if any_pose {
            if part.pose.is_empty() {
                out.pose
                    .resize_with(out.pose.len() + n, PoseRanges::default);
            } else {
                out.pose.extend(part.pose);
            }
        }
        if any_segm {
            if part.segm.is_empty() {
                out.segm.resize(out.segm.len() + n, NO_SEGMENTATION);
            } else {
                out.segm.extend(part.segm);
            }
        }
    }
    out.finish()
}

struct Dataset<'py> {
    py: Python<'py>,
    raw_base: usize,
    segm_spans: bool,
}
impl<'de, 'py> DeserializeSeed<'de> for Dataset<'py> {
    type Value = (Py<PyDict>, Columns);
    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        d.deserialize_map(self)
    }
}
impl<'de, 'py> Visitor<'de> for Dataset<'py> {
    type Value = (Py<PyDict>, Columns);
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("COCO dataset")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let metadata = PyDict::new(self.py);
        let mut memo = HashMap::new();
        let mut rows = None;
        while let Some(key) = map.next_key::<String>()? {
            if key == "annotations" {
                if rows.is_some() {
                    return Err(serde::de::Error::custom(
                        "duplicate annotation arrays require ordinary JSON loading",
                    ));
                }
                rows = Some(map.next_value_seed(Rows(false, self.raw_base, self.segm_spans))?);
                metadata
                    .set_item("annotations", self.py.None())
                    .map_err(serde::de::Error::custom)?;
            } else {
                let value = map.next_value_seed(super::json::Builder {
                    py: self.py,
                    memo: &mut memo,
                })?;
                metadata
                    .set_item(key, value)
                    .map_err(serde::de::Error::custom)?;
            }
        }
        Ok((
            metadata.unbind(),
            rows.ok_or_else(|| serde::de::Error::custom("missing annotations"))?,
        ))
    }
}

#[pyclass(module = "ultrafast_pycocotools._ufcoco")]
pub struct CompactBbox {
    // Keep an owned snapshot: changing or deleting the input file cannot change later API reads.
    // Shared with evaluators that decode masks from it lazily.
    raw: Arc<Vec<u8>>,
    rows: Vec<Row>,
    ground_truth: Option<Vec<GroundTruth>>,
    boxes: Arc<Vec<[f64; 4]>>,
    results: bool,
    pose: Vec<PoseRanges>,
    /// Segmentation value spans by row (shorter when trailing rows have none),
    /// or `None` when the snapshot is too large for 32-bit offsets.
    segm: Option<Vec<SegmSpan>>,
}

/// `(image slot, category slot)` for one file row, or [`UNSELECTED`] when
/// either ID is outside the evaluation.
type Slot = (u32, u32);
const UNSELECTED: Slot = (u32::MAX, u32::MAX);

impl CompactBbox {
    /// Resolve every row's image and category slot once. Each later pass reads
    /// this column instead of repeating two map lookups per row.
    pub(super) fn slots(&self, images: &IdMap<u32>, categories: &IdMap<u32>) -> Vec<Slot> {
        self.rows
            .iter()
            .map(
                |row| match (images.get(&row.image), categories.get(&row.category)) {
                    (Some(&image), Some(&category)) => (image, category),
                    _ => UNSELECTED,
                },
            )
            .collect()
    }

    pub(super) fn groups(slots: &[Slot], use_cats: bool) -> GroupSet {
        slots
            .iter()
            .filter(|&&slot| slot != UNSELECTED)
            .map(|&(image, category)| (image, if use_cats { category } else { 0 }))
            .collect()
    }

    pub fn instances(&self, is_gt: bool, slots: &[Slot]) -> Instances {
        let n = slots.iter().filter(|&&slot| slot != UNSELECTED).count();
        let geom = if n == self.rows.len() {
            GeomStore::SharedBboxes(Arc::clone(&self.boxes))
        } else {
            GeomStore::Bboxes(Vec::with_capacity(n))
        };
        let mut out = super::new_instances_with_geom(n, geom);
        for (index, (row, &(image, category))) in self.rows.iter().zip(slots).enumerate() {
            if (image, category) == UNSELECTED {
                continue;
            }
            let (id, area, crowd) = match &self.ground_truth {
                Some(fields) => {
                    let gt = &fields[index];
                    (gt.id, gt.area, gt.crowd)
                }
                None => (
                    index as i64 + 1,
                    self.boxes[index][2] * self.boxes[index][3],
                    false,
                ),
            };
            out.ids.push(id);
            out.img_slot.push(image);
            out.cat_slot.push(category);
            out.scores.push(row.score);
            out.areas.push(area);
            out.iscrowd.push(crowd);
            out.ignore.push(is_gt && crowd);
            out.lvis_mark.push(false);
            if let GeomStore::Bboxes(ref mut boxes) = out.geom {
                boxes.push(self.boxes[index]);
            }
        }
        out
    }

    /// Decode independent DT coordinate spans straight into disjoint final slices.
    /// The scalar rows and stable cap selection remain unchanged.
    fn parallel_pose_detections(
        &self,
        out: &mut Instances,
        slots: &[Slot],
        keep: &[bool],
    ) -> PyResult<bool> {
        let rows: Vec<_> = slots
            .iter()
            .enumerate()
            .filter_map(|(index, &slot)| (slot != UNSELECTED).then_some(index))
            .collect();
        // Preserve the existing handling of absent/null geometry and empty inputs.
        if rows.is_empty()
            || rows
                .iter()
                .any(|&i| self.pose.get(i).and_then(|r| r.points.as_ref()).is_none())
        {
            return Ok(false);
        }
        let mut joints = 0;
        let mut scratch = Vec::new();
        let mut visible = Vec::new();
        let range = self.pose[rows[0]].points.as_ref().unwrap();
        let mut de = serde_json::Deserializer::from_slice(&self.raw[range.clone()]);
        KeypointCoordinates::<false> {
            data: &mut scratch,
            visible: &mut visible,
            k: &mut joints,
            is_gt: false,
            instances: 0,
        }
        .deserialize(&mut de)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if joints == 0 {
            return Ok(false);
        }
        let GeomStore::Keypoints {
            data, offsets, k, ..
        } = &mut out.geom
        else {
            unreachable!()
        };
        *k = joints;
        let mut retained = Vec::new();
        let mut discarded = Vec::new();
        let sparse = keep.iter().any(|&flag| !flag);
        if sparse {
            offsets.reserve_exact(rows.len());
        }
        for (&index, &retain) in rows.iter().zip(keep) {
            if sparse {
                offsets.push(if retain {
                    retained.len() * joints * 2
                } else {
                    usize::MAX
                });
            }
            if retain {
                retained.push(index);
            } else {
                discarded.push(index);
            }
        }
        data.resize(retained.len() * joints * 2, 0.0);
        let parse = |index: usize, destination: &mut [f64]| -> Result<(), serde_json::Error> {
            let ranges = &self.pose[index];
            let range = ranges.points.as_ref().unwrap();
            let raw = &self.raw[range.clone()];
            if !super::pose_numbers::decode(raw, destination, joints) {
                let mut de = serde_json::Deserializer::from_slice(raw);
                CoordinateSlice {
                    destination,
                    joints,
                }
                .deserialize(&mut de)?;
                de.end()?;
            }
            if let Some(range) = &ranges.count {
                serde_json::from_slice::<Crowd>(&self.raw[range.clone()])?;
            }
            Ok(())
        };
        let (written, checked) = rayon::join(
            || {
                data.par_chunks_mut(joints * 2)
                    .zip(retained.par_iter())
                    .try_for_each(|(destination, &index)| parse(index, destination))
            },
            || {
                discarded
                    .par_iter()
                    .try_for_each(|&index| parse(index, &mut []))
            },
        );
        written
            .and(checked)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(true)
    }

    fn pose_instances(
        &self,
        out: &mut Instances,
        is_gt: bool,
        slots: &[Slot],
        keep: Option<&[bool]>,
    ) -> PyResult<()> {
        if !is_gt && self.parallel_pose_detections(out, slots, keep.unwrap())? {
            return Ok(());
        }
        let GeomStore::Keypoints {
            data,
            visible,
            offsets,
            k,
        } = &mut out.geom
        else {
            unreachable!()
        };
        let sparse = keep.is_some_and(|flags| flags.iter().any(|&flag| !flag));
        let retained = keep.map_or(out.ids.len(), |flags| {
            flags.iter().filter(|&&flag| flag).count()
        });
        if sparse {
            offsets.reserve_exact(out.ids.len());
        }
        let mut selected = 0;
        for (index, &slot) in slots.iter().enumerate() {
            if slot == UNSELECTED {
                continue;
            }
            if is_gt {
                out.bboxes.push(self.boxes[index]);
            }
            let retain = keep.map_or(true, |flags| flags[selected]);
            if sparse {
                offsets.push(if retain { data.len() } else { usize::MAX });
            }
            let ranges = self.pose.get(index);
            if let Some(range) = ranges.and_then(|r| r.points.as_ref()) {
                let mut de = serde_json::Deserializer::from_slice(&self.raw[range.clone()]);
                let parsed = if retain {
                    KeypointCoordinates::<true> {
                        data,
                        visible,
                        k,
                        is_gt,
                        instances: retained,
                    }
                    .deserialize(&mut de)
                } else {
                    KeypointCoordinates::<false> {
                        data,
                        visible,
                        k,
                        is_gt,
                        instances: retained,
                    }
                    .deserialize(&mut de)
                };
                parsed
                    .and_then(|()| de.end())
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            }
            // Preserve validation of this field on detection inputs too.
            let count = ranges
                .and_then(|r| r.count.as_ref())
                .map(|range| serde_json::from_slice::<Crowd>(&self.raw[range.clone()]))
                .transpose()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if is_gt {
                let has_keypoints = match count {
                    Some(Crowd::Bool(b)) => b,
                    Some(Crowd::Int(n)) => n != 0,
                    None => false,
                };
                out.ignore[selected] |= !has_keypoints;
            }
            selected += 1;
        }
        Ok(())
    }

    /// Segmentation masks from the spans recorded at load time.
    ///
    /// Each selected row's `segmentation` value is parsed on its own, in
    /// parallel, instead of re-parsing the whole file. Unused rows are still
    /// parsed so malformed geometry is rejected exactly as before. A plain
    /// compressed `counts` string is kept as a reference into the snapshot and
    /// decoded by the engine when an IoU needs it; every other form is
    /// rasterised here with the same routine as the pipelined path.
    fn span_masks(
        &self,
        spans: &[SegmSpan],
        sizes: &[Option<(u32, u32)>],
        slots: &[Slot],
        opposing_groups: Option<&GroupSet>,
        use_cats: bool,
        detection_keep: Option<&[bool]>,
    ) -> PyResult<Vec<MaskRef>> {
        let rows: Vec<u32> = (0..slots.len() as u32)
            .filter(|&i| slots[i as usize] != UNSELECTED)
            .collect();
        let base = self.raw.as_ptr() as usize;
        let mask = |selected: usize, scratch: &mut rle::PolyScratch| -> Result<MaskRef, String> {
            let index = rows[selected] as usize;
            let (image, category) = slots[index];
            let (offset, length) = spans.get(index).copied().unwrap_or(NO_SEGMENTATION);
            let parsed = if (offset, length) == NO_SEGMENTATION {
                None
            } else if let Some((coords, ends)) = super::pose_numbers::decode_polygons(
                &self.raw[offset as usize..offset as usize + length as usize],
            ) {
                Some(SpanSegm::Raw(RawSegm::Poly { coords, ends }))
            } else {
                let value = &self.raw[offset as usize..offset as usize + length as usize];
                let mut de = serde_json::Deserializer::from_slice(value);
                let parsed = SpanGeometry(base)
                    .deserialize(&mut de)
                    .and_then(|parsed| de.end().map(|()| parsed))
                    .map_err(|e| e.to_string())?;
                Some(parsed)
            };
            let (h, w) = sizes[image as usize]
                .ok_or_else(|| format!("no image entry for image_id {}", self.rows[index].image))?;
            let group = (image, if use_cats { category } else { 0 });
            if opposing_groups.is_some_and(|groups| !groups.contains(&group))
                || detection_keep.is_some_and(|keep| !keep[selected])
            {
                return Ok(MaskRef::Ready(Default::default()));
            }
            Ok(match parsed {
                Some(SpanSegm::Encoded {
                    start,
                    len,
                    escaped,
                }) => MaskRef::Encoded {
                    start,
                    len,
                    h,
                    w,
                    escaped,
                },
                Some(SpanSegm::Raw(raw)) => MaskRef::Ready(super::raw_to_rle(&raw, h, w, scratch)),
                None => MaskRef::Ready(super::raw_to_rle(
                    &RawSegm::FromBbox(self.boxes[index]),
                    h,
                    w,
                    scratch,
                )),
            })
        };
        let parallel: Result<Vec<MaskRef>, String> = (0..rows.len())
            .into_par_iter()
            .map_init(rle::PolyScratch::default, |scratch, selected| {
                mask(selected, scratch)
            })
            .collect();
        parallel
            .or_else(|_| -> Result<Vec<MaskRef>, String> {
                // Report the first failing row in file order, as the sequential path does.
                let mut scratch = rle::PolyScratch::default();
                for selected in 0..rows.len() {
                    mask(selected, &mut scratch)?;
                }
                unreachable!("a parallel mask failure must fail sequentially")
            })
            .map_err(PyValueError::new_err)
    }

    /// Read only the requested geometry from the immutable JSON snapshot.
    /// No Python annotation dictionaries or indexes are needed for evaluation.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn geometry_instances(
        &self,
        is_gt: bool,
        kind: IouType,
        sizes: &IdMap<(u32, u32)>,
        image_ids: &[i64],
        slots: &[Slot],
        categories: usize,
        boundary_dilation: f64,
        timings: &ExtractTimings,
        opposing_groups: Option<&GroupSet>,
        use_cats: bool,
        max_det: usize,
    ) -> PyResult<Instances> {
        let mut out = self.instances(is_gt, slots);
        out.geom = super::new_geometry(out.len(), kind);
        if kind == IouType::Keypoints {
            let start = Instant::now();
            let keep = (!is_gt).then(|| {
                ufcoco_core::eval::detection_geometry_keep(&out, categories, use_cats, max_det)
            });
            self.pose_instances(&mut out, is_gt, slots, keep.as_deref())?;
            let counter = if is_gt {
                &timings.gt_read_ns
            } else {
                &timings.dt_read_ns
            };
            counter.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            return Ok(out);
        }
        let mask_count = out.len();
        let detection_keep = (!is_gt && kind != IouType::Keypoints).then(|| {
            ufcoco_core::eval::detection_geometry_keep(&out, categories, use_cats, max_det)
        });
        // Image sizes by slot, so the per-row geometry pass needs no hashing.
        let sizes: Vec<Option<(u32, u32)>> =
            image_ids.iter().map(|id| sizes.get(id).copied()).collect();
        if let (IouType::Segm, Some(spans)) = (kind, &self.segm) {
            let start = Instant::now();
            let masks = self.span_masks(
                spans,
                &sizes,
                slots,
                opposing_groups,
                use_cats,
                detection_keep.as_deref(),
            )?;
            out.geom = GeomStore::EncodedMasks {
                raw: Arc::clone(&self.raw),
                masks,
            };
            let counter = if is_gt {
                &timings.gt_read_ns
            } else {
                &timings.dt_read_ns
            };
            counter.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            return Ok(out);
        }
        std::thread::scope(|scope| -> PyResult<()> {
            let (tx, rx) = std::sync::mpsc::sync_channel(2);
            let worker = scope.spawn(|| {
                super::rasterise_chunks(
                    rx,
                    kind == IouType::Boundary,
                    boundary_dilation,
                    timings,
                    mask_count,
                )
            });
            let start = Instant::now();
            let mut geometry = GeometryRows {
                source: self,
                sizes: &sizes,
                slots,
                chunk: Vec::with_capacity(if kind == IouType::Keypoints { 0 } else { CHUNK }),
                sender: &tx,
                timings,
                opposing_groups,
                use_cats,
                detection_keep: detection_keep.as_deref(),
            };
            let mut de = serde_json::Deserializer::from_slice(&self.raw);
            let result = if self.results {
                (&mut geometry).deserialize(&mut de)
            } else {
                de.deserialize_map(&mut geometry)
            }
            .and_then(|()| de.end());
            if result.is_ok() && !geometry.chunk.is_empty() {
                tx.send(std::mem::take(&mut geometry.chunk))
                    .map_err(|_| PyValueError::new_err("mask rasteriser stopped"))?;
            }
            let counter = if is_gt {
                &timings.gt_read_ns
            } else {
                &timings.dt_read_ns
            };
            counter.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            drop(geometry);
            drop(tx);
            let masks = worker.join().expect("mask worker panicked");
            result.map_err(|e| PyValueError::new_err(e.to_string()))?;
            super::attach_masks(&mut out.geom, masks);
            Ok(())
        })?;
        Ok(out)
    }
}

#[derive(Deserialize)]
struct MaskGeometry {
    #[serde(default)]
    segmentation: Option<RawSegm>,
}

/// A `segmentation` value parsed from its own span.
enum SpanSegm {
    /// A compressed `counts` string at `raw[start..start + len]`; see
    /// [`MaskRef::Encoded`] for `escaped`.
    Encoded {
        start: usize,
        len: u32,
        escaped: bool,
    },
    Raw(RawSegm),
}

/// Parse one `segmentation` value like [`RawSegm`], but keep an unescaped
/// compressed `counts` string as a snapshot offset. The field is the address
/// of the snapshot start.
struct SpanGeometry(usize);
impl<'de> DeserializeSeed<'de> for SpanGeometry {
    type Value = SpanSegm;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<SpanSegm, D::Error> {
        d.deserialize_any(self)
    }
}
impl<'de> Visitor<'de> for SpanGeometry {
    type Value = SpanSegm;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("polygon rings or an RLE object")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<SpanSegm, A::Error> {
        let mut coords = Vec::new();
        let mut ends = Vec::new();
        while let Some(ring) = seq.next_element::<Vec<Coordinate>>()? {
            coords.extend(ring.into_iter().map(|value| value.0));
            ends.push(coords.len() as u32);
        }
        Ok(SpanSegm::Raw(RawSegm::Poly { coords, ends }))
    }
    // Callers try `pose_numbers::decode_polygons` on the span first.
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<SpanSegm, A::Error> {
        let mut counts = None;
        while let Some(key) = map.next_key::<MaskField>()? {
            match key {
                MaskField::Counts => {
                    let value: &'de serde_json::value::RawValue = map.next_value()?;
                    counts = Some(span_counts(value, self.0).map_err(serde::de::Error::custom)?);
                }
                MaskField::Other => {
                    map.next_value::<serde::de::IgnoredAny>()?;
                }
            }
        }
        counts.ok_or_else(|| serde::de::Error::custom("RLE segmentation is missing counts"))
    }
}

/// Classify a `counts` value from its validated JSON text. A string whose only
/// escapes are `\\` pairs stays in the snapshot; any other form is decoded
/// with the same [`Counts`] visitor as the whole-file path.
fn span_counts(value: &serde_json::value::RawValue, base: usize) -> Result<SpanSegm, String> {
    let text = value.get().as_bytes();
    if let [b'"', inner @ .., b'"'] = text {
        let mut escaped = false;
        let mut plain_pairs_only = true;
        let mut bytes = inner.iter();
        while let Some(&byte) = bytes.next() {
            if byte == b'\\' {
                escaped = true;
                plain_pairs_only &= bytes.next() == Some(&b'\\');
            }
        }
        if plain_pairs_only {
            return Ok(SpanSegm::Encoded {
                start: inner.as_ptr() as usize - base,
                len: inner.len() as u32,
                escaped,
            });
        }
    }
    let mut de = serde_json::Deserializer::from_str(value.get());
    Counts
        .deserialize(&mut de)
        .map(SpanSegm::Raw)
        .map_err(|e| e.to_string())
}

// Stream coordinates straight into evaluator storage; validate skipped rows too.
struct KeypointCoordinates<'a, const RETAIN: bool> {
    data: &'a mut Vec<f64>,
    visible: &'a mut Vec<bool>,
    k: &'a mut usize,
    is_gt: bool,
    instances: usize,
}
impl<'de, const RETAIN: bool> DeserializeSeed<'de> for KeypointCoordinates<'_, RETAIN> {
    type Value = ();
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
        d.deserialize_option(self)
    }
}
impl<'de, const RETAIN: bool> Visitor<'de> for KeypointCoordinates<'_, RETAIN> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("x/y/visibility triplets")
    }
    fn visit_none<E: serde::de::Error>(self) -> Result<(), E> {
        Ok(())
    }
    fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
        d.deserialize_seq(self)
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        let mut joints = 0;
        while let Some(x) = seq.next_element::<Coordinate>()? {
            let y = seq
                .next_element::<Coordinate>()?
                .ok_or_else(|| serde::de::Error::custom("incomplete keypoint triplet"))?;
            let v = seq
                .next_element::<Coordinate>()?
                .ok_or_else(|| serde::de::Error::custom("incomplete keypoint triplet"))?;
            if RETAIN {
                self.data.extend_from_slice(&[x.0, y.0]);
            }
            if self.is_gt {
                self.visible.push(v.0 > 0.0);
            }
            joints += 1;
        }
        if *self.k != 0 && joints != *self.k {
            return Err(serde::de::Error::custom("inconsistent keypoint counts"));
        }
        if *self.k == 0 && joints != 0 {
            *self.k = joints;
            self.data
                .reserve_exact(self.instances * joints * 2 - self.data.len());
            if self.is_gt {
                self.visible
                    .reserve_exact(self.instances * joints - self.visible.len());
            }
        }
        Ok(())
    }
}

/// A checked, allocation-free row decoder. An empty destination validates
/// discarded DT geometry without retaining it; DT visibility is still converted.
struct CoordinateSlice<'a> {
    destination: &'a mut [f64],
    joints: usize,
}
impl<'de> DeserializeSeed<'de> for CoordinateSlice<'_> {
    type Value = ();
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
        d.deserialize_seq(self)
    }
}
impl<'de> Visitor<'de> for CoordinateSlice<'_> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("keypoint x/y/visibility triplets")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        let mut count = 0;
        while let Some(x) = seq.next_element::<Coordinate>()? {
            let y = seq
                .next_element::<Coordinate>()?
                .ok_or_else(|| serde::de::Error::custom("incomplete keypoint triplet"))?;
            let _v = seq
                .next_element::<Coordinate>()?
                .ok_or_else(|| serde::de::Error::custom("incomplete keypoint triplet"))?;
            if count >= self.joints {
                return Err(serde::de::Error::custom("inconsistent keypoint counts"));
            }
            if !self.destination.is_empty() {
                self.destination[count * 2] = x.0;
                self.destination[count * 2 + 1] = y.0;
            }
            count += 1;
        }
        if count != self.joints {
            return Err(serde::de::Error::custom("inconsistent keypoint counts"));
        }
        Ok(())
    }
}

struct Coordinate(f64);
impl<'de> Deserialize<'de> for Coordinate {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct CoordinateVisitor;
        impl<'de> Visitor<'de> for CoordinateVisitor {
            type Value = Coordinate;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a numeric keypoint coordinate")
            }
            fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Self::Value, E> {
                Ok(Coordinate(value))
            }
            fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(Coordinate(value as f64))
            }
            fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(Coordinate(value as f64))
            }
            fn visit_bool<E: serde::de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(Coordinate(u8::from(value) as f64))
            }
        }
        d.deserialize_any(CoordinateVisitor)
    }
}

#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum MaskField {
    Counts,
    #[serde(other)]
    Other,
}

impl<'de> Deserialize<'de> for RawSegm {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct MaskVisitor;
        impl<'de> Visitor<'de> for MaskVisitor {
            type Value = RawSegm;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("polygon rings or an RLE object")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<RawSegm, A::Error> {
                let mut coords = Vec::new();
                let mut ends = Vec::new();
                while let Some(ring) = seq.next_element::<Vec<Coordinate>>()? {
                    coords.extend(ring.into_iter().map(|value| value.0));
                    ends.push(coords.len() as u32);
                }
                Ok(RawSegm::Poly { coords, ends })
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<RawSegm, A::Error> {
                let mut counts = None;
                while let Some(key) = map.next_key::<MaskField>()? {
                    match key {
                        MaskField::Counts => counts = Some(map.next_value_seed(Counts)?),
                        MaskField::Other => {
                            map.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                counts.ok_or_else(|| serde::de::Error::custom("RLE segmentation is missing counts"))
            }
        }
        d.deserialize_any(MaskVisitor)
    }
}

struct RleCount(u32);
impl<'de> Deserialize<'de> for RleCount {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct CountVisitor;
        impl<'de> Visitor<'de> for CountVisitor {
            type Value = RleCount;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a uint32 run length")
            }
            fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<RleCount, E> {
                u32::try_from(value).map(RleCount).map_err(E::custom)
            }
            fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<RleCount, E> {
                u32::try_from(value).map(RleCount).map_err(E::custom)
            }
            fn visit_bool<E: serde::de::Error>(self, value: bool) -> Result<RleCount, E> {
                Ok(RleCount(u32::from(value)))
            }
        }
        d.deserialize_any(CountVisitor)
    }
}

struct Counts;
impl<'de> DeserializeSeed<'de> for Counts {
    type Value = RawSegm;
    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<RawSegm, D::Error> {
        d.deserialize_any(self)
    }
}
impl<'de> Visitor<'de> for Counts {
    type Value = RawSegm;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("RLE counts")
    }
    fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<RawSegm, E> {
        Ok(RawSegm::Compressed(value.as_bytes().to_vec()))
    }
    fn visit_string<E: serde::de::Error>(self, value: String) -> Result<RawSegm, E> {
        Ok(RawSegm::Compressed(value.into_bytes()))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<RawSegm, A::Error> {
        let mut counts = Vec::new();
        while let Some(value) = seq.next_element::<RleCount>()? {
            counts.push(value.0);
        }
        Ok(RawSegm::Uncompressed(counts))
    }
}

struct GeometryRows<'a> {
    source: &'a CompactBbox,
    /// Image `(height, width)` by image slot; `None` when the image has no entry.
    sizes: &'a [Option<(u32, u32)>],
    slots: &'a [Slot],
    chunk: Vec<(RawSegm, u32, u32)>,
    sender: &'a std::sync::mpsc::SyncSender<Vec<(RawSegm, u32, u32)>>,
    timings: &'a ExtractTimings,
    opposing_groups: Option<&'a GroupSet>,
    use_cats: bool,
    detection_keep: Option<&'a [bool]>,
}

impl<'de> DeserializeSeed<'de> for &mut GeometryRows<'_> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
        d.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for &mut GeometryRows<'_> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("annotation geometry")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        while let Some(key) = map.next_key::<String>()? {
            if key == "annotations" {
                map.next_value_seed(&mut *self)?;
            } else {
                map.next_value::<serde::de::IgnoredAny>()?;
            }
        }
        Ok(())
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        let mut selected = 0;
        for (index, (row, &(image, category))) in
            self.source.rows.iter().zip(self.slots).enumerate()
        {
            if (image, category) == UNSELECTED {
                seq.next_element::<serde::de::IgnoredAny>()?;
                continue;
            }
            let record = seq
                .next_element::<MaskGeometry>()?
                .ok_or_else(|| serde::de::Error::custom("missing mask annotation"))?;
            let (h, w) = self.sizes[image as usize].ok_or_else(|| {
                serde::de::Error::custom(format!("no image entry for image_id {}", row.image))
            })?;
            let mut raw = record
                .segmentation
                .unwrap_or(RawSegm::FromBbox(self.source.boxes[index]));
            // Parse geometry and check image metadata even for unused masks.
            // Only the expensive rasterisation/storage is omitted. Crowds
            // and ignored annotations remain in the opposing group set.
            let group = (image, if self.use_cats { category } else { 0 });
            if self
                .opposing_groups
                .is_some_and(|groups| !groups.contains(&group))
                || self.detection_keep.is_some_and(|keep| !keep[selected])
            {
                raw = RawSegm::Unused;
            }
            self.chunk.push((raw, h, w));
            if self.chunk.len() >= CHUNK {
                let chunk = std::mem::replace(&mut self.chunk, Vec::with_capacity(CHUNK));
                let start = Instant::now();
                self.sender
                    .send(chunk)
                    .map_err(|_| serde::de::Error::custom("mask rasteriser stopped"))?;
                self.timings
                    .read_blocked_ns
                    .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            selected += 1;
        }
        Ok(())
    }
}

#[pymethods]
impl CompactBbox {
    /// Materialize only federated detections, without building the public
    /// annotation indexes or releasing the immutable snapshot. The official
    /// global image cap precedes category/federated filtering.
    fn lvis_annotations(
        &self,
        py: Python<'_>,
        image_ids: Vec<i64>,
        categories: HashSet<i64>,
        verified: HashMap<i64, HashSet<i64>>,
        max_det: Option<usize>,
    ) -> PyResult<Option<Py<PyList>>> {
        if !self.results {
            return Ok(None);
        }
        let mut by_image: HashMap<i64, Vec<usize>> = HashMap::new();
        for (index, row) in self.rows.iter().enumerate() {
            if verified.contains_key(&row.image) {
                by_image.entry(row.image).or_default().push(index);
            }
        }
        let mut selected = Vec::new();
        for image in image_ids {
            if let Some(indices) = by_image.get_mut(&image) {
                if let Some(cap) = max_det {
                    if indices.len() > cap {
                        // Compact JSON scores are finite; equality preserves
                        // file order, including positive/negative zero.
                        indices.sort_by(|&a, &b| {
                            self.rows[b].score.partial_cmp(&self.rows[a].score).unwrap()
                        });
                        indices.truncate(cap);
                    }
                }
                selected.extend(indices.iter().copied().filter(|&index| {
                    let category = self.rows[index].category;
                    categories.contains(&category) && verified[&image].contains(&category)
                }));
            }
        }
        let mut positions = vec![usize::MAX; self.rows.len()];
        for (position, &index) in selected.iter().enumerate() {
            positions[index] = position;
        }
        let mut visitor = SelectedAnnotations {
            py,
            positions: &positions,
            memo: HashMap::with_capacity(32),
            records: std::iter::repeat_with(|| None)
                .take(selected.len())
                .collect(),
        };
        let mut de = serde_json::Deserializer::from_slice(&self.raw);
        de.deserialize_seq(&mut visitor)
            .and_then(|()| de.end())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let annotations = PyList::new(py, visitor.records.into_iter().map(Option::unwrap))?;
        super::prepare_bbox_results(py, &annotations)?;
        for (ann, index) in annotations.iter().zip(selected) {
            // loadRes assigns IDs before filtering, not within the subset.
            ann.set_item("id", index + 1)?;
        }
        Ok(Some(annotations.unbind()))
    }

    fn annotations(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = super::json::parse_bytes(py, &self.raw)?;
        let annotations = if self.results {
            value.bind(py).clone()
        } else {
            value.bind(py).get_item("annotations")?
        };
        if self.results {
            super::prepare_bbox_results(py, annotations.cast::<PyList>()?)?;
        }
        Ok(annotations.unbind())
    }
    fn valid_images(&self, images: Vec<i64>) -> bool {
        let allowed: HashSet<i64> = images.into_iter().collect();
        self.rows.iter().all(|r| allowed.contains(&r.image))
    }
    #[getter]
    fn annotation_count(&self) -> usize {
        self.rows.len()
    }
    #[getter]
    fn snapshot_bytes(&self) -> usize {
        self.raw.len()
    }
    #[getter]
    fn column_bytes(&self) -> usize {
        self.rows.capacity() * std::mem::size_of::<Row>()
            + self
                .ground_truth
                .as_ref()
                .map_or(0, |v| v.capacity() * std::mem::size_of::<GroundTruth>())
            + self.boxes.capacity() * std::mem::size_of::<[f64; 4]>()
            + self.pose.capacity() * std::mem::size_of::<PoseRanges>()
    }
}

struct SelectedAnnotations<'a, 'py> {
    py: Python<'py>,
    positions: &'a [usize],
    memo: super::json::Memo,
    records: Vec<Option<Py<PyAny>>>,
}

impl<'de> Visitor<'de> for &mut SelectedAnnotations<'_, '_> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a detection annotation array")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        for &position in self.positions {
            if position == usize::MAX {
                seq.next_element::<serde::de::IgnoredAny>()?;
            } else {
                self.records[position] = Some(
                    seq.next_element_seed(super::json::Builder {
                        py: self.py,
                        memo: &mut self.memo,
                    })?
                    .ok_or_else(|| serde::de::Error::custom("missing selected annotation"))?,
                );
            }
        }
        Ok(())
    }
}

#[pyfunction]
#[pyo3(signature = (path, results=false))]
pub fn load_compact_bbox(
    py: Python<'_>,
    path: &str,
    results: bool,
) -> PyResult<(Py<PyDict>, CompactBbox)> {
    let raw = std::fs::read(path).map_err(|e| PyIOError::new_err(e.to_string()))?;
    let segm_spans = u32::try_from(raw.len()).is_ok_and(|n| n < u32::MAX);
    let parallel = if results {
        parallel_results(&raw, segm_spans)
    } else {
        None
    };
    let (metadata, (rows, ground_truth, boxes, pose, segm)) = match parallel {
        Some(columns) => (PyDict::new(py).unbind(), columns),
        None => {
            let mut de = serde_json::Deserializer::from_slice(&raw);
            let parsed = if results {
                Rows(true, raw.as_ptr() as usize, segm_spans)
                    .deserialize(&mut de)
                    .map(|rows| (PyDict::new(py).unbind(), rows))
            } else {
                Dataset {
                    py,
                    raw_base: raw.as_ptr() as usize,
                    segm_spans,
                }
                .deserialize(&mut de)
            };
            let parsed = parsed.map_err(|e| PyValueError::new_err(e.to_string()))?;
            de.end().map_err(|e| PyValueError::new_err(e.to_string()))?;
            parsed
        }
    };
    Ok((
        metadata,
        CompactBbox {
            raw: Arc::new(raw),
            rows,
            ground_truth,
            boxes: Arc::new(boxes),
            results,
            pose,
            segm: segm_spans.then_some(segm),
        },
    ))
}
