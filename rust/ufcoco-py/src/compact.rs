//! Immutable file snapshots with native bbox columns and lazy Python annotations.
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::de::{DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;
use ufcoco_core::eval::{GeomStore, Instances, IouType};

use super::{ExtractTimings, RawSegm, CHUNK};

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
struct Record {
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
    #[serde(default, rename = "segmentation")]
    _segmentation: Option<serde::de::IgnoredAny>,
    #[serde(default, rename = "keypoints")]
    _keypoints: Option<serde::de::IgnoredAny>,
    #[serde(default, rename = "num_keypoints")]
    _num_keypoints: Option<serde::de::IgnoredAny>,
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

type Columns = (Vec<Row>, Option<Vec<GroundTruth>>, Vec<[f64; 4]>);

struct Rows(bool);
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
        let mut rows = Vec::new();
        let mut ground_truth = if self.0 { None } else { Some(Vec::new()) };
        let mut boxes = Vec::new();
        let mut ids = HashSet::new();
        while let Some(record) = seq.next_element::<Record>()? {
            if self.0 && record.caption {
                return Err(serde::de::Error::custom(
                    "caption results require caption loading",
                ));
            }
            let id = if self.0 {
                rows.len() as i64 + 1
            } else {
                record
                    .id
                    .ok_or_else(|| serde::de::Error::custom("missing GT id"))?
            };
            if !self.0 && !ids.insert(id) {
                return Err(serde::de::Error::custom(
                    "duplicate GT ids require dictionary indexing",
                ));
            }
            let bbox = record.bbox.map(|x| x.0);
            let area = if self.0 {
                bbox[2] * bbox[3]
            } else {
                record.area.unwrap_or(bbox[2] * bbox[3])
            };
            let crowd = !self.0
                && match record.iscrowd {
                    Some(Crowd::Bool(x)) => x,
                    Some(Crowd::Int(x)) => x != 0,
                    None => false,
                };
            boxes.push(bbox);
            rows.push(Row {
                image: record.image_id,
                category: record.category_id,
                score: record.score.unwrap_or(0.0),
            });
            if let Some(fields) = &mut ground_truth {
                fields.push(GroundTruth { id, area, crowd });
            }
        }
        rows.shrink_to_fit();
        boxes.shrink_to_fit();
        if let Some(fields) = &mut ground_truth {
            fields.shrink_to_fit();
        }
        Ok((rows, ground_truth, boxes))
    }
}

struct Dataset<'py> {
    py: Python<'py>,
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
                rows = Some(map.next_value_seed(Rows(false))?);
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
    raw: Vec<u8>,
    rows: Vec<Row>,
    ground_truth: Option<Vec<GroundTruth>>,
    boxes: Arc<Vec<[f64; 4]>>,
    results: bool,
}

impl CompactBbox {
    pub(super) fn groups(
        &self,
        images: &HashMap<i64, u32>,
        categories: &HashMap<i64, u32>,
        use_cats: bool,
    ) -> HashSet<(u32, u32)> {
        self.rows
            .iter()
            .filter_map(|row| {
                let image = *images.get(&row.image)?;
                let category = *categories.get(&row.category)?;
                Some((image, if use_cats { category } else { 0 }))
            })
            .collect()
    }

    pub fn instances(
        &self,
        is_gt: bool,
        images: &HashMap<i64, u32>,
        categories: &HashMap<i64, u32>,
    ) -> Instances {
        let n = self
            .rows
            .iter()
            .filter(|r| images.contains_key(&r.image) && categories.contains_key(&r.category))
            .count();
        let geom = if n == self.rows.len() {
            GeomStore::SharedBboxes(Arc::clone(&self.boxes))
        } else {
            GeomStore::Bboxes(Vec::with_capacity(n))
        };
        let mut out = super::new_instances_with_geom(n, geom);
        for (index, row) in self.rows.iter().enumerate() {
            let (Some(&image), Some(&category)) =
                (images.get(&row.image), categories.get(&row.category))
            else {
                continue;
            };
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

    /// Read only the requested geometry from the immutable JSON snapshot.
    /// No Python annotation dictionaries or indexes are needed for evaluation.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn geometry_instances(
        &self,
        is_gt: bool,
        kind: IouType,
        sizes: &HashMap<i64, (u32, u32)>,
        images: &HashMap<i64, u32>,
        categories: &HashMap<i64, u32>,
        boundary_dilation: f64,
        timings: &ExtractTimings,
        opposing_groups: Option<&HashSet<(u32, u32)>>,
        use_cats: bool,
    ) -> PyResult<Instances> {
        let mut out = self.instances(is_gt, images, categories);
        out.geom = super::new_geometry(out.len(), kind);
        let mask_count = if kind == IouType::Keypoints {
            0
        } else {
            out.len()
        };
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
                out: &mut out,
                kind,
                is_gt,
                sizes,
                images,
                categories,
                chunk: Vec::with_capacity(if kind == IouType::Keypoints { 0 } else { CHUNK }),
                sender: &tx,
                timings,
                opposing_groups,
                use_cats,
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

#[derive(Deserialize)]
struct KeypointGeometry {
    #[serde(default)]
    keypoints: Option<Vec<Coordinate>>,
    #[serde(default)]
    num_keypoints: Option<Crowd>,
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
    out: &'a mut Instances,
    kind: IouType,
    is_gt: bool,
    sizes: &'a HashMap<i64, (u32, u32)>,
    images: &'a HashMap<i64, u32>,
    categories: &'a HashMap<i64, u32>,
    chunk: Vec<(RawSegm, u32, u32)>,
    sender: &'a std::sync::mpsc::SyncSender<Vec<(RawSegm, u32, u32)>>,
    timings: &'a ExtractTimings,
    opposing_groups: Option<&'a HashSet<(u32, u32)>>,
    use_cats: bool,
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
        for (index, row) in self.source.rows.iter().enumerate() {
            if !self.images.contains_key(&row.image) || !self.categories.contains_key(&row.category)
            {
                seq.next_element::<serde::de::IgnoredAny>()?;
                continue;
            }
            if self.kind == IouType::Keypoints {
                let record = seq
                    .next_element::<KeypointGeometry>()?
                    .ok_or_else(|| serde::de::Error::custom("missing keypoint annotation"))?;
                if self.is_gt {
                    let visible = match record.num_keypoints {
                        Some(Crowd::Bool(b)) => b,
                        Some(Crowd::Int(n)) => n != 0,
                        None => false,
                    };
                    self.out.ignore[selected] |= !visible;
                }
                self.out.bboxes.push(self.source.boxes[index]);
                if let GeomStore::Keypoints { data, k } = &mut self.out.geom {
                    let points = record.keypoints.unwrap_or_default();
                    if *k == 0 && !points.is_empty() {
                        *k = points.len() / 3;
                    }
                    data.extend(points.into_iter().map(|coordinate| coordinate.0));
                }
            } else {
                let record = seq
                    .next_element::<MaskGeometry>()?
                    .ok_or_else(|| serde::de::Error::custom("missing mask annotation"))?;
                let (h, w) = self.sizes.get(&row.image).copied().ok_or_else(|| {
                    serde::de::Error::custom(format!("no image entry for image_id {}", row.image))
                })?;
                let mut raw = record
                    .segmentation
                    .unwrap_or(RawSegm::FromBbox(self.source.boxes[index]));
                // Parse geometry and check image metadata even for unused masks.
                // Only the expensive rasterisation/storage is omitted. Crowds
                // and ignored annotations remain in the opposing group set.
                let group = (
                    self.images[&row.image],
                    if self.use_cats {
                        self.categories[&row.category]
                    } else {
                        0
                    },
                );
                if self
                    .opposing_groups
                    .is_some_and(|groups| !groups.contains(&group))
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
            }
            selected += 1;
        }
        Ok(())
    }
}

#[pymethods]
impl CompactBbox {
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
    let mut de = serde_json::Deserializer::from_slice(&raw);
    let parsed = if results {
        Rows(true)
            .deserialize(&mut de)
            .map(|rows| (PyDict::new(py).unbind(), rows))
    } else {
        Dataset { py }.deserialize(&mut de)
    };
    let (metadata, (rows, ground_truth, boxes)) =
        parsed.map_err(|e| PyValueError::new_err(e.to_string()))?;
    de.end().map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        metadata,
        CompactBbox {
            raw,
            rows,
            ground_truth,
            boxes: Arc::new(boxes),
            results,
        },
    ))
}
