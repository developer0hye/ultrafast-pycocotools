//! Immutable file snapshots with native bbox columns and lazy Python annotations.
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::de::{DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;
use ufcoco_core::eval::{GeomStore, Instances};

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
