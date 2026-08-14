//! A JSON reader that builds Python objects directly.
//!
//! Why bother, when `json.load` is already C? Because on a large annotation
//! file the parse is the biggest single item on the clock — Objects365 val is
//! 269 MB and `json.load` spends 4.1 s on it, more than the entire evaluation
//! — and because `json` builds an intermediate view of every number and string
//! that we can skip.
//!
//! The parse feeds a serde visitor that constructs `PyDict` / `PyList` /
//! `PyFloat` as it goes: no `serde_json::Value` tree is ever built, so peak
//! memory is the mapped file plus the Python objects, not both plus a Rust
//! copy.
//!
//! **Float parsing has to agree with CPython bit for bit.** An annotation's
//! `area` decides whether it counts as small/medium/large, so a one-ULP
//! difference on a value sitting exactly on 32^2 would move AP_small. Both
//! sides are correctly rounded (Rust's `dec2flt`, CPython's `_Py_dg_strtod`),
//! so they agree by construction — and `tests/test_json_loader.py` checks it
//! against a real annotation file rather than trusting that argument.

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use serde::de::{Deserialize, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use std::collections::HashMap;
use std::fmt;

/// Interning table for object keys.
///
/// A COCO file repeats `"image_id"`, `"bbox"`, `"category_id"` once per
/// annotation — millions of times. Without a memo we would allocate a fresh
/// `str` object for each, which costs more memory than the values do.
/// CPython's `json` interns keys the same way.
type Memo = HashMap<String, Py<PyString>>;

struct Builder<'a, 'py> {
    py: Python<'py>,
    memo: &'a mut Memo,
}

impl<'a, 'py> Builder<'a, 'py> {
    fn key(&mut self, s: &str) -> Py<PyString> {
        if let Some(k) = self.memo.get(s) {
            return k.clone_ref(self.py);
        }
        let k: Py<PyString> = PyString::new(self.py, s).unbind();
        self.memo.insert(s.to_owned(), k.clone_ref(self.py));
        k
    }
}

impl<'de, 'a, 'py> DeserializeSeed<'de> for Builder<'a, 'py> {
    type Value = Py<PyAny>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(self)
    }
}

impl<'de, 'a, 'py> Visitor<'de> for Builder<'a, 'py> {
    type Value = Py<PyAny>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("any JSON value")
    }

    fn visit_bool<E>(self, v: bool) -> Result<Self::Value, E> {
        Ok(pyo3::types::PyBool::new(self.py, v)
            .to_owned()
            .into_any()
            .unbind())
    }
    fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        v.into_pyobject(self.py)
            .map(|b| b.into_any().unbind())
            .map_err(|e| E::custom(format!("{e:?}")))
    }
    fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        v.into_pyobject(self.py)
            .map(|b| b.into_any().unbind())
            .map_err(|e| E::custom(format!("{e:?}")))
    }
    fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        v.into_pyobject(self.py)
            .map(|b| b.into_any().unbind())
            .map_err(|e| E::custom(format!("{e:?}")))
    }
    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
        Ok(PyString::new(self.py, v).into_any().unbind())
    }
    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(self.py.None())
    }
    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(self.py.None())
    }
    fn visit_some<D>(self, d: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        d.deserialize_any(self)
    }

    fn visit_seq<A>(mut self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        // Appended one at a time on purpose. Gathering into a `Vec` first and
        // handing `PyList::new` the finished length looks strictly better — it
        // replaces CPython's ~90 growth steps on the 431k-element `annotations`
        // array with a single right-sized allocation — and it measured *slower*
        // on every file tried (163 MB: 0.789s -> 0.818s; 269 MB: 1.744s ->
        // 1.813s). The outer array is one array; the small ones are millions.
        // Every `bbox` and every `counts` would pay a heap allocation for its
        // Vec and then a copy out of it, and 862k extra allocations cost more
        // than 90 reallocations save.
        let list = PyList::empty(self.py);
        while let Some(item) = seq.next_element_seed(Builder {
            py: self.py,
            memo: self.memo,
        })? {
            list.append(item)
                .map_err(|e| serde::de::Error::custom(format!("{e:?}")))?;
        }
        // Reborrow to satisfy the borrow checker on the loop above.
        let _ = &mut self;
        Ok(list.into_any().unbind())
    }

    fn visit_map<A>(mut self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let dict = PyDict::new(self.py);
        while let Some(k) = map.next_key::<std::borrow::Cow<str>>()? {
            let key = self.key(&k);
            let value = map.next_value_seed(Builder {
                py: self.py,
                memo: self.memo,
            })?;
            dict.set_item(key, value)
                .map_err(|e| serde::de::Error::custom(format!("{e:?}")))?;
        }
        Ok(dict.into_any().unbind())
    }
}

/// Read and parse the file, building nothing. Benchmarking only.
///
/// Splits the loader's cost in two. Everything this does — the read, the
/// tokenizer, the number and string parsing — `load_json` also does; what it
/// skips is allocating the `PyDict`/`PyList`/`PyFloat` objects and hashing the
/// keys. The difference between the two says which half to attack, and the
/// answer decided against a SIMD tokenizer: see DESIGN.md.
#[pyfunction]
pub fn parse_json_only(py: Python<'_>, path: &str) -> PyResult<usize> {
    let bytes = std::fs::read(path).map_err(|e| PyIOError::new_err(e.to_string()))?;
    let n = bytes.len();
    py.detach(|| {
        let mut de = serde_json::Deserializer::from_slice(&bytes);
        serde::de::IgnoredAny::deserialize(&mut de)
            .map_err(|e| PyValueError::new_err(format!("{path}: {e}")))?;
        de.end()
            .map_err(|e| PyValueError::new_err(format!("{path}: trailing data: {e}")))
    })?;
    Ok(n)
}

/// Read the file and nothing else. Benchmarking only: the floor both of the
/// above sit on, so neither gets credited with the disk.
#[pyfunction]
pub fn read_file_only(path: &str) -> PyResult<usize> {
    std::fs::read(path)
        .map(|b| b.len())
        .map_err(|e| PyIOError::new_err(e.to_string()))
}

/// Parse a JSON file into Python objects.
///
/// Equivalent to `json.load(open(path))` for the subset JSON actually
/// contains; duplicate object keys keep the last value, as CPython does.
#[pyfunction]
pub fn load_json(py: Python<'_>, path: &str) -> PyResult<Py<PyAny>> {
    let bytes = std::fs::read(path).map_err(|e| PyIOError::new_err(e.to_string()))?;
    let mut memo: Memo = HashMap::with_capacity(64);
    let mut de = serde_json::Deserializer::from_slice(&bytes);
    // COCO annotation files nest only a few levels, but a hostile file could
    // nest deeply enough to blow the stack; serde_json's default limit
    // (128) is what protects us and is left in place.
    let seed = Builder {
        py,
        memo: &mut memo,
    };
    let value = seed
        .deserialize(&mut de)
        .map_err(|e| PyValueError::new_err(format!("{path}: {e}")))?;
    de.end()
        .map_err(|e| PyValueError::new_err(format!("{path}: trailing data: {e}")))?;
    Ok(value)
}
