// =============================================================================
// tree.rs
//
// the **schema tree** — `Tree` describes everything that can live in a
// checkpoint as a graph of `Group` nodes, each holding a list of named
// `Attr`s and `Dataset`s. one Schema feeds every backend (HDF5 / JSON /
// ascii tree / symbi-display table) — write + read + display walk the
// same tree, never two parallel mirror functions.
//
// nodes are the stable identity, edges are name-keyed lookups, leaves
// carry typed payloads. data lives behind a borrowing `DataRef` so building the tree
// for write is allocation-free; reads materialize into the owned `DataBuf`.
// =============================================================================

use crate::attr::Attr;

// ---- DType: the typed-array discriminant ----------------------------------

/// the array element type a Dataset carries: the primitive numeric types the
/// substrate stores (f64 fields, u64 indices).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F64,
    F32,
    I64,
    U64,
    U8,
}

impl DType {
    pub fn size_bytes(self) -> usize {
        match self {
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::F32 => 4,
            Self::U8 => 1,
        }
    }
}

// ---- DataRef / DataBuf: borrow vs owned -----------------------------------

/// borrow of a contiguous slice of one of the supported scalar types — the
/// write-side payload. holds a reference, so building a Tree for write does
/// not copy field data.
#[derive(Debug, Clone, Copy)]
pub enum DataRef<'a> {
    F64(&'a [f64]),
    F32(&'a [f32]),
    I64(&'a [i64]),
    U64(&'a [u64]),
    U8(&'a [u8]),
}

impl DataRef<'_> {
    pub fn dtype(&self) -> DType {
        match self {
            Self::F64(_) => DType::F64,
            Self::F32(_) => DType::F32,
            Self::I64(_) => DType::I64,
            Self::U64(_) => DType::U64,
            Self::U8(_) => DType::U8,
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::F64(s) => s.len(),
            Self::F32(s) => s.len(),
            Self::I64(s) => s.len(),
            Self::U64(s) => s.len(),
            Self::U8(s) => s.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// owned counterpart of `DataRef` — what the read side materializes into.
#[derive(Debug, Clone)]
pub enum DataBuf {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I64(Vec<i64>),
    U64(Vec<u64>),
    U8(Vec<u8>),
}

impl DataBuf {
    pub fn dtype(&self) -> DType {
        match self {
            Self::F64(_) => DType::F64,
            Self::F32(_) => DType::F32,
            Self::I64(_) => DType::I64,
            Self::U64(_) => DType::U64,
            Self::U8(_) => DType::U8,
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::F64(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::U8(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn as_f64(&self) -> Option<&[f64]> {
        if let Self::F64(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub fn as_u64(&self) -> Option<&[u64]> {
        if let Self::U64(v) = self {
            Some(v)
        } else {
            None
        }
    }
}

// ---- Dataset --------------------------------------------------------------

/// a named N-d array node. `shape.iter().product() == data.len()`.
#[derive(Debug)]
pub struct Dataset<'a> {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: DataRef<'a>,
}

impl<'a> Dataset<'a> {
    /// canonical constructor; checks `shape x dtype` matches `data.len()`
    /// in debug builds. shape is a Vec because rank varies (1D primitive
    /// flat arrays, 2D mesh arrays, etc.).
    pub fn new(name: impl Into<String>, shape: Vec<usize>, data: DataRef<'a>) -> Self {
        let name = name.into();
        let expected: usize = shape.iter().product();
        debug_assert_eq!(
            expected,
            data.len(),
            "Dataset '{}': shape {:?} = {} elements but data has {}",
            name,
            shape,
            expected,
            data.len(),
        );
        Self { name, shape, data }
    }
}

// ---- Tree (the schema node) -----------------------------------------------

/// a hierarchical schema node. groups carry `attrs` + `datasets` + child
/// `groups`. order-preserving so the resulting file layout is deterministic.
#[derive(Debug)]
pub struct Tree<'a> {
    pub name: String,
    pub attrs: Vec<(String, Attr)>,
    pub datasets: Vec<Dataset<'a>>,
    pub groups: Vec<Tree<'a>>,
}

impl<'a> Tree<'a> {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            attrs: Vec::new(),
            datasets: Vec::new(),
            groups: Vec::new(),
        }
    }

    /// builder: attach a typed attribute.
    pub fn with_attr(mut self, key: impl Into<String>, value: impl Into<Attr>) -> Self {
        self.attrs.push((key.into(), value.into()));
        self
    }

    /// builder: attach a dataset.
    pub fn with_dataset(mut self, ds: Dataset<'a>) -> Self {
        self.datasets.push(ds);
        self
    }

    /// builder: attach a child group.
    pub fn with_group(mut self, group: Tree<'a>) -> Self {
        self.groups.push(group);
        self
    }

    /// in-place push: attribute.
    pub fn push_attr(&mut self, key: impl Into<String>, value: impl Into<Attr>) {
        self.attrs.push((key.into(), value.into()));
    }
    /// in-place push: dataset.
    pub fn push_dataset(&mut self, ds: Dataset<'a>) {
        self.datasets.push(ds);
    }
    /// in-place push: child group.
    pub fn push_group(&mut self, group: Tree<'a>) {
        self.groups.push(group);
    }

    pub fn find_group(&self, name: &str) -> Option<&Tree<'a>> {
        self.groups.iter().find(|g| g.name == name)
    }
    pub fn find_attr(&self, name: &str) -> Option<&Attr> {
        self.attrs
            .iter()
            .find_map(|(k, v)| (k == name).then_some(v))
    }
    pub fn find_dataset(&self, name: &str) -> Option<&Dataset<'a>> {
        self.datasets.iter().find(|d| d.name == name)
    }
}

// ---- TreeBuf — the owned tree the read side materializes into ------------

/// owned counterpart of `Tree` — what the read side returns. groups carry
/// owned attrs + owned datasets (Vec<f64> etc.).
#[derive(Debug)]
pub struct DatasetBuf {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: DataBuf,
}

#[derive(Debug, Default)]
pub struct TreeBuf {
    pub name: String,
    pub attrs: Vec<(String, Attr)>,
    pub datasets: Vec<DatasetBuf>,
    pub groups: Vec<TreeBuf>,
}

impl TreeBuf {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    pub fn find_group(&self, name: &str) -> Option<&TreeBuf> {
        self.groups.iter().find(|g| g.name == name)
    }
    pub fn find_attr(&self, name: &str) -> Option<&Attr> {
        self.attrs
            .iter()
            .find_map(|(k, v)| (k == name).then_some(v))
    }
    pub fn find_dataset(&self, name: &str) -> Option<&DatasetBuf> {
        self.datasets.iter().find(|d| d.name == name)
    }
}
