// =============================================================================
// hdf5.rs
//
// `Hdf5Backend` — production checkpoint backend. translates a borrowed Tree
// into hdf5-metno API calls. produces the canonical on-disk layout, so
// existing checkpoint files + every plot script (`scripts/plot_*.py`) read
// unchanged.
//
// the implementation is intentionally a SINGLE-PASS recursive walk: for each
// group node, create the HDF5 group, drain its attrs + datasets + children.
// no parallel reader/writer code — both directions share the same Tree shape.
// =============================================================================

use std::path::Path;

use crate::attr::Attr;
use crate::backend::IoBackend;
use crate::error::{IoError, Result};
use crate::tree::{DataBuf, DataRef, Dataset, DatasetBuf, Tree, TreeBuf};

/// the production HDF5 backend.
pub struct Hdf5Backend;

impl IoBackend for Hdf5Backend {
    fn write(&self, path: &Path, tree: &Tree<'_>) -> Result<()> {
        let file = hdf5_metno::File::create(path)
            .map_err(|e| IoError::Backend(format!("create file {path:?}: {e}")))?;
        // root-level attrs (e.g., format_version) and child groups
        write_group_attrs(&file, &tree.attrs)?;
        for ds in &tree.datasets {
            write_dataset(&FileOrGroup::File(&file), ds)?;
        }
        for sub in &tree.groups {
            write_subtree(&file, sub)?;
        }
        Ok(())
    }

    fn read(&self, path: &Path) -> Result<TreeBuf> {
        let file = hdf5_metno::File::open(path)
            .map_err(|e| IoError::Backend(format!("open file {path:?}: {e}")))?;
        let mut root = TreeBuf::new("");
        read_group_attrs(&FileOrGroupRead::File(&file), &mut root.attrs)?;
        // root datasets (rare but supported for completeness)
        for ds_name in list_datasets(&FileOrGroupRead::File(&file))? {
            let ds = read_dataset(&FileOrGroupRead::File(&file), &ds_name)?;
            root.datasets.push(ds);
        }
        // walk children
        for grp_name in list_groups(&FileOrGroupRead::File(&file))? {
            let child = read_subtree(&file, &grp_name)?;
            root.groups.push(child);
        }
        Ok(root)
    }
}

impl Hdf5Backend {
    /// length of the first axis of a root dataset, WITHOUT reading the data — the
    /// row count that drives a chunked read loop.
    pub fn dataset_len(&self, path: &Path, name: &str) -> Result<usize> {
        let file = hdf5_metno::File::open(path)
            .map_err(|e| IoError::Backend(format!("open file {path:?}: {e}")))?;
        let ds = FileOrGroupRead::File(&file).dataset(name)?;
        Ok(ds.shape().first().copied().unwrap_or(0))
    }

    /// read the rows `[start, start + count)` of EVERY root dataset (a flat-table / SoA
    /// slice), plus all root attrs, into a TreeBuf — without ever materializing the full
    /// columns. `start`/`count` are clamped to each dataset's length, so the last chunk is
    /// naturally short and an out-of-range start yields empty columns. subgroups are NOT
    /// walked (the photon catalog is a flat root table). this is the bounded-memory
    /// counterpart to `read`: reduce a huge events file chunk-by-chunk, discarding each slice.
    pub fn read_root_slice(&self, path: &Path, start: usize, count: usize) -> Result<TreeBuf> {
        let file = hdf5_metno::File::open(path)
            .map_err(|e| IoError::Backend(format!("open file {path:?}: {e}")))?;
        let mut root = TreeBuf::new("");
        read_group_attrs(&FileOrGroupRead::File(&file), &mut root.attrs)?;
        for ds_name in list_datasets(&FileOrGroupRead::File(&file))? {
            let ds = read_dataset_slice(&FileOrGroupRead::File(&file), &ds_name, start, count)?;
            root.datasets.push(ds);
        }
        Ok(root)
    }
}

// ----- adapter for writing attrs/datasets to either a File or a Group --

#[allow(dead_code)] // `create_group` reserved for future per-write group helper
enum FileOrGroup<'a> {
    File(&'a hdf5_metno::File),
    Group(&'a hdf5_metno::Group),
}

impl<'a> FileOrGroup<'a> {
    fn create_dataset<T: hdf5_metno::H5Type>(
        &self,
        name: &str,
        shape: &[usize],
    ) -> Result<hdf5_metno::Dataset> {
        let builder = match self {
            Self::File(f) => f.new_dataset::<T>(),
            Self::Group(g) => g.new_dataset::<T>(),
        };
        builder
            .shape(shape)
            .create(name)
            .map_err(|e| IoError::Backend(format!("create dataset '{name}': {e}")))
    }
    #[allow(dead_code)] // part of the FileOrGroup API surface; kept for parity with `group()` below.
    fn create_group(&self, name: &str) -> Result<hdf5_metno::Group> {
        let r = match self {
            Self::File(f) => f.create_group(name),
            Self::Group(g) => g.create_group(name),
        };
        r.map_err(|e| IoError::Backend(format!("create group '{name}': {e}")))
    }
    fn new_attr_bool(&self, name: &str) -> Result<hdf5_metno::Attribute> {
        let r = match self {
            Self::File(f) => f.new_attr::<u8>().create(name),
            Self::Group(g) => g.new_attr::<u8>().create(name),
        };
        r.map_err(|e| IoError::Backend(format!("create attr '{name}': {e}")))
    }
    fn new_attr_i64(&self, name: &str) -> Result<hdf5_metno::Attribute> {
        let r = match self {
            Self::File(f) => f.new_attr::<i64>().create(name),
            Self::Group(g) => g.new_attr::<i64>().create(name),
        };
        r.map_err(|e| IoError::Backend(format!("create attr '{name}': {e}")))
    }
    fn new_attr_u64(&self, name: &str) -> Result<hdf5_metno::Attribute> {
        let r = match self {
            Self::File(f) => f.new_attr::<u64>().create(name),
            Self::Group(g) => g.new_attr::<u64>().create(name),
        };
        r.map_err(|e| IoError::Backend(format!("create attr '{name}': {e}")))
    }
    fn new_attr_f64(&self, name: &str) -> Result<hdf5_metno::Attribute> {
        let r = match self {
            Self::File(f) => f.new_attr::<f64>().create(name),
            Self::Group(g) => g.new_attr::<f64>().create(name),
        };
        r.map_err(|e| IoError::Backend(format!("create attr '{name}': {e}")))
    }
    fn new_attr_str(&self, name: &str) -> Result<hdf5_metno::Attribute> {
        // string metadata rides as a variable-length unicode HDF5 ATTRIBUTE,
        // matching the frozen v2.0 python reader contract (`meta_group.attrs[..]`
        // decoded via `decode_str`).
        use hdf5_metno::types::VarLenUnicode;
        let r = match self {
            Self::File(f) => f.new_attr::<VarLenUnicode>().create(name),
            Self::Group(g) => g.new_attr::<VarLenUnicode>().create(name),
        };
        r.map_err(|e| IoError::Backend(format!("create str attr '{name}': {e}")))
    }
}

fn write_group_attrs(file: &hdf5_metno::File, attrs: &[(String, Attr)]) -> Result<()> {
    write_attrs(&FileOrGroup::File(file), attrs)
}

fn write_attrs(target: &FileOrGroup<'_>, attrs: &[(String, Attr)]) -> Result<()> {
    for (name, value) in attrs {
        match value {
            Attr::Bool(v) => target
                .new_attr_bool(name)?
                .write_scalar(&(if *v { 1u8 } else { 0u8 }))
                .map_err(|e| IoError::Backend(format!("write bool attr '{name}': {e}")))?,
            Attr::I64(v) => target
                .new_attr_i64(name)?
                .write_scalar(v)
                .map_err(|e| IoError::Backend(format!("write i64 attr '{name}': {e}")))?,
            Attr::U64(v) => target
                .new_attr_u64(name)?
                .write_scalar(v)
                .map_err(|e| IoError::Backend(format!("write u64 attr '{name}': {e}")))?,
            Attr::F64(v) => target
                .new_attr_f64(name)?
                .write_scalar(v)
                .map_err(|e| IoError::Backend(format!("write f64 attr '{name}': {e}")))?,
            Attr::Str(s) => {
                use hdf5_metno::types::VarLenUnicode;
                let v: VarLenUnicode = s
                    .parse()
                    .map_err(|e| IoError::Backend(format!("encode str attr '{name}': {e}")))?;
                target
                    .new_attr_str(name)?
                    .write_scalar(&v)
                    .map_err(|e| IoError::Backend(format!("write str attr '{name}': {e}")))?;
            }
        }
    }
    Ok(())
}

fn write_dataset(target: &FileOrGroup<'_>, ds: &Dataset<'_>) -> Result<()> {
    match ds.data {
        DataRef::F64(d) => target
            .create_dataset::<f64>(&ds.name, &ds.shape)?
            .write_raw(d)
            .map_err(|e| IoError::Backend(format!("write f64 ds '{}': {e}", ds.name))),
        DataRef::F32(d) => target
            .create_dataset::<f32>(&ds.name, &ds.shape)?
            .write_raw(d)
            .map_err(|e| IoError::Backend(format!("write f32 ds '{}': {e}", ds.name))),
        DataRef::I64(d) => target
            .create_dataset::<i64>(&ds.name, &ds.shape)?
            .write_raw(d)
            .map_err(|e| IoError::Backend(format!("write i64 ds '{}': {e}", ds.name))),
        DataRef::U64(d) => target
            .create_dataset::<u64>(&ds.name, &ds.shape)?
            .write_raw(d)
            .map_err(|e| IoError::Backend(format!("write u64 ds '{}': {e}", ds.name))),
        DataRef::U8(d) => target
            .create_dataset::<u8>(&ds.name, &ds.shape)?
            .write_raw(d)
            .map_err(|e| IoError::Backend(format!("write u8 ds '{}': {e}", ds.name))),
    }
}

fn write_subtree(parent: &hdf5_metno::File, sub: &Tree<'_>) -> Result<()> {
    let grp = parent
        .create_group(&sub.name)
        .map_err(|e| IoError::Backend(format!("create group '{}': {e}", sub.name)))?;
    write_subtree_into(&grp, sub)
}

fn write_subtree_into(grp: &hdf5_metno::Group, sub: &Tree<'_>) -> Result<()> {
    write_attrs(&FileOrGroup::Group(grp), &sub.attrs)?;
    for ds in &sub.datasets {
        write_dataset(&FileOrGroup::Group(grp), ds)?;
    }
    for child in &sub.groups {
        let child_grp = grp
            .create_group(&child.name)
            .map_err(|e| IoError::Backend(format!("create group '{}': {e}", child.name)))?;
        write_subtree_into(&child_grp, child)?;
    }
    Ok(())
}

// ----- READ side -----------------------------------------------------------

#[allow(dead_code)] // `group` reserved for descend helpers
enum FileOrGroupRead<'a> {
    File(&'a hdf5_metno::File),
    Group(&'a hdf5_metno::Group),
}

impl<'a> FileOrGroupRead<'a> {
    fn attr_names(&self) -> Result<Vec<String>> {
        Ok(match self {
            Self::File(f) => f.attr_names(),
            Self::Group(g) => g.attr_names(),
        }
        .map_err(|e| IoError::Backend(format!("list attrs: {e}")))?)
    }
    fn member_names(&self) -> Result<Vec<String>> {
        Ok(match self {
            Self::File(f) => f.member_names(),
            Self::Group(g) => g.member_names(),
        }
        .map_err(|e| IoError::Backend(format!("list members: {e}")))?)
    }
    fn attr(&self, name: &str) -> Result<hdf5_metno::Attribute> {
        match self {
            Self::File(f) => f.attr(name),
            Self::Group(g) => g.attr(name),
        }
        .map_err(|_| IoError::MissingPath(name.into()))
    }
    fn dataset(&self, name: &str) -> Result<hdf5_metno::Dataset> {
        match self {
            Self::File(f) => f.dataset(name),
            Self::Group(g) => g.dataset(name),
        }
        .map_err(|_| IoError::MissingPath(name.into()))
    }
    #[allow(dead_code)] // part of the FileOrGroupRead API surface; reserved for nested-group reads.
    fn group(&self, name: &str) -> Result<hdf5_metno::Group> {
        match self {
            Self::File(f) => f.group(name),
            Self::Group(g) => g.group(name),
        }
        .map_err(|_| IoError::MissingPath(name.into()))
    }
    fn is_group(&self, name: &str) -> bool {
        match self {
            Self::File(f) => f.group(name).is_ok(),
            Self::Group(g) => g.group(name).is_ok(),
        }
    }
    fn is_dataset(&self, name: &str) -> bool {
        match self {
            Self::File(f) => f.dataset(name).is_ok(),
            Self::Group(g) => g.dataset(name).is_ok(),
        }
    }
}

fn read_group_attrs(src: &FileOrGroupRead<'_>, out: &mut Vec<(String, Attr)>) -> Result<()> {
    use hdf5_metno::types::{FloatSize, IntSize, TypeDescriptor};
    for name in src.attr_names()? {
        let attr = src.attr(&name)?;
        let dt = attr
            .dtype()
            .and_then(|d| d.to_descriptor())
            .map_err(|e| IoError::Backend(format!("dtype probe '{name}': {e}")))?;
        let parsed = match dt {
            TypeDescriptor::Float(FloatSize::U8) => attr.read_scalar::<f64>().map(Attr::F64),
            TypeDescriptor::Float(FloatSize::U4) => {
                attr.read_scalar::<f32>().map(|v| Attr::F64(v as f64))
            }
            TypeDescriptor::Integer(IntSize::U8) => attr.read_scalar::<i64>().map(Attr::I64),
            TypeDescriptor::Integer(IntSize::U4) => {
                attr.read_scalar::<i32>().map(|v| Attr::I64(v as i64))
            }
            TypeDescriptor::Unsigned(IntSize::U8) => attr.read_scalar::<u64>().map(Attr::U64),
            TypeDescriptor::Unsigned(IntSize::U4) => {
                attr.read_scalar::<u32>().map(|v| Attr::U64(v as u64))
            }
            // on-disk convention: bool attrs ride as u8.
            TypeDescriptor::Unsigned(IntSize::U1) => {
                attr.read_scalar::<u8>().map(|v| Attr::Bool(v != 0))
            }
            TypeDescriptor::Boolean => attr.read_scalar::<bool>().map(Attr::Bool),
            // variable-length string attrs (regime / coord_system / timestepping / ...).
            TypeDescriptor::VarLenUnicode => attr
                .read_scalar::<hdf5_metno::types::VarLenUnicode>()
                .map(|v| Attr::Str(v.to_string())),
            TypeDescriptor::VarLenAscii => attr
                .read_scalar::<hdf5_metno::types::VarLenAscii>()
                .map(|v| Attr::Str(v.to_string())),
            other => {
                return Err(IoError::Backend(format!(
                    "unsupported attr type at '{name}': {other:?}"
                )));
            }
        }
        .map_err(|e| IoError::Backend(format!("read attr '{name}': {e}")))?;
        out.push((name, parsed));
    }
    Ok(())
}

fn list_datasets(src: &FileOrGroupRead<'_>) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for name in src.member_names()? {
        if src.is_dataset(&name) && !src.is_group(&name) {
            out.push(name);
        }
    }
    Ok(out)
}

fn list_groups(src: &FileOrGroupRead<'_>) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for name in src.member_names()? {
        if src.is_group(&name) {
            out.push(name);
        }
    }
    Ok(out)
}

fn read_dataset(src: &FileOrGroupRead<'_>, name: &str) -> Result<DatasetBuf> {
    use hdf5_metno::types::{FloatSize, IntSize, TypeDescriptor};
    let ds = src.dataset(name)?;
    let shape = ds.shape();
    let dt = ds
        .dtype()
        .and_then(|d| d.to_descriptor())
        .map_err(|e| IoError::Backend(format!("dtype probe '{name}': {e}")))?;
    let data = match dt {
        TypeDescriptor::Float(FloatSize::U8) => DataBuf::F64(
            ds.read_raw::<f64>()
                .map_err(|e| IoError::Backend(format!("read f64 ds '{name}': {e}")))?,
        ),
        TypeDescriptor::Float(FloatSize::U4) => DataBuf::F32(
            ds.read_raw::<f32>()
                .map_err(|e| IoError::Backend(format!("read f32 ds '{name}': {e}")))?,
        ),
        TypeDescriptor::Integer(IntSize::U8) => DataBuf::I64(
            ds.read_raw::<i64>()
                .map_err(|e| IoError::Backend(format!("read i64 ds '{name}': {e}")))?,
        ),
        TypeDescriptor::Unsigned(IntSize::U8) => DataBuf::U64(
            ds.read_raw::<u64>()
                .map_err(|e| IoError::Backend(format!("read u64 ds '{name}': {e}")))?,
        ),
        TypeDescriptor::Unsigned(IntSize::U1) => DataBuf::U8(
            ds.read_raw::<u8>()
                .map_err(|e| IoError::Backend(format!("read u8 ds '{name}': {e}")))?,
        ),
        other => {
            return Err(IoError::Backend(format!(
                "unsupported dataset type at '{name}': {other:?}"
            )));
        }
    };
    Ok(DatasetBuf {
        name: name.into(),
        shape,
        data,
    })
}

/// read only rows `[start, start + count)` of a 1D root dataset via an HDF5 hyperslab,
/// clamped to the dataset length. mirrors `read_dataset`'s dtype dispatch but reads a
/// hyperslab slice, so memory is O(count) in the requested rows. 2D+ datasets are rejected
/// (the catalog columns are all 1D).
fn read_dataset_slice(
    src: &FileOrGroupRead<'_>,
    name: &str,
    start: usize,
    count: usize,
) -> Result<DatasetBuf> {
    use hdf5_metno::types::{FloatSize, IntSize, TypeDescriptor};
    let ds = src.dataset(name)?;
    let full = ds.shape();
    if full.len() != 1 {
        return Err(IoError::Backend(format!(
            "read_dataset_slice '{name}': expected 1D dataset, got shape {full:?}"
        )));
    }
    let len = full[0];
    let s = start.min(len);
    let c = count.min(len - s);
    let sel = s..s + c; // From<Range<usize>> for hdf5 Selection
    let slice = |what: &str| format!("slice {what} ds '{name}' [{s}..{}]", s + c);
    let dt = ds
        .dtype()
        .and_then(|d| d.to_descriptor())
        .map_err(|e| IoError::Backend(format!("dtype probe '{name}': {e}")))?;
    let data = match dt {
        TypeDescriptor::Float(FloatSize::U8) => DataBuf::F64(
            ds.read_slice_1d::<f64, _>(sel)
                .map_err(|e| IoError::Backend(format!("{}: {e}", slice("f64"))))?
                .to_vec(),
        ),
        TypeDescriptor::Float(FloatSize::U4) => DataBuf::F32(
            ds.read_slice_1d::<f32, _>(sel)
                .map_err(|e| IoError::Backend(format!("{}: {e}", slice("f32"))))?
                .to_vec(),
        ),
        TypeDescriptor::Integer(IntSize::U8) => DataBuf::I64(
            ds.read_slice_1d::<i64, _>(sel)
                .map_err(|e| IoError::Backend(format!("{}: {e}", slice("i64"))))?
                .to_vec(),
        ),
        TypeDescriptor::Unsigned(IntSize::U8) => DataBuf::U64(
            ds.read_slice_1d::<u64, _>(sel)
                .map_err(|e| IoError::Backend(format!("{}: {e}", slice("u64"))))?
                .to_vec(),
        ),
        TypeDescriptor::Unsigned(IntSize::U1) => DataBuf::U8(
            ds.read_slice_1d::<u8, _>(sel)
                .map_err(|e| IoError::Backend(format!("{}: {e}", slice("u8"))))?
                .to_vec(),
        ),
        other => {
            return Err(IoError::Backend(format!(
                "unsupported dataset type at '{name}': {other:?}"
            )));
        }
    };
    Ok(DatasetBuf {
        name: name.into(),
        shape: vec![c],
        data,
    })
}

fn read_subtree(parent: &hdf5_metno::File, name: &str) -> Result<TreeBuf> {
    let grp = parent
        .group(name)
        .map_err(|_| IoError::MissingPath(name.into()))?;
    read_subtree_into(&grp, name)
}

fn read_subtree_into(grp: &hdf5_metno::Group, name: &str) -> Result<TreeBuf> {
    let mut out = TreeBuf::new(name);
    read_group_attrs(&FileOrGroupRead::Group(grp), &mut out.attrs)?;
    for ds_name in list_datasets(&FileOrGroupRead::Group(grp))? {
        out.datasets
            .push(read_dataset(&FileOrGroupRead::Group(grp), &ds_name)?);
    }
    for child_name in list_groups(&FileOrGroupRead::Group(grp))? {
        let child_grp = grp
            .group(&child_name)
            .map_err(|_| IoError::MissingPath(child_name.clone()))?;
        out.groups.push(read_subtree_into(&child_grp, &child_name)?);
    }
    Ok(out)
}
