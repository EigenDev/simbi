// =============================================================================
// stream.rs
//
// an HDF5 checkpoint file under construction. small groups (metadata, mesh, bodies) land as
// trees; large field datasets are declared once at their global shape and filled slab by slab
// from bounded host buffers, so a partitioned run writes each tile's owned block straight into
// the global dataset and never assembles a global array. the file is built under a temporary
// name and takes its final name by rename after every write and the close succeeded, so a
// reader only ever sees complete files. `read_slab` is the matching bounded read for a restart
// that reconstructs a tile from any partition of the same global grid.
//
// usage:
//   let stream = Hdf5Stream::create(Path::new("run.chkpt.h5"))?;
//   stream.write_tree(&metadata_tree)?;
//   stream.declare_f64("level_0/partition_0/hydro/primitives", "rho", &[nz, ny, nx])?;
//   stream.write_slab("level_0/partition_0/hydro/primitives/rho", &[z0, y0, x0], &[dz, dy, dx], &buf)?;
//   stream.publish()?;
// =============================================================================
use std::path::{Path, PathBuf};

use crate::error::{IoError, Result};
use crate::hdf5::{FileOrGroup, write_dataset, write_group_attrs, write_subtree};
use crate::tree::Tree;
use hdf5_metno::{Hyperslab, Selection, SliceOrIndex};

pub struct Hdf5Stream {
    file: Option<hdf5_metno::File>,
    temporary: PathBuf,
    target: PathBuf,
}

/// the temporary name a file is built under: a dotfile beside the target carrying the writer's
/// process id, which every checkpoint glob skips and a crashed writer leaves behind harmlessly.
pub fn temporary_name(target: &Path) -> Result<PathBuf> {
    let file_name = target.file_name().ok_or_else(|| {
        IoError::MissingPath(format!("checkpoint path has no file name: {target:?}"))
    })?;
    Ok(target.with_file_name(format!(
        ".{}.tmp.{}",
        file_name.to_string_lossy(),
        std::process::id()
    )))
}

impl Hdf5Stream {
    /// open the file under its temporary name; a stale temporary from an earlier writer is replaced.
    pub fn create(target: &Path) -> Result<Self> {
        let temporary = temporary_name(target)?;
        if temporary.exists() {
            std::fs::remove_file(&temporary)?;
        }
        let file = hdf5_metno::File::create(&temporary)
            .map_err(|e| IoError::Backend(format!("create file {temporary:?}: {e}")))?;
        Ok(Self {
            file: Some(file),
            temporary,
            target: target.to_path_buf(),
        })
    }

    fn file(&self) -> &hdf5_metno::File {
        self.file.as_ref().expect("the stream is open until publish")
    }

    /// write a tree at the root: its attributes, its datasets and its groups, exactly as the
    /// whole-tree backend does.
    pub fn write_tree(&self, tree: &Tree<'_>) -> Result<()> {
        let file = self.file();
        write_group_attrs(file, &tree.attrs)?;
        for ds in &tree.datasets {
            write_dataset(&FileOrGroup::File(file), ds)?;
        }
        for sub in &tree.groups {
            write_subtree(file, sub)?;
        }
        Ok(())
    }

    /// the group at `path` (segments separated by `/`), created along the way when absent.
    fn group(&self, path: &str) -> Result<hdf5_metno::Group> {
        let file = self.file();
        let mut current: Option<hdf5_metno::Group> = None;
        for segment in path.split('/').filter(|s| !s.is_empty()) {
            let next = match &current {
                None => match file.group(segment) {
                    Ok(g) => g,
                    Err(_) => file
                        .create_group(segment)
                        .map_err(|e| IoError::Backend(format!("create group '{segment}': {e}")))?,
                },
                Some(parent) => match parent.group(segment) {
                    Ok(g) => g,
                    Err(_) => parent
                        .create_group(segment)
                        .map_err(|e| IoError::Backend(format!("create group '{segment}': {e}")))?,
                },
            };
            current = Some(next);
        }
        current.ok_or_else(|| IoError::MissingPath(format!("empty group path '{path}'")))
    }

    /// declare an f64 dataset `name` in the group at `group_path` with the global `shape`;
    /// its contents arrive through `write_slab`.
    pub fn declare_f64(&self, group_path: &str, name: &str, shape: &[usize]) -> Result<()> {
        let group = self.group(group_path)?;
        group
            .new_dataset::<f64>()
            .shape(shape)
            .create(name)
            .map(|_| ())
            .map_err(|e| IoError::Backend(format!("declare dataset '{group_path}/{name}': {e}")))
    }

    /// write `data`, laid out in row-major order over `count`, into the block of the dataset at
    /// `dataset_path` that starts at `start`.
    pub fn write_slab(
        &self,
        dataset_path: &str,
        start: &[usize],
        count: &[usize],
        data: &[f64],
    ) -> Result<()> {
        let volume: usize = count.iter().product();
        if data.len() != volume {
            return Err(IoError::ShapeMismatch {
                path: dataset_path.into(),
                expected: vec![volume],
                actual: vec![data.len()],
            });
        }
        if volume == 0 {
            return Ok(());
        }
        let ds = self
            .file()
            .dataset(dataset_path)
            .map_err(|e| IoError::Backend(format!("open dataset '{dataset_path}': {e}")))?;
        let view = ndarray::ArrayViewD::from_shape(ndarray::IxDyn(count), data).map_err(|e| {
            IoError::Backend(format!("slab view for '{dataset_path}' over {count:?}: {e}"))
        })?;
        ds.write_slice(view, slab_selection(start, count))
            .map_err(|e| IoError::Backend(format!("write slab of '{dataset_path}' at {start:?} x {count:?}: {e}")))
    }

    /// close the file and give it its final name. the temporary is removed on any failure.
    pub fn publish(mut self) -> Result<PathBuf> {
        let file = self.file.take().expect("the stream is open until publish");
        if let Err(e) = file.close() {
            let _ = std::fs::remove_file(&self.temporary);
            return Err(IoError::Backend(format!("close {:?}: {e}", self.temporary)));
        }
        if let Err(error) = std::fs::rename(&self.temporary, &self.target) {
            let _ = std::fs::remove_file(&self.temporary);
            return Err(error.into());
        }
        Ok(self.target.clone())
    }
}

impl Drop for Hdf5Stream {
    /// a stream dropped before `publish` leaves no file behind.
    fn drop(&mut self) {
        if let Some(file) = self.file.take() {
            let _ = file.close();
            let _ = std::fs::remove_file(&self.temporary);
        }
    }
}

fn slab_selection(start: &[usize], count: &[usize]) -> Selection {
    let slices: Vec<SliceOrIndex> = start
        .iter()
        .zip(count)
        .map(|(&s, &c)| SliceOrIndex::SliceCount {
            start: s,
            step: 1,
            count: c,
            block: 1,
        })
        .collect();
    Selection::from(Hyperslab::from(slices))
}

/// read the block of the f64 dataset at `dataset_path` in `path` that starts at `start` and
/// spans `count`, in row-major order over `count`.
pub fn read_slab(path: &Path, dataset_path: &str, start: &[usize], count: &[usize]) -> Result<Vec<f64>> {
    let volume: usize = count.iter().product();
    if volume == 0 {
        return Ok(Vec::new());
    }
    let file = hdf5_metno::File::open(path)
        .map_err(|e| IoError::Backend(format!("open file {path:?}: {e}")))?;
    let ds = file
        .dataset(dataset_path)
        .map_err(|e| IoError::Backend(format!("open dataset '{dataset_path}': {e}")))?;
    let shape = ds.shape();
    for (ax, (&s, &c)) in start.iter().zip(count).enumerate() {
        if s + c > shape[ax] {
            return Err(IoError::ShapeMismatch {
                path: dataset_path.into(),
                expected: shape.clone(),
                actual: start.iter().zip(count).map(|(s, c)| s + c).collect(),
            });
        }
    }
    let arr = ds
        .read_slice::<f64, _, ndarray::IxDyn>(slab_selection(start, count))
        .map_err(|e| IoError::Backend(format!("read slab of '{dataset_path}' at {start:?} x {count:?}: {e}")))?;
    Ok(arr.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::Attr;
    use crate::hdf5::Hdf5Backend;
    use crate::backend::IoBackend;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("symbi_io_stream_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn slabs_assemble_the_global_dataset_and_the_file_publishes_by_rename() {
        let dir = scratch("assemble");
        let target = dir.join("run.h5");
        let stream = Hdf5Stream::create(&target).unwrap();
        stream
            .write_tree(&Tree::new("").with_attr("format_version", Attr::Str("test".into())))
            .unwrap();
        stream.declare_f64("level_0/hydro", "rho", &[2, 3, 4]).unwrap();
        // two slabs split along the slowest axis: rows z = 0 and z = 1.
        let lower: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let upper: Vec<f64> = (0..12).map(|i| 100.0 + i as f64).collect();
        stream.write_slab("level_0/hydro/rho", &[0, 0, 0], &[1, 3, 4], &lower).unwrap();
        stream.write_slab("level_0/hydro/rho", &[1, 0, 0], &[1, 3, 4], &upper).unwrap();
        assert!(!target.exists(), "the target appears at publish alone");
        assert!(temporary_name(&target).unwrap().exists());
        let published = stream.publish().unwrap();
        assert_eq!(published, target);
        assert!(!temporary_name(&target).unwrap().exists());
        let tree = Hdf5Backend.read(&target).unwrap();
        let ds = tree.find_group("level_0").unwrap().find_group("hydro").unwrap().find_dataset("rho").unwrap();
        let full = ds.data.as_f64().unwrap();
        assert_eq!(ds.shape, vec![2, 3, 4]);
        assert_eq!(&full[..12], &lower[..]);
        assert_eq!(&full[12..], &upper[..]);
        // a partial column read: y = 1..3, x = 2..4 of the upper row.
        let block = read_slab(&target, "level_0/hydro/rho", &[1, 1, 2], &[1, 2, 2]).unwrap();
        assert_eq!(block, vec![106.0, 107.0, 110.0, 111.0]);
    }

    #[test]
    fn a_slab_of_the_wrong_volume_is_refused_and_a_dropped_stream_leaves_nothing() {
        let dir = scratch("refuse");
        let target = dir.join("run.h5");
        {
            let stream = Hdf5Stream::create(&target).unwrap();
            stream.declare_f64("g", "f", &[2, 2]).unwrap();
            let err = stream.write_slab("g/f", &[0, 0], &[2, 2], &[1.0, 2.0, 3.0]).unwrap_err();
            assert!(matches!(err, IoError::ShapeMismatch { .. }), "{err:?}");
        }
        assert!(!target.exists());
        assert!(!temporary_name(&target).unwrap().exists());
    }

    #[test]
    fn a_missing_directory_fails_at_create_with_no_file() {
        let target = scratch("missing").join("absent").join("run.h5");
        assert!(Hdf5Stream::create(&target).is_err());
        assert!(!target.exists());
    }

    #[test]
    fn a_slab_past_the_dataset_bounds_is_refused_on_read() {
        let dir = scratch("bounds");
        let target = dir.join("run.h5");
        let stream = Hdf5Stream::create(&target).unwrap();
        stream.declare_f64("g", "f", &[2, 2]).unwrap();
        stream.write_slab("g/f", &[0, 0], &[2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        stream.publish().unwrap();
        assert!(read_slab(&target, "g/f", &[1, 0], &[2, 2]).is_err());
        assert_eq!(read_slab(&target, "g/f", &[1, 0], &[1, 2]).unwrap(), vec![3.0, 4.0]);
    }
}
