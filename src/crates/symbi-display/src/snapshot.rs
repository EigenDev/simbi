// =============================================================================
// snapshot.rs
//
// on-disk live snapshot for read-only attach. a headless solver (a batch/cluster
// run with no tty) serializes a `DiagnosticView` plus the full bundle of
// selectable fields to `<rundir>/.simbi-live/snapshot.bin` every diagnostic
// cadence; a `simbi attach <rundir>` client polls that file and renders it
// through the shared `live` layer. bundling every field (density, and pressure /
// W / |B| when present) lets the client's `f`-key switch fields with no producer
// round-trip, so the transport stays strictly one-way.
//
// the write is atomic (temp file in the same directory + rename) so a polling
// reader never observes a torn frame; postcard keeps the float-heavy field
// bundle compact on a shared filesystem.
//
// usage:
//   // producer (each cadence):
//   Snapshot { view, fields }.write_atomic(rundir)?;
//   // consumer (poll loop):
//   let snap = Snapshot::read(&snapshot_path(rundir))?;
// =============================================================================

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::live::{DiagnosticView, FieldSlice};

/// the snapshot directory under a run's output directory.
const DIR: &str = ".simbi-live";
/// the snapshot file within `DIR`.
const FILE: &str = "snapshot.bin";

/// a complete render frame plus every selectable field, so a read-only client can
/// cycle fields locally. `fields` is ordered density-first, matching the solver's
/// field index; the client picks `fields[field_kind]`.
#[derive(Serialize, Deserialize)]
pub struct Snapshot {
    pub view: DiagnosticView,
    pub fields: Vec<FieldSlice>,
}

/// the snapshot path for a run's output directory: `<rundir>/.simbi-live/snapshot.bin`.
pub fn snapshot_path(rundir: &Path) -> PathBuf {
    rundir.join(DIR).join(FILE)
}

impl Snapshot {
    /// serialize + write atomically (temp file + rename within the same directory,
    /// so the rename is atomic and a concurrent reader sees only a complete file).
    /// best-effort: the caller ignores the error so a full disk never kills a run.
    pub fn write_atomic(&self, rundir: &Path) -> io::Result<()> {
        let dir = rundir.join(DIR);
        fs::create_dir_all(&dir)?;
        let bytes = postcard::to_allocvec(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let tmp = dir.join(concat!("snapshot.bin", ".tmp"));
        fs::write(&tmp, &bytes)?;
        fs::rename(&tmp, dir.join(FILE))
    }

    /// read + deserialize a snapshot file.
    pub fn read(path: &Path) -> io::Result<Snapshot> {
        let bytes = fs::read(path)?;
        postcard::from_bytes(&bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

/// remove a run's snapshot directory. called when a run ends so a stale snapshot
/// does not outlive it; a still-attached client keeps showing its last frame.
/// idempotent — a missing directory is not an error.
pub fn cleanup(rundir: &Path) -> io::Result<()> {
    match fs::remove_dir_all(rundir.join(DIR)) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live::{Colormap, DiagnosticView};

    fn sample_view() -> DiagnosticView {
        DiagnosticView {
            app_title: "test — kh.toml".into(),
            regime: "RHD".into(),
            attached: String::new(),
            paused: false,
            frame: 7,
            t: 1.25,
            step: 420,
            dt: 3.1e-4,
            wall_secs: 12.0,
            throughput_mzcups: 148.0,
            tab: 0,
            config_scroll: 0,
            throughput_hist: vec![140.0, 148.0],
            dt_hist: vec![3.0e-4, 3.1e-4],
            mass_drift: Some(vec![2.4e-13]),
            energy_drift: None,
            div_b: None,
            max_w: Some(4.8),
            cfl: 0.4,
            cfl_max: 0.8,
            blocks_per_level: vec![64],
            log: vec![("00:01".into(), "started".into())],
            config: vec![("Physics".into(), "regime".into(), "rhd".into())],
            field: None,
            field_count: 0,
            host: Some(crate::hostinfo::HostStats {
                hostname: "node42".into(),
                cpu_count: 64,
                threads: 32,
                mem_rss: 3_221_225_472,
                mem_total: 68_719_476_736,
            }),
        }
    }

    fn sample_field(name: &str, v: f32) -> FieldSlice {
        FieldSlice {
            label: name.into(),
            width: 2,
            height: 2,
            data: vec![v, v + 1.0, v + 2.0, v + 3.0],
            vmin: v as f64,
            vmax: (v + 3.0) as f64,
            cmap: Colormap::Inferno,
        }
    }

    // the snapshot is the attach wire contract: a run written by one build must
    // deserialize field-for-field in the client. round-trip through the real
    // atomic write + read path.
    #[test]
    fn snapshot_round_trips_through_disk() {
        let rundir =
            std::env::temp_dir().join(format!("simbi-snapshot-test-{}", std::process::id()));
        let fields = vec![sample_field("density", 1.0), sample_field("pressure", 5.0)];
        let snap = Snapshot {
            view: sample_view(),
            fields,
        };
        snap.write_atomic(&rundir).unwrap();

        // the atomic write leaves no dangling temp file.
        assert!(!rundir.join(DIR).join("snapshot.bin.tmp").exists());

        let got = Snapshot::read(&snapshot_path(&rundir)).unwrap();
        assert_eq!(got.fields.len(), 2);
        assert_eq!(got.fields[0].label, "density");
        assert_eq!(got.fields[1].data, vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(got.view.step, 420);
        assert_eq!(got.view.mass_drift, Some(vec![2.4e-13]));
        assert_eq!(got.view.energy_drift, None);
        let host = got.view.host.expect("host stats round-trip");
        assert_eq!(host.hostname, "node42");
        assert_eq!(host.mem_total, 68_719_476_736);

        let _ = fs::remove_dir_all(&rundir);
    }
}
