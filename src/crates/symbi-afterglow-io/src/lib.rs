// =============================================================================
// symbi-afterglow-io
//
// the checkpoint adapter: read a symbi HDF5 checkpoint (any geometry the hydro ran —
// Cartesian / Spherical / Cylindrical, in 1/2/3D) into the afterglow's neutral, Cartesian
// `Cell` list. this is the last hop that lets the afterglow module observe real simulation
// output end-to-end.
//
// it owns the geometry's axis-role layout (the bit that differs from sim to sim):
//   Cartesian 3D : (x1,x2,x3) = (x, y, z)
//   Spherical 3D : (x1,x2,x3) = (r, theta, phi)
//   Spherical 2D : (x1,x2)    = (r, theta), phi synthesized (axisymmetric)
//   Spherical 1D : (x1)       = (r), theta & phi synthesized
//   Cylindrical 3D: (x1,x2,x3)= (r, phi, z)
//   Cylindrical 2D: (x1,x2)   = (r, z),    phi synthesized  [note: x2 is z, the third role]
// for reduced-dimension (axisymmetric / radial) runs it synthesizes the missing angular grid
// and broadcasts the data over it, weighting cell volume by the geometry's volume element.
//
// the checkpoint is in code units; `CgsScales` converts length/density/pressure/time to CGS.
// velocity is the relativistic three-velocity (already in units of c). proper cell volumes are
// computed from the coordinate cell widths and the geometry's volume factor.
//
// usage:
//  let cells = read_cells(path, &scales, &Synth::default(), dt_seconds)?;
//  let events = symbi_afterglow::generate_events_from_cells(&cells, &micro, seed, 4, max);
// =============================================================================

use std::path::Path;

use symbi_afterglow::{Cell, Coords};
use symbi_io::backend::IoBackend;
use symbi_io::hdf5::Hdf5Backend;
use symbi_io::{DataBuf, IoError, Result, TreeBuf};

/// code-unit -> CGS conversion scales for a checkpoint. velocity needs no scale (it is the
/// relativistic three-velocity, already in units of c).
#[derive(Clone, Copy, Debug)]
pub struct CgsScales {
    /// cm per code length.
    pub length: f64,
    /// g/cm^3 per code density.
    pub density: f64,
    /// erg/cm^3 per code pressure.
    pub pressure: f64,
    /// seconds per code time.
    pub time: f64,
}

/// synthesized angular resolution for reduced-dimension runs: `n_phi` azimuthal cells for an
/// axisymmetric (2D) snapshot, plus `n_theta` polar cells for a 1D radial snapshot.
#[derive(Clone, Copy, Debug)]
pub struct Synth {
    pub n_theta: usize,
    pub n_phi: usize,
}

impl Default for Synth {
    fn default() -> Self {
        Synth {
            n_theta: 64,
            n_phi: 128,
        }
    }
}

/// the raw arrays pulled from one checkpoint, before geometry mapping.
struct Raw {
    coords: Coords,
    dims: usize,
    centers: Vec<Vec<f64>>, // x1..xD cell centers (code units)
    rho: Vec<f64>,
    pre: Vec<f64>,
    vel: Vec<Vec<f64>>, // v1..vD (units of c)
    time_code: f64,
}

fn f64_dataset(g: &TreeBuf, name: &str) -> Result<Vec<f64>> {
    let ds = g
        .find_dataset(name)
        .ok_or_else(|| IoError::MissingPath(name.into()))?;
    match &ds.data {
        DataBuf::F64(v) => Ok(v.clone()),
        _ => Err(IoError::TypeMismatch {
            path: name.into(),
            expected: "f64",
            actual: "?",
        }),
    }
}

fn usize_dataset(g: &TreeBuf, name: &str) -> Result<Vec<usize>> {
    let ds = g
        .find_dataset(name)
        .ok_or_else(|| IoError::MissingPath(name.into()))?;
    match &ds.data {
        DataBuf::U64(v) => Ok(v.iter().map(|&x| x as usize).collect()),
        DataBuf::I64(v) => Ok(v.iter().map(|&x| x as usize).collect()),
        _ => Err(IoError::TypeMismatch {
            path: name.into(),
            expected: "int",
            actual: "?",
        }),
    }
}

// string metadata (regime / coord_system / ..) rides as an HDF5 string attribute
// (the v2.0 convention symbi-io writes and the python reader reads); it is not a byte
// dataset.
fn str_attr(g: &TreeBuf, name: &str) -> Result<String> {
    g.find_attr(name)
        .ok_or_else(|| IoError::MissingPath(name.into()))?
        .as_str(name)
        .map(|s| s.to_string())
}

fn parse_coords(s: &str) -> Result<Coords> {
    let l = s.to_ascii_lowercase();
    if l.starts_with("cart") {
        Ok(Coords::Cartesian)
    } else if l.starts_with("spher") {
        Ok(Coords::Spherical)
    } else if l.starts_with("cylind") {
        Ok(Coords::Cylindrical)
    } else {
        Err(IoError::Backend(format!("unknown coord_system '{s}'")))
    }
}

fn read_raw(path: &Path) -> Result<Raw> {
    let tree = Hdf5Backend.read(path)?;
    let meta = tree
        .find_group("metadata")
        .ok_or_else(|| IoError::MissingPath("metadata".into()))?;
    let dims = meta
        .find_attr("dimensions")
        .ok_or_else(|| IoError::MissingPath("metadata/dimensions".into()))?
        .as_u64("metadata/dimensions")? as usize;
    let time_code = meta
        .find_attr("time")
        .ok_or_else(|| IoError::MissingPath("metadata/time".into()))?
        .as_f64("metadata/time")?;
    let coords = parse_coords(&str_attr(meta, "coord_system")?)?;

    let level0 = tree
        .find_group("level_0")
        .ok_or_else(|| IoError::MissingPath("level_0".into()))?;
    let mesh = level0
        .find_group("mesh")
        .ok_or_else(|| IoError::MissingPath("level_0/mesh".into()))?;

    // v2.0 mesh: cell centers are rebuilt from the geometry description
    // (global_cells + per-dim start/end attrs); no coordinate arrays are stored.
    let geometry = mesh
        .find_group("geometry")
        .ok_or_else(|| IoError::MissingPath("level_0/mesh/geometry".into()))?;
    let global_cells = usize_dataset(mesh, "global_cells")?;
    let centers = (0..dims)
        .map(|ax| -> Result<Vec<f64>> {
            let dim = geometry
                .find_group(&format!("dim_{ax}"))
                .ok_or_else(|| IoError::MissingPath(format!("level_0/mesh/geometry/dim_{ax}")))?;
            let start = dim
                .find_attr("start")
                .ok_or_else(|| IoError::MissingPath("dim/start".into()))?
                .as_f64("start")?;
            let end = dim
                .find_attr("end")
                .ok_or_else(|| IoError::MissingPath("dim/end".into()))?
                .as_f64("end")?;
            let n = global_cells[ax];
            let dx = (end - start) / n as f64;
            Ok((0..n).map(|i| start + (i as f64 + 0.5) * dx).collect())
        })
        .collect::<Result<_>>()?;

    // v2.0 fields: primitives live under partition_0/hydro.
    let part = level0
        .find_group("partition_0")
        .ok_or_else(|| IoError::MissingPath("level_0/partition_0".into()))?;
    let hydro = part
        .find_group("hydro")
        .ok_or_else(|| IoError::MissingPath("level_0/partition_0/hydro".into()))?;
    let prim = hydro
        .find_group("primitives")
        .ok_or_else(|| IoError::MissingPath("level_0/partition_0/hydro/primitives".into()))?;

    let vel = (0..dims)
        .map(|ax| f64_dataset(prim, &format!("v{}", ax + 1)))
        .collect::<Result<_>>()?;
    let rho = f64_dataset(prim, "rho")?;
    let pre = f64_dataset(prim, "pre")?;

    Ok(Raw {
        coords,
        dims,
        centers,
        rho,
        pre,
        vel,
        time_code,
    })
}

/// per-cell width (code units) from cell centers: arithmetic midpoints, boundary cells use the
/// adjacent gap. works for uniform or stretched (e.g., log-radial) grids.
fn cell_widths(centers: &[f64]) -> Vec<f64> {
    let n = centers.len();
    (0..n)
        .map(|i| {
            if n == 1 {
                centers[0].abs().max(1.0)
            } else if i == 0 {
                centers[1] - centers[0]
            } else if i == n - 1 {
                centers[n - 1] - centers[n - 2]
            } else {
                0.5 * (centers[i + 1] - centers[i - 1])
            }
        })
        .collect()
}

/// map raw checkpoint arrays to Cartesian cells, applying the geometry's axis roles, synthesizing
/// the missing angular grid for reduced-dimension runs, and weighting each cell's proper volume.
fn build_cells(raw: &Raw, scales: &CgsScales, synth: &Synth, t_emission: f64) -> Result<Vec<Cell>> {
    let l = scales.length;
    let (rho_s, pre_s) = (scales.density, scales.pressure);
    let dims = raw.dims;
    let cs = raw.coords;

    // real-data extents and widths.
    let n: Vec<usize> = raw.centers.iter().map(|c| c.len()).collect();
    let w: Vec<Vec<f64>> = raw.centers.iter().map(|c| cell_widths(c)).collect();
    // flat index with x1 fastest (on-disk layout is reversed-shape: [.., N2, N1]).
    let flat = |i: &[usize]| -> usize {
        let mut idx = 0usize;
        let mut stride = 1usize;
        for ax in 0..dims {
            idx += i[ax] * stride;
            stride *= n[ax];
        }
        idx
    };

    let mut cells = Vec::new();
    // a small helper to push one cell from role-ordered (coord, velocity, volume).
    let push = |coord: [f64; 3], v: [f64; 3], idx: usize, vol: f64, out: &mut Vec<Cell>| {
        out.push(Cell::from_coords(
            cs,
            coord,
            v,
            raw.rho[idx] * rho_s,
            raw.pre[idx] * pre_s,
            vol,
            t_emission,
        ));
    };

    match (cs, dims) {
        // ---- spherical (r, theta, phi) ----
        (Coords::Spherical, 3) => {
            for i3 in 0..n[2] {
                for i2 in 0..n[1] {
                    for i1 in 0..n[0] {
                        let idx = flat(&[i1, i2, i3]);
                        let (r, th, ph) = (
                            raw.centers[0][i1] * l,
                            raw.centers[1][i2],
                            raw.centers[2][i3],
                        );
                        let vol = r * r * th.sin().abs() * (w[0][i1] * l) * w[1][i2] * w[2][i3];
                        let v = [raw.vel[0][idx], raw.vel[1][idx], raw.vel[2][idx]];
                        push([r, th, ph], v, idx, vol, &mut cells);
                    }
                }
            }
        }
        (Coords::Spherical, 2) => {
            let dphi = 2.0 * std::f64::consts::PI / synth.n_phi as f64;
            for kphi in 0..synth.n_phi {
                let ph = (kphi as f64 + 0.5) * dphi;
                for i2 in 0..n[1] {
                    for i1 in 0..n[0] {
                        let idx = flat(&[i1, i2]);
                        let (r, th) = (raw.centers[0][i1] * l, raw.centers[1][i2]);
                        let vol = r * r * th.sin().abs() * (w[0][i1] * l) * w[1][i2] * dphi;
                        let v = [raw.vel[0][idx], raw.vel[1][idx], 0.0]; // (vr, vtheta, 0)
                        push([r, th, ph], v, idx, vol, &mut cells);
                    }
                }
            }
        }
        (Coords::Spherical, 1) => {
            let dphi = 2.0 * std::f64::consts::PI / synth.n_phi as f64;
            let dth = std::f64::consts::PI / synth.n_theta as f64;
            for kth in 0..synth.n_theta {
                let th = (kth as f64 + 0.5) * dth;
                for kphi in 0..synth.n_phi {
                    let ph = (kphi as f64 + 0.5) * dphi;
                    for i1 in 0..n[0] {
                        let idx = i1;
                        let r = raw.centers[0][i1] * l;
                        let vol = r * r * th.sin().abs() * (w[0][i1] * l) * dth * dphi;
                        let v = [raw.vel[0][idx], 0.0, 0.0]; // (vr, 0, 0)
                        push([r, th, ph], v, idx, vol, &mut cells);
                    }
                }
            }
        }
        // ---- cylindrical (r, phi, z) ----
        (Coords::Cylindrical, 3) => {
            for i3 in 0..n[2] {
                for i2 in 0..n[1] {
                    for i1 in 0..n[0] {
                        let idx = flat(&[i1, i2, i3]);
                        let (r, ph, z) = (
                            raw.centers[0][i1] * l,
                            raw.centers[1][i2],
                            raw.centers[2][i3] * l,
                        );
                        let vol = r * (w[0][i1] * l) * w[1][i2] * (w[2][i3] * l);
                        let v = [raw.vel[0][idx], raw.vel[1][idx], raw.vel[2][idx]];
                        push([r, ph, z], v, idx, vol, &mut cells);
                    }
                }
            }
        }
        (Coords::Cylindrical, 2) => {
            // axisymmetric (r, z): x1 = r, x2 = z (the third coordinate role); phi synthesized.
            let dphi = 2.0 * std::f64::consts::PI / synth.n_phi as f64;
            for kphi in 0..synth.n_phi {
                let ph = (kphi as f64 + 0.5) * dphi;
                for i2 in 0..n[1] {
                    for i1 in 0..n[0] {
                        let idx = flat(&[i1, i2]);
                        let (r, z) = (raw.centers[0][i1] * l, raw.centers[1][i2] * l);
                        let vol = r * (w[0][i1] * l) * dphi * (w[1][i2] * l);
                        // stored v2 is v_z (role 2); v_phi (role 1) is unresolved -> 0.
                        let v = [raw.vel[0][idx], 0.0, raw.vel[1][idx]];
                        push([r, ph, z], v, idx, vol, &mut cells);
                    }
                }
            }
        }
        // ---- cartesian (x, y, z) ----
        (Coords::Cartesian, 3) => {
            for i3 in 0..n[2] {
                for i2 in 0..n[1] {
                    for i1 in 0..n[0] {
                        let idx = flat(&[i1, i2, i3]);
                        let coord = [
                            raw.centers[0][i1] * l,
                            raw.centers[1][i2] * l,
                            raw.centers[2][i3] * l,
                        ];
                        let vol = (w[0][i1] * l) * (w[1][i2] * l) * (w[2][i3] * l);
                        let v = [raw.vel[0][idx], raw.vel[1][idx], raw.vel[2][idx]];
                        push(coord, v, idx, vol, &mut cells);
                    }
                }
            }
        }
        (c, d) => {
            return Err(IoError::Backend(format!(
                "unsupported geometry/dimensionality for afterglow imaging: {c:?} in {d}D \
                 (a localized outflow needs spherical 1/2/3D, cylindrical 2/3D, or cartesian 3D)"
            )));
        }
    }

    Ok(cells)
}

/// read one checkpoint into Cartesian cells. the emission timestep each cell radiates over (the
/// checkpoint cadence) is the caller's `Microphysics.dt` at `generate_events_from_cells` — it is
/// not a property of the cell, so it is not stored here.
pub fn read_cells(path: &Path, scales: &CgsScales, synth: &Synth) -> Result<Vec<Cell>> {
    let raw = read_raw(path)?;
    let t_emission = raw.time_code * scales.time;
    build_cells(&raw, scales, synth, t_emission)
}

/// read a time sequence of checkpoints into one concatenated cell list — the multi-checkpoint
/// EATS workflow (each cell carries its checkpoint's lab emission time, so the EATS integrates
/// over the outflow's evolution). returns `(cells, dt_seconds)` where `dt_seconds` is the
/// checkpoint cadence to use as `Microphysics.dt` (the gap between the first two checkpoints).
/// assumes a ~uniform cadence (the common case: fixed dump interval); a strongly non-uniform
/// cadence would need a per-checkpoint emission window, which the single-`dt` weighting here does
/// not model. loads every checkpoint's primitives into memory — for very large sequences, call
/// `read_cells` per file and weight each batch with its own `Microphysics.dt`.
pub fn read_sequence(
    paths: &[&Path],
    scales: &CgsScales,
    synth: &Synth,
) -> Result<(Vec<Cell>, f64)> {
    if paths.is_empty() {
        return Ok((Vec::new(), 0.0));
    }
    let raws: Vec<Raw> = paths.iter().map(|p| read_raw(p)).collect::<Result<_>>()?;
    let times: Vec<f64> = raws.iter().map(|r| r.time_code * scales.time).collect();

    let mut cells = Vec::new();
    for (i, raw) in raws.iter().enumerate() {
        let mut c = build_cells(raw, scales, synth, times[i])?;
        cells.append(&mut c);
    }
    let dt0 = if times.len() > 1 {
        (times[1] - times[0]).abs()
    } else {
        times[0].max(1.0)
    };
    Ok((cells, dt0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scales() -> CgsScales {
        CgsScales {
            length: 1.0e15,
            density: 1.0e-24,
            pressure: 1.0e-3,
            time: 1.0e5,
        }
    }

    // spherical 2D (r, theta) broadcasts over a synthesized phi grid: cell count = Nr*Ntheta*Nphi,
    // a radial velocity yields beta_vec along the position, volumes are positive.
    #[test]
    fn spherical_2d_axisymmetric_broadcast() {
        let nr = 4;
        let nth = 3;
        let centers = vec![
            (0..nr).map(|i| 1.0 + i as f64).collect::<Vec<_>>(),
            (0..nth).map(|j| 0.3 + 0.4 * j as f64).collect::<Vec<_>>(),
        ];
        // reversed-shape flat layout: idx = i1 + Nr*i2.
        let len = nr * nth;
        let raw = Raw {
            coords: Coords::Spherical,
            dims: 2,
            centers,
            rho: vec![1.0; len],
            pre: vec![1.0; len],
            vel: vec![vec![0.5; len], vec![0.0; len]], // purely radial (vr=0.5, vtheta=0)
            time_code: 10.0,
        };
        let synth = Synth {
            n_theta: 8,
            n_phi: 6,
        };
        let cells = build_cells(&raw, &scales(), &synth, 1.0e6).unwrap();
        assert_eq!(cells.len(), nr * nth * synth.n_phi);
        assert!(
            cells.iter().all(|c| c.volume > 0.0),
            "positive proper volumes"
        );
        // a radial velocity is parallel to the position direction.
        for c in &cells {
            let rmag =
                (c.position[0].powi(2) + c.position[1].powi(2) + c.position[2].powi(2)).sqrt();
            let bmag =
                (c.beta_vec[0].powi(2) + c.beta_vec[1].powi(2) + c.beta_vec[2].powi(2)).sqrt();
            let cos = (c.position[0] * c.beta_vec[0]
                + c.position[1] * c.beta_vec[1]
                + c.position[2] * c.beta_vec[2])
                / (rmag * bmag);
            assert!(
                (cos - 1.0).abs() < 1e-9,
                "radial velocity should align with r-hat"
            );
        }
    }

    // cylindrical 2D (r, z): the second mesh axis is z (the third coordinate role), and the
    // stored v2 maps to v_z — so beta_vec has the right out-of-plane (z) component.
    #[test]
    fn cylindrical_2d_axis_roles() {
        let raw = Raw {
            coords: Coords::Cylindrical,
            dims: 2,
            centers: vec![vec![2.0], vec![5.0]], // r=2, z=5 (single cell)
            rho: vec![1.0],
            pre: vec![1.0],
            vel: vec![vec![0.0], vec![0.6]], // v1=vr=0, v2=vz=0.6 (purely vertical)
            time_code: 1.0,
        };
        let synth = Synth {
            n_theta: 1,
            n_phi: 4,
        };
        let cells = build_cells(&raw, &scales(), &synth, 1.0e6).unwrap();
        assert_eq!(cells.len(), synth.n_phi);
        // v2 = vz -> the velocity is purely along +z for every azimuth (cylindrical z is lab z).
        for c in &cells {
            assert!(
                c.beta_vec[2] > 0.59 && c.beta_vec[2] < 0.61,
                "v2 must map to lab v_z"
            );
            assert!(
                c.beta_vec[0].abs() < 1e-9 && c.beta_vec[1].abs() < 1e-9,
                "no in-plane velocity"
            );
        }
    }

    // an unsupported combination (cartesian 2D slab) errors clearly and does not silently
    // produce a degenerate image.
    #[test]
    fn unsupported_geometry_errors() {
        let raw = Raw {
            coords: Coords::Cartesian,
            dims: 2,
            centers: vec![vec![0.0, 1.0], vec![0.0, 1.0]],
            rho: vec![1.0; 4],
            pre: vec![1.0; 4],
            vel: vec![vec![0.0; 4], vec![0.0; 4]],
            time_code: 1.0,
        };
        assert!(build_cells(&raw, &scales(), &Synth::default(), 0.0).is_err());
    }
}
