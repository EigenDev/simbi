// =============================================================================
// afterglow.rs
//
// pyo3 binding for the symbi-afterglow synchrotron post-processor. the python
// `simbi afterglow generate` workflow crosses into rust here:
//   - generate_photon_events: python dicts (sim_cond, qscales, fields, mesh) ->
//     rust SimConditions / QuantScales / HydroFields / Mesh, dispatch on the mesh
//     data_dim (1 = the spherical BMK path), returns an opaque PhotonEvents handle.
//   - monte_carlo_radiative_transfer: run MCRT in place on the handle.
//   - write_photon_events: serialize the handle to HDF5, byte-compatible with the
//     python `postprocess.read_photon_events` schema (root datasets + root attrs).
//
// the handle wraps a `Vec<PhotonEvent>` so it round-trips between the three calls
// without ever materializing per-event python objects (cheapest boundary). the
// HDF5 write reuses the symbi-io `Hdf5Backend` Tree writer — no bespoke hdf5 code.
//
// usage (from python):
//  from simbi.libs import cpu_ext
//  ev = cpu_ext.generate_photon_events(sim_cond=.., qscales=.., fields=.., mesh=..,
//                                      seed=.., max_events=..)
//  cpu_ext.monte_carlo_radiative_transfer(ev, sim_cond=.., qscales=.., fields=..,
//                                         mesh=.., seed=.., include_scattering=..,
//                                         include_pair_production=..)
//  cpu_ext.write_photon_events(output_path, ev, sim_cond=.., qscales=..)
// =============================================================================

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use symbi_afterglow::event::PhotonEvent;
use symbi_afterglow::observe::{
    compute_lightcurve_from_events, compute_skymap as observe_compute_skymap,
};
use symbi_afterglow::transfer::{
    compute_skymap_deposit_spherical, generate_photon_events, generate_photon_events_spherical,
    monte_carlo_radiative_transfer,
};
use symbi_afterglow::units::{Frequency, Length};
use symbi_afterglow::{HydroFields, Mesh, QuantScales, SimConditions};

use symbi_io::tree::{DataBuf, DataRef, Dataset, TreeBuf};
use symbi_io::{Hdf5Backend, IoBackend, Tree};

// =============================================================================
// the opaque event-catalog handle
// =============================================================================

/// an opaque catalog of lab-frame photon packets — the data product handed between
/// the three afterglow calls. python treats it as a handle (len + absorbed count);
/// the per-event fields live in rust and only cross to numpy at the HDF5 write.
#[pyclass(name = "PhotonEvents")]
pub struct PhotonEvents {
    events: Vec<PhotonEvent>,
}

#[pymethods]
impl PhotonEvents {
    /// total number of packets in the catalog.
    fn __len__(&self) -> usize {
        self.events.len()
    }

    /// number of packets flagged absorbed by the transfer step.
    #[getter]
    fn n_absorbed(&self) -> usize {
        self.events.iter().filter(|e| e.absorbed).count()
    }

    /// number of surviving (unabsorbed) packets.
    #[getter]
    fn n_surviving(&self) -> usize {
        self.events.iter().filter(|e| !e.absorbed).count()
    }

    /// merge another catalog into this one (consuming the other's packets), so a
    /// multi-checkpoint run concatenates per-file catalogs without a python list.
    fn extend(&mut self, other: &mut PhotonEvents) {
        self.events.append(&mut other.events);
    }
}

// =============================================================================
// dict -> rust conversion
// =============================================================================

/// required f64 from a dict, erroring with the missing key name.
fn req_f64(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<f64> {
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing '{key}'")))?
        .extract()
}

/// optional f64 with a default.
fn opt_f64(dict: &Bound<'_, PyDict>, key: &str, default: f64) -> f64 {
    dict.get_item(key)
        .ok()
        .flatten()
        .and_then(|v| v.extract::<f64>().ok())
        .unwrap_or(default)
}

/// required flat f64 array from the fields dict (gamma_beta / rho / pre).
fn req_field<'a>(dict: &Bound<'a, PyDict>, key: &str) -> PyResult<Vec<f64>> {
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("fields missing '{key}'")))?
        .extract()
}

/// build `QuantScales` from the python qscales dict (code->CGS factors).
fn quant_scales(dict: &Bound<'_, PyDict>) -> PyResult<QuantScales> {
    use symbi_afterglow::units::{EnergyDensity, MassDensity, Time, Velocity};
    Ok(QuantScales {
        time:     Time::new(req_f64(dict, "time")?),
        pre:      EnergyDensity::new(req_f64(dict, "pre")?),
        rho:      MassDensity::new(req_f64(dict, "rho")?),
        velocity: Velocity::new(req_f64(dict, "velocity")?),
        length:   Length::new(req_f64(dict, "length")?),
    })
}

/// build `SimConditions` from the python sim_cond dict. `nus` defaults to an empty
/// list (the monte-carlo path samples its own frequencies); `theta_obs` / `redshift`
/// / `d_l` are carried for the observer reductions but unused by generation.
fn sim_conditions(dict: &Bound<'_, PyDict>) -> PyResult<SimConditions> {
    let nus: Vec<Frequency> = dict
        .get_item("nus")?
        .and_then(|v| v.extract::<Vec<f64>>().ok())
        .unwrap_or_default()
        .into_iter()
        .map(Frequency::new)
        .collect();
    Ok(SimConditions {
        dt:              req_f64(dict, "dt")?,
        theta_obs:       opt_f64(dict, "theta_obs", 0.0),
        adiabatic_index: req_f64(dict, "adiabatic_index")?,
        current_time:    req_f64(dict, "current_time")?,
        p:               req_f64(dict, "p")?,
        redshift:        opt_f64(dict, "z", 0.0),
        eps_e:           req_f64(dict, "eps_e")?,
        eps_b:           req_f64(dict, "eps_b")?,
        d_l:             Length::new(opt_f64(dict, "d_L", 1.0e28)),
        nus,
    })
}

/// pull (x1, x2, x3, data_dim) out of the mesh dict. x1 is mandatory; x2/x3 are
/// optional (a 1d run broadcasts over them). data_dim defaults to 1.
struct MeshData {
    x1:       Vec<f64>,
    x2:       Vec<f64>,
    x3:       Option<Vec<f64>>,
    data_dim: i64,
}

fn mesh_data(dict: &Bound<'_, PyDict>) -> PyResult<MeshData> {
    let x1: Vec<f64> = dict
        .get_item("x1")?
        .ok_or_else(|| PyValueError::new_err("mesh missing 'x1'"))?
        .extract()?;
    let x2: Vec<f64> = dict
        .get_item("x2")?
        .and_then(|v| v.extract::<Vec<f64>>().ok())
        // a degenerate single-cell theta lets the cartesian generator run for 1d data.
        .unwrap_or_else(|| vec![std::f64::consts::FRAC_PI_2]);
    let x3: Option<Vec<f64>> =
        dict.get_item("x3")?.and_then(|v| v.extract::<Vec<f64>>().ok());
    let data_dim = dict
        .get_item("data_dim")?
        .and_then(|v| v.extract::<i64>().ok())
        .unwrap_or(1);
    Ok(MeshData { x1, x2, x3, data_dim })
}

// =============================================================================
// the three bound functions
// =============================================================================

/// generate relativistically-beamed synchrotron photon packets from a hydro snapshot.
/// dispatches on `mesh["data_dim"]`: 1 uses the spherical generator (a synthesized
/// equal-solid-angle sphere from the 1d radial profile — the BMK target); 2/3 use the
/// general mesh generator. `seed` makes the catalog reproducible, `max_events` caps it.
#[pyfunction]
#[pyo3(name = "generate_photon_events")]
#[pyo3(signature = (sim_cond, qscales, fields, mesh, seed=0, max_events=1_000_000, photons_per_cell=0))]
#[allow(clippy::too_many_arguments)]
fn generate_photon_events_py(
    sim_cond: &Bound<'_, PyDict>,
    qscales: &Bound<'_, PyDict>,
    fields: &Bound<'_, PyDict>,
    mesh: &Bound<'_, PyDict>,
    seed: u64,
    max_events: u64,
    photons_per_cell: u64,
) -> PyResult<PhotonEvents> {
    let cond = sim_conditions(sim_cond)?;
    let scales = quant_scales(qscales)?;
    let rho = req_field(fields, "rho")?;
    let gamma_beta = req_field(fields, "gamma_beta")?;
    let pre = req_field(fields, "pre")?;
    if rho.len() != gamma_beta.len() || rho.len() != pre.len() {
        return Err(PyValueError::new_err(
            "fields rho / gamma_beta / pre must have equal length",
        ));
    }
    let hf = HydroFields { rho: &rho, gamma_beta: &gamma_beta, pre: &pre };
    let md = mesh_data(mesh)?;

    let events = if md.data_dim <= 1 {
        // the spherical BMK path: synthesize a full sphere from the 1d radial profile.
        // angular resolution and per-direction packet budget are derived from the cap so
        // a user only sets max_events; full sphere (theta_max = pi).
        let ppd = if photons_per_cell > 0 { photons_per_cell } else { 1 };
        generate_photon_events_spherical(
            &cond, &scales, &hf, &md.x1, seed, std::f64::consts::PI, 64, 128, ppd, max_events,
        )
    } else {
        let mesh = Mesh {
            x1:       &md.x1,
            x2:       &md.x2,
            x3:       md.x3.as_deref(),
            data_dim: md.data_dim,
        };
        generate_photon_events(&cond, &scales, &hf, &mesh, seed, max_events, photons_per_cell)
    };

    Ok(PhotonEvents { events })
}

/// propagate the catalog through the medium in place (synchrotron self-absorption +
/// thomson scattering, optional pair production), filling optical_depth and flipping
/// absorbed / n_scatter. operates on the data_dim==1 spherical catalog by indexing the
/// radial profile (cell_id is the radial index), so x2/x3 are not needed here.
#[pyfunction]
#[pyo3(name = "monte_carlo_radiative_transfer")]
#[pyo3(signature = (events, sim_cond, qscales, fields, mesh, seed=0, include_scattering=true, include_pair_production=false))]
#[allow(clippy::too_many_arguments)]
fn monte_carlo_radiative_transfer_py(
    events: &mut PhotonEvents,
    sim_cond: &Bound<'_, PyDict>,
    qscales: &Bound<'_, PyDict>,
    fields: &Bound<'_, PyDict>,
    mesh: &Bound<'_, PyDict>,
    seed: u64,
    include_scattering: bool,
    include_pair_production: bool,
) -> PyResult<()> {
    let cond = sim_conditions(sim_cond)?;
    let scales = quant_scales(qscales)?;
    let rho = req_field(fields, "rho")?;
    let gamma_beta = req_field(fields, "gamma_beta")?;
    let pre = req_field(fields, "pre")?;
    let hf = HydroFields { rho: &rho, gamma_beta: &gamma_beta, pre: &pre };
    let md = mesh_data(mesh)?;
    let mesh = Mesh {
        x1:       &md.x1,
        x2:       &md.x2,
        x3:       md.x3.as_deref(),
        data_dim: md.data_dim,
    };
    monte_carlo_radiative_transfer(
        &mut events.events,
        &cond,
        &scales,
        &hf,
        &mesh,
        seed,
        include_scattering,
        include_pair_production,
    );
    Ok(())
}

/// reduce the catalog into a resolved sky-plane image at a given line of sight. the
/// observer direction is an arbitrary unit vector, so one catalog serves EVERY viewing
/// angle without regeneration: per call the EATS surface t_obs = (1+z)(t_em - r.n/c) is
/// selected, the observer-direction doppler boost delta^doppler_power is recomputed toward
/// n (the limb-brightening), and the in-window packets are projected onto the plane
/// perpendicular to n and binned into n_pix*n_pix equal-area pixels.
///
/// returns (intensity flat row-major [n_pix*n_pix], n_pix, half_width [cm]); the caller
/// reshapes to [n_pix, n_pix]. doppler_power: 3 = specific intensity, 4 = bolometric.
#[pyfunction]
#[pyo3(name = "skymap_from_events")]
#[pyo3(signature = (events, observer_direction, observer_time, time_window=0.1, energy_min=0.0, energy_max=1.0e30, redshift=0.0, doppler_power=3.0, n_pix=256, half_width=0.0))]
#[allow(clippy::too_many_arguments)]
fn skymap_from_events_py(
    events: &PhotonEvents,
    observer_direction: Vec<f64>,
    observer_time: f64,
    time_window: f64,
    energy_min: f64,
    energy_max: f64,
    redshift: f64,
    doppler_power: f64,
    n_pix: usize,
    half_width: f64,
) -> PyResult<(Vec<f64>, usize, f64)> {
    if observer_direction.len() != 3 {
        return Err(PyValueError::new_err(
            "observer_direction must be a length-3 unit vector",
        ));
    }
    let nhat = [observer_direction[0], observer_direction[1], observer_direction[2]];
    // half_width > 0 fixes the field of view (a shared grid for streaming accumulation); 0 auto-sizes.
    let img = observe_compute_skymap(
        &events.events,
        nhat,
        observer_time,
        time_window,
        energy_min,
        energy_max,
        redshift,
        doppler_power,
        n_pix,
        half_width,
    );
    Ok((img.intensity, img.n_pix, img.half_width))
}

/// write the catalog to HDF5, matching the python `postprocess.read_photon_events`
/// schema exactly: one root dataset per per-event field plus the run metadata as root
/// attributes. the writer is the symbi-io `Hdf5Backend` Tree walker.
#[pyfunction]
#[pyo3(name = "write_photon_events")]
#[pyo3(signature = (output_path, events, sim_cond, qscales))]
fn write_photon_events_py(
    output_path: &str,
    events: &PhotonEvents,
    sim_cond: &Bound<'_, PyDict>,
    qscales: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let ev = &events.events;
    let n = ev.len();

    // soa columns, in the `read_photon_events` dataset order.
    let t_emission: Vec<f64> = ev.iter().map(|e| e.t_emission).collect();
    let x: Vec<f64> = ev.iter().map(|e| e.x).collect();
    let y: Vec<f64> = ev.iter().map(|e| e.y).collect();
    let z: Vec<f64> = ev.iter().map(|e| e.z).collect();
    // `energy` is the comoving energy weight (the flux/intensity accumulator).
    let energy: Vec<f64> = ev.iter().map(|e| e.energy_weight).collect();
    let px: Vec<f64> = ev.iter().map(|e| e.px).collect();
    let py: Vec<f64> = ev.iter().map(|e| e.py).collect();
    let pz: Vec<f64> = ev.iter().map(|e| e.pz).collect();
    let stokes_i: Vec<f64> = ev.iter().map(|e| e.stokes_i).collect();
    let stokes_q: Vec<f64> = ev.iter().map(|e| e.stokes_q).collect();
    let stokes_u: Vec<f64> = ev.iter().map(|e| e.stokes_u).collect();
    let stokes_v: Vec<f64> = ev.iter().map(|e| e.stokes_v).collect();
    let doppler: Vec<f64> = ev.iter().map(|e| e.doppler_factor).collect();
    let lorentz: Vec<f64> = ev.iter().map(|e| e.lorentz_factor()).collect();
    // the lab-frame fluid velocity vector — REQUIRED to round-trip the observer-direction
    // doppler (delta = 1/(gamma(1 - beta.n))) when a saved catalog is reduced; not assumed radial.
    let beta_x: Vec<f64> = ev.iter().map(|e| e.beta_vec[0]).collect();
    let beta_y: Vec<f64> = ev.iter().map(|e| e.beta_vec[1]).collect();
    let beta_z: Vec<f64> = ev.iter().map(|e| e.beta_vec[2]).collect();
    let optical_depth: Vec<f64> = ev.iter().map(|e| e.optical_depth).collect();
    let cell_id: Vec<u64> = ev.iter().map(|e| e.cell_id as u64).collect();
    // absorbed is read back as `.astype(bool)`; store as u8 (0/1).
    let absorbed: Vec<u8> = ev.iter().map(|e| e.absorbed as u8).collect();
    let n_scatter: Vec<u64> = ev.iter().map(|e| e.n_scatter as u64).collect();

    // emitter frequencies the catalog carries (the `frequencies` dataset, optional in
    // the reader but useful for spectral binning).
    let frequencies: Vec<f64> = ev.iter().map(|e| e.nu_emit).collect();

    let shape = vec![n];
    let mut tree = Tree::new("");
    tree.push_dataset(Dataset::new("t_emission", shape.clone(), DataRef::F64(&t_emission)));
    tree.push_dataset(Dataset::new("x", shape.clone(), DataRef::F64(&x)));
    tree.push_dataset(Dataset::new("y", shape.clone(), DataRef::F64(&y)));
    tree.push_dataset(Dataset::new("z", shape.clone(), DataRef::F64(&z)));
    tree.push_dataset(Dataset::new("energy", shape.clone(), DataRef::F64(&energy)));
    tree.push_dataset(Dataset::new("px", shape.clone(), DataRef::F64(&px)));
    tree.push_dataset(Dataset::new("py", shape.clone(), DataRef::F64(&py)));
    tree.push_dataset(Dataset::new("pz", shape.clone(), DataRef::F64(&pz)));
    tree.push_dataset(Dataset::new("stokes_I", shape.clone(), DataRef::F64(&stokes_i)));
    tree.push_dataset(Dataset::new("stokes_Q", shape.clone(), DataRef::F64(&stokes_q)));
    tree.push_dataset(Dataset::new("stokes_U", shape.clone(), DataRef::F64(&stokes_u)));
    tree.push_dataset(Dataset::new("stokes_V", shape.clone(), DataRef::F64(&stokes_v)));
    tree.push_dataset(Dataset::new("doppler_factor", shape.clone(), DataRef::F64(&doppler)));
    tree.push_dataset(Dataset::new("lorentz_factor", shape.clone(), DataRef::F64(&lorentz)));
    tree.push_dataset(Dataset::new("beta_x", shape.clone(), DataRef::F64(&beta_x)));
    tree.push_dataset(Dataset::new("beta_y", shape.clone(), DataRef::F64(&beta_y)));
    tree.push_dataset(Dataset::new("beta_z", shape.clone(), DataRef::F64(&beta_z)));
    tree.push_dataset(Dataset::new("optical_depth", shape.clone(), DataRef::F64(&optical_depth)));
    tree.push_dataset(Dataset::new("cell_id", shape.clone(), DataRef::U64(&cell_id)));
    tree.push_dataset(Dataset::new("absorbed", shape.clone(), DataRef::U8(&absorbed)));
    tree.push_dataset(Dataset::new("n_scatter", shape.clone(), DataRef::U64(&n_scatter)));
    tree.push_dataset(Dataset::new("frequencies", shape.clone(), DataRef::F64(&frequencies)));

    // run metadata as root attributes (the `meta_t` reader contract).
    tree.push_attr("dt", req_f64(sim_cond, "dt")?);
    tree.push_attr("theta_obs", opt_f64(sim_cond, "theta_obs", 0.0));
    tree.push_attr("adiabatic_index", req_f64(sim_cond, "adiabatic_index")?);
    tree.push_attr("current_time", req_f64(sim_cond, "current_time")?);
    tree.push_attr("p", req_f64(sim_cond, "p")?);
    tree.push_attr("z", opt_f64(sim_cond, "z", 0.0));
    tree.push_attr("eps_e", req_f64(sim_cond, "eps_e")?);
    tree.push_attr("eps_b", req_f64(sim_cond, "eps_b")?);
    tree.push_attr("d_L", opt_f64(sim_cond, "d_L", 1.0e28));
    tree.push_attr("time_scale", req_f64(qscales, "time")?);
    tree.push_attr("pre_scale", req_f64(qscales, "pre")?);
    tree.push_attr("rho_scale", req_f64(qscales, "rho")?);
    tree.push_attr("v_scale", req_f64(qscales, "velocity")?);
    tree.push_attr("length_scale", req_f64(qscales, "length")?);
    tree.push_attr("n_events", n as u64);
    // hydro_type: 0 = SRHD (unpolarized synchrotron); the reader reads it as an attr.
    tree.push_attr("hydro_type", 0u64);

    Hdf5Backend
        .write(std::path::Path::new(output_path), &tree)
        .map_err(|e| PyValueError::new_err(format!("write_photon_events: {e}")))?;
    Ok(())
}

/// read a catalog written by `write_photon_events` back into a handle, so a saved catalog can be
/// reduced (skymap_from_events / monte_carlo_radiative_transfer) WITHOUT regenerating — the
/// generate-once, reduce-many path. beta_vec is read from beta_x/y/z; a catalog written before
/// those existed falls back to a RADIAL reconstruction from position + the stored lorentz_factor.
#[pyfunction]
#[pyo3(name = "read_photon_events")]
fn read_photon_events_py(path: &str) -> PyResult<PhotonEvents> {
    let tree: TreeBuf = Hdf5Backend
        .read(std::path::Path::new(path))
        .map_err(|e| PyValueError::new_err(format!("read_photon_events: {e}")))?;
    events_from_tree(&tree)
}

/// total packet count in a saved catalog, read from the `t_emission` column's length WITHOUT
/// loading any column — the row count that bounds a `read_photon_events_chunk` loop.
#[pyfunction]
#[pyo3(name = "photon_event_count")]
fn photon_event_count_py(path: &str) -> PyResult<usize> {
    Hdf5Backend
        .dataset_len(std::path::Path::new(path), "t_emission")
        .map_err(|e| PyValueError::new_err(format!("photon_event_count: {e}")))
}

/// read only packets `[start, start + count)` of a saved catalog into a handle, so a huge
/// events file is reduced (lightcurve/skymap) chunk-by-chunk at O(count) memory — the
/// generate-once, chunk-read-many path. `start`/`count` are clamped to the file length, so the
/// final chunk is short and an out-of-range start returns an empty handle (loop terminator).
#[pyfunction]
#[pyo3(name = "read_photon_events_chunk")]
fn read_photon_events_chunk_py(path: &str, start: usize, count: usize) -> PyResult<PhotonEvents> {
    let tree: TreeBuf = Hdf5Backend
        .read_root_slice(std::path::Path::new(path), start, count)
        .map_err(|e| PyValueError::new_err(format!("read_photon_events_chunk: {e}")))?;
    events_from_tree(&tree)
}

/// build the packet vector from a catalog TreeBuf (whole-file `read` OR a hyperslab chunk —
/// both produce the same SoA columns, just different lengths). shared by the full and chunked
/// readers so the column->struct mapping (incl. the radial beta_vec fallback) lives in ONE place.
fn events_from_tree(tree: &TreeBuf) -> PyResult<PhotonEvents> {
    let f64col = |name: &str| -> PyResult<Vec<f64>> {
        tree.find_dataset(name)
            .and_then(|d| d.data.as_f64())
            .map(|s| s.to_vec())
            .ok_or_else(|| PyValueError::new_err(format!("catalog missing f64 dataset '{name}'")))
    };

    let t_emission = f64col("t_emission")?;
    let n = t_emission.len();
    let x = f64col("x")?;
    let y = f64col("y")?;
    let z = f64col("z")?;
    let energy = f64col("energy")?;
    let px = f64col("px")?;
    let py = f64col("py")?;
    let pz = f64col("pz")?;
    let stokes_i = f64col("stokes_I")?;
    let stokes_q = f64col("stokes_Q")?;
    let stokes_u = f64col("stokes_U")?;
    let stokes_v = f64col("stokes_V")?;
    let doppler = f64col("doppler_factor")?;
    let optical_depth = f64col("optical_depth")?;
    let frequencies = f64col("frequencies").unwrap_or_else(|_| vec![0.0; n]);

    let cell_id: Vec<u64> = tree
        .find_dataset("cell_id")
        .and_then(|d| d.data.as_u64())
        .map(|s| s.to_vec())
        .unwrap_or_else(|| vec![0; n]);
    let n_scatter: Vec<u64> = tree
        .find_dataset("n_scatter")
        .and_then(|d| d.data.as_u64())
        .map(|s| s.to_vec())
        .unwrap_or_else(|| vec![0; n]);
    let absorbed: Vec<bool> = match tree.find_dataset("absorbed").map(|d| &d.data) {
        Some(DataBuf::U8(v)) => v.iter().map(|&b| b != 0).collect(),
        _ => vec![false; n],
    };

    // beta_vec: prefer the stored vector; else reconstruct radially from position + lorentz.
    let (beta_x, beta_y, beta_z) =
        match (f64col("beta_x"), f64col("beta_y"), f64col("beta_z")) {
            (Ok(bx), Ok(by), Ok(bz)) => (bx, by, bz),
            _ => {
                let lorentz = f64col("lorentz_factor").unwrap_or_else(|_| vec![1.0; n]);
                let mut bx = vec![0.0; n];
                let mut by = vec![0.0; n];
                let mut bz = vec![0.0; n];
                for ii in 0..n {
                    let w = lorentz[ii].max(1.0);
                    let beta_mag = (1.0 - 1.0 / (w * w)).max(0.0).sqrt();
                    let r = (x[ii] * x[ii] + y[ii] * y[ii] + z[ii] * z[ii]).sqrt();
                    if r > 0.0 {
                        bx[ii] = beta_mag * x[ii] / r;
                        by[ii] = beta_mag * y[ii] / r;
                        bz[ii] = beta_mag * z[ii] / r;
                    }
                }
                (bx, by, bz)
            }
        };

    let mut events = Vec::with_capacity(n);
    for ii in 0..n {
        events.push(PhotonEvent {
            t_emission: t_emission[ii],
            x: x[ii],
            y: y[ii],
            z: z[ii],
            nu_emit: frequencies[ii],
            energy_weight: energy[ii],
            px: px[ii],
            py: py[ii],
            pz: pz[ii],
            stokes_i: stokes_i[ii],
            stokes_q: stokes_q[ii],
            stokes_u: stokes_u[ii],
            stokes_v: stokes_v[ii],
            doppler_factor: doppler[ii],
            beta_vec: [beta_x[ii], beta_y[ii], beta_z[ii]],
            optical_depth: optical_depth[ii],
            cell_id: cell_id[ii] as u32,
            absorbed: absorbed[ii],
            n_scatter: n_scatter[ii] as u32,
        });
    }
    Ok(PhotonEvents { events })
}

/// reduce a catalog into an observer light curve F_nu(t): per-frequency flux density [mJy]
/// over the observer-time bins, via the EATS. additive across catalogs, so a multi-checkpoint
/// light curve STREAMS — reduce each checkpoint's events into the bins and discard them, never
/// holding the full event set. returns (bin-center times [day], fluxes flat [n_time*n_freq],
/// frequencies [Hz]).
#[pyfunction]
#[pyo3(name = "lightcurve_from_events")]
#[pyo3(signature = (events, observer_direction, frequencies, redshift, luminosity_distance, time_bins, doppler_power=3.0, frac_bandwidth=0.1))]
fn lightcurve_from_events_py(
    events: &PhotonEvents,
    observer_direction: Vec<f64>,
    frequencies: Vec<f64>,
    redshift: f64,
    luminosity_distance: f64,
    time_bins: Vec<f64>,
    doppler_power: f64,
    frac_bandwidth: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if observer_direction.len() != 3 {
        return Err(PyValueError::new_err(
            "observer_direction must be a length-3 unit vector",
        ));
    }
    let nhat = [observer_direction[0], observer_direction[1], observer_direction[2]];
    let lc = compute_lightcurve_from_events(
        &events.events,
        nhat,
        &frequencies,
        redshift,
        luminosity_distance,
        &time_bins,
        doppler_power,
        frac_bandwidth,
    );
    Ok((lc.times, lc.fluxes, lc.frequencies))
}

/// DETERMINISTIC (noise-free) sky-map deposition for a 1d spherical blast — Zrake+2018 eq. 1-2.
/// deposits each cell's lab-frame monochromatic emissivity onto the sky plane (no photon sampling),
/// gated by the EATS window. `sim_cond["dt"]` carries the snapshot's lab-time interval (the same
/// trapezoidal weight the photon generator uses), `obs_time`/`time_window` are in DAYS, and
/// `half_width` [cm] is the caller-fixed image extent (so frames over many snapshots share a grid).
/// returns the raw deposit image (row-major `[iy*n_pix+ix]`); the caller calibrates to mJy/mas^2.
#[pyfunction]
#[pyo3(name = "skymap_deposit_spherical")]
#[pyo3(signature = (sim_cond, qscales, fields, mesh, observer_direction, obs_time, time_window,
                    frequency, redshift, half_width, n_pix=256, doppler_power=2.0))]
#[allow(clippy::too_many_arguments)]
fn skymap_deposit_spherical_py(
    sim_cond: &Bound<'_, PyDict>,
    qscales: &Bound<'_, PyDict>,
    fields: &Bound<'_, PyDict>,
    mesh: &Bound<'_, PyDict>,
    observer_direction: Vec<f64>,
    obs_time: f64,
    time_window: f64,
    frequency: f64,
    redshift: f64,
    half_width: f64,
    n_pix: usize,
    doppler_power: f64,
) -> PyResult<Vec<f64>> {
    if observer_direction.len() != 3 {
        return Err(PyValueError::new_err(
            "observer_direction must be a length-3 unit vector",
        ));
    }
    let cond = sim_conditions(sim_cond)?;
    let scales = quant_scales(qscales)?;
    let rho = req_field(fields, "rho")?;
    let gamma_beta = req_field(fields, "gamma_beta")?;
    let pre = req_field(fields, "pre")?;
    if rho.len() != gamma_beta.len() || rho.len() != pre.len() {
        return Err(PyValueError::new_err(
            "fields rho / gamma_beta / pre must have equal length",
        ));
    }
    let hf = HydroFields { rho: &rho, gamma_beta: &gamma_beta, pre: &pre };
    let md = mesh_data(mesh)?;

    const SECONDS_PER_DAY: f64 = 86_400.0;
    let nhat = [observer_direction[0], observer_direction[1], observer_direction[2]];
    let emit_dt_s = (cond.dt * scales.time).value();
    let obs_time_s = obs_time * SECONDS_PER_DAY;
    let half_window_s = 0.5 * time_window * SECONDS_PER_DAY;

    Ok(compute_skymap_deposit_spherical(
        &cond, &scales, &hf, &md.x1, nhat, obs_time_s, half_window_s, frequency, redshift,
        doppler_power, n_pix, half_width, emit_dt_s, std::f64::consts::PI, 256, 50,
    ))
}

/// register the afterglow pyfunctions + the event-handle class on the parent module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PhotonEvents>()?;
    m.add_function(wrap_pyfunction!(generate_photon_events_py, m)?)?;
    m.add_function(wrap_pyfunction!(monte_carlo_radiative_transfer_py, m)?)?;
    m.add_function(wrap_pyfunction!(skymap_from_events_py, m)?)?;
    m.add_function(wrap_pyfunction!(lightcurve_from_events_py, m)?)?;
    m.add_function(wrap_pyfunction!(write_photon_events_py, m)?)?;
    m.add_function(wrap_pyfunction!(read_photon_events_py, m)?)?;
    m.add_function(wrap_pyfunction!(photon_event_count_py, m)?)?;
    m.add_function(wrap_pyfunction!(read_photon_events_chunk_py, m)?)?;
    m.add_function(wrap_pyfunction!(skymap_deposit_spherical_py, m)?)?;
    Ok(())
}
