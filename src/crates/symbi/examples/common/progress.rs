// =============================================================================
// progress.rs
//
// **B10 — live progress widget integration**. builds the (Category, Property,
// Value) row tables `symbi_display::Table::set_system_info` and
// `set_problem_setup` consume. one match per fact, gathered from:
//   • cfg-time backend selection (`feature = "cuda"` → GPU else CPU)
//   • `symbi_xpu::cuda::device_info()` (when cuda is built)
//   • `/proc/cpuinfo` + `/proc/meminfo` (Linux host info)
//   • `std::thread::available_parallelism()`
//   • the `SimState` (regime spec, metric, eos, geometry, timestepping, cfl).
//
// the resulting rows display once at the top of every example run; the
// benchmark sub-table updates per-refresh with iteration / time / dt / MZCS.
// =============================================================================

#![allow(dead_code)]

use std::fs;

use symbi::sim::state::{SimStateGeneric, Timestepping};
use symbi_geometry::{Geometry, Metric};
use symbi_hydro::eos::Eos;
use symbi_hydro::regime::Regime;
use symbi_xpu::{ExecutionSpace, MemorySpace};

// =============================================================================
// system info — backend-aware. cuda builds show the GPU; cpu builds show the
// CPU. linux-only host probing for now (the production env is Linux).
// =============================================================================

/// the (category, property, value) triples a `symbi_display::Table` consumes
/// via `set_system_info`. owned strings so the table can borrow them.
pub fn build_system_info_rows() -> Vec<[String; 3]> {
    let mut rows: Vec<[String; 3]> = Vec::new();

    // backend label — derived at compile time. matches what kernels actually run on.
    let backend: &str = if cfg!(feature = "cuda") { "GPU (CUDA)" } else { "CPU (host)" };
    rows.push(["Backend".to_string(), "Active".to_string(), backend.to_string()]);

    // host CPU info — cheap one-shot probe of /proc/cpuinfo.
    if let Some((model, _phys, _flags)) = cpu_info() {
        rows.push(["CPU".to_string(), "Model".to_string(), model]);
    }
    if let Ok(n) = std::thread::available_parallelism() {
        rows.push(["".to_string(), "Threads".to_string(), n.get().to_string()]);
    }
    if let Some(ram) = total_ram_bytes() {
        rows.push(["".to_string(), "RAM".to_string(), format_bytes(ram)]);
    }

    // GPU info — only attempt if built with cuda. on success this populates
    // device name + VRAM + compute capability + device count.
    #[cfg(feature = "cuda")]
    {
        if let Ok(info) = symbi_xpu::cuda::device_info() {
            rows.push(["GPU".to_string(), "Device".to_string(), info.name.clone()]);
            let (mj, mn) = info.compute_capability;
            rows.push(["".to_string(), "Compute".to_string(), format!("{mj}.{mn}")]);
            rows.push(["".to_string(), "VRAM".to_string(), format_bytes(info.total_memory_bytes)]);
            if info.device_count > 1 {
                rows.push(["".to_string(), "Device count".to_string(), info.device_count.to_string()]);
            }
        }
    }
    rows
}

fn cpu_info() -> Option<(String, Option<String>, Option<String>)> {
    let txt = fs::read_to_string("/proc/cpuinfo").ok()?;
    let mut model = None;
    let mut phys = None;
    let mut flags = None;
    for line in txt.lines() {
        if let Some(v) = line.strip_prefix("model name") {
            if model.is_none() {
                model = v.split(':').nth(1).map(|s| s.trim().to_string());
            }
        }
        if let Some(v) = line.strip_prefix("physical id") {
            if phys.is_none() {
                phys = v.split(':').nth(1).map(|s| s.trim().to_string());
            }
        }
        if let Some(v) = line.strip_prefix("flags") {
            if flags.is_none() {
                flags = v.split(':').nth(1).map(|s| s.trim().to_string());
            }
        }
    }
    Some((model.unwrap_or_else(|| "unknown".into()), phys, flags))
}

fn total_ram_bytes() -> Option<u64> {
    let txt = fs::read_to_string("/proc/meminfo").ok()?;
    for line in txt.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let n: u64 = rest.trim().split_whitespace().next()?.parse().ok()?;
            return Some(n * 1024); // kB → bytes
        }
    }
    None
}

/// "1.5 GB", "256 KB", "8 B".
pub fn format_bytes(n: u64) -> String {
    const KB: u64 = 1_024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if      n >= GB { format!("{:.1} GB", n as f64 / GB as f64) }
    else if n >= MB { format!("{:.1} MB", n as f64 / MB as f64) }
    else if n >= KB { format!("{:.1} KB", n as f64 / KB as f64) }
    else            { format!("{n} B") }
}

// =============================================================================
// problem setup — every fact that's known up front and stays fixed.
// =============================================================================

pub fn build_problem_setup_rows<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim:          &SimStateGeneric<R, D, DOF, M, E, S, Mem, f64>,
    solver_label: &str,
    recon_label:  &str,
) -> Vec<[String; 3]>
where
    R:   Regime<f64, D>,
    M:   Metric<f64, D> + Copy,
    E:   Eos<f64>,
    S:   ExecutionSpace,
    Mem: MemorySpace,
{
    let mut rows: Vec<[String; 3]> = Vec::new();

    // regime + geometry from the static specs.
    rows.push(["Physics".to_string(), "Regime".to_string(), R::SPEC.name.to_string()]);
    rows.push(["".to_string(), "Coordinates".to_string(),
        geometry_name(sim.physics.metric.geometry()).to_string()]);
    rows.push(["".to_string(), "EOS γ".to_string(), format!("{:.4}", sim.physics.eos.gamma())]);

    // grid (size + ghost halos).
    let dims: Vec<String> = (0..D).map(|ax| sim.geom.interior.spaces[ax].size().to_string()).collect();
    rows.push(["Grid".to_string(), "Dimensions".to_string(), format!("{}D", D)]);
    rows.push(["".to_string(), "Resolution".to_string(), dims.join(" × ")]);
    rows.push(["".to_string(), "Cells (interior)".to_string(),
        format_cell_count(sim.geom.interior.volume() as u64)]);
    rows.push(["".to_string(), "Halos".to_string(), sim.geom.ng.to_string()]);

    // numerics.
    rows.push(["Numerics".to_string(), "Riemann".to_string(), solver_label.to_string()]);
    rows.push(["".to_string(), "Reconstruction".to_string(), recon_label.to_string()]);
    rows.push(["".to_string(), "Timestepping".to_string(),
        timestepping_name(sim.timestepping).to_string()]);
    rows.push(["".to_string(), "CFL".to_string(), format!("{:.3}", sim.cfl)]);

    rows
}

fn geometry_name(g: Geometry) -> &'static str {
    match g {
        Geometry::Cartesian   => "cartesian",
        Geometry::Spherical   => "spherical",
        Geometry::Cylindrical => "cylindrical",
    }
}

fn timestepping_name(t: Timestepping) -> &'static str {
    match t {
        Timestepping::Euler => "forward-Euler",
        Timestepping::Rk2   => "SSP-RK2 (Heun)",
        Timestepping::Rk3   => "SSP-RK3 (Shu-Osher)",
    }
}

fn format_cell_count(n: u64) -> String {
    if      n >= 1_000_000_000 { format!("{:.2} B", n as f64 / 1e9) }
    else if n >= 1_000_000     { format!("{:.2} M", n as f64 / 1e6) }
    else if n >= 1_000         { format!("{:.1} K", n as f64 / 1e3) }
    else                       { n.to_string() }
}
