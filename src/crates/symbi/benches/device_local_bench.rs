// =============================================================================
// device_local_bench.rs
//
// managed against device-local storage on identical inputs: the 2d entropy wave (periodic,
// gamma = 5/3, plm + hlle + ssp-rk2) at a fixed pinned step, mirrored from one host state into
// each space and evolved by the same kernel set and driver. per space it reports:
// - steady step time: the median of `SYMBI_BENCH_REPS` timed segments of `SYMBI_BENCH_STEPS`
//   steps each, after a warm-up segment that compiles and caches every kernel
// - transfer volume: the bytes the initial mirror uploads, and the bytes one checkpoint
//   download moves (zero for managed storage, which the writer reads in place)
// - checkpoint cost: the wall time of one checkpoint, device-local including its download
// the final states of the two spaces are compared bit for bit. on the cpu build the spaces are
// host memory and its opaque stand-in, both run by the cpu kernels: the numbers check the
// harness, not a device. one json line per space, then one comparison line.
//
// usage:
//   cargo bench -p symbi --bench device_local_bench
//   SYMBI_BENCH_N=1024 cargo bench -p symbi --features hip --bench device_local_bench
// =============================================================================

use std::time::Instant;

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::device_local::{download_state, mirror_sim, resident_bytes, state_bytes};
use symbi_xpu::{HostMemory, MemoryStorage};

#[cfg(not(feature = "gpu"))]
mod spaces {
    pub type Space = symbi_xpu::CpuSpace;
    pub type Managed = symbi_xpu::HostMemory;
    pub type Local = symbi_xpu::HostOpaque;
}
#[cfg(feature = "hip")]
mod spaces {
    pub type Space = symbi_xpu::HipSpace;
    pub type Managed = symbi_xpu::HipManaged;
    pub type Local = symbi_xpu::HipDeviceLocal;
}
use spaces::{Local, Managed, Space};

type HostSim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, Space, HostMemory>;
type SpaceSim<Mem> = SimStateGeneric<Newtonian, 2, 2, Cartesian, IdealGas<f64>, Space, Mem>;

const GAMMA: f64 = 5.0 / 3.0;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn wave(n: usize, pinned_dt: f64) -> HostSim {
    let dx = 1.0 / n as f64;
    let mut sim = HostSim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n, n])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.4)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim construction")
        .set_initial(|[x, _]| {
            let rho = 1.0 + 0.2 * (std::f64::consts::TAU * x).sin();
            Prim::adiabatic(Density(rho), Tensor::new([1.0, 0.0]), Pressure(1.0))
        })
        .build();
    sim.store.max_dt = pinned_dt;
    sim
}

struct Measured {
    step_seconds: f64,
    upload_bytes: usize,
    checkpoint_download_bytes: usize,
    checkpoint_seconds: f64,
    state: Vec<Vec<f64>>,
}

fn state_of<Mem: MemoryStorage>(sim: &SpaceSim<Mem>) -> Vec<Vec<f64>> {
    let (c, p) = (&sim.fields.cons, &sim.fields.prim);
    let mut fields = vec![&c.den, &c.mom[0], &c.mom[1], &p.rho, &p.vel[0], &p.vel[1]];
    fields.extend(c.nrg.iter());
    fields.extend(p.pre.iter());
    fields.iter().map(|f| f.download().expect("download")).collect()
}

/// the steady step time of `sim` and the cost of one checkpoint written through `checkpoint`
fn measure<Mem: MemoryStorage>(
    mut sim: SpaceSim<Mem>,
    pinned_dt: f64,
    steps: usize,
    reps: usize,
    mut checkpoint: impl FnMut(&SpaceSim<Mem>) -> usize,
) -> Measured {
    let upload_bytes = resident_bytes(&sim.store);
    let kernels = AdiabaticSubstrateKernelSet::<Mem, f64, 2>::new(GAMMA, 0.4, &sim.geom.allocated);
    let mut t = sim.time;
    t += steps as f64 * pinned_dt;
    evolve(&mut sim, &kernels, t).expect("the warm-up segment");
    let mut segments: Vec<f64> = (0..reps)
        .map(|_| {
            t += steps as f64 * pinned_dt;
            let start = Instant::now();
            evolve(&mut sim, &kernels, t).expect("a timed segment");
            start.elapsed().as_secs_f64() / steps as f64
        })
        .collect();
    segments.sort_by(f64::total_cmp);
    let start = Instant::now();
    let checkpoint_download_bytes = checkpoint(&sim);
    let checkpoint_seconds = start.elapsed().as_secs_f64();
    Measured {
        step_seconds: segments[reps / 2],
        upload_bytes,
        checkpoint_download_bytes,
        checkpoint_seconds,
        state: state_of(&sim),
    }
}

fn report(space: &str, n: usize, m: &Measured) {
    println!(
        "{{\"space\":\"{space}\",\"n\":{n},\"step_seconds\":{:.6e},\"mzcps\":{:.3},\"upload_bytes\":{},\
         \"checkpoint_download_bytes\":{},\"checkpoint_seconds\":{:.6e}}}",
        m.step_seconds,
        (n * n) as f64 / m.step_seconds / 1e6,
        m.upload_bytes,
        m.checkpoint_download_bytes,
        m.checkpoint_seconds
    );
}

fn main() {
    let n = env_usize("SYMBI_BENCH_N", 256);
    let steps = env_usize("SYMBI_BENCH_STEPS", 20);
    let reps = env_usize("SYMBI_BENCH_REPS", 5);
    // a power of two at most 0.05 / n, below the cfl step of the wave (about 0.1 / n), so both
    // spaces take the same steps
    let pinned_dt = 2f64.powi((0.05 / n as f64).log2().floor() as i32);
    let dir = std::env::temp_dir().join(format!("symbi_device_local_bench_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("bench directory");
    let source = wave(n, pinned_dt);

    let managed_path = dir.join("managed.h5");
    let managed = mirror_sim::<_, 2, 2, _, _, _, HostMemory, Managed>(&source, 0).expect("the managed mirror");
    let managed = measure(managed, pinned_dt, steps, reps, |sim| {
        symbi_sim::checkpoint::write_checkpoint(sim, managed_path.to_str().unwrap(), &symbi_io::Metadata::new())
            .expect("the managed checkpoint");
        0
    });
    report("managed", n, &managed);

    let local_path = dir.join("local.h5");
    let local = mirror_sim::<_, 2, 2, _, _, _, HostMemory, Local>(&source, 0).expect("the device-local mirror");
    let mut staging = wave(n, pinned_dt);
    let local = measure(local, pinned_dt, steps, reps, |sim| {
        download_state(&sim.store, &mut staging.store).expect("the checkpoint download");
        symbi_sim::checkpoint::write_checkpoint(&staging, local_path.to_str().unwrap(), &symbi_io::Metadata::new())
            .expect("the device-local checkpoint");
        state_bytes(&sim.store)
    });
    report("device_local", n, &local);

    let identical = managed.state.iter().zip(&local.state).all(|(a, b)| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
    println!(
        "{{\"n\":{n},\"pinned_dt\":{pinned_dt:e},\"steps_per_segment\":{steps},\"segments\":{reps},\
         \"step_time_ratio_local_over_managed\":{:.4},\"final_states_bit_identical\":{identical}}}",
        local.step_seconds / managed.step_seconds
    );
    let _ = std::fs::remove_dir_all(&dir);
    assert!(identical, "managed and device-local runs diverged on identical inputs");
}
