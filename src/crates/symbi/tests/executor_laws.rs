// =============================================================================
// executor_laws.rs
//
// the allocation-free executor law: the steady-state step loop performs
// no per-call bulk allocation. kernels, scratch fields, and dispatch tables are
// allocated once at attach / first dispatch; after warmup, a step's allocator
// traffic is bounded by a small named budget (per-dispatch kernel-name strings,
// smallvec spills, reduction partials). a regression that allocates per cell or
// per call — a fresh scratch field, a cloned buffer — multiplies the count and
// fails the law with the measured number attached.
//
// the counting allocator is process-global, so this binary holds ONLY this law.
// =============================================================================

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection};
use symbi_xpu::{CpuSpace, HostMemory};

static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

// system allocator wrapped with call/byte counters; the law reads the deltas.
struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const GAMMA: f64 = 1.4;
const N: usize = 32;
const L: f64 = 1.0;

#[test]
fn e2_steady_state_step_loop_allocation_is_bounded() {
    type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let dx = 2.0 * L / N as f64;
    // a central gravitating + accreting body drives the full dispatch surface:
    // godunov, cfl, ghost fill, body source, AND both body-feedback passes
    // (gravity reaction + drain) with their workspace-cached scratch.
    let bodies = BodyCollection::new().add(Body::black_hole(
        0,
        Tensor::new([0.0, 0.0]),
        Tensor::zeros(),
        1.0,  // mass
        0.1,  // radius
        0.2,  // softening
        0.5,  // sink rate
        0.0,  // sink delta
        0.15, // accretion radius
    ));
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim construction failed")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.0, 0.0]),
            pre: 1.0,
        })
        .build()
        .with_bodies(bodies);
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sim.geom.allocated);

    // warmup: first steps build kernels, size the feedback scratch, and grow
    // every lazily-initialized table. the law measures the steady state after.
    evolve(&mut sim, &sub, 0.05).expect("warmup evolve failed");
    let warm_iter = sim.iteration;
    assert!(warm_iter >= 2, "warmup must take steps (got {warm_iter})");

    let calls_before = ALLOC_CALLS.load(Ordering::Relaxed);
    let bytes_before = ALLOC_BYTES.load(Ordering::Relaxed);
    evolve(&mut sim, &sub, 0.30).expect("measured evolve failed");
    let steps = sim.iteration - warm_iter;
    assert!(steps >= 4, "measurement window too short ({steps} steps)");
    let calls_per_step = (ALLOC_CALLS.load(Ordering::Relaxed) - calls_before) / steps;
    let bytes_per_step = (ALLOC_BYTES.load(Ordering::Relaxed) - bytes_before) / steps;
    println!(
        "e2: {calls_per_step} allocator calls/step, {bytes_per_step} bytes/step over {steps} steps"
    );

    // the named budget, measured at 126 calls / 15.3 KB per step (deterministic
    // across runs). the residue is small-object traffic: per-dispatch kernel-name
    // strings and binding smallvecs, reduction partials, body scalar vecs — each
    // O(bytes), none O(grid). the bounds carry ~2x headroom. bytes is the
    // sensitive axis for field-scale regressions: re-allocating the 7-field
    // feedback scratch per call measures 89 KB/step on this grid and fails the
    // byte bound; a per-cell or per-face allocation blows both bounds.
    assert!(
        calls_per_step <= 250,
        "e2 violated: {calls_per_step} allocator calls/step (budget 250) — \
         a dispatch-path allocation regressed",
    );
    assert!(
        bytes_per_step <= 30_000,
        "e2 violated: {bytes_per_step} bytes/step (budget 30000) — \
         something grid-sized is being allocated per call",
    );
}
