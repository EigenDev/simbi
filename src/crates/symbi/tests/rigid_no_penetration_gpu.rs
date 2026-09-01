// =============================================================================
// rigid_no_penetration_gpu.rs
//
// the cartesian device twin of the rigid-wall gate: a runtime-JIT shaped rigid
// sphere (a CSG shape, routing the shaped kernel) on the cartesian chart, run on
// device memory and asserted bit-close to
// the CPU run. this exercises the cartesian bounding-ball bbox branch of the
// shaped-wall dispatch on device — the index-aligned support box, where the
// cylindrical twin routes to whole-interior dispatch. the per-body force receipt
// (a deterministic slab-order fold) lands the same value on both backends.
//
// runs on the host GPU (NVRTC needs no nvcc). run:
//   cargo test -p symbi --features cuda --test rigid_no_penetration_gpu
// =============================================================================
#![cfg(feature = "cuda")]

use symbi::regimes::substrate_kernels::dispatch_penalize;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ib::sdf::SdfExpr;
use symbi_ib::{Body, BodyCollection, SurfaceSpec};
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};
use symbi_xpu::{CpuSpace, ExecutionSpace, HostMemory, MemorySpace};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.3;
const N: usize = 64;
const L: f64 = 1.0;
const DX: f64 = 2.0 * L / N as f64;
const R_BODY: f64 = 0.25;
const V_INF: f64 = 0.3;

type Sim<S, Mem> = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, S, Mem>;

fn build<S: ExecutionSpace, Mem: MemorySpace>() -> Sim<S, Mem> {
    let sim = Sim::<S, Mem>::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([DX; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([V_INF, 0.0]),
            pre: 1.0,
        })
        .build();
    let mut sim = sim.with_bodies(
        BodyCollection::new().add(
            Body::rigid_sphere(
                0,
                Tensor::new([0.0, 0.0]),
                Tensor::new([0.0, 0.0]),
                1.0,
                R_BODY,
                0.1,
                false,
            )
            .with_surface(SurfaceSpec::Porous {
                porosity: 0.0,
                k_eta_n: 1.0e3,
                k_eta_t: 0.0,
            }),
        ),
    );
    // the CSG shape routes the runtime shaped kernel (host cranelift / device NVRTC), whose
    // cartesian bbox floors the bounding ball to an index box — the branch under test.
    sim.immersed.as_mut().unwrap().shapes[0] =
        Some(SdfExpr::<f64, 3>::sphere([0.0, 0.0, 0.0], R_BODY));
    sim
}

type HostSim = Sim<CpuSpace, HostMemory>;
type DevSim = Sim<CudaSpace, UnifiedMemory>;

fn cons_rel_gap(h: &HostSim, d: &DevSim) -> f64 {
    let mut gap = 0.0_f64;
    let cmp = |hf: &symbi_grid::Field<f64, 2, HostMemory>,
               df: &symbi_grid::Field<f64, 2, UnifiedMemory>,
               gap: &mut f64| {
        for c in h.geom.interior.iter() {
            let (a, b) = (*hf.view().at(c), *df.view().at(c));
            assert!(b.is_finite(), "non-finite device cons at {c:?}: {b}");
            *gap = gap.max((a - b).abs() / a.abs().max(1.0));
        }
    };
    cmp(&h.fields.cons.den, &d.fields.cons.den, &mut gap);
    for k in 0..2 {
        cmp(&h.fields.cons.mom[k], &d.fields.cons.mom[k], &mut gap);
    }
    let (hn, dn) = (
        h.fields.cons.nrg_field().unwrap(),
        d.fields.cons.nrg_field().unwrap(),
    );
    cmp(hn, dn, &mut gap);
    gap
}

#[test]
fn cartesian_shaped_wall_penalize_matches_cpu_on_device() {
    let h = build::<CpuSpace, HostMemory>();
    let d = build::<CudaSpace, UnifiedMemory>();
    dispatch_penalize(&h, 1e-3, GAMMA, 1.0, 3.0);
    dispatch_penalize(&d, 1e-3, GAMMA, 1.0, 3.0);
    symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();

    let gap = cons_rel_gap(&h, &d);
    assert!(
        gap < 1e-9,
        "cartesian shaped-wall cons host!=device: rel gap {gap:e}"
    );
    let hf = h.immersed.as_ref().unwrap().diagnostics.consolidate()[0].force_delta;
    let df = d.immersed.as_ref().unwrap().diagnostics.consolidate()[0].force_delta;
    // combined rel+abs tolerance against the force magnitude: the transverse component is a
    // numerical zero (mirror cancellation about the stream axis), where the host slab-order and
    // device block-order folds differ at round-off — bounded by the absolute floor.
    let scale = hf[0].abs().max(hf[1].abs());
    for k in 0..2 {
        let diff = (hf[k] - df[k]).abs();
        assert!(
            diff < 1e-9 * scale + 1e-11,
            "force receipt[{k}] host {} != device {} (abs diff {diff:e}, scale {scale:e})",
            hf[k],
            df[k],
        );
    }
}

#[test]
fn cartesian_shaped_wall_evolved_matches_cpu_on_device() {
    let mut h = build::<CpuSpace, HostMemory>();
    let mut d = build::<CudaSpace, UnifiedMemory>();
    let hk = AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, CFL, &h.geom.allocated);
    let dk =
        AdiabaticSubstrateKernelSet::<UnifiedMemory, f64, 2>::new(GAMMA, CFL, &d.geom.allocated);
    evolve(&mut h, &hk, 0.3).expect("host run");
    evolve(&mut d, &dk, 0.3).expect("device run");
    symbi::regimes::substrate_gpu::device_sync::<UnifiedMemory>();
    let gap = cons_rel_gap(&h, &d);
    assert!(
        gap < 1e-7,
        "evolved cartesian shaped-wall cons host!=device: rel gap {gap:e}"
    );
}
