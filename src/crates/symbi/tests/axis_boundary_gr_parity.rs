// =============================================================================
// axis_boundary_gr_parity.rs
//
// the axis boundary on a 2.5D spherical (r, theta) grid in curved spacetime. the valencia
// state stores contravariant coordinate components: across the pole the basis vector
// d/dtheta reverses while d/dr and d/dphi continue unchanged, so v^theta and B^theta are odd
// there and every other component, with the scalars, is even. the gates read the production
// ghost fills: the generic lattice fill on the schwarzschild chart (the hydro wedge, whose
// momentum carries the azimuthal lift), and the kerr fills of both the hydro and the MHD
// stacks, which continue the frame-dragging variable w = v^phi + q v^r with
// q = gamma_{r phi}/gamma_{phi phi} even in theta, so the v^phi ghost is the image value to
// roundoff.
// =============================================================================
use std::f64::consts::PI;
use symbi::prelude::KernelSet;
use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{KerrKS, Metric, SchwarzschildKS};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rhd::Rhd;
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::{Prim, PrimG};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim<M> = SimStateGeneric<Rmhd, 2, 3, M, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = RmhdSubstrateKernelSet<HostMemory, f64, 2>;
type SimH<M> = SimStateGeneric<Rhd, 2, 3, M, IdealGas<f64>, CpuSpace, HostMemory>;
type KernH = RhdSubstrateKernelSet<HostMemory, f64, 2>;

const GAMMA: f64 = 4.0 / 3.0;
const CFL: f64 = 0.3;
const R_LO: f64 = 3.0;
const R_HI: f64 = 5.0;
const N: [usize; 2] = [8, 8];

/// a smooth, asymmetric contravariant state with every component nonzero and every physical
/// speed well below light on the shell r in [3, 5].
fn probe_gas(r: f64, th: f64) -> PrimG<f64, 3> {
    Prim::adiabatic(
        Density(1.0 + 0.1 * r + 0.05 * th),
        Tensor::new([
            0.02 * th.sin() + 0.01 * r,
            0.004 * th + 0.001 * r,
            0.01 * th + 0.002 * r,
        ]),
        Pressure(0.1 + 0.01 * th),
    )
}

fn probe_state(r: f64, th: f64) -> MhdPrim<f64, 3> {
    MhdPrim::new(
        probe_gas(r, th),
        Tensor::new([0.01 + 0.002 * th, 0.002 * th, 0.03 * th + 0.004 * r]),
    )
}

/// the probe on one chart: axis at the pole, a reflecting equator, outflow radial walls. a
/// macro because the builder's seeding step is bound to each concrete metric.
macro_rules! probe_sim {
    ($metric:expr) => {{
        let sim = Sim::build(Rmhd, IdealGas { gamma: GAMMA }, $metric)
            .cells(N)
            .spacing([(R_HI - R_LO) / N[0] as f64, 0.5 * PI / N[1] as f64])
            .origin([R_LO, 0.0])
            .boundaries(Boundaries::per_axis([
                [BoundaryType::Outflow, BoundaryType::Outflow],
                [BoundaryType::Axis, BoundaryType::Reflect],
            ]))
            .cfl(CFL)
            .allocate()
            .expect("curved 2.5D construction")
            .set_initial(|[r, th]| probe_state(r, th))
            .seed_faces(|axis, [r, th]| match axis {
                0 => 0.01 + 0.002 * th,
                _ => 0.002 * th + 0.0001 * r,
            })
            .build();
        let k = Kern::new(GAMMA, CFL, 1.0, &sim.geom.allocated);
        (sim, k)
    }};
}

macro_rules! probe_gas_sim {
    ($metric:expr) => {{
        let sim = SimH::build(Rhd, IdealGas { gamma: GAMMA }, $metric)
            .cells(N)
            .spacing([(R_HI - R_LO) / N[0] as f64, 0.5 * PI / N[1] as f64])
            .origin([R_LO, 0.0])
            .boundaries(Boundaries::per_axis([
                [BoundaryType::Outflow, BoundaryType::Outflow],
                [BoundaryType::Axis, BoundaryType::Reflect],
            ]))
            .cfl(CFL)
            .allocate()
            .expect("curved 2.5D hydro construction")
            .set_initial(|[r, th]| probe_gas(r, th))
            .build();
        let k = KernH::new(GAMMA, CFL, &sim.geom.allocated);
        (sim, k)
    }};
}

/// the gas fields of a cell, with the magnetic slots zero, so one parity table serves both
/// stacks.
fn gas_cell<M: Metric<f64, 2> + Copy>(sim: &SimH<M>, c: [isize; 2]) -> [f64; 8] {
    let p = &sim.fields.prim;
    [
        *p.rho.view().at(c),
        *p.vel[0].view().at(c),
        *p.vel[1].view().at(c),
        *p.vel[2].view().at(c),
        *p.pre.as_ref().expect("adiabatic pressure").view().at(c),
        0.0,
        0.0,
        0.0,
    ]
}

fn cell<M: Metric<f64, 2> + Copy>(sim: &Sim<M>, c: [isize; 2]) -> [f64; 8] {
    let p = &sim.fields.prim;
    let m = sim.fields.mhd.as_ref().expect("mhd fields");
    [
        *p.rho.view().at(c),
        *p.vel[0].view().at(c),
        *p.vel[1].view().at(c),
        *p.vel[2].view().at(c),
        *p.pre.as_ref().expect("adiabatic pressure").view().at(c),
        *m.bcell[0].view().at(c),
        *m.bcell[1].view().at(c),
        *m.bcell[2].view().at(c),
    ]
}

const NAMES: [&str; 8] = [
    "rho", "v^r", "v^theta", "v^phi", "pre", "B^r", "B^theta", "B^phi",
];
/// contravariant parity across the pole: only the theta components reverse.
const POLE_SIGN: [f64; 8] = [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0];

/// every pole ghost row against its image; `slack` is the relative tolerance allowed on the
/// azimuthal slots (zero demands bit equality). `read` is the stack's cell reader; a slot
/// that vanishes on the first interior row (the hydro stack's magnetic slots) is skipped in
/// the liveness check.
fn check_pole(
    allocated: &symbi_algebra::Domain<2>,
    interior: &symbi_algebra::Domain<2>,
    read: impl Fn([isize; 2]) -> [f64; 8],
    slack: f64,
    live: &[usize],
    label: &str,
) -> usize {
    let jlo = interior.spaces[1].lo;
    let mut n = 0;
    for i in interior.spaces[0].lo..interior.spaces[0].hi {
        for j in allocated.spaces[1].lo..jlo {
            let g = read([i, j]);
            let s = read([i, 2 * jlo - 1 - j]);
            for k in 0..8 {
                let want = POLE_SIGN[k] * s[k];
                let azimuthal = k == 3 || k == 7;
                let ok = if g[k] == 0.0 && s[k] == 0.0 {
                    true
                } else if azimuthal && slack > 0.0 {
                    (g[k] - want).abs() <= slack * want.abs()
                } else {
                    g[k] == want
                };
                assert!(
                    ok,
                    "{label}: {} at ghost {:?} = {} vs {want} (image {})",
                    NAMES[k],
                    [i, j],
                    g[k],
                    s[k]
                );
            }
            n += 1;
        }
    }
    assert!(n >= 16, "{label}: the pole band was read ({n} cells)");
    let first = read([interior.spaces[0].lo, jlo]);
    for &k in live {
        assert!(
            first[k] != 0.0,
            "{label}: {} vanished on the first row",
            NAMES[k]
        );
    }
    n
}

const GAS_LIVE: [usize; 2] = [2, 3];
const MHD_LIVE: [usize; 4] = [2, 3, 6, 7];

#[test]
fn the_schwarzschild_pole_reverses_only_the_theta_components() {
    // the generic lattice fill under a curved spacetime: the azimuthal slot binds the even
    // sign, so v^phi copies bit for bit.
    let (sim, k) = probe_gas_sim!(SchwarzschildKS { mass: 1.0 });
    k.c2p(&sim);
    k.ghost_fill(&sim);
    let (a, i) = (sim.geom.allocated.clone(), sim.geom.interior.clone());
    check_pole(
        &a,
        &i,
        |c| gas_cell(&sim, c),
        0.0,
        &GAS_LIVE,
        "schwarzschild hydro axis",
    );
}

#[test]
fn the_kerr_hydro_pole_continues_the_dragging_variable_to_the_even_image() {
    let (sim, k) = probe_gas_sim!(KerrKS {
        mass: 1.0,
        spin: 0.6
    });
    k.c2p(&sim);
    k.ghost_fill(&sim);
    let (a, i) = (sim.geom.allocated.clone(), sim.geom.interior.clone());
    check_pole(
        &a,
        &i,
        |c| gas_cell(&sim, c),
        1e-12,
        &GAS_LIVE,
        "kerr hydro axis",
    );
}

#[test]
fn the_kerr_mhd_pole_continues_the_dragging_variable_to_the_even_image() {
    // q = gamma_{r phi}/gamma_{phi phi} is built from sin^2 and cos^2 of theta, so the ghost
    // centroid's q equals the image's and w - q v^r returns the image v^phi within roundoff;
    // the cell B^phi rides the same manifold.
    let (sim, k) = probe_sim!(KerrKS {
        mass: 1.0,
        spin: 0.6
    });
    k.c2p(&sim);
    k.ghost_fill(&sim);
    let (a, i) = (sim.geom.allocated.clone(), sim.geom.interior.clone());
    check_pole(&a, &i, |c| cell(&sim, c), 1e-12, &MHD_LIVE, "kerr mhd axis");
}
