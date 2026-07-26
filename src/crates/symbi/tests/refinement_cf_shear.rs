// =============================================================================
// refine_cf_shear.rs
//
// coarse-fine SUPERSONIC TANGENTIAL SHEAR gate, one test per regime
// (newtonian, isothermal, nmhd, imhd, rmhd): a kinetic-dominated parallel
// shear flow vx(y) = V0 sin(2 pi y) with uniform rho, p, and uniform B along
// x — an exact equilibrium in EVERY regime (all flux divergences vanish for
// parallel flow; v x B = 0 kills the induction) — crossing a static nested
// box, held for 50 root steps at full cfl on a fully periodic domain. the CF
// y-faces at +/-0.25 carry the peak tangential speed (mach ~8 for the
// non-relativistic regimes, lorentz-boosted mach ~10 for rmhd), the shear
// layer at y = 0 sits inside the fine level, and the CF x-faces see
// supersonic normal through-flow. the only forcing is the interface
// truncation itself, so the gate pins the property production runs depend
// on: sustained supersonic shear along a coarse-fine face must not
// destabilize the interface coupling in any regime (the adiabatic regimes
// run ~97% kinetic-energy dominated — the energy-equation cancellation mode
// is armed; the isothermal regimes pin the advection/CT coupling).
//
// this is the gate family member the binary-disk investigation found
// missing: the conservation/divb gates push waves THROUGH interfaces; none
// holds kinetic-dominated flow ALONG one.
//
// usage:
//  cargo test -p symbi --release --test refine_cf_shear
// =============================================================================

use std::f64::consts::TAU;
use std::sync::atomic::Ordering;

use symbi::regimes::substrate::IsoSubstrateKernelSet;
use symbi::regimes::substrate_isothermal_mhd::IsothermalMhdSubstrateKernelSet3D;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet3D;
use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet3D;
use symbi::sim::evolve::KernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::{Eos, IdealGas, Isothermal};
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::isothermal_mhd::IsothermalMhd;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::regime::Regime;
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::{Cons, Prim};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const N: usize = 16;
const CS: f64 = 0.25;
const V0: f64 = 2.0;
const STEPS: u64 = 50;

type Sim<R, E> = SimState<R, 3, Cartesian, E, CpuSpace, HostMemory>;
type Hier<R, E, K> = Hierarchy<R, 3, 3, Cartesian, E, CpuSpace, HostMemory, K>;

fn vx(y: f64) -> f64 {
    V0 * (TAU * y).sin()
}

/// a 2-level periodic hierarchy on [-0.5, 0.5)^3 with the middle half
/// refined, both levels filled by `fill` (absolute coordinates — the same
/// closure seeds both).
fn two_level<R, E, K>(
    regime: R,
    eos: E,
    make_kernels: impl Fn(&Sim<R, E>) -> K,
    fill: impl Fn(&Sim<R, E>),
) -> Hier<R, E, K>
where
    R: Regime<f64, 3> + Copy,
    E: Eos<f64> + Copy + Send + Sync,
    K: KernelSet<3, 3, HostMemory, f64>,
{
    let dx = 1.0 / N as f64;
    let coarse = Sim::build(regime, eos, Cartesian)
        .cells([N; 3])
        .origin([-0.5; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .finish()
        .unwrap();
    fill(&coarse);
    let ck = make_kernels(&coarse);
    let regions = [RefinementRegion {
        x_lo: [-0.25; 3],
        x_hi: [0.25; 3],
    }];
    let hier =
        Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, &make_kernels).unwrap();
    fill(&hier.levels[1].state);
    hier
}

/// uniform staggered + cell B along x: exactly div-free, coarse-fine
/// consistent, and force-free for parallel flow.
fn fill_uniform_bx<R, E>(sim: &Sim<R, E>, b0: f64)
where
    R: Regime<f64, 3>,
    E: Eos<f64>,
{
    let mhd = sim
        .fields
        .mhd
        .as_ref()
        .expect("mhd regime must allocate mhd fields");
    for c in &sim.geom.interior.extend(0, 0, 1) {
        mhd.bface[0].view_mut().set(c, b0);
    }
    for aa in 1..3 {
        for c in &sim.geom.interior.extend(aa, 0, 1) {
            mhd.bface[aa].view_mut().set(c, 0.0);
        }
    }
    mhd.bface_initialized.store(true, Ordering::Relaxed);
    for c in sim.geom.interior.iter() {
        mhd.bcell[0].view_mut().set(c, b0);
        mhd.bcell[1].view_mut().set(c, 0.0);
        mhd.bcell[2].view_mut().set(c, 0.0);
    }
}

/// the equilibrium must stay a perturbation on every level: finite state,
/// bounded density, positive pressure (when the regime carries energy),
/// finite cell B (when magnetized).
fn assert_bounded<R, E, K>(hier: &Hier<R, E, K>, label: &str)
where
    R: Regime<f64, 3> + Copy,
    E: Eos<f64> + Copy + Send + Sync,
    K: KernelSet<3, 3, HostMemory, f64>,
{
    for (ll, lvl) in hier.levels.iter().enumerate() {
        for c in lvl.state.geom.interior.iter() {
            let den = *lvl.state.fields.cons.den.view().at(c);
            assert!(
                den.is_finite() && (0.2..5.0).contains(&den),
                "{label} L{ll}: density out of bounds at {c:?}: {den:e}"
            );
            if let Some(pre) = lvl.state.fields.prim.pre_field() {
                let p = *pre.view().at(c);
                assert!(
                    p.is_finite() && p > 0.0,
                    "{label} L{ll}: non-positive/non-finite pressure at {c:?}: {p:e}"
                );
            }
            if let Some(mhd) = lvl.state.fields.mhd.as_ref() {
                for aa in 0..3 {
                    let b = *mhd.bcell[aa].view().at(c);
                    assert!(b.is_finite(), "{label} L{ll}: non-finite B{aa} at {c:?}");
                }
            }
        }
    }
}

#[test]
fn newtonian_shear_tangent_to_cf_faces_stays_bounded() {
    let mut hier = two_level(
        Newtonian,
        IdealGas { gamma: GAMMA },
        |s| AdiabaticSubstrateKernelSet::<HostMemory, f64, 3>::new(GAMMA, CFL, &s.geom.allocated),
        |s| {
            let cnrg = s.fields.cons.nrg_field().unwrap();
            for c in s.geom.interior.iter() {
                let prim = Prim {
                    rho: 1.0,
                    vel: Tensor::new([vx(s.geom.centroid(c)[1]), 0.0, 0.0]),
                    pre: CS * CS / GAMMA,
                };
                let cons = Regime::to_conserved(&s.physics.regime, &s.physics.eos, &prim);
                s.fields.cons.den.view_mut().set(c, cons.den);
                for dd in 0..3 {
                    s.fields.cons.mom[dd].view_mut().set(c, cons.mom[dd]);
                }
                cnrg.view_mut().set(c, cons.nrg);
            }
        },
    );
    hier.evolve_steps(STEPS).unwrap();
    assert_bounded(&hier, "newtonian");
}

#[test]
fn isothermal_shear_tangent_to_cf_faces_stays_bounded() {
    let mut hier = two_level(
        IsoNewtonian,
        Isothermal { cs: CS },
        |s| IsoSubstrateKernelSet::<HostMemory, f64, 3>::new(CS, CFL, &s.geom.allocated),
        |s| {
            for c in s.geom.interior.iter() {
                s.fields.cons.den.view_mut().set(c, 1.0);
                s.fields.cons.mom[0]
                    .view_mut()
                    .set(c, vx(s.geom.centroid(c)[1]));
                s.fields.cons.mom[1].view_mut().set(c, 0.0);
                s.fields.cons.mom[2].view_mut().set(c, 0.0);
            }
        },
    );
    hier.evolve_steps(STEPS).unwrap();
    assert_bounded(&hier, "isothermal");
}

#[test]
fn nmhd_shear_tangent_to_cf_faces_stays_bounded() {
    // plasma beta = 1 at the uniform pressure: super-alfvenic mach-8 shear.
    let b0 = (2.0 * CS * CS / GAMMA).sqrt();
    let mut hier = two_level(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        |s| {
            NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(
                GAMMA,
                CFL,
                1.0,
                &s.geom.allocated,
            )
        },
        |s| {
            fill_uniform_bx(s, b0);
            for c in s.geom.interior.iter() {
                let prim = MhdPrim {
                    hydro: Prim {
                        rho: 1.0,
                        vel: Tensor::new([vx(s.geom.centroid(c)[1]), 0.0, 0.0]),
                        pre: CS * CS / GAMMA,
                    },
                    mag: Tensor::new([b0, 0.0, 0.0]),
                };
                let cons = s.physics.regime.to_conserved(&s.physics.eos, &prim);
                s.fields.cons.scatter(
                    c,
                    Cons {
                        den: cons.den,
                        mom: cons.mom,
                        nrg: cons.nrg,
                    },
                );
            }
        },
    );
    hier.evolve_steps(STEPS).unwrap();
    assert_bounded(&hier, "nmhd");
}

#[test]
fn imhd_shear_tangent_to_cf_faces_stays_bounded() {
    // plasma beta = 1 at the isothermal pressure rho * cs^2.
    let b0 = (2.0 * CS * CS).sqrt();
    let mut hier = two_level(
        IsothermalMhd,
        Isothermal { cs: CS },
        |s| {
            IsothermalMhdSubstrateKernelSet3D::<HostMemory, f64>::new(
                CS,
                CFL,
                1.0,
                &s.geom.allocated,
            )
        },
        |s| {
            fill_uniform_bx(s, b0);
            for c in s.geom.interior.iter() {
                s.fields.cons.den.view_mut().set(c, 1.0);
                s.fields.cons.mom[0]
                    .view_mut()
                    .set(c, vx(s.geom.centroid(c)[1]));
                s.fields.cons.mom[1].view_mut().set(c, 0.0);
                s.fields.cons.mom[2].view_mut().set(c, 0.0);
            }
        },
    );
    hier.evolve_steps(STEPS).unwrap();
    assert_bounded(&hier, "imhd");
}

#[test]
fn rmhd_shear_tangent_to_cf_faces_stays_bounded() {
    // v0 = 0.8c at p = 0.01: lorentz-boosted mach ~10, tau kinetic-dominated;
    // beta = 1 keeps the magnetization modest (no c2p stiffness, full coupling).
    let (v0, p0): (f64, f64) = (0.8, 0.01);
    let b0 = (2.0 * p0).sqrt();
    let mut hier = two_level(
        Rmhd,
        IdealGas { gamma: GAMMA },
        |s| RmhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, CFL, 1.0, &s.geom.allocated),
        |s| {
            fill_uniform_bx(s, b0);
            for c in s.geom.interior.iter() {
                let y = s.geom.centroid(c)[1];
                let prim = MhdPrim {
                    hydro: Prim {
                        rho: 1.0,
                        vel: Tensor::new([v0 * (TAU * y).sin(), 0.0, 0.0]),
                        pre: p0,
                    },
                    mag: Tensor::new([b0, 0.0, 0.0]),
                };
                let cons = s.physics.regime.to_conserved(&s.physics.eos, &prim);
                s.fields.cons.scatter(
                    c,
                    Cons {
                        den: cons.den,
                        mom: cons.mom,
                        nrg: cons.nrg,
                    },
                );
            }
        },
    );
    hier.evolve_steps(STEPS).unwrap();
    assert_bounded(&hier, "rmhd");
}
