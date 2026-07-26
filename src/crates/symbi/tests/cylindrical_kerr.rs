// =============================================================================
// cylindrical_kerr.rs
//
// the spinning-kerr cylindrical chart (R, phi, z), full 3D: the rank-1
// kerr-schild metric on the diag(1, R^2, 1) base with the frame dragging in
// the covariant l_phi. two oracle-free gates on an annular grid whose inner
// radius sits ABOVE the metric guard M/2 (no frozen clamped core on-grid, so
// the chart equivalences hold to roundoff everywhere):
// - a = 0 reduces the chart EXACTLY to the cylindrical kerr-schild one: the
//   evolved states agree to accumulated roundoff (different expression trees,
//   identical algebra);
// - the frame dragging is real and antisymmetric in the spin: a phi-uniform
//   (axisymmetric) state evolves identically in rho for +-a while the
//   azimuthal velocity develops NONZERO and flips sign exactly with a.
// =============================================================================

use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{KerrKSCylindrical, SchwarzschildKSCylindrical};
use symbi_hydro::Rhd;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 4.0 / 3.0;
const CFL: f64 = 0.3;
const NR: usize = 24;
const NP: usize = 16;
const NZ: usize = 24;
const R_LO: f64 = 1.0; // above the M/2 = 0.5 guard: no clamped cell on-grid
const R_HI: f64 = 6.0;
const Z_HALF: f64 = 3.0;
const MASS: f64 = 1.0;
const T_FINAL: f64 = 0.5;

type KerrSim = SimState<Rhd, 3, KerrKSCylindrical<f64>, IdealGas<f64>, CpuSpace, HostMemory>;
type KsSim = SimState<Rhd, 3, SchwarzschildKSCylindrical<f64>, IdealGas<f64>, CpuSpace, HostMemory>;

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

macro_rules! build_run {
    ($ty:ty, $metric:expr) => {{
        let dr = (R_HI - R_LO) / NR as f64;
        let dp = TWO_PI / NP as f64;
        let dz = 2.0 * Z_HALF / NZ as f64;
        let sim = <$ty>::build(Rhd, IdealGas { gamma: GAMMA }, $metric)
            .cells([NR, NP, NZ])
            .origin([R_LO, 0.0, -Z_HALF])
            .spacing([dr, dp, dz])
            .boundaries(Boundaries(std::array::from_fn(|a| {
                if a == 1 {
                    [BoundaryType::Periodic; 2]
                } else {
                    [BoundaryType::Outflow; 2]
                }
            })))
            .cfl(CFL)
            .timestepping(Timestepping::Rk2)
            .allocate()
            .expect("sim")
            .set_initial(|_| Prim {
                rho: 1.0,
                vel: Tensor::new([0.0; 3]),
                pre: 0.1,
            })
            .build();
        let kern =
            RhdSubstrateKernelSet::<HostMemory, f64, 3>::new(GAMMA, CFL, &sim.geom.allocated);
        let mut sim = sim;
        evolve(&mut sim, &kern, T_FINAL).expect("evolve");
        let mut den = Vec::new();
        let mut sphi = Vec::new();
        for c in sim.geom.interior.iter() {
            den.push(*sim.fields.cons.den.view().at(c));
            sphi.push(*sim.fields.cons.mom[1].view().at(c));
        }
        (den, sphi)
    }};
}

#[test]
fn zero_spin_reduces_to_the_kerr_schild_chart() {
    let (den_k, _) = build_run!(
        KerrSim,
        KerrKSCylindrical {
            mass: MASS,
            spin: 0.0
        }
    );
    let (den_s, _) = build_run!(KsSim, SchwarzschildKSCylindrical { mass: MASS });
    // non-vacuous: the infall genuinely developed.
    let dmax = den_k.iter().cloned().fold(0.0_f64, f64::max);
    assert!(dmax > 1.02, "no accretion developed (max den {dmax:.4})");
    // no clamped core on-grid: the two expression trees agree to roundoff EVERYWHERE.
    let err = den_k
        .iter()
        .zip(&den_s)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        err < 1e-11,
        "a=0 cylindrical kerr vs kerr_schild: max den err {err:e}"
    );
}

#[test]
fn frame_dragging_is_real_and_antisymmetric_in_the_spin() {
    let (den_p, sphi_p) = build_run!(
        KerrSim,
        KerrKSCylindrical {
            mass: MASS,
            spin: 0.9
        }
    );
    let (den_m, sphi_m) = build_run!(
        KerrSim,
        KerrKSCylindrical {
            mass: MASS,
            spin: -0.9
        }
    );
    // a phi-uniform state stays discretely axisymmetric, and the metric's only
    // odd-in-a piece is the azimuthal l: rho evolves IDENTICALLY for +-a.
    let derr = den_p
        .iter()
        .zip(&den_m)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(derr < 1e-12, "rho not spin-even: {derr:e}");
    // the dragging is real: the azimuthal covariant momentum develops nonzero...
    let smax = sphi_p.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(
        smax > 1e-8,
        "no frame dragging developed (max |S_phi| {smax:e})"
    );
    // ...and flips sign exactly with the spin.
    let aerr = sphi_p
        .iter()
        .zip(&sphi_m)
        .map(|(a, b)| (a + b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        aerr < 1e-12 * smax.max(1.0),
        "S_phi not spin-odd: {aerr:e} (scale {smax:e})"
    );
}
