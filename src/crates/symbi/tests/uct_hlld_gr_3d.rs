// =============================================================================
// uct_hlld_gr_3d.rs
//
// the 3D cartesian GR UCT-HLLD chain, end to end through the production RMHD
// substrate: the wave-sum corner EMFs on the kerr-schild charts dispatch for
// all three edge orientations and the staggered face div(B) stays at machine
// zero under evolve (the curl-form update telescopes regardless of the
// densitization convention, so a discretely div-free IC must stay div-free).
// gates:
// - schwarzschild KS: a magnetized swirl in a box OUTSIDE the horizon
//   (r > 2M everywhere) evolves under UCT-HLLD with div(B) preserved;
// - spinning kerr: the same contract on the a != 0 chart (nonzero shift
//   enters the transport velocity and the moving-interface fan speeds);
// - the M -> 0 oracle: the kerr-schild metric at zero mass IS minkowski
//   (alpha = 1, beta = 0, gamma = delta, tetrad = identity), so the GR
//   UCT-HLLD run must match the FLAT UCT-HLLD run (gated in uct_hlld_3d.rs)
//   on identical initial data to roundoff.
// =============================================================================

use std::f64::consts::PI;

use symbi::regimes::substrate_kernels::Solver;
use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet3D;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{Cartesian, KerrKSCartesian, SchwarzschildKSCartesian};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 8;
const GAMMA: f64 = 4.0 / 3.0;
const CFL: f64 = 0.3;
const MASS: f64 = 0.2;
const B0: f64 = 0.1;
const T_FINAL: f64 = 0.25;
const DIVB_TOL: f64 = 1e-12;

// the conserved discrete object on a curved chart is the DENSITIZED face flux
// sqrt(gamma)|face * B: the GR curl updates bface by curl(Etilde)/sqrt(gamma),
// so the raw-B divergence drifts with the metric gradient while the densitized
// one telescopes exactly. sqrt(gamma) is evaluated at each face's own center
// (face on its axis, cell-centered transversely) via the SAME carrier-generic
// `Metric::sqrt_det_gamma` the kernel traces. flat: sqrt(gamma) = 1, the
// ordinary staggered divergence.
fn max_divb<M>(
    sim: &SimState<Rmhd, 3, M, IdealGas<f64>, CpuSpace, HostMemory>,
    metric: &M,
    x_lo: f64,
    inv_d: [f64; 3],
) -> (f64, f64)
where
    M: symbi_geometry::Metric<f64, 3>,
{
    let mhd = sim.fields.mhd.as_ref().expect("mhd");
    let dx = 1.0 / inv_d[0];
    let ilo: [isize; 3] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
    let face = |i: isize| x_lo + i as f64 * dx;
    let center = |i: isize| (face(i) + face(i + 1)) * 0.5;
    let face_pos = |c: [isize; 3], a: usize, hi: bool| -> Tensor<f64, 3> {
        let idx: [isize; 3] = std::array::from_fn(|d| c[d] - ilo[d]);
        Tensor::new(std::array::from_fn(|d| {
            if d == a {
                face(idx[d] + if hi { 1 } else { 0 })
            } else {
                center(idx[d])
            }
        }))
    };
    let (mut max_div, mut max_b) = (0.0_f64, 0.0_f64);
    for c in sim.geom.interior.iter() {
        let mut div = 0.0_f64;
        for a in 0..3 {
            let b_lo = *mhd.bface[a].view().at(c);
            let mut n = c;
            n[a] += 1;
            let b_hi = *mhd.bface[a].view().at(n);
            let g_lo = metric.sqrt_det_gamma(face_pos(c, a, false));
            let g_hi = metric.sqrt_det_gamma(face_pos(c, a, true));
            div += (g_hi * b_hi - g_lo * b_lo) * inv_d[a];
            max_b = max_b.max(b_lo.abs());
        }
        max_div = max_div.max(div.abs());
    }
    (max_div, max_b)
}

// a discretely div-free magnetized swirl: uniform B_x on the faces (zero face
// divergence by construction), a gentle position-dependent velocity so every
// edge EMF carries nonzero curved-metric physics.
fn swirl_prim(x: f64, y: f64, z: f64) -> MhdPrim<f64, 3> {
    let s = 2.0 * PI;
    MhdPrim {
        hydro: Prim {
            rho: 1.0,
            vel: Tensor::new([
                0.1 * (s * y).sin(),
                0.1 * (s * z).sin(),
                0.1 * (s * x).sin(),
            ]),
            pre: 0.5,
        },
        mag: Tensor::new([B0, 0.0, 0.0]),
    }
}

macro_rules! gr_divb_gate {
    ($metric:expr, $metric_ty:ty, $x_lo:expr, $t_final:expr, $min_steps:expr, $what:literal) => {{
        type Sim = SimState<Rmhd, 3, $metric_ty, IdealGas<f64>, CpuSpace, HostMemory>;
        let dx = 1.0 / N as f64;
        // seed the DENSITIZED flux uniform: bface = B0 / sqrt(gamma)(face), so
        // the densitized divergence is exactly zero at t = 0 (flat: sqrt = 1).
        let metric = $metric;
        let mut sim = Sim::build(Rmhd, IdealGas { gamma: GAMMA }, $metric)
            .cells([N; 3])
            .origin([$x_lo; 3])
            .spacing([dx; 3])
            .boundaries(Boundaries::uniform(BoundaryType::Outflow))
            .cfl(CFL)
            .allocate()
            .expect("gr sim")
            .set_initial(|[x, y, z]| swirl_prim(x, y, z))
            .seed_faces(|axis, [x, y, z]| {
                if axis == 0 {
                    B0 / symbi_geometry::Metric::<f64, 3>::sqrt_det_gamma(&metric, Tensor::new([x, y, z]))
                } else {
                    0.0
                }
            })
            .build();
        let inv_d = [N as f64; 3];
        let (div0, _) = max_divb(&sim, &metric, $x_lo, inv_d);
        assert!(div0 < 1e-13, "{}: densitized IC not div-free: {div0:e}", $what);
        let sub = RmhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, CFL, 1.0, &sim.geom.allocated)
            .with_solver(Solver::Hlld)
            .expect("hlld")
            .ct_method(CtMethod::Uct);
        let mut steps: u64 = 0;
        evolve_with_callback(&mut sim, &sub, $t_final, 1, |s| {
            let (max_div, max_b) = max_divb(s, &metric, $x_lo, inv_d);
            let rel = max_div / max_b.max(1.0);
            assert!(
                rel < DIVB_TOL,
                "{}: div(B) grew at iter {} t={:.3e}: rel {rel:e}",
                $what, s.iteration, s.time,
            );
            steps = s.iteration;
        })
        .expect("gr evolve");
        assert!(steps >= $min_steps, "{}: only {steps} steps — gate barely exercised", $what);
        sim
    }};
}

#[test]
fn schwarzschild_ks_3d_uct_hlld_preserves_divb() {
    // the box sits outside the horizon: x_lo = 1.2 puts every cell at
    // r >= sqrt(3) * 1.2 > 2M = 0.4.
    gr_divb_gate!(SchwarzschildKSCartesian { mass: MASS }, SchwarzschildKSCartesian<f64>, 1.2, T_FINAL, 5, "ks 3d uct-hlld");
}

#[test]
fn spinning_kerr_3d_uct_hlld_preserves_divb() {
    gr_divb_gate!(
        KerrKSCartesian { mass: MASS, spin: 0.5 },
        KerrKSCartesian<f64>,
        1.2,
        T_FINAL,
        5,
        "kerr 3d uct-hlld"
    );
}

#[test]
fn zero_mass_ks_3d_matches_flat_to_roundoff() {
    // M = 0 collapses the kerr-schild chart to minkowski exactly (the metric
    // factors evaluate to exact 1.0 / 0.0), so the GR UCT-HLLD chain must
    // reproduce the flat chain on identical initial data. the two kernels
    // assemble algebraically identical arithmetic through different f64
    // operation orders, so the comparison is roundoff-tight; the differing
    // operation orders rule out a bitwise match.
    // ONE step: t_final sits below both charts' CFL estimates, so the loop's
    // dt = min(cfl, t_final - t) clamp pins the SAME dt on both chains (the
    // GR and flat wave-speed maps are different, both valid, bounds — free
    // stepping would diverge in step count while agreeing on physics).
    const T_ONE: f64 = 0.005;
    let sim_gr = gr_divb_gate!(SchwarzschildKSCartesian { mass: 0.0 }, SchwarzschildKSCartesian<f64>, 1.2, T_ONE, 1, "ks M=0");

    type FlatSim = SimState<Rmhd, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let dx = 1.0 / N as f64;
    let mut flat = FlatSim::build(Rmhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .origin([1.2; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("flat sim")
        .set_initial(|[x, y, z]| swirl_prim(x, y, z))
        .seed_faces(|axis, _| if axis == 0 { B0 } else { 0.0 })
        .build();
    let sub = RmhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, CFL, 1.0, &flat.geom.allocated)
        .with_solver(Solver::Hlld)
        .expect("hlld")
        .ct_method(CtMethod::Uct);
    evolve_with_callback(&mut flat, &sub, T_ONE, 1, |_| {}).expect("flat evolve");

    assert_eq!(sim_gr.iteration, flat.iteration, "step counts diverged at M = 0");
    let mhd_g = sim_gr.fields.mhd.as_ref().expect("mhd");
    let mhd_f = flat.fields.mhd.as_ref().expect("mhd");
    let mut worst = 0.0_f64;
    for c in sim_gr.geom.interior.iter() {
        let pairs = [
            (*sim_gr.fields.cons.den.view().at(c), *flat.fields.cons.den.view().at(c)),
            (*sim_gr.fields.cons.mom[0].view().at(c), *flat.fields.cons.mom[0].view().at(c)),
            (*sim_gr.fields.cons.mom[1].view().at(c), *flat.fields.cons.mom[1].view().at(c)),
            (*sim_gr.fields.cons.mom[2].view().at(c), *flat.fields.cons.mom[2].view().at(c)),
            (*mhd_g.bface[0].view().at(c), *mhd_f.bface[0].view().at(c)),
            (*mhd_g.bface[1].view().at(c), *mhd_f.bface[1].view().at(c)),
            (*mhd_g.bface[2].view().at(c), *mhd_f.bface[2].view().at(c)),
        ];
        for (g, f) in pairs {
            worst = worst.max((g - f).abs() / f.abs().max(1.0));
        }
    }
    assert!(worst < 1e-12, "M = 0 kerr-schild diverges from flat: rel {worst:e}");
}
