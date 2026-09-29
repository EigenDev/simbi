// =============================================================================
// uct_hlld_kerr_2d.rs
//
// the 2.5D cartesian (x, y) UCT-HLLD chain on the spinning kerr chart, end to
// end through the production RMHD substrate. gates:
// - spin 0.5, mass 1: a magnetized swirl in a box outside the horizon evolves
//   under the wave-sum corner EMF with the densitized face divergence
//   sqrt(gamma) B at round-off (the curl-form update telescopes for any face
//   weight, so a discretely div-free start stays div-free);
// - the zero-mass oracle: the kerr-schild metric at M = 0 is minkowski for any
//   spin (H = 0: alpha = 1, beta = 0, gamma = delta), so one step of the kerr
//   UCT-HLLD chain at M = 0, spin 0.5 matches the flat UCT-HLLD chain on
//   identical initial data to round-off. the two chains assemble algebraically
//   identical arithmetic through different operation orders.
//
// run: cargo test -p symbi --test uct_hlld_kerr_2d
// =============================================================================

use std::f64::consts::PI;

use symbi::regimes::substrate_kernels::Solver;
use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{Cartesian, KerrKSCartesian};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 16;
const GAMMA: f64 = 4.0 / 3.0;
const CFL: f64 = 0.3;
const B0: f64 = 0.1;
const X_LO: f64 = 1.2;
const DIVB_TOL: f64 = 1e-12;

type KerrSim = SimStateGeneric<Rmhd, 2, 3, KerrKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory>;
type FlatSim = SimStateGeneric<Rmhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = RmhdSubstrateKernelSet<HostMemory, f64, 2>;

/// a gentle position-dependent swirl carrying the out-of-plane velocity, under a uniform B_x
fn swirl([x, y]: [f64; 2]) -> MhdPrim<f64, 3> {
    let s = 2.0 * PI;
    MhdPrim::new(
        Prim::adiabatic(
            Density(1.0),
            Tensor::new([0.1 * (s * y).sin(), 0.1 * (s * x).sin(), 0.05 * (s * (x + y)).cos()]),
            Pressure(0.5),
        ),
        Tensor::new([B0, 0.0, 0.0]),
    )
}

fn kernels(alloc: &symbi_algebra::Domain<2>) -> Kern {
    Kern::new(GAMMA, CFL, 1.0, alloc)
        .with_solver(Solver::Hlld)
        .expect("hlld")
        .ct_method(CtMethod::Uct)
}

fn kerr(metric: KerrKSCartesian<f64>) -> KerrSim {
    let dx = 1.0 / N as f64;
    KerrSim::build(Rmhd, IdealGas { gamma: GAMMA }, metric)
        .cells([N; 2])
        .origin([X_LO; 2])
        .spacing([dx; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("kerr sim")
        .set_initial(swirl)
        // the densitized flux sqrt(gamma) B_x is uniform on the x faces, so its divergence is zero
        .seed_faces(|axis, x| {
            if axis == 0 {
                B0 / symbi_geometry::Metric::<f64, 2>::sqrt_det_gamma(&metric, Tensor::new(x))
            } else {
                0.0
            }
        })
        .build()
}

/// (largest densitized face divergence, largest |B_face|) over the interior
fn max_divb(s: &KerrSim, metric: &KerrKSCartesian<f64>) -> (f64, f64) {
    let mhd = s.fields.mhd.as_ref().expect("mhd");
    let dx = 1.0 / N as f64;
    let lo: [isize; 2] = std::array::from_fn(|a| s.geom.interior.spaces[a].lo);
    let face_pos = |c: [isize; 2], a: usize, hi: bool| -> Tensor<f64, 2> {
        Tensor::new(std::array::from_fn(|d| {
            let i = (c[d] - lo[d]) as f64;
            if d == a { X_LO + (i + if hi { 1.0 } else { 0.0 }) * dx } else { X_LO + (i + 0.5) * dx }
        }))
    };
    let (mut div_max, mut b_max) = (0.0f64, 0.0f64);
    for c in s.geom.interior.iter() {
        let mut div = 0.0;
        for a in 0..2 {
            let mut n = c;
            n[a] += 1;
            let (b_lo, b_hi) = (*mhd.bface[a].view().at(c), *mhd.bface[a].view().at(n));
            let g_lo = symbi_geometry::Metric::<f64, 2>::sqrt_det_gamma(metric, face_pos(c, a, false));
            let g_hi = symbi_geometry::Metric::<f64, 2>::sqrt_det_gamma(metric, face_pos(c, a, true));
            div += (g_hi * b_hi - g_lo * b_lo) / dx;
            b_max = b_max.max(b_lo.abs());
        }
        div_max = div_max.max(div.abs());
    }
    (div_max, b_max)
}

#[test]
fn spinning_kerr_2d_uct_hlld_preserves_divb() {
    let metric = KerrKSCartesian { mass: 1.0, spin: 0.5 };
    let mut s = kerr(metric);
    let (div0, _) = max_divb(&s, &metric);
    assert!(div0 < 1e-13, "the densitized start is not div-free: {div0:e}");
    let k = kernels(&s.geom.allocated);
    let mut steps = 0u64;
    evolve_with_callback(&mut s, &k, 0.5, 1, |st| {
        let (div, b) = max_divb(st, &metric);
        let rel = div / b.max(1.0);
        assert!(rel < DIVB_TOL, "div B grew at iteration {} t = {:.3e}: rel {rel:e}", st.iteration, st.time);
        steps = st.iteration;
    })
    .expect("kerr evolve");
    assert!(steps >= 5, "only {steps} steps; the gate barely ran");
}

#[test]
fn zero_mass_kerr_2d_uct_hlld_matches_flat_to_roundoff() {
    // one step: t_final sits below both charts' CFL steps, so both chains take dt = t_final
    const T_ONE: f64 = 0.005;
    let mut gr = kerr(KerrKSCartesian { mass: 0.0, spin: 0.5 });
    let k = kernels(&gr.geom.allocated);
    evolve_with_callback(&mut gr, &k, T_ONE, 1, |_| {}).expect("kerr evolve");
    let dx = 1.0 / N as f64;
    let mut flat = FlatSim::build(Rmhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 2])
        .origin([X_LO; 2])
        .spacing([dx; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("flat sim")
        .set_initial(swirl)
        .seed_faces(|axis, _| if axis == 0 { B0 } else { 0.0 })
        .build();
    let k = kernels(&flat.geom.allocated);
    evolve_with_callback(&mut flat, &k, T_ONE, 1, |_| {}).expect("flat evolve");
    assert_eq!(gr.iteration, flat.iteration, "the step counts diverged at M = 0");
    let (mg, mf) = (gr.fields.mhd.as_ref().unwrap(), flat.fields.mhd.as_ref().unwrap());
    let mut worst = 0.0f64;
    let mut moved = 0.0f64;
    for c in gr.geom.interior.iter() {
        let mut pairs = vec![(*gr.fields.cons.den.view().at(c), *flat.fields.cons.den.view().at(c))];
        for k in 0..3 {
            pairs.push((*gr.fields.cons.mom[k].view().at(c), *flat.fields.cons.mom[k].view().at(c)));
        }
        for a in 0..2 {
            pairs.push((*mg.bface[a].view().at(c), *mf.bface[a].view().at(c)));
        }
        for (g, f) in pairs {
            worst = worst.max((g - f).abs() / f.abs().max(1.0));
        }
        moved = moved.max((*mf.bface[1].view().at(c)).abs());
    }
    assert!(moved > 1e-8, "the step never induced B_y; the comparison is blind to the corner EMF");
    assert!(worst < 1e-12, "the zero-mass kerr chain departs from flat by rel {worst:e}");
}
