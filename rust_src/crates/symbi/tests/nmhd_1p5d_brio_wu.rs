// =============================================================================
// nmhd_1p5d_brio_wu.rs
//
// increment-3 validation of the 1.5D Newtonian-MHD substrate (docs/design/30): the
// canonical Brio & Wu (1988) MHD shock tube on a GENUINE D=1 grid (DOF=3). 1.5D has
// NO constrained transport — C(1,2)=0 edges (the StaggerComplex empty case) — so:
//   - the normal field Bx is carried but NEVER curled: it MUST stay at its constant IC
//     (0.75) to machine precision (the defining 1.5D property + trivial div B = dBx/dx),
//   - the transverse By,Bz ride the ordinary induction-flux divergence (bcell_godunov),
//   - the gas (rho, v, p) shocks via the HLLE flux.
// asserts: Bx const (machine), physicality, the unshocked end states survive at the
// boundaries, and the characteristic Brio-Wu structure (By sign change = the compound
// wave; a compressed intermediate density) appears. exact-reference comparison is a
// follow-up; this gate pins the 1.5D machinery end-to-end.
//
// IC (gamma=2, x0=0.5): left rho=1, p=1, B=(0.75, 1, 0); right rho=0.125, p=0.1,
// B=(0.75, -1, 0); v=0 both sides.
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::{MhdCons, MhdPrim};
use symbi_hydro::newtonian_mhd::{nmhd_recover, NewtonianMhd};
use symbi_hydro::state::{Cons, Prim};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 1, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

const NX: usize = 400;
const GAMMA: f64 = 2.0;
const CFL: f64 = 0.4;
const BX: f64 = 0.75;
const T_FINAL: f64 = 0.1;

fn make_sim() -> Sim {
    let dx = 1.0 / NX as f64;
    // the normal field Bx is constant (the 1.5D parameter): seed_faces_uniform sets the
    // (thin) face field over its FULL domain incl. ghosts to 0.75 and marks it initialized,
    // so the flux normal-B override reads 0.75 at every face and evolve() does not re-derive it.
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([NX])
        .spacing([dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("brio-wu 1.5d sim construction failed")
        .set_initial(|[x]| {
            let (rho, p, by) = if x < 0.5 { (1.0, 1.0, 1.0) } else { (0.125, 0.1, -1.0) };
            MhdPrim {
                hydro: Prim { rho, vel: Tensor::new([0.0, 0.0, 0.0]), pre: p },
                mag: Tensor::new([BX, by, 0.0]),
            }
        })
        .seed_faces_uniform([BX])
        .build()
}

fn recover(sim: &Sim, c: [isize; 1]) -> (f64, f64) {
    let mhd = sim.fields.mhd.as_ref().unwrap();
    let cnrg = sim.fields.cons.nrg_field().unwrap();
    let cons = MhdCons::<f64, 3> {
        hydro: Cons {
            den: *sim.fields.cons.den.view().at(c),
            mom: Tensor::new([
                *sim.fields.cons.mom[0].view().at(c),
                *sim.fields.cons.mom[1].view().at(c),
                *sim.fields.cons.mom[2].view().at(c),
            ]),
            nrg: *cnrg.view().at(c),
        },
        mag: Tensor::new([
            *mhd.bcell[0].view().at(c),
            *mhd.bcell[1].view().at(c),
            *mhd.bcell[2].view().at(c),
        ]),
    };
    let prim = nmhd_recover(&IdealGas { gamma: GAMMA }, &cons);
    (prim.rho, prim.pre)
}

#[test]
fn nmhd_1p5d_brio_wu_shock_tube() {
    let mut sim = make_sim();
    let sub = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, CFL, /* theta */ 1.5, &sim.geom.allocated);

    // Bx must stay EXACTLY constant under evolve — the crux of the no-CT 1.5D scheme.
    let assert_bx_const = |s: &Sim| {
        let mhd = s.fields.mhd.as_ref().unwrap();
        for c in s.geom.interior.iter() {
            let bx = *mhd.bcell[0].view().at(c);
            assert!((bx - BX).abs() < 1e-12, "Bx drifted from {BX} to {bx} at {c:?} (iter {})", s.iteration);
        }
    };

    let mut steps = 0u64;
    evolve_with_callback(&mut sim, &sub, T_FINAL, 1, |s| {
        assert_bx_const(s);
        steps = s.iteration;
    })
    .expect("brio-wu 1.5d evolve failed");
    assert!(steps >= 20, "brio-wu produced only {steps} steps — gate barely exercised");

    let mhd = sim.fields.mhd.as_ref().unwrap();
    let by = |i: isize| *mhd.bcell[1].view().at([i]);

    // physicality everywhere + collect density bounds and the By sign-change (compound wave).
    let mut rho_min = f64::INFINITY;
    let mut rho_max = f64::NEG_INFINITY;
    let mut by_sign_changes = 0;
    let mut prev_by = by(0);
    for i in 0..NX as isize {
        let (rho, p) = recover(&sim, [i]);
        assert!(rho.is_finite() && rho > 0.0, "cell {i}: rho={rho}");
        assert!(p.is_finite() && p > 0.0, "cell {i}: p={p}");
        assert!(rho > 0.05 && rho < 1.05, "cell {i}: rho={rho} out of Brio-Wu bounds");
        rho_min = rho_min.min(rho);
        rho_max = rho_max.max(rho);
        let b = by(i);
        if b * prev_by < 0.0 {
            by_sign_changes += 1;
        }
        prev_by = b;
    }

    // unshocked end states survive at the boundaries (waves have not reached them by t=0.1).
    let (rho_l, p_l) = recover(&sim, [3]);
    let (rho_r, p_r) = recover(&sim, [NX as isize - 4]);
    assert!((rho_l - 1.0).abs() < 0.05, "left end rho={rho_l} (expected ~1.0)");
    assert!((p_l - 1.0).abs() < 0.05, "left end p={p_l} (expected ~1.0)");
    assert!((rho_r - 0.125).abs() < 0.02, "right end rho={rho_r} (expected ~0.125)");
    assert!((p_r - 0.1).abs() < 0.02, "right end p={p_r} (expected ~0.1)");
    assert!((by(3) - 1.0).abs() < 0.05, "left end By={} (expected ~1.0)", by(3));
    assert!((by(NX as isize - 4) + 1.0).abs() < 0.05, "right end By={} (expected ~-1.0)", by(NX as isize - 4));

    // the Brio-Wu compound wave: By transitions +1 -> -1, so it changes sign in the interior.
    assert!(by_sign_changes >= 1, "By never changed sign — compound-wave structure absent");
    // the wave structure developed: the rarefaction / contact / slow-shock region drove the
    // density into the intermediate states, well below the left unshocked value.
    assert!(rho_min < 0.6, "solution did not develop the rarefaction/contact drop: rho_min={rho_min}");

    eprintln!(
        "[brio-wu 1.5d] DONE iter={} t={:.4e} rho in [{:.3},{:.3}] By sign-changes={}",
        sim.iteration, sim.time, rho_min, rho_max, by_sign_changes,
    );
}
