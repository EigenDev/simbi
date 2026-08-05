// =============================================================================
// refined_ppm_drain.rs
//
// a draining accretor inside a steeply stratified atmosphere on a refined
// hierarchy, evolved with ppm reconstruction and quartic coarse-fine
// prolongation — the combination where a poisoned ghost or source state shows
// up as a persistent fofc freeze (the first-order redo replays fluxes and
// sources but re-reads the same prolonged ghost values, so an inadmissible
// ghost defeats every redo tier and escalates to the freeze-streak halt).
// the plm twin runs the identical setup as the control: a failure in both is
// the setup, a failure only under ppm is the reconstruction stack.
//
// run: cargo test -p symbi --test refined_ppm_drain -- --nocapture
// =============================================================================

use symbi::prelude::Solver;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_discretize::Recon;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, SurfaceSpec};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const N: usize = 32;
const K0: f64 = 0.6;
const GM: f64 = 3.0;
/// four fine cells, the production accretor geometry.
const R_ACC: f64 = 4.0 / (2.0 * N as f64);

type Sim = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 3>;
type Hier = Hierarchy<Newtonian, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

/// the isentropic atmosphere in hydrostatic balance against the softened point
/// mass, from the bernoulli invariant `gamma K0/(gamma-1) rho^(gamma-1) - GM/r
/// = const`, normalized to rho = 1 at the domain corner and regularized to the
/// accretion radius inside the mask.
fn atmosphere(x: [f64; 3]) -> Prim<f64, 3> {
    let r = x.iter().map(|c| c * c).sum::<f64>().sqrt().max(R_ACC);
    let r_ref = 3.0_f64.sqrt() * 0.5;
    let a = (GAMMA - 1.0) / (GAMMA * K0);
    let rho = (1.0 + a * GM * (1.0 / r - 1.0 / r_ref)).powf(1.0 / (GAMMA - 1.0));
    Prim {
        rho,
        vel: Tensor::new([0.0; 3]),
        pre: K0 * rho.powf(GAMMA),
    }
}

/// the atmosphere with a mach-0.0625 solenoidal (abc-flow) velocity seed riding
/// the local sound speed — the production initial condition: the run starts
/// moving everywhere, so every coarse-fine ghost prolongation and every ppm
/// stencil sees a stratified state in motion from the first stage.
fn stirred_atmosphere(x: [f64; 3]) -> Prim<f64, 3> {
    const MACH: f64 = 0.0625;
    let base = atmosphere(x);
    let cs = (GAMMA * base.pre / base.rho).sqrt();
    let k = 2.0 * std::f64::consts::PI;
    let vel = Tensor::new([
        MACH * cs * (2.0 * k * x[1]).sin() * (3.0 * k * x[2]).cos(),
        MACH * cs * (2.0 * k * x[2]).sin() * (3.0 * k * x[0]).cos(),
        MACH * cs * (2.0 * k * x[0]).sin() * (3.0 * k * x[1]).cos(),
    ]);
    Prim {
        rho: base.rho,
        vel,
        pre: base.pre,
    }
}

fn build(recon: Recon, ng: usize, prolong: ProlongOrder, stirred: bool) -> Hier {
    let seed = if stirred { stirred_atmosphere } else { atmosphere };
    let dx = 1.0 / N as f64;
    let kset = move |s: &Sim| {
        Kset::new(GAMMA, CFL, &s.geom.allocated)
            .with_solver(Solver::HllcLm)
            .expect("solver/regime mismatch")
            .reconstruction(recon)
    };
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N, N])
        .origin([-0.5, -0.5, -0.5])
        .spacing([dx, dx, dx])
        .ghosts(ng)
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim construction failed")
        .set_initial(seed)
        .build();
    let ck = kset(&coarse);
    let regions = [RefinementRegion {
        x_lo: [-0.25; 3],
        x_hi: [0.25; 3],
    }];
    let hier = Hierarchy::with_refinement(coarse, ck, &regions, prolong, kset)
        .unwrap()
        .with_bodies(
            BodyCollection::new().add(
                // porosity 1 is the pure drain channel: no wall, mass and energy
                // removed inside the mask at the porous kernel's drain rate.
                Body::black_hole(
                    0,
                    Tensor::new([0.0; 3]),
                    Tensor::zeros(),
                    GM,
                    R_ACC,
                    R_ACC,
                    0.0,
                    1.0,
                    R_ACC,
                )
                .with_surface(SurfaceSpec::Porous {
                    porosity: 1.0,
                    k_eta_n: 50.0,
                    k_eta_t: 0.0,
                }),
            ),
        );
    for lvl in 1..hier.levels.len() {
        hier.levels[lvl].state.seed_cells(seed);
    }
    hier
}

fn run_and_scan(recon: Recon, ng: usize, prolong: ProlongOrder, stirred: bool) -> (f64, f64) {
    let mut hier = build(recon, ng, prolong, stirred);
    hier.evolve_steps(60).unwrap();
    let (mut rho_min, mut pre_min) = (f64::INFINITY, f64::INFINITY);
    for lvl in hier.levels.iter() {
        let st = &lvl.state;
        let rho = st.fields.prim.rho.view();
        let pre_f = st.fields.prim.pre_field().expect("adiabatic pre");
        let pre = pre_f.view();
        for c in st.geom.interior.iter() {
            let (r, p) = (*rho.at(c), *pre.at(c));
            assert!(
                r.is_finite() && p.is_finite(),
                "non-finite state at {c:?}"
            );
            rho_min = rho_min.min(r);
            pre_min = pre_min.min(p);
        }
    }
    (rho_min, pre_min)
}

/// the production combination (ppm + quartic prolongation + porous drain +
/// hllc_lm on a stratified refined atmosphere) completes 60 root steps with a
/// positive state everywhere; a persistent fofc freeze panics inside evolve.
#[test]
fn a_draining_accretor_in_a_stratified_atmosphere_survives_under_ppm() {
    let (rho_min, pre_min) = run_and_scan(Recon::Ppm, 3, ProlongOrder::Quartic, false);
    println!("ppm+quartic quiescent: min rho {rho_min:.6e}, min pre {pre_min:.6e}");
    assert!(rho_min > 0.0 && pre_min > 0.0, "non-positive state survived c2p");
}

/// the plm control on the identical setup: a failure here too would indict the
/// setup, not the reconstruction stack.
#[test]
fn the_plm_control_survives_the_same_drain() {
    let (rho_min, pre_min) = run_and_scan(Recon::Plm, 3, ProlongOrder::Ppm, false);
    println!("plm control: min rho {rho_min:.6e}, min pre {pre_min:.6e}");
    assert!(rho_min > 0.0 && pre_min > 0.0, "non-positive state survived c2p");
}

/// the full production initial condition: the same drain with the mach-0.0625
/// solenoidal velocity seed, so the stratified state is in motion from stage
/// one everywhere the parabola and the quartic prolongation read.
#[test]
fn the_stirred_production_initial_condition_survives_under_ppm() {
    let (rho_min, pre_min) = run_and_scan(Recon::Ppm, 3, ProlongOrder::Quartic, true);
    println!("ppm+quartic stirred: min rho {rho_min:.6e}, min pre {pre_min:.6e}");
    assert!(rho_min > 0.0 && pre_min > 0.0, "non-positive state survived c2p");
}
