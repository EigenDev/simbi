// =============================================================================
// mesh_motion_dye.rs
//
// the passive scalar on a homologously expanding mesh.
//
// expansion is a GEOMETRIC operation: a(t) stretches the physical cell width on the expanding
// axes, and the conserved densities dilute accordingly. `D_chi = rho chi` is a density like `rho`,
// so both dilute by the same factor and the CONCENTRATION `chi` is invariant under pure expansion.
//
// the failure mode is narrow and silent: the dye divergence divides by a cell width, and if it
// resolves the COMOVING width while the gas resolves the physical one, the dye is advected against
// a grid a factor a(t) too small. a uniform dye is the sharpest probe — under the correct rule it
// stays exactly uniform however far the mesh expands, and any mismatch between the two widths
// shows up immediately as structure in a field that has no reason to develop any.
//
// run: cargo test -p symbi --test mesh_motion_dye
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{Cartesian, MotionState};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const N: usize = 32;
const CHI: f64 = 0.4;
const ADOT: f64 = 0.5;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

fn build(motion: MotionState<f64>) -> (Sim, Kset) {
    let dx = 1.0 / N as f64;
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 2])
        .origin([-0.5; 2])
        .spacing([dx; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("sim")
        // cold gas coasting homologously: v = a_dot x, so the gas rides the mesh and the only
        // thing acting on the dye is the expansion itself.
        .set_initial(|[x, y]| Prim {
            rho: 1.0,
            vel: Tensor::new([ADOT * x, ADOT * y]),
            pre: 1e-6,
        })
        .build()
        .with_passive_scalar()
        .expect("chi alloc");
    sim.motion = motion;
    let cons_chi = sim.fields.cons.chi_field().expect("cons chi");
    let prim_chi = sim.fields.prim.chi_field().expect("prim chi");
    for c in sim.geom.allocated.clone().iter() {
        let rho = *sim.fields.cons.den.view().at(c);
        cons_chi.view_mut().set(c, rho * CHI);
        prim_chi.view_mut().set(c, CHI);
    }
    let k = Kset::new(GAMMA, CFL, &sim.geom.allocated);
    (sim, k)
}

#[test]
fn a_uniform_dye_survives_homologous_expansion() {
    let (mut sim, k) = build(MotionState::homologous(1.0, ADOT));
    let a0 = sim.motion.a;
    evolve(&mut sim, &k, 0.4).expect("expanding dye evolution");

    // the premise: the mesh genuinely expanded. a static mesh holds a uniform dye uniform for
    // free, collapsing the comoving and physical widths onto the same number.
    let growth = sim.motion.a / a0;
    assert!(
        growth > 1.1,
        "the mesh grew by only {growth}; the gate cannot separate comoving from physical width"
    );

    let cons_chi = sim.fields.cons.chi_field().expect("cons chi");
    let (mut worst, mut worst_at) = (0.0_f64, [0isize; 2]);
    for c in sim.geom.interior.iter() {
        let rho = *sim.fields.cons.den.view().at(c);
        if rho <= 0.0 {
            continue;
        }
        let err = (*cons_chi.view().at(c) / rho - CHI).abs();
        if err > worst {
            worst = err;
            worst_at = c;
        }
    }
    assert!(
        worst < 1e-10,
        "expansion changed the dye concentration by {worst:e} at {worst_at:?} (seeded {CHI}, \
         mesh grew {growth}x); a pure dilution leaves the concentration invariant"
    );
}
