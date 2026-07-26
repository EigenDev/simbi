// =============================================================================
// fragment_wind.rs
//
// bonded rigid fragments coupled to gas through the real evolve() loop:
// the penalization books each fragment's drag receipts, evolve_bodies hands
// them to the bonded subcycle as frozen external loads, and the fragments
// move under gas drag + bond forces together.
//
//   - a bonded pair broadside to a wind accelerates downstream as a cluster:
//     both fragments gain +x velocity, the bond holds the separation, and the
//     two velocities stay close (the bond transmits the drag imbalance).
//   - a fragment tethered to a clamped anchor snaps its weak bond once the
//     accumulated drag stretches it past the tensile strength, then drifts
//     downstream freely.
//
// run: cargo test -p symbi --test fragment_wind
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, Bond, BondMaterial, FragmentPhysics, SurfaceSpec};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const N: usize = 48;
const L: f64 = 1.0;
// added-mass stability: the explicit two-way coupling hands the body the mask
// gas's momentum change each step, so a fragment lighter than the gas it
// displaces (rho pi r^2 ~ 0.07 here) overshoots its velocity correction and
// the mismatch grows without bound (dt collapses chasing it). the fragment
// mass must sit well above the displaced gas mass.
const FRAG_MASS: f64 = 0.6;
const FRAG_RADIUS: f64 = 0.15;
const WIND: f64 = 0.5;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

fn fragment(x: f64, y: f64, mobile: bool) -> Body<f64, 2> {
    Body::rigid_sphere(
        0,
        Tensor::new([x, y]),
        Tensor::zeros(),
        FRAG_MASS,
        FRAG_RADIUS,
        1e-3,
        true,
    )
    .with_surface(SurfaceSpec::Porous {
        porosity: 0.0,
        k_eta_n: 50.0,
        k_eta_t: 50.0,
    })
    .with_two_way_coupling(mobile)
}

fn wind_sim(bodies: BodyCollection<f64, 2>) -> Sim {
    let dx = 2.0 * L / N as f64;
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([WIND, 0.0]),
            pre: 1.0,
        })
        .build()
        .with_bodies(bodies)
}

#[test]
fn bonded_pair_rides_the_wind_as_a_cluster() {
    let coll = BodyCollection::new()
        .add_fragment(fragment(0.0, -0.35, true))
        .add_fragment(fragment(0.0, 0.35, true));
    let mat = BondMaterial {
        k_n: 50.0,
        gamma: 0.5,
        ..BondMaterial::rigid()
    };
    let bonds = vec![Bond::form(0, 1, coll.get(0), coll.get(1), mat)];
    let mut sim = wind_sim(coll);
    sim.attach_fragment_physics(FragmentPhysics {
        bonds,
        contacts: None,
        gravity: None,
    });

    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sim.geom.allocated);
    evolve(&mut sim, &sub, 0.15).expect("wind evolution with fragments failed");

    let im = sim.immersed.as_ref().unwrap();
    let (b0, b1) = (im.bodies.get(0), im.bodies.get(1));
    for (tag, b) in [("lower", b0), ("upper", b1)] {
        assert!(
            b.velocity[0] > 0.005,
            "{tag} fragment should accelerate downstream: v_x = {}",
            b.velocity[0]
        );
        assert!(
            b.position[0] > 1e-4,
            "{tag} fragment should drift downstream: x = {}",
            b.position[0]
        );
    }
    // the bond transmits the drag imbalance, so the pair moves as one cluster.
    let dv = (b0.velocity[0] - b1.velocity[0]).abs();
    assert!(
        dv < 0.2 * b0.velocity[0].max(b1.velocity[0]),
        "cluster velocities diverged: {} vs {}",
        b0.velocity[0],
        b1.velocity[0]
    );
    let sys = im.fragment_physics.as_ref().unwrap();
    assert_eq!(sys.intact_bonds(), 1, "the bond must survive a gentle wind");
    let dy = b1.position[1] - b0.position[1];
    let dx = b1.position[0] - b0.position[0];
    let sep = (dx * dx + dy * dy).sqrt();
    assert!(
        (sep - 0.7).abs() < 0.05,
        "bonded separation should hold near rest length 0.7: {sep}"
    );
}

#[test]
fn wind_drag_snaps_a_tethered_fragment() {
    // the anchor is a kinematic fragment (a clamp); the free fragment
    // downstream stretches the weak bond under drag until the tensile
    // envelope parts it.
    let coll = BodyCollection::new()
        .add_fragment(fragment(-0.2, 0.0, false))
        .add_fragment(fragment(0.5, 0.0, true));
    let weak = BondMaterial {
        k_n: 50.0,
        gamma: 0.5,
        sigma_t: 0.02,
        ..BondMaterial::rigid()
    };
    let bonds = vec![Bond::form(0, 1, coll.get(0), coll.get(1), weak)];
    let mut sim = wind_sim(coll);
    sim.attach_fragment_physics(FragmentPhysics {
        bonds,
        contacts: None,
        gravity: None,
    });

    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sim.geom.allocated);
    evolve(&mut sim, &sub, 0.8).expect("wind evolution with tethered fragment failed");

    let im = sim.immersed.as_ref().unwrap();
    let sys = im.fragment_physics.as_ref().unwrap();
    assert_eq!(sys.intact_bonds(), 0, "drag should snap the weak tether");
    let anchor = im.bodies.get(0);
    let free = im.bodies.get(1);
    assert!(
        (anchor.position[0] + 0.2).abs() < 1e-12,
        "the clamped anchor must not move: x = {}",
        anchor.position[0]
    );
    assert!(
        free.position[0] > 0.52,
        "the freed fragment should drift downstream: x = {}",
        free.position[0]
    );
}
