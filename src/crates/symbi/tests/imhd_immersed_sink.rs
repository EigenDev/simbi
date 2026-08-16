// =============================================================================
// imhd_immersed_sink.rs
//
// the immersed-body drain under isothermal MHD. isothermal has no energy equation
// (nrg is a zst), so there is no 1/2|B|^2 in any energy to corrupt — the 1/2|B|^2
// sandwich is a correct no-op and the drain simply removes (den, mom) while the
// magnetic field is left to constrained transport. the wall/drain still relaxes on the
// fast magnetosonic speed c_fast = sqrt(cs^2 + c_a^2), so a low-beta iso sink is stiffer.
// this pins: the sink drains plasma, B is untouched, and c_fast bites.
// =============================================================================

use symbi::regimes::substrate_isothermal_mhd::IsothermalMhdSubstrateKernelSet3D;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::Isothermal;
use symbi_hydro::isothermal_mhd::IsothermalMhd;
use symbi_hydro::mhd_state::MhdPrimG;
use symbi_hydro::state::PrimG;
use symbi_ib::sdf::SdfExpr;
use symbi_ib::{Body, BodyCollection, SurfaceSpec};
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<IsothermalMhd, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;

const N: usize = 24;
const L: f64 = 1.0;
const CS: f64 = 1.0;

fn make_sim(b0: f64) -> Sim {
    let dx = 2.0 * L / N as f64;
    // uniform gas threaded by a uniform Bx = b0 (div-free); isothermal closure p = cs^2 rho.
    Sim::build(IsothermalMhd, Isothermal { cs: CS }, Cartesian)
        .cells([N, N, N])
        .origin([-L, -L, -L])
        .spacing([dx, dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("iso-mhd sim construction failed")
        .set_initial(move |_| MhdPrimG::<f64, 3, IsoModel> {
            hydro: PrimG {
                rho: 1.0,
                vel: Tensor::new([0.0, 0.0, 0.0]),
                pre: Default::default(),
            },
            mag: Tensor::new([b0, 0.0, 0.0]),
        })
        .seed_faces(move |axis, _| if axis == 0 { b0 } else { 0.0 })
        .build()
}

fn total_mass(s: &Sim) -> f64 {
    s.geom
        .interior
        .iter()
        .map(|c| *s.fields.cons.den.view().at(c))
        .sum()
}

fn bcell_snapshot(s: &Sim) -> Vec<f64> {
    let m = s.fields.mhd.as_ref().unwrap();
    let mut v = Vec::new();
    for c in s.geom.interior.iter() {
        for k in 0..3 {
            v.push(*m.bcell[k].view().at(c));
        }
    }
    v
}

fn setup(b0: f64) -> Sim {
    let mut sim = make_sim(b0);
    // a shaped subgrid sink at the origin: a pure-drain porous wall (porosity 1), removing plasma.
    sim = sim.with_bodies(BodyCollection::new().add(
        Body::rigid_sphere(0, Tensor::zeros(), Tensor::zeros(), 1.0, 0.3, 1.0, true).with_surface(
            SurfaceSpec::Porous {
                porosity: 1.0,
                k_eta_n: 50.0,
                k_eta_t: 50.0,
            },
        ),
    ));
    sim.immersed.as_mut().unwrap().shapes[0] = Some(SdfExpr::<f64, 3>::cuboid(
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25],
    ));
    sim
}

fn drained_mass(sim: &Sim) -> f64 {
    let m0 = total_mass(sim);
    let sub = IsothermalMhdSubstrateKernelSet3D::<HostMemory, f64>::new(
        CS,
        0.3,
        1.0,
        &sim.geom.allocated,
    );
    sub.penalize(sim, 1e-3);
    m0 - total_mass(sim)
}

#[test]
fn imhd_sink_drains_plasma_and_leaves_the_field() {
    let sim = setup(0.5);
    let mass0 = total_mass(&sim);
    let bcell0 = bcell_snapshot(&sim);

    let dm = drained_mass(&sim);

    assert!(dm > 0.0, "the iso sink drained no plasma: {dm}");
    assert!(total_mass(&sim) < mass0);
    // constrained transport owns B; the drain never touches it (bit-exact). there is no energy
    // slot, so there is nothing else to check -- the sandwich correctly does nothing here.
    assert_eq!(
        bcell_snapshot(&sim),
        bcell0,
        "the iso drain modified the magnetic field"
    );
}

#[test]
fn imhd_c_fast_makes_a_low_beta_sink_stiffer() {
    // c_fast = sqrt(cs^2 + c_a^2): with cs = 1 and b0 = 5 (c_a ~ 5), the strong-field iso sink is
    // several-fold stiffer than the weak-field one -- the same rate lift as the adiabatic path.
    let weak = drained_mass(&setup(0.01));
    let strong = drained_mass(&setup(5.0));
    assert!(
        strong > 1.5 * weak,
        "the low-beta iso sink was not stiffer: weak drained {weak}, strong drained {strong}"
    );
}
