// =============================================================================
// ohmic_heating.rs
//
// resistive MHD conserves TOTAL energy and books the Ohmic heating into the gas AUTOMATICALLY: nrg is
// the total energy (conserved by the godunov flux), and the CT `bcell_from_bface` reconciliation moves
// the resistively-dissipated magnetic energy 1/2 B^2 into the gas internal energy -- no separate Joule
// source term. a decaying sheared field B_x = B0 sin(k y) in a periodic box loses magnetic energy to
// resistivity; that energy reappears as gas heat and the total energy is invariant to round-off.
// (this pins the property; there is no leak, so no Joule-heating term is added.)
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<NewtonianMhd, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

const N: usize = 16;
const GAMMA: f64 = 5.0 / 3.0;
const B0: f64 = 0.1;
const T_FINAL: f64 = 0.1;

fn make() -> Sim {
    let dx = 1.0 / N as f64;
    let k = 2.0 * std::f64::consts::PI;
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N, N])
        .spacing([dx, dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("3d mhd sim")
        .set_initial(|[_x, y, _z]| MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.0, 0.0, 0.0]),
                pre: 1.0,
            },
            mag: Tensor::new([B0 * (k * y).sin(), 0.0, 0.0]),
        })
        .seed_faces(|axis, x| {
            if axis == 0 {
                B0 * (k * x[1]).sin()
            } else {
                0.0
            }
        })
        .build()
}

// (total energy, magnetic energy, gas internal energy) over the interior.
fn energies(s: &Sim) -> (f64, f64, f64) {
    let m = s.fields.mhd.as_ref().unwrap();
    let den = &s.fields.cons.den;
    let nrg = s.fields.cons.nrg_field().unwrap();
    let (mut e, mut me, mut ie) = (0.0, 0.0, 0.0);
    for c in s.geom.interior.iter() {
        let rho = *den.view().at(c);
        let mut msq = 0.0;
        for k in 0..3 {
            let mo = *s.fields.cons.mom[k].view().at(c);
            msq += mo * mo;
        }
        let mut bsq = 0.0;
        for k in 0..3 {
            let b = *m.bcell[k].view().at(c);
            bsq += b * b;
        }
        let tot = *nrg.view().at(c);
        e += tot;
        me += 0.5 * bsq;
        ie += tot - 0.5 * msq / rho - 0.5 * bsq;
    }
    (e, me, ie)
}

#[test]
fn resistive_mhd_conserves_energy_and_heats_the_gas() {
    let mut sim = make();
    let (e0, me0, ie0) = energies(&sim);
    let sub = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 3>::new(
        GAMMA,
        0.3,
        1.0,
        &sim.geom.allocated,
    )
    .with_resistivity(0.05);
    evolve(&mut sim, &sub, T_FINAL).expect("resistive mhd evolve failed");
    let (e1, me1, ie1) = energies(&sim);

    let me_loss = me0 - me1;
    let ie_gain = ie1 - ie0;
    assert!(
        me_loss > 0.05 * me0,
        "resistivity should have dissipated substantial magnetic energy: {me0} -> {me1}"
    );
    // TOTAL energy conserved to round-off (the conservative total-energy flux form). the Ohmic
    // dissipation redistributes energy within the conserved total, so the sum is unchanged.
    let rel_drift = (e1 - e0).abs() / e0;
    assert!(
        rel_drift < 1e-10,
        "resistive MHD did not conserve total energy: relative drift {rel_drift:.3e}"
    );
    // the dissipated magnetic energy became gas internal energy: the gas heated by ~the magnetic loss.
    assert!(
        ie_gain > 0.0,
        "the gas did not heat: internal {ie0} -> {ie1}"
    );
    // nearly ALL the dissipated magnetic energy became gas HEAT (a sub-percent sliver goes to kinetic
    // energy from the resistive dynamics; E = KE + ME + IE is the exactly-conserved total above).
    assert!(
        (ie_gain - me_loss).abs() < 0.02 * me_loss,
        "gas heating {ie_gain} did not track the magnetic-energy loss {me_loss}"
    );
}
