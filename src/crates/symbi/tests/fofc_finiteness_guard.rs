// =============================================================================
// fofc_finiteness_guard.rs
//
// the FOFC physicality gate guards the FULL state vector, not just density (and pressure where the
// energy is modelled): a cell whose spliced first-order VELOCITY is non-finite — a NaN/inf momentum
// with a finite density — must be FROZEN to the admissible stage input, not kept. this is the gap
// the review flagged for iso, whose only other guard is the density, so a NaN momentum would
// otherwise ride through the FOFC select until the next flux divergence poisoned the density a step
// later.
//
// A/B on the select KERNEL, isolated from the flux dynamics (a live NaN propagates into the density
// within one godunov, so a sim-level test cannot attribute the flag): a hand-built state gives a
// physical uniform stage input (v = 0) and a first-order result with a finite rho/pre but a NaN
// velocity band. `fofc_select` must leave EVERY cell's momentum finite (the band froze to u_stage,
// the rest kept the first-order value). without the velocity finiteness check the NaN band is kept
// and the finiteness assertion fails.
// =============================================================================

use symbi::regimes::fofc::fofc_select;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::{IdealGas, Isothermal};
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::{Prim, PrimG};
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 16;
// the right half of the interior gets a NaN first-order velocity (finite density + pressure).
fn nan_band(i: isize) -> bool {
    i >= (N / 2) as isize
}

#[test]
fn fofc_select_freezes_nonfinite_momentum_adiabatic() {
    type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let gamma = 1.4;
    let e_int = 1.0 / (gamma - 1.0); // internal energy of rho=1, v=0, p=1
    let dx = 1.0 / N as f64;

    let s = Sim::build(Newtonian, IdealGas { gamma }, Cartesian)
        .cells([N, N])
        .origin([0.0, 0.0])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.0, 0.0]),
            pre: 1.0,
        })
        .build();
    let us = &s.workspace.u_stage;
    let us_nrg = us.nrg_field().expect("u_stage nrg");
    let cons = &s.fields.cons;
    let c_nrg = cons.nrg_field().expect("cons nrg");
    let prim = &s.fields.prim;
    let p_pre = prim.pre_field().expect("prim pre");
    for c in s.geom.interior.iter() {
        // physical stage input everywhere (the freeze parachute).
        us.den.view_mut().set(c, 1.0);
        us.mom[0].view_mut().set(c, 0.0);
        us.mom[1].view_mut().set(c, 0.0);
        us_nrg.view_mut().set(c, e_int);
        // the spliced first-order result: finite density + pressure everywhere; NaN velocity band.
        cons.den.view_mut().set(c, 1.0);
        c_nrg.view_mut().set(c, e_int);
        prim.rho.view_mut().set(c, 1.0);
        p_pre.view_mut().set(c, 1.0);
        let v = if nan_band(c[0]) { f64::NAN } else { 0.0 };
        cons.mom[0].view_mut().set(c, v);
        cons.mom[1].view_mut().set(c, 0.0);
        prim.vel[0].view_mut().set(c, v);
        prim.vel[1].view_mut().set(c, 0.0);
    }

    fofc_select(&s, "adiabatic", "", &s.workspace.u_stage, &s.fields.cons, &s.fields.prim);

    let mut band = 0usize;
    for c in s.geom.interior.iter() {
        let m0 = *s.fields.cons.mom[0].view().at(c);
        assert!(m0.is_finite(), "fofc_select kept a non-finite momentum at {c:?}");
        if nan_band(c[0]) {
            assert_eq!(m0, 0.0, "a NaN-velocity cell must freeze to the stage input at {c:?}");
            band += 1;
        }
    }
    assert!(band > N, "NaN band too small to be meaningful ({band})");
}

#[test]
fn fofc_select_freezes_nonfinite_momentum_iso() {
    type Sim = SimState<IsoNewtonian, 2, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
    let cs = 0.5;
    let dx = 1.0 / N as f64;

    let s = Sim::build(IsoNewtonian, Isothermal { cs }, Cartesian)
        .cells([N, N])
        .origin([0.0, 0.0])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|_| PrimG::<f64, 2, IsoModel> {
            rho: 1.0,
            vel: Tensor::new([0.0, 0.0]),
            pre: Default::default(),
        })
        .build();
    let us = &s.workspace.u_stage;
    let cons = &s.fields.cons;
    let prim = &s.fields.prim;
    for c in s.geom.interior.iter() {
        us.den.view_mut().set(c, 1.0);
        us.mom[0].view_mut().set(c, 0.0);
        us.mom[1].view_mut().set(c, 0.0);
        // iso: no pressure field; density is the only scalar guard, so a NaN velocity band is exactly
        // the case that would ride through without the added velocity finiteness check.
        cons.den.view_mut().set(c, 1.0);
        prim.rho.view_mut().set(c, 1.0);
        let v = if nan_band(c[0]) { f64::NAN } else { 0.0 };
        cons.mom[0].view_mut().set(c, v);
        cons.mom[1].view_mut().set(c, 0.0);
        prim.vel[0].view_mut().set(c, v);
        prim.vel[1].view_mut().set(c, 0.0);
    }

    fofc_select(&s, "iso", "", &s.workspace.u_stage, &s.fields.cons, &s.fields.prim);

    let mut band = 0usize;
    for c in s.geom.interior.iter() {
        let m0 = *s.fields.cons.mom[0].view().at(c);
        assert!(m0.is_finite(), "iso fofc_select kept a non-finite momentum at {c:?}");
        if nan_band(c[0]) {
            assert_eq!(m0, 0.0, "a NaN-velocity iso cell must freeze to the stage input at {c:?}");
            band += 1;
        }
    }
    assert!(band > N, "NaN band too small to be meaningful ({band})");
}
