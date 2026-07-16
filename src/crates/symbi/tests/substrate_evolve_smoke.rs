// smoke: a full time-marched isothermal Euler evolution where EVERY operator in
// the RK2 step (ghost_fill, c2p, cfl, flux, godunov_euler, godunov_rk2, snapshot)
// is the substrate-generated kernel — no hand-written kernel touched.
use symbi::regimes::substrate::IsoSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::Isothermal;
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::state::PrimG;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<IsoNewtonian, 1, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
const CS: f64 = 1.0;

#[test]
fn full_substrate_isothermal_euler_evolution() {
    let n = 128usize;
    let dx = 1.0 / n as f64;
    let pi = std::f64::consts::PI;
    let amp = 0.01;
    // rho = 1 + amp*sin(2 pi x); mom = rho*amp*cs*sin => vel = mom/rho = amp*cs*sin.
    let mut sim = Sim::build(IsoNewtonian, Isothermal { cs: CS }, Cartesian)
        .cells([n])
        .spacing([dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .allocate()
        .unwrap()
        .set_initial(|x| {
            let s = (2.0 * pi * x[0]).sin();
            PrimG {
                rho: 1.0 + amp * s,
                vel: Tensor::new([amp * CS * s]),
                pre: Default::default(),
            }
        })
        .build();

    let mass0: f64 = sim.geom.interior.iter().map(|c| *sim.fields.cons.den.view().at(c)).sum::<f64>() * dx;

    let sub = IsoSubstrateKernelSet::<HostMemory, f64, 1>::new(CS, 0.4, &sim.geom.allocated);
    // march to t=0.25 (a quarter sound-crossing); the whole loop is substrate.
    evolve(&mut sim, &sub, 0.25).expect("evolution failed");

    // everything finite + positive density, mass conserved (periodic), state moved.
    for c in sim.geom.interior.iter() {
        let r = *sim.fields.cons.den.view().at(c);
        assert!(r.is_finite() && r > 0.0, "bad density {r} at {c:?}");
    }
    let mass1: f64 = sim.geom.interior.iter().map(|c| *sim.fields.cons.den.view().at(c)).sum::<f64>() * dx;
    assert!((mass1 - mass0).abs() < 1e-12 * mass0, "mass drift {:e}", (mass1 - mass0)/mass0);
    assert!(sim.iteration > 5, "took only {} steps", sim.iteration);
    println!("SUBSTRATE EVOLVE: {} steps to t={:.3}, mass rel-drift {:e}",
             sim.iteration, sim.time, (mass1 - mass0) / mass0);
}

#[test]
fn locally_isothermal_cs2_derived_from_ic_and_held() {
    use symbi::symbi_grid::Field;
    let n = 64usize;
    let dx = 1.0 / n as f64;
    let pi = std::f64::consts::PI;
    // a VARYING local temperature cs^2(x) = p_IC(x)/rho_IC(x): a locally isothermal IC.
    // cs^2 is a DERIVED quantity — the user sets density + pressure, nothing else.
    let cs2_of = |x: f64| 0.5 + 0.3 * (2.0 * pi * x).sin();
    let rho_of = |x: f64| 1.0 + 0.05 * (2.0 * pi * x).cos();
    // at rest; the cs^2 gradient drives it.
    let mut sim = Sim::build(IsoNewtonian, Isothermal { cs: 1.0 }, Cartesian)
        .cells([n])
        .spacing([dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .allocate()
        .unwrap()
        .set_initial(|x| PrimG {
            rho: rho_of(x[0]),
            vel: Tensor::new([0.0]),
            pre: Default::default(),
        })
        .build();

    // p_IC = cs^2(x) * rho — a SEPARATE field the substrate derives cs^2 from (not a sim field).
    let pre_ic = Field::<f64, 1, HostMemory>::zeros(&sim.geom.allocated).unwrap();
    for c in sim.geom.interior.iter() {
        let x = (c[0] as f64 + 0.5) * dx;
        pre_ic.view_mut().set(c, cs2_of(x) * rho_of(x));
    }

    let sub = IsoSubstrateKernelSet::<HostMemory, f64, 1>::new(1.0, 0.4, &sim.geom.allocated);
    // cs^2 = p/rho, the isothermal sound speed squared (per-cell, NOT a global scalar).
    sub.compute_isothermal_cs2(&sim.fields.cons.den, &pre_ic, &sim.geom.interior);
    for c in sim.geom.interior.iter() {
        let x = (c[0] as f64 + 0.5) * dx;
        assert!((*sub.cs2.view().at(c) - cs2_of(x)).abs() < 1e-12, "cs^2 mis-derived at x={x}");
    }

    evolve(&mut sim, &sub, 0.1).expect("locally isothermal evolution failed");

    // cs^2(x) is read-only — still the derived per-cell temperature after the run, which
    // stayed physical (the varying sound speed flowed through c2p / flux / cfl).
    for c in sim.geom.interior.iter() {
        let x = (c[0] as f64 + 0.5) * dx;
        assert!((*sub.cs2.view().at(c) - cs2_of(x)).abs() < 1e-12, "cs^2 drifted (not read-only)");
        let r = *sim.fields.cons.den.view().at(c);
        assert!(r.is_finite() && r > 0.0, "bad density {r} at {c:?}");
    }
    assert!(sim.iteration > 3, "took only {} steps", sim.iteration);
}

#[test]
fn locally_isothermal_ghost_temperature_is_the_clamped_interior_value() {
    // a locally isothermal run derives cs^2(x) = p_IC/rho_IC over the INTERIOR; the ghost
    // cells must then receive the clamped zero-gradient continuation of that field. left
    // at the constructor's uniform cs^2 they poison every boundary flux: the ghost-pressure
    // pass books p = cs2_ghost * rho into the boundary reconstruction, and a cold disk edge
    // (cs^2 ~ 1e-2) against a cs = 1 default is a ~100x spurious wall pressure.
    use symbi::sim::evolve::KernelSet;
    use symbi::symbi_grid::Field;
    type Sim2 = SimState<IsoNewtonian, 2, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
    let n = 16usize;
    let dx = 1.0 / n as f64;
    // a cold, spatially varying temperature, far from the constructor's cs = 1.
    let cs2_of = |x: f64, y: f64| 0.01 * (1.0 + 0.5 * x + 0.25 * y);
    let rho_of = |x: f64, y: f64| 1.0 + 0.05 * x - 0.02 * y;
    let sim = Sim2::build(IsoNewtonian, Isothermal { cs: CS }, Cartesian)
        .cells([n, n])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .unwrap()
        .set_initial(|x| PrimG {
            rho: rho_of(x[0], x[1]),
            vel: Tensor::new([0.0, 0.0]),
            pre: Default::default(),
        })
        .build();

    let pre_ic = Field::<f64, 2, HostMemory>::zeros(&sim.geom.allocated).unwrap();
    for c in sim.geom.interior.iter() {
        let (x, y) = ((c[0] as f64 + 0.5) * dx, (c[1] as f64 + 0.5) * dx);
        pre_ic.view_mut().set(c, cs2_of(x, y) * rho_of(x, y));
    }
    let sub = IsoSubstrateKernelSet::<HostMemory, f64, 2>::new(CS, 0.4, &sim.geom.allocated);
    sub.compute_isothermal_cs2(&sim.fields.cons.den, &pre_ic, &sim.geom.interior);
    sub.extend_cs2_into_ghosts(&sim.geom.allocated, &sim.geom.interior);

    // every ghost cell (faces, edges, corners) holds the per-axis-clamped interior value.
    let (mut ghosts, mut corners) = (0usize, 0usize);
    for c in sim.geom.allocated.iter() {
        if sim.geom.interior.contains(c) {
            continue;
        }
        ghosts += 1;
        let clamped: [isize; 2] = std::array::from_fn(|ax| {
            c[ax].clamp(sim.geom.interior.spaces[ax].lo, sim.geom.interior.spaces[ax].hi - 1)
        });
        if clamped != c {
            let both_out = (0..2).all(|ax| clamped[ax] != c[ax]);
            if both_out {
                corners += 1;
            }
        }
        let got = *sub.cs2.view().at(c);
        let want = *sub.cs2.view().at(clamped);
        assert!(
            (got - want).abs() < 1e-15,
            "ghost cs2 at {c:?} = {got}, want clamped interior {want} (uniform constructor cs^2 leaked?)"
        );
        assert!(got < 0.1, "ghost cs2 at {c:?} = {got} is the constructor's global value, not the IC's");
    }
    assert!(ghosts > 0 && corners > 0, "expected face and corner ghosts to be checked");

    // integration: recover the interior prims, then fill ghosts — the ghost pressure
    // must obey the LOCAL closure p = cs2*rho, not the constructor's uniform cs^2.
    sub.c2p(&sim);
    sub.ghost_fill(&sim);
    for c in sim.geom.allocated.iter() {
        if sim.geom.interior.contains(c) {
            continue;
        }
        let rho = *sim.fields.prim.rho.view().at(c);
        let p = *sub.pre.view().at(c);
        let cs2 = *sub.cs2.view().at(c);
        assert!(
            (p - cs2 * rho).abs() < 1e-14 && p < 0.1 * rho,
            "ghost pressure at {c:?}: p = {p}, cs2*rho = {}, rho = {rho}",
            cs2 * rho
        );
    }
}
