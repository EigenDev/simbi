// =============================================================================
// body_gravity_evolve.rs
//
// the immersed-boundary forward gravity, END TO END on the substrate: a fixed central
// point mass pulls a fluid inward through the real evolve() loop (the body_source
// KernelSet method, dispatched per RK stage when has_bodies()).
//
//   - cpu: uniform fluid at rest around a central mass develops INWARD radial momentum,
//     stays finite, conserves mass (gravity touches mom/nrg, not den).
//   - cuda: the body_gravity_source kernel runs on device and matches CPU < 1e-9.
//
// run: cargo test -p symbi --test body_gravity_evolve            (cpu)
//      cargo test -p symbi --features cuda --test body_gravity_evolve   (+ gpu)
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::{KernelSet, evolve};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const N: usize = 32;
const L: f64 = 1.0; // domain [-L, L]^2, central mass at origin.
const MASS: f64 = 1.0;
const SOFT: f64 = 0.2;

fn central_mass() -> BodyCollection<f64, 2> {
    BodyCollection::new().add(Body::gravitational(
        0,
        Tensor::new([0.0, 0.0]),
        Tensor::zeros(),
        MASS,
        0.1,  // radius
        SOFT, // softening
    ))
}

// -----------------------------------------------------------------------------
// regression: the FOFC redo must re-apply the immersed-body source. a FOFC-firing
// substage RESTORES the stage-input state and re-runs the godunov + additive source; without
// re-applying the body source there too, a FO/freeze-selected cell near a body loses its gravity
// kick for that substage. discriminating A/B on ONE forward-Euler substage (so a flagged cell
// cannot recover gravity on a later substage): identical FOFC-firing ICs, one run WITH the central
// mass and one WITHOUT. the flux/godunov is bit-identical from identical ICs (the body source is
// applied AFTER it), so mom_B - mom_A isolates the body impulse — which must be present and INWARD
// at every checked cell, INCLUDING the ones that fired FOFC (where the bug drops it).
// -----------------------------------------------------------------------------
#[test]
fn fofc_redo_preserves_body_gravity() {
    use symbi::regimes::fofc::{fofc_reset_stats, fofc_stats};
    use symbi::sim::refinement::Hierarchy;
    type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let dx = 2.0 * L / N as f64;
    // two streams moving APART across x = 0.4 (a double rarefaction): a near-vacuum opens there, so
    // the high-order c2p goes negative and FOFC fires along the strip (the redo restores + re-fluxes).
    // gravity is velocity-independent, so the body impulse D = mom_B - mom_A is unaffected by the flow.
    let ic = |[x, _y]: [f64; 2]| Prim {
        rho: 1.0,
        vel: Tensor::new([if x > 0.4 { 6.0 } else { -6.0 }, 0.0]),
        pre: 1e-8, // near-zero internal energy -> the diverging flux over-removes it -> p < 0 -> FOFC
    };
    // FOFC runs only through the AMR hierarchy's level_stage (the uni-grid evolve() has no fofc
    // phase), so drive the single-level hierarchy one Euler step (one substage -> a flagged cell
    // cannot recover the body source on a later substage). returns the stepped state.
    let run = |with_body: bool| -> Sim {
        let s = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
            .cells([N, N])
            .origin([-L, -L])
            .spacing([dx, dx])
            .timestepping(Timestepping::Euler)
            .boundaries(Boundaries::uniform(BoundaryType::Outflow))
            .allocate()
            .expect("sim")
            .set_initial(ic)
            .build();
        let s = if with_body { s.with_bodies(central_mass()) } else { s };
        let kset = AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &s.geom.allocated);
        let mut hier = Hierarchy::single(s, kset);
        hier.evolve(1e-4).expect("hierarchy step"); // one Euler step (t_final < first dt)
        assert_eq!(hier.levels[0].state.iteration, 1, "expected one step");
        hier.levels.into_iter().next().unwrap().state
    };

    fofc_reset_stats();
    let b = run(true); // central mass
    let (fired, _) = fofc_stats();
    assert!(fired > 0, "FOFC did not fire — the gate does not exercise the redo path");
    let a = run(false); // no body

    // the body pulls toward the origin: the impulse D = mom_B - mom_A must point INWARD
    // (D . (-r_hat) > 0). the flux is identical in A and B over one step, so D is purely the body
    // source. WITHOUT the fix, the redo restores u_stage + re-runs the godunov and never re-applies
    // the body source, so EVERY cell has D == 0 exactly; a single cell with an inward kick is
    // therefore impossible without the fix. the majority bound (not all) tolerates the far cells
    // whose softened kick falls below the 1e-6 test floor; the freeze tier is covered by the
    // dedicated `fofc_freeze_preserves_body_gravity` gate below. no cell may get a spurious OUTWARD
    // kick.
    let mut checked = 0usize;
    let mut got_body = 0usize;
    let mut worst_outward = 0.0_f64;
    for c in b.geom.interior.iter() {
        let x = -L + (c[0] as f64 + 0.5) * dx;
        let y = -L + (c[1] as f64 + 0.5) * dx;
        let r = (x * x + y * y).sqrt();
        if !(0.25..0.85).contains(&r) {
            continue; // skip the softened core + the boundary rows
        }
        let dbx = *b.fields.cons.mom[0].view().at(c) - *a.fields.cons.mom[0].view().at(c);
        let dby = *b.fields.cons.mom[1].view().at(c) - *a.fields.cons.mom[1].view().at(c);
        let inward = -(dbx * x + dby * y) / r; // impulse projected inward (toward the mass)
        if inward > 1e-6 {
            got_body += 1;
        }
        worst_outward = worst_outward.min(inward);
        checked += 1;
    }
    assert!(checked > 20, "too few cells checked ({checked})");
    assert!(
        got_body > checked / 2,
        "the FOFC redo dropped the body source: only {got_body}/{checked} cells got the inward \
         gravity kick in a FOFC-firing substage (0 without the fix)",
    );
    assert!(
        worst_outward > -1e-6,
        "a cell got a spurious OUTWARD kick from the FOFC redo: {worst_outward:e}",
    );
}

// -----------------------------------------------------------------------------
// FOFC FREEZE-tier body source: the last-resort freeze holds the stage input `u_stage` (pre-body)
// on a cell no first-order flux can update admissibly. the plain `fofc_select` leaves that cell with
// ZERO body impulse; the `_with_body` freeze-select instead evolves the parachute by the body source
// (gravity + accretion) over the substage, guarded to a physical state.
//
// this is an A/B on the SELECT KERNEL itself, isolated from the flux dynamics: the freeze set in a
// live run is body-dependent (the redo's body_apply can rescue a cell from the freeze) and cannot be
// cleanly separated at the sim output, so the kernel is driven directly. a hand-built state gives a
// PHYSICAL uniform stage input (u_stage: rho=1, v=0, p=1) and a first-order result whose PRIM is
// unphysical (p < 0) in an inner band (so the select FREEZES there) and physical in an outer band
// (so the select KEEPS the marker there). the only difference between the two kernels is the body
// source, so `with_body - plain` on the frozen cells is exactly the inward gravity impulse; without
// the with-body select it is identically zero and got_body collapses.
// -----------------------------------------------------------------------------
#[test]
fn fofc_freeze_preserves_body_gravity() {
    use symbi::regimes::fofc::{fofc_select, fofc_select_with_body};
    type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let dx = 2.0 * L / N as f64;
    let dt = 0.01;
    let e_int = 1.0 / (GAMMA - 1.0); // internal energy of rho=1, v=0, p=1
    const KEPT_MARKER: f64 = 7.0; // a kept (physical) cell retains this first-order momentum
    let frozen_band = |r: f64| (0.30..0.60).contains(&r);
    let kept_band = |r: f64| (0.70..0.90).contains(&r);

    // build a sim carrying the central body and impose the hand-built select inputs: a physical
    // uniform u_stage everywhere, a first-order cons whose prim is unphysical in the frozen band.
    // the select is pointwise (no stencil), so only the interior needs filling.
    let build = || -> Sim {
        let s = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
            .cells([N, N])
            .origin([-L, -L])
            .spacing([dx, dx])
            .boundaries(Boundaries::uniform(BoundaryType::Outflow))
            .allocate()
            .expect("sim")
            .set_initial(|_| Prim {
                rho: 1.0,
                vel: Tensor::new([0.0, 0.0]),
                pre: 1.0,
            })
            .build()
            .with_bodies(central_mass());
        let us = &s.workspace.u_stage;
        let us_nrg = us.nrg_field().expect("u_stage nrg");
        let cons = &s.fields.cons;
        let c_nrg = cons.nrg_field().expect("cons nrg");
        let prim = &s.fields.prim;
        let p_pre = prim.pre_field().expect("prim pre");
        for c in s.geom.interior.iter() {
            let x = -L + (c[0] as f64 + 0.5) * dx;
            let y = -L + (c[1] as f64 + 0.5) * dx;
            let r = (x * x + y * y).sqrt();
            us.den.view_mut().set(c, 1.0);
            us.mom[0].view_mut().set(c, 0.0);
            us.mom[1].view_mut().set(c, 0.0);
            us_nrg.view_mut().set(c, e_int);
            cons.den.view_mut().set(c, 1.0);
            cons.mom[0].view_mut().set(c, KEPT_MARKER);
            cons.mom[1].view_mut().set(c, 0.0);
            c_nrg.view_mut().set(c, e_int);
            prim.rho.view_mut().set(c, 1.0);
            // unphysical first-order pressure in the frozen band forces the select to freeze there.
            p_pre.view_mut().set(c, if frozen_band(r) { -1.0 } else { 1.0 });
        }
        s
    };

    let sb = build();
    fofc_select_with_body(&sb, "adiabatic", dt, GAMMA);
    let sp = build();
    fofc_select(&sp, "adiabatic", "", &sp.workspace.u_stage, &sp.fields.cons, &sp.fields.prim);

    let mut frozen = 0usize;
    let mut got_body = 0usize;
    let mut kept_ok = 0usize;
    let mut worst_outward = 0.0_f64;
    for c in sb.geom.interior.iter() {
        let x = -L + (c[0] as f64 + 0.5) * dx;
        let y = -L + (c[1] as f64 + 0.5) * dx;
        let r = (x * x + y * y).sqrt();
        let bx = *sb.fields.cons.mom[0].view().at(c);
        let by = *sb.fields.cons.mom[1].view().at(c);
        let px = *sp.fields.cons.mom[0].view().at(c);
        let py = *sp.fields.cons.mom[1].view().at(c);
        if frozen_band(r) {
            // the difference is the body impulse (both selects freeze to u_stage; only with_body
            // adds gravity). u_stage is at rest, so the impulse is pure inward gravity.
            let inward = -((bx - px) * x + (by - py) * y) / r;
            if inward > 1e-6 {
                got_body += 1;
            }
            worst_outward = worst_outward.min(inward);
            frozen += 1;
        } else if kept_band(r) {
            // a physical first-order cell is KEPT by both selects: momentum stays the marker exactly.
            if (bx - KEPT_MARKER).abs() < 1e-12 && (px - KEPT_MARKER).abs() < 1e-12 {
                kept_ok += 1;
            }
        }
    }
    assert!(frozen > 8, "too few frozen cells to test ({frozen})");
    assert!(kept_ok > 8, "physical cells must be kept unchanged by the freeze select ({kept_ok})");
    assert!(
        got_body == frozen,
        "the with-body freeze select dropped the body source: only {got_body}/{frozen} frozen \
         cells got the inward gravity kick (0 for the plain freeze select)",
    );
    assert!(
        worst_outward > -1e-9,
        "a frozen cell got a spurious OUTWARD body kick: {worst_outward:e}",
    );
}

// -----------------------------------------------------------------------------
// ISOTHERMAL freeze-tier body source (thin-disk sims): the energy-free twin of the gate above. the
// iso freeze parachute uses the isothermal EOS (p = cs^2 * rho, always positive with the density) so
// only the density guard applies, and the eos param is cs rather than gamma. same A/B on the two
// select kernels: `with_body` must add the inward gravity impulse to every frozen cell.
// -----------------------------------------------------------------------------
#[test]
fn fofc_freeze_preserves_body_gravity_iso() {
    use symbi::regimes::fofc::{fofc_select, fofc_select_with_body};
    use symbi_hydro::energy::IsoModel;
    use symbi_hydro::eos::Isothermal;
    use symbi_hydro::isothermal::IsoNewtonian;
    use symbi_hydro::state::PrimG;
    type Sim = SimState<IsoNewtonian, 2, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
    let dx = 2.0 * L / N as f64;
    let dt = 0.01;
    let cs = 0.5;
    const KEPT_MARKER: f64 = 7.0;
    let frozen_band = |r: f64| (0.30..0.60).contains(&r);
    let kept_band = |r: f64| (0.70..0.90).contains(&r);

    // physical uniform stage input (v = 0) + a first-order result whose prim DENSITY is unphysical in
    // the frozen band. iso keeps pressure in a separate cs^2 buffer (no sim prim pre), so its select
    // — like the plain iso `fofc_select` — gates freeze on the density alone; a negative x_rho fires it.
    let build = || -> Sim {
        let s = Sim::build(IsoNewtonian, Isothermal { cs }, Cartesian)
            .cells([N, N])
            .origin([-L, -L])
            .spacing([dx, dx])
            .boundaries(Boundaries::uniform(BoundaryType::Outflow))
            .allocate()
            .expect("sim")
            .set_initial(|_| PrimG::<f64, 2, IsoModel> {
                rho: 1.0,
                vel: Tensor::new([0.0, 0.0]),
                pre: Default::default(),
            })
            .build()
            .with_bodies(central_mass());
        let us = &s.workspace.u_stage;
        let cons = &s.fields.cons;
        let prim = &s.fields.prim;
        for c in s.geom.interior.iter() {
            let x = -L + (c[0] as f64 + 0.5) * dx;
            let y = -L + (c[1] as f64 + 0.5) * dx;
            let r = (x * x + y * y).sqrt();
            us.den.view_mut().set(c, 1.0);
            us.mom[0].view_mut().set(c, 0.0);
            us.mom[1].view_mut().set(c, 0.0);
            cons.den.view_mut().set(c, 1.0);
            cons.mom[0].view_mut().set(c, KEPT_MARKER);
            cons.mom[1].view_mut().set(c, 0.0);
            // negative first-order density in the frozen band -> select freezes; physical elsewhere.
            prim.rho.view_mut().set(c, if frozen_band(r) { -1.0 } else { 1.0 });
        }
        s
    };

    let sb = build();
    fofc_select_with_body(&sb, "iso", dt, cs);
    let sp = build();
    fofc_select(&sp, "iso", "", &sp.workspace.u_stage, &sp.fields.cons, &sp.fields.prim);

    let mut frozen = 0usize;
    let mut got_body = 0usize;
    let mut kept_ok = 0usize;
    let mut worst_outward = 0.0_f64;
    for c in sb.geom.interior.iter() {
        let x = -L + (c[0] as f64 + 0.5) * dx;
        let y = -L + (c[1] as f64 + 0.5) * dx;
        let r = (x * x + y * y).sqrt();
        let bx = *sb.fields.cons.mom[0].view().at(c);
        let by = *sb.fields.cons.mom[1].view().at(c);
        let px = *sp.fields.cons.mom[0].view().at(c);
        let py = *sp.fields.cons.mom[1].view().at(c);
        if frozen_band(r) {
            let inward = -((bx - px) * x + (by - py) * y) / r;
            if inward > 1e-6 {
                got_body += 1;
            }
            worst_outward = worst_outward.min(inward);
            frozen += 1;
        } else if kept_band(r) && (bx - KEPT_MARKER).abs() < 1e-12 && (px - KEPT_MARKER).abs() < 1e-12 {
            kept_ok += 1;
        }
    }
    assert!(frozen > 8, "too few frozen cells to test ({frozen})");
    assert!(kept_ok > 8, "physical cells must be kept unchanged by the iso freeze select ({kept_ok})");
    assert!(
        got_body == frozen,
        "the iso with-body freeze select dropped the body source: only {got_body}/{frozen} frozen \
         cells got the inward gravity kick (0 for the plain freeze select)",
    );
    assert!(
        worst_outward > -1e-9,
        "an iso frozen cell got a spurious OUTWARD body kick: {worst_outward:e}",
    );
}

#[test]
fn central_gravity_pulls_fluid_inward_through_evolve() {
    type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let dx = 2.0 * L / N as f64;
    // uniform fluid AT REST: no pressure gradient, so the ONLY force is gravity -> the fluid
    // accelerates purely radially inward. (mom=0, nrg=1/(gamma-1) <=> vel=0, pre=1.0.)
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim construction failed")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.0, 0.0]),
            pre: 1.0,
        })
        .build()
        .with_bodies(central_mass());
    assert!(
        sim.has_bodies(),
        "with_bodies must register the central mass"
    );

    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sim.geom.allocated);
    // a few steps — enough for gravity to act, before any infall collapse.
    evolve(&mut sim, &sub, 0.05).expect("evolution with body gravity failed");

    let rho = &sim.fields.prim.rho;
    let mut checked = 0usize;
    for c in sim.geom.interior.iter() {
        let x = -L + (c[0] as f64 + 0.5) * dx;
        let y = -L + (c[1] as f64 + 0.5) * dx;
        let r = (x * x + y * y).sqrt();
        let den = *sim.fields.cons.den.view().at(c);
        let mx = *sim.fields.cons.mom[0].view().at(c);
        let my = *sim.fields.cons.mom[1].view().at(c);

        assert!(
            den.is_finite() && den > 0.0,
            "density must stay positive/finite at {c:?}: {den}"
        );
        assert!(
            mx.is_finite() && my.is_finite(),
            "momentum non-finite at {c:?}"
        );
        // gravity conserves mass; with no pressure gradient + outflow BC, den stays ~1 in the
        // interior over a few steps.
        assert!(
            *rho.view().at(c) > 0.0,
            "prim.rho must stay positive at {c:?}"
        );

        // away from the softened core + the boundary, the radial momentum must be INWARD.
        if (0.3..0.8).contains(&r)
            && c[0] >= 3
            && c[0] < N as isize - 3
            && c[1] >= 3
            && c[1] < N as isize - 3
        {
            let mom_r = (mx * x + my * y) / r; // momentum projected on the outward radial unit
            assert!(
                mom_r < 0.0,
                "fluid must be pulled inward at r={r:.3} {c:?}: mom_r={mom_r:.3e}"
            );
            checked += 1;
        }
    }
    assert!(checked > 20, "too few interior cells checked ({checked})");
}

// -----------------------------------------------------------------------------
// GPU<->CPU parity of the body_gravity_source kernel.
// -----------------------------------------------------------------------------
#[cfg(feature = "gpu")]
#[test]
fn body_gravity_gpu_matches_cpu() {
    use symbi_algebra::Domain;
    use symbi_grid::Field;
    use symbi_xpu::{DeviceSpace, DeviceMemory};
    use symbi_xpu::{ExecutionSpace, MemorySpace};

    fn build<S: ExecutionSpace, Mem: MemorySpace>()
    -> SimState<Newtonian, 2, Cartesian, IdealGas<f64>, S, Mem> {
        let dx = 2.0 * L / N as f64;
        SimState::<Newtonian, 2, Cartesian, IdealGas<f64>, S, Mem>::build(
            Newtonian,
            IdealGas { gamma: GAMMA },
            Cartesian,
        )
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim construction failed")
        // a non-trivial state so the diff is meaningful (varying rho + nonzero mom). conserved
        // den=1+0.5g, mom=(0.1y,-0.1x), nrg=2+g inverted to prim:
        //   vel[k]=mom[k]/rho, pre=(gamma-1)*(nrg-0.5*rho*|v|^2).
        .set_initial(|[x, y]| {
            let g = (-(x * x + y * y) / 0.3).exp();
            let rho = 1.0 + 0.5 * g;
            let (mx, my) = (0.1 * y, -0.1 * x);
            let nrg = 2.0 + g;
            let (vx, vy) = (mx / rho, my / rho);
            let pre = (GAMMA - 1.0) * (nrg - 0.5 * rho * (vx * vx + vy * vy));
            Prim {
                rho,
                vel: Tensor::new([vx, vy]),
                pre,
            }
        })
        .build()
        // a BLACK HOLE (gravity + Bondi-Hoyle accretion) so the GPU diff covers both effects.
        .with_bodies(BodyCollection::new().add(Body::black_hole(
            0,
            Tensor::new([0.0, 0.0]),
            Tensor::zeros(),
            MASS,
            0.1,
            SOFT,
            8.0,
            0.3,
            0.5, // sink_rate, sink_delta, accretion_radius
        )))
    }

    fn cmp<MH: MemorySpace, MD: MemorySpace>(
        dom: &Domain<2>,
        host: &Field<f64, 2, MH>,
        dev: &Field<f64, 2, MD>,
        what: &str,
    ) {
        for c in dom.iter() {
            let (h, g) = (*host.view().at(c), *dev.view().at(c));
            assert!(g.is_finite(), "{what} at {c:?} non-finite on GPU");
            let rel = (g - h).abs() / h.abs().max(1.0);
            assert!(
                rel < 1e-9,
                "{what} at {c:?}: gpu {g} != cpu {h} (rel {rel:e})"
            );
        }
    }

    let host = build::<CpuSpace, HostMemory>();
    let dev = build::<DeviceSpace, DeviceMemory>();
    let hset =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &host.geom.allocated);
    let dset =
        AdiabaticSubstrateKernelSet::<DeviceMemory, f64, 2>::new(GAMMA, 0.4, &dev.geom.allocated);

    hset.body_source(&host, 0.01);
    dset.body_source(&dev, 0.01);
    // launches are asynchronous: drain the device queue before the host reads
    // the unified buffers (the host-read barrier).
    symbi::regimes::substrate_gpu::device_sync::<DeviceMemory>();

    let interior = &host.geom.interior;
    for k in 0..2 {
        cmp(
            interior,
            &host.fields.cons.mom[k],
            &dev.fields.cons.mom[k],
            "cons.mom (body grav)",
        );
    }
    let (hn, dn) = (
        host.fields.cons.nrg_field().unwrap(),
        dev.fields.cons.nrg_field().unwrap(),
    );
    cmp(interior, hn, dn, "cons.nrg (body grav)");
    // gravity does not touch density.
    cmp(
        interior,
        &host.fields.cons.den,
        &dev.fields.cons.den,
        "cons.den (unchanged)",
    );
}

#[test]
fn black_hole_records_accretion_without_changing_mass() {
    // a black hole (gravity + accretion) embedded in dense fluid: the fluid is removed by the
    // sink + the accretion RECORDED into the diagnostic (total_accreted_mass), but the BH's
    // GRAVITATING mass is held FIXED (fixed-potential sink — the central potential must not
    // drift).
    type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let dx = 2.0 * L / N as f64;
    let m_init = 1.0;
    // dense fluid at rest; the inner region (within r_acc) is the accretion reservoir. (den=2,
    // mom=0, nrg=1/(gamma-1) <=> rho=2, vel=0, pre=1.0.)
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim construction failed")
        .set_initial(|_| Prim {
            rho: 2.0,
            vel: Tensor::new([0.0, 0.0]),
            pre: 1.0,
        })
        .build()
        .with_bodies(BodyCollection::new().add(Body::black_hole(
            0,
            Tensor::new([0.0, 0.0]),
            Tensor::zeros(),
            m_init,
            0.1,
            SOFT,
            10.0,
            1.0,
            0.5, // sink_rate, sink_delta, accretion_radius
        )));

    let mut fluid_mass0 = 0.0;
    for c in sim.geom.interior.iter() {
        fluid_mass0 += *sim.fields.cons.den.view().at(c);
    }
    fluid_mass0 *= dx * dx;

    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sim.geom.allocated);
    evolve(&mut sim, &sub, 0.05).expect("evolution with accreting black hole failed");

    let body = sim.immersed.as_ref().unwrap().bodies.get(0);
    // the gravitating mass is UNCHANGED — the accretion never touches body.mass.
    assert_eq!(
        body.mass, m_init,
        "BH gravitating mass must stay fixed: {} -> {}",
        m_init, body.mass
    );
    // the would-be accretion is recorded in the diagnostic.
    if let symbi_ib::BodyKind::BlackHole {
        total_accreted_mass,
        ..
    } = body.kind
    {
        assert!(
            total_accreted_mass > 0.0,
            "total_accreted_mass must record accretion: {total_accreted_mass}"
        );
        assert!(
            total_accreted_mass.is_finite(),
            "total_accreted_mass non-finite"
        );
        // the recorded accretion ~ the fluid mass the sink removed (same sign + order; the
        // outflow BC + 1st-order feedback timing make this approximate, not exact).
        let mut fluid_mass1 = 0.0;
        for c in sim.geom.interior.iter() {
            fluid_mass1 += *sim.fields.cons.den.view().at(c);
        }
        fluid_mass1 *= dx * dx;
        assert!(
            fluid_mass1 < fluid_mass0,
            "fluid must lose mass to the sink: {fluid_mass0} -> {fluid_mass1}"
        );
        let fluid_loss = fluid_mass0 - fluid_mass1;
        assert!(
            total_accreted_mass > 0.1 * fluid_loss && total_accreted_mass < 10.0 * fluid_loss,
            "recorded accretion {total_accreted_mass} not consistent with fluid loss {fluid_loss}"
        );
    } else {
        panic!("body 0 should be a BlackHole");
    }
    // the disk-on-BH force is recorded + finite (a diagnostic; it does not drive the prescribed orbit).
    assert!(
        body.force[0].is_finite() && body.force[1].is_finite(),
        "BH force non-finite"
    );
}

// -----------------------------------------------------------------------------
// GPU<->CPU parity of the backward feedback (body_feedback kernel + device reduction).
// -----------------------------------------------------------------------------
#[cfg(feature = "gpu")]
#[test]
fn body_feedback_gpu_matches_cpu() {
    use symbi_xpu::{DeviceSpace, DeviceMemory};
    use symbi_xpu::{ExecutionSpace, MemorySpace};

    fn build<S: ExecutionSpace, Mem: MemorySpace>()
    -> SimState<Newtonian, 2, Cartesian, IdealGas<f64>, S, Mem> {
        let dx = 2.0 * L / N as f64;
        // same non-trivial conserved state as body_gravity_gpu_matches_cpu, inverted to prim.
        SimState::<Newtonian, 2, Cartesian, IdealGas<f64>, S, Mem>::build(
            Newtonian,
            IdealGas { gamma: GAMMA },
            Cartesian,
        )
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim construction failed")
        .set_initial(|[x, y]| {
            let g = (-(x * x + y * y) / 0.3).exp();
            let rho = 1.0 + 0.5 * g;
            let (mx, my) = (0.1 * y, -0.1 * x);
            let nrg = 2.0 + g;
            let (vx, vy) = (mx / rho, my / rho);
            let pre = (GAMMA - 1.0) * (nrg - 0.5 * rho * (vx * vx + vy * vy));
            Prim {
                rho,
                vel: Tensor::new([vx, vy]),
                pre,
            }
        })
        .build()
        .with_bodies(BodyCollection::new().add(Body::black_hole(
            0,
            Tensor::new([0.0, 0.0]),
            Tensor::zeros(),
            MASS,
            0.1,
            SOFT,
            8.0,
            0.3,
            0.5,
        )))
    }

    let host = build::<CpuSpace, HostMemory>();
    let dev = build::<DeviceSpace, DeviceMemory>();
    let hset =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &host.geom.allocated);
    let dset =
        AdiabaticSubstrateKernelSet::<DeviceMemory, f64, 2>::new(GAMMA, 0.4, &dev.geom.allocated);

    hset.body_feedback(&host, 0.01);
    dset.body_feedback(&dev, 0.01);

    let hd = host.immersed.as_ref().unwrap().diagnostics.consolidate();
    let dd = dev.immersed.as_ref().unwrap().diagnostics.consolidate();
    assert_eq!(hd.len(), dd.len(), "delta count mismatch");
    let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(1.0);
    for (h, d) in hd.iter().zip(dd.iter()) {
        for k in 0..2 {
            assert!(
                rel(h.force_delta[k], d.force_delta[k]) < 1e-9,
                "force[{k}] body {}: cpu {} gpu {}",
                h.idx,
                h.force_delta[k],
                d.force_delta[k]
            );
        }
        assert!(
            rel(h.torque_delta[2], d.torque_delta[2]) < 1e-9,
            "torque_z body {}: cpu {} gpu {}",
            h.idx,
            h.torque_delta[2],
            d.torque_delta[2]
        );
        assert!(
            rel(h.mass_delta, d.mass_delta) < 1e-9,
            "mass body {}: cpu {} gpu {}",
            h.idx,
            h.mass_delta,
            d.mass_delta
        );
    }
    // body 0 (the BH) actually accreted + felt a force.
    assert!(
        hd[0].mass_delta > 0.0,
        "BH accreted nothing: {}",
        hd[0].mass_delta
    );
}

// -----------------------------------------------------------------------------
// curvilinear (polar r-phi) body gravity: a central mass must produce PURELY RADIAL
// gravity in the physical (r, phi) momentum components.
// -----------------------------------------------------------------------------
#[test]
fn curvilinear_central_gravity_is_radial() {
    use symbi_geometry::Cylindrical;
    // a 2D cylindrical r-phi (disk-plane) sim; central mass at the cartesian origin.
    type CylSim = SimState<Newtonian, 2, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>;
    let (nr, nphi) = (16usize, 8usize);
    let (r_lo, dr) = (1.0_f64, 1.0 / 16.0);
    let (phi_lo, dphi) = (0.2_f64, 1.0 / 8.0);
    // fluid at rest; only gravity acts. (den=1, mom=0, nrg=1/(gamma-1) <=> rho=1, vel=0, pre=1.0.)
    let sim = CylSim::build(Newtonian, IdealGas { gamma: GAMMA }, Cylindrical)
        .cells([nr, nphi])
        .origin([r_lo, phi_lo])
        .spacing([dr, dphi])
        .boundaries(Boundaries::per_axis([
            [BoundaryType::Outflow, BoundaryType::Outflow],
            [BoundaryType::Periodic, BoundaryType::Periodic],
        ]))
        .allocate()
        .expect("cylindrical sim construction failed")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.0, 0.0]),
            pre: 1.0,
        })
        .build()
        .with_bodies(BodyCollection::new().add(Body::gravitational(
            0,
            Tensor::new([0.0, 0.0]),
            Tensor::zeros(),
            MASS,
            0.1,
            SOFT,
        )));

    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sim.geom.allocated);
    sub.body_source(&sim, 0.01); // direct forward source (no hydro kernels needed)

    let mut checked = 0usize;
    for c in sim.geom.interior.iter() {
        let mom_r = *sim.fields.cons.mom[0].view().at(c); // physical radial momentum
        let mom_phi = *sim.fields.cons.mom[1].view().at(c); // physical azimuthal momentum
        let den = *sim.fields.cons.den.view().at(c);
        assert!(
            mom_r.is_finite() && mom_phi.is_finite(),
            "non-finite at {c:?}"
        );
        // central gravity is purely radial inward: mom_r < 0, mom_phi == 0 (the projection
        // onto phi_hat cancels for a central force).
        assert!(
            mom_r < 0.0,
            "radial momentum must be inward at {c:?}: {mom_r:e}"
        );
        assert!(
            mom_phi.abs() < 1e-12,
            "azimuthal momentum must vanish (central) at {c:?}: {mom_phi:e}"
        );
        assert_eq!(den, 1.0, "gravity must not change density at {c:?}: {den}");
        checked += 1;
    }
    assert!(checked > 8, "too few cells checked ({checked})");
}

#[cfg(feature = "gpu")]
#[test]
fn curvilinear_body_source_gpu_matches_cpu() {
    use symbi_geometry::Cylindrical;
    use symbi_xpu::{DeviceSpace, DeviceMemory};
    use symbi_xpu::{ExecutionSpace, MemorySpace};

    fn build<S: ExecutionSpace, Mem: MemorySpace>()
    -> SimState<Newtonian, 2, Cylindrical, IdealGas<f64>, S, Mem> {
        let (nr, nphi) = (16usize, 8usize);
        SimState::<Newtonian, 2, Cylindrical, IdealGas<f64>, S, Mem>::build(
            Newtonian,
            IdealGas { gamma: GAMMA },
            Cylindrical,
        )
        .cells([nr, nphi])
        .origin([1.0, 0.2])
        .spacing([1.0 / 16.0, 1.0 / 8.0])
        .boundaries(Boundaries::per_axis([
            [BoundaryType::Outflow, BoundaryType::Outflow],
            [BoundaryType::Periodic, BoundaryType::Periodic],
        ]))
        .allocate()
        .expect("cyl sim")
        // conserved den=1+0.2*c[0]/16, mom=(0.05,-0.03), nrg=2 inverted to prim. the radial index
        // recovers from the center coordinate: c[0] = (r - r_lo)/dr - 0.5 = (r - 1.0)*16 - 0.5.
        .set_initial(|[r, _phi]| {
            let ir = (r - 1.0) * 16.0 - 0.5;
            let rho = 1.0 + 0.2 * ir / 16.0;
            let (vr, vphi) = (0.05 / rho, -0.03 / rho);
            let pre = (GAMMA - 1.0) * (2.0 - 0.5 * rho * (vr * vr + vphi * vphi));
            Prim {
                rho,
                vel: Tensor::new([vr, vphi]),
                pre,
            }
        })
        .build()
        // off-origin BH so gravity has a non-trivial phi component + accretion exercises the sink.
        .with_bodies(BodyCollection::new().add(Body::black_hole(
            0,
            Tensor::new([0.3, 0.4]),
            Tensor::zeros(),
            MASS,
            0.1,
            SOFT,
            6.0,
            0.5,
            0.4,
        )))
    }

    let host = build::<CpuSpace, HostMemory>();
    let dev = build::<DeviceSpace, DeviceMemory>();
    let hset =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &host.geom.allocated);
    let dset =
        AdiabaticSubstrateKernelSet::<DeviceMemory, f64, 2>::new(GAMMA, 0.4, &dev.geom.allocated);
    hset.body_source(&host, 0.01);
    dset.body_source(&dev, 0.01);
    // launches are asynchronous: drain the device queue before the host reads
    // the unified buffers (the host-read barrier).
    symbi::regimes::substrate_gpu::device_sync::<DeviceMemory>();

    let interior = &host.geom.interior;
    let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(1.0);
    for c in interior.iter() {
        for fld in ["den", "mom0", "mom1", "nrg"] {
            let (h, d) = match fld {
                "den" => (
                    *host.fields.cons.den.view().at(c),
                    *dev.fields.cons.den.view().at(c),
                ),
                "mom0" => (
                    *host.fields.cons.mom[0].view().at(c),
                    *dev.fields.cons.mom[0].view().at(c),
                ),
                "mom1" => (
                    *host.fields.cons.mom[1].view().at(c),
                    *dev.fields.cons.mom[1].view().at(c),
                ),
                _ => (
                    *host.fields.cons.nrg_field().unwrap().view().at(c),
                    *dev.fields.cons.nrg_field().unwrap().view().at(c),
                ),
            };
            assert!(
                d.is_finite() && rel(h, d) < 1e-9,
                "{fld} at {c:?}: cpu {h} gpu {d}"
            );
        }
    }
}

// -----------------------------------------------------------------------------
// 3D spherical body gravity: a central mass must produce PURELY RADIAL gravity in the
// physical (r, theta, phi) components.
// -----------------------------------------------------------------------------
#[test]
fn spherical_central_gravity_is_radial() {
    use symbi_geometry::Spherical;
    type SphSim = SimState<Newtonian, 3, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
    let (nr, nth, nph) = (8usize, 6usize, 6usize);
    // den=1, mom=0, nrg=1/(gamma-1) <=> rho=1, vel=0, pre=1.0.
    let sim = SphSim::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells([nr, nth, nph])
        .origin([1.0, 0.5, 0.2])
        .spacing([1.0 / 8.0, 1.0 / 6.0, 1.0 / 6.0])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("spherical sim construction failed")
        .set_initial(|_| Prim {
            rho: 1.0,
            vel: Tensor::new([0.0, 0.0, 0.0]),
            pre: 1.0,
        })
        .build()
        .with_bodies(BodyCollection::new().add(Body::gravitational(
            0,
            Tensor::new([0.0, 0.0, 0.0]),
            Tensor::zeros(),
            MASS,
            0.1,
            SOFT,
        )));

    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 3>::new(GAMMA, 0.4, &sim.geom.allocated);
    sub.body_source(&sim, 0.01);

    let mut checked = 0usize;
    for c in sim.geom.interior.iter() {
        let mom_r = *sim.fields.cons.mom[0].view().at(c);
        let mom_th = *sim.fields.cons.mom[1].view().at(c);
        let mom_ph = *sim.fields.cons.mom[2].view().at(c);
        assert!(
            mom_r < 0.0,
            "radial momentum must be inward at {c:?}: {mom_r:e}"
        );
        assert!(
            mom_th.abs() < 1e-12,
            "theta momentum must vanish (central) at {c:?}: {mom_th:e}"
        );
        assert!(
            mom_ph.abs() < 1e-12,
            "phi momentum must vanish (central) at {c:?}: {mom_ph:e}"
        );
        checked += 1;
    }
    assert!(checked > 8, "too few cells checked ({checked})");
}

#[cfg(feature = "gpu")]
#[test]
fn spherical_3d_body_gpu_matches_cpu() {
    use symbi_geometry::Spherical;
    use symbi_xpu::{DeviceSpace, DeviceMemory};
    use symbi_xpu::{ExecutionSpace, MemorySpace};

    fn build<S: ExecutionSpace, Mem: MemorySpace>()
    -> SimState<Newtonian, 3, Spherical, IdealGas<f64>, S, Mem> {
        SimState::<Newtonian, 3, Spherical, IdealGas<f64>, S, Mem>::build(
            Newtonian,
            IdealGas { gamma: GAMMA },
            Spherical,
        )
        .cells([8, 6, 6])
        .origin([1.0, 0.5, 0.2])
        .spacing([1.0 / 8.0, 1.0 / 6.0, 1.0 / 6.0])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sph sim")
        // conserved den=1+0.1*c[0]/8, mom=(0.04,-0.02,0.01), nrg=2 inverted to prim. radial index
        // recovers from the center coordinate: c[0] = (r - r_lo)/dr - 0.5 = (r - 1.0)*8 - 0.5.
        .set_initial(|[r, _th, _ph]| {
            let ir = (r - 1.0) * 8.0 - 0.5;
            let rho = 1.0 + 0.1 * ir / 8.0;
            let (vr, vth, vph) = (0.04 / rho, -0.02 / rho, 0.01 / rho);
            let pre = (GAMMA - 1.0) * (2.0 - 0.5 * rho * (vr * vr + vth * vth + vph * vph));
            Prim {
                rho,
                vel: Tensor::new([vr, vth, vph]),
                pre,
            }
        })
        .build()
        // off-origin BH: gravity has theta/phi components + accretion + a 3D torque in feedback.
        .with_bodies(BodyCollection::new().add(Body::black_hole(
            0,
            Tensor::new([0.5, 0.3, 0.7]),
            Tensor::zeros(),
            MASS,
            0.1,
            SOFT,
            5.0,
            0.5,
            0.4,
        )))
    }

    let host = build::<CpuSpace, HostMemory>();
    let dev = build::<DeviceSpace, DeviceMemory>();
    let hset =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 3>::new(GAMMA, 0.4, &host.geom.allocated);
    let dset =
        AdiabaticSubstrateKernelSet::<DeviceMemory, f64, 3>::new(GAMMA, 0.4, &dev.geom.allocated);

    // forward source diff.
    hset.body_source(&host, 0.01);
    dset.body_source(&dev, 0.01);
    // launches are asynchronous: drain the device queue before the host reads
    // the unified buffers (the host-read barrier).
    symbi::regimes::substrate_gpu::device_sync::<DeviceMemory>();
    let interior = &host.geom.interior;
    let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(1.0);
    for c in interior.iter() {
        for k in 0..3 {
            let (h, d) = (
                *host.fields.cons.mom[k].view().at(c),
                *dev.fields.cons.mom[k].view().at(c),
            );
            assert!(
                d.is_finite() && rel(h, d) < 1e-9,
                "mom[{k}] at {c:?}: cpu {h} gpu {d}"
            );
        }
        let (h, d) = (
            *host.fields.cons.den.view().at(c),
            *dev.fields.cons.den.view().at(c),
        );
        assert!(rel(h, d) < 1e-9, "den at {c:?}: cpu {h} gpu {d}");
    }
    // backward 3D-torque feedback diff.
    hset.body_feedback(&host, 0.01);
    dset.body_feedback(&dev, 0.01);
    let hd = host.immersed.as_ref().unwrap().diagnostics.consolidate();
    let dd = dev.immersed.as_ref().unwrap().diagnostics.consolidate();
    for (h, d) in hd.iter().zip(dd.iter()) {
        for k in 0..3 {
            assert!(
                rel(h.force_delta[k], d.force_delta[k]) < 1e-9,
                "force[{k}]: {} {}",
                h.force_delta[k],
                d.force_delta[k]
            );
            assert!(
                rel(h.torque_delta[k], d.torque_delta[k]) < 1e-9,
                "torque[{k}]: {} {}",
                h.torque_delta[k],
                d.torque_delta[k]
            );
        }
        assert!(
            rel(h.mass_delta, d.mass_delta) < 1e-9,
            "mass: {} {}",
            h.mass_delta,
            d.mass_delta
        );
    }
    // a 3D torque component (tx or ty) is actually non-trivial for the off-axis body.
    assert!(
        hd[0].torque_delta[0].abs() + hd[0].torque_delta[1].abs() > 0.0,
        "3D torque should be non-zero"
    );
}
