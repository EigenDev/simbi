// =============================================================================
// refinement_restart_depth.rs
//
// resuming a checkpoint at a greater refinement depth than it was written with — the bootstrap
// ladder, where a rung converges at its own resolution and the next rung resumes it with one more
// level. without it every depth pays the initial transient again at its own (far more expensive)
// resolution.
//
// three properties make it correct, and each is checkable independently:
//
//   - the levels the file carries are loaded, and the levels beyond it are injected from their
//     parents. an off-by-one either way is silent: injecting over a level that should have been
//     loaded discards converged data and replaces it with a coarser copy of itself, which is
//     smooth, finite and plausible.
//   - injection is exactly conservative on cell averages, which is the entire reason a
//     piecewise-constant operator is sufficient here.
//   - level `i` must occupy the same region at every depth. that is a property of a config's
//     refinement schedule, not of the code — a schedule deriving its regions from the level count
//     violates it — so it is verified against the file rather than assumed, and the violation is an
//     error rather than a field laid over the wrong region.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;
type Hier = Hierarchy<Newtonian, 1, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const N: usize = 64;
const L0: f64 = 1.0;

/// a smooth, strongly structured state so an injected level inherits real variation: a flat
/// profile would make injection indistinguishable from any other initialization.
fn ic(x: [f64; 1]) -> Prim<f64, 1> {
    let g = (-(x[0] * x[0]) / 0.02).exp();
    Prim {
        rho: 1.0 + 3.0 * g,
        vel: Tensor::new([0.3 * x[0]]),
        pre: 1.0 + 2.0 * g,
    }
}

/// the region schedule: level `l` covers the inner half of its parent about the origin. fixed
/// geometry — level `l`'s box does not depend on how many levels follow it, which is what makes a
/// deeper restart meaningful at all.
fn region(ll: usize) -> RefinementRegion<1> {
    let half = L0 / (1u64 << ll) as f64;
    RefinementRegion {
        x_lo: [-half],
        x_hi: [half],
    }
}

fn ladder(levels: usize, seed: bool) -> Hier {
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .origin([-L0])
        .spacing([2.0 * L0 / N as f64])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("root allocates")
        .set_initial(ic)
        .build();
    let ck = Kset::new(GAMMA, CFL, &coarse.geom.allocated);
    let regions: Vec<RefinementRegion<1>> = (1..levels).map(region).collect();
    let mut hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, |s| {
        Kset::new(GAMMA, CFL, &s.geom.allocated)
    })
    .expect("the ladder builds");
    if seed {
        for level in hier.levels.iter_mut().skip(1) {
            level.state.seed_cells(ic);
        }
    }
    hier
}

/// the conserved mass and energy a level holds over the region a finer level would cover — the
/// integral injection must preserve.
fn mass_energy_over(sim: &Sim, lo: f64, hi: f64) -> (f64, f64) {
    let bg = sim.geom.block_geometry(sim.physics.metric);
    let nrg = sim
        .fields
        .cons
        .nrg_field()
        .expect("adiabatic carries energy");
    let (mut m, mut e) = (0.0, 0.0);
    for c in sim.geom.interior.iter() {
        let x = sim.geom.centroid(c)[0];
        if x < lo || x >= hi {
            continue;
        }
        let dv = bg.labframe_volume(c, sim.motion.a);
        m += *sim.fields.cons.den.view().at(c) * dv;
        e += *nrg.view().at(c) * dv;
    }
    (m, e)
}

#[test]
fn an_injected_level_holds_exactly_what_its_parent_held_over_the_same_region() {
    // the justification for piecewise-constant injection, stated as an equality. a coarse cell's
    // average replicated to its children preserves the integral over that cell exactly, so the new
    // level enters the run carrying precisely the mass and energy its parent held there. anything
    // else means the ladder manufactures or destroys conserved quantity at every rung.
    let mut hier = ladder(3, true);
    let deep = {
        // a fourth level, built but never seeded: exactly the state a deeper restart is in after
        // the file's levels have been loaded.
        let mut h = ladder(4, false);
        for ll in 1..3 {
            h.levels[ll].state.seed_cells(ic);
        }
        h.inject_level_from_parent(3).expect("injection succeeds");
        h
    };
    let _ = &mut hier;

    let r = region(3);
    let (want_m, want_e) = mass_energy_over(&deep.levels[2].state, r.x_lo[0], r.x_hi[0]);
    let (got_m, got_e) = mass_energy_over(&deep.levels[3].state, r.x_lo[0], r.x_hi[0]);

    // the premise: the region must actually hold something, or preserving it is trivial.
    assert!(
        want_m > 0.0 && want_e > 0.0,
        "the injected region holds no mass or energy ({want_m}, {want_e}); preserving it proves \
         nothing"
    );
    assert!(
        (got_m / want_m - 1.0).abs() < 1.0e-13,
        "the injected level holds mass {got_m} against its parent's {want_m} over the same region \
         (relative {:e}); injection is exactly conservative on cell averages, so this is an \
         equality and not a tolerance",
        (got_m / want_m - 1.0).abs()
    );
    assert!(
        (got_e / want_e - 1.0).abs() < 1.0e-13,
        "the injected level holds energy {got_e} against its parent's {want_e} (relative {:e})",
        (got_e / want_e - 1.0).abs()
    );

    // and the primitives were recovered: a level whose conserved state was written but never run
    // through c2p reports zero density everywhere to the first diagnostic that reads it.
    let rho = *deep.levels[3]
        .state
        .fields
        .prim
        .rho
        .view()
        .at(deep.levels[3].state.geom.interior.iter().next().unwrap());
    assert!(
        rho > 0.0,
        "the injected level's primitives are still empty (rho = {rho}); the conserved-to-primitive \
         recovery did not run"
    );
}

#[test]
fn injection_is_refused_where_it_would_break_a_constraint() {
    // the root has no parent to inject from, and an out-of-range level is a caller error rather
    // than a data condition. both are refused by name rather than panicking or quietly no-opping.
    let mut hier = ladder(2, true);
    let root = hier.inject_level_from_parent(0).unwrap_err();
    assert!(root.contains("root"), "{root}");
    let past = hier.inject_level_from_parent(7).unwrap_err();
    assert!(past.contains("level(s)"), "{past}");
}

#[test]
fn a_checkpoint_reports_the_levels_it_actually_carries() {
    // the count that decides which levels are loaded and which are injected. taken from the file
    // rather than from the config, so a truncated or hand-edited checkpoint cannot make a level
    // silently start from zeros while the run reports a successful restart.
    let dir = std::env::temp_dir().join(format!("restart_depth_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");

    for levels in [1usize, 3, 5] {
        let hier = ladder(levels, true);
        let path = dir.join(format!("l{levels}.h5"));
        let states: Vec<&Sim> = hier.levels.iter().map(|l| &l.state).collect();
        symbi_sim::checkpoint::write_hierarchy_checkpoint(
            &states,
            path.to_str().expect("utf-8"),
            &Default::default(),
        )
        .expect("checkpoint written");

        let got = symbi_sim::checkpoint::checkpoint_level_count(path.to_str().expect("utf-8"))
            .expect("the level count reads");
        assert_eq!(
            got, levels,
            "a checkpoint written with {levels} level(s) reports {got}; the count decides which \
             levels are loaded and which are initialized from their parents"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_level_whose_region_moved_between_runs_is_refused() {
    // the compatibility property a deeper restart rests on: level `i` occupies the same region at
    // every depth. that holds for a schedule of fixed regions and fails for one deriving its
    // regions from the level count — and the failure is invisible, since loading a level's data
    // onto a different region yields a field that is smooth, finite and wrong everywhere.
    let dir = std::env::temp_dir().join(format!("restart_geom_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("c.h5");

    let hier = ladder(3, true);
    let states: Vec<&Sim> = hier.levels.iter().map(|l| &l.state).collect();
    symbi_sim::checkpoint::write_hierarchy_checkpoint(
        &states,
        path.to_str().expect("utf-8"),
        &Default::default(),
    )
    .expect("checkpoint written");
    let p = path.to_str().expect("utf-8");

    // the matching hierarchy verifies cleanly on every level it shares with the file.
    let same = ladder(3, true);
    for ll in 0..3 {
        symbi_sim::checkpoint::verify_checkpoint_level_geometry(&same.levels[ll].state, p, ll)
            .unwrap_or_else(|e| {
                panic!("an identical hierarchy failed verification at level {ll}: {e}")
            });
    }

    // a schedule that placed level 1 somewhere else — the shape a level-count-dependent region
    // takes — must be refused, naming the level.
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .origin([-L0])
        .spacing([2.0 * L0 / N as f64])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("root allocates")
        .set_initial(ic)
        .build();
    let ck = Kset::new(GAMMA, CFL, &coarse.geom.allocated);
    let moved = [RefinementRegion {
        x_lo: [-L0 / 4.0],
        x_hi: [L0 / 4.0],
    }];
    let other = Hierarchy::with_refinement(coarse, ck, &moved, ProlongOrder::Ppm, |s| {
        Kset::new(GAMMA, CFL, &s.geom.allocated)
    })
    .expect("the shifted ladder builds");

    // the root is unchanged, so it must still verify — the check must discriminate, not refuse
    // everything once one level differs.
    symbi_sim::checkpoint::verify_checkpoint_level_geometry(&other.levels[0].state, p, 0)
        .expect("the root geometry is identical and must verify");

    let err = symbi_sim::checkpoint::verify_checkpoint_level_geometry(&other.levels[1].state, p, 1)
        .expect_err("a level covering a different region must be refused");
    let msg = format!("{err}");
    assert!(
        msg.contains("level 1"),
        "the refusal must name the level that moved: {msg}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_moving_mesh_checkpoint_verifies_against_the_comoving_rebuild() {
    // homologous mesh motion scales the stored bounds by a(t) at write time, while a
    // restart rebuilds the comoving (a = 1) grid and re-derives a(t) from the resume
    // time — a is a pure function of time, never integrated state. the region check
    // must therefore compare comoving against comoving: a checkpoint written
    // mid-expansion must verify against the fresh build, or every moving-mesh restart
    // is refused with a spurious "region moved" whose two spans differ by exactly a.
    let dir = std::env::temp_dir().join(format!("restart_motion_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("m.h5");

    let mut hier = ladder(1, true);
    hier.levels[0].state.motion.a = 1.7;
    hier.levels[0].state.motion.a_dot = 0.3;
    hier.levels[0].state.motion.homologous = true;
    write_at(&hier, &path);
    let p = path.to_str().expect("utf-8");

    let fresh = ladder(1, true);
    symbi_sim::checkpoint::verify_checkpoint_level_geometry(&fresh.levels[0].state, p, 0)
        .expect(
            "the comoving regions are identical; the stored bounds differ only by the \
             checkpoint's own scale factor, which is motion, not a region change",
        );
    let _ = std::fs::remove_dir_all(&dir);
}

/// every conserved cell of every level, in level then cell order — the bitwise fingerprint of a
/// hierarchy's state.
fn fingerprint(hier: &Hier) -> Vec<u64> {
    let mut out = Vec::new();
    for level in hier.levels.iter() {
        let sim = &level.state;
        let nrg = sim
            .fields
            .cons
            .nrg_field()
            .expect("adiabatic carries energy");
        for c in sim.geom.interior.iter() {
            out.push(sim.fields.cons.den.view().at(c).to_bits());
            out.push(sim.fields.cons.mom[0].view().at(c).to_bits());
            out.push(nrg.view().at(c).to_bits());
        }
    }
    out
}

fn write_at(hier: &Hier, path: &std::path::Path) {
    let states: Vec<&Sim> = hier.levels.iter().map(|l| &l.state).collect();
    symbi_sim::checkpoint::write_hierarchy_checkpoint(
        &states,
        path.to_str().expect("utf-8"),
        &Default::default(),
    )
    .expect("checkpoint written");
}

#[test]
fn restarting_at_the_same_depth_reproduces_the_state_bitwise() {
    // the off-by-one catcher. if the loaded/injected split is wrong by one in either direction the
    // run still starts and still looks physical: injecting over the finest level replaces converged
    // data with a coarser copy of itself, which is smooth, finite and entirely plausible. only a
    // bitwise comparison against the pre-checkpoint state distinguishes them.
    let dir = std::env::temp_dir().join(format!("restart_idem_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("c.h5");

    let mut original = ladder(4, true);
    original.evolve_steps(3).expect("evolve");
    let before = fingerprint(&original);
    write_at(&original, &path);

    let mut restarted = ladder(4, false);
    let stored = restarted
        .restore_from_checkpoint(path.to_str().expect("utf-8"))
        .expect("same-depth restart");
    assert_eq!(stored, 4, "a 4-level checkpoint must load all 4 levels");
    assert_eq!(
        fingerprint(&restarted),
        before,
        "a same-depth restart did not reproduce the checkpoint bitwise. one of the levels was \
         injected from its parent instead of loaded, which replaces converged data with a coarser \
         copy of itself and leaves a state that is smooth, finite and wrong"
    );

    // the premise: the levels must actually differ from a fresh injection, or "loaded" and
    // "injected" produce the same bits and the comparison above is vacuous.
    let mut all_injected = ladder(4, false);
    all_injected.levels[0].state.seed_cells(ic);
    for ll in 1..4 {
        all_injected
            .inject_level_from_parent(ll)
            .expect("injection succeeds");
    }
    assert_ne!(
        fingerprint(&all_injected),
        before,
        "an entirely injected hierarchy matches the evolved checkpoint bitwise; loading and \
         injecting are indistinguishable here, so nothing above discriminates"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_deeper_restart_loads_what_exists_and_injects_only_the_rest() {
    // the ladder step itself. the levels the file carries must come back bitwise — the whole point
    // is that their convergence is inherited rather than recomputed — while the new level is built
    // from its parent.
    let dir = std::env::temp_dir().join(format!("restart_deep_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("c.h5");

    let mut shallow = ladder(3, true);
    shallow.evolve_steps(3).expect("evolve");
    let before = fingerprint(&shallow);
    write_at(&shallow, &path);

    let mut deeper = ladder(4, false);
    let stored = deeper
        .restore_from_checkpoint(path.to_str().expect("utf-8"))
        .expect("deeper restart");
    assert_eq!(stored, 3, "a 3-level checkpoint carries 3 levels");
    assert_eq!(deeper.levels.len(), 4, "the run builds one level more");

    // the loaded levels, bitwise. `fingerprint` walks every level, so the deeper hierarchy's
    // fingerprint is compared over its first three levels only.
    let got: Vec<u64> = fingerprint(&deeper)
        .into_iter()
        .take(before.len())
        .collect();
    assert_eq!(
        got, before,
        "the levels the checkpoint carries did not come back bitwise on a deeper restart. \
         inheriting their converged state is the entire reason to ladder rather than start over"
    );

    // and the new level holds its parent's state over its own region, rather than zeros or the
    // initial condition.
    let r = region(3);
    let (want_m, _) = mass_energy_over(&deeper.levels[2].state, r.x_lo[0], r.x_hi[0]);
    let (got_m, _) = mass_energy_over(&deeper.levels[3].state, r.x_lo[0], r.x_hi[0]);
    assert!(
        (got_m / want_m - 1.0).abs() < 1.0e-13,
        "the new level holds mass {got_m} against its parent's {want_m} over the same region"
    );

    // the injected level must share its parent's clock: a level whose time disagreed would
    // desynchronize the subcycle on the first root step.
    assert_eq!(
        deeper.levels[3].state.time, deeper.levels[2].state.time,
        "the injected level's clock does not match its parent's"
    );
    assert!(
        deeper.levels[3].state.time > 0.0,
        "the restart did not restore the elapsed time, so the clock check above is vacuous"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_restart_may_not_drop_levels() {
    // the reverse direction is a different operation: discarding a level means restricting its data
    // onto its parent, and silently ignoring the extra level would throw away exactly the
    // resolution the checkpoint was run to obtain.
    let dir = std::env::temp_dir().join(format!("restart_drop_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("c.h5");
    write_at(&ladder(4, true), &path);

    let mut shallower = ladder(2, false);
    let err = shallower
        .restore_from_checkpoint(path.to_str().expect("utf-8"))
        .expect_err("a shallower restart must be refused");
    assert!(err.contains("never drop"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_restart_onto_a_moved_region_is_refused_by_the_restart_itself() {
    // the geometry check has to be wired, not merely available. every other assertion about it
    // calls the verifier directly, so a restart path that never invoked it would leave them all
    // green while loading a converged level's data onto a region it never occupied — a field that
    // is smooth, finite and wrong everywhere, discovered as an unexplained profile weeks later.
    let dir = std::env::temp_dir().join(format!("restart_wired_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("c.h5");
    let p = path.to_str().expect("utf-8");

    let mut original = ladder(3, true);
    original.evolve_steps(2).expect("evolve");
    write_at(&original, &path);

    // the premise: the same schedule must restart cleanly, or the refusal below could be any
    // unrelated failure.
    ladder(3, false)
        .restore_from_checkpoint(p)
        .expect("an identical schedule must restart");

    // a schedule whose level 1 sits somewhere else — the shape a level-count-dependent region takes.
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .origin([-L0])
        .spacing([2.0 * L0 / N as f64])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("root allocates")
        .set_initial(ic)
        .build();
    let ck = Kset::new(GAMMA, CFL, &coarse.geom.allocated);
    let moved = [
        RefinementRegion {
            x_lo: [-L0 / 4.0],
            x_hi: [L0 / 4.0],
        },
        region(2),
    ];
    let mut other = Hierarchy::with_refinement(coarse, ck, &moved, ProlongOrder::Ppm, |s| {
        Kset::new(GAMMA, CFL, &s.geom.allocated)
    })
    .expect("the shifted ladder builds");

    let err = other
        .restore_from_checkpoint(p)
        .expect_err("a restart onto a moved region must be refused");
    assert!(
        err.contains("level 1"),
        "the refusal must name the level whose region moved: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// =============================================================================
// the checkpoint's mesh description is written in reversed (storage) axis order — `global_cells` as
// [nx_D, .., nx_1] and the geometry groups named by storage slot — so that a reader's plot axes are
// not transposed. every gate above is one-dimensional, where that reversal is the identity, and a
// cubic grid hides it just as well. an anisotropic multi-dimensional grid is the only shape that
// exposes it, and reading the wrong axis rejects every legitimate restart it touches.
// =============================================================================

type Sim2 = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

#[test]
fn an_anisotropic_grid_verifies_against_its_own_axes() {
    let (nx, ny) = (32usize, 8usize);
    let (lx, ly) = (1.0f64, 0.25f64);
    let ic2 = |x: [f64; 2]| Prim {
        rho: 1.0 + 0.3 * (x[0] + x[1]),
        vel: Tensor::new([0.1, -0.2]),
        pre: 1.0,
    };
    let sim = Sim2::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([nx, ny])
        .origin([-0.5 * lx, -0.5 * ly])
        .spacing([lx / nx as f64, ly / ny as f64])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("sim")
        .set_initial(ic2)
        .build();

    // the premise: the two axes must differ in both cell count and physical extent, or a swapped
    // read agrees by coincidence and this proves nothing.
    assert_ne!(nx, ny, "the cell counts must differ");
    assert!(
        (lx - ly).abs() > 1.0e-12,
        "the physical extents must differ"
    );

    let dir = std::env::temp_dir().join(format!("restart_aniso_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("c.h5");
    let p = path.to_str().expect("utf-8");
    symbi_sim::checkpoint::write_checkpoint(&sim, p, &Default::default())
        .expect("checkpoint written");

    symbi_sim::checkpoint::verify_checkpoint_level_geometry(&sim, p, 0).unwrap_or_else(|e| {
        panic!(
            "a {nx}x{ny} grid failed to verify against the checkpoint it just wrote: {e}. the mesh \
             description is stored in reversed axis order, so reading it forward compares axis 0 \
             against the last axis's extent"
        )
    });

    // and the check must still discriminate on an anisotropic grid: a genuinely different grid is
    // refused. swapping the two axes is the sharpest case — same total cells, same total volume.
    let swapped = Sim2::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([ny, nx])
        .origin([-0.5 * ly, -0.5 * lx])
        .spacing([ly / ny as f64, lx / nx as f64])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("sim")
        .set_initial(ic2)
        .build();
    let err = symbi_sim::checkpoint::verify_checkpoint_level_geometry(&swapped, p, 0)
        .expect_err("a transposed grid must be refused");
    assert!(
        format!("{err}").contains("axis"),
        "the refusal must name the axis that disagrees: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
