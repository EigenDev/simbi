// =============================================================================
// covered_core_exclusion.rs
//
// a covered coarse cell beyond every stage kernel's read reach from the evolved
// cells takes no part in the run-wide decisions. the inactive core of a covered
// region is restored to the stage input every stage and replaced by the
// restriction, so a core cell reaches the evolved cells only through the stage
// kernels of the donor shell and the ghost fill, which read within the level's
// ghost width. on a three-level 2.5D newtonian MHD hierarchy with a draining
// sink (plm, rk2, contact EMF; donor shell 3, ghost width 2 on the root and 3 on
// the first refined level):
//   - the excluded core starts past the donor shell and the ghost width, at depth
//     5 on the root and 6 on the first refined level;
//   - poisoning every excluded cell with NaN before each of three root steps
//     changes no value outside the excluded cores, on any level;
//   - a perturbation one cell inside the read reach (depth 4) changes evolved
//     cells, so the comparison above can see an influence;
//   - an inadmissible state within the read reach still ends the run, and an
//     inadmissible state in the excluded core changes nothing;
//   - a checkpoint publishes NaN primitives for an excluded cell without an
//     admissible recovery, names the excluded box of a level that withholds one
//     (and of no other level), keeps its conserved
//     state, and a restart from it continues as the uninterrupted run does;
//   - with a rigid body, whose wall acts on covered coarse cells and sums their
//     force onto the body, every covered cell stays in the decisions.
//
// run: cargo test -p symbi --test covered_core_exclusion
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::{Domain, Tensor};
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, MagneticSpec, SurfaceSpec};
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Kern = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const N: usize = 64;
const DX: f64 = 1.0 / N as f64;
const DT: f64 = 1.0e-3;
const STEPS: usize = 3;

fn regions() -> [RefinementRegion<2>; 2] {
    [
        RefinementRegion { x_lo: [0.0625, 0.0625], x_hi: [0.9375, 0.9375] },
        RefinementRegion { x_lo: [0.3125, 0.3125], x_hi: [0.6875, 0.6875] },
    ]
}

fn kset(sim: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, 1.0, &sim.geom.allocated)
}

fn build() -> Hier {
    let bx = |y: f64| 0.4 + 0.2 * (2.0 * std::f64::consts::PI * y + 0.3).sin();
    let by = |x: f64| 0.2 + 0.2 * (2.0 * std::f64::consts::PI * x + 0.7).cos();
    let bump = |x: f64, y: f64| 0.3 * (-(((x - 0.5) / 0.07).powi(2) + ((y - 0.48) / 0.09).powi(2))).exp();
    let root = Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .spacing([DX; 2])
        .origin([0.0; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("root")
        .set_initial(|x| {
            let b = bump(x[0], x[1]);
            MhdPrim::new(
                Prim::adiabatic(Density(1.0 + b), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0 + b)),
                Tensor::new([bx(x[1]), by(x[0]), 0.1]),
            )
        })
        .seed_faces(|d, x| if d == 0 { bx(x[1]) } else { by(x[0]) })
        .build();
    let k = kset(&root);
    let h = Hier::with_refinement(root, k, &regions(), ProlongOrder::Ppm, kset).expect("hierarchy");
    h.seed_fine_from_coarse().expect("seed");
    let sink = BodyCollection::new().add(
        Body::black_hole(0, Tensor::new([0.5, 0.5]), Tensor::zeros(), 0.01, 0.08, 0.05, 0.0, 1.0, 0.08)
            .with_surface(SurfaceSpec::Drain)
            .with_magnetic(MagneticSpec::None),
    );
    let mut h = h.with_bodies(sink);
    h.prime();
    h.establish_decision_exclusion();
    h
}

fn excluded(h: &Hier, level: usize) -> Option<Domain<2>> {
    h.levels[level].state.composite_ownership.decision_excluded.clone()
}

/// every conserved, cell-field and primitive value of every interior cell outside the excluded
/// cores, per level
fn evolved_values(h: &Hier) -> Vec<Vec<([isize; 2], Vec<u64>)>> {
    h.levels
        .iter()
        .map(|l| {
            let s = &l.state;
            let core = s.composite_ownership.decision_excluded.clone();
            let m = s.fields.mhd.as_ref().expect("mhd");
            s.geom
                .interior
                .iter()
                .filter(|c| !core.as_ref().is_some_and(|d| d.contains(*c)))
                .map(|c| {
                    let cons = &s.fields.cons;
                    let mut v = vec![*cons.den.view().at(c), *cons.nrg_field().expect("nrg").view().at(c)];
                    v.extend((0..3).map(|k| *cons.mom[k].view().at(c)));
                    v.extend((0..3).map(|k| *m.bcell[k].view().at(c)));
                    v.push(*s.fields.prim.rho.view().at(c));
                    v.push(*s.fields.prim.pre.as_ref().expect("pre").view().at(c));
                    (c, v.iter().map(|x| x.to_bits()).collect())
                })
                .collect()
        })
        .collect()
}

/// set every conserved, cell-field and primitive value of `cells` on `level` from `value`
fn overwrite(h: &Hier, level: usize, cells: &[[isize; 2]], value: impl Fn(f64) -> f64) {
    let s = &h.levels[level].state;
    let m = s.fields.mhd.as_ref().expect("mhd");
    let cons = &s.fields.cons;
    let mut fields: Vec<&symbi_grid::Field<f64, 2, HostMemory>> = vec![&cons.den, cons.nrg_field().expect("nrg"), &s.fields.prim.rho];
    fields.extend(cons.mom.iter());
    fields.extend(m.bcell.b.iter());
    fields.push(s.fields.prim.pre.as_ref().expect("pre"));
    fields.extend(s.fields.prim.vel.iter());
    for c in cells {
        for f in &fields {
            let x = *f.view().at(*c);
            f.view_mut().set(*c, value(x));
        }
    }
}

/// leave a cell's gas with a negative internal energy: its total energy below its kinetic and
/// field energy
fn make_inadmissible(h: &Hier, level: usize, c: [isize; 2]) {
    let s = &h.levels[level].state;
    let cons = &s.fields.cons;
    let m = s.fields.mhd.as_ref().expect("mhd");
    let rho = *cons.den.view().at(c);
    let kinetic: f64 = 0.5 * (0..3).map(|k| cons.mom[k].view().at(c).powi(2)).sum::<f64>() / rho;
    let field: f64 = 0.5 * (0..3).map(|k| m.bcell[k].view().at(c).powi(2)).sum::<f64>();
    cons.nrg_field().expect("nrg").view_mut().set(c, kinetic + field - 0.05);
}

/// the covered cell of `level` at `depth` from the coverage edge, on the row through the center
fn cell_at_depth(h: &Hier, level: usize, depth: isize) -> [isize; 2] {
    let cov = h.levels[level].state.composite_ownership.coverage.clone().expect("covered");
    [cov.spaces[0].lo + depth, (cov.spaces[1].lo + cov.spaces[1].hi) / 2 + 3]
}

fn step(h: &mut Hier) {
    h.step_root_with_dt(DT);
    assert!(h.crash.is_none(), "the step halted: {:?}", h.crash);
}

#[test]
fn the_excluded_core_starts_past_the_donor_shell_and_the_ghost_width() {
    let h = build();
    for level in 0..2 {
        let s = &h.levels[level].state;
        let cov = s.composite_ownership.coverage.clone().expect("covered");
        let core = s.composite_ownership.inactive.clone().expect("inactive core");
        let ghost = s.geom.interior.spaces[0].lo - s.geom.allocated.spaces[0].lo;
        let shell = core.spaces[0].lo - cov.spaces[0].lo;
        assert_eq!((shell, ghost), [(3, 2), (3, 3)][level], "level {level}: the donor shell or the ghost width changed");
        let ex = excluded(&h, level).expect("an excluded core");
        for axis in 0..2 {
            assert_eq!(ex.spaces[axis].lo - cov.spaces[axis].lo, shell + ghost, "level {level} axis {axis}");
            assert_eq!(cov.spaces[axis].hi - ex.spaces[axis].hi, shell + ghost, "level {level} axis {axis}");
        }
    }
    assert!(excluded(&h, 2).is_none(), "the finest level has no covered cells");
}

#[test]
fn poisoning_the_excluded_core_changes_no_evolved_value() {
    let (mut clean, mut poisoned) = (build(), build());
    assert!(evolved_values(&clean) == evolved_values(&poisoned), "two builds differ; the comparison is meaningless");
    for _ in 0..STEPS {
        poisoned.establish_decision_exclusion();
        for level in 0..2 {
            let cells: Vec<[isize; 2]> = excluded(&poisoned, level).expect("excluded core").iter().collect();
            assert!(cells.len() > 100, "level {level}: {} excluded cells", cells.len());
            overwrite(&poisoned, level, &cells, |_| f64::NAN);
        }
        step(&mut clean);
        step(&mut poisoned);
        assert_eq!(clean.levels[0].state.time.to_bits(), poisoned.levels[0].state.time.to_bits(), "the clocks parted");
        assert!(evolved_values(&clean) == evolved_values(&poisoned), "a NaN in the excluded core reached an evolved value");
    }
}

#[test]
fn a_perturbation_within_the_read_reach_reaches_the_evolved_cells() {
    let (mut clean, mut perturbed) = (build(), build());
    for level in 0..2 {
        let c = cell_at_depth(&perturbed, level, 4);
        overwrite(&perturbed, level, &[c], |x| x * (1.0 + 1e-9));
    }
    step(&mut clean);
    step(&mut perturbed);
    let (a, b) = (evolved_values(&clean), evolved_values(&perturbed));
    let changed: usize = a.iter().zip(&b).map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p != q).count()).sum();
    assert!(changed > 10, "a perturbation at depth 4 changed {changed} evolved values; the poisoning gate cannot see an influence");
}

#[test]
fn an_inadmissible_state_within_the_read_reach_ends_the_run() {
    let mut h = build();
    let c = cell_at_depth(&h, 1, 4);
    make_inadmissible(&h, 1, c);
    h.step_root_with_dt(DT);
    let crash = h.crash.as_ref().expect("an inadmissible state at depth 4 was accepted");
    assert!(format!("{crash:?}").contains("FOFC"), "the run ended for another reason: {crash:?}");
}

#[test]
fn an_inadmissible_state_in_the_excluded_core_changes_nothing() {
    let (mut clean, mut marked) = (build(), build());
    for level in 0..2 {
        let c = cell_at_depth(&marked, level, 8);
        assert!(excluded(&marked, level).expect("core").contains(c));
        make_inadmissible(&marked, level, c);
    }
    for _ in 0..STEPS {
        step(&mut clean);
        step(&mut marked);
    }
    assert!(evolved_values(&clean) == evolved_values(&marked), "an inadmissible excluded state reached an evolved value");
}

#[test]
fn a_checkpoint_withholds_unrecovered_excluded_primitives_and_restarts_identically() {
    let dir = std::env::temp_dir().join(format!("covered_core_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("c.h5");
    let mut original = build();
    step(&mut original);
    let c = cell_at_depth(&original, 1, 8);
    make_inadmissible(&original, 1, c);
    original.levels[1].kernels.c2p(&original.levels[1].state);
    let den = *original.levels[1].state.fields.cons.den.view().at(c);
    let raw_pre = *original.levels[1].state.fields.prim.pre.as_ref().expect("pre").view().at(c);
    assert!(raw_pre < 0.0 || !raw_pre.is_finite(), "the cell recovered an admissible pressure {raw_pre}; the premise is gone");
    let states: Vec<&Sim> = original.levels.iter().map(|l| &l.state).collect();
    symbi_sim::checkpoint::write_hierarchy_checkpoint(&states, path.to_str().expect("utf-8"), &Default::default())
        .expect("checkpoint");
    // the live state keeps its recovered values after the write
    assert_eq!(*original.levels[1].state.fields.prim.pre.as_ref().expect("pre").view().at(c), raw_pre);
    // the file names the box whose NaN primitives are withheld on purpose
    {
        let file = hdf5_metno::File::open(&path).expect("open checkpoint");
        let read = |name: &str| -> Vec<i64> { file.dataset(name).expect(name).read_raw().expect("read") };
        let core = excluded(&original, 1).expect("excluded core");
        assert_eq!(read("level_1/partition_0/unread_core/start"), vec![core.spaces[0].lo as i64, core.spaces[1].lo as i64]);
        assert_eq!(read("level_1/partition_0/unread_core/fin"), vec![core.spaces[0].hi as i64, core.spaces[1].hi as i64]);
        assert!(file.dataset("level_2/partition_0/unread_core/start").is_err(), "the finest level records an excluded core");
        // the root excludes cells too, and every one of them recovers: it records no box
        assert!(excluded(&original, 0).is_some(), "the root excludes no cells; the premise is gone");
        assert!(file.dataset("level_0/partition_0/unread_core/start").is_err(), "a level whose excluded cells all recover records a box");
    }

    let mut restarted = build();
    restarted.restore_from_checkpoint(path.to_str().expect("utf-8")).expect("restart");
    assert!(
        restarted.levels[1].state.fields.prim.pre.as_ref().expect("pre").view().at(c).is_nan(),
        "the checkpoint published a recovered pressure for an excluded cell without an admissible recovery"
    );
    assert_eq!(*restarted.levels[1].state.fields.cons.den.view().at(c), den, "the conserved state was not kept");
    // the restored hierarchy recovers its primitives from the loaded conserved state, as a resumed
    // run does before its first step; the excluded cell recovers its raw, inadmissible values
    restarted.prime();
    for _ in 0..STEPS {
        step(&mut original);
        step(&mut restarted);
    }
    assert!(evolved_values(&original) == evolved_values(&restarted), "the restart departs from the uninterrupted run");
    let _ = std::fs::remove_dir_all(&dir);
}


#[test]
fn a_rigid_body_keeps_every_covered_cell_in_the_decisions() {
    let mut h = build();
    assert!(excluded(&h, 0).is_some(), "the sink-only hierarchy excludes no core; the premise is gone");
    let wall = Body::rigid_sphere(1, Tensor::new([0.5, 0.5]), Tensor::zeros(), 1.0e12, 0.05, 1.0, true)
        .with_surface(SurfaceSpec::Porous { porosity: 0.0, k_eta_n: 50.0, k_eta_t: 50.0 });
    h = h.with_bodies(BodyCollection::new().add(wall));
    h.establish_decision_exclusion();
    for level in 0..3 {
        assert!(excluded(&h, level).is_none(), "level {level} excludes cells beside a rigid body");
    }
}
