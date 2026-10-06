// =============================================================================
// decomp_mhd_fofc_cut.rs
//
// the first-order flux correction of a magnetized flow with the troubled cells
// against a tile cut on one side. both tiles at a cut hold the interface face
// and each writes its copy from its own curl of the edge EMFs, so the copies
// agree only when the two tiles correct the cut's edges identically: the clean
// tile learns of its neighbor's trouble and redoes its side, both tiles splice
// the induction flux of the column beyond the cut by that column's own flags,
// and both reconstruct the redo's fluxes on the stage-entry face field.
//
// the fixture is the periodic magnetized relativistic shock of the translation
// gate with the collision column one cell right of the cut, which puts the
// last-stage troubled columns c - 5 ..= c - 2 in the left tile's cut column
// and leaves the right tile's first column clean. the four-tile arm adds a cut
// along y, so every troubled row also meets a tile corner.
//
// per arm the premises are asserted (the correction fired, a step ended with
// flags against the cut on the left alone, no freeze), then: every conserved
// and magnetic field equals the one-tile run bit for bit; the two copies of
// every shared interface face, periodic seams included, are bit-for-bit equal;
// div B is at round-off in every tile from its own faces; and div B is at
// round-off on the grid assembled with the upper tile's copy of each shared
// face.
//
// run: cargo test -p symbi --test decomp_mhd_fofc_cut
// =============================================================================

use std::ops::ControlFlow;
use symbi::prelude::SimSubstrate;
use symbi::regimes::fofc::{fofc_reset_stats, fofc_stats};
use symbi::sim::decomp::{LocalCopy, Partition, Schedule, Topology, evolve_scheduled, unflatten};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_sim::substrate_seam::Solver;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 4.0 / 3.0;
const B0Y: f64 = 0.2;
const NX: usize = 256;
const NY: usize = 8;
const DX: f64 = 1.0 / NX as f64;
const DY: f64 = 1.0 / NY as f64;
const CUT: usize = NX / 2;

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

type Sim = SimStateGeneric<Rmhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Store = FieldStore<2, 3, HostMemory, f64>;

#[derive(Clone, Copy)]
struct Fixture {
    speed: f64,
    t_final: f64,
    /// the collision column, chosen per arm so a last-stage troubled column is `CUT - 1`
    collide: usize,
}

// a tile placed as production places it: the whole grid's origin and the index offset of its
// first cell
fn tile(ext: [(usize, usize); 2], bnd: Boundaries<2>, ts: Timestepping, fixture: Fixture) -> Sim {
    Sim::build(Rmhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([ext[0].1, ext[1].1])
        .origin([0.0; 2])
        .spacing([DX, DY])
        .index_offset([ext[0].0 as isize, ext[1].0 as isize])
        .boundaries(bnd)
        .cfl(0.1)
        .timestepping(ts)
        .allocate()
        .expect("sim")
        .set_initial(move |[x, _y]| {
            let column = (x / DX).floor() as usize;
            let behind = (column + NX - fixture.collide) % NX >= NX / 2;
            let vx = if behind { fixture.speed } else { -(fixture.speed - 0.01) };
            MhdPrim::new(
                Prim::adiabatic(Density(1.0), Tensor::new([vx, 0.0, 0.0]), Pressure(1e-4)),
                Tensor::new([0.0, B0Y, 0.0]),
            )
        })
        .seed_faces_uniform([0.0, B0Y])
        .build()
}

struct Arm {
    /// (name, values on the global grid, x index first); faces over their NX by NY distinct ones
    fields: Vec<(String, Vec<f64>)>,
    fallbacks: u64,
    freezes: u64,
    left_only_steps: u64,
    /// shared interface faces whose two copies differ, over every cut and seam
    unequal_shared_faces: usize,
    shared_faces: usize,
    max_tile_div: f64,
    max_assembled_div: f64,
}

fn run(cuts: [Vec<usize>; 2], ct: CtMethod, ts: Timestepping, fixture: Fixture) -> Arm {
    fofc_reset_stats();
    let partition = Partition::explicit([NX, NY], cuts).expect("cuts inside the grid");
    let counts = partition.counts();
    let n = partition.n_tiles();
    let extents: Vec<[(usize, usize); 2]> =
        (0..n).map(|f| partition.tile_extents(unflatten(f, counts))).collect();
    let mut sims: Vec<Sim> = extents
        .iter()
        .map(|ext| {
            let bnd = Boundaries(std::array::from_fn(|a| {
                let kind = if counts[a] > 1 {
                    BoundaryType::CoarseFine
                } else {
                    BoundaryType::Periodic
                };
                [kind, kind]
            }));
            tile(*ext, bnd, ts, fixture)
        })
        .collect();
    let kernels: Vec<_> = sims
        .iter()
        .map(|sim| {
            let mut k = sim
                .substrate()
                .with_solver(Solver::Hlld)
                .expect("solver")
                .ct_method(ct);
            k.theta = 1.5;
            k
        })
        .collect();
    let mut left_only_steps = 0u64;
    {
        let mut stores: Vec<&mut Store> = sims.iter_mut().map(|s| &mut **s).collect();
        let kernel_refs: Vec<_> = kernels.iter().collect();
        let schedule = Schedule::derive(counts, stores[0].geom.ng, &Topology::wrapping([true; 2]));
        let devices = vec![0i32; n];
        evolve_scheduled(
            &mut stores,
            &kernel_refs,
            &schedule,
            &devices,
            ts,
            0.0,
            fixture.t_final,
            1,
            &LocalCopy,
            |_, _, stores| {
                // flags against the x cut at CUT, in any row: the left tiles end at CUT
                let column_flagged = |global: usize| {
                    stores.iter().zip(&extents).any(|(s, ext)| {
                        if global < ext[0].0 || global >= ext[0].0 + ext[0].1 {
                            return false;
                        }
                        let i = s.geom.interior.spaces[0].lo + (global - ext[0].0) as isize;
                        let view = s.workspace.fofc_flag.view();
                        s.geom.interior.iter().any(|c| c[0] == i && *view.at(c) != 0.0)
                    })
                };
                if n > 1 && column_flagged(CUT - 1) && !column_flagged(CUT) {
                    left_only_steps += 1;
                }
                ControlFlow::Continue(())
            },
        );
    }
    let (fallbacks, freezes) = fofc_stats();

    // assemble: cells from their owner; a shared face from the upper tile, the closing face of
    // a periodic axis being the first tile's opening face
    let names = ["den", "nrg", "mom0", "mom1", "mom2", "bcell0", "bcell1", "bcell2"];
    let mut cells = vec![vec![f64::NAN; NX * NY]; names.len()];
    let mut faces = [vec![f64::NAN; NX * NY], vec![f64::NAN; NX * NY]];
    let (mut unequal_shared_faces, mut shared_faces) = (0usize, 0usize);
    let mut max_tile_div = 0.0f64;
    for (sim, ext) in sims.iter().zip(&extents) {
        let lo = [sim.geom.interior.spaces[0].lo, sim.geom.interior.spaces[1].lo];
        let mhd = sim.fields.mhd.as_ref().expect("mhd fields");
        let cons = &sim.fields.cons;
        let sources = [
            &cons.den,
            cons.nrg_field().expect("energy"),
            &cons.mom[0],
            &cons.mom[1],
            &cons.mom[2],
            &mhd.bcell.b[0],
            &mhd.bcell.b[1],
            &mhd.bcell.b[2],
        ];
        let (bx, by) = (mhd.bface.b[0].view(), mhd.bface.b[1].view());
        for c in sim.geom.interior.iter() {
            let g = [ext[0].0 + (c[0] - lo[0]) as usize, ext[1].0 + (c[1] - lo[1]) as usize];
            for (k, f) in sources.iter().enumerate() {
                cells[k][g[0] * NY + g[1]] = *f.view().at(c);
            }
            faces[0][g[0] * NY + g[1]] = *bx.at(c);
            faces[1][g[0] * NY + g[1]] = *by.at(c);
            let div = (*bx.at([c[0] + 1, c[1]]) - *bx.at(c)) / DX
                + (*by.at([c[0], c[1] + 1]) - *by.at(c)) / DY;
            max_tile_div = max_tile_div.max(div.abs());
        }
    }
    // the lower tile's closing copy of each shared face against the assembled (upper) copy
    for (sim, ext) in sims.iter().zip(&extents) {
        let lo = [sim.geom.interior.spaces[0].lo, sim.geom.interior.spaces[1].lo];
        let mhd = sim.fields.mhd.as_ref().expect("mhd fields");
        for d in 0..2 {
            if counts[d] == 1 {
                continue;
            }
            let other = 1 - d;
            let view = mhd.bface.b[d].view();
            for t in 0..ext[other].1 {
                let mut c = [0isize; 2];
                c[d] = lo[d] + ext[d].1 as isize;
                c[other] = lo[other] + t as isize;
                let mut g = [0usize; 2];
                g[d] = (ext[d].0 + ext[d].1) % [NX, NY][d];
                g[other] = ext[other].0 + t;
                shared_faces += 1;
                if view.at(c).to_bits() != faces[d][g[0] * NY + g[1]].to_bits() {
                    unequal_shared_faces += 1;
                }
            }
        }
    }
    let mut max_assembled_div = 0.0f64;
    for i in 0..NX {
        for j in 0..NY {
            let at = |d: usize, i: usize, j: usize| faces[d][(i % NX) * NY + (j % NY)];
            let div = (at(0, i + 1, j) - at(0, i, j)) / DX + (at(1, i, j + 1) - at(1, i, j)) / DY;
            max_assembled_div = max_assembled_div.max(div.abs());
        }
    }
    let mut fields: Vec<(String, Vec<f64>)> =
        names.iter().map(|s| s.to_string()).zip(cells).collect();
    let [f0, f1] = faces;
    fields.push(("bface0".into(), f0));
    fields.push(("bface1".into(), f1));
    Arm {
        fields,
        fallbacks,
        freezes,
        left_only_steps,
        unequal_shared_faces,
        shared_faces,
        max_tile_div,
        max_assembled_div,
    }
}

fn assert_cut_arms_match_one_tile(
    ct: CtMethod,
    ts: Timestepping,
    fixture: Fixture,
    arms: &[(&str, [Vec<usize>; 2])],
    label: &str,
) {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let one = run([Vec::new(), Vec::new()], ct, ts, fixture);
    assert!(one.fallbacks > 0, "{label}: the correction never fired");
    assert_eq!(one.freezes, 0, "{label}: the one-tile run froze a cell");
    assert!(one.max_tile_div < 1e-11, "{label}: one-tile div B {:e}", one.max_tile_div);
    for (layout, cuts) in arms {
        let tag = format!("{label}, {layout}");
        let cut = run(cuts.clone(), ct, ts, fixture);
        assert!(
            cut.left_only_steps > 0,
            "{tag}: no step ended with the left of the cut flagged and the right clean"
        );
        assert_eq!(cut.freezes, 0, "{tag}: the freeze tier acted");
        assert!(cut.shared_faces > 0);
        assert_eq!(
            cut.unequal_shared_faces, 0,
            "{tag}: {} of {} shared interface faces differ between their two tiles",
            cut.unequal_shared_faces, cut.shared_faces
        );
        for ((name, want), (_, got)) in one.fields.iter().zip(&cut.fields) {
            let differing = want.iter().zip(got).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(differing, 0, "{tag}: {name} differs from the one-tile run in {differing} places");
        }
        assert_eq!(cut.fallbacks, one.fallbacks, "{tag}: the arms booked different corrections");
        assert!(cut.max_tile_div < 1e-11, "{tag}: tile-local div B {:e}", cut.max_tile_div);
        assert!(
            cut.max_assembled_div < 1e-11,
            "{tag}: assembled div B {:e}",
            cut.max_assembled_div
        );
    }
}

fn two_tiles() -> (&'static str, [Vec<usize>; 2]) {
    ("two tiles", [vec![CUT], Vec::new()])
}

fn four_tiles() -> (&'static str, [Vec<usize>; 2]) {
    ("four tiles", [vec![CUT], vec![NY / 2]])
}

// last-stage troubled columns relative to the collision column c (from the translation gate):
//  forward Euler, Contact:  c - 128 ..= c - 118 and c + 117 ..= c + 127
//  UCT, either scheme:      c - 5 ..= c - 2 and c + 1 ..= c + 5
//  Contact under RK2, fast: c - 127 alone
// each arm picks c so that column CUT - 1 is troubled and column CUT is clean.
const MODERATE: f64 = 0.95;

#[test]
fn uct_euler() {
    let fixture = Fixture { speed: MODERATE, t_final: 0.06, collide: CUT + 1 };
    assert_cut_arms_match_one_tile(CtMethod::Uct, Timestepping::Euler, fixture, &[two_tiles()], "uct, euler");
}

#[test]
fn uct_rk2() {
    let fixture = Fixture { speed: MODERATE, t_final: 0.06, collide: CUT + 1 };
    assert_cut_arms_match_one_tile(CtMethod::Uct, Timestepping::Rk2, fixture, &[two_tiles()], "uct, rk2");
}

// the corner arms run on their own, so one layout's failure leaves the others' evidence.

#[test]
fn uct_euler_corner() {
    let fixture = Fixture { speed: MODERATE, t_final: 0.06, collide: CUT + 1 };
    assert_cut_arms_match_one_tile(CtMethod::Uct, Timestepping::Euler, fixture, &[four_tiles()], "uct, euler");
}

#[test]
fn uct_rk2_corner() {
    let fixture = Fixture { speed: MODERATE, t_final: 0.06, collide: CUT + 1 };
    assert_cut_arms_match_one_tile(CtMethod::Uct, Timestepping::Rk2, fixture, &[four_tiles()], "uct, rk2");
}

#[test]
fn contact_euler_corner() {
    let fixture = Fixture { speed: MODERATE, t_final: 0.06, collide: (CUT + 117) % NX };
    assert_cut_arms_match_one_tile(CtMethod::Contact, Timestepping::Euler, fixture, &[four_tiles()], "contact, euler");
}

#[test]
fn contact_rk2_corner() {
    let fixture = Fixture { speed: 0.98, t_final: 0.02, collide: (CUT - 1 + 127) % NX };
    assert_cut_arms_match_one_tile(CtMethod::Contact, Timestepping::Rk2, fixture, &[four_tiles()], "contact, rk2");
}

#[test]
fn contact_euler() {
    let fixture = Fixture { speed: MODERATE, t_final: 0.06, collide: (CUT + 117) % NX };
    assert_cut_arms_match_one_tile(CtMethod::Contact, Timestepping::Euler, fixture, &[two_tiles()], "contact, euler");
}

#[test]
fn contact_rk2() {
    let fixture = Fixture { speed: 0.98, t_final: 0.02, collide: (CUT - 1 + 127) % NX };
    assert_cut_arms_match_one_tile(CtMethod::Contact, Timestepping::Rk2, fixture, &[two_tiles()], "contact, rk2");
}
