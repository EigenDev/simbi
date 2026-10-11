// =============================================================================
// refinement_seed_boundary.rs
//
// the seed of a refined level beside a domain face. the seed prolongs the
// parent's conserved state over the fine interior, and the prolongation stencil
// of the fine cells beside a face the patch reaches reads the parent's ghost
// cells there; the parent's ghosts hold the state the face's boundary condition
// prescribes when the seed reads them. gates, on the single hierarchy alone:
// - beside an outflow, a reflecting, a driven and a zero-gradient face, the fine
//   cells seed as on a grid extended past the face whose cells hold the state the
//   face's own fill prescribes;
// - a nonuniform state periodic in the cell index seeds the fine cells beside
//   the seam as it seeds the same cells shifted into the interior;
// - a magnetized state seeds the gas, cell field and staggered faces beside an
//   outflow face as from cells holding the face's fill, and beside a periodic
//   seam as in the interior;
// - a patch clear of every face seeds as it did before the boundary fill.
//
// run: cargo test -p symbi --test refinement_seed_boundary
// =============================================================================
use symbi::regimes::substrate_kernels::GradientBc;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi::sim::substrate_seam::KernelSet;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::NEWTONIAN_SPEC;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_source_compile::SourceConfig;
use symbi_source_compile::expr_bridge::build_boundary_dag;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;
type MSim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type MKern = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 2>;
type MHier = Hierarchy<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, MKern>;

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 32;
const DX: f64 = 1.0 / N as f64;
const RHO: f64 = 1.3;
const PRE: f64 = 0.9;

/// the face kinds of the x axis; y keeps outflow walls
#[derive(Clone, Copy, Debug)]
enum Face {
    Outflow,
    Reflecting,
    Periodic,
    Driven,
    ZeroGradient,
}

/// a constant dirichlet prescription [rho, v1, v2, p]
fn const_dag(vel: [f64; 2]) -> String {
    format!(
        r#"{{
        "kind": "dirichlet", "dim": 2, "outputs": [0, 1, 2, 3], "params": [],
        "nodes": [ {{"op":"CONSTANT","value":{RHO}}}, {{"op":"CONSTANT","value":{}}},
                   {{"op":"CONSTANT","value":{}}}, {{"op":"CONSTANT","value":{PRE}}} ]
    }}"#,
        vel[0], vel[1]
    )
}

fn kset_for(face: Face, vel: [f64; 2]) -> impl Fn(&Sim) -> Kern {
    move |sim: &Sim| {
        let k = Kern::new(GAMMA, CFL, &sim.geom.allocated);
        match face {
            Face::Driven => {
                let json = const_dag(vel);
                let cfg = SourceConfig::from_json(&json).expect("parse boundary config");
                let built = build_boundary_dag(&cfg, &NEWTONIAN_SPEC).expect("lower boundary dag");
                k.with_driven_boundary(built, cfg.params.clone()).0
            }
            Face::ZeroGradient => k.with_gradient_boundary(GradientBc::Neumann(vec![0.0; 4])).0,
            _ => k,
        }
    }
}

fn x_faces(face: Face) -> [BoundaryType; 2] {
    match face {
        Face::Outflow => [BoundaryType::Outflow; 2],
        Face::Reflecting => [BoundaryType::Reflect; 2],
        Face::Periodic => [BoundaryType::Periodic; 2],
        Face::Driven => [BoundaryType::Driven(0); 2],
        Face::ZeroGradient => [BoundaryType::Neumann(0); 2],
    }
}

fn build(face: Face, region: RefinementRegion<2>, ic: impl Fn([f64; 2]) -> Prim<f64, 2> + Copy, vel: [f64; 2]) -> Hier {
    let bnd = Boundaries::per_axis([x_faces(face), [BoundaryType::Outflow; 2]]);
    let root = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .spacing([DX; 2])
        .origin([0.0; 2])
        .boundaries(bnd)
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("root")
        .set_initial(ic)
        .build();
    let make = kset_for(face, vel);
    let k = make(&root);
    Hier::with_refinement(root, k, &[region], ProlongOrder::Ppm, make).expect("hierarchy")
}

/// the fine level's conserved state over its interior as (den, mom0, mom1, nrg) per cell
fn fine_cons(h: &Hier) -> Vec<([isize; 2], [f64; 4])> {
    let s = &h.levels[1].state;
    let c = &s.fields.cons;
    s.geom
        .interior
        .iter()
        .map(|x| ([x[0], x[1]], [*c.den.view().at(x), *c.mom[0].view().at(x), *c.mom[1].view().at(x), *c.nrg_field().unwrap().view().at(x)]))
        .collect()
}

/// a nonuniform state along x and y, smooth across the x faces
fn varied(x: [f64; 2]) -> Prim<f64, 2> {
    let (u, v) = (x[0], x[1]);
    Prim::adiabatic(
        Density(1.0 + 0.3 * u + 0.2 * (3.0 * v).sin() + 0.4 * u * u),
        Tensor::new([0.3 - 0.5 * u, 0.1 * (2.0 * v).cos()]),
        Pressure(1.0 + 0.6 * u + 0.1 * v),
    )
}

const EXT: usize = 2;

/// the primitive ghost columns the face's boundary condition prescribes beside the root's x-lo
/// face: the root's own fill, after recovering its primitives from the seeded conserved state
fn x_lo_ghost_prims(h: &Hier) -> std::collections::HashMap<[isize; 2], Prim<f64, 2>> {
    let l = &h.levels[0];
    l.kernels.c2p(&l.state);
    l.kernels.ghost_fill(&l.state);
    let s = &l.state;
    let (lo, ng) = (s.geom.interior.spaces[0].lo, s.geom.ng as isize);
    let pre = s.fields.prim.pre_field().unwrap();
    let mut out = std::collections::HashMap::new();
    for j in s.geom.interior.spaces[1].lo..s.geom.interior.spaces[1].hi {
        for i in lo - ng..lo {
            let c = [i, j];
            let prim = Prim::adiabatic(
                Density(*s.fields.prim.rho.view().at(c)),
                Tensor::new([*s.fields.prim.vel[0].view().at(c), *s.fields.prim.vel[1].view().at(c)]),
                Pressure(*pre.view().at(c)),
            );
            out.insert([i - lo, j - s.geom.interior.spaces[1].lo], prim);
        }
    }
    out
}

#[test]
fn the_fine_cells_beside_a_physical_face_seed_as_from_the_cells_its_boundary_prescribes() {
    // the same problem on a grid extended by two cells past the x-lo face, those cells holding
    // the primitive state the face's own fill prescribes: there the seed reads real cells where
    // the bounded grid's seed reads ghosts, so both seed the patch's fine cells identically
    let region = RefinementRegion { x_lo: [0.0, 0.375], x_hi: [0.25, 0.625] };
    for face in [Face::Outflow, Face::Reflecting, Face::Driven, Face::ZeroGradient] {
        let vel = [0.4, -0.2];
        let bounded = build(face, region, varied, vel);
        let probe = build(face, region, varied, vel);
        let ghosts = x_lo_ghost_prims(&probe);
        let cells = [N + EXT, N];
        let root = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
            .cells(cells)
            .spacing([DX; 2])
            .origin([-(EXT as f64) * DX, 0.0])
            .boundaries(Boundaries::uniform(BoundaryType::Outflow))
            .cfl(CFL)
            .timestepping(Timestepping::Rk2)
            .allocate()
            .expect("extended root")
            .set_initial_indexed(|idx, x| {
                let i = idx[0] - EXT as isize;
                if i < 0 { ghosts[&[i, idx[1]]] } else { varied(x) }
            })
            .build();
        let make = |s: &Sim| Kern::new(GAMMA, CFL, &s.geom.allocated);
        let k = make(&root);
        let extended = Hier::with_refinement(root, k, &[region], ProlongOrder::Ppm, make).expect("extended hierarchy");
        // premise: the face's ghosts differ from what a zero ghost would give
        assert!(ghosts.values().all(|p| p.rho() > 0.5), "{face:?}: the face's fill left an empty ghost");
        bounded.seed_fine_from_coarse().expect("seed");
        extended.seed_fine_from_coarse().expect("seed");
        let (a, b) = (fine_cons(&bounded), fine_cons(&extended));
        let lo_a = bounded.levels[1].state.geom.interior.spaces[0].lo;
        let lo_b = extended.levels[1].state.geom.interior.spaces[0].lo;
        let index: std::collections::HashMap<[isize; 2], [f64; 4]> = b.into_iter().collect();
        let mut compared = 0;
        for (c, v) in &a {
            if c[0] - lo_a >= 4 {
                continue;
            }
            let w = index[&[lo_b + (c[0] - lo_a), c[1]]];
            assert_eq!(*v, w, "{face:?}: fine cell {c:?} beside the face seeds {v:?}; seeded from the prescribed cells it is {w:?}");
            compared += 1;
        }
        assert!(compared > 0, "{face:?}: no fine cell beside the face was compared");
    }
}

/// a state periodic in the root cell index along x, shifted by `shift` cells
fn indexed(shift: usize) -> impl Fn([f64; 2]) -> Prim<f64, 2> + Copy {
    move |x: [f64; 2]| {
        let i = ((x[0] / DX).floor() as usize + shift) % N;
        let phase = 2.0 * std::f64::consts::PI * i as f64 / N as f64;
        Prim::adiabatic(Density(1.0 + 0.3 * phase.sin()), Tensor::new([0.5 + 0.1 * phase.cos(), 0.1]), Pressure(1.0 + 0.2 * (2.0 * phase).cos()))
    }
}

#[test]
fn a_periodic_patch_seeds_the_seam_as_the_interior() {
    // the patch spans the periodic x axis; the state shifted by 16 cells brings the seam's
    // neighborhood to the middle of the axis, where the seed reads interior cells alone
    let region = RefinementRegion { x_lo: [0.0, 0.375], x_hi: [1.0, 0.625] };
    let vel = [0.0, 0.0];
    let seam = build(Face::Periodic, region, indexed(0), vel);
    let middle = build(Face::Periodic, region, indexed(16), vel);
    seam.seed_fine_from_coarse().expect("seed");
    middle.seed_fine_from_coarse().expect("seed");
    let (a, b) = (fine_cons(&seam), fine_cons(&middle));
    let lo = seam.levels[1].state.geom.interior.spaces[0].lo;
    // fine cells within four of the seam on either side, against the same cells 32 fine cells over
    let mut compared = 0;
    for (c, v) in &a {
        let i = c[0] - lo;
        if i >= 4 && i < 2 * N as isize - 4 {
            continue;
        }
        let shifted = [lo + (i + N as isize).rem_euclid(2 * N as isize), c[1]];
        let w = b.iter().find(|(d, _)| *d == shifted).map(|(_, w)| *w).expect("the shifted cell");
        assert_eq!(*v, w, "fine cell {c:?} beside the seam seeds {v:?}; the same cell in the interior seeds {w:?}");
        compared += 1;
    }
    assert!(compared > 0, "no fine cell beside the seam was compared");
}

/// a stable digest of the fine level's seeded conserved state
fn digest(h: &Hier) -> u64 {
    let mut acc = 0xcbf2_9ce4_8422_2325u64;
    for (_, v) in fine_cons(h) {
        for x in v {
            acc ^= x.to_bits();
            acc = acc.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    acc
}

#[test]
fn a_patch_clear_of_every_face_seeds_as_before() {
    // a nonuniform state under an interior patch: the seed reads interior parent cells alone, and
    // the digest was recorded before the parent's ghosts were filled for the seed
    let region = RefinementRegion { x_lo: [0.25, 0.375], x_hi: [0.75, 0.625] };
    let h = build(Face::Outflow, region, indexed(0), [0.0, 0.0]);
    h.seed_fine_from_coarse().expect("seed");
    let got = digest(&h);
    eprintln!("interior seed digest {got:#018x}");
    assert_eq!(got, INTERIOR_DIGEST, "the interior patch's seed changed");
}

const INTERIOR_DIGEST: u64 = 0x6061_5ef5_cfad_f145;

/// a magnetized state periodic in the root cell index along x, shifted by `shift` cells: the gas
/// varies along x, the x-face field B_x is uniform and the y-face field B_y varies along x, so the
/// field is divergence-free on the faces
fn m_indexed(shift: usize) -> (impl Fn([f64; 2]) -> MhdPrim<f64, 3> + Copy, impl Fn(usize, [f64; 2]) -> f64 + Copy) {
    let phase = move |x: f64| {
        let i = ((x / DX).floor() as usize + shift) % N;
        2.0 * std::f64::consts::PI * i as f64 / N as f64
    };
    let by = move |x: f64| 0.2 + 0.1 * phase(x).sin();
    let cells = move |x: [f64; 2]| {
        let p = phase(x[0]);
        MhdPrim::new(
            Prim::adiabatic(Density(1.0 + 0.3 * p.sin()), Tensor::new([0.4 + 0.1 * p.cos(), 0.1, 0.0]), Pressure(1.0 + 0.2 * (2.0 * p).cos())),
            Tensor::new([0.3, by(x[0]), 0.05]),
        )
    };
    let faces = move |d: usize, x: [f64; 2]| if d == 0 { 0.3 } else { by(x[0]) };
    (cells, faces)
}

fn m_build(bnd: Boundaries<2>, region: RefinementRegion<2>, shift: usize) -> MHier {
    let (cells, faces) = m_indexed(shift);
    let root = MSim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .spacing([DX; 2])
        .origin([0.0; 2])
        .boundaries(bnd)
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("root")
        .set_initial(cells)
        .seed_faces(faces)
        .build();
    let make = |s: &MSim| MKern::new(GAMMA, CFL, 1.0, &s.geom.allocated);
    let k = make(&root);
    MHier::with_refinement(root, k, &[region], ProlongOrder::Ppm, make).expect("hierarchy")
}

/// the fine level's conserved gas, cell field and staggered faces over its interior, keyed by
/// (kind, axis, index)
fn m_fine(h: &MHier) -> std::collections::HashMap<(u8, usize, [isize; 2]), f64> {
    let s = &h.levels[1].state;
    let mhd = s.fields.mhd.as_ref().unwrap();
    let mut out = std::collections::HashMap::new();
    for x in s.geom.interior.iter() {
        let c = [x[0], x[1]];
        out.insert((0, 0, c), *s.fields.cons.den.view().at(x));
        out.insert((1, 0, c), *s.fields.cons.nrg_field().unwrap().view().at(x));
        for k in 0..3 {
            out.insert((2, k, c), *mhd.bcell[k].view().at(x));
        }
        for d in 0..2 {
            out.insert((3, d, c), *mhd.bface[d].view().at(x));
        }
    }
    out
}

#[test]
fn a_periodic_magnetized_patch_seeds_the_seam_as_the_interior() {
    let bnd = Boundaries::per_axis([[BoundaryType::Periodic; 2], [BoundaryType::Outflow; 2]]);
    let region = RefinementRegion { x_lo: [0.0, 0.375], x_hi: [1.0, 0.625] };
    let seam = m_build(bnd.clone(), region, 0);
    let middle = m_build(bnd, region, 16);
    seam.seed_fine_from_coarse().expect("seed");
    middle.seed_fine_from_coarse().expect("seed");
    let (a, b) = (m_fine(&seam), m_fine(&middle));
    let lo = seam.levels[1].state.geom.interior.spaces[0].lo;
    let mut compared = 0;
    for ((kind, axis, c), v) in &a {
        let i = c[0] - lo;
        if i >= 4 && i < 2 * N as isize - 4 {
            continue;
        }
        let shifted = [lo + (i + N as isize).rem_euclid(2 * N as isize), c[1]];
        let w = b[&(*kind, *axis, shifted)];
        assert_eq!(*v, w, "kind {kind} axis {axis}: fine cell {c:?} beside the seam seeds {v}; the same cell in the interior seeds {w}");
        compared += 1;
    }
    assert!(compared > 0, "no fine cell beside the seam was compared");
}

#[test]
fn the_magnetized_fine_cells_beside_an_outflow_face_seed_as_from_the_cells_its_boundary_prescribes() {
    // the magnetized analog of the bounded/extended comparison: the extension's cells and faces hold
    // the primitives, cell field and staggered faces the outflow face's own fill writes
    let region = RefinementRegion { x_lo: [0.0, 0.375], x_hi: [0.25, 0.625] };
    let bnd = Boundaries::uniform(BoundaryType::Outflow);
    let bounded = m_build(bnd.clone(), region, 3);
    let probe = m_build(bnd, region, 3);
    let l = &probe.levels[0];
    l.kernels.c2p(&l.state);
    l.kernels.ghost_fill(&l.state);
    let s0 = &l.state;
    let (ilo, jlo) = (s0.geom.interior.spaces[0].lo, s0.geom.interior.spaces[1].lo);
    let mhd = s0.fields.mhd.as_ref().unwrap();
    let pre = s0.fields.prim.pre_field().unwrap();
    let cell_ghost = |i: isize, j: isize| -> MhdPrim<f64, 3> {
        let c = [ilo + i, jlo + j];
        MhdPrim::new(
            Prim::adiabatic(
                Density(*s0.fields.prim.rho.view().at(c)),
                Tensor::new(std::array::from_fn(|k| *s0.fields.prim.vel[k].view().at(c))),
                Pressure(*pre.view().at(c)),
            ),
            Tensor::new(std::array::from_fn(|k| *mhd.bcell[k].view().at(c))),
        )
    };
    let face_ghost = |d: usize, i: isize, j: isize| *mhd.bface[d].view().at([ilo + i, jlo + j]);
    let (cells, faces) = m_indexed(3);
    let ext = EXT as isize;
    let root = MSim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N + EXT, N])
        .spacing([DX; 2])
        .origin([-(EXT as f64) * DX, 0.0])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("extended root")
        .set_initial_indexed(|idx, x| if idx[0] < ext { cell_ghost(idx[0] - ext, idx[1]) } else { cells(x) })
        .seed_faces(|d, x| {
            // the face's index on the bounded grid: x faces sit on lattice lines, y faces at centers
            let i = if d == 0 { (x[0] / DX).round() as isize } else { (x[0] / DX).floor() as isize };
            let j = if d == 1 { (x[1] / DX).round() as isize } else { (x[1] / DX).floor() as isize };
            // the x faces carry no ghost layer along x; the y faces' ghost columns hold the fill
            let held = i >= -(s0.geom.ng as isize) && j >= -(s0.geom.ng as isize) && j < (N + s0.geom.ng) as isize;
            if i < 0 && d == 1 && held { face_ghost(d, i, j) } else { faces(d, x) }
        })
        .build();
    let make = |s: &MSim| MKern::new(GAMMA, CFL, 1.0, &s.geom.allocated);
    let k = make(&root);
    let extended = MHier::with_refinement(root, k, &[region], ProlongOrder::Ppm, make).expect("extended hierarchy");
    bounded.seed_fine_from_coarse().expect("seed");
    extended.seed_fine_from_coarse().expect("seed");
    let (a, b) = (m_fine(&bounded), m_fine(&extended));
    let lo_a = bounded.levels[1].state.geom.interior.spaces[0].lo;
    let lo_b = extended.levels[1].state.geom.interior.spaces[0].lo;
    let mut compared = 0;
    for ((kind, axis, c), v) in &a {
        if c[0] - lo_a >= 4 {
            continue;
        }
        let w = b[&(*kind, *axis, [lo_b + (c[0] - lo_a), c[1]])];
        assert_eq!(*v, w, "kind {kind} axis {axis}: fine cell {c:?} beside the face seeds {v}; seeded from the prescribed cells it is {w}");
        compared += 1;
    }
    assert!(compared > 0, "no fine cell beside the face was compared");
}
