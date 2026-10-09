// =============================================================================
// refinement_covered_core_reach.rs
//
// how far into the covered core of a refined level the step reads before the
// next restriction replaces it. a three-level 2.5D newtonian MHD hierarchy with
// a draining sink in the finest box takes two root steps; one copy has level
// 1's covered cells at depth >= d inside level 2's coverage scaled by
// 1 + 1e-6 (density and energy, admissible), and both take one more root step.
// perturbing depth 3 -- the first layer past the three-cell donor shell -- moves
// leaves on every level; perturbing depth 4 or deeper leaves every leaf, every
// covered cell, the next timestep, the sink's accreted mass and force history
// and the first-order correction counts bit for bit unchanged. the deeper core
// is stepped and recovered and read by nothing the step publishes; its recovery
// still feeds the guards' halt decisions.
//
// run: cargo test -p symbi --test refinement_covered_core_reach
// =============================================================================
use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, MagneticSpec, SurfaceSpec};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Kern = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const N: usize = 64;
const DX: f64 = 1.0 / N as f64;

fn regions() -> [RefinementRegion<2>; 2] {
    [
        RefinementRegion { x_lo: [0.0625, 0.0625], x_hi: [0.9375, 0.9375] },
        RefinementRegion { x_lo: [0.3125, 0.3125], x_hi: [0.6875, 0.6875] },
    ]
}

// face-normal components each independent of their own axis: discretely divergence-free
fn bx(y: f64) -> f64 {
    0.4 + 0.2 * (2.0 * std::f64::consts::PI * y + 0.3).sin()
}

fn by(x: f64) -> f64 {
    0.2 + 0.2 * (2.0 * std::f64::consts::PI * x + 0.7).cos()
}

fn bump(x: f64, y: f64) -> f64 {
    0.3 * (-(((x - 0.5) / 0.07).powi(2) + ((y - 0.48) / 0.09).powi(2))).exp()
}

fn kset(sim: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, 1.0, &sim.geom.allocated)
}

/// a root tile placed as production places it, seeding the faces of its own lattice
fn build_root(cells: [usize; 2], offset: [usize; 2], bnd: Boundaries<2>) -> Sim {
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin([0.0; 2])
        .index_offset([offset[0] as isize, offset[1] as isize])
        .boundaries(bnd)
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("root sim construction failed")
        .set_initial(|x| {
            let b = bump(x[0], x[1]);
            MhdPrim::new(
                Prim::adiabatic(Density(1.0 + b), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0 + b)),
                Tensor::new([bx(x[1]), by(x[0]), 0.1]),
            )
        })
        .seed_faces(|d, x| {
            let inside = (0..2).all(|a| {
                let (lo, hi) = (offset[a] as f64 * DX, (offset[a] + cells[a]) as f64 * DX);
                let eps = 1e-9 * DX;
                if a == d { x[a] >= lo - eps && x[a] <= hi + eps } else { x[a] > lo && x[a] < hi }
            });
            match (inside, d) {
                (false, _) => 0.0,
                (true, 0) => bx(x[1]),
                (true, _) => by(x[0]),
            }
        })
        .build()
}

fn sink() -> BodyCollection<f64, 2> {
    BodyCollection::new().add(
        Body::black_hole(0, Tensor::new([0.5, 0.5]), Tensor::zeros(), 0.01, 0.08, 0.05, 1.0, 1.0, 0.08)
            .with_surface(SurfaceSpec::Drain)
            .with_magnetic(MagneticSpec::None),
    )
}

fn build_single() -> Hier {
    let root = build_root([N, N], [0, 0], Boundaries::uniform(BoundaryType::Outflow));
    let k = kset(&root);
    let h = Hier::with_refinement(root, k, &regions(), ProlongOrder::Ppm, kset).expect("single hierarchy");
    h.seed_fine_from_coarse().expect("seed fine");
    let mut h = h.with_bodies(sink());
    h.prime();
    h
}


fn cons_bits(h: &Hier, level: usize, keep: &dyn Fn([isize; 2]) -> bool) -> Vec<([isize; 2], [u64; 5])> {
    let s = &h.levels[level].state;
    let c = &s.fields.cons;
    s.geom.interior.iter().filter(|x| keep([x[0], x[1]])).map(|x| {
        ([x[0], x[1]], [c.den.view().at(x).to_bits(), c.mom[0].view().at(x).to_bits(), c.mom[1].view().at(x).to_bits(), c.mom[2].view().at(x).to_bits(), c.nrg_field().unwrap().view().at(x).to_bits()])
    }).collect()
}

fn covers(h: &Hier, level: usize, c: [isize; 2]) -> bool {
    h.levels[level].coverage.as_ref().is_some_and(|cv| (0..2).all(|a| c[a] >= cv.spaces[a].lo && c[a] < cv.spaces[a].hi))
}

fn depth(h: &Hier, level: usize, c: [isize; 2]) -> isize {
    let cv = h.levels[level].coverage.as_ref().unwrap();
    (0..2).map(|a| (c[a] - cv.spaces[a].lo).min(cv.spaces[a].hi - 1 - c[a])).min().unwrap()
}

/// the composite outcome of one root step after perturbing level 1's covered cells at depth >= `d`
/// inside level 2's coverage: the cells of every level's leaves and covered region, the next
/// timestep, the sink's record and the correction counts that differ from the unperturbed run
fn reach_of_perturbation(d: isize) -> Vec<String> {
    let mut a = build_single();
    let mut b = build_single();
    for _ in 0..2 {
        let dt = a.root_cfl_dt();
        a.step_root_with_dt(dt);
        b.step_root_with_dt(dt);
    }
    {
        let s = &b.levels[1].state;
        let cells: Vec<_> = s.geom.interior.iter().filter(|c| covers(&b, 1, [c[0], c[1]]) && depth(&b, 1, [c[0], c[1]]) >= d).collect();
        assert!(!cells.is_empty(), "no covered cell at depth {d}; the probe perturbs nothing");
        for c in &cells {
            let den = *s.fields.cons.den.view().at(*c);
            *s.fields.cons.den.view_mut().at_mut(*c) = den * (1.0 + 1e-6);
            let e = *s.fields.cons.nrg_field().unwrap().view().at(*c);
            *s.fields.cons.nrg_field().unwrap().view_mut().at_mut(*c) = e * (1.0 + 1e-6);
        }
    }
    symbi::regimes::fofc::fofc_reset_stats();
    let dt = a.root_cfl_dt();
    a.step_root_with_dt(dt);
    let fa = symbi::regimes::fofc::fofc_stats();
    symbi::regimes::fofc::fofc_reset_stats();
    b.step_root_with_dt(dt);
    let fb = symbi::regimes::fofc::fofc_stats();
    let mut report = Vec::new();
    for level in 0..3 {
        let leaf = |h: &Hier, c: [isize; 2]| level == 2 || !covers(h, level, c);
        let (xa, xb) = (cons_bits(&a, level, &|c| leaf(&a, c)), cons_bits(&b, level, &|c| leaf(&b, c)));
        let diff = xa.iter().zip(&xb).filter(|(p, q)| p.1 != q.1).count();
        if diff > 0 {
            report.push(format!("level {level} leaves: {diff} cells"));
        }
        if level < 2 {
            let (ca, cb) = (cons_bits(&a, level, &|c| covers(&a, level, c)), cons_bits(&b, level, &|c| covers(&b, level, c)));
            let diff = ca.iter().zip(&cb).filter(|(p, q)| p.1 != q.1).count();
            if diff > 0 {
                report.push(format!("level {level} covered: {diff} cells"));
            }
        }
    }
    if a.root_cfl_dt().to_bits() != b.root_cfl_dt().to_bits() {
        report.push("the next timestep".into());
    }
    let record = |h: &Hier| {
        let i = h.levels.last().unwrap().state.immersed.as_ref().unwrap();
        let mut m = 0.0;
        i.bodies.visit_accretion(|x| {
            if let symbi_ib::BodyKind::BlackHole { total_accreted_mass, .. } = &x.kind {
                m += *total_accreted_mass;
            }
        });
        (i.history.force().iter().map(|f| f.to_bits()).collect::<Vec<_>>(), m.to_bits())
    };
    if record(&a) != record(&b) {
        report.push("the sink's record".into());
    }
    if fa != fb {
        report.push(format!("the correction counts {fa:?} against {fb:?}"));
    }
    report
}

#[test]
fn the_covered_core_past_the_donor_shell_and_one_layer_reaches_nothing_the_step_publishes() {
    // depth 3, the first core layer past the three-cell donor shell, feeds the composite solution
    let first_core_layer = reach_of_perturbation(3);
    assert!(
        first_core_layer.iter().any(|r| r.starts_with("level 2 leaves")),
        "perturbing the first core layer reached no finest cell ({first_core_layer:?}); the probe is blind"
    );
    // every deeper layer is discarded by the next restriction before any published value reads it
    for d in [4, 8] {
        let reached = reach_of_perturbation(d);
        assert!(reached.is_empty(), "perturbing the covered core at depth >= {d} reached {reached:?}");
    }
}
