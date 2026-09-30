// =============================================================================
// pole_axis_emf.rs
//
// the shared radial axis edge of a 3D spherical grid spanning both poles. at a
// pole every azimuthal copy of a radial edge is one physical segment of the
// axis; summed over a pole cell's faces, the constrained-transport curl leaves
// dr (E_r(phi_k+1) - E_r(phi_k)) from the two copies bounding the cell's
// zero-area pole face, so the cell's discrete magnetic flux is preserved
// exactly when every copy carries one EMF. the stage reconciles each ring to
// the sector-weighted quadrature of its copies between the complete EMF and
// the curl. the gates, on newtonian MHD with non-axisymmetric seeded data:
// - under contact and under uct, the unnormalized flux out of every cell,
//   sum_d (W B)^+ - (W B)^- with W the CT face weight, is unchanged by a run
//   to roundoff, the pole rows included and checked apart;
// - the same stage curled from the unreconciled EMF changes a pole cell's
//   flux at the scale of the field, so the gate discriminates;
// - a troubled pole cell corrected by the first-order redo leaves the flux
//   unchanged, the correction shown to fire;
// - the ring weights are each copy's share of the ring on uneven widths.
//
// run: cargo test -p symbi --test pole_axis_emf
// =============================================================================
use std::f64::consts::PI;
use std::ops::ControlFlow;
use symbi::prelude::{KernelSet, SimSubstrate};
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Spherical;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::axis_ring::ring_value;
use symbi_sim::substrate_seam::Solver;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<NewtonianMhd, 3, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;

const GAMMA: f64 = 5.0 / 3.0;
const R_LO: f64 = 1.0;
const R_HI: f64 = 2.0;
const N: [usize; 3] = [6, 8, 12];
const T_FINAL: f64 = 0.2;

/// the spherical orthonormal components of a uniform Cartesian vector at (theta, phi)
fn spherical_of(v: [f64; 3], th: f64, ph: f64) -> [f64; 3] {
    let (st, ct, sp, cp) = (th.sin(), th.cos(), ph.sin(), ph.cos());
    [
        v[0] * st * cp + v[1] * st * sp + v[2] * ct,
        v[0] * ct * cp + v[1] * ct * sp - v[2] * st,
        -v[0] * sp + v[1] * cp,
    ]
}

/// a deterministic perturbation in [-1, 1) from a cell position
fn seeded(x: [f64; 3]) -> f64 {
    let h = (x[0] * 12.9898 + x[1] * 78.233 + x[2] * 37.719).sin() * 43758.5453;
    2.0 * (h - h.floor()) - 1.0
}

fn make(ct: CtMethod) -> (Sim, impl KernelSet<3, 3, HostMemory, f64>) {
    let sim = Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Spherical)
        .cells(N)
        .spacing([(R_HI - R_LO) / N[0] as f64, PI / N[1] as f64, 2.0 * PI / N[2] as f64])
        .origin([R_LO, 0.0, 0.0])
        .boundaries(Boundaries::per_axis([
            [BoundaryType::Outflow, BoundaryType::Outflow],
            [BoundaryType::Axis, BoundaryType::Axis],
            [BoundaryType::Periodic, BoundaryType::Periodic],
        ]))
        .cfl(0.3)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("spherical 3D construction")
        .set_initial(|x| {
            let [r, th, ph] = x;
            let v = spherical_of([0.03, 0.02, 0.01], th, ph);
            let kick = 0.01 * seeded(x);
            MhdPrim::new(
                Prim::adiabatic(
                    Density(1.0 + 0.1 * r + 0.05 * th.cos() + 0.02 * th.sin() * ph.cos() + kick),
                    Tensor::new([v[0] + kick, v[1] - kick, v[2] + 0.5 * kick]),
                    Pressure(1.0 + 0.03 * th.sin() * ph.sin()),
                ),
                Tensor::new(spherical_of([0.1, 0.07, 0.05], th, ph)),
            )
        })
        .seed_faces(|axis, [_r, th, ph]| spherical_of([0.1, 0.07, 0.05], th, ph)[axis])
        .build();
    let mut k = sim.substrate().with_solver(Solver::Hlld).expect("solver").ct_method(ct);
    k.theta = 1.5;
    (sim, k)
}

/// the unnormalized discrete magnetic flux out of every interior cell with the CT face weights
fn cell_flux(sim: &Sim) -> Vec<([isize; 3], f64)> {
    let bg = sim.geom.block_geometry(sim.physics.metric);
    let m = sim.fields.mhd.as_ref().expect("mhd fields");
    sim.geom
        .interior
        .iter()
        .map(|c| {
            let mut flux = 0.0;
            for d in 0..3 {
                let mut hi = c;
                hi[d] += 1;
                flux += bg.ct_face_weight(hi, d) * *m.bface[d].view().at(hi)
                    - bg.ct_face_weight(c, d) * *m.bface[d].view().at(c);
            }
            (c, flux)
        })
        .collect()
}

/// the scale of the weighted face flux, the size a flux change is roundoff against
fn flux_scale(sim: &Sim) -> f64 {
    let bg = sim.geom.block_geometry(sim.physics.metric);
    let m = sim.fields.mhd.as_ref().expect("mhd fields");
    sim.geom
        .interior
        .iter()
        .flat_map(|c| (0..3).map(move |d| (c, d)))
        .map(|(c, d)| (bg.ct_face_weight(c, d) * *m.bface[d].view().at(c)).abs())
        .fold(0.0, f64::max)
}

/// the largest change of the cell flux between two measurements over the pole row `j` alone, or
/// over every cell off the poles
fn largest_change(a: &[([isize; 3], f64)], b: &[([isize; 3], f64)], rows: impl Fn(isize) -> bool) -> f64 {
    a.iter().zip(b).filter(|((c, _), _)| rows(c[1])).map(|((_, x), (_, y))| (y - x).abs()).fold(0.0, f64::max)
}

/// run to T_FINAL on the production single-grid driver; returns the troubled-cell total
fn run(sim: Sim, k: impl KernelSet<3, 3, HostMemory, f64>) -> (Sim, u64) {
    let mut hier = Hierarchy::single(sim, k);
    let d = hier.evolve_with_callback(T_FINAL, 1, |_| ControlFlow::Continue(())).expect("the run");
    let troubled = d.guards.troubled_cells.total;
    (hier.levels.remove(0).state, troubled)
}

// the tolerance on a preserved flux: each cell sum takes six weighted faces of size up to the
// flux scale, rounded once per stage over a handful of steps, measured at ~1e-16 of the scale
// (2.4e-17 against 6.2e-4 before the redo arm); 1e-12 of the scale is roundoff by four orders
// and three orders below the smallest leak the unreconciled ring produces (see the mutation)
const ROUNDOFF: f64 = 1.0e-12;

fn assert_every_cell_keeps_its_flux(ct: CtMethod) {
    let (sim, k) = make(ct);
    k.c2p(&sim);
    k.ghost_fill(&sim);
    let before = cell_flux(&sim);
    let scale = flux_scale(&sim);
    let last = (N[1] - 1) as isize;
    let (sim, _) = run(sim, k);
    assert!(sim.iteration >= 8, "{ct:?}: the run took {} steps", sim.iteration);
    let after = cell_flux(&sim);
    for (label, rows) in [
        ("the theta = 0 pole row", Box::new(|j: isize| j == 0) as Box<dyn Fn(isize) -> bool>),
        ("the theta = pi pole row", Box::new(move |j: isize| j == last)),
        ("every row off the poles", Box::new(move |j: isize| j != 0 && j != last)),
    ] {
        let change = largest_change(&before, &after, rows);
        assert!(change <= ROUNDOFF * scale, "{ct:?}: {label} changed its flux by {change:e} against a scale of {scale:e}");
    }
}

#[test]
fn contact_keeps_every_cells_flux_at_both_poles() {
    assert_every_cell_keeps_its_flux(CtMethod::Contact);
}

#[test]
fn uct_keeps_every_cells_flux_at_both_poles() {
    assert_every_cell_keeps_its_flux(CtMethod::Uct);
}

#[test]
fn a_stage_curled_from_the_unreconciled_ring_changes_the_pole_cells_flux() {
    for ct in [CtMethod::Contact, CtMethod::Uct] {
        let run_stage = |reconcile: bool| {
            let (sim, k) = make(ct);
            k.c2p(&sim);
            k.ghost_fill(&sim);
            let before = cell_flux(&sim);
            let scale = flux_scale(&sim);
            let dt = k.cfl(&sim);
            sim.mark_stage_input_captured();
            k.snapshot_stage(&sim);
            k.wave_speeds(&sim);
            for d in 0..3 {
                k.flux(&sim, d);
            }
            k.efield(&sim);
            k.godunov_stage(&sim, dt, 0.0, 1.0);
            k.stage_emf(&sim, dt, 0);
            if reconcile {
                symbi_sim::axis_ring::reconcile_whole_rings(&sim);
            }
            k.stage_curl(&sim, dt, 0);
            let last = (N[1] - 1) as isize;
            (largest_change(&before, &cell_flux(&sim), |j| j == 0 || j == last), scale)
        };
        let (leak, scale) = run_stage(false);
        let (kept, _) = run_stage(true);
        assert!(leak > 1.0e-9 * scale, "{ct:?}: the unreconciled ring left the pole flux within {leak:e}; the gate is vacuous");
        assert!(kept <= ROUNDOFF * scale, "{ct:?}: the reconciled stage changed a pole cell's flux by {kept:e}");
    }
}

#[test]
fn a_corrected_pole_cell_keeps_its_flux() {
    for ct in [CtMethod::Contact, CtMethod::Uct] {
        let (sim, k) = make(ct);
        k.c2p(&sim);
        k.ghost_fill(&sim);
        let before = cell_flux(&sim);
        let scale = flux_scale(&sim);
        // a pole cell flagged troubled in the corrector of step 1: the first-order redo
        // recomputes the stage EMF around it
        let int = &sim.geom.interior;
        let cell = [int.spaces[0].lo + 2, int.spaces[1].lo, int.spaces[2].lo + 3];
        *sim.workspace.troubled_once.lock().unwrap() = Some((1, 1, cell, false));
        let (sim, troubled) = run(sim, k);
        assert!(troubled > 0, "{ct:?}: the flagged pole cell was never corrected; the gate is vacuous");
        let change = largest_change(&before, &cell_flux(&sim), |_| true);
        assert!(change <= ROUNDOFF * scale, "{ct:?}: the corrected run changed a cell's flux by {change:e}");
    }
}

#[test]
fn each_copy_weighs_its_share_of_the_ring() {
    // widths 1, 2, 3 around the ring: the copy at the start of the cell of width w_k serves the
    // cells of widths w_{k-1} and w_k, so its weight is their mean, (3+1)/2, (1+2)/2, (2+3)/2
    let copies = [(1.0, 1.0), (2.0, 0.0), (3.0, 0.0)];
    assert_eq!(ring_value(&copies), 2.0 / 6.0);
    let copies = [(1.0, 0.0), (2.0, 1.0), (3.0, 0.0)];
    assert_eq!(ring_value(&copies), 1.5 / 6.0);
    // a ring whose copies agree reconciles to their common value
    let copies = [(0.5, -0.25), (1.5, -0.25), (0.75, -0.25), (0.25, -0.25)];
    assert_eq!(ring_value(&copies), -0.25);
}
