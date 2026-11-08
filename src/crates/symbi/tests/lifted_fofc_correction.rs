// =============================================================================
// lifted_fofc_correction.rs
//
// the first-order flux correction on the lifted (r, theta) and (R, z) planes with the azimuthal
// velocity and a point mass, through the production single-hierarchy driver. root step 2's first
// stage has one cell raised troubled, so the correction splices the first-order flux of every
// conserved component, the azimuthal momentum's included, onto that cell's faces, recomputes the
// update and its face gravity work from the spliced fluxes, and recovers again:
// - the correction fires: the accepted guard totals count the troubled cell, the receipts book the
//   same count with the cell where it lies, and the corrected run's state differs from the clean
//   run's
// - every accepted stage is admissible (the run completes; a failed recovery stops it)
// - mass, angular momentum about the axis and E + rho Phi close against the production receipts:
//   the change of each volume total plus its wall outflow, at roundoff of the totals
//
// run: cargo test -p symbi --test lifted_fofc_correction
// =============================================================================

use std::f64::consts::PI;
use std::ops::ControlFlow;

use symbi::prelude::SimSubstrate;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{Cylindrical, Geometry, Spherical};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, SofteningKind};
use symbi_xpu::{CpuSpace, HostMemory};

const NI: usize = 16;
const NJ: usize = 8;
const TROUBLED: [isize; 2] = [7, 3];
const STEPS: u64 = 4;
const U: f64 = 1.1102230246251565e-16;

macro_rules! lifted {
    ($chart:expr, $origin:expr, $extent:expr, $troubled:expr) => {{
        type S<C> = SimStateGeneric<Newtonian, 2, 3, C, IdealGas<f64>, CpuSpace, HostMemory>;
        let walls = [[BoundaryType::Outflow, BoundaryType::Outflow], [BoundaryType::Axis, BoundaryType::Axis]];
        let sim = S::build(Newtonian, IdealGas { gamma: 5.0 / 3.0 }, $chart)
            .cells([NI, NJ])
            .origin($origin)
            .spacing([$extent[0] / NI as f64, $extent[1] / NJ as f64])
            .boundaries(Boundaries::per_axis(walls))
            .cfl(0.2)
            .timestepping(Timestepping::Rk2)
            .allocate()
            .expect("the plane")
            .set_initial(|x: [f64; 2]| Prim::adiabatic(Density(1.0 + 0.2 * x[1].sin()), Tensor::new([-0.05, 0.02, 0.4 / x[0]]), Pressure(0.5)))
            .build();
        if $troubled {
            // root step 2 enters at iteration 1
            *sim.workspace.troubled_once.lock().unwrap() = Some((1, 0, TROUBLED, false));
        }
        let kernels = sim.substrate();
        let mass = BodyCollection::new()
            .add(Body::gravitational(0, Tensor::zeros(), Tensor::zeros(), 1.0, 0.0, 0.1).with_softening_kind(SofteningKind::Compact));
        let mut hier = Hierarchy::single(sim, kernels).with_bodies(mass);
        hier.prime();
        hier
    }};
}

/// (mass, angular momentum, E + rho Phi) over the interior, at the volume-weighted centroids
fn totals(store: &FieldStore<2, 3, HostMemory>) -> [f64; 3] {
    let g = &store.geom;
    let (c, d) = (&store.fields.cons, &g.interior);
    let face = |ax: usize, i: isize| g.x_lo[ax] + i as f64 * g.dx[ax];
    let mut out = [0.0; 3];
    for x in d.iter() {
        let (a0, b0, a1, b1) = (face(0, x[0]), face(0, x[0] + 1), face(1, x[1]), face(1, x[1] + 1));
        let (volume, lever, radius) = match g.coords {
            Geometry::Spherical => {
                let r_c = (b0.powi(4) - a0.powi(4)) / 4.0 / ((b0.powi(3) - a0.powi(3)) / 3.0);
                let t_c = ((b1.sin() - b1 * b1.cos()) - (a1.sin() - a1 * a1.cos())) / (a1.cos() - b1.cos());
                (2.0 * PI * (b0.powi(3) - a0.powi(3)) / 3.0 * (a1.cos() - b1.cos()), r_c * t_c.sin(), r_c)
            }
            _ => {
                let r_c = (b0.powi(3) - a0.powi(3)) / 3.0 / ((b0 * b0 - a0 * a0) / 2.0);
                let z_c = 0.5 * (a1 + b1);
                (PI * (b0 * b0 - a0 * a0) * (b1 - a1), r_c, (r_c * r_c + z_c * z_c).sqrt())
            }
        };
        let rho = *c.den.view().at(x);
        // the azimuthal momentum is the coordinate the two grid axes leave out: phi after (r, theta),
        // the middle one of (R, phi, z)
        let slot = 3 - g.axes[0] - g.axes[1];
        out[0] += rho * volume;
        out[1] += *c.mom[slot].view().at(x) * lever * volume;
        out[2] += (*c.nrg_field().expect("nrg").view().at(x) - rho / radius) * volume;
    }
    out
}

fn bits(store: &FieldStore<2, 3, HostMemory>) -> Vec<u64> {
    let (c, d) = (&store.fields.cons, &store.geom.interior);
    let mut out = Vec::new();
    for f in [&c.den, c.nrg_field().expect("nrg"), &c.mom[0], &c.mom[1], &c.mom[2]] {
        let v = f.view();
        out.extend(d.iter().map(|x| v.at(x).to_bits()));
    }
    out
}

macro_rules! gate {
    ($label:expr, $corrected:expr, $clean:expr) => {{
        let mut corrected = $corrected;
        let start = totals(&corrected.levels[0].state);
        let diagnostics = corrected
            .evolve_with_callback(10.0, 1, |h| if h.levels[0].state.iteration >= STEPS { ControlFlow::Break(()) } else { ControlFlow::Continue(()) })
            .expect("the corrected run");
        let (_, receipts) = symbi_sim::conservation_ledger::ledger_report();
        assert!(corrected.crash.is_none(), "{}: the corrected run stopped: {:?}", $label, corrected.crash);
        assert_eq!(corrected.levels[0].state.iteration, STEPS, "{}: the corrected run stopped short", $label);
        assert!(diagnostics.guards.troubled_cells.total > 0, "{}: the correction never counted a troubled cell", $label);
        // the receipts book every corrected cell where it lies: the guard totals' count, the forced
        // cell among them
        assert_eq!(receipts.corrected_cell_stages, diagnostics.guards.troubled_cells.total, "{}: the receipts count other corrections than the guards", $label);
        assert!(receipts.max_corrected_in_stage >= 1, "{}: no stage booked a corrected cell", $label);
        let map = symbi_sim::conservation_ledger::corrections_report();
        assert!(map.iter().any(|(c, _)| c[..2] == TROUBLED), "{}: the forced cell is missing from the correction map {map:?}", $label);
        assert_eq!(map.iter().map(|(_, n)| n).sum::<u64>(), receipts.corrected_cell_stages, "{}: the map and the count disagree", $label);
        let mut clean = $clean;
        clean
            .evolve_with_callback(10.0, 1, |h| if h.levels[0].state.iteration >= STEPS { ControlFlow::Break(()) } else { ControlFlow::Continue(()) })
            .expect("the clean run");
        assert!(bits(&corrected.levels[0].state) != bits(&clean.levels[0].state), "{}: the correction left the state unchanged; it never acted", $label);
        // the receipts: every wall's outflow and the change of each total close; the bound is the
        // roundoff of summing the totals, a few ulps per cell times the cells, per step
        let end = totals(&corrected.levels[0].state);
        let out = receipts.total_outflow();
        let n = (NI * NJ) as f64 * STEPS as f64;
        for (k, (name, outflow)) in [("mass", out.mass), ("angular momentum", out.angular_momentum), ("E + rho Phi", out.energy)].into_iter().enumerate() {
            let residual = end[k] - start[k] + outflow;
            let bound = 4.0 * n * U * start[k].abs();
            assert!(residual.abs() <= bound, "{}: {name} residual {residual:e} beyond {bound:e}", $label);
        }
        assert!(out.mass != 0.0, "{}: no mass crossed a wall; the closure checks the interior alone", $label);
    }};
}

#[test]
fn the_correction_fires_and_conserves_on_the_spherical_plane() {
    gate!(
        "sphere",
        lifted!(Spherical, [1.0, 0.0], [1.0, PI], true),
        lifted!(Spherical, [1.0, 0.0], [1.0, PI], false)
    );
}

#[test]
fn the_correction_fires_and_conserves_on_the_cylindrical_plane() {
    gate!(
        "cylinder",
        lifted!(Cylindrical, [0.5, -1.0], [1.0, 2.0], true),
        lifted!(Cylindrical, [0.5, -1.0], [1.0, 2.0], false)
    );
}
