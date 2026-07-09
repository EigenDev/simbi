// =============================================================================
// gradient_boundary_cpu.rs
//
// a NEUMANN boundary (the registry-driven convenience short-circuit) prescribes its ghost prim
// state as `U_ghost = u_edge + q*dist` per primitive variable, from the boundary-adjacent interior
// ("edge") cell. the standard ghost-fill SKIPS the Neumann face (Neumann -> BcType::Skip); the
// gradient-boundary pass then fills it from the edge + the registered per-variable gradients.
//
// x_lo is Neumann with per-variable gradients q = [rho:0.5, vx:-0.3, vy:0.2, pre:1.0]; the interior
// is uniform, so the edge value is known and each x_lo ghost at distance `dist` inward must hold
// exactly u_edge + q*dist. checked on the face band (transverse interior; corners excluded, like the
// driven-boundary pass).
// =============================================================================

use symbi::prelude::*;
use symbi::regimes::substrate_kernels::GradientBc;
use symbi::sim::evolve::KernelSet;

type Sim = SimCpu<Newtonian, 2, Cartesian, IdealGas<f64>>;

const RHO: f64 = 2.0;
const VX: f64 = 1.0;
const PRE: f64 = 3.0;
// per-variable outward gradients in prim order [rho, vel_0, vel_1, pre].
const Q: [f64; 4] = [0.5, -0.3, 0.2, 1.0];

#[test]
fn neumann_boundary_extrapolates_from_the_edge_cell() {
    // x_lo Neumann, x_hi outflow, y periodic.
    let boundaries = Boundaries::<2>::per_axis([
        [BoundaryType::Neumann(0), BoundaryType::Outflow],
        [BoundaryType::Periodic, BoundaryType::Periodic],
    ]);
    let sim = Sim::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([8, 8])
        .bounds([0.0, 0.0], [1.0, 1.0])
        .boundaries(boundaries)
        .finish()
        .unwrap();
    // populate the PRIM state directly (the ghost fill reads prim; in the evolve loop c2p fills it
    // from cons each step, but this isolated test skips the step). uniform interior, so the edge
    // value is exactly (RHO, VX, 0, PRE).
    let pre_f = sim.fields.prim.pre_field().unwrap();
    for c in sim.geom.interior.iter() {
        sim.fields.prim.rho.view_mut().set(c, RHO);
        sim.fields.prim.vel[0].view_mut().set(c, VX);
        sim.fields.prim.vel[1].view_mut().set(c, 0.0);
        pre_f.view_mut().set(c, PRE);
    }

    let (sub, id) = sim
        .substrate()
        .with_gradient_boundary(GradientBc::Neumann(Q.to_vec()));
    assert_eq!(id, 0, "first registration is id 0 (matches Neumann(0))");

    sub.ghost_fill(&sim);

    // the edge (first interior cell) centroid along x: bounds start 0, dx = 1/8 -> 0.0625.
    let x_edge = 0.5 * (1.0 / 8.0);
    let mut checked = 0usize;
    for c in sim.geom.allocated.iter() {
        let x = sim.geom.cell_coord(c);
        let is_xlo_ghost = x[0] < 0.0;
        let y_interior = x[1] > 0.0 && x[1] < 1.0;
        if is_xlo_ghost && y_interior {
            let dist = (x[0] - x_edge).abs();
            let rho = *sim.fields.prim.rho.view().at(c);
            let v0 = *sim.fields.prim.vel[0].view().at(c);
            let v1 = *sim.fields.prim.vel[1].view().at(c);
            let p = *sim.fields.prim.pre_field().unwrap().view().at(c);
            assert!((rho - (RHO + Q[0] * dist)).abs() < 1e-12, "x_lo ghost rho at {x:?}: {rho}");
            assert!((v0 - (VX + Q[1] * dist)).abs() < 1e-12, "x_lo ghost vel_0 at {x:?}: {v0}");
            assert!((v1 - (Q[2] * dist)).abs() < 1e-12, "x_lo ghost vel_1 at {x:?}: {v1}");
            assert!((p - (PRE + Q[3] * dist)).abs() < 1e-12, "x_lo ghost pre at {x:?}: {p}");
            checked += 1;
        }
    }
    assert!(checked > 0, "no x_lo ghost-band cells found to check");
}

#[test]
fn robin_boundary_solves_the_mixed_relation() {
    // per-variable (a,b,c): rho Dirichlet (a=1,b=0), vel_0 Neumann (a=0,b=1), vel_1 + pre general.
    let abc: Vec<[f64; 3]> = vec![
        [1.0, 0.0, 2.0],  // rho:   a*U_face = c -> U_face = 2
        [0.0, 1.0, 0.5],  // vel_0: b*dU/dn = c -> dU/dn = 0.5
        [1.3, 0.7, 0.9],  // vel_1: general
        [2.0, 0.5, 7.0],  // pre:   general
    ];
    let boundaries = Boundaries::<2>::per_axis([
        [BoundaryType::Robin(0), BoundaryType::Outflow],
        [BoundaryType::Periodic, BoundaryType::Periodic],
    ]);
    let sim = Sim::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([8, 8])
        .bounds([0.0, 0.0], [1.0, 1.0])
        .boundaries(boundaries)
        .finish()
        .unwrap();
    let pre_f = sim.fields.prim.pre_field().unwrap();
    for c in sim.geom.interior.iter() {
        sim.fields.prim.rho.view_mut().set(c, RHO);
        sim.fields.prim.vel[0].view_mut().set(c, VX);
        sim.fields.prim.vel[1].view_mut().set(c, 0.0);
        pre_f.view_mut().set(c, PRE);
    }

    let (sub, _id) = sim.substrate().with_gradient_boundary(GradientBc::Robin(abc.clone()));
    sub.ghost_fill(&sim);

    let x_edge = 0.5 * (1.0 / 8.0);
    let edge = [RHO, VX, 0.0, PRE];
    let mut checked = 0usize;
    for c in sim.geom.allocated.iter() {
        let x = sim.geom.cell_coord(c);
        if x[0] < 0.0 && x[1] > 0.0 && x[1] < 1.0 {
            let h = (x[0] - x_edge).abs();
            let g = [
                *sim.fields.prim.rho.view().at(c),
                *sim.fields.prim.vel[0].view().at(c),
                *sim.fields.prim.vel[1].view().at(c),
                *pre_f.view().at(c),
            ];
            // each variable's ghost satisfies its prescribed relation a*U_face + b*dU/dn = c.
            for v in 0..4 {
                let u_face = (edge[v] + g[v]) / 2.0;
                let dudn = (g[v] - edge[v]) / h;
                let lhs = abc[v][0] * u_face + abc[v][1] * dudn;
                assert!((lhs - abc[v][2]).abs() < 1e-10, "robin var {v} at {x:?}: {lhs} != {}", abc[v][2]);
            }
            checked += 1;
        }
    }
    assert!(checked > 0, "no x_lo ghost-band cells found to check");
}
