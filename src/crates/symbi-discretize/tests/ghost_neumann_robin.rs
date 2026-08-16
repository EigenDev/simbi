// =============================================================================
// ghost_neumann_robin.rs
//
// numeric checks for the prescribed-gradient (neumann) and mixed (robin) lattice-map ghost
// fills. both reuse the outflow edge source coord (map_type = 3 -> arg = the boundary-adjacent
// interior cell) and apply the `symbi_hydro::boundary_term` lift per primitive variable, using
// the outward edge->ghost centroid separation `dist` computed in-kernel from the geometry.
//
// setup: a 1D grid, lo-side ghosts filled from the edge cell at index `EDGE`. the interior state
// is uniform, so the edge value is known; each ghost at index i sits `|i - EDGE| * dx` inward-
// distance out, and the fill must reproduce u_edge + q*dist (neumann) / the robin solve.
// =============================================================================

mod harness;
use harness::KernelRun;

use symbi_discretize::{Spacing, neumann_ghost_fill_gv, robin_ghost_fill_gv};

const NX: usize = 5; // allocated [0,5); interior starts at edge, lo ghosts are {0, 1}.
const EDGE: usize = 2;
const X0: f64 = 0.0;
const DX: f64 = 0.1;

// uniform interior state (so the edge value is exactly these).
const RHO: f64 = 2.0;
const VEL: f64 = 1.0;
const PRE: f64 = 3.0;

// outward edge->ghost distance for a lo-side ghost at index i: |centroid(i) - centroid(edge)|.
fn dist(i: usize) -> f64 {
    ((i as f64 + 0.5) - (EDGE as f64 + 0.5)).abs() * DX
}

fn close(a: f64, b: f64, ctx: &str) {
    assert!((a - b).abs() < 1e-12, "{ctx}: {a} != {b}");
}

#[test]
fn neumann_ghost_extrapolates_the_prescribed_gradient() {
    let (q_rho, q_vel, q_pre) = (0.5, -0.3, 1.0);
    // non-uniform density: only the edge cell holds rho; the ghosts hold a decoy. this pins that the
    // fill reads the edge (outflow source); a read of the ghost's own value would return the stale decoy.
    let out = KernelRun::new(neumann_ghost_fill_gv(1, 1, true, &[Spacing::Uniform]))
        .grid([NX])
        .compute_window([0], [2]) // the two lo ghosts
        .field_with("prim_rho", |c| if c[0] == EDGE { RHO } else { -7.0 })
        .fields(&[("prim_v0", VEL), ("prim_pre", PRE)])
        .scalars(&[
            ("map_type_0", 3.0), // outflow edge source
            ("arg_0", EDGE as f64),
            ("x_lo_0", X0),
            ("dx_0", DX),
            ("neu_q_rho", q_rho),
            ("neu_q_v0", q_vel),
            ("neu_q_pre", q_pre),
        ])
        .run();

    for i in 0..2 {
        let d = dist(i);
        close(
            out.get([i], "prim_rho"),
            RHO + q_rho * d,
            &format!("neumann rho ghost {i}"),
        );
        close(
            out.get([i], "prim_v0"),
            VEL + q_vel * d,
            &format!("neumann vel ghost {i}"),
        );
        close(
            out.get([i], "prim_pre"),
            PRE + q_pre * d,
            &format!("neumann pre ghost {i}"),
        );
    }
    // zero-gradient recovers the plain outflow copy (edge value at every ghost).
    let copy = KernelRun::new(neumann_ghost_fill_gv(1, 1, true, &[Spacing::Uniform]))
        .grid([NX])
        .compute_window([0], [2])
        .fields(&[("prim_rho", RHO), ("prim_v0", VEL), ("prim_pre", PRE)])
        .scalars(&[
            ("map_type_0", 3.0),
            ("arg_0", EDGE as f64),
            ("x_lo_0", X0),
            ("dx_0", DX),
            ("neu_q_rho", 0.0),
            ("neu_q_v0", 0.0),
            ("neu_q_pre", 0.0),
        ])
        .run();
    close(
        copy.get([0], "prim_rho"),
        RHO,
        "zero-gradient copy is the outflow fill",
    );
}

#[test]
fn robin_ghost_solves_the_mixed_relation() {
    // rho: a=2, b=0 -> dirichlet U_face = c/a. vel: a=0, b=1 -> neumann dU/dn = c/b. pre: general.
    let out = KernelRun::new(robin_ghost_fill_gv(1, 1, true, &[Spacing::Uniform]))
        .grid([NX])
        .compute_window([0], [2])
        .fields(&[("prim_rho", RHO), ("prim_v0", VEL), ("prim_pre", PRE)])
        .scalars(&[
            ("map_type_0", 3.0),
            ("arg_0", EDGE as f64),
            ("x_lo_0", X0),
            ("dx_0", DX),
            ("rob_a_rho", 2.0),
            ("rob_b_rho", 0.0),
            ("rob_c_rho", 6.0),
            ("rob_a_v0", 0.0),
            ("rob_b_v0", 1.0),
            ("rob_c_v0", 0.4),
            ("rob_a_pre", 1.3),
            ("rob_b_pre", 0.7),
            ("rob_c_pre", 2.1),
        ])
        .run();

    for i in 0..2 {
        let h = dist(i);
        // rho: dirichlet -> U_face = (u_edge + U_ghost)/2 = c/a = 3.
        let rho_g = out.get([i], "prim_rho");
        close(
            (RHO + rho_g) / 2.0,
            3.0,
            &format!("robin->dirichlet rho face {i}"),
        );
        // vel: neumann -> (U_ghost - u_edge)/h = c/b = 0.4.
        let vel_g = out.get([i], "prim_v0");
        close(
            (vel_g - VEL) / h,
            0.4,
            &format!("robin->neumann vel gradient {i}"),
        );
        // pre: general -> a*U_face + b*dU/dn = c.
        let pre_g = out.get([i], "prim_pre");
        let u_face = (PRE + pre_g) / 2.0;
        let dudn = (pre_g - PRE) / h;
        close(
            1.3 * u_face + 0.7 * dudn,
            2.1,
            &format!("robin pre relation {i}"),
        );
    }
}
