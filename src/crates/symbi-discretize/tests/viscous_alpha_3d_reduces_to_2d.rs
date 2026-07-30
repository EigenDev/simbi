// =============================================================================
// viscous_alpha_3d_reduces_to_2d.rs
//
// the 3D cartesian adiabatic alpha-disk viscous operator against its 2D twin.
//
// the alpha law sets nu from the LOCAL sound speed and the keplerian frequency at the CYLINDRICAL
// radius about the rotation axis, so nothing in nu depends on height. give the operator a state
// that is uniform in z with no vertical velocity, and the vertical derivatives of the stress
// tensor vanish identically: the in-plane momentum and the viscous heating must then reproduce the
// 2D kernel exactly, and the vertical momentum must not move at all.
//
// that is the sharpest available check on a new 3D kernel, because it pins the whole in-plane
// stress — shear, divergence, and heating — against an operator that already has gates, rather
// than against a hand-computed number. a wrong z-term, a mis-indexed stencil slice, or a nu that
// accidentally reads the spherical radius all break it.
//
// run: cargo test -p symbi-discretize --test viscous_alpha_3d_reduces_to_2d
// =============================================================================

mod harness;
use harness::KernelRun;

use symbi_discretize::{viscous_adiabatic_alpha_gv, viscous_adiabatic_alpha_gv_3d};

const N: usize = 6;
const NZ: usize = 5;

// a sheared, stratified in-plane state. the shear drives the stress, the density and pressure
// gradients make the LOCAL cs^2 (hence nu) vary from cell to cell, and none of it depends on z.
fn rho_at(i: usize, j: usize) -> f64 {
    1.0 + 0.3 * (i as f64) - 0.2 * (j as f64)
}
fn pre_at(i: usize, j: usize) -> f64 {
    2.0 + 0.15 * (i as f64) * (j as f64)
}
fn vx_at(i: usize, j: usize) -> f64 {
    0.4 * (j as f64) - 0.1 * (i as f64)
}
fn vy_at(i: usize, j: usize) -> f64 {
    -0.25 * (i as f64) + 0.05 * (j as f64)
}

fn scalars() -> Vec<(&'static str, f64)> {
    vec![
        ("dt", 1e-3),
        ("alpha", 0.05),
        ("gamma", 5.0 / 3.0),
        ("body_0_mass", 3.0),
        ("body_0_pos_0", -1.5),
        ("body_0_pos_1", 0.75),
        ("dx_0", 0.1),
        ("dx_1", 0.12),
        ("dx_2", 0.09),
        ("x_lo_0", -0.4),
        ("x_lo_1", -0.3),
        ("x_lo_2", -0.2),
    ]
}

#[test]
fn a_z_invariant_state_reproduces_the_two_dimensional_operator() {
    let two = KernelRun::new(viscous_adiabatic_alpha_gv())
        .grid([N, N])
        .compute_window([1, 1], [N - 2, N - 2])
        .field_with("prim_rho", |c| rho_at(c[0], c[1]))
        .field_with("prim_pre", |c| pre_at(c[0], c[1]))
        .field_with("prim_v0", |c| vx_at(c[0], c[1]))
        .field_with("prim_v1", |c| vy_at(c[0], c[1]))
        .field_with("mom0", |c| rho_at(c[0], c[1]) * vx_at(c[0], c[1]))
        .field_with("mom1", |c| rho_at(c[0], c[1]) * vy_at(c[0], c[1]))
        .fields(&[("nrg", 7.0)])
        .scalars(&scalars())
        .run();

    let three = KernelRun::new(viscous_adiabatic_alpha_gv_3d())
        .grid([N, N, NZ])
        .compute_window([1, 1, 1], [N - 2, N - 2, NZ - 2])
        .field_with("prim_rho", |c| rho_at(c[0], c[1]))
        .field_with("prim_pre", |c| pre_at(c[0], c[1]))
        .field_with("prim_v0", |c| vx_at(c[0], c[1]))
        .field_with("prim_v1", |c| vy_at(c[0], c[1]))
        .fields(&[("prim_v2", 0.0)])
        .field_with("mom0", |c| rho_at(c[0], c[1]) * vx_at(c[0], c[1]))
        .field_with("mom1", |c| rho_at(c[0], c[1]) * vy_at(c[0], c[1]))
        .fields(&[("mom2", 0.0), ("nrg", 7.0)])
        .scalars(&scalars())
        .run();

    // interior cells only: the stencil reaches one cell each way, and the 3D run needs a z
    // neighbour on both sides for the vertical terms it must find to be zero.
    let mut checked = 0usize;
    for i in 1..N - 1 {
        for j in 1..N - 1 {
            for k in 1..NZ - 1 {
                for w in ["mom_out_0", "mom_out_1", "nrg_out"] {
                    let a = two.get([i, j], w);
                    let b = three.get([i, j, k], w);
                    assert!(
                        (a - b).abs() < 1e-12,
                        "{w} at ({i},{j},{k}): 3d {b} != 2d {a}"
                    );
                }
                // no vertical shear and no vertical velocity: the z momentum is untouched.
                let mz = three.get([i, j, k], "mom_out_2");
                assert!(
                    mz.abs() < 1e-14,
                    "the 3d operator moved z momentum ({mz}) on a z-invariant state at \
                     ({i},{j},{k})"
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "no interior cells were compared");

    // the premise: the operator actually did something. a zero update would satisfy the
    // equivalence above trivially.
    let moved = (1..N - 1)
        .flat_map(|i| (1..N - 1).map(move |j| (i, j)))
        .map(|(i, j)| {
            let m0 = rho_at(i, j) * vx_at(i, j);
            (two.get([i, j], "mom_out_0") - m0).abs()
        })
        .fold(0.0_f64, f64::max);
    assert!(
        moved > 1e-9,
        "the viscous update moved in-plane momentum by only {moved:e}; the comparison is vacuous"
    );
}

// the z-invariant reduction above cannot see the VERTICAL structure of nu: with no vertical
// gradients the z stress terms are zero, so the nu values at the k±1 stencil cells never reach the
// result. that is precisely where the adiabatic operator differs from its isothermal twin — the
// isothermal cs is one global constant, so its nu really is z-invariant, while the adiabatic cs^2
// = gamma p / rho is read per stencil cell and varies with height through the stratification.
//
// this gate gives the state a vertical structure and asserts the operator responds to it, which
// pins the k-slices of the nu stencil that the reduction leaves untested.
#[test]
fn vertical_stratification_reaches_the_viscosity() {
    // the same in-plane state, now with a vertical velocity shear so the z stress terms are live,
    // and a vertically stratified pressure so the local cs^2 varies with height.
    let run = |stratified: bool| {
        KernelRun::new(viscous_adiabatic_alpha_gv_3d())
            .grid([N, N, NZ])
            .compute_window([1, 1, 1], [N - 2, N - 2, NZ - 2])
            .field_with("prim_rho", |c| rho_at(c[0], c[1]))
            .field_with("prim_pre", move |c| {
                let vertical = if stratified { 0.4 * c[2] as f64 } else { 0.0 };
                pre_at(c[0], c[1]) + vertical
            })
            .field_with("prim_v0", |c| vx_at(c[0], c[1]))
            .field_with("prim_v1", |c| vy_at(c[0], c[1]))
            // a vertical shear of the in-plane flow: dv_x/dz is nonzero, so the xz stress lives
            // and the nu values at k±1 enter the update.
            .field_with("prim_v2", |c| 0.05 * c[2] as f64)
            .field_with("mom0", |c| rho_at(c[0], c[1]) * vx_at(c[0], c[1]))
            .field_with("mom1", |c| rho_at(c[0], c[1]) * vy_at(c[0], c[1]))
            .field_with("mom2", |c| rho_at(c[0], c[1]) * 0.05 * c[2] as f64)
            .fields(&[("nrg", 7.0)])
            .scalars(&scalars())
            .run()
    };
    let flat = run(false);
    let strat = run(true);

    let mut worst = 0.0_f64;
    for i in 1..N - 1 {
        for j in 1..N - 1 {
            for k in 1..NZ - 1 {
                for w in ["mom_out_0", "mom_out_1", "mom_out_2", "nrg_out"] {
                    worst = worst.max((flat.get([i, j, k], w) - strat.get([i, j, k], w)).abs());
                }
            }
        }
    }
    // a vertically varying pressure changes the LOCAL sound speed, hence nu, hence the update.
    // an operator that read one z-independent nu (the isothermal rule) would give zero here.
    assert!(
        worst > 1e-9,
        "vertical stratification changed the viscous update by only {worst:e}; the local cs^2 is \
         not reaching nu, so the operator is behaving like the isothermal twin"
    );
}
