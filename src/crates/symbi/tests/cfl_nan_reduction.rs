// =============================================================================
// cfl_nan_reduction.rs
//
// no silent floors: a single NaN cell in the
// wave-speed scratch must propagate through the CFL max-reduction so the
// downstream dt guard fires. f64::max(finite, NaN) == finite silently drops the
// NaN, which would let garbage advance past check_dt_or_panic to checkpoint.
//
// this pins the host fold path (runs without cuda). the device kernel + its
// host-side partials fold get the same NaN-propagation; the cuda-gated diff test
// in substrate_rmhd_gpu.rs exercises that on a GPU.
// =============================================================================

use symbi::kernels::support::cfl_from_lambda;
use symbi::regimes::substrate_gpu::{field_max_reduce, field_reduce};
use symbi::sim::evolve::check_dt;
use symbi_algebra::{Domain, Space};
use symbi_grid::Field;
use symbi_ir::emit::ReductionOp;
use symbi_xpu::HostMemory;

fn host_3d(alloc_hi: isize, lo: isize, hi: isize) -> (Domain<3>, Domain<3>) {
    let alloc = Domain::new([
        Space {
            name: "x",
            lo: 0,
            hi: alloc_hi,
        },
        Space {
            name: "y",
            lo: 0,
            hi: alloc_hi,
        },
        Space {
            name: "z",
            lo: 0,
            hi: alloc_hi,
        },
    ]);
    let interior = Domain::new([
        Space { name: "x", lo, hi },
        Space { name: "y", lo, hi },
        Space { name: "z", lo, hi },
    ]);
    (alloc, interior)
}

// one NaN cell among finite cells must make the max-reduction NaN (a reduction that
// returned the finite max would swallow it).
#[test]
fn single_nan_cell_propagates_through_max_reduce() {
    let (alloc, interior) = host_3d(8, 2, 6);
    let f = Field::<f64, 3, HostMemory>::zeros(&alloc).unwrap();
    for c in alloc.iter() {
        f.view_mut()
            .set(c, 0.5 + 0.001 * (c[0] + 5 * c[1] + 11 * c[2]) as f64);
    }
    // poison exactly one interior cell.
    f.view_mut().set([3, 3, 3], f64::NAN);

    let lambda_max = field_max_reduce(&f, &interior);
    assert!(
        lambda_max.is_nan(),
        "a single NaN cell was silently dropped by the max-reduction (got {lambda_max}); \
         the no-silent-floors dt guard depends on NaN surfacing here"
    );
}

// min must propagate NaN identically (positivity/lower-bound reductions rely on it).
#[test]
fn single_nan_cell_propagates_through_min_reduce() {
    let (alloc, interior) = host_3d(8, 2, 6);
    let f = Field::<f64, 3, HostMemory>::zeros(&alloc).unwrap();
    for c in alloc.iter() {
        f.view_mut()
            .set(c, 0.5 + 0.001 * (c[0] + 5 * c[1] + 11 * c[2]) as f64);
    }
    f.view_mut().set([4, 2, 5], f64::NAN);

    let m = field_reduce(&f, &interior, ReductionOp::Min);
    assert!(
        m.is_nan(),
        "min-reduction silently dropped a NaN cell (got {m})"
    );
}

// finite fields are unaffected — NaN-propagation must not perturb the normal path.
#[test]
fn finite_field_max_min_unchanged() {
    let (alloc, interior) = host_3d(8, 2, 6);
    let f = Field::<f64, 3, HostMemory>::zeros(&alloc).unwrap();
    for c in alloc.iter() {
        f.view_mut()
            .set(c, 0.5 + 0.001 * (c[0] + 5 * c[1] + 11 * c[2]) as f64);
    }
    let cells: Vec<f64> = interior.iter().map(|c| *f.view().at(c)).collect();
    let hmax = cells.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let hmin = cells.iter().cloned().fold(f64::INFINITY, f64::min);
    assert_eq!(field_reduce(&f, &interior, ReductionOp::Max), hmax);
    assert_eq!(field_reduce(&f, &interior, ReductionOp::Min), hmin);
}

// the full chain: NaN lambda_max -> NaN dt -> check_dt returns Err. a NaN-propagating reduction is
// what arms the fallible guard.
#[test]
fn nan_lambda_max_trips_dt_guard() {
    let (alloc, interior) = host_3d(8, 2, 6);
    let f = Field::<f64, 3, HostMemory>::zeros(&alloc).unwrap();
    for c in alloc.iter() {
        f.view_mut()
            .set(c, 0.5 + 0.001 * (c[0] + 5 * c[1] + 11 * c[2]) as f64);
    }
    f.view_mut().set([3, 3, 3], f64::NAN);
    let lambda_max = field_max_reduce(&f, &interior);
    let dt = cfl_from_lambda(lambda_max, 0.4);
    let err = check_dt(dt, 0, 0.0).expect_err("NaN dt must surface as Err");
    assert!(
        err.detail.contains("invalid dt"),
        "diagnostic preserved: {}",
        err.detail
    );
}
