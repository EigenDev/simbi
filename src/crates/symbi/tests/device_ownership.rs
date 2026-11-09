// =============================================================================
// device_ownership.rs
//
// the device a tile's halo copy and a device reduction run on, on a node with two or more gpus:
// - a staged copy between two fields of logical device 1 gathers and scatters on device 1, with
//   device 0 current on the calling thread, and moves the strip exactly
// - reductions alternating between devices 0 and 1 return each field's exact minimum and sum, and
//   each reduces into scratch allocated in its own device's context
// the device-binding checks need a second device: on one visible gpu the test fails by name.
//
// run (gpu build): cargo test -p symbi --features hip --test device_ownership
// =============================================================================
#![cfg(feature = "gpu")]

use symbi::sim::decomp::{HaloTransport, StagedCopy, strip_launches_by_device};
use symbi_algebra::{Domain, Space};
use symbi_exec::engine::{field_reduce, field_sum_exact, last_reduction_scratch_device};
use symbi_grid::Field;
use symbi_ir::emit::ReductionOp;
use symbi_xpu::{DeviceMemory, device_count, with_device};

fn plane(lo: [isize; 2], hi: [isize; 2]) -> Domain<2> {
    Domain::new([Space { name: "x", lo: lo[0], hi: hi[0] }, Space { name: "y", lo: lo[1], hi: hi[1] }])
}

fn require_two_devices() {
    let n = device_count().expect("the device count");
    assert!(n >= 2, "{n} gpu visible: a copy or reduction on device 1 lands on device 0, and the checks are vacuous");
}

#[test]
fn a_staged_copy_runs_on_its_tiles_device_whatever_the_caller_has_current() {
    require_two_devices();
    let whole = plane([0, 0], [8, 8]);
    let (src, dst) = with_device(1, || {
        (Field::<f64, 2, DeviceMemory>::zeros(&whole).expect("src"), Field::<f64, 2, DeviceMemory>::zeros(&whole).expect("dst"))
    });
    for ii in 0..8 {
        for jj in 0..8 {
            src.set([ii, jj], (10 * ii + jj) as f64 + 0.5);
        }
    }
    let from = plane([2, 0], [4, 8]);
    let to = plane([5, 0], [7, 8]);
    assert_eq!(symbi_xpu::current_device(), 0, "the caller's current device");
    let before = strip_launches_by_device();
    StagedCopy.copy_region(&src, &from, &dst, &to, 1, 1);
    let after = strip_launches_by_device();
    assert_eq!(after[1] - before[1], 2, "the gather and the scatter run on device 1");
    assert_eq!(after[0], before[0], "the copy launched on the caller's device 0");
    for ii in 0..8 {
        for jj in 0..8 {
            let want = if (5..7).contains(&ii) { (10 * (ii - 3) + jj) as f64 + 0.5 } else { 0.0 };
            assert_eq!(*dst.at([ii, jj]), want, "cell ({ii}, {jj})");
        }
    }
}

#[test]
fn reductions_alternating_between_devices_reduce_into_their_own_devices_scratch() {
    require_two_devices();
    let whole = plane([0, 0], [64, 48]);
    for (round, dev) in [0, 1, 0, 1, 1, 0].into_iter().enumerate() {
        let field = with_device(dev, || Field::<f64, 2, DeviceMemory>::zeros(&whole).expect("field"));
        let mut expected_sum = 0.0;
        for ii in 0..64 {
            for jj in 0..48 {
                // integers and halves: every partial sum is exact, so the expected sum is too
                let value = (ii * 48 + jj) as f64 * 0.5 - 7.0 - round as f64;
                field.set([ii, jj], value);
                expected_sum += value;
            }
        }
        let min = with_device(dev, || field_reduce(&field, &whole, ReductionOp::Min));
        assert_eq!(min, -7.0 - round as f64, "round {round}, device {dev}: minimum");
        assert_eq!(last_reduction_scratch_device(), dev, "round {round}: the minimum reduced into another device's scratch");
        let sum = with_device(dev, || field_sum_exact(&field, &whole).value());
        assert_eq!(sum, expected_sum, "round {round}, device {dev}: exact sum");
        assert_eq!(last_reduction_scratch_device(), dev, "round {round}: the exact sum reduced into another device's scratch");
    }
}
