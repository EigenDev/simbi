// =============================================================================
// device_bindings.rs
//
// the boundary between host-resident and device-resident storage at kernel binding:
// - host-resident storage (host memory, the opaque cpu stand-in) binds as host slices
// - device-resident storage binds as its address and length, read or written as declared
// - the cpu executor, the cover executor and the host fold refuse device-resident storage
// - the alias checks hold on device bindings as on host ones
// - a view refuses a coordinate outside it in every build
// device-resident storage is reached on a host-only build through `DeviceResidentProbe`, host
// memory declaring itself device-resident: every binding decision reads the declaration alone.
//
// usage:
//   cargo test -p symbi-exec --test device_bindings
// =============================================================================

use symbi_algebra::{Domain, Space};
use symbi_aot::{Buf, BufHandle, KernelInvocation};
use symbi_exec::policy::{bind_input, bind_output, disjoint_host_buffers};
use symbi_grid::Field;
use symbi_xpu::{HostMemory, HostOpaque, MemoryStorage, NoHostAccess};

/// host memory declaring itself device-resident and cpu-executed
struct DeviceResidentProbe;

impl MemoryStorage for DeviceResidentProbe {
    type HostAccess = NoHostAccess;
    const IS_HOST_ACCESSIBLE: bool = false;
    const IS_HOST_RESIDENT: bool = false;
    const IS_DEVICE_ACCESSIBLE: bool = false;
    const IS_UNIFIED: bool = false;
    const PREFERRED_ALIGNMENT: usize = HostMemory::PREFERRED_ALIGNMENT;

    fn allocate(bytes: usize) -> symbi_xpu::Result<*mut u8> {
        HostMemory::allocate(bytes)
    }

    fn deallocate(ptr: *mut u8, bytes: usize) {
        HostMemory::deallocate(ptr, bytes)
    }

    unsafe fn copy_to_host(dst: *mut u8, src: *const u8, bytes: usize) -> symbi_xpu::Result<()> {
        unsafe { HostMemory::copy_to_host(dst, src, bytes) }
    }

    unsafe fn copy_from_host(dst: *mut u8, src: *const u8, bytes: usize) -> symbi_xpu::Result<()> {
        unsafe { HostMemory::copy_from_host(dst, src, bytes) }
    }
}

fn dom() -> Domain<1> {
    Domain::new([Space { name: "x", lo: 0, hi: 8 }])
}

fn field<Mem: MemoryStorage>() -> Field<f64, 1, Mem> {
    Field::zeros(&dom()).expect("field")
}

#[test]
fn host_resident_storage_binds_as_host_slices() {
    let (host, opaque) = (field::<HostMemory>(), field::<HostOpaque>());
    let (h_in, o_out) = unsafe { (bind_input(&host), bind_output(&opaque)) };
    assert!(matches!(h_in, BufHandle::Host(d) if d.len() == 8 && d.as_ptr() == host.as_ptr()));
    assert!(matches!(o_out, BufHandle::HostMut(ref d) if d.len() == 8 && d.as_ptr() == opaque.as_ptr()));
}

#[test]
fn device_resident_storage_binds_as_address_and_length_in_its_declared_role() {
    let (read, written) = (field::<DeviceResidentProbe>(), field::<DeviceResidentProbe>());
    let (input, output) = unsafe { (bind_input(&read), bind_output(&written)) };
    assert!(matches!(input, BufHandle::Device(_)), "a device input bound as a host slice");
    assert!(matches!(output, BufHandle::DeviceMut(_)), "a device output bound as a host slice");
    assert_eq!((input.as_ptr(), input.len(), input.is_output()), (read.as_ptr(), 8, false));
    assert_eq!((output.as_ptr(), output.len(), output.is_output()), (written.as_ptr() as *const f64, 8, true));
}

#[test]
fn an_empty_allocation_binds_as_an_empty_slice() {
    let empty = Field::<f64, 1, DeviceResidentProbe>::zeros(&Domain::new([Space { name: "x", lo: 0, hi: 0 }])).unwrap();
    assert!(matches!(unsafe { bind_input(&empty) }, BufHandle::Host(d) if d.is_empty()));
}

fn noop(_: &[symbi_aot::CpuField<'_, f64>], _: &mut [symbi_aot::CpuFieldMut<'_, f64>], _: &[u32], _: &[i32], _: &[i32], _: &[f64]) {}

#[test]
#[should_panic(expected = "a device-resident buffer reached the cpu executor")]
fn the_cpu_executor_refuses_a_device_binding() {
    let (read, written) = (field::<DeviceResidentProbe>(), field::<DeviceResidentProbe>());
    let layouts = [([0i32], [8u32], 8usize); 2];
    let buffers = disjoint_host_buffers::<f64, 1, DeviceResidentProbe>("probe", &[&read], &[&written], &layouts);
    let inv = KernelInvocation { buffers, grid: &[8], dom_lo: &[0], ints: &[], scalars: &[] };
    // the storage is cpu-executed, so the dispatch takes the cpu path and must refuse
    symbi_exec::engine::dispatch::<f64, DeviceResidentProbe, _>(inv, "", "probe", noop);
}

#[test]
fn the_cover_executor_declines_device_resident_storage() {
    let (read, written) = (field::<DeviceResidentProbe>(), field::<DeviceResidentProbe>());
    let ran = symbi_exec::policy::dispatch_fields_cover::<f64, DeviceResidentProbe, 1>(
        "probe",
        &dom(),
        [4],
        &[&read],
        &[&written],
        &[],
        &[],
    );
    assert!(!ran, "the cover executor sliced device-resident storage");
}

#[test]
#[should_panic(expected = "the CPU executor reads host-resident storage")]
fn the_host_fold_refuses_device_resident_storage() {
    let f = field::<DeviceResidentProbe>();
    let _ = symbi_exec::engine::field_reduce(&f, &dom(), symbi_ir::emit::ReductionOp::Max);
}

#[test]
#[should_panic(expected = "two OUTPUT bindings resolve to the same allocation")]
fn two_device_outputs_on_one_allocation_are_refused() {
    let f = field::<DeviceResidentProbe>();
    let l = [([0i32], [8u32], 8usize); 2];
    let _ = disjoint_host_buffers::<f64, 1, DeviceResidentProbe>("probe", &[], &[&f, &f], &l);
}

#[test]
#[should_panic(expected = "a binding is both an input and an output")]
fn a_device_allocation_bound_as_input_and_output_is_refused() {
    let f = field::<DeviceResidentProbe>();
    let l = [([0i32], [8u32], 8usize); 2];
    let _ = disjoint_host_buffers::<f64, 1, DeviceResidentProbe>("probe", &[&f], &[&f], &l);
}

#[test]
fn a_device_binding_carries_its_layout_into_the_invocation() {
    let f = field::<DeviceResidentProbe>();
    let l = [([0i32], [8u32], 8usize)];
    let bufs: Vec<Buf<'_, f64>> = disjoint_host_buffers::<f64, 1, DeviceResidentProbe>("probe", &[&f], &[], &l);
    assert_eq!((bufs[0].lo, bufs[0].extent), (&[0i32][..], &[8u32][..]));
    assert_eq!(bufs[0].handle.as_ptr(), f.as_ptr());
}

#[test]
#[should_panic(expected = "lies outside the view")]
fn a_view_refuses_a_coordinate_outside_it() {
    let f = field::<HostMemory>();
    let _ = *f.view().at([8]);
}

#[test]
#[should_panic(expected = "lies outside the view")]
fn a_mutable_view_refuses_a_coordinate_outside_it() {
    let f = field::<HostMemory>();
    f.view_mut().set([-1], 1.0);
}
