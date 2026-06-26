// =============================================================================
// aot_kernel_invocation.rs
//
// the structured binding ABI (docs/design/15 §5, step 3a) — proof that routing a
// generated kernel through `KernelInvocation::run_cpu` produces BYTE-IDENTICAL
// output to calling the descriptor wrapper directly with `&[CpuField]` /
// `&mut [CpuFieldMut]`. the invocation is the backend-NEUTRAL seam (handle +
// layout + params); the CPU mapping just splits the buffers back into the
// generated fn's args. (the device-pointer handle + GPU launch arrive in 3c.)
// =============================================================================

use symbi_aot::{Buf, BufHandle, CpuField, CpuFieldMut, KernelInvocation, godunov_mass_1d};

const N: usize = 8;
const DT: f64 = 0.01;
const DX: f64 = 0.5;

// godunov_mass_1d buffers (signature order): [cons.den (in), mass_flux (in),
// cons.den_new (out)]; scalars [dt, dx_0]. mass_flux is a face field (N+1 entries).
fn inputs() -> (Vec<f64>, Vec<f64>) {
    let rho: Vec<f64> = (0..N).map(|i| if i < N / 2 { 1.0 } else { 0.125 }).collect();
    let flux: Vec<f64> = (0..=N).map(|i| 0.3 - 0.05 * i as f64).collect();
    (rho, flux)
}

#[test]
fn kernel_invocation_run_cpu_matches_direct_descriptor_call() {
    let lo = [0i32];
    let ext_n = [N as u32];
    let ext_n1 = [(N + 1) as u32];
    let grid = [N as u32];
    let dom_lo = [0i32];
    let scalars = [DT, DX];

    // --- direct descriptor call (the existing path) ---
    let (rho, flux) = inputs();
    let mut rho_new_direct = vec![0.0_f64; N];
    godunov_mass_1d(
        &[
            CpuField::from_layout(&rho, &lo, &ext_n),
            CpuField::from_layout(&flux, &lo, &ext_n1),
        ],
        &mut [CpuFieldMut::from_layout(&mut rho_new_direct, &lo, &ext_n)],
        &grid, &dom_lo, &[], &scalars,
    );

    // --- through the structured invocation seam ---
    let (rho2, flux2) = inputs();
    let mut rho_new_inv = vec![0.0_f64; N];
    let inv = KernelInvocation {
        buffers: vec![
            Buf { handle: BufHandle::Host(&rho2), lo: &lo, extent: &ext_n },
            Buf { handle: BufHandle::Host(&flux2), lo: &lo, extent: &ext_n1 },
            Buf { handle: BufHandle::HostMut(&mut rho_new_inv), lo: &lo, extent: &ext_n },
        ],
        grid: &grid,
        dom_lo: &dom_lo,
        ints: &[],
        scalars: &scalars,
    };
    inv.run_cpu(godunov_mass_1d);

    // byte-identical: the seam is a pure re-binding of the same args.
    assert_eq!(rho_new_inv, rho_new_direct, "invocation seam diverged from the direct call");
    // and the kernel actually ran (not a no-op).
    assert!(rho_new_direct.iter().zip(&rho).any(|(a, b)| a != b), "godunov was a no-op");
}
