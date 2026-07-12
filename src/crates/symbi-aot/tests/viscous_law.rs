// =============================================================================
// viscous_law.rs
//
// the constant-nu viscous kernel's gate (docs/design/54 step 1): the compiled
// halo-1 stencil kernel is BIT-IDENTICAL to the f64 host chain built from the
// same carrier-generic `viscous_mom_update_2d` (read prim.vel/prim.rho on the
// 3x3 stencil -> conservative shear-stress flux divergence -> accumulate into
// cons.mom). dispatched over the interior so the +-1 stencil stays in bounds.
// =============================================================================

use symbi_aot::{kernel_by_name, CpuField, CpuFieldMut};
use symbi_algebra::Tensor;
use symbi_hydro::viscous::viscous_mom_update_2d;

const N: usize = 24;
const X_LO: f64 = -0.6;
const DX: f64 = 0.05;
const NU: f64 = 0.02;
const DT: f64 = 0.001;

fn cc(i: usize) -> f64 {
    X_LO + (i as f64 + 0.5) * DX
}

#[test]
fn compiled_viscous_iso_matches_the_f64_chain_bitwise() {
    let (kernel, ir) = kernel_by_name::<f64>("viscous_iso_2d").expect("viscous kernel");

    let n2 = N * N;
    let (mut rho, mut v0, mut v1) = (vec![0.0; n2], vec![0.0; n2], vec![0.0; n2]);
    let (mut m0, mut m1) = (vec![0.0; n2], vec![0.0; n2]);
    for jj in 0..N {
        for ii in 0..N {
            let (x, y) = (cc(ii), cc(jj));
            let c = ii + jj * N;
            rho[c] = 1.0 + 0.2 * (2.0 * x).sin() * (1.5 * y).cos();
            v0[c] = 0.3 * (x + y).cos();
            v1[c] = -0.2 * (x - 2.0 * y).sin();
            m0[c] = rho[c] * v0[c];
            m1[c] = rho[c] * v1[c];
        }
    }
    let m0_in = m0.clone();
    let m1_in = m1.clone();

    let scalar = |name: &str| -> f64 {
        match name {
            "dt" => DT,
            "nu" => NU,
            "dx_0" | "dx_1" => DX,
            other => panic!("unexpected scalar '{other}'"),
        }
    };
    let (mut ints, mut scalars) = (Vec::new(), Vec::new());
    for (bind, is_int) in symbi_ir::kernel_scalar_params_typed_from_ir(ir) {
        let v = scalar(&bind.name());
        if is_int {
            ints.push(v as i32)
        } else {
            scalars.push(v)
        }
    }

    let lo = [0i32; 2];
    let ext = [N as u32; 2];
    // dispatch over the INTERIOR only: the +-1 stencil reads must stay in bounds.
    let disp_lo = [1i32; 2];
    let disp_ext = [(N - 2) as u32; 2];
    {
        let inputs = [
            CpuField::from_layout(&rho, &lo, &ext),
            CpuField::from_layout(&v0, &lo, &ext),
            CpuField::from_layout(&v1, &lo, &ext),
            CpuField::from_layout(&m0_in, &lo, &ext),
            CpuField::from_layout(&m1_in, &lo, &ext),
        ];
        let mut outs = [
            CpuFieldMut::from_layout(&mut m0, &lo, &ext),
            CpuFieldMut::from_layout(&mut m1, &lo, &ext),
        ];
        kernel(&inputs, &mut outs, &disp_ext, &disp_lo, &ints, &scalars);
    }

    // the f64 chain, over the same interior.
    let mut checked = 0usize;
    for jj in 1..N - 1 {
        for ii in 1..N - 1 {
            let mut vst = [[Tensor::<f64, 2>::zeros(); 3]; 3];
            let mut rst = [[0.0f64; 3]; 3];
            for dj in 0..3 {
                for di in 0..3 {
                    let c = (ii + di - 1) + (jj + dj - 1) * N;
                    vst[dj][di] = Tensor::new([v0[c], v1[c]]);
                    rst[dj][di] = rho[c];
                }
            }
            let dmom = viscous_mom_update_2d(&vst, &rst, NU, DX, DX, DT);
            let c = ii + jj * N;
            let e0 = m0_in[c] + dmom[0];
            let e1 = m1_in[c] + dmom[1];
            assert_eq!(m0[c].to_bits(), e0.to_bits(), "mom0 at ({ii},{jj})");
            assert_eq!(m1[c].to_bits(), e1.to_bits(), "mom1 at ({ii},{jj})");
            // the operator actually did something (not a trivial zero everywhere).
            if (m0[c] - m0_in[c]).abs() > 1e-14 {
                checked += 1;
            }
        }
    }
    assert!(checked > 20, "the viscous operator never produced a nonzero force");
}
