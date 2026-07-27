// =============================================================================
// viscous_law.rs
//
// the constant-nu viscous kernel's gate: the compiled
// halo-1 stencil kernel is BIT-IDENTICAL to the f64 host chain built from the
// same carrier-generic `viscous_mom_update_2d` (read prim.vel/prim.rho on the
// 3x3 stencil -> conservative shear-stress flux divergence -> accumulate into
// cons.mom). dispatched over the interior so the +-1 stencil stays in bounds.
// =============================================================================

use symbi_algebra::Tensor;
use symbi_aot::{CpuField, CpuFieldMut, kernel_by_name};
use symbi_hydro::viscous::{viscous_mom_update_2d, viscous_mom_update_3d};

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
            let dmom = viscous_mom_update_2d(&vst, &rst, &[[NU; 3]; 3], DX, DX, DT);
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
    assert!(
        checked > 20,
        "the viscous operator never produced a nonzero force"
    );
}

// the alpha kernel. nu(x) = alpha c_s^2 / Omega_k(r) is a
// SPATIALLY VARYING viscosity (face-averaged in the core); the compiled kernel
// must be bit-identical to the f64 chain that computes the same per-cell nu.
#[test]
fn compiled_viscous_iso_alpha_matches_the_f64_chain_bitwise() {
    let (kernel, ir) = kernel_by_name::<f64>("viscous_iso_alpha_2d").expect("alpha viscous kernel");

    const ALPHA: f64 = 0.1;
    const CS: f64 = 0.1;
    const GM: f64 = 1.0;
    const BODY: [f64; 2] = [0.05, -0.03];

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
    let (m0_in, m1_in) = (m0.clone(), m1.clone());

    let scalar = |name: &str| -> f64 {
        match name {
            "dt" => DT,
            "alpha" => ALPHA,
            "cs" => CS,
            "body_0_mass" => GM,
            "body_0_pos_0" => BODY[0],
            "body_0_pos_1" => BODY[1],
            "dx_0" | "dx_1" => DX,
            "x_lo_0" | "x_lo_1" => X_LO,
            "map_kind_0" | "map_kind_1" | "map_param_0" | "map_param_1" => 0.0,
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

    // the per-cell alpha viscosity: nu(x) = alpha cs^2 / Omega_k(r).
    let nu_at = |ii: usize, jj: usize| -> f64 {
        let (x, y) = (cc(ii), cc(jj));
        let (rx, ry) = (x - BODY[0], y - BODY[1]);
        let r = (rx * rx + ry * ry).sqrt().max(1e-30);
        let omega_k = (GM / (r * r * r)).sqrt().max(1e-30);
        ALPHA * CS * CS / omega_k
    };

    let mut checked = 0usize;
    for jj in 1..N - 1 {
        for ii in 1..N - 1 {
            let mut vst = [[Tensor::<f64, 2>::zeros(); 3]; 3];
            let mut rst = [[0.0f64; 3]; 3];
            let mut nst = [[0.0f64; 3]; 3];
            for dj in 0..3 {
                for di in 0..3 {
                    let (si, sj) = (ii + di - 1, jj + dj - 1);
                    let c = si + sj * N;
                    vst[dj][di] = Tensor::new([v0[c], v1[c]]);
                    rst[dj][di] = rho[c];
                    nst[dj][di] = nu_at(si, sj);
                }
            }
            let dmom = viscous_mom_update_2d(&vst, &rst, &nst, DX, DX, DT);
            let c = ii + jj * N;
            assert_eq!(
                m0[c].to_bits(),
                (m0_in[c] + dmom[0]).to_bits(),
                "mom0 ({ii},{jj})"
            );
            assert_eq!(
                m1[c].to_bits(),
                (m1_in[c] + dmom[1]).to_bits(),
                "mom1 ({ii},{jj})"
            );
            if (m0[c] - m0_in[c]).abs() > 1e-14 {
                checked += 1;
            }
        }
    }
    assert!(
        checked > 20,
        "the alpha viscous operator never produced a nonzero force"
    );
}

// the 3D constant-nu gate: the compiled 27-cell stencil kernel is bit-identical
// to the f64 chain built from the same `viscous_mom_update_3d`.
const N3: usize = 12;

#[test]
fn compiled_viscous_iso_3d_matches_the_f64_chain_bitwise() {
    let (kernel, ir) = kernel_by_name::<f64>("viscous_iso_3d").expect("viscous 3d kernel");

    let n3 = N3 * N3 * N3;
    let at = |i: usize, j: usize, k: usize| i + j * N3 + k * N3 * N3;
    let (mut rho, mut v0, mut v1, mut v2) =
        (vec![0.0; n3], vec![0.0; n3], vec![0.0; n3], vec![0.0; n3]);
    let (mut m0, mut m1, mut m2) = (vec![0.0; n3], vec![0.0; n3], vec![0.0; n3]);
    for kk in 0..N3 {
        for jj in 0..N3 {
            for ii in 0..N3 {
                let (x, y, z) = (cc(ii), cc(jj), cc(kk));
                let c = at(ii, jj, kk);
                rho[c] = 1.0 + 0.2 * (2.0 * x).sin() * (1.5 * y).cos() * (z).sin();
                v0[c] = 0.3 * (x + y - z).cos();
                v1[c] = -0.2 * (x - 2.0 * y + z).sin();
                v2[c] = 0.15 * (0.5 * x + y - 1.5 * z).cos();
                m0[c] = rho[c] * v0[c];
                m1[c] = rho[c] * v1[c];
                m2[c] = rho[c] * v2[c];
            }
        }
    }
    let (m0_in, m1_in, m2_in) = (m0.clone(), m1.clone(), m2.clone());

    let scalar = |name: &str| -> f64 {
        match name {
            "dt" => DT,
            "nu" => NU,
            "dx_0" | "dx_1" | "dx_2" => DX,
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

    let lo = [0i32; 3];
    let ext = [N3 as u32; 3];
    let disp_lo = [1i32; 3];
    let disp_ext = [(N3 - 2) as u32; 3];
    {
        let inputs = [
            CpuField::from_layout(&rho, &lo, &ext),
            CpuField::from_layout(&v0, &lo, &ext),
            CpuField::from_layout(&v1, &lo, &ext),
            CpuField::from_layout(&v2, &lo, &ext),
            CpuField::from_layout(&m0_in, &lo, &ext),
            CpuField::from_layout(&m1_in, &lo, &ext),
            CpuField::from_layout(&m2_in, &lo, &ext),
        ];
        let mut outs = [
            CpuFieldMut::from_layout(&mut m0, &lo, &ext),
            CpuFieldMut::from_layout(&mut m1, &lo, &ext),
            CpuFieldMut::from_layout(&mut m2, &lo, &ext),
        ];
        kernel(&inputs, &mut outs, &disp_ext, &disp_lo, &ints, &scalars);
    }

    let mut checked = 0usize;
    for kk in 1..N3 - 1 {
        for jj in 1..N3 - 1 {
            for ii in 1..N3 - 1 {
                let mut vst = [[[Tensor::<f64, 3>::zeros(); 3]; 3]; 3];
                let mut rst = [[[0.0f64; 3]; 3]; 3];
                for dk in 0..3 {
                    for dj in 0..3 {
                        for di in 0..3 {
                            let c = at(ii + di - 1, jj + dj - 1, kk + dk - 1);
                            vst[dk][dj][di] = Tensor::new([v0[c], v1[c], v2[c]]);
                            rst[dk][dj][di] = rho[c];
                        }
                    }
                }
                let dmom = viscous_mom_update_3d(&vst, &rst, &[[[NU; 3]; 3]; 3], [DX; 3], DT);
                let c = at(ii, jj, kk);
                assert_eq!(
                    m0[c].to_bits(),
                    (m0_in[c] + dmom[0]).to_bits(),
                    "mom0 ({ii},{jj},{kk})"
                );
                assert_eq!(
                    m1[c].to_bits(),
                    (m1_in[c] + dmom[1]).to_bits(),
                    "mom1 ({ii},{jj},{kk})"
                );
                assert_eq!(
                    m2[c].to_bits(),
                    (m2_in[c] + dmom[2]).to_bits(),
                    "mom2 ({ii},{jj},{kk})"
                );
                if (m2[c] - m2_in[c]).abs() > 1e-14 {
                    checked += 1;
                }
            }
        }
    }
    assert!(
        checked > 20,
        "the 3D viscous operator never produced a nonzero z-force"
    );
}

// the 3D alpha gate: nu(x,y) = alpha cs^2 / Omega_k(R) with R the CYLINDRICAL
// radius (z does not enter), so every k-slice of the nu stencil is equal.
#[test]
fn compiled_viscous_iso_alpha_3d_matches_the_f64_chain_bitwise() {
    let (kernel, ir) =
        kernel_by_name::<f64>("viscous_iso_alpha_3d").expect("alpha viscous 3d kernel");

    const ALPHA: f64 = 0.1;
    const CS: f64 = 0.1;
    const GM: f64 = 1.0;
    const BODY: [f64; 2] = [0.05, -0.03];

    let n3 = N3 * N3 * N3;
    let at = |i: usize, j: usize, k: usize| i + j * N3 + k * N3 * N3;
    let (mut rho, mut v0, mut v1, mut v2) =
        (vec![0.0; n3], vec![0.0; n3], vec![0.0; n3], vec![0.0; n3]);
    let (mut m0, mut m1, mut m2) = (vec![0.0; n3], vec![0.0; n3], vec![0.0; n3]);
    for kk in 0..N3 {
        for jj in 0..N3 {
            for ii in 0..N3 {
                let (x, y, z) = (cc(ii), cc(jj), cc(kk));
                let c = at(ii, jj, kk);
                rho[c] = 1.0 + 0.2 * (2.0 * x).sin() * (1.5 * y).cos() * (z).sin();
                v0[c] = 0.3 * (x + y - z).cos();
                v1[c] = -0.2 * (x - 2.0 * y + z).sin();
                v2[c] = 0.15 * (0.5 * x + y - 1.5 * z).cos();
                m0[c] = rho[c] * v0[c];
                m1[c] = rho[c] * v1[c];
                m2[c] = rho[c] * v2[c];
            }
        }
    }
    let (m0_in, m1_in, m2_in) = (m0.clone(), m1.clone(), m2.clone());

    let scalar = |name: &str| -> f64 {
        match name {
            "dt" => DT,
            "alpha" => ALPHA,
            "cs" => CS,
            "body_0_mass" => GM,
            "body_0_pos_0" => BODY[0],
            "body_0_pos_1" => BODY[1],
            "dx_0" | "dx_1" | "dx_2" => DX,
            "x_lo_0" | "x_lo_1" | "x_lo_2" => X_LO,
            "map_kind_0" | "map_kind_1" | "map_kind_2" | "map_param_0" | "map_param_1" | "map_param_2" => 0.0,
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

    let lo = [0i32; 3];
    let ext = [N3 as u32; 3];
    let disp_lo = [1i32; 3];
    let disp_ext = [(N3 - 2) as u32; 3];
    {
        let inputs = [
            CpuField::from_layout(&rho, &lo, &ext),
            CpuField::from_layout(&v0, &lo, &ext),
            CpuField::from_layout(&v1, &lo, &ext),
            CpuField::from_layout(&v2, &lo, &ext),
            CpuField::from_layout(&m0_in, &lo, &ext),
            CpuField::from_layout(&m1_in, &lo, &ext),
            CpuField::from_layout(&m2_in, &lo, &ext),
        ];
        let mut outs = [
            CpuFieldMut::from_layout(&mut m0, &lo, &ext),
            CpuFieldMut::from_layout(&mut m1, &lo, &ext),
            CpuFieldMut::from_layout(&mut m2, &lo, &ext),
        ];
        kernel(&inputs, &mut outs, &disp_ext, &disp_lo, &ints, &scalars);
    }

    // cylindrical R: z does not enter Omega_k. the stencil position must match the
    // kernel arithmetic bit-for-bit: the cell centroid is `(lo_face + hi_face)/2`
    // with the uniform face `x_lo + i*dx`, and the offset cell sits at `centroid +
    // (off)*dx` — NOT `x_lo + (i+off+0.5)*dx`, which rounds differently and the
    // sqrt/division in nu(x) amplifies.
    let face = |i_f: f64| X_LO + i_f * DX;
    let centroid = |c: usize| (face(c as f64) + face(c as f64 + 1.0)) * 0.5;
    let nu_at = |cx: f64, cy: f64| -> f64 {
        let (rx, ry) = (cx - BODY[0], cy - BODY[1]);
        let r = (rx * rx + ry * ry).sqrt().max(1e-30);
        let omega_k = (GM / (r * r * r)).sqrt().max(1e-30);
        ALPHA * CS * CS / omega_k
    };

    let mut checked = 0usize;
    for kk in 1..N3 - 1 {
        for jj in 1..N3 - 1 {
            for ii in 1..N3 - 1 {
                let mut vst = [[[Tensor::<f64, 3>::zeros(); 3]; 3]; 3];
                let mut rst = [[[0.0f64; 3]; 3]; 3];
                let mut nst = [[[0.0f64; 3]; 3]; 3];
                for dk in 0..3 {
                    for dj in 0..3 {
                        for di in 0..3 {
                            let (si, sj, sk) = (ii + di - 1, jj + dj - 1, kk + dk - 1);
                            let c = at(si, sj, sk);
                            vst[dk][dj][di] = Tensor::new([v0[c], v1[c], v2[c]]);
                            rst[dk][dj][di] = rho[c];
                            let xk = centroid(ii) + (di as f64 - 1.0) * DX;
                            let yk = centroid(jj) + (dj as f64 - 1.0) * DX;
                            nst[dk][dj][di] = nu_at(xk, yk);
                        }
                    }
                }
                let dmom = viscous_mom_update_3d(&vst, &rst, &nst, [DX; 3], DT);
                let c = at(ii, jj, kk);
                assert_eq!(
                    m0[c].to_bits(),
                    (m0_in[c] + dmom[0]).to_bits(),
                    "mom0 ({ii},{jj},{kk})"
                );
                assert_eq!(
                    m1[c].to_bits(),
                    (m1_in[c] + dmom[1]).to_bits(),
                    "mom1 ({ii},{jj},{kk})"
                );
                assert_eq!(
                    m2[c].to_bits(),
                    (m2_in[c] + dmom[2]).to_bits(),
                    "mom2 ({ii},{jj},{kk})"
                );
                if (m2[c] - m2_in[c]).abs() > 1e-14 {
                    checked += 1;
                }
            }
        }
    }
    assert!(
        checked > 20,
        "the 3D alpha viscous operator never produced a nonzero z-force"
    );
}

// the GENERAL ORTHOGONAL kernel on the cylindrical chart: bit-identical to the f64
// `viscous_mom_update_orthogonal_2d` fed the scale factors h = (1, R) that the
// kernel reads from CylindricalRPhi::scale_factors (R = the cell centroid + offset).
// this is the ONE kernel every curvilinear chart routes through.
#[test]
fn compiled_viscous_iso_ortho_cyl_matches_the_f64_chain_bitwise() {
    use symbi_hydro::viscous::viscous_mom_update_orthogonal_2d;
    let (kernel, ir) = kernel_by_name::<f64>("viscous_iso_ortho_cyl_2d").expect("ortho cyl kernel");

    const R_LO: f64 = 0.1;
    const DR: f64 = 0.05;
    const PHI_LO: f64 = 0.0;
    const DPHI: f64 = 0.1;

    let n2 = N * N;
    let (mut rho, mut v0, mut v1) = (vec![0.0; n2], vec![0.0; n2], vec![0.0; n2]);
    let (mut m0, mut m1) = (vec![0.0; n2], vec![0.0; n2]);
    for jj in 0..N {
        for ii in 0..N {
            let (rr, pp) = (
                R_LO + (ii as f64 + 0.5) * DR,
                PHI_LO + (jj as f64 + 0.5) * DPHI,
            );
            let c = ii + jj * N;
            rho[c] = 1.0 + 0.2 * (2.0 * rr).sin() * (1.5 * pp).cos();
            v0[c] = 0.3 * (rr + pp).cos();
            v1[c] = (1.0 / rr).sqrt() - 0.2 * (rr - 2.0 * pp).sin();
            m0[c] = rho[c] * v0[c];
            m1[c] = rho[c] * v1[c];
        }
    }
    let (m0_in, m1_in) = (m0.clone(), m1.clone());

    let scalar = |name: &str| -> f64 {
        match name {
            "dt" => DT,
            "nu" => NU,
            "dx_0" => DR,
            "dx_1" => DPHI,
            "x_lo_0" => R_LO,
            "x_lo_1" => PHI_LO,
            "map_kind_0" | "map_kind_1" | "map_param_0" | "map_param_1" => 0.0,
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

    // f64 chain: h1 = 1, h2 = R = r_c + (di-1) dr (the CylindricalRPhi scale factor),
    // r_c the volume-weighted centroid, then the general orthogonal operator.
    let rl = |i: usize| R_LO + i as f64 * DR;
    let rh = |i: usize| R_LO + (i as f64 + 1.0) * DR;
    let r_c = |i: usize| {
        let (a, b) = (rh(i), rl(i));
        ((a * a * a - b * b * b) / 3.0) / ((a * a - b * b) / 2.0)
    };
    let ones = [[1.0f64; 3]; 3];
    let mut checked = 0usize;
    for jj in 1..N - 1 {
        for ii in 1..N - 1 {
            let rc = r_c(ii);
            let mut vst = [[Tensor::<f64, 2>::zeros(); 3]; 3];
            let mut rst = [[0.0f64; 3]; 3];
            let mut h2 = [[0.0f64; 3]; 3];
            for dj in 0..3 {
                for di in 0..3 {
                    let c = (ii + di - 1) + (jj + dj - 1) * N;
                    vst[dj][di] = Tensor::new([v0[c], v1[c]]);
                    rst[dj][di] = rho[c];
                    h2[dj][di] = rc + (di as f64 - 1.0) * DR;
                }
            }
            let dmom = viscous_mom_update_orthogonal_2d(
                &vst,
                &rst,
                &[[NU; 3]; 3],
                &ones,
                &h2,
                DR,
                DPHI,
                DT,
            );
            let c = ii + jj * N;
            assert_eq!(
                m0[c].to_bits(),
                (m0_in[c] + dmom[0]).to_bits(),
                "mom0 ({ii},{jj})"
            );
            assert_eq!(
                m1[c].to_bits(),
                (m1_in[c] + dmom[1]).to_bits(),
                "mom1 ({ii},{jj})"
            );
            if (m1[c] - m1_in[c]).abs() > 1e-14 {
                checked += 1;
            }
        }
    }
    assert!(
        checked > 20,
        "the general orthogonal operator never produced a torque"
    );
}

// the GENERAL ORTHOGONAL ALPHA kernel on the cylindrical chart: nu(R) = alpha cs^2 /
// Omega_k(R) is filled into the ortho operator's nu stencil, R = x0 the radial
// coordinate. bit-identical to the f64 chain.
#[test]
fn compiled_viscous_iso_alpha_ortho_cyl_matches_the_f64_chain_bitwise() {
    use symbi_hydro::viscous::viscous_mom_update_orthogonal_2d;
    let (kernel, ir) =
        kernel_by_name::<f64>("viscous_iso_alpha_ortho_cyl_2d").expect("alpha ortho cyl kernel");

    const R_LO: f64 = 0.1;
    const DR: f64 = 0.05;
    const PHI_LO: f64 = 0.0;
    const DPHI: f64 = 0.1;
    const ALPHA: f64 = 0.1;
    const CS: f64 = 0.1;
    const GM: f64 = 1.0;

    let n2 = N * N;
    let (mut rho, mut v0, mut v1) = (vec![0.0; n2], vec![0.0; n2], vec![0.0; n2]);
    let (mut m0, mut m1) = (vec![0.0; n2], vec![0.0; n2]);
    for jj in 0..N {
        for ii in 0..N {
            let (rr, pp) = (
                R_LO + (ii as f64 + 0.5) * DR,
                PHI_LO + (jj as f64 + 0.5) * DPHI,
            );
            let c = ii + jj * N;
            rho[c] = 1.0 + 0.2 * (2.0 * rr).sin() * (1.5 * pp).cos();
            v0[c] = 0.3 * (rr + pp).cos();
            v1[c] = (GM / rr).sqrt() - 0.2 * (rr - 2.0 * pp).sin();
            m0[c] = rho[c] * v0[c];
            m1[c] = rho[c] * v1[c];
        }
    }
    let (m0_in, m1_in) = (m0.clone(), m1.clone());

    let scalar = |name: &str| -> f64 {
        match name {
            "dt" => DT,
            "alpha" => ALPHA,
            "cs" => CS,
            "body_0_mass" => GM,
            "dx_0" => DR,
            "dx_1" => DPHI,
            "x_lo_0" => R_LO,
            "x_lo_1" => PHI_LO,
            "map_kind_0" | "map_kind_1" | "map_param_0" | "map_param_1" => 0.0,
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

    let rl = |i: usize| R_LO + i as f64 * DR;
    let rh = |i: usize| R_LO + (i as f64 + 1.0) * DR;
    let r_c = |i: usize| {
        let (a, b) = (rh(i), rl(i));
        ((a * a * a - b * b * b) / 3.0) / ((a * a - b * b) / 2.0)
    };
    let ones = [[1.0f64; 3]; 3];
    let mut checked = 0usize;
    for jj in 1..N - 1 {
        for ii in 1..N - 1 {
            let rc = r_c(ii);
            let mut vst = [[Tensor::<f64, 2>::zeros(); 3]; 3];
            let mut rst = [[0.0f64; 3]; 3];
            let mut h2 = [[0.0f64; 3]; 3];
            let mut nst = [[0.0f64; 3]; 3];
            for dj in 0..3 {
                for di in 0..3 {
                    let c = (ii + di - 1) + (jj + dj - 1) * N;
                    vst[dj][di] = Tensor::new([v0[c], v1[c]]);
                    rst[dj][di] = rho[c];
                    let x0 = rc + (di as f64 - 1.0) * DR;
                    h2[dj][di] = x0;
                    let r = x0.max(1e-30);
                    let omega_k = (GM / (r * r * r)).sqrt().max(1e-30);
                    nst[dj][di] = ALPHA * CS * CS / omega_k;
                }
            }
            let dmom = viscous_mom_update_orthogonal_2d(&vst, &rst, &nst, &ones, &h2, DR, DPHI, DT);
            let c = ii + jj * N;
            assert_eq!(
                m0[c].to_bits(),
                (m0_in[c] + dmom[0]).to_bits(),
                "mom0 ({ii},{jj})"
            );
            assert_eq!(
                m1[c].to_bits(),
                (m1_in[c] + dmom[1]).to_bits(),
                "mom1 ({ii},{jj})"
            );
            if (m1[c] - m1_in[c]).abs() > 1e-14 {
                checked += 1;
            }
        }
    }
    assert!(
        checked > 20,
        "the general orthogonal alpha operator never produced a torque"
    );
}
