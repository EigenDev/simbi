// =============================================================================
// penalize_law.rs
//
// the [Drain] penalization kernel's gates (docs/design/50 step 2): the
// compiled kernel is BIT-IDENTICAL to the f64 host chain built from the same
// carrier-generic functions (sphere SDF chi -> Relax -> penalize_cell), the
// per-cell deltas equal the gas's conserved loss exactly, and the blob
// declares its support ball.
// =============================================================================

use symbi_algebra::Tensor;
use symbi_aot::{kernel_by_name, CpuField, CpuFieldMut};
use symbi_hydro::energy::Adiabatic;
use symbi_hydro::state::ConsG;
use symbi_ib::penalize::{penalize_cell, BodyKin, Property, Relax};
use symbi_ib::sdf::{chi, SdfExpr};
use symbi_ir::kernel_output_support_from_ir;

const N: usize = 48;
const X_LO: f64 = -1.2;
const DX: f64 = 0.05;
const POS: [f64; 2] = [0.11, -0.07];
const RACC: f64 = 0.15;
const C_DRAIN: f64 = 1.0;
const DT: f64 = 0.008;
const GAMMA: f64 = 1.4;

fn cell_center(i: usize) -> f64 {
    X_LO + (i as f64 + 0.5) * DX
}

#[test]
fn compiled_drain_penalize_matches_the_f64_chain_bitwise() {
    let (kernel, ir) = kernel_by_name::<f64>("penalize_drain_2d").expect("kernel in registry");
    assert!(
        kernel_output_support_from_ir(ir).is_some(),
        "the penalize kernel must declare its support ball",
    );

    // nontrivial cons over the grid, axis 0 contiguous.
    let n2 = N * N;
    let (mut den, mut mx, mut my, mut nrg) =
        (vec![0.0; n2], vec![0.0; n2], vec![0.0; n2], vec![0.0; n2]);
    for jj in 0..N {
        for ii in 0..N {
            let (x, y) = (cell_center(ii), cell_center(jj));
            let c = ii + jj * N;
            den[c] = 1.0 + 0.2 * (2.0 * x).sin() * (1.5 * y).cos();
            mx[c] = den[c] * 0.3 * (x + y).cos();
            my[c] = den[c] * -0.2 * (x - 2.0 * y).sin();
            nrg[c] = den[c] * (1.8 + 0.5 * (0.3 * (x * y)).sin());
        }
    }
    let host_in = (den.clone(), mx.clone(), my.clone(), nrg.clone());

    // scalar order from the manifest, resolved by name.
    let scalar = |name: &str| -> f64 {
        match name {
            "dt" => DT,
            "gamma" => GAMMA,
            "c_drain" => C_DRAIN,
            "x_lo_0" | "x_lo_1" => X_LO,
            "dx_0" | "dx_1" => DX,
            "map_kind_0" | "map_kind_1" => 0.0,
            "body_0_pos_0" => POS[0],
            "body_0_pos_1" => POS[1],
            "body_0_racc" => RACC,
            other => panic!("unexpected scalar '{other}'"),
        }
    };
    let (mut ints, mut scalars) = (Vec::new(), Vec::new());
    for (bind, is_int) in symbi_ir::kernel_scalar_params_typed_from_ir(ir) {
        let v = scalar(&bind.name());
        if is_int { ints.push(v as i32) } else { scalars.push(v) }
    }
    let lo = [0i32; 2];
    let ext = [N as u32; 2];
    let mut pm = vec![7.7; n2];
    let mut pfx = vec![7.7; n2];
    let mut pfy = vec![7.7; n2];
    let mut pe = vec![7.7; n2];
    let mut pt = vec![7.7; n2];
    {
        // in-place cons: the same buffers appear as inputs and outputs.
        let inputs = [
            CpuField::from_layout(&den, &lo, &ext),
            CpuField::from_layout(&mx, &lo, &ext),
            CpuField::from_layout(&my, &lo, &ext),
            CpuField::from_layout(&nrg, &lo, &ext),
        ];
        // SAFETY-free aliasing dance: run_parallel-style in-place needs the
        // same buffer in both lists; the generated kernel reads before it
        // writes per cell. bind fresh output views over clones instead — the
        // kernel's outputs land in the clones, inputs stay pristine.
        let mut den_o = den.clone();
        let mut mx_o = mx.clone();
        let mut my_o = my.clone();
        let mut nrg_o = nrg.clone();
        let mut outs = [
            CpuFieldMut::from_layout(&mut den_o, &lo, &ext),
            CpuFieldMut::from_layout(&mut mx_o, &lo, &ext),
            CpuFieldMut::from_layout(&mut my_o, &lo, &ext),
            CpuFieldMut::from_layout(&mut nrg_o, &lo, &ext),
            CpuFieldMut::from_layout(&mut pm, &lo, &ext),
            CpuFieldMut::from_layout(&mut pfx, &lo, &ext),
            CpuFieldMut::from_layout(&mut pfy, &lo, &ext),
            CpuFieldMut::from_layout(&mut pe, &lo, &ext),
            CpuFieldMut::from_layout(&mut pt, &lo, &ext),
        ];
        kernel(&inputs, &mut outs, &[N as u32; 2], &[0i32; 2], &ints, &scalars);
        drop(outs);
        den = den_o;
        mx = mx_o;
        my = my_o;
        nrg = nrg_o;
    }

    // the f64 host chain: the SAME carrier-generic functions per cell.
    let sphere = SdfExpr::<f64, 2>::sphere(POS, RACC);
    // the kernel's volume comes from the geometry scaffold: per-axis widths
    // as FACE differences (x_lo + (i+1)dx) - (x_lo + i dx), the product
    // reciprocated twice (dv = 1/inv_volume). mirror the exact chain.
    let width = |i: usize| (X_LO + (i as f64 + 1.0) * DX) - (X_LO + i as f64 * DX);
    let mut interior_nonzero = 0usize;
    for jj in 0..N {
        for ii in 0..N {
            let c = ii + jj * N;
            let cons = ConsG::<f64, 2, Adiabatic> {
                den: host_in.0[c],
                mom: Tensor::new([host_in.1[c], host_in.2[c]]),
                nrg: host_in.3[c],
            };
            // the kernel's centroid is the arithmetic mid of the FACE
            // positions, not x_lo + (i+0.5)dx — mirror it to the bit.
            let mid = |i: usize| {
                ((X_LO + i as f64 * DX) + (X_LO + (i as f64 + 1.0) * DX)) * 0.5
            };
            let x = [mid(ii), mid(jj)];
            let dv = 1.0 / (1.0 / (width(ii) * width(jj)));
            let ch = chi(sphere.dist(x), DX);
            let mom_sq = cons.mom.dot(&cons.mom);
            let cs = symbi_ib::drain::sound_speed_from_cons(cons.den, mom_sq, cons.nrg, GAMMA);
            let inv_tau = cs / (C_DRAIN * DX);
            let kin = BodyKin::<f64, 2> { u_solid: Tensor::zeros(), omega: Tensor::zeros(), e_wall: 0.0 };
            let mut acc = Relax::none();
            Property::Drain { inv_tau }.contribute(ch, &kin, &mut acc);
            let (out, delta) = penalize_cell(&cons, &acc, Tensor::zeros(), DT, dv, 0);

            assert_eq!(den[c].to_bits(), out.den.to_bits(), "den at ({ii},{jj})");
            assert_eq!(mx[c].to_bits(), out.mom[0].to_bits(), "mom0 at ({ii},{jj})");
            assert_eq!(my[c].to_bits(), out.mom[1].to_bits(), "mom1 at ({ii},{jj})");
            assert_eq!(nrg[c].to_bits(), out.nrg.to_bits(), "nrg at ({ii},{jj})");
            assert_eq!(pm[c].to_bits(), delta.mass_delta.to_bits(), "mass at ({ii},{jj})");
            assert_eq!(pfx[c].to_bits(), delta.force_delta[0].to_bits(), "fx at ({ii},{jj})");
            assert_eq!(pe[c].to_bits(), delta.energy_delta.to_bits(), "energy at ({ii},{jj})");
            // the angular-momentum receipt: the z moment of the force receipt
            // about the body center, same helper, same bits.
            let x_rel = Tensor::new([x[0] - POS[0], x[1] - POS[1]]);
            let tq = symbi_ib::moment(&x_rel, &delta.force_delta);
            assert_eq!(pt[c].to_bits(), tq[2].to_bits(), "torque at ({ii},{jj})");
            // gate 5, per cell: the delta IS the gas's loss.
            assert_eq!(
                pm[c].to_bits(),
                ((host_in.0[c] - den[c]) * dv).to_bits(),
                "conservation at ({ii},{jj})",
            );
            if pm[c] != 0.0 {
                interior_nonzero += 1;
            }
        }
    }
    assert!(interior_nonzero > 20, "the drain never fired — the gate is vacuous");
}

// the ISOTHERMAL kernel: constant sound speed, no energy channel — the same
// bit-identity gate against the same carrier-generic chain at IsoModel.
#[test]
fn compiled_iso_drain_penalize_matches_the_f64_chain_bitwise() {
    use symbi_hydro::energy::IsoModel;
    let (kernel, ir) = kernel_by_name::<f64>("penalize_drain_iso_2d").expect("iso kernel");
    assert!(kernel_output_support_from_ir(ir).is_some());
    const CS: f64 = 0.8;

    let n2 = N * N;
    let (mut den, mut mx, mut my) = (vec![0.0; n2], vec![0.0; n2], vec![0.0; n2]);
    for jj in 0..N {
        for ii in 0..N {
            let (x, y) = (cell_center(ii), cell_center(jj));
            let c = ii + jj * N;
            den[c] = 1.0 + 0.2 * (2.0 * x).sin() * (1.5 * y).cos();
            mx[c] = den[c] * 0.3 * (x + y).cos();
            my[c] = den[c] * -0.2 * (x - 2.0 * y).sin();
        }
    }
    let host_in = (den.clone(), mx.clone(), my.clone());

    let scalar = |name: &str| -> f64 {
        match name {
            "dt" => DT,
            "cs" => CS,
            "c_drain" => C_DRAIN,
            "x_lo_0" | "x_lo_1" => X_LO,
            "dx_0" | "dx_1" => DX,
            "map_kind_0" | "map_kind_1" => 0.0,
            "body_0_pos_0" => POS[0],
            "body_0_pos_1" => POS[1],
            "body_0_racc" => RACC,
            other => panic!("unexpected scalar '{other}'"),
        }
    };
    let (mut ints, mut scalars) = (Vec::new(), Vec::new());
    for (bind, is_int) in symbi_ir::kernel_scalar_params_typed_from_ir(ir) {
        let v = scalar(&bind.name());
        if is_int { ints.push(v as i32) } else { scalars.push(v) }
    }
    let lo = [0i32; 2];
    let ext = [N as u32; 2];
    let mut pm = vec![7.7; n2];
    let mut pfx = vec![7.7; n2];
    let mut pfy = vec![7.7; n2];
    let mut pt = vec![7.7; n2];
    {
        let inputs = [
            CpuField::from_layout(&den, &lo, &ext),
            CpuField::from_layout(&mx, &lo, &ext),
            CpuField::from_layout(&my, &lo, &ext),
        ];
        let mut den_o = den.clone();
        let mut mx_o = mx.clone();
        let mut my_o = my.clone();
        let mut outs = [
            CpuFieldMut::from_layout(&mut den_o, &lo, &ext),
            CpuFieldMut::from_layout(&mut mx_o, &lo, &ext),
            CpuFieldMut::from_layout(&mut my_o, &lo, &ext),
            CpuFieldMut::from_layout(&mut pm, &lo, &ext),
            CpuFieldMut::from_layout(&mut pfx, &lo, &ext),
            CpuFieldMut::from_layout(&mut pfy, &lo, &ext),
            CpuFieldMut::from_layout(&mut pt, &lo, &ext),
        ];
        kernel(&inputs, &mut outs, &[N as u32; 2], &[0i32; 2], &ints, &scalars);
        drop(outs);
        den = den_o;
        mx = mx_o;
        my = my_o;
    }

    let sphere = SdfExpr::<f64, 2>::sphere(POS, RACC);
    let width = |i: usize| (X_LO + (i as f64 + 1.0) * DX) - (X_LO + i as f64 * DX);
    let mut fired = 0usize;
    for jj in 0..N {
        for ii in 0..N {
            let c = ii + jj * N;
            let cons = ConsG::<f64, 2, IsoModel> {
                den: host_in.0[c],
                mom: Tensor::new([host_in.1[c], host_in.2[c]]),
                nrg: Default::default(),
            };
            let mid = |i: usize| ((X_LO + i as f64 * DX) + (X_LO + (i as f64 + 1.0) * DX)) * 0.5;
            let ch = chi(sphere.dist([mid(ii), mid(jj)]), DX);
            let inv_tau = CS / (C_DRAIN * DX);
            let kin = BodyKin::<f64, 2> { u_solid: Tensor::zeros(), omega: Tensor::zeros(), e_wall: 0.0 };
            let mut acc = Relax::none();
            Property::Drain { inv_tau }.contribute(ch, &kin, &mut acc);
            let dv = 1.0 / (1.0 / (width(ii) * width(jj)));
            let (out, delta) = penalize_cell(&cons, &acc, Tensor::zeros(), DT, dv, 0);
            assert_eq!(den[c].to_bits(), out.den.to_bits(), "den at ({ii},{jj})");
            assert_eq!(mx[c].to_bits(), out.mom[0].to_bits(), "mom0 at ({ii},{jj})");
            assert_eq!(my[c].to_bits(), out.mom[1].to_bits(), "mom1 at ({ii},{jj})");
            assert_eq!(pm[c].to_bits(), delta.mass_delta.to_bits(), "mass at ({ii},{jj})");
            let x_rel = Tensor::new([mid(ii) - POS[0], mid(jj) - POS[1]]);
            let tq = symbi_ib::moment(&x_rel, &delta.force_delta);
            assert_eq!(pt[c].to_bits(), tq[2].to_bits(), "torque at ({ii},{jj})");
            if pm[c] != 0.0 {
                fired += 1;
            }
        }
    }
    assert!(fired > 20, "the iso drain never fired");
}
