// =============================================================================
// penalize_law.rs
//
// the [Drain] penalization kernel's gates: the
// compiled kernel is bit-identical to the f64 host chain built from the same
// carrier-generic functions (sphere SDF chi -> Relax -> penalize_cell), the
// per-cell deltas equal the gas's conserved loss exactly, and the blob
// declares its support ball.
// =============================================================================

use symbi_algebra::Tensor;
use symbi_aot::{CpuField, CpuFieldMut, kernel_by_name};
use symbi_hydro::energy::Adiabatic;
use symbi_hydro::state::ConsG;
use symbi_ib::penalize::{BodyKin, Property, Relax, penalize_cell};
use symbi_ib::sdf::{SdfExpr, chi};
use symbi_ir::kernel_output_support_from_ir;

const N: usize = 48;
const X_LO: f64 = -1.2;
const DX: f64 = 0.05;
const POS: [f64; 2] = [0.11, -0.07];
const RACC: f64 = 0.15;
const C_DRAIN: f64 = 1.0;
/// the accretor mass. chosen so the free-fall floor binds: with `RACC = 0.15` the free-fall rate
/// `sqrt(MASS/RACC^3)` is 34.4 against a sound-crossing rate `CS/(C_DRAIN*DX)` of about 20, so
/// these oracles exercise the floor arm rather than passing through the sound-crossing arm and
/// leaving the new path unchecked.
const MASS: f64 = 4.0;

/// the f64 twin of the kernel's `spherical_drain_rate`: the faster of the sound-crossing rate and
/// free fall at the mask radius. takes the radius explicitly -- the cases here use several.
fn drain_rate_floor(sound_rate: f64, racc: f64) -> f64 {
    sound_rate.max((MASS / (racc * racc * racc)).sqrt())
}
const DT: f64 = 0.008;
const GAMMA: f64 = 1.4;

fn cell_center(i: usize) -> f64 {
    X_LO + (i as f64 + 0.5) * DX
}

/// run a penalize kernel: `inputs` are the in-place cons reads (3 for iso: den,
/// mom0, mom1; 4 for adiabatic: + nrg). the output count (cons clones first,
/// then the delta scratch) comes off the artifact's buffer manifest, so a
/// receipt added to the kernels (torque, normal-force split) stays automatically
/// in sync with this harness. returns every output buffer.
fn run_pen(
    kern: symbi_aot::KernelFn<f64>,
    ir: &str,
    inputs: &[&[f64]],
    scalar: impl Fn(&str) -> f64,
) -> Vec<Vec<f64>> {
    let n_out = symbi_ir::kernel_bindings_from_ir(ir)
        .iter()
        .filter(|(_, is_output)| *is_output)
        .count();
    let n2 = inputs[0].len();
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
    let ext = [(n2 as f64).sqrt() as u32; 2];
    let in_fields: Vec<CpuField<f64>> = inputs
        .iter()
        .map(|b| CpuField::from_layout(b, &lo, &ext))
        .collect();
    let mut out: Vec<Vec<f64>> = (0..n_out)
        .map(|k| {
            if k < inputs.len() {
                inputs[k].to_vec()
            } else {
                vec![7.7; n2]
            }
        })
        .collect();
    {
        let mut out_fields: Vec<CpuFieldMut<f64>> = out
            .iter_mut()
            .map(|b| CpuFieldMut::from_layout(b, &lo, &ext))
            .collect();
        kern(&in_fields, &mut out_fields, &ext, &lo, &ints, &scalars);
    }
    out
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
            // hydro: zero alfven contribution reduces the magnetosonic signal
            // speed sqrt(cs^2 + c_a2) to the sound-speed form exactly.
            "c_a2" => 0.0,
            "x_lo_0" | "x_lo_1" => X_LO,
            "dx_0" | "dx_1" => DX,
            "map_kind_0" | "map_kind_1" | "map_param_0" | "map_param_1" => 0.0,
            "body_0_pos_0" => POS[0],
            "body_0_pos_1" => POS[1],
            "body_0_racc" => RACC,
            "body_0_mass" => MASS,
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
    let mut pm = vec![7.7; n2];
    let mut pfx = vec![7.7; n2];
    let mut pfy = vec![7.7; n2];
    let mut pe = vec![7.7; n2];
    let mut pt = vec![7.7; n2];
    // the normal-projected force receipt occupies the two trailing outputs
    let mut pfnx = vec![7.7; n2];
    let mut pfny = vec![7.7; n2];
    {
        // in-place cons: the same buffers appear as inputs and outputs.
        let inputs = [
            CpuField::from_layout(&den, &lo, &ext),
            CpuField::from_layout(&mx, &lo, &ext),
            CpuField::from_layout(&my, &lo, &ext),
            CpuField::from_layout(&nrg, &lo, &ext),
        ];
        // safety-free aliasing dance: run_parallel-style in-place needs the
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
            CpuFieldMut::from_layout(&mut pfnx, &lo, &ext),
            CpuFieldMut::from_layout(&mut pfny, &lo, &ext),
        ];
        kernel(
            &inputs,
            &mut outs,
            &[N as u32; 2],
            &[0i32; 2],
            &ints,
            &scalars,
        );
        drop(outs);
        den = den_o;
        mx = mx_o;
        my = my_o;
        nrg = nrg_o;
    }

    // the f64 host chain: the same carrier-generic functions per cell.
    let sphere = SdfExpr::<f64, 2>::sphere(POS, RACC);
    // the kernel's volume comes from the geometry scaffold: per-axis widths
    // as face differences (x_lo + (i+1)dx) - (x_lo + i dx), the product
    // reciprocated twice (dv = 1/inv_volume). mirror the exact chain.
    let width = |i: usize| (X_LO + (i as f64 + 1.0) * DX) - (X_LO + i as f64 * DX);
    let mut interior_nonzero = 0usize;
    for jj in 0..N {
        for ii in 0..N {
            let c = ii + jj * N;
            let cons = ConsG::<f64, 2, Adiabatic> {
                chi: Default::default(),
                den: host_in.0[c],
                mom: Tensor::new([host_in.1[c], host_in.2[c]]),
                nrg: host_in.3[c],
            };
            // the kernel's centroid is the arithmetic mid of the face
            // positions; the algebraically equal x_lo + (i+0.5)dx differs in the last
            // bit, so mirror the face-mid form.
            let mid = |i: usize| ((X_LO + i as f64 * DX) + (X_LO + (i as f64 + 1.0) * DX)) * 0.5;
            let x = [mid(ii), mid(jj)];
            let dv = 1.0 / (1.0 / (width(ii) * width(jj)));
            let ch = chi(sphere.dist(x), DX);
            let mom_sq = cons.mom.dot(&cons.mom);
            let cs = symbi_ib::drain::sound_speed_from_cons(cons.den, mom_sq, cons.nrg, GAMMA);
            let inv_tau = drain_rate_floor(cs / (C_DRAIN * DX), RACC);
            let kin = BodyKin::<f64, 2> {
                u_solid: Tensor::zeros(),
                omega: Tensor::zeros(),
                e_wall: 0.0,
            };
            let mut acc = Relax::none();
            Property::Drain { inv_tau }.contribute(ch, &kin, &mut acc);
            let (out, delta) = penalize_cell(&cons, &acc, Tensor::zeros(), DT, dv, 0);

            assert_eq!(den[c].to_bits(), out.den.to_bits(), "den at ({ii},{jj})");
            assert_eq!(mx[c].to_bits(), out.mom[0].to_bits(), "mom0 at ({ii},{jj})");
            assert_eq!(my[c].to_bits(), out.mom[1].to_bits(), "mom1 at ({ii},{jj})");
            assert_eq!(nrg[c].to_bits(), out.nrg.to_bits(), "nrg at ({ii},{jj})");
            assert_eq!(
                pm[c].to_bits(),
                delta.mass_delta.to_bits(),
                "mass at ({ii},{jj})"
            );
            assert_eq!(
                pfx[c].to_bits(),
                delta.force_delta[0].to_bits(),
                "fx at ({ii},{jj})"
            );
            assert_eq!(
                pe[c].to_bits(),
                delta.energy_delta.to_bits(),
                "energy at ({ii},{jj})"
            );
            // the angular-momentum receipt: the z moment of the force receipt
            // about the body center, same helper, same bits.
            let x_rel = Tensor::new([x[0] - POS[0], x[1] - POS[1]]);
            let tq = symbi_ib::moment(&x_rel, &delta.force_delta);
            assert_eq!(pt[c].to_bits(), tq[2].to_bits(), "torque at ({ii},{jj})");
            // per cell: the delta equals the gas's loss.
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
    assert!(
        interior_nonzero > 20,
        "the drain never fired — the gate is vacuous"
    );
}

// the isothermal kernel: constant sound speed, no energy channel — the same
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
            "map_kind_0" | "map_kind_1" | "map_param_0" | "map_param_1" => 0.0,
            "body_0_pos_0" => POS[0],
            "body_0_pos_1" => POS[1],
            "body_0_racc" => RACC,
            "body_0_mass" => MASS,
            "c_a2" => 0.0,
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
    let mut pm = vec![7.7; n2];
    let mut pfx = vec![7.7; n2];
    let mut pfy = vec![7.7; n2];
    let mut pt = vec![7.7; n2];
    // the normal-projected force receipt occupies the two trailing outputs
    let mut pfnx = vec![7.7; n2];
    let mut pfny = vec![7.7; n2];
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
            CpuFieldMut::from_layout(&mut pfnx, &lo, &ext),
            CpuFieldMut::from_layout(&mut pfny, &lo, &ext),
        ];
        kernel(
            &inputs,
            &mut outs,
            &[N as u32; 2],
            &[0i32; 2],
            &ints,
            &scalars,
        );
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
                chi: Default::default(),
                den: host_in.0[c],
                mom: Tensor::new([host_in.1[c], host_in.2[c]]),
                nrg: Default::default(),
            };
            let mid = |i: usize| ((X_LO + i as f64 * DX) + (X_LO + (i as f64 + 1.0) * DX)) * 0.5;
            let ch = chi(sphere.dist([mid(ii), mid(jj)]), DX);
            let inv_tau = drain_rate_floor(CS / (C_DRAIN * DX), RACC);
            let kin = BodyKin::<f64, 2> {
                u_solid: Tensor::zeros(),
                omega: Tensor::zeros(),
                e_wall: 0.0,
            };
            let mut acc = Relax::none();
            Property::Drain { inv_tau }.contribute(ch, &kin, &mut acc);
            let dv = 1.0 / (1.0 / (width(ii) * width(jj)));
            let (out, delta) = penalize_cell(&cons, &acc, Tensor::zeros(), DT, dv, 0);
            assert_eq!(den[c].to_bits(), out.den.to_bits(), "den at ({ii},{jj})");
            assert_eq!(mx[c].to_bits(), out.mom[0].to_bits(), "mom0 at ({ii},{jj})");
            assert_eq!(my[c].to_bits(), out.mom[1].to_bits(), "mom1 at ({ii},{jj})");
            assert_eq!(
                pm[c].to_bits(),
                delta.mass_delta.to_bits(),
                "mass at ({ii},{jj})"
            );
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

// the cylindrical (R, phi) iso drain gate: the mask distance maps the coordinate
// centroid to Cartesian, so the mask is a coordinate ball (a cylinder about the
// axis) and the cell volume is the curvilinear int r dr dphi. bit-identical to the
// f64 chain built from cell_geometry's cylindrical formulas + the CylindricalRPhi
// map. a body on the axis (R = 0 == Cartesian origin) is the central-accretor case.
#[test]
fn compiled_iso_drain_penalize_cylindrical_matches_the_f64_chain_bitwise() {
    use symbi_geometry::{CylindricalRPhi, Metric};
    use symbi_hydro::energy::IsoModel;
    let (kernel, ir) = kernel_by_name::<f64>("penalize_drain_iso_cyl_2d").expect("cyl kernel");
    // a cylindrical kernel dispatches over the whole interior: the cartesian mask
    // region forms a coordinate ball only on the identity chart, so the support
    // derivation declines off it.
    assert!(kernel_output_support_from_ir(ir).is_none());

    const CS: f64 = 0.8;
    const R_LO: f64 = 0.0;
    const DR: f64 = 0.05;
    const PHI_LO: f64 = 0.0;
    const DPHI: f64 = 0.2;
    const CENTER: [f64; 2] = [0.0, 0.0]; // Cartesian body position (on the axis)
    const RACC_C: f64 = 0.15;

    let n2 = N * N;
    let (mut den, mut mx, mut my) = (vec![0.0; n2], vec![0.0; n2], vec![0.0; n2]);
    for jj in 0..N {
        for ii in 0..N {
            let c = ii + jj * N;
            let (rr, pp) = (
                R_LO + (ii as f64 + 0.5) * DR,
                PHI_LO + (jj as f64 + 0.5) * DPHI,
            );
            den[c] = 1.0 + 0.2 * (2.0 * rr).sin() * (1.5 * pp).cos();
            mx[c] = den[c] * 0.3 * (rr + pp).cos();
            my[c] = den[c] * -0.2 * (rr - 2.0 * pp).sin();
        }
    }
    let host_in = (den.clone(), mx.clone(), my.clone());

    let scalar = |name: &str| -> f64 {
        match name {
            "dt" => DT,
            "cs" => CS,
            "c_drain" => C_DRAIN,
            "x_lo_0" => R_LO,
            "x_lo_1" => PHI_LO,
            "dx_0" => DR,
            "dx_1" => DPHI,
            "map_kind_0" | "map_kind_1" | "map_param_0" | "map_param_1" => 0.0,
            "body_0_pos_0" => CENTER[0],
            "body_0_pos_1" => CENTER[1],
            "body_0_racc" => RACC_C,
                "body_0_mass" => MASS,
            "c_a2" => 0.0,
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
    let (mut pm, mut pfx, mut pfy, mut pt) =
        (vec![7.7; n2], vec![7.7; n2], vec![7.7; n2], vec![7.7; n2]);
    // the normal-projected force receipt occupies the two trailing outputs
    let (mut pfnx, mut pfny) = (vec![7.7; n2], vec![7.7; n2]);
    {
        let inputs = [
            CpuField::from_layout(&den, &lo, &ext),
            CpuField::from_layout(&mx, &lo, &ext),
            CpuField::from_layout(&my, &lo, &ext),
        ];
        let (mut den_o, mut mx_o, mut my_o) = (den.clone(), mx.clone(), my.clone());
        let mut outs = [
            CpuFieldMut::from_layout(&mut den_o, &lo, &ext),
            CpuFieldMut::from_layout(&mut mx_o, &lo, &ext),
            CpuFieldMut::from_layout(&mut my_o, &lo, &ext),
            CpuFieldMut::from_layout(&mut pm, &lo, &ext),
            CpuFieldMut::from_layout(&mut pfx, &lo, &ext),
            CpuFieldMut::from_layout(&mut pfy, &lo, &ext),
            CpuFieldMut::from_layout(&mut pt, &lo, &ext),
            CpuFieldMut::from_layout(&mut pfnx, &lo, &ext),
            CpuFieldMut::from_layout(&mut pfny, &lo, &ext),
        ];
        kernel(
            &inputs,
            &mut outs,
            &[N as u32; 2],
            &[0i32; 2],
            &ints,
            &scalars,
        );
        drop(outs);
        den = den_o;
        mx = mx_o;
        my = my_o;
    }

    // the f64 chain: the cylindrical cell geometry (volume-weighted radial centroid
    // c_R = (2/3)(rh^3-rl^3)/(rh^2-rl^2), volume int r dr dphi) + the CylindricalRPhi
    // map, mirrored to the bit (gv_powi is repeated multiplication; dv is the double
    // reciprocal of inv_volume).
    let sphere = SdfExpr::<f64, 2>::sphere(CENTER, RACC_C);
    let metric = CylindricalRPhi;
    let rl = |i: usize| R_LO + i as f64 * DR;
    let rh = |i: usize| R_LO + (i as f64 + 1.0) * DR;
    let ir2 = |i: usize| (rh(i) * rh(i) - rl(i) * rl(i)) / 2.0;
    let cnum = |i: usize| (rh(i) * rh(i) * rh(i) - rl(i) * rl(i) * rl(i)) / 3.0;
    let cr = |i: usize| cnum(i) / ir2(i);
    let cphi = |j: usize| ((PHI_LO + j as f64 * DPHI) + (PHI_LO + (j as f64 + 1.0) * DPHI)) * 0.5;
    // the phi extent is the face difference (phi_hi - phi_lo), which rounds
    // differently from dphi itself — the cell volume int r dr dphi uses it.
    let iphi = |j: usize| (PHI_LO + (j as f64 + 1.0) * DPHI) - (PHI_LO + j as f64 * DPHI);
    let min_w = DR.min(DPHI);
    let mut fired = 0usize;
    for jj in 0..N {
        for ii in 0..N {
            let c = ii + jj * N;
            let cons = ConsG::<f64, 2, IsoModel> {
                chi: Default::default(),
                den: host_in.0[c],
                mom: Tensor::new([host_in.1[c], host_in.2[c]]),
                nrg: Default::default(),
            };
            let xc = metric.to_cartesian(Tensor::new([cr(ii), cphi(jj)]));
            let ch = chi(sphere.dist([xc[0], xc[1]]), min_w);
            let inv_tau = drain_rate_floor(CS / (C_DRAIN * min_w), RACC_C);
            let kin = BodyKin::<f64, 2> {
                u_solid: Tensor::zeros(),
                omega: Tensor::zeros(),
                e_wall: 0.0,
            };
            let mut acc = Relax::none();
            Property::Drain { inv_tau }.contribute(ch, &kin, &mut acc);
            let dv = 1.0 / (1.0 / (ir2(ii) * iphi(jj) * 1.0));
            let (out, delta) = penalize_cell(&cons, &acc, Tensor::zeros(), DT, dv, 0);
            assert_eq!(den[c].to_bits(), out.den.to_bits(), "den at ({ii},{jj})");
            assert_eq!(mx[c].to_bits(), out.mom[0].to_bits(), "mom0 at ({ii},{jj})");
            assert_eq!(my[c].to_bits(), out.mom[1].to_bits(), "mom1 at ({ii},{jj})");
            assert_eq!(
                pm[c].to_bits(),
                delta.mass_delta.to_bits(),
                "mass at ({ii},{jj})"
            );
            // lab-frame torque: rotate the physical force to Cartesian, cross with r_cart.
            let e = metric.vector_to_cartesian(
                Tensor::new([cr(ii), cphi(jj)]),
                symbi_algebra::Physical::new(delta.force_delta),
            );
            let f_cart = Tensor::new([e[0], e[1]]);
            let x_rel = Tensor::new([xc[0] - CENTER[0], xc[1] - CENTER[1]]);
            let tq = symbi_ib::moment(&x_rel, &f_cart);
            assert_eq!(pt[c].to_bits(), tq[2].to_bits(), "torque at ({ii},{jj})");
            if pm[c] != 0.0 {
                fired += 1;
            }
        }
    }
    assert!(
        fired > 20,
        "the cylindrical drain never fired — the gate is vacuous"
    );
}

// the cylindrical (R, phi) torque-free accretor gate: the surface normal is
// rotated into the physical frame (e_R for a centered accretor, so tangential ==
// phi == the angular-momentum direction), and the torque is the lab-frame
// r_cart x F_cart. bit-identical to the f64 chain, and xi = 0 reduces to the
// cylindrical iso drain kernel bit-for-bit.
#[test]
fn compiled_torque_free_penalize_cylindrical_matches_and_reduces_at_xi0() {
    use symbi_geometry::{CylindricalRPhi, Metric};
    use symbi_hydro::energy::IsoModel;
    let (tf, ir) =
        kernel_by_name::<f64>("penalize_torque_free_iso_cyl_2d").expect("cyl torque-free kernel");
    // a cylindrical kernel dispatches over the whole interior: the cartesian mask
    // region forms a coordinate ball only on the identity chart, so the support
    // derivation declines off it.
    assert!(kernel_output_support_from_ir(ir).is_none());
    let (drain, drain_ir) =
        kernel_by_name::<f64>("penalize_drain_iso_cyl_2d").expect("cyl drain kernel");

    const CS: f64 = 0.8;
    const R_LO: f64 = 0.0;
    const DR: f64 = 0.05;
    const PHI_LO: f64 = 0.0;
    const DPHI: f64 = 0.2;
    const CENTER: [f64; 2] = [0.0, 0.0]; // centered accretor on the axis
    const RACC_C: f64 = 0.15;
    const VEL: [f64; 2] = [0.0, 0.0];

    let n2 = N * N;
    let (mut den0, mut mx0, mut my0) = (vec![0.0; n2], vec![0.0; n2], vec![0.0; n2]);
    for jj in 0..N {
        for ii in 0..N {
            let c = ii + jj * N;
            let (rr, pp) = (
                R_LO + (ii as f64 + 0.5) * DR,
                PHI_LO + (jj as f64 + 0.5) * DPHI,
            );
            den0[c] = 1.0 + 0.2 * (2.0 * rr).sin() * (1.5 * pp).cos();
            mx0[c] = den0[c] * 0.3 * (rr + pp).cos();
            my0[c] = den0[c] * -0.2 * (rr - 2.0 * pp).sin();
        }
    }

    #[allow(clippy::type_complexity)]
    let run = |kern: symbi_aot::KernelFn<f64>,
               kern_ir,
               xi_val: f64|
     -> (
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
    ) {
        let scalar = |name: &str| -> f64 {
            match name {
                "dt" => DT,
                "cs" => CS,
                "c_drain" => C_DRAIN,
                "xi" => xi_val,
                "x_lo_0" => R_LO,
                "x_lo_1" => PHI_LO,
                "dx_0" => DR,
                "dx_1" => DPHI,
                "map_kind_0" | "map_kind_1" | "map_param_0" | "map_param_1" => 0.0,
                "body_0_pos_0" => CENTER[0],
                "body_0_pos_1" => CENTER[1],
                "body_0_racc" => RACC_C,
                "body_0_mass" => MASS,
                "body_0_vel_0" => VEL[0],
                "body_0_vel_1" => VEL[1],
                "c_a2" => 0.0,
                other => panic!("unexpected scalar '{other}'"),
            }
        };
        let (mut ints, mut scalars) = (Vec::new(), Vec::new());
        for (bind, is_int) in symbi_ir::kernel_scalar_params_typed_from_ir(kern_ir) {
            let v = scalar(&bind.name());
            if is_int {
                ints.push(v as i32)
            } else {
                scalars.push(v)
            }
        }
        let lo = [0i32; 2];
        let ext = [N as u32; 2];
        let inputs = [
            CpuField::from_layout(&den0, &lo, &ext),
            CpuField::from_layout(&mx0, &lo, &ext),
            CpuField::from_layout(&my0, &lo, &ext),
        ];
        let (mut d, mut mx, mut my) = (den0.clone(), mx0.clone(), my0.clone());
        let (mut pm, mut pfx, mut pfy, mut pt) =
            (vec![7.7; n2], vec![7.7; n2], vec![7.7; n2], vec![7.7; n2]);
        // the normal-projected force receipt occupies the two trailing outputs
        let (mut pfnx, mut pfny) = (vec![7.7; n2], vec![7.7; n2]);
        {
            let mut outs = [
                CpuFieldMut::from_layout(&mut d, &lo, &ext),
                CpuFieldMut::from_layout(&mut mx, &lo, &ext),
                CpuFieldMut::from_layout(&mut my, &lo, &ext),
                CpuFieldMut::from_layout(&mut pm, &lo, &ext),
                CpuFieldMut::from_layout(&mut pfx, &lo, &ext),
                CpuFieldMut::from_layout(&mut pfy, &lo, &ext),
                CpuFieldMut::from_layout(&mut pt, &lo, &ext),
                CpuFieldMut::from_layout(&mut pfnx, &lo, &ext),
                CpuFieldMut::from_layout(&mut pfny, &lo, &ext),
            ];
            kern(
                &inputs,
                &mut outs,
                &[N as u32; 2],
                &[0i32; 2],
                &ints,
                &scalars,
            );
        }
        (d, mx, my, pm, pfx, pfy, pt)
    };

    // (1) bit-identity against the f64 chain at xi = 0.7.
    const XI: f64 = 0.7;
    let out = run(tf, ir, XI);
    let metric = CylindricalRPhi;
    let sphere = SdfExpr::<f64, 2>::sphere(CENTER, RACC_C);
    let rl = |i: usize| R_LO + i as f64 * DR;
    let rh = |i: usize| R_LO + (i as f64 + 1.0) * DR;
    let ir2 = |i: usize| (rh(i) * rh(i) - rl(i) * rl(i)) / 2.0;
    let cnum = |i: usize| (rh(i) * rh(i) * rh(i) - rl(i) * rl(i) * rl(i)) / 3.0;
    let cr = |i: usize| cnum(i) / ir2(i);
    let cphi = |j: usize| ((PHI_LO + j as f64 * DPHI) + (PHI_LO + (j as f64 + 1.0) * DPHI)) * 0.5;
    let iphi = |j: usize| (PHI_LO + (j as f64 + 1.0) * DPHI) - (PHI_LO + j as f64 * DPHI);
    let min_w = DR.min(DPHI);
    let mut fired = 0usize;
    for jj in 0..N {
        for ii in 0..N {
            let c = ii + jj * N;
            let cons = ConsG::<f64, 2, IsoModel> {
                chi: Default::default(),
                den: den0[c],
                mom: Tensor::new([mx0[c], my0[c]]),
                nrg: Default::default(),
            };
            let xc = metric.to_cartesian(Tensor::new([cr(ii), cphi(jj)]));
            let ch = chi(sphere.dist([xc[0], xc[1]]), min_w);
            let inv_tau = drain_rate_floor(CS / (C_DRAIN * min_w), RACC_C);
            // the physical-frame normal: Cartesian r_hat rotated into the ortho basis.
            let xrel = [xc[0] - CENTER[0], xc[1] - CENTER[1]];
            let r = (xrel[0] * xrel[0] + xrel[1] * xrel[1]).sqrt();
            let invr = 1.0 / r.max(1e-300);
            let nphys = metric.vector_from_cartesian(
                Tensor::new([cr(ii), cphi(jj)]),
                symbi_algebra::Embedded::new(Tensor::new([xrel[0] * invr, xrel[1] * invr])),
            );
            let normal = Tensor::new([nphys[0], nphys[1]]);
            // the body's translational velocity is a cartesian world vector,
            // rotated into the cell's physical basis before the tangential
            // split — the same transform the kernel traces.
            let us = metric.vector_from_cartesian(
                Tensor::new([cr(ii), cphi(jj)]),
                symbi_algebra::Embedded::new(Tensor::new(VEL)),
            );
            let kin = BodyKin::<f64, 2> {
                u_solid: Tensor::new([us[0], us[1]]),
                omega: Tensor::zeros(),
                e_wall: 0.0,
            };
            let mut acc = Relax::none();
            Property::TorqueFreeAccretor { inv_tau, xi: XI }.contribute(ch, &kin, &mut acc);
            let dv = 1.0 / (1.0 / (ir2(ii) * iphi(jj) * 1.0));
            let (expect, delta) = penalize_cell(&cons, &acc, normal, DT, dv, 0);
            assert_eq!(
                out.0[c].to_bits(),
                expect.den.to_bits(),
                "den at ({ii},{jj})"
            );
            assert_eq!(
                out.1[c].to_bits(),
                expect.mom[0].to_bits(),
                "mom0 at ({ii},{jj})"
            );
            assert_eq!(
                out.2[c].to_bits(),
                expect.mom[1].to_bits(),
                "mom1 at ({ii},{jj})"
            );
            assert_eq!(
                out.3[c].to_bits(),
                delta.mass_delta.to_bits(),
                "mass at ({ii},{jj})"
            );
            // the force receipt is booked in the cartesian world frame: local
            // physical components rotate cell to cell, so only the cartesian-frame
            // sum represents the net force on the body.
            let e = metric.vector_to_cartesian(
                Tensor::new([cr(ii), cphi(jj)]),
                symbi_algebra::Physical::new(delta.force_delta),
            );
            assert_eq!(out.4[c].to_bits(), e[0].to_bits(), "fx at ({ii},{jj})");
            assert_eq!(out.5[c].to_bits(), e[1].to_bits(), "fy at ({ii},{jj})");
            let tq = symbi_ib::moment(&Tensor::new([xrel[0], xrel[1]]), &Tensor::new([e[0], e[1]]));
            assert_eq!(out.6[c].to_bits(), tq[2].to_bits(), "torque at ({ii},{jj})");
            if out.3[c] != 0.0 {
                fired += 1;
            }
        }
    }
    assert!(fired > 20, "the cylindrical torque-free drain never fired");

    // (2) xi = 0: the torque-free kernel equals the cylindrical iso drain, bit for bit.
    let tf0 = run(tf, ir, 0.0);
    let dr = run(drain, drain_ir, 0.0);
    for c in 0..n2 {
        assert_eq!(tf0.0[c].to_bits(), dr.0[c].to_bits(), "xi=0 den at {c}");
        assert_eq!(tf0.1[c].to_bits(), dr.1[c].to_bits(), "xi=0 mom0 at {c}");
        assert_eq!(tf0.2[c].to_bits(), dr.2[c].to_bits(), "xi=0 mom1 at {c}");
        assert_eq!(tf0.3[c].to_bits(), dr.3[c].to_bits(), "xi=0 mass at {c}");
        assert_eq!(tf0.6[c].to_bits(), dr.6[c].to_bits(), "xi=0 torque at {c}");
    }
}

// the isothermal torque-free accretor kernel. the compiled
// kernel is bit-identical to the f64 chain (guarded sphere normal ->
// TorqueFreeAccretor contribute + retention floor -> penalize_cell at IsoModel),
// and xi = 0 reduces to the iso drain kernel bit-for-bit (the tangential
// coupling vanishes as an exact zero — the drain-reduction anchor).
#[test]
fn compiled_iso_torque_free_penalize_matches_the_f64_chain_and_reduces_at_xi0() {
    use symbi_hydro::energy::IsoModel;
    let (tf, ir) =
        kernel_by_name::<f64>("penalize_torque_free_iso_2d").expect("torque-free kernel");
    assert!(kernel_output_support_from_ir(ir).is_some());
    let (drain, drain_ir) =
        kernel_by_name::<f64>("penalize_drain_iso_2d").expect("iso drain kernel");

    const CS: f64 = 0.8;
    const VEL: [f64; 2] = [0.05, -0.03];

    let n2 = N * N;
    let (mut den0, mut mx0, mut my0) = (vec![0.0; n2], vec![0.0; n2], vec![0.0; n2]);
    for jj in 0..N {
        for ii in 0..N {
            let (x, y) = (cell_center(ii), cell_center(jj));
            let c = ii + jj * N;
            den0[c] = 1.0 + 0.2 * (2.0 * x).sin() * (1.5 * y).cos();
            mx0[c] = den0[c] * 0.3 * (x + y).cos();
            my0[c] = den0[c] * -0.2 * (x - 2.0 * y).sin();
        }
    }

    // one scalar closure (a superset) serves both kernels: the iso drain pulls
    // only its subset (no xi / vel).
    #[allow(clippy::type_complexity)]
    let run = |kern: symbi_aot::KernelFn<f64>,
               kern_ir,
               xi_val: f64|
     -> (
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
    ) {
        let scalar = |name: &str| -> f64 {
            match name {
                "dt" => DT,
                "cs" => CS,
                "c_drain" => C_DRAIN,
                "xi" => xi_val,
                "x_lo_0" | "x_lo_1" => X_LO,
                "dx_0" | "dx_1" => DX,
                "map_kind_0" | "map_kind_1" | "map_param_0" | "map_param_1" => 0.0,
                "body_0_pos_0" => POS[0],
                "body_0_pos_1" => POS[1],
                "body_0_racc" => RACC,
            "body_0_mass" => MASS,
                "body_0_vel_0" => VEL[0],
                "body_0_vel_1" => VEL[1],
                "c_a2" => 0.0,
                other => panic!("unexpected scalar '{other}'"),
            }
        };
        let (mut ints, mut scalars) = (Vec::new(), Vec::new());
        for (bind, is_int) in symbi_ir::kernel_scalar_params_typed_from_ir(kern_ir) {
            let v = scalar(&bind.name());
            if is_int {
                ints.push(v as i32)
            } else {
                scalars.push(v)
            }
        }
        let lo = [0i32; 2];
        let ext = [N as u32; 2];
        let inputs = [
            CpuField::from_layout(&den0, &lo, &ext),
            CpuField::from_layout(&mx0, &lo, &ext),
            CpuField::from_layout(&my0, &lo, &ext),
        ];
        let (mut d, mut mx, mut my) = (den0.clone(), mx0.clone(), my0.clone());
        let (mut pm, mut pfx, mut pfy, mut pt) =
            (vec![7.7; n2], vec![7.7; n2], vec![7.7; n2], vec![7.7; n2]);
        // the normal-projected force receipt occupies the two trailing outputs
        let (mut pfnx, mut pfny) = (vec![7.7; n2], vec![7.7; n2]);
        {
            let mut outs = [
                CpuFieldMut::from_layout(&mut d, &lo, &ext),
                CpuFieldMut::from_layout(&mut mx, &lo, &ext),
                CpuFieldMut::from_layout(&mut my, &lo, &ext),
                CpuFieldMut::from_layout(&mut pm, &lo, &ext),
                CpuFieldMut::from_layout(&mut pfx, &lo, &ext),
                CpuFieldMut::from_layout(&mut pfy, &lo, &ext),
                CpuFieldMut::from_layout(&mut pt, &lo, &ext),
                CpuFieldMut::from_layout(&mut pfnx, &lo, &ext),
                CpuFieldMut::from_layout(&mut pfny, &lo, &ext),
            ];
            kern(
                &inputs,
                &mut outs,
                &[N as u32; 2],
                &[0i32; 2],
                &ints,
                &scalars,
            );
        }
        (d, mx, my, pm, pfx, pfy, pt)
    };

    // (1) bit-identity against the f64 chain at a partial dial xi = 0.7.
    const XI: f64 = 0.7;
    let out = run(tf, ir, XI);
    let sphere = SdfExpr::<f64, 2>::sphere(POS, RACC);
    let mid = |i: usize| ((X_LO + i as f64 * DX) + (X_LO + (i as f64 + 1.0) * DX)) * 0.5;
    let width = |i: usize| (X_LO + (i as f64 + 1.0) * DX) - (X_LO + i as f64 * DX);
    let mut fired = 0usize;
    for jj in 0..N {
        for ii in 0..N {
            let c = ii + jj * N;
            let cons = ConsG::<f64, 2, IsoModel> {
                chi: Default::default(),
                den: den0[c],
                mom: Tensor::new([mx0[c], my0[c]]),
                nrg: Default::default(),
            };
            let x = [mid(ii), mid(jj)];
            let ch = chi(sphere.dist(x), DX);
            let inv_tau = drain_rate_floor(CS / (C_DRAIN * DX), RACC);
            let x_rel = Tensor::new([x[0] - POS[0], x[1] - POS[1]]);
            let r = x_rel.dot(&x_rel).sqrt();
            let normal = x_rel.scale(1.0 / r.max(1e-300));
            let kin = BodyKin::<f64, 2> {
                u_solid: Tensor::new(VEL),
                omega: Tensor::zeros(),
                e_wall: 0.0,
            };
            let mut acc = Relax::none();
            Property::TorqueFreeAccretor { inv_tau, xi: XI }.contribute(ch, &kin, &mut acc);
            let dv = width(ii) * width(jj);
            let (expect, delta) = penalize_cell(&cons, &acc, normal, DT, dv, 0);
            assert_eq!(
                out.0[c].to_bits(),
                expect.den.to_bits(),
                "den at ({ii},{jj})"
            );
            assert_eq!(
                out.1[c].to_bits(),
                expect.mom[0].to_bits(),
                "mom0 at ({ii},{jj})"
            );
            assert_eq!(
                out.2[c].to_bits(),
                expect.mom[1].to_bits(),
                "mom1 at ({ii},{jj})"
            );
            assert_eq!(
                out.3[c].to_bits(),
                delta.mass_delta.to_bits(),
                "mass at ({ii},{jj})"
            );
            assert_eq!(
                out.4[c].to_bits(),
                delta.force_delta[0].to_bits(),
                "fx at ({ii},{jj})"
            );
            assert_eq!(
                out.5[c].to_bits(),
                delta.force_delta[1].to_bits(),
                "fy at ({ii},{jj})"
            );
            let tq = symbi_ib::moment(&x_rel, &delta.force_delta);
            assert_eq!(out.6[c].to_bits(), tq[2].to_bits(), "torque at ({ii},{jj})");
            if out.3[c] != 0.0 {
                fired += 1;
            }
        }
    }
    assert!(fired > 20, "the torque-free drain never fired");

    // (2) xi = 0: the torque-free kernel equals the iso drain kernel, bit for bit.
    let tf0 = run(tf, ir, 0.0);
    let dr = run(drain, drain_ir, 0.0);
    for c in 0..n2 {
        assert_eq!(tf0.0[c].to_bits(), dr.0[c].to_bits(), "xi=0 den at {c}");
        assert_eq!(tf0.1[c].to_bits(), dr.1[c].to_bits(), "xi=0 mom0 at {c}");
        assert_eq!(tf0.2[c].to_bits(), dr.2[c].to_bits(), "xi=0 mom1 at {c}");
        assert_eq!(tf0.3[c].to_bits(), dr.3[c].to_bits(), "xi=0 mass at {c}");
        assert_eq!(tf0.6[c].to_bits(), dr.6[c].to_bits(), "xi=0 torque at {c}");
    }
}

// the [PorousAccretor] kernel: the compiled kernel is
// bit-identical to the f64 chain (guarded sphere normal -> PorousAccretor
// contribute -> penalize_cell), free-slip (k_eta_t = 0) leaves the tangential
// velocity bit-untouched through the compiled path, and porosity = 1 reduces
// the porous kernel to the drain kernel bit-for-bit on the same inputs.
#[test]
fn compiled_porous_penalize_matches_the_f64_chain_and_reduces_at_p1() {
    let (porous, ir) = kernel_by_name::<f64>("penalize_porous_2d").expect("porous kernel");
    assert!(kernel_output_support_from_ir(ir).is_some());
    let (drain, drain_ir) = kernel_by_name::<f64>("penalize_drain_2d").expect("drain kernel");

    const P: f64 = 0.4;
    const K_ETA_N: f64 = 30.0;
    const K_ETA_T: f64 = 0.0; // free-slip
    const VEL: [f64; 2] = [0.05, -0.03];

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
            nrg[c] = 2.0 / (GAMMA - 1.0) + 0.5 * (mx[c] * mx[c] + my[c] * my[c]) / den[c];
        }
    }
    let host_in = (den.clone(), mx.clone(), my.clone(), nrg.clone());

    #[allow(clippy::type_complexity)]
    let run = |kern: symbi_aot::KernelFn<f64>,
               kern_ir,
               p_val: f64|
     -> (
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
    ) {
        let scalar = |name: &str| -> f64 {
            match name {
                "dt" => DT,
                "gamma" => GAMMA,
                "c_drain" => C_DRAIN,
                "porosity" => p_val,
                "k_eta_n" => K_ETA_N,
                "k_eta_t" => K_ETA_T,
                "x_lo_0" | "x_lo_1" => X_LO,
                "dx_0" | "dx_1" => DX,
                "map_kind_0" | "map_kind_1" | "map_param_0" | "map_param_1" => 0.0,
                "body_0_pos_0" => POS[0],
                "body_0_pos_1" => POS[1],
                "body_0_racc" => RACC,
            "body_0_mass" => MASS,
                "body_0_vel_0" => VEL[0],
                "body_0_vel_1" => VEL[1],
                "c_a2" => 0.0,
                other => panic!("unexpected scalar '{other}'"),
            }
        };
        let (mut ints, mut scalars) = (Vec::new(), Vec::new());
        for (bind, is_int) in symbi_ir::kernel_scalar_params_typed_from_ir(kern_ir) {
            let v = scalar(&bind.name());
            if is_int {
                ints.push(v as i32)
            } else {
                scalars.push(v)
            }
        }
        let lo = [0i32; 2];
        let ext = [N as u32; 2];
        let mut pm = vec![7.7; n2];
        let mut pfx = vec![7.7; n2];
        let mut pfy = vec![7.7; n2];
        let mut pe = vec![7.7; n2];
        let mut pt = vec![7.7; n2];
        // the normal-projected force receipt occupies the two trailing outputs
        let mut pfnx = vec![7.7; n2];
        let mut pfny = vec![7.7; n2];
        let mut den_o = host_in.0.clone();
        let mut mx_o = host_in.1.clone();
        let mut my_o = host_in.2.clone();
        let mut nrg_o = host_in.3.clone();
        {
            let inputs = [
                CpuField::from_layout(&host_in.0, &lo, &ext),
                CpuField::from_layout(&host_in.1, &lo, &ext),
                CpuField::from_layout(&host_in.2, &lo, &ext),
                CpuField::from_layout(&host_in.3, &lo, &ext),
            ];
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
                CpuFieldMut::from_layout(&mut pfnx, &lo, &ext),
                CpuFieldMut::from_layout(&mut pfny, &lo, &ext),
            ];
            kern(
                &inputs,
                &mut outs,
                &[N as u32; 2],
                &[0i32; 2],
                &ints,
                &scalars,
            );
        }
        (den_o, mx_o, my_o, nrg_o, pm, pfx, pfy, pe, pt)
    };

    let out = run(porous, ir, P);

    // the f64 chain, mirrored to the bit (face-mid centroid, double-reciprocal
    // volume, guarded normal).
    let sphere = SdfExpr::<f64, 2>::sphere(POS, RACC);
    let width = |i: usize| (X_LO + (i as f64 + 1.0) * DX) - (X_LO + i as f64 * DX);
    let mid = |i: usize| ((X_LO + i as f64 * DX) + (X_LO + (i as f64 + 1.0) * DX)) * 0.5;
    let mut fired = 0usize;
    let mut tangential_checked = 0usize;
    for jj in 0..N {
        for ii in 0..N {
            let c = ii + jj * N;
            let cons = ConsG::<f64, 2, Adiabatic> {
                chi: Default::default(),
                den: host_in.0[c],
                mom: Tensor::new([host_in.1[c], host_in.2[c]]),
                nrg: host_in.3[c],
            };
            let x = [mid(ii), mid(jj)];
            let dv = 1.0 / (1.0 / (width(ii) * width(jj)));
            let ch = chi(sphere.dist(x), DX);
            let mom_sq = cons.mom.dot(&cons.mom);
            let cs = symbi_ib::drain::sound_speed_from_cons(cons.den, mom_sq, cons.nrg, GAMMA);
            let inv_tau = drain_rate_floor(cs / (C_DRAIN * DX), RACC);
            let rate_scale = cs / DX;
            let x_rel = Tensor::new([x[0] - POS[0], x[1] - POS[1]]);
            let r = x_rel.dot(&x_rel).sqrt();
            let normal = x_rel.scale(1.0 / r.max(1e-300));
            let kin = BodyKin::<f64, 2> {
                u_solid: Tensor::new(VEL),
                omega: Tensor::zeros(),
                e_wall: 0.0,
            };
            let mut acc = Relax::none();
            Property::PorousAccretor {
                p: P,
                inv_tau,
                inv_eta_n: K_ETA_N * rate_scale,
                inv_eta_t: K_ETA_T * rate_scale,
            }
            .contribute(ch, &kin, &mut acc);
            let (expect, delta) = penalize_cell(&cons, &acc, normal, DT, dv, 0);

            assert_eq!(
                out.0[c].to_bits(),
                expect.den.to_bits(),
                "den at ({ii},{jj})"
            );
            assert_eq!(
                out.1[c].to_bits(),
                expect.mom[0].to_bits(),
                "mom0 at ({ii},{jj})"
            );
            assert_eq!(
                out.2[c].to_bits(),
                expect.mom[1].to_bits(),
                "mom1 at ({ii},{jj})"
            );
            assert_eq!(
                out.3[c].to_bits(),
                expect.nrg.to_bits(),
                "nrg at ({ii},{jj})"
            );
            assert_eq!(
                out.4[c].to_bits(),
                delta.mass_delta.to_bits(),
                "mass at ({ii},{jj})"
            );
            assert_eq!(
                out.5[c].to_bits(),
                delta.force_delta[0].to_bits(),
                "fx at ({ii},{jj})"
            );
            assert_eq!(
                out.6[c].to_bits(),
                delta.force_delta[1].to_bits(),
                "fy at ({ii},{jj})"
            );
            assert_eq!(
                out.7[c].to_bits(),
                delta.energy_delta.to_bits(),
                "energy at ({ii},{jj})"
            );
            let tq = symbi_ib::moment(&x_rel, &delta.force_delta);
            assert_eq!(out.8[c].to_bits(), tq[2].to_bits(), "torque at ({ii},{jj})");
            if out.4[c] != 0.0 {
                fired += 1;
            }
            // free-slip through the compiled path: inside the ball, the
            // tangential projection of the velocity change is exactly the
            // drain scaling's (u unchanged) — pin the tangential component of
            // u against the pre-state where the drain leaves u invariant.
            if ch > 0.5 && r > 1e-12 {
                let u0 = cons.mom.scale(1.0 / cons.den);
                let u1 = Tensor::new([out.1[c], out.2[c]]).scale(1.0 / out.0[c]);
                let t = Tensor::new([-normal[1], normal[0]]);
                let (du0, du1) = ((u0 - kin.u_solid).dot(&t), (u1 - kin.u_solid).dot(&t));
                assert!(
                    (du1 - du0).abs() <= 1e-13 * du0.abs().max(1e-30),
                    "free-slip tangential drift at ({ii},{jj}): {du0} -> {du1}",
                );
                tangential_checked += 1;
            }
        }
    }
    assert!(fired > 20, "the porous drain never fired");
    assert!(tangential_checked > 5, "the free-slip check never ran");

    // porosity = 1: the porous kernel equals the drain kernel, bit for bit.
    let porous_p1 = run(porous, ir, 1.0);
    let drain_out = run(drain, drain_ir, 1.0);
    for c in 0..n2 {
        assert_eq!(
            porous_p1.0[c].to_bits(),
            drain_out.0[c].to_bits(),
            "p=1 den at {c}"
        );
        assert_eq!(
            porous_p1.1[c].to_bits(),
            drain_out.1[c].to_bits(),
            "p=1 mom0 at {c}"
        );
        assert_eq!(
            porous_p1.2[c].to_bits(),
            drain_out.2[c].to_bits(),
            "p=1 mom1 at {c}"
        );
        assert_eq!(
            porous_p1.3[c].to_bits(),
            drain_out.3[c].to_bits(),
            "p=1 nrg at {c}"
        );
        assert_eq!(
            porous_p1.4[c].to_bits(),
            drain_out.4[c].to_bits(),
            "p=1 mass at {c}"
        );
        assert_eq!(
            porous_p1.8[c].to_bits(),
            drain_out.8[c].to_bits(),
            "p=1 torque at {c}"
        );
    }
}

// the isothermal porous twin: porosity = 1 equals the isothermal drain, bit for bit
// (the wall channels carry an exact (1 - p) = 0). proves the regime twin is wired.
#[test]
fn iso_porous_reduces_to_iso_drain_at_p1() {
    let (porous, pir) = kernel_by_name::<f64>("penalize_porous_iso_2d").expect("iso porous");
    let (drain, dir) = kernel_by_name::<f64>("penalize_drain_iso_2d").expect("iso drain");
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
    let sc = |name: &str| -> f64 {
        match name {
            "dt" => DT,
            "cs" => 0.8,
            "c_drain" => C_DRAIN,
            "porosity" => 1.0,
            "k_eta_n" => 30.0,
            "k_eta_t" => 5.0,
            "x_lo_0" | "x_lo_1" => X_LO,
            "dx_0" | "dx_1" => DX,
            "map_kind_0" | "map_kind_1" | "map_param_0" | "map_param_1" => 0.0,
            "body_0_pos_0" => POS[0],
            "body_0_pos_1" => POS[1],
            "body_0_racc" => RACC,
            "body_0_mass" => MASS,
            "body_0_vel_0" => 0.05,
            "body_0_vel_1" => -0.03,
            "c_a2" => 0.0,
            other => panic!("unexpected '{other}'"),
        }
    };
    let inp: [&[f64]; 3] = [&den, &mx, &my];
    let p = run_pen(porous, pir, &inp, &sc);
    let d = run_pen(drain, dir, &inp, &sc);
    let mut fired = 0usize;
    for c in 0..n2 {
        for k in 0..4 {
            assert_eq!(p[k][c].to_bits(), d[k][c].to_bits(), "p=1 out{k} at {c}");
        }
        if p[3][c] != 0.0 {
            fired += 1;
        }
    }
    assert!(fired > 20, "the iso porous drain never fired");
}

// the adiabatic torque-free twin: xi = 0 equals the adiabatic drain, bit for bit.
#[test]
fn adiabatic_torque_free_reduces_to_drain_at_xi0() {
    let (tf, tir) = kernel_by_name::<f64>("penalize_torque_free_2d").expect("adiabatic tf");
    let (drain, dir) = kernel_by_name::<f64>("penalize_drain_2d").expect("adiabatic drain");
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
    let sc = |name: &str| -> f64 {
        match name {
            "dt" => DT,
            "gamma" => GAMMA,
            "c_drain" => C_DRAIN,
            // hydro: zero alfven contribution reduces the magnetosonic signal
            // speed sqrt(cs^2 + c_a2) to the sound-speed form exactly.
            "c_a2" => 0.0,
            "xi" => 0.0,
            "x_lo_0" | "x_lo_1" => X_LO,
            "dx_0" | "dx_1" => DX,
            "map_kind_0" | "map_kind_1" | "map_param_0" | "map_param_1" => 0.0,
            "body_0_pos_0" => POS[0],
            "body_0_pos_1" => POS[1],
            "body_0_racc" => RACC,
            "body_0_mass" => MASS,
            "body_0_vel_0" => 0.05,
            "body_0_vel_1" => -0.03,
            other => panic!("unexpected '{other}'"),
        }
    };
    let inp: [&[f64]; 4] = [&den, &mx, &my, &nrg];
    let t = run_pen(tf, tir, &inp, &sc);
    let d = run_pen(drain, dir, &inp, &sc);
    let mut fired = 0usize;
    for c in 0..n2 {
        for k in 0..5 {
            assert_eq!(t[k][c].to_bits(), d[k][c].to_bits(), "xi=0 out{k} at {c}");
        }
        if t[4][c] != 0.0 {
            fired += 1;
        }
    }
    assert!(fired > 20, "the adiabatic torque-free drain never fired");
}

// off-center body: with the body position given in Cartesian (the convention on
// any grid), the cylindrical drain masks a physical ball around that Cartesian
// point — a curved region in (R, phi) enclosing the off-axis Cartesian point. bit-identical to the
// f64 chain, proving the general (non-e_r) off-center case.
#[test]
fn off_center_cylindrical_drain_masks_a_ball_around_a_cartesian_point() {
    use symbi_geometry::{CylindricalRPhi, Metric};
    use symbi_hydro::energy::IsoModel;
    let (kernel, ir) = kernel_by_name::<f64>("penalize_drain_iso_cyl_2d").expect("cyl drain");
    const CS: f64 = 0.8;
    const R_LO: f64 = 0.0;
    const DR: f64 = 0.05;
    const PHI_LO: f64 = 0.0;
    const DPHI: f64 = 0.2;
    const CENTER: [f64; 2] = [0.3, 0.2]; // off-center Cartesian body position
    const RACC_C: f64 = 0.18;

    let n2 = N * N;
    let (mut den, mut mx, mut my) = (vec![0.0; n2], vec![0.0; n2], vec![0.0; n2]);
    for jj in 0..N {
        for ii in 0..N {
            let c = ii + jj * N;
            let (rr, pp) = (
                R_LO + (ii as f64 + 0.5) * DR,
                PHI_LO + (jj as f64 + 0.5) * DPHI,
            );
            den[c] = 1.0 + 0.2 * (2.0 * rr).sin() * (1.5 * pp).cos();
            mx[c] = den[c] * 0.3 * (rr + pp).cos();
            my[c] = den[c] * -0.2 * (rr - 2.0 * pp).sin();
        }
    }
    let sc = |name: &str| -> f64 {
        match name {
            "dt" => DT,
            "cs" => CS,
            "c_drain" => C_DRAIN,
            "x_lo_0" => R_LO,
            "x_lo_1" => PHI_LO,
            "dx_0" => DR,
            "dx_1" => DPHI,
            "map_kind_0" | "map_kind_1" | "map_param_0" | "map_param_1" => 0.0,
            "body_0_pos_0" => CENTER[0],
            "body_0_pos_1" => CENTER[1],
            "body_0_racc" => RACC_C,
                "body_0_mass" => MASS,
            "c_a2" => 0.0,
            other => panic!("unexpected '{other}'"),
        }
    };
    let inp: [&[f64]; 3] = [&den, &mx, &my];
    let o = run_pen(kernel, ir, &inp, &sc);

    let metric = CylindricalRPhi;
    let sphere = SdfExpr::<f64, 2>::sphere(CENTER, RACC_C);
    let rl = |i: usize| R_LO + i as f64 * DR;
    let rh = |i: usize| R_LO + (i as f64 + 1.0) * DR;
    let ir2 = |i: usize| (rh(i) * rh(i) - rl(i) * rl(i)) / 2.0;
    let cnum = |i: usize| (rh(i) * rh(i) * rh(i) - rl(i) * rl(i) * rl(i)) / 3.0;
    let cr = |i: usize| cnum(i) / ir2(i);
    let cphi = |j: usize| ((PHI_LO + j as f64 * DPHI) + (PHI_LO + (j as f64 + 1.0) * DPHI)) * 0.5;
    let iphi = |j: usize| (PHI_LO + (j as f64 + 1.0) * DPHI) - (PHI_LO + j as f64 * DPHI);
    let min_w = DR.min(DPHI);
    let mut fired = 0usize;
    for jj in 0..N {
        for ii in 0..N {
            let c = ii + jj * N;
            let cons = ConsG::<f64, 2, IsoModel> {
                chi: Default::default(),
                den: den[c],
                mom: Tensor::new([mx[c], my[c]]),
                nrg: Default::default(),
            };
            let xc = metric.to_cartesian(Tensor::new([cr(ii), cphi(jj)]));
            let ch = chi(sphere.dist([xc[0], xc[1]]), min_w);
            let inv_tau = drain_rate_floor(CS / (C_DRAIN * min_w), RACC_C);
            let kin = BodyKin::<f64, 2> {
                u_solid: Tensor::zeros(),
                omega: Tensor::zeros(),
                e_wall: 0.0,
            };
            let mut acc = Relax::none();
            Property::Drain { inv_tau }.contribute(ch, &kin, &mut acc);
            let dv = 1.0 / (1.0 / (ir2(ii) * iphi(jj) * 1.0));
            let (out, delta) = penalize_cell(&cons, &acc, Tensor::zeros(), DT, dv, 0);
            assert_eq!(o[0][c].to_bits(), out.den.to_bits(), "den at ({ii},{jj})");
            assert_eq!(
                o[3][c].to_bits(),
                delta.mass_delta.to_bits(),
                "mass at ({ii},{jj})"
            );
            if o[3][c] != 0.0 {
                fired += 1;
            }
        }
    }
    assert!(
        fired > 8,
        "the off-center mask never fired around the Cartesian point"
    );
}
