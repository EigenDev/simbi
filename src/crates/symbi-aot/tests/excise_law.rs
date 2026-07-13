// =============================================================================
// excise_law.rs
//
// the horizon-excision kernels' gates: the compiled onion-sweep fill pair +
// the valencia conserved rebuild are BIT-IDENTICAL to the f64 host chain built
// from the same carrier-generic pieces (`onion_fill_cell`, `RhdGr::to_conserved`
// on the guarded cartesian kerr-schild metric), run as the same sequence
// (K sweeps, then the rebuild). the geometry mirror uses the kernel's own
// centroid arithmetic — faces x_lo + i dx, centroid = face midpoint.
// a uniform state with self-consistent conserved fields must round-trip
// bitwise: the fill copies equal values and the rebuild recomputes the very
// arithmetic that produced the inputs.
// =============================================================================

use symbi_aot::{kernel_by_name, CpuField, CpuFieldMut};
use symbi_algebra::Tensor;
use symbi_geometry::{Metric, SchwarzschildKSCartesian};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::regime::Regime;
use symbi_hydro::spatial_metric::{Gamma, GammaInv, SpatialMetric};
use symbi_hydro::state::Prim;
use symbi_hydro::RhdGr;
use symbi_ib::excise::{onion_fill_cell, onion_pass_count};

const N: usize = 24;
const X_LO: f64 = -0.6;
const DX: f64 = 0.05;
const R_EXC: f64 = 0.35;
const MASS: f64 = 0.3;
const GAMMA: f64 = 4.0 / 3.0;

fn face(ii: usize) -> f64 {
    X_LO + ii as f64 * DX
}

// the kernel's centroid: the face midpoint, NOT x_lo + (i + 0.5) dx (one ulp apart).
fn xc(ii: usize) -> f64 {
    (face(ii) + face(ii + 1)) * 0.5
}

fn to_conserved_at(xx: f64, yy: f64, prim: &Prim<f64, 2>) -> symbi_hydro::state::Cons<f64, 2> {
    let m = SchwarzschildKSCartesian { mass: MASS };
    let x = Tensor::new([xx, yy]);
    let metric = SpatialMetric::<f64, 2>::new(
        Gamma::new(m.spatial_metric(x)),
        GammaInv::new(m.spatial_metric_inv(x)),
    );
    RhdGr { metric, alpha: 1.0 }.to_conserved(&IdealGas { gamma: GAMMA }, prim)
}

struct Grid {
    rho: Vec<f64>,
    v0: Vec<f64>,
    v1: Vec<f64>,
    pre: Vec<f64>,
    den: Vec<f64>,
    m0: Vec<f64>,
    m1: Vec<f64>,
    nrg: Vec<f64>,
}

fn smooth_grid() -> Grid {
    let n2 = N * N;
    let mut g = Grid {
        rho: vec![0.0; n2],
        v0: vec![0.0; n2],
        v1: vec![0.0; n2],
        pre: vec![0.0; n2],
        den: vec![0.0; n2],
        m0: vec![0.0; n2],
        m1: vec![0.0; n2],
        nrg: vec![0.0; n2],
    };
    for jj in 0..N {
        for ii in 0..N {
            let (x, y) = (xc(ii), xc(jj));
            let c = ii + jj * N;
            g.rho[c] = 1.0 + 0.3 * (2.0 * x).sin() * (1.5 * y).cos();
            g.v0[c] = 0.15 * (x + y).cos();
            g.v1[c] = -0.1 * (x - 2.0 * y).sin();
            g.pre[c] = 0.05 + 0.02 * (x * y).cos();
            // arbitrary smooth conserved fields: the rebuild overwrites the excised
            // ones and must pass the live ones through untouched.
            g.den[c] = 2.0 + 0.1 * (3.0 * x).cos();
            g.m0[c] = 0.4 * (y - x).sin();
            g.m1[c] = 0.2 * (x + 0.5 * y).cos();
            g.nrg[c] = 0.9 + 0.05 * (2.0 * y).sin();
        }
    }
    g
}

/// run the compiled sequence: K x (fill -> writeback), then the rebuild.
/// dispatched over the interior so the fill's +-1 stencil stays in bounds.
fn run_compiled(g: &mut Grid) {
    let (fill, fill_ir) = kernel_by_name::<f64>("excise_fill_2d").expect("fill kernel");
    let (wb, wb_ir) = kernel_by_name::<f64>("excise_writeback_2d").expect("writeback kernel");
    let (p2c, p2c_ir) = kernel_by_name::<f64>("excise_p2c_cart_ks_2d").expect("p2c kernel");

    let scalar = |name: &str| -> f64 {
        match name {
            "gamma" => GAMMA,
            "schwarzschild_mass" => MASS,
            "excision_radius" => R_EXC,
            "x_lo_0" | "x_lo_1" => X_LO,
            "dx_0" | "dx_1" => DX,
            "map_kind_0" | "map_kind_1" => 0.0,
            other => panic!("unexpected scalar '{other}'"),
        }
    };
    let bind_scalars = |ir| -> (Vec<i32>, Vec<f64>) {
        let (mut ints, mut scalars) = (Vec::new(), Vec::new());
        for (bind, is_int) in symbi_ir::kernel_scalar_params_typed_from_ir(ir) {
            let v = scalar(&bind.name());
            if is_int {
                ints.push(v as i32)
            } else {
                scalars.push(v)
            }
        }
        (ints, scalars)
    };
    let (fill_ints, fill_scalars) = bind_scalars(fill_ir);
    let (wb_ints, wb_scalars) = bind_scalars(wb_ir);
    let (p2c_ints, p2c_scalars) = bind_scalars(p2c_ir);

    let lo = [0i32; 2];
    let ext = [N as u32; 2];
    let disp_lo = [1i32; 2];
    let disp_ext = [(N - 2) as u32; 2];

    let n2 = N * N;
    let mut exc = [vec![0.0f64; n2], vec![0.0; n2], vec![0.0; n2], vec![0.0; n2]];
    for _ in 0..onion_pass_count(R_EXC, DX) {
        {
            let inputs = [
                CpuField::from_layout(&g.rho, &lo, &ext),
                CpuField::from_layout(&g.v0, &lo, &ext),
                CpuField::from_layout(&g.v1, &lo, &ext),
                CpuField::from_layout(&g.pre, &lo, &ext),
            ];
            let [e0, e1, e2, e3] = &mut exc;
            let mut outs = [
                CpuFieldMut::from_layout(e0, &lo, &ext),
                CpuFieldMut::from_layout(e1, &lo, &ext),
                CpuFieldMut::from_layout(e2, &lo, &ext),
                CpuFieldMut::from_layout(e3, &lo, &ext),
            ];
            fill(&inputs, &mut outs, &disp_ext, &disp_lo, &fill_ints, &fill_scalars);
        }
        {
            let inputs = [
                CpuField::from_layout(&exc[0], &lo, &ext),
                CpuField::from_layout(&exc[1], &lo, &ext),
                CpuField::from_layout(&exc[2], &lo, &ext),
                CpuField::from_layout(&exc[3], &lo, &ext),
            ];
            let mut outs = [
                CpuFieldMut::from_layout(&mut g.rho, &lo, &ext),
                CpuFieldMut::from_layout(&mut g.v0, &lo, &ext),
                CpuFieldMut::from_layout(&mut g.v1, &lo, &ext),
                CpuFieldMut::from_layout(&mut g.pre, &lo, &ext),
            ];
            wb(&inputs, &mut outs, &disp_ext, &disp_lo, &wb_ints, &wb_scalars);
        }
    }
    {
        let (den_in, m0_in, m1_in, nrg_in) =
            (g.den.clone(), g.m0.clone(), g.m1.clone(), g.nrg.clone());
        let inputs = [
            CpuField::from_layout(&g.rho, &lo, &ext),
            CpuField::from_layout(&g.v0, &lo, &ext),
            CpuField::from_layout(&g.v1, &lo, &ext),
            CpuField::from_layout(&g.pre, &lo, &ext),
            CpuField::from_layout(&den_in, &lo, &ext),
            CpuField::from_layout(&m0_in, &lo, &ext),
            CpuField::from_layout(&m1_in, &lo, &ext),
            CpuField::from_layout(&nrg_in, &lo, &ext),
        ];
        let mut outs = [
            CpuFieldMut::from_layout(&mut g.den, &lo, &ext),
            CpuFieldMut::from_layout(&mut g.m0, &lo, &ext),
            CpuFieldMut::from_layout(&mut g.m1, &lo, &ext),
            CpuFieldMut::from_layout(&mut g.nrg, &lo, &ext),
        ];
        p2c(&inputs, &mut outs, &disp_ext, &disp_lo, &p2c_ints, &p2c_scalars);
    }
}

/// the f64 chain: the same sweep + rebuild sequence from the same carrier code.
fn run_reference(g: &mut Grid) {
    for _ in 0..onion_pass_count(R_EXC, DX) {
        let (rho, v0, v1, pre) = (g.rho.clone(), g.v0.clone(), g.v1.clone(), g.pre.clone());
        let at = |f: &[f64], ii: isize, jj: isize| f[ii as usize + jj as usize * N];
        for jj in 1..(N - 1) as isize {
            for ii in 1..(N - 1) as isize {
                let st = |di: isize, dj: isize| -> [f64; 4] {
                    [
                        at(&rho, ii + di, jj + dj),
                        at(&v0, ii + di, jj + dj),
                        at(&v1, ii + di, jj + dj),
                        at(&pre, ii + di, jj + dj),
                    ]
                };
                let filled = onion_fill_cell(
                    st(0, 0),
                    st(1, 1),
                    st(1, -1),
                    st(-1, 1),
                    st(-1, -1),
                    [xc(ii as usize), xc(jj as usize)],
                    R_EXC,
                );
                let c = ii as usize + jj as usize * N;
                g.rho[c] = filled[0];
                g.v0[c] = filled[1];
                g.v1[c] = filled[2];
                g.pre[c] = filled[3];
            }
        }
    }
    for jj in 1..N - 1 {
        for ii in 1..N - 1 {
            let (x, y) = (xc(ii), xc(jj));
            if (x * x + y * y).sqrt() < R_EXC {
                let c = ii + jj * N;
                let prim = Prim::<f64, 2> {
                    rho: g.rho[c],
                    vel: Tensor::new([g.v0[c], g.v1[c]]),
                    pre: g.pre[c],
                };
                let cons = to_conserved_at(x, y, &prim);
                g.den[c] = cons.den;
                g.m0[c] = cons.mom[0];
                g.m1[c] = cons.mom[1];
                g.nrg[c] = cons.nrg;
            }
        }
    }
}

#[test]
fn compiled_excise_sequence_matches_the_f64_chain_bitwise() {
    let mut compiled = smooth_grid();
    let mut reference = smooth_grid();
    let input = smooth_grid();

    run_compiled(&mut compiled);
    run_reference(&mut reference);

    let mut n_excised = 0usize;
    for jj in 0..N {
        for ii in 0..N {
            let c = ii + jj * N;
            let (x, y) = (xc(ii), xc(jj));
            let live = (x * x + y * y).sqrt() >= R_EXC;
            for (name, a, b) in [
                ("rho", &compiled.rho, &reference.rho),
                ("v0", &compiled.v0, &reference.v0),
                ("v1", &compiled.v1, &reference.v1),
                ("pre", &compiled.pre, &reference.pre),
                ("den", &compiled.den, &reference.den),
                ("m0", &compiled.m0, &reference.m0),
                ("m1", &compiled.m1, &reference.m1),
                ("nrg", &compiled.nrg, &reference.nrg),
            ] {
                assert_eq!(
                    a[c].to_bits(),
                    b[c].to_bits(),
                    "{name} at ({ii},{jj}): compiled {} vs reference {}",
                    a[c],
                    b[c]
                );
            }
            if live {
                // a live cell is bitwise untouched by the whole sequence.
                assert_eq!(compiled.rho[c].to_bits(), input.rho[c].to_bits(), "live rho at ({ii},{jj})");
                assert_eq!(compiled.den[c].to_bits(), input.den[c].to_bits(), "live den at ({ii},{jj})");
            } else {
                n_excised += 1;
            }
        }
    }
    assert!(n_excised > 100, "the excision ball must be deep (got {n_excised} cells)");
}

#[test]
fn uniform_state_round_trips_bitwise() {
    // uniform primitives with per-cell self-consistent conserved fields: the fill
    // copies equal values and the rebuild recomputes the arithmetic that produced
    // the inputs — the whole sequence is the bitwise identity.
    let mut g = smooth_grid();
    let uni = Prim::<f64, 2> { rho: 1.2, vel: Tensor::new([0.1, -0.05]), pre: 0.03 };
    for jj in 0..N {
        for ii in 0..N {
            let c = ii + jj * N;
            g.rho[c] = uni.rho;
            g.v0[c] = uni.vel[0];
            g.v1[c] = uni.vel[1];
            g.pre[c] = uni.pre;
            let cons = to_conserved_at(xc(ii), xc(jj), &uni);
            g.den[c] = cons.den;
            g.m0[c] = cons.mom[0];
            g.m1[c] = cons.mom[1];
            g.nrg[c] = cons.nrg;
        }
    }
    let before = Grid {
        rho: g.rho.clone(),
        v0: g.v0.clone(),
        v1: g.v1.clone(),
        pre: g.pre.clone(),
        den: g.den.clone(),
        m0: g.m0.clone(),
        m1: g.m1.clone(),
        nrg: g.nrg.clone(),
    };
    run_compiled(&mut g);
    for c in 0..N * N {
        assert_eq!(g.rho[c].to_bits(), before.rho[c].to_bits(), "rho at {c}");
        assert_eq!(g.v0[c].to_bits(), before.v0[c].to_bits(), "v0 at {c}");
        assert_eq!(g.v1[c].to_bits(), before.v1[c].to_bits(), "v1 at {c}");
        assert_eq!(g.pre[c].to_bits(), before.pre[c].to_bits(), "pre at {c}");
        assert_eq!(g.den[c].to_bits(), before.den[c].to_bits(), "den at {c}");
        assert_eq!(g.m0[c].to_bits(), before.m0[c].to_bits(), "m0 at {c}");
        assert_eq!(g.m1[c].to_bits(), before.m1[c].to_bits(), "m1 at {c}");
        assert_eq!(g.nrg[c].to_bits(), before.nrg[c].to_bits(), "nrg at {c}");
    }
}
