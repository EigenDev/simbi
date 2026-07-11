// =============================================================================
// support_law.rs
//
// the support-validation law (docs/design/48 part 3): a kernel's DECLARED
// output support must hold on the COMPILED kernel — every output exactly zero
// (in f64, not approximately) at every cell outside the declared region, for
// nontrivial field inputs. the declaration drives dispatch decisions (the
// body-feedback reduction box), so a support declared wider than reality is
// harmless and a support declared narrower is a silent physics loss — exactly
// what this sampler catches.
//
// gates:
//   - the drain kernel's blob carries a Ball declaration (losing it fails loudly)
//   - outputs are exactly zero outside the ball, nonzero somewhere inside
//   - every cell of the run region is assign-written (the scratch-reuse
//     contract the cached feedback scratch relies on)
//   - bug injection: a shrunken ball leaves nonzero cells outside it — the
//     sampler distinguishes a too-narrow declaration
// =============================================================================

use symbi_aot::{kernel_by_name, CpuField, CpuFieldMut};
use symbi_ir::{kernel_output_support_from_ir, kernel_scalar_params_typed_from_ir, Support};

const N: usize = 96;
const X_LO: f64 = -1.2;
const DX: f64 = 0.025;
const BODY_POS: [f64; 2] = [0.13, -0.07];
const RACC: f64 = 0.1;
const SENTINEL: f64 = 7.7;

// the drain kernel's scalar table, by manifest name. any name outside this
// vocabulary panics — a kernel that grows a param must extend the sampler.
fn scalar_value(name: &str) -> f64 {
    match name {
        "dt" => 0.01,
        "gamma" => 1.4,
        "x_lo_0" | "x_lo_1" => X_LO,
        "dx_0" | "dx_1" => DX,
        "map_kind_0" | "map_kind_1" => 0.0,
        "body_0_mass" => 1.0,
        "body_0_soft" => 0.05,
        "body_0_pos_0" => BODY_POS[0],
        "body_0_pos_1" => BODY_POS[1],
        "body_0_racc" => RACC,
        "body_0_sink" => 5.0,
        other => panic!("support sampler: unresolved scalar param '{other}'"),
    }
}

// the kernel's own cell-center map: uniform faces x_lo + i*dx, centroid at the
// arithmetic mid.
fn cell_center(i: usize) -> f64 {
    X_LO + (i as f64 + 0.5) * DX
}

// nontrivial, smooth, positive-pressure conserved fields — the support must
// hold for ANY field values, so nothing here is tuned to the body.
fn cons_fields() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut den = Vec::with_capacity(N * N);
    let mut mx = Vec::with_capacity(N * N);
    let mut my = Vec::with_capacity(N * N);
    let mut nrg = Vec::with_capacity(N * N);
    for jj in 0..N {
        for ii in 0..N {
            let (x, y) = (cell_center(ii), cell_center(jj));
            let rho = 1.0 + 0.25 * (3.0 * x).sin() * (2.0 * y).cos();
            den.push(rho);
            mx.push(rho * 0.3 * (x + 2.0 * y).cos());
            my.push(rho * -0.2 * (2.0 * x - y).sin());
            nrg.push(2.0 + 0.5 * (x + y).sin());
        }
    }
    (den, mx, my, nrg)
}

// run body_feedback_drain_2d over the full grid into 7 sentinel-filled outputs.
fn run_drain() -> (Vec<Vec<f64>>, Support) {
    let name = "body_feedback_drain_2d";
    let (kernel, ir) = kernel_by_name::<f64>(name).expect("drain kernel in the registry");
    let support = kernel_output_support_from_ir(ir)
        .expect("the drain kernel must DECLARE its output support (docs/design/48 part 3)");

    let (den, mx, my, nrg) = cons_fields();
    let lo = [0i32; 2];
    let ext = [N as u32; 2];
    let inputs = [
        CpuField::from_layout(&den, &lo, &ext),
        CpuField::from_layout(&mx, &lo, &ext),
        CpuField::from_layout(&my, &lo, &ext),
        CpuField::from_layout(&nrg, &lo, &ext),
    ];
    let mut outs: Vec<Vec<f64>> = (0..7).map(|_| vec![SENTINEL; N * N]).collect();
    {
        let mut out_fields: Vec<CpuFieldMut> = outs
            .iter_mut()
            .map(|o| CpuFieldMut::from_layout(o, &lo, &ext))
            .collect();
        // scalar args in the kernel's own type-sorted manifest order.
        let mut ints = Vec::new();
        let mut scalars = Vec::new();
        for (bind, is_int) in kernel_scalar_params_typed_from_ir(ir) {
            let v = scalar_value(&bind.name());
            if is_int {
                ints.push(v as i32);
            } else {
                scalars.push(v);
            }
        }
        kernel(&inputs, &mut out_fields, &[N as u32; 2], &[0i32; 2], &ints, &scalars);
    }
    (outs, support)
}

#[test]
fn drain_outputs_are_exactly_zero_outside_the_declared_ball() {
    let (outs, support) = run_drain();
    let (center, radius) = support
        .eval_ball(&|n| scalar_value(n))
        .expect("the drain support must be a Ball");
    assert_eq!(center, BODY_POS.to_vec());
    assert_eq!(radius, RACC + 20.0 * DX, "the declared radius drifted from ibm::DRAIN_SUPPORT_WIDTHS");
    assert!(
        radius < (N as f64) * DX * 0.5,
        "the ball covers the whole grid — the outside sample set is empty and the law vacuous",
    );

    let mut outside = 0usize;
    let mut inside_nonzero = false;
    for ii in 0..N {
        for jj in 0..N {
            let d = ((cell_center(ii) - center[0]).powi(2)
                + (cell_center(jj) - center[1]).powi(2))
            .sqrt();
            let cell = ii + jj * N;
            for (k, o) in outs.iter().enumerate() {
                assert_ne!(
                    o[cell], SENTINEL,
                    "output {k} not assign-written at ({ii},{jj}) — the cached-scratch reuse \
                     contract (no re-zeroing) is broken",
                );
                // a boundary-cell distance can differ from the kernel's own by
                // rounding; a one-ulp band at the surface stays unjudged.
                if d > radius * (1.0 + 1e-12) {
                    assert_eq!(
                        o[cell], 0.0,
                        "output {k} nonzero at ({ii},{jj}), distance {d:.6} > declared radius \
                         {radius:.6} — the declared support is NARROWER than the kernel's reality",
                    );
                }
            }
            if d > radius * (1.0 + 1e-12) {
                outside += 1;
            } else if outs[5][cell] != 0.0 {
                inside_nonzero = true; // output 5 = absorbed mass
            }
        }
    }
    assert!(outside > N * N / 10, "too few outside-ball cells ({outside}) — weak sample");
    assert!(
        inside_nonzero,
        "no absorbed mass anywhere inside the ball — the kernel never drained and the law is vacuous",
    );
}

// bug injection: a ball shrunk below the true support must FAIL the sampler —
// nonzero outputs exist outside it. proves the law can distinguish a too-narrow
// declaration (the failure mode that silently drops physics from a reduction).
#[test]
fn a_shrunken_ball_is_caught_by_the_sampler() {
    let (outs, support) = run_drain();
    let (center, radius) = support.eval_ball(&|n| scalar_value(n)).unwrap();
    let shrunk = radius * 0.25;
    let mut violations = 0usize;
    for ii in 0..N {
        for jj in 0..N {
            let d = ((cell_center(ii) - center[0]).powi(2)
                + (cell_center(jj) - center[1]).powi(2))
            .sqrt();
            if d > shrunk && outs.iter().any(|o| o[ii + jj * N] != 0.0) {
                violations += 1;
            }
        }
    }
    assert!(
        violations > 0,
        "no nonzero cell outside a quarter-radius ball — the sampler could never catch a \
         too-narrow declaration",
    );
}
