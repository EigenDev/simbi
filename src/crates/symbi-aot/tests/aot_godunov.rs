// =============================================================================
// aot_godunov.rs
//
// run the BUILD-TIME-GENERATED godunov kernel (compiled into this crate from a
// substrate RegimeSpec via emit_kernel_cpu) and diff against the hand-written
// reference. since the SAME kernel matches hand-written via the #4
// interpreter, this transitively proves the AOT-compiled Rust ≡ the interpreter
// — and, more importantly, that the generated Rust COMPILES and RUNS.
//
// the generated signature (see the header in OUT_DIR/godunov_generated.rs):
//   godunov_mass_1d(
//       buf0: &[f64],      // rho      (input)
//       buf1: &[f64],      // mass_flux (input, face field, n+1 entries)
//       buf2: &mut [f64],  // rho_new  (output)
//       grid_size_0: i64, dom_lo_0: i64,
//       buf_lo_0_0: i64, buf_lo_1_0: i64, buf_lo_2_0: i64,
//       dt: f64, dx_0: f64,
//   )
// =============================================================================

use symbi_aot::NamedKernel;

// thin shim mapping a "raw slices + scattered lo" call shape to the
// view-struct ABI. tests stay focused on the kernel's NUMERICS.
fn godunov_mass_1d<S: symbi_aot::Scalar + symbi_aot::OrderedNumeric + Send + Sync>(
    rho: &[S], flux: &[S], rho_new: &mut [S],
    grid_size_0: i32, dom_lo_0: i32,
    _lo0: i32, _lo1: i32, _lo2: i32,
    dt: S, dx_0: S,
) {
    let grid = [grid_size_0 as u32];
    let dom = [dom_lo_0];
    NamedKernel::new("godunov_mass_1d")
        .input("cons.den", rho).input("mass_flux[0]", flux)
        .output("cons.den_new", rho_new)
        .grid(&grid).dom_lo(&dom)
        .scalar("dt", dt).scalar("dx_0", dx_0)
        .run();
}

#[test]
fn aot_mass_godunov_step_matches_hand_written() {
    let n = 8usize;
    let (dt, dx) = (0.01_f64, 0.5_f64);
    // a Sod-like discontinuity and a (given) face flux of n+1 entries.
    let rho: Vec<f64> = (0..n).map(|i| if i < n / 2 { 1.0 } else { 0.125 }).collect();
    let flux: Vec<f64> = (0..=n).map(|i| 0.3 - 0.05 * i as f64).collect();
    let mut rho_new = vec![0.0_f64; n];

    godunov_mass_1d(&rho, &flux, &mut rho_new, n as i32, 0, 0, 0, 0, dt, dx);

    // the hand-written density update of hydro_godunov_euler:
    //   rho_new[i] = rho[i] - dt/dx * (flux[i+1] - flux[i]).
    for i in 0..n {
        let expected = rho[i] - dt / dx * (flux[i + 1] - flux[i]);
        assert!(
            (rho_new[i] - expected).abs() < 1e-12,
            "cell {i}: AOT {} != hand-written {}", rho_new[i], expected,
        );
    }
    assert!(rho_new.iter().zip(&rho).any(|(a, b)| a != b), "the AOT godunov step was a no-op");
}

#[test]
fn aot_mass_godunov_runs_at_f32() {
    // the SAME generic kernel `godunov_mass_1d<S: Scalar>` instantiated at f32 — S
    // inferred from the &[f32] buffers + f32 scalars. proves the precision-generic
    // codegen: one kernel, run at f32, Scalar-for-f32 forwarding
    // to f32::* correctly. no separate f32 kernel, no dispatch — just inference.
    let n = 8usize;
    let (dt, dx) = (0.01_f32, 0.5_f32);
    let rho: Vec<f32> = (0..n).map(|i| if i < n / 2 { 1.0 } else { 0.125 }).collect();
    let flux: Vec<f32> = (0..=n).map(|i| 0.3 - 0.05 * i as f32).collect();
    let mut rho_new = vec![0.0_f32; n];

    godunov_mass_1d(&rho, &flux, &mut rho_new, n as i32, 0, 0, 0, 0, dt, dx);

    for i in 0..n {
        let expected = rho[i] - dt / dx * (flux[i + 1] - flux[i]);
        assert!(
            (rho_new[i] - expected).abs() < 1e-6,
            "cell {i}: AOT f32 {} != hand-written {}", rho_new[i], expected,
        );
    }
    assert!(rho_new.iter().zip(&rho).any(|(a, b)| a != b), "the f32 godunov step was a no-op");
}

#[test]
fn aot_mass_godunov_conserves_over_many_steps() {
    // march a smooth density bump under constant-velocity upwind advection for
    // many steps. total mass telescopes to the net boundary flux; the bump stays
    // interior so both edge cells hold the background (rho = 1) throughout and
    // the net boundary flux is zero -> mass conserved. proves the AOT kernel
    // iterates correctly, not just one step.
    let n = 64usize;
    let dx = 1.0 / n as f64;
    let a = 1.0_f64; // advection speed (> 0, so upwind takes the left cell)
    let dt = 0.4 * dx / a; // cfl 0.4
    let mut rho: Vec<f64> = (0..n)
        .map(|i| {
            let x = (i as f64 + 0.5) * dx;
            1.0 + 0.5 * (-((x - 0.3) * (x - 0.3)) / 0.002).exp() // gaussian bump near x=0.3
        })
        .collect();

    let mass0: f64 = rho.iter().sum::<f64>() * dx;

    // 50 steps -> t = 0.3125, bump centre 0.3 -> ~0.61, well clear of the edges.
    let steps = 50;
    for _ in 0..steps {
        // upwind face flux (a > 0): F[j] = a * rho[left cell]; transmissive edges.
        let mut flux = vec![0.0_f64; n + 1];
        for j in 0..=n {
            let left = if j == 0 { 0 } else { j - 1 };
            flux[j] = a * rho[left];
        }
        let mut rho_new = vec![0.0_f64; n];
        godunov_mass_1d(&rho, &flux, &mut rho_new, n as i32, 0, 0, 0, 0, dt, dx);
        rho = rho_new;
    }

    for (i, &r) in rho.iter().enumerate() {
        assert!(r.is_finite() && r > 0.0, "cell {i}: rho = {r}");
    }
    let mass1: f64 = rho.iter().sum::<f64>() * dx;
    assert!(
        (mass1 - mass0).abs() < 1e-9 * mass0,
        "mass drift over {steps} steps: {} (rel {:e})", mass1 - mass0, (mass1 - mass0) / mass0,
    );
    // and the bump actually moved right (advection happened, not a no-op).
    let peak = rho.iter().cloned().fold(f64::MIN, f64::max);
    let peak_idx = rho.iter().position(|&r| r == peak).unwrap();
    assert!(peak_idx as f64 * dx > 0.45, "bump did not advect right: peak at x={}", peak_idx as f64 * dx);
}
