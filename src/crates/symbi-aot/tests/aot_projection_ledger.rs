// =============================================================================
// aot_projection_ledger.rs
//
// pins for the production GRMHD admissible-boundary projection kernel, which
// emits the projection-ledger diagnostic channels alongside its candidate
// writes:
// - manifest: the kernel writes the candidate (D, S_i, tau) slots plus exactly
//   the four diagnostic channels (`xd_theta`, `xd_d_den`, `xd_d_nrg_seg`,
//   `xd_d_nrg_raise`);
// - passthrough: an admissible cell writes theta = 1, zero receipts, and the
//   candidate state passes through untouched;
// - conservation identity: a projected cell's receipts equal the observed state
//   deltas to the last bit — the den receipt is the D-slot delta and the two
//   energy receipts sum to the tau-slot delta, so the ledger books exactly what
//   the projection moved.
// =============================================================================

use symbi_aot::NamedKernel;

const PROD: &str = "rmhd_fofc_project_ks_3d";

fn manifest(name: &str) -> (Vec<String>, Vec<String>) {
    let (_, ir) = symbi_aot::kernel_by_name::<f64>(name)
        .unwrap_or_else(|| panic!("kernel '{name}' is not baked"));
    let bindings = symbi_ir::kernel_bindings_from_ir(ir);
    let inputs = bindings
        .iter()
        .filter(|(_, is_out)| !is_out)
        .map(|(b, _)| format!("{b:?}"))
        .collect();
    let writes = bindings
        .iter()
        .filter(|(_, is_out)| *is_out)
        .map(|(b, _)| format!("{b:?}"))
        .collect();
    (inputs, writes)
}

#[test]
fn the_production_kernel_writes_the_candidate_plus_four_diagnostic_channels() {
    let (_inputs, writes) = manifest(PROD);
    // the candidate conserved writes come first; the four diagnostic channels
    // are appended.
    let diag: Vec<_> = writes.iter().filter(|w| w.contains("xd_")).collect();
    assert_eq!(diag.len(), 4, "the four diagnostic channels: {writes:?}");
    let candidate = writes.len() - 4;
    assert!(candidate >= 5, "the (D, S_0..2, tau) candidate writes: {writes:?}");
}

/// two cells on a far-region cartesian kerr-schild patch: cell 0 an admissible
/// candidate, cell 1 an out-of-cone one (momentum far past the energy). the
/// stage gas — the eulerian-rebuild anchor's primitives — is calm on both.
struct Setup {
    x_den: Vec<f64>,
    x_mom: [Vec<f64>; 3],
    x_nrg: Vec<f64>,
}

fn setup() -> Setup {
    Setup {
        x_den: vec![1.0, 1.0],
        x_mom: [vec![0.1, 10.0], vec![0.0, 0.0], vec![0.0, 0.0]],
        x_nrg: vec![2.0, 0.05],
    }
}

#[allow(clippy::type_complexity)]
fn run() -> (Vec<f64>, [Vec<f64>; 3], Vec<f64>, [Vec<f64>; 4]) {
    let s = setup();
    let n = 2usize;
    let mut x_den = s.x_den.clone();
    let [mut x_mom_0, mut x_mom_1, mut x_mom_2] = s.x_mom.clone();
    let mut x_nrg = s.x_nrg.clone();
    let prim_rho = vec![1.0, 1.0];
    let prim_vel = [vec![0.1, 0.1], vec![0.0, 0.0], vec![0.0, 0.0]];
    let prim_pre = vec![0.2, 0.2];
    let bcell = [vec![0.05, 0.05], vec![0.0, 0.0], vec![0.0, 0.0]];
    let (mut xd_theta, mut xd_d_den, mut xd_d_seg, mut xd_d_raise) =
        (vec![-1.0; n], vec![-1.0; n], vec![-1.0; n], vec![-1.0; n]);

    let grid = [n as u32, 1, 1];
    let dom = [0, 0, 0];
    let mut k = NamedKernel::new(PROD);
    k = k.input("mhd.bcell[0]", &bcell[0]);
    k = k.input("mhd.bcell[1]", &bcell[1]);
    k = k.input("mhd.bcell[2]", &bcell[2]);
    k = k.input("prim.rho", &prim_rho);
    k = k.input("prim.vel_0", &prim_vel[0]);
    k = k.input("prim.vel_1", &prim_vel[1]);
    k = k.input("prim.vel_2", &prim_vel[2]);
    k = k.input("prim.pre", &prim_pre);
    k = k.output("x_den", &mut x_den);
    k = k.output("x_mom_0", &mut x_mom_0);
    k = k.output("x_mom_1", &mut x_mom_1);
    k = k.output("x_mom_2", &mut x_mom_2);
    k = k.output("x_nrg", &mut x_nrg);
    k = k.output("xd_theta", &mut xd_theta);
    k = k.output("xd_d_den", &mut xd_d_den);
    k = k.output("xd_d_nrg_seg", &mut xd_d_seg);
    k = k.output("xd_d_nrg_raise", &mut xd_d_raise);
    k = k.grid(&grid);
    k = k.dom_lo(&dom);
    // a far-region patch (r ~ 17 M): the metric is genuinely curved and
    // regular, uniform spacing on every axis.
    let grid_scalars: Vec<(String, f64)> = (0..3)
        .flat_map(|ax| {
            [
                (format!("x_lo_{ax}"), 10.0),
                (format!("dx_{ax}"), 0.5),
                (format!("map_kind_{ax}"), 0.0),
                (format!("map_param_{ax}"), 0.0),
            ]
        })
        .collect();
    for (name, value) in &grid_scalars {
        k = k.scalar(name, *value);
    }
    k = k.scalar("schwarzschild_mass", 1.0);
    k = k.scalar("gamma", 5.0 / 3.0);
    k.run();
    (
        x_den,
        [x_mom_0, x_mom_1, x_mom_2],
        x_nrg,
        [xd_theta, xd_d_den, xd_d_seg, xd_d_raise],
    )
}

#[test]
fn an_admissible_cell_passes_through_with_an_empty_receipt() {
    let s = setup();
    let (den, mom, nrg, diag) = run();
    assert_eq!(diag[0][0], 1.0, "theta = 1 on an admissible cell");
    assert_eq!(diag[1][0], 0.0);
    assert_eq!(diag[2][0], 0.0);
    assert_eq!(diag[3][0], 0.0);
    assert_eq!(den[0], s.x_den[0]);
    assert_eq!(nrg[0], s.x_nrg[0]);
    for k in 0..3 {
        assert_eq!(mom[k][0], s.x_mom[k][0]);
    }
}

#[test]
fn receipts_equal_the_observed_state_deltas_exactly() {
    let s = setup();
    let (den, _mom, nrg, diag) = run();
    // cell 1 projects: theta in [0, 1), the den receipt is the D-slot delta,
    // and the segment plus raise receipts sum to the tau-slot delta — the
    // ledger books exactly the state the projection moved, to the last bit.
    assert!(diag[0][1] < 1.0, "the out-of-cone cell projects");
    assert!(diag[0][1] >= 0.0);
    assert_eq!(diag[1][1], den[1] - s.x_den[1], "den receipt = D-slot delta");
    let d_nrg = nrg[1] - s.x_nrg[1];
    assert_eq!(
        diag[2][1] + diag[3][1],
        d_nrg,
        "segment + raise = the tau-slot delta"
    );
}
