// =============================================================================
// aot_anchor_experiment.rs
//
// pins for the projection-anchor experiment kernels (measurement apparatus
// for the anchor study):
// - structural: the two convention arms share every write and every scalar,
//   and their field inputs differ only in the anchor slots (the rebuilt arm's
//   stage-gas primitives against the stage arm's stage conserved slots); the
//   production kernel's manifest is the arms' manifest with the diagnostic
//   channels removed.
// - numeric: on identical inputs the rebuilt arm's candidate outputs equal
//   the production kernel's exactly — the diagnostics are observers.
// - receipts: an admissible cell writes the empty identities exactly
//   (theta = 1, zero deltas, outputs equal inputs), and a projected cell's
//   receipts equal the observed state deltas analytically — the mass receipt
//   is the den-slot delta and the two energy receipts sum to the nrg-slot
//   delta, to the last bit.
// =============================================================================

use symbi_aot::NamedKernel;

const PROD: &str = "rmhd_fofc_project_ks_3d";
const REBUILT: &str = "rmhd_fofc_project_ks_expt_rebuilt_3d";
const STAGE: &str = "rmhd_fofc_project_ks_expt_stage_3d";

fn manifest(name: &str) -> (Vec<String>, Vec<String>, Vec<String>) {
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
    let scalars = symbi_ir::kernel_scalar_params_typed_from_ir(ir)
        .iter()
        .map(|(s, _)| format!("{s:?}"))
        .collect();
    (inputs, writes, scalars)
}

#[test]
fn the_arms_differ_only_in_anchor_inputs() {
    let (prod_in, prod_w, prod_s) = manifest(PROD);
    let (reb_in, reb_w, reb_s) = manifest(REBUILT);
    let (stg_in, stg_w, stg_s) = manifest(STAGE);

    // one scalar manifest across all three.
    assert_eq!(prod_s, reb_s);
    assert_eq!(reb_s, stg_s);

    // the arms share every write; the production writes are the arms' writes
    // with the diagnostic channels removed.
    assert_eq!(reb_w, stg_w);
    assert_eq!(prod_w[..], reb_w[..prod_w.len()]);
    let diag: Vec<_> = reb_w[prod_w.len()..].to_vec();
    assert_eq!(diag.len(), 4, "the four diagnostic channels: {diag:?}");
    for d in &diag {
        assert!(d.contains("xd_"), "diagnostic write {d}");
    }

    // the rebuilt arm reads exactly what production reads.
    assert_eq!(prod_in, reb_in);

    // the stage arm swaps the anchor slots and only those: stage-gas
    // primitives out, stage conserved slots in.
    let only_reb: Vec<_> = reb_in.iter().filter(|i| !stg_in.contains(i)).collect();
    let only_stg: Vec<_> = stg_in.iter().filter(|i| !reb_in.contains(i)).collect();
    assert!(
        only_reb.iter().all(|i| i.contains("Prim")),
        "rebuilt-only inputs are the stage-gas primitives: {only_reb:?}"
    );
    assert!(
        only_stg.iter().all(|i| i.contains("us_")),
        "stage-only inputs are the stage conserved slots: {only_stg:?}"
    );
    assert_eq!(only_reb.len(), 5, "{only_reb:?}");
    assert_eq!(only_stg.len(), 5, "{only_stg:?}");
}

/// two cells on a far-region cartesian kerr-schild patch: cell 0 carries an
/// admissible candidate, cell 1 an out-of-cone one (momentum far past the
/// energy). the stage gas is calm on both.
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
fn run(
    name: &str,
    diagnostics: bool,
) -> (Vec<f64>, [Vec<f64>; 3], Vec<f64>, Option<[Vec<f64>; 4]>) {
    let s = setup();
    let n = 2usize;
    let mut x_den = s.x_den.clone();
    let [mut x_mom_0, mut x_mom_1, mut x_mom_2] = s.x_mom.clone();
    let mut x_nrg = s.x_nrg.clone();
    let prim_rho = vec![1.0, 1.0];
    let prim_vel = [vec![0.1, 0.1], vec![0.0, 0.0], vec![0.0, 0.0]];
    let prim_pre = vec![0.2, 0.2];
    let us_den = vec![1.0, 1.0];
    let us_mom = [vec![0.1, 0.1], vec![0.0, 0.0], vec![0.0, 0.0]];
    let us_nrg = vec![1.0, 1.0];
    let bcell = [vec![0.05, 0.05], vec![0.0, 0.0], vec![0.0, 0.0]];
    let (mut xd_theta, mut xd_d_den, mut xd_d_seg, mut xd_d_raise) =
        (vec![-1.0; n], vec![-1.0; n], vec![-1.0; n], vec![-1.0; n]);

    let grid = [n as u32, 1, 1];
    let dom = [0, 0, 0];
    let mut k = NamedKernel::new(name);
    k = k.input("mhd.bcell[0]", &bcell[0]);
    k = k.input("mhd.bcell[1]", &bcell[1]);
    k = k.input("mhd.bcell[2]", &bcell[2]);
    if name == STAGE {
        k = k.input("us_den", &us_den);
        k = k.input("us_mom_0", &us_mom[0]);
        k = k.input("us_mom_1", &us_mom[1]);
        k = k.input("us_mom_2", &us_mom[2]);
        k = k.input("us_nrg", &us_nrg);
    } else {
        k = k.input("prim.rho", &prim_rho);
        k = k.input("prim.vel_0", &prim_vel[0]);
        k = k.input("prim.vel_1", &prim_vel[1]);
        k = k.input("prim.vel_2", &prim_vel[2]);
        k = k.input("prim.pre", &prim_pre);
    }
    k = k.output("x_den", &mut x_den);
    k = k.output("x_mom_0", &mut x_mom_0);
    k = k.output("x_mom_1", &mut x_mom_1);
    k = k.output("x_mom_2", &mut x_mom_2);
    k = k.output("x_nrg", &mut x_nrg);
    if diagnostics {
        k = k.output("xd_theta", &mut xd_theta);
        k = k.output("xd_d_den", &mut xd_d_den);
        k = k.output("xd_d_nrg_seg", &mut xd_d_seg);
        k = k.output("xd_d_nrg_raise", &mut xd_d_raise);
    }
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
        diagnostics.then_some([xd_theta, xd_d_den, xd_d_seg, xd_d_raise]),
    )
}

#[test]
fn diagnostics_observe_without_touching_the_candidate() {
    let (p_den, p_mom, p_nrg, _) = run(PROD, false);
    let (r_den, r_mom, r_nrg, _) = run(REBUILT, true);
    assert_eq!(p_den, r_den);
    assert_eq!(p_nrg, r_nrg);
    for k in 0..3 {
        assert_eq!(p_mom[k], r_mom[k]);
    }
}

#[test]
fn receipts_equal_the_observed_state_deltas_exactly() {
    let s = setup();
    let (den, _mom, nrg, diag) = run(REBUILT, true);
    let diag = diag.unwrap();

    // cell 0 is admissible: the empty identities hold exactly and the state
    // passes through untouched.
    assert_eq!(diag[0][0], 1.0, "theta = 1 on an admissible cell");
    assert_eq!(diag[1][0], 0.0);
    assert_eq!(diag[2][0], 0.0);
    assert_eq!(diag[3][0], 0.0);
    assert_eq!(den[0], s.x_den[0]);
    assert_eq!(nrg[0], s.x_nrg[0]);

    // cell 1 projects: theta in [0, 1), the mass receipt is the den-slot
    // delta, and the two energy receipts sum to the nrg-slot delta — the
    // receipts are the analytic state deltas, to the last bit.
    assert!(diag[0][1] < 1.0, "the out-of-cone cell projects");
    assert!(diag[0][1] >= 0.0);
    assert_eq!(diag[1][1], den[1] - s.x_den[1], "mass receipt = den delta");
    let d_nrg = nrg[1] - s.x_nrg[1];
    assert_eq!(
        diag[2][1] + diag[3][1],
        d_nrg,
        "segment + raise = the nrg-slot delta"
    );

    // the stage arm runs the same shared tail on its own anchor.
    let (sden, _smom, snrg, sdiag) = run(STAGE, true);
    let sdiag = sdiag.unwrap();
    assert_eq!(sdiag[1][1], sden[1] - s.x_den[1]);
    assert_eq!(sdiag[2][1] + sdiag[3][1], snrg[1] - s.x_nrg[1]);
}
