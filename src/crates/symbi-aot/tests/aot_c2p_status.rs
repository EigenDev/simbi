// =============================================================================
// aot_c2p_status.rs
//
// end-to-end validation of the c2p status channel on the compiled kernels:
// the recovery kernel writes its accept/reject fact alongside the raw
// candidate, so a rejected cell carries exactly `INVALID_PRIMITIVE` (64.0) on
// the status channel while the candidate outputs stay the raw computation —
// data and control ride separate channels. one rejected case per family
// condition:
// - adiabatic: magnetic-free energy deficit -> negative pressure;
// - isothermal: zero density -> the infinite velocity it produces;
// - newtonian MHD: a non-finite cell-B poisons the stripped pressure;
// - isothermal MHD: a non-finite cell-B with a clean gas state — the
//   magnetic clause alone rejects, and the gas outputs stay exact.
// each family also runs a physical control that writes status zero.
// =============================================================================

use symbi_aot::NamedKernel;

const INVALID: f64 = 64.0;
const GAMMA: f64 = 5.0 / 3.0;

fn adiabatic_1d(den: &[f64], mom: &[f64], nrg: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = den.len();
    let (mut rho, mut vel, mut pre) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut status = vec![-1.0; n];
    NamedKernel::new("adiabatic_c2p_1d")
        .input("cons.den", den)
        .input("cons.mom_0", mom)
        .input("cons.nrg", nrg)
        .output("prim.rho", &mut rho)
        .output("prim.vel_0", &mut vel)
        .output("prim.pre", &mut pre)
        .output("scratch", &mut status)
        .grid(&[n as u32])
        .dom_lo(&[0])
        .scalar("gamma", GAMMA)
        .run();
    (rho, vel, pre, status)
}

#[test]
fn adiabatic_energy_deficit_rejects_with_raw_candidate() {
    // cell 0: physical. cell 1: kinetic energy exceeds the total, so the
    // recovered pressure is negative.
    let den = [1.0, 1.0];
    let mom = [0.5, 10.0];
    let nrg = [2.0, 1.0];
    let (rho, _vel, pre, status) = adiabatic_1d(&den, &mom, &nrg);
    assert_eq!(status[0], 0.0);
    assert_eq!(status[1], INVALID);
    // the candidate stays the raw unfloored computation.
    assert!(pre[1] < 0.0, "raw negative pressure survives: {}", pre[1]);
    assert_eq!(rho[1], 1.0);
}

fn iso_1d(den: &[f64], mom: &[f64], cs2: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = den.len();
    let (mut rho, mut vel, mut pre) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut status = vec![-1.0; n];
    NamedKernel::new("iso_c2p_1d")
        .input("cons.den", den)
        .input("cons.mom_0", mom)
        .input("cs2", cs2)
        .output("prim.rho", &mut rho)
        .output("prim.vel_0", &mut vel)
        .output("prim.pre", &mut pre)
        .output("scratch", &mut status)
        .grid(&[n as u32])
        .dom_lo(&[0])
        .run();
    let _ = pre;
    (rho, vel, status)
}

#[test]
fn isothermal_zero_density_rejects_with_infinite_velocity_preserved() {
    let den = [1.0, 0.0];
    let mom = [0.5, 1.0];
    let cs2 = [1.0, 1.0];
    let (rho, vel, status) = iso_1d(&den, &mom, &cs2);
    assert_eq!(status[0], 0.0);
    assert_eq!(status[1], INVALID);
    // the raw IEEE division survives on the candidate.
    assert_eq!(rho[1], 0.0);
    assert!(
        vel[1].is_infinite(),
        "raw infinite velocity survives: {}",
        vel[1]
    );
}

fn mhd_1d(
    name: &str,
    den: &[f64],
    mom: &[[f64; 2]; 3],
    nrg: Option<&[f64]>,
    mag: &[[f64; 2]; 3],
    has_pre: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = den.len();
    let (mut rho, mut v0, mut pre) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let (mut v1, mut v2) = (vec![0.0; n], vec![0.0; n]);
    let mut status = vec![-1.0; n];
    let grid = [n as u32];
    let dom = [0];
    let mut k = NamedKernel::new(name);
    k = k.input("cons.den", den);
    k = k.input("cons.mom_0", &mom[0]);
    k = k.input("cons.mom_1", &mom[1]);
    k = k.input("cons.mom_2", &mom[2]);
    if let Some(nrg) = nrg {
        k = k.input("cons.nrg", nrg);
    }
    k = k.input("cons.mag_0", &mag[0]);
    k = k.input("cons.mag_1", &mag[1]);
    k = k.input("cons.mag_2", &mag[2]);
    k = k.output("prim.rho", &mut rho);
    k = k.output("prim.vel_0", &mut v0);
    k = k.output("prim.vel_1", &mut v1);
    k = k.output("prim.vel_2", &mut v2);
    if has_pre {
        k = k.output("prim.pre", &mut pre);
    }
    k = k.output("scratch", &mut status);
    k = k.grid(&grid);
    k = k.dom_lo(&dom);
    if has_pre {
        k = k.scalar("gamma", GAMMA);
    }
    k.run();
    (rho, v0, pre, status)
}

#[test]
fn newtonian_mhd_nonfinite_field_rejects_with_raw_candidate() {
    let den = [1.0, 1.0];
    let mom = [[0.5, 0.5], [0.0, 0.0], [0.0, 0.0]];
    let nrg = [2.0, 2.0];
    let mag = [[0.1, f64::NAN], [0.0, 0.0], [0.0, 0.0]];
    let (_rho, _v0, pre, status) = mhd_1d("nmhd_c2p_1d", &den, &mom, Some(&nrg), &mag, true);
    assert_eq!(status[0], 0.0);
    assert_eq!(status[1], INVALID);
    // the NaN field poisons the stripped pressure; the raw NaN survives.
    assert!(pre[1].is_nan(), "raw NaN pressure survives: {}", pre[1]);
}

#[test]
fn isothermal_mhd_magnetic_clause_rejects_alone() {
    // the gas state is clean on both cells; cell 1 carries a NaN cell-B, so
    // the rejection comes from the magnetic finiteness clause alone and the
    // gas candidate outputs stay exact.
    let den = [1.0, 2.0];
    let mom = [[0.5, 1.0], [0.0, 0.0], [0.0, 0.0]];
    let mag = [[0.1, f64::NAN], [0.0, 0.0], [0.0, 0.0]];
    let (rho, v0, _pre, status) = mhd_1d("imhd_c2p_1d", &den, &mom, None, &mag, false);
    assert_eq!(status[0], 0.0);
    assert_eq!(status[1], INVALID);
    assert_eq!(rho[1], 2.0);
    assert_eq!(v0[1], 0.5);
}
