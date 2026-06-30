// =============================================================================
// carrier_oracle.rs
//
// the CARRIER ORACLE: a kernel test whose reference is the SAME carrier-generic
// physics run natively at S = f64 — never hand-derived algebra. the adiabatic c2p
// builder traces symbi-hydro's `Cons::to_primitive` at S = Gv; here we run that one
// physics source at S = f64 and assert the Gv kernel (traced -> lowered -> CPU
// interpreted) reproduces it. it catches any trace/lower/interp divergence AND
// deletes the duplicated expected-value math the hand-written tests carry.
//
// two flavours:
//   - round-trip:  prim --to_conserved (f64)--> cons --c2p (Gv kernel)--> prim'   ;  prim' == prim
//   - equivalence: cons --to_primitive (f64)--> prim_ref   ==   cons --c2p (Gv kernel)--> prim
// =============================================================================

mod harness;
use harness::{KernelRun, Out};

use symbi_algebra::Tensor;
use symbi_discretize::{
    adiabatic_c2p_gv, adiabatic_flux_gv, adiabatic_hllc_flux_gv, imhd_c2p_gv, imhd_flux_gv,
    imhd_hlld_flux_gv, imhd_wave_speed_map_gv, iso_c2p_gv, iso_flux_gv, iso_wave_speed_map_gv,
    nmhd_c2p_gv, nmhd_flux_gv, nmhd_hllc_flux_gv, nmhd_hlld_flux_gv, nmhd_wave_speed_map_gv,
    rmhd_c2p_gv, rmhd_flux_gv, rmhd_hllc_flux_gv, rmhd_hlld_flux_gv, rmhd_wave_speed_map_gv,
    srhd_c2p_gv, srhd_flux_gv, srhd_hllc_flux_gv, srhd_wave_speed_map_gv, Coords, GvKernel, Spacing, Spacetime,
};
use symbi_hydro::eos::{IdealGas, Isothermal};
use symbi_hydro::energy::Zero;
use symbi_hydro::mhd_state::{IsoMhdCons, IsoMhdPrim, MhdCons, MhdPrim};
use symbi_hydro::isothermal_mhd::IsothermalMhd;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::regime::Regime;
use symbi_hydro::riemann::{hllc, hllc_rmhd, hllc_srhd, hlld_rmhd, hlle};
use symbi_hydro::dissipation::ShockwaveLimiter;
use symbi_hydro::state::PrimG;
use symbi_hydro::rmhd::{Rmhd, rmhd_magnetosonic_cfl_speeds};
use symbi_hydro::srhd::Srhd;
use symbi_hydro::state::{Cons, Prim};
use symbi_ir::graph::NodeId;
use symbi_ir::MeshScalar;

const N: usize = 6;
const GAMMA: f64 = 1.4;
const NCOMP: usize = 2; // carry a transverse velocity component (c2p is pointwise; grid is 1D)

const EOS: IdealGas<f64> = IdealGas { gamma: GAMMA };

// a smooth family of admissible primitive states across the grid (rho, pre > 0).
fn prim_at(i: usize) -> Prim<f64, NCOMP> {
    let x = i as f64;
    let mut vel = Tensor::zeros();
    vel[0] = 0.1 + 0.05 * x;
    vel[1] = -0.2 + 0.03 * x;
    Prim { rho: 1.0 + 0.1 * x, vel, pre: 0.5 + 0.15 * x }
}

// the conserved state that prim_at(i) maps to, via the native f64 physics (the single source).
fn cons_at(i: usize) -> Cons<f64, NCOMP> {
    prim_at(i).to_conserved(&EOS)
}

#[test]
fn adiabatic_c2p_round_trips_against_native_physics() {
    // reference = the INPUT prim; impl = native p2c then the Gv c2p kernel. zero expected algebra.
    let out = KernelRun::new(adiabatic_c2p_gv::<NCOMP>())
        .grid([N])
        .field_with("cons_den", |c| cons_at(c[0]).den)
        .field_with("cons_mom_0", |c| cons_at(c[0]).mom[0])
        .field_with("cons_mom_1", |c| cons_at(c[0]).mom[1])
        .field_with("cons_nrg", |c| cons_at(c[0]).nrg)
        .scalars(&[("gamma", GAMMA)])
        .run();

    for i in 0..N {
        let p = prim_at(i);
        out.expect(
            [i],
            &[
                ("prim_rho", p.rho),
                ("prim_vel_0", p.vel[0]),
                ("prim_vel_1", p.vel[1]),
                ("prim_pre", p.pre),
            ],
            1e-12,
        );
    }
}

#[test]
fn adiabatic_c2p_matches_native_carrier() {
    // reference = native f64 c2p of an arbitrary admissible cons; impl = the Gv kernel. the
    // expected prim is the physics fn ITSELF at S = f64, not a re-derivation of its algebra.
    fn cons_raw(i: usize) -> Cons<f64, NCOMP> {
        let x = i as f64;
        let mut mom = Tensor::zeros();
        mom[0] = 0.3 + 0.1 * x;
        mom[1] = -0.15 + 0.05 * x;
        Cons { den: 1.2 + 0.2 * x, mom, nrg: 2.0 + 0.3 * x }
    }

    let out = KernelRun::new(adiabatic_c2p_gv::<NCOMP>())
        .grid([N])
        .field_with("cons_den", |c| cons_raw(c[0]).den)
        .field_with("cons_mom_0", |c| cons_raw(c[0]).mom[0])
        .field_with("cons_mom_1", |c| cons_raw(c[0]).mom[1])
        .field_with("cons_nrg", |c| cons_raw(c[0]).nrg)
        .scalars(&[("gamma", GAMMA)])
        .run();

    for i in 0..N {
        let want = cons_raw(i).to_primitive(&EOS); // native f64 — the single-source reference
        out.expect(
            [i],
            &[
                ("prim_rho", want.rho),
                ("prim_vel_0", want.vel[0]),
                ("prim_vel_1", want.vel[1]),
                ("prim_pre", want.pre),
            ],
            1e-12,
        );
    }
}

// =============================================================================
// iso c2p — the LOCALLY-isothermal recovery: `Cons::to_primitive` with the `Isothermal`
// eos, which reads cs^2 from the nrg slot (the gv builder feeds the prescribed per-cell
// `cs2` field through that slot). recovery is `rho = den`, `vel = mom/den`, `pre = cs2*rho`.
// the kernel reads `cons_den` / `cons_mom_{k}` / `cs2` — NO gamma scalar (cs2 is a field).
// round-trip: a known prim -> (den, mom, cs2=pre/rho) -> the gv kernel -> the input prim.
// =============================================================================

const ISO_NCOMP: usize = 2;
// the iso eos's `cs` is irrelevant — `recover_pressure` reads cs^2 from nrg, not self.cs.
const ISO_EOS: Isothermal<f64> = Isothermal { cs: 0.0 };

// a smooth family of admissible iso primitives + their prescribed per-cell sound speed.
fn iso_prim_at(i: usize) -> (Prim<f64, ISO_NCOMP>, f64) {
    let x = i as f64;
    let mut vel = Tensor::zeros();
    vel[0] = 0.15 + 0.04 * x;
    vel[1] = -0.1 + 0.02 * x;
    let rho = 1.0 + 0.1 * x;
    let cs2 = 0.6 + 0.05 * x; // the local temperature (sound-speed-squared)
    (Prim { rho, vel, pre: cs2 * rho }, cs2)
}

// the conserved iso state carrying cs2 in the nrg slot — built by the native f64 physics.
fn iso_cons_at(i: usize) -> Cons<f64, ISO_NCOMP> {
    let (p, _cs2) = iso_prim_at(i);
    // nrg = cs^2 via Isothermal::conserved_energy uses self.cs (=0); instead set nrg = cs2
    // directly to model the LOCALLY-isothermal field, exactly as the gv builder feeds it.
    let mut c = p.to_conserved(&ISO_EOS);
    c.nrg = iso_prim_at(i).1; // the prescribed per-cell cs^2 field
    c
}

#[test]
fn iso_c2p_round_trips_against_native_physics() {
    // reference = the INPUT prim; impl = the Gv iso c2p kernel on (den, mom, cs2). the native
    // `Cons::to_primitive(&Isothermal)` IS what the builder traces — round-trip, zero algebra.
    let out = KernelRun::new(iso_c2p_gv::<ISO_NCOMP>())
        .grid([N])
        .field_with("cons_den", |c| iso_cons_at(c[0]).den)
        .field_with("cons_mom_0", |c| iso_cons_at(c[0]).mom[0])
        .field_with("cons_mom_1", |c| iso_cons_at(c[0]).mom[1])
        .field_with("cs2", |c| iso_cons_at(c[0]).nrg)
        .run();

    for i in 0..N {
        let p = iso_prim_at(i).0;
        out.expect(
            [i],
            &[
                ("prim_rho", p.rho),
                ("prim_vel_0", p.vel[0]),
                ("prim_vel_1", p.vel[1]),
                ("prim_pre", p.pre),
            ],
            1e-12,
        );
    }
}

#[test]
fn iso_c2p_matches_native_carrier() {
    // reference = native f64 `Cons::to_primitive(&Isothermal)` of an arbitrary admissible cons
    // (cs2 in the nrg slot); impl = the Gv kernel. direct equivalence — the physics fn ITSELF.
    fn cons_raw(i: usize) -> Cons<f64, ISO_NCOMP> {
        let x = i as f64;
        let mut mom = Tensor::zeros();
        mom[0] = 0.3 + 0.1 * x;
        mom[1] = -0.12 + 0.04 * x;
        Cons { den: 1.2 + 0.2 * x, mom, nrg: 0.5 + 0.05 * x } // nrg = the prescribed cs^2
    }

    let out = KernelRun::new(iso_c2p_gv::<ISO_NCOMP>())
        .grid([N])
        .field_with("cons_den", |c| cons_raw(c[0]).den)
        .field_with("cons_mom_0", |c| cons_raw(c[0]).mom[0])
        .field_with("cons_mom_1", |c| cons_raw(c[0]).mom[1])
        .field_with("cs2", |c| cons_raw(c[0]).nrg)
        .run();

    for i in 0..N {
        let want = cons_raw(i).to_primitive(&ISO_EOS); // recover_pressure reads cs^2 from nrg
        out.expect(
            [i],
            &[
                ("prim_rho", want.rho),
                ("prim_vel_0", want.vel[0]),
                ("prim_vel_1", want.vel[1]),
                ("prim_pre", want.pre),
            ],
            1e-12,
        );
    }
}

// =============================================================================
// srhd c2p — the ITERATIVE relativistic recovery (`srhd_recover` = a carrier-generic Newton
// on the pressure root). round-trip ONLY (it guarantees admissibility): a KNOWN admissible
// prim (subluminal velocity, positive pressure) -> cons via `Srhd::to_conserved::<f64>` ->
// the Gv kernel (IterateInline, count=20 == build.rs) -> the input prim. tolerance ~1e-9
// (iterative, not ULP-exact). the native reference is the INPUT prim — the round-trip
// closes through the SAME `srhd_recover` the builder traces at S=Gv.
// =============================================================================

const SRHD_GAMMA: f64 = 5.0 / 3.0;
const SRHD_EOS: IdealGas<f64> = IdealGas { gamma: SRHD_GAMMA };
const SRHD_ITERS: usize = 20; // matches build.rs's baked Newton count

// a smooth family of admissible (subluminal, positive-pressure) relativistic primitives.
fn srhd_prim_at(i: usize) -> Prim<f64, 3> {
    let x = i as f64;
    let mut vel = Tensor::zeros();
    vel[0] = 0.2 + 0.05 * x; // |v| stays well below 1 across the grid
    vel[1] = -0.15 + 0.02 * x;
    vel[2] = 0.1;
    Prim { rho: 1.0 + 0.1 * x, vel, pre: 0.5 + 0.15 * x }
}

fn srhd_cons_at(i: usize) -> Cons<f64, 3> {
    Srhd.to_conserved(&SRHD_EOS, &srhd_prim_at(i)) // native f64 p2c — the single source
}

#[test]
fn srhd_c2p_round_trips_against_native_physics() {
    let out = KernelRun::new(srhd_c2p_gv::<3>(SRHD_ITERS))
        .grid([N])
        .field_with("cons_den", |c| srhd_cons_at(c[0]).den)
        .field_with("cons_mom_0", |c| srhd_cons_at(c[0]).mom[0])
        .field_with("cons_mom_1", |c| srhd_cons_at(c[0]).mom[1])
        .field_with("cons_mom_2", |c| srhd_cons_at(c[0]).mom[2])
        .field_with("cons_nrg", |c| srhd_cons_at(c[0]).nrg)
        .scalars(&[("gamma", SRHD_GAMMA)])
        .run();

    for i in 0..N {
        let p = srhd_prim_at(i);
        out.expect(
            [i],
            &[
                ("prim_rho", p.rho),
                ("prim_vel_0", p.vel[0]),
                ("prim_vel_1", p.vel[1]),
                ("prim_vel_2", p.vel[2]),
                ("prim_pre", p.pre),
            ],
            1e-9,
        );
    }
}

// =============================================================================
// rmhd c2p — the ITERATIVE KKC false-position recovery (`rmhd_recover`, a 6-state bracketed
// iterate over the magnetosonic master function). round-trip ONLY: a KNOWN admissible prim
// (incl. a B field) -> cons via `Rmhd::to_conserved::<f64>` -> the Gv kernel (multi-acc
// IterateInline, count=100 == build.rs) -> the input prim. B passes through unchanged (it is
// CT-evolved, not recovered). tolerance ~1e-9 (iterative). RMHD vectors are always 3-comp.
// =============================================================================

const RMHD_GAMMA: f64 = 5.0 / 3.0;
const RMHD_EOS: IdealGas<f64> = IdealGas { gamma: RMHD_GAMMA };
const RMHD_ITERS: usize = 100; // matches build.rs's baked false-position count

// a smooth family of admissible RMHD primitives (subluminal, positive pressure, modest B).
fn rmhd_prim_at(i: usize) -> MhdPrim<f64, 3> {
    let x = i as f64;
    let mut vel = Tensor::zeros();
    vel[0] = 0.15 + 0.03 * x;
    vel[1] = -0.1 + 0.02 * x;
    vel[2] = 0.05;
    let mut mag = Tensor::zeros();
    mag[0] = 0.2;
    mag[1] = 0.3 + 0.02 * x;
    mag[2] = -0.1;
    MhdPrim { hydro: Prim { rho: 1.0 + 0.1 * x, vel, pre: 0.5 + 0.1 * x }, mag }
}

fn rmhd_cons_at(i: usize) -> MhdCons<f64, 3> {
    Rmhd.to_conserved(&RMHD_EOS, &rmhd_prim_at(i)) // native f64 p2c — the single source
}

#[test]
fn rmhd_c2p_round_trips_against_native_physics() {
    let out = KernelRun::new(rmhd_c2p_gv(RMHD_ITERS))
        .grid([N])
        .field_with("cons_den", |c| rmhd_cons_at(c[0]).den)
        .field_with("cons_mom_0", |c| rmhd_cons_at(c[0]).mom[0])
        .field_with("cons_mom_1", |c| rmhd_cons_at(c[0]).mom[1])
        .field_with("cons_mom_2", |c| rmhd_cons_at(c[0]).mom[2])
        .field_with("cons_nrg", |c| rmhd_cons_at(c[0]).nrg)
        .field_with("cons_mag_0", |c| rmhd_cons_at(c[0]).mag[0])
        .field_with("cons_mag_1", |c| rmhd_cons_at(c[0]).mag[1])
        .field_with("cons_mag_2", |c| rmhd_cons_at(c[0]).mag[2])
        .scalars(&[("gamma", RMHD_GAMMA)])
        .run();

    for i in 0..N {
        let p = rmhd_prim_at(i);
        out.expect(
            [i],
            &[
                ("prim_rho", p.rho),
                ("prim_vel_0", p.vel[0]),
                ("prim_vel_1", p.vel[1]),
                ("prim_vel_2", p.vel[2]),
                ("prim_pre", p.pre),
            ],
            1e-9,
        );
    }
}

// =============================================================================
// face flux — PLM reconstruction (a stencil) composed with `riemann::hlle`. for a UNIFORM
// state the reconstruction returns the cell value (zero slope) and HLLE returns the PHYSICAL
// flux = the regime's native `to_flux::<f64>(prim, nhat, eos)` along the sweep axis. so: fill
// a uniform admissible state, compute only an interior cell (recon reads i-2..i+1, so start at
// i>=2), reference = the native flux. theta=1 == plain minmod (the hydro default).
// =============================================================================

// the cartesian euler flux `adiabatic_flux_gv::<D>` / `srhd_flux_gv::<D>` is an `ndim == D`
// stencil kernel swept along axis `dir`. recon reads i-2..i+1 along `dir`, so the buffer needs
// >= 4 cells on `dir` and we compute the SINGLE interior cell at `dir == 2` (indices 0..3
// uniform -> HLLE returns the physical flux). the non-swept axes are length 1.
const NSWEEP: usize = 4; // recon stencil width: i-2..i+1 around i=2 reads 0..3
const FCELL: usize = 2; // the interior cell along the sweep axis
fn flux_grid<const D: usize>(dir: usize) -> [usize; D] {
    let mut g = [1usize; D];
    g[dir] = NSWEEP;
    g
}
fn flux_window<const D: usize>(dir: usize) -> ([i32; D], [usize; D]) {
    let mut lo = [0i32; D];
    let mut size = [1usize; D];
    lo[dir] = FCELL as i32;
    size[dir] = 1; // compute exactly the interior cell
    (lo, size)
}
fn flux_cell<const D: usize>(dir: usize) -> [usize; D] {
    let mut c = [0usize; D];
    c[dir] = FCELL;
    c
}

// run a cartesian euler-family flux kernel on a UNIFORM state, computing only the interior
// cell along sweep `dir`. binds rho/vel_{0..D-1}/pre + gamma/theta; returns the run output.
fn run_uniform_euler_flux<const D: usize>(
    kernel: (GvKernel, Vec<(String, symbi_ir::FieldBind, NodeId)>),
    prim: &Prim<f64, D>,
    gamma: f64,
    dir: usize,
) -> Out {
    let mut fields: Vec<(&str, f64)> = vec![("prim_rho", prim.rho), ("prim_pre", prim.pre)];
    let vkeys: Vec<String> = (0..D).map(|k| format!("prim_v{k}")).collect();
    for k in 0..D {
        fields.push((vkeys[k].as_str(), prim.vel[k]));
    }
    let (lo, size) = flux_window::<D>(dir);
    // static-mesh binding: the flux's grid velocity is
    // `vface = mesh_adot_{dir} * (x_lo_dir + coord*dx_dir) + mesh_vtrans_{dir}`
    // (mesh_face_velocity_gv). both rates zero => vface = 0 => the kernel is
    // bit-identical to the no-motion flux, so the native (static) reference
    // holds; x_lo/dx are then immaterial. the mesh names come from `MeshScalar`,
    // the SAME source the trace declares them with — so this binding cannot
    // drift from the kernel.
    let adot = MeshScalar::Adot(dir as u8).name();
    let vtrans = MeshScalar::Vtrans(dir as u8).name();
    let x_lo = format!("x_lo_{dir}");
    let dx = format!("dx_{dir}");
    KernelRun::new(kernel)
        .grid(flux_grid::<D>(dir))
        .compute_window(lo, size)
        .fields(&fields)
        .scalars(&[
            ("gamma", gamma), ("theta", 1.0),
            (adot.as_str(), 0.0), (vtrans.as_str(), 0.0),
            (x_lo.as_str(), 0.0), (dx.as_str(), 1.0),
        ])
        .run()
}

#[test]
fn adiabatic_flux_matches_native_physics() {
    // reference = `Newtonian::to_flux::<f64>(prim, unit(0), IdealGas)`; impl = the Gv flux kernel
    // on a uniform state. the HLLE of L==R IS the physical flux — the single source.
    let prim = Prim::<f64, 2> { rho: 1.3, vel: Tensor::new([0.4, -0.2]), pre: 0.7 };
    let eos = IdealGas { gamma: GAMMA };
    let f = symbi_hydro::newtonian::Newtonian.to_flux(&prim, &Tensor::unit(0), &eos);
    let out = run_uniform_euler_flux::<2>(adiabatic_flux_gv::<2>(0), &prim, GAMMA, 0);
    out.expect(
        flux_cell::<2>(0),
        &[("flux_den", f.den), ("flux_mom_0", f.mom[0]), ("flux_mom_1", f.mom[1]), ("flux_nrg", f.nrg)],
        1e-12,
    );
}

#[test]
fn iso_flux_matches_native_physics() {
    // the iso flux IS the Newtonian flux at gamma=1 with `pre = cs2*rho` (the locally-iso
    // closure), MINUS the energy flux (iso has no energy law). reference = the native iso
    // physics: `IsoNewtonian` is the Newtonian regime, so `Newtonian::to_flux` at gamma=1 over
    // a prim whose pre = cs2*rho gives the iso mass+momentum flux (the builder's single source).
    let iso_gamma = 1.0;
    let cs2 = 0.6;
    let rho = 1.3;
    let prim = Prim::<f64, 2> { rho, vel: Tensor::new([0.4, -0.2]), pre: cs2 * rho };
    let eos = IdealGas { gamma: iso_gamma };
    let f = symbi_hydro::newtonian::Newtonian.to_flux(&prim, &Tensor::unit(0), &eos);
    let out = run_uniform_euler_flux::<2>(iso_flux_gv::<2>(0), &prim, iso_gamma, 0);
    // no flux_nrg write for iso.
    out.expect(
        flux_cell::<2>(0),
        &[("flux_den", f.den), ("flux_mom_0", f.mom[0]), ("flux_mom_1", f.mom[1])],
        1e-12,
    );
}

#[test]
fn srhd_flux_matches_native_physics() {
    // reference = `Srhd::to_flux::<f64>(prim, unit(0), IdealGas)`. the SRHD to_flux is algebraic
    // (no iteration) given a prim, so the uniform-state HLLE reproduces it to ~1e-12.
    let prim = Prim::<f64, 2> { rho: 1.0, vel: Tensor::new([0.3, -0.1]), pre: 1.0 };
    let eos = IdealGas { gamma: SRHD_GAMMA };
    let f = Srhd.to_flux(&prim, &Tensor::unit(0), &eos);
    let out = run_uniform_euler_flux::<2>(srhd_flux_gv::<2>(0), &prim, SRHD_GAMMA, 0);
    out.expect(
        flux_cell::<2>(0),
        &[("flux_den", f.den), ("flux_mom_0", f.mom[0]), ("flux_mom_1", f.mom[1]), ("flux_nrg", f.nrg)],
        1e-12,
    );
}

#[test]
fn rmhd_flux_matches_native_physics() {
    // reference = `Rmhd::to_flux::<f64>(prim, unit(0), IdealGas)` — algebraic given a prim (the
    // wave speeds the HLLE uses ARE the quartic, but to_flux itself is closed-form). impl = the
    // Gv rmhd flux on a uniform 3-component state. writes 8 fluxes (D, S_{0,1,2}, tau, B_{0,1,2}).
    let prim = MhdPrim::<f64, 3> {
        hydro: Prim { rho: 1.0, vel: Tensor::new([0.2, -0.1, 0.05]), pre: 1.0 },
        mag: Tensor::new([0.3, 0.2, -0.1]),
    };
    let eos = IdealGas { gamma: RMHD_GAMMA };
    let f = Rmhd.to_flux(&prim, &Tensor::unit(0), &eos);
    // rmhd_flux_gv reads the materialized per-cell Davis speeds (ws_l/ws_r), produced in the
    // live solver by rmhd_wave_speeds_cell_gv = `Rmhd::wave_speeds`. for a UNIFORM state HLLE
    // returns the physical flux for any s_l < 0 < s_r (the diffusive U_R - U_L term vanishes),
    // so binding the exact quartic speeds uniformly reproduces `to_flux`.
    let (sl, sr) = Rmhd.wave_speeds(&eos, &prim, &Tensor::unit(0));
    let fields: Vec<(&str, f64)> = vec![
        ("prim_rho", prim.rho),
        ("prim_v0", prim.vel[0]),
        ("prim_v1", prim.vel[1]),
        ("prim_v2", prim.vel[2]),
        ("prim_pre", prim.pre),
        ("prim_b0", prim.mag[0]),
        ("prim_b1", prim.mag[1]),
        ("prim_b2", prim.mag[2]),
        ("bface_n", prim.mag[0]),
        ("ws_l", sl),
        ("ws_r", sr),
    ];
    let out = KernelRun::new(rmhd_flux_gv(1, 0, 0))
        .grid([NSWEEP])
        .compute_window([FCELL as i32], [1])
        .fields(&fields)
        .scalars(&[("gamma", RMHD_GAMMA), ("theta", 1.0)])
        .run();
    out.expect(
        [FCELL],
        &[
            ("flux_den", f.den),
            ("flux_mom_0", f.mom[0]),
            ("flux_mom_1", f.mom[1]),
            ("flux_mom_2", f.mom[2]),
            ("flux_nrg", f.nrg),
            ("flux_mag_0", f.mag[0]),
            ("flux_mag_1", f.mag[1]),
            ("flux_mag_2", f.mag[2]),
        ],
        1e-12,
    );
}

// =============================================================================
// wave-speed maps — the map folds the per-cell carrier-generic `wave_speeds_axis` TOGETHER
// with the geometry inverse-width into `lambda = max_d (max(|sl|,|sr|) * inv_w_d)`. in general
// the output is physics*geometry and the geometry factor is NOT carrier-generic. BUT for a
// CARTESIAN-UNIFORM 1D grid the inverse width is the trivial known constant `inv_dx_0`; binding
// `inv_dx_0 = 1` (dx = 1) collapses the geometry to the identity, leaving `lambda = max(|sl|,
// |sr|)` — the PURE carrier-generic physics. that is a clean f64 reference, so we oracle the
// cartesian-uniform case. (curvilinear / non-uniform maps couple an in-kernel metric width
// with no carrier-generic f64 reference; their physics is already covered by the flux oracles
// + the rmhd_wave_speeds test, so we do NOT invent a geometry reference for them.)
// =============================================================================

const CART_1D: [Spacing; 1] = [Spacing::Uniform];
const AXES_1D: [usize; 1] = [0];

#[test]
fn iso_wave_speed_map_matches_native_physics() {
    // gamma=1.4 here drives the adiabatic Newtonian speed |v_0| + cs; with dx=1 the map IS
    // `max(|sl|,|sr|)` from `Newtonian::wave_speeds_axis`. (the same builder drives the iso CFL
    // at gamma=1; here we exercise it carrier-generically against the Newtonian f64 reference.)
    let (rho, v0, pre) = (1.3_f64, 0.4_f64, 0.7_f64);
    let eos = IdealGas { gamma: GAMMA };
    let prim = Prim::<f64, 3> { rho, vel: Tensor::new([v0, 0.0, 0.0]), pre };
    let (sl, sr) = symbi_hydro::newtonian::Newtonian.wave_speeds_axis(&eos, &prim, 0);
    let want = sl.abs().max(sr.abs()); // inv_dx_0 = 1
    // static mesh: per-axis grid velocity v_g = mesh_adot_0*xc + mesh_vtrans_0
    // (euler_wave_speed_map_gv). zero rates => v_g = 0 => |s - 0| = |s|,
    // bit-identical to the static reference; x_lo_0/dx_0 are then immaterial. mesh
    // names from `MeshScalar` (the same source the trace uses).
    let (adot0, vtrans0) = (MeshScalar::Adot(0).name(), MeshScalar::Vtrans(0).name());
    let out = KernelRun::new(iso_wave_speed_map_gv(Coords::Cartesian, &CART_1D, &AXES_1D, 1))
        .grid([N])
        .fields(&[("prim_rho", rho), ("prim_v0", v0), ("prim_pre", pre)])
        .scalars(&[("gamma", GAMMA), ("inv_dx_0", 1.0),
            (adot0.as_str(), 0.0), (vtrans0.as_str(), 0.0), ("x_lo_0", 0.0), ("dx_0", 1.0)])
        .run();
    out.expect([2], &[("lambda", want)], 1e-12);
}

#[test]
fn srhd_wave_speed_map_matches_native_physics() {
    // the relativistic Mignone-Bodo per-axis speed (`Srhd::wave_speeds_axis`); dx=1 -> the map
    // IS `max(|sl|,|sr|)`. the SAME core the SRHD flux's HLLE consumes.
    let (rho, v0, pre) = (1.0_f64, 0.3_f64, 1.0_f64);
    let eos = IdealGas { gamma: SRHD_GAMMA };
    let prim = Prim::<f64, 3> { rho, vel: Tensor::new([v0, 0.0, 0.0]), pre };
    let (sl, sr) = Srhd.wave_speeds_axis(&eos, &prim, 0);
    let want = sl.abs().max(sr.abs());
    // static mesh: zero grid velocity (see iso_wave_speed_map test above).
    let (adot0, vtrans0) = (MeshScalar::Adot(0).name(), MeshScalar::Vtrans(0).name());
    let out = KernelRun::new(srhd_wave_speed_map_gv(Coords::Cartesian, Spacetime::Minkowski, &CART_1D, &AXES_1D, 1))
        .grid([N])
        .fields(&[("prim_rho", rho), ("prim_v0", v0), ("prim_pre", pre)])
        .scalars(&[("gamma", SRHD_GAMMA), ("inv_dx_0", 1.0),
            (adot0.as_str(), 0.0), (vtrans0.as_str(), 0.0), ("x_lo_0", 0.0), ("dx_0", 1.0)])
        .run();
    out.expect([2], &[("lambda", want)], 1e-12);
}

#[test]
fn rmhd_wave_speed_map_matches_native_physics() {
    // the CFL map traces `rmhd_magnetosonic_cfl_speeds` (the cheap c_f^2 = c_s^2 + c_A^2 upper
    // bound), NOT the full Mignone & Del Zanna quartic — the quartic stays on the Riemann/flux
    // path only. so the oracle MUST be the same magnetosonic bound at native f64, not
    // `wave_speeds_axis` (which over-tightens to the exact characteristic and disagrees by ~2%).
    // dx=1 -> the map IS `max(|sl|,|sr|)`; the bound reads the full 3-velocity + 3-B-field.
    let prim = MhdPrim::<f64, 3> {
        hydro: Prim { rho: 1.0, vel: Tensor::new([0.2, -0.1, 0.05]), pre: 1.0 },
        mag: Tensor::new([0.3, 0.2, -0.1]),
    };
    let eos = IdealGas { gamma: RMHD_GAMMA };
    let nhat = Tensor::<f64, 3>::unit(0);
    let (sl, sr) = rmhd_magnetosonic_cfl_speeds(&eos, &prim, &nhat);
    let want = sl.abs().max(sr.abs());
    let out = KernelRun::new(rmhd_wave_speed_map_gv(Coords::Cartesian, &CART_1D, &AXES_1D, 1))
        .grid([N])
        .fields(&[
            ("prim_rho", prim.rho),
            ("prim_v0", prim.vel[0]),
            ("prim_v1", prim.vel[1]),
            ("prim_v2", prim.vel[2]),
            ("prim_pre", prim.pre),
            ("prim_b0", prim.mag[0]),
            ("prim_b1", prim.mag[1]),
            ("prim_b2", prim.mag[2]),
        ])
        .scalars(&[("gamma", RMHD_GAMMA), ("inv_dx_0", 1.0)])
        .run();
    out.expect([2], &[("lambda", want)], 1e-12);
}

// =============================================================================
// newtonian MHD — the non-relativistic ideal-MHD regime. ALGEBRAIC c2p (no
// iteration), closed-form fast-magnetosonic speeds. these oracle the SAME
// `NewtonianMhd` carrier-generic physics validated at f64 in symbi-hydro:
//   - c2p round-trip: prim --p2c (f64)--> cons --nmhd_c2p_gv kernel--> prim'  (1e-12,
//     algebraic, NOT iterative -> ULP-tight, unlike RMHD's 1e-9).
//   - flux: uniform state -> HLLE returns the physical `NewtonianMhd::to_flux`.
//   - wave-speed map: dx=1 -> lambda == max(|sl|,|sr|) from the exact magnetosonic.
// proves the regime traces, lowers, CPU-interprets, and bit-matches f64 (CLAUDE.md 4.3).
// =============================================================================

const NMHD_GAMMA: f64 = 5.0 / 3.0;
const NMHD_EOS: IdealGas<f64> = IdealGas { gamma: NMHD_GAMMA };

// a smooth family of admissible newtonian-MHD primitives (positive rho/pre, modest B).
fn nmhd_prim_at(i: usize) -> MhdPrim<f64, 3> {
    let x = i as f64;
    let mut vel = Tensor::zeros();
    vel[0] = 0.15 + 0.03 * x;
    vel[1] = -0.1 + 0.02 * x;
    vel[2] = 0.05;
    let mut mag = Tensor::zeros();
    mag[0] = 0.2;
    mag[1] = 0.3 + 0.02 * x;
    mag[2] = -0.1;
    MhdPrim { hydro: Prim { rho: 1.0 + 0.1 * x, vel, pre: 0.5 + 0.1 * x }, mag }
}

fn nmhd_cons_at(i: usize) -> MhdCons<f64, 3> {
    NewtonianMhd.to_conserved(&NMHD_EOS, &nmhd_prim_at(i)) // native f64 p2c — the single source
}

#[test]
fn nmhd_c2p_round_trips_against_native_physics() {
    // algebraic recovery (nmhd_recover) -> ULP-tight round-trip. B passes through.
    let out = KernelRun::new(nmhd_c2p_gv())
        .grid([N])
        .field_with("cons_den", |c| nmhd_cons_at(c[0]).den)
        .field_with("cons_mom_0", |c| nmhd_cons_at(c[0]).mom[0])
        .field_with("cons_mom_1", |c| nmhd_cons_at(c[0]).mom[1])
        .field_with("cons_mom_2", |c| nmhd_cons_at(c[0]).mom[2])
        .field_with("cons_nrg", |c| nmhd_cons_at(c[0]).nrg)
        .field_with("cons_mag_0", |c| nmhd_cons_at(c[0]).mag[0])
        .field_with("cons_mag_1", |c| nmhd_cons_at(c[0]).mag[1])
        .field_with("cons_mag_2", |c| nmhd_cons_at(c[0]).mag[2])
        .scalars(&[("gamma", NMHD_GAMMA)])
        .run();

    for i in 0..N {
        let p = nmhd_prim_at(i);
        out.expect(
            [i],
            &[
                ("prim_rho", p.rho),
                ("prim_vel_0", p.vel[0]),
                ("prim_vel_1", p.vel[1]),
                ("prim_vel_2", p.vel[2]),
                ("prim_pre", p.pre),
            ],
            1e-12,
        );
    }
}

#[test]
fn nmhd_flux_matches_native_physics() {
    // reference = `NewtonianMhd::to_flux::<f64>(prim, unit(0), IdealGas)` — algebraic given a
    // prim. impl = the Gv nmhd flux on a uniform 3-component state (HLLE of L==R IS the physical
    // flux). writes 8 fluxes (D, S_{0,1,2}, nrg, B_{0,1,2}).
    let prim = MhdPrim::<f64, 3> {
        hydro: Prim { rho: 1.0, vel: Tensor::new([0.2, -0.1, 0.05]), pre: 1.0 },
        mag: Tensor::new([0.3, 0.2, -0.1]),
    };
    let eos = IdealGas { gamma: NMHD_GAMMA };
    let f = NewtonianMhd.to_flux(&prim, &Tensor::unit(0), &eos);
    let fields: Vec<(&str, f64)> = vec![
        ("prim_rho", prim.rho),
        ("prim_v0", prim.vel[0]),
        ("prim_v1", prim.vel[1]),
        ("prim_v2", prim.vel[2]),
        ("prim_pre", prim.pre),
        ("prim_b0", prim.mag[0]),
        ("prim_b1", prim.mag[1]),
        ("prim_b2", prim.mag[2]),
        ("bface_n", prim.mag[0]),
    ];
    let out = KernelRun::new(nmhd_flux_gv(1, 0, 0))
        .grid([NSWEEP])
        .compute_window([FCELL as i32], [1])
        .fields(&fields)
        .scalars(&[("gamma", NMHD_GAMMA), ("theta", 1.0)])
        .run();
    out.expect(
        [FCELL],
        &[
            ("flux_den", f.den),
            ("flux_mom_0", f.mom[0]),
            ("flux_mom_1", f.mom[1]),
            ("flux_mom_2", f.mom[2]),
            ("flux_nrg", f.nrg),
            ("flux_mag_0", f.mag[0]),
            ("flux_mag_1", f.mag[1]),
            ("flux_mag_2", f.mag[2]),
        ],
        1e-12,
    );
}

#[test]
fn nmhd_wave_speed_map_matches_native_physics() {
    // the CFL map traces the EXACT closed-form magnetosonic `NewtonianMhd::wave_speeds` (cheap,
    // so no separate bound is needed — unlike RMHD). dx=1 -> the map IS `max(|sl|,|sr|)`.
    let prim = MhdPrim::<f64, 3> {
        hydro: Prim { rho: 1.0, vel: Tensor::new([0.2, -0.1, 0.05]), pre: 1.0 },
        mag: Tensor::new([0.3, 0.2, -0.1]),
    };
    let eos = IdealGas { gamma: NMHD_GAMMA };
    let nhat = Tensor::<f64, 3>::unit(0);
    let (sl, sr) = NewtonianMhd.wave_speeds(&eos, &prim, &nhat);
    let want = sl.abs().max(sr.abs());
    let out = KernelRun::new(nmhd_wave_speed_map_gv(Coords::Cartesian, &CART_1D, &AXES_1D, 1))
        .grid([N])
        .fields(&[
            ("prim_rho", prim.rho),
            ("prim_v0", prim.vel[0]),
            ("prim_v1", prim.vel[1]),
            ("prim_v2", prim.vel[2]),
            ("prim_pre", prim.pre),
            ("prim_b0", prim.mag[0]),
            ("prim_b1", prim.mag[1]),
            ("prim_b2", prim.mag[2]),
        ])
        .scalars(&[("gamma", NMHD_GAMMA), ("inv_dx_0", 1.0)])
        .run();
    out.expect([2], &[("lambda", want)], 1e-12);
}

#[test]
fn nmhd_builders_render_to_cpu_and_cuda() {
    // the LOWERABILITY half of the carrier gate (CLAUDE.md 4.3): all three NMHD
    // builders must emit non-empty CPU (rust) AND CUDA source — write-once-run-
    // everywhere at the source level. the GPU emit is the whole point of NMHD.
    KernelRun::new(nmhd_c2p_gv()).grid([N]).assert_lowers();
    KernelRun::new(nmhd_flux_gv(1, 0, 0)).grid([NSWEEP]).assert_lowers();
    KernelRun::new(nmhd_hllc_flux_gv(1, 0, 0)).grid([NSWEEP]).assert_lowers();
    KernelRun::new(nmhd_hlld_flux_gv(1, 0, 0)).grid([NSWEEP]).assert_lowers();
    KernelRun::new(nmhd_wave_speed_map_gv(Coords::Cartesian, &CART_1D, &AXES_1D, 1))
        .grid([N])
        .assert_lowers();
}

#[test]
fn nmhd_hllc_hlld_flux_match_native_physics_on_uniform_state() {
    // consistency: for a uniform state the HLLC / HLLD flux collapses to F(U) = to_flux,
    // exactly like HLLE. validates that the select-heavy solvers TRACE + lower + interp
    // and bit-match the native f64 physics through the Gv carrier.
    let prim = MhdPrim::<f64, 3> {
        hydro: Prim { rho: 1.0, vel: Tensor::new([0.2, -0.1, 0.05]), pre: 1.0 },
        mag: Tensor::new([0.3, 0.2, -0.1]),
    };
    let eos = IdealGas { gamma: NMHD_GAMMA };
    let f = NewtonianMhd.to_flux(&prim, &Tensor::unit(0), &eos);
    let fields: Vec<(&str, f64)> = vec![
        ("prim_rho", prim.rho),
        ("prim_v0", prim.vel[0]),
        ("prim_v1", prim.vel[1]),
        ("prim_v2", prim.vel[2]),
        ("prim_pre", prim.pre),
        ("prim_b0", prim.mag[0]),
        ("prim_b1", prim.mag[1]),
        ("prim_b2", prim.mag[2]),
        ("bface_n", prim.mag[0]),
    ];
    let want = &[
        ("flux_den", f.den),
        ("flux_mom_0", f.mom[0]),
        ("flux_mom_1", f.mom[1]),
        ("flux_mom_2", f.mom[2]),
        ("flux_nrg", f.nrg),
        ("flux_mag_0", f.mag[0]),
        ("flux_mag_1", f.mag[1]),
        ("flux_mag_2", f.mag[2]),
    ];
    for (label, kernel) in [("hllc", nmhd_hllc_flux_gv(1, 0, 0)), ("hlld", nmhd_hlld_flux_gv(1, 0, 0))] {
        let out = KernelRun::new(kernel)
            .grid([NSWEEP])
            .compute_window([FCELL as i32], [1])
            .fields(&fields)
            .scalars(&[("gamma", NMHD_GAMMA), ("theta", 1.0)])
            .run();
        eprintln!("[nmhd {label}] uniform-state flux oracle");
        out.expect([FCELL], want, 1e-12);
    }
}

// =============================================================================
// ISOTHERMAL MHD carrier oracle (Mignone 2007) — the same gate as NMHD over the
// no-energy state: c2p writes (rho, v) only, flux writes (D, S, B) only, closure
// scalar is `cs`. the flux reads `bface_n` (the staggered normal-B coupling).
// =============================================================================
const IMHD_CS: f64 = 1.0;

fn imhd_prim_at(i: usize) -> IsoMhdPrim<f64, 3> {
    let x = i as f64;
    let mut vel = Tensor::zeros();
    vel[0] = 0.15 + 0.03 * x;
    vel[1] = -0.1 + 0.02 * x;
    vel[2] = 0.05;
    let mut mag = Tensor::zeros();
    mag[0] = 0.2;
    mag[1] = 0.3 + 0.02 * x;
    mag[2] = -0.1;
    IsoMhdPrim { hydro: PrimG { rho: 1.0 + 0.1 * x, vel, pre: Zero::default() }, mag }
}

fn imhd_cons_at(i: usize) -> IsoMhdCons<f64, 3> {
    IsothermalMhd.to_conserved(&Isothermal { cs: IMHD_CS }, &imhd_prim_at(i))
}

#[test]
fn imhd_c2p_round_trips_against_native_physics() {
    // trivial recovery (rho=den, v=mom/den) -> ULP-tight round-trip. B passes through. no nrg/pre.
    let out = KernelRun::new(imhd_c2p_gv())
        .grid([N])
        .field_with("cons_den", |c| imhd_cons_at(c[0]).den)
        .field_with("cons_mom_0", |c| imhd_cons_at(c[0]).mom[0])
        .field_with("cons_mom_1", |c| imhd_cons_at(c[0]).mom[1])
        .field_with("cons_mom_2", |c| imhd_cons_at(c[0]).mom[2])
        .field_with("cons_mag_0", |c| imhd_cons_at(c[0]).mag[0])
        .field_with("cons_mag_1", |c| imhd_cons_at(c[0]).mag[1])
        .field_with("cons_mag_2", |c| imhd_cons_at(c[0]).mag[2])
        .run();
    for i in 0..N {
        let p = imhd_prim_at(i);
        out.expect(
            [i],
            &[
                ("prim_rho", p.rho),
                ("prim_vel_0", p.vel[0]),
                ("prim_vel_1", p.vel[1]),
                ("prim_vel_2", p.vel[2]),
            ],
            1e-12,
        );
    }
}

#[test]
fn imhd_flux_and_hlld_match_native_physics_on_uniform_state() {
    // uniform state: HLLE / the 3-state HLLD collapse to F(U) = to_flux, bit-matching native
    // f64 physics through the Gv carrier. bface_n = the normal mag (the staggered coupling).
    let prim = IsoMhdPrim::<f64, 3> {
        hydro: PrimG { rho: 1.0, vel: Tensor::new([0.2, -0.1, 0.05]), pre: Zero::default() },
        mag: Tensor::new([0.3, 0.2, -0.1]),
    };
    let eos = Isothermal { cs: IMHD_CS };
    let f = IsothermalMhd.to_flux(&prim, &Tensor::unit(0), &eos);
    let fields: Vec<(&str, f64)> = vec![
        ("prim_rho", prim.rho),
        ("prim_v0", prim.vel[0]),
        ("prim_v1", prim.vel[1]),
        ("prim_v2", prim.vel[2]),
        ("prim_b0", prim.mag[0]),
        ("prim_b1", prim.mag[1]),
        ("prim_b2", prim.mag[2]),
        ("bface_n", prim.mag[0]),
    ];
    let want = &[
        ("flux_den", f.den),
        ("flux_mom_0", f.mom[0]),
        ("flux_mom_1", f.mom[1]),
        ("flux_mom_2", f.mom[2]),
        ("flux_mag_0", f.mag[0]),
        ("flux_mag_1", f.mag[1]),
        ("flux_mag_2", f.mag[2]),
    ];
    for (label, kernel) in [("hlle", imhd_flux_gv(1, 0, 0)), ("hlld", imhd_hlld_flux_gv(1, 0, 0))] {
        let out = KernelRun::new(kernel)
            .grid([NSWEEP])
            .compute_window([FCELL as i32], [1])
            .fields(&fields)
            .scalars(&[("cs", IMHD_CS), ("theta", 1.0)])
            .run();
        eprintln!("[imhd {label}] uniform-state flux oracle");
        out.expect([FCELL], want, 1e-12);
    }
}

#[test]
fn imhd_builders_render_to_cpu_and_cuda() {
    KernelRun::new(imhd_c2p_gv()).grid([N]).assert_lowers();
    KernelRun::new(imhd_flux_gv(1, 0, 0)).grid([NSWEEP]).assert_lowers();
    KernelRun::new(imhd_hlld_flux_gv(1, 0, 0)).grid([NSWEEP]).assert_lowers();
    KernelRun::new(imhd_wave_speed_map_gv(Coords::Cartesian, &CART_1D, &AXES_1D, 1))
        .grid([N])
        .assert_lowers();
}

// =============================================================================
// HLLC / HLLD carrier oracle — the highest-risk kernels (most select-heavy /
// NaN-prone): the contact-resolving HLLC family (adiabatic / srhd / rmhd) and the
// 5-wave rmhd HLLD. the review flagged these as having NO f64 == Gv oracle. the
// reference is the SAME riemann function run NATIVELY at S = f64 (never re-derived
// algebra) on the SAME reconstructed L/R the kernel sees. uniform state -> the
// PLM slope is zero so L == R == the cell value, which isolates the Riemann
// branch; the non-uniform limiter oracle below then drives the reconstruction
// select-branches. theta = 1 (plain minmod), the Standard shock-smoother arm
// (the trace-time default), and a static mesh (vface = 0) match the trace.
// =============================================================================

#[test]
fn adiabatic_hllc_flux_matches_native_physics_on_uniform_state() {
    // uniform state: hllc(L == R) collapses to F(U), bit-matching the native hllc at f64.
    let prim = Prim::<f64, 2> { rho: 1.3, vel: Tensor::new([0.4, -0.2]), pre: 0.7 };
    let eos = IdealGas { gamma: GAMMA };
    let f = hllc(&eos, &prim, &prim, &Tensor::unit(0), 0.0, ShockwaveLimiter::Standard);
    let out = run_uniform_euler_flux::<2>(adiabatic_hllc_flux_gv::<2>(0), &prim, GAMMA, 0);
    out.expect(
        flux_cell::<2>(0),
        &[("flux_den", f.den), ("flux_mom_0", f.mom[0]), ("flux_mom_1", f.mom[1]), ("flux_nrg", f.nrg)],
        1e-12,
    );
}

#[test]
fn srhd_hllc_flux_matches_native_physics_on_uniform_state() {
    // a grazing v^2 -> 1 state stresses the relativistic Lorentz factor + Mignone-Bodo
    // contact quadratic (the select-heavy region) at both carriers.
    let prim = Prim::<f64, 2> { rho: 1.0, vel: Tensor::new([0.9, -0.2]), pre: 1.0 };
    let eos = IdealGas { gamma: SRHD_GAMMA };
    let f = hllc_srhd(&eos, &prim, &prim, &Tensor::unit(0), 0.0, ShockwaveLimiter::Standard);
    let out = run_uniform_euler_flux::<2>(srhd_hllc_flux_gv::<2>(0), &prim, SRHD_GAMMA, 0);
    out.expect(
        flux_cell::<2>(0),
        &[("flux_den", f.den), ("flux_mom_0", f.mom[0]), ("flux_mom_1", f.mom[1]), ("flux_nrg", f.nrg)],
        1e-12,
    );
}

// the rmhd flux fields (8-component MHD primitive + the staggered normal-B coupling),
// bound uniformly. shared by the rmhd HLLC + HLLD uniform oracles.
fn rmhd_uniform_flux_fields(prim: &MhdPrim<f64, 3>) -> Vec<(&'static str, f64)> {
    vec![
        ("prim_rho", prim.rho),
        ("prim_v0", prim.vel[0]),
        ("prim_v1", prim.vel[1]),
        ("prim_v2", prim.vel[2]),
        ("prim_pre", prim.pre),
        ("prim_b0", prim.mag[0]),
        ("prim_b1", prim.mag[1]),
        ("prim_b2", prim.mag[2]),
        ("bface_n", prim.mag[0]),
    ]
}

fn rmhd_flux_want(f: &MhdCons<f64, 3>) -> [(&'static str, f64); 8] {
    [
        ("flux_den", f.den),
        ("flux_mom_0", f.mom[0]),
        ("flux_mom_1", f.mom[1]),
        ("flux_mom_2", f.mom[2]),
        ("flux_nrg", f.nrg),
        ("flux_mag_0", f.mag[0]),
        ("flux_mag_1", f.mag[1]),
        ("flux_mag_2", f.mag[2]),
    ]
}

#[test]
fn rmhd_hllc_flux_matches_native_physics_on_uniform_state() {
    // strongly magnetized, relativistic uniform state: hllc_rmhd(L == R) -> F(U), bit-matching
    // the native hllc_rmhd at f64 (the null vs non-null normal-B branch is taken identically).
    let prim = MhdPrim::<f64, 3> {
        hydro: Prim { rho: 1.0, vel: Tensor::new([0.4, -0.2, 0.1]), pre: 1.0 },
        mag: Tensor::new([0.6, 0.3, -0.2]),
    };
    let eos = IdealGas { gamma: RMHD_GAMMA };
    let f = hllc_rmhd(&Rmhd, &eos, &prim, &prim, &Tensor::unit(0), 0.0, ShockwaveLimiter::Standard);
    let out = KernelRun::new(rmhd_hllc_flux_gv(1, 0, 0))
        .grid([NSWEEP])
        .compute_window([FCELL as i32], [1])
        .fields(&rmhd_uniform_flux_fields(&prim))
        .scalars(&[("gamma", RMHD_GAMMA), ("theta", 1.0)])
        .run();
    out.expect([FCELL], &rmhd_flux_want(&f), 1e-12);
}

#[test]
fn rmhd_hlld_flux_matches_native_physics_on_uniform_state() {
    // the 5-wave HLLD — the most select-heavy / NaN-prone kernel (15-step secant on p*,
    // eager HLLE fallback, success-mask select). uniform state: hlld_rmhd(L == R) collapses
    // to F(U), bit-matching the native hlld_rmhd at f64 through the Gv carrier.
    let prim = MhdPrim::<f64, 3> {
        hydro: Prim { rho: 1.0, vel: Tensor::new([0.3, -0.15, 0.05]), pre: 1.0 },
        mag: Tensor::new([0.5, 0.3, -0.2]),
    };
    let eos = IdealGas { gamma: RMHD_GAMMA };
    let f = hlld_rmhd(&Rmhd, &eos, &prim, &prim, &Tensor::unit(0), 0.0);
    let out = KernelRun::new(rmhd_hlld_flux_gv(1, 0, 0))
        .grid([NSWEEP])
        .compute_window([FCELL as i32], [1])
        .fields(&rmhd_uniform_flux_fields(&prim))
        .scalars(&[("gamma", RMHD_GAMMA), ("theta", 1.0)])
        .run();
    out.expect([FCELL], &rmhd_flux_want(&f), 1e-12);
}

#[test]
fn hllc_hlld_builders_render_to_cpu_and_cuda() {
    // the lowerability half of the carrier gate (CLAUDE.md 4.3): every HLLC/HLLD builder
    // must emit non-empty CPU (rust) AND CUDA source. an unlowerable op panics here.
    KernelRun::new(adiabatic_hllc_flux_gv::<2>(0)).grid([1, NSWEEP]).assert_lowers();
    KernelRun::new(srhd_hllc_flux_gv::<2>(0)).grid([1, NSWEEP]).assert_lowers();
    KernelRun::new(rmhd_hllc_flux_gv(1, 0, 0)).grid([NSWEEP]).assert_lowers();
    KernelRun::new(rmhd_hlld_flux_gv(1, 0, 0)).grid([NSWEEP]).assert_lowers();
}

// =============================================================================
// NON-UNIFORM reconstruction oracle — every other flux oracle uses a UNIFORM state,
// so the PLM/theta-MC slope is zero and the limiter's select-branches are NEVER
// numerically exercised. here we drive a 4-cell stencil with SIGN-CHANGING,
// ASYMMETRIC slopes (a local extremum + a steep one-sided gradient) so minmod3's
// all_pos / all_neg / clamp-to-zero arms are all taken. the reference is the SAME
// theta-MC formula at f64 (the single source the substrate `plm_reconstruct_theta`
// and `plm_theta_gv` share) feeding the PUBLIC native `hlle`, asserted bit-equal to
// the Gv kernel evaluated on the f64 interpreter — the true f64 == Gv carrier check
// across the limiter branches.
// =============================================================================

// the theta-MC slope (matches plm_theta_gv's `minmod3((vc-vl)*theta, 0.5*(vr-vl), (vr-vc)*theta)`):
// the common-signed minimum-magnitude argument iff the three share a strict sign, else 0.
fn minmod3_f64(x: f64, y: f64, z: f64) -> f64 {
    let mn = x.min(y).min(z);
    let mx = x.max(y).max(z);
    if mn > 0.0 {
        mn
    } else if mx < 0.0 {
        mx
    } else {
        0.0
    }
}

// reconstruct (left_at_face_i, right_at_face_i) for the cell-centred series `q` at face index i,
// the f64 mirror of plm_theta_gv: left uses cells i-2..i, right uses cells i-1..i+1.
fn plm_theta_f64(q: &[f64], i: usize, theta: f64) -> (f64, f64) {
    let slope = |vl: f64, vc: f64, vr: f64| {
        minmod3_f64((vc - vl) * theta, 0.5 * (vr - vl), (vr - vc) * theta)
    };
    let left = q[i - 1] + 0.5 * slope(q[i - 2], q[i - 1], q[i]);
    let right = q[i] - 0.5 * slope(q[i - 1], q[i], q[i + 1]);
    (left, right)
}

#[test]
fn euler_flux_nonuniform_reconstruction_drives_limiter() {
    // a 4-cell density/pressure/velocity series chosen so the theta-MC slope hits each
    // minmod3 arm at the face cell FCELL: a local extremum (clamp to 0) on one field,
    // a monotone steep gradient (sign-consistent) on another. theta = 1.5 (compression
    // in [1,2]) makes the *theta vs *0.5 selection non-trivial.
    let theta = 1.5_f64;
    // the series are chosen so the theta-MC limiter takes ALL THREE minmod3 arms across the
    // two reconstruction stencils at FCELL (left = cells {0,1,2}, right = cells {1,2,3}):
    //   rho: all-positive arm (both stencils, magnitude-limited).
    //   v0:  all-negative arm (left) AND the sign-mixed CLAMP-to-zero arm (right, local extremum).
    //   v1:  all-negative arm (both, the transverse component is reconstructed too).
    //   pre: all-positive arm (left) AND the CLAMP-to-zero arm (right, local extremum).
    let rho = [1.0, 1.4, 1.5, 3.0];
    let v0 = [0.3, -0.2, -0.25, 0.4];
    let v1 = [0.2, 0.1, -0.05, -0.3];
    let pre = [0.6, 1.2, 1.25, 0.5];

    let eos = IdealGas { gamma: GAMMA };
    // f64 reference: reconstruct L/R with the single-source theta-MC, then native HLLE.
    let (rho_l, rho_r) = plm_theta_f64(&rho, FCELL, theta);
    let (v0_l, v0_r) = plm_theta_f64(&v0, FCELL, theta);
    let (v1_l, v1_r) = plm_theta_f64(&v1, FCELL, theta);
    let (pre_l, pre_r) = plm_theta_f64(&pre, FCELL, theta);
    // guard that the designed clamp-to-zero arm is actually taken (a zero slope leaves the
    // RIGHT face value equal to the cell value) — else the test would silently stop exercising
    // the limiter branch it exists to cover.
    assert_eq!(v0_r, v0[FCELL], "v0 right slope must clamp to 0 (sign-mixed local extremum)");
    assert_eq!(pre_r, pre[FCELL], "pre right slope must clamp to 0 (sign-mixed local extremum)");
    let left = Prim::<f64, 2> { rho: rho_l, vel: Tensor::new([v0_l, v1_l]), pre: pre_l };
    let right = Prim::<f64, 2> { rho: rho_r, vel: Tensor::new([v0_r, v1_r]), pre: pre_r };
    let f = hlle::<f64, 2, _>(&symbi_hydro::newtonian::Newtonian, &eos, &left, &right, &Tensor::unit(0), 0.0);

    // Gv kernel: the SAME adiabatic flux on the per-cell series, evaluated on the f64
    // interpreter. the kernel reconstructs internally (its select-branches now driven by
    // the non-uniform input), so its output at FCELL must equal the f64 reference.
    let out = KernelRun::new(adiabatic_flux_gv::<2>(0))
        .grid([NSWEEP, 1])
        .compute_window([FCELL as i32, 0], [1, 1])
        .field_with("prim_rho", move |c| rho[c[0]])
        .field_with("prim_v0", move |c| v0[c[0]])
        .field_with("prim_v1", move |c| v1[c[0]])
        .field_with("prim_pre", move |c| pre[c[0]])
        .scalars(&[
            ("gamma", GAMMA), ("theta", theta),
            ("mesh_adot_0", 0.0), ("mesh_vtrans_0", 0.0), ("x_lo_0", 0.0), ("dx_0", 1.0),
        ])
        .run();
    out.expect(
        [FCELL, 0],
        &[("flux_den", f.den), ("flux_mom_0", f.mom[0]), ("flux_mom_1", f.mom[1]), ("flux_nrg", f.nrg)],
        1e-12,
    );
}
