// =============================================================================
// ct_z_invariance_assessment.rs
//
// the z-invariance record of a z-uniform magnetized equilibrium at rest,
// B_axis1 = B0 sin(k (x - x_min)) on [1, 3] x [0, 1] with reflecting walls on
// axis 0 and periodic axis 1, over the matrix
//   {Contact, UCT} constrained transport x {cylindrical r-z, cartesian x-y}
// at matched resolution, CFL number, timestep policy and seed (UCT is refused
// on the flat r-z plane and that arm is skipped). the continuum state is
// z-uniform for all time, so every z-dependent value is numerical, and a
// grid-scale z mode is divergence-free: a div B check cannot see it.
//
// three ignored diagnostics, each bounded by its end time or step count:
// - z_invariance_record: per step, the seed-mode projection of the axis-1 face
//   field (r-weighted on the cylinder), the z-Nyquist amplitude
//   max_x |mean_z (-1)^z F| and the z-departure max |F - mean_z F| of both face
//   fields, density, both poloidal velocities and pressure, and
//   max |div B| dx / B0, to t = ZI_T (default 0.3).
// - z_invariance_localization: exact (bitwise) z-uniformity of every field
//   after the build (interior), after the first recovery and ghost fill (the
//   state fields over their full allocation), and at each hook point of each
//   stage (interior), for ZI_STEPS steps (default 40) or until the first
//   operation whose output departs.
// - z_invariance_first_stage_replay: stage ZI_REPLAY_STAGE (0 or 1) of the
//   first step replayed one kernel-set operation at a time with the scan after
//   each, plus the periodic consistency of every z-ghost column with its image
//   after the fluxes; ZI_DOMAINS=1 prints each field's allocation.
// ZI_SEED adds a z-alternating velocity of that amplitude on grid axis
// ZI_SEED_AXIS (0 = across the field, the coupling one) to every arm alike.
//
// run:
//   ZI_N=48 cargo test -p symbi --test ct_z_invariance_assessment z_invariance_record -- --ignored --nocapture
//   ZI_N=48 ZI_SEED=1e-14 cargo test -p symbi --test ct_z_invariance_assessment z_invariance_record -- --ignored --nocapture
//   ZI_N=48 ZI_STEPS=60 cargo test -p symbi --test ct_z_invariance_assessment z_invariance_localization -- --ignored --nocapture
//   ZI_N=48 ZI_REPLAY_STAGE=1 cargo test -p symbi --test ct_z_invariance_assessment z_invariance_first_stage_replay -- --ignored --nocapture
// =============================================================================

use std::f64::consts::PI;
use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::evolve::{KernelSet, evolve_with_callback};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{Cartesian, Cylindrical};
use symbi_grid::Field;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::stage::{HookPoint, StageArgs, fold_stage};
use symbi_xpu::{CpuSpace, HostMemory};

type Cyl =
    SimStateGeneric<NewtonianMhd, 2, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Cart =
    SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Kern = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 2>;
type F = Field<f64, 2, HostMemory>;

const GAMMA: f64 = 5.0 / 3.0;
const X_MIN: f64 = 1.0;
const X_MAX: f64 = 3.0;
const Z_MAX: f64 = 1.0;
const B0: f64 = 1e-2;
const K: f64 = 4.0 * PI / (X_MAX - X_MIN);

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn seed(x: f64) -> f64 {
    B0 * (K * (x - X_MIN)).sin()
}

fn boundaries() -> Boundaries<2> {
    Boundaries::per_axis([
        [BoundaryType::Reflect, BoundaryType::Reflect],
        [BoundaryType::Periodic, BoundaryType::Periodic],
    ])
}

/// the gas at rest, with an optional z-alternating velocity of amplitude `ZI_SEED` (default 0)
/// on grid axis `ZI_SEED_AXIS` (default 0, across the field): the matched perturbation the
/// seeded control injects in every arm. `v1_slot` is the component slot of grid axis 1.
fn gas(z: f64, n: usize, v1_slot: usize) -> Prim<f64, 3> {
    let amplitude: f64 = env_or("ZI_SEED", 0.0);
    let axis: usize = env_or("ZI_SEED_AXIS", 0);
    let j = (z / (Z_MAX / n as f64)).floor() as i64;
    let mut v = [0.0; 3];
    v[if axis == 0 { 0 } else { v1_slot }] = if j % 2 == 0 { amplitude } else { -amplitude };
    Prim::adiabatic(Density(1.0), Tensor::new(v), Pressure(1.0))
}

// the axis-1 field is the z component (slot 2) of the cylindrical r-phi-z triple and the y
// component (slot 1) of the cartesian x-y-z triple
fn cyl(n: usize, cfl: f64) -> Cyl {
    Cyl::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cylindrical)
        .cells([n, n])
        .bounds([X_MIN, 0.0], [X_MAX, Z_MAX])
        .boundaries(boundaries())
        .cfl(cfl)
        .allocate()
        .expect("cylindrical sim")
        .set_initial(|[x, z]| MhdPrim::new(gas(z, n, 2), Tensor::new([0.0, 0.0, seed(x)])))
        .seed_faces(|axis, [x, _]| if axis == 1 { seed(x) } else { 0.0 })
        .build()
}

fn cart(n: usize, cfl: f64) -> Cart {
    Cart::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n, n])
        .bounds([X_MIN, 0.0], [X_MAX, Z_MAX])
        .boundaries(boundaries())
        .cfl(cfl)
        .allocate()
        .expect("cartesian sim")
        .set_initial(|[x, z]| MhdPrim::new(gas(z, n, 1), Tensor::new([0.0, seed(x), 0.0])))
        .seed_faces(|axis, [x, _]| if axis == 1 { seed(x) } else { 0.0 })
        .build()
}

fn kernels(cfl: f64, ct: CtMethod, alloc: &symbi_algebra::Domain<2>) -> Kern {
    Kern::new(GAMMA, cfl, 1.0, alloc).ct_method(ct)
}

/// the store-level views an arm exposes to the measures: the axis-1 velocity slot and whether
/// the chart weights by r
struct Chart {
    v1_slot: usize,
    cylindrical: bool,
}

/// the largest departure from z-uniformity of `f` over rows x in `xs` and columns z in `zs`,
/// with the row and the spread (min, max) there
fn z_departure(f: &F, xs: std::ops::Range<isize>, zs: std::ops::Range<isize>) -> (f64, isize) {
    let v = f.view();
    let mut worst = (0.0f64, 0isize);
    for ix in xs {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for iz in zs.clone() {
            let x = *v.at([ix, iz]);
            lo = lo.min(x);
            hi = hi.max(x);
        }
        if hi - lo > worst.0 {
            worst = (hi - lo, ix);
        }
    }
    worst
}

/// max over rows of |mean_z (-1)^z F|
fn z_nyquist(f: &F, xs: std::ops::Range<isize>, zs: std::ops::Range<isize>) -> f64 {
    let v = f.view();
    let n = zs.len() as f64;
    xs.map(|ix| {
        (zs.clone()
            .map(|iz| {
                if iz % 2 == 0 {
                    *v.at([ix, iz])
                } else {
                    -*v.at([ix, iz])
                }
            })
            .sum::<f64>()
            / n)
            .abs()
    })
    .fold(0.0, f64::max)
}

fn full(f: &F, axis: usize) -> std::ops::Range<isize> {
    f.domain().spaces[axis].lo..f.domain().spaces[axis].hi
}

macro_rules! arm_impl {
    ($name:ident, $sim:ty, $make:ident) => {
        mod $name {
            use super::*;

            fn interior(s: &$sim) -> (std::ops::Range<isize>, std::ops::Range<isize>) {
                let i = &s.geom.interior.spaces;
                (i[0].lo..i[0].hi, i[1].lo..i[1].hi)
            }

            fn x_cell(s: &$sim, ix: isize) -> f64 {
                X_MIN + ((ix - s.geom.interior.spaces[0].lo) as f64 + 0.5) * s.geom.dx[0]
            }

            fn x_face(s: &$sim, ix: isize) -> f64 {
                X_MIN + (ix - s.geom.interior.spaces[0].lo) as f64 * s.geom.dx[0]
            }

            /// projection, nyquist(B0f, B1f), departure(B0f, B1f, rho, v0, v1, p), div B dx / B0
            pub fn measures(s: &$sim, chart: &Chart) -> [f64; 10] {
                let m = s.fields.mhd.as_ref().unwrap();
                let (xs, zs) = interior(s);
                let b0 = &m.bface.b[0];
                let b1 = &m.bface.b[1];
                let w = |x: f64| if chart.cylindrical { x } else { 1.0 };
                let (mut num, mut den) = (0.0, 0.0);
                for ix in xs.clone() {
                    let (x, shape) = (x_cell(s, ix), (K * (x_cell(s, ix) - X_MIN)).sin());
                    for iz in zs.clone() {
                        num += *b1.view().at([ix, iz]) * shape * w(x);
                        den += shape * shape * w(x);
                    }
                }
                let mut div = 0.0f64;
                let (dx, dz) = (s.geom.dx[0], s.geom.dx[1]);
                for ix in xs.clone() {
                    for iz in zs.clone() {
                        let radial = (w(x_face(s, ix + 1)) * *b0.view().at([ix + 1, iz]) - w(x_face(s, ix)) * *b0.view().at([ix, iz]))
                            / (dx * w(x_cell(s, ix)));
                        let axial = (*b1.view().at([ix, iz + 1]) - *b1.view().at([ix, iz])) / dz;
                        div = div.max((radial + axial).abs());
                    }
                }
                let p = &s.fields.prim;
                let dep = |f: &F| z_departure(f, xs.clone(), zs.clone()).0;
                [
                    num / den / B0,
                    z_nyquist(b0, xs.clone(), zs.clone()) / B0,
                    z_nyquist(b1, xs.clone(), zs.clone()) / B0,
                    dep(b0) / B0,
                    dep(b1) / B0,
                    dep(&p.rho),
                    dep(&p.vel[0]),
                    dep(&p.vel[chart.v1_slot]),
                    dep(p.pre_field().unwrap()),
                    div * dx.min(dz) / B0,
                ]
            }

            pub fn record(n: usize, cfl: f64, ct: CtMethod, t_final: f64, stride: usize, chart: &Chart, label: &str) {
                let mut s = $make(n, cfl);
                let k = kernels(cfl, ct, &s.geom.allocated);
                let row = |step: usize, t: f64, v: [f64; 10]| {
                    println!(
                        "T {label} step {step:>5} t {t:.4} proj {:.6} nyq B0f {:.3e} B1f {:.3e} | dep B0f {:.3e} B1f {:.3e} rho {:.3e} v0 {:.3e} v1 {:.3e} p {:.3e} | divB {:.3e}",
                        v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9]
                    )
                };
                row(0, 0.0, measures(&s, chart));
                let step = std::cell::Cell::new(0usize);
                let first = std::cell::Cell::new(None::<usize>);
                let last = std::cell::RefCell::new((0.0, [0.0; 10]));
                let outcome = evolve_with_callback(&mut s, &k, t_final, 1, |st| {
                    step.set(step.get() + 1);
                    let v = measures(st, chart);
                    if first.get().is_none() && v[3..9].iter().any(|d| *d > 0.0) {
                        first.set(Some(step.get()));
                    }
                    if step.get() % stride.max(1) == 0 {
                        row(step.get(), st.time, v);
                    }
                    *last.borrow_mut() = (st.time, v);
                });
                let (t, v) = *last.borrow();
                row(step.get(), t, v);
                println!(
                    "E {label} steps {} t {t:.4} first z-departure at step {:?} outcome {}",
                    step.get(),
                    first.get(),
                    if outcome.is_ok() { "ok".to_string() } else { format!("{:?}", outcome.err()) }
                );
            }

            /// every field whose z-uniformity the localization checks, by name
            fn fields(s: &$sim) -> Vec<(String, &F)> {
                let mut out: Vec<(String, &F)> = Vec::new();
                let c = &s.fields.cons;
                out.push(("cons.den".into(), &c.den));
                for (ii, f) in c.mom.iter().enumerate() {
                    out.push((format!("cons.mom{ii}"), f));
                }
                out.push(("cons.nrg".into(), c.nrg_field().unwrap()));
                let p = &s.fields.prim;
                out.push(("prim.rho".into(), &p.rho));
                for (ii, f) in p.vel.iter().enumerate() {
                    out.push((format!("prim.vel{ii}"), f));
                }
                out.push(("prim.pre".into(), p.pre_field().unwrap()));
                for (dd, fl) in s.fields.flux.iter().enumerate() {
                    out.push((format!("flux{dd}.den"), &fl.den));
                    for (ii, f) in fl.mom.iter().enumerate() {
                        out.push((format!("flux{dd}.mom{ii}"), f));
                    }
                    out.push((format!("flux{dd}.nrg"), fl.nrg_field().unwrap()));
                }
                let w = &s.workspace;
                for (tag, c) in [("u_n", &w.u_n), ("u_stage", &w.u_stage)] {
                    out.push((format!("{tag}.den"), &c.den));
                    for (ii, f) in c.mom.iter().enumerate() {
                        out.push((format!("{tag}.mom{ii}"), f));
                    }
                    out.push((format!("{tag}.nrg"), c.nrg_field().unwrap()));
                }
                let m = s.fields.mhd.as_ref().unwrap();
                for (dd, f) in m.bface.b.iter().enumerate() {
                    out.push((format!("bface{dd}"), f));
                }
                for (dd, f) in m.bface_n.b.iter().enumerate() {
                    out.push((format!("bface_n{dd}"), f));
                }
                for (dd, f) in m.efield_n.e.iter().enumerate() {
                    out.push((format!("efield_n{dd}"), f));
                }
                for (dd, f) in m.wave_speed_l.iter().chain(m.wave_speed_r.iter()).enumerate() {
                    out.push((format!("wave_speed{dd}"), f));
                }
                for (dd, row) in m.v_upwind.iter().enumerate() {
                    for (ii, f) in row.iter().enumerate() {
                        out.push((format!("v_upwind{dd}.{ii}"), f));
                    }
                }
                for (dd, f) in m.bface_stage.b.iter().enumerate() {
                    out.push((format!("bface_stage{dd}"), f));
                }
                for (dd, f) in m.bcell_stage.b.iter().enumerate() {
                    out.push((format!("bcell_stage{dd}"), f));
                }
                for (dd, f) in m.bcell.b.iter().enumerate() {
                    out.push((format!("bcell{dd}"), f));
                }
                for (dd, f) in m.efield.e.iter().enumerate() {
                    out.push((format!("efield{dd}"), f));
                }
                for (dd, bf) in m.bflux.iter().enumerate() {
                    for (ii, f) in bf.f.iter().enumerate() {
                        out.push((format!("bflux{dd}.{ii}"), f));
                    }
                }
                out
            }

            /// the fields departing from exact z-uniformity, over their full allocation once the
            /// ghosts are filled and over the interior before
            fn broken(s: &$sim, ghosts: bool) -> Vec<String> {
                let (xs, zs) = interior(s);
                fields(s)
                    .into_iter()
                    .filter_map(|(name, f)| {
                        // the ghost fill carries primitives and face fields; conserved values,
                        // fluxes and EMFs are derived and meaningful on the interior
                        let state = name.starts_with("prim") || name.starts_with("bface") || name.starts_with("bcell");
                        let (d, row) = if ghosts && state {
                            z_departure(f, full(f, 0), full(f, 1))
                        } else {
                            z_departure(f, xs.clone(), zs.clone())
                        };
                        (d > 0.0).then(|| {
                            // the z columns of the row whose value departs from the row's first
                            let (zs_all, v) = (if ghosts && state { full(f, 1) } else { zs.clone() }, f.view());
                            let base = *v.at([row, zs_all.start]);
                            let odd: Vec<isize> = zs_all.filter(|&iz| *v.at([row, iz]) != base).collect();
                            let shown: Vec<String> = odd.iter().take(8).map(|z| z.to_string()).collect();
                            format!("{name} (dep {d:.3e} at row {row}, {} z columns differ from z0: {})", odd.len(), shown.join(","))
                        })
                    })
                    .collect()
            }

            /// the first rk2 stage of the first step replayed one operation at a time, scanning
            /// the interior after each
            pub fn replay(n: usize, cfl: f64, ct: CtMethod, label: &str) {
                let s = $make(n, cfl);
                let k = kernels(cfl, ct, &s.geom.allocated);
                println!("R {label} interior rows {:?} z {:?}", interior(&s).0, interior(&s).1);
                if std::env::var("ZI_DOMAINS").is_ok() {
                    for (name, f) in fields(&s) {
                        let d = f.domain();
                        println!("D {label} {name}: x {}..{} z {}..{}", d.spaces[0].lo, d.spaces[0].hi, d.spaces[1].lo, d.spaces[1].hi);
                    }
                }
                let scan = |at: &str| {
                    let b = broken(&s, false);
                    println!("R {label} {at}: {}", if b.is_empty() { "z-uniform".to_string() } else { b.join(", ") });
                };
                let dt = cfl * s.geom.dx[0].min(s.geom.dx[1]) / (GAMMA + B0 * B0).sqrt();
                let stage: usize = env_or("ZI_REPLAY_STAGE", 0);
                let stages = Timestepping::Rk2.stages();
                let (a0, ac) = stages[stage];
                k.c2p(&s);
                k.ghost_fill(&s);
                k.snapshot_retry(&s);
                k.snapshot(&s);
                if stage == 1 {
                    let args = StageArgs {
                        dt,
                        a0: stages[0].0,
                        ac: stages[0].1,
                        stage: 0,
                        n_stages: 2,
                        injection_weight: symbi_sim::driver::downstream_injection_weight(stages, 0),
                        allow_elision: false,
                        exit_fraction: symbi_sim::driver::stage_exit_fraction(stages, 0),
                    };
                    let _ = fold_stage(&s, &k, args, &mut |_| {});
                    let b = broken(&s, true);
                    println!("R {label} after stage 0 (full): {}", if b.is_empty() { "z-uniform".to_string() } else { b.join(", ") });
                }
                scan("snapshot");
                k.snapshot_stage_faces(&s);
                s.mark_stage_input_captured();
                k.snapshot_stage(&s);
                scan("stage snapshots");
                k.wave_speeds(&s);
                k.flux(&s, 0);
                k.flux(&s, 1);
                scan("flux");
                // periodic consistency of the z-ghost layers: each ghost column against its image
                let (xs, zs) = interior(&s);
                let nz = zs.len() as isize;
                for (name, f) in fields(&s) {
                    // the ghost fill carries primitives and face fields; conserved ghosts hold no state
                    if name.starts_with("cons") || name.starts_with("u_") {
                        continue;
                    }
                    let (fz, v) = (full(f, 1), f.view());
                    let mut worst = (0.0f64, 0isize, 0isize);
                    for ix in xs.clone() {
                        for iz in fz.clone().filter(|iz| !zs.contains(iz)) {
                            // the corner stencils read the axis-1-staggered fluxes on interior faces alone and
                            // the others one ghost cell deep; the state fields carry the full ghost band
                            let state = name.starts_with("prim") || name.starts_with("bface") || name.starts_with("bcell");
                            let z_staggered = name.starts_with("flux1") || name.starts_with("bflux1");
                            if !state && (z_staggered || (iz != zs.start - 1 && iz != zs.end)) {
                                continue;
                            }
                            let image = if iz < zs.start { iz + nz } else { iz - nz };
                            if !fz.contains(&image) {
                                continue;
                            }
                            let d = (*v.at([ix, iz]) - *v.at([ix, image])).abs();
                            if d > worst.0 {
                                worst = (d, ix, iz);
                            }
                        }
                    }
                    if worst.0 > 0.0 {
                        println!("P {label} {name}: ghost z {} differs from its periodic image by {:.3e} at row {}", worst.2, worst.0, worst.1);
                    }
                }
                k.efield(&s);
                scan("efield");
                k.godunov_stage(&s, dt, a0, ac);
                scan("godunov_stage");
                k.post_godunov(&s, dt, stage as u8 + 1);
                scan("post_godunov");
                k.source_apply(&s, ac * dt);
                scan("source_apply");
                k.body_source(&s, ac * dt);
                scan("body_source");
            }

            pub fn localize(n: usize, cfl: f64, ct: CtMethod, steps: usize, label: &str) {
                let s = $make(n, cfl);
                let k = kernels(cfl, ct, &s.geom.allocated);
                let report = |at: &str, b: Vec<String>| {
                    println!("L {label} {at}: {}", if b.is_empty() { "z-uniform".to_string() } else { b.join(", ") });
                    !b.is_empty()
                };
                if report("after build (interior)", broken(&s, false)) {
                    return;
                }
                k.c2p(&s);
                k.ghost_fill(&s);
                if report("after first recovery + ghost fill", broken(&s, true)) {
                    return;
                }
                let dt = cfl * s.geom.dx[0].min(s.geom.dx[1]) / (GAMMA + B0 * B0).sqrt();
                let stages = Timestepping::Rk2.stages();
                for step in 0..steps {
                    k.snapshot_retry(&s);
                    k.snapshot(&s);
                    for stage in 0..2 {
                        let args = StageArgs {
                            dt,
                            a0: stages[stage].0,
                            ac: stages[stage].1,
                            stage,
                            n_stages: 2,
                            injection_weight: symbi_sim::driver::downstream_injection_weight(stages, stage),
                            allow_elision: false,
                            exit_fraction: symbi_sim::driver::stage_exit_fraction(stages, stage),
                        };
                        let hit = std::cell::Cell::new(false);
                        let mut hook = |hp: HookPoint| {
                            if hit.get() {
                                return;
                            }
                            let name = match hp {
                                HookPoint::AfterFlux => "after flux",
                                HookPoint::BeforeGhostFill => "before ghost fill",
                                HookPoint::AfterGhostFill => "after ghost fill",
                                HookPoint::BeforeC2p => "before recovery",
                            };
                            hit.set(report(&format!("step {step} stage {stage} {name}"), broken(&s, false)));
                        };
                        let _ = fold_stage(&s, &k, args, &mut hook);
                        if hit.get() || report(&format!("step {step} stage {stage} end"), broken(&s, true)) {
                            return;
                        }
                    }
                }
                println!("L {label} z-uniform through {steps} steps");
            }
        }
    };
}

arm_impl!(cyl_arm, Cyl, cyl);
arm_impl!(cart_arm, Cart, cart);

fn matrix() -> Vec<(CtMethod, bool, String)> {
    let mut out = Vec::new();
    for (ct, name) in [(CtMethod::Contact, "contact"), (CtMethod::Uct, "uct")] {
        for cylindrical in [true, false] {
            // the flat 2D UCT edge EMF covers the out-of-plane E_z edge; the r-z plane refuses UCT
            if ct == CtMethod::Uct && cylindrical {
                println!(
                    "X uct/cyl refused: the flat cylindrical r-z plane builds the Contact edge EMF alone"
                );
                continue;
            }
            out.push((
                ct,
                cylindrical,
                format!("{name}/{}", if cylindrical { "cyl" } else { "cart" }),
            ));
        }
    }
    out
}

#[test]
#[ignore = "diagnostic record; prints the z-invariance matrix"]
fn z_invariance_record() {
    let n: usize = env_or("ZI_N", 48);
    let cfl: f64 = env_or("ZI_CFL", 0.3);
    let t_final: f64 = env_or("ZI_T", 0.3);
    let stride: usize = env_or("ZI_STRIDE", 8);
    for (ct, cylindrical, label) in matrix() {
        let label = format!("{label}/N{n}");
        if cylindrical {
            cyl_arm::record(
                n,
                cfl,
                ct,
                t_final,
                stride,
                &Chart {
                    v1_slot: 2,
                    cylindrical: true,
                },
                &label,
            );
        } else {
            cart_arm::record(
                n,
                cfl,
                ct,
                t_final,
                stride,
                &Chart {
                    v1_slot: 1,
                    cylindrical: false,
                },
                &label,
            );
        }
    }
}

#[test]
#[ignore = "diagnostic record; locates the first operation that breaks z-uniformity"]
fn z_invariance_localization() {
    let n: usize = env_or("ZI_N", 48);
    let cfl: f64 = env_or("ZI_CFL", 0.3);
    let steps: usize = env_or("ZI_STEPS", 40);
    for (ct, cylindrical, label) in matrix() {
        let label = format!("{label}/N{n}");
        if cylindrical {
            cyl_arm::localize(n, cfl, ct, steps, &label);
        } else {
            cart_arm::localize(n, cfl, ct, steps, &label);
        }
    }
}

#[test]
#[ignore = "diagnostic record; replays the first stage one operation at a time"]
fn z_invariance_first_stage_replay() {
    let n: usize = env_or("ZI_N", 48);
    let cfl: f64 = env_or("ZI_CFL", 0.3);
    cyl_arm::replay(n, cfl, CtMethod::Contact, &format!("contact/cyl/N{n}"));
    cart_arm::replay(n, cfl, CtMethod::Contact, &format!("contact/cart/N{n}"));
}
