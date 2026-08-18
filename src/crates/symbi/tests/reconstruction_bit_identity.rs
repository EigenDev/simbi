// =============================================================================
// reconstruction_bit_identity.rs
//
// byte-level pin of the plm evolution path: a short shocked 1d run and a smooth
// 2d run, final conserved state serialized as little-endian f64 bytes and
// compared against a recorded baseline file. any reconstruction variant added
// beside plm leaves these bytes untouched — a mismatch means the plm numerics
// moved, which a bit comparison reports without any tolerance in the way.
//
// the baseline lives in the gitignored workspace tree, machine-local and outside
// the repo. when the file is absent the run records the current bytes and
// reports that no gate was exerted; deleting a baseline file is the
// deliberate act that re-pins after an intentional change to plm numerics.
//
// run: cargo test -p symbi --test reconstruction_bit_identity
// =============================================================================

use std::path::PathBuf;

use symbi::prelude::Solver;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const THETA: f64 = 1.5;

fn baseline_path(case: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../workspace/baselines/recon_bit_identity")
        .join(format!("{case}.f64"))
}

/// compare the serialized state against the recorded baseline, or record it when
/// no baseline exists yet. a mismatch reports the count and first offset of the
/// differing f64 lanes.
fn gate_or_record(case: &str, bytes: &[u8]) {
    let path = baseline_path(case);
    if !path.exists() {
        std::fs::create_dir_all(path.parent().unwrap()).expect("baseline dir");
        std::fs::write(&path, bytes).expect("baseline write");
        eprintln!(
            "{case}: RECORDED {} bytes at {} — no gate exerted this run",
            bytes.len(),
            path.display()
        );
        return;
    }
    let base = std::fs::read(&path).expect("baseline read");
    assert_eq!(
        base.len(),
        bytes.len(),
        "{case}: serialized length changed ({} -> {} bytes)",
        base.len(),
        bytes.len()
    );
    let mut diff_count = 0usize;
    let mut first_lane = None;
    let mut max_abs = 0.0_f64;
    for (ll, (a, b)) in base.chunks(8).zip(bytes.chunks(8)).enumerate() {
        if a != b {
            diff_count += 1;
            first_lane.get_or_insert(ll);
            let (va, vb) = (
                f64::from_le_bytes(a.try_into().unwrap()),
                f64::from_le_bytes(b.try_into().unwrap()),
            );
            max_abs = max_abs.max((va - vb).abs());
        }
    }
    assert!(
        diff_count == 0,
        "{case}: {diff_count} of {} f64 lanes differ from baseline (first at lane {}, \
         max |delta| = {max_abs:.3e})",
        bytes.len() / 8,
        first_lane.unwrap()
    );
    eprintln!("{case}: bit-identical to baseline ({} lanes)", bytes.len() / 8);
}

/// interior conserved state as little-endian f64 bytes: den, then each momentum
/// component, then energy, each in interior iteration order.
fn serialize_cons<const D: usize>(
    sim: &SimState<Newtonian, D, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>,
) -> Vec<u8>
where
    Cartesian: symbi_geometry::Metric<f64, D>,
{
    let mut bytes = Vec::new();
    let mut push_field = |field: &symbi_grid::Field<f64, D, HostMemory>| {
        for c in sim.geom.interior.iter() {
            bytes.extend_from_slice(&field.view().at(c).to_le_bytes());
        }
    };
    push_field(&sim.fields.cons.den);
    for dd in 0..D {
        push_field(&sim.fields.cons.mom[dd]);
    }
    push_field(sim.fields.cons.nrg_field().expect("adiabatic cons.nrg"));
    bytes
}

/// sod shock tube: shock + contact + rarefaction, so the limiter is genuinely
/// engaged rather than idling on a smooth profile.
#[test]
fn plm_sod_1d_bit_identity() {
    const N: usize = 128;
    let dx = 1.0 / N as f64;
    type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .spacing([dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|[x]| {
            if x < 0.5 {
                Prim {
                    rho: 1.0,
                    vel: Tensor::new([0.0]),
                    pre: 1.0,
                }
            } else {
                Prim {
                    rho: 0.125,
                    vel: Tensor::new([0.0]),
                    pre: 0.1,
                }
            }
        })
        .build();
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, CFL, &sim.geom.allocated)
            .with_solver(Solver::Hllc)
            .expect("solver/regime mismatch")
            .theta(THETA);
    evolve(&mut sim, &sub, 0.1).expect("evolve failed");
    gate_or_record("sod_1d_plm_hllc_rk2", &serialize_cons(&sim));
}

/// low-mach vortical flow under HLLC-LM (the published fleischmann ramp): at
/// mach ~0.06 every face sits on the phi ramp, and this pin holds the
/// low-dissipation arm byte-stable against future change. the LM benefit itself
/// is gated behaviorally in `lm_clamp_laws.rs` (the ramp stays strictly less
/// dissipative than classical HLLC on this flow).
#[test]
fn hllc_plus_low_mach_vortex_bit_identity() {
    const N: usize = 48;
    const MACH: f64 = 0.06;
    let dx = 1.0 / N as f64;
    type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|[x, y]| {
            let tau = std::f64::consts::TAU;
            // taylor-green cell: solenoidal, smooth, periodic; cs = 1 at this state
            Prim {
                rho: 1.0,
                vel: Tensor::new([
                    -MACH * (tau * y).sin() * (tau * x).cos(),
                    MACH * (tau * x).sin() * (tau * y).cos(),
                ]),
                pre: 1.0 / GAMMA,
            }
        })
        .build();
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, CFL, &sim.geom.allocated)
            .with_solver(Solver::HllcPlus)
            .expect("solver/regime mismatch")
            .theta(THETA);
    evolve(&mut sim, &sub, 0.5).expect("evolve failed");
    gate_or_record("vortex_2d_plm_hllc_plus_rk2", &serialize_cons(&sim));
}

/// smooth 2d diagonal advection on a periodic box: the multi-dimensional sweep
/// order and transverse indexing, with the limiter in its smooth regime.
#[test]
fn plm_bump_2d_bit_identity() {
    const N: usize = 48;
    let dx = 1.0 / N as f64;
    type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|[x, y]| Prim {
            rho: 1.0 + 0.2 * (-(((x - 0.5) / 0.1).powi(2) + ((y - 0.5) / 0.1).powi(2))).exp(),
            vel: Tensor::new([1.0, 0.5]),
            pre: 1.0,
        })
        .build();
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, CFL, &sim.geom.allocated)
            .with_solver(Solver::Hlle)
            .expect("solver/regime mismatch")
            .theta(THETA);
    evolve(&mut sim, &sub, 0.05).expect("evolve failed");
    gate_or_record("bump_2d_plm_hlle_rk2", &serialize_cons(&sim));
}
