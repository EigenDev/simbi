// =============================================================================
// mhd_recovery_status.rs
//
// the status the magnetized conserved-to-primitive recovery stores for each cell,
// on the build's default memory (host memory in a cpu build, managed device memory
// in a gpu build), through the production kernel set's c2p. a 3D store of finite,
// positive-density states with a finite cell field, one interior cell planted at
// negative density:
// - every other interior cell stores status 0 and the primitives of its own
//   conserved state (rho = den, v = mom / den)
// - the planted cell stores INVALID_PRIMITIVE, so the scan reaches it and rejects it
// for the isothermal and the newtonian regime. a failure prints, for the first
// mismatched cells, the stored status beside the conserved and cell-field values
// the host reads at that cell.
//
// run: cargo test -p symbi --test mhd_recovery_status
//      (with --features hip on a gpu node: the device recovery)
// =============================================================================

use symbi::prelude::KernelSet;
use symbi::regimes::substrate_isothermal_mhd::IsothermalMhdSubstrateKernelSet3D;
use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet3D;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::c2p_result::ErrorCode;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::{IdealGas, Isothermal};
use symbi_hydro::isothermal_mhd::IsothermalMhd;
use symbi_hydro::mhd_state::{MhdPrim, MhdPrimG};
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::{Prim, PrimG};
use symbi_xpu::{DefaultMemory, DefaultSpace};

const N: usize = 8;
const PLANTED: [isize; 3] = [3, 4, 5];

fn sync() {
    #[cfg(feature = "gpu")]
    if <DefaultMemory as symbi_xpu::MemoryStorage>::IS_DEVICE_ACCESSIBLE {
        symbi_xpu::ctx_sync();
    }
}

fn velocity([x, y, z]: [f64; 3]) -> [f64; 3] {
    [0.2 * (6.0 * y).sin(), -0.1 * (5.0 * x).cos(), 0.05 * z]
}

fn field([x, y, _z]: [f64; 3]) -> [f64; 3] {
    [0.5 * (4.0 * y).sin(), 0.3 * (3.0 * x).cos(), 0.0]
}

/// every interior cell's stored status and primitives against its own conserved state; the
/// planted cell must store the rejection. returns the first mismatches, described with the values
/// the host reads
fn mismatches<R, E>(sim: &SimState<R, 3, Cartesian, E, DefaultSpace, DefaultMemory>) -> Vec<String>
where
    R: symbi_hydro::regime::Regime<f64, 3>,
    E: symbi_hydro::eos::Eos<f64>,
{
    sync();
    let mhd = sim.fields.mhd.as_ref().expect("a magnetized store");
    let mut out = Vec::new();
    for c in sim.geom.interior.iter() {
        let status = *sim.fields.c2p_error.view().at(c) as u8;
        let den = *sim.fields.cons.den.view().at(c);
        let mom: Vec<f64> = sim.fields.cons.mom.iter().map(|f| *f.view().at(c)).collect();
        let bcell: Vec<f64> = mhd.bcell.b.iter().map(|f| *f.view().at(c)).collect();
        let want = if c == PLANTED { ErrorCode::INVALID_PRIMITIVE.0 } else { 0 };
        let rho = *sim.fields.prim.rho.view().at(c);
        let vel: Vec<f64> = sim.fields.prim.vel.iter().map(|f| *f.view().at(c)).collect();
        // the isothermal recovery forms mom * (1 / den), one rounding from the quotient the
        // newtonian recovery forms; 1e-15 relative holds either
        let recovered = c == PLANTED
            || (rho == den && vel.iter().zip(&mom).all(|(v, m)| (*v - m / den).abs() <= 1e-15 * (m / den).abs().max(1.0)));
        if status != want || !recovered {
            out.push(format!(
                "cell {c:?}: status {status} (want {want}), den {den:e}, mom {mom:?}, bcell {bcell:?}, prim rho {rho:e} vel {vel:?}"
            ));
        }
        if out.len() >= 8 {
            break;
        }
    }
    out
}

#[test]
fn the_isothermal_recovery_accepts_every_valid_cell_and_rejects_the_planted_one() {
    type Sim = SimState<IsothermalMhd, 3, Cartesian, Isothermal<f64>, DefaultSpace, DefaultMemory>;
    let dx = 1.0 / N as f64;
    let sim: Sim = Sim::build(IsothermalMhd, Isothermal { cs: 1.0 }, Cartesian)
        .cells([N, N, N])
        .spacing([dx, dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(0.3)
        .allocate()
        .expect("isothermal mhd store")
        .set_initial(|x| {
            MhdPrimG::<f64, 3, IsoModel>::new(
                PrimG::isothermal(Density(1.0 + 0.5 * x[0]), Tensor::new(velocity(x))),
                Tensor::new(field(x)),
            )
        })
        .seed_faces(|axis, x| field(x)[axis])
        .build();
    sim.fields.cons.den.view_mut().set(PLANTED, -1.0);
    assert!(sim.geom.interior.contains(PLANTED), "the planted cell lies outside the interior the recovery covers");
    let kernels = IsothermalMhdSubstrateKernelSet3D::<DefaultMemory, f64>::new(1.0, 0.3, 1.0, &sim.geom.allocated);
    kernels.c2p(&sim);
    let bad = mismatches(&sim);
    assert!(bad.is_empty(), "isothermal mhd recovery status:\n  {}", bad.join("\n  "));
}

#[test]
fn the_newtonian_recovery_accepts_every_valid_cell_and_rejects_the_planted_one() {
    type Sim = SimState<NewtonianMhd, 3, Cartesian, IdealGas<f64>, DefaultSpace, DefaultMemory>;
    let dx = 1.0 / N as f64;
    let sim: Sim = Sim::build(NewtonianMhd, IdealGas { gamma: 5.0 / 3.0 }, Cartesian)
        .cells([N, N, N])
        .spacing([dx, dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(0.3)
        .allocate()
        .expect("newtonian mhd store")
        .set_initial(|x| {
            MhdPrim::new(
                Prim::adiabatic(Density(1.0 + 0.5 * x[0]), Tensor::new(velocity(x)), Pressure(1.0)),
                Tensor::new(field(x)),
            )
        })
        .seed_faces(|axis, x| field(x)[axis])
        .build();
    sim.fields.cons.den.view_mut().set(PLANTED, -1.0);
    assert!(sim.geom.interior.contains(PLANTED), "the planted cell lies outside the interior the recovery covers");
    let kernels = NewtonianMhdSubstrateKernelSet3D::<DefaultMemory, f64>::new(5.0 / 3.0, 0.3, 1.0, &sim.geom.allocated);
    kernels.c2p(&sim);
    let bad = mismatches(&sim);
    assert!(bad.is_empty(), "newtonian mhd recovery status:\n  {}", bad.join("\n  "));
}
