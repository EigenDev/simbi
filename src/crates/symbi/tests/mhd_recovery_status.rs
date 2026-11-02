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
// the host reads at that cell. the isothermal case also runs with the cell field
// zero, the velocity zero, and both zero: each switches off one of the
// acceptance conditions (finite field, finite velocity), so the variants that
// fail name the condition the stored status rejects. on a gpu build a failure
// prints the kernel source the device renders from the registry's ir.
// the probe gate runs the isothermal recovery's probe kernels on the same store
// through the production binding (dispatch_named, each probe's own field in the
// status slot): each carries one loaded operand or one acceptance condition on
// the status channel. at every interior cell the loaded operands equal the
// host's conserved values bit for bit, each condition reads 0 or 1, `accepted`
// is their conjunction, the stored status is 0 exactly where `accepted` is 1,
// and every condition holds except rho_positive at the planted cell. a failure
// prints every channel at the first mismatched cells; every run writes the
// production and probe kernel sources with their buffer bindings and the device
// compile options to stderr.
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

/// the source the device compiles for `name`: the registry's ir rendered for the build's target
#[cfg(feature = "gpu")]
fn device_source(name: &str) -> String {
    let (_, ir) = symbi::regimes::substrate_kernels::expect_kernel::<f64>(name);
    #[cfg(feature = "hip")]
    let target = symbi_ir::emit::Target::Hip;
    #[cfg(not(feature = "hip"))]
    let target = symbi_ir::emit::Target::Cuda;
    let desc = symbi_ir::render_from_ir(ir, target, symbi_ir::emit::Precision::F64);
    let mut bindings = desc.field_bindings.clone();
    bindings.sort_by_key(|b| b.buffer_index);
    let listed: Vec<String> = bindings.iter().map(|b| format!("  buffer {} {:?} output {}", b.buffer_index, b.field, b.is_output)).collect();
    format!("{name} buffers (render order):\n{}\n{}", listed.join("\n"), desc.source)
}

/// the options the device compiler receives for every kernel
#[cfg(feature = "hip")]
fn compile_options() -> String {
    format!("hiprtc options {:?} (hiprtc defaults otherwise)", symbi_xpu::hiprtc::compile_options())
}

#[cfg(not(feature = "hip"))]
fn compile_options() -> String {
    "(no hiprtc compile in this build)".to_string()
}

/// a cpu build runs the kernel's host twin, which carries no device source
#[cfg(not(feature = "gpu"))]
fn device_source(name: &str) -> String {
    format!("(a cpu build runs the {name} host twin)")
}

type IsoSim = SimState<IsothermalMhd, 3, Cartesian, Isothermal<f64>, DefaultSpace, DefaultMemory>;

/// the isothermal store with the velocity and the cell field each kept or zeroed, and the planted cell
fn isothermal_store(keep_velocity: bool, keep_field: bool) -> IsoSim {
    type Sim = IsoSim;
    let dx = 1.0 / N as f64;
    let v = move |x: [f64; 3]| if keep_velocity { velocity(x) } else { [0.0; 3] };
    let b = move |x: [f64; 3]| if keep_field { field(x) } else { [0.0; 3] };
    let sim: Sim = Sim::build(IsothermalMhd, Isothermal { cs: 1.0 }, Cartesian)
        .cells([N, N, N])
        .spacing([dx, dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(0.3)
        .allocate()
        .expect("isothermal mhd store")
        .set_initial(|x| {
            MhdPrimG::<f64, 3, IsoModel>::new(
                PrimG::isothermal(Density(1.0 + 0.5 * x[0]), Tensor::new(v(x))),
                Tensor::new(b(x)),
            )
        })
        .seed_faces(|axis, x| b(x)[axis])
        .build();
    sim.fields.cons.den.view_mut().set(PLANTED, -1.0);
    assert!(sim.geom.interior.contains(PLANTED), "the planted cell lies outside the interior the recovery covers");
    sim
}

/// the isothermal recovery over the store with the velocity and the cell field each kept or zeroed
fn isothermal_case(keep_velocity: bool, keep_field: bool) {
    let sim = isothermal_store(keep_velocity, keep_field);
    let kernels = IsothermalMhdSubstrateKernelSet3D::<DefaultMemory, f64>::new(1.0, 0.3, 1.0, &sim.geom.allocated);
    kernels.c2p(&sim);
    let bad = mismatches(&sim);
    assert!(
        bad.is_empty(),
        "isothermal mhd recovery status (velocity kept {keep_velocity}, field kept {keep_field}):\n  {}\nthe device source:\n{}",
        bad.join("\n  "),
        device_source("imhd_c2p_3d")
    );
}

#[test]
fn the_isothermal_recovery_accepts_every_valid_cell_and_rejects_the_planted_one() {
    isothermal_case(true, true);
}

#[test]
fn the_isothermal_recovery_with_a_zero_cell_field_accepts_every_valid_cell() {
    isothermal_case(true, false);
}

#[test]
fn the_isothermal_recovery_at_rest_accepts_every_valid_cell() {
    isothermal_case(false, true);
}

#[test]
fn the_isothermal_recovery_at_rest_with_a_zero_cell_field_accepts_every_valid_cell() {
    isothermal_case(false, false);
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

#[test]
fn the_isothermal_recovery_probes_report_the_loaded_operands_and_each_condition() {
    use symbi_discretize::gv::IMHD_C2P_PROBE_CHANNELS;
    let sim = isothermal_store(true, true);
    let kernels = IsothermalMhdSubstrateKernelSet3D::<DefaultMemory, f64>::new(1.0, 0.3, 1.0, &sim.geom.allocated);
    kernels.c2p(&sim);
    let store: &symbi::sim::state::FieldStore<3, 3, DefaultMemory, f64> = &sim;
    let probes: Vec<(&str, symbi_grid::Field<f64, 3, DefaultMemory>)> = IMHD_C2P_PROBE_CHANNELS
        .iter()
        .map(|channel| {
            let field = symbi_grid::Field::zeros(&sim.geom.allocated).expect("probe field");
            symbi::regimes::substrate_kernels::dispatch_named(
                store,
                &sim.fields.cons.den,
                Some(&field),
                0,
                &format!("imhd_c2p_probe_{channel}_3d"),
                &sim.geom.interior,
                &[],
                &[],
            );
            (*channel, field)
        })
        .collect();
    sync();
    let mhd = sim.fields.mhd.as_ref().expect("a magnetized store");
    let mut bad = Vec::new();
    for c in sim.geom.interior.iter() {
        let at = |name: &str| *probes.iter().find(|(n, _)| *n == name).expect("a channel").1.view().at(c);
        let mut host = vec![("den", *sim.fields.cons.den.view().at(c))];
        for k in 0..3 {
            host.push((["mom_0", "mom_1", "mom_2"][k], *sim.fields.cons.mom[k].view().at(c)));
        }
        for k in 0..3 {
            host.push((["mag_0", "mag_1", "mag_2"][k], *mhd.bcell.b[k].view().at(c)));
        }
        let conditions = &IMHD_C2P_PROBE_CHANNELS[7..15];
        let status = *sim.fields.c2p_error.view().at(c);
        let mut faults = Vec::new();
        for (name, value) in &host {
            if at(name).to_bits() != value.to_bits() {
                faults.push(format!("{name} loaded {:#018x}, host {:#018x}", at(name).to_bits(), value.to_bits()));
            }
        }
        for name in conditions {
            let want = if c == PLANTED && *name == "rho_positive" { 0.0 } else { 1.0 };
            if at(name) != want {
                faults.push(format!("{name} = {} (want {want})", at(name)));
            }
        }
        let all = conditions.iter().all(|n| at(n) == 1.0);
        if at("accepted") != if all { 1.0 } else { 0.0 } {
            faults.push(format!("accepted = {} against the conjunction {all}", at("accepted")));
        }
        if (status == 0.0) != (at("accepted") == 1.0) {
            faults.push(format!("stored status {status} against accepted {}", at("accepted")));
        }
        if !faults.is_empty() {
            let channels: Vec<String> = probes.iter().map(|(n, f)| format!("{n} {:e} ({:#018x})", f.view().at(c), f.view().at(c).to_bits())).collect();
            bad.push(format!("cell {c:?}: {}\n    every channel: {}", faults.join("; "), channels.join(", ")));
        }
        if bad.len() >= 8 {
            break;
        }
    }
    // the record of what the device compiled, kept with every run's log
    eprintln!("{}\n{}\n{}", compile_options(), device_source("imhd_c2p_3d"), device_source("imhd_c2p_probe_rho_positive_3d"));
    assert!(bad.is_empty(), "isothermal mhd recovery probes (sources and compile options on stderr):\n  {}", bad.join("\n  "));
}
