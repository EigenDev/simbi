// =============================================================================
// conservation_ledger_kernels.rs
//
// the conservation ledger's kernel path (`ledger_gv` integrands reduced where the data lives)
// against its host sums, the independent reference, on the lifted (r, theta) and (R, z) planes
// after production stages of a rotating, flowing gas about a point mass:
// - the wall transport of mass, angular momentum and E + rho Phi and the gross inflow, through
//   the outward-only wall and the plain outflow wall
// - the volume integrals of total energy, kinetic energy and angular momentum
// - the lowest density and specific internal energy with their cells, on the evolved state and
//   with a planted non-finite cell and a planted negative-energy cell
// the sums differ by reassociation alone: the kernel path sums exactly and rounds once, the host
// folds in sequence, so for n terms of one sign they differ by at most (n + 1) u |sum|, u = 2^-53
// (Higham 2002, eq. 4.4). the extremes agree bit for bit, cell for cell.
//
// run: cargo test -p symbi --test conservation_ledger_kernels
// =============================================================================

use std::f64::consts::PI;

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::regimes::wall_ledger;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{Cylindrical, Spherical};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, SofteningKind};
use symbi_sim::conservation_ledger::{Extreme, Transport};
use symbi_sim::driver::{downstream_injection_weight, needs_step_snapshot, stage_exit_fraction, stage_schedule};
use symbi_sim::stage::{StageArgs, StageOutcome, fold_stage};
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const NI: usize = 16;
const NJ: usize = 8;
const U: f64 = 1.1102230246251565e-16;

type Store = FieldStore<2, 3, HostMemory, f64>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

macro_rules! plane {
    ($chart:expr, $origin:expr, $extent:expr, $vr:expr, $outward:expr) => {{
        let walls = [[BoundaryType::Outflow, BoundaryType::Outflow], [BoundaryType::Axis, BoundaryType::Axis]];
        let sim: SimStateGeneric<Newtonian, 2, 3, _, IdealGas<f64>, CpuSpace, HostMemory> = SimStateGeneric::build(Newtonian, IdealGas { gamma: GAMMA }, $chart)
            .cells([NI, NJ])
            .origin($origin)
            .spacing([$extent[0] / NI as f64, $extent[1] / NJ as f64])
            .boundaries(Boundaries::per_axis(walls))
            .cfl(0.1)
            .timestepping(Timestepping::Rk2)
            .allocate()
            .expect("the grid")
            .set_initial(|x: [f64; 2]| Prim::adiabatic(Density(1.0 + 0.3 * (x[1]).sin()), Tensor::new([$vr, 0.01, 0.4 / x[0].max(0.5)]), Pressure(0.2)))
            .build()
            .with_bodies(BodyCollection::new().add(
                Body::gravitational(0, Tensor::zeros(), Tensor::zeros(), 1.0, 0.0, 0.1).with_softening_kind(SofteningKind::Compact),
            ));
        let mut k = Kern::new(GAMMA, 0.1, &sim.geom.allocated);
        k.cell_momentum_gravity_work = false;
        let k = k.with_outward_walls([[false, $outward], [false, false], [false, false]]);
        (sim, k)
    }};
}

/// `steps` production steps; the stored fluxes are the last stage's
macro_rules! march {
    ($sim:expr, $k:expr, $steps:expr) => {{
        let k: &Kern = $k;
        KernelSet::c2p(k, &*$sim);
        KernelSet::ghost_fill(k, &*$sim);
        for _ in 0..$steps {
            let dt = KernelSet::cfl(k, &*$sim);
            assert!(dt > 0.0, "timestep {dt}");
            $sim.dt = dt;
            let stages = $sim.timestepping.stages();
            if needs_step_snapshot(stages) {
                KernelSet::snapshot(k, &*$sim);
            }
            for spec in stage_schedule(stages) {
                let store: &Store = &$sim;
                let outcome = fold_stage(
                    store,
                    k,
                    StageArgs {
                        dt,
                        a0: spec.a0,
                        ac: spec.ac,
                        stage: spec.index,
                        n_stages: stages.len(),
                        injection_weight: downstream_injection_weight(stages, spec.index),
                        allow_elision: true,
                        exit_fraction: stage_exit_fraction(stages, spec.index),
                    },
                    &mut |_| {},
                );
                assert_eq!(outcome, StageOutcome::Accepted);
            }
            $sim.time += dt;
            $sim.iteration += 1;
        }
    }};
}

fn close(label: &str, kernel: f64, host: f64, terms: usize) {
    let bound = (terms as f64 + 1.0) * U * host.abs();
    assert!((kernel - host).abs() <= bound, "{label}: kernel {kernel:e} against host {host:e}, beyond the reassociation bound {bound:e}");
}

/// the energy-flux and potential-energy-flux parts of one wall's E + rho Phi transport
fn wall_parts(store: &Store, k: &Kern, ax: usize, side: usize) -> (f64, f64) {
    use symbi::regimes::substrate_kernels::{LedgerReduce, ledger_reduce};
    use symbi_discretize::LedgerChannel;
    let mut plane = store.geom.interior.clone();
    let face = if side == 0 { plane.spaces[ax].lo } else { plane.spaces[ax].hi };
    plane.spaces[ax].lo = face;
    plane.spaces[ax].hi = face + 1;
    let sum = |channel| ledger_reduce(store, &k.cfl_scratch, channel, &plane, 0.0, 1.0, LedgerReduce::Sum);
    (sum(LedgerChannel::WallEnergy { axis: ax }), sum(LedgerChannel::WallPotential { axis: ax }))
}

fn same_extreme(label: &str, kernel: Extreme, host: Extreme) {
    let same_value = kernel.value.to_bits() == host.value.to_bits() || (kernel.value.is_nan() && host.value.is_nan());
    assert!(same_value && kernel.cell == host.cell, "{label}: kernel {kernel:?} against host {host:?}");
}

fn compare(label: &str, store: &Store, k: &Kern) {
    let kernel = wall_ledger::wall_transport_kernels(store, &k.cfl_scratch).expect("the kernel wall transport");
    let host = wall_ledger::wall_transport_host(store).expect("the host wall transport");
    let mut booked = 0;
    for ax in 0..2 {
        for side in 0..2 {
            let (a, b): (Transport, Transport) = (kernel[ax][side], host[ax][side]);
            let n = if ax == 0 { NJ } else { NI };
            close(&format!("{label} wall {ax}/{side} mass"), a.mass, b.mass, n);
            close(&format!("{label} wall {ax}/{side} mass in"), a.mass_in, b.mass_in, n);
            close(&format!("{label} wall {ax}/{side} angular momentum"), a.angular_momentum, b.angular_momentum, n);
            // E + rho Phi through a wall adds the energy flux and the potential-energy flux, of
            // opposite signs here, so the reassociation bound runs over both magnitudes. the kernel
            // path also evaluates Phi at the face through its cartesian position, as the face
            // gravity work does, where the host takes the radius directly; the conversion and the
            // norm round six times, so the two potentials differ by at most 6 u per face
            let (energy_part, potential_part) = wall_parts(store, k, ax, side);
            let bound = (n as f64 + 2.0) * U * (energy_part.abs() + potential_part.abs()) + 6.0 * U * potential_part.abs();
            assert!(
                (a.energy - b.energy).abs() <= bound,
                "{label} wall {ax}/{side} energy: kernel {:e} against host {:e}, beyond {bound:e}",
                a.energy,
                b.energy
            );
            booked += usize::from(b.mass != 0.0);
        }
    }
    // the polar walls are the symmetry axis and the outward-only wall refuses inflow, so the inner
    // radial wall alone may carry mass
    assert!(booked >= 1, "{label}: no wall carries mass; the wall comparison is vacuous");
    let (a, b) = (wall_ledger::energy_partition_kernels(store, &k.cfl_scratch), wall_ledger::energy_partition_host(store).expect("host partition"));
    assert!(b.angular_momentum > 0.0 && b.kinetic > 0.0, "{label}: the gas carries no rotation");
    close(&format!("{label} energy"), a.energy, b.energy, NI * NJ);
    close(&format!("{label} kinetic"), a.kinetic, b.kinetic, NI * NJ);
    close(&format!("{label} angular momentum"), a.angular_momentum, b.angular_momentum, NI * NJ);
    let (a, b) = (wall_ledger::stage_extremes_kernels(store, &k.cfl_scratch), wall_ledger::stage_extremes_host(store).expect("host extremes"));
    same_extreme(&format!("{label} density"), a.0, b.0);
    same_extreme(&format!("{label} specific internal energy"), a.1, b.1);
}

fn plant(store: &Store, cell: [isize; 2], den: f64, nrg: f64) {
    store.fields.cons.den.view_mut().set(cell, den);
    store.fields.cons.nrg_field().expect("nrg").view_mut().set(cell, nrg);
}

#[test]
fn the_kernel_ledger_matches_the_host_sums_on_the_spherical_plane() {
    for (outward, vr) in [(true, -0.2), (false, -0.2), (true, 0.2)] {
        let (mut sim, k) = plane!(Spherical, [1.0, 0.0], [1.0, PI], vr, outward);
        march!(sim, &k, 3);
        compare(&format!("sphere vr {vr} outward {outward}"), &sim, &k);
        if !outward && vr < 0.0 {
            let host = wall_ledger::wall_transport_host(&*sim).expect("host");
            assert!(host[0][1].mass_in > 0.0, "the plain wall's inflow case books no gross inflow");
        }
    }
}

#[test]
fn the_kernel_ledger_matches_the_host_sums_on_the_cylindrical_plane() {
    let (mut sim, k) = plane!(Cylindrical, [0.5, -1.0], [1.0, 2.0], -0.2, true);
    march!(sim, &k, 3);
    compare("cylinder", &sim, &k);
}

#[test]
fn the_kernel_extremes_report_planted_cells_as_the_host_scan_does() {
    let (mut sim, k) = plane!(Spherical, [1.0, 0.0], [1.0, PI], -0.2, true);
    march!(sim, &k, 1);
    // a negative internal energy at one cell, then a non-finite density at a later cell and an
    // earlier one: the first non-finite cell in the scan order wins
    plant(&sim, [5, 3], 0.5, 1e-6);
    let (a, b) = (wall_ledger::stage_extremes_kernels(&*sim, &k.cfl_scratch), wall_ledger::stage_extremes_host(&*sim).expect("host"));
    assert!(b.1.value < 0.0 && b.1.cell == [5, 3, 0], "the planted negative energy is not the host's minimum: {:?}", b.1);
    same_extreme("negative energy", a.1, b.1);
    plant(&sim, [9, 6], f64::NAN, 1.0);
    plant(&sim, [2, 4], f64::INFINITY, 1.0);
    let (a, b) = (wall_ledger::stage_extremes_kernels(&*sim, &k.cfl_scratch), wall_ledger::stage_extremes_host(&*sim).expect("host"));
    assert!(b.0.value.is_nan() && b.0.cell == [2, 4, 0], "the host scan's first non-finite density: {:?}", b.0);
    same_extreme("non-finite density", a.0, b.0);
    same_extreme("non-finite specific energy", a.1, b.1);
}
