// =============================================================================
// device_local_hydro.rs
//
// flat cartesian hydrodynamics on storage host code cannot dereference, run by the same kernel
// set and the same `evolve` driver as host-dereferenceable storage. two spaces are compared:
// - `Managed`: host-dereferenceable storage (host memory on the cpu build, hip managed memory
//   under the hip feature)
// - `Local`: storage reached through synchronized transfers alone (`HostOpaque`, host-resident
//   and run by the cpu kernels, on the cpu build; `HipDeviceLocal` under the hip feature)
// each test builds its initial state on the host and mirrors it into the space under test.
// - the mirror and the state transfers move every value bit for bit
// - a store outside the device-local scope is refused, naming the feature
// - managed and device-local storage evolve the same bits
// - a smooth entropy wave converges at second order on device-local storage
// - a run restarted through a checkpoint reproduces the straight run's recovered state (ghosts
//   included) and conserved interior bit for bit; the conserved ghost band differs and is reported
// - no kernel of the step reads the conserved ghost band, the correction and replay paths included
// - a run that fails at its initial recovery releases its storage, and the next run proceeds
//
// usage:
//   cargo test -p symbi --test device_local_hydro
//   cargo test -p symbi --features hip --test device_local_hydro -- --test-threads=1
// =============================================================================
#![cfg(any(feature = "hip", not(feature = "gpu")))]

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_grid::Field;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::device_local::{check_scope, download_state, mirror_sim, state_bytes};
use symbi_xpu::{HostMemory, MemoryStorage};

#[cfg(not(feature = "gpu"))]
mod spaces {
    pub type Space = symbi_xpu::CpuSpace;
    pub type Managed = symbi_xpu::HostMemory;
    pub type Local = symbi_xpu::HostOpaque;
}
#[cfg(feature = "hip")]
mod spaces {
    pub type Space = symbi_xpu::HipSpace;
    pub type Managed = symbi_xpu::HipManaged;
    pub type Local = symbi_xpu::HipDeviceLocal;
}
use spaces::{Local, Managed, Space};

type HostSim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, Space, HostMemory>;
type SpaceSim<Mem> = SimStateGeneric<Newtonian, 2, 2, Cartesian, IdealGas<f64>, Space, Mem>;

const GAMMA: f64 = 5.0 / 3.0;
const AMP: f64 = 0.2;
const V: f64 = 1.0;
const P: f64 = 1.0;
// a power of two below the cfl step of every grid here, so every run takes the same steps and
// every step lands on a binary-exact time
const PINNED_DT: f64 = 1.0 / 512.0;

fn rho_exact(x: f64) -> f64 {
    1.0 + AMP * (std::f64::consts::TAU * x).sin()
}

/// the entropy wave `rho = 1 + A sin(2 pi x)`, `v = (V, 0)`, `p = P`, an exact euler solution that
/// returns to its initial profile after one period on the periodic unit square
fn host(n: usize, pressure: impl Fn(f64) -> f64) -> HostSim {
    let dx = 1.0 / n as f64;
    HostSim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n, n])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.4)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim construction")
        .set_initial(|[x, _]| Prim::adiabatic(Density(rho_exact(x)), Tensor::new([V, 0.0]), Pressure(pressure(x))))
        .build()
}

fn wave(n: usize) -> HostSim {
    host(n, |_| P)
}

fn kernels<Mem: MemoryStorage>(sim: &SpaceSim<Mem>) -> AdiabaticSubstrateKernelSet<Mem, f64, 2> {
    AdiabaticSubstrateKernelSet::<Mem, f64, 2>::new(GAMMA, 0.4, &sim.geom.allocated)
}

/// `sim` evolved to `t_end` by the adiabatic kernel set in its own space
fn run<Mem: MemoryStorage>(sim: &mut SpaceSim<Mem>, t_end: f64) -> symbi_xpu::Result<()> {
    let kernels = kernels(sim);
    evolve(sim, &kernels, t_end)
}

fn mirror<Mem: MemoryStorage>(sim: &HostSim) -> SpaceSim<Mem> {
    mirror_sim::<_, 2, 2, _, _, _, HostMemory, Mem>(sim, 0).expect("the mirror")
}

/// the conserved fields of `store` (density, momenta, energy), downloaded whole, ghosts included
fn conserved<Mem: MemoryStorage>(store: &FieldStore<2, 2, Mem, f64>) -> Vec<Vec<f64>> {
    let c = &store.fields.cons;
    let mut fields: Vec<&Field<f64, 2, Mem>> = vec![&c.den, &c.mom[0], &c.mom[1]];
    fields.extend(c.nrg.iter());
    fields.iter().map(|f| f.download().expect("download")).collect()
}

/// the recovered fields of `store` (density, velocities, pressure, recovery status), downloaded
/// whole, ghosts included
fn recovered<Mem: MemoryStorage>(store: &FieldStore<2, 2, Mem, f64>) -> Vec<Vec<f64>> {
    let p = &store.fields.prim;
    let mut fields: Vec<&Field<f64, 2, Mem>> = vec![&p.rho, &p.vel[0], &p.vel[1]];
    fields.extend(p.pre.iter());
    fields.push(&store.fields.c2p_error);
    fields.iter().map(|f| f.download().expect("download")).collect()
}

/// every field of the evolved state, whole
fn evolved<Mem: MemoryStorage>(store: &FieldStore<2, 2, Mem, f64>) -> Vec<Vec<f64>> {
    let mut fields = conserved(store);
    fields.extend(recovered(store));
    fields
}

/// the flat indices of the allocation's cells inside (`inside = true`) or outside the interior
fn cells<Mem: MemoryStorage>(store: &FieldStore<2, 2, Mem, f64>, inside: bool) -> Vec<usize> {
    let (allocated, interior) = (&store.geom.allocated, &store.geom.interior);
    allocated.iter().filter(|&c| interior.contains(c) == inside).map(|c| allocated.flat_index(c)).collect()
}

fn select(fields: Vec<Vec<f64>>, at: &[usize]) -> Vec<Vec<f64>> {
    fields.into_iter().map(|f| at.iter().map(|&ii| f[ii]).collect()).collect()
}

/// every conserved field's ghost band overwritten with NaN, through the synchronized transfers
fn poison_conserved_ghosts<Mem: MemoryStorage>(store: &FieldStore<2, 2, Mem, f64>) {
    let ghosts = cells(store, false);
    let c = &store.fields.cons;
    let mut fields = vec![&c.den, &c.mom[0], &c.mom[1]];
    fields.extend(c.nrg.iter());
    for f in fields {
        let mut values = f.download().expect("download");
        for &ii in &ghosts {
            values[ii] = f64::NAN;
        }
        f.upload(&values).expect("upload");
    }
}

fn assert_same_bits(a: &[Vec<f64>], b: &[Vec<f64>], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: field count");
    for (kk, (fa, fb)) in a.iter().zip(b).enumerate() {
        let differing = fa.iter().zip(fb).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        assert_eq!(differing, 0, "{what}: field {kk} differs in {differing} of {} values", fa.len());
    }
}

#[test]
fn the_mirror_and_the_state_transfers_move_every_value_bit_for_bit() {
    let source = wave(16);
    let device = mirror::<Local>(&source);
    assert!(device.store.host().is_none(), "the device-local store reached a host alias");
    assert_same_bits(&evolved(&source.store), &evolved(&device.store), "mirror");
    let workspace_flag = device.store.workspace.fofc_flag.download().expect("download");
    assert_eq!(workspace_flag.len(), source.geom.allocated.volume(), "the workspace mirrors the allocation");

    let mut target = wave(16);
    for f in [&target.fields.cons.den, &target.fields.prim.rho] {
        f.upload(&vec![-1.0; f.domain().volume()]).expect("upload");
    }
    download_state(&device.store, &mut target.store).expect("the download");
    assert_same_bits(&evolved(&source.store), &evolved(&target.store), "download");

    // nine fields: density, two momenta and energy; density, two velocities and pressure; status
    assert_eq!(state_bytes(&device.store), 9 * source.geom.allocated.volume() * 8);
}

#[test]
fn a_store_outside_the_device_local_scope_is_refused_by_feature() {
    let cases: Vec<(&str, Box<dyn Fn(&mut HostSim)>)> = vec![
        ("mesh motion", Box::new(|s| s.store.motion.a_dot = 0.5)),
        ("Axis boundary", Box::new(|s| s.store.boundaries = Boundaries::uniform(BoundaryType::Axis))),
        ("CoarseFine boundary", Box::new(|s| s.store.boundaries = Boundaries::uniform(BoundaryType::CoarseFine))),
        ("a refined level", Box::new(|s| s.store.composite_ownership.coverage = Some(s.geom.interior.clone()))),
        ("an isothermal closure", Box::new(|s| s.store.fields.cs2 = Some(Field::zeros(&s.geom.allocated).unwrap()))),
        ("a curvilinear chart", Box::new(|s| s.store.geom.coords = symbi_geometry::Geometry::Cylindrical)),
        ("a tile of a decomposed grid", Box::new(|s| s.store.geom.index_offset = [4, 0])),
    ];
    assert!(check_scope(&wave(8).store).is_ok(), "the plain wave lies inside the scope");
    for (feature, mutate) in cases {
        let mut sim = wave(8);
        mutate(&mut sim);
        let refused = mirror_sim::<_, 2, 2, _, _, _, HostMemory, Local>(&sim, 0).map(|_| ());
        let reason = refused.expect_err(feature).0;
        assert!(reason.contains(feature), "refusal names `{reason}`, expected `{feature}`");
    }
}

#[test]
fn managed_and_device_local_storage_evolve_the_same_bits() {
    let mut source = wave(32);
    source.store.max_dt = PINNED_DT;
    let mut managed = mirror::<Managed>(&source);
    let mut local = mirror::<Local>(&source);
    let t_end = 24.0 * PINNED_DT;
    run(&mut managed, t_end).expect("the managed run");
    run(&mut local, t_end).expect("the device-local run");
    assert_eq!((managed.iteration, managed.time), (24, t_end), "the managed run took its pinned steps");
    assert_eq!((local.iteration, local.time), (managed.iteration, managed.time));
    assert_same_bits(&evolved(&managed.store), &evolved(&local.store), "managed vs device-local");
}

/// the L1 density error of the entropy wave after one period on device-local storage
fn l1_after_one_period(n: usize) -> f64 {
    let mut sim = mirror::<Local>(&wave(n));
    run(&mut sim, 1.0).expect("the device-local run");
    let mut host_copy = wave(n);
    download_state(&sim.store, &mut host_copy.store).expect("the download");
    let (rho, dx) = (&host_copy.fields.prim.rho, 1.0 / n as f64);
    let interior: Vec<[isize; 2]> = host_copy.geom.interior.iter().collect();
    interior.iter().map(|&c| (*rho.view().at(c) - rho_exact((c[0] as f64 + 0.5) * dx)).abs()).sum::<f64>()
        / interior.len() as f64
}

#[test]
fn a_smooth_entropy_wave_converges_at_second_order_on_device_local_storage() {
    let (coarse, fine) = (l1_after_one_period(64), l1_after_one_period(128));
    let ratio = coarse / fine;
    eprintln!("device-local entropy wave: L1(64) = {coarse:.3e}, L1(128) = {fine:.3e}, ratio {ratio:.2}");
    // plm + ssp-rk2 is second order: the ratio sits near 4 and a first-order collapse near 2; 3 is
    // the threshold of the cpu order gate on the same wave at the same resolutions. at 32/64 the
    // amplitude-0.2 wave measures 2.63, short of the asymptotic regime
    assert!(fine < coarse && ratio > 3.0, "not second order: L1(64) = {coarse:.3e}, L1(128) = {fine:.3e}");
}

#[test]
fn a_restart_reproduces_the_recovered_state_and_the_conserved_interior() {
    let dir = std::env::temp_dir().join(format!("symbi_device_local_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("half.h5");
    let (t_half, t_end) = (16.0 * PINNED_DT, 32.0 * PINNED_DT);
    let mut source = wave(32);
    source.store.max_dt = PINNED_DT;

    let mut straight = mirror::<Local>(&source);
    run(&mut straight, t_end).expect("the straight run");

    let mut first = mirror::<Local>(&source);
    run(&mut first, t_half).expect("the first segment");
    let mut staging = wave(32);
    download_state(&first.store, &mut staging.store).expect("the checkpoint download");
    symbi_sim::checkpoint::write_checkpoint(&staging, path.to_str().unwrap(), &symbi_io::Metadata::new())
        .expect("the checkpoint");
    drop(first);

    let mut restored = wave(32);
    let meta = symbi_sim::checkpoint::load_checkpoint(&mut restored, path.to_str().unwrap()).expect("the restart");
    restored.store.time = meta.time;
    restored.store.iteration = meta.iteration;
    restored.store.max_dt = PINNED_DT;
    let mut second = mirror::<Local>(&restored);
    run(&mut second, t_end).expect("the second segment");

    assert_eq!((second.iteration, second.time), (straight.iteration, straight.time));
    // the recovered state agrees everywhere, ghosts included, and the conserved state on the
    // interior. the conserved ghost band differs: the straight run holds whatever the band last
    // received, the restart the conserved image of the checkpointed halo primitives. no kernel
    // of the step reads the band (`no_kernel_of_the_step_reads_the_conserved_ghost_band`)
    assert_same_bits(&recovered(&straight.store), &recovered(&second.store), "straight vs restarted, recovered");
    let interior = cells(&straight.store, true);
    assert_same_bits(
        &select(conserved(&straight.store), &interior),
        &select(conserved(&second.store), &interior),
        "straight vs restarted, conserved interior",
    );
    let ghosts = cells(&straight.store, false);
    let (a, b) = (select(conserved(&straight.store), &ghosts), select(conserved(&second.store), &ghosts));
    let differing: usize = a.iter().zip(&b).map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count()).sum();
    eprintln!("restart: conserved ghost band differs in {differing} of {} values", ghosts.len() * a.len());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_run_that_fails_releases_its_storage_and_the_next_run_proceeds() {
    // a negative pressure on the two columns either side of x = 0.5: the initial recovery fails on
    // the device-local status reduction and the run returns the error
    let poisoned = host(16, |x| if (x - 0.5).abs() < 0.05 { -1.0 } else { P });
    let nrg = poisoned.fields.cons.nrg.as_ref().expect("an energy field");
    let poisoned_cells = poisoned.geom.interior.iter().filter(|&c| *nrg.view().at(c) < 0.0).count();
    assert_eq!(poisoned_cells, 2 * 16, "the setup seeded no inadmissible energy; the failure is vacuous");
    let mut failing = mirror::<Local>(&poisoned);
    let failed = run(&mut failing, 0.1).expect_err("a negative pressure evolved");
    assert!(failed.detail.contains("c2p failed on initial conditions"), "unexpected failure: {}", failed.detail);
    assert_eq!(failing.iteration, 0, "the failed run took a step");
    drop(failing);

    let mut next = mirror::<Local>(&wave(16));
    run(&mut next, 4.0 * PINNED_DT).expect("the run after a failure");
    assert!(next.iteration > 0);
}

#[test]
fn no_kernel_of_the_step_reads_the_conserved_ghost_band() {
    // two device-local runs from one state, one with its conserved ghost band overwritten by NaN
    // before the first step and after every step. a kernel reading the band would carry a NaN
    // into the recovered state or the interior. an unrecoverable cell scheduled at step 3 drives
    // the first-order correction, its freeze, and the whole-step replay from the step-entry
    // snapshot through the same test
    let mut source = wave(32);
    source.store.max_dt = PINNED_DT;
    let center = [16isize, 16];
    *source.store.workspace.unrecoverable_once.lock().unwrap() = vec![(3, 0, center, false)];
    let t_end = 12.0 * PINNED_DT;

    let mut clean = mirror::<Local>(&source);
    run(&mut clean, t_end).expect("the clean run");
    let mut poisoned = mirror::<Local>(&source);
    poison_conserved_ghosts(&poisoned.store);
    let kernels = kernels(&poisoned);
    symbi::sim::evolve::evolve_with_callback(&mut poisoned, &kernels, t_end, 1, |sim| poison_conserved_ghosts(&sim.store))
        .expect("the poisoned run");

    for sim in [&clean, &poisoned] {
        let slots = sim.store.workspace.unrecoverable_once.lock().unwrap().clone();
        assert_eq!(slots, vec![(3, 0, center, true)], "the scheduled cell never fired; the correction paths went untested");
    }
    assert_eq!((poisoned.iteration, poisoned.time), (clean.iteration, clean.time));
    let band = select(conserved(&poisoned.store), &cells(&poisoned.store, false));
    assert!(band.iter().flatten().all(|v| v.is_nan()), "the conserved ghost band holds values other than the poison");
    assert_same_bits(&recovered(&clean.store), &recovered(&poisoned.store), "clean vs poisoned band, recovered");
    let interior = cells(&clean.store, true);
    assert_same_bits(
        &select(conserved(&clean.store), &interior),
        &select(conserved(&poisoned.store), &interior),
        "clean vs poisoned band, conserved interior",
    );
}
