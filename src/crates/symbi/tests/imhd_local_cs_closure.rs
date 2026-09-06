// =============================================================================
// imhd_local_cs_closure.rs
//
// the isothermal MHD closure with a prescribed, spatially varying sound speed cs^2(x): a read-only
// Eulerian field every isothermal kernel reads. a uniform disk in solid-body rotation is held by
// the outward pressure gradient of cs^2(r) = a + b r^2 alone, v_phi = r sqrt(2 b); under the
// constant sound speed of its mid-annulus the same disk has no gradient to hold it and flies
// outward, so the balance proves the radial profile reaches the flux. the field survives a
// checkpoint bit for bit and the resumed run continues identically; on a refined hierarchy the
// fine level's field is the prolongation of the root's over its interior and a two-cell band,
// continued outward beyond.
// =============================================================================

use symbi::regimes::substrate_isothermal_mhd::IsothermalMhdSubstrateKernelSet;
use symbi::sim::checkpoint::{load_checkpoint, write_checkpoint};
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_grid::Field;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::Isothermal;
use symbi_hydro::isothermal_mhd::IsothermalMhd;
use symbi_hydro::mhd_state::MhdPrimG;
use symbi_hydro::quantity::Density;
use symbi_hydro::state::PrimG;
use symbi_io::Metadata;
use symbi_refinement::refinement::transfer::prolong_field;
use symbi_xpu::{CpuSpace, HostMemory};

type Disk = SimStateGeneric<IsothermalMhd, 2, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory, f64>;
type Box3 = SimStateGeneric<IsothermalMhd, 3, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory, f64>;
type Kset3 = IsothermalMhdSubstrateKernelSet<HostMemory, f64, 3>;

const CS2_CENTER: f64 = 0.01;
const CS2_SLOPE: f64 = 0.02;
const R_MID: f64 = 1.2;

fn cs2_of(r: f64) -> f64 {
    CS2_CENTER + CS2_SLOPE * r * r
}

// a disk of uniform density in solid-body rotation, v_phi = r sqrt(2 b): with rho uniform the
// radial momentum balance v_phi^2 / r = d(cs^2)/dr = 2 b r holds exactly.
fn disk(n: usize, profile: bool) -> Disk {
    let half = 2.0;
    let dx = 2.0 * half / n as f64;
    let cs_uniform = cs2_of(R_MID).sqrt();
    let sim = Disk::build(IsothermalMhd, Isothermal { cs: cs_uniform }, Cartesian)
        .cells([n; 2])
        .origin([-half; 2])
        .spacing([dx; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(0.3)
        .allocate()
        .expect("disk construction")
        .set_initial(move |[x, y]| {
            let omega = (2.0 * CS2_SLOPE).sqrt();
            let (vx, vy) = (-omega * y, omega * x);
            MhdPrimG::<f64, 3, IsoModel>::new(PrimG::isothermal(Density(1.0), Tensor::new([vx, vy, 0.0])), Tensor::new([0.0; 3]))
        })
        .seed_faces(|_, _| 0.0)
        .build();
    if profile {
        let pre: Field<f64, 2, HostMemory> = Field::zeros(&sim.geom.allocated).unwrap();
        for c in sim.geom.interior.iter() {
            let x = sim.geom.x_lo[0] + (c[0] as f64 + 0.5) * sim.geom.dx[0];
            let y = sim.geom.x_lo[1] + (c[1] as f64 + 0.5) * sim.geom.dx[1];
            pre.set(c, cs2_of((x * x + y * y).sqrt()));
        }
        sim.set_isothermal_cs2_from_pressure(&pre);
    }
    sim
}

// the largest radial speed over the annulus 0.8 <= r <= 1.6 relative to the local rotation speed.
fn radial_residual(sim: &Disk) -> f64 {
    let mut worst = 0.0f64;
    for c in sim.geom.interior.iter() {
        let x = sim.geom.x_lo[0] + (c[0] as f64 + 0.5) * sim.geom.dx[0];
        let y = sim.geom.x_lo[1] + (c[1] as f64 + 0.5) * sim.geom.dx[1];
        let r = (x * x + y * y).sqrt();
        if !(0.8..=1.6).contains(&r) {
            continue;
        }
        let vr = (*sim.fields.prim.vel[0].at(c) * x + *sim.fields.prim.vel[1].at(c) * y) / r;
        worst = worst.max(vr.abs() / (r * (2.0 * CS2_SLOPE).sqrt()));
    }
    worst
}

#[test]
fn a_disk_held_by_the_radial_sound_speed_stays_in_balance_where_the_constant_closure_flies_apart() {
    let steps = 20;
    let run = |profile: bool| -> f64 {
        let sim = disk(64, profile);
        let kset = IsothermalMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(cs2_of(R_MID).sqrt(), 0.3, 1.0, &sim.geom.allocated);
        let mut hier = Hierarchy::single(sim, kset);
        hier.prime();
        for _ in 0..steps {
            hier.step_root_with_dt(5.0e-3);
        }
        radial_residual(&hier.levels[0].state)
    };
    let (balanced, constant) = (run(true), run(false));
    println!("radial residual over the annulus: cs^2(r) profile {balanced:.3e}, constant cs {constant:.3e}");
    assert!(constant > 0.0, "the constant-closure control shows no radial flow; the comparison is vacuous");
    assert!(balanced < 0.2 * constant, "the radial sound-speed profile did not reach the kernels: residual {balanced:.3e} against the constant closure's {constant:.3e}");
    assert!(balanced < 2.0e-2, "the balanced disk drifts radially: {balanced:.3e} of the rotation speed");
}

fn cs2_bits(sim: &Disk) -> Vec<u64> {
    let f = sim.fields.cs2.as_ref().expect("closure field");
    sim.geom.allocated.iter().map(|c| f.at(c).to_bits()).collect()
}

fn prim_bits(sim: &Disk) -> Vec<u64> {
    let mut v = Vec::new();
    for c in sim.geom.interior.iter() {
        v.push(sim.fields.prim.rho.at(c).to_bits());
        for k in 0..3 {
            v.push(sim.fields.prim.vel[k].at(c).to_bits());
        }
    }
    v
}

#[test]
fn the_closure_field_survives_a_checkpoint_and_the_resumed_run_continues_identically() {
    let build = || {
        let sim = disk(32, true);
        let kset = IsothermalMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(cs2_of(R_MID).sqrt(), 0.3, 1.0, &sim.geom.allocated);
        (sim, kset)
    };
    let (mut written, kset) = build();
    let mut hier = Hierarchy::single(written, kset);
    hier.prime();
    for _ in 0..3 {
        hier.step_root_with_dt(5.0e-3);
    }
    written = std::mem::replace(&mut hier.levels[0].state, disk(32, true));
    let path = std::env::temp_dir().join(format!("symbi_iso_cs2_restart_{}.h5", std::process::id()));
    write_checkpoint(&written, path.to_str().unwrap(), &Metadata::new()).expect("checkpoint written");
    let (mut resumed, kset_r) = (disk(32, false), IsothermalMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(cs2_of(R_MID).sqrt(), 0.3, 1.0, &written.geom.allocated));
    assert!(cs2_bits(&resumed) != cs2_bits(&written), "the fresh store already carries the profile; the restart is vacuous");
    load_checkpoint(&mut resumed, path.to_str().unwrap()).expect("checkpoint restored");
    let _ = std::fs::remove_file(&path);
    assert!(cs2_bits(&resumed) == cs2_bits(&written), "the closure field did not survive the restart");
    let kset_w = IsothermalMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(cs2_of(R_MID).sqrt(), 0.3, 1.0, &written.geom.allocated);
    let mut a = Hierarchy::single(written, kset_w);
    let mut b = Hierarchy::single(resumed, kset_r);
    a.prime();
    b.prime();
    for _ in 0..3 {
        a.step_root_with_dt(5.0e-3);
        b.step_root_with_dt(5.0e-3);
    }
    assert!(prim_bits(&a.levels[0].state) == prim_bits(&b.levels[0].state), "the resumed run diverges from the written one");
}

// the fine level's closure field on a refined hierarchy: the prolongation of the root's over the
// fine interior and a two-cell band, continued by the nearest band value beyond it.
#[test]
fn the_fine_level_carries_the_prolonged_closure_field_with_its_ghost_band() {
    const N: usize = 16;
    let dx = 1.0 / N as f64;
    let k = 2.0 * std::f64::consts::PI;
    let root = Box3::build(IsothermalMhd, Isothermal { cs: 1.0 }, Cartesian)
        .cells([N; 3])
        .origin([-0.5; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("root construction")
        .set_initial(|_| MhdPrimG::<f64, 3, IsoModel>::new(PrimG::isothermal(Density(1.0), Tensor::new([0.0; 3])), Tensor::new([0.1, 0.0, 0.0])))
        .seed_faces(|axis, _| if axis == 0 { 0.1 } else { 0.0 })
        .build();
    let pre: Field<f64, 3, HostMemory> = Field::zeros(&root.geom.allocated).unwrap();
    for c in root.geom.interior.iter() {
        let x = root.geom.x_lo[0] + (c[0] as f64 + 0.5) * dx;
        let y = root.geom.x_lo[1] + (c[1] as f64 + 0.5) * dx;
        pre.set(c, 1.0 + 0.5 * (k * x).sin() * (k * y).cos());
    }
    root.set_isothermal_cs2_from_pressure(&pre);
    let kset = |s: &Box3| Kset3::new(1.0, 0.3, 1.0, &s.geom.allocated);
    let ck = kset(&root);
    let regions = [RefinementRegion { x_lo: [-0.25; 3], x_hi: [0.25; 3] }];
    let hier = Hierarchy::with_refinement(root, ck, &regions, ProlongOrder::Ppm, kset).unwrap();
    hier.seed_fine_from_coarse().expect("fine seed");
    let coarse = &hier.levels[0].state;
    let fine = &hier.levels[1].state;
    let (ccs, fcs) = (coarse.fields.cs2.as_ref().unwrap(), fine.fields.cs2.as_ref().unwrap());
    let mut band = fine.geom.interior.clone();
    for ax in 0..3 {
        band = band.extend(ax, 2, 2);
    }
    let expected: Field<f64, 3, HostMemory> = Field::zeros(&fine.geom.allocated).unwrap();
    let zero: Field<f64, 3, HostMemory> = Field::zeros(&coarse.geom.allocated).unwrap();
    prolong_field(ccs, &zero, &expected, &band, ProlongOrder::Ppm, 0.0);
    let mut varies = false;
    for c in band.iter() {
        assert_eq!(fcs.at(c).to_bits(), expected.at(c).to_bits(), "the fine closure field at {c:?} is not the prolongation of the root's");
        varies |= (*fcs.at(c) - 1.0).abs() > 1e-3;
    }
    assert!(varies, "the fine closure field is uniform; the prolongation is vacuous");
    for c in fine.geom.allocated.iter() {
        if band.contains(c) {
            continue;
        }
        let nearest: [isize; 3] = std::array::from_fn(|a| c[a].clamp(band.spaces[a].lo, band.spaces[a].hi - 1));
        assert_eq!(fcs.at(c).to_bits(), fcs.at(nearest).to_bits(), "the fine ghost at {c:?} is not the band's nearest value");
    }
    let mut hier = hier;
    hier.prime();
    hier.step_root_with_dt(1.0e-3);
    assert!(hier.levels[1].state.geom.interior.iter().all(|c| hier.levels[1].state.fields.prim.rho.at(c).is_finite()));
}

