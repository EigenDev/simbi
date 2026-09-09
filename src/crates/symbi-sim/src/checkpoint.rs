// =============================================================================
// checkpoint.rs
//
// HDF5 checkpoint API for SimState. a thin builder that maps SimState onto a
// `symbi_io::Tree` schema and hands off to `symbi_io::Hdf5Backend`. the I/O
// concerns (file format, field naming, error handling) live in the symbi-io
// crate; this module only describes what SimState contributes to the schema.
//
// extras ride in as a typed `symbi_io::Metadata` — callers build
// `Metadata::new().with("key", value)` with naked typed values — and
// `write_checkpoint` / `load_checkpoint` / `read_checkpoint_meta` all return
// `Result<_, symbi_io::IoError>`.
// =============================================================================

use std::path::Path;

use crate::state::*;
use crate::tracers as symbi_sim_tracers;
use symbi_geometry::Metric;
use symbi_hydro::eos::Eos;
use symbi_hydro::regime::Regime;
use symbi_xpu::{ExecutionSpace, MemorySpace};

pub use symbi_io::{Attr, IoError, Metadata, Result};
use symbi_io::{DataRef, Dataset, Hdf5Backend, Hdf5Stream, IoBackend, Tree, TreeBuf};
use symbi_io::{dataset_shape, read_attrs, read_group, read_slab};

/// the homologous mesh-motion factor applied to axis `ax` of a `d`-dimensional
/// grid: cartesian expands every axis, spherical the radius only, cylindrical
/// the in-plane r and axial z slots. shared by the checkpoint writer (which
/// stores physical, scaled bounds) and the restart region check (which must
/// unscale them back to the comoving grid) so the two cannot disagree about
/// which axes a stored bound was scaled by.
fn motion_axis_scale(geometry: symbi_geometry::Geometry, ax: usize, d: usize, a: f64) -> f64 {
    match geometry {
        symbi_geometry::Geometry::Cartesian => a,
        symbi_geometry::Geometry::Spherical => {
            if ax == 0 {
                a
            } else {
                1.0
            }
        }
        symbi_geometry::Geometry::Cylindrical => {
            if ax == 0 || ax == d - 1 {
                a
            } else {
                1.0
            }
        }
    }
}


pub fn time_at_or_after(time: f64, boundary: f64) -> bool {
    let tolerance = 32.0 * f64::EPSILON * time.abs().max(boundary.abs());
    time >= boundary || (time - boundary).abs() <= tolerance
}

// =============================================================================
// Snapshot — the materialized Vec<f64> buffers a write needs to borrow
// when building the Tree. one struct, holds every field's interior data,
// no copies during Tree construction.
// =============================================================================



/// visit every cell of `domain` in axis-0-fastest order (x varies fastest) — the on-disk
/// checkpoint layout, so numpy `arr.reshape((Nz, Ny, Nx))` puts physical x on the horizontal
/// `imshow` axis. gather (`extract_field`) and scatter (`restore_field`) must share this one
/// walk: if their orders diverge, a written-then-loaded D>=2 field comes back transposed.
fn for_each_cell_axis0<const D: usize>(
    domain: &symbi_algebra::Domain<D>,
    mut visit: impl FnMut([isize; D]),
) {
    let vol = domain.volume();
    let mut coord: [isize; D] = std::array::from_fn(|ax| domain.spaces[ax].lo);
    for _ in 0..vol {
        visit(coord);
        for ax in 0..D {
            coord[ax] += 1;
            if coord[ax] < domain.spaces[ax].hi {
                break;
            }
            coord[ax] = domain.spaces[ax].lo;
        }
    }
}




// =============================================================================
// build_tree — the schema description SimState contributes. one place that
// owns the on-disk layout; the Hdf5Backend (or any other) walks it.
// =============================================================================

fn regime_name<R: Regime<f64, D>, const D: usize>(r: &R) -> &'static str {
    // the checkpoint carries the regime in the configuration vocabulary (newtonian, isothermal,
    // rhd, rmhd, nmhd, imhd), so a restart parses it back into the same enum the run was
    // configured from. `has_energy() == false` marks the isothermal (IsoModel) regimes, which
    // carry no energy equation.
    if r.is_mhd() {
        if r.is_relativistic() {
            "rmhd"
        } else if r.has_energy() {
            "nmhd"
        } else {
            "imhd"
        }
    } else if r.is_relativistic() {
        "rhd"
    } else if r.has_energy() {
        "newtonian"
    } else {
        "isothermal"
    }
}

fn coord_name(g: symbi_geometry::Geometry) -> &'static str {
    match g {
        symbi_geometry::Geometry::Cartesian => "cartesian",
        symbi_geometry::Geometry::Spherical => "spherical",
        symbi_geometry::Geometry::Cylindrical => "cylindrical",
    }
}

fn spacetime_name(s: symbi_geometry::Spacetime) -> &'static str {
    match s {
        symbi_geometry::Spacetime::Minkowski => "minkowski",
        symbi_geometry::Spacetime::SchwarzschildKS => "schwarzschild_ks",
        symbi_geometry::Spacetime::KerrKS => "kerr_ks",
    }
}

fn timestepping_name(t: Timestepping) -> &'static str {
    match t {
        Timestepping::Euler => "euler",
        Timestepping::Rk2 => "rk2",
        Timestepping::Rk3 => "rk3",
    }
}

/// the global `/metadata` group — time/physics/scheme attrs (same across an AMR
/// hierarchy, so authored from the coarse level) + the coarse mesh datasets for
/// single-level readers.
/// the physics a checkpoint identifies its run by, read once from a state: the regime and its
/// closure, the equation of state's index, the isothermal sound speed when the closure has no
/// energy, and the coordinate chart. every tile of a decomposed run shares it.
#[derive(Clone, Debug)]
pub struct PhysicsIdentity {
    pub regime: &'static str,
    pub is_mhd: bool,
    pub is_relativistic: bool,
    pub has_energy: bool,
    pub gamma: f64,
    pub sound_speed: Option<f64>,
    pub geometry: symbi_geometry::Geometry,
}

impl PhysicsIdentity {
    pub fn of<R, const D: usize, const DOF: usize, M, E, S, Mem>(
        sim: &SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    ) -> Self
    where
        R: Regime<f64, D>,
        M: Metric<f64, D> + Copy,
        E: Eos<f64>,
        S: ExecutionSpace,
        Mem: MemorySpace,
    {
        let has_energy = sim.physics.regime.has_energy();
        Self {
            regime: regime_name(&sim.physics.regime),
            is_mhd: sim.physics.regime.is_mhd(),
            is_relativistic: sim.physics.regime.is_relativistic(),
            has_energy,
            gamma: sim.physics.eos.gamma(),
            sound_speed: (!has_energy).then(|| {
                sim.physics.eos.sound_speed(
                    symbi_hydro::quantity::Density(1.0),
                    symbi_hydro::quantity::Pressure(1.0),
                )
            }),
            geometry: sim.physics.metric.geometry(),
        }
    }
}

/// the coarse level's mesh facts the metadata group records for single-level readers: interior
/// cell counts, physical cell widths and physical lower bounds per axis, all scaled by the
/// mesh motion.
pub struct MeshFacts {
    pub resolution: Vec<u64>,
    pub dx_phys: Vec<f64>,
    pub x_lo_phys: Vec<f64>,
}

fn build_metadata_group<'a, const D: usize, const DOF: usize, Mem: MemorySpace>(
    identity: &'a PhysicsIdentity,
    store: &'a FieldStore<D, DOF, Mem, f64>,
    facts: &'a MeshFacts,
    extras: &'a Metadata,
) -> Tree<'a> {
    // builtins. user extras can override any name here — explicit win.
    let mut builtins: Vec<(&str, Attr)> = vec![
        ("gamma", Attr::F64(identity.gamma)),
        ("cfl", Attr::F64(store.cfl)),
        ("time", Attr::F64(store.time)),
        ("dt", Attr::F64(store.dt)),
        ("iteration", Attr::U64(store.iteration as u64)),
        ("dimensions", Attr::U64(D as u64)),
        ("halo_radius", Attr::U64(store.geom.ng as u64)),
        ("scale_factor", Attr::F64(store.motion.a)),
        ("scale_factor_dot", Attr::F64(store.motion.a_dot)),
        ("homologous", Attr::Bool(store.motion.homologous)),
        ("regime", Attr::Str(identity.regime.into())),
        ("is_mhd", Attr::Bool(identity.is_mhd)),
        ("is_relativistic", Attr::Bool(identity.is_relativistic)),
        (
            "timestepping",
            Attr::Str(timestepping_name(store.timestepping).into()),
        ),
        ("coord_system", Attr::Str(coord_name(identity.geometry).into())),
        // the background spacetime chart — orthogonal to coord_system. GR readers need
        // this to select the metric (lapse, shift, densitization) when reducing fluxes.
        (
            "spacetime",
            Attr::Str(spacetime_name(store.geom.spacetime).into()),
        ),
    ];
    // the curved-spacetime scalar params (schwarzschild_mass, kerr_spin) ride as
    // named attrs so a reader can reconstruct the metric; empty on a flat background.
    for (name, value) in &store.geom.spacetime_scalars {
        builtins.push((name.as_str(), Attr::F64(*value)));
    }
    // isothermal regimes close with p = cs^2 rho at a constant sound speed
    // and store no pressure dataset; record cs so readers can reconstruct
    // pressure-dependent fields.
    if let Some(cs) = identity.sound_speed {
        builtins.push(("sound_speed", Attr::F64(cs)));
    }
    let mut meta = Tree::new("metadata");
    // explicit user extras win. start with them, then fill in any built-in
    // the user didn't override.
    for (k, v) in extras {
        meta.push_attr(k.to_string(), v.clone());
    }
    for (k, v) in builtins {
        if extras.get(k).is_some() {
            continue;
        }
        meta.push_attr(k.to_string(), v);
    }
    // coarsest-level mesh info (backward compat with single-level readers).
    meta.push_dataset(Dataset::new(
        "resolution",
        vec![D],
        DataRef::U64(&facts.resolution),
    ));
    meta.push_dataset(Dataset::new("dx", vec![D], DataRef::F64(&facts.dx_phys)));
    meta.push_dataset(Dataset::new("x_lo", vec![D], DataRef::F64(&facts.x_lo_phys)));
    meta
}



// =============================================================================
// body state round-trip: the per-body kinematic + accretion ledger a restart
// must restore. bodies re-attach from the config on restart, so without this
// group a moving body's orbit phase and a sink's cumulative accreted mass
// silently reset — wrong physics for binaries, a broken cumulative ledger for
// accretors.
// =============================================================================

/// the materialized per-body state buffers a write borrows (Snapshot-style).
struct BodyStateSnap {
    pos: Vec<f64>,             // [nb, D]
    vel: Vec<f64>,             // [nb, D]
    mass: Vec<f64>,            // [nb]
    accreted: Vec<f64>,        // [nb] cumulative rest mass (0 for non-sinks)
    rate: Vec<f64>,            // [nb] instantaneous Mdot (0 for non-sinks)
    accreted_energy: Vec<f64>, // [nb] cumulative covariant (killing) energy (GR horizon only)
    energy_rate: Vec<f64>,     // [nb] instantaneous Edot (GR horizon only)
    slip_heat: Vec<f64>,       // [nb] cumulative magnetic-slip heat released by the body's shell
    slip_heat_rate: Vec<f64>,  // [nb] its rate over the last step
    ang_mom: Vec<f64>,         // [nb, 3] world-frame angular momentum L = I omega
    ke_trans: Vec<f64>,        // [nb] translational kinetic energy 0.5 m |v|^2
    ke_rot: Vec<f64>,          // [nb] rotational kinetic energy 0.5 omega.I.omega
    orientation: Vec<f64>,     // [nb, 3, 3] row-major rotation matrix (evolved spin state)
    omega: Vec<f64>,           // [nb, 3] angular-velocity vector
    shape_json: Vec<String>,   // per-body CSG wire (empty = the analytic sphere), for viz
    nb: usize,
}

fn body_state_snap<const D: usize>(im: &ImmersedBodies<D>) -> BodyStateSnap {
    let nb = im.bodies.len();
    let mut snap = BodyStateSnap {
        pos: Vec::with_capacity(nb * D),
        vel: Vec::with_capacity(nb * D),
        mass: Vec::with_capacity(nb),
        accreted: Vec::with_capacity(nb),
        rate: Vec::with_capacity(nb),
        accreted_energy: Vec::with_capacity(nb),
        energy_rate: Vec::with_capacity(nb),
        slip_heat: Vec::with_capacity(nb),
        slip_heat_rate: Vec::with_capacity(nb),
        ang_mom: Vec::with_capacity(nb * 3),
        ke_trans: Vec::with_capacity(nb),
        ke_rot: Vec::with_capacity(nb),
        orientation: Vec::with_capacity(nb * 9),
        omega: Vec::with_capacity(nb * 3),
        shape_json: Vec::with_capacity(nb),
        nb,
    };
    for b in 0..nb {
        // the CSG shape wire (empty for the analytic sphere) — a self-describing silhouette for viz.
        snap.shape_json.push(
            im.shapes
                .get(b)
                .and_then(|s| s.as_ref())
                .map(|s| s.to_json())
                .unwrap_or_default(),
        );
        let body = im.bodies.get(b);
        for a in 0..D {
            snap.pos.push(body.position[a]);
            snap.vel.push(body.velocity[a]);
        }
        // the evolved rigid-body rotation: the orientation matrix (row-major) + angular velocity.
        for i in 0..3 {
            for j in 0..3 {
                snap.orientation.push(body.orientation[i][j]);
            }
        }
        for k in 0..3 {
            snap.omega.push(body.omega[k]);
        }
        // the rigid-body ledgers: world-frame angular momentum + the split kinetic energy, so viz can
        // draw an L glyph and plot the translational/rotational energy budget.
        let l = body.angular_momentum();
        for k in 0..3 {
            snap.ang_mom.push(l[k]);
        }
        snap.ke_trans.push(body.translational_ke());
        snap.ke_rot.push(body.rotational_ke());
        snap.mass.push(body.mass);
        let (acc, rate) = match body.kind {
            symbi_ib::BodyKind::BlackHole {
                total_accreted_mass,
                accretion_rate,
                ..
            } => (total_accreted_mass, accretion_rate),
            // the GR horizon books the shell-flux rest-mass ledger into the same datasets (Mdot).
            symbi_ib::BodyKind::Horizon {
                total_accreted_mass,
                mdot,
                ..
            } => (total_accreted_mass, mdot),
            _ => (0.0, 0.0),
        };
        snap.accreted.push(acc);
        snap.rate.push(rate);
        // the GR horizon also books the covariant (killing) energy ledger (Edot, cumulative E).
        let (acc_e, rate_e) = match body.kind {
            symbi_ib::BodyKind::Horizon {
                total_accreted_energy,
                edot,
                ..
            } => (total_accreted_energy, edot),
            _ => (0.0, 0.0),
        };
        snap.accreted_energy.push(acc_e);
        snap.energy_rate.push(rate_e);
        snap.slip_heat.push(body.slip_heat_total);
        snap.slip_heat_rate.push(body.slip_heat_rate);
    }
    snap
}

/// flattened tracer state for the checkpoint: positions row-major [n, D],
/// ids and flags as f64 (ids stay exact below 2^53; flags are 0/1).
struct TracerSnap {
    n: usize,
    x: Vec<f64>,
    id: Vec<u64>,
    cohort: Vec<u64>,
    owner: Vec<u64>,
    run_seed: u64,
    next_id: u64,
    injection_remainder: f64,
    escaped: Vec<f64>,
    crossed: Vec<f64>,
    crossing_time: Vec<f64>,
    weight: Vec<f64>,
}

fn tracer_snap<const D: usize>(tr: &symbi_sim_tracers::TracerSet<D>) -> TracerSnap {
    let n = tr.len();
    let mut x = Vec::with_capacity(n * D);
    for p in &tr.x {
        x.extend_from_slice(&p[..]);
    }
    TracerSnap {
        n,
        x,
        id: tr.id.clone(),
        cohort: tr.cohort.iter().map(|&cohort| cohort as u64).collect(),
        owner: tr.owner.iter().map(|owner| owner.0).collect(),
        run_seed: tr.run_seed,
        next_id: tr.next_id,
        injection_remainder: tr.injection_remainder,
        escaped: tr.flags.iter().map(|f| f.escaped as u8 as f64).collect(),
        crossed: tr
            .flags
            .iter()
            .map(|f| f.crossed_sink as u8 as f64)
            .collect(),
        crossing_time: tr.flags.iter().map(|f| f.crossing_time).collect(),
        weight: vec![tr.weight],
    }
}

fn tracer_group<const D: usize>(snap: &TracerSnap) -> Tree<'_> {
    Tree::new("tracers")
        .with_attr("n_tracers", snap.n as u64)
        .with_attr("run_seed", snap.run_seed)
        .with_attr("next_id", snap.next_id)
        .with_attr("injection_remainder", snap.injection_remainder)
        .with_dataset(Dataset::new(
            "position",
            vec![snap.n, D],
            DataRef::F64(&snap.x),
        ))
        .with_dataset(Dataset::new("id", vec![snap.n], DataRef::U64(&snap.id)))
        .with_dataset(Dataset::new(
            "cohort",
            vec![snap.n],
            DataRef::U64(&snap.cohort),
        ))
        .with_dataset(Dataset::new(
            "owner",
            vec![snap.n],
            DataRef::U64(&snap.owner),
        ))
        .with_dataset(Dataset::new(
            "escaped",
            vec![snap.n],
            DataRef::F64(&snap.escaped),
        ))
        .with_dataset(Dataset::new(
            "crossed_sink",
            vec![snap.n],
            DataRef::F64(&snap.crossed),
        ))
        .with_dataset(Dataset::new(
            "crossing_time",
            vec![snap.n],
            DataRef::F64(&snap.crossing_time),
        ))
        .with_dataset(Dataset::new("weight", vec![1], DataRef::F64(&snap.weight)))
}

struct ContinuousTracerSnap {
    n: usize,
    order: u64,
    x: Vec<f64>,
    step_x: Vec<f64>,
    id: Vec<u64>,
    cohort: Vec<u64>,
    owner: Vec<u64>,
    escaped: Vec<u64>,
    crossed_sink: Vec<u64>,
    crossing_time: Vec<f64>,
    random_counter: Vec<u64>,
    weight: Vec<f64>,
    run_seed: u64,
    next_id: u64,
    injection_remainder: f64,
}

fn continuous_tracer_snap<const D: usize, Mem: MemorySpace>(
    tracers: &symbi_sim_tracers::ContinuousTracerSet<D, Mem>,
) -> ContinuousTracerSnap {
    assert!(
        Mem::IS_HOST_ACCESSIBLE,
        "continuous tracer checkpointing requires host-accessible storage"
    );
    let n = tracers.len;
    unsafe {
        ContinuousTracerSnap {
            n,
            order: tracers.order as u64,
            x: (0..n)
                .flat_map(|ii| (0..D).map(move |dd| *tracers.x[dd].as_ptr::<f64>().add(ii)))
                .collect(),
            step_x: (0..n)
                .flat_map(|ii| (0..D).map(move |dd| *tracers.step_x[dd].as_ptr::<f64>().add(ii)))
                .collect(),
            id: std::slice::from_raw_parts(tracers.id.as_ptr::<u64>(), n).to_vec(),
            cohort: std::slice::from_raw_parts(tracers.cohort.as_ptr::<u16>(), n)
                .iter()
                .map(|value| *value as u64)
                .collect(),
            owner: std::slice::from_raw_parts(
                tracers.owner.as_ptr::<crate::mass_transport::ContainerId>(),
                n,
            )
            .iter()
            .map(|owner| owner.0)
            .collect(),
            escaped: std::slice::from_raw_parts(tracers.escaped.as_ptr::<u8>(), n)
                .iter()
                .map(|value| *value as u64)
                .collect(),
            crossed_sink: std::slice::from_raw_parts(tracers.crossed_sink.as_ptr::<u8>(), n)
                .iter()
                .map(|value| *value as u64)
                .collect(),
            crossing_time: std::slice::from_raw_parts(tracers.crossing_time.as_ptr::<f64>(), n)
                .to_vec(),
            random_counter: std::slice::from_raw_parts(tracers.random_counter.as_ptr::<u64>(), n)
                .to_vec(),
            weight: vec![tracers.weight],
            run_seed: tracers.run_seed,
            next_id: tracers.next_id,
            injection_remainder: tracers.injection_remainder,
        }
    }
}

fn continuous_tracer_group<const D: usize>(snap: &ContinuousTracerSnap) -> Tree<'_> {
    Tree::new("continuous_tracers")
        .with_attr("n_tracers", snap.n as u64)
        .with_attr("order", snap.order)
        .with_attr("run_seed", snap.run_seed)
        .with_attr("next_id", snap.next_id)
        .with_attr("injection_remainder", snap.injection_remainder)
        .with_dataset(Dataset::new(
            "position",
            vec![snap.n, D],
            DataRef::F64(&snap.x),
        ))
        .with_dataset(Dataset::new(
            "step_position",
            vec![snap.n, D],
            DataRef::F64(&snap.step_x),
        ))
        .with_dataset(Dataset::new("id", vec![snap.n], DataRef::U64(&snap.id)))
        .with_dataset(Dataset::new(
            "cohort",
            vec![snap.n],
            DataRef::U64(&snap.cohort),
        ))
        .with_dataset(Dataset::new(
            "owner",
            vec![snap.n],
            DataRef::U64(&snap.owner),
        ))
        .with_dataset(Dataset::new(
            "escaped",
            vec![snap.n],
            DataRef::U64(&snap.escaped),
        ))
        .with_dataset(Dataset::new(
            "crossed_sink",
            vec![snap.n],
            DataRef::U64(&snap.crossed_sink),
        ))
        .with_dataset(Dataset::new(
            "crossing_time",
            vec![snap.n],
            DataRef::F64(&snap.crossing_time),
        ))
        .with_dataset(Dataset::new(
            "random_counter",
            vec![snap.n],
            DataRef::U64(&snap.random_counter),
        ))
        .with_dataset(Dataset::new("weight", vec![1], DataRef::F64(&snap.weight)))
}

fn combine_continuous_tracer_snaps(
    mut snaps: impl Iterator<Item = ContinuousTracerSnap>,
) -> Option<ContinuousTracerSnap> {
    let mut combined = snaps.next()?;
    for snap in snaps {
        assert_eq!(snap.order, combined.order);
        assert_eq!(snap.run_seed, combined.run_seed);
        assert_eq!(snap.next_id, combined.next_id);
        assert_eq!(
            snap.injection_remainder.to_bits(),
            combined.injection_remainder.to_bits()
        );
        assert_eq!(snap.weight, combined.weight);
        combined.n += snap.n;
        combined.x.extend(snap.x);
        combined.step_x.extend(snap.step_x);
        combined.id.extend(snap.id);
        combined.cohort.extend(snap.cohort);
        combined.owner.extend(snap.owner);
        combined.escaped.extend(snap.escaped);
        combined.crossed_sink.extend(snap.crossed_sink);
        combined.crossing_time.extend(snap.crossing_time);
        combined.random_counter.extend(snap.random_counter);
    }
    Some(combined)
}

/// the dataset naming the magnetic-slip heat by its fate under the run's closure: deposited in
/// the gas (adiabatic) or exported to the cooling bath (isothermal).
fn slip_heat_dataset_name(has_energy: bool) -> &'static str {
    if has_energy {
        "magnetic_slip_heating"
    } else {
        "exported_slip_heat"
    }
}

fn body_state_group<'a, const D: usize>(snap: &'a BodyStateSnap, heat_name: &str) -> Tree<'a> {
    let nb = snap.nb;
    let mut t = Tree::new("bodies")
        .with_attr("n_bodies", nb as u64)
        .with_dataset(Dataset::new(
            "position",
            vec![nb, D],
            DataRef::F64(&snap.pos),
        ))
        .with_dataset(Dataset::new(
            "velocity",
            vec![nb, D],
            DataRef::F64(&snap.vel),
        ))
        .with_dataset(Dataset::new("mass", vec![nb], DataRef::F64(&snap.mass)))
        .with_dataset(Dataset::new(
            "total_accreted_mass",
            vec![nb],
            DataRef::F64(&snap.accreted),
        ))
        .with_dataset(Dataset::new(
            "accretion_rate",
            vec![nb],
            DataRef::F64(&snap.rate),
        ))
        .with_dataset(Dataset::new(
            "total_accreted_energy",
            vec![nb],
            DataRef::F64(&snap.accreted_energy),
        ))
        .with_dataset(Dataset::new(
            "accretion_energy_rate",
            vec![nb],
            DataRef::F64(&snap.energy_rate),
        ))
        .with_dataset(Dataset::new(
            heat_name.to_string(),
            vec![nb],
            DataRef::F64(&snap.slip_heat),
        ))
        .with_dataset(Dataset::new(
            format!("{heat_name}_rate"),
            vec![nb],
            DataRef::F64(&snap.slip_heat_rate),
        ))
        .with_dataset(Dataset::new(
            "orientation",
            vec![nb, 3, 3],
            DataRef::F64(&snap.orientation),
        ))
        .with_dataset(Dataset::new(
            "omega",
            vec![nb, 3],
            DataRef::F64(&snap.omega),
        ))
        .with_dataset(Dataset::new(
            "angular_momentum",
            vec![nb, 3],
            DataRef::F64(&snap.ang_mom),
        ))
        .with_dataset(Dataset::new(
            "ke_translational",
            vec![nb],
            DataRef::F64(&snap.ke_trans),
        ))
        .with_dataset(Dataset::new(
            "ke_rotational",
            vec![nb],
            DataRef::F64(&snap.ke_rot),
        ));
    // the per-body CSG shape wire as a string attr (empty = analytic sphere); viz reconstructs the
    // body silhouette from it + the position/orientation.
    for (b, wire) in snap.shape_json.iter().enumerate() {
        t.push_attr(format!("shape_{b}"), wire.clone());
    }
    t
}

/// root attribute naming the measure the stored conserved state carries. absent means the
/// undensitized Valencia state (`D`, `S_i`, `ehat` per unit coordinate volume).
const CONSERVED_DENSITIZATION_ATTR: &str = "conserved_densitization";
/// the value written for the free-index-down GR-hydro state
/// `sqrt(-g)[rho u^t, T^t_i, -(T^t_t + rho u^t)]`.
const SQRT_MINUS_G: &str = "sqrt_minus_g";

/// the measure the run's conserved state carries, or `None` when it is undensitized. relativistic
/// hydro on a curved spacetime stores the fully densitized state; every other configuration —
/// flat spacetime, and GR MHD, whose induction and CT seam are still Valencia — does not. the two
/// states differ by a per-cell factor `sqrt(-g)(x)`, so reloading one as the other is silently
/// wrong rather than loud, which is why the file records which it holds.
fn conserved_densitization<const D: usize, const DOF: usize, Mem: MemorySpace>(
    store: &FieldStore<D, DOF, Mem, f64>,
) -> Option<&'static str> {
    let curved = store.geom.spacetime != symbi_geometry::Spacetime::Minkowski;
    (curved && store.fields.mhd.is_none()).then_some(SQRT_MINUS_G)
}


fn reduction_op_tag(op: symbi_ir::emit::ReductionOp) -> String {
    match op {
        symbi_ir::emit::ReductionOp::Add => "add",
        symbi_ir::emit::ReductionOp::Min => "min",
        symbi_ir::emit::ReductionOp::Max => "max",
        symbi_ir::emit::ReductionOp::Mul => {
            unreachable!("a product is refused at census registration")
        }
    }
    .to_string()
}

// =============================================================================
// public API: write_checkpoint / load_checkpoint / read_checkpoint_meta
// =============================================================================

/// write a checkpoint. typed `Metadata` carries naked typed values,
/// so there is no `to_string()` boilerplate at call sites:
///
/// ```ignore
/// let extras = Metadata::new()
///     .with("problem", "kepler")
///     .with("ring_r0", 1.0)
///     .with("gm",      gm);
/// write_checkpoint(&sim, "kepler_0001.h5", &extras)?;
/// ```
pub fn write_checkpoint<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    path: &str,
    extras: &Metadata,
) -> Result<()>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    write_partitioned_checkpoint::<R, D, DOF, Mem>(
        &PhysicsIdentity::of(sim),
        &[LevelTiles::whole(&sim.store)],
        path,
        extras,
    )
}

/// write every level of a hierarchy, each held whole by one state, into one file: `levels[0]`
/// is the coarse level and authors the global metadata.
pub fn write_hierarchy_checkpoint<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    levels: &[&SimStateGeneric<R, D, DOF, M, E, S, Mem>],
    path: &str,
    extras: &Metadata,
) -> Result<()>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    let tiles: Vec<LevelTiles<'_, D, DOF, Mem>> =
        levels.iter().map(|s| LevelTiles::whole(&s.store)).collect();
    write_partitioned_checkpoint::<R, D, DOF, Mem>(&PhysicsIdentity::of(levels[0]), &tiles, path, extras)
}

// =============================================================================
// CheckpointMeta — typed view of the metadata group, parsed once from the
// TreeBuf the read side returns.
// =============================================================================

#[derive(Clone, Debug)]
pub struct CheckpointMeta {
    pub time: f64,
    pub dt: f64,
    pub iteration: u64,
    pub gamma: f64,
    pub dimensions: u64,
    pub regime: String,
    pub coord_system: String,
}

fn read_meta_from(tree: &TreeBuf) -> Result<CheckpointMeta> {
    let m = tree
        .find_group("metadata")
        .ok_or_else(|| IoError::MissingPath("metadata".into()))?;
    Ok(CheckpointMeta {
        time: m
            .find_attr("time")
            .ok_or_else(|| IoError::MissingPath("metadata/time".into()))?
            .as_f64("metadata/time")?,
        dt: m
            .find_attr("dt")
            .ok_or_else(|| IoError::MissingPath("metadata/dt".into()))?
            .as_f64("metadata/dt")?,
        iteration: m
            .find_attr("iteration")
            .ok_or_else(|| IoError::MissingPath("metadata/iteration".into()))?
            .as_u64("metadata/iteration")?,
        gamma: m
            .find_attr("gamma")
            .ok_or_else(|| IoError::MissingPath("metadata/gamma".into()))?
            .as_f64("metadata/gamma")?,
        dimensions: m
            .find_attr("dimensions")
            .ok_or_else(|| IoError::MissingPath("metadata/dimensions".into()))?
            .as_u64("metadata/dimensions")?,
        // strings live as byte-array datasets (on-disk convention)
        regime: read_str_dataset(m, "regime").unwrap_or_else(|_| "unknown".into()),
        coord_system: read_str_dataset(m, "coord_system").unwrap_or_else(|_| "unknown".into()),
    })
}

fn read_str_dataset(tree: &TreeBuf, name: &str) -> Result<String> {
    let ds = tree
        .find_dataset(name)
        .ok_or_else(|| IoError::MissingPath(name.into()))?;
    match &ds.data {
        symbi_io::DataBuf::U8(b) => Ok(String::from_utf8_lossy(b).into_owned()),
        other => Err(IoError::TypeMismatch {
            path: name.into(),
            expected: "u8 (string)",
            actual: match other {
                symbi_io::DataBuf::F64(_) => "f64",
                _ => "?",
            },
        }),
    }
}

pub fn read_checkpoint_meta(path: &str) -> Result<CheckpointMeta> {
    let tree = Hdf5Backend.read(Path::new(path))?;
    read_meta_from(&tree)
}

/// load a checkpoint into an existing SimState. restores cons (and
/// prim if present, and bface if present) from disk; returns the typed
/// `CheckpointMeta` for the caller to consume time/iteration/etc.
pub fn load_checkpoint<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &mut SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    path: &str,
) -> Result<CheckpointMeta>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    load_checkpoint_level(sim, path, 0)
}

/// how many refinement levels a checkpoint carries.
///
/// a restart may run deeper than the file it resumes from — that is the whole point of a
/// bootstrap ladder, where each rung converges at its own resolution and the next adds one. the
/// levels the file has are loaded; the rest are initialized from their parents. counting them is
/// what tells the two apart, and the count comes from the file rather than from the config so a
/// hand-edited or truncated checkpoint cannot make a level silently start from zeros.
pub fn checkpoint_level_count(path: &str) -> Result<usize> {
    let tree = Hdf5Backend.read(Path::new(path))?;
    let mut n = 0usize;
    while tree.find_group(&format!("level_{n}")).is_some() {
        n += 1;
    }
    if n == 0 {
        return Err(IoError::MissingPath("level_0".into()));
    }
    Ok(n)
}

/// check that a checkpoint's level `level_index` describes the same grid this run built for it.
///
/// a deeper restart only works because level `i` occupies the same region at every depth — true
/// when a config's refinement regions are fixed geometry, false for any schedule that derives them
/// from the level count. in the second case the loaded data would be laid over a different region
/// and produce a field that is smooth, finite and wrong everywhere, with the error appearing as an
/// unexplained profile rather than as a failure.
///
/// so the property is verified rather than assumed: cell counts and physical bounds per axis, from
/// the file's own mesh description.
pub fn verify_checkpoint_level_geometry<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    path: &str,
    level_index: usize,
) -> Result<()>
where
    R: Regime<f64, D>,
    M: symbi_geometry::Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: symbi_xpu::ExecutionSpace,
    Mem: symbi_xpu::MemorySpace,
{
    let tree = Hdf5Backend.read(Path::new(path))?;
    let name = format!("level_{level_index}");
    let mesh = tree
        .find_group(&name)
        .and_then(|level| level.find_group("mesh"))
        .ok_or_else(|| IoError::MissingPath(format!("{name}/mesh")))?;

    let cells = match mesh.find_dataset("global_cells").map(|d| &d.data) {
        Some(symbi_io::DataBuf::U64(v)) => v.clone(),
        _ => return Err(IoError::MissingPath(format!("{name}/mesh/global_cells"))),
    };
    for ax in 0..D {
        let want = sim.geom.interior.spaces[ax].size();
        // `mesh/global_cells` is written in reversed (storage) axis order, matching the reversed
        // dataset shapes so a reader's plot axes are not transposed — see `mesh_cells`. reading it
        // forward compares axis 0 against the last axis's count, which agrees only on a cubic grid
        // and rejects every anisotropic one.
        let got = *cells.get(D - 1 - ax).unwrap_or(&0) as usize;
        if got != want {
            return Err(IoError::Backend(format!(
                "{path}: level {level_index} was written with {got} cell(s) on axis {ax} but this \
                 run builds {want}. a deeper restart requires level {level_index} to occupy the \
                 same grid it did in the checkpoint; a refinement schedule whose regions depend on \
                 the level count does not satisfy that, and loading across the mismatch would \
                 place the data on the wrong region."
            )));
        }
    }

    let geometry = mesh
        .find_group("geometry")
        .ok_or_else(|| IoError::MissingPath(format!("{name}/mesh/geometry")))?;
    // the writer stores physical bounds — the comoving faces scaled by the mesh-motion
    // factor a(t) at write time on the expanding axes — while this sim is freshly built
    // on the comoving grid (a = 1; the motion re-derives a(t) from the resume time, it
    // is never integrated state). unscale the stored bounds by the checkpoint's own
    // scale factor so the comparison is comoving against comoving; a checkpoint from a
    // static-mesh run carries a = 1 and is unchanged.
    let a_checkpoint = tree
        .find_group("metadata")
        .and_then(|meta| meta.find_attr("scale_factor"))
        .and_then(|attr| match attr {
            Attr::F64(v) => Some(*v),
            _ => None,
        })
        .unwrap_or(1.0);
    for ax in 0..D {
        // the geometry groups are named by storage slot, `(0..D).rev().enumerate()` in the writer,
        // so slot `D - 1 - ax` holds axis `ax`. the same reversal as `global_cells`, and equally
        // invisible on a cubic grid or in one dimension.
        let slot = D - 1 - ax;
        let dim = geometry
            .find_group(&format!("dim_{slot}"))
            .ok_or_else(|| IoError::MissingPath(format!("{name}/mesh/geometry/dim_{slot}")))?;
        let bound = |key: &str| match dim.find_attr(key) {
            Some(Attr::F64(v)) => Some(*v),
            _ => None,
        };
        let (Some(start), Some(end)) = (bound("start"), bound("end")) else {
            return Err(IoError::MissingPath(format!(
                "{name}/mesh/geometry/dim_{slot}/start|end"
            )));
        };
        let unscale = motion_axis_scale(sim.physics.metric.geometry(), ax, D, a_checkpoint);
        let start = start / unscale;
        let end = end / unscale;
        // the run's comoving bounds, by the writer's own face arithmetic (the coordinate
        // maps when present, the uniform formula otherwise) so the two sides can only
        // differ by a genuine region mismatch, never by formula drift.
        let lo_index = sim.geom.interior.spaces[ax].lo;
        let hi_index = sim.geom.interior.spaces[ax].hi;
        let (lo, hi) = match &sim.geom.maps {
            Some(maps) => (maps[ax].face(lo_index), maps[ax].face(hi_index)),
            None => {
                let lo = sim.geom.x_lo[ax] + lo_index as f64 * sim.geom.dx[ax];
                (
                    lo,
                    lo + sim.geom.dx[ax] * sim.geom.interior.spaces[ax].size() as f64,
                )
            }
        };
        // relative to the level's own extent: an absolute tolerance would be meaningless across a
        // ladder whose finest level is orders of magnitude smaller than its root.
        let scale = (hi - lo).abs().max(1.0e-300);
        if (start - lo).abs() / scale > 1.0e-9 || (end - hi).abs() / scale > 1.0e-9 {
            return Err(IoError::Backend(format!(
                "{path}: level {level_index} axis {ax} spans [{start:e}, {end:e}] in the checkpoint \
                 (comoving, unscaled by its a = {a_checkpoint}) but [{lo:e}, {hi:e}] in this run. \
                 level {level_index} must occupy the same comoving region at every depth for a \
                 deeper restart to be meaningful."
            )));
        }
    }
    Ok(())
}

pub fn load_checkpoint_level<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &mut SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    path: &str,
    level_index: usize,
) -> Result<CheckpointMeta>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    let tree = Hdf5Backend.read(Path::new(path))?;
    // the conserved group is the restart primary, so its measure must match what this run
    // evolves. a densitized GR-hydro run reading an undensitized file (or the reverse) differs by
    // a per-cell sqrt(-g) and would restart onto a physically different state without any symptom
    // the first step could not explain away.
    let stored = match tree.find_attr(CONSERVED_DENSITIZATION_ATTR) {
        Some(Attr::Str(s)) => Some(s.as_str()),
        _ => None,
    };
    let expected = conserved_densitization(&sim.store);
    if stored != expected {
        return Err(IoError::Backend(format!(
            "{path}: the checkpoint stores a {} conserved state but this run evolves a {} one",
            stored.unwrap_or("undensitized"),
            expected.unwrap_or("undensitized"),
        )));
    }
    let meta = read_meta_from(&tree)?;
    sim.time = meta.time;
    sim.dt = meta.dt;
    sim.iteration = meta.iteration;

    let level_name = format!("level_{level_index}");
    let level_0 = tree
        .find_group(&level_name)
        .ok_or_else(|| IoError::MissingPath(level_name.clone()))?;
    // the level's own clock, when the file carries one; older files hold the root's clock alone.
    if let Some(a) = level_0.find_attr("time") {
        sim.time = a.as_f64(&format!("{level_name}/time"))?;
    }
    if let Some(a) = level_0.find_attr("dt") {
        sim.dt = a.as_f64(&format!("{level_name}/dt"))?;
    }
    if let Some(a) = level_0.find_attr("iteration") {
        sim.iteration = a.as_u64(&format!("{level_name}/iteration"))?;
    }
    // the isothermal closure's cs^2(x), when the file carries it; older files leave the
    // constructor's uniform value.
    if let Some(cs2) = sim.fields.cs2.as_ref() {
        if level_0.find_dataset("iso_cs2").is_some() {
            restore_field(level_0, "iso_cs2", cs2, cs2.domain())?;
        }
    }
    let interior = sim.geom.interior.clone();

    // conserved (primary — c2p will derive prims on restart). RegimeSpec-driven.
    let cons = level_0
        .find_group("conserved")
        .ok_or_else(|| IoError::MissingPath("level_0/conserved".into()))?;
    for fs in R::SPEC.fields {
        let n = symbi_io::component_count(fs, DOF);
        for idx in 0..n {
            let name = symbi_io::dataset_name(fs, idx);
            match fs.name {
                "den" => restore_field(
                    cons,
                    &name,
                    &sim.fields.cons.den,
                    sim.fields.cons.den.domain(),
                )?,
                "mom" => restore_field(
                    cons,
                    &name,
                    &sim.fields.cons.mom[idx],
                    sim.fields.cons.mom[idx].domain(),
                )?,
                "nrg" => {
                    if let Some(nrg) = sim.fields.cons.nrg_field() {
                        restore_field(cons, &name, nrg, nrg.domain())?;
                    }
                }
                "mag" => {
                    if let Some(ref mhd) = sim.fields.mhd {
                        restore_field(cons, &name, &mhd.bcell[idx], mhd.bcell[idx].domain())?;
                    }
                }
                other => panic!("checkpoint read: unknown conserved field '{other}'"),
            }
        }
    }
    // the passive scalar rides outside the spec iteration (run-level opt-in):
    // restored iff this run allocated it and the file carries it — a dyed
    // restart of an undyed file starts from chi = 0 rather than failing.
    if let Some(chi) = sim.fields.cons.chi_field() {
        if cons.find_dataset("chi").is_some() {
            restore_field(cons, "chi", chi, chi.domain())?;
        }
    }

    // primitives (optional). the canonical v2 tree nests visualization fields
    // under partition_0/hydro; accept the former flat location for old files.
    let hydro = level_0
        .find_group("partition_0")
        .and_then(|partition| partition.find_group("hydro"));
    let primitives = hydro
        .and_then(|group| group.find_group("primitives"))
        .or_else(|| level_0.find_group("primitives"));
    if let Some(prim) = primitives {
        for fs in R::SPEC.primitive_fields {
            let n = symbi_io::component_count(fs, DOF);
            for idx in 0..n {
                let name = symbi_io::dataset_name(fs, idx);
                match fs.name {
                    "rho" => restore_field(
                        prim,
                        &name,
                        &sim.fields.prim.rho,
                        sim.fields.prim.rho.domain(),
                    )?,
                    "vel" => restore_field(
                        prim,
                        &name,
                        &sim.fields.prim.vel[idx],
                        sim.fields.prim.vel[idx].domain(),
                    )?,
                    "pre" => {
                        if let Some(pre) = sim.fields.prim.pre_field() {
                            restore_field(prim, &name, pre, pre.domain())?;
                        }
                    }
                    "bcell" => {
                        if let Some(ref mhd) = sim.fields.mhd {
                            if prim.find_dataset(&name).is_some() {
                                restore_field(
                                    prim,
                                    &name,
                                    &mhd.bcell[idx],
                                    mhd.bcell[idx].domain(),
                                )?;
                            }
                        }
                    }
                    other => panic!("checkpoint read: unknown primitive field '{other}'"),
                }
            }
        }
        // the passive-scalar concentration, outside the spec iteration like its
        // conserved counterpart.
        if let Some(chi) = sim.fields.prim.chi_field() {
            if prim.find_dataset("chi").is_some() {
                restore_field(prim, "chi", chi, chi.domain())?;
            }
        }
    }

    // face-centered B (CT truth — restores div(B)=0 exactly)
    if let Some(ref mhd) = sim.fields.mhd {
        let magnetic = hydro
            .and_then(|group| group.find_group("magnetic"))
            .or_else(|| level_0.find_group("magnetic"));
        if let Some(mag) = magnetic {
            let mut all_ok = true;
            for d in 0..D {
                let face_dom = interior.extend(d, 0, 1);
                let name = format!("B{}", d + 1);
                let restored = mag.find_group(&name).map_or_else(
                    || restore_field(mag, &name, &mhd.bface[d], &face_dom),
                    |face| restore_field(face, "data", &mhd.bface[d], &face_dom),
                );
                if restored.is_err() {
                    all_ok = false;
                }
            }
            if all_ok {
                mhd.bface_initialized
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    // restore per-body kinematic + accretion state over the config-attached
    // collection: without this, a restart resets a moving body's orbit phase
    // and a sink's cumulative accreted mass (the diagnostics.dat ledger would
    // step down at the seam). checkpoints written before the group exists
    // restore fields only — bodies keep their config values.
    // the tracer population: restored whenever the file carries the group —
    // the run continues the previous population (flags, ids, weight intact)
    // regardless of what the fresh-run config would have seeded.
    if let Some(tg) = tree.find_group("tracers") {
        let getf = |name: &str| -> Result<Vec<f64>> {
            Ok(tg
                .find_dataset(name)
                .ok_or_else(|| IoError::MissingPath(format!("tracers/{name}")))?
                .data
                .as_f64()
                .ok_or_else(|| IoError::MissingPath(format!("tracers/{name}: not f64")))?
                .to_vec())
        };
        let getu = |name: &str| -> Result<Vec<u64>> {
            Ok(tg
                .find_dataset(name)
                .ok_or_else(|| IoError::MissingPath(format!("tracers/{name}")))?
                .data
                .as_u64()
                .ok_or_else(|| IoError::MissingPath(format!("tracers/{name}: not u64")))?
                .to_vec())
        };
        let xs = getf("position")?;
        let ids = getu("id")?;
        let cohorts = getu("cohort")?;
        let owners = getu("owner")?;
        let (esc, crx, ct) = (
            getf("escaped")?,
            getf("crossed_sink")?,
            getf("crossing_time")?,
        );
        let weight = getf("weight")?.first().copied().unwrap_or(0.0);
        let n = ids.len();
        let run_seed = tg
            .find_attr("run_seed")
            .ok_or_else(|| IoError::MissingPath("tracers/run_seed".to_string()))?
            .as_u64("tracers/run_seed")?;
        let next_id = tg
            .find_attr("next_id")
            .ok_or_else(|| IoError::MissingPath("tracers/next_id".to_string()))?
            .as_u64("tracers/next_id")?;
        let injection_remainder = tg
            .find_attr("injection_remainder")
            .ok_or_else(|| IoError::MissingPath("tracers/injection_remainder".to_string()))?
            .as_f64("tracers/injection_remainder")?;
        let mut tr = symbi_sim_tracers::TracerSet::<D> {
            weight,
            run_seed,
            next_id,
            injection_remainder,
            ..Default::default()
        };
        for i in 0..n {
            let mut p = [0.0; D];
            for a in 0..D {
                p[a] = xs[i * D + a];
            }
            tr.x.push(p);
            tr.id.push(ids[i]);
            tr.cohort.push(
                u16::try_from(cohorts[i])
                    .map_err(|_| IoError::Backend("tracer cohort exceeds u16".to_string()))?,
            );
            tr.owner.push(crate::mass_transport::ContainerId(owners[i]));
            tr.flags.push(symbi_sim_tracers::TracerFlags {
                escaped: esc[i] != 0.0,
                crossed_sink: crx[i] != 0.0,
                crossing_time: ct[i],
            });
        }
        tr.step_owner = tr.owner.clone();
        tr.step_flags = tr.flags.clone();
        sim.tracers = Some(tr);
        let geometry = sim.geom.block_geometry(sim.physics.metric);
        let layout = symbi_sim_tracers::TransportLayout::single(&sim.geom.interior);
        symbi_sim_tracers::refresh_derived_positions_store(&mut sim.store, &geometry, layout);
    }
    if let Some(group) = tree.find_group("continuous_tracers") {
        let getf = |name: &str| -> Result<Vec<f64>> {
            Ok(group
                .find_dataset(name)
                .ok_or_else(|| IoError::MissingPath(format!("continuous_tracers/{name}")))?
                .data
                .as_f64()
                .ok_or_else(|| IoError::MissingPath(format!("continuous_tracers/{name}: not f64")))?
                .to_vec())
        };
        let getu = |name: &str| -> Result<Vec<u64>> {
            Ok(group
                .find_dataset(name)
                .ok_or_else(|| IoError::MissingPath(format!("continuous_tracers/{name}")))?
                .data
                .as_u64()
                .ok_or_else(|| IoError::MissingPath(format!("continuous_tracers/{name}: not u64")))?
                .to_vec())
        };
        let x = getf("position")?;
        let step_x = getf("step_position")?;
        let id = getu("id")?;
        let cohort = getu("cohort")?;
        let owner = getu("owner")?;
        let escaped = getu("escaped")?;
        let crossed_sink = getu("crossed_sink")?;
        let crossing_time = getf("crossing_time")?;
        let random_counter = getu("random_counter")?;
        let weight = getf("weight")?.first().copied().unwrap_or(0.0);
        let n = id.len();
        if [
            x.len() / D,
            step_x.len() / D,
            cohort.len(),
            owner.len(),
            escaped.len(),
            crossed_sink.len(),
            crossing_time.len(),
            random_counter.len(),
        ]
        .into_iter()
        .any(|length| length != n)
        {
            return Err(IoError::Backend(
                "continuous tracer checkpoint arrays have inconsistent lengths".to_string(),
            ));
        }
        let order = match group
            .find_attr("order")
            .ok_or_else(|| IoError::MissingPath("continuous_tracers/order".to_string()))?
            .as_u64("continuous_tracers/order")?
        {
            2 => crate::mass_transport::ItoOrder::Two,
            3 => crate::mass_transport::ItoOrder::Three,
            value => {
                return Err(IoError::Backend(format!(
                    "unsupported continuous tracer order {value}"
                )));
            }
        };
        let mut tracers = symbi_sim_tracers::ContinuousTracerSet::<D, Mem>::allocate(n, order)
            .map_err(IoError::Backend)?;
        tracers.weight = weight;
        tracers.run_seed = group
            .find_attr("run_seed")
            .ok_or_else(|| IoError::MissingPath("continuous_tracers/run_seed".to_string()))?
            .as_u64("continuous_tracers/run_seed")?;
        tracers.next_id = group
            .find_attr("next_id")
            .ok_or_else(|| IoError::MissingPath("continuous_tracers/next_id".to_string()))?
            .as_u64("continuous_tracers/next_id")?;
        tracers.injection_remainder = group
            .find_attr("injection_remainder")
            .ok_or_else(|| {
                IoError::MissingPath("continuous_tracers/injection_remainder".to_string())
            })?
            .as_f64("continuous_tracers/injection_remainder")?;
        for ii in 0..n {
            tracers
                .push_host(symbi_sim_tracers::ContinuousTracerRecord {
                    x: std::array::from_fn(|dd| x[ii * D + dd]),
                    step_x: std::array::from_fn(|dd| step_x[ii * D + dd]),
                    id: id[ii],
                    cohort: u16::try_from(cohort[ii]).map_err(|_| {
                        IoError::Backend("continuous tracer cohort exceeds u16".to_string())
                    })?,
                    owner: crate::mass_transport::ContainerId(owner[ii]),
                    escaped: u8::try_from(escaped[ii]).map_err(|_| {
                        IoError::Backend("continuous tracer escaped flag exceeds u8".to_string())
                    })?,
                    crossed_sink: u8::try_from(crossed_sink[ii]).map_err(|_| {
                        IoError::Backend(
                            "continuous tracer crossed-sink flag exceeds u8".to_string(),
                        )
                    })?,
                    crossing_time: crossing_time[ii],
                    random_counter: random_counter[ii],
                })
                .map_err(IoError::Backend)?;
        }
        sim.continuous_tracers = Some(tracers);
    }

    if let (Some(bodies_g), Some(im)) = (tree.find_group("bodies"), sim.immersed.as_mut()) {
        restore_bodies::<D>(bodies_g, im)?;
    }

    Ok(meta)
}

/// restore the per-body kinematic and accretion state recorded under `bodies` over the
/// config-attached collection `im`: without it a restart resets a moving body's orbit phase and
/// a sink's cumulative accreted mass. files written before a dataset existed keep the config
/// value for that quantity.
fn restore_bodies<const D: usize>(bodies_g: &TreeBuf, im: &mut ImmersedBodies<D>) -> Result<()> {
    let get = |name: &str| -> Result<Vec<f64>> {
        Ok(bodies_g
            .find_dataset(name)
            .ok_or_else(|| IoError::MissingPath(format!("bodies/{name}")))?
            .data
            .as_f64()
            .ok_or_else(|| IoError::MissingPath(format!("bodies/{name}: not f64")))?
            .to_vec())
    };
    let (pos, vel) = (get("position")?, get("velocity")?);
    let (mass, accreted, rate) = (
        get("mass")?,
        get("total_accreted_mass")?,
        get("accretion_rate")?,
    );
    let nb = im.bodies.len().min(mass.len());
    // the slip heat under either closure's name; files written before the receipt existed
    // leave the counters at zero.
    let optional = |name: &str| -> Option<Vec<f64>> {
        bodies_g.find_dataset(name).and_then(|d| d.data.as_f64().map(|v| v.to_vec()))
    };
    let heat = optional("magnetic_slip_heating").or_else(|| optional("exported_slip_heat"));
    let heat_rate = optional("magnetic_slip_heating_rate").or_else(|| optional("exported_slip_heat_rate"));
    for b in 0..nb {
        let body = im.bodies.get_mut(b);
        for a in 0..D {
            body.position[a] = pos[b * D + a];
            body.velocity[a] = vel[b * D + a];
        }
        body.mass = mass[b];
        // the slip heat belongs to the body that owns the slip; a gravity-only proxy of it on
        // a coarser level carries the coupling stripped and books nothing.
        if matches!(body.spec.magnetic, symbi_ib::MagneticSpec::Slip { .. }) {
            if let Some(h) = heat.as_ref() {
                body.slip_heat_total = h[b];
            }
            if let Some(h) = heat_rate.as_ref() {
                body.slip_heat_rate = h[b];
            }
        }
        if let symbi_ib::BodyKind::BlackHole {
            total_accreted_mass,
            accretion_rate,
            ..
        } = &mut body.kind
        {
            *total_accreted_mass = accreted[b];
            *accretion_rate = rate[b];
        }
    }
    // restore the evolved rigid-body rotation (orientation matrix + angular velocity) so a
    // spinning / tumbling body resumes its exact pose. checkpoints written before these datasets
    // existed keep the config values (identity orientation, prescribed omega).
    let get_opt = |name: &str| -> Option<Vec<f64>> {
        bodies_g
            .find_dataset(name)?
            .data
            .as_f64()
            .map(|d| d.to_vec())
    };
    if let (Some(orient), Some(omega)) = (get_opt("orientation"), get_opt("omega")) {
        for b in 0..nb {
            let body = im.bodies.get_mut(b);
            for i in 0..3 {
                for j in 0..3 {
                    body.orientation[i][j] = orient[b * 9 + i * 3 + j];
                }
            }
            for k in 0..3 {
                body.omega[k] = omega[b * 3 + k];
            }
        }
    }
    Ok(())
}

fn restore_field<const D: usize, Mem: MemorySpace>(
    tree: &TreeBuf,
    name: &str,
    field: &symbi_grid::Field<f64, D, Mem>,
    domain: &symbi_algebra::Domain<D>,
) -> Result<()> {
    let ds = tree
        .find_dataset(name)
        .ok_or_else(|| IoError::MissingPath(name.into()))?;
    let data = ds.data.as_f64().ok_or_else(|| IoError::TypeMismatch {
        path: name.into(),
        expected: "f64",
        actual: "non-f64",
    })?;
    let vol = domain.volume();
    if data.len() != vol {
        return Err(IoError::ShapeMismatch {
            path: name.into(),
            expected: vec![vol],
            actual: vec![data.len()],
        });
    }
    let view = field.view_mut();
    // same axis-0-fastest walk as `extract_field` — written-then-loaded is the identity by
    // construction. a `(0..D).rev()` walk would transpose every D>=2 restart.
    let mut ii = 0usize;
    for_each_cell_axis0(domain, |coord| {
        view.set(coord, data[ii]);
        ii += 1;
    });
    Ok(())
}

// =============================================================================
// tests
//
// the checkpoint mesh-coordinate gate for a nonzero interior origin lives in-crate
// because it exercises `SimStateGeneric::new_at` — the absolute-index amr-internal
// constructor (pub(crate); the public path is `SimBuilder`, which always grids at
// interior_lo = [0; D]). amr fine levels live at absolute indices, so their written
// mesh/x{1,2,3} centers must equal geom.centroid of the actual interior cells.
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use symbi_geometry::Cartesian;
    use symbi_hydro::eos::IdealGas;
    use symbi_hydro::newtonian::Newtonian;
    use symbi_xpu::{CpuSpace, HostMemory};

    type Sim = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

    // the regime slug a checkpoint records is the configuration vocabulary, so a restart parses
    // it back into the enum the run was configured from.
    #[test]
    fn the_checkpoint_regime_slug_is_the_configuration_name() {
        assert_eq!(regime_name::<Newtonian, 3>(&Newtonian), "newtonian");
        assert_eq!(
            regime_name::<symbi_hydro::newtonian_mhd::NewtonianMhd, 3>(
                &symbi_hydro::newtonian_mhd::NewtonianMhd
            ),
            "nmhd"
        );
    }

    #[test]
    fn mesh_centers_respect_the_interior_origin() {
        // a fine-level-like state: interior [8, 12)^3 on a global origin at -1.
        let interior_lo = [8isize; 3];
        let sim = Sim::new_at(
            Newtonian,
            IdealGas { gamma: 5.0 / 3.0 },
            Cartesian,
            interior_lo,
            [4; 3],
            [-1.0; 3],
            [0.125; 3],
            2,
            Boundaries::uniform(BoundaryType::Outflow),
            0.4,
            Timestepping::Rk2,
            0,
        )
        .unwrap();

        let dir = std::env::temp_dir().join("symbi_checkpoint_offset_origin");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("offset.h5");
        let path = path.to_str().unwrap();
        write_checkpoint(&sim, path, &Metadata::new()).unwrap();

        // the frozen v2.0 reader rebuilds cell centers from the geometry
        // description (global_cells + per-dim start/end), so verify that honors
        // the interior origin — reconstruct centers exactly as the reader does
        // (center_i = start + (i + 0.5) * (end - start) / n) and match centroid.
        let tree = Hdf5Backend.read(std::path::Path::new(path)).unwrap();
        let geometry = tree
            .find_group("level_0")
            .unwrap()
            .find_group("mesh")
            .unwrap()
            .find_group("geometry")
            .unwrap();
        for ax in 0..3 {
            let dim = geometry.find_group(&format!("dim_{ax}")).unwrap();
            let start = dim.find_attr("start").unwrap().as_f64("start").unwrap();
            let end = dim.find_attr("end").unwrap().as_f64("end").unwrap();
            let n = 4usize;
            let dx = (end - start) / n as f64;
            for ii in 0..n {
                let x = start + (ii as f64 + 0.5) * dx;
                let mut coord = [0isize; 3];
                coord[ax] = interior_lo[ax] + ii as isize;
                let expect = sim.geom.centroid(coord)[ax];
                assert!(
                    (x - expect).abs() < 1e-14,
                    "dim_{ax}[{ii}] reconstructs to {x} but the interior cell center is {expect}"
                );
            }
        }
    }

    #[test]
    fn checkpoint_records_geometric_spacing_parameters() {
        let ratio = 0.9_f64;
        let cells = 8_usize;
        let width = (ratio - 1.0) / (ratio.powf(cells as f64) - 1.0);
        let maps = [
            symbi_geometry::AxisMap::Geometric {
                start: 0.0,
                width,
                ratio,
            },
            symbi_geometry::AxisMap::Uniform {
                start: 0.0,
                dx: 1.0,
            },
            symbi_geometry::AxisMap::Uniform {
                start: 0.0,
                dx: 1.0,
            },
        ];
        let mut sim = Sim::new(
            Newtonian,
            IdealGas { gamma: 5.0 / 3.0 },
            Cartesian,
            [cells, 1, 1],
            [0.0; 3],
            [1.0 / cells as f64, 1.0, 1.0],
            2,
            Boundaries::uniform(BoundaryType::Outflow),
            0.4,
            Timestepping::Rk2,
            0,
        )
        .unwrap();
        sim.geom.set_maps(maps);

        let dir = std::env::temp_dir().join("symbi_checkpoint_geometric_spacing");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("geometric.h5");
        write_checkpoint(&sim, path.to_str().unwrap(), &Metadata::new()).unwrap();

        let tree = Hdf5Backend.read(&path).unwrap();
        let dim = tree
            .find_group("level_0")
            .unwrap()
            .find_group("mesh")
            .unwrap()
            .find_group("geometry")
            .unwrap()
            .find_group("dim_2")
            .unwrap();
        assert_eq!(
            dim.find_attr("type").unwrap().as_str("type").unwrap(),
            "geometric"
        );
        assert!((dim.find_attr("ratio").unwrap().as_f64("ratio").unwrap() - ratio).abs() < 1.0e-14);
        assert!((dim.find_attr("start").unwrap().as_f64("start").unwrap()).abs() < 1.0e-14);
        assert!((dim.find_attr("end").unwrap().as_f64("end").unwrap() - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn checkpoint_roundtrip_preserves_field_layout_2d() {
        // **restart round-trip gate**: write a checkpoint then load it; the conserved state must
        // come back identical. a non-square grid (5x3) seeded with an asymmetric pattern
        // (value = i + 100*j, distinct per field) makes any axis transpose between the gather
        // (`extract_field`) and scatter (`restore_field`) a loud failure — the bug that shipped
        // when the two walks used opposite axis orders (`0..D` vs `(0..D).rev()`).
        type Sim2 = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
        let build = || {
            Sim2::new_at(
                Newtonian,
                IdealGas { gamma: 5.0 / 3.0 },
                Cartesian,
                [0isize, 0],
                [5usize, 3],
                [0.0, 0.0],
                [0.2, 1.0 / 3.0],
                2,
                Boundaries::uniform(BoundaryType::Outflow),
                0.4,
                Timestepping::Rk2,
                0,
            )
            .unwrap()
        };

        let sim = build();
        let nrg = sim
            .fields
            .cons
            .nrg_field()
            .expect("Newtonian cons.nrg")
            .clone();
        for c in sim.geom.interior.iter() {
            let (i, j) = (c[0] as f64, c[1] as f64);
            sim.fields.cons.den.view_mut().set(c, 1.0 + i + 100.0 * j);
            sim.fields.cons.mom[0]
                .view_mut()
                .set(c, 10.0 + i + 100.0 * j);
            sim.fields.cons.mom[1]
                .view_mut()
                .set(c, 20.0 + i + 100.0 * j);
            nrg.view_mut().set(c, 30.0 + i + 100.0 * j);
        }

        let dir = std::env::temp_dir().join("symbi_checkpoint_roundtrip_2d");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rt.h5");
        let path = path.to_str().unwrap();
        write_checkpoint(&sim, path, &Metadata::new()).unwrap();

        let mut loaded = build();
        load_checkpoint(&mut loaded, path).unwrap();

        let lnrg = loaded.fields.cons.nrg_field().unwrap();
        for c in sim.geom.interior.iter() {
            assert_eq!(
                *loaded.fields.cons.den.view().at(c),
                *sim.fields.cons.den.view().at(c),
                "cons.den transposed/garbled at {c:?}"
            );
            for k in 0..2 {
                assert_eq!(
                    *loaded.fields.cons.mom[k].view().at(c),
                    *sim.fields.cons.mom[k].view().at(c),
                    "cons.mom_{k} transposed/garbled at {c:?}"
                );
            }
            assert_eq!(
                *lnrg.view().at(c),
                *nrg.view().at(c),
                "cons.nrg transposed/garbled at {c:?}"
            );
        }
    }

    #[test]
    fn hierarchy_restart_restores_each_level_from_its_own_group() {
        type Sim1 = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
        let build = || {
            Sim1::new_at(
                Newtonian,
                IdealGas { gamma: 5.0 / 3.0 },
                Cartesian,
                [0isize],
                [4usize],
                [0.0],
                [0.25],
                2,
                Boundaries::uniform(BoundaryType::Outflow),
                0.4,
                Timestepping::Rk2,
                0,
            )
            .unwrap()
        };
        let coarse = build();
        let fine = build();
        for coord in coarse.geom.interior.iter() {
            coarse.fields.cons.den.view_mut().set(coord, 2.0);
            fine.fields.cons.den.view_mut().set(coord, 7.0);
        }
        let dir = std::env::temp_dir().join("symbi_hierarchy_restart_levels");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hierarchy.h5");
        let path = path.to_str().unwrap();
        write_hierarchy_checkpoint(&[&coarse, &fine], path, &Metadata::new()).unwrap();

        let mut loaded_coarse = build();
        let mut loaded_fine = build();
        load_checkpoint_level(&mut loaded_coarse, path, 0).unwrap();
        load_checkpoint_level(&mut loaded_fine, path, 1).unwrap();
        for coord in loaded_coarse.geom.interior.iter() {
            assert_eq!(*loaded_coarse.fields.cons.den.view().at(coord), 2.0);
            assert_eq!(*loaded_fine.fields.cons.den.view().at(coord), 7.0);
        }
    }

    #[test]
    fn checkpoint_roundtrip_preserves_mass_transport_tracers() {
        let build = || {
            Sim::new(
                Newtonian,
                IdealGas { gamma: 5.0 / 3.0 },
                Cartesian,
                [2, 1, 1],
                [0.0; 3],
                [0.5, 1.0, 1.0],
                2,
                Boundaries::uniform(BoundaryType::Outflow),
                0.4,
                Timestepping::Rk2,
                0,
            )
            .unwrap()
        };
        let mut sim = build();
        sim.tracers = Some(symbi_sim_tracers::TracerSet {
            x: vec![[91.0, 92.0, 93.0], [94.0, 95.0, 96.0]],
            id: vec![u64::MAX - 1, u64::MAX],
            cohort: vec![7, 9],
            flags: vec![Default::default(); 2],
            weight: 3.5,
            owner: vec![
                crate::mass_transport::ContainerId(0),
                crate::mass_transport::ContainerId(1),
            ],
            step_owner: vec![
                crate::mass_transport::ContainerId(0),
                crate::mass_transport::ContainerId(1),
            ],
            step_flags: vec![Default::default(); 2],
            run_seed: u64::MAX - 7,
            next_id: u64::MAX - 2,
            injection_remainder: 0.25,
        });

        let dir = std::env::temp_dir().join("symbi_checkpoint_mass_transport_tracers");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tracers.h5");
        write_checkpoint(&sim, path.to_str().unwrap(), &Metadata::new()).unwrap();

        let mut restored = build();
        load_checkpoint(&mut restored, path.to_str().unwrap()).unwrap();
        let expected = sim.tracers.as_ref().unwrap();
        let actual = restored.tracers.as_ref().unwrap();
        assert_eq!(actual.id, expected.id);
        assert_eq!(actual.cohort, expected.cohort);
        assert_eq!(actual.owner, expected.owner);
        assert_eq!(actual.step_owner, expected.owner);
        assert_eq!(actual.x, vec![[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]]);
        assert_ne!(actual.x, expected.x);
        assert_eq!(actual.run_seed, expected.run_seed);
        assert_eq!(actual.next_id, expected.next_id);
        assert_eq!(
            actual.injection_remainder.to_bits(),
            expected.injection_remainder.to_bits()
        );
        assert_eq!(actual.weight.to_bits(), expected.weight.to_bits());
    }

    #[test]
    fn checkpoint_roundtrip_preserves_continuous_tracer_state() {
        let build = || {
            Sim::new(
                Newtonian,
                IdealGas { gamma: 5.0 / 3.0 },
                Cartesian,
                [2, 1, 1],
                [0.0; 3],
                [0.5, 1.0, 1.0],
                2,
                Boundaries::uniform(BoundaryType::Outflow),
                0.4,
                Timestepping::Rk2,
                0,
            )
            .unwrap()
        };
        let mut sim = build();
        let mut tracers = symbi_sim_tracers::ContinuousTracerSet::<3, HostMemory>::allocate(
            2,
            crate::mass_transport::ItoOrder::Three,
        )
        .unwrap();
        tracers.weight = 3.5;
        tracers.run_seed = u64::MAX - 7;
        tracers.next_id = u64::MAX - 2;
        tracers.injection_remainder = 0.25;
        for record in [
            symbi_sim_tracers::ContinuousTracerRecord {
                x: [1.0, 2.0, 3.0],
                step_x: [0.5, 1.5, 2.5],
                id: u64::MAX - 1,
                cohort: 7,
                owner: crate::mass_transport::ContainerId(4),
                escaped: 0,
                crossed_sink: 1,
                crossing_time: 2.25,
                random_counter: 19,
            },
            symbi_sim_tracers::ContinuousTracerRecord {
                x: [4.0, 5.0, 6.0],
                step_x: [3.5, 4.5, 5.5],
                id: u64::MAX,
                cohort: 9,
                owner: crate::mass_transport::ContainerId(5),
                escaped: 1,
                crossed_sink: 0,
                crossing_time: 3.25,
                random_counter: 23,
            },
        ] {
            tracers.push_host(record).unwrap();
        }
        sim.continuous_tracers = Some(tracers);

        let dir = std::env::temp_dir().join("symbi_checkpoint_continuous_tracers");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tracers.h5");
        write_checkpoint(&sim, path.to_str().unwrap(), &Metadata::new()).unwrap();

        let mut restored = build();
        load_checkpoint(&mut restored, path.to_str().unwrap()).unwrap();
        let expected = continuous_tracer_snap(sim.continuous_tracers.as_ref().unwrap());
        let actual = continuous_tracer_snap(restored.continuous_tracers.as_ref().unwrap());
        assert_eq!(actual.order, expected.order);
        assert_eq!(actual.x, expected.x);
        assert_eq!(actual.step_x, expected.step_x);
        assert_eq!(actual.id, expected.id);
        assert_eq!(actual.cohort, expected.cohort);
        assert_eq!(actual.owner, expected.owner);
        assert_eq!(actual.escaped, expected.escaped);
        assert_eq!(actual.crossed_sink, expected.crossed_sink);
        assert_eq!(actual.crossing_time, expected.crossing_time);
        assert_eq!(actual.random_counter, expected.random_counter);
        assert_eq!(actual.weight, expected.weight);
        assert_eq!(actual.run_seed, expected.run_seed);
        assert_eq!(actual.next_id, expected.next_id);
        assert_eq!(
            actual.injection_remainder.to_bits(),
            expected.injection_remainder.to_bits()
        );
    }

    #[test]
    fn checkpoint_saves_full_allocated_field_including_ghosts() {
        // **truncation gate**: a cell-centered dataset must carry the full allocated extent
        // (interior + 2*ng) — otherwise a restart loses the halo the
        // next stencil reads before the first ghost-fill, and the reader's `halo_width` trim
        // would over-cut interior-only data. seed every allocated cell (ghosts included) with a
        // coord-unique value; assert (a) the on-disk dataset volume is the padded volume, and
        // (b) every ghost cell survives the write -> load round-trip byte-for-byte.
        type Sim2 = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
        let ng = 2usize;
        let build = || {
            Sim2::new_at(
                Newtonian,
                IdealGas { gamma: 5.0 / 3.0 },
                Cartesian,
                [0isize, 0],
                [5usize, 3],
                [0.0, 0.0],
                [0.2, 1.0 / 3.0],
                ng,
                Boundaries::uniform(BoundaryType::Outflow),
                0.4,
                Timestepping::Rk2,
                0,
            )
            .unwrap()
        };

        let sim = build();
        // a value unique per (coord) so a misplaced or dropped ghost is a loud failure.
        let seed = |c: [isize; 2]| 1000.0 + c[0] as f64 + 31.0 * c[1] as f64;
        let alloc = sim.fields.cons.den.domain().clone();
        for c in alloc.iter() {
            sim.fields.cons.den.view_mut().set(c, seed(c));
        }

        let dir = std::env::temp_dir().join("symbi_checkpoint_fullfield");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("full.h5");
        let path = path.to_str().unwrap();
        write_checkpoint(&sim, path, &Metadata::new()).unwrap();

        // (a) the den dataset must hold the padded volume (5+2*ng)x(3+2*ng).
        let tree = Hdf5Backend.read(std::path::Path::new(path)).unwrap();
        let den_ds = tree
            .find_group("level_0")
            .unwrap()
            .find_group("conserved")
            .unwrap()
            .find_dataset("den")
            .unwrap();
        let on_disk = den_ds.data.as_f64().unwrap().len();
        assert_eq!(
            on_disk,
            alloc.volume(),
            "den dataset holds {on_disk} cells but the allocated field has {} (interior would be 15)",
            alloc.volume()
        );

        // (b) every allocated cell — especially the ghosts outside the interior — round-trips.
        let mut loaded = build();
        load_checkpoint(&mut loaded, path).unwrap();
        let interior = sim.geom.interior.clone();
        let mut ghost_checked = 0usize;
        for c in alloc.iter() {
            assert_eq!(
                *loaded.fields.cons.den.view().at(c),
                seed(c),
                "cons.den lost/garbled at {c:?} (ghost={})",
                !interior.contains(c)
            );
            if !interior.contains(c) {
                ghost_checked += 1;
            }
        }
        assert!(
            ghost_checked > 0,
            "test seeded no ghosts — allocation has no halo"
        );
    }

    #[test]
    fn body_state_round_trips_through_a_restart() {
        // the restart contract: a moving sink's orbit phase and cumulative
        // accreted mass survive write -> fresh-config sim -> load. without the
        // bodies group, restart resets both to config values silently.
        type Sim2 = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
        let build = || {
            let mut s = Sim2::new_at(
                Newtonian,
                IdealGas { gamma: 5.0 / 3.0 },
                Cartesian,
                [0isize, 0],
                [4usize, 4],
                [0.0, 0.0],
                [0.25, 0.25],
                2,
                Boundaries::uniform(BoundaryType::Outflow),
                0.4,
                Timestepping::Rk2,
                0,
            )
            .unwrap();
            s.attach_bodies(
                symbi_ib::BodyCollection::new().add(symbi_ib::Body::black_hole(
                    0,
                    symbi_algebra::Tensor::new([0.5, 0.5]),
                    symbi_algebra::Tensor::zeros(),
                    1.0,
                    0.1,
                    0.05,
                    0.5,
                    0.0,
                    0.1,
                )),
            );
            s
        };
        let mut sim = build();
        {
            let body = sim.immersed.as_mut().unwrap().bodies.get_mut(0);
            body.position = symbi_algebra::Tensor::new([0.7, 0.3]); // orbit advanced
            body.velocity = symbi_algebra::Tensor::new([-0.1, 0.2]);
            body.mass = 1.25;
            // an evolved rotation state (a spinning/tumbling body): R_z(90) + a tilted omega.
            body.orientation = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
            body.omega = symbi_algebra::Tensor::new([0.1, -0.2, 0.3]);
            if let symbi_ib::BodyKind::BlackHole {
                total_accreted_mass,
                accretion_rate,
                ..
            } = &mut body.kind
            {
                *total_accreted_mass = 0.042;
                *accretion_rate = 3.5e-3;
            }
        }
        let dir = std::env::temp_dir().join("symbi_checkpoint_bodystate");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bodystate.h5");
        let path = path.to_str().unwrap();
        write_checkpoint(&sim, path, &Metadata::new()).unwrap();

        let mut restored = build(); // fresh: body back at config values
        load_checkpoint(&mut restored, path).unwrap();
        let b = restored.immersed.as_ref().unwrap().bodies.get(0);
        assert_eq!(b.position[0], 0.7);
        assert_eq!(b.position[1], 0.3);
        assert_eq!(b.velocity[0], -0.1);
        assert_eq!(b.mass, 1.25);
        // the evolved rotation survives the restart (else a tumbling body resets to identity/config).
        assert_eq!(
            b.orientation,
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        );
        assert_eq!([b.omega[0], b.omega[1], b.omega[2]], [0.1, -0.2, 0.3]);
        match b.kind {
            symbi_ib::BodyKind::BlackHole {
                total_accreted_mass,
                accretion_rate,
                ..
            } => {
                assert_eq!(total_accreted_mass, 0.042);
                assert_eq!(accretion_rate, 3.5e-3);
            }
            _ => panic!("body kind lost through the round trip"),
        }
    }

    #[test]
    fn body_diagnostics_series_lands_in_the_checkpoint() {
        // the Mdot(t)/F_acc(t) accretion series: pushed per
        // step by evolve_bodies, flushed into every checkpoint as the
        // `body_diagnostics` group. shapes: time/dt [len], mass_delta [len, nb],
        // force [len, nb, D]. this pins the group layout the steady-state
        // detector reads.
        type Sim2 = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
        let mut sim = Sim2::new_at(
            Newtonian,
            IdealGas { gamma: 5.0 / 3.0 },
            Cartesian,
            [0isize, 0],
            [4usize, 4],
            [0.0, 0.0],
            [0.25, 0.25],
            2,
            Boundaries::uniform(BoundaryType::Outflow),
            0.4,
            Timestepping::Rk2,
            0,
        )
        .unwrap();
        sim.attach_bodies(
            symbi_ib::BodyCollection::new().add(symbi_ib::Body::black_hole(
                0,
                symbi_algebra::Tensor::new([0.5, 0.5]),
                symbi_algebra::Tensor::zeros(),
                1.0,
                0.1,
                0.05,
                0.5,
                0.0,
                0.1,
            )),
        );

        // two steps' worth of exchanges, distinct so ordering bugs are loud.
        let im = sim.immersed.as_mut().unwrap();
        let mut d = symbi_ib::BodyDelta::<f64, 2>::new(0);
        d.mass_delta = 0.25;
        d.force_delta = symbi_algebra::Tensor::new([1.0, -2.0]);
        im.history.push(0.1, 0.1, &[d]);
        d.mass_delta = 0.5;
        im.history.push(0.2, 0.1, &[d]);

        let dir = std::env::temp_dir().join("symbi_checkpoint_bodydiag");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bodydiag.h5");
        let path = path.to_str().unwrap();
        write_checkpoint(&sim, path, &Metadata::new()).unwrap();

        let tree = Hdf5Backend.read(std::path::Path::new(path)).unwrap();
        let diag = tree
            .find_group("body_diagnostics")
            .expect("body_diagnostics group missing from the checkpoint");
        let time = diag
            .find_dataset("time")
            .unwrap()
            .data
            .as_f64()
            .unwrap()
            .to_vec();
        let mass = diag
            .find_dataset("mass_delta")
            .unwrap()
            .data
            .as_f64()
            .unwrap()
            .to_vec();
        let force = diag
            .find_dataset("force")
            .unwrap()
            .data
            .as_f64()
            .unwrap()
            .to_vec();
        assert_eq!(time, vec![0.1, 0.2]);
        assert_eq!(mass, vec![0.25, 0.5]);
        assert_eq!(force, vec![1.0, -2.0, 1.0, -2.0]);
    }

    #[test]
    fn isothermal_checkpoints_record_the_sound_speed() {
        // the isothermal eos closes with p = cs^2 rho and stores no pressure
        // dataset, so cs must travel in metadata for readers to reconstruct
        // pressure-dependent fields. energy regimes must not carry the attr:
        // their sound speed varies per cell and a constant would be a lie.
        use symbi_hydro::IsoNewtonian;
        use symbi_hydro::eos::Isothermal;

        type IsoSim = SimState<IsoNewtonian, 2, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
        let iso = IsoSim::new_at(
            IsoNewtonian,
            Isothermal { cs: 0.75 },
            Cartesian,
            [0isize, 0],
            [4usize, 4],
            [0.0, 0.0],
            [0.25, 0.25],
            2,
            Boundaries::uniform(BoundaryType::Outflow),
            0.4,
            Timestepping::Rk2,
            0,
        )
        .unwrap();

        let dir = std::env::temp_dir().join("symbi_checkpoint_iso_cs");
        std::fs::create_dir_all(&dir).unwrap();
        let iso_path = dir.join("iso.h5");
        write_checkpoint(&iso, iso_path.to_str().unwrap(), &Metadata::new()).unwrap();
        let tree = Hdf5Backend.read(&iso_path).unwrap();
        let meta = tree.find_group("metadata").unwrap();
        let cs = meta
            .find_attr("sound_speed")
            .expect("isothermal metadata must carry sound_speed")
            .as_f64("sound_speed")
            .unwrap();
        assert_eq!(cs, 0.75);

        type AdiSim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
        let adi = AdiSim::new_at(
            Newtonian,
            IdealGas { gamma: 5.0 / 3.0 },
            Cartesian,
            [0isize, 0],
            [4usize, 4],
            [0.0, 0.0],
            [0.25, 0.25],
            2,
            Boundaries::uniform(BoundaryType::Outflow),
            0.4,
            Timestepping::Rk2,
            0,
        )
        .unwrap();
        let adi_path = dir.join("adi.h5");
        write_checkpoint(&adi, adi_path.to_str().unwrap(), &Metadata::new()).unwrap();
        let tree = Hdf5Backend.read(&adi_path).unwrap();
        let meta = tree.find_group("metadata").unwrap();
        assert!(
            meta.find_attr("sound_speed").is_none(),
            "energy-regime metadata must not carry a constant sound_speed"
        );
    }
}

/// the recorded census groups of a store, one per registration carrying at least one sample.
///
/// extracted so both checkpoint writers emit them. the uni-grid writer and the hierarchy writer
/// build their trees separately, and a census group written by only one of them leaves every run
/// on the other driver recording nothing — a checkpoint with no census group reads exactly like a
/// run that registered none, so the omission carries no signal at all.
fn census_groups<'a, const D: usize, const DOF: usize, Mem: symbi_xpu::MemorySpace>(
    sim: &'a FieldStore<D, DOF, Mem, f64>,
) -> Vec<Tree<'a>> {
    // the registered binned reductions, one group each. like the body series these cover
    // this run segment only and restart empty on checkpoint load, so a restart chain
    // concatenates offline rather than the run carrying its whole history forward.
    let mut out = Vec::new();
    for registered in &sim.censuses {
        let spec = registered.evaluator.spec();
        let history = &registered.history;
        if history.is_empty() {
            continue;
        }
        let (n, n_seg, n_val) = (history.len(), history.n_segments(), history.n_values());
        let mut group = Tree::new(&format!("census/{}", spec.name()))
            .with_attr("n_segments", n_seg as u64)
            .with_attr("n_values", n_val as u64)
            // the accumulator labels, in the order the `values` axis carries them, so a
            // reader names a column without re-deriving the registration order.
            .with_attr("value_names", spec.value_names().join(","))
            .with_attr("op", reduction_op_tag(spec.op()))
            // the size of the compiled per-cell graph: what a census actually costs, since
            // the cost scales with the dag rather than with the accumulator count.
            .with_attr("node_count", registered.evaluator.node_count() as u64)
            // an accumulating census stores one row folded from many samples rather than a row
            // apiece, so the row alone does not say what it is an accumulation of. the count and
            // the two endpoints make it self-describing: a reader forms the time average by
            // dividing an additive row by the count, and two run segments combine as a
            // count-weighted sum without either having stored its samples.
            .with_attr("accumulated", u64::from(history.accumulate()))
            .with_attr("cadence", spec.cadence().tag().to_string())
            .with_dataset(Dataset::new("time", vec![n], DataRef::F64(history.time())))
            // segment-major within a sample. a reader reshapes the segment axis to the
            // per-axis bin counts in registration order, last axis varying fastest.
            .with_dataset(Dataset::new(
                "values",
                vec![n, n_seg, n_val],
                DataRef::F64(history.values()),
            ))
            // cells that fell outside the binning. a census that silently under-covers its
            // domain is indistinguishable from a physics result, so the shortfall travels
            // with the numbers.
            .with_dataset(Dataset::new(
                "dropped",
                vec![n],
                DataRef::U64(history.dropped()),
            ))
            // which level produced each row, and the span it covers. an accumulating row is folded
            // from many samples, so without the count there is no way to recover a time average
            // from a running sum, and without the level a per-level row is indistinguishable from a
            // composite one. all ones and all zeros respectively in the ordinary case, which costs
            // sixteen bytes a row and makes every row self-describing.
            .with_dataset(Dataset::new(
                "level",
                vec![n],
                DataRef::U64(history.level()),
            ))
            .with_dataset(Dataset::new(
                "n_samples",
                vec![n],
                DataRef::U64(history.n_samples()),
            ))
            .with_dataset(Dataset::new(
                "t_start",
                vec![n],
                DataRef::F64(history.t_start()),
            ));
        // the edges are a property of the registration, not of a sample, so they are
        // written once per axis rather than per row.
        for (k, axis) in spec.axes().iter().enumerate() {
            group = group
                .with_attr(&format!("axis{k}_name"), axis.name().to_string())
                .with_dataset(Dataset::new(
                    &format!("axis{k}_edges"),
                    vec![axis.edges().len()],
                    DataRef::F64(axis.edges()),
                ));
        }
        out.push(group);
    }
    out
}

// =============================================================================
// partitioned output: every tile streams its owned block of each global dataset
// =============================================================================

/// the global grid of one level as the checkpoint records it: the interior cell counts and the
/// global index of the first interior cell per axis, the halo width every cell dataset carries,
/// and the coordinate description the mesh group is rebuilt from. the tiles of a decomposed run
/// partition this grid; a single state is the whole of it.
#[derive(Clone, Debug)]
pub struct GlobalGrid<const D: usize> {
    pub cells: [usize; D],
    pub interior_lo: [isize; D],
    pub ng: usize,
    pub x_lo: [f64; D],
    pub dx: [f64; D],
    pub maps: Option<[symbi_geometry::AxisMap; D]>,
}

impl<const D: usize> GlobalGrid<D> {
    /// the grid of a store that is its whole level.
    pub fn of_state<const DOF: usize, Mem: MemorySpace>(sim: &FieldStore<D, DOF, Mem, f64>) -> Self {
        let interior = &sim.geom.interior;
        Self {
            cells: std::array::from_fn(|ax| interior.spaces[ax].size()),
            interior_lo: std::array::from_fn(|ax| interior.spaces[ax].lo),
            ng: sim.geom.ng,
            x_lo: std::array::from_fn(|ax| sim.geom.x_lo[ax]),
            dx: std::array::from_fn(|ax| sim.geom.dx[ax]),
            maps: sim.geom.maps,
        }
    }
}

/// one tile of a level for output: a read-only view of its fields and the global index of its
/// first interior cell per axis.
pub struct TileView<'a, const D: usize, const DOF: usize, Mem: MemorySpace> {
    pub state: &'a FieldStore<D, DOF, Mem, f64>,
    pub offset: [isize; D],
}

/// the tiles of one level and the grid they partition. tile 0 supplies the level's clock.
pub struct LevelTiles<'a, const D: usize, const DOF: usize, Mem: MemorySpace> {
    pub tiles: Vec<TileView<'a, D, DOF, Mem>>,
    pub grid: GlobalGrid<D>,
}

impl<'a, const D: usize, const DOF: usize, Mem: MemorySpace> LevelTiles<'a, D, DOF, Mem> {
    /// a level held whole by one store.
    pub fn whole(state: &'a FieldStore<D, DOF, Mem, f64>) -> Self {
        let grid = GlobalGrid::of_state(state);
        Self {
            tiles: vec![TileView {
                state,
                offset: grid.interior_lo,
            }],
            grid,
        }
    }
}

/// the host staging budget in cells per slab, from `SYMBI_CHECKPOINT_STAGING_MB` (64 MB when
/// unset): a hard cap on the staging buffer, every tile block being cut into boxes of at most
/// this many cells across as many axes as it takes.
pub fn staging_budget_cells() -> usize {
    let mb: usize = std::env::var("SYMBI_CHECKPOINT_STAGING_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&mb| mb > 0)
        .unwrap_or(64);
    mb * (1 << 20) / std::mem::size_of::<f64>()
}

/// the row-major shape and origin of a local region in the file's index space, which runs
/// storage order (axis D-1 slowest) with the file origin `file_of(local coordinate)`.
fn slab_start_and_count<const D: usize>(
    region: &symbi_algebra::Domain<D>,
    file_of: impl Fn(usize, isize) -> usize,
) -> (Vec<usize>, Vec<usize>) {
    let start = (0..D)
        .rev()
        .map(|ax| file_of(ax, region.spaces[ax].lo))
        .collect();
    let count = (0..D).rev().map(|ax| region.spaces[ax].size()).collect();
    (start, count)
}

/// split `region` into boxes of at most `budget` cells: the slowest axis is cut into row groups,
/// and a row group that still exceeds the budget on its own is cut along the next faster axis,
/// down to single cells, so the cap is hard at any budget of one cell or more.
fn chunk_region<const D: usize>(
    region: &symbi_algebra::Domain<D>,
    budget: usize,
) -> Vec<symbi_algebra::Domain<D>> {
    fn split<const D: usize>(
        region: &symbi_algebra::Domain<D>,
        axis: usize,
        budget: usize,
        out: &mut Vec<symbi_algebra::Domain<D>>,
    ) {
        if region.volume() <= budget {
            out.push(region.clone());
            return;
        }
        let row_volume: usize = (0..axis).map(|ax| region.spaces[ax].size()).product();
        let (lo, hi) = (region.spaces[axis].lo, region.spaces[axis].hi);
        if row_volume <= budget {
            let rows = (budget / row_volume.max(1)).max(1) as isize;
            let mut r0 = lo;
            while r0 < hi {
                let r1 = (r0 + rows).min(hi);
                out.push(region.slab(axis, (r0, r1)));
                r0 = r1;
            }
        } else {
            for r in lo..hi {
                split(&region.slab(axis, (r, r + 1)), axis - 1, budget, out);
            }
        }
    }
    let mut out = Vec::new();
    if region.volume() > 0 {
        split(region, D - 1, budget.max(1), &mut out);
    }
    out
}

/// stream one field's `region` into the dataset at `dataset_path` through one reused staging
/// buffer, one box of at most `budget` cells at a time.
fn stream_region<const D: usize, Mem: MemorySpace>(
    stream: &Hdf5Stream,
    dataset_path: &str,
    field: &symbi_grid::Field<f64, D, Mem>,
    region: &symbi_algebra::Domain<D>,
    file_of: &dyn Fn(usize, isize) -> usize,
    budget: usize,
    staging: &mut Vec<f64>,
) -> Result<()> {
    for chunk in chunk_region(region, budget) {
        staging.clear();
        for_each_cell_axis0(&chunk, |coord| staging.push(*field.view().at(coord)));
        let (start, count) = slab_start_and_count(&chunk, file_of);
        stream.write_slab(dataset_path, &start, &count, staging)?;
    }
    Ok(())
}

/// the level's mesh facts for the metadata group: global interior counts, physical widths and
/// lower bounds scaled by the mesh motion, in grid axis order.
fn mesh_facts<const D: usize>(geometry: symbi_geometry::Geometry, a: f64, grid: &GlobalGrid<D>) -> MeshFacts {
    let scale = |ax: usize| motion_axis_scale(geometry, ax, D, a);
    MeshFacts {
        resolution: grid.cells.iter().map(|&c| c as u64).collect(),
        dx_phys: (0..D).map(|ax| grid.dx[ax] * scale(ax)).collect(),
        x_lo_phys: (0..D).map(|ax| grid.x_lo[ax] * scale(ax)).collect(),
    }
}

/// the cell-centered datasets a state contributes, in checkpoint order: the conserved bucket,
/// the primitive bucket, and the isothermal closure, each as (group path under the level, name,
/// field).
fn cell_datasets<'a, R, const D: usize, const DOF: usize, Mem>(
    sim: &'a FieldStore<D, DOF, Mem, f64>,
) -> Vec<(&'static str, String, &'a symbi_grid::Field<f64, D, Mem>)>
where
    R: Regime<f64, D>,
    Mem: MemorySpace,
{
    let mut out: Vec<(&'static str, String, &symbi_grid::Field<f64, D, Mem>)> = Vec::new();
    for fs in R::SPEC.primitive_fields {
        for idx in 0..symbi_io::component_count(fs, DOF) {
            let field = match fs.name {
                "rho" => Some(&sim.fields.prim.rho),
                "vel" => Some(&sim.fields.prim.vel[idx]),
                "pre" => sim.fields.prim.pre_field(),
                "bcell" => sim.fields.mhd.as_ref().map(|m| &m.bcell[idx]),
                other => panic!("checkpoint write: unknown primitive field '{other}'"),
            };
            if let Some(field) = field {
                out.push(("partition_0/hydro/primitives", symbi_io::dataset_name(fs, idx), field));
            }
        }
    }
    if let Some(chi) = sim.fields.prim.chi_field() {
        out.push(("partition_0/hydro/primitives", "chi".to_string(), chi));
    }
    for fs in R::SPEC.fields {
        for idx in 0..symbi_io::component_count(fs, DOF) {
            let field = match fs.name {
                "den" => Some(&sim.fields.cons.den),
                "mom" => Some(&sim.fields.cons.mom[idx]),
                "nrg" => sim.fields.cons.nrg_field(),
                "mag" => sim.fields.mhd.as_ref().map(|m| &m.bcell[idx]),
                other => panic!("checkpoint write: unknown conserved field '{other}'"),
            };
            if let Some(field) = field {
                out.push(("conserved", symbi_io::dataset_name(fs, idx), field));
            }
        }
    }
    if let Some(chi) = sim.fields.cons.chi_field() {
        out.push(("conserved", "chi".to_string(), chi));
    }
    if let Some(cs2) = sim.fields.cs2.as_ref() {
        out.push(("", "iso_cs2".to_string(), cs2));
    }
    out
}

/// the union of the tiles' tracer populations in stable id order, the same reassembly a gathered
/// output state performs: ids and cohorts concatenate, the next free id is the largest any tile
/// reached, the injection remainders sum, and the one particle weight every tile shares is kept.
fn combine_tracer_snaps(snaps: Vec<TracerSnap>) -> Option<TracerSnap> {
    let mut iter = snaps.into_iter();
    let mut combined = iter.next()?;
    let d = if combined.n > 0 { combined.x.len() / combined.n } else { 0 };
    for snap in iter {
        combined.n += snap.n;
        combined.x.extend(snap.x);
        combined.id.extend(snap.id);
        combined.cohort.extend(snap.cohort);
        combined.owner.extend(snap.owner);
        combined.escaped.extend(snap.escaped);
        combined.crossed.extend(snap.crossed);
        combined.crossing_time.extend(snap.crossing_time);
        combined.next_id = combined.next_id.max(snap.next_id);
        combined.injection_remainder += snap.injection_remainder;
    }
    let mut perm: Vec<usize> = (0..combined.n).collect();
    perm.sort_by_key(|&i| combined.id[i]);
    let take = |v: &Vec<f64>| -> Vec<f64> { perm.iter().map(|&i| v[i]).collect() };
    let take_u = |v: &Vec<u64>| -> Vec<u64> { perm.iter().map(|&i| v[i]).collect() };
    let x: Vec<f64> = perm
        .iter()
        .flat_map(|&i| combined.x[i * d..(i + 1) * d].iter().copied())
        .collect();
    combined.x = x;
    combined.id = take_u(&combined.id);
    combined.cohort = take_u(&combined.cohort);
    combined.owner = take_u(&combined.owner);
    combined.escaped = take(&combined.escaped);
    combined.crossed = take(&combined.crossed);
    combined.crossing_time = take(&combined.crossing_time);
    Some(combined)
}

/// write a checkpoint from the tiles of every level, streaming each tile's owned block of every
/// field into the global datasets through a bounded staging buffer: cell datasets carry the
/// global halo, which the tiles on the domain boundary own; face datasets carry the global
/// interior faces plus the closing face on each axis, owned by the last tile along it; every
/// element has exactly one writer. tile 0 of each level supplies the clock, tile 0 of the coarse
/// level the physics identity, the finest level's tile 0 the body state, and the tiles together
/// the tracer population. the file takes its name only once every write succeeded.
pub fn write_partitioned_checkpoint<R, const D: usize, const DOF: usize, Mem>(
    identity: &PhysicsIdentity,
    levels: &[LevelTiles<'_, D, DOF, Mem>],
    path: &str,
    extras: &Metadata,
) -> Result<()>
where
    R: Regime<f64, D>,
    Mem: MemorySpace,
{
    write_partitioned_checkpoint_with_budget::<R, D, DOF, Mem>(identity, levels, path, extras, staging_budget_cells())
}

/// `write_partitioned_checkpoint` at an explicit staging budget in cells per slab.
pub fn write_partitioned_checkpoint_with_budget<R, const D: usize, const DOF: usize, Mem>(
    identity: &PhysicsIdentity,
    levels: &[LevelTiles<'_, D, DOF, Mem>],
    path: &str,
    extras: &Metadata,
    budget: usize,
) -> Result<()>
where
    R: Regime<f64, D>,
    Mem: MemorySpace,
{
    if levels.is_empty() || levels.iter().any(|l| l.tiles.is_empty()) {
        return Err(IoError::MissingPath("hierarchy has no levels or a level has no tiles".into()));
    }
    #[cfg(feature = "gpu")]
    if Mem::IS_DEVICE_ACCESSIBLE {
        symbi_xpu::ctx_sync();
    }
    let authority = levels[0].tiles[0].state;
    let facts = mesh_facts(identity.geometry, authority.motion.a, &levels[0].grid);
    let mut root = Tree::new("")
        .with_attr("format_version", "2.0")
        .with_attr("symbi_version", "0.1.0");
    if let Some(tag) = conserved_densitization(authority) {
        root = root.with_attr(CONSERVED_DENSITIZATION_ATTR, tag);
    }
    root.push_group(build_metadata_group(identity, authority, &facts, extras));

    // the small per-level groups: mesh geometry, ownership, the face domains and the level clock.
    struct LevelSmall {
        mesh_cells: Vec<u64>,
        owned_start: Vec<i64>,
        owned_fin: Vec<i64>,
        face_domains: Vec<(Vec<i64>, Vec<i64>)>,
    }
    let smalls: Vec<LevelSmall> = levels
        .iter()
        .map(|level| {
            let grid = &level.grid;
            let faces = level.tiles[0]
                .state
                .fields
                .mhd
                .as_ref()
                .is_some_and(|m| m.bface_initialized.load(std::sync::atomic::Ordering::Relaxed));
            LevelSmall {
                mesh_cells: (0..D).rev().map(|ax| grid.cells[ax] as u64).collect(),
                owned_start: vec![0; D],
                owned_fin: (0..D).rev().map(|ax| grid.cells[ax] as i64).collect(),
                face_domains: if faces {
                    (0..D)
                        .map(|d| {
                            let start: Vec<i64> = (0..D).map(|ax| grid.interior_lo[ax] as i64).collect();
                            let fin: Vec<i64> = (0..D)
                                .map(|ax| grid.interior_lo[ax] as i64 + grid.cells[ax] as i64 + if ax == d { 1 } else { 0 })
                                .collect();
                            (start, fin)
                        })
                        .collect()
                } else {
                    Vec::new()
                },
            }
        })
        .collect();
    for (idx, (level, small)) in levels.iter().zip(&smalls).enumerate() {
        let sim = level.tiles[0].state;
        let grid = &level.grid;
        let geometry_kind = identity.geometry;
        let mut geometry = Tree::new("geometry").with_attr("metric", coord_name(geometry_kind));
        for (slot, ax) in (0..D).rev().enumerate() {
            let lo_index = grid.interior_lo[ax];
            let hi_index = lo_index + grid.cells[ax] as isize;
            let scale = motion_axis_scale(geometry_kind, ax, D, sim.motion.a);
            let (start, end) = match &grid.maps {
                Some(maps) => (maps[ax].face(lo_index) * scale, maps[ax].face(hi_index) * scale),
                None => {
                    let start = facts_x_lo(&grid.x_lo, grid.dx[ax], lo_index, ax, scale);
                    (start, start + grid.dx[ax] * scale * grid.cells[ax] as f64)
                }
            };
            let (spacing_label, spacing_ratio) = match &grid.maps {
                Some(maps) => match maps[ax] {
                    symbi_geometry::AxisMap::Uniform { .. } => ("linear", 1.0),
                    symbi_geometry::AxisMap::Log { .. } => ("log", 1.0),
                    symbi_geometry::AxisMap::Geometric { ratio, .. } => ("geometric", ratio),
                },
                None => ("linear", 1.0),
            };
            geometry.push_group(
                Tree::new(format!("dim_{slot}"))
                    .with_attr("start", start)
                    .with_attr("end", end)
                    .with_attr("type", spacing_label)
                    .with_attr("ratio", spacing_ratio),
            );
        }
        let mesh = Tree::new("mesh")
            .with_attr("halo_width", grid.ng as u64)
            .with_dataset(Dataset::new("global_cells", vec![D], DataRef::U64(&small.mesh_cells)))
            .with_group(geometry);
        let mut hydro = Tree::new("hydro").with_group(Tree::new("primitives"));
        if !small.face_domains.is_empty() {
            let mut magnetic = Tree::new("magnetic");
            for (d, (start, fin)) in small.face_domains.iter().enumerate() {
                magnetic.push_group(
                    Tree::new(format!("B{}", d + 1)).with_group(
                        Tree::new("domain")
                            .with_dataset(Dataset::new("start", vec![D], DataRef::I64(start)))
                            .with_dataset(Dataset::new("fin", vec![D], DataRef::I64(fin))),
                    ),
                );
            }
            hydro.push_group(magnetic);
        }
        let partition_0 = Tree::new("partition_0")
            .with_dataset(Dataset::new("owned_start", vec![D], DataRef::I64(&small.owned_start)))
            .with_dataset(Dataset::new("owned_fin", vec![D], DataRef::I64(&small.owned_fin)))
            .with_group(hydro);
        root.push_group(
            Tree::new(format!("level_{idx}"))
                .with_attr("scale_factor_a", sim.motion.a)
                .with_attr("scale_factor_adot", sim.motion.a_dot)
                .with_attr("time", sim.time)
                .with_attr("dt", sim.dt)
                .with_attr("iteration", Attr::U64(sim.iteration))
                .with_group(mesh)
                .with_group(partition_0)
                .with_group(Tree::new("conserved")),
        );
    }
    // the sidecars every tile shares: the finest level's body state, the tiles' tracer union,
    // the continuous tracers, the body series, and, for a level held whole, its censuses.
    let finest_bodies = levels
        .iter()
        .filter_map(|l| l.tiles[0].state.immersed.as_ref())
        .last()
        .map(body_state_snap);
    let heat_name = slip_heat_dataset_name(authority.fields.cons.nrg_field().is_some());
    let _ = identity.has_energy;
    if let Some(bs) = finest_bodies.as_ref() {
        root.push_group(body_state_group::<D>(bs, heat_name));
    }
    // a level held whole writes its population in its own order; tiles combine theirs in id order.
    let tracers = levels
        .iter()
        .find(|l| l.tiles.iter().any(|t| t.state.tracers.is_some()))
        .and_then(|l| {
            let snaps: Vec<TracerSnap> =
                l.tiles.iter().filter_map(|t| t.state.tracers.as_ref().map(tracer_snap)).collect();
            if snaps.len() == 1 { snaps.into_iter().next() } else { combine_tracer_snaps(snaps) }
        });
    if let Some(ts) = tracers.as_ref() {
        root.push_group(tracer_group::<D>(ts));
    }
    let continuous = combine_continuous_tracer_snaps(
        levels
            .iter()
            .flat_map(|l| l.tiles.iter())
            .filter_map(|t| t.state.continuous_tracers.as_ref().map(continuous_tracer_snap)),
    );
    if let Some(snap) = continuous.as_ref() {
        root.push_group(continuous_tracer_group::<D>(snap));
    }
    if let Some(im) = levels
        .iter()
        .filter_map(|l| l.tiles[0].state.immersed.as_ref())
        .find(|im| !im.history.is_empty())
    {
        let (n, nb) = (im.history.len(), im.history.n_bodies());
        root.push_group(
            Tree::new("body_diagnostics")
                .with_attr("n_bodies", nb as u64)
                .with_dataset(Dataset::new("time", vec![n], DataRef::F64(im.history.time())))
                .with_dataset(Dataset::new("dt", vec![n], DataRef::F64(im.history.dt())))
                .with_dataset(Dataset::new("mass_delta", vec![n, nb], DataRef::F64(im.history.mass_delta())))
                .with_dataset(Dataset::new("energy_delta", vec![n, nb], DataRef::F64(im.history.energy_delta())))
                .with_dataset(Dataset::new("force", vec![n, nb, D], DataRef::F64(im.history.force())))
                .with_dataset(Dataset::new("force_normal", vec![n, nb, D], DataRef::F64(im.history.force_normal())))
                .with_dataset(Dataset::new("torque", vec![n, nb, 3], DataRef::F64(im.history.torque()))),
        );
    }
    if levels[0].tiles.len() == 1 {
        for group in census_groups(authority) {
            root.push_group(group);
        }
    }

    let stream = Hdf5Stream::create(Path::new(path))?;
    stream.write_tree(&root)?;
    let mut staging: Vec<f64> = Vec::new();
    for (idx, level) in levels.iter().enumerate() {
        let grid = &level.grid;
        let level_path = format!("level_{idx}");
        let cell_shape: Vec<usize> = (0..D).rev().map(|ax| grid.cells[ax] + 2 * grid.ng).collect();
        let names = cell_datasets::<R, D, DOF, Mem>(level.tiles[0].state);
        for (group, name, _) in &names {
            let group_path = if group.is_empty() { level_path.clone() } else { format!("{level_path}/{group}") };
            stream.declare_f64(&group_path, name, &cell_shape)?;
        }
        for d in 0..D {
            if level.tiles[0].state.fields.mhd.as_ref().is_some_and(|m| m.bface_initialized.load(std::sync::atomic::Ordering::Relaxed)) {
                let face_shape: Vec<usize> = (0..D)
                    .rev()
                    .map(|ax| grid.cells[ax] + if ax == d { 1 } else { 0 })
                    .collect();
                stream.declare_f64(&format!("{level_path}/partition_0/hydro/magnetic/B{}", d + 1), "data", &face_shape)?;
            }
        }
        for tile in &level.tiles {
            let sim = tile.state;
            let interior = &sim.geom.interior;
            let alloc = sim.fields.cons.den.domain();
            let ng = grid.ng as isize;
            let first: [bool; D] = std::array::from_fn(|ax| tile.offset[ax] == grid.interior_lo[ax]);
            let last: [bool; D] = std::array::from_fn(|ax| {
                tile.offset[ax] + interior.spaces[ax].size() as isize == grid.interior_lo[ax] + grid.cells[ax] as isize
            });
            // a tile's cell block: its interior, extended into the global halo on every side it
            // touches the domain boundary. the file's cell index of local coordinate `c` on an
            // axis is the global interior index plus the halo width.
            let cell_region = {
                let mut r = interior.clone();
                for ax in 0..D {
                    let lo = if first[ax] { alloc.spaces[ax].lo } else { interior.spaces[ax].lo };
                    let hi = if last[ax] { alloc.spaces[ax].hi } else { interior.spaces[ax].hi };
                    r = r.slab(ax, (lo, hi));
                }
                r
            };
            let shift: [isize; D] = std::array::from_fn(|ax| tile.offset[ax] - interior.spaces[ax].lo - grid.interior_lo[ax]);
            let cell_file_of = |ax: usize, c: isize| (c + shift[ax] + ng) as usize;
            for (group, name, field) in cell_datasets::<R, D, DOF, Mem>(sim) {
                let dataset_path = if group.is_empty() {
                    format!("{level_path}/{name}")
                } else {
                    format!("{level_path}/{group}/{name}")
                };
                stream_region(&stream, &dataset_path, field, &cell_region, &cell_file_of, budget, &mut staging)?;
            }
            if let Some(mhd) = sim.fields.mhd.as_ref().filter(|m| m.bface_initialized.load(std::sync::atomic::Ordering::Relaxed)) {
                for d in 0..D {
                    // a tile's faces along `d`: those at the low side of its interior cells, plus
                    // the closing face when the tile ends the axis; interior extents elsewhere.
                    let mut region = interior.clone();
                    if last[d] {
                        region = region.extend(d, 0, 1);
                    }
                    let face_file_of = |ax: usize, c: isize| (c + shift[ax]) as usize;
                    stream_region(
                        &stream,
                        &format!("{level_path}/partition_0/hydro/magnetic/B{}/data", d + 1),
                        &mhd.bface[d],
                        &region,
                        &face_file_of,
                        budget,
                        &mut staging,
                    )?;
                }
            }
        }
    }
    stream.publish()?;
    Ok(())
}

/// the physical lower bound of a uniform axis at global interior index `lo_index`.
fn facts_x_lo(x_lo: &[f64], dx: f64, lo_index: isize, ax: usize, scale: f64) -> f64 {
    x_lo[ax] * scale + lo_index as f64 * dx * scale
}

#[cfg(test)]
mod partitioned_tests_support {
    use super::*;
    use symbi_geometry::Cartesian;
    use symbi_hydro::eos::IdealGas;
    use symbi_hydro::newtonian_mhd::NewtonianMhd;
    use symbi_xpu::{CpuSpace, HostMemory};

    pub type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    pub const NG: usize = 2;

    /// the analytic value of field `tag` at global cell `g`: distinct per field and per cell.
    pub fn value(tag: usize, g: [isize; 2]) -> f64 {
        1.0 + tag as f64 * 0.01 + 0.5 * g[0] as f64 + 0.125 * g[1] as f64
    }

    /// a blank 2.5D MHD tile whose interior starts at global index `offset` with `cells` cells,
    /// on a log-spaced first axis.
    pub fn blank(offset: [isize; 2], cells: [usize; 2]) -> Sim {
        let dx = [0.1, 0.25];
        let x_lo = [1.0 * 10f64.powf(0.05 * offset[0] as f64), offset[1] as f64 * dx[1]];
        let mut sim = Sim::new(
            NewtonianMhd,
            IdealGas { gamma: 5.0 / 3.0 },
            Cartesian,
            cells,
            x_lo,
            dx,
            NG,
            Boundaries::uniform(BoundaryType::Outflow),
            0.4,
            Timestepping::Rk2,
            0,
        )
        .unwrap();
        sim.geom.set_maps([
            symbi_geometry::AxisMap::Log { start: x_lo[0], log_slope: 0.05 },
            symbi_geometry::AxisMap::Uniform { start: x_lo[1], dx: dx[1] },
        ]);
        sim
    }

    /// the distinct cell fields a state stores, each once: the cell-centered magnetic field
    /// appears under both the primitive and the conserved bucket and carries one value.
    pub fn unique_cell_fields(sim: &Sim) -> Vec<(String, &symbi_grid::Field<f64, 2, HostMemory>)> {
        let mut out: Vec<(String, &symbi_grid::Field<f64, 2, HostMemory>)> = Vec::new();
        for (_, name, field) in cell_datasets::<NewtonianMhd, 2, 3, HostMemory>(&sim.store) {
            if !out.iter().any(|(_, f)| std::ptr::eq(*f, field)) {
                out.push((name, field));
            }
        }
        out
    }

    /// `blank` with every cell and face field, halos included, set to the analytic function of
    /// the global index, so tiles cut from one grid agree with the whole grid wherever both hold
    /// a value.
    pub fn tile(offset: [isize; 2], cells: [usize; 2]) -> Sim {
        let sim = blank(offset, cells);
        let int_lo = [sim.geom.interior.spaces[0].lo, sim.geom.interior.spaces[1].lo];
        let global = |c: [isize; 2]| [c[0] - int_lo[0] + offset[0], c[1] - int_lo[1] + offset[1]];
        for (tag, (_, field)) in unique_cell_fields(&sim).into_iter().enumerate() {
            let view = field.view_mut();
            for c in field.domain().iter() {
                view.set(c, value(tag, global(c)));
            }
        }
        let mhd = sim.fields.mhd.as_ref().unwrap();
        for d in 0..2 {
            let view = mhd.bface[d].view_mut();
            for c in mhd.bface[d].domain().iter() {
                view.set(c, value(100 + d, global(c)));
            }
        }
        mhd.bface_initialized.store(true, std::sync::atomic::Ordering::Relaxed);
        sim
    }

    /// every cell of the tile whose global index lies inside the file's cell box, and every face
    /// of its owned face domain, against the analytic function.
    pub fn assert_tile_matches_the_function(sim: &Sim, offset: [isize; 2], grid: &GlobalGrid<2>, label: &str) {
        let int_lo = [sim.geom.interior.spaces[0].lo, sim.geom.interior.spaces[1].lo];
        let global = |c: [isize; 2]| [c[0] - int_lo[0] + offset[0], c[1] - int_lo[1] + offset[1]];
        let ng = grid.ng as isize;
        let inside = |g: [isize; 2]| (0..2).all(|ax| g[ax] >= -ng && g[ax] < grid.cells[ax] as isize + ng);
        let mut checked = 0;
        for (tag, (name, field)) in unique_cell_fields(sim).into_iter().enumerate() {
            for c in field.domain().iter() {
                let g = global(c);
                if !inside(g) {
                    continue;
                }
                let got = *field.view().at(c);
                assert_eq!(got.to_bits(), value(tag, g).to_bits(), "{label}: {name} at {c:?} (global {g:?}) = {got}");
                checked += 1;
            }
        }
        let mhd = sim.fields.mhd.as_ref().unwrap();
        for d in 0..2 {
            for c in sim.geom.interior.extend(d, 0, 1).iter() {
                let got = *mhd.bface[d].view().at(c);
                assert_eq!(got.to_bits(), value(100 + d, global(c)).to_bits(), "{label}: B{} at {c:?}", d + 1);
                checked += 1;
            }
        }
        assert!(checked > 0, "{label}: nothing compared");
    }

    pub fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("symbi_partitioned_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}

#[cfg(test)]
mod partitioned_tests {
    use super::partitioned_tests_support::*;
    use super::*;
    use symbi_hydro::newtonian_mhd::NewtonianMhd;
    use symbi_io::TreeBuf;
    use symbi_xpu::HostMemory;

    fn assert_trees_equal(a: &TreeBuf, b: &TreeBuf, path: &str) {
        assert_eq!(a.attrs.len(), b.attrs.len(), "{path}: attribute count");
        for (x, y) in a.attrs.iter().zip(&b.attrs) {
            assert_eq!(x.0, y.0, "{path}: attribute name");
            assert_eq!(format!("{:?}", x.1), format!("{:?}", y.1), "{path}: attribute {}", x.0);
        }
        assert_eq!(a.datasets.len(), b.datasets.len(), "{path}: dataset count");
        for (x, y) in a.datasets.iter().zip(&b.datasets) {
            assert_eq!(x.name, y.name, "{path}: dataset name");
            assert_eq!(x.shape, y.shape, "{path}/{}: shape", x.name);
            match (x.data.as_f64(), y.data.as_f64()) {
                (Some(p), Some(q)) => {
                    assert_eq!(p.len(), q.len(), "{path}/{}: length", x.name);
                    for (i, (u, v)) in p.iter().zip(q).enumerate() {
                        assert_eq!(u.to_bits(), v.to_bits(), "{path}/{}[{i}]: {u} vs {v}", x.name);
                    }
                }
                _ => assert_eq!(format!("{:?}", x.data), format!("{:?}", y.data), "{path}/{}", x.name),
            }
        }
        assert_eq!(
            a.groups.iter().map(|g| g.name.clone()).collect::<Vec<_>>(),
            b.groups.iter().map(|g| g.name.clone()).collect::<Vec<_>>(),
            "{path}: groups"
        );
        for (x, y) in a.groups.iter().zip(&b.groups) {
            assert_trees_equal(x, y, &format!("{path}/{}", x.name));
        }
    }

    #[test]
    fn tiles_cut_unevenly_along_an_axis_write_the_whole_grid_file() {
        let dir = scratch("tiles");
        let whole = tile([0, 0], [8, 6]);
        let left = tile([0, 0], [3, 6]);
        let right = tile([3, 0], [5, 6]);
        let grid = GlobalGrid::of_state(&whole.store);
        let whole_path = dir.join("whole.h5");
        let tiled_path = dir.join("tiled.h5");
        write_partitioned_checkpoint::<NewtonianMhd, 2, 3, HostMemory>(&PhysicsIdentity::of(&whole), &[LevelTiles::whole(&whole.store)], whole_path.to_str().unwrap(), &Metadata::new()).unwrap();
        let level = LevelTiles {
            tiles: vec![
                TileView { state: &left.store, offset: [0, 0] },
                TileView { state: &right.store, offset: [3, 0] },
            ],
            grid,
        };
        // a staging budget of five cells forces every block through several slabs.
        write_partitioned_checkpoint_with_budget::<NewtonianMhd, 2, 3, HostMemory>(&PhysicsIdentity::of(&whole), &[level], tiled_path.to_str().unwrap(), &Metadata::new(), 5).unwrap();
        let a = Hdf5Backend.read(&whole_path).unwrap();
        let b = Hdf5Backend.read(&tiled_path).unwrap();
        assert_trees_equal(&a, &b, "");
    }

    #[test]
    fn the_staging_cap_holds_below_one_row_and_down_to_one_cell() {
        let region = symbi_algebra::Domain::new([
            symbi_algebra::Space { name: "x", lo: -2, hi: 6 },
            symbi_algebra::Space { name: "y", lo: 0, hi: 3 },
        ]);
        for budget in [1usize, 3, 5, 8, 9, 100] {
            let chunks = chunk_region(&region, budget);
            assert!(chunks.iter().all(|c| c.volume() <= budget), "budget {budget}: a box exceeds it");
            assert_eq!(chunks.iter().map(|c| c.volume()).sum::<usize>(), region.volume(), "budget {budget}: coverage");
            let mut seen = std::collections::HashSet::new();
            for c in &chunks {
                for coord in c.iter() {
                    assert!(seen.insert(coord), "budget {budget}: cell {coord:?} twice");
                }
            }
        }
        assert_eq!(chunk_region(&region, 1).len(), 24, "one cell per box");
        // a file written at the one-cell cap matches the whole-block write.
        let dir = scratch("cap");
        let whole = tile([0, 0], [8, 6]);
        let a = dir.join("wide.h5");
        let b = dir.join("cell.h5");
        let id = PhysicsIdentity::of(&whole);
        write_partitioned_checkpoint::<NewtonianMhd, 2, 3, HostMemory>(&id, &[LevelTiles::whole(&whole.store)], a.to_str().unwrap(), &Metadata::new()).unwrap();
        write_partitioned_checkpoint_with_budget::<NewtonianMhd, 2, 3, HostMemory>(&id, &[LevelTiles::whole(&whole.store)], b.to_str().unwrap(), &Metadata::new(), 1).unwrap();
        assert_trees_equal(&Hdf5Backend.read(&a).unwrap(), &Hdf5Backend.read(&b).unwrap(), "");
    }

    #[test]
    fn a_write_that_cannot_open_its_file_publishes_nothing() {
        let sim = tile([0, 0], [4, 4]);
        let target = scratch("sealed").join("absent").join("run.h5");
        let err = write_partitioned_checkpoint::<NewtonianMhd, 2, 3, HostMemory>(&PhysicsIdentity::of(&sim), &[LevelTiles::whole(&sim.store)], target.to_str().unwrap(), &Metadata::new()).unwrap_err();
        assert!(matches!(err, IoError::Backend(_)), "{err:?}");
        assert!(!target.exists());
    }
}

// =============================================================================
// partitioned restart: a tile reads its own block of each global dataset
// =============================================================================

/// read `region` of the dataset at `dataset_path` into `field`, one box of at most `budget`
/// cells at a time.
fn read_region<const D: usize, Mem: MemorySpace>(
    path: &Path,
    dataset_path: &str,
    field: &symbi_grid::Field<f64, D, Mem>,
    region: &symbi_algebra::Domain<D>,
    file_of: &dyn Fn(usize, isize) -> usize,
    budget: usize,
) -> Result<()> {
    let view = field.view_mut();
    for chunk in chunk_region(region, budget) {
        let (start, count) = slab_start_and_count(&chunk, file_of);
        let data = read_slab(path, dataset_path, &start, &count)?;
        let mut ii = 0usize;
        for_each_cell_axis0(&chunk, |coord| {
            view.set(coord, data[ii]);
            ii += 1;
        });
    }
    Ok(())
}

fn attr_of<'a>(attrs: &'a [(String, Attr)], name: &str) -> Option<&'a Attr> {
    attrs.iter().find(|(k, _)| k == name).map(|(_, v)| v)
}

/// load level `level_index` of the checkpoint at `path` into one tile of a partition of the
/// level's grid: the tile's cells over its allocated box clipped to the file's cell box, so a
/// cut halo arrives from the neighbor's interior and a domain halo from the file's own, its
/// faces over its owned face domain (the shared cut face included), the level clock, and the
/// body state. the file carries the global grid alone, so the partition it is read into is
/// free. tracer populations are refused, since they need repartitioning by owner.
pub fn load_partitioned_level<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &mut SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    path: &str,
    level_index: usize,
    offset: [isize; D],
    grid: &GlobalGrid<D>,
) -> Result<()>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    let file = Path::new(path);
    let root = read_attrs(file, "")?;
    let stored = attr_of(&root, CONSERVED_DENSITIZATION_ATTR).map(|a| format!("{a:?}"));
    let expected = conserved_densitization(&sim.store).map(|t| format!("{:?}", Attr::Str(t.into())));
    if stored != expected {
        return Err(IoError::Backend(format!(
            "checkpoint '{path}' stores conserved variables with densitization {stored:?}; this run evolves {expected:?}"
        )));
    }
    let meta = read_attrs(file, "metadata")?;
    let regime = attr_of(&meta, "regime")
        .ok_or_else(|| IoError::MissingPath("metadata/regime".into()))?
        .as_str("metadata/regime")?;
    if regime != regime_name(&sim.physics.regime) {
        return Err(IoError::Backend(format!(
            "checkpoint '{path}' holds a {regime} run; this run is {}",
            regime_name(&sim.physics.regime)
        )));
    }
    let level_path = format!("level_{level_index}");
    let level_attrs = read_attrs(file, &level_path)?;
    if let Some(a) = attr_of(&level_attrs, "time") {
        sim.time = a.as_f64(&format!("{level_path}/time"))?;
    }
    if let Some(a) = attr_of(&level_attrs, "iteration") {
        sim.iteration = a.as_u64(&format!("{level_path}/iteration"))?;
    }
    let mesh = read_group(file, &format!("{level_path}/mesh"))?;
    let stored_cells: Vec<u64> = mesh
        .find_dataset("global_cells")
        .and_then(|d| d.data.as_u64().map(|v| v.to_vec()))
        .ok_or_else(|| IoError::MissingPath(format!("{level_path}/mesh/global_cells")))?;
    let expected_cells: Vec<u64> = (0..D).rev().map(|ax| grid.cells[ax] as u64).collect();
    if stored_cells != expected_cells {
        return Err(IoError::ShapeMismatch {
            path: format!("{level_path}/mesh/global_cells"),
            expected: expected_cells.iter().map(|&c| c as usize).collect(),
            actual: stored_cells.iter().map(|&c| c as usize).collect(),
        });
    }
    let halo = mesh
        .find_attr("halo_width")
        .ok_or_else(|| IoError::MissingPath(format!("{level_path}/mesh/halo_width")))?
        .as_u64("halo_width")? as usize;
    if halo != grid.ng {
        return Err(IoError::Backend(format!(
            "checkpoint '{path}' carries a halo of {halo} cells; this run allocates {}",
            grid.ng
        )));
    }
    let budget = staging_budget_cells();
    let interior = sim.geom.interior.clone();
    let alloc = sim.fields.cons.den.domain().clone();
    let ng = grid.ng as isize;
    let shift: [isize; D] = std::array::from_fn(|ax| offset[ax] - interior.spaces[ax].lo - grid.interior_lo[ax]);
    // the tile's cells whose file index lies inside the file's cell box.
    let cell_region = {
        let mut r = alloc.clone();
        for ax in 0..D {
            let lo = alloc.spaces[ax].lo.max(-shift[ax] - ng);
            let hi = alloc.spaces[ax].hi.min(grid.cells[ax] as isize + ng - shift[ax]);
            r = r.slab(ax, (lo, hi));
        }
        r
    };
    let cell_file_of = |ax: usize, c: isize| (c + shift[ax] + ng) as usize;
    for (group, name, field) in cell_datasets::<R, D, DOF, Mem>(&sim.store) {
        let dataset_path = if group.is_empty() {
            format!("{level_path}/{name}")
        } else {
            format!("{level_path}/{group}/{name}")
        };
        read_region(file, &dataset_path, field, &cell_region, &cell_file_of, budget)?;
    }
    if let Some(mhd) = sim.fields.mhd.as_ref() {
        let face_file_of = |ax: usize, c: isize| (c + shift[ax]) as usize;
        for d in 0..D {
            let dataset_path = format!("{level_path}/partition_0/hydro/magnetic/B{}/data", d + 1);
            dataset_shape(file, &dataset_path)?;
            let region = interior.extend(d, 0, 1);
            read_region(file, &dataset_path, &mhd.bface[d], &region, &face_file_of, budget)?;
        }
        mhd.bface_initialized.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    if read_group(file, "tracers").is_ok() {
        return Err(IoError::Backend(format!(
            "checkpoint '{path}' carries a tracer population; a partitioned restart does not repartition tracers"
        )));
    }
    if let Some(im) = sim.immersed.as_mut() {
        if let Ok(bodies) = read_group(file, "bodies") {
            restore_bodies::<D>(&bodies, im)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod partitioned_restart_tests {
    use super::partitioned_tests_support::*;
    use super::*;
    use symbi_hydro::newtonian_mhd::NewtonianMhd;
    use symbi_xpu::HostMemory;

    #[test]
    fn a_restart_reconstructs_tiles_of_a_different_cut_and_the_whole_grid() {
        let dir = scratch("restart");
        let path = dir.join("two.h5");
        let source = [tile([0, 0], [3, 6]), tile([3, 0], [5, 6])];
        let grid = GlobalGrid::of_state(&tile([0, 0], [8, 6]).store);
        let level = LevelTiles {
            tiles: vec![
                TileView { state: &source[0].store, offset: [0, 0] },
                TileView { state: &source[1].store, offset: [3, 0] },
            ],
            grid: grid.clone(),
        };
        write_partitioned_checkpoint::<NewtonianMhd, 2, 3, HostMemory>(&PhysicsIdentity::of(&source[0]), &[level], path.to_str().unwrap(), &Metadata::new()).unwrap();
        let path = path.to_str().unwrap();
        // three tiles cut where the source had none, each starting blank.
        let cuts = [([0isize, 0], [2usize, 6]), ([2, 0], [3, 6]), ([5, 0], [3, 6])];
        for (offset, cells) in cuts {
            let mut fresh = blank(offset, cells);
            load_partitioned_level(&mut fresh, path, 0, offset, &grid).unwrap();
            assert_tile_matches_the_function(&fresh, offset, &grid, "three-cut tile");
        }
        let mut whole = blank([0, 0], [8, 6]);
        load_partitioned_level(&mut whole, path, 0, [0, 0], &grid).unwrap();
        assert_tile_matches_the_function(&whole, [0, 0], &grid, "whole from tiles");
        // the classic whole-grid loader reads the same file.
        let mut classic = blank([0, 0], [8, 6]);
        load_checkpoint_level(&mut classic, path, 0).unwrap();
        assert_tile_matches_the_function(&classic, [0, 0], &grid, "classic loader");
    }

    #[test]
    fn a_restart_refuses_a_grid_of_another_size_or_regime_halo() {
        let dir = scratch("refuse_restart");
        let path = dir.join("one.h5");
        let source = tile([0, 0], [8, 6]);
        write_partitioned_checkpoint::<NewtonianMhd, 2, 3, HostMemory>(&PhysicsIdentity::of(&source), &[LevelTiles::whole(&source.store)], path.to_str().unwrap(), &Metadata::new()).unwrap();
        let mut other = blank([0, 0], [8, 4]);
        let wrong = GlobalGrid::of_state(&other.store);
        let err = load_partitioned_level(&mut other, path.to_str().unwrap(), 0, [0, 0], &wrong).unwrap_err();
        assert!(matches!(err, IoError::ShapeMismatch { .. }), "{err:?}");
    }
}
