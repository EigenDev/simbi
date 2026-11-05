// =============================================================================
// torus_ledger.rs
//
// the adiabatic torus on the log (r, theta) plane through the production stage, with a per-stage
// ledger the frontend's checkpoints cannot supply. each stage runs fold_stage on the admitted
// kernels; hooks read the stage's stored fluxes and its output, so the driver adds no numerics.
// - bounds: a physical end time, an absolute step cap and a wall-time limit per launch. reaching a
//   budget ends the run as incomplete with a resumable checkpoint, exit code 75
// - samples: rows at exact multiples of the sample interval, the step clipped to land on them, so
//   two resolutions compare at identical physical times
// - every stage: lowest density, pressure and specific internal energy with their locations
// - the ledger: wall and shell transport of mass, angular momentum and E + rho Phi, weighted by the
//   stage weights of rk2, closing the volume totals; the closure residuals are in each row
// - a stage output with a non-finite value or a non-positive density or internal energy ends
//   the run as inadmissible at that stage, exit code 1, with its location
//
// - provenance: the method, the source revision and the binary's digest are written once to
//   out/provenance.json and into every restart file. a resume whose method, revision, binary,
//   sample interval or grid differ from the file's is refused by name, exit code 2, with no file
//   touched. a seed under a directory marked outside the authorized execution scope is refused
//
// inputs: a frontend checkpoint holding the initial state (--seed), or --resume from out/restart.h5
// outputs: out/provenance.json, out/rows.jsonl, out/status.json, out/restart.h5,
//   out/snapshot_NNNNNN.h5
//
// usage:
//   torus_ledger --seed t0.h5 --out run --ni 84 --nj 44 --t-end 1 --sample-dt 0.001 \
//     --snapshot-every 10 --max-steps 2000000 --max-wall-seconds 86400 --revision abc \
//     --binary-sha256 def [--resume]
// =============================================================================

use std::f64::consts::PI;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use symbi::regimes::substrate_newton::{AdiabaticSubstrateKernelSet, admits_face_gravity_work};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_discretize::ibm::body_potential;
use symbi_geometry::{AxisMap, Spherical};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, SofteningKind};
use symbi_sim::checkpoint::{load_checkpoint, write_checkpoint};
use symbi_sim::driver::{downstream_injection_weight, needs_step_snapshot, stage_exit_fraction, stage_schedule};
use symbi_sim::stage::{HookPoint, StageArgs, StageOutcome, fold_stage};
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

pub type Sim = SimStateGeneric<Newtonian, 2, 3, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
pub type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
pub type Fields = PartitionFieldsGeneric<2, 3, HostMemory, f64>;

pub const GAMMA: f64 = 5.0 / 3.0;
pub const CFL: f64 = 0.1;
pub const THETA: f64 = 1.5;
pub const R_LO: f64 = 0.01;
pub const R_HI: f64 = 4.0;
pub const SOFTENING: f64 = 0.009;
pub const MASS: f64 = 1.0;
/// the torus's pressure maximum sits at r = 1; inner and outer minima split there
pub const SPLIT_RADIUS: f64 = 1.0;
/// the interior shells' target radii; each resolves to the nearest face of the coarsest grid
pub const SHELL_RADII: [f64; 3] = [0.5, 1.0, 2.0];
/// the coarsest grid's radial cell count; every grid is a multiple of it, so shells share a radius
pub const COARSE_NI: usize = 84;

pub const EXIT_COMPLETE: i32 = 0;
pub const EXIT_FAILED: i32 = 1;
pub const EXIT_BUDGET: i32 = 75;
pub const EXIT_REFUSED: i32 = 2;
/// a directory holding this file, or under one that does, is outside the authorized execution scope
pub const OUTSIDE_SCOPE_MARKER: &str = "UNLEDGERED_OUTSIDE_AUTHORIZED_SCOPE";

#[derive(Clone, Debug)]
pub struct Bounds {
    pub t_end: f64,
    pub sample_dt: f64,
    /// a snapshot and restart file every this many samples
    pub snapshot_every: u64,
    /// the absolute iteration cap, counted from the start of the simulation
    pub max_steps: u64,
    pub max_wall_seconds: f64,
}

/// the state a resumed run must continue with, in the restart file's metadata
#[derive(Clone, Debug)]
pub struct Ledger {
    /// volume totals at the start of the run: mass, angular momentum, E + rho Phi
    pub start: [f64; 3],
    /// outward time integrals through the radial walls and polar walls: mass, L, E + rho Phi
    pub inner: [f64; 3],
    pub outer: [f64; 3],
    pub polar: [f64; 3],
    /// (net outward, gross inward, gross outward) mass through each shell, then the two walls
    pub transport: [[f64; 3]; 5],
    /// the rows this run has written
    pub rows: u64,
    /// the multiple of the sample interval last landed on
    pub sample: u64,
    /// the every-stage minima since the last row, which a stop between two rows carries over
    pub window: Extremes,
}

const WINDOW_EXTREMES: usize = 8;
const LEDGER_FIELDS: usize = 3 * 4 + 15 + 2 + 3 * WINDOW_EXTREMES;

impl Default for Ledger {
    fn default() -> Self {
        Ledger {
            start: [0.0; 3],
            inner: [0.0; 3],
            outer: [0.0; 3],
            polar: [0.0; 3],
            transport: [[0.0; 3]; 5],
            rows: 0,
            sample: 0,
            window: Extremes::empty(),
        }
    }
}

impl Ledger {
    fn to_values(&self) -> [f64; LEDGER_FIELDS] {
        let mut out = [0.0; LEDGER_FIELDS];
        let flat = self.start.iter().chain(&self.inner).chain(&self.outer).chain(&self.polar).chain(self.transport.iter().flatten());
        for (slot, value) in out.iter_mut().zip(flat) {
            *slot = *value;
        }
        out[27] = self.rows as f64;
        out[28] = self.sample as f64;
        for (kk, extreme) in self.window.listed().iter().enumerate() {
            out[29 + 3 * kk] = extreme.value;
            out[30 + 3 * kk] = extreme.cell[0] as f64;
            out[31 + 3 * kk] = extreme.cell[1] as f64;
        }
        out
    }

    fn from_values(v: &[f64]) -> Self {
        assert_eq!(v.len(), LEDGER_FIELDS, "the restart file's ledger length");
        let three = |at: usize| [v[at], v[at + 1], v[at + 2]];
        Ledger {
            start: three(0),
            inner: three(3),
            outer: three(6),
            polar: three(9),
            transport: std::array::from_fn(|kk| three(12 + 3 * kk)),
            rows: v[27] as u64,
            sample: v[28] as u64,
            window: Extremes::from_listed(std::array::from_fn(|kk| Extreme { value: v[29 + 3 * kk], cell: [v[30 + 3 * kk] as usize, v[31 + 3 * kk] as usize] })),
        }
    }
}

/// the log-radial, uniform-polar geometry in closed form, one row of radial and polar weights
pub struct Mesh {
    pub ni: usize,
    pub nj: usize,
    pub r_face: Vec<f64>,
    pub t_face: Vec<f64>,
    pub r_c: Vec<f64>,
    pub t_c: Vec<f64>,
    pub phi_c: Vec<f64>,
    pub phi_face: Vec<f64>,
    /// the interior shells' face index in this grid and their radius
    pub shells: [(usize, f64); 3],
}

fn potential(r: f64) -> f64 {
    body_potential([r, 0.0, 0.0], MASS, SOFTENING, SofteningKind::Compact.as_scalar())
}

impl Mesh {
    pub fn new(ni: usize, nj: usize) -> Self {
        assert_eq!(ni % COARSE_NI, 0, "{ni} radial cells: every grid nests the {COARSE_NI}-cell grid");
        let slope = (R_HI / R_LO).log10() / ni as f64;
        let r_face: Vec<f64> = (0..=ni).map(|ii| R_LO * 10f64.powf(ii as f64 * slope)).collect();
        let t_face: Vec<f64> = (0..=nj).map(|jj| jj as f64 * PI / nj as f64).collect();
        let r_c: Vec<f64> = (0..ni)
            .map(|ii| {
                let (a, b) = (r_face[ii], r_face[ii + 1]);
                (b.powi(4) - a.powi(4)) / 4.0 / ((b.powi(3) - a.powi(3)) / 3.0)
            })
            .collect();
        let t_c: Vec<f64> = (0..nj)
            .map(|jj| {
                let (a, b) = (t_face[jj], t_face[jj + 1]);
                ((b.sin() - b * b.cos()) - (a.sin() - a * a.cos())) / (a.cos() - b.cos())
            })
            .collect();
        let refine = ni / COARSE_NI;
        let shells = SHELL_RADII.map(|radius| {
            let coarse_slope = (R_HI / R_LO).log10() / COARSE_NI as f64;
            let coarse = ((radius / R_LO).log10() / coarse_slope).round() as usize;
            (coarse * refine, r_face[coarse * refine])
        });
        Mesh {
            ni,
            nj,
            phi_c: r_c.iter().map(|r| potential(*r)).collect(),
            phi_face: r_face.iter().map(|r| potential(*r)).collect(),
            r_face,
            t_face,
            r_c,
            t_c,
            shells,
        }
    }

    pub fn volume(&self, ii: usize, jj: usize) -> f64 {
        let (a, b) = (self.r_face[ii], self.r_face[ii + 1]);
        2.0 * PI * (b.powi(3) - a.powi(3)) / 3.0 * (self.t_face[jj].cos() - self.t_face[jj + 1].cos())
    }

    pub fn lever(&self, ii: usize, jj: usize) -> f64 {
        self.r_c[ii] * self.t_c[jj].sin()
    }

    fn radial_area(&self, ff: usize, jj: usize) -> f64 {
        2.0 * PI * self.r_face[ff] * self.r_face[ff] * (self.t_face[jj].cos() - self.t_face[jj + 1].cos())
    }

    fn polar_area(&self, ii: usize, ff: usize) -> f64 {
        PI * (self.r_face[ii + 1] * self.r_face[ii + 1] - self.r_face[ii] * self.r_face[ii]) * self.t_face[ff].sin()
    }
}

/// a location on the grid with the value it holds
#[derive(Clone, Copy, Debug)]
pub struct Extreme {
    pub value: f64,
    pub cell: [usize; 2],
}

impl Extreme {
    const HIGH: Extreme = Extreme { value: f64::INFINITY, cell: [0, 0] };
    const LOW: Extreme = Extreme { value: f64::NEG_INFINITY, cell: [0, 0] };

    fn lower(&mut self, value: f64, cell: [usize; 2]) {
        if value < self.value {
            *self = Extreme { value, cell };
        }
    }

    fn raise(&mut self, value: f64, cell: [usize; 2]) {
        if value > self.value {
            *self = Extreme { value, cell };
        }
    }

    fn json(&self, mesh: &Mesh) -> String {
        format!(
            "{{\"value\":{:?},\"i\":{},\"j\":{},\"r\":{:?},\"theta\":{:?}}}",
            self.value, self.cell[0], self.cell[1], mesh.r_c[self.cell[0]], mesh.t_c[self.cell[1]]
        )
    }
}

/// the minima over one state, overall and by side of the split radius
#[derive(Clone, Copy, Debug)]
pub struct Extremes {
    pub rho_min: Extreme,
    pub rho_max: Extreme,
    pub pressure_min: [Extreme; 2],
    pub specific_min: [Extreme; 2],
    pub speed_max: Extreme,
    pub mach_max: Extreme,
    /// the first cell holding a non-finite value or a non-positive density or internal energy
    pub rejected: Option<[usize; 2]>,
}

impl Extremes {
    fn listed(&self) -> [Extreme; WINDOW_EXTREMES] {
        [self.rho_min, self.rho_max, self.pressure_min[0], self.pressure_min[1], self.specific_min[0], self.specific_min[1], self.speed_max, self.mach_max]
    }

    fn from_listed(e: [Extreme; WINDOW_EXTREMES]) -> Self {
        Extremes {
            rho_min: e[0],
            rho_max: e[1],
            pressure_min: [e[2], e[3]],
            specific_min: [e[4], e[5]],
            speed_max: e[6],
            mach_max: e[7],
            rejected: None,
        }
    }

    fn empty() -> Self {
        Extremes {
            rho_min: Extreme::HIGH,
            rho_max: Extreme::LOW,
            pressure_min: [Extreme::HIGH; 2],
            specific_min: [Extreme::HIGH; 2],
            speed_max: Extreme::LOW,
            mach_max: Extreme::LOW,
            rejected: None,
        }
    }

    fn merge(&mut self, other: &Extremes) {
        self.rho_min.lower(other.rho_min.value, other.rho_min.cell);
        self.rho_max.raise(other.rho_max.value, other.rho_max.cell);
        for side in 0..2 {
            self.pressure_min[side].lower(other.pressure_min[side].value, other.pressure_min[side].cell);
            self.specific_min[side].lower(other.specific_min[side].value, other.specific_min[side].cell);
        }
        self.speed_max.raise(other.speed_max.value, other.speed_max.cell);
        self.mach_max.raise(other.mach_max.value, other.mach_max.cell);
        self.rejected = self.rejected.or(other.rejected);
    }

    fn json(&self, mesh: &Mesh) -> String {
        format!(
            "{{\"rho_min\":{},\"rho_max\":{},\"pressure_min_inner\":{},\"pressure_min_outer\":{},\"specific_internal_min_inner\":{},\"specific_internal_min_outer\":{},\"speed_max\":{},\"mach_max\":{}}}",
            self.rho_min.json(mesh),
            self.rho_max.json(mesh),
            self.pressure_min[0].json(mesh),
            self.pressure_min[1].json(mesh),
            self.specific_min[0].json(mesh),
            self.specific_min[1].json(mesh),
            self.speed_max.json(mesh),
            self.mach_max.json(mesh),
        )
    }
}

/// scans one conserved state's interior
pub fn extremes_of(fields: &Fields, mesh: &Mesh) -> Extremes {
    let c = &fields.cons;
    let (den, m0, m1, m2, nrg) = (c.den.view(), c.mom[0].view(), c.mom[1].view(), c.mom[2].view(), c.nrg_field().expect("cons.nrg").view());
    let mut out = Extremes::empty();
    for jj in 0..mesh.nj {
        for ii in 0..mesh.ni {
            let at = [ii as isize, jj as isize];
            let (rho, m, e) = (*den.at(at), [*m0.at(at), *m1.at(at), *m2.at(at)], *nrg.at(at));
            let momentum_sq = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
            let internal = e - 0.5 * momentum_sq / rho;
            let specific = internal / rho;
            let cell = [ii, jj];
            if !(rho > 0.0) || !rho.is_finite() || !e.is_finite() || !momentum_sq.is_finite() || !(specific > 0.0) || !specific.is_finite() {
                out.rejected = out.rejected.or(Some(cell));
            }
            let side = usize::from(mesh.r_c[ii] >= SPLIT_RADIUS);
            let pressure = (GAMMA - 1.0) * internal;
            let speed = momentum_sq.sqrt() / rho;
            out.rho_min.lower(rho, cell);
            out.rho_max.raise(rho, cell);
            out.pressure_min[side].lower(pressure, cell);
            out.specific_min[side].lower(specific, cell);
            out.speed_max.raise(speed, cell);
            out.mach_max.raise(speed / (GAMMA * pressure / rho).sqrt(), cell);
        }
    }
    out
}

/// volume totals of a state: mass, angular momentum, E + rho Phi, then E, kinetic and internal energy
pub fn totals_of(fields: &Fields, mesh: &Mesh) -> [f64; 6] {
    let c = &fields.cons;
    let (den, m0, m1, m2, nrg) = (c.den.view(), c.mom[0].view(), c.mom[1].view(), c.mom[2].view(), c.nrg_field().expect("cons.nrg").view());
    let mut sum = [0.0; 6];
    for jj in 0..mesh.nj {
        for ii in 0..mesh.ni {
            let at = [ii as isize, jj as isize];
            let volume = mesh.volume(ii, jj);
            let (rho, e) = (*den.at(at), *nrg.at(at));
            let kinetic = 0.5 * (m0.at(at).powi(2) + m1.at(at).powi(2) + m2.at(at).powi(2)) / rho;
            sum[0] += rho * volume;
            sum[1] += *m2.at(at) * mesh.lever(ii, jj) * volume;
            sum[2] += (e + rho * mesh.phi_c[ii]) * volume;
            sum[3] += e * volume;
            sum[4] += kinetic * volume;
            sum[5] += (e - kinetic) * volume;
        }
    }
    sum
}

/// adds one stage's boundary and shell transport, weighted for its share of the step
fn accumulate(fields: &Fields, mesh: &Mesh, ledger: &mut Ledger, weight: f64) {
    let radial = &fields.flux[0];
    let polar = &fields.flux[1];
    let (rho, phi, nrg) = (radial.den.view(), radial.mom[2].view(), radial.nrg_field().expect("flux.nrg").view());
    for jj in 0..mesh.nj {
        let t_lever = mesh.t_c[jj].sin();
        let wall = |ff: usize, slot: &mut [f64; 3]| {
            let at = [ff as isize, jj as isize];
            let area = mesh.radial_area(ff, jj);
            let mass = *rho.at(at);
            slot[0] += weight * area * mass;
            slot[1] += weight * area * mesh.r_face[ff] * t_lever * *phi.at(at);
            slot[2] += weight * area * (*nrg.at(at) + mesh.phi_face[ff] * mass);
        };
        wall(0, &mut ledger.inner);
        wall(mesh.ni, &mut ledger.outer);
        let faces = [mesh.shells[0].0, mesh.shells[1].0, mesh.shells[2].0, 0, mesh.ni];
        for (slot, ff) in faces.iter().enumerate() {
            let rate = weight * mesh.radial_area(*ff, jj) * *rho.at([*ff as isize, jj as isize]);
            let row = &mut ledger.transport[slot];
            row[0] += rate;
            row[1] += (-rate).max(0.0);
            row[2] += rate.max(0.0);
        }
    }
    let (p_rho, p_phi, p_nrg) = (polar.den.view(), polar.mom[2].view(), polar.nrg_field().expect("flux.nrg").view());
    for ii in 0..mesh.ni {
        for (ff, sign) in [(0usize, -1.0), (mesh.nj, 1.0)] {
            let at = [ii as isize, ff as isize];
            let area = mesh.polar_area(ii, ff);
            let lever = mesh.r_c[ii] * mesh.t_face[ff].sin();
            ledger.polar[0] += sign * weight * area * *p_rho.at(at);
            ledger.polar[1] += sign * weight * area * lever * *p_phi.at(at);
            ledger.polar[2] += sign * weight * area * (*p_nrg.at(at) + mesh.phi_c[ii] * *p_rho.at(at));
        }
    }
}

/// the torus's grid, body and kernels; the gas comes from the checkpoint a run loads
pub fn build(ni: usize, nj: usize) -> (Sim, Kern) {
    let slope = (R_HI / R_LO).log10() / ni as f64;
    let dtheta = PI / nj as f64;
    let walls = [[BoundaryType::Outflow, BoundaryType::Outflow], [BoundaryType::Axis, BoundaryType::Axis]];
    let sim: Sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells([ni, nj])
        .origin([R_LO, 0.0])
        .spacing([(R_HI - R_LO) / ni as f64, dtheta])
        .coord_maps(Some([AxisMap::Log { start: R_LO, log_slope: slope }, AxisMap::Uniform { start: 0.0, dx: dtheta }]))
        .boundaries(Boundaries::per_axis(walls))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("the torus grid")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0; 3]), Pressure(1.0)))
        .build()
        .with_bodies(BodyCollection::new().add(
            Body::gravitational(0, Tensor::zeros(), Tensor::zeros(), MASS, 0.0, SOFTENING).with_softening_kind(SofteningKind::Compact),
        ));
    let kernels = Kern::new(GAMMA, CFL, &sim.geom.allocated).theta(THETA);
    (sim, kernels)
}

/// how a launch ended
#[derive(Clone, Debug, PartialEq)]
pub enum Ending {
    Complete,
    /// a budget ended the launch: the run is incomplete, and its restart file continues it
    Budget(&'static str),
    /// a stage output left the admissible set, or the timestep was not positive
    Failed(String),
    /// the launch's identity disagrees with what it would continue; no step ran and no file changed
    Refused(String),
}

impl Ending {
    pub fn exit_code(&self) -> i32 {
        match self {
            Ending::Complete => EXIT_COMPLETE,
            Ending::Budget(_) => EXIT_BUDGET,
            Ending::Failed(_) => EXIT_FAILED,
            Ending::Refused(_) => EXIT_REFUSED,
        }
    }
}

pub struct Launch<'a> {
    pub out: &'a Path,
    pub bounds: &'a Bounds,
    pub identity: &'a Identity,
}

/// what a restart file must agree with to be continued
#[derive(Clone, Debug, PartialEq)]
pub struct Identity {
    pub method: String,
    pub revision: String,
    pub binary_sha256: String,
    pub sample_dt: f64,
}

impl Identity {
    fn into_metadata(&self, extras: symbi_io::Metadata) -> symbi_io::Metadata {
        extras
            .with("torus_method", self.method.as_str())
            .with("torus_revision", self.revision.as_str())
            .with("torus_binary_sha256", self.binary_sha256.as_str())
            .with("torus_sample_dt", format!("{:016x}", self.sample_dt.to_bits()))
    }

    /// each field a restart file records differently from this launch, named with both values
    fn disagreements(&self, restart: &Path) -> Vec<String> {
        let recorded = |key: &str| read_text_attr(restart, key).unwrap_or_else(|| "absent".to_string());
        let sample_dt = format!("{:016x}", self.sample_dt.to_bits());
        [
            ("torus_method", self.method.as_str()),
            ("torus_revision", self.revision.as_str()),
            ("torus_binary_sha256", self.binary_sha256.as_str()),
            ("torus_sample_dt", sample_dt.as_str()),
        ]
        .into_iter()
        .filter_map(|(key, launch)| {
            let held = recorded(key);
            (held != launch).then(|| format!("{key}: the restart file records {held:?}, this launch is {launch:?}"))
        })
        .collect()
    }
}

/// the scheme this driver runs on this grid, read from the kernels' selection
pub fn method_of(sim: &Sim, kernels: &Kern, ni: usize, nj: usize) -> String {
    let bodies = sim.immersed.as_ref().expect("the point mass").bodies.bodies();
    let face_work = !kernels.cell_momentum_gravity_work && !kernels.azimuthal_source_control && admits_face_gravity_work(sim.geom.coords, 3, bodies);
    format!(
        "regime=newtonian-adiabatic gamma={GAMMA:?} solver=hlle reconstruction=plm theta={THETA:?} timestepping=rk2 cfl={CFL:?} azimuthal={} gravity_energy_work={} body=point-mass mass={MASS:?} softening=compact softening_length={SOFTENING:?} grid={ni}x{nj} radial=log[{R_LO:?},{R_HI:?}] polar=uniform[0,pi] walls=outflow,outflow,axis,axis",
        if kernels.azimuthal_source_control { "source-form" } else { "torque-form" },
        if face_work { "face-mass-flux" } else { "cell-momentum" },
    )
}

fn read_text_attr(path: &Path, key: &str) -> Option<String> {
    symbi_io::read_attrs(path, "metadata").ok()?.into_iter().find(|(k, _)| k == key).and_then(|(_, v)| match v {
        symbi_io::Attr::Str(text) => Some(text),
        _ => None,
    })
}

/// the nearest directory at or above `path` holding the outside-scope marker
pub fn outside_scope_ancestor(path: &Path) -> Option<PathBuf> {
    let absolute = std::fs::canonicalize(path).ok()?;
    absolute.ancestors().find(|dir| dir.join(OUTSIDE_SCOPE_MARKER).exists()).map(Path::to_path_buf)
}

fn write_restart(sim: &Sim, out: &Path, ledger: &Ledger, identity: &Identity, name: &str) {
    let temporary = out.join(format!("{name}.partial"));
    let values: Vec<String> = ledger.to_values().iter().map(|v| format!("{:016x}", v.to_bits())).collect();
    let extras = identity.into_metadata(symbi_io::Metadata::new().with("torus_ledger", values.join(" ")));
    write_checkpoint(sim, temporary.to_str().expect("a utf-8 path"), &extras).expect("the restart checkpoint");
    std::fs::rename(&temporary, out.join(name)).expect("publishing the restart checkpoint");
}

fn read_ledger(path: &Path) -> Ledger {
    let attrs = symbi_io::read_attrs(path, "metadata").expect("the restart file's metadata");
    let text = attrs
        .iter()
        .find(|(k, _)| k == "torus_ledger")
        .map(|(_, v)| match v {
            symbi_io::Attr::Str(text) => text.clone(),
            other => panic!("{path:?}: torus_ledger is a {}", other.type_name()),
        })
        .unwrap_or_else(|| panic!("{path:?} holds no torus_ledger metadata; it is not this driver's restart file"));
    let values: Vec<f64> = text
        .split_whitespace()
        .map(|word| f64::from_bits(u64::from_str_radix(word, 16).expect("a ledger word")))
        .collect();
    Ledger::from_values(&values)
}

/// the rows file holding exactly the rows the restart file counted
fn truncated_rows(path: &Path, keep: u64) -> std::io::Result<()> {
    let text = std::fs::read_to_string(path)?;
    let kept: Vec<&str> = text.lines().take(keep as usize).collect();
    assert_eq!(kept.len() as u64, keep, "{path:?} holds fewer rows than the restart file counts");
    let temporary = path.with_extension("partial");
    let mut body = kept.join("\n");
    if keep > 0 {
        body.push('\n');
    }
    std::fs::write(&temporary, body)?;
    std::fs::rename(temporary, path)
}

fn write_status(launch: &Launch, sim: &Sim, ending: &Ending, wall: f64, detail: &str) {
    let state = match ending {
        Ending::Complete => "complete".to_string(),
        Ending::Budget(which) => format!("incomplete: {which} budget"),
        Ending::Failed(reason) => format!("failed: {reason}"),
        Ending::Refused(reason) => format!("refused: {reason}"),
    };
    let text = format!(
        "{{\"state\":{state:?},\"time\":{:?},\"step\":{},\"target_time\":{:?},\"max_steps\":{},\"max_wall_seconds\":{:?},\"launch_wall_seconds\":{wall:?},\"revision\":{:?},\"binary_sha256\":{:?},\"method\":{:?},\"detail\":{detail:?}}}\n",
        sim.time, sim.iteration, launch.bounds.t_end, launch.bounds.max_steps, launch.bounds.max_wall_seconds, launch.identity.revision, launch.identity.binary_sha256, launch.identity.method
    );
    let temporary = launch.out.join("status.partial");
    std::fs::write(&temporary, text).expect("the status file");
    std::fs::rename(temporary, launch.out.join("status.json")).expect("publishing the status file");
}

/// each total's departure from its start and the transport out of the domain, over its start
pub fn closure_of(fields: &Fields, mesh: &Mesh, ledger: &Ledger) -> [f64; 3] {
    let totals = totals_of(fields, mesh);
    std::array::from_fn(|kk| (totals[kk] - ledger.start[kk] + ledger.outer[kk] - ledger.inner[kk] + ledger.polar[kk]) / ledger.start[kk].abs())
}

fn row_json(sim: &Sim, mesh: &Mesh, ledger: &Ledger, wall: f64) -> String {
    let totals = totals_of(&sim.fields, mesh);
    let closure = closure_of(&sim.fields, mesh, ledger);
    let at = extremes_of(&sim.fields, mesh);
    let sample = ledger.sample;
    let surfaces: Vec<String> = (0..5)
        .map(|slot| {
            let (name, radius) = match slot {
                0..=2 => (format!("shell_{}", slot), mesh.shells[slot].1),
                3 => ("inner_wall".to_string(), mesh.r_face[0]),
                _ => ("outer_wall".to_string(), mesh.r_face[mesh.ni]),
            };
            let t = ledger.transport[slot];
            format!("{name:?}:{{\"r\":{radius:?},\"net_outward\":{:?},\"gross_inward\":{:?},\"gross_outward\":{:?}}}", t[0], t[1], t[2])
        })
        .collect();
    format!(
        "{{\"sample\":{sample},\"t\":{:?},\"step\":{},\"dt\":{:?},\"wall\":{wall:?},\"state\":{},\"every_stage\":{},\"mass\":{:?},\"angular_momentum\":{:?},\"total_plus_potential\":{:?},\"total\":{:?},\"kinetic\":{:?},\"internal\":{:?},\"outflow\":{{\"inner\":{:?},\"outer\":{:?},\"polar\":{:?}}},\"closure\":{{\"mass\":{:?},\"angular_momentum\":{:?},\"total_plus_potential\":{:?}}},\"mass_transport\":{{{}}}}}",
        sim.time,
        sim.iteration,
        sim.dt,
        at.json(mesh),
        ledger.window.json(mesh),
        totals[0],
        totals[1],
        totals[2],
        totals[3],
        totals[4],
        totals[5],
        ledger.inner,
        ledger.outer,
        ledger.polar,
        closure[0],
        closure[1],
        closure[2],
        surfaces.join(",")
    )
}

/// runs the production stage from the state `sim` holds until an ending, appending rows
pub fn march(sim: &mut Sim, kernels: &Kern, mesh: &Mesh, ledger: &mut Ledger, launch: &Launch) -> Ending {
    let started = Instant::now();
    let bounds = launch.bounds;
    let rows_path = launch.out.join("rows.jsonl");
    let mut rows = std::fs::OpenOptions::new().create(true).append(true).open(&rows_path).expect("the rows file");
    let ending = loop {
        if sim.time >= bounds.t_end {
            break Ending::Complete;
        }
        if sim.iteration >= bounds.max_steps {
            break Ending::Budget("step");
        }
        if started.elapsed().as_secs_f64() >= bounds.max_wall_seconds {
            break Ending::Budget("wall");
        }
        if !kernels.cell_momentum_gravity_work && !admits_face_gravity_work(sim.geom.coords, 3, sim.immersed.as_ref().expect("the point mass").bodies.bodies()) {
            break Ending::Failed("the body left the face work's admission".to_string());
        }
        let next_sample = ledger.sample + 1;
        let target = (next_sample as f64 * bounds.sample_dt).min(bounds.t_end);
        let cfl_dt = KernelSet::cfl(kernels, &*sim);
        if !(cfl_dt > 0.0) {
            break Ending::Failed(format!("the timestep is {cfl_dt} at t = {:?}", sim.time));
        }
        let landing = cfl_dt >= target - sim.time;
        let dt = if landing { target - sim.time } else { cfl_dt };
        sim.dt = dt;
        let stages = sim.timestepping.stages();
        assert_eq!(stages.len(), 2, "the ledger weights the two stages of rk2 by one half each");
        if needs_step_snapshot(stages) {
            KernelSet::snapshot(kernels, &*sim);
        }
        let mut failure = None;
        for spec in stage_schedule(stages) {
            let store_ref: &FieldStore<2, 3, HostMemory, f64> = &*sim;
            let mut output = None;
            let outcome = fold_stage(
                store_ref,
                kernels,
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
                &mut |point| match point {
                    HookPoint::AfterFlux => accumulate(&store_ref.fields, mesh, ledger, 0.5 * dt),
                    HookPoint::BeforeC2p => output = Some(extremes_of(&store_ref.fields, mesh)),
                    _ => {}
                },
            );
            if outcome != StageOutcome::Accepted {
                failure = Some(format!("stage {} of step {} asked for a retry, which this fixture has no path for", spec.index, sim.iteration));
                break;
            }
            let output = output.expect("the stage reached its recovery");
            ledger.window.merge(&output);
            if let Some(cell) = output.rejected {
                failure = Some(format!(
                    "stage {} of step {} at t = {:?} left the admissible set at cell [{}, {}], r = {:?}, theta = {:?}",
                    spec.index, sim.iteration, sim.time, cell[0], cell[1], mesh.r_c[cell[0]], mesh.t_c[cell[1]]
                ));
                break;
            }
        }
        if let Some(reason) = failure {
            break Ending::Failed(reason);
        }
        sim.time = if landing { target } else { sim.time + dt };
        sim.iteration += 1;
        if landing {
            ledger.sample = next_sample;
            ledger.rows += 1;
            let line = row_json(sim, mesh, ledger, started.elapsed().as_secs_f64());
            writeln!(rows, "{line}").expect("a row");
            rows.flush().expect("flushing the rows");
            ledger.window = Extremes::empty();
            if next_sample % bounds.snapshot_every == 0 {
                write_restart(sim, launch.out, ledger, launch.identity, &format!("snapshot_{next_sample:06}.h5"));
                write_restart(sim, launch.out, ledger, launch.identity, "restart.h5");
            }
        }
    };
    let wall = started.elapsed().as_secs_f64();
    match &ending {
        Ending::Failed(reason) => {
            write_restart(sim, launch.out, ledger, launch.identity, "failed_state.h5");
            write_status(launch, sim, &ending, wall, reason);
        }
        Ending::Refused(_) => unreachable!("a refusal ends a launch before its march"),
        Ending::Budget(_) | Ending::Complete => {
            write_restart(sim, launch.out, ledger, launch.identity, "restart.h5");
            write_status(launch, sim, &ending, wall, "");
        }
    }
    ending
}

/// the run's identity and its seed's own build record, written once at the first launch
fn write_provenance(out: &Path, identity: &Identity, seed: &Path) {
    let seed_attr = |key: &str| read_text_attr(seed, key).unwrap_or_else(|| "absent".to_string());
    let text = format!(
        "{{\"method\":{:?},\"revision\":{:?},\"binary_sha256\":{:?},\"sample_dt\":{:?},\"seed\":{:?},\"seed_build_git_sha\":{:?},\"seed_build_source_id\":{:?},\"seed_config_sha256\":{:?}}}\n",
        identity.method,
        identity.revision,
        identity.binary_sha256,
        identity.sample_dt,
        seed.display().to_string(),
        seed_attr("build_git_sha"),
        seed_attr("build_source_id"),
        seed_attr("config_sha256"),
    );
    let temporary = out.join("provenance.partial");
    std::fs::write(&temporary, text).expect("the provenance file");
    std::fs::rename(temporary, out.join("provenance.json")).expect("publishing the provenance file");
}

#[derive(Clone, Debug)]
pub struct Args {
    pub seed: Option<PathBuf>,
    pub out: PathBuf,
    pub ni: usize,
    pub nj: usize,
    pub bounds: Bounds,
    pub revision: String,
    pub binary_sha256: String,
    pub resume: bool,
    /// the cell momentum's gravity work; the gates set it to reproduce the failing update
    pub cell_work: bool,
}

/// the run's state from its seed or its restart file, then the march
pub fn run(args: &Args) -> Ending {
    std::fs::create_dir_all(&args.out).expect("the output directory");
    let (mut sim, mut kernels) = build(args.ni, args.nj);
    kernels.cell_momentum_gravity_work = args.cell_work;
    let mesh = Mesh::new(args.ni, args.nj);
    let rows_path = args.out.join("rows.jsonl");
    let identity = Identity {
        method: method_of(&sim, &kernels, args.ni, args.nj),
        revision: args.revision.clone(),
        binary_sha256: args.binary_sha256.clone(),
        sample_dt: args.bounds.sample_dt,
    };
    let mut ledger = if args.resume {
        let restart = args.out.join("restart.h5");
        let disagreements = identity.disagreements(&restart);
        if !disagreements.is_empty() {
            return Ending::Refused(format!("{restart:?} belongs to another run. {}", disagreements.join("; ")));
        }
        load_checkpoint(&mut sim, restart.to_str().expect("a utf-8 path")).expect("the restart file");
        let ledger = read_ledger(&restart);
        truncated_rows(&rows_path, ledger.rows).expect("the rows file");
        ledger
    } else {
        assert!(!rows_path.exists(), "{rows_path:?} exists: resume the run with --resume or choose another directory");
        let seed = args.seed.as_ref().expect("a first launch reads --seed");
        if let Some(marked) = outside_scope_ancestor(seed) {
            return Ending::Refused(format!("the seed {seed:?} lies under {marked:?}, marked outside the authorized execution scope"));
        }
        load_checkpoint(&mut sim, seed.to_str().expect("a utf-8 path")).expect("the seed checkpoint");
        write_provenance(&args.out, &identity, seed);
        Ledger::default()
    };
    KernelSet::c2p(&kernels, &sim);
    KernelSet::ghost_fill(&kernels, &sim);
    if !args.resume {
        let totals = totals_of(&sim.fields, &mesh);
        ledger.start = [totals[0], totals[1], totals[2]];
        ledger.sample = (sim.time / args.bounds.sample_dt).floor() as u64;
    }
    let launch = Launch { out: &args.out, bounds: &args.bounds, identity: &identity };
    march(&mut sim, &kernels, &mesh, &mut ledger, &launch)
}

fn parse_args() -> Args {
    let words: Vec<String> = std::env::args().skip(1).collect();
    let value = |name: &str| words.iter().position(|w| w == name).map(|at| words.get(at + 1).unwrap_or_else(|| panic!("{name} needs a value")).clone());
    let need = |name: &str| value(name).unwrap_or_else(|| panic!("{name} is required"));
    let number = |name: &str| need(name).parse::<f64>().unwrap_or_else(|e| panic!("{name}: {e}"));
    let count = |name: &str| need(name).parse::<u64>().unwrap_or_else(|e| panic!("{name}: {e}"));
    Args {
        seed: value("--seed").map(PathBuf::from),
        out: PathBuf::from(need("--out")),
        ni: count("--ni") as usize,
        nj: count("--nj") as usize,
        bounds: Bounds {
            t_end: number("--t-end"),
            sample_dt: number("--sample-dt"),
            snapshot_every: count("--snapshot-every"),
            max_steps: count("--max-steps"),
            max_wall_seconds: number("--max-wall-seconds"),
        },
        revision: need("--revision"),
        binary_sha256: need("--binary-sha256"),
        resume: words.iter().any(|w| w == "--resume"),
        cell_work: false,
    }
}

// runs on a thread with the frontend's stack: the kernels' recursion depth matches its run thread
fn main() {
    let args = parse_args();
    let ending = std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(move || run(&args))
        .expect("the run thread")
        .join()
        .expect("the run");
    println!("{ending:?}");
    std::process::exit(ending.exit_code());
}
