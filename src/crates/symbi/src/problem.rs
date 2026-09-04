// =============================================================================
// problem.rs
//
// the rust scientific surface, at its smallest: a `Problem` states a regime,
// a geometry, an initial gas state, its boundaries, and its source
// contributions in physics vocabulary alone, and `realize` turns it into a
// seeded engine state plus the kernel set that steps it. the surface names
// gas, gravity and walls; the compiler's graphs, kernels and manifests stay
// behind it.
//
// a source contribution reaches the engine through the source admission door
// every configured source passes: it is lowered to a declared `SourceSpec`
// and `AdmittedSources::admit_specs` holds the built program to that
// declaration on the regime before any step runs. a rejected contribution
// surfaces as `ProblemError::SourceRejected` at realization.
//
// the slice covers a one-dimensional cartesian newtonian gas. every
// realized problem runs on the host through the same `evolve` loop the
// configured path drives.
//
// usage:
//   struct Sod;
//   impl Problem for Sod { ... }
//   let mut run = realize(&Sod)?;
//   run.advance_to(0.1)?;
//   let gas = run.state();
// =============================================================================

use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::NEWTONIAN_SPEC;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ir::SourceProgram;
use symbi_source_compile::AdmittedSources;
use symbi_source_compile::source_spec::source_params::ReadFamily;
use symbi_source_compile::source_spec::user_params::{UserParam, UserVocabulary};
use symbi_source_compile::source_spec::{SourceSpec, user_defined_source};
use symbi_xpu::{CpuSpace, HostMemory};

use crate::regimes::regime_substrate::SimSubstrate;
use crate::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use crate::sim::evolve::evolve;
use crate::sim::state::{Boundaries as EngineBoundaries, BoundaryType, ConfigError, SimState};

/// the fluid regime and its closure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Regime {
    /// a newtonian ideal gas with adiabatic index `gamma`.
    Newtonian { gamma: f64 },
}

impl Regime {
    pub fn newtonian(gamma: f64) -> Self {
        Self::Newtonian { gamma }
    }
}

/// a one-dimensional cartesian line of `cells` uniform cells spanning
/// `[lo, hi]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    lo: f64,
    hi: f64,
    cells: usize,
}

impl Geometry {
    pub fn cartesian_line(lo: f64, hi: f64, cells: usize) -> Self {
        Self { lo, hi, cells }
    }

    /// the uniform cell width.
    pub fn spacing(&self) -> f64 {
        (self.hi - self.lo) / self.cells as f64
    }

    pub fn cells(&self) -> usize {
        self.cells
    }
}

/// the condition on one domain face.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    /// zero-gradient extrapolation: waves leave the domain.
    Outflow,
    /// a solid wall: the normal velocity mirrors.
    Reflecting,
    /// the two faces of the axis are identified.
    Periodic,
}

/// the conditions on the low and high faces of the line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Boundaries {
    pub lo: Boundary,
    pub hi: Boundary,
}

impl Boundaries {
    pub fn new(lo: Boundary, hi: Boundary) -> Self {
        Self { lo, hi }
    }

    pub fn uniform(both: Boundary) -> Self {
        Self::new(both, both)
    }
}

/// the primitive state of the gas at one point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GasState {
    pub density: f64,
    pub velocity: f64,
    pub pressure: f64,
}

impl GasState {
    /// gas at rest.
    pub fn at_rest(density: f64, pressure: f64) -> Self {
        Self {
            density,
            velocity: 0.0,
            pressure,
        }
    }
}

/// the cell the initial-state generator samples: its centroid and width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cell {
    pub center: f64,
    pub width: f64,
}

/// a source contribution to the gas equations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Source {
    /// a constant acceleration along the line: momentum gains
    /// `rho * acceleration`, energy gains `rho * v * acceleration`.
    UniformGravity { acceleration: f64 },
}

/// a scientific problem: what the gas is, where it lives, how it starts,
/// what bounds it, and what acts on it.
pub trait Problem {
    fn regime(&self) -> Regime;
    fn geometry(&self) -> Geometry;
    fn boundaries(&self) -> Boundaries;
    fn initial_state(&self, cell: Cell) -> GasState;
    /// the source contributions; empty for a free gas.
    fn sources(&self) -> Vec<Source> {
        Vec::new()
    }
}

/// the failures realization surfaces ahead of any step.
#[derive(Debug)]
pub enum ProblemError {
    /// the grid the geometry describes cannot be allocated.
    Grid(ConfigError),
    /// a source contribution failed the admission door; the verdict names
    /// the offending leaf or target.
    SourceRejected(String),
    /// the slice carries at most one source contribution.
    TooManySources(usize),
    /// the engine failed while stepping.
    Engine(symbi_xpu::XpuError),
}

impl std::fmt::Display for ProblemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Grid(e) => write!(f, "grid: {e:?}"),
            Self::SourceRejected(verdict) => write!(f, "source rejected: {verdict}"),
            Self::TooManySources(n) => write!(f, "{n} sources; the slice admits at most one"),
            Self::Engine(e) => write!(f, "engine: {e:?}"),
        }
    }
}

impl std::error::Error for ProblemError {}

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Kernels = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;

/// a problem realized on the engine: the seeded state and the kernel set
/// that steps it.
pub struct Realized {
    sim: Sim,
    kernels: Kernels,
}

impl Realized {
    /// march the gas to time `t_final`.
    pub fn advance_to(&mut self, t_final: f64) -> Result<(), ProblemError> {
        evolve(&mut self.sim, &self.kernels, t_final).map_err(ProblemError::Engine)
    }

    pub fn time(&self) -> f64 {
        self.sim.time
    }

    pub fn steps(&self) -> u64 {
        self.sim.iteration
    }

    /// the primitive state of every interior cell, in position order.
    pub fn state(&self) -> Vec<GasState> {
        let pre = self
            .sim
            .fields
            .prim
            .pre_field()
            .expect("newtonian gas carries pressure");
        self.sim
            .geom
            .interior
            .iter()
            .map(|c| GasState {
                density: *self.sim.fields.prim.rho.view().at(c),
                velocity: *self.sim.fields.prim.vel[0].view().at(c),
                pressure: *pre.view().at(c),
            })
            .collect()
    }

    /// the total mass on the interior.
    pub fn mass(&self) -> f64 {
        let dx = self.sim.geom.cell_width([0], 0);
        self.sim
            .geom
            .interior
            .iter()
            .map(|c| *self.sim.fields.cons.den.view().at(c))
            .sum::<f64>()
            * dx
    }

    /// the total momentum along the line on the interior.
    pub fn momentum(&self) -> f64 {
        let dx = self.sim.geom.cell_width([0], 0);
        self.sim
            .geom
            .interior
            .iter()
            .map(|c| *self.sim.fields.cons.mom[0].view().at(c))
            .sum::<f64>()
            * dx
    }
}

/// seed the engine from a problem and admit its source contributions.
pub fn realize(problem: &impl Problem) -> Result<Realized, ProblemError> {
    let Regime::Newtonian { gamma } = problem.regime();
    let geometry = problem.geometry();
    let width = geometry.spacing();
    let sim = Sim::build(Newtonian, IdealGas { gamma }, Cartesian)
        .cells([geometry.cells])
        .bounds([geometry.lo], [geometry.hi])
        .boundaries(engine_boundaries(problem.boundaries()))
        .allocate()
        .map_err(ProblemError::Grid)?
        .set_initial(|x| {
            let gas = problem.initial_state(Cell {
                center: x[0],
                width,
            });
            Prim::adiabatic(
                Density(gas.density),
                Tensor::new([gas.velocity]),
                Pressure(gas.pressure),
            )
        })
        .build();

    let sources = problem.sources();
    let kernels = match sources.as_slice() {
        [] => sim.substrate(),
        [Source::UniformGravity { acceleration }] => {
            let specs = uniform_gravity_specs();
            let refs: Vec<&SourceSpec> = specs.iter().collect();
            let admitted = AdmittedSources::admit_specs(&refs, &NEWTONIAN_SPEC, 1)
                .map_err(|verdict| ProblemError::SourceRejected(format!("{verdict:?}")))?;
            sim.substrate()
                .with_runtime_source(admitted, vec![*acceleration])
        }
        more => return Err(ProblemError::TooManySources(more.len())),
    };
    Ok(Realized { sim, kernels })
}

fn engine_boundaries(b: Boundaries) -> EngineBoundaries<1> {
    let face = |side: Boundary| match side {
        Boundary::Outflow => BoundaryType::Outflow,
        Boundary::Reflecting => BoundaryType::Reflect,
        Boundary::Periodic => BoundaryType::Periodic,
    };
    EngineBoundaries::per_axis([[face(b.lo), face(b.hi)]])
}

/// the runtime knob the acceleration binds to: the first numbered scalar of
/// the runtime source's parameter vector.
const ACCELERATION_KNOB: &str = "p0";

/// `S_mom = rho * g` along the line observes the density alone.
const GRAVITY_MOMENTUM: UserVocabulary = UserVocabulary::Families {
    reads: &[ReadFamily::Rho],
    parameters: &[UserParam::Scalar(ACCELERATION_KNOB)],
};

/// `S_nrg = rho * v * g` observes the density and the velocity.
const GRAVITY_ENERGY: UserVocabulary = UserVocabulary::Families {
    reads: &[ReadFamily::Rho, ReadFamily::Vel],
    parameters: &[UserParam::Scalar(ACCELERATION_KNOB)],
};

/// the momentum and energy contributions of a uniform acceleration along
/// axis 0, each carrying the declaration the admission door holds it to.
fn uniform_gravity_specs() -> [SourceSpec; 2] {
    [
        user_defined_source("mom", GRAVITY_MOMENTUM, gravity_momentum_program),
        user_defined_source("nrg", GRAVITY_ENERGY, gravity_energy_program),
    ]
}

fn gravity_momentum_program(d: usize) -> SourceProgram {
    SourceProgram::trace(|cx| {
        let rho = cx.scalar("rho");
        let g = cx.scalar(ACCELERATION_KNOB);
        (0..d)
            .map(|k| if k == 0 { rho * g } else { cx.lit(0.0) })
            .collect()
    })
}

fn gravity_energy_program(_d: usize) -> SourceProgram {
    SourceProgram::trace(|cx| {
        let rho = cx.scalar("rho");
        let vel = cx.scalar("vel_0");
        let g = cx.scalar(ACCELERATION_KNOB);
        vec![rho * vel * g]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_source_compile::simulation_laws::CompositionError;
    use symbi_source_compile::source_spec::source_params::Read;

    // the gravity contributions pass the admission door on the newtonian
    // regime, and a program observing a leaf outside the declaration is
    // refused by the same door: the rust surface is held to the gate, so a
    // green admission here is evidence rather than a formality.
    #[test]
    fn uniform_gravity_is_admitted_and_an_undeclared_read_is_refused() {
        let specs = uniform_gravity_specs();
        let refs: Vec<&SourceSpec> = specs.iter().collect();
        let admitted =
            AdmittedSources::admit_specs(&refs, &NEWTONIAN_SPEC, 1).expect("declared gravity");
        let targets: Vec<&str> = admitted.pairs().iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(targets, ["mom", "nrg"]);

        fn reads_pressure(_d: usize) -> SourceProgram {
            SourceProgram::trace(|cx| vec![cx.scalar("rho") * cx.scalar("pre")])
        }
        let off_declaration = user_defined_source("mom", GRAVITY_MOMENTUM, reads_pressure);
        match AdmittedSources::admit_specs(&[&off_declaration], &NEWTONIAN_SPEC, 1) {
            Err(CompositionError::UndeclaredRead { read, target, .. }) => {
                assert_eq!(read, Read::Pre);
                assert_eq!(target, "mom");
            }
            other => panic!("expected UndeclaredRead, got {other:?}"),
        }
    }
}
