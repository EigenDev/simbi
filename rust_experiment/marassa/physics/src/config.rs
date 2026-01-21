// =============================================================================
// config.rs
//
// compile-time physics configuration and dispatch system.
// uses zero-sized types (zst) and const generics to select physics regime,
// dimensionality, solver, and reconstruction at compile time with zero overhead.
//
// design philosophy:
//   - all configuration known at compile time
//   - zero runtime cost (monomorphization eliminates abstractions)
//   - type-safe: invalid combinations rejected by compiler
//   - field layout determined by configuration type
//
// inspired by game engine ecs systems and simbi's c++ dispatch.
//
// usage:
//   type MyConfig = PhysicsConfig<Newtonian, Dim1, HlleSolver, PlmReconstruction>;
//   type StateFields = <MyConfig as Configuration>::StateFields;
//
//   let solver = Solver::<MyConfig>::new(device, domain, gamma);
//   solver.step(dt);
// =============================================================================

use core::marker::PhantomData;

// =============================================================================
// regime markers (zst)
// =============================================================================

/// newtonian hydrodynamics
pub struct Newtonian;

/// special relativistic hydrodynamics
pub struct Srhd;

/// magnetohydrodynamics
pub struct Mhd;

/// relativistic magnetohydrodynamics
pub struct Rmhd;

/// trait bound for valid physics regimes
pub trait Regime {
    /// human-readable name
    const NAME: &'static str;

    /// number of conserved variables per cell (excluding scalars)
    const NCONS: usize;

    /// whether this regime includes magnetic fields
    const HAS_MAGNETIC: bool = false;
}

impl Regime for Newtonian {
    const NAME: &'static str = "Newtonian";
    const NCONS: usize = 3; // rho, mom, energy
}

impl Regime for Srhd {
    const NAME: &'static str = "SRHD";
    const NCONS: usize = 3; // D, S, tau
}

impl Regime for Mhd {
    const NAME: &'static str = "MHD";
    const NCONS: usize = 3; // rho, mom, energy (+ magnetic in state)
    const HAS_MAGNETIC: bool = true;
}

impl Regime for Rmhd {
    const NAME: &'static str = "RMHD";
    const NCONS: usize = 3; // D, S, tau (+ magnetic in state)
    const HAS_MAGNETIC: bool = true;
}

// =============================================================================
// dimensionality markers (zst)
// =============================================================================

/// 1d simulation
pub struct Dim1;

/// 2d simulation
pub struct Dim2;

/// 3d simulation
pub struct Dim3;

/// trait bound for valid dimensionality
pub trait Dimensionality {
    const NDIM: usize;
    const NAME: &'static str;
}

impl Dimensionality for Dim1 {
    const NDIM: usize = 1;
    const NAME: &'static str = "1D";
}

impl Dimensionality for Dim2 {
    const NDIM: usize = 2;
    const NAME: &'static str = "2D";
}

impl Dimensionality for Dim3 {
    const NDIM: usize = 3;
    const NAME: &'static str = "3D";
}

// =============================================================================
// solver markers (zst)
// =============================================================================

/// hlle approximate riemann solver
pub struct HlleSolver;

/// hllc solver (contact wave resolution)
pub struct HllcSolver;

/// exact riemann solver
pub struct ExactSolver;

/// trait bound for valid solvers
pub trait RiemannSolver {
    const NAME: &'static str;
    const DIFFUSIVE: bool;
}

impl RiemannSolver for HlleSolver {
    const NAME: &'static str = "HLLE";
    const DIFFUSIVE: bool = true;
}

impl RiemannSolver for HllcSolver {
    const NAME: &'static str = "HLLC";
    const DIFFUSIVE: bool = false;
}

impl RiemannSolver for ExactSolver {
    const NAME: &'static str = "Exact";
    const DIFFUSIVE: bool = false;
}

// =============================================================================
// reconstruction markers (zst)
// =============================================================================

/// piecewise constant (first-order)
pub struct PcmReconstruction;

/// piecewise linear (second-order)
pub struct PlmReconstruction;

/// piecewise parabolic (third-order)
pub struct PpmReconstruction;

/// weno (fifth-order)
pub struct WenoReconstruction;

/// trait bound for valid reconstruction schemes
pub trait ReconstructionScheme {
    const NAME: &'static str;
    const ORDER: usize;
    const STENCIL_SIZE: usize;
}

impl ReconstructionScheme for PcmReconstruction {
    const NAME: &'static str = "PCM";
    const ORDER: usize = 1;
    const STENCIL_SIZE: usize = 1;
}

impl ReconstructionScheme for PlmReconstruction {
    const NAME: &'static str = "PLM";
    const ORDER: usize = 2;
    const STENCIL_SIZE: usize = 3;
}

impl ReconstructionScheme for PpmReconstruction {
    const NAME: &'static str = "PPM";
    const ORDER: usize = 3;
    const STENCIL_SIZE: usize = 5;
}

impl ReconstructionScheme for WenoReconstruction {
    const NAME: &'static str = "WENO5";
    const ORDER: usize = 5;
    const STENCIL_SIZE: usize = 5;
}

// =============================================================================
// physics configuration (compile-time dispatch)
// =============================================================================

/// complete physics configuration.
/// all parameters known at compile time via zero-sized types.
pub struct PhysicsConfig<R, D, S, Rec> {
    _regime: PhantomData<R>,
    _dim: PhantomData<D>,
    _solver: PhantomData<S>,
    _reconstruction: PhantomData<Rec>,
}

/// trait that defines valid physics configurations.
/// implementations specify field layout and primitive/conserved types.
pub trait Configuration {
    type Regime: Regime;
    type Dim: Dimensionality;
    type Solver: RiemannSolver;
    type Reconstruction: ReconstructionScheme;

    /// number of spatial dimensions
    const NDIM: usize = Self::Dim::NDIM;

    /// stencil size for reconstruction
    const STENCIL_SIZE: usize = Self::Reconstruction::STENCIL_SIZE;

    /// configuration name for logging
    fn name() -> String {
        format!(
            "{}-{}-{}-{}",
            Self::Regime::NAME,
            Self::Dim::NAME,
            Self::Solver::NAME,
            Self::Reconstruction::NAME
        )
    }
}

// =============================================================================
// implementations for valid configurations
// =============================================================================

// newtonian 1d
impl Configuration for PhysicsConfig<Newtonian, Dim1, HlleSolver, PlmReconstruction> {
    type Regime = Newtonian;
    type Dim = Dim1;
    type Solver = HlleSolver;
    type Reconstruction = PlmReconstruction;
}

impl Configuration for PhysicsConfig<Newtonian, Dim1, HlleSolver, PcmReconstruction> {
    type Regime = Newtonian;
    type Dim = Dim1;
    type Solver = HlleSolver;
    type Reconstruction = PcmReconstruction;
}

// newtonian 2d
impl Configuration for PhysicsConfig<Newtonian, Dim2, HlleSolver, PlmReconstruction> {
    type Regime = Newtonian;
    type Dim = Dim2;
    type Solver = HlleSolver;
    type Reconstruction = PlmReconstruction;
}

// newtonian 3d
impl Configuration for PhysicsConfig<Newtonian, Dim3, HlleSolver, PlmReconstruction> {
    type Regime = Newtonian;
    type Dim = Dim3;
    type Solver = HlleSolver;
    type Reconstruction = PlmReconstruction;
}

// =============================================================================
// field layout specification
// =============================================================================

/// describes the memory layout of simulation fields.
/// known entirely at compile time based on configuration.
pub trait FieldLayout {
    /// index of density field
    const RHO: usize = 0;

    /// index of first velocity component
    const VX: usize = 1;

    /// index of pressure field
    const PRESSURE: usize;

    /// total number of fields
    const NFIELDS: usize;
}

// 1d layout
impl<R, S, Rec> FieldLayout for PhysicsConfig<R, Dim1, S, Rec>
where
    R: Regime,
    S: RiemannSolver,
    Rec: ReconstructionScheme,
{
    const PRESSURE: usize = 2; // rho, vx, p
    const NFIELDS: usize = 3;
}

// 2d layout
impl<R, S, Rec> FieldLayout for PhysicsConfig<R, Dim2, S, Rec>
where
    R: Regime,
    S: RiemannSolver,
    Rec: ReconstructionScheme,
{
    const PRESSURE: usize = 3; // rho, vx, vy, p
    const NFIELDS: usize = 4;
}

// 3d layout
impl<R, S, Rec> FieldLayout for PhysicsConfig<R, Dim3, S, Rec>
where
    R: Regime,
    S: RiemannSolver,
    Rec: ReconstructionScheme,
{
    const PRESSURE: usize = 4; // rho, vx, vy, vz, p
    const NFIELDS: usize = 5;
}

// =============================================================================
// configuration validation
// =============================================================================

/// validates that a configuration is supported.
/// use as a const assertion in solver initialization.
pub const fn validate_config<C: Configuration>() {
    // add validation rules here
    // compiler will evaluate at compile time
}

// =============================================================================
// type aliases for common configurations
// =============================================================================

/// 1d newtonian euler with hlle and plm
pub type Euler1D = PhysicsConfig<Newtonian, Dim1, HlleSolver, PlmReconstruction>;

/// 2d newtonian euler with hlle and plm
pub type Euler2D = PhysicsConfig<Newtonian, Dim2, HlleSolver, PlmReconstruction>;

/// 3d newtonian euler with hlle and plm
pub type Euler3D = PhysicsConfig<Newtonian, Dim3, HlleSolver, PlmReconstruction>;

/// 1d newtonian euler with hlle and pcm (first-order)
pub type Euler1DFirstOrder = PhysicsConfig<Newtonian, Dim1, HlleSolver, PcmReconstruction>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_configuration_constants() {
        // verify compile-time constants are correct
        assert_eq!(<Euler1D as FieldLayout>::NFIELDS, 3);
        assert_eq!(Euler1D::NDIM, 1);
        assert_eq!(Euler1D::STENCIL_SIZE, 3);

        assert_eq!(<Euler2D as FieldLayout>::NFIELDS, 4);
        assert_eq!(Euler2D::NDIM, 2);

        assert_eq!(<Euler3D as FieldLayout>::NFIELDS, 5);
        assert_eq!(Euler3D::NDIM, 3);
    }

    #[test]
    fn test_field_layout() {
        assert_eq!(<Euler1D as FieldLayout>::RHO, 0);
        assert_eq!(<Euler1D as FieldLayout>::VX, 1);
        assert_eq!(<Euler1D as FieldLayout>::PRESSURE, 2);

        assert_eq!(<Euler2D as FieldLayout>::PRESSURE, 3);
        assert_eq!(<Euler3D as FieldLayout>::PRESSURE, 4);
    }

    #[test]
    fn test_configuration_names() {
        assert_eq!(Euler1D::name(), "Newtonian-1D-HLLE-PLM");
        assert_eq!(Euler2D::name(), "Newtonian-2D-HLLE-PLM");
        assert_eq!(Euler1DFirstOrder::name(), "Newtonian-1D-HLLE-PCM");
    }

    #[test]
    fn test_regime_properties() {
        assert_eq!(Newtonian::NAME, "Newtonian");
        assert_eq!(Newtonian::NCONS, 3);
        assert!(!Newtonian::HAS_MAGNETIC);

        assert_eq!(Mhd::NAME, "MHD");
        assert!(Mhd::HAS_MAGNETIC);
    }

    #[test]
    fn test_zero_sized_types() {
        // verify markers are truly zero-sized
        use core::mem::size_of;
        assert_eq!(size_of::<Newtonian>(), 0);
        assert_eq!(size_of::<Dim1>(), 0);
        assert_eq!(size_of::<HlleSolver>(), 0);
        assert_eq!(size_of::<PlmReconstruction>(), 0);
        assert_eq!(size_of::<Euler1D>(), 0);
    }

    #[test]
    fn test_configuration_validation() {
        // should compile without error
        validate_config::<Euler1D>();
        validate_config::<Euler2D>();
        validate_config::<Euler3D>();
    }
}
