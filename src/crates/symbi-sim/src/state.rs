// =============================================================================
// state.rs
//
// simulation state built on symbi-grid Field and the symbi-xpu
// Executor, with types that work on both CPU and GPU via the xpu layer.
//
// SoA layout: conserved state is stored as separate fields per component
// (den, mom[D], nrg). this is optimal for GPU
// coalesced access. the gather/scatter bridge provides AoS access for
// pointwise physics.
//
// all structs are pure data — no physics methods. behavior lives in free
// functions (ops, evolution driver).
//
// usage:
//   let sim = SimState::<Newtonian, 2, Cartesian, IdealGas<f64>>::build(regime, eos, metric)
//       .cells([nx, ny]).bounds([0., 0.], [1., 1.]).boundaries(BoundaryType::Periodic)
//       .allocate()?.set_initial(ic).build();
//   evolve(&mut sim, &ops, t_final)?;
// =============================================================================

use symbi_algebra::{Domain, OrderedNumeric, Space, Tensor};
use symbi_geometry::{Metric, MotionState};
use symbi_grid::Field;
use symbi_grid::centering::Cell;
use symbi_hydro::energy::{EnergyModel, EnergySlot};
use symbi_hydro::eos::Eos;
use symbi_hydro::regime::Regime;
use symbi_hydro::state::{Cons, ConsG, Prim, PrimG, SeedableCons};
use symbi_ir::algebra::Scalar;
use symbi_xpu::{DefaultMemory, DefaultSpace, ExecutionSpace, Executor, MemorySpace};

// =============================================================================
// energy/pressure FIELD slot — the field-layer analog of
// `symbi_hydro::energy::EnergySlot`. encodes energy presence at the TYPE level so
// `cons.nrg` / `prim.pre` are a real `Field` for energy regimes and a zero-sized
// `FieldZero` for isothermal — retiring the runtime `Option<Field>`. lives HERE (not on
// `EnergyModel`) because `symbi-hydro` cannot see `Field` (no `symbi-grid` dep).
// =============================================================================

/// the uniform surface of an energy/pressure field slot — a real `Field` (energy regimes) or the
/// zero-sized [`FieldZero`] (isothermal). generic code resolves a slot through this without knowing
/// which; regime-specific code names the concrete `Field` directly.
pub trait EnergyFieldSlot<Sc: Scalar + OrderedNumeric, const D: usize, Mem: MemorySpace>:
    Sized
{
    /// allocate the slot (a zeroed `Field`, or the free `FieldZero`).
    fn alloc(domain: &Domain<D>) -> symbi_xpu::Result<Self>;
    /// the backing field, if present (`None` for `FieldZero`). the ONE place absence is handled.
    fn as_field(&self) -> Option<&Field<Sc, D, Mem>>;
    /// the device pointer for the kernel ABI manifest (`0` when absent — the null slot).
    fn ptr(&self) -> u64;
}

impl<Sc: Scalar + OrderedNumeric, const D: usize, Mem: MemorySpace> EnergyFieldSlot<Sc, D, Mem>
    for Field<Sc, D, Mem>
{
    fn alloc(domain: &Domain<D>) -> symbi_xpu::Result<Self> {
        Field::zeros(domain)
    }
    fn as_field(&self) -> Option<&Field<Sc, D, Mem>> {
        Some(self)
    }
    fn ptr(&self) -> u64 {
        self.as_ptr() as u64
    }
}

/// zero-sized energy/pressure field slot for isothermal regimes — the field-layer `Zero<S>`.
#[derive(Clone, Copy, Debug, Default)]
pub struct FieldZero;

impl<Sc: Scalar + OrderedNumeric, const D: usize, Mem: MemorySpace> EnergyFieldSlot<Sc, D, Mem>
    for FieldZero
{
    fn alloc(_domain: &Domain<D>) -> symbi_xpu::Result<Self> {
        Ok(FieldZero)
    }
    fn as_field(&self) -> Option<&Field<Sc, D, Mem>> {
        None
    }
    fn ptr(&self) -> u64 {
        0
    }
}

/// bridge from an energy MARKER (`Adiabatic` / `IsoModel`, foreign `symbi-hydro` types) to its
/// field slot. impl'd here (local trait + foreign type) so `SimStateGeneric<R, ..>` can pick the
/// field storage from `R::Energy` at the type level.
pub trait FieldEnergy {
    type Slot<Sc: Scalar + OrderedNumeric, const D: usize, Mem: MemorySpace>: EnergyFieldSlot<Sc, D, Mem>;
}

impl FieldEnergy for symbi_hydro::energy::Adiabatic {
    type Slot<Sc: Scalar + OrderedNumeric, const D: usize, Mem: MemorySpace> = Field<Sc, D, Mem>;
}

impl FieldEnergy for symbi_hydro::energy::IsoModel {
    type Slot<Sc: Scalar + OrderedNumeric, const D: usize, Mem: MemorySpace> = FieldZero;
}

// the energy/pressure field storage stays `Option<Field>`. the
// additive foundation above (`FieldEnergy`/`EnergyFieldSlot`/`FieldZero` + `Regime::Energy`) is
// the type-level slot resolver; the field containers do not route their `nrg`/`pre` storage
// through it (routing them would touch ~180 access sites across ~55 files plus the foreign-`Field`
// trait scope).

// =============================================================================
// SoA field containers
// =============================================================================

/// conserved state in SoA layout: separate field per component.
/// GPU-optimal: each kernel reads one contiguous array at a time. nrg is None for
/// isothermal regimes (no energy equation).
///
/// `NDIM` is the GRID dimension (the field storage `Field<Sc, NDIM, M>`); `DOF` the VECTOR
/// (momentum-component) dimension — decoupled, so axisymmetric (r,z) hydro carries `DOF=3`
/// momentum components (the v_phi swirl) on an `NDIM=2` grid. the `ConsFields<D>` alias fills
/// `DOF = NDIM = D` (the natural case), so every existing site is unchanged.
pub struct ConsFieldsGeneric<
    const NDIM: usize,
    const DOF: usize,
    M: MemorySpace = DefaultMemory,
    Sc: Scalar + OrderedNumeric = f64,
> {
    pub den: Field<Sc, NDIM, M>,
    pub mom: [Field<Sc, NDIM, M>; DOF],
    pub nrg: Option<Field<Sc, NDIM, M>>,
    /// the conserved passive scalar `D_chi = rho * chi` (dye). `None` unless the
    /// run declares one; absent from the positional pointer ABI — chi kernels
    /// bind it by manifest name only.
    pub chi: Option<Field<Sc, NDIM, M>>,
}

/// the natural case: vector dimension == grid dimension.
pub type ConsFields<const D: usize, M = DefaultMemory, Sc = f64> = ConsFieldsGeneric<D, D, M, Sc>;

impl<const NDIM: usize, const DOF: usize, M: MemorySpace, Sc: Scalar + OrderedNumeric>
    ConsFieldsGeneric<NDIM, DOF, M, Sc>
{
    /// fill a u64 pointer array with [den, mom[0], ..., mom[D-1], nrg].
    /// the array must have at least D+2 elements.
    /// for isothermal regimes (nrg = None), writes a null pointer.
    pub fn fill_ptr_array(&self, arr: *mut u64) {
        unsafe {
            *arr.add(0) = self.den.as_ptr() as u64;
            for dd in 0..DOF {
                *arr.add(1 + dd) = self.mom[dd].as_ptr() as u64;
            }
            *arr.add(DOF + 1) = self.nrg.as_ref().map_or(0u64, |f| f.as_ptr() as u64);
        }
    }

    /// pack all field pointers into kernel args (den, mom[0..DOF], nrg).
    /// for isothermal regimes (nrg = None), pushes a null pointer.
    /// the GPU kernel must check for null and skip energy operations.
    pub fn push_ptrs_to(&self, args: &mut symbi_xpu::KernelArgs) {
        args.push(&(self.den.as_ptr() as u64));
        for dd in 0..DOF {
            args.push(&(self.mom[dd].as_ptr() as u64));
        }
        let nrg_ptr = self.nrg.as_ref().map_or(0u64, |f| f.as_ptr() as u64);
        args.push(&nrg_ptr);
    }

    /// the cell-centered conserved fields carried by the decomposition halo exchange and
    /// the output gather, in a fixed order: den, mom[0..DOF], nrg (adiabatic only), chi
    /// (the passive scalar, run-level opt-in). the set is derived HERE, beside the fields,
    /// so an optional slot cannot be silently dropped by a hand-maintained list in the
    /// transport; a `None` slot contributes nothing, so hydro/iso and non-dye runs are
    /// unchanged. `bcell`/`bface` live on the mhd side-car and are appended by the caller.
    pub fn exchange_fields(&self) -> Vec<&Field<Sc, NDIM, M>> {
        let mut fields: Vec<&Field<Sc, NDIM, M>> = Vec::with_capacity(DOF + 3);
        fields.push(&self.den);
        fields.extend(self.mom.iter());
        fields.extend(self.nrg.as_ref());
        fields.extend(self.chi.as_ref());
        fields
    }

    /// return all scalar field raw pointers in order: den, mom[0..DOF], nrg.
    /// for isothermal regimes (nrg = None), pushes a null pointer.
    pub fn all_ptrs(&self) -> Vec<*const Sc> {
        let mut ptrs = Vec::with_capacity(DOF + 2);
        ptrs.push(self.den.as_ptr());
        for dd in 0..DOF {
            ptrs.push(self.mom[dd].as_ptr());
        }
        ptrs.push(self.nrg.as_ref().map_or(std::ptr::null(), |f| f.as_ptr()));
        ptrs
    }

    /// allocate all fields including energy. default for non-isothermal regimes.
    pub fn zeros(domain: &Domain<NDIM>) -> symbi_xpu::Result<Self> {
        Self::zeros_with_energy(domain, true)
    }

    /// allocate fields. `nrg` is allocated IFF `has_energy` — isothermal regimes get
    /// `nrg = None` (no energy equation, no slot), symmetric with `prim.pre`. this makes
    /// `has_energy()` truthful (the allocated buffer set matches the regime's energy
    /// semantics) and stops iso wasting a per-cell field the kernels never bind.
    pub fn zeros_with_energy(domain: &Domain<NDIM>, has_energy: bool) -> symbi_xpu::Result<Self> {
        Ok(Self {
            den: Field::zeros(domain)?,
            mom: array_field_zeros(domain)?,
            nrg: if has_energy {
                Some(Field::zeros(domain)?)
            } else {
                None
            },
            chi: None,
        })
    }

    /// allocate the passive-scalar slot (a run-level opt-in, unlike the
    /// regime-static `nrg`).
    pub fn alloc_chi(&mut self, domain: &Domain<NDIM>) -> symbi_xpu::Result<()> {
        self.chi = Some(Field::zeros(domain)?);
        Ok(())
    }

    /// whether this field set includes energy.
    pub fn has_energy(&self) -> bool {
        self.nrg_field().is_some()
    }

    /// **the passive-scalar-slot accessor**: the ONE accessor for the `chi`
    /// field, mirroring `nrg_field`. `None` when the run carries no dye.
    #[inline]
    pub fn chi_field(&self) -> Option<&Field<Sc, NDIM, M>> {
        self.chi.as_ref()
    }

    /// **the energy-slot accessor**: the ONE
    /// accessor for the `nrg` field. ALL readers route through this, so swapping the representation
    /// (`Option<Field>` -> a type-level `E::Slot`) is a one-place change behind this method; without
    /// the accessor it would be a 180-site sweep. returns the backing field, or `None` when energy is absent (isothermal).
    #[inline]
    pub fn nrg_field(&self) -> Option<&Field<Sc, NDIM, M>> {
        self.nrg.as_ref()
    }

    /// gather AoS from SoA at a coordinate.
    /// returns nrg=0.0 when energy field is absent (isothermal).
    #[inline]
    pub fn gather(&self, coord: [isize; NDIM]) -> Cons<Sc, DOF> {
        Cons {
            den: *self.den.view().at(coord),
            mom: Tensor::new(std::array::from_fn(|dd| *self.mom[dd].view().at(coord))),
            nrg: self.nrg.as_ref().map_or(Sc::ZERO, |f| *f.view().at(coord)),
        }
    }

    /// scatter AoS to SoA at a coordinate.
    /// skips nrg write when energy field is absent (isothermal).
    #[inline]
    pub fn scatter(&self, coord: [isize; NDIM], val: Cons<Sc, DOF>) {
        self.den.view_mut().set(coord, val.den);
        for dd in 0..DOF {
            self.mom[dd].view_mut().set(coord, val.mom[dd]);
        }
        if let Some(ref nrg) = self.nrg {
            nrg.view_mut().set(coord, val.nrg);
        }
    }

    /// gather AoS from SoA with a specific energy model.
    /// isothermal: nrg slot becomes Zero<f64> (ZST). adiabatic: nrg slot is f64.
    #[inline]
    pub fn gather_as<E: EnergyModel>(&self, coord: [isize; NDIM]) -> ConsG<Sc, DOF, E> {
        ConsG {
            den: *self.den.view().at(coord),
            mom: Tensor::new(std::array::from_fn(|dd| *self.mom[dd].view().at(coord))),
            nrg: E::Slot::<Sc>::from_scalar(
                self.nrg.as_ref().map_or(Sc::ZERO, |f| *f.view().at(coord)),
            ),
        }
    }

    /// scatter AoS to SoA with a specific energy model.
    /// isothermal: writes 0.0 for nrg (via Zero::value()). adiabatic: writes real nrg.
    #[inline]
    pub fn scatter_from<E: EnergyModel>(&self, coord: [isize; NDIM], val: ConsG<Sc, DOF, E>) {
        self.den.view_mut().set(coord, val.den);
        for dd in 0..DOF {
            self.mom[dd].view_mut().set(coord, val.mom[dd]);
        }
        if let Some(ref nrg) = self.nrg {
            nrg.view_mut().set(coord, val.nrg.value());
        }
    }
}

/// primitive state in SoA layout. pre is None for isothermal regimes
/// (pressure is derived from the eos and never stored). `NDIM` = grid dim, `DOF` = velocity-component
/// dim (decoupled — the `PrimFields<D>` alias fills `DOF = NDIM = D`).
pub struct PrimFieldsGeneric<
    const NDIM: usize,
    const DOF: usize,
    M: MemorySpace = DefaultMemory,
    Sc: Scalar + OrderedNumeric = f64,
> {
    pub rho: Field<Sc, NDIM, M>,
    pub vel: [Field<Sc, NDIM, M>; DOF],
    pub pre: Option<Field<Sc, NDIM, M>>,
    /// the primitive passive-scalar concentration `chi = D_chi / rho`. `None`
    /// unless the run declares one; outside the positional pointer ABI.
    pub chi: Option<Field<Sc, NDIM, M>>,
}

/// the natural case: velocity dimension == grid dimension.
pub type PrimFields<const D: usize, M = DefaultMemory, Sc = f64> = PrimFieldsGeneric<D, D, M, Sc>;

impl<const NDIM: usize, const DOF: usize, M: MemorySpace, Sc: Scalar + OrderedNumeric>
    PrimFieldsGeneric<NDIM, DOF, M, Sc>
{
    /// fill a u64 pointer array with [rho, vel[0], ..., vel[DOF-1], pre].
    /// for isothermal regimes (pre = None), writes a null pointer.
    pub fn fill_ptr_array(&self, arr: *mut u64) {
        unsafe {
            *arr.add(0) = self.rho.as_ptr() as u64;
            for dd in 0..DOF {
                *arr.add(1 + dd) = self.vel[dd].as_ptr() as u64;
            }
            *arr.add(DOF + 1) = self.pre.as_ref().map_or(0u64, |f| f.as_ptr() as u64);
        }
    }

    /// pack all field pointers into kernel args (rho, vel[0..DOF], pre).
    /// for isothermal regimes (pre = None), pushes a null pointer.
    pub fn push_ptrs_to(&self, args: &mut symbi_xpu::KernelArgs) {
        args.push(&(self.rho.as_ptr() as u64));
        for dd in 0..DOF {
            args.push(&(self.vel[dd].as_ptr() as u64));
        }
        let pre_ptr = self.pre.as_ref().map_or(0u64, |f| f.as_ptr() as u64);
        args.push(&pre_ptr);
    }

    /// allocate all fields including pressure. default for non-isothermal regimes.
    pub fn zeros(domain: &Domain<NDIM>) -> symbi_xpu::Result<Self> {
        Self::zeros_with_pressure(domain, true)
    }

    /// allocate fields. when has_pressure is false, pre is None (isothermal).
    pub fn zeros_with_pressure(
        domain: &Domain<NDIM>,
        has_pressure: bool,
    ) -> symbi_xpu::Result<Self> {
        Ok(Self {
            rho: Field::zeros(domain)?,
            vel: array_field_zeros(domain)?,
            pre: if has_pressure {
                Some(Field::zeros(domain)?)
            } else {
                None
            },
            chi: None,
        })
    }

    /// allocate the passive-scalar concentration slot (run-level opt-in).
    pub fn alloc_chi(&mut self, domain: &Domain<NDIM>) -> symbi_xpu::Result<()> {
        self.chi = Some(Field::zeros(domain)?);
        Ok(())
    }

    /// **the pressure-slot accessor**: the ONE
    /// accessor for the `pre` field. ALL readers route through this so the representation is swappable
    /// in one place. `None` when pressure is not stored (isothermal — derived from the EOS).
    #[inline]
    pub fn pre_field(&self) -> Option<&Field<Sc, NDIM, M>> {
        self.pre.as_ref()
    }

    /// **the passive-scalar-slot accessor**, mirroring `pre_field`.
    #[inline]
    pub fn chi_field(&self) -> Option<&Field<Sc, NDIM, M>> {
        self.chi.as_ref()
    }

    /// the cell-centered primitive components the flux stage reconstructs from and the
    /// decomposition exchanges each step, in a fixed order: rho, vel[0..DOF], pre
    /// (adiabatic only), chi (the passive scalar, run-level opt-in). derived beside the
    /// fields for the same reason as the conserved set — an optional slot cannot be
    /// forgotten by a transport-side list; a `None` slot contributes nothing.
    pub fn exchange_fields(&self) -> Vec<&Field<Sc, NDIM, M>> {
        let mut fields: Vec<&Field<Sc, NDIM, M>> = Vec::with_capacity(DOF + 2);
        fields.push(&self.rho);
        fields.extend(self.vel.iter());
        fields.extend(self.pre.as_ref());
        fields.extend(self.chi.as_ref());
        fields
    }

    #[inline]
    pub fn gather(&self, coord: [isize; NDIM]) -> Prim<Sc, DOF> {
        Prim {
            rho: *self.rho.view().at(coord),
            vel: Tensor::new(std::array::from_fn(|dd| *self.vel[dd].view().at(coord))),
            pre: self.pre.as_ref().map_or(Sc::ZERO, |f| *f.view().at(coord)),
        }
    }

    #[inline]
    pub fn scatter(&self, coord: [isize; NDIM], val: Prim<Sc, DOF>) {
        self.rho.view_mut().set(coord, val.rho);
        for dd in 0..DOF {
            self.vel[dd].view_mut().set(coord, val.vel[dd]);
        }
        if let Some(ref pre) = self.pre {
            pre.view_mut().set(coord, val.pre);
        }
    }

    /// gather AoS from SoA with a specific energy model.
    /// isothermal: pre slot becomes Zero<f64> (ZST). adiabatic: pre slot is f64.
    #[inline]
    pub fn gather_as<E: EnergyModel>(&self, coord: [isize; NDIM]) -> PrimG<Sc, DOF, E> {
        PrimG {
            rho: *self.rho.view().at(coord),
            vel: Tensor::new(std::array::from_fn(|dd| *self.vel[dd].view().at(coord))),
            pre: E::Slot::<Sc>::from_scalar(
                self.pre.as_ref().map_or(Sc::ZERO, |f| *f.view().at(coord)),
            ),
        }
    }

    /// scatter AoS to SoA with a specific energy model.
    /// isothermal: writes 0.0 for pre (via Zero::value()). adiabatic: writes real pre.
    #[inline]
    pub fn scatter_from<E: EnergyModel>(&self, coord: [isize; NDIM], val: PrimG<Sc, DOF, E>) {
        self.rho.view_mut().set(coord, val.rho);
        for dd in 0..DOF {
            self.vel[dd].view_mut().set(coord, val.vel[dd]);
        }
        if let Some(ref pre) = self.pre {
            pre.view_mut().set(coord, val.pre.value());
        }
    }
}

// =============================================================================
// partition fields + workspace
// =============================================================================

/// all field storage for one partition. `NDIM` = grid dim, `DOF` = vector (momentum)
/// component dim; the `PartitionFields<D>` alias fills `DOF = NDIM = D`.
/// the MHD staggered fields stay keyed on `NDIM` (RMHD is full-3D, so its B is NDIM-component).
pub struct PartitionFieldsGeneric<
    const NDIM: usize,
    const DOF: usize,
    M: MemorySpace = DefaultMemory,
    Sc: Scalar + OrderedNumeric = f64,
> {
    pub cons: ConsFieldsGeneric<NDIM, DOF, M, Sc>,
    pub prim: PrimFieldsGeneric<NDIM, DOF, M, Sc>,
    pub flux: [ConsFieldsGeneric<NDIM, DOF, M, Sc>; NDIM],
    /// per-cell c2p validity status. zero means the recovered primitive lies in
    /// the strict admissible interior; nonzero means recovery produced a state
    /// rejected by the same predicate used by fofc.
    pub c2p_error: Field<Sc, NDIM, M>,
    /// MHD staggered fields. None for pure hydro regimes.
    pub mhd: Option<MhdStaggeredFields<NDIM, DOF, M, Sc>>,
    /// external source term field (gravity, bodies, cooling, etc.).
    /// written by the source pass before godunov, read by godunov.
    /// zeroed at the start of each step. None when no sources are active.
    pub source: Option<ConsFieldsGeneric<NDIM, DOF, M, Sc>>,
}

/// the natural case: vector dimension == grid dimension.
pub type PartitionFields<const D: usize, M = DefaultMemory, Sc = f64> =
    PartitionFieldsGeneric<D, D, M, Sc>;

// =============================================================================
// MHD staggered FieldGroups
//
// each of the per-axis Field arrays inside MhdStaggeredFields is
// wrapped in a `#[derive(FieldGroup)]` struct that carries its centering at
// the type level (Cell / Face / Edge — axis-erased). this lets
// chalkboard kernels accept `&BfaceFields<D, M>` etc. and the macro emits
// per-D per-member access automatically.
//
// `Index<usize>` / `IndexMut<usize>` impls preserve the bare-array indexing
// syntax (`bcell[d]`) so call sites that read or write `mhd.bcell[d]` use the
// same syntax whether the field is a bare array or one of these groups.
// =============================================================================

/// cell-centered magnetic field group: one Field per B-component.
/// internal Field array uses default `Cell` centering; per-kernel call sites
/// may use typed Cell/Face/Edge centering. the FieldGroup struct distinction
/// (BcellFields != BfaceFields)
/// already separates the storage at the type level.
// N = vector-component count (DOF for MHD), decoupled from the grid dimension D:
// the cell-centered B is a DOF-vector on a D-axis grid, so the array length (N=DOF)
// differs from the Field spatial dim (D) whenever D != DOF (1.5D / 2.5D MHD).
pub struct BcellFields<
    const D: usize,
    const N: usize,
    M: MemorySpace = DefaultMemory,
    Sc: Scalar + OrderedNumeric = f64,
> {
    pub b: [Field<Sc, D, M>; N],
}

impl<const D: usize, const N: usize, M: MemorySpace, Sc: Scalar + OrderedNumeric>
    std::ops::Index<usize> for BcellFields<D, N, M, Sc>
{
    type Output = Field<Sc, D, M>;
    #[inline]
    fn index(&self, dd: usize) -> &Self::Output {
        &self.b[dd]
    }
}

impl<const D: usize, const N: usize, M: MemorySpace, Sc: Scalar + OrderedNumeric>
    std::ops::IndexMut<usize> for BcellFields<D, N, M, Sc>
{
    #[inline]
    fn index_mut(&mut self, dd: usize) -> &mut Self::Output {
        &mut self.b[dd]
    }
}

/// face-centered magnetic field group: bface[d] lives on the d-perpendicular
/// face. CT-evolved "truth"; bcell is interpolated from this.
pub struct BfaceFields<
    const D: usize,
    M: MemorySpace = DefaultMemory,
    Sc: Scalar + OrderedNumeric = f64,
> {
    pub b: [Field<Sc, D, M>; D],
}

impl<const D: usize, M: MemorySpace, Sc: Scalar + OrderedNumeric> std::ops::Index<usize>
    for BfaceFields<D, M, Sc>
{
    type Output = Field<Sc, D, M>;
    #[inline]
    fn index(&self, dd: usize) -> &Self::Output {
        &self.b[dd]
    }
}

impl<const D: usize, M: MemorySpace, Sc: Scalar + OrderedNumeric> std::ops::IndexMut<usize>
    for BfaceFields<D, M, Sc>
{
    #[inline]
    fn index_mut(&mut self, dd: usize) -> &mut Self::Output {
        &mut self.b[dd]
    }
}

/// edge-centered electric field group: efield[d] lives on edges parallel
/// to axis d. transient; recomputed each stage from fluxes.
pub struct EfieldFields<
    const D: usize,
    M: MemorySpace = DefaultMemory,
    Sc: Scalar + OrderedNumeric = f64,
> {
    pub e: [Field<Sc, D, M>; D],
}

impl<const D: usize, M: MemorySpace, Sc: Scalar + OrderedNumeric> std::ops::Index<usize>
    for EfieldFields<D, M, Sc>
{
    type Output = Field<Sc, D, M>;
    #[inline]
    fn index(&self, dd: usize) -> &Self::Output {
        &self.e[dd]
    }
}

impl<const D: usize, M: MemorySpace, Sc: Scalar + OrderedNumeric> std::ops::IndexMut<usize>
    for EfieldFields<D, M, Sc>
{
    #[inline]
    fn index_mut(&mut self, dd: usize) -> &mut Self::Output {
        &mut self.e[dd]
    }
}

/// per-axis B-flux group: f[c] is the c-th magnetic-component flux on an
/// axis-d face. the owning MhdStaggeredFields uses `[BfluxFields<D, M>; D]`,
/// indexed by axis-of-flux.
pub struct BfluxFields<
    const D: usize,
    const N: usize,
    M: MemorySpace = DefaultMemory,
    Sc: Scalar + OrderedNumeric = f64,
> {
    pub f: [Field<Sc, D, M>; N],
}

impl<const D: usize, const N: usize, M: MemorySpace, Sc: Scalar + OrderedNumeric>
    std::ops::Index<usize> for BfluxFields<D, N, M, Sc>
{
    type Output = Field<Sc, D, M>;
    #[inline]
    fn index(&self, dd: usize) -> &Self::Output {
        &self.f[dd]
    }
}

impl<const D: usize, const N: usize, M: MemorySpace, Sc: Scalar + OrderedNumeric>
    std::ops::IndexMut<usize> for BfluxFields<D, N, M, Sc>
{
    #[inline]
    fn index_mut(&mut self, dd: usize) -> &mut Self::Output {
        &mut self.f[dd]
    }
}

/// the step-entry state an explicit step is replayed from after it is REJECTED: the gas
/// conserved state plus both representations of the magnetic field. rolling the face field
/// back exactly is what keeps `div(B) = 0` across a rejection — re-curling from a restored
/// `bface` reproduces the accepted-step history rather than accumulating the rejected curl.
pub struct MhdStepSnapshot<
    const D: usize,
    const DOF: usize,
    M: MemorySpace = DefaultMemory,
    Sc: Scalar + OrderedNumeric = f64,
> {
    pub cons: ConsFieldsGeneric<D, DOF, M, Sc>,
    pub bcell: BcellFields<D, DOF, M, Sc>,
    pub bface: BfaceFields<D, M, Sc>,
}

/// MHD field storage: cell-centered B + staggered CT fields.
/// the cell-centered B is used for reconstruction and flux computation.
/// the face-centered B is the CT "truth" (2D/3D only).
pub struct MhdStaggeredFields<
    const D: usize,
    const DOF: usize,
    M: MemorySpace = DefaultMemory,
    Sc: Scalar + OrderedNumeric = f64,
> {
    /// the step-entry rollback state, present only where a step can be rejected: the GRMHD
    /// physical-constraint-preserving redo halves the timestep and replays when its source-free
    /// low-order anchor is itself inadmissible. every other MHD regime accepts every step it
    /// takes, so the snapshot would be a permanently dead allocation the size of the whole
    /// conserved + magnetic state.
    pub step_snapshot: Option<MhdStepSnapshot<D, DOF, M, Sc>>,

    /// cell-centered B: bcell[c] on the allocated domain (same as cons/prim), one
    /// per DOF vector component. the D in-plane components [0..D) are interpolated
    /// from bface after CT; the (DOF-D) out-of-plane components [D..DOF) have no
    /// face to stagger on and are carried/evolved cell-centered directly (1.5D /
    /// 2.5D MHD). at D=DOF this is the fully interpolated B.
    pub bcell: BcellFields<D, DOF, M, Sc>,

    /// RK2 snapshot of bcell at the start of a step (bcell^n). the godunov
    /// flux-evolves bcell as a conserved component (the gas scale/add steps
    /// include B); the RK2 combine needs bcell^n. the flux-predicted bcell is
    /// the b_old the magnetic-energy correction reads before CT overwrites bcell.
    pub bcell_n: BcellFields<D, DOF, M, Sc>,

    /// FOFC: the STAGE-INPUT cell B (snapshot in `snapshot_stage`, alongside `u_stage`). the
    /// face-based CT redo restores `bcell <- bcell_stage` before re-running the stage from the
    /// spliced fluxes, so the recomputed edge EMF reads the stage-input B and the cell-B predictor
    /// combines from the correct base. only touched on a firing MHD substage.
    pub bcell_stage: BcellFields<D, DOF, M, Sc>,

    /// face-centered B: bface[d] on interior.extend(d, 0, 1) with ±1 transverse
    /// halo on each axis tt != d (the MHD/RMHD face domains).
    /// this is the CT "truth" — evolved by discrete curl of E.
    /// only used in 2D/3D.
    pub bface: BfaceFields<D, M, Sc>,

    /// FOFC: snapshot of `bface` taken in `post_godunov` immediately BEFORE the corrector/euler curl
    /// (bface is untouched until then, so this is `bface^n`). the C2 CT redo restores `bface <-
    /// bface_n` and re-applies the curl from the SPLICED edge EMF (HO off the fallback region, FO on
    /// it), so the curl is applied exactly ONCE. only used in 2D/3D, only touched on a firing substage.
    pub bface_n: BfaceFields<D, M, Sc>,

    /// edge-centered E: efield[d] on edge_domain(d).
    /// transient — recomputed each stage from fluxes and prims.
    pub efield: EfieldFields<D, M, Sc>,

    /// saved E from RK2 stage 1 (for time-averaging).
    pub efield_n: EfieldFields<D, M, Sc>,

    /// FOFC: save of the HIGH-ORDER edge EMF (`= efield` at FOFC entry). the C2 CT redo splices the
    /// edge EMF `edge_flag ? E_FO(Contact) : E_HO`, keeping the saved HO EMF here on edges touching no
    /// flagged cell so their face field is bit-unchanged (I5). only touched on a firing substage.
    pub efield_ho: EfieldFields<D, M, Sc>,

    /// B-field flux: bflux[d] is the per-axis BfluxFields group; bflux[d][k]
    /// is the k-th B-component flux at d-perpendicular faces.
    /// used in 1D to update B via flux divergence (no CT).
    /// in 2D/3D, the induction flux is used to compute E at faces.
    /// inner length DOF: each face carries all DOF B-component fluxes.
    pub bflux: [BfluxFields<D, DOF, M, Sc>; D],

    /// FOFC: the HIGH-ORDER induction flux save (mirror of the gas `flux_ho`). the face-based CT
    /// redo saves `bflux -> bflux_ho` before the first-order redo overwrites `bflux`, then splices
    /// FO-on-flagged faces from `bflux_ho` (HO) and the live `bflux` (FO) so the recomputed edge EMF
    /// and cell-B predictor are HO off the fallback region, FO on it. only touched on a firing
    /// MHD substage.
    pub bflux_ho: [BfluxFields<D, DOF, M, Sc>; D],

    /// per-face wave speeds from the Riemann solver.
    /// wave_speed_l[d] and wave_speed_r[d] store the left (negative) and
    /// right (positive) signal speeds at each face in direction d.
    /// alpha_plus = max(0, wave_speed_r), alpha_minus = -min(0, wave_speed_l).
    /// used by UCT-HLL emf averaging (Mignone & Del Zanna 2021, Eq. 32).
    pub wave_speed_l: [Field<Sc, D, M>; D],
    pub wave_speed_r: [Field<Sc, D, M>; D],

    /// per-face upwind transverse velocity from the Riemann solver.
    /// v_upwind[dir][t] stores the upwind-weighted transverse velocity
    /// component t at faces in direction dir (Eq. 29):
    ///   v_bar_t = (alpha^+ * v_t^L + alpha^- * v_t^R) / (alpha^+ + alpha^-)
    /// used by UCT-HLL emf construction (Eq. 33).
    pub v_upwind: [[Field<Sc, D, M>; D]; D],

    /// whether bface has been explicitly initialized.
    /// set by mhd_init_bface_from_bcell or by direct user writes.
    /// evolve() uses this to auto-init bface from bcell on first call
    /// when the user hasn't set face values directly.
    pub bface_initialized: std::sync::atomic::AtomicBool,
}

impl<const D: usize, const DOF: usize, M: MemorySpace, Sc: Scalar + OrderedNumeric>
    MhdStaggeredFields<D, DOF, M, Sc>
{
    /// allocate MHD fields from the cell-centered domains.
    /// allocated_domain: full domain including ghost cells (for bcell, bflux).
    /// interior: interior domain (for bface, efield).
    /// `rejectable` allocates the step-entry rollback snapshot (see `step_snapshot`).
    pub fn zeros(
        allocated: &Domain<D>,
        interior: &Domain<D>,
        rejectable: bool,
    ) -> symbi_xpu::Result<Self> {
        // cell-centered B: same domain as cons/prim, DOF vector components.
        let bcell = BcellFields {
            b: array_field_zeros::<D, DOF, M, Cell, Sc>(allocated)?,
        };
        // RK2 snapshot of bcell (same domain).
        let bcell_n = BcellFields {
            b: array_field_zeros::<D, DOF, M, Cell, Sc>(allocated)?,
        };
        // FOFC stage-input cell-B snapshot (same domain).
        let bcell_stage = BcellFields {
            b: array_field_zeros::<D, DOF, M, Cell, Sc>(allocated)?,
        };

        // face-centered B: one extra in normal direction; for MHD the CT
        // stencil needs a TRANSVERSE halo. ±2 (not ±1): the faithful UCT edge EMF
        // (Mignone & Del Zanna) PLM-reconstructs the staggered transverse field to
        // the edge, whose minmod slope reaches the second transverse neighbour. ±1
        // suffices for bface→bcell + curl-of-E (which read 1 neighbour); the extra
        // layer is filled by the same owned→alloc ghost-fill driver and is harmless
        // to the narrower readers. on-disk checkpoint is interior-only, unaffected.
        let mut bface_vec: Vec<Field<Sc, D, M>> = Vec::with_capacity(D);
        let mut bface_n_vec: Vec<Field<Sc, D, M>> = Vec::with_capacity(D);
        let mut face_doms: Vec<Domain<D>> = Vec::with_capacity(D);
        for dd in 0..D {
            let mut face_dom = interior.extend(dd, 0, 1);
            for tt in 0..D {
                if tt != dd {
                    face_dom = face_dom.extend(tt, -2, 2);
                }
            }
            bface_vec.push(Field::zeros(&face_dom)?);
            bface_n_vec.push(Field::zeros(&face_dom)?); // FOFC bface^n snapshot (same face domain)
            face_doms.push(face_dom);
        }
        let bface = BfaceFields {
            b: bface_vec.try_into().unwrap_or_else(|_| unreachable!()),
        };
        let bface_n = BfaceFields {
            b: bface_n_vec.try_into().unwrap_or_else(|_| unreachable!()),
        };
        let step_snapshot = if rejectable {
            let mut snapshot_faces: Vec<Field<Sc, D, M>> = Vec::with_capacity(D);
            for face_dom in &face_doms {
                snapshot_faces.push(Field::zeros(face_dom)?);
            }
            Some(MhdStepSnapshot {
                cons: ConsFieldsGeneric::zeros_with_energy(allocated, true)?,
                bcell: BcellFields {
                    b: array_field_zeros::<D, DOF, M, Cell, Sc>(allocated)?,
                },
                bface: BfaceFields {
                    b: snapshot_faces.try_into().unwrap_or_else(|_| unreachable!()),
                },
            })
        } else {
            None
        };

        // edge-centered E: extra in both transverse directions.
        // for D=2, all efield slots use the corner domain (extend in both
        // directions) because the only physical E-field is Ez at corners.
        let mut efield_vec: Vec<Field<Sc, D, M>> = Vec::with_capacity(D);
        let mut efield_n_vec: Vec<Field<Sc, D, M>> = Vec::with_capacity(D);
        let mut efield_ho_vec: Vec<Field<Sc, D, M>> = Vec::with_capacity(D);
        for dd in 0..D {
            let mut edge_dom = interior.clone();
            for ax in 0..D {
                if ax != dd || D == 2 {
                    edge_dom = edge_dom.extend(ax, 0, 1);
                }
            }
            efield_vec.push(Field::zeros(&edge_dom)?);
            efield_n_vec.push(Field::zeros(&edge_dom)?);
            efield_ho_vec.push(Field::zeros(&edge_dom)?); // FOFC HO-EMF save (same edge domain)
        }
        let efield = EfieldFields {
            e: efield_vec.try_into().unwrap_or_else(|_| unreachable!()),
        };
        let efield_n = EfieldFields {
            e: efield_n_vec.try_into().unwrap_or_else(|_| unreachable!()),
        };
        let efield_ho = EfieldFields {
            e: efield_ho_vec.try_into().unwrap_or_else(|_| unreachable!()),
        };

        // B-field flux arrays: per-axis group, same domain as hydro flux[d];
        // inner length DOF (one flux field per B-component).
        let mut bflux_outer: Vec<BfluxFields<D, DOF, M, Sc>> = Vec::with_capacity(D);
        let mut bflux_ho_outer: Vec<BfluxFields<D, DOF, M, Sc>> = Vec::with_capacity(D);
        for _dd in 0..D {
            bflux_outer.push(BfluxFields {
                f: array_field_zeros::<D, DOF, M, Cell, Sc>(allocated)?,
            });
            // FOFC HO induction-flux save: same per-axis DOF-component layout as bflux.
            bflux_ho_outer.push(BfluxFields {
                f: array_field_zeros::<D, DOF, M, Cell, Sc>(allocated)?,
            });
        }
        let bflux = bflux_outer.try_into().unwrap_or_else(|_| unreachable!());
        let bflux_ho = bflux_ho_outer.try_into().unwrap_or_else(|_| unreachable!());

        // per-face wave speeds and upwind transverse velocities
        let wave_speed_l = array_field_zeros(allocated)?;
        let wave_speed_r = array_field_zeros(allocated)?;
        let mut v_upwind_outer = Vec::with_capacity(D);
        for _dd in 0..D {
            v_upwind_outer.push(array_field_zeros(allocated)?);
        }

        Ok(MhdStaggeredFields {
            step_snapshot,
            bcell,
            bcell_n,
            bcell_stage,
            bface,
            bface_n,
            efield,
            efield_n,
            efield_ho,
            bflux,
            bflux_ho,
            wave_speed_l,
            wave_speed_r,
            v_upwind: v_upwind_outer.try_into().unwrap_or_else(|_| unreachable!()),
            bface_initialized: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// gather cell-centered B at a coordinate.
    pub fn gather_bcell(&self, coord: [isize; D]) -> symbi_algebra::Tensor<Sc, D> {
        symbi_algebra::Tensor::new(std::array::from_fn(|dd| *self.bcell[dd].view().at(coord)))
    }

    /// scatter cell-centered B at a coordinate.
    pub fn scatter_bcell(&self, coord: [isize; D], mag: symbi_algebra::Tensor<Sc, D>) {
        for dd in 0..D {
            self.bcell[dd].view_mut().set(coord, mag[dd]);
        }
    }

    /// push bcell pointers to GPU kernel args (b0 [, b1 [, b2]]).
    pub fn push_bcell_ptrs_to(&self, args: &mut symbi_xpu::KernelArgs) {
        for dd in 0..D {
            args.push(&(self.bcell[dd].as_ptr() as u64));
        }
    }

    /// push bflux pointers for one direction to GPU kernel args.
    pub fn push_bflux_ptrs_to(&self, dir: usize, args: &mut symbi_xpu::KernelArgs) {
        for dd in 0..D {
            args.push(&(self.bflux[dir][dd].as_ptr() as u64));
        }
    }
}

/// RK workspace for one partition. `NDIM` = grid dim, `DOF` = vector component dim
/// the `RkWorkspace<D>` alias fills `DOF = NDIM = D`.
pub struct RkWorkspaceGeneric<
    const NDIM: usize,
    const DOF: usize,
    M: MemorySpace = DefaultMemory,
    Sc: Scalar + OrderedNumeric = f64,
> {
    pub u_n: ConsFieldsGeneric<NDIM, DOF, M, Sc>,
    pub prim_n: PrimFieldsGeneric<NDIM, DOF, M, Sc>,
    /// the per-STAGE conserved snapshot. distinct from `u_n` (the per-STEP `u^n`
    /// held for the `a0*u_n` SSP term): `u_stage` is the stage-INPUT cons, taken
    /// before each godunov stage so the additive source pass evaluates `S` at the
    /// same state the fused stage does — the bit-for-bit `fused == plain + additive`
    /// invariant (see `godunov_with_fused_source` S2 proof). dead weight unless an
    /// additive source overlay is active (the step loop gates the snapshot).
    pub u_stage: ConsFieldsGeneric<NDIM, DOF, M, Sc>,
    /// when set, every `u_stage` binding resolves to `u_n` instead. at the FIRST stage of a
    /// multi-stage SSP scheme `snapshot` has just copied `cons -> u_n` and nothing has touched cons
    /// since, so `u_n` IS the stage input — copying cons a second time into `u_stage` moves a
    /// full-grid conserved set for no information. `u_stage` is read-only once written (only
    /// `snapshot_stage` writes it), so the alias can never be written through. the driver sets it per
    /// stage; `binding.rs` is the single site that honours it.
    pub stage_input_is_un: std::sync::atomic::AtomicBool,
    /// disables the stage-0 alias, forcing the `cons -> u_stage` copy at every stage. the
    /// REFERENCE path: a reference run evolves the same state both ways and asserts a bit-identical
    /// trajectory, so the elision cannot silently change physics. `true` (elide) in production.
    pub elide_stage_snapshot: std::sync::atomic::AtomicBool,
    /// first-order flux-correction scratch: the HIGH-ORDER per-direction conserved fluxes, saved
    /// before FOFC redoes the substage at first order (which overwrites `fields.flux`). the
    /// face-based splice reads HO here and FO from the live `fields.flux`, choosing per face by the
    /// fallback flag so every face carries ONE flux -> the re-godunov telescopes conservatively.
    /// only touched when a substage fires FOFC; a no-op otherwise and for regimes without FOFC.
    pub flux_ho: [ConsFieldsGeneric<NDIM, DOF, M, Sc>; NDIM],
    /// the per-cell FOFC fallback flag over the allocated domain: 1 where the high-order c2p is
    /// unphysical, else 0, with boundary-consistent ghosts (a face is first-order iff either
    /// adjacent cell is flagged). the splice stencil reads it at the two cells sharing each face.
    pub fofc_flag: Field<Sc, NDIM, M>,
    /// body-feedback reduction scratch, allocated on first feedback dispatch (body-free sims
    /// never touch it, and pay neither the memory nor a per-call allocation). the feedback
    /// kernels assign-write every cell of their dispatch region before the reduction reads
    /// it, so reuse across calls needs no re-zeroing. sized by the first caller — the split
    /// cartesian path needs D+5 fields, the combined curvilinear path MAX_SOURCE_BODIES*(D+5);
    /// a sim's geometry picks exactly one path for its lifetime.
    pub body_scratch: std::sync::OnceLock<Vec<Field<Sc, NDIM, M>>>,
}

/// the natural case: vector dimension == grid dimension.
pub type RkWorkspace<const D: usize, M = DefaultMemory, Sc = f64> = RkWorkspaceGeneric<D, D, M, Sc>;

// =============================================================================
// partition geometry
// =============================================================================

/// domain topology for one partition.
pub struct PartitionGeometry<const D: usize> {
    /// cell widths per axis (uniform grid approximation).
    pub dx: [f64; D],

    /// physical coordinate lower bounds.
    pub x_lo: [f64; D],

    /// full allocated domain (interior + ghost).
    pub allocated: Domain<D>,

    /// interior domain (owned cells, no ghosts).
    pub interior: Domain<D>,

    /// ghost zone width.
    pub ng: usize,

    /// the coordinate system (from the metric `M::geometry()`) — drives curvilinear
    /// kernel selection (`_sph` / `_cyl` suffix) and the geometric source terms.
    pub coords: symbi_geometry::Geometry,

    /// the spacetime background (from the metric `M::spacetime()`) — ORTHOGONAL to `coords`:
    /// `Minkowski` for every flat run, a curved variant (Schwarzschild, ...) for GR. drives the
    /// lapse / sqrt(gamma) densitization selector in the kernel; flat -> no-op.
    pub spacetime: symbi_geometry::Spacetime,

    /// the curved-spacetime runtime scalar params (from `M::spacetime_scalars()`), `(wire-name,
    /// value)` — e.g. `[("schwarzschild_mass", M)]`. EMPTY for flat. the godunov dispatch resolves
    /// the kernel's spacetime scalars (the lapse `schwarzschild_mass`) against this by name.
    pub spacetime_scalars: Vec<(String, f64)>,

    /// grid axis -> coordinate index map. identity for cartesian / spherical / 3D, so grid
    /// axis d IS coordinate d. the AMBIGUOUS case is the cylindrical 2D plane, where MHD
    /// carries a 3-vector B on a 2-axis grid (DOF > ndim) and the two physical planes are
    /// indistinguishable by DOF alone: r-z axisymmetric (`[0, 2]`, out-of-plane phi) vs r-phi
    /// disk (`[0, 1]`, out-of-plane z). this records WHICH plane, set at construction (default
    /// r-z for back-compat) and overridable via [`SimStateGeneric::with_cyl_plane`]. the MHD
    /// constrained-transport seam (`StaggerComplex` edges, the metric curl, the kernel-name
    /// suffix) reads it; hydro ignores it (DOF == ndim disambiguates the cyl plane there).
    pub axes: [usize; D],

    /// per-axis coordinate maps. when Some, overrides dx/x_lo for
    /// non-uniform grids (logarithmic radial spacing, etc.).
    pub maps: Option<[symbi_geometry::AxisMap; D]>,
}

/// the cylindrical 2D plane an MHD sim grids — the only place the grid-axis set is ambiguous
/// (both planes carry a 3-vector B, so DOF can't tell them apart). `RZ` = axisymmetric (r, z),
/// out-of-plane swirl phi; `RPhi` = the disk (r, phi), out-of-plane vertical z.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CylPlane {
    Rz,
    RPhi,
}

/// the default grid-axis -> coordinate map: identity everywhere EXCEPT the cylindrical 2D grid,
/// which defaults to r-z (`[0, 2]`) — the established axisymmetric convention (back-compat).
pub fn default_grid_axes<const D: usize>(coords: symbi_geometry::Geometry) -> [usize; D] {
    match coords {
        symbi_geometry::Geometry::Cylindrical if D == 2 => {
            std::array::from_fn(|d| if d == 0 { 0 } else { 2 })
        }
        _ => std::array::from_fn(|d| d),
    }
}

/// a fluent builder for [`SimStateGeneric`] — named setters + sane defaults in place of `new`'s
/// 11 positional args (and its bare `ng` / `device_id` magic numbers + 3D-shaped `[bc; 6]`).
/// start with `SimStateGeneric::<..>::build(regime, eos, metric)`, set the grid + domain, and
/// `finish()`. defaults: `ghosts(2)`, `cfl(0.4)`, RK2, outflow boundaries, device 0. the type
/// parameters still come from the `Sim` alias; the builder removes the call-site hazards.
pub struct SimBuilder<R, const D: usize, const DOF: usize, M, E, S, Mem, Sc, St = NeedsGrid>
where
    R: Regime<Sc, D>,
    M: Metric<Sc, D>,
    E: Eos<Sc>,
    S: ExecutionSpace,
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    regime: R,
    eos: E,
    metric: M,
    n_cells: Option<[usize; D]>,
    x_lo: [f64; D],
    dx: Option<[f64; D]>,
    bounds_hi: Option<[f64; D]>,
    ng: usize,
    boundaries: Boundaries<D>,
    cfl: f64,
    timestepping: Timestepping,
    device_id: i64,
    cyl_plane: Option<CylPlane>,
    /// per-axis non-uniform coordinate maps (e.g. log-radial spacing). `None` -> the uniform
    /// `x_lo + i*dx` grid (bit-identical to the prior behavior).
    coord_maps: Option<[symbi_geometry::AxisMap; D]>,
    /// the partially-built sim, present once `allocate` has run (states `NeedsCells` / `Ready`).
    /// `None` in the `NeedsGrid` config phase.
    sim: Option<SimStateGeneric<R, D, DOF, M, E, S, Mem, Sc>>,
    _marker: std::marker::PhantomData<(S, Mem, Sc, St)>,
}

/// typestate markers for [`SimBuilder`] (safe-path-only frontend). the builder moves
/// `NeedsGrid -> NeedsCells -> Ready`; `build()` is callable ONLY at `Ready`, so a sim with
/// un-seeded fields (or un-seeded MHD faces) is unrepresentable at the type level.
pub struct NeedsGrid;
/// grid allocated, conserved fields not yet seeded.
pub struct NeedsCells;
/// fully seeded (cells, and for MHD the staggered faces) — ready to `build()`.
pub struct Ready;

impl<R, const D: usize, const DOF: usize, M, E, S, Mem, Sc>
    SimBuilder<R, D, DOF, M, E, S, Mem, Sc, NeedsGrid>
where
    R: Regime<Sc, D>,
    M: Metric<Sc, D>,
    E: Eos<Sc>,
    S: ExecutionSpace,
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    /// the grid resolution (interior cell count per axis). REQUIRED.
    pub fn cells(mut self, n: [usize; D]) -> Self {
        self.n_cells = Some(n);
        self
    }
    /// the physical origin (coordinate of index 0). default `[0; D]`.
    pub fn origin(mut self, x_lo: [f64; D]) -> Self {
        self.x_lo = x_lo;
        self
    }
    /// the cell widths directly. mutually exclusive with [`Self::bounds`] (last one wins).
    pub fn spacing(mut self, dx: [f64; D]) -> Self {
        self.dx = Some(dx);
        self.bounds_hi = None;
        self
    }
    /// per-axis coordinate maps for a non-uniform grid (log-radial spacing). `None` keeps the
    /// uniform grid. when set, the curvilinear kernel selects its `_logr` variant and reads the
    /// log decade-slope as the per-axis dx parameter (the host geometry honors the maps directly).
    pub fn coord_maps(mut self, maps: Option<[symbi_geometry::AxisMap; D]>) -> Self {
        self.coord_maps = maps;
        self
    }
    /// the physical box `[lo, hi]` per axis — `dx` is derived as `(hi - lo) / cells` at `finish`
    /// (so call `cells` too). the ergonomic common case: the domain is known and the spacing is derived.
    pub fn bounds(mut self, lo: [f64; D], hi: [f64; D]) -> Self {
        self.x_lo = lo;
        self.bounds_hi = Some(hi);
        self.dx = None;
        self
    }
    /// ghost-zone width. default 2.
    pub fn ghosts(mut self, ng: usize) -> Self {
        self.ng = ng;
        self
    }
    /// the per-axis boundary conditions. accepts `Boundaries<D>` or a bare `BoundaryType` (uniform)
    /// via `Into`. `Boundaries::per_axis([..]).axis(..)` for non-uniform (e.g., periodic-z, outflow-r).
    pub fn boundaries(mut self, bc: impl Into<Boundaries<D>>) -> Self {
        self.boundaries = bc.into();
        self
    }
    /// the CFL number. default 0.4.
    pub fn cfl(mut self, cfl: f64) -> Self {
        self.cfl = cfl;
        self
    }
    /// the time integrator. default RK2.
    pub fn timestepping(mut self, t: Timestepping) -> Self {
        self.timestepping = t;
        self
    }
    /// the CUDA device id (unused on CPU). default 0.
    pub fn device(mut self, id: i64) -> Self {
        self.device_id = id;
        self
    }
    /// select the cylindrical 2D MHD plane (r-z vs r-phi disk) — see [`SimStateGeneric::with_cyl_plane`].
    pub fn cyl_plane(mut self, plane: CylPlane) -> Self {
        self.cyl_plane = Some(plane);
        self
    }

    /// construct the simulation. errors if `cells` was never set, or if neither `spacing` nor
    /// `bounds` was given. applies the cyl-plane selection if requested.
    pub fn finish(self) -> symbi_xpu::Result<SimStateGeneric<R, D, DOF, M, E, S, Mem, Sc>> {
        let n = self.n_cells.expect("SimBuilder: .cells([..]) is required");
        let dx = self.dx.unwrap_or_else(|| {
            let hi = self
                .bounds_hi
                .expect("SimBuilder: set the spacing via .spacing([..]) or .bounds(lo, hi)");
            std::array::from_fn(|d| (hi[d] - self.x_lo[d]) / n[d] as f64)
        });
        let sim = SimStateGeneric::new(
            self.regime,
            self.eos,
            self.metric,
            n,
            self.x_lo,
            dx,
            self.ng,
            self.boundaries,
            self.cfl,
            self.timestepping,
            self.device_id,
        )?;
        Ok(match self.cyl_plane {
            Some(p) => sim.with_cyl_plane(p),
            None => sim,
        })
    }

    /// validate the grid config, then ALLOCATE the sim's fields — the typestate gate from the
    /// config phase (`NeedsGrid`) into the seeding phase (`NeedsCells`). errors BEFORE any
    /// allocation if `cells` is unset, neither `spacing` nor `bounds` was given, or any of
    /// cells / spacing / cfl is non-positive. on success the partially-built sim is carried in
    /// the returned `NeedsCells` builder (cyl-plane applied), ready for `set_initial` / `seed_faces`.
    pub fn allocate(
        self,
    ) -> Result<SimBuilder<R, D, DOF, M, E, S, Mem, Sc, NeedsCells>, ConfigError>
    where
        E: Copy,
        M: Copy,
    {
        let n = self.n_cells.ok_or(ConfigError::MissingCells)?;
        let dx = match self.dx {
            Some(dx) => dx,
            None => {
                let hi = self.bounds_hi.ok_or(ConfigError::MissingSpacing)?;
                std::array::from_fn(|d| (hi[d] - self.x_lo[d]) / n[d] as f64)
            }
        };
        for d in 0..D {
            if n[d] == 0 {
                return Err(ConfigError::NonPositive {
                    field: "cells",
                    value: 0.0,
                });
            }
            if !(dx[d] > 0.0) {
                return Err(ConfigError::NonPositive {
                    field: "spacing",
                    value: dx[d],
                });
            }
        }
        if !(self.cfl > 0.0) {
            return Err(ConfigError::NonPositive {
                field: "cfl",
                value: self.cfl,
            });
        }
        let sim = SimStateGeneric::new(
            self.regime,
            self.eos,
            self.metric,
            n,
            self.x_lo,
            dx,
            self.ng,
            self.boundaries,
            self.cfl,
            self.timestepping,
            self.device_id,
        )
        .map_err(ConfigError::Alloc)?;
        let sim = match self.cyl_plane {
            Some(p) => sim.with_cyl_plane(p),
            None => sim,
        };
        let sim = match self.coord_maps {
            Some(maps) => sim.with_maps(maps),
            None => sim,
        };
        Ok(SimBuilder {
            regime: self.regime,
            eos: self.eos,
            metric: self.metric,
            n_cells: self.n_cells,
            x_lo: self.x_lo,
            dx: Some(dx),
            bounds_hi: self.bounds_hi,
            ng: self.ng,
            boundaries: self.boundaries,
            cfl: self.cfl,
            timestepping: self.timestepping,
            device_id: self.device_id,
            cyl_plane: self.cyl_plane,
            coord_maps: self.coord_maps,
            sim: Some(sim),
            _marker: std::marker::PhantomData,
        })
    }
}

// =============================================================================
// typestate seeding phase: NeedsCells -> (NeedsCells) -> Ready.
//
// the builder owns the allocated sim from here. `set_initial` seeds the cell-centered state via
// the `seed_cells` / `seed_cell` internals; for MHD it leaves the staggered
// faces owed (still NeedsCells) until `seed_faces` sets them (and the bface_initialized flag).
// `build()` is reachable ONLY at Ready, so an un-seeded sim (or un-seeded MHD faces) can't be built.
// =============================================================================

/// the typestate `set_initial` lands in, keyed on the regime's conserved state: pure hydro
/// (`ConsG`) is fully seeded -> `Ready`; MHD (`MhdConsG`) still owes the staggered faces ->
/// `NeedsCells`. ONE `set_initial` method routes through this associated state, so the two cases
/// don't collide as duplicate inherent definitions. impl'd on the two CONCRETE cons types (no
/// blanket, no coherence overlap). the empty `Magnetic`/`NonMagnetic` markers (symbi-hydro) carry
/// the same hydro-vs-mhd distinction for the `seed_faces` gating.
pub trait AfterSetInitial {
    type State;
}
impl<Sc: Scalar, const N: usize, En: EnergyModel> AfterSetInitial for ConsG<Sc, N, En> {
    type State = Ready;
}
impl<Sc: Scalar, const N: usize, En: EnergyModel> AfterSetInitial
    for symbi_hydro::mhd_state::MhdConsG<Sc, N, En>
{
    type State = NeedsCells;
}

impl<R, const D: usize, const DOF: usize, M, E, S, Mem, Sc, St>
    SimBuilder<R, D, DOF, M, E, S, Mem, Sc, St>
where
    R: Regime<Sc, D>,
    M: Metric<Sc, D>,
    E: Eos<Sc>,
    S: ExecutionSpace,
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    /// rebuild the builder in a new typestate, carrying every field through unchanged. the ONE
    /// state-transition seam — pure phantom retag, no allocation.
    fn retag<St2>(self) -> SimBuilder<R, D, DOF, M, E, S, Mem, Sc, St2> {
        SimBuilder {
            regime: self.regime,
            eos: self.eos,
            metric: self.metric,
            n_cells: self.n_cells,
            x_lo: self.x_lo,
            dx: self.dx,
            bounds_hi: self.bounds_hi,
            ng: self.ng,
            boundaries: self.boundaries,
            cfl: self.cfl,
            timestepping: self.timestepping,
            device_id: self.device_id,
            cyl_plane: self.cyl_plane,
            coord_maps: self.coord_maps,
            sim: self.sim,
            _marker: std::marker::PhantomData,
        }
    }
}

// cell seeding (any regime): set_initial routes to Ready (hydro) or NeedsCells (MHD) via the
// AfterSetInitial associated state. ONE method, no duplicate-definition clash.
impl<R, const D: usize, const DOF: usize, M, E, S, Mem, Sc>
    SimBuilder<R, D, DOF, M, E, S, Mem, Sc, NeedsCells>
where
    R: Regime<Sc, D> + Regime<Sc, DOF>,
    M: Metric<Sc, D> + Metric<Sc, DOF>,
    E: Eos<Sc>,
    S: ExecutionSpace,
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
    <R as Regime<Sc, DOF>>::Cons: SeedableCons<Sc, DOF> + AfterSetInitial,
{
    /// seed EVERY interior cell from a primitive closure over the cell CENTER coordinate. routes to
    /// `Ready` (pure hydro — fully seeded) or `NeedsCells` (MHD — faces still owed) via the
    /// `AfterSetInitial` associated state. routes through `SimStateGeneric::seed_cells`.
    pub fn set_initial(
        self,
        prim_at: impl Fn([f64; D]) -> <R as Regime<Sc, DOF>>::Prim,
    ) -> SimBuilder<
        R,
        D,
        DOF,
        M,
        E,
        S,
        Mem,
        Sc,
        <<R as Regime<Sc, DOF>>::Cons as AfterSetInitial>::State,
    > {
        self.sim
            .as_ref()
            .expect("NeedsCells builder carries an allocated sim")
            .seed_cells(prim_at);
        self.retag()
    }

    /// seed every interior cell from a closure over its (index, center-coordinate) — for index-based
    /// ICs (e.g., kh-noise). same state routing as `set_initial`.
    pub fn set_initial_indexed(
        self,
        prim_at: impl Fn([isize; D], [f64; D]) -> <R as Regime<Sc, DOF>>::Prim,
    ) -> SimBuilder<
        R,
        D,
        DOF,
        M,
        E,
        S,
        Mem,
        Sc,
        <<R as Regime<Sc, DOF>>::Cons as AfterSetInitial>::State,
    > {
        {
            let sim = self
                .sim
                .as_ref()
                .expect("NeedsCells builder carries an allocated sim");
            for c in sim.geom.interior.iter() {
                let x = sim.geom.cell_coord(c);
                sim.seed_cell(c, &prim_at(c, x));
            }
        }
        self.retag()
    }
}

// MHD face seeding (Cons: Magnetic): seed_faces / seed_faces_uniform set the staggered faces (+
// the bface_initialized flag) and reach Ready. distinct method names from set_initial — no clash.
impl<R, const D: usize, const DOF: usize, M, E, S, Mem, Sc>
    SimBuilder<R, D, DOF, M, E, S, Mem, Sc, NeedsCells>
where
    R: Regime<Sc, D> + Regime<Sc, DOF>,
    M: Metric<Sc, D>,
    E: Eos<Sc>,
    S: ExecutionSpace,
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
    <R as Regime<Sc, DOF>>::Cons: SeedableCons<Sc, DOF> + symbi_hydro::state::Magnetic,
{
    /// seed the staggered face-normal B `bface[d]` from a closure over the face's physical position,
    /// for every axis d, then reach `Ready`. sets `bface_initialized` (via `seed_face_with`), so the
    /// CT has its divergence-free ground truth.
    pub fn seed_faces(
        self,
        f: impl Fn(usize, [f64; D]) -> Sc,
    ) -> SimBuilder<R, D, DOF, M, E, S, Mem, Sc, Ready> {
        {
            let sim = self
                .sim
                .as_ref()
                .expect("NeedsCells builder carries an allocated sim");
            for d in 0..D {
                sim.seed_face_with(d, |x| f(d, x));
            }
        }
        self.retag()
    }

    /// seed each face-normal B `bface[d]` to a UNIFORM value `b0[d]` (the common case — a uniform
    /// field threading the domain), then reach `Ready`.
    pub fn seed_faces_uniform(self, b0: [Sc; D]) -> SimBuilder<R, D, DOF, M, E, S, Mem, Sc, Ready> {
        {
            let sim = self
                .sim
                .as_ref()
                .expect("NeedsCells builder carries an allocated sim");
            for d in 0..D {
                sim.seed_face(d, b0[d]);
            }
        }
        self.retag()
    }

    /// seed every staggered face `bface[d]` from a per-axis flat buffer (`faces[d]` feeds axis
    /// `d`, axis-0-fastest over the interior face domain), then reach `Ready` — the array analog
    /// of [`seed_faces`] for the python `staggered_bfields` generators.
    pub fn seed_faces_indexed(
        self,
        faces: &[Vec<Sc>],
    ) -> SimBuilder<R, D, DOF, M, E, S, Mem, Sc, Ready> {
        {
            let sim = self
                .sim
                .as_ref()
                .expect("NeedsCells builder carries an allocated sim");
            for d in 0..D {
                sim.seed_face_indexed(d, &faces[d]);
            }
        }
        self.retag()
    }
}

// build() is reachable ONLY at Ready: the conserved fields (and, for MHD, the staggered faces) are
// guaranteed seeded by the typestate.
impl<R, const D: usize, const DOF: usize, M, E, S, Mem, Sc>
    SimBuilder<R, D, DOF, M, E, S, Mem, Sc, Ready>
where
    R: Regime<Sc, D>,
    M: Metric<Sc, D>,
    E: Eos<Sc>,
    S: ExecutionSpace,
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    /// take the fully-seeded simulation.
    pub fn build(self) -> SimStateGeneric<R, D, DOF, M, E, S, Mem, Sc> {
        self.sim
            .expect("Ready builder carries an allocated, seeded sim")
    }
}

/// staggering of a single axis: a quantity sampled at the lower cell FACE (the
/// index plane `coord[ax]`) or at the cell CENTER (`coord[ax] + 1/2`). the per-axis
/// choice is the ONLY thing distinguishing a cell-centered field, a face-normal
/// field (`bface[d]`: Face on `d`, Center elsewhere), and an edge field — see
/// [`PartitionGeometry::stagger_coord`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loc {
    /// the lower cell face on this axis (index plane, no half-cell shift).
    Face,
    /// the cell center on this axis (half-cell shift).
    Center,
}

impl<const D: usize> PartitionGeometry<D> {
    /// physical centroid of a cell (every axis at [`Loc::Center`]).
    #[inline]
    pub fn centroid(&self, coord: [isize; D]) -> [f64; D] {
        self.stagger_coord(coord, [Loc::Center; D])
    }

    /// physical position of a STAGGERED quantity: per axis, [`Loc::Face`] samples the
    /// lower cell face and [`Loc::Center`] the cell center. this is the SINGLE source
    /// of the half-cell offset — every staggered IC reads its coordinates here, with no
    /// hand-written `coord*dx` vs `(coord+0.5)*dx` (the Orszag-Tang point-symmetry
    /// bug class). honors non-uniform maps (Face -> map.face, Center -> map.center).
    #[inline]
    pub fn stagger_coord(&self, coord: [isize; D], loc: [Loc; D]) -> [f64; D] {
        std::array::from_fn(|ax| match (self.maps.as_ref(), loc[ax]) {
            (Some(maps), Loc::Face) => maps[ax].face(coord[ax]),
            (Some(maps), Loc::Center) => maps[ax].center(coord[ax]),
            (None, Loc::Face) => self.x_lo[ax] + (coord[ax] as f64) * self.dx[ax],
            (None, Loc::Center) => self.x_lo[ax] + (coord[ax] as f64 + 0.5) * self.dx[ax],
        })
    }

    /// physical position of `bface[dir]`: on the lower-`dir` face, cell-centered in the
    /// transverse axes — the canonical staggering of a face-normal B-field.
    #[inline]
    pub fn face_coord(&self, coord: [isize; D], dir: usize) -> [f64; D] {
        self.stagger_coord(
            coord,
            std::array::from_fn(|ax| if ax == dir { Loc::Face } else { Loc::Center }),
        )
    }

    /// physical position of a CELL CENTER — the index->coordinate bridge an IC closure wants.
    /// map-aware (non-uniform grids) via `stagger_coord`. `[x_lo + (i+1/2)*dx]` on a uniform grid.
    #[inline]
    pub fn cell_coord(&self, coord: [isize; D]) -> [f64; D] {
        self.stagger_coord(coord, std::array::from_fn(|_| Loc::Center))
    }

    /// cell width along axis ax at a given index.
    #[inline]
    pub fn cell_width(&self, coord: [isize; D], ax: usize) -> f64 {
        if let Some(ref maps) = self.maps {
            maps[ax].width(coord[ax])
        } else {
            self.dx[ax]
        }
    }

    /// set per-axis coordinate maps for non-uniform grids.
    pub fn set_maps(&mut self, maps: [symbi_geometry::AxisMap; D]) {
        self.maps = Some(maps);
    }

    /// create a BlockGeometry that uses these maps (if set).
    pub fn block_geometry<M: symbi_geometry::Metric<f64, D> + Copy>(
        &self,
        metric: M,
    ) -> symbi_geometry::BlockGeometry<M, f64, D> {
        if let Some(maps) = self.maps {
            symbi_geometry::BlockGeometry::with_maps(metric, maps)
        } else {
            symbi_geometry::BlockGeometry::uniform(metric, self.x_lo, self.dx)
        }
    }
}

// =============================================================================
// mesh + boundary config
// =============================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BoundaryType {
    Periodic,
    Outflow,
    Reflect,
    /// filled by prolongation from a coarser AMR level.
    /// standard ghost fill skips these faces.
    CoarseFine,
    /// a DRIVEN boundary: the ghost state is PRESCRIBED by a user DAG
    /// (`build_boundary_dag`). the `u16` indexes the kernel-set's
    /// boundary-DAG side table. standard ghost fill SKIPS these faces (like `CoarseFine`); the
    /// driven-boundary pass fills them by evaluating the DAG over the face's ghost band. enum stays
    /// `Copy`/`Eq` — the DAG lives in the side table, only its id rides here.
    Driven(u16),
    /// a NEUMANN boundary: the ghost holds a prescribed OUTWARD normal derivative `dU/dn = q` per
    /// primitive variable, `U_ghost = u_edge + q*dist`. the `u16` indexes the kernel-set's
    /// gradient-BC side table (the per-variable coefficients). standard ghost fill SKIPS these faces;
    /// the gradient-boundary pass fills them from the boundary-adjacent interior cell. a convenience
    /// short-circuit for the classical prescribed-gradient wall — a custom boundary is the general path.
    Neumann(u16),
    /// a ROBIN boundary: the ghost enforces `a*U_face + b*dU/dn = c` per primitive variable. the
    /// `u16` indexes the same gradient-BC side table as `Neumann` (the entry carries the `(a,b,c)`
    /// triples). standard ghost fill SKIPS these faces; the gradient-boundary pass fills them.
    /// degenerates to Dirichlet (`b=0`) and Neumann (`a=0`).
    Robin(u16),
}

/// per-axis boundary conditions, `D`-shaped (not the 3D-padded `[BoundaryType; 6]`): one `[lo, hi]`
/// pair per axis. replaces the flat 6-array in `SimStateGeneric` / `SimBuilder` so the boundary
/// count is dimension-correct and the lo/hi accessors name the face explicitly.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Boundaries<const D: usize>(pub [[BoundaryType; 2]; D]);

impl<const D: usize> Boundaries<D> {
    /// ONE boundary type for every face (the common uniform case).
    pub const fn uniform(bc: BoundaryType) -> Self {
        Self([[bc; 2]; D])
    }
    /// per-axis `[lo, hi]` table.
    pub const fn per_axis(p: [[BoundaryType; 2]; D]) -> Self {
        Self(p)
    }
    /// override one axis's `[lo, hi]` pair (fluent).
    pub fn axis(mut self, ax: usize, lo: BoundaryType, hi: BoundaryType) -> Self {
        self.0[ax] = [lo, hi];
        self
    }
    /// the low-side boundary on axis `ax`.
    pub fn lo(&self, ax: usize) -> BoundaryType {
        self.0[ax][0]
    }
    /// the high-side boundary on axis `ax`.
    pub fn hi(&self, ax: usize) -> BoundaryType {
        self.0[ax][1]
    }
}

/// a bare `BoundaryType` means the uniform-all-faces case (`Boundaries::uniform`). lets the builder
/// `.boundaries(BoundaryType::Periodic)` shorthand coexist with the explicit `Boundaries<D>` form.
impl<const D: usize> From<BoundaryType> for Boundaries<D> {
    fn from(bc: BoundaryType) -> Self {
        Self::uniform(bc)
    }
}

/// the configuration errors `SimBuilder::allocate` surfaces BEFORE allocating fields (and the
/// solver/regime mismatch the phase-2 wiring will check). a typed-result config
/// seam.
#[derive(Debug)]
pub enum ConfigError {
    /// `.cells([..])` was never set.
    MissingCells,
    /// neither `.spacing([..])` nor `.bounds(lo, hi)` was given.
    MissingSpacing,
    /// a grid scalar (cells / spacing / cfl) was non-positive.
    NonPositive { field: &'static str, value: f64 },
    /// the requested Riemann solver is invalid for the regime family (see [`Solver::valid_for`]).
    SolverRegimeMismatch {
        solver: crate::substrate_seam::Solver,
        regime: crate::substrate_seam::RegimeKind,
    },
    /// the underlying xpu field allocation failed.
    Alloc(symbi_xpu::XpuError),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::MissingCells => write!(f, "config: .cells([..]) is required"),
            ConfigError::MissingSpacing => {
                write!(
                    f,
                    "config: set the spacing via .spacing([..]) or .bounds(lo, hi)"
                )
            }
            ConfigError::NonPositive { field, value } => {
                write!(f, "config: {field} must be positive (got {value:e})")
            }
            ConfigError::SolverRegimeMismatch { solver, regime } => {
                write!(
                    f,
                    "config: solver {solver:?} is invalid for regime {regime:?}"
                )
            }
            ConfigError::Alloc(e) => write!(f, "config: field allocation failed: {e}"),
        }
    }
}

impl std::error::Error for ConfigError {}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Timestepping {
    Euler,
    Rk2,
    Rk3,
}

impl Timestepping {
    /// the explicit SSP scheme as Shu-Osher convex coefficients: one `(a0, ac)` row per stage,
    /// where stage `i` updates `cons = a0*u_n + ac*(cons - dt*div(F) + dt*S)`. forward-Euler is
    /// the single row `(0, 1)`; SSP-RK2 (Heun) is the predictor `(0,1)` + the trapezoidal
    /// corrector `(1/2, 1/2)`; SSP-RK3 (Shu-Osher) adds the `(3/4, 1/4)` + `(1/3, 2/3)` rows.
    /// `a0 + ac == 1` for every row (SSP consistency). this table IS the integrator — the one
    /// `godunov_stage` kernel reads `(a0, ac)` as runtime scalars, so adding a scheme is a row,
    /// never a new kernel.
    pub fn stages(self) -> &'static [(f64, f64)] {
        match self {
            Self::Euler => &[(0.0, 1.0)],
            Self::Rk2 => &[(0.0, 1.0), (0.5, 0.5)],
            Self::Rk3 => &[(0.0, 1.0), (0.75, 0.25), (1.0 / 3.0, 2.0 / 3.0)],
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reconstruction {
    Pcm,
    Plm,
    // Weno,  // future
    // Ppm,   // future
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Solver {
    Hlle,
    Hllc,
    HllcLm,
    Hlld,
}

/// constrained transport method for edge E-field computation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CtMethod {
    /// CT-Contact (Gardiner & Stone 2005): upwind selection based on
    /// density flux sign. simple, moderate dissipation.
    Contact,
    /// UCT (Mignone & Del Zanna 2021, Eq. 30-33): upwind constrained
    /// transport using the master emf composition formula. the flux and
    /// diffusion coefficients are derived from whatever Riemann solver
    /// is selected (HLL coefficients from Eq. 32 for HLLE/HLLC, HLLD
    /// coefficients from Eq. 44-45 when HLLD is the base solver).
    Uct,
}

// =============================================================================
// simulation state
// =============================================================================

/// immersed-body side-car: present iff the sim has bodies (attached via `with_bodies`).
/// groups the body collection with its feedback accumulator so the body-free common case
/// carries neither, and "bodies without an accumulator" is unrepresentable. only the grid
/// dimension `NDIM` parametrizes it (the bodies are f64).
pub struct ImmersedBodies<const NDIM: usize> {
    pub bodies: symbi_ib::BodyCollection<f64, NDIM>,
    pub diagnostics: symbi_ib::DiagnosticAccumulator<f64, NDIM>,
    /// the per-step body-gas exchange series (Mdot(t) = mass_delta/dt, drag
    /// F_acc(t)) — appended by `evolve_bodies`, written into every checkpoint
    /// as the `body_diagnostics` group, consumed by the steady-state detector.
    /// restarts empty on load; earlier segments live in earlier checkpoints.
    pub history: symbi_ib::BodyHistory<NDIM>,
    /// per-body immersed-boundary SHAPE (body-local CSG signed distance), parallel to
    /// `bodies`. `None` = the analytic sphere of radius `body.radius` (the AOT penalization
    /// kernel). `Some(sdf)` = an arbitrary shape whose penalization kernel is runtime-built +
    /// JIT-compiled per distinct geometry (the sphere geometry is baked constants; the body
    /// position stays a runtime scalar, so a moving body rides the same kernel).
    pub shapes: Vec<Option<symbi_ib::sdf::SdfExpr<f64, 3>>>,
    /// the bonded-fragment pair physics (bonds + contact + mutual gravity),
    /// present iff the collection carries wall-only fragments. the fragment
    /// subcycle integrates fragment motion with the gas force/torque held
    /// frozen; the legacy body integrator owns only the source prefix.
    pub fragment_physics: Option<symbi_ib::FragmentPhysics>,
    /// accepted per-cell mass removal for each body during the latest
    /// penalization pass. each body row follows `geom.interior.iter()` order.
    accretion_receipts: std::sync::Mutex<Vec<Vec<f64>>>,
    /// the GLOBAL fast-magnetosonic Alfven stiffness c_a2 = max_interior |B|^2/rho for the wall
    /// relaxation. the max is a domain-global property, so under domain decomposition the
    /// decomposed loop reduces the per-tile maxima and publishes the global value here; the
    /// penalize then relaxes every tile's wall at the monolithic rate (a per-tile local max makes
    /// the same wall cell relax differently in a tile than in the monolithic run). NaN (the
    /// default) means "compute the local max", which IS the global max on a monolithic single
    /// grid. raw f64 bits for Sync interior mutability (set through the shared FieldStore inside
    /// the tiled loop).
    c_a2_override: std::sync::atomic::AtomicU64,
}

impl<const NDIM: usize> ImmersedBodies<NDIM> {
    /// the published global Alfven stiffness, or `None` if unset (compute the local max).
    pub fn c_a2_override(&self) -> Option<f64> {
        let v = f64::from_bits(
            self.c_a2_override
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        (!v.is_nan()).then_some(v)
    }
    /// publish the global Alfven stiffness (the decomposed loop's cross-tile max).
    pub fn set_c_a2_override(&self, v: f64) {
        self.c_a2_override
            .store(v.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }

    pub fn reset_accretion_receipts(&self, cell_count: usize) {
        let mut receipts = self.accretion_receipts.lock().unwrap();
        receipts.clear();
        receipts.resize_with(self.bodies.len(), || vec![0.0; cell_count]);
    }

    pub fn record_accretion_receipt(&self, body: usize, values: impl IntoIterator<Item = f64>) {
        let mut receipts = self.accretion_receipts.lock().unwrap();
        if let Some(row) = receipts.get_mut(body) {
            for (slot, value) in row.iter_mut().zip(values) {
                *slot = value;
            }
        }
    }

    pub fn accretion_receipts(&self) -> Vec<Vec<f64>> {
        self.accretion_receipts.lock().unwrap().clone()
    }
}

/// the LOCAL fast-magnetosonic Alfven stiffness c_a2 = max over this store's interior of
/// |B|^2/rho, the wall-relaxation rate lift for a magnetized immersed body. 0 off MHD (no
/// magnetic field), where the rate reduces to the sound speed exactly. on a monolithic single
/// grid this local max IS the global max; the decomposed loop reduces it across tiles and
/// publishes the global value via `ImmersedBodies::set_c_a2_override`.
pub fn local_c_a2_max<const NDIM: usize, const DOF: usize, Mem, Sc>(
    store: &FieldStore<NDIM, DOF, Mem, Sc>,
) -> f64
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    store.fields.mhd.as_ref().map_or(0.0, |mhd| {
        let den = store.fields.cons.den.view();
        let bcell: Vec<_> = (0..DOF).map(|k| mhd.bcell[k].view()).collect();
        let mut m = 0.0_f64;
        for c in store.geom.interior.iter() {
            let rho = (*den.at(c)).to_f64();
            if rho <= 0.0 {
                continue;
            }
            let mut bsq = 0.0;
            for view in &bcell {
                let b = (*view.at(c)).to_f64();
                bsq += b * b;
            }
            m = m.max(bsq / rho);
        }
        m
    })
}

/// the simulation's mutable SUBSTANCE: every buffer + grid + time-state a kernel reads
/// or writes. parametrized ONLY by storage shape — grid dim `NDIM`,
/// vector dim `DOF`, memory space `Mem`, scalar `Sc` — NOT by the physics tags (`R`/`M`/`E`)
/// or the executor (`S`). this is the keystone decoupling: a `KernelSet`
/// takes `&FieldStore` and so carries 4 params, and the energy/schema bounds
/// that ripple off `R` become LOCAL to this one struct; without it they would be an 80-site sweep.
pub struct FieldStore<
    const NDIM: usize,
    const DOF: usize,
    Mem: MemorySpace = DefaultMemory,
    Sc: Scalar + OrderedNumeric = f64,
> {
    // ---- fields (allocated once, live forever) ----
    pub fields: PartitionFieldsGeneric<NDIM, DOF, Mem, Sc>,
    pub workspace: RkWorkspaceGeneric<NDIM, DOF, Mem, Sc>,

    // ---- geometry ----
    pub geom: PartitionGeometry<NDIM>,
    pub boundaries: Boundaries<NDIM>,

    /// discrete mass-transport tracers. position is derived output state;
    /// material ownership is authoritative.
    pub tracers: Option<crate::tracers::TracerSet<NDIM>>,
    /// continuous-position ito tracers in the store's execution memory.
    pub continuous_tracers: Option<crate::tracers::ContinuousTracerSet<NDIM, Mem>>,
    /// accepted-flux moment rates consumed by continuous ito tracers.
    pub ito_coefficients: Option<crate::tracers::ItoCoefficientFields<NDIM, Mem>>,
    /// full-step accepted mass-transfer receipt accumulated across hydro stages.
    pub ito_transport: Option<crate::tracers::ItoTransportReceipt<NDIM, Mem>>,

    // ---- time state ----
    pub time: f64,
    pub dt: f64,
    /// an upper clamp on the CFL time step (`dt = min(dt_cfl, max_dt)`); 0 disables. pins the
    /// dt SEQUENCE across runs whose CFL estimators differ (kernel cross-validation, temporal
    /// convergence studies) — two clamped runs from the same state take bitwise-identical steps.
    pub max_dt: f64,
    pub iteration: u64,
    pub cfl: f64,
    pub timestepping: Timestepping,

    // ---- mesh motion (ALE) ----
    pub motion: MotionState<f64>,
    // traced scale-factor law a(t)/a_dot(t); when present the time loop evaluates it EXACTLY each
    // (sub)stage. None = static / linear.
    pub motion_law: Option<symbi_hydro::motion_law::MotionLaw>,

    // ---- immersed bodies (optional side-car) ----
    pub immersed: Option<ImmersedBodies<NDIM>>,
}

/// the TYPE-LEVEL physics tags: regime, metric, eos. pure config — never read by a kernel
/// dispatch (the concrete `KernelSet` bakes `R::SPEC` / `eos_param` at construction); the
/// sim-level helpers (`seed_cell`, `cons_at`, `to_conserved`) read them. holding them apart
/// from `FieldStore` is what keeps the `R::Energy` / `R::Schema` bounds off the kernel path.
pub struct Physics<R, M, E> {
    pub regime: R,
    pub metric: M,
    pub eos: E,
}

/// the execution side-car: the only `S`-bearing state. split out so the `ExecutionSpace`
/// param touches neither the storage nor the kernel signatures.
pub struct Context<S: ExecutionSpace> {
    pub exec: Executor<S>,
}

/// complete simulation state = `FieldStore` (substance) + `Physics` (tags) + `Context`
/// (executor). generic over regime, GRID dim (`NDIM`),
/// VECTOR dim (`DOF`), metric, eos. the `SimState<R, D, M, ..>` alias fills
/// `DOF = NDIM = D` (the natural case); axisymmetric hydro uses `SimStateGeneric<R, 2, 3,
/// Cylindrical, ..>` directly (2D grid, 3-vector momentum with the v_phi swirl).
///
impl<const NDIM: usize, const DOF: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric>
    FieldStore<NDIM, DOF, Mem, Sc>
{
    /// THE stage-INPUT conserved set — the state an SSP stage's sources and its FOFC fallback must
    /// evaluate against. it is `u_n` at the first stage of a multi-stage scheme (where `snapshot`
    /// has already captured it and the driver elides the redundant `cons -> u_stage` copy), and the
    /// `u_stage` snapshot otherwise. every reader routes here; branching on
    /// `stage_input_is_un` at each call site is how a buffer alias drifts into a correctness bug.
    /// whether this simulation has immersed bodies (an attached collection
    /// with at least one body — `immersed.is_some()` alone disagrees with this
    /// in the zero-body-collection state the historical parse bug produced).
    pub fn has_bodies(&self) -> bool {
        self.immersed
            .as_ref()
            .map_or(false, |im| !im.bodies.is_empty())
    }

    /// whether this run carries mass-transport tracers.
    #[inline]
    pub fn has_tracers(&self) -> bool {
        self.tracers.is_some() || self.continuous_tracers.is_some()
    }

    /// whether this run carries the passive scalar (dye): the cons `chi` slot is
    /// allocated. every chi consumer gates on this, so an undyed run pays nothing.
    #[inline]
    pub fn has_passive_scalar(&self) -> bool {
        self.fields.cons.chi_field().is_some()
    }

    #[inline]
    pub fn stage_input(&self) -> &ConsFieldsGeneric<NDIM, DOF, Mem, Sc> {
        if self
            .workspace
            .stage_input_is_un
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            &self.workspace.u_n
        } else {
            &self.workspace.u_stage
        }
    }

    /// the stage-INPUT cell B — the CT twin of [`Self::stage_input`]: `bcell_n`
    /// (the step-entry snapshot, written unconditionally with `u_n`) at the first
    /// stage of a multi-stage scheme, where the driver elides `snapshot_stage`
    /// (which is what captures `bcell -> bcell_stage`); the `bcell_stage`
    /// snapshot otherwise. the MHD FOFC redo restores `bcell` from this so the
    /// recomputed edge EMF reads the true stage-input field — a direct
    /// `bcell_stage` read at stage 0 hands it a stale buffer.
    pub fn bcell_stage_input(&self) -> &crate::state::BcellFields<NDIM, DOF, Mem, Sc> {
        let mhd = self
            .fields
            .mhd
            .as_ref()
            .expect("bcell_stage_input requires MHD fields");
        if self
            .workspace
            .stage_input_is_un
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            &mhd.bcell_n
        } else {
            &mhd.bcell_stage
        }
    }
}

/// **the storage seam:** `SimStateGeneric` `Deref`s to its `FieldStore`. this is a
/// DELIBERATE seam; the `Deref` is not inheritance: the `FieldStore` IS the sim's
/// substance (1300+ `sim.fields` / `sim.geom` / `sim.time` accesses), while `physics` /
/// `ctx` are rare type-level side-cars reached explicitly (`sim.physics.regime`,
/// `sim.ctx.exec`). routing the substance through ONE target keeps every storage access —
/// and every `kernels.flux(sim, ..)` (which coerces `&Sim -> &FieldStore`) — unchanged.
pub struct SimStateGeneric<
    R: Regime<Sc, NDIM>,
    const NDIM: usize,
    const DOF: usize,
    M: Metric<Sc, NDIM>,
    E: Eos<Sc>,
    S: ExecutionSpace = DefaultSpace,
    Mem: MemorySpace = DefaultMemory,
    Sc: Scalar + OrderedNumeric = f64,
> {
    // ---- mutable substance: the only thing kernels see ----
    pub store: FieldStore<NDIM, DOF, Mem, Sc>,
    // ---- type-level physics tags ----
    pub physics: Physics<R, M, E>,
    // ---- execution side-car ----
    pub ctx: Context<S>,
}

impl<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, Sc> std::ops::Deref
    for SimStateGeneric<R, NDIM, DOF, M, E, S, Mem, Sc>
where
    R: Regime<Sc, NDIM>,
    M: Metric<Sc, NDIM>,
    E: Eos<Sc>,
    S: ExecutionSpace,
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    type Target = FieldStore<NDIM, DOF, Mem, Sc>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.store
    }
}

impl<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, Sc> std::ops::DerefMut
    for SimStateGeneric<R, NDIM, DOF, M, E, S, Mem, Sc>
where
    R: Regime<Sc, NDIM>,
    M: Metric<Sc, NDIM>,
    E: Eos<Sc>,
    S: ExecutionSpace,
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.store
    }
}

/// the natural case: vector dimension == grid dimension (all existing sites use this).
pub type SimState<R, const D: usize, M, E, S = DefaultSpace, Mem = DefaultMemory, Sc = f64> =
    SimStateGeneric<R, D, D, M, E, S, Mem, Sc>;

// =============================================================================
// construction
// =============================================================================

/// helper: create [Field<f64, D, M, C>; D] with zeros (workaround for
/// unstable try_from_fn). centering defaults to Cell so existing callers
/// keep working unchanged; pass an explicit C turbofish for face / edge
/// arrays.
// `N` component fields on a `D`-dimensional grid. `N` is the VECTOR (component) dimension,
// decoupled from the grid `D` — `N == D` for the natural case, `N > D` for an
// axisymmetric vector (the v_phi swirl on an (r,z) grid). `N` is inferred from the target array.
pub fn array_field_zeros<
    const D: usize,
    const N: usize,
    M: MemorySpace,
    C: symbi_grid::centering::Centering,
    Sc: Scalar + OrderedNumeric,
>(
    domain: &Domain<D>,
) -> symbi_xpu::Result<[Field<Sc, D, M, C>; N]> {
    let mut fields: Vec<Field<Sc, D, M, C>> = Vec::with_capacity(N);
    for _ in 0..N {
        fields.push(Field::<Sc, D, M, C>::zeros(domain)?);
    }
    Ok(fields.try_into().unwrap_or_else(|_| unreachable!()))
}

/// helper: create [ConsFields<D>; D] with optional energy field.
fn array_cons_zeros_with_energy<
    const D: usize,
    const DOF: usize,
    M: MemorySpace,
    Sc: Scalar + OrderedNumeric,
>(
    domain: &Domain<D>,
    has_energy: bool,
) -> symbi_xpu::Result<[ConsFieldsGeneric<D, DOF, M, Sc>; D]> {
    let mut fields = Vec::with_capacity(D);
    for _ in 0..D {
        fields.push(ConsFieldsGeneric::zeros_with_energy(domain, has_energy)?);
    }
    Ok(fields.try_into().unwrap_or_else(|_| unreachable!()))
}

pub fn axis_name(ax: usize) -> &'static str {
    match ax {
        0 => "i",
        1 => "j",
        2 => "k",
        _ => panic!("axis out of range"),
    }
}

// the uniform IC seam: seed a cell from a primitive, regime-agnostically. invokes the
// regime at the VECTOR dimension DOF (not the grid dim) so the state vector is full even
// on a 1.5D/2.5D grid, then routes through to_conserved + the EnergyModel-generic
// scatter_from + (for MHD) bcell <- the magnetic 3-vector. ONE entry point for every
// regime/EOS — no hand-built `Cons { den, mom, nrg }`, no iso-vs-adiabatic improvisation.
impl<R, const D: usize, const DOF: usize, M, E, S, Mem, Sc>
    SimStateGeneric<R, D, DOF, M, E, S, Mem, Sc>
where
    R: Regime<Sc, D> + Regime<Sc, DOF>,
    M: Metric<Sc, D> + Metric<Sc, DOF>,
    E: Eos<Sc>,
    S: ExecutionSpace,
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
    <R as Regime<Sc, DOF>>::Cons: symbi_hydro::state::SeedableCons<Sc, DOF>,
{
    /// seed one cell from a primitive: conserved gas state <- to_conserved(prim) (scattered
    /// via the EnergyModel-generic scatter_from), and — for MHD regimes — the cell-centered B
    /// <- the primitive's magnetic 3-vector. the staggered bface is seeded separately by the
    /// IC (face values are not a function of a single cell's primitive).
    ///
    /// on a curved spacetime the conserved momentum is the Valencia COVARIANT `S_i = rho h W^2
    /// gamma_ij v^j`, so the seed evaluates the spatial metric at the cell and stores the covariant
    /// state (via `to_conserved_covariant`) — the metric radius is the VOLUME-WEIGHTED radial
    /// centroid, the SAME point the metric-aware c2p inverts at, so the storage↔recovery round-trip
    /// is exact per cell. flat (Minkowski) keeps the orthonormal `to_conserved`.
    pub fn seed_cell(&self, coord: [isize; D], prim: &<R as Regime<Sc, DOF>>::Prim) {
        use symbi_hydro::spatial_metric::{Gamma, GammaInv, SpatialMetric};
        use symbi_hydro::state::SeedableCons;
        let cons = if matches!(self.geom.spacetime, symbi_geometry::Spacetime::Minkowski) {
            <R as Regime<Sc, DOF>>::to_conserved(&self.physics.regime, &self.physics.eos, prim)
        } else {
            // the metric point must be the SAME point the in-kernel geometry evaluates at
            // (`cell_geometry_gv`), so the covariant storage <-> metric-aware c2p round-trip
            // is exact per cell. the spelling is CHART-dependent:
            //   cartesian  — the face midpoint (lo + hi)/2 on every gridded axis; slot 0 is
            //                a plain length coordinate (treating it as a radius and applying the
            //                spherical volume-weighted formula to x mislocates the metric by
            //                O(dx^2/x), and degenerates on a sign-straddling cell of an
            //                origin-containing box).
            //   spherical  — the volume-weighted radial centroid r_vw = (3/4)(rh^4 - rl^4)/
            //                (rh^3 - rl^3) from the cell's radial faces (map-aware, so
            //                log-radial grids match too); angular slots take cell centers.
            let x_dof: Tensor<Sc, DOF> = if matches!(
                <M as Metric<Sc, DOF>>::geometry(&self.physics.metric),
                symbi_geometry::Geometry::Cartesian
            ) {
                Tensor::new(std::array::from_fn(|k| {
                    if k < D {
                        let lo = self.geom.face_coord(coord, k)[k];
                        let mut coord_hi = coord;
                        coord_hi[k] += 1;
                        let hi = self.geom.face_coord(coord_hi, k)[k];
                        Sc::from_f64((lo + hi) * 0.5)
                    } else {
                        // the cartesian kerr-schild D = 2 instantiation is the z = 0
                        // equatorial slice; an ungridded slot sits on the plane.
                        Sc::ZERO
                    }
                }))
            } else {
                let chart = <M as Metric<Sc, DOF>>::geometry(&self.physics.metric);
                // EVERY gridded slot takes its moment from the shared owner the in-kernel
                // geometry uses — under the chart's volume element the radial weight differs
                // between spherical (r^2 dr) and cylindrical (R dR), and the spherical polar slot
                // is sin-weighted rather than centered.
                // grid axis `d` carries COORDINATE slot `axes[d]`, which is not the identity on
                // the cylindrical r-z plane (axes = [0, 2]: grid axis 1 is the z coordinate,
                // slot 2). indexing the slot by the grid axis puts z in the azimuthal slot and
                // leaves slot 2 at zero, so a metric forming the spherical radius from the
                // cylindrical pair, r = sqrt(R^2 + z^2), reads r = R everywhere off the
                // midplane. the centroid moment is likewise a property of the COORDINATE, not
                // of the grid axis.
                // the moment depends on the MEASURE the scheme's cell average is taken against.
                // the area-weighted law integrates the chart's volume element and reads the
                // volume-weighted centroid; the densitized relativistic-hydro law integrates the
                // plain coordinate volume (its measure rides inside the conserved variable) and
                // reads the arithmetic midpoint, which is what the in-kernel c2p undensitizes at.
                let densitized = self.fields.mhd.is_none();
                let centroid_of = |d: usize, slot: usize| -> f64 {
                    let lo = self.geom.face_coord(coord, d)[d];
                    let mut coord_hi = coord;
                    coord_hi[d] += 1;
                    let hi = self.geom.face_coord(coord_hi, d)[d];
                    if densitized {
                        (lo + hi) * 0.5
                    } else {
                        symbi_geometry::volume_weighted_centroid(chart, slot, lo, hi)
                    }
                };
                Tensor::new(std::array::from_fn(|slot| {
                    match self.geom.axes.iter().position(|&a| a == slot) {
                        Some(d) => Sc::from_f64(centroid_of(d, slot)),
                        // an ungridded POLAR slot (DOF-lifted vectors on a 1D radial grid) takes
                        // the exact equatorial pi/2; zero would degenerate
                        // gamma_{phi phi} = r^2 sin^2(theta). every other ungridded slot is a
                        // symmetry direction the metric never reads.
                        None if slot == 1 && chart == symbi_geometry::Geometry::Spherical => {
                            Sc::from_f64(std::f64::consts::FRAC_PI_2)
                        }
                        None => Sc::ZERO,
                    }
                }))
            };
            let sm = SpatialMetric::new(
                Gamma::new(<M as Metric<Sc, DOF>>::spatial_metric(
                    &self.physics.metric,
                    x_dof,
                )),
                GammaInv::new(<M as Metric<Sc, DOF>>::spatial_metric_inv(
                    &self.physics.metric,
                    x_dof,
                )),
            );
            let alpha = <M as Metric<Sc, DOF>>::lapse(&self.physics.metric, x_dof);
            let shift = <M as Metric<Sc, DOF>>::shift(&self.physics.metric, x_dof);
            // the FULL-chart spatial measure, including the directions the momentum block
            // suppresses: `volume_factor` is r^2 sin(theta)/sqrt(f) on a 1D radial spherical grid
            // where the 1x1 determinant would give only 1/sqrt(f). paired with the lapse it is the
            // sqrt(-g) the densitized relativistic state carries.
            let sqrt_gamma = <M as Metric<Sc, DOF>>::volume_factor(&self.physics.metric, x_dof);
            <R as Regime<Sc, DOF>>::to_conserved_covariant(
                &self.physics.regime,
                &self.physics.eos,
                prim,
                &sm,
                alpha,
                shift,
                sqrt_gamma,
            )
        };
        self.fields.cons.scatter_from(coord, cons.hydro_part());
        if let Some(mag) = cons.mag_part() {
            if let Some(mhd) = self.fields.mhd.as_ref() {
                for k in 0..DOF {
                    mhd.bcell[k].view_mut().set(coord, mag[k]);
                }
            }
        }
    }

    /// seed EVERY interior cell from a closure over its physical CENTER position — the
    /// index->coordinate loop every IC otherwise hand-rolls. `sim.seed_cells(|x| prim_at(x))`
    /// replaces `for c in interior { let x = ...; sim.seed_cell(c, &prim_at(x)); }`. for MHD,
    /// pair with `seed_face` for the staggered face B (the CT ground truth).
    pub fn seed_cells(&self, f: impl Fn([f64; D]) -> <R as Regime<Sc, DOF>>::Prim) {
        for c in self.geom.interior.iter() {
            let x = self.geom.cell_coord(c);
            self.seed_cell(c, &f(x));
        }
    }

    /// gather the regime's CONSERVED state at a cell — the inverse of `seed_cell`'s scatter, with
    /// the cell-centered B folded in for MHD. `sim.cons_at(c)` replaces the per-test hand-rebuild
    /// `MhdCons { hydro: Cons { den: *..view().at(c), mom: Tensor::new([..]), nrg: .. }, mag: .. }`.
    pub fn cons_at(&self, coord: [isize; D]) -> <R as Regime<Sc, DOF>>::Cons {
        use symbi_hydro::state::SeedableCons;
        type Energy<R, const DOF: usize, Sc> =
            <<R as Regime<Sc, DOF>>::Cons as SeedableCons<Sc, DOF>>::Energy;
        let hydro = self.fields.cons.gather_as::<Energy<R, DOF, Sc>>(coord);
        let mag = self.fields.mhd.as_ref().map(|mhd| {
            Tensor::<Sc, DOF>::new(std::array::from_fn(|k| *mhd.bcell[k].view().at(coord)))
        });
        <<R as Regime<Sc, DOF>>::Cons as SeedableCons<Sc, DOF>>::from_parts(hydro, mag)
    }

    /// recover the regime's PRIMITIVE at a cell (c2p) — `sim.prim_at(c)` replaces building the
    /// Cons by hand + calling the regime recover. returns the recovered primitive (the c2p error
    /// code is dropped; assert physicality yourself, or use `regime.to_primitive` for the full
    /// `C2pResult`).
    pub fn prim_at(&self, coord: [isize; D]) -> <R as Regime<Sc, DOF>>::Prim {
        <R as Regime<Sc, DOF>>::to_primitive(
            &self.physics.regime,
            &self.physics.eos,
            &self.cons_at(coord),
        )
        .value
    }
}

// the inherent impl, generic over the vector dimension `DOF`: `new` builds
// a `DOF`-component state, so the `SimState<R,D,M,..>` alias gives the natural `DOF = D` and
// `SimStateGeneric<R, 2, 3, ..>::new` gives axisymmetric (3-vector momentum on a 2D grid).
impl<R, const D: usize, const DOF: usize, M, E, S, Mem, Sc>
    SimStateGeneric<R, D, DOF, M, E, S, Mem, Sc>
where
    R: Regime<Sc, D>,
    M: Metric<Sc, D>,
    E: Eos<Sc>,
    S: ExecutionSpace,
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    /// construct a simulation. allocates all fields once. no further allocations.
    /// crate-internal: the public constructor is [`SimBuilder`] (`Self::build`).
    pub(crate) fn new(
        regime: R,
        eos: E,
        metric: M,
        n_cells: [usize; D],
        x_lo: [f64; D],
        dx: [f64; D],
        ng: usize,
        boundaries: Boundaries<D>,
        cfl: f64,
        timestepping: Timestepping,
        device_id: i64,
    ) -> symbi_xpu::Result<Self> {
        Self::new_at(
            regime,
            eos,
            metric,
            [0; D],
            n_cells,
            x_lo,
            dx,
            ng,
            boundaries,
            cfl,
            timestepping,
            device_id,
        )
    }

    /// construct a simulation whose interior starts at an arbitrary ABSOLUTE index
    /// `interior_lo` (amr levels share one global index space: a fine level covering
    /// coarse cells [cov_lo, cov_hi) lives at [r*cov_lo, r*cov_hi)). `x_lo` stays the
    /// GLOBAL physical origin — the coordinate of index 0 — so `geom.centroid` is
    /// correct on every level with the same formula.
    /// crate-internal: the amr hierarchy builds fine levels at absolute indices; the public
    /// constructor is [`SimBuilder`] (which always uses `interior_lo = [0; D]`).
    #[allow(clippy::too_many_arguments)]
    pub fn new_at(
        regime: R,
        eos: E,
        metric: M,
        interior_lo: [isize; D],
        n_cells: [usize; D],
        x_lo: [f64; D],
        dx: [f64; D],
        ng: usize,
        boundaries: Boundaries<D>,
        cfl: f64,
        timestepping: Timestepping,
        device_id: i64,
    ) -> symbi_xpu::Result<Self> {
        let ng_i = ng as isize;
        let allocated = Domain::new(std::array::from_fn(|ax| Space {
            name: axis_name(ax),
            lo: interior_lo[ax] - ng_i,
            hi: interior_lo[ax] + n_cells[ax] as isize + ng_i,
        }));
        let interior = allocated.contract(ng_i);

        let geom = PartitionGeometry {
            dx,
            x_lo,
            allocated: allocated.clone(),
            interior,
            ng,
            coords: metric.geometry(),
            spacetime: metric.spacetime(),
            spacetime_scalars: metric
                .spacetime_scalars()
                .into_iter()
                .map(|(n, v)| (n.to_string(), v.to_f64()))
                .collect(),
            axes: default_grid_axes::<D>(metric.geometry()),
            maps: None,
        };
        let has_energy = regime.has_energy();

        // allocate MHD fields if the regime has magnetic fields. only GRMHD (magnetized, with
        // energy, on a curved background) can reject a step: its physical-constraint-preserving
        // redo falls back to halving the timestep when the source-free low-order anchor is
        // itself inadmissible. that alone allocates the step-entry rollback snapshot.
        let mhd = if regime.is_mhd() {
            let rejectable = has_energy && geom.spacetime != symbi_geometry::Spacetime::Minkowski;
            Some(MhdStaggeredFields::zeros(
                &allocated,
                &geom.interior,
                rejectable,
            )?)
        } else {
            None
        };

        // prim.pre allocation is REGIME-uniform across CPU and GPU: adiabatic
        // allocates it (the pressure primitive); isothermal does NOT — iso's pressure lives in the
        // kernel-set's substrate-owned `self.pre` (= cs^2*rho), bound by every iso kernel via the
        // `pre` override on CPU AND GPU. an `|| S::IS_DEVICE` term would allocate a DEAD placeholder on
        // iso-GPU (a positional-ABI "derefs pre unconditionally" path that is unused; nothing
        // writes or reads `sim.fields.prim.pre` for iso) — a CPU/GPU storage divergence avoided here.
        let alloc_pre = has_energy;

        let fields = PartitionFieldsGeneric {
            cons: ConsFieldsGeneric::zeros_with_energy(&allocated, has_energy)?,
            prim: PrimFieldsGeneric::zeros_with_pressure(&allocated, alloc_pre)?,
            flux: array_cons_zeros_with_energy(&allocated, has_energy)?,
            c2p_error: Field::zeros(&allocated)?,
            mhd,
            source: None,
        };

        let workspace = RkWorkspaceGeneric {
            u_n: ConsFieldsGeneric::zeros_with_energy(&allocated, has_energy)?,
            prim_n: PrimFieldsGeneric::zeros_with_pressure(&allocated, alloc_pre)?,
            u_stage: ConsFieldsGeneric::zeros_with_energy(&allocated, has_energy)?,
            stage_input_is_un: std::sync::atomic::AtomicBool::new(false),
            elide_stage_snapshot: std::sync::atomic::AtomicBool::new(true),
            flux_ho: array_cons_zeros_with_energy(&allocated, has_energy)?,
            fofc_flag: Field::zeros(&allocated)?,
            body_scratch: std::sync::OnceLock::new(),
        };

        let exec = Executor::<S>::new(device_id)?;

        Ok(Self {
            store: FieldStore {
                fields,
                workspace,
                geom,
                boundaries,
                time: 0.0,
                dt: 0.0,
                max_dt: 0.0,
                iteration: 0,
                cfl,
                timestepping,
                motion: MotionState::static_mesh(),
                motion_law: None,
                immersed: None,
                tracers: None,
                continuous_tracers: None,
                ito_coefficients: None,
                ito_transport: None,
            },
            physics: Physics {
                regime,
                metric,
                eos,
            },
            ctx: Context { exec },
        })
    }

    /// attach an immersed body collection to the simulation.
    /// also creates the diagnostic accumulator and source field for body feedback.
    pub fn with_bodies(mut self, bodies: symbi_ib::BodyCollection<f64, D>) -> Self {
        self.attach_bodies(bodies);
        self
    }

    /// allocate the passive-scalar (dye) slots: conserved `D_chi = rho chi`, the
    /// primitive concentration, and the rk step snapshot. a run-level opt-in
    /// (unlike the regime-static energy slot); every chi consumer gates on
    /// `has_passive_scalar()`, so an unallocated dye costs nothing anywhere.
    pub fn with_passive_scalar(mut self) -> symbi_xpu::Result<Self> {
        let allocated = self.geom.allocated.clone();
        self.fields.cons.alloc_chi(&allocated)?;
        self.fields.prim.alloc_chi(&allocated)?;
        self.workspace.u_n.alloc_chi(&allocated)?;
        // the dye rides in the conserved vector, so the step-entry rollback snapshot must carry
        // it too — otherwise a rejected step would replay from an undyed conserved state.
        if let Some(snapshot) = self
            .fields
            .mhd
            .as_mut()
            .and_then(|mhd| mhd.step_snapshot.as_mut())
        {
            snapshot.cons.alloc_chi(&allocated)?;
        }
        Ok(self)
    }

    /// select the cylindrical 2D MHD plane (r-z axisymmetric vs r-phi disk) — the grid-axis set
    /// the constrained-transport seam reads. only meaningful for a Cylindrical D==2 sim; a no-op
    /// otherwise (the axis set is unambiguous identity / `[0,2]` default). r-z is the default, so
    /// call this with `RPhi` to grid the disk plane (out-of-plane vertical B_z). must be set
    /// BEFORE seeding / evolving (it picks which `_cyl_rz` / `_cyl_rphi` kernels dispatch).
    pub fn with_cyl_plane(mut self, plane: CylPlane) -> Self {
        if self.geom.coords == symbi_geometry::Geometry::Cylindrical && D == 2 {
            self.geom.axes = match plane {
                CylPlane::Rz => std::array::from_fn(|d| if d == 0 { 0 } else { 2 }),
                CylPlane::RPhi => std::array::from_fn(|d| d), // [0, 1]
            };
        }
        self
    }

    /// attach per-axis non-uniform coordinate maps (log-radial spacing). the host geometry
    /// (centroids/faces/widths) honors the maps directly, and the curvilinear kernel dispatch
    /// reads them to select the `_logr` variant and pass the log decade-slope as the dx parameter.
    pub fn with_maps(mut self, maps: [symbi_geometry::AxisMap; D]) -> Self {
        self.geom.set_maps(maps);
        self
    }

    /// seed a staggered FACE-normal B component `bface[d]` from a closure over the face's
    /// physical position, and mark the staggered field initialized. dissolves the per-IC loop
    /// `for c in bface[d].domain() { set(c, f(face_coord)); } + bface_initialized.store(...)`.
    /// the CT ground truth: seed the faces with `seed_face*`, the cells with `seed_cells`.
    pub fn seed_face_with(&self, d: usize, f: impl Fn([f64; D]) -> Sc) {
        let mhd = self
            .fields
            .mhd
            .as_ref()
            .expect("seed_face requires MHD fields");
        for c in mhd.bface[d].domain().clone().iter() {
            mhd.bface[d]
                .view_mut()
                .set(c, f(self.geom.face_coord(c, d)));
        }
        mhd.bface_initialized
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// seed a staggered face-normal B component to a UNIFORM value (the common case — e.g., a
    /// uniform Bx / vertical B_z / toroidal B_phi threading the domain).
    pub fn seed_face(&self, d: usize, value: Sc) {
        self.seed_face_with(d, |_| value);
    }

    /// seed a staggered face-normal B component `bface[d]` from a flat buffer in axis-0-fastest
    /// order over the INTERIOR face domain (`interior` extended +1 on axis `d`) — the layout the
    /// python `staggered_bfields` generators yield. the index analog of [`seed_face_with`] for
    /// array-sourced ICs (the CT divergence-free ground truth).
    pub fn seed_face_indexed(&self, d: usize, data: &[Sc]) {
        let mhd = self
            .fields
            .mhd
            .as_ref()
            .expect("seed_face requires MHD fields");
        let dom = self.geom.interior.extend(d, 0, 1);
        assert_eq!(
            data.len(),
            dom.volume(),
            "seed_face_indexed[{d}]: {} values for a {}-face domain",
            data.len(),
            dom.volume()
        );
        let mut coord: [isize; D] = std::array::from_fn(|ax| dom.spaces[ax].lo);
        for &val in data {
            mhd.bface[d].view_mut().set(coord, val);
            for ax in 0..D {
                coord[ax] += 1;
                if coord[ax] < dom.spaces[ax].hi {
                    break;
                }
                coord[ax] = dom.spaces[ax].lo;
            }
        }
        mhd.bface_initialized
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// start a fluent [`SimBuilder`]: named setters + sane defaults (ng=2, cfl=0.4, RK2, outflow,
    /// device 0) in place of `new`'s 11 positional args. `Sim::build(regime, eos, metric)
    /// .cells([nx,ny]).bounds([0.,0.],[1.,1.]).boundaries(Periodic).finish()?`.
    pub fn build(regime: R, eos: E, metric: M) -> SimBuilder<R, D, DOF, M, E, S, Mem, Sc> {
        SimBuilder {
            regime,
            eos,
            metric,
            n_cells: None,
            x_lo: [0.0; D],
            dx: None,
            bounds_hi: None,
            ng: 2,
            boundaries: Boundaries::uniform(BoundaryType::Outflow),
            cfl: 0.4,
            timestepping: Timestepping::Rk2,
            device_id: 0,
            cyl_plane: None,
            coord_maps: None,
            sim: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// the by-reference form of [`Self::with_bodies`] — the amr hierarchy
    /// attaches per-level collections to already-constructed level states.
    pub fn attach_bodies(&mut self, bodies: symbi_ib::BodyCollection<f64, D>) {
        // wall-only fragments dispatch over per-body support boxes, which exist
        // only on cartesian charts today; off-cartesian every fragment pass
        // would sweep the full interior. fail loud instead of degrading.
        assert!(
            bodies.fragment_count() == 0 || self.geom.coords == symbi_geometry::Geometry::Cartesian,
            "fragments require a cartesian chart (per-fragment support boxes); \
             chart is {:?}",
            self.geom.coords,
        );
        let n = bodies.len();
        let has_energy = self.physics.regime.has_energy();
        if self.fields.source.is_none() {
            if let Ok(src_field) =
                ConsFieldsGeneric::zeros_with_energy(&self.geom.allocated, has_energy)
            {
                self.fields.source = Some(src_field);
            }
        }
        self.immersed = Some(ImmersedBodies {
            bodies,
            diagnostics: symbi_ib::DiagnosticAccumulator::new(n),
            history: symbi_ib::BodyHistory::new(n),
            // default: every body is its analytic sphere; a config shape is attached separately.
            shapes: vec![None; n],
            fragment_physics: None,
            accretion_receipts: std::sync::Mutex::new(Vec::new()),
            // NaN = "compute the local max"; the decomposed loop publishes a global value.
            c_a2_override: std::sync::atomic::AtomicU64::new(f64::NAN.to_bits()),
        });
    }

    /// attach the bonded-fragment pair physics (bonds + contact + mutual
    /// gravity). required whenever the collection carries fragments — without
    /// it a two-way fragment would feel wall forces but never move, so the
    /// mismatch fails loud here rather than silently freezing bodies.
    pub fn attach_fragment_physics(&mut self, physics: symbi_ib::FragmentPhysics) {
        let im = self
            .immersed
            .as_mut()
            .expect("attach_fragment_physics: no immersed bodies attached");
        for bond in &physics.bonds {
            assert!(
                bond.i < im.bodies.len() && bond.j < im.bodies.len(),
                "bond ({}, {}) references a body outside the collection of {}",
                bond.i,
                bond.j,
                im.bodies.len(),
            );
        }
        im.fragment_physics = Some(physics);
    }

    /// attach per-body immersed-boundary shapes (parallel to the body collection); `None` entries
    /// keep the analytic sphere. a no-op when the sim has no bodies. the length must match.
    pub fn attach_body_shapes(&mut self, shapes: Vec<Option<symbi_ib::sdf::SdfExpr<f64, 3>>>) {
        if let Some(im) = self.immersed.as_mut() {
            assert_eq!(
                shapes.len(),
                im.shapes.len(),
                "attach_body_shapes: {} shapes for {} bodies",
                shapes.len(),
                im.shapes.len(),
            );
            im.shapes = shapes;
        }
    }
}

// =============================================================================
// per-cadence conservation + constraint reduction
// =============================================================================

/// a host-side reduction of a level's interior for the live diagnostics: total
/// conserved mass and (when the regime carries an energy equation) total energy,
/// plus (for mhd) the peak magnetic-monopole error max|div B|. absolute values;
/// the display derives the relative drift against the t=0 baseline.
pub struct ConservationDiag {
    /// sum over interior cells of den * cell_volume (rest-mass density D, or rho).
    pub mass: f64,
    /// sum of nrg * cell_volume (tau or E); None for isothermal regimes.
    pub energy: Option<f64>,
    /// max over interior cells of |div B| from the staggered faces; None off mhd.
    pub div_b: Option<f64>,
    /// max Lorentz factor W = 1/sqrt(1 - v^2) over the interior; None for
    /// non-relativistic regimes (where v is unbounded and W is undefined).
    pub max_w: Option<f64>,
}

impl<R, const D: usize, const DOF: usize, M, E, S, Mem>
    SimStateGeneric<R, D, DOF, M, E, S, Mem, f64>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    /// reduce total mass / energy and max|div B| over this level's interior. cell
    /// volumes come from the block geometry so the sums are correct on curvilinear
    /// grids (r^2 sin(theta) etc.) as well as cartesian. returns `None` when the
    /// fields are not host-accessible (a device-resident gpu run), so the caller
    /// simply omits the diagnostics.
    pub fn conservation_diag(&self) -> Option<ConservationDiag> {
        if !Mem::IS_HOST_ACCESSIBLE {
            return None;
        }
        let bg = self.geom.block_geometry(self.physics.metric);
        // lab-frame (physical) cell volumes: on a homologously expanding (ALE) mesh
        // the conserved density multiplies the PHYSICAL volume = comoving * a^n (n
        // per geometry), so total mass/energy stay constant as
        // a(t) evolves. a static mesh (a = 1) leaves the comoving volume unchanged.
        let a = self.motion.a;
        let den = self.fields.cons.den.view();
        let nrg = self.fields.cons.nrg.as_ref().map(|f| f.view());
        let mut mass = 0.0_f64;
        let mut energy = 0.0_f64;
        for c in self.geom.interior.iter() {
            let vol = bg.labframe_volume(c, a);
            mass += *den.at(c) * vol;
            if let Some(nv) = nrg.as_ref() {
                energy += *nv.at(c) * vol;
            }
        }

        // max|div B| from the staggered faces, area-weighted so it is the true
        // constrained-transport divergence on curvilinear grids: div B =
        // (1/V) sum_d (A_d^+ B_d^+ - A_d^- B_d^-). face areas + cell volume come
        // from the block geometry; on a cartesian grid A_d = prod of the transverse
        // widths and this reduces to the plain (B_hi - B_lo)/dx form. the +1 normal
        // face exists on bface (allocated interior.extend(d, 0, 1)).
        let div_b = self.fields.mhd.as_ref().map(|mhd| {
            let mut max_div = 0.0_f64;
            for c in self.geom.interior.iter() {
                let vol = bg.volume(c);
                if vol <= 0.0 {
                    continue;
                }
                let mut flux = 0.0_f64;
                for d in 0..D {
                    let mut hi = c;
                    hi[d] += 1;
                    let b_lo = *mhd.bface[d].view().at(c);
                    let b_hi = *mhd.bface[d].view().at(hi);
                    flux += bg.face_area(hi, d) * b_hi - bg.face_area(c, d) * b_lo;
                }
                max_div = max_div.max((flux / vol).abs());
            }
            max_div
        });

        // max Lorentz factor: relativistic regimes only (v is the orthonormal
        // 3-velocity, |v| < 1, so W = 1/sqrt(1 - sum v_k^2)). non-relativistic v is
        // unbounded, so W is left None.
        let max_w = if <R as Regime<f64, D>>::SPEC.is_relativistic {
            let vel: [_; DOF] = std::array::from_fn(|k| self.fields.prim.vel[k].view());
            let mut mw = 1.0_f64;
            for c in self.geom.interior.iter() {
                let mut v2 = 0.0_f64;
                for v in vel.iter() {
                    let vk = *v.at(c);
                    v2 += vk * vk;
                }
                if v2 < 1.0 {
                    mw = mw.max(1.0 / (1.0 - v2).sqrt());
                }
            }
            Some(mw)
        } else {
            None
        };

        Some(ConservationDiag {
            mass,
            energy: nrg.map(|_| energy),
            div_b,
            max_w,
        })
    }

    /// how many fields the live heatmap can cycle through (density + pressure /
    /// Lorentz W / |B| when the regime carries them). bounds the `f`-key cycle.
    pub fn field_count(&self) -> usize {
        self.available_kinds().len()
    }

    /// the selectable fields, in cycle order (density always first).
    fn available_kinds(&self) -> Vec<(FieldKind, &'static str)> {
        let mut v = vec![(FieldKind::Density, "density")];
        if self.fields.prim.pre.is_some() {
            v.push((FieldKind::Pressure, "pressure"));
        }
        if <R as Regime<f64, D>>::SPEC.is_relativistic {
            v.push((FieldKind::LorentzW, "lorentz W"));
        }
        if self.fields.mhd.is_some() {
            v.push((FieldKind::MagField, "|B|"));
        }
        v
    }

    /// map a cycle index to the field it selects (wraps).
    pub fn nth_field(&self, index: usize) -> (FieldKind, &'static str) {
        let k = self.available_kinds();
        k[index % k.len()]
    }

    /// the selected scalar field at a cell (host access assumed by the caller).
    pub fn field_value(&self, c: [isize; D], kind: FieldKind) -> f64 {
        match kind {
            FieldKind::Density => *self.fields.prim.rho.view().at(c) as f64,
            FieldKind::Pressure => self
                .fields
                .prim
                .pre
                .as_ref()
                .map(|p| *p.view().at(c) as f64)
                .unwrap_or(0.0),
            FieldKind::LorentzW => {
                let mut v2 = 0.0_f64;
                for k in 0..DOF {
                    let vk = *self.fields.prim.vel[k].view().at(c) as f64;
                    v2 += vk * vk;
                }
                if v2 < 1.0 {
                    1.0 / (1.0 - v2).sqrt()
                } else {
                    1.0
                }
            }
            FieldKind::MagField => match self.fields.mhd.as_ref() {
                Some(mhd) => {
                    let mut b2 = 0.0_f64;
                    for k in 0..DOF {
                        let bk = *mhd.bcell[k].view().at(c) as f64;
                        b2 += bk * bk;
                    }
                    b2.sqrt()
                }
                None => 0.0,
            },
        }
    }

    /// decimate the selected field (by cycle `index`) to a screen-sized slice for
    /// the live heatmap: block-average the interior (mid-plane in axes >= 2 for the
    /// 3D z-slice) to <= `max_dim` per axis, so cost is bounded by the SCREEN size.
    /// a 1D grid yields a 1-row line profile. `None` off host memory.
    pub fn field_slice(&self, max_dim: usize, index: usize) -> Option<FieldDecimation> {
        self.field_slice_oriented(max_dim, index, 0, 0)
    }

    /// the decimated 2D display slice with a selectable ORIENTATION on a 3D grid
    /// (orient 0 = the z mid-plane (x, y), 1 = the y mid-plane (x, z), 2 = the x
    /// mid-plane (y, z); 1D/2D grids ignore it) and a ZOOM exponent: the display
    /// axes sample a centered 1/2^zoom-extent window, decimated to the same
    /// screen resolution — each step doubles the magnification about the domain
    /// center (where the hole / disk / body of interest conventionally sits).
    pub fn field_slice_oriented(
        &self,
        max_dim: usize,
        index: usize,
        orient: usize,
        zoom: usize,
    ) -> Option<FieldDecimation> {
        if !Mem::IS_HOST_ACCESSIBLE {
            return None;
        }
        let (kind, name) = self.nth_field(index);
        // a 2D ANGULAR chart draws in its physical shape: spherical (r, theta) as the
        // meridional half-plane, the cylindrical (R, phi) disk plane as the disk. the
        // (R, z) plane is already a faithful rectangle and 1D/3D keep the index view.
        if D == 2 {
            let angular = match self.geom.coords {
                symbi_geometry::Geometry::Spherical => true,
                symbi_geometry::Geometry::Cylindrical => self.geom.axes[1] == 1,
                symbi_geometry::Geometry::Cartesian => false,
            };
            if angular {
                return self.field_slice_polar(max_dim, kind, name);
            }
        }
        let interior = &self.geom.interior;
        // the display axes: (horizontal, vertical) index-space axes of the slice
        // plane; the remaining axis (3D only) holds its mid-plane index. 2D always
        // shows (0, 1).
        let (ah, av) = if D >= 3 {
            match orient % 3 {
                1 => (0usize, 2usize),
                2 => (1, 2),
                _ => (0, 1),
            }
        } else {
            (0, 1)
        };
        // the zoomed window on a display axis: a centered span of size/2^zoom,
        // at least 4 cells so a deep zoom on a tiny grid stays a real picture.
        let windowed = |sp: &symbi_algebra::Space| -> symbi_algebra::Space {
            if zoom == 0 {
                return sp.clone();
            }
            let size = sp.size() as isize;
            let span = (size >> zoom.min(4)).max(4.min(size));
            let mid = sp.lo + size / 2;
            symbi_algebra::Space {
                name: sp.name,
                lo: mid - span / 2,
                hi: mid - span / 2 + span,
            }
        };
        let sp0 = windowed(&interior.spaces[ah]);
        let sp0 = &sp0;
        let nx = sp0.size();
        if nx == 0 {
            return None;
        }
        let m = max_dim.max(1);
        let sx = ((nx + m - 1) / m).max(1);
        let out_w = (nx + sx - 1) / sx;
        // base coord: mid-plane on every axis; the display axes are overwritten below.
        let mut c: [isize; D] = std::array::from_fn(|ax| {
            let s = &interior.spaces[ax];
            s.lo + (s.size() / 2) as isize
        });
        let mut vmin = f64::INFINITY;
        let mut vmax = f64::NEG_INFINITY;

        // 1D grid: a line profile (height = 1), block-averaged along axis 0.
        let sp1_w = interior.spaces.get(av).map(|sp| windowed(sp));
        let Some(sp1) = sp1_w.as_ref() else {
            let mut data = Vec::with_capacity(out_w);
            for i in 0..out_w {
                let x0 = sp0.lo + (i * sx) as isize;
                let x1 = (x0 + sx as isize).min(sp0.hi);
                let (mut sum, mut cnt) = (0.0_f64, 0u32);
                let mut xx = x0;
                while xx < x1 {
                    c[ah] = xx;
                    sum += self.field_value(c, kind);
                    cnt += 1;
                    xx += 1;
                }
                let v = sum / cnt.max(1) as f64;
                vmin = vmin.min(v);
                vmax = vmax.max(v);
                data.push(v as f32);
            }
            return Some(FieldDecimation {
                width: out_w,
                height: 1,
                data,
                vmin,
                vmax,
                name: name.into(),
                preserve_aspect: false,
            });
        };

        // 2D (or 3D mid-slice): block-average each sx x sy footprint.
        let ny = sp1.size();
        if ny == 0 {
            return None;
        }
        let sy = ((ny + m - 1) / m).max(1);
        let out_h = (ny + sy - 1) / sy;
        let mut data = Vec::with_capacity(out_w * out_h);
        for j in 0..out_h {
            let y0 = sp1.lo + (j * sy) as isize;
            let y1 = (y0 + sy as isize).min(sp1.hi);
            for i in 0..out_w {
                let x0 = sp0.lo + (i * sx) as isize;
                let x1 = (x0 + sx as isize).min(sp0.hi);
                let (mut sum, mut cnt) = (0.0_f64, 0u32);
                let mut yy = y0;
                while yy < y1 {
                    if let Some(slot) = c.get_mut(av) {
                        *slot = yy;
                    }
                    let mut xx = x0;
                    while xx < x1 {
                        c[ah] = xx;
                        sum += self.field_value(c, kind);
                        cnt += 1;
                        xx += 1;
                    }
                    yy += 1;
                }
                let v = sum / cnt.max(1) as f64;
                vmin = vmin.min(v);
                vmax = vmax.max(v);
                data.push(v as f32);
            }
        }
        Some(FieldDecimation {
            width: out_w,
            height: out_h,
            data,
            vmin,
            vmax,
            name: name.into(),
            preserve_aspect: false,
        })
    }

    /// the polar (physical-shape) decimation of a 2D angular chart, by INVERSE
    /// sampling: each display pixel maps to (display radius, angle) -> nearest
    /// grid cell -> field value, so cost is screen-bounded exactly like the
    /// index-space decimation. pixels outside the annulus/wedge are NaN and the
    /// renderer leaves them blank — the central hole marks the inner boundary.
    ///
    /// the display radius is the radial INDEX fraction, so a log-radial grid gets
    /// a log-polar view (equal display area per cell shell — the inner decades of
    /// an accretion run keep their visual weight) and a uniform grid a linear one,
    /// with no coordinate-map inversion.
    ///
    /// spherical (r, theta): the meridional half-plane x = r sin(theta),
    /// y = r cos(theta) (pole up). cylindrical (R, phi): the full disk
    /// x = R cos(phi), y = R sin(phi), with the angle wrapped when the grid spans
    /// the full circle and a NaN wedge otherwise.
    fn field_slice_polar(
        &self,
        max_dim: usize,
        kind: FieldKind,
        name: &str,
    ) -> Option<FieldDecimation> {
        let interior = &self.geom.interior;
        let sp0 = &interior.spaces[0];
        let sp1 = interior.spaces.get(1)?;
        let (nr, na) = (sp0.size(), sp1.size());
        if nr == 0 || na == 0 {
            return None;
        }
        let meridional = self.geom.coords == symbi_geometry::Geometry::Spherical;
        let a_lo = self.geom.x_lo[1];
        let a_hi = a_lo + na as f64 * self.geom.dx[1];
        let two_pi = 2.0 * std::f64::consts::PI;
        // the central hole: a fixed display fraction marking the inner radial
        // boundary (excision surface, inner ghost) when the grid starts off zero.
        let s0 = if self.geom.x_lo[0] > 0.0 { 0.08 } else { 0.0 };

        // the TIGHT display bounding box of the annular sector, so a wedge fills
        // its panel (a full-circle box would leave it floating in mostly-NaN space): evaluate
        // the sector's (x, y) at both radii for the angular endpoints and every
        // quarter-turn extremum inside the span. meridional (x, y) = rho (sin, cos)
        // with theta from the +y pole; disk (x, y) = rho (cos, sin).
        let (mut x_min, mut x_max, mut y_min, mut y_max) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        {
            let mut angles = vec![a_lo, a_hi];
            let mut k = (a_lo / std::f64::consts::FRAC_PI_2).ceil() as i64;
            while (k as f64) * std::f64::consts::FRAC_PI_2 <= a_hi {
                angles.push(k as f64 * std::f64::consts::FRAC_PI_2);
                k += 1;
            }
            for &ang in &angles {
                for rho in [s0, 1.0] {
                    let (x, y) = if meridional {
                        (rho * ang.sin(), rho * ang.cos())
                    } else {
                        (rho * ang.cos(), rho * ang.sin())
                    };
                    x_min = x_min.min(x);
                    x_max = x_max.max(x);
                    y_min = y_min.min(y);
                    y_max = y_max.max(y);
                }
            }
        }
        let (span_x, span_y) = (x_max - x_min, y_max - y_min);
        if span_x <= 0.0 || span_y <= 0.0 {
            return None;
        }

        let m = max_dim.max(2);
        // fit the sector's aspect into m: the renderer preserves it (letterbox).
        let aspect = span_x / span_y;
        let (out_w, out_h) = if aspect >= 1.0 {
            (m, ((m as f64 / aspect) as usize).max(1))
        } else {
            (((m as f64 * aspect) as usize).max(1), m)
        };
        let mut data = Vec::with_capacity(out_w * out_h);
        let mut vmin = f64::INFINITY;
        let mut vmax = f64::NEG_INFINITY;
        let mut c: [isize; D] = std::array::from_fn(|ax| {
            let s = &interior.spaces[ax];
            s.lo + (s.size() / 2) as isize
        });
        // one inverse sample: display point -> (radius fraction, angle) -> nearest cell.
        let mut sample_at = |x_d: f64, y_d: f64| -> f64 {
            let s = (x_d * x_d + y_d * y_d).sqrt();
            if !(s0..=1.0).contains(&s) {
                return f64::NAN;
            }
            let u_r = if s0 < 1.0 { (s - s0) / (1.0 - s0) } else { 0.0 };
            let ang = if meridional {
                // theta from the +y pole, in [0, pi] for x_d >= 0.
                x_d.atan2(y_d)
            } else {
                // fold the atan2 branch into the grid's own angular span: a full
                // circle wraps everywhere, a wedge leaves u_a > 1 (NaN) outside —
                // including wedges that cross the cut.
                a_lo + (y_d.atan2(x_d) - a_lo).rem_euclid(two_pi)
            };
            let u_a = (ang - a_lo) / (a_hi - a_lo);
            if !(0.0..=1.0).contains(&u_a) {
                return f64::NAN;
            }
            let ir = ((u_r * nr as f64) as usize).min(nr - 1);
            let ia = ((u_a * na as f64) as usize).min(na - 1);
            c[0] = sp0.lo + ir as isize;
            c[1] = sp1.lo + ia as isize;
            self.field_value(c, kind)
        };
        // supersample each pixel and average the in-domain sub-samples, with tap
        // counts matched to the pixel's FOOTPRINT in cells: a display pixel spans
        // ~nr/out radial cells (and, near the center, many angular cells), and any
        // tap count below that lets a thin feature — a 2-cell blast shell — fall
        // between taps on some rays, aliasing a smooth ring into a dotted arc.
        // footprint-matched taps give the same block-average semantics as the
        // index-space decimation; worst case a few million lookups per cadence,
        // the same order as the one grid pass the rectangle path already pays.
        let px = (span_x / out_w as f64).max(span_y / out_h as f64);
        let clamp_taps = |t: f64| (t.ceil() as usize).clamp(3, 16);
        // radial cells per pixel is uniform in the index-fraction display map; the
        // radial direction rotates against the display axes around the arc, so the
        // sub-grid is UNIFORM at this density in both axes (direction-agnostic).
        let ss_r = clamp_taps(nr as f64 * px / (1.0 - s0).max(1e-12));
        let ang_span = a_hi - a_lo;
        for jj in 0..out_h {
            for ii in 0..out_w {
                // near the center a pixel also spans many ANGULAR cells: coverage
                // du_a ~ px / (s * span) grows as 1/s. take the denser requirement.
                let yc = y_max - (jj as f64 + 0.5) / out_h as f64 * span_y;
                let xc = x_min + (ii as f64 + 0.5) / out_w as f64 * span_x;
                let s_c = (xc * xc + yc * yc).sqrt().max(s0.max(1e-3));
                let ss = ss_r.max(clamp_taps(na as f64 * px / (s_c * ang_span)));

                let (mut sum, mut cnt) = (0.0f64, 0u32);
                for sj in 0..ss {
                    // rows top -> bottom = display y decreasing (pole / +y up).
                    let y_d =
                        y_max - (jj as f64 + (sj as f64 + 0.5) / ss as f64) / out_h as f64 * span_y;
                    for si in 0..ss {
                        let x_d = x_min
                            + (ii as f64 + (si as f64 + 0.5) / ss as f64) / out_w as f64 * span_x;
                        let v = sample_at(x_d, y_d);
                        if !v.is_nan() {
                            sum += v;
                            cnt += 1;
                        }
                    }
                }
                let v = if cnt == 0 { f64::NAN } else { sum / cnt as f64 };
                if !v.is_nan() {
                    vmin = vmin.min(v);
                    vmax = vmax.max(v);
                }
                data.push(v as f32);
            }
        }
        if !vmin.is_finite() {
            return None;
        }
        Some(FieldDecimation {
            width: out_w,
            height: out_h,
            data,
            vmin,
            vmax,
            name: format!(
                "{name} · {}",
                if meridional { "meridional" } else { "disk" }
            ),
            preserve_aspect: true,
        })
    }
}

/// the scalar fields the live heatmap can display.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldKind {
    Density,
    Pressure,
    LorentzW,
    MagField,
}

/// a screen-sized decimated field for the live heatmap. row-major `width * height`
/// (`height == 1` is a 1D line profile); the display crate wraps this with a
/// colormap. `name` is the field label (e.g. "density", "|B|").
pub struct FieldDecimation {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
    pub vmin: f64,
    pub vmax: f64,
    pub name: String,
    /// true for a physical-shape (polar) slice: the renderer must letterbox to the
    /// slice's aspect ratio; stretching to the panel would draw a circle as
    /// an ellipse and defeat the projection. index-space rectangles stretch as before.
    pub preserve_aspect: bool,
}

// =============================================================================
// tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_geometry::Cartesian;
    use symbi_hydro::eos::IdealGas;
    use symbi_hydro::newtonian::Newtonian;
    use symbi_xpu::{CpuSpace, HostMemory};

    // THE atlas invariant: the decomposition transport set is DERIVED from the store's
    // populated slots, not a hand-maintained list. a passive scalar (dye), allocated as a
    // run-level opt-in, must appear in the exchanged set the moment it exists and must be
    // absent otherwise -- a hand-listed set is exactly what silently dropped it before.
    #[test]
    fn exchange_set_tracks_the_optional_passive_scalar_slot() {
        let domain = Domain::<2>::new([
            symbi_algebra::Space {
                name: "x",
                lo: 0,
                hi: 16,
            },
            symbi_algebra::Space {
                name: "y",
                lo: 0,
                hi: 16,
            },
        ]);

        // adiabatic hydro: den + 2 momenta + nrg = 4 conserved, rho + 2 vel + pre = 4 prim.
        let mut cons = ConsFields::<2, HostMemory>::zeros(&domain).unwrap();
        let mut prim = PrimFields::<2, HostMemory>::zeros(&domain).unwrap();
        assert_eq!(cons.exchange_fields().len(), 4, "den, mom[0], mom[1], nrg");
        assert_eq!(prim.exchange_fields().len(), 4, "rho, vel[0], vel[1], pre");

        // allocating the dye extends BOTH sets by exactly one, with no other change.
        cons.alloc_chi(&domain).unwrap();
        prim.alloc_chi(&domain).unwrap();
        assert_eq!(cons.exchange_fields().len(), 5, "dye adds cons.chi");
        assert_eq!(prim.exchange_fields().len(), 5, "dye adds prim.chi");

        // the appended field IS the chi slot (the gather/exchange zip relies on chi being
        // last so global and tile sets stay aligned when both carry it).
        let cons_last = *cons.exchange_fields().last().unwrap() as *const _;
        assert_eq!(cons_last, cons.chi_field().unwrap() as *const _);
        let prim_last = *prim.exchange_fields().last().unwrap() as *const _;
        assert_eq!(prim_last, prim.chi_field().unwrap() as *const _);
    }

    // isothermal drops the energy/pressure slot, so the derived set must shrink -- the
    // enumeration reads the actual slots, it does not assume the adiabatic arity.
    #[test]
    fn exchange_set_omits_the_absent_isothermal_energy_slot() {
        let domain = Domain::<2>::new([
            symbi_algebra::Space {
                name: "x",
                lo: 0,
                hi: 16,
            },
            symbi_algebra::Space {
                name: "y",
                lo: 0,
                hi: 16,
            },
        ]);
        let cons = ConsFields::<2, HostMemory>::zeros_with_energy(&domain, false).unwrap();
        let prim = PrimFields::<2, HostMemory>::zeros_with_pressure(&domain, false).unwrap();
        assert_eq!(
            cons.exchange_fields().len(),
            3,
            "den, mom[0], mom[1]; no nrg"
        );
        assert_eq!(
            prim.exchange_fields().len(),
            3,
            "rho, vel[0], vel[1]; no pre"
        );
    }

    #[test]
    fn sim_construction_1d() {
        let sim = SimState::<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>::new(
            Newtonian,
            IdealGas { gamma: 1.4 },
            Cartesian,
            [100],
            [0.0],
            [0.01],
            2,
            Boundaries::uniform(BoundaryType::Outflow),
            0.4,
            Timestepping::Rk2,
            0,
        )
        .unwrap();

        assert_eq!(sim.geom.interior.volume(), 100);
        assert_eq!(sim.geom.allocated.volume(), 104); // 100 + 2*2
        assert_eq!(sim.time, 0.0);
        assert_eq!(sim.iteration, 0);
    }

    #[test]
    fn sim_construction_2d() {
        let sim = SimState::<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>::new(
            Newtonian,
            IdealGas { gamma: 5.0 / 3.0 },
            Cartesian,
            [256, 256],
            [0.0, 0.0],
            [1.0 / 256.0, 1.0 / 256.0],
            2,
            Boundaries::uniform(BoundaryType::Periodic),
            0.4,
            Timestepping::Rk2,
            0,
        )
        .unwrap();

        assert_eq!(sim.geom.interior.volume(), 256 * 256);
        assert_eq!(sim.geom.allocated.volume(), 260 * 260);
    }

    #[test]
    fn sim_construction_3d() {
        let sim = SimState::<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>::new(
            Newtonian,
            IdealGas { gamma: 5.0 / 3.0 },
            Cartesian,
            [32, 32, 32],
            [0.0, 0.0, 0.0],
            [1.0 / 32.0, 1.0 / 32.0, 1.0 / 32.0],
            2,
            Boundaries::uniform(BoundaryType::Periodic),
            0.4,
            Timestepping::Rk2,
            0,
        )
        .unwrap();

        assert_eq!(sim.geom.interior.volume(), 32 * 32 * 32);
        assert_eq!(sim.geom.allocated.volume(), 36 * 36 * 36);
    }

    #[test]
    fn gather_scatter_roundtrip() {
        let sim = SimState::<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>::new(
            Newtonian,
            IdealGas { gamma: 1.4 },
            Cartesian,
            [10, 10],
            [0.0, 0.0],
            [0.1, 0.1],
            2,
            Boundaries::uniform(BoundaryType::Outflow),
            0.4,
            Timestepping::Euler,
            0,
        )
        .unwrap();

        let val = Cons {
            den: 1.5,
            mom: Tensor::new([0.3, -0.2]),
            nrg: 2.5,
        };

        sim.fields.cons.scatter([3, 4], val);
        let got = sim.fields.cons.gather([3, 4]);

        assert_eq!(got.den, 1.5);
        assert_eq!(got.mom[0], 0.3);
        assert_eq!(got.mom[1], -0.2);
        assert_eq!(got.nrg, 2.5);
    }

    #[test]
    fn prim_gather_scatter() {
        let sim = SimState::<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>::new(
            Newtonian,
            IdealGas { gamma: 1.4 },
            Cartesian,
            [10, 10],
            [0.0, 0.0],
            [0.1, 0.1],
            2,
            Boundaries::uniform(BoundaryType::Outflow),
            0.4,
            Timestepping::Euler,
            0,
        )
        .unwrap();

        let val = Prim {
            rho: 2.0,
            vel: Tensor::new([0.5, -0.3]),
            pre: 1.0,
        };

        sim.fields.prim.scatter([5, 5], val);
        let got = sim.fields.prim.gather([5, 5]);

        assert_eq!(got.rho, 2.0);
        assert_eq!(got.vel[0], 0.5);
        assert_eq!(got.vel[1], -0.3);
        assert_eq!(got.pre, 1.0);
    }

    #[test]
    fn centroid_2d() {
        let sim = SimState::<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>::new(
            Newtonian,
            IdealGas { gamma: 1.4 },
            Cartesian,
            [100, 100],
            [0.0, 0.0],
            [0.01, 0.01],
            2,
            Boundaries::uniform(BoundaryType::Outflow),
            0.4,
            Timestepping::Euler,
            0,
        )
        .unwrap();

        let c = sim.geom.centroid([0, 0]);
        assert!((c[0] - 0.005).abs() < 1e-14);
        assert!((c[1] - 0.005).abs() < 1e-14);

        let c = sim.geom.centroid([50, 50]);
        assert!((c[0] - 0.505).abs() < 1e-14);
        assert!((c[1] - 0.505).abs() < 1e-14);
    }

    #[test]
    fn stagger_coord_face_vs_center() {
        // distinct per-axis spacing so an axis mix-up is caught.
        let sim = SimState::<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>::new(
            Newtonian,
            IdealGas { gamma: 1.4 },
            Cartesian,
            [8, 8, 8],
            [0.0, 0.0, 0.0],
            [0.1, 0.2, 0.4],
            2,
            Boundaries::uniform(BoundaryType::Outflow),
            0.4,
            Timestepping::Euler,
            0,
        )
        .unwrap();
        let g = &sim.geom;
        let approx = |a: f64, b: f64| (a - b).abs() < 1e-14;

        // bface[0]: on the x-face (i*dx, exact), cell-centered in y/z.
        let f0 = g.face_coord([2, 3, 4], 0);
        assert!(approx(f0[0], 0.20) && approx(f0[1], 0.70) && approx(f0[2], 1.80));
        // bface[1]: cell-centered in x, on the y-face (j*dy), centered in z.
        let f1 = g.face_coord([2, 3, 4], 1);
        assert!(approx(f1[0], 0.25) && approx(f1[1], 0.60) && approx(f1[2], 1.80));

        // face_coord(dir) is exactly half a cell below centroid on `dir`, equal elsewhere.
        let cc = g.centroid([2, 3, 4]);
        assert!(approx(cc[0] - f0[0], 0.5 * 0.1) && approx(cc[1], f0[1]) && approx(cc[2], f0[2]));

        // general stagger: an edge (Face, Face, Center) is the lower x/y edge, centered in z.
        let e = g.stagger_coord([2, 3, 4], [Loc::Face, Loc::Face, Loc::Center]);
        assert!(approx(e[0], 0.20) && approx(e[1], 0.60) && approx(e[2], 1.80));
        // all-Center reproduces the centroid.
        let ctr = g.stagger_coord([2, 3, 4], [Loc::Center; 3]);
        assert!(approx(ctr[0], cc[0]) && approx(ctr[1], cc[1]) && approx(ctr[2], cc[2]));
    }
}
