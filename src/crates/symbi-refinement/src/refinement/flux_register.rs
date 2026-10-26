// =============================================================================
// flux_register.rs
//
// flux correction (refluxing) for fixed mesh refinement.
// single-coverage cap: one register per coarse-fine level-pair, sized to
// the level's single refined box (`coverage`) — there is no per-patch register
// fan-out, because a level refines exactly one box. accumulates the
// mismatch between coarse- and fine-level face fluxes at coarse-fine
// boundaries, then applies a conservative correction to the coarse state:
//   accumulate_coarse:  R -= F_coarse * A_coarse * w
//   accumulate_fine:    U = sum(F_fine), then R += (A_fine * w) * U
//   apply:              U[cell] += sign * R / V[cell]
//     sign = -1 on lo faces (the face is the corrected cell's right face),
//     sign = +1 on hi faces (the corrected cell's left face).
// per ssp stage the weight is `w = dt * ac_i * prod_{k>i} ac_k` (the stage's
// effective flux weight after the convex recombination) so the per-step
// weights sum to dt for every scheme.
//
// levels share absolute index space: the fine faces covering coarse face
// `coord` start at `ratio * coord` — no coverage offset. component-wise over
// den / mom[0..DOF] / optional nrg / optional chi (energy-optional regimes skip
// nrg; runs without a passive scalar skip chi).
//
// every fine accumulation is one event: a unit U holds the unscaled sum of the
// fine faces coincident with each register face cell (formed from -0.0, so it
// is the sum exactly), and the register takes R + coefficient * U with the
// coefficient A_fine * w. a register face has one of three roles. a local face
// holds both the fine faces and the coarse cell it corrects on this tile; it
// forms each unit and applies it at once. a contributing face lies on a patch
// face that coincides with a decomposition cut: the fine faces are here and the
// corrected coarse cell is on the neighbor tile, so the face holds one stage's
// unit and its coefficient, which the driver sends to the neighbor before the
// face is reset; the coarse flux and the application belong to the neighbor. a
// receiving face is that neighbor's side: it accumulates the coarse flux at its
// own stages, applies each landed unit with the coefficient that came with it,
// in stage order, and is applied to the corrected cell. local and received units
// take the same operation on the same operands, so the register agrees bit for
// bit with a single grid's on every backend, fused multiply-add included. a
// unit carries one gas event and one out-of-plane event at most; a second is
// refused rather than combined.
//
// two execution paths: uniform-cartesian geometry runs substrate kernels
// (field_fill / field_axpy_shift / refine_acc_face — cpu and gpu through the
// kernel dispatch; the face area and cell volume are constant scales), while
// curvilinear geometry keeps the per-coordinate host loops (cpu-only until
// the register kernels learn in-kernel geometry). the budget diagnostics
// (debug_*_den) are maintained by the host path only; the uniform path
// poisons them with NaN so a stale read is loud.
//
// usage:
//  let reg = FluxRegister::new(&coverage, &coarse_interior, has_energy, has_dye)?;
//  reg.zero();
//  reg.accumulate_coarse(&flux, &geo, dir, w);
//  reg.accumulate_fine(&fine_flux, &fine_geo, dir, w, ratio);
//  reg.apply(&cons, &geo);
// =============================================================================

use symbi_algebra::{Domain, Space};
use symbi_geometry::{BlockGeometry, Metric};
use symbi_grid::Field;
use symbi_xpu::MemorySpace;

use symbi_ir::KernelId;
use symbi_sim::state::{BcellFields, BfluxFields, ConsFieldsGeneric, axis_name};
use symbi_substrate::regimes::substrate_kernels::dispatch_fields_each;

/// the part a register face plays on this tile
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceRole {
    /// the fine faces and the corrected coarse cell are both on this tile
    Local,
    /// the fine faces are here, the corrected cell on the neighbor across a cut: the face
    /// holds one stage's fine unit for the driver to send
    Contributing,
    /// the corrected cell is here, the fine faces on the neighbor across a cut: the face
    /// takes the coarse flux here and the fine units from its inbox
    Receiving,
}

/// per coarse-fine interface accumulator. faces[2*ax + side] is the register
/// for the coverage boundary slab on that side; None where the coverage
/// touches the coarse interior boundary and no register crosses the cut there.
pub struct FluxRegister<const D: usize, const DOF: usize, Mem: MemorySpace> {
    faces: [Option<ConsFieldsGeneric<D, DOF, Mem>>; 6],
    domains: [Option<Domain<D>>; 6],
    roles: [FaceRole; 6],
    /// the landing slab of a receiving face: the neighbor's unit arrives here and is folded
    /// into the face
    inboxes: [Option<ConsFieldsGeneric<D, DOF, Mem>>; 6],
    /// the landing slabs of a receiving face's out-of-plane components
    oop_inboxes: [Option<Vec<Field<f64, D, Mem>>>; 6],
    /// the out-of-plane magnetic components of a partially gridded chart (the cell-centered
    /// B_z of a 2.5D grid), conserved by their induction-flux divergence and refluxed like the
    /// gas: one register face field per component, `DOF - D` of them, magnetized runs only.
    oop: [Option<Vec<Field<f64, D, Mem>>>; 6],
    /// the coefficient of a transported face's unit: on a contributing face the sender's
    /// `area * w`, sent with the unscaled fine-flux sums; on a receiving face the coefficient
    /// that landed, which the fold applies as `R + coefficient * S`, the operation the local
    /// accumulation performs with the same operands
    coefficients: [Option<Field<f64, D, Mem>>; 6],
    /// the accumulation events a contributing face's unit holds since it was last sent: bit 0
    /// the gas sums, bit 1 the out-of-plane sums. a unit carries at most one event of each, so
    /// the receiver's fold sees the operands the local accumulation sees
    pending: std::cell::Cell<[u8; 6]>,
    /// a local face's unit: each fine accumulation forms its unscaled sums here and applies
    /// them as `R + coefficient * U`, the operation a receiving face applies to a landed unit
    local_units: [std::cell::OnceCell<Vec<Field<f64, D, Mem>>>; 6],
    /// sign-weighted running den totals of the coarse / fine accumulations
    /// (conservation-budget diagnostics: the pure-coarse outside-mass change
    /// must equal -debug_coarse_den, the fine interior change -debug_fine_den).
    pub debug_coarse_den: std::cell::Cell<f64>,
    pub debug_fine_den: std::cell::Cell<f64>,
}

/// the axis whose low register face SIMBI_REFLUX_PROBE names: every register operation on the
/// first cell of that face prints its x-momentum operands and result, as bit patterns, once the
/// devices have completed. unset, nothing is read or printed
fn probe_axis() -> Option<usize> {
    static AXIS: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *AXIS.get_or_init(|| std::env::var("SIMBI_REFLUX_PROBE").ok().and_then(|v| v.trim().parse().ok()))
}

/// the x-momentum's position in a face's component list
const PROBE_COMPONENT: usize = 1;

fn probe_cell<const D: usize>(dom: &Domain<D>) -> [isize; D] {
    std::array::from_fn(|ax| dom.spaces[ax].lo)
}

/// wait for every kernel on the current device, so a host read of a register field that follows
/// sees what those kernels wrote
fn sync_current<Mem: MemorySpace>() {
    #[cfg(feature = "gpu")]
    if Mem::IS_DEVICE_ACCESSIBLE {
        symbi_xpu::ctx_sync();
    }
}

fn probe_sync<Mem: MemorySpace>() {
    sync_current::<Mem>();
}

fn probe_line(op: &str, role: FaceRole, cell: &[isize], values: &[(&str, f64)]) {
    use std::io::Write;
    let fields: Vec<String> = values.iter().map(|(k, v)| format!("{k}={:016x}", v.to_bits())).collect();
    let line = format!(
        "SIMBI reflux probe: pid={} op={op} role={role:?} cell={cell:?} {}\n",
        std::process::id(),
        fields.join(" ")
    );
    let _ = std::io::stderr().write_all(line.as_bytes());
}

/// a fine accumulation event of the gas sums, and of the out-of-plane sums
const EVENT_GAS: u8 = 1;
const EVENT_OUT_OF_PLANE: u8 = 2;

/// form unit `U = S`, the unscaled sum of the fine faces coincident with each register face
/// cell, in the accumulation kernel's summation order: the unit starts at -0.0, the additive
/// identity of every value including a signed zero, and takes `-0.0 + 1.0 * S`, which is S
fn form_unit<const D: usize, Mem: MemorySpace>(kernel: &str, dom: &Domain<D>, fine: &Field<f64, D, Mem>, unit: &Field<f64, D, Mem>) {
    let fill = KernelId::FieldFill { ndim: D as u8 }.name();
    dispatch_fields_each::<f64, Mem, D>(fill, dom, &[], &[unit], &[], &[-0.0]);
    dispatch_fields_each::<f64, Mem, D>(kernel, dom, &[fine], &[unit], &[], &[1.0]);
}

/// apply unit U to register R as `R + coefficient * U`: one operation for a local face's own
/// unit and a receiving face's landed one, so both see the same operation and operands
fn apply_unit<const D: usize, Mem: MemorySpace>(dom: &Domain<D>, unit: &Field<f64, D, Mem>, register: &Field<f64, D, Mem>, coefficient: f64) {
    let axpy = KernelId::FieldAxpyShift { ndim: D as u8 }.name();
    dispatch_fields_each::<f64, Mem, D>(axpy, dom, &[unit], &[register], &[0i32; 3][..D], &[coefficient]);
}

/// the fine faces of register face cell `c` normal to `dir`, in the accumulation kernel's
/// summation order: the transverse offsets counted with the lowest transverse axis fastest
fn probe_fine<const D: usize, Mem: MemorySpace>(fine: &Field<f64, D, Mem>, c: [isize; D], dir: usize) -> Vec<f64> {
    let transverse: Vec<usize> = (0..D).filter(|&t| t != dir).collect();
    (0..1usize << transverse.len())
        .map(|bits| {
            let at: [isize; D] = std::array::from_fn(|ax| {
                let off = transverse.iter().position(|&t| t == ax).map_or(0, |k| ((bits >> k) & 1) as isize);
                2 * c[ax] + off
            });
            *fine.at(at)
        })
        .collect()
}

/// the conserved components of `c` as a flat field list: den, mom[0..DOF],
/// then nrg when present. pairs of lists zip positionally — both sides of
/// every register operation carry the same component order.
fn comps<const D: usize, const DOF: usize, Mem: MemorySpace>(
    c: &ConsFieldsGeneric<D, DOF, Mem>,
) -> Vec<&Field<f64, D, Mem>> {
    let mut v: Vec<&Field<f64, D, Mem>> = Vec::with_capacity(DOF + 2);
    v.push(&c.den);
    for dd in 0..DOF {
        v.push(&c.mom[dd]);
    }
    if let Some(nrg) = c.nrg_field() {
        v.push(nrg);
    }
    // the dye rides last so the positional zip stays stable for runs without one. the conserved
    // dye is refluxed exactly like mass: its interface flux is a stored quantity, so the
    // fine-summed minus coarse mismatch corrects the covered coarse cells the same way.
    if let Some(chi) = c.chi_field() {
        v.push(chi);
    }
    v
}

impl<const D: usize, const DOF: usize, Mem: MemorySpace> FluxRegister<D, DOF, Mem> {
    /// allocate register faces at the boundary of `coverage` (absolute coarse
    /// indices) within `coarse_interior`; skip faces flush with the coarse
    /// interior boundary.
    /// `has_energy` must match the registered sim's `cons.has_energy()`, and `has_dye` its
    /// `cons.chi_field().is_some()` — the register's per-face cons carries the same component set
    /// (den / mom / optional nrg / optional chi) so the positional reflux zip (`comps`) stays
    /// aligned. iso (no nrg) passes `false`; a run without a passive scalar passes `false` for the
    /// dye.
    pub fn new(
        coverage: &Domain<D>,
        coarse_interior: &Domain<D>,
        has_energy: bool,
        has_dye: bool,
        n_oop: usize,
    ) -> symbi_xpu::Result<Self> {
        let mut faces: [Option<ConsFieldsGeneric<D, DOF, Mem>>; 6] = std::array::from_fn(|_| None);
        let mut domains: [Option<Domain<D>>; 6] = std::array::from_fn(|_| None);
        let mut oop: [Option<Vec<Field<f64, D, Mem>>>; 6] = std::array::from_fn(|_| None);

        for ax in 0..D {
            for side in 0..2usize {
                if side == 0 && coverage.spaces[ax].lo == coarse_interior.spaces[ax].lo {
                    continue;
                }
                if side == 1 && coverage.spaces[ax].hi == coarse_interior.spaces[ax].hi {
                    continue;
                }
                let face_pos = if side == 0 {
                    coverage.spaces[ax].lo
                } else {
                    coverage.spaces[ax].hi
                };
                let face_domain = Domain::new(std::array::from_fn(|aa| {
                    if aa == ax {
                        Space {
                            name: axis_name(aa),
                            lo: face_pos,
                            hi: face_pos + 1,
                        }
                    } else {
                        Space {
                            name: axis_name(aa),
                            lo: coverage.spaces[aa].lo,
                            hi: coverage.spaces[aa].hi,
                        }
                    }
                }));
                let mut face = ConsFieldsGeneric::zeros_with_energy(&face_domain, has_energy)?;
                if has_dye {
                    face.alloc_chi(&face_domain)?;
                }
                faces[2 * ax + side] = Some(face);
                if n_oop > 0 {
                    let mut v = Vec::with_capacity(n_oop);
                    for _ in 0..n_oop {
                        v.push(Field::zeros(&face_domain)?);
                    }
                    oop[2 * ax + side] = Some(v);
                }
                domains[2 * ax + side] = Some(face_domain);
            }
        }
        Ok(FluxRegister {
            faces,
            domains,
            roles: [FaceRole::Local; 6],
            inboxes: std::array::from_fn(|_| None),
            oop_inboxes: std::array::from_fn(|_| None),
            oop,
            coefficients: std::array::from_fn(|_| None),
            pending: std::cell::Cell::new([0; 6]),
            local_units: std::array::from_fn(|_| std::cell::OnceCell::new()),
            debug_coarse_den: std::cell::Cell::new(0.0),
            debug_fine_den: std::cell::Cell::new(0.0),
        })
    }

    /// the face slab of `coverage` on `(ax, side)`, in the coverage's index space
    fn face_domain(coverage: &Domain<D>, ax: usize, side: usize) -> Domain<D> {
        let face_pos = if side == 0 { coverage.spaces[ax].lo } else { coverage.spaces[ax].hi };
        Domain::new(std::array::from_fn(|aa| {
            if aa == ax {
                Space { name: axis_name(aa), lo: face_pos, hi: face_pos + 1 }
            } else {
                Space { name: axis_name(aa), lo: coverage.spaces[aa].lo, hi: coverage.spaces[aa].hi }
            }
        }))
    }

    fn alloc_face(
        domain: &Domain<D>,
        has_energy: bool,
        has_dye: bool,
    ) -> symbi_xpu::Result<ConsFieldsGeneric<D, DOF, Mem>> {
        let mut face = ConsFieldsGeneric::zeros_with_energy(domain, has_energy)?;
        if has_dye {
            face.alloc_chi(domain)?;
        }
        Ok(face)
    }

    /// add contributing faces on the coverage sides named by `faces[ax][side]`: patch
    /// faces flush with this tile's interior boundary that are the patch's real faces,
    /// whose corrected coarse cell lies on the neighbor across the cut
    pub fn with_contributing_faces(
        mut self,
        coverage: &Domain<D>,
        faces: [[bool; 2]; D],
        has_energy: bool,
        has_dye: bool,
        n_oop: usize,
    ) -> symbi_xpu::Result<Self> {
        for ax in 0..D {
            for side in 0..2 {
                if !faces[ax][side] {
                    continue;
                }
                let idx = 2 * ax + side;
                assert!(self.faces[idx].is_none(), "a contributing face replaces no local face");
                let domain = Self::face_domain(coverage, ax, side);
                self.faces[idx] = Some(Self::alloc_face(&domain, has_energy, has_dye)?);
                self.oop[idx] = Self::alloc_oop(&domain, n_oop)?;
                self.coefficients[idx] = Some(Field::zeros(&domain)?);
                self.domains[idx] = Some(domain);
                self.roles[idx] = FaceRole::Contributing;
            }
        }
        Ok(self)
    }

    fn alloc_oop(domain: &Domain<D>, n_oop: usize) -> symbi_xpu::Result<Option<Vec<Field<f64, D, Mem>>>> {
        if n_oop == 0 {
            return Ok(None);
        }
        let mut v = Vec::with_capacity(n_oop);
        for _ in 0..n_oop {
            v.push(Field::zeros(domain)?);
        }
        Ok(Some(v))
    }

    /// a register of receiving faces alone, for a tile whose interior holds the coarse
    /// cells corrected by a neighbor's fine faces: `slabs` names each face by `(ax, side)`
    /// of the patch it bounds and the slab domain in this tile's index space (the cell at
    /// the face for a hi face, the cell above the corrected one for a lo face, as a local
    /// register's slab lies)
    pub fn receiving(
        slabs: &[(usize, usize, Domain<D>)],
        has_energy: bool,
        has_dye: bool,
        n_oop: usize,
    ) -> symbi_xpu::Result<Self> {
        let mut faces: [Option<ConsFieldsGeneric<D, DOF, Mem>>; 6] = std::array::from_fn(|_| None);
        let mut inboxes: [Option<ConsFieldsGeneric<D, DOF, Mem>>; 6] = std::array::from_fn(|_| None);
        let mut oop: [Option<Vec<Field<f64, D, Mem>>>; 6] = std::array::from_fn(|_| None);
        let mut oop_inboxes: [Option<Vec<Field<f64, D, Mem>>>; 6] = std::array::from_fn(|_| None);
        let mut domains: [Option<Domain<D>>; 6] = std::array::from_fn(|_| None);
        let mut coefficients: [Option<Field<f64, D, Mem>>; 6] = std::array::from_fn(|_| None);
        let mut roles = [FaceRole::Local; 6];
        for (ax, side, domain) in slabs {
            let idx = 2 * ax + side;
            assert!(faces[idx].is_none(), "one receiving face per side");
            coefficients[idx] = Some(Field::zeros(domain)?);
            faces[idx] = Some(Self::alloc_face(domain, has_energy, has_dye)?);
            inboxes[idx] = Some(Self::alloc_face(domain, has_energy, has_dye)?);
            oop[idx] = Self::alloc_oop(domain, n_oop)?;
            oop_inboxes[idx] = Self::alloc_oop(domain, n_oop)?;
            domains[idx] = Some(domain.clone());
            roles[idx] = FaceRole::Receiving;
        }
        Ok(FluxRegister {
            faces,
            domains,
            roles,
            inboxes,
            oop_inboxes,
            oop,
            coefficients,
            pending: std::cell::Cell::new([0; 6]),
            local_units: std::array::from_fn(|_| std::cell::OnceCell::new()),
            debug_coarse_den: std::cell::Cell::new(f64::NAN),
            debug_fine_den: std::cell::Cell::new(f64::NAN),
        })
    }

    /// `receiving` over the faces `faces[ax][side]` of `coverage`, the patch box in this tile's
    /// index space, lying beyond its interior
    pub fn receiving_on(
        coverage: &Domain<D>,
        faces: [[bool; 2]; D],
        has_energy: bool,
        has_dye: bool,
        n_oop: usize,
    ) -> symbi_xpu::Result<Self> {
        let mut slabs = Vec::new();
        for ax in 0..D {
            for side in 0..2 {
                if faces[ax][side] {
                    slabs.push((ax, side, Self::face_domain(coverage, ax, side)));
                }
            }
        }
        Self::receiving(&slabs, has_energy, has_dye, n_oop)
    }

    /// the register fields of face `idx`: the gas components, then the out-of-plane ones
    fn all_comps(&self, idx: usize) -> Vec<&Field<f64, D, Mem>> {
        let mut v = comps(self.faces[idx].as_ref().unwrap());
        if let Some(oop) = self.oop[idx].as_ref() {
            v.extend(oop.iter());
        }
        v
    }

    /// the inbox fields of receiving face `idx`, ordered like `all_comps`
    fn all_inbox(&self, idx: usize) -> Vec<&Field<f64, D, Mem>> {
        let mut v = comps(self.inboxes[idx].as_ref().unwrap());
        if let Some(oop) = self.oop_inboxes[idx].as_ref() {
            v.extend(oop.iter());
        }
        v
    }

    /// the fields a transported face moves: a contributing face's unscaled sums and its
    /// coefficient, a receiving face's inbox and the coefficient that lands with it
    fn transported(&self, idx: usize) -> Vec<&Field<f64, D, Mem>> {
        let mut v = match self.roles[idx] {
            FaceRole::Contributing => self.all_comps(idx),
            FaceRole::Receiving => self.all_inbox(idx),
            FaceRole::Local => return Vec::new(),
        };
        v.push(self.coefficients[idx].as_ref().expect("a transported face carries a coefficient"));
        v
    }

    /// clear the inboxes without folding them: a rejected attempt's landed units are discarded
    pub fn clear_inboxes(&self) {
        let fill = KernelId::FieldFill { ndim: D as u8 }.name();
        for (_, inbox, dom) in self.inboxes() {
            for field in inbox {
                dispatch_fields_each::<f64, Mem, D>(fill, dom, &[], &[field], &[], &[0.0]);
            }
        }
    }

    /// the contributing faces: `(2 * ax + side, components in register order, slab domain)`,
    /// the out-of-plane components after the gas ones, the coefficient last
    pub fn contributing(&self) -> Vec<(usize, Vec<&Field<f64, D, Mem>>, &Domain<D>)> {
        (0..2 * D)
            .filter(|&idx| self.roles[idx] == FaceRole::Contributing)
            .map(|idx| (idx, self.transported(idx), self.domains[idx].as_ref().unwrap()))
            .collect()
    }

    /// the inboxes of the receiving faces, keyed and ordered like `contributing`
    pub fn inboxes(&self) -> Vec<(usize, Vec<&Field<f64, D, Mem>>, &Domain<D>)> {
        (0..2 * D)
            .filter(|&idx| self.roles[idx] == FaceRole::Receiving)
            .map(|idx| (idx, self.transported(idx), self.domains[idx].as_ref().unwrap()))
            .collect()
    }

    pub fn has_contributing(&self) -> bool {
        self.roles.iter().any(|r| *r == FaceRole::Contributing)
    }

    pub fn has_receiving(&self) -> bool {
        self.roles.iter().any(|r| *r == FaceRole::Receiving)
    }

    /// reset the contributing faces after their units left, so the next stage's unit holds that
    /// stage's events alone. a unit starts at -0.0, the additive identity for every value
    /// including a signed zero, so the unscaled accumulation `-0.0 + 1.0 * S` is S exactly
    pub fn zero_contributions(&self) {
        let name = KernelId::FieldFill { ndim: D as u8 }.name();
        for (idx, fields, dom) in self.contributing() {
            let _ = idx;
            for field in fields {
                dispatch_fields_each::<f64, Mem, D>(name, dom, &[], &[field], &[], &[-0.0]);
            }
        }
        self.pending.set([0; 6]);
    }

    /// record accumulation event `bit` (1 the gas sums, 2 the out-of-plane sums) on contributing
    /// face `idx` and write its coefficient beside the unit. a second event of one kind before the unit is sent
    /// would combine two operands the receiver cannot separate, and a coefficient that differs
    /// from the unit's would leave one fold coefficient for two scales: both are refused
    fn record_event(&self, idx: usize, bit: u8, coefficient: f64, dom: &Domain<D>) {
        let mut pending = self.pending.get();
        assert!(
            pending[idx] & bit == 0,
            "a second accumulation into the transported unit of register face {idx} before it was sent"
        );
        let field = self.coefficients[idx].as_ref().expect("a contributing face carries a coefficient");
        if pending[idx] != 0 {
            sync_current::<Mem>();
            let held = *field.at(std::array::from_fn(|ax| dom.spaces[ax].lo));
            assert_eq!(held.to_bits(), coefficient.to_bits(), "register face {idx}: two coefficients in one transported unit");
        }
        let fill = KernelId::FieldFill { ndim: D as u8 }.name();
        dispatch_fields_each::<f64, Mem, D>(fill, dom, &[], &[field], &[], &[coefficient]);
        pending[idx] |= bit;
        self.pending.set(pending);
    }

    /// fold every landed unit into its receiving face, `R + coefficient * S` with the
    /// coefficient that landed with the unscaled sums S: the operation, and the operands, of
    /// the local accumulation. then clear the inbox
    pub fn fold_inboxes(&self) {
        let fill = KernelId::FieldFill { ndim: D as u8 }.name();
        for (idx, inbox, dom) in self.inboxes() {
            let face = self.all_comps(idx);
            let probed = probe_axis() == Some(idx / 2) && idx % 2 == 0;
            // every cell of the face holds the one coefficient its unit was accumulated with
            sync_current::<Mem>();
            let coefficient = *inbox.last().expect("a receiving face lands a coefficient").at(probe_cell(dom));
            for (comp, (rf, uf)) in face.into_iter().zip(&inbox).enumerate() {
                let probe = probed && comp == PROBE_COMPONENT;
                let c = probe_cell(dom);
                if probe {
                    probe_sync::<Mem>();
                }
                let (before, unit) = if probe { (*rf.at(c), *uf.at(c)) } else { (0.0, 0.0) };
                apply_unit::<D, Mem>(dom, uf, rf, coefficient);
                if probe {
                    probe_sync::<Mem>();
                    probe_line("fold", self.roles[idx], &c, &[("register", before), ("unit", unit), ("scale", coefficient), ("after", *rf.at(c))]);
                }
            }
            for uf in inbox {
                dispatch_fields_each::<f64, Mem, D>(fill, dom, &[], &[uf], &[], &[0.0]);
            }
        }
    }

    /// the component count of a face: den, the momenta, energy and dye when present, then
    /// the out-of-plane magnetic components
    pub fn n_components(&self) -> usize {
        (0..2 * D).find(|&idx| self.faces[idx].is_some()).map(|idx| self.all_comps(idx).len()).unwrap_or(0)
    }

    /// the slabs the contributions plan moves, in `(2 * ax + side) * n_slot + slot` order with
    /// `n_slot = n_components() + 1`: a contributing face's components and its coefficient, a
    /// receiving face's inbox and the landed coefficient, `filler` where this tile has neither
    /// (the plan references no such slot)
    pub fn plan_fields<'a>(&'a self, filler: &'a Field<f64, D, Mem>) -> Vec<&'a Field<f64, D, Mem>> {
        let n_comp = self.n_components() + 1;
        let mut out = Vec::with_capacity(2 * D * n_comp);
        for idx in 0..2 * D {
            let slab = match self.roles[idx] {
                FaceRole::Local => None,
                _ => Some(self.transported(idx)),
            };
            match slab {
                Some(c) => out.extend(c),
                None => out.extend(std::iter::repeat_n(filler, n_comp)),
            }
        }
        out
    }

    /// the out-of-plane magnetic components of register face `(ax, side)` for a gate
    pub fn oop_face(&self, ax: usize, side: usize) -> Option<(&[Field<f64, D, Mem>], &Domain<D>)> {
        let idx = 2 * ax + side;
        Some((self.oop[idx].as_deref()?, self.domains[idx].as_ref()?))
    }

    /// the register face `(ax, side)` for a gate: its components in register order
    pub fn face(&self, ax: usize, side: usize) -> Option<(Vec<&Field<f64, D, Mem>>, &Domain<D>)> {
        let idx = 2 * ax + side;
        Some((comps(self.faces[idx].as_ref()?), self.domains[idx].as_ref()?))
    }

    pub fn zero(&self) {
        self.debug_coarse_den.set(0.0);
        self.debug_fine_den.set(0.0);
        self.pending.set([0; 6]);
        for (face, dom) in self.faces.iter().chain(self.inboxes.iter()).zip(self.domains.iter().cycle()) {
            if let (Some(reg), Some(dom)) = (face, dom) {
                for field in comps(reg) {
                    for coord in dom.iter() {
                        field.view_mut().set(coord, 0.0);
                    }
                }
            }
        }
    }

    /// subtract the coarse flux through the register faces normal to `dir`:
    /// `R -= F_coarse * A * w`.
    pub fn accumulate_coarse<M: Metric<f64, D> + Copy>(
        &self,
        flux: &[ConsFieldsGeneric<D, DOF, Mem>; D],
        geo: &BlockGeometry<M, f64, D>,
        dir: usize,
        w: f64,
    ) {
        for side in 0..2usize {
            let idx = 2 * dir + side;
            if self.roles[idx] == FaceRole::Contributing {
                continue;
            }
            let sign = if side == 0 { -1.0 } else { 1.0 };
            if let (Some(reg), Some(dom)) = (&self.faces[idx], &self.domains[idx]) {
                for (cc, (rf, ff)) in comps(reg).into_iter().zip(comps(&flux[dir])).enumerate() {
                    for coord in dom.iter() {
                        let a = geo.face_area(coord, dir);
                        let r = *rf.view().at(coord);
                        let contrib = *ff.view().at(coord) * a * w;
                        rf.view_mut().set(coord, r - contrib);
                        if cc == 0 {
                            self.debug_coarse_den
                                .set(self.debug_coarse_den.get() - sign * contrib);
                        }
                    }
                }
            }
        }
    }

    /// add the fine flux through the same interface: `R += sum(F_fine * A_fine) * w`,
    /// summing the `ratio^(D-1)` fine faces covering each coarse face. absolute
    /// indices: the fine face block for coarse face `coord` starts at `ratio * coord`.
    pub fn accumulate_fine<M: Metric<f64, D> + Copy>(
        &self,
        fine_flux: &[ConsFieldsGeneric<D, DOF, Mem>; D],
        fine_geo: &BlockGeometry<M, f64, D>,
        dir: usize,
        w: f64,
        ratio: usize,
    ) {
        let r = ratio as isize;
        for side in 0..2usize {
            let idx = 2 * dir + side;
            if self.roles[idx] == FaceRole::Receiving {
                continue;
            }
            // a transported unit holds unscaled sums and a coefficient, which the uniform
            // cartesian accumulation produces; the metric-weighted accumulation keeps its faces
            // local
            assert!(
                self.roles[idx] != FaceRole::Contributing,
                "a metric-weighted fine accumulation into the transported unit of register face {idx}"
            );
            let sign = if side == 0 { -1.0 } else { 1.0 };
            if let (Some(reg), Some(dom)) = (&self.faces[idx], &self.domains[idx]) {
                for (cc, (rf, ff)) in comps(reg)
                    .into_iter()
                    .zip(comps(&fine_flux[dir]))
                    .enumerate()
                {
                    for coord in dom.iter() {
                        let fine_base: [isize; D] = std::array::from_fn(|ax| coord[ax] * r);
                        let mut sum = 0.0;
                        for_each_transverse::<D>(dir, ratio, |off| {
                            let mut fc = fine_base;
                            for (ax, o) in off.iter().enumerate() {
                                if ax != dir {
                                    fc[ax] += o;
                                }
                            }
                            sum += *ff.view().at(fc) * fine_geo.face_area(fc, dir);
                        });
                        let rv = *rf.view().at(coord);
                        rf.view_mut().set(coord, rv + sum * w);
                        if cc == 0 {
                            self.debug_fine_den
                                .set(self.debug_fine_den.get() + sign * sum * w);
                        }
                    }
                }
            }
        }
    }

    /// apply the accumulated mismatch to the coarse cells abutting the
    /// interface: lo faces correct the cell just below (`u -= R/V`), hi faces
    /// the cell at the face (`u += R/V`).
    pub fn apply<M: Metric<f64, D> + Copy>(
        &self,
        cons: &ConsFieldsGeneric<D, DOF, Mem>,
        geo: &BlockGeometry<M, f64, D>,
    ) {
        for ax in 0..D {
            for side in 0..2usize {
                let idx = 2 * ax + side;
                if self.roles[idx] == FaceRole::Contributing {
                    continue;
                }
                if let (Some(reg), Some(dom)) = (&self.faces[idx], &self.domains[idx]) {
                    let sign: f64 = if side == 0 { -1.0 } else { 1.0 };
                    for (rf, uf) in comps(reg).into_iter().zip(comps(cons)) {
                        for coord in dom.iter() {
                            let mut cell = coord;
                            if side == 0 {
                                cell[ax] -= 1;
                            }
                            let inv_vol = 1.0 / geo.volume(cell);
                            let u = *uf.view().at(cell);
                            uf.view_mut()
                                .set(cell, u + sign * *rf.view().at(coord) * inv_vol);
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// the uniform-cartesian kernel path (cpu + gpu)
// =============================================================================

impl<const D: usize, const DOF: usize, Mem: MemorySpace> FluxRegister<D, DOF, Mem> {
    /// zero all register faces via the fill kernel; the budget diagnostics go
    /// NaN (host-path only).
    pub fn zero_uniform(&self) {
        self.debug_coarse_den.set(f64::NAN);
        self.debug_fine_den.set(f64::NAN);
        self.pending.set([0; 6]);
        let name = KernelId::FieldFill { ndim: D as u8 }.name();
        for (face, dom) in self.faces.iter().chain(self.inboxes.iter()).zip(self.domains.iter().cycle()) {
            if let (Some(reg), Some(dom)) = (face, dom) {
                for field in comps(reg) {
                    dispatch_fields_each::<f64, Mem, D>(name, dom, &[], &[field], &[], &[0.0]);
                }
            }
        }
        for (fields, dom) in self.oop.iter().chain(self.oop_inboxes.iter()).zip(self.domains.iter().cycle()) {
            if let (Some(fields), Some(dom)) = (fields, dom) {
                for field in fields {
                    dispatch_fields_each::<f64, Mem, D>(name, dom, &[], &[field], &[], &[0.0]);
                }
            }
        }
    }

    /// `R -= F_coarse * A * w` with the constant cartesian face area
    /// `A = prod(dx[t != dir])`.
    pub fn accumulate_coarse_uniform(
        &self,
        flux: &[ConsFieldsGeneric<D, DOF, Mem>; D],
        dx: &[f64; D],
        dir: usize,
        w: f64,
    ) {
        let area: f64 = (0..D).filter(|&t| t != dir).map(|t| dx[t]).product();
        let name = KernelId::FieldAxpyShift { ndim: D as u8 }.name();
        let ints = [0i32; 3];
        let scalars = [-area * w];
        for side in 0..2usize {
            let idx = 2 * dir + side;
            if self.roles[idx] == FaceRole::Contributing {
                continue;
            }
            if let (Some(reg), Some(dom)) = (&self.faces[idx], &self.domains[idx]) {
                for (comp, (rf, ff)) in comps(reg).into_iter().zip(comps(&flux[dir])).enumerate() {
                    let probe = probe_axis() == Some(dir) && side == 0 && comp == PROBE_COMPONENT;
                    let c = probe_cell(dom);
                    if probe {
                        probe_sync::<Mem>();
                    }
                    let (before, operand) = if probe { (*rf.at(c), *ff.at(c)) } else { (0.0, 0.0) };
                    dispatch_fields_each::<f64, Mem, D>(
                        name,
                        dom,
                        &[ff],
                        &[rf],
                        &ints[..D],
                        &scalars,
                    );
                    if probe {
                        probe_sync::<Mem>();
                        probe_line("coarse", self.roles[idx], &c, &[("register", before), ("flux", operand), ("scale", scalars[0]), ("after", *rf.at(c))]);
                    }
                }
            }
        }
    }

    /// `R += (A_fine * w) * U` with U the unscaled sum over the `ratio^(D-1)` fine faces per
    /// coarse face (the child-sum kernel; absolute indices): applied on a local face, held with
    /// its coefficient on a contributing face for the receiving face across the cut.
    pub fn accumulate_fine_uniform(
        &self,
        fine_flux: &[ConsFieldsGeneric<D, DOF, Mem>; D],
        fine_dx: &[f64; D],
        dir: usize,
        w: f64,
    ) {
        let area: f64 = (0..D).filter(|&t| t != dir).map(|t| fine_dx[t]).product();
        for side in 0..2usize {
            let idx = 2 * dir + side;
            if self.roles[idx] == FaceRole::Receiving {
                continue;
            }
            if let (Some(reg), Some(dom)) = (&self.faces[idx], &self.domains[idx]) {
                let pairs: Vec<_> = comps(reg).into_iter().zip(comps(&fine_flux[dir])).collect();
                self.accumulate_event(idx, EVENT_GAS, area * w, dom, dir, &pairs, 0);
            }
        }
    }

    /// one fine accumulation event on register face `idx` over `(register field, fine field)`
    /// pairs, the unit slots from `first_slot`: each pair's unscaled sum S of the coincident
    /// fine faces forms a unit U = S, and the face takes `R + coefficient * U`. a local face
    /// applies its unit at once; a contributing face holds it, with the coefficient, for the
    /// receiving face across the cut, which applies it with the same operation and operands
    #[allow(clippy::too_many_arguments)]
    fn accumulate_event(
        &self,
        idx: usize,
        bit: u8,
        coefficient: f64,
        dom: &Domain<D>,
        dir: usize,
        pairs: &[(&Field<f64, D, Mem>, &Field<f64, D, Mem>)],
        first_slot: usize,
    ) {
        let kernel = KernelId::RefineAccFace { axis: dir as u8, ndim: D as u8 }.name();
        let contributing = self.roles[idx] == FaceRole::Contributing;
        if contributing {
            self.record_event(idx, bit, coefficient, dom);
        }
        let probed = probe_axis() == Some(dir) && idx % 2 == 0 && bit == EVENT_GAS;
        for (k, (rf, ff)) in pairs.iter().enumerate() {
            let probe = probed && k == PROBE_COMPONENT;
            let c = probe_cell(dom);
            if probe {
                probe_sync::<Mem>();
            }
            let before = if probe { *rf.at(c) } else { 0.0 };
            let fine = if probe { probe_fine(ff, c, dir) } else { Vec::new() };
            if contributing {
                // the unit is the contributing face itself
                form_unit::<D, Mem>(kernel, dom, ff, rf);
            } else {
                let unit = self.local_unit(idx, first_slot + k, dom);
                form_unit::<D, Mem>(kernel, dom, ff, unit);
                apply_unit::<D, Mem>(dom, unit, rf, coefficient);
            }
            if probe {
                probe_sync::<Mem>();
                let scale = if contributing { 1.0 } else { coefficient };
                let mut values: Vec<(&str, f64)> = vec![("register", before), ("scale", scale)];
                let names = ["fine0", "fine1", "fine2", "fine3"];
                values.extend(names.iter().copied().zip(fine.iter().copied()));
                values.push(("after", *rf.at(c)));
                probe_line("fine", self.roles[idx], &c, &values);
            }
        }
    }

    /// slot `slot` of local face `idx`'s unit, allocated over the face's slab at first use
    fn local_unit(&self, idx: usize, slot: usize, dom: &Domain<D>) -> &Field<f64, D, Mem> {
        let slots = self.local_units[idx].get_or_init(|| {
            (0..self.all_comps(idx).len()).map(|_| Field::zeros(dom).expect("a local register unit")).collect()
        });
        &slots[slot]
    }

    /// apply the correction to the coarse cells abutting the interface with
    /// the constant cartesian volume: lo faces correct the cell below
    /// (`u -= R/V`, the register read at `c + e_ax`), hi faces the cell at the
    /// face (`u += R/V`, read at `c`).
    pub fn apply_uniform(&self, cons: &ConsFieldsGeneric<D, DOF, Mem>, dx: &[f64; D]) {
        let inv_vol = 1.0 / dx.iter().product::<f64>();
        let name = KernelId::FieldAxpyShift { ndim: D as u8 }.name();
        for ax in 0..D {
            for side in 0..2usize {
                let idx = 2 * ax + side;
                if self.roles[idx] == FaceRole::Contributing {
                    continue;
                }
                if let (Some(reg), Some(dom)) = (&self.faces[idx], &self.domains[idx]) {
                    let (cell_dom, arg, sign) = if side == 0 {
                        // corrected cells sit one below the face plane.
                        let spaces: [Space; D] = std::array::from_fn(|aa| {
                            let s = &dom.spaces[aa];
                            let shift = if aa == ax { 1 } else { 0 };
                            Space {
                                name: s.name,
                                lo: s.lo - shift,
                                hi: s.hi - shift,
                            }
                        });
                        let mut ints = [0i32; 3];
                        ints[ax] = 1;
                        (Domain::new(spaces), ints, -1.0)
                    } else {
                        (dom.clone(), [0i32; 3], 1.0)
                    };
                    let scalars = [sign * inv_vol];
                    for (comp, (rf, uf)) in comps(reg).into_iter().zip(comps(cons)).enumerate() {
                        let probe = probe_axis() == Some(ax) && side == 0 && comp == PROBE_COMPONENT;
                        let c = probe_cell(dom);
                        let cell: [isize; D] = std::array::from_fn(|aa| c[aa] - isize::from(aa == ax));
                        if probe {
                            probe_sync::<Mem>();
                        }
                        let (before, register) = if probe { (*uf.at(cell), *rf.at(c)) } else { (0.0, 0.0) };
                        dispatch_fields_each::<f64, Mem, D>(
                            name,
                            &cell_dom,
                            &[rf],
                            &[uf],
                            &arg[..D],
                            &scalars,
                        );
                        if probe {
                            probe_sync::<Mem>();
                            probe_line("apply", self.roles[idx], &cell, &[("cell", before), ("register", register), ("scale", scalars[0]), ("after", *uf.at(cell))]);
                        }
                    }
                }
            }
        }
    }
}

impl<const D: usize, const DOF: usize, Mem: MemorySpace> FluxRegister<D, DOF, Mem> {
    /// `R -= F_coarse * A * w` for the out-of-plane magnetic components' induction fluxes
    /// through the register faces normal to `dir`, constant cartesian face area.
    pub fn accumulate_coarse_uniform_oop(
        &self,
        bflux: &[BfluxFields<D, DOF, Mem>; D],
        dx: &[f64; D],
        dir: usize,
        w: f64,
    ) {
        let area: f64 = (0..D).filter(|&t| t != dir).map(|t| dx[t]).product();
        let name = KernelId::FieldAxpyShift { ndim: D as u8 }.name();
        let ints = [0i32; 3];
        let scalars = [-area * w];
        for side in 0..2usize {
            let idx = 2 * dir + side;
            if self.roles[idx] == FaceRole::Contributing {
                continue;
            }
            if let (Some(reg), Some(dom)) = (&self.oop[idx], &self.domains[idx]) {
                for (k, rf) in reg.iter().enumerate() {
                    dispatch_fields_each::<f64, Mem, D>(name, dom, &[&bflux[dir].f[D + k]], &[rf], &ints[..D], &scalars);
                }
            }
        }
    }

    /// `R += (A_fine * w) * U` for the out-of-plane magnetic components, U the unscaled sum over
    /// the fine faces covering each coarse register face, as `accumulate_fine_uniform` forms
    /// and applies or holds it.
    pub fn accumulate_fine_uniform_oop(
        &self,
        fine_bflux: &[BfluxFields<D, DOF, Mem>; D],
        fine_dx: &[f64; D],
        dir: usize,
        w: f64,
    ) {
        let area: f64 = (0..D).filter(|&t| t != dir).map(|t| fine_dx[t]).product();
        for side in 0..2usize {
            let idx = 2 * dir + side;
            if self.roles[idx] == FaceRole::Receiving {
                continue;
            }
            if let (Some(reg), Some(dom), Some(gas)) = (&self.oop[idx], &self.domains[idx], &self.faces[idx]) {
                let pairs: Vec<_> = reg.iter().enumerate().map(|(k, rf)| (rf, &fine_bflux[dir].f[D + k])).collect();
                self.accumulate_event(idx, EVENT_OUT_OF_PLANE, area * w, dom, dir, &pairs, comps(gas).len());
            }
        }
    }

    /// apply the out-of-plane magnetic corrections to the coarse cell field abutting the
    /// interface, the constant cartesian volume: lo faces correct the cell below, hi faces the
    /// cell at the face, as `apply_uniform` does for the gas.
    pub fn apply_uniform_oop(&self, bcell: &BcellFields<D, DOF, Mem>, dx: &[f64; D]) {
        let inv_vol = 1.0 / dx.iter().product::<f64>();
        let name = KernelId::FieldAxpyShift { ndim: D as u8 }.name();
        for ax in 0..D {
            for side in 0..2usize {
                let idx = 2 * ax + side;
                if self.roles[idx] == FaceRole::Contributing {
                    continue;
                }
                if let (Some(reg), Some(dom)) = (&self.oop[idx], &self.domains[idx]) {
                    let (cell_dom, arg, sign) = if side == 0 {
                        let spaces: [Space; D] = std::array::from_fn(|aa| {
                            let s = &dom.spaces[aa];
                            let shift = if aa == ax { 1 } else { 0 };
                            Space {
                                name: s.name,
                                lo: s.lo - shift,
                                hi: s.hi - shift,
                            }
                        });
                        let mut ints = [0i32; 3];
                        ints[ax] = 1;
                        (Domain::new(spaces), ints, -1.0)
                    } else {
                        (dom.clone(), [0i32; 3], 1.0)
                    };
                    let scalars = [sign * inv_vol];
                    for (k, rf) in reg.iter().enumerate() {
                        dispatch_fields_each::<f64, Mem, D>(name, &cell_dom, &[rf], &[&bcell.b[D + k]], &arg[..D], &scalars);
                    }
                }
            }
        }
    }
}

/// visit every transverse fine-face offset (`0..ratio` on each axis != dir,
/// zero on `dir`): 1 face in 1d, `ratio` in 2d, `ratio^2` in 3d.
fn for_each_transverse<const D: usize>(dir: usize, ratio: usize, mut f: impl FnMut(&[isize; D])) {
    let mut off = [0isize; D];
    visit(&mut off, 0, dir, ratio, &mut f);
    fn visit<const D: usize>(
        off: &mut [isize; D],
        ax: usize,
        dir: usize,
        ratio: usize,
        f: &mut impl FnMut(&[isize; D]),
    ) {
        if ax == D {
            f(off);
            return;
        }
        if ax == dir {
            off[ax] = 0;
            visit(off, ax + 1, dir, ratio, f);
        } else {
            for kk in 0..ratio as isize {
                off[ax] = kk;
                visit(off, ax + 1, dir, ratio, f);
            }
        }
    }
}

// =============================================================================
// tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_geometry::Cartesian;
    use symbi_xpu::HostMemory;

    type Cons1 = ConsFieldsGeneric<1, 1, HostMemory>;

    fn cons_filled(domain: &Domain<1>, den: f64, mom: f64, nrg: f64) -> Cons1 {
        let c = Cons1::zeros(domain).unwrap();
        for coord in domain.iter() {
            c.den.view_mut().set(coord, den);
            c.mom[0].view_mut().set(coord, mom);
            c.nrg_field().unwrap().view_mut().set(coord, nrg);
        }
        c
    }

    fn setup_1d() -> (
        FluxRegister<1, 1, HostMemory>,
        BlockGeometry<Cartesian, f64, 1>,
        BlockGeometry<Cartesian, f64, 1>,
    ) {
        let interior = Domain::new([Space {
            name: "i",
            lo: 0,
            hi: 10,
        }]);
        let coverage = Domain::new([Space {
            name: "i",
            lo: 3,
            hi: 7,
        }]);
        let reg = FluxRegister::new(&coverage, &interior, true, false, 0).unwrap();
        let coarse_geo =
            BlockGeometry::uniform(Cartesian, [0.0], [0.1], std::array::from_fn(|d| d));
        let fine_geo = BlockGeometry::uniform(Cartesian, [0.0], [0.05], std::array::from_fn(|d| d));
        (reg, coarse_geo, fine_geo)
    }

    /// the operands a gpu run recorded at the first cell of a refined box's low x register face
    /// (x-momentum, the first fine accumulation of a step: the register after two coarse
    /// accumulations, the fine coefficient `area * w` and the four fine fluxes), and the results
    /// the single device and the old transport recorded there. both register kernels compile
    /// `dst + scale * x` to one fused multiply-add (`v_fmac_f64`), which `f64::mul_add` is
    const CAPTURED_REGISTER: u64 = 0xbe87c83d621ea656;
    const CAPTURED_COEFFICIENT: u64 = 0x3e7dba4cbaa64fec;
    const CAPTURED_FINE: u64 = 0x3fb9999999999999;
    const LOCAL_RESULT: u64 = 0xbe81d62e0996fcc1;
    const OLD_TRANSPORTED_RESULT: u64 = 0xbe81d62e0996fcc0;

    #[test]
    fn the_transported_unit_reproduces_the_local_fused_accumulation_on_the_captured_operands() {
        let (r, s, f) = (f64::from_bits(CAPTURED_REGISTER), f64::from_bits(CAPTURED_COEFFICIENT), f64::from_bits(CAPTURED_FINE));
        // the fine-face sum in the kernel's order
        let sum = (f + f) + (f + f);
        // the single device: one fused operation on the register
        let local = s.mul_add(sum, r);
        assert_eq!(local.to_bits(), LOCAL_RESULT, "the local accumulation");
        // the old transport: the sender scaled onto a zero unit, the receiver added with 1.0
        let old_unit = s.mul_add(sum, 0.0);
        let old = 1.0f64.mul_add(old_unit, r);
        assert_eq!(old.to_bits(), OLD_TRANSPORTED_RESULT, "the old transported fold");
        assert_ne!(old.to_bits(), local.to_bits(), "the captured operands separate the two protocols; the gate is vacuous");
        // the operand-preserving transport: the sender accumulates the sum onto -0.0 with 1.0,
        // the receiver applies the coefficient that came with it
        let unit = 1.0f64.mul_add(sum, -0.0);
        assert_eq!(unit.to_bits(), sum.to_bits(), "the unit carries the sum exactly");
        let folded = s.mul_add(unit, r);
        assert_eq!(folded.to_bits(), LOCAL_RESULT, "the operand-preserving fold equals the local accumulation");
    }

    #[test]
    fn a_unit_started_at_negative_zero_carries_a_signed_zero_sum_exactly() {
        for sum in [0.0f64, -0.0, 1.0e-300, -2.5] {
            assert_eq!(1.0f64.mul_add(sum, -0.0).to_bits(), sum.to_bits(), "fused: {sum:e}");
            assert_eq!((-0.0 + 1.0 * sum).to_bits(), sum.to_bits(), "separately rounded: {sum:e}");
        }
    }

    #[test]
    fn register_allocates_interface_faces_only() {
        let (reg, _, _) = setup_1d();
        assert!(reg.faces[0].is_some() && reg.faces[1].is_some());
        assert_eq!(reg.domains[0].as_ref().unwrap().spaces[0].lo, 3);
        assert_eq!(reg.domains[1].as_ref().unwrap().spaces[0].lo, 7);

        // coverage flush with the interior boundary: no register face there.
        let interior = Domain::new([Space {
            name: "i",
            lo: 0,
            hi: 10,
        }]);
        let coverage = Domain::new([Space {
            name: "i",
            lo: 0,
            hi: 5,
        }]);
        let touching =
            FluxRegister::<1, 1, HostMemory>::new(&coverage, &interior, true, false, 0).unwrap();
        assert!(touching.faces[0].is_none() && touching.faces[1].is_some());
    }

    #[test]
    fn matching_coarse_and_fine_fluxes_cancel() {
        let (reg, coarse_geo, fine_geo) = setup_1d();
        reg.zero();
        let alloc_c = Domain::new([Space {
            name: "i",
            lo: -2,
            hi: 12,
        }]);
        let alloc_f = Domain::new([Space {
            name: "i",
            lo: 4,
            hi: 16,
        }]);
        let coarse_flux = [cons_filled(&alloc_c, 1.0, 0.5, 2.0)];
        let fine_flux = [cons_filled(&alloc_f, 1.0, 0.5, 2.0)];

        reg.accumulate_coarse(&coarse_flux, &coarse_geo, 0, 1.0);
        reg.accumulate_fine(&fine_flux, &fine_geo, 0, 1.0, 2);

        for idx in [0usize, 1] {
            let dom = reg.domains[idx].as_ref().unwrap();
            let f = reg.faces[idx].as_ref().unwrap();
            for coord in dom.iter() {
                assert!(f.den.view().at(coord).abs() < 1e-14);
                assert!(f.mom[0].view().at(coord).abs() < 1e-14);
                assert!(f.nrg_field().unwrap().view().at(coord).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn apply_signs_and_volume_weighting() {
        let (reg, coarse_geo, _) = setup_1d();
        reg.zero();
        let lo = reg.faces[0].as_ref().unwrap();
        lo.den.view_mut().set([3], 0.1);
        let hi = reg.faces[1].as_ref().unwrap();
        hi.den.view_mut().set([7], 0.3);

        let alloc = Domain::new([Space {
            name: "i",
            lo: -2,
            hi: 12,
        }]);
        let cons = cons_filled(&alloc, 1.0, 0.0, 1.0);
        reg.apply(&cons, &coarse_geo);

        // dx = 0.1 -> inv_vol = 10; lo face corrects cell 2 with -R/V, hi corrects cell 7 with +R/V.
        assert!((*cons.den.view().at([2]) - 0.0).abs() < 1e-13);
        assert!((*cons.den.view().at([7]) - 4.0).abs() < 1e-13);
        // untouched elsewhere.
        assert!((*cons.den.view().at([5]) - 1.0).abs() < 1e-14);
    }
}
