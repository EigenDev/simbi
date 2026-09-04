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
//   accumulate_fine:    R += sum(F_fine * A_fine) * w
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
use symbi_sim::state::{ConsFieldsGeneric, axis_name};
use symbi_substrate::regimes::substrate_kernels::dispatch_fields_each;

/// per coarse-fine interface accumulator. faces[2*ax + side] is the register
/// for the coverage boundary slab on that side; None where the coverage
/// touches the coarse interior boundary (no coarse cell to correct there).
pub struct FluxRegister<const D: usize, const DOF: usize, Mem: MemorySpace> {
    faces: [Option<ConsFieldsGeneric<D, DOF, Mem>>; 6],
    domains: [Option<Domain<D>>; 6],
    /// sign-weighted running den totals of the coarse / fine accumulations
    /// (conservation-budget diagnostics: the pure-coarse outside-mass change
    /// must equal -debug_coarse_den, the fine interior change -debug_fine_den).
    pub debug_coarse_den: std::cell::Cell<f64>,
    pub debug_fine_den: std::cell::Cell<f64>,
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
    ) -> symbi_xpu::Result<Self> {
        let mut faces: [Option<ConsFieldsGeneric<D, DOF, Mem>>; 6] = std::array::from_fn(|_| None);
        let mut domains: [Option<Domain<D>>; 6] = std::array::from_fn(|_| None);

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
                domains[2 * ax + side] = Some(face_domain);
            }
        }
        Ok(FluxRegister {
            faces,
            domains,
            debug_coarse_den: std::cell::Cell::new(0.0),
            debug_fine_den: std::cell::Cell::new(0.0),
        })
    }

    pub fn zero(&self) {
        self.debug_coarse_den.set(0.0);
        self.debug_fine_den.set(0.0);
        for (face, dom) in self.faces.iter().zip(self.domains.iter()) {
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
        let name = KernelId::FieldFill { ndim: D as u8 }.name();
        for (face, dom) in self.faces.iter().zip(self.domains.iter()) {
            if let (Some(reg), Some(dom)) = (face, dom) {
                for field in comps(reg) {
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
            if let (Some(reg), Some(dom)) = (&self.faces[idx], &self.domains[idx]) {
                for (rf, ff) in comps(reg).into_iter().zip(comps(&flux[dir])) {
                    dispatch_fields_each::<f64, Mem, D>(
                        name,
                        dom,
                        &[ff],
                        &[rf],
                        &ints[..D],
                        &scalars,
                    );
                }
            }
        }
    }

    /// `R += sum(F_fine) * A_fine * w` over the `ratio^(D-1)` fine faces per
    /// coarse face (the accumulating child-sum kernel; absolute indices).
    pub fn accumulate_fine_uniform(
        &self,
        fine_flux: &[ConsFieldsGeneric<D, DOF, Mem>; D],
        fine_dx: &[f64; D],
        dir: usize,
        w: f64,
    ) {
        let area: f64 = (0..D).filter(|&t| t != dir).map(|t| fine_dx[t]).product();
        let name = KernelId::RefineAccFace {
            axis: dir as u8,
            ndim: D as u8,
        }
        .name();
        let scalars = [area * w];
        for side in 0..2usize {
            let idx = 2 * dir + side;
            if let (Some(reg), Some(dom)) = (&self.faces[idx], &self.domains[idx]) {
                for (rf, ff) in comps(reg).into_iter().zip(comps(&fine_flux[dir])) {
                    dispatch_fields_each::<f64, Mem, D>(name, dom, &[ff], &[rf], &[], &scalars);
                }
            }
        }
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
                    for (rf, uf) in comps(reg).into_iter().zip(comps(cons)) {
                        dispatch_fields_each::<f64, Mem, D>(
                            name,
                            &cell_dom,
                            &[rf],
                            &[uf],
                            &arg[..D],
                            &scalars,
                        );
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
        let reg = FluxRegister::new(&coverage, &interior, true, false).unwrap();
        let coarse_geo =
            BlockGeometry::uniform(Cartesian, [0.0], [0.1], std::array::from_fn(|d| d));
        let fine_geo = BlockGeometry::uniform(Cartesian, [0.0], [0.05], std::array::from_fn(|d| d));
        (reg, coarse_geo, fine_geo)
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
            FluxRegister::<1, 1, HostMemory>::new(&coverage, &interior, true, false).unwrap();
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
