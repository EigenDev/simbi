// =============================================================================
// emf_register.rs
//
// the edge-EMF reflux register — the constrained-
// transport analogue of the flux register. the CT curl applies, per coarse
// face bface_a(c) with (p1, p2) cyclic of a:
//   bface_a(c) += dt * ( (E_p1(c+e_p2) - E_p1(c)) / dx_p2
//                      - (E_p2(c+e_p1) - E_p2(c)) / dx_p1 )
// (the rmhd_ct_curl kernel formula). across a coarse-fine interface the
// restriction replaces the covered faces (interface included) with the fine
// area-averages — equivalent to a CT update with the time/length-averaged
// fine edge EMFs. the outside faces adjacent to the interface edges were
// updated with the coarse EMFs; this register accumulates the mismatch
//   dPhi(edge) = sum_substeps dt_f * (length-avg fine EMF) - dt * (coarse EMF)
// and corrects each adjacent non-restricted face with the curl coefficient —
// div(curl) = 0 cell-by-cell, so the staggered divB stays at machine zero
// across the level jump.
//
// slab-field storage, all kernels (cpu + gpu through the dispatch boundary): for
// each edge direction t and each coarse-fine side plane (axis b != t, side s,
// skipped where the coverage touches the coarse interior), one thin Field
// over the in-plane t-edge slab (t cells x a nodes at the plane's b node,
// a = the third axis). a box-corner edge lies in two side slabs — both copies
// accumulate the identical value (same inputs, same kernels) and each side's
// apply consumes its own copy for a different face set, so no dedup is needed:
//   zero            — field_fill per slab
//   accumulate_*    — field_axpy_shift (coarse, arg 0, scale -dt) /
//                     refine_acc_edge (fine sub-edge pair, scale dt_f/ratio)
//   apply           — one field_axpy_shift gather per slab: the outside
//                     a-faces at b-cell (cov.lo-1 | cov.hi) each read their
//                     single in-plane edge (arg_b = +1 | 0) with the curl
//                     coefficient sign(t, a, side) * 1/dx_b. the restricted
//                     faces never appear in these sets by construction.
//
// both levels' effective per-step EMF is read from the efield buffers after a
// level's stage loop (post_godunov writes the rk2 time-average in place before
// the single curl; euler keeps the raw stage EMF). cartesian only — the curl
// coefficients are 1/dx.
//
// a slab has a role like a flux-register face. a local slab holds the fine
// sub-edges and the corrected outside faces on this tile. a contributing slab
// lies on a patch plane coinciding with a decomposition cut: the fine sub-edges
// are here and the corrected faces on the neighbor, so the slab accumulates one
// substep's unit from zero for the driver to send and clear; the coarse EMF and
// the application belong to the neighbor. a receiving slab is that neighbor's
// side: the coarse EMF at its own root step, each substep's unit folded from an
// inbox with `R += unit` in substep order, then applied to its outside faces.
// a box-corner edge of a cut plane lies in the transported slab and in the
// transverse plane's local slab of the tile holding the fine sub-edges; each
// copy accumulates the identical value, the local one corrects that tile's
// copy of the faces at the corner, and the neighbor's copy of a face on the
// cut takes the corrected value from the tile that holds the fine faces.
//
// usage:
//  let reg = EmfRegister::new(&coverage, &coarse_interior)?;
//  reg.zero();
//  reg.accumulate_coarse(&mhd.efield, dt);
//  reg.accumulate_fine(&fine_mhd.efield, fine_dt);
//  reg.apply(&mhd.bface, &inv_dx);
// =============================================================================

use symbi_algebra::{Domain, Space};
use symbi_grid::Field;
use symbi_xpu::MemorySpace;

use symbi_ir::KernelId;
use symbi_sim::state::{BfaceFields, EfieldFields, axis_name};

pub use super::flux_register::FaceRole;
use symbi_substrate::regimes::substrate_kernels::dispatch_fields_each;

/// the refinement ratio baked into the registered edge kernels.
const RATIO: i64 = 2;

/// one accumulator slab: the in-plane edges of direction `tt` on the coverage
/// side plane (`bb`, `side`).
struct EdgeSlab<const D: usize, Mem: MemorySpace> {
    /// the edge direction as a physical axis: a grid axis in 3D, the missing axis (2) on a
    /// 2.5D grid, whose corner EMF runs along it.
    tt: usize,
    /// the efield storage slot holding that edge's EMF.
    slot: usize,
    bb: usize,
    /// the axis of the faces the edge's curl corrects, the third axis of (t, b).
    aa: usize,
    side: usize,
    dom: Domain<D>,
    reg: Field<f64, D, Mem>,
    role: FaceRole,
    /// the landing slab of a receiving slab: the neighbor's unit arrives here and is folded
    inbox: Option<Field<f64, D, Mem>>,
}

/// per coarse-fine interface edge-EMF accumulator, slab-field storage.
pub struct EmfRegister<const D: usize, Mem: MemorySpace> {
    slabs: Vec<EdgeSlab<D, Mem>>,
    /// the coverage box (absolute coarse indices) — fixes the outside-face
    /// gather sets.
    coverage: Domain<D>,
}

impl<const D: usize, Mem: MemorySpace> EmfRegister<D, Mem> {
    /// build the side-plane edge slabs from the coverage box (absolute coarse
    /// indices); planes flush with the coarse interior boundary have no
    /// outside cells and are skipped. in 3D every grid axis carries edges; on a 2.5D grid the
    /// one edge family is the corner EMF along the missing axis, stored in slot 0, whose curl
    /// corrects the in-plane faces.
    pub fn new(coverage: &Domain<D>, coarse_interior: &Domain<D>) -> symbi_xpu::Result<Self> {
        assert!(D == 2 || D == 3, "EmfRegister: the staggered CT stack is 2.5D or 3D");
        let mut slabs = Vec::new();
        for (tt, slot) in Self::families() {
            for bb in 0..D {
                if bb == tt {
                    continue;
                }
                for side in 0..2usize {
                    if side == 0 && coverage.spaces[bb].lo == coarse_interior.spaces[bb].lo {
                        continue;
                    }
                    if side == 1 && coverage.spaces[bb].hi == coarse_interior.spaces[bb].hi {
                        continue;
                    }
                    slabs.push(Self::slab(coverage, tt, slot, bb, side, FaceRole::Local)?);
                }
            }
        }
        Ok(EmfRegister {
            slabs,
            coverage: coverage.clone(),
        })
    }

    /// (edge axis, storage slot) families: the three grid axes in 3D, the missing axis in 2D
    fn families() -> Vec<(usize, usize)> {
        if D == 3 { vec![(0, 0), (1, 1), (2, 2)] } else { vec![(2, 0)] }
    }

    /// the in-plane t-edge slab of `coverage` on the side plane `(bb, side)`
    fn slab(
        coverage: &Domain<D>,
        tt: usize,
        slot: usize,
        bb: usize,
        side: usize,
        role: FaceRole,
    ) -> symbi_xpu::Result<EdgeSlab<D, Mem>> {
        let aa = 3 - tt - bb;
        let plane = if side == 0 { coverage.spaces[bb].lo } else { coverage.spaces[bb].hi };
        let dom = Domain::new(std::array::from_fn(|ax| {
            let s = &coverage.spaces[ax];
            let (lo, hi) = if ax == bb {
                (plane, plane + 1)
            } else if ax == aa {
                // a-node range, both extremes (the box corners).
                (s.lo, s.hi + 1)
            } else {
                // t-cell range along the edge direction.
                (s.lo, s.hi)
            };
            Space { name: axis_name(ax), lo, hi }
        }));
        let reg = Field::zeros(&dom)?;
        let inbox = if role == FaceRole::Receiving { Some(Field::zeros(&dom)?) } else { None };
        Ok(EdgeSlab { tt, slot, bb, aa, side, dom, reg, role, inbox })
    }

    /// add contributing slabs on the coverage planes named by `faces[bb][side]`: patch
    /// planes flush with this tile's interior boundary that are the patch's real faces,
    /// whose corrected outside faces lie on the neighbor across the cut
    pub fn with_contributing_planes(mut self, coverage: &Domain<D>, faces: [[bool; 2]; D]) -> symbi_xpu::Result<Self> {
        for (tt, slot) in Self::families() {
            for bb in 0..D {
                if bb == tt {
                    continue;
                }
                for side in 0..2usize {
                    if !faces[bb][side] {
                        continue;
                    }
                    assert!(
                        !self.slabs.iter().any(|s| s.tt == tt && s.bb == bb && s.side == side),
                        "a contributing slab replaces no local slab"
                    );
                    self.slabs.push(Self::slab(coverage, tt, slot, bb, side, FaceRole::Contributing)?);
                }
            }
        }
        Ok(self)
    }

    /// a register of receiving slabs alone, for a tile whose interior holds the outside
    /// faces a neighbor's fine sub-edges correct: `coverage` is the patch box in this tile's
    /// index space (lying beyond its interior), `faces[bb][side]` the patch planes on the
    /// cuts it sits beside
    pub fn receiving(coverage: &Domain<D>, faces: [[bool; 2]; D]) -> symbi_xpu::Result<Self> {
        assert!(D == 2 || D == 3, "EmfRegister: the staggered CT stack is 2.5D or 3D");
        let mut slabs = Vec::new();
        for (tt, slot) in Self::families() {
            for bb in 0..D {
                if bb == tt {
                    continue;
                }
                for side in 0..2usize {
                    if faces[bb][side] {
                        slabs.push(Self::slab(coverage, tt, slot, bb, side, FaceRole::Receiving)?);
                    }
                }
            }
        }
        Ok(EmfRegister { slabs, coverage: coverage.clone() })
    }

    /// the coverage box the slabs bound, in this tile's index space
    pub fn coverage(&self) -> &Domain<D> {
        &self.coverage
    }

    /// the field slot of a slab in the contributions plan: `(2 * bb + side) * families + family`
    fn plan_slot(tt: usize, bb: usize, side: usize) -> usize {
        let families = Self::families();
        let family = families.iter().position(|(t, _)| *t == tt).expect("a known edge family");
        (2 * bb + side) * families.len() + family
    }

    /// the contributing slabs: `(plan slot, register field, slab domain)`
    pub fn contributing(&self) -> Vec<(usize, &Field<f64, D, Mem>, &Domain<D>)> {
        self.slabs
            .iter()
            .filter(|s| s.role == FaceRole::Contributing)
            .map(|s| (Self::plan_slot(s.tt, s.bb, s.side), &s.reg, &s.dom))
            .collect()
    }

    /// the inboxes of the receiving slabs, keyed like `contributing`
    pub fn inboxes(&self) -> Vec<(usize, &Field<f64, D, Mem>, &Domain<D>)> {
        self.slabs
            .iter()
            .filter(|s| s.role == FaceRole::Receiving)
            .map(|s| (Self::plan_slot(s.tt, s.bb, s.side), s.inbox.as_ref().unwrap(), &s.dom))
            .collect()
    }

    pub fn has_contributing(&self) -> bool {
        self.slabs.iter().any(|s| s.role == FaceRole::Contributing)
    }

    pub fn has_receiving(&self) -> bool {
        self.slabs.iter().any(|s| s.role == FaceRole::Receiving)
    }

    /// the slabs the contributions plan moves, `2 * D * families` slots: a contributing
    /// slab's register, a receiving slab's inbox, `filler` where this tile has neither
    pub fn plan_fields<'a>(&'a self, filler: &'a Field<f64, D, Mem>) -> Vec<&'a Field<f64, D, Mem>> {
        let mut out = vec![filler; 2 * D * Self::families().len()];
        for (slot, field, _) in self.contributing().into_iter().chain(self.inboxes()) {
            out[slot] = field;
        }
        out
    }

    /// zero the contributing slabs after their units left
    pub fn zero_contributions(&self) {
        let name = KernelId::FieldFill { ndim: D as u8 }.name();
        for (_, field, dom) in self.contributing() {
            dispatch_fields_each::<f64, Mem, D>(name, dom, &[], &[field], &[], &[0.0]);
        }
    }

    /// clear the inboxes without folding them: a rejected attempt's landed units are discarded
    pub fn clear_inboxes(&self) {
        let fill = KernelId::FieldFill { ndim: D as u8 }.name();
        for (_, inbox, dom) in self.inboxes() {
            dispatch_fields_each::<f64, Mem, D>(fill, dom, &[], &[inbox], &[], &[0.0]);
        }
    }

    /// fold every landed unit into its receiving slab, `R += unit`, then clear the inbox
    pub fn fold_inboxes(&self) {
        let axpy = KernelId::FieldAxpyShift { ndim: D as u8 }.name();
        let fill = KernelId::FieldFill { ndim: D as u8 }.name();
        let ints = [0i32; 3];
        for slab in self.slabs.iter().filter(|s| s.role == FaceRole::Receiving) {
            let inbox = slab.inbox.as_ref().unwrap();
            dispatch_fields_each::<f64, Mem, D>(axpy, &slab.dom, &[inbox], &[&slab.reg], &ints[..D], &[1.0]);
            dispatch_fields_each::<f64, Mem, D>(fill, &slab.dom, &[], &[inbox], &[], &[0.0]);
        }
    }

    /// the slab of edge family `tt` on the plane `(bb, side)` for a gate: its register and domain
    pub fn slab_of(&self, tt: usize, bb: usize, side: usize) -> Option<(&Field<f64, D, Mem>, &Domain<D>)> {
        self.slabs.iter().find(|s| s.tt == tt && s.bb == bb && s.side == side).map(|s| (&s.reg, &s.dom))
    }

    pub fn zero(&self) {
        let name = KernelId::FieldFill { ndim: D as u8 }.name();
        for slab in &self.slabs {
            dispatch_fields_each::<f64, Mem, D>(name, &slab.dom, &[], &[&slab.reg], &[], &[0.0]);
            if let Some(inbox) = slab.inbox.as_ref() {
                dispatch_fields_each::<f64, Mem, D>(name, &slab.dom, &[], &[inbox], &[], &[0.0]);
            }
        }
    }

    /// subtract the coarse effective EMF: `R -= dt * E_t(edge)`. call after
    /// the coarse level's stage loop (efield then holds the EMF the step's
    /// curl actually applied).
    pub fn accumulate_coarse(&self, efield: &EfieldFields<D, Mem>, dt: f64) {
        let name = KernelId::FieldAxpyShift { ndim: D as u8 }.name();
        let ints = [0i32; 3];
        for slab in self.slabs.iter().filter(|s| s.role != FaceRole::Contributing) {
            dispatch_fields_each::<f64, Mem, D>(
                name,
                &slab.dom,
                &[&efield[slab.slot]],
                &[&slab.reg],
                &ints[..D],
                &[-dt],
            );
        }
    }

    /// add the fine effective EMF, length-averaged over the two fine
    /// sub-edges of each coarse edge (absolute indices: the sub-edges sit at
    /// `2g` and `2g + e_t`): `R += (dt_f / 2) * (E_f(2g) + E_f(2g + e_t))`.
    /// call after each fine substep's stage loop.
    pub fn accumulate_fine(&self, efield: &EfieldFields<D, Mem>, dt_f: f64) {
        for slab in self.slabs.iter().filter(|s| s.role != FaceRole::Receiving) {
            // a 3D coarse edge covers two fine sub-edges, length-averaged; a 2.5D coarse corner
            // edge runs along the missing axis and covers the one coincident fine corner.
            let (name, scale) = if D == 3 {
                (
                    KernelId::RefineAccEdge {
                        axis: slab.tt as u8,
                        ndim: D as u8,
                    }
                    .name()
                    .to_string(),
                    dt_f / RATIO as f64,
                )
            } else {
                (format!("refine_acc_node_{D}d"), dt_f)
            };
            dispatch_fields_each::<f64, Mem, D>(
                &name,
                &slab.dom,
                &[&efield[slab.slot]],
                &[&slab.reg],
                &[],
                &[scale],
            );
        }
    }

    /// correct the outside faces: per slab, the a-faces (a = the third axis)
    /// of the cells just outside the plane each gather their single in-plane
    /// edge with the curl coefficient. with the covered faces (interface
    /// included) replaced by the fine restriction, this closes div(curl) = 0
    /// on every outside cell.
    pub fn apply(&self, bface: &BfaceFields<D, Mem>, inv_dx: &[f64; D]) {
        self.apply_with(bface, inv_dx, None);
    }

    /// `apply` on a curved spacetime: the GR curl updates a face's B by the curl of the densitized
    /// EMF over the face's own sqrt(gamma), so each corrected face takes the correction on its flux
    /// sqrt(gamma) B, the faces of the slab staged in the face metric's scratch and carried back
    pub fn apply_densitized(&self, bface: &BfaceFields<D, Mem>, inv_dx: &[f64; D], metric: &super::transfer::FaceMetric<D, Mem>) {
        self.apply_with(bface, inv_dx, Some(metric));
    }

    fn apply_with(&self, bface: &BfaceFields<D, Mem>, inv_dx: &[f64; D], metric: Option<&super::transfer::FaceMetric<D, Mem>>) {
        let name = KernelId::FieldAxpyShift { ndim: D as u8 }.name();
        let product = KernelId::FieldProduct { ndim: D as u8 }.name();
        for slab in self.slabs.iter().filter(|s| s.role != FaceRole::Contributing) {
            let (tt, bb, side, aa) = (slab.tt, slab.bb, slab.side, slab.aa);
            // the curl coefficient of edge E_t in face bface_a: with
            // (p1, p2) cyclic of a, t == p1 pairs the faces (g: -1/dx_p2,
            // g - e_p2: +1/dx_p2) and t == p2 the faces (g: +1/dx_p1,
            // g - e_p1: -1/dx_p1); the offset axis is b in both cases. the
            // lo-side outside face is `g - e_b`, the hi-side one is `g`.
            let p1 = (aa + 1) % 3;
            let t_is_p1 = tt == p1;
            let sign = match (t_is_p1, side == 0) {
                (true, true) => 1.0,
                (true, false) => -1.0,
                (false, true) => -1.0,
                (false, false) => 1.0,
            };
            let scale = sign * inv_dx[bb];
            // exec = the outside a-face slab: b at the outside cell, the edge
            // read shifted back onto the plane node.
            let cov = &self.coverage;
            let dom = Domain::new(std::array::from_fn(|ax| {
                let s = &cov.spaces[ax];
                let (lo, hi) = if ax == bb {
                    if side == 0 {
                        (s.lo - 1, s.lo)
                    } else {
                        (s.hi, s.hi + 1)
                    }
                } else if ax == aa {
                    (s.lo, s.hi + 1)
                } else {
                    (s.lo, s.hi)
                };
                Space {
                    name: axis_name(ax),
                    lo,
                    hi,
                }
            }));
            let mut ints = [0i32; 3];
            if side == 0 {
                ints[bb] = 1;
            }
            match metric {
                None => dispatch_fields_each::<f64, Mem, D>(name, &dom, &[&slab.reg], &[&bface[aa]], &ints[..D], &[scale]),
                Some(m) => {
                    dispatch_fields_each::<f64, Mem, D>(product, &dom, &[&m.sqrtg[aa], &bface[aa]], &[&m.scratch[aa]], &[], &[]);
                    dispatch_fields_each::<f64, Mem, D>(name, &dom, &[&slab.reg], &[&m.scratch[aa]], &ints[..D], &[scale]);
                    dispatch_fields_each::<f64, Mem, D>(product, &dom, &[&m.scratch[aa], &m.inv_sqrtg[aa]], &[&bface[aa]], &[], &[]);
                }
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

    fn cov3() -> Domain<3> {
        Domain::new([
            Space {
                name: "i",
                lo: 4,
                hi: 8,
            },
            Space {
                name: "j",
                lo: 4,
                hi: 8,
            },
            Space {
                name: "k",
                lo: 0,
                hi: 2,
            },
        ])
    }

    fn interior3() -> Domain<3> {
        Domain::new([
            Space {
                name: "i",
                lo: 0,
                hi: 16,
            },
            Space {
                name: "j",
                lo: 0,
                hi: 16,
            },
            Space {
                name: "k",
                lo: 0,
                hi: 2,
            },
        ])
    }

    #[test]
    fn slabs_cover_the_cf_side_planes_only() {
        // z touches the interior on both sides -> no z-plane slabs; per edge
        // direction t the b-planes are the CF transverse axes only.
        let reg = EmfRegister::<3, symbi_xpu::HostMemory>::new(&cov3(), &interior3()).unwrap();
        // t = 0 (x): planes b in {1} (y CF both sides; z touches) -> 2 slabs.
        // t = 1 (y): planes b in {0} -> 2. t = 2 (z): b in {0, 1} -> 4.
        assert_eq!(reg.slabs.len(), 8);
        // a z-edge slab on the x-lo plane: x fixed at node 4, y spans nodes
        // [4, 9), z spans cells [0, 2).
        let s = reg
            .slabs
            .iter()
            .find(|s| s.tt == 2 && s.bb == 0 && s.side == 0)
            .expect("z-edge slab on the x-lo plane");
        assert_eq!((s.dom.spaces[0].lo, s.dom.spaces[0].hi), (4, 5));
        assert_eq!((s.dom.spaces[1].lo, s.dom.spaces[1].hi), (4, 9));
        assert_eq!((s.dom.spaces[2].lo, s.dom.spaces[2].hi), (0, 2));
    }

    #[test]
    fn corner_edges_live_in_two_slabs() {
        // the box corner edge (t = 2 at x-node 4, y-node 4) appears in both
        // the x-lo and y-lo slabs — each side's apply consumes its own copy
        // for a different face set, so the overlap is by design.
        let reg = EmfRegister::<3, symbi_xpu::HostMemory>::new(&cov3(), &interior3()).unwrap();
        let holds = |s: &EdgeSlab<3, symbi_xpu::HostMemory>| s.tt == 2 && s.dom.contains([4, 4, 0]);
        let count = reg.slabs.iter().filter(|s| holds(s)).count();
        assert_eq!(count, 2, "corner edge should appear in exactly two slabs");
    }
}
