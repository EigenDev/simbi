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
use symbi_substrate::regimes::substrate_kernels::dispatch_fields_each;

/// the refinement ratio baked into the registered edge kernels.
const RATIO: i64 = 2;

/// one accumulator slab: the in-plane edges of direction `tt` on the coverage
/// side plane (`bb`, `side`).
struct EdgeSlab<const D: usize, Mem: MemorySpace> {
    tt: usize,
    bb: usize,
    side: usize,
    dom: Domain<D>,
    reg: Field<f64, D, Mem>,
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
    /// outside cells and are skipped. the staggered CT machinery is 3d-only.
    pub fn new(coverage: &Domain<D>, coarse_interior: &Domain<D>) -> symbi_xpu::Result<Self> {
        assert!(D == 3, "EmfRegister: the staggered CT stack is 3d-only");
        let mut slabs = Vec::new();
        for tt in 0..3 {
            for bb in 0..3 {
                if bb == tt {
                    continue;
                }
                let aa = 3 - tt - bb;
                for side in 0..2usize {
                    if side == 0 && coverage.spaces[bb].lo == coarse_interior.spaces[bb].lo {
                        continue;
                    }
                    if side == 1 && coverage.spaces[bb].hi == coarse_interior.spaces[bb].hi {
                        continue;
                    }
                    let plane = if side == 0 {
                        coverage.spaces[bb].lo
                    } else {
                        coverage.spaces[bb].hi
                    };
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
                        Space {
                            name: axis_name(ax),
                            lo,
                            hi,
                        }
                    }));
                    let reg = Field::zeros(&dom)?;
                    slabs.push(EdgeSlab {
                        tt,
                        bb,
                        side,
                        dom,
                        reg,
                    });
                }
            }
        }
        Ok(EmfRegister {
            slabs,
            coverage: coverage.clone(),
        })
    }

    pub fn zero(&self) {
        let name = KernelId::FieldFill { ndim: D as u8 }.name();
        for slab in &self.slabs {
            dispatch_fields_each::<f64, Mem, D>(name, &slab.dom, &[], &[&slab.reg], &[], &[0.0]);
        }
    }

    /// subtract the coarse effective EMF: `R -= dt * E_t(edge)`. call after
    /// the coarse level's stage loop (efield then holds the EMF the step's
    /// curl actually applied).
    pub fn accumulate_coarse(&self, efield: &EfieldFields<D, Mem>, dt: f64) {
        let name = KernelId::FieldAxpyShift { ndim: D as u8 }.name();
        let ints = [0i32; 3];
        for slab in &self.slabs {
            dispatch_fields_each::<f64, Mem, D>(
                name,
                &slab.dom,
                &[&efield[slab.tt]],
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
        let scale = dt_f / RATIO as f64;
        for slab in &self.slabs {
            let name = KernelId::RefineAccEdge {
                axis: slab.tt as u8,
                ndim: D as u8,
            }
            .name();
            dispatch_fields_each::<f64, Mem, D>(
                name,
                &slab.dom,
                &[&efield[slab.tt]],
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
        let name = KernelId::FieldAxpyShift { ndim: D as u8 }.name();
        for slab in &self.slabs {
            let (tt, bb, side) = (slab.tt, slab.bb, slab.side);
            let aa = 3 - tt - bb;
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
            dispatch_fields_each::<f64, Mem, D>(
                name,
                &dom,
                &[&slab.reg],
                &[&bface[aa]],
                &ints[..D],
                &[scale],
            );
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
