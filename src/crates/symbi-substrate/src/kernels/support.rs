// =============================================================================
// kernels/support.rs
//
// regime-independent plumbing shared across every KernelSet implementation.
// nothing here depends on the physics regime — only on Domain<D>, Field<D>,
// ghost classification, and boundary conditions.
//
// these helpers exist because ~60% of a KernelSet impl is identical across
// regimes: face-domain construction, ghost region analysis with per-axis
// bc parameterization, cfl scaling.
//
// usage:
//   // face-parallel dispatch domain for axis `dir`
//   let face_dom = interior.face_domain(dir);
//
//   // ghost fill: regime provides the per-region dispatch closure
//   ghost_fill_driver::<D>(interior, allocated, &bc)
//       .drive(|region, params| {
//           my_ghost_kernel(&fields..., params.as_args()).on(&region.domain);
//       });
//
//   // cfl: provide the map kernel and reduction result, get the timestep
//   let dt = cfl_from_smax(s_max, cfl_number, dx_min);
// =============================================================================

use symbi_algebra::{Domain, Space};
use symbi_grid::ghost::{BcType, FaceSide, GhostRegion, GhostType, analyze_ghost_regions};
use symbi_sim::state::{Boundaries, BoundaryType};

// =============================================================================
// FaceDomain: one-face-wider-along-dir dispatch domain
// =============================================================================

/// extension trait: face-parallel dispatch domain for axis `dir`.
/// the returned domain is the interior extended by one cell on the `hi`
/// side of `dir` (faces are one more than cells along the flux axis).
pub trait FaceDomain<const D: usize> {
    fn face_domain(&self, dir: usize) -> Domain<D>;
}

impl<const D: usize> FaceDomain<D> for Domain<D> {
    fn face_domain(&self, dir: usize) -> Domain<D> {
        assert!(dir < D, "face_domain: dir {} out of range for D={}", dir, D);
        Domain::new(std::array::from_fn(|ax| Space {
            name: self.spaces[ax].name,
            lo: self.spaces[ax].lo,
            hi: self.spaces[ax].hi + if ax == dir { 1 } else { 0 },
        }))
    }
}

// =============================================================================
// BoundaryType -> BcType lookup (regime-independent)
// =============================================================================

/// convert the per-axis `Boundaries<D>` into a per-axis [lo, hi] `BcType` table.
pub fn to_bc_array<const D: usize>(boundaries: &Boundaries<D>) -> [[BcType; 2]; D] {
    std::array::from_fn(|ax| {
        let conv = |bt: BoundaryType| match bt {
            BoundaryType::Periodic => BcType::Periodic,
            BoundaryType::Outflow => BcType::Outflow,
            BoundaryType::Reflect => BcType::Reflect,
            BoundaryType::Axis => BcType::Axis,
            BoundaryType::CoarseFine => BcType::Skip,
            // driven faces are skipped by the standard pullback; the
            // driven-boundary pass prescribes their ghost state from the DAG afterward.
            BoundaryType::Driven(_) => BcType::Skip,
            // neumann / robin faces are likewise skipped here; the gradient-boundary pass fills
            // them from the boundary-adjacent interior cell + the registry coefficients.
            BoundaryType::Neumann(_) | BoundaryType::Robin(_) => BcType::Skip,
        };
        [conv(boundaries.lo(ax)), conv(boundaries.hi(ax))]
    })
}

/// the same table for a lone advected scalar carried alongside the prim state (the dye
/// concentration), which has no entry in the per-variable gradient registry and no slot in the
/// driven DAG's prim writes.
///
/// gradient faces become outflow rather than skip: a prescribed normal derivative applies per
/// primitive variable, and a scalar with no prescription has `dchi/dn = 0`, which is exactly the
/// nearest-interior copy the outflow map performs. leaving them skipped instead would hand the
/// sweep an unwritten ghost band.
///
/// driven faces stay skipped: the dye of injected fluid is independent of the interior state, so it
/// is prescribed by the driven pass from an explicit `chi` slot. coarse-fine faces stay skipped for
/// the usual reason -- the refinement prolongation owns them.
pub fn to_bc_array_scalar<const D: usize>(boundaries: &Boundaries<D>) -> [[BcType; 2]; D] {
    std::array::from_fn(|ax| {
        let conv = |bt: BoundaryType| match bt {
            BoundaryType::Periodic => BcType::Periodic,
            BoundaryType::Outflow => BcType::Outflow,
            BoundaryType::Reflect | BoundaryType::Axis => BcType::Reflect,
            BoundaryType::CoarseFine => BcType::Skip,
            BoundaryType::Driven(_) => BcType::Skip,
            BoundaryType::Neumann(_) | BoundaryType::Robin(_) => BcType::Outflow,
        };
        [conv(boundaries.lo(ax)), conv(boundaries.hi(ax))]
    })
}

/// the azimuth axis of a chart whose azimuth is gridded, with the interior window a polar
/// half-turn rotates within: spherical (r, theta, phi) and cylindrical (R, phi, z) in three
/// dimensions. every other layout carries its azimuth out of plane or has no axis at all.
pub fn polar_turn_for<const D: usize>(
    coords: symbi_geometry::Geometry,
    interior: &Domain<D>,
) -> Option<PolarTurn> {
    use symbi_geometry::Geometry;
    let axis = match (coords, D) {
        (Geometry::Spherical, 3) => 2,
        (Geometry::Cylindrical, 3) => 1,
        _ => return None,
    };
    Some(PolarTurn {
        axis,
        lo: interior.spaces[axis].lo,
        n: interior.spaces[axis].hi - interior.spaces[axis].lo,
    })
}

/// the gridded azimuth a polar axis face rotates along: its axis index and the interior
/// window [lo, lo + n) the rotation wraps within. `n` is even, so a shift of n/2 cells is
/// exactly a half-turn.
#[derive(Clone, Copy, Debug)]
pub struct PolarTurn {
    pub axis: usize,
    pub lo: isize,
    pub n: isize,
}

/// the sign an in-plane vector component `ax` picks up under the region's lattice map: its own
/// axis's reflect sign, and, when it is the gridded azimuth a polar face rotates along, the
/// azimuthal parity of the state's basis as well (odd for orthonormal components, even for
/// contravariant ones), exactly as the out-of-plane azimuth of a 2.5D layout.
pub fn axis_vel_sign<const D: usize>(
    p: &GhostMapParams<D>,
    ax: usize,
    basis: symbi_geometry::ComponentBasis,
) -> f64 {
    let azimuthal = if p.turn[ax] != 0 {
        match basis {
            symbi_geometry::ComponentBasis::Orthonormal => -1.0,
            symbi_geometry::ComponentBasis::Contravariant => 1.0,
        }
    } else {
        1.0
    };
    p.vel_sign[ax] * azimuthal
}

/// the sign every out-of-plane vector component picks up across the region's faces, by the
/// basis the state is stored in. an orthonormal (physical) azimuthal component rides the unit
/// vector phi-hat = d/dphi / (r sin theta), whose normalization changes sign with sin theta
/// through the pole, so it is odd there and the axis face binds -1. a contravariant coordinate
/// component rides d/dphi itself, which continues unchanged through the pole (the ghost cell's
/// coordinate basis vector equals the image point's), so it is even and the axis face binds +1.
/// the basis comes from `Spacetime::component_basis`, the same predicate the seeding and the
/// metric-aware recovery follow.
pub fn axis_oop_sign<const D: usize>(
    p: &GhostMapParams<D>,
    ax: usize,
    basis: symbi_geometry::ComponentBasis,
) -> f64 {
    match basis {
        symbi_geometry::ComponentBasis::Orthonormal => p.oop_sign[ax],
        symbi_geometry::ComponentBasis::Contravariant => 1.0,
    }
}

// =============================================================================
// GhostFillDriver: region analysis + per-axis map parameters
// =============================================================================

/// per-axis ghost-remap parameters consumed by a ghost-fill kernel.
/// the encoding matches kernels_shared::iso_ghost_fill_2d's convention:
///
///   map_type: 0 = skip, 1 = periodic, 2 = reflect, 3 = outflow/clamp
///   start, len:   periodic wrap window ([start, start+len))
///   pivot:        reflect pivot (2*face - 1)
///   clamp_val:    outflow clamp target (face boundary)
///   vel_sign:     -1.0 for axes that reflect (flips normal-vector sign)
///   oop_sign:     -1.0 for axes that are a coordinate axis (flips every
///                 out-of-plane vector component; the kernel multiplies the
///                 per-axis signs, so a corner of two axis faces copies)
///   turn, turn_lo: on a gridded azimuth, crossing a polar axis face is a half-turn: the
///                 source is rotated by `turn` cells (half the period) along the azimuth axis,
///                 wrapping within [turn_lo, turn_lo + 2 turn). zero everywhere else.
///
/// when `map_type[ax] == 0` the axis is passthrough; the kernel must
/// interpret that as "leave coord[ax] alone."
#[derive(Clone, Debug)]
pub struct GhostMapParams<const D: usize> {
    pub map_type: [f64; D],
    pub start: [f64; D],
    pub len: [f64; D],
    pub pivot: [f64; D],
    pub clamp_val: [f64; D],
    pub vel_sign: [f64; D],
    pub oop_sign: [f64; D],
    pub turn: [i32; D],
    pub turn_lo: [i32; D],
    /// the lattice-map source-coord arg, one integer per axis,
    /// for the substrate `iso_ghost_fill` kernel: a signed periodic shift
    /// (`+len` on a low-side ghost, `-len` on a high side), a reflect `pivot2`, or
    /// an outflow edge cell. `src[ax] = c+arg | arg-c | arg` by `map_type`. derived
    /// from the i64 domain bounds (the side is known here), cast to the kernel's
    /// 32-bit index ABI — never through float.
    pub arg: [i32; D],
}

/// build the launch domain for the (`axis`, `side`) sweep step of `drive_sweep`.
///
/// - axis `axis`: a halo slab (allocated-side <-> interior-side boundary)
/// - axes in `done` (axes already swept): full allocated extent — the slab
///   covers x-halo positions whose `axis`-halo will be filled now (this is
///   what fills the corners/edges via successive sweeps)
/// - other axes (not yet swept): interior extent — their halos haven't been
///   filled yet, so must not be read from
fn sweep_domain<const D: usize>(
    allocated: &Domain<D>,
    interior: &Domain<D>,
    axis: usize,
    side: FaceSide,
    done: &[bool; D],
) -> Domain<D> {
    let spaces = std::array::from_fn(|ax| {
        if ax == axis {
            match side {
                FaceSide::Minus => Space {
                    name: allocated.spaces[ax].name,
                    lo: allocated.spaces[ax].lo,
                    hi: interior.spaces[ax].lo,
                },
                FaceSide::Plus => Space {
                    name: allocated.spaces[ax].name,
                    lo: interior.spaces[ax].hi,
                    hi: allocated.spaces[ax].hi,
                },
                FaceSide::None => unreachable!(),
            }
        } else if done[ax] {
            allocated.spaces[ax].clone()
        } else {
            interior.spaces[ax].clone()
        }
    });
    Domain::new(spaces)
}

/// driver: classify ghost regions, build per-axis map parameters, and
/// invoke the caller's per-region dispatch closure.
pub struct GhostFillDriver<'a, const D: usize> {
    allocated: &'a Domain<D>,
    interior: &'a Domain<D>,
    bc: [[BcType; 2]; D],
    polar: Option<PolarTurn>,
}

impl<'a, const D: usize> GhostFillDriver<'a, D> {
    pub fn new(allocated: &'a Domain<D>, interior: &'a Domain<D>, bc: [[BcType; 2]; D]) -> Self {
        GhostFillDriver {
            allocated,
            interior,
            bc,
            polar: None,
        }
    }

    /// the gridded azimuth an axis face rotates along. with it set, an `Axis` face on any other
    /// axis maps its source through the mirror and a half-turn of the azimuth; without it an
    /// `Axis` face is the plain mirror (the azimuth is out of plane). the window comes from
    /// the cell interior even for a face-anchored field, so the closing periodic face rotates
    /// onto its physical partner.
    pub fn with_polar_turn(mut self, polar: Option<PolarTurn>) -> Self {
        if let Some(t) = polar {
            assert!(t.n % 2 == 0, "a polar half-turn needs an even azimuth cell count, got {}", t.n);
        }
        self.polar = polar;
        self
    }

    /// for each ghost region contributing a non-skip fill, compute the
    /// per-axis map parameters and invoke `dispatch(region, params)`.
    ///
    /// 26-box scheme: classifies `allocated.difference(interior)`
    /// into up to `3^D - 1` axis-aligned regions (faces + edges + corners)
    /// and dispatches once per non-skip box. correct, but pays per-launch
    /// overhead x 26 per step and launches with shapes the dispatcher's
    /// block-picker handles poorly (tiny corners, axis-thin edges). prefer
    /// `drive_sweep` for hot paths — same semantics, 6 launches max.
    pub fn drive<F>(&self, mut dispatch: F)
    where
        F: FnMut(&GhostRegion<D>, &GhostMapParams<D>),
    {
        let regions = analyze_ghost_regions(self.allocated, self.interior);

        for region in &regions {
            if self.all_skip(region) {
                continue;
            }
            let params = self.build_params(region);
            dispatch(region, &params);
        }
    }

    /// **axis-sequential sweep**: fill the halo in 3 axis passes (x2 sides per
    /// axis = up to `2*D` launches), each pass a single thick slab. produces
    /// the same result as `drive` for periodic / reflect / outflow / skip BCs,
    /// but with `2*D` launches in place of `3^D - 1` and each launch is a
    /// rectangular slab the block-picker handles well.
    ///
    /// **invariant**: after sweep `k`, every halo cell whose halo-axes are a
    /// subset of `{0..=k}` is filled. each sweep `k > 0` reads from cells
    /// already filled by earlier sweeps — that's why sweep `k`'s domain
    /// extends over the full allocated extent on axes `0..k` (so e.g., a
    /// y-sweep at an x-halo position reads the x-halo source that sweep 0
    /// just produced). without this expansion, xy-edges and xyz-corners
    /// would never be filled.
    ///
    /// dispatches the same callback signature as `drive` — the caller treats
    /// each sweep like a single-axis ghost region.
    pub fn drive_sweep<F>(&self, mut dispatch: F)
    where
        F: FnMut(&GhostRegion<D>, &GhostMapParams<D>),
    {
        let mut done = [false; D];
        for axis in 0..D {
            for side in [FaceSide::Minus, FaceSide::Plus] {
                let bc = match side {
                    FaceSide::Minus => self.bc[axis][0],
                    FaceSide::Plus => self.bc[axis][1],
                    FaceSide::None => unreachable!(),
                };
                if bc == BcType::Skip {
                    continue;
                }

                let domain = sweep_domain(self.allocated, self.interior, axis, side, &done);
                if domain.volume() == 0 {
                    continue;
                }

                // synthetic single-axis region: only `axis` is in-halo on `side`,
                // all other axes are passthrough. build_params reads `directions`
                // and assigns `map_type[ax]` only for non-None axes — others stay
                // at 0 (the kernel's passthrough semantics).
                let mut directions = [FaceSide::None; D];
                directions[axis] = side;
                let region = GhostRegion {
                    domain,
                    ghost_type: GhostType::Face,
                    directions,
                };
                let params = self.build_params(&region);
                dispatch(&region, &params);
            }
            done[axis] = true;
        }
    }

    fn all_skip(&self, region: &GhostRegion<D>) -> bool {
        (0..D).all(|ax| match region.directions[ax] {
            FaceSide::None => true,
            FaceSide::Minus => self.bc[ax][0] == BcType::Skip,
            FaceSide::Plus => self.bc[ax][1] == BcType::Skip,
        })
    }

    fn build_params(&self, region: &GhostRegion<D>) -> GhostMapParams<D> {
        let mut p = GhostMapParams {
            map_type: [0.0; D],
            start: [0.0; D],
            len: [0.0; D],
            pivot: [0.0; D],
            clamp_val: [0.0; D],
            vel_sign: [1.0; D],
            oop_sign: [1.0; D],
            turn: [0; D],
            turn_lo: [0; D],
            arg: [0; D],
        };

        for ax in 0..D {
            let side = region.directions[ax];
            if side == FaceSide::None {
                continue;
            }

            let bc_type = match side {
                FaceSide::Minus => self.bc[ax][0],
                FaceSide::Plus => self.bc[ax][1],
                FaceSide::None => unreachable!(),
            };

            let lo = self.interior.spaces[ax].lo;
            let hi = self.interior.spaces[ax].hi;

            match bc_type {
                BcType::Periodic => {
                    p.map_type[ax] = 1.0;
                    p.start[ax] = lo as f64;
                    p.len[ax] = (hi - lo) as f64;
                    // signed shift: a low-side ghost reads one period up, a
                    // high-side ghost one period down (the region is one-sided, so
                    // the shift is uniform — no modulo needed).
                    let period = (hi - lo) as i32;
                    p.arg[ax] = if side == FaceSide::Minus {
                        period
                    } else {
                        -period
                    };
                }
                BcType::Reflect | BcType::Axis => {
                    p.map_type[ax] = 2.0;
                    let face = if side == FaceSide::Minus { lo } else { hi };
                    p.pivot[ax] = (2 * face - 1) as f64;
                    p.vel_sign[ax] = -1.0;
                    if bc_type == BcType::Axis {
                        p.oop_sign[ax] = -1.0;
                        if let Some(t) = self.polar.filter(|t| t.axis != ax) {
                            p.turn[t.axis] = (t.n / 2) as i32;
                            p.turn_lo[t.axis] = t.lo as i32;
                        }
                    }
                    p.arg[ax] = (2 * face - 1) as i32;
                }
                BcType::Outflow => {
                    p.map_type[ax] = 3.0;
                    let edge = if side == FaceSide::Minus { lo } else { hi - 1 };
                    p.clamp_val[ax] = edge as f64;
                    p.arg[ax] = edge as i32;
                }
                BcType::Skip => {}
            }
        }

        p
    }
}

// =============================================================================
// CFL scaling
// =============================================================================

/// standard CFL timestep from a pre-reduced max wave speed and the minimum
/// cell width. kernels compute `s_max` themselves (map + reduction); this
/// helper applies the scaling `cfl * dx_min / s_max`.
///
/// this is the isotropic form. the anisotropic-correct form is `cfl_from_lambda`,
/// which a call site with a per-axis wave-speed map uses instead.
#[inline]
pub fn cfl_from_smax(s_max: f64, cfl_number: f64, dx_min: f64) -> f64 {
    cfl_number * dx_min / s_max
}

/// strict anisotropic CFL: kernels compute per-cell
/// `lambda = max_d ((|v_d| + cs) / dx_d)`, host reduces to `lambda_max`,
/// this helper returns `dt = cfl / lambda_max`. matches the per-axis
/// limiter (cfl * dx_d / s_max_d picked per cell + axis) without
/// collapsing to `dx_min` and losing anisotropy.
#[inline]
pub fn cfl_from_lambda(lambda_max: f64, cfl_number: f64) -> f64 {
    cfl_number / lambda_max
}

// =============================================================================
// tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_domain_extends_only_named_axis() {
        let interior = Domain::new([
            Space {
                name: "x",
                lo: 2,
                hi: 10,
            },
            Space {
                name: "y",
                lo: 2,
                hi: 10,
            },
        ]);
        let fx = interior.face_domain(0);
        assert_eq!(fx.spaces[0].hi, 11);
        assert_eq!(fx.spaces[1].hi, 10);
        let fy = interior.face_domain(1);
        assert_eq!(fy.spaces[0].hi, 10);
        assert_eq!(fy.spaces[1].hi, 11);
    }

    #[test]
    fn bc_array_converts_boundaries() {
        let bcs = Boundaries::<3>::per_axis([
            [BoundaryType::Periodic, BoundaryType::Periodic],
            [BoundaryType::Outflow, BoundaryType::Reflect],
            [BoundaryType::CoarseFine, BoundaryType::Periodic],
        ]);
        let arr = to_bc_array::<3>(&bcs);
        assert_eq!(arr[0], [BcType::Periodic, BcType::Periodic]);
        assert_eq!(arr[1], [BcType::Outflow, BcType::Reflect]);
        assert_eq!(arr[2], [BcType::Skip, BcType::Periodic]);
    }

    #[test]
    fn cfl_scaling() {
        assert_eq!(cfl_from_smax(2.0, 0.4, 0.1), 0.4 * 0.1 / 2.0);
    }

    #[test]
    fn ghost_driver_visits_non_skip_regions() {
        let alloc = Domain::new([
            Space {
                name: "x",
                lo: 0,
                hi: 12,
            },
            Space {
                name: "y",
                lo: 0,
                hi: 12,
            },
        ]);
        let interior = Domain::new([
            Space {
                name: "x",
                lo: 2,
                hi: 10,
            },
            Space {
                name: "y",
                lo: 2,
                hi: 10,
            },
        ]);
        let bc = [
            [BcType::Periodic, BcType::Periodic],
            [BcType::Reflect, BcType::Outflow],
        ];
        let mut visited = 0;
        GhostFillDriver::<2>::new(&alloc, &interior, bc).drive(|_region, _params| {
            visited += 1;
        });
        // 2D ghost layout: 4 faces + 4 corners = 8 regions. none are skip.
        assert_eq!(visited, 8);
    }

    #[test]
    fn ghost_driver_skips_all_skip_faces() {
        let alloc = Domain::new([
            Space {
                name: "x",
                lo: 0,
                hi: 12,
            },
            Space {
                name: "y",
                lo: 0,
                hi: 12,
            },
        ]);
        let interior = Domain::new([
            Space {
                name: "x",
                lo: 2,
                hi: 10,
            },
            Space {
                name: "y",
                lo: 2,
                hi: 10,
            },
        ]);
        // every contact is Skip -> every region should be filtered.
        let bc = [[BcType::Skip, BcType::Skip], [BcType::Skip, BcType::Skip]];
        let mut visited = 0;
        GhostFillDriver::<2>::new(&alloc, &interior, bc).drive(|_region, _params| {
            visited += 1;
        });
        assert_eq!(visited, 0);
    }

    #[test]
    fn ghost_driver_reflect_sets_vel_sign() {
        let alloc = Domain::new([
            Space {
                name: "x",
                lo: 0,
                hi: 12,
            },
            Space {
                name: "y",
                lo: 0,
                hi: 12,
            },
        ]);
        let interior = Domain::new([
            Space {
                name: "x",
                lo: 2,
                hi: 10,
            },
            Space {
                name: "y",
                lo: 2,
                hi: 10,
            },
        ]);
        let bc = [
            [BcType::Reflect, BcType::Outflow],
            [BcType::Periodic, BcType::Periodic],
        ];
        let mut saw_lo_x_reflect = false;
        GhostFillDriver::<2>::new(&alloc, &interior, bc).drive(|region, params| {
            if region.directions[0] == FaceSide::Minus {
                // axis 0 reflects on the Minus side -> vel_sign[0] should be -1.
                assert_eq!(params.vel_sign[0], -1.0);
                saw_lo_x_reflect = true;
            }
        });
        assert!(
            saw_lo_x_reflect,
            "expected at least one lo-x reflect region"
        );
    }

    #[test]
    fn ghost_driver_axis_flips_the_normal_and_the_out_of_plane_signs() {
        let alloc = Domain::new([
            Space {
                name: "r",
                lo: 0,
                hi: 12,
            },
            Space {
                name: "theta",
                lo: 0,
                hi: 12,
            },
        ]);
        let interior = Domain::new([
            Space {
                name: "r",
                lo: 2,
                hi: 10,
            },
            Space {
                name: "theta",
                lo: 2,
                hi: 10,
            },
        ]);
        let bc = [
            [BcType::Outflow, BcType::Outflow],
            [BcType::Axis, BcType::Reflect],
        ];
        let mut saw = [false; 2];
        GhostFillDriver::<2>::new(&alloc, &interior, bc).drive(|region, params| {
            match region.directions[1] {
                FaceSide::Minus => {
                    assert_eq!(params.map_type[1], 2.0, "the axis shares the mirror map");
                    assert_eq!(params.vel_sign[1], -1.0);
                    assert_eq!(params.oop_sign[1], -1.0);
                    saw[0] = true;
                }
                FaceSide::Plus => {
                    assert_eq!(params.vel_sign[1], -1.0);
                    assert_eq!(params.oop_sign[1], 1.0, "a reflecting wall keeps the azimuthal sign");
                    saw[1] = true;
                }
                FaceSide::None => {}
            }
            assert_eq!(params.oop_sign[0], 1.0, "an outflow face carries no azimuthal flip");
        });
        assert!(saw[0] && saw[1], "expected both theta faces to dispatch");
    }

    #[test]
    fn ghost_driver_polar_turn_rotates_the_azimuth_by_half_its_period() {
        let sp = |name, lo, hi| Space { name, lo, hi };
        let alloc = Domain::new([sp("r", 0, 12), sp("theta", 0, 12), sp("phi", 0, 12)]);
        let interior = Domain::new([sp("r", 2, 10), sp("theta", 2, 10), sp("phi", 2, 10)]);
        let bc = [
            [BcType::Outflow, BcType::Outflow],
            [BcType::Axis, BcType::Axis],
            [BcType::Periodic, BcType::Periodic],
        ];
        let turn = polar_turn_for::<3>(symbi_geometry::Geometry::Spherical, &interior);
        let mut saw_theta = 0;
        GhostFillDriver::<3>::new(&alloc, &interior, bc)
            .with_polar_turn(turn)
            .drive_sweep(|region, params| {
                if region.directions[1] != FaceSide::None {
                    assert_eq!(params.map_type[1], 2.0, "the pole is the mirror on theta");
                    assert_eq!(params.turn[2], 4, "half of the 8 azimuth cells");
                    assert_eq!(params.turn_lo[2], 2);
                    assert_eq!(params.turn[1], 0, "the mirrored axis itself is unrotated");
                    saw_theta += 1;
                } else {
                    assert_eq!(params.turn, [0; 3], "only a polar face rotates the source");
                }
            });
        assert_eq!(saw_theta, 2);
        let plane = Domain::new([sp("r", 2, 10), sp("theta", 2, 10)]);
        assert!(
            polar_turn_for::<2>(symbi_geometry::Geometry::Spherical, &plane).is_none(),
            "a 2.5D layout keeps its azimuth out of plane"
        );
    }

    // ----- drive_sweep tests -----

    /// in 2D with all-periodic BCs, drive_sweep should dispatch exactly 4 times
    /// (2 axes x 2 sides) — replacing the 8-region `drive` (4 faces + 4 corners).
    #[test]
    fn drive_sweep_2d_dispatches_four_times() {
        let alloc = Domain::new([
            Space {
                name: "x",
                lo: 0,
                hi: 12,
            },
            Space {
                name: "y",
                lo: 0,
                hi: 12,
            },
        ]);
        let interior = Domain::new([
            Space {
                name: "x",
                lo: 2,
                hi: 10,
            },
            Space {
                name: "y",
                lo: 2,
                hi: 10,
            },
        ]);
        let bc = [
            [BcType::Periodic, BcType::Periodic],
            [BcType::Periodic, BcType::Periodic],
        ];
        let mut dispatches = 0;
        GhostFillDriver::<2>::new(&alloc, &interior, bc).drive_sweep(|_region, _params| {
            dispatches += 1;
        });
        assert_eq!(
            dispatches, 4,
            "drive_sweep should dispatch 2*D = 4 times in 2D"
        );
    }

    /// in 3D, drive_sweep dispatches 2*D = 6 times.
    #[test]
    fn drive_sweep_3d_dispatches_six_times() {
        let alloc = Domain::new([
            Space {
                name: "x",
                lo: 0,
                hi: 12,
            },
            Space {
                name: "y",
                lo: 0,
                hi: 12,
            },
            Space {
                name: "z",
                lo: 0,
                hi: 5,
            },
        ]);
        let interior = Domain::new([
            Space {
                name: "x",
                lo: 2,
                hi: 10,
            },
            Space {
                name: "y",
                lo: 2,
                hi: 10,
            },
            Space {
                name: "z",
                lo: 2,
                hi: 3,
            },
        ]);
        let bc = [
            [BcType::Periodic, BcType::Periodic],
            [BcType::Periodic, BcType::Periodic],
            [BcType::Periodic, BcType::Periodic],
        ];
        let mut dispatches = 0;
        GhostFillDriver::<3>::new(&alloc, &interior, bc).drive_sweep(|_region, _params| {
            dispatches += 1;
        });
        assert_eq!(
            dispatches, 6,
            "drive_sweep should dispatch 2*D = 6 times in 3D"
        );
    }

    /// drive_sweep should skip dispatches whose side has BcType::Skip — exactly
    /// the way `drive` would. all-skip -> zero dispatches.
    #[test]
    fn drive_sweep_skips_skip_sides() {
        let alloc = Domain::new([
            Space {
                name: "x",
                lo: 0,
                hi: 12,
            },
            Space {
                name: "y",
                lo: 0,
                hi: 12,
            },
        ]);
        let interior = Domain::new([
            Space {
                name: "x",
                lo: 2,
                hi: 10,
            },
            Space {
                name: "y",
                lo: 2,
                hi: 10,
            },
        ]);
        // only one side non-skip: lo-x periodic
        let bc = [
            [BcType::Periodic, BcType::Skip],
            [BcType::Skip, BcType::Skip],
        ];
        let mut dispatches = 0;
        GhostFillDriver::<2>::new(&alloc, &interior, bc).drive_sweep(|_region, _params| {
            dispatches += 1;
        });
        assert_eq!(dispatches, 1, "only the non-skip side should dispatch");
    }

    /// the coverage invariant: the union of all sweep domains covers the same
    /// halo cells as `allocated.difference(interior)`. proves the sweep scheme
    /// fills the same set of cells the 26-box scheme does — no cell missed.
    #[test]
    fn drive_sweep_covers_full_halo() {
        let alloc = Domain::new([
            Space {
                name: "x",
                lo: 0,
                hi: 8,
            },
            Space {
                name: "y",
                lo: 0,
                hi: 8,
            },
            Space {
                name: "z",
                lo: 0,
                hi: 5,
            },
        ]);
        let interior = Domain::new([
            Space {
                name: "x",
                lo: 2,
                hi: 6,
            },
            Space {
                name: "y",
                lo: 2,
                hi: 6,
            },
            Space {
                name: "z",
                lo: 2,
                hi: 3,
            },
        ]);
        let bc = [
            [BcType::Periodic, BcType::Periodic],
            [BcType::Reflect, BcType::Outflow],
            [BcType::Periodic, BcType::Periodic],
        ];

        // expected total halo cells = allocated.volume() - interior.volume()
        let expected_halo_count = alloc.volume() - interior.volume();

        // build the union of sweep domains. sweeps are disjoint by construction
        // because each sweep `k > 0` only writes the halo on axis k (at axes
        // 0..k allocated, axes >k interior), and a cell can be in only one
        // sweep — the first axis where it's in the halo.
        let mut sweep_cell_count = 0usize;
        GhostFillDriver::<3>::new(&alloc, &interior, bc).drive_sweep(|region, _params| {
            sweep_cell_count += region.domain.volume();
        });

        assert_eq!(
            sweep_cell_count, expected_halo_count,
            "sweep coverage must equal allocated.volume() - interior.volume()",
        );
    }
}
