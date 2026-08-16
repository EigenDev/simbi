// =============================================================================
// geometry.rs
//
// in-kernel geometry: index->physical metric, finite-volume factors, and the divergence operators.
// =============================================================================

use super::*;
use symbi_algebra::Tensor;
use symbi_geometry::{
    Geometry, KerrKS, KerrKSCartesian, KerrKSCylindrical, Metric, SchwarzschildKS,
    SchwarzschildKSCartesian, SchwarzschildKSCylindrical, volume_weighted_centroid,
};

/// one cartesian-uniform finite-volume divergence sum over the gridded axes:
/// `sum_i (F_i[coord+e_i] - F_i[coord]) / dx_i`. `base` names the per-direction flux field
/// (`{base}_{i}`, runtime `{base}[{i}]`) — `mass_flux` / `mom_flux_{k}` / `nrg_flux`. the lo
/// read is the direct cell read, the hi a `+e_i` field_shifted (LoadAt); dt is the caller's.
fn gv_divergence_cartesian(base: &str, ndim: u8, spacing: &[Spacing]) -> Gv {
    let mut acc: Option<Gv> = None;
    for ii in 0..ndim {
        let key = format!("{base}_{ii}");
        let rt = format!("{base}[{ii}]");
        let f_lo = Gv::field_shifted(&key, &rt, ndim, ii, 0); // == Gv::field (offset 0)
        let f_hi = Gv::field_shifted(&key, &rt, ndim, ii, 1);
        // the width is the cell's own: on a graded axis every cell carries its own face
        // separation, and differencing over a single `dx` would misplace the divergence in every
        // cell whose width differs from it. `gv_axis_width` reduces to the `dx_i` scalar on an
        // unmapped axis.
        let dx = gv_axis_width(ii as usize, spacing[ii as usize]);
        let term = (f_hi - f_lo) / dx;
        acc = Some(match acc {
            None => term,
            Some(a) => a + term,
        });
    }
    acc.expect("godunov divergence needs ndim >= 1")
}

/// the analytic area-weighted curvilinear divergence: `(1/V) sum_i (F_i[+e_i]*A_hi_i -
/// F_i*A_lo_i)` — each face flux weighted by its face area ahead of the telescope, the cell sum
/// scaled by `1/V`. the gv mirror of `finite_volume::divergence_sum_weighted`; `geo` carries
/// the in-kernel per-cell areas + inverse volume from `cell_geometry_gv`.
fn gv_divergence_weighted(base: &str, ndim: u8, geo: &CellGeometryGv) -> Gv {
    let mut acc: Option<Gv> = None;
    for ii in 0..ndim {
        let key = format!("{base}_{ii}");
        let rt = format!("{base}[{ii}]");
        let f_lo = Gv::field_shifted(&key, &rt, ndim, ii, 0);
        let f_hi = Gv::field_shifted(&key, &rt, ndim, ii, 1);
        let d = ii as usize;
        let diff = f_hi * geo.area_hi[d] - f_lo * geo.area_lo[d];
        acc = Some(match acc {
            None => diff,
            Some(a) => a + diff,
        });
    }
    acc.expect("godunov divergence needs ndim >= 1") * geo.inv_volume
}

/// the plain coordinate divergence `sum_i (F_i[+e_i] - F_i) / width_i` — the raw face
/// difference over the cell width on every chart. this is the divergence of a fully densitized
/// conservation law `d_t U + d_j F^j = S`, where the measure `sqrt(-g)` rides inside both `U`
/// and `F` and already carries every geometric factor. the widths come from the per-cell index
/// map, so a log-spaced axis is exact.
pub(crate) fn gv_divergence_coord(base: &str, ndim: u8, spacing: &[Spacing]) -> Gv {
    let (_, _, width) = gv_faces(spacing, ndim as usize);
    let mut acc: Option<Gv> = None;
    for ii in 0..ndim {
        let key = format!("{base}_{ii}");
        let rt = format!("{base}[{ii}]");
        let f_lo = Gv::field_shifted(&key, &rt, ndim, ii, 0);
        let f_hi = Gv::field_shifted(&key, &rt, ndim, ii, 1);
        let term = (f_hi - f_lo) / width[ii as usize];
        acc = Some(match acc {
            None => term,
            Some(a) => a + term,
        });
    }
    acc.expect("godunov divergence needs ndim >= 1")
}

/// the per-direction inverse divergence operator for `base`: cartesian-uniform `(F_hi -
/// F_lo)/dx_i`, else the area-weighted `(1/V)(F_hi*A_hi - F_lo*A_lo)` from `geo`.
pub(crate) fn gv_divergence(
    base: &str,
    ndim: u8,
    geo: &Option<CellGeometryGv>,
    spacing: &[Spacing],
) -> Gv {
    match geo {
        None => gv_divergence_cartesian(base, ndim, spacing),
        Some(g) => gv_divergence_weighted(base, ndim, g),
    }
}

/// the GR lapse weight `alpha(x)` for the spatial-RHS densitization (valencia 3+1). the conserved
/// update `d_t(sqrt(gamma) U) + d_i(sqrt(-g) F) = sqrt(-g) S` reduces, on a static diagonal
/// background, to weighting the flux divergence + the geometric momentum source by the lapse
/// (`sqrt(-g) = alpha sqrt(gamma)`; in schwarzschild coordinates `sqrt(-g) = sqrt(gamma_flat)`,
/// which leaves the face areas flat and folds `1/sqrt(gamma) = alpha/sqrt(gamma_flat)` into a single
/// `alpha` on the RHS). a flat spacetime has `alpha = 1` -> `None`,
/// so the RHS is left as written and bit-identical on a flat metric. a GR metric (schwarzschild)
/// returns `Some(alpha)` dispatched `Coords -> concrete Metric -> metric.lapse(centroid)` as a
/// traced Gv expression in the cell coordinate (the coordinate-dispatch pattern).
pub(crate) fn gv_lapse_weight(
    coords: Coords,
    spacetime: Spacetime,
    coord_centroid: &[Gv],
) -> Option<Gv> {
    match (spacetime, coords) {
        // flat (minkowski) lapse alpha = 1: the densitization is the identity, so the weight is
        // elided from the graph (the unity multiply stays out) -> bit-identical.
        (Spacetime::Minkowski, _) => None,
        // cartesian kerr-schild is gridded on cartesian axes: alpha = 1/sqrt(1 + 2M/|x|) is
        // evaluated at the full cartesian position (the metric computes r = |x| internally). the
        // spherical shortcut r = coord_centroid[0] would use the x-coordinate as the radius —
        // wrong, and asymmetric under x <-> y.
        (Spacetime::SchwarzschildKS, Coords::Cartesian) => {
            let x = Tensor::<Gv, 3>::new(std::array::from_fn(|c| {
                coord_centroid.get(c).copied().unwrap_or(Gv::ZERO)
            }));
            Some(
                SchwarzschildKSCartesian {
                    mass: Gv::scalar("schwarzschild_mass"),
                }
                .lapse(x),
            )
        }
        // cylindrical kerr-schild: alpha = 1/sqrt(1 + 2M/r), r = sqrt(R^2 + z^2) the spherical radius
        // at the full (R, phi, z) position — the metric reads both slots 0 and 2 (the cylindrical
        // R and z). the same radial-shortcut trap the cartesian arm names above.
        (Spacetime::SchwarzschildKS, Coords::Cylindrical) => {
            let x = Tensor::<Gv, 3>::new(std::array::from_fn(|c| {
                coord_centroid.get(c).copied().unwrap_or(Gv::ZERO)
            }));
            Some(
                SchwarzschildKSCylindrical {
                    mass: Gv::scalar("schwarzschild_mass"),
                }
                .lapse(x),
            )
        }
        // cartesian spinning kerr: alpha = 1/sqrt(1 + 2H |l|^2) at the full cartesian position
        // (the metric solves the oblate-spheroidal radius internally); the radius on this chart
        // is a function of all three cartesian slots.
        (Spacetime::KerrKS, Coords::Cartesian) => {
            let x = Tensor::<Gv, 3>::new(std::array::from_fn(|c| {
                coord_centroid.get(c).copied().unwrap_or(Gv::ZERO)
            }));
            let mass = Gv::scalar("schwarzschild_mass");
            let spin = Gv::scalar("kerr_spin");
            Some(KerrKSCartesian { mass, spin }.lapse(x))
        }
        (Spacetime::KerrKS, Coords::Cylindrical) => {
            let x = Tensor::<Gv, 3>::new(std::array::from_fn(|c| {
                coord_centroid.get(c).copied().unwrap_or(Gv::ZERO)
            }));
            let mass = Gv::scalar("schwarzschild_mass");
            let spin = Gv::scalar("kerr_spin");
            Some(KerrKSCylindrical { mass, spin }.lapse(x))
        }
        // spherical charts: r = the radial centroid (coordinate slot 0). the schwarzschild /
        // kerr-schild lapses are radial-only; the spinning-kerr lapse also reads the polar centroid
        // (slot 1) through Sigma = r^2 + a^2 cos^2(theta).
        _ => Some(gv_metric_lapse_at(
            spacetime,
            coord_centroid[0],
            coord_centroid.get(1).copied(),
        )),
    }
}

/// the per-axis arithmetic cell midpoints. a cell's stored value is its cell average, and the
/// point where a smooth field's average equals its point value to second order depends on the
/// measure the average is taken against: the area-weighted law integrates against the chart's
/// volume element and reads the volume-weighted centroid, while the densitized law integrates
/// against the plain coordinate volume — its measure rides inside the conserved variable — and
/// reads the midpoint. on a spherical radial axis the two differ by `dr^2/(6r)`, which offsets a
/// pointwise connection source from the flux difference it must balance at exactly the order of
/// the truncation error.
pub(crate) fn gv_cell_midpoints(spacing: &[Spacing], ndim: usize) -> Vec<Gv> {
    let (lo, hi, _) = gv_faces(spacing, ndim);
    let half = Gv::from_f64(0.5);
    (0..ndim).map(|d| (lo[d] + hi[d]) * half).collect()
}

/// the full-chart spatial measure `sqrt(det gamma)` at a 3-slot coordinate position, dispatched
/// `(Spacetime, Coords) -> concrete Metric -> Metric::volume_factor`. always evaluated at the
/// metric's full spatial dimension, so a reduced grid still carries the suppressed directions: on a
/// 1D radial spherical grid the measure is `r^2 sin(theta) sqrt(gamma_rr)`, of which the 1x1 radial
/// block alone would keep only `sqrt(gamma_rr)`. paired with the lapse it gives the four-volume
/// densitization `sqrt(-g) = alpha sqrt(det gamma)`. curved spacetimes alone reach this; the
/// flat measure is the coordinate volume element.
pub(crate) fn gv_metric_volume_factor_at(
    spacetime: Spacetime,
    coords: Coords,
    x: Tensor<Gv, 3>,
) -> Gv {
    let mass = Gv::scalar("schwarzschild_mass");
    match (spacetime, coords) {
        (Spacetime::SchwarzschildKS, Coords::Cartesian) => {
            SchwarzschildKSCartesian { mass }.volume_factor(x)
        }
        (Spacetime::SchwarzschildKS, Coords::Cylindrical) => {
            SchwarzschildKSCylindrical { mass }.volume_factor(x)
        }
        (Spacetime::SchwarzschildKS, _) => SchwarzschildKS { mass }.volume_factor(x),
        (Spacetime::KerrKS, Coords::Cartesian) => KerrKSCartesian {
            mass,
            spin: Gv::scalar("kerr_spin"),
        }
        .volume_factor(x),
        (Spacetime::KerrKS, Coords::Cylindrical) => KerrKSCylindrical {
            mass,
            spin: Gv::scalar("kerr_spin"),
        }
        .volume_factor(x),
        (Spacetime::KerrKS, _) => KerrKS {
            mass,
            spin: Gv::scalar("kerr_spin"),
        }
        .volume_factor(x),
        (Spacetime::Minkowski, _) => {
            unreachable!("the flat measure is the coordinate volume element")
        }
    }
}

/// the analytic lapse alpha(r) as a traced Gv, dispatched `Spacetime -> concrete Metric ->
/// Metric::lapse` — the single codegen seam for the GR lapse. every consumer (the densitization cell
/// weight `gv_lapse_weight`, the CFL/shift kernels) reads the lapse here, so a new analytic background
/// is a new `Metric` impl + one arm. `M` rides as the
/// host-filled scalar `schwarzschild_mass` so the kernel stays M-agnostic. curved spacetimes alone
/// reach this (the flat weight is elided by the caller); a flat call is a bug.
/// `theta` is required by the spinning-kerr arm (Sigma depends on the polar angle); the
/// radial-only backgrounds read `r` alone, and a theta-less caller requesting kerr is a
/// bake-time bug.
pub(crate) fn gv_metric_lapse_at(spacetime: Spacetime, r: Gv, theta: Option<Gv>) -> Gv {
    let mass = Gv::scalar("schwarzschild_mass");
    match spacetime {
        // alpha = sqrt(1 - 2M/r) (schwarzschild coords) / alpha = 1/sqrt(1 + 2M/r) (kerr-schild),
        // each from its `Metric` impl (the single source of the lapse expression).
        Spacetime::SchwarzschildKS => SchwarzschildKS { mass }.lapse(Tensor::new([r])),
        Spacetime::KerrKS => {
            let spin = Gv::scalar("kerr_spin");
            let th = theta.expect("the kerr lapse requires the polar coordinate");
            KerrKS { mass, spin }.lapse(Tensor::new([r, th, Gv::ZERO]))
        }
        Spacetime::Minkowski => unreachable!("flat lapse is elided by the densitization caller"),
    }
}

/// the analytic lapse square alpha^2(r) from `Metric::lapse_sq` — the CFL radial coordinate-speed
/// factor alpha sqrt(gamma^{rr}) = alpha^2 for the det-g-flat family (schwarzschild alpha^2 = f;
/// kerr-schild alpha^2 = 1/(1 + 2M/r)). the closed form, so the genericized wave-speed map
/// reproduces the pre-refactor `f` node bitwise (squaring `lapse()` rounds differently). curved
/// spacetimes alone reach this.
pub(crate) fn gv_metric_lapse_sq_at(spacetime: Spacetime, r: Gv, theta: Option<Gv>) -> Gv {
    let mass = Gv::scalar("schwarzschild_mass");
    match spacetime {
        Spacetime::SchwarzschildKS => SchwarzschildKS { mass }.lapse_sq(Tensor::new([r])),
        Spacetime::KerrKS => {
            let spin = Gv::scalar("kerr_spin");
            let th = theta.expect("the kerr lapse-square requires the polar coordinate");
            KerrKS { mass, spin }.lapse_sq(Tensor::new([r, th, Gv::ZERO]))
        }
        Spacetime::Minkowski => unreachable!("flat lapse-square is elided by the CFL caller"),
    }
}

/// the analytic radial shift beta^r(r) from `Metric::shift` — nonzero on a shifted background
/// (kerr-schild beta^r = 2M/(r + 2M)); the static diagonal cases (minkowski, schwarzschild) have
/// beta = 0 -> None, so the caller elides the shift term and the arithmetic stays bit-identical
/// (the `- 0` stays out of the graph).
pub(crate) fn gv_metric_shift_r_at(spacetime: Spacetime, r: Gv, theta: Option<Gv>) -> Option<Gv> {
    match spacetime {
        Spacetime::Minkowski => None,
        Spacetime::SchwarzschildKS => {
            let mass = Gv::scalar("schwarzschild_mass");
            Some(SchwarzschildKS { mass }.shift(Tensor::new([r]))[0])
        }
        Spacetime::KerrKS => {
            let mass = Gv::scalar("schwarzschild_mass");
            let spin = Gv::scalar("kerr_spin");
            let th = theta.expect("the kerr shift requires the polar coordinate");
            Some(KerrKS { mass, spin }.shift(Tensor::new([r, th, Gv::ZERO]))[0])
        }
    }
}

/// the single statement of which chart expression realizes which curved background.
/// selects the kerr-schild metric struct for a `(spacetime, chart)` pair -- spherical
/// (the bare names), cartesian, or cylindrical, each computing its own radius from the
/// chart's coordinates -- creates its runtime scalars (`schwarzschild_mass`, and
/// `kerr_spin` on the spinning arms) in trace order, and hands the struct to `$body`.
/// the six structs are distinct types, so the selection is a macro; every gr
/// kernel builder (flux fan, c2p, wave speeds, ct emf, cell geometry) expands this
/// macro with a small `Metric<Gv, 3>`-generic adapter as the body, and adding a chart
/// is one new arm here, shared by every builder. `$kernel` names the caller in the
/// flat-spacetime unreachable message.
macro_rules! with_ks_metric {
    ($spacetime:expr, $coords:expr, $kernel:literal, |$m:ident| $body:expr) => {{
        let mass = Gv::scalar("schwarzschild_mass");
        match ($spacetime, $coords) {
            (Spacetime::SchwarzschildKS, $crate::coords::Coords::Cartesian) => {
                let $m = ::symbi_geometry::SchwarzschildKSCartesian { mass };
                $body
            }
            (Spacetime::SchwarzschildKS, $crate::coords::Coords::Cylindrical) => {
                let $m = ::symbi_geometry::SchwarzschildKSCylindrical { mass };
                $body
            }
            (Spacetime::SchwarzschildKS, _) => {
                let $m = ::symbi_geometry::SchwarzschildKS { mass };
                $body
            }
            (Spacetime::KerrKS, $crate::coords::Coords::Cartesian) => {
                let $m = ::symbi_geometry::KerrKSCartesian {
                    mass,
                    spin: Gv::scalar("kerr_spin"),
                };
                $body
            }
            (Spacetime::KerrKS, $crate::coords::Coords::Cylindrical) => {
                let $m = ::symbi_geometry::KerrKSCylindrical {
                    mass,
                    spin: Gv::scalar("kerr_spin"),
                };
                $body
            }
            (Spacetime::KerrKS, _) => {
                let $m = ::symbi_geometry::KerrKS {
                    mass,
                    spin: Gv::scalar("kerr_spin"),
                };
                $body
            }
            (Spacetime::Minkowski, _) => {
                unreachable!(concat!($kernel, " is baked only for a curved spacetime"))
            }
        }
    }};
}
pub(crate) use with_ks_metric;

/// `true` iff the flat unweighted `(F_hi-F_lo)/dx` divergence applies — a cartesian chart on
/// uniform spacing, where the grid scalars carry the whole geometry.
pub(crate) fn is_cartesian_uniform(coords: Coords, spacing: &[Spacing]) -> bool {
    coords == Coords::Cartesian && spacing.iter().all(|&s| s == Spacing::Uniform)
}

// =============================================================================
// in-kernel geometry — the substrate metric expressed in Gv. `Gv::coord` is the
// index->physical bridge; the coordinate-system formulas are a build-time `match` on
// `Coords` (the kernel is generated per geometry, so the branch is resolved at trace
// time). this is the foundation every curvilinear operator (CFL
// widths, godunov divergence, geometric sources, CT curl) traces through.
// =============================================================================

/// the face at `coord + offset` along axis `ax` as a physical position (offset 0 = lo face,
/// 1 = hi face). `x_lo_{ax}` + `dx_{ax}` are the grid scalars (dx = width for Uniform, the
/// log-slope for Log). the integer coord promotes to f64 against the scalars at lowering.
pub(crate) fn gv_axis_face_at(ax: usize, spacing: Spacing, offset: i64) -> Gv {
    let coord = Gv::coord(ax as u8);
    let i = if offset == 0 {
        coord
    } else {
        coord + Gv::from_f64(offset as f64)
    };
    gv_axis_face_at_index(ax, spacing, i)
}

/// physical coordinate width of the current cell on one logical grid axis.
/// the runtime map selector makes this local on uniform, logarithmic, and
/// geometrically graded meshes.
pub(crate) fn gv_axis_width(ax: usize, spacing: Spacing) -> Gv {
    let map_kind = Gv::scalar(&format!("map_kind_{ax}"));
    Gv::cond(
        map_kind.cmp_gt(Gv::from_f64(0.5)),
        || gv_axis_face_at(ax, spacing, 1) - gv_axis_face_at(ax, spacing, 0),
        || Gv::scalar(&format!("dx_{ax}")),
    )
}

/// the lower face position of the cell at an arbitrary integer index expression `i` along
/// grid axis `ax` — the index-general form of [`gv_axis_face_at`] (which passes the thread
/// coord). the lattice-map ghost fill evaluates metric coefficients at the source cell,
/// whose index is a runtime map expression.
pub(crate) fn gv_axis_face_at_index(ax: usize, _spacing: Spacing, i: Gv) -> Gv {
    let start = Gv::scalar(&format!("x_lo_{ax}"));
    let param = Gv::scalar(&format!("dx_{ax}"));
    // spacing is a runtime per-axis value: `map_kind_{ax}` selects the face-
    // position map (0 = uniform, 1 = log, 2 = geometric cell widths), so one kernel per
    // (regime, geometry) serves every spacing (log-r, log-theta, ...) and a moving mesh updates
    // `x_lo`/`dx` on the fly while the map kind stays fixed. the face position comes from the
    // runtime map alone; the bake-time `spacing` enum stays in the signature through the
    // transition.
    //
    // this is a real branch (`cond` -> `Op::IfElse`): the log `pow` is emitted
    // inside the `if` arm and runs on a log axis alone — the uniform arm stays plain arithmetic.
    // `map_kind` is per-launch-uniform (same for every cell/lane), so every lane takes one arm.
    let map_kind = Gv::scalar(&format!("map_kind_{ax}"));
    Gv::cond(
        map_kind.cmp_gt(Gv::from_f64(1.5)),
        || {
            let ratio = Gv::scalar(&format!("map_param_{ax}"));
            start + param * (ratio.powf(i) - Gv::ONE) / (ratio - Gv::ONE)
        },
        || {
            Gv::cond(
                map_kind.cmp_gt(Gv::from_f64(0.5)),
                || start * Gv::from_f64(10.0).powf(i * param),
                || start + i * param,
            )
        },
    )
}

/// the cell-center position between the bracketing faces `lo` and `hi` on grid axis `ax`,
/// per the runtime spacing map: a log axis (`map_kind = 1`) centers at the geometric mean
/// `sqrt(lo*hi)`, every other map at the arithmetic midpoint `(lo + hi)/2`. this is the
/// in-kernel mirror of the host `AxisMap::center` — the single position definition
/// `stagger_coord(Center)` reports and `set_initial` seeds primitives at. the well-balanced
/// ladder must use this definition: its machine-exactness is the statement that the body
/// potential is evaluated at the exact positions where the seeded column satisfies its
/// discrete equilibrium, and an arithmetic midpoint on a log axis sits O((dr/r)^2 r) off
/// every cell — the balance would then hold a column displaced from the evolved one.
/// `map_kind` is per-launch-uniform, so every lane takes one arm; on a uniform or
/// geometric map the selected arm is the arithmetic midpoint, value-identical to the
/// unconditional spelling.
pub(crate) fn gv_axis_center_between(ax: usize, lo: Gv, hi: Gv) -> Gv {
    let map_kind = Gv::scalar(&format!("map_kind_{ax}"));
    let is_log = map_kind.cmp_gt(Gv::from_f64(0.5)) & map_kind.cmp_lt(Gv::from_f64(1.5));
    Gv::cond(
        is_log,
        || (lo * hi).sqrt(),
        || (lo + hi) * Gv::from_f64(0.5),
    )
}

/// the diagonal scale factor `h_dir(pos)` — the metric lame coefficient. cartesian: 1;
/// spherical: (1, r, r*sin(theta)); cylindrical: (1, r, 1). `pos` is coordinate-indexed
/// (pos[0]=r, pos[1]=theta). the `match` is build-time (Coords is the codegen geometry).
pub(crate) fn gv_scale_factor(coords: Coords, dir: usize, pos: &[Gv]) -> Gv {
    match (coords, dir) {
        (Coords::Cartesian, _) => Gv::ONE,
        (Coords::Spherical, 1) => pos[0],                // r
        (Coords::Spherical, 2) => pos[0] * pos[1].sin(), // r*sin(theta)
        (Coords::Spherical, _) => Gv::ONE,
        (Coords::Cylindrical, 1) => pos[0], // r (phi direction)
        (Coords::Cylindrical, _) => Gv::ONE,
    }
}

/// the coordinate value for an ungridded metric slot `c` — the symmetry default the GR kernels
/// fill in for a coordinate the grid leaves out. spherical: the polar angle defaults to the equator
/// (theta = pi/2, sin theta = 1); the azimuth and every cartesian / cylindrical suppressed axis
/// default to 0. this is the single chart authority for ungridded fills — the GR flux / c2p /
/// wave-speed / godunov position builders all read it, so
/// spherical stays bit-identical (same values) and cartesian / cylindrical fall out at zero.
pub(crate) fn gv_ungridded_slot(coords: Coords, c: usize) -> Gv {
    match (coords, c) {
        (Coords::Spherical, 1) => Gv::from_f64(std::f64::consts::FRAC_PI_2),
        _ => Gv::ZERO,
    }
}

/// per-cell physical inverse widths `1 / (h_d * width_d)` per gridded axis — the metric-
/// correct CFL length scale (the wave crosses the physical extent `h_d * \Delta coord_d`, the
/// coordinate width scaled by the lame factor `h_d`), computed in-kernel from the cell index.
/// `axes[d]` is the coordinate gridded axis `d` maps to.
/// (the cartesian-uniform CFL still uses the host's precomputed `inv_dx_d` scalar — this is
/// the curvilinear / non-uniform path.)
pub fn cell_inv_phys_widths_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: usize,
) -> Vec<Gv> {
    let half = Gv::from_f64(0.5);
    let lo: Vec<Gv> = (0..ndim)
        .map(|d| gv_axis_face_at(d, spacing[d], 0))
        .collect();
    let hi: Vec<Gv> = (0..ndim)
        .map(|d| gv_axis_face_at(d, spacing[d], 1))
        .collect();
    let width: Vec<Gv> = (0..ndim).map(|d| hi[d] - lo[d]).collect();
    // coordinate-indexed cell center: scale_factor reads pos by coordinate, so place each
    // gridded axis's center at its coordinate slot (symmetry slots stay 0, and the scale
    // factor reads the gridded ones).
    let mut center = vec![Gv::ZERO; 3];
    for d in 0..ndim {
        center[axes[d]] = (lo[d] + hi[d]) * half;
    }
    (0..ndim)
        .map(|d| {
            let h = gv_scale_factor(coords, axes[d], &center); // h of the coordinate this axis is
            Gv::ONE / (h * width[d]) // 1 / (h_d * width_d)
        })
        .collect()
}

/// per-cell finite-volume geometric factors in Gv:
/// inverse cell volume, per-axis lo/hi face areas, and
/// volume-weighted centroids, all from the cell index. the foundation the curvilinear
/// godunov (area-weighted divergence) + the geometric momentum source trace through.
#[derive(Clone)]
pub struct CellGeometryGv {
    pub inv_volume: Gv,
    pub area_lo: Vec<Gv>,
    pub area_hi: Vec<Gv>,
    pub centroid: Vec<Gv>,
}

/// `a^n` for a small literal power `n >= 1` as repeated multiply — exact, and the graph carries
/// plain multiplies, so the analytic radial integrals stay byte-form-identical across rebuilds.
pub(crate) fn gv_powi(a: Gv, n: u32) -> Gv {
    let mut acc = a;
    for _ in 1..n {
        acc = acc * a;
    }
    acc
}

/// per axis: `(lo face, hi face, width)` from the index map.
fn gv_faces(spacing: &[Spacing], ndim: usize) -> (Vec<Gv>, Vec<Gv>, Vec<Gv>) {
    let lo: Vec<Gv> = (0..ndim)
        .map(|d| gv_axis_face_at(d, spacing[d], 0))
        .collect();
    let hi: Vec<Gv> = (0..ndim)
        .map(|d| gv_axis_face_at(d, spacing[d], 1))
        .collect();
    let width: Vec<Gv> = (0..ndim).map(|d| hi[d] - lo[d]).collect();
    (lo, hi, width)
}

/// build the per-cell finite-volume geometric factors in Gv (cartesian / spherical /
/// cylindrical), axis-role driven. `axes[d]`
/// is the coordinate gridded axis `d` represents (identity for cartesian/spherical; the cyl
/// r-z swirl folds phi). analytic exact-integral factors + volume-weighted centroids.
pub fn cell_geometry_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: usize,
) -> CellGeometryGv {
    let (lo, hi, width) = gv_faces(spacing, ndim);
    match coords {
        Coords::Cartesian => cartesian_geometry_gv(&lo, &hi, &width, ndim),
        Coords::Spherical => spherical_geometry_gv(&lo, &hi, &width, ndim, None),
        Coords::Cylindrical => cylindrical_geometry_gv(&lo, &hi, &width, axes, ndim),
    }
}

/// the covariant (valencia) finite-volume geometry: face weights are integrals of the
/// densitized volume element `alpha sqrt(gamma) = r^2 sin(theta)` (the det-g-flat family)
/// over the face, and the divergence they build is the coordinate form
/// `(1/sqrt(gamma)) d_i (alpha sqrt(gamma) F^i)` — what the covariant momentum S_i and the
/// contravariant fluxes v^i require. it differs from the flat (orthonormal) geometry in the
/// angular face weights alone: the theta face carries `int r^2 dr` where the physical area
/// carries `int r dr` (the arc-length measure) — an orthonormal-form angular divergence
/// applied to the covariant S_theta is short by a factor r in every theta-direction force
/// (the pressure gradient in particular, visible to a state with theta structure and
/// non-radial flow). radial faces and the volume coincide with the flat geometry, so 1D radial GR
/// is untouched. spherical-only (the curved backgrounds are spherical).
pub fn cell_geometry_covariant_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: usize,
    // the kerr spin `a` (as the `kerr_spin` scalar) when the densitized measure is the kerr
    // `alpha sqrt(gamma) = Sigma sin(theta) = (r^2 + a^2 cos^2 theta) sin(theta)`; `None` for
    // the det-g-flat family (schwarzschild, kerr-schild) whose measure is `r^2 sin(theta)`.
    kerr_spin: Option<Gv>,
) -> CellGeometryGv {
    let (lo, hi, width) = gv_faces(spacing, ndim);
    match coords {
        // det-g-flat cartesian: alpha sqrt(gamma) = 1 = the flat cartesian coordinate volume, and
        // every direction is equivalent on this chart, so the covariant geometry is the flat
        // cartesian geometry (the spherical int r^2 dr angular correction is spherical-specific).
        Coords::Cartesian => cartesian_geometry_gv(&lo, &hi, &width, ndim),
        // the angular (theta) face carries the covariant int r^2 dr measure where the flat geometry
        // carries the arc-length int r dr — the sole flat-vs-covariant difference (see the type doc).
        Coords::Spherical => spherical_geometry_gv(&lo, &hi, &width, ndim, Some(kerr_spin)),
        // cylindrical: the flat cylindrical geometry already uses the coordinate R-measure (volume
        // int R dR, face areas with sqrt(gamma_phi-phi) = R), so physical == coordinate == covariant
        // (h_phi = R appears identically in the volume and every face, where spherical takes an
        // arc-length shortcut). alpha sqrt(gamma) = R is the det-g-flat identity; a = 0, so the
        // kerr moment vanishes.
        Coords::Cylindrical => cylindrical_geometry_gv(&lo, &hi, &width, axes, ndim),
    }
}

/// the out-of-plane induction divergence geometry for the flat spherical (r,theta) plane:
/// the toroidal physical component obeys `d_t B_phi = -(1/r)[d_r(r F^r) + d_theta F^theta]`
/// ((curl E)_phi with in-plane lame factors h_r = 1, h_theta = r, so the in-plane measure is
/// r dr dtheta; the gas r^2 sin(theta) measure would inject spurious
/// `-F^r/r - cot(theta) F^theta/r` sources). equivalently the conservation law
/// `d_t(r B_phi) + d_r(r F^r) + d_theta F^theta = 0` on the measure `r dr dtheta`: face
/// weights (r_face, 1), volume `r_c dr dtheta` with the arithmetic midpoint r_c — exact,
/// the r-weight is linear in r.
pub(crate) fn oop_curl_geometry_sph_rtheta_gv(spacing: &[Spacing]) -> CellGeometryGv {
    let (lo, hi, width) = gv_faces(spacing, 2);
    let half = Gv::from_f64(0.5);
    let r_c = (lo[0] + hi[0]) * half;
    let th_c = (lo[1] + hi[1]) * half;
    CellGeometryGv {
        inv_volume: Gv::ONE / (r_c * width[0] * width[1]),
        area_lo: vec![lo[0] * width[1], width[0]],
        area_hi: vec![hi[0] * width[1], width[0]],
        centroid: vec![r_c, th_c],
    }
}

// cartesian: V = prod(width); A_dir = prod_{j!=dir}(width); centroid = arithmetic mid.
fn cartesian_geometry_gv(lo: &[Gv], hi: &[Gv], width: &[Gv], ndim: usize) -> CellGeometryGv {
    let mut vol = width[0];
    for d in 1..ndim {
        vol = vol * width[d];
    }
    let inv_volume = Gv::ONE / vol;
    let half = Gv::from_f64(0.5);
    let mut area_lo = Vec::with_capacity(ndim);
    let mut area_hi = Vec::with_capacity(ndim);
    let mut centroid = Vec::with_capacity(ndim);
    for dir in 0..ndim {
        let mut a: Option<Gv> = None;
        for (j, &w) in width.iter().enumerate() {
            if j == dir {
                continue;
            }
            a = Some(match a {
                None => w,
                Some(acc) => acc * w,
            });
        }
        let area = a.unwrap_or(Gv::ONE); // 1D: unit perpendicular face
        area_lo.push(area);
        area_hi.push(area);
        centroid.push((lo[dir] + hi[dir]) * half); // flat cell centroid = arithmetic mid
    }
    CellGeometryGv {
        inv_volume,
        area_lo,
        area_hi,
        centroid,
    }
}

// spherical (r, theta, phi): analytic exact-integral factors, volume-weighted centroids
// (the radial centroid is volume-weighted). `covariant` selects
// the coordinate-form (alpha sqrt(gamma)) angular face weights for the GR path; `None` keeps the
// physical (arc-length) areas — see `cell_geometry_covariant_gv`.
// `covariant`: `None` = the physical (orthonormal) geometry; `Some(None)` = the coordinate-form
// r^2 sin(theta) measure (det-g-flat GR); `Some(Some(a))` = the kerr Sigma sin(theta) measure.
fn spherical_geometry_gv(
    lo: &[Gv],
    hi: &[Gv],
    width: &[Gv],
    ndim: usize,
    covariant: Option<Option<Gv>>,
) -> CellGeometryGv {
    let pi = std::f64::consts::PI;
    let (rl, rh) = (lo[0], hi[0]);
    let ir1 = (gv_powi(rh, 3) - gv_powi(rl, 3)) / Gv::from_f64(3.0); // int r^2 dr
    let ir2 = (gv_powi(rh, 2) - gv_powi(rl, 2)) / Gv::from_f64(2.0); // int r dr
    let centroid_r = volume_weighted_centroid(Geometry::Spherical, 0, rl, rh);

    let (i_theta, sin_tl, sin_th, centroid_t) = if ndim >= 2 {
        let (tl, th) = (lo[1], hi[1]);
        let (ctl, cth) = (tl.cos(), th.cos());
        let it = ctl - cth; // cos(tl) - cos(th)
        (
            it,
            tl.sin(),
            th.sin(),
            volume_weighted_centroid(Geometry::Spherical, 1, tl, th),
        )
    } else {
        let z = Gv::ZERO;
        (Gv::from_f64(2.0), z, z, Gv::from_f64(pi / 2.0)) // cos(0)-cos(pi)=2; centroid at pi/2
    };
    let i_phi = if ndim >= 3 {
        width[2]
    } else {
        Gv::from_f64(2.0 * pi)
    };

    // the kerr Sigma-measure moments: i_c2s = int cos^2(theta) sin(theta) dtheta over the cell
    // (the a^2 companion of i_theta), c2 at the theta faces, and the radial width. the spinning
    // kerr measure is the one that reads them.
    let kerr = covariant.clone().flatten();
    let (i_c2s, c2_lo, c2_hi, wr) = if ndim >= 2 {
        let (tl, th) = (lo[1], hi[1]);
        let (ctl, cth) = (tl.cos(), th.cos());
        let third = Gv::from_f64(1.0 / 3.0);
        (
            (gv_powi(ctl, 3) - gv_powi(cth, 3)) * third,
            ctl * ctl,
            cth * cth,
            rh - rl,
        )
    } else {
        // full sphere: int cos^2 sin over [0, pi] = 2/3.
        (Gv::from_f64(2.0 / 3.0), Gv::ZERO, Gv::ZERO, rh - rl)
    };

    // volume: physical and det-g-flat covariant share int r^2 sin = ir1 * i_theta; the kerr
    // measure adds the a^2 moment: int Sigma sin = ir1 * i_theta + a^2 * wr * i_c2s.
    let vol = match &kerr {
        Some(a) => ir1 * i_theta * i_phi + *a * *a * wr * i_c2s * i_phi,
        None => ir1 * i_theta * i_phi,
    };
    let inv_volume = Gv::ONE / vol;
    let omega = i_theta * i_phi; // angular solid-angle measure for the r-face

    let mut area_lo = vec![Gv::ZERO; ndim];
    let mut area_hi = vec![Gv::ZERO; ndim];
    let mut centroid = vec![Gv::ZERO; ndim];
    // r-face weight: r_f^2 * Omega, plus the kerr a^2 cos^2 moment of the Sigma measure.
    let r_face_weight = |rf: Gv| match &kerr {
        Some(a) => gv_powi(rf, 2) * omega + *a * *a * i_c2s * i_phi,
        None => gv_powi(rf, 2) * omega,
    };
    area_lo[0] = r_face_weight(rl);
    area_hi[0] = r_face_weight(rh);
    centroid[0] = centroid_r;
    // the angular radial moment: physical (orthonormal) faces carry the arc-length measure
    // int r dr; the covariant (coordinate) form carries the alpha sqrt(gamma) measure, int r^2 dr
    // (+ the kerr a^2 cos^2(theta_face) width moment).
    let ir_ang = if covariant.is_some() { ir1 } else { ir2 };
    if ndim >= 2 {
        let t_face_weight = |sin_f: Gv, c2_f: Gv| match &kerr {
            Some(a) => sin_f * (ir1 + *a * *a * c2_f * wr) * i_phi,
            None => sin_f * ir_ang * i_phi,
        };
        area_lo[1] = t_face_weight(sin_tl, c2_lo);
        area_hi[1] = t_face_weight(sin_th, c2_hi);
        centroid[1] = centroid_t;
    }
    if ndim >= 3 {
        // phi-face weight: physical = Ir2 * dtheta (arc length); covariant = the full measure
        // over the (r, theta) face.
        let aphi = match &kerr {
            Some(a) => ir1 * i_theta + *a * *a * wr * i_c2s,
            None if covariant.is_some() => ir1 * i_theta,
            None => ir2 * width[1],
        };
        area_lo[2] = aphi;
        area_hi[2] = aphi;
        centroid[2] = (lo[2] + hi[2]) * Gv::from_f64(0.5); // arithmetic mid (uniform in phi)
    }
    CellGeometryGv {
        inv_volume,
        area_lo,
        area_hi,
        centroid,
    }
}

// cylindrical (coords 0=r, 1=phi, 2=z): h=(1,r,1), sqrt(g)=r. axis-role driven — one builder
// serves (r,phi)/(r,z)/(r,phi,z); an ungridded coordinate is a symmetry axis (its full-extent
// measure cancels in the divergence).
fn cylindrical_geometry_gv(
    lo: &[Gv],
    hi: &[Gv],
    width: &[Gv],
    axes: &[usize],
    ndim: usize,
) -> CellGeometryGv {
    let pi = std::f64::consts::PI;
    let grid_of = |coord: usize| -> Option<usize> { axes.iter().position(|&c| c == coord) };
    let r_ax = grid_of(0).expect("cylindrical: the radial coordinate (0) must be gridded");
    let phi_ax = grid_of(1);
    let z_ax = grid_of(2);

    let (rl, rh) = (lo[r_ax], hi[r_ax]);
    let ir2 = (gv_powi(rh, 2) - gv_powi(rl, 2)) / Gv::from_f64(2.0); // int r dr
    let centroid_r = volume_weighted_centroid(Geometry::Cylindrical, 0, rl, rh);
    let dr = rh - rl;

    // transverse measures: gridded -> the grid width; symmetry -> the full extent constant.
    let i_phi = match phi_ax {
        Some(a) => width[a],
        None => Gv::from_f64(2.0 * pi),
    };
    let i_z = match z_ax {
        Some(a) => width[a],
        None => Gv::ONE,
    };
    let inv_volume = Gv::ONE / (ir2 * i_phi * i_z);

    let half = Gv::from_f64(0.5);
    let mut area_lo = vec![Gv::ZERO; ndim];
    let mut area_hi = vec![Gv::ZERO; ndim];
    let mut centroid = vec![Gv::ZERO; ndim];
    area_lo[r_ax] = rl * i_phi * i_z; // r-face A = r_face * i_phi * i_z
    area_hi[r_ax] = rh * i_phi * i_z;
    centroid[r_ax] = centroid_r;
    if let Some(a) = phi_ax {
        let aphi = dr * i_z; // phi-face A = dr * i_z
        area_lo[a] = aphi;
        area_hi[a] = aphi;
        centroid[a] = (lo[a] + hi[a]) * half;
    }
    if let Some(a) = z_ax {
        let az = ir2 * i_phi; // z-face A = Ir2 * i_phi
        area_lo[a] = az;
        area_hi[a] = az;
        centroid[a] = (lo[a] + hi[a]) * half;
    }
    CellGeometryGv {
        inv_volume,
        area_lo,
        area_hi,
        centroid,
    }
}

/// the geometry probe: write `inv_volume` + the dir-0 lo/hi face
/// areas + the dir-0 volume-weighted centroid, so a host test bit-diffs them against the
/// analytic formulas (incl. log spacing). identity axes (the probe is always natural-order).
pub fn geometry_probe_gv(
    coords: Coords,
    spacing: &[Spacing],
    ndim: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let axes: Vec<usize> = (0..ndim).collect();
    let g = cell_geometry_gv(coords, spacing, &axes, ndim);
    let writes = vec![
        (
            "inv_volume".to_string(),
            "inv_volume".into(),
            g.inv_volume.node(),
        ),
        (
            "area_lo_0".to_string(),
            "area_lo_0".into(),
            g.area_lo[0].node(),
        ),
        (
            "area_hi_0".to_string(),
            "area_hi_0".into(),
            g.area_hi[0].node(),
        ),
        (
            "centroid_0".to_string(),
            "centroid_0".into(),
            g.centroid[0].node(),
        ),
    ];
    (end_trace(), writes)
}

#[cfg(test)]
mod local_width_tests {
    use super::*;
    use symbi_ir::backends::interp::{Backend, Cpu};
    use symbi_ir::gv::{begin_trace, end_trace, with_trace};
    use symbi_ir::passes::scalarize::{LoweredFn, scalarize_kernel};

    fn eval_width(values: &[(&str, f64)]) -> f64 {
        begin_trace();
        let output = gv_axis_width(0, Spacing::Uniform).node();
        let lowered = with_trace(|trace| {
            let scalarized = scalarize_kernel(trace.graph(), &[output]);
            let ty = trace.graph().ty(output).clone();
            LoweredFn {
                name: "cell_width".to_string(),
                params: scalarized.params,
                body: scalarized.body,
                results: vec![scalarized.outputs[0].clone()],
                result_element: ty.element,
                result_shape: ty.shape,
            }
        });
        let inputs = lowered
            .params
            .iter()
            .map(|param| {
                values
                    .iter()
                    .find(|(name, _)| *name == param.name.as_str())
                    .map(|(_, value)| *value)
                    .unwrap_or_else(|| panic!("missing width parameter {}", param.name))
            })
            .collect::<Vec<_>>();
        let value = Cpu.eval_elemental(&lowered, &inputs)[0];
        end_trace();
        value
    }

    #[test]
    fn geometric_width_is_local_to_the_cell() {
        let values = [
            ("x_lo_0", 0.2),
            ("dx_0", 0.03),
            ("map_kind_0", 2.0),
            ("map_param_0", 1.2),
            ("_coord_0", 5.0),
        ];
        let width = eval_width(&values);
        let expected = 0.03 * 1.2_f64.powi(5);
        assert!((width - expected).abs() <= 32.0 * f64::EPSILON * expected);
        assert_ne!(
            width, values[1].1,
            "graded width collapsed to the base width"
        );
    }

    #[test]
    fn logarithmic_width_is_local_to_the_cell() {
        let values = [
            ("x_lo_0", 0.5),
            ("dx_0", 0.04),
            ("map_kind_0", 1.0),
            ("map_param_0", 0.0),
            ("_coord_0", 7.0),
        ];
        let width = eval_width(&values);
        let expected = 0.5 * (10.0_f64.powf(8.0 * 0.04) - 10.0_f64.powf(7.0 * 0.04));
        assert!((width - expected).abs() <= 64.0 * f64::EPSILON * expected);
    }
}
