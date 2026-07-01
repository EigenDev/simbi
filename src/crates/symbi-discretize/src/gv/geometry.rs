// =============================================================================
// geometry.rs
//
// in-kernel geometry: index->physical metric, finite-volume factors, and the divergence operators.
// =============================================================================

use super::*;
use symbi_geometry::{Metric, Schwarzschild, SchwarzschildKS};
use symbi_algebra::Tensor;


/// one cartesian-uniform finite-volume divergence sum over the gridded axes:
/// `sum_i (F_i[coord+e_i] - F_i[coord]) / dx_i`. `base` names the per-direction flux field
/// (`{base}_{i}`, runtime `{base}[{i}]`) — `mass_flux` / `mom_flux_{k}` / `nrg_flux`. the lo
/// read is the direct cell read, the hi a `+e_i` field_shifted (LoadAt); dt is the caller's.
fn gv_divergence_cartesian(base: &str, ndim: u8) -> Gv {
    let mut acc: Option<Gv> = None;
    for ii in 0..ndim {
        let key = format!("{base}_{ii}");
        let rt = format!("{base}[{ii}]");
        let f_lo = Gv::field_shifted(&key, &rt, ndim, ii, 0); // == Gv::field (offset 0)
        let f_hi = Gv::field_shifted(&key, &rt, ndim, ii, 1);
        let dx = Gv::scalar(&format!("dx_{ii}"));
        let term = (f_hi - f_lo) / dx;
        acc = Some(match acc {
            None => term,
            Some(a) => a + term,
        });
    }
    acc.expect("godunov divergence needs ndim >= 1")
}


/// the analytic AREA-WEIGHTED curvilinear divergence: `(1/V) sum_i (F_i[+e_i]*A_hi_i -
/// F_i*A_lo_i)` — each face flux weighted by its face area BEFORE the telescope, the cell sum
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


/// the per-direction inverse divergence operator for `base`: cartesian-uniform `(F_hi -
/// F_lo)/dx_i`, else the area-weighted `(1/V)(F_hi*A_hi - F_lo*A_lo)` from `geo`.
pub(crate) fn gv_divergence(base: &str, ndim: u8, geo: &Option<CellGeometryGv>) -> Gv {
    match geo {
        None => gv_divergence_cartesian(base, ndim),
        Some(g) => gv_divergence_weighted(base, ndim, g),
    }
}

/// the area-weighted curvilinear divergence with the RADIAL face fluxes weighted by the lapse at
/// the face. Font (2008) Eq (34): for a static background the conserved mass/energy flux is the
/// CONTRAVARIANT-velocity flux `F^i = U (v^i - beta^i/alpha)`, which on a diagonal metric carries
/// `v^r = alpha * V_rhat` — i.e. one factor of the LAPSE per radial face vs the orthonormal flat
/// flux the riemann solver produces. angular faces sit at the cell radius (constant lapse) and need
/// no per-face weight; the single cell lapse the godunov already applies covers them. `face_lapse =
/// (alpha_lo, alpha_hi)` are the lapses at the cell's lower/upper RADIAL faces (grid axis 0).
fn gv_divergence_radial_lapse_weighted(
    base: &str,
    ndim: u8,
    geo: &CellGeometryGv,
    face_lapse: (Gv, Gv),
) -> Gv {
    let mut acc: Option<Gv> = None;
    for ii in 0..ndim {
        let key = format!("{base}_{ii}");
        let rt = format!("{base}[{ii}]");
        let f_lo = Gv::field_shifted(&key, &rt, ndim, ii, 0);
        let f_hi = Gv::field_shifted(&key, &rt, ndim, ii, 1);
        let d = ii as usize;
        // grid axis 0 is the radial axis for the curvilinear (spherical) instances; weight its two
        // faces by the lapse there. the other (angular) faces are at constant radius -> unit weight.
        let (w_lo, w_hi) = if ii == 0 { (face_lapse.0, face_lapse.1) } else { (Gv::ONE, Gv::ONE) };
        let diff = w_hi * f_hi * geo.area_hi[d] - w_lo * f_lo * geo.area_lo[d];
        acc = Some(match acc {
            None => diff,
            Some(a) => a + diff,
        });
    }
    acc.expect("godunov divergence needs ndim >= 1") * geo.inv_volume
}

/// the inverse divergence for the mass / energy conserved scalars on a static GR background: the
/// area-weighted divergence with the radial faces lapse-weighted (Font Eq 34 contravariant flux).
/// `face_lapse = None` (flat spacetime) reduces to the plain `gv_divergence` -> bit-identical.
pub(crate) fn gv_divergence_lapse(
    base: &str,
    ndim: u8,
    geo: &Option<CellGeometryGv>,
    face_lapse: Option<(Gv, Gv)>,
) -> Gv {
    match (geo, face_lapse) {
        (Some(g), Some(fl)) => gv_divergence_radial_lapse_weighted(base, ndim, g, fl),
        _ => gv_divergence(base, ndim, geo),
    }
}


/// the GR LAPSE WEIGHT `alpha(x)` for the spatial-RHS densitization (Valencia 3+1). the conserved
/// update `d_t(sqrt(gamma) U) + d_i(sqrt(-g) F) = sqrt(-g) S` reduces, on a STATIC DIAGONAL
/// background, to weighting the flux divergence + the geometric momentum source by the lapse
/// (`sqrt(-g) = alpha sqrt(gamma)`; the Schwarzschild coordinate gift `sqrt(-g) = sqrt(gamma_flat)`
/// leaves the face areas flat and folds `1/sqrt(gamma) = alpha/sqrt(gamma_flat)` into a single
/// `alpha` on the RHS). flat spacetimes (EVERY realized metric today) have `alpha = 1` -> `None`,
/// so the RHS is untouched and BIT-IDENTICAL — the de-risk seam. a GR metric (Schwarzschild, B3.1)
/// returns `Some(alpha)` dispatched `Coords -> concrete Metric -> metric.lapse(centroid)` as a
/// traced Gv expression in the cell coordinate (the established B1 source-dispatch pattern).
pub(crate) fn gv_lapse_weight(coords: Coords, spacetime: Spacetime, coord_centroid: &[Gv]) -> Option<Gv> {
    let _ = coords; // the spatial coords select the concrete `Metric` impl in the GR arms.
    match spacetime {
        // flat (Minkowski) lapse alpha = 1: no densitization -> the weight is ELIDED from the graph
        // (no unity multiply) -> bit-identical.
        Spacetime::Minkowski => None,
        // r = the radial centroid (coordinate slot 0); the lapse is radial-only, so the D=1
        // evaluation gives the correct alpha for any sim dimension.
        _ => Some(gv_metric_lapse_at(spacetime, coord_centroid[0])),
    }
}

/// the analytic lapse alpha(r) as a traced Gv, dispatched `Spacetime -> concrete Metric ->
/// Metric::lapse` — the SINGLE codegen seam for the GR lapse. every densitization consumer (the cell
/// weight `gv_lapse_weight`, the radial face weight `gv_radial_face_lapse`) reads the lapse HERE, so a
/// new analytic background is a new `Metric` impl + one arm, not a lapse formula re-inlined per
/// consumer. `M` rides as the host-filled scalar `schwarzschild_mass` so the kernel stays M-agnostic.
/// flat spacetime never reaches this (its weight is elided by the caller); calling it is a bug.
pub(crate) fn gv_metric_lapse_at(spacetime: Spacetime, r: Gv) -> Gv {
    let mass = Gv::scalar("schwarzschild_mass");
    match spacetime {
        // alpha = sqrt(1 - 2M/r) (schwarzschild coords) / alpha = 1/sqrt(1 + 2M/r) (kerr-schild),
        // each from its `Metric` impl (the SINGLE source of the lapse expression).
        Spacetime::Schwarzschild => Schwarzschild { mass }.lapse(Tensor::new([r])),
        Spacetime::KerrSchild => SchwarzschildKS { mass }.lapse(Tensor::new([r])),
        Spacetime::Minkowski => unreachable!("flat lapse is elided by the densitization caller"),
    }
}

/// the analytic lapse SQUARE alpha^2(r) from `Metric::lapse_sq` — the CFL radial coordinate-speed
/// factor alpha sqrt(gamma^{rr}) = alpha^2 for the det-g-flat family (schwarzschild alpha^2 = f;
/// kerr-schild alpha^2 = 1/(1 + 2M/r)). the closed form (NOT `lapse().powi(2)`) so the genericized
/// wave-speed map reproduces the pre-refactor `f` node bitwise. flat never reaches this.
pub(crate) fn gv_metric_lapse_sq_at(spacetime: Spacetime, r: Gv) -> Gv {
    let mass = Gv::scalar("schwarzschild_mass");
    match spacetime {
        Spacetime::Schwarzschild => Schwarzschild { mass }.lapse_sq(Tensor::new([r])),
        Spacetime::KerrSchild => SchwarzschildKS { mass }.lapse_sq(Tensor::new([r])),
        Spacetime::Minkowski => unreachable!("flat lapse-square is elided by the CFL caller"),
    }
}

/// the analytic radial shift beta^r(r) from `Metric::shift` — nonzero ONLY for a shifted background
/// (kerr-schild beta^r = 2M/(r + 2M)); the static diagonal cases (Minkowski, Schwarzschild) have
/// beta = 0 -> None so the caller elides the shift term (bit-identical, no `- 0`).
pub(crate) fn gv_metric_shift_r_at(spacetime: Spacetime, r: Gv) -> Option<Gv> {
    match spacetime {
        Spacetime::Minkowski | Spacetime::Schwarzschild => None,
        Spacetime::KerrSchild => {
            let mass = Gv::scalar("schwarzschild_mass");
            Some(SchwarzschildKS { mass }.shift(Tensor::new([r]))[0])
        }
    }
}

/// the static-background lapse at the cell's lower/upper RADIAL faces (grid axis 0), for the
/// Font (2008) Eq (34) mass/energy flux weighting: on a ZERO-SHIFT background the conserved scalar
/// flux transports with the CONTRAVARIANT `v^r = alpha V_rhat`, so each radial face flux carries the
/// lapse there (vs the orthonormal flat flux the riemann solver returns). shares the `gv_metric_lapse_at`
/// seam with `gv_lapse_weight`; flat spacetime -> None.
pub(crate) fn gv_radial_face_lapse(spacetime: Spacetime, spacing: &[Spacing]) -> Option<(Gv, Gv)> {
    match spacetime {
        Spacetime::Minkowski => None,
        // both static-diagonal (Schwarzschild) and kerr-schild carry the SAME contravariant face
        // weight alpha_face on the flat mass/energy flux (v^r = alpha V_rhat on the flux the riemann
        // solver returns). the KS shift piece `- b U` is a SEPARATE additive flux-field term (it reads
        // the conserved U, so it is folded into the flux fields, not applied as a face multiply here);
        // once in the flux field it inherits this same alpha_face weight.
        Spacetime::Schwarzschild | Spacetime::KerrSchild => Some((
            gv_metric_lapse_at(spacetime, gv_axis_face_at(0, spacing[0], 0)),
            gv_metric_lapse_at(spacetime, gv_axis_face_at(0, spacing[0], 1)),
        )),
    }
}


/// `true` iff the flat unweighted `(F_hi-F_lo)/dx` divergence applies (no in-kernel metric).
pub(crate) fn is_cartesian_uniform(coords: Coords, spacing: &[Spacing]) -> bool {
    coords == Coords::Cartesian && spacing.iter().all(|&s| s == Spacing::Uniform)
}


// =============================================================================
// in-kernel GEOMETRY — the substrate metric expressed in Gv. `Gv::coord` is the
// index->physical bridge; the coordinate-system formulas are a BUILD-TIME `match` on
// `Coords` (the kernel is generated per geometry, so the branch is resolved at trace
// time, not a runtime select). this is the foundation every curvilinear operator (CFL
// widths, godunov divergence, geometric sources, CT curl) traces through.
// =============================================================================

/// the face at `coord + offset` along axis `ax` as a physical position (offset 0 = lo face,
/// 1 = hi face). `x_lo_{ax}` + `dx_{ax}` are the grid scalars (dx = width for Uniform, the
/// log-slope for Log). the integer coord promotes to f64 against the scalars at lowering.
pub(crate) fn gv_axis_face_at(ax: usize, spacing: Spacing, offset: i64) -> Gv {
    let coord = Gv::coord(ax as u8);
    let start = Gv::scalar(&format!("x_lo_{ax}"));
    let param = Gv::scalar(&format!("dx_{ax}"));
    let i = if offset == 0 { coord } else { coord + Gv::from_f64(offset as f64) };
    match spacing {
        Spacing::Uniform => start + i * param,                      // start + i*dx
        Spacing::Log => start * Gv::from_f64(10.0).powf(i * param), // start * 10^(i*slope)
    }
}


/// the diagonal scale factor `h_dir(pos)` — the metric Lame coefficient. Cartesian: 1;
/// Spherical: (1, r, r*sin(theta)); Cylindrical: (1, r, 1). `pos` is coordinate-indexed
/// (pos[0]=r, pos[1]=theta). the `match` is build-time (Coords is the codegen geometry).
pub(crate) fn gv_scale_factor(coords: Coords, dir: usize, pos: &[Gv]) -> Gv {
    match (coords, dir) {
        (Coords::Cartesian, _) => Gv::ONE,
        (Coords::Spherical, 1) => pos[0],                  // r
        (Coords::Spherical, 2) => pos[0] * pos[1].sin(),   // r*sin(theta)
        (Coords::Spherical, _) => Gv::ONE,
        (Coords::Cylindrical, 1) => pos[0],                // r (phi direction)
        (Coords::Cylindrical, _) => Gv::ONE,
    }
}


/// per-cell PHYSICAL inverse widths `1 / (h_d * width_d)` per gridded axis — the metric-
/// correct CFL length scale (the wave crosses the physical extent `h_d * Δcoord_d`, not the
/// coordinate width), computed in-kernel from the cell index.
/// `axes[d]` is the coordinate gridded axis `d` maps to.
/// (the cartesian-UNIFORM CFL still uses the host's precomputed `inv_dx_d` scalar — this is
/// the curvilinear / non-uniform path.)
pub fn cell_inv_phys_widths_gv(coords: Coords, spacing: &[Spacing], axes: &[usize], ndim: usize) -> Vec<Gv> {
    let half = Gv::from_f64(0.5);
    let lo: Vec<Gv> = (0..ndim).map(|d| gv_axis_face_at(d, spacing[d], 0)).collect();
    let hi: Vec<Gv> = (0..ndim).map(|d| gv_axis_face_at(d, spacing[d], 1)).collect();
    let width: Vec<Gv> = (0..ndim).map(|d| hi[d] - lo[d]).collect();
    // coordinate-indexed cell center: scale_factor reads pos by coordinate, so place each
    // gridded axis's center at its coordinate slot (symmetry slots stay 0, never read).
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


/// `a^n` for a small literal power `n >= 1` as repeated multiply — exact (no Pow), so the
/// analytic radial integrals stay byte-form-identical across rebuilds.
fn gv_powi(a: Gv, n: u32) -> Gv {
    let mut acc = a;
    for _ in 1..n {
        acc = acc * a;
    }
    acc
}


/// per axis: `(lo face, hi face, width)` from the index map.
fn gv_faces(spacing: &[Spacing], ndim: usize) -> (Vec<Gv>, Vec<Gv>, Vec<Gv>) {
    let lo: Vec<Gv> = (0..ndim).map(|d| gv_axis_face_at(d, spacing[d], 0)).collect();
    let hi: Vec<Gv> = (0..ndim).map(|d| gv_axis_face_at(d, spacing[d], 1)).collect();
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
        Coords::Spherical => spherical_geometry_gv(&lo, &hi, &width, ndim),
        Coords::Cylindrical => cylindrical_geometry_gv(&lo, &hi, &width, axes, ndim),
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
    CellGeometryGv { inv_volume, area_lo, area_hi, centroid }
}


// spherical (r, theta, phi): analytic exact-integral factors, volume-weighted centroids
// (the radial centroid is volume-weighted, not the coordinate center).
fn spherical_geometry_gv(lo: &[Gv], hi: &[Gv], width: &[Gv], ndim: usize) -> CellGeometryGv {
    let pi = std::f64::consts::PI;
    let (rl, rh) = (lo[0], hi[0]);
    let ir1 = (gv_powi(rh, 3) - gv_powi(rl, 3)) / Gv::from_f64(3.0); // int r^2 dr
    let ir2 = (gv_powi(rh, 2) - gv_powi(rl, 2)) / Gv::from_f64(2.0); // int r dr
    let ir_cnum = (gv_powi(rh, 4) - gv_powi(rl, 4)) / Gv::from_f64(4.0); // int r^3 dr
    let centroid_r = ir_cnum / ir1; // (3/4)(rh^4-rl^4)/(rh^3-rl^3)

    let (i_theta, sin_tl, sin_th, centroid_t) = if ndim >= 2 {
        let (tl, th) = (lo[1], hi[1]);
        let (ctl, cth) = (tl.cos(), th.cos());
        let it = ctl - cth; // cos(tl) - cos(th)
        // volume-weighted theta centroid: [(sin th - th cos th)]_{tl}^{th} / Itheta.
        let num = (th.sin() - th * cth) - (tl.sin() - tl * ctl);
        (it, tl.sin(), th.sin(), num / it)
    } else {
        let z = Gv::ZERO;
        (Gv::from_f64(2.0), z, z, Gv::from_f64(pi / 2.0)) // cos(0)-cos(pi)=2; centroid at pi/2
    };
    let i_phi = if ndim >= 3 { width[2] } else { Gv::from_f64(2.0 * pi) };

    let inv_volume = Gv::ONE / (ir1 * i_theta * i_phi);
    let omega = i_theta * i_phi; // angular solid-angle measure for the r-face

    let mut area_lo = vec![Gv::ZERO; ndim];
    let mut area_hi = vec![Gv::ZERO; ndim];
    let mut centroid = vec![Gv::ZERO; ndim];
    area_lo[0] = gv_powi(rl, 2) * omega; // r-face A = r_face^2 * Omega
    area_hi[0] = gv_powi(rh, 2) * omega;
    centroid[0] = centroid_r;
    if ndim >= 2 {
        area_lo[1] = ir2 * sin_tl * i_phi; // theta-face A = Ir2 * sin(theta_face) * Iphi
        area_hi[1] = ir2 * sin_th * i_phi;
        centroid[1] = centroid_t;
    }
    if ndim >= 3 {
        let aphi = ir2 * width[1]; // phi-face A = Ir2 * dtheta
        area_lo[2] = aphi;
        area_hi[2] = aphi;
        centroid[2] = (lo[2] + hi[2]) * Gv::from_f64(0.5); // arithmetic mid (uniform in phi)
    }
    CellGeometryGv { inv_volume, area_lo, area_hi, centroid }
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
    let ir_cnum = (gv_powi(rh, 3) - gv_powi(rl, 3)) / Gv::from_f64(3.0);
    let centroid_r = ir_cnum / ir2; // (2/3)(rh^3-rl^3)/(rh^2-rl^2)
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
    CellGeometryGv { inv_volume, area_lo, area_hi, centroid }
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
        ("inv_volume".to_string(), "inv_volume".into(), g.inv_volume.node()),
        ("area_lo_0".to_string(), "area_lo_0".into(), g.area_lo[0].node()),
        ("area_hi_0".to_string(), "area_hi_0".into(), g.area_hi[0].node()),
        ("centroid_0".to_string(), "centroid_0".into(), g.centroid[0].node()),
    ];
    (end_trace(), writes)
}
