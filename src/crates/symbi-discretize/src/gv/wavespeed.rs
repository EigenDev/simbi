// =============================================================================
// wavespeed.rs
//
// cfl wave-speed map + per-cell wave-speed kernel builders (the geometry-folded characteristic speed).
// =============================================================================

use super::*;
use symbi_algebra::Matrix;
use symbi_geometry::grhd_source::{grhd_covariant_source, grmhd_covariant_source};
use symbi_geometry::{KerrKS, Metric, Schwarzschild, SchwarzschildKS, SchwarzschildKSCartesian, SchwarzschildKSCylindrical};
use symbi_ir::dual::Dual;


/// trace the newtonian-MHD CFL wave-speed map — `NewtonianMhd::wave_speeds` (the
/// EXACT closed-form fast magnetosonic, not a bound; it is already cheap) folded
/// with the geometry inverse-width into `lambda = max_d (max(|sl|,|sr|) inv_w_d)`.
/// the SAME speed the flux's HLLE consumes (one physics, two consumers).
pub fn nmhd_wave_speed_map_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let rho = Gv::field("prim_rho", FieldRef::PrimRho);
    let vel: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8)));
    let pre = Gv::field("prim_pre", FieldRef::PrimPre);
    let mag: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_b{k}"), FieldRef::PrimMag(k as u8)));
    let gamma = Gv::scalar("gamma");
    let eos = IdealGas { gamma };
    let prim = MhdPrim::<Gv, 3> {
        hydro: Prim { rho, vel: Tensor::new(vel), pre },
        mag: Tensor::new(mag),
    };
    let inv_w = cfl_inv_widths_gv(coords, spacing, axes, ndim);
    let mut lambda = Gv::ZERO;
    for d in 0..ndim {
        let nhat = Tensor::<Gv, 3>::unit(axes[d]);
        let (sl, sr) = NewtonianMhd.wave_speeds(&eos, &prim, &nhat);
        lambda = lambda.max(sl.abs().max(sr.abs()) * inv_w[d]);
    }
    let writes = wave_speed_map_writes(lambda.node());
    (end_trace(), writes)
}


/// isothermal-MHD CFL wave-speed map — `IsothermalMhd::wave_speeds` (fast
/// magnetosonic at a^2 = cs^2) folded with the geometry inverse-width.
pub fn imhd_wave_speed_map_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let rho = Gv::field("prim_rho", FieldRef::PrimRho);
    let vel: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8)));
    let mag: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_b{k}"), FieldRef::PrimMag(k as u8)));
    let cs = Gv::scalar("cs");
    let eos = Isothermal { cs };
    let prim = IsoMhdPrim::<Gv, 3> {
        hydro: PrimG { rho, vel: Tensor::new(vel), pre: Zero::default() },
        mag: Tensor::new(mag),
    };
    let inv_w = cfl_inv_widths_gv(coords, spacing, axes, ndim);
    let mut lambda = Gv::ZERO;
    for d in 0..ndim {
        let nhat = Tensor::<Gv, 3>::unit(axes[d]);
        let (sl, sr) = IsothermalMhd.wave_speeds(&eos, &prim, &nhat);
        lambda = lambda.max(sl.abs().max(sr.abs()) * inv_w[d]);
    }
    let writes = wave_speed_map_writes(lambda.node());
    (end_trace(), writes)
}


// =============================================================================
// CFL wave-speed MAP — symbi-hydro's `Regime::wave_speeds_axis` traced at S=Gv into the
// COMPLETE timestep kernel: per gridded axis the characteristic speed `s_d = max(|sl|,|sr|)`,
// folded against the per-cell inverse PHYSICAL width into `lambda = max_d (s_d * inv_w_d)`.
// ONE gv trace owns physics + geometry + reduction — the regime supplies only
// `wave_speeds_axis` (the SAME function the flux's HLLE uses), the geometry is the in-kernel
// Gv metric (`cfl_inv_widths_gv`). NO Expr quartic, NO splice, NO hand-written iso speed.
//
// `axes[d]` is the COORDINATE gridded axis `d` maps to: identity for cartesian/spherical,
// `[0, 2]` for the cyl r-z swirl. the Euler map reads the normal velocity `vel[axes[d]]`
// directly (`wave_speeds_axis` reads only the normal) and leaves the non-gridded velocity
// slots ZERO — so the swirl CFL reads v_r/v_z but never the folded v_phi, and those zeroed
// slots never enter the graph (dead). the host reduces `lambda` by max -> dt = cfl/lambda_max.
// =============================================================================

/// the per-gridded-axis inverse PHYSICAL width `1/(h_d*width_d)` for the CFL — the Gv mirror
/// of the retired `flux::cfl_inv_widths`. cartesian + uniform: the host's precomputed
/// `inv_dx_d` scalar (one per gridded axis); curvilinear OR non-uniform: the per-cell width
/// from the index (`cell_inv_phys_widths_gv`, so log zones + angular `h_d=r` are tracked). the
/// regime never writes the width — it only supplies wave speeds — so it can never desync it.
fn cfl_inv_widths_gv(coords: Coords, spacing: &[Spacing], axes: &[usize], ndim: usize) -> Vec<Gv> {
    let uniform_cartesian =
        coords == Coords::Cartesian && spacing.iter().all(|&s| s == Spacing::Uniform);
    if uniform_cartesian {
        (0..ndim).map(|d| Gv::scalar(&format!("inv_dx_{d}"))).collect()
    } else {
        cell_inv_phys_widths_gv(coords, spacing, axes, ndim)
    }
}


/// the lambda write list every wave-speed map returns: one scratch output `lambda`.
fn wave_speed_map_writes(root: NodeId) -> Vec<(String, FieldBind, NodeId)> {
    vec![("lambda".to_string(), FieldRef::Scratch.into(), root)]
}

/// the gridded cell-CENTER coordinate on axis `d`, SPACING-AWARE: the geometric mean of the bounding
/// faces on a `Log` axis, the arithmetic midpoint on a `Uniform` one. every GR wave-speed map MUST
/// evaluate the metric (lapse alpha, shift beta^r, the h_c = sqrt(gamma_cc) scale factors) at this
/// radius — the uniform `x_lo + (i + 1/2) dx` formula evaluates the metric at ~r_min for EVERY cell
/// on a log grid (alpha^2/beta^r/scale factors suppressed by f(r_min)/f(r)), overestimating dt into a
/// silent CFL violation. bit-identical to the old uniform formula on a `Uniform` axis
/// (`face_at(0) + 1/2 dx = x_lo + (i + 1/2) dx`). single source shared by every wave-speed map.
fn gv_cell_center(d: usize, spacing: &[Spacing]) -> Gv {
    let half = Gv::from_f64(0.5);
    match spacing[d] {
        Spacing::Uniform => gv_axis_face_at(d, spacing[d], 0) + half * Gv::scalar(&format!("dx_{d}")),
        Spacing::Log => (gv_axis_face_at(d, spacing[d], 0) * gv_axis_face_at(d, spacing[d], 1)).sqrt(),
    }
}


/// trace the COMPLETE ideal-gas Euler CFL wave-speed map at `S = Gv` — the Newtonian regime
/// (which also drives the isothermal CFL at gamma->1) or `Rhd`. reads rho/pre + the gridded
/// normal velocities `vel[axes[d]]` (non-gridded slots left ZERO; `wave_speeds_axis` reads
/// only the normal, so they stay dead) + gamma, then folds `lambda = max_d (max(|sl|,|sr|) *
/// inv_w_d)` over the gridded axes with the in-kernel geometry widths. ONE trace: physics +
/// metric + reduction, replacing the splice-into-`flux::wave_speed_map` composition. always
/// 3-component (the swirl shares the form); `coords`/`spacing`/`axes` select plane + metric.
pub fn euler_wave_speed_map_gv<R>(
    regime: &R,
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>)
where
    R: Regime<Gv, 3, Prim = Prim<Gv, 3>, Cons = Cons<Gv, 3>>,
{
    begin_trace();
    let rho = Gv::field("prim_rho", FieldRef::PrimRho);
    // the gridded normal velocities only; the non-gridded slots (cyl r-z's v_phi) stay ZERO and
    // never enter the graph — `wave_speeds_axis` reads only the normal velocity `vel[axes[d]]`.
    // the cell-CENTER coordinate on the grid axis carrying COORDINATE `target` (radial = 0, polar =
    // 1), for the physical-velocity scale factors below. spacing-aware (log grids evaluate the metric
    // scale factors at the geometric-mean radius, not ~r_min). `None` if `target` is not a grid axis.
    let coord_at = |target: usize| -> Option<Gv> {
        axes.iter().position(|&c| c == target).map(|d| gv_cell_center(d, spacing))
    };
    let mut vel = [Gv::ZERO; 3];
    for d in 0..ndim {
        let c = axes[d];
        let raw = Gv::field(&format!("prim_v{c}"), FieldRef::PrimVel(c as u8));
        // Valencia storage: `prim.vel` is the CONTRAVARIANT v^i; the SR characteristic speed is a
        // function of the PHYSICAL velocity V^c = h_c v^c, with the metric scale factor h_c =
        // sqrt(gamma_cc). spherical GR: h_r = sqrt(gamma_rr) = 1/alpha (det-g-flat), h_theta = r,
        // h_phi = r sin(theta). the per-axis coordinate factor (alpha^2 radial / alpha angular) applied
        // below completes the Banyuls-Font coordinate speed. flat -> h = 1 (untouched, bit-identical).
        vel[c] = if matches!(spacetime, Spacetime::Minkowski) {
            raw
        } else {
            let r = coord_at(0).expect("GR wave-speed map needs a radial axis");
            match c {
                0 => raw / gv_metric_lapse_at(spacetime, r, None),
                1 => raw * r,
                2 => raw * r * coord_at(1).expect("phi scale factor needs a polar axis").sin(),
                _ => raw,
            }
        };
    }
    let pre = Gv::field("prim_pre", FieldRef::PrimPre);
    let gamma = Gv::scalar("gamma");
    let eos = IdealGas { gamma };
    let prim = Prim::<Gv, 3> { rho, vel: Tensor::new(vel), pre };
    let inv_w = cfl_inv_widths_gv(coords, spacing, axes, ndim);
    // mesh motion: the cfl signal speed is RELATIVE to the grid, `|s -+ v_g|`
    // with per-axis `v_g = mesh_adot_d * x_centroid + mesh_vtrans_d`
    // (uniform-spacing centroid; the dispatch binds the homologous hubble
    // rate on expanding axes, the translation rate on axis 0, zero
    // otherwise — the static binding makes v_g exactly zero, and `s - 0` /
    // `|s|` are bit-identical).
    let half = Gv::from_f64(0.5);
    // GR coordinate CFL: the Banyuls-Font coordinate signal speed
    //   lambda_coord^c = alpha sqrt(gamma^{cc}) lambda^{SR} - beta^c.
    // for the det-g-flat family (Schwarzschild, Kerr-Schild) the RADIAL factor alpha sqrt(gamma^{rr})
    // = alpha^2 (Schwarzschild f = 1-2M/r; kerr-schild 1/(1+2M/r)), the ANGULAR factor = alpha. a
    // ZERO-shift background keeps the multiplicative form `base * factor` (bit-identical to the
    // pre-genericization kernel); a SHIFTED background (kerr-schild beta^r != 0) subtracts beta^r per
    // characteristic root BEFORE the max|.|, which the multiplicative form cannot express. the lapse /
    // shift are radial-only, evaluated at the uniform-centroid cell radius (coord slot 0); flat ->
    // None -> the SR CFL untouched (bit-identical). the factors come from the `Metric` trait (the
    // single ADM seam), not per-spacetime formulas inlined here.
    let gr_radius: Option<Gv> = match spacetime {
        Spacetime::Minkowski => None,
        _ => {
            let d_r = axes.iter().position(|&c| c == 0).expect("GR wave-speed map needs a radial axis");
            // spacing-aware: the lapse/shift are evaluated at the geometric-mean radius on a log grid,
            // not the uniform ~r_min the old `x_lo + (i+1/2) dx` gave (silent CFL violation on log GR).
            Some(gv_cell_center(d_r, spacing))
        }
    };
    let mut lambda = Gv::ZERO;
    for d in 0..ndim {
        let (sl, sr) = regime.wave_speeds_axis(&eos, &prim, axes[d]);
        let xc = Gv::scalar(&format!("x_lo_{d}"))
            + (Gv::coord(d as u8) + half) * Gv::scalar(&format!("dx_{d}"));
        let vg = Gv::scalar(&MeshScalar::Adot(d as u8).name()) * xc
            + Gv::scalar(&MeshScalar::Vtrans(d as u8).name());
        let base = (sl - vg).abs().max((sr - vg).abs()) * inv_w[d];
        let contrib = match gr_radius {
            // flat: the SR CFL |s - vg| * inv_phys_width, untouched -> bit-identical.
            None => base,
            Some(r) => {
                let is_radial = axes[d] == 0;
                // radial factor = alpha^2 = alpha sqrt(gamma^{rr}); angular factor = alpha.
                let factor = if is_radial {
                    gv_metric_lapse_sq_at(spacetime, r, None)
                } else {
                    gv_metric_lapse_at(spacetime, r, None)
                };
                // the radial shift beta^r (kerr-schild only); zero-shift backgrounds -> None.
                match if is_radial { gv_metric_shift_r_at(spacetime, r, None) } else { None } {
                    // zero shift (Schwarzschild + every angular axis): `base * factor` -> bit-identical.
                    None => base * factor,
                    // shifted (kerr-schild radial): lambda_coord = factor*(s - vg) - beta^r, carried
                    // per characteristic root through the abs/max (the shift breaks the multiplicative
                    // factoring). both roots < 0 at r <= 2M -> domain of dependence entirely interior.
                    Some(beta) => {
                        let ll = factor * (sl - vg) - beta;
                        let lr = factor * (sr - vg) - beta;
                        ll.abs().max(lr.abs()) * inv_w[d]
                    }
                }
            }
        };
        lambda = lambda.max(contrib);
    }
    let writes = wave_speed_map_writes(lambda.node());
    (end_trace(), writes)
}


/// the GENERIC curved-background CFL wave-speed map — the coordinate LIGHT-CONE bound per
/// gridded axis, for ANY spacetime the codegen enum carries:
/// `lambda_d = (alpha sqrt(gamma^{dd}) + |beta^d|) h_d / (h_d dx_d)` (the flat scale factor
/// `h_d` cancels against the physical inverse width, leaving the coordinate speed over the
/// coordinate width). every characteristic of ANY matter — magnetosonic waves included — lies
/// inside the coordinate light cone, so the bound is unconditionally CFL-safe and
/// state-INDEPENDENT (a pure-geometry kernel). tighter state-dependent maps (the factored
/// banyuls-font forms) stay per-regime specializations; this is the safe generic fallback the
/// GRMHD path uses. the metric evaluates at its full three coordinates: gridded slots at the
/// cell center, the ungridded polar slot at the exact equatorial pi/2, the azimuthal at zero.
pub fn gr_light_cone_wave_speed_map_gv(
    spacetime: Spacetime,
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let inv_w = cfl_inv_widths_gv(coords, spacing, axes, ndim);
    // gridded cell-center positions, spacing-aware (log = geometric mean of faces, uniform = midpoint)
    // via the shared `gv_cell_center`; ungridded slots take the exact equatorial/azimuthal constant.
    let x = Tensor::<Gv, 3>::new(std::array::from_fn(|c| {
        match axes.iter().position(|&a| a == c) {
            Some(d) => gv_cell_center(d, spacing),
            None => gv_ungridded_slot(coords, c),
        }
    }));
    let mass = Gv::scalar("schwarzschild_mass");
    // the light-cone speed alpha sqrt(gamma^{dd}) + |beta^d| at the FULL position, dispatched by
    // (spacetime, chart): the kerr-schild spacetime is expressed in spherical, CARTESIAN, or
    // CYLINDRICAL coordinates — the metric computes r = |x| (cartesian) / sqrt(R^2 + z^2)
    // (cylindrical) internally, so the wrong-chart spherical metric (which would read x[0] as the
    // radius) is never used.
    let (alpha, gi, beta) = match (spacetime, coords) {
        (Spacetime::Schwarzschild, _) => {
            let g = Schwarzschild { mass };
            (g.lapse(x), g.spatial_metric_inv(x), <Schwarzschild<Gv> as Metric<Gv, 3>>::shift(&g, x))
        }
        (Spacetime::KerrSchild, Coords::Cartesian) => {
            let g = SchwarzschildKSCartesian { mass };
            (g.lapse(x), g.spatial_metric_inv(x), <SchwarzschildKSCartesian<Gv> as Metric<Gv, 3>>::shift(&g, x))
        }
        (Spacetime::KerrSchild, Coords::Cylindrical) => {
            let g = SchwarzschildKSCylindrical { mass };
            (g.lapse(x), g.spatial_metric_inv(x), <SchwarzschildKSCylindrical<Gv> as Metric<Gv, 3>>::shift(&g, x))
        }
        (Spacetime::KerrSchild, _) => {
            let g = SchwarzschildKS { mass };
            (g.lapse(x), g.spatial_metric_inv(x), <SchwarzschildKS<Gv> as Metric<Gv, 3>>::shift(&g, x))
        }
        (Spacetime::Kerr, _) => {
            let g = KerrKS { mass, spin: Gv::scalar("kerr_spin") };
            (g.lapse(x), g.spatial_metric_inv(x), <KerrKS<Gv> as Metric<Gv, 3>>::shift(&g, x))
        }
        (Spacetime::Minkowski, _) => unreachable!("the light-cone map is baked only for a curved spacetime"),
    };
    // per gridded axis: coordinate light-cone speed times the coordinate inverse width — the
    // flat physical inv width times the flat scale factor h_d at the cell center.
    let pos: Vec<Gv> = (0..3).map(|c| x[c]).collect();
    let mut lambda = Gv::ZERO;
    for d in 0..ndim {
        let c = axes[d];
        let lam_c = alpha * gi[(c, c)].sqrt() + beta[c].abs();
        let h_flat = gv_scale_factor(coords, c, &pos);
        lambda = lambda.max(lam_c * h_flat * inv_w[d]);
    }
    let lambda = gv_state_finite_guard(lambda);
    let writes = wave_speed_map_writes(lambda.node());
    (end_trace(), writes)
}


/// the fail-loud guard for a STATE-INDEPENDENT wave-speed map: a pure-geometry lambda breaks
/// the "NaN state -> NaN wave speed -> NaN dt -> halt" backstop (a blown-up run marches to
/// t_final and writes garbage checkpoints). probe the conserved state at this cell and force
/// lambda -> +inf when it is non-finite: dt collapses to ZERO and the driver's crash guard
/// halts. the inf is built at RUNTIME as lambda/0 (an inf literal does not survive the json
/// ir); `probe - probe` is NaN for BOTH NaN and +-inf inputs; the good path divides by one,
/// bit-transparent. den + tau suffice: any physics NaN reaches them within one step's fluxes.
fn gv_state_finite_guard(lambda: Gv) -> Gv {
    let probe = Gv::field("cons_den", FieldRef::cons_den())
        + Gv::field("cons_nrg", FieldRef::cons_nrg());
    let diff = probe - probe;
    lambda / Gv::select(diff.cmp_eq(diff), Gv::ONE, Gv::ZERO)
}


/// the SPINNING-KERR CFL wave-speed map — the coordinate LIGHT-CONE bound per gridded axis:
/// `lambda_d = alpha sqrt(gamma^{dd}) + |beta^d|`, folded over axes with the in-kernel physical
/// inverse widths. every fluid characteristic lies inside the coordinate light cone, so the bound
/// is unconditionally CFL-safe; it is state-INDEPENDENT (a pure-geometry kernel — dt is set by the
/// metric alone). the exact banyuls-font per-axis speeds of the radial-only backgrounds do not
/// factor for kerr (the theta-dependent lapse and the non-diagonal gamma^{rr} break the
/// alpha^2-times-SR-speed identity), and the light cone gives up at most the O(1 - |v| - cs)
/// interior margin. static metric: no mesh-motion terms.
pub fn kerr_wave_speed_map_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    assert!(
        coords == Coords::Spherical && ndim == 2 && axes == [0, 1],
        "the kerr wave-speed map is the (r, theta) swirl instance"
    );
    begin_trace();
    let inv_w = cfl_inv_widths_gv(coords, spacing, axes, ndim);
    // spacing-aware cell centers: the radial axis may be LOG (kerr log-radial is baked), so the
    // metric (lapse, shift, gamma^{cc}) must evaluate at the geometric-mean radius, not the uniform
    // ~r_min. theta is uniform -> bit-identical to the old midpoint.
    let r = gv_cell_center(0, spacing);
    let th = gv_cell_center(1, spacing);
    let mass = Gv::scalar("schwarzschild_mass");
    let spin = Gv::scalar("kerr_spin");
    let g = KerrKS { mass, spin };
    let x = Tensor::<Gv, 3>::new([r, th, Gv::ZERO]);
    let alpha = g.lapse(x);
    let gi = g.spatial_metric_inv(x);
    let beta_r = g.shift(x)[0];
    // radial: (alpha sqrt(gamma^rr) + beta^r) / dr; the physical inv width for the flat radial
    // scale factor h_r = 1 IS the coordinate 1/dr.
    let lam_r = (alpha * gi[(0, 0)].sqrt() + beta_r) * inv_w[0];
    // polar: alpha sqrt(gamma^{theta theta}) = alpha/sqrt(Sigma) over the coordinate dtheta; the
    // physical inv width carries the flat h_theta = r, so multiply the ratio r/sqrt(Sigma) back.
    let lam_t = alpha * gi[(1, 1)].sqrt() * r * inv_w[1];
    let lambda = gv_state_finite_guard(lam_r.max(lam_t));
    let writes = wave_speed_map_writes(lambda.node());
    (end_trace(), writes)
}


/// the Newtonian / isothermal CFL wave-speed map (gamma->1 drives isothermal, 1.4 adiabatic) —
/// `Newtonian::wave_speeds_axis` (`|v_d| + cs`) traced to the complete timestep kernel.
pub fn iso_wave_speed_map_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    // newtonian / isothermal are non-relativistic -> always flat spacetime.
    euler_wave_speed_map_gv(&Newtonian, coords, Spacetime::Minkowski, spacing, axes, ndim)
}


/// the RHD CFL wave-speed map — the relativistic Mignone-Bodo per-axis speed (`Rhd::
/// wave_speeds_axis`, the SAME core the RHD flux's HLLE consumes) traced to the timestep kernel.
pub fn rhd_wave_speed_map_gv(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    euler_wave_speed_map_gv(&Rhd, coords, spacetime, spacing, axes, ndim)
}


/// trace the RMHD CFL wave-speed map at `S = Gv` — the MAGNETOSONIC UPPER BOUND
/// (`rmhd_magnetosonic_cfl_speeds`), NOT the full Mignone & Del Zanna quartic. the CFL needs
/// only a stable upper bound on the signal speed, and the bound is ~25x cheaper than the
/// quartic (~30 ops + 1 sqrt vs ~750 ops + ~10 transcendentals, ALL of which trace into the
/// kernel because `S::select` evaluates every arm). the quartic stays on the Riemann/flux
/// path (`rmhd_flux_gv` -> extremal_speeds), where HLLE diffusion needs the tight estimate.
/// see docs/c9fbdcb_perf_study/02. reads the full 3-velocity + 3-magnetic-field (vsq/bsq).
/// RMHD is fixed 3D (identity axes); the folded geometry + max-reduction is the same single-
/// trace form as the Euler map.
pub fn rmhd_wave_speed_map_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let rho = Gv::field("prim_rho", FieldRef::PrimRho);
    let vel: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8)));
    let pre = Gv::field("prim_pre", FieldRef::PrimPre);
    let mag: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_b{k}"), FieldRef::PrimMag(k as u8)));
    let gamma = Gv::scalar("gamma");
    let eos = IdealGas { gamma };
    let prim = MhdPrim::<Gv, 3> {
        hydro: Prim { rho, vel: Tensor::new(vel), pre },
        mag: Tensor::new(mag),
    };
    let inv_w = cfl_inv_widths_gv(coords, spacing, axes, ndim);
    let mut lambda = Gv::ZERO;
    for d in 0..ndim {
        let nhat = Tensor::<Gv, 3>::unit(axes[d]);
        let (sl, sr) = rmhd_magnetosonic_cfl_speeds(&eos, &prim, &nhat);
        lambda = lambda.max(sl.abs().max(sr.abs()) * inv_w[d]);
    }
    let writes = wave_speed_map_writes(lambda.node());
    (end_trace(), writes)
}


/// trace the PER-CELL RMHD wave speeds — the EXACT Mignone & Del Zanna quartic
/// (`Rmhd::wave_speeds`, raw min/max, NO zero-clamp) evaluated once per cell for each of the
/// 3 directions, writing `wave_speed_l[d] = lambda_min^d` and `wave_speed_r[d] = lambda_max^d`.
///
/// this lifts the wave speed OFF the per-face flux (where it was recomputed for L and R at
/// every face — the dominant 166-register, 12-transcendental cost) onto the cell index space:
/// the quartic's direction-INDEPENDENT guts (rho/h/c_s^2/b_mu^2/...) are CSE-shared across the
/// 3 directions; only vn/bn and the resolvent differ. the flux then reads the two adjacent
/// cells' stored speeds for the Davis fan (`hlle_with_speeds`), and CFL folds the same fields
/// — one computation, three consumers (flux, CFL, UCT-HLL). the zero-clamp for the HLL fan is
/// applied at the FLUX (min/max of the two cells, clamped), so the stored values are raw.
/// the CURVED-SPACETIME per-cell RMHD wave speeds — the SHIFTED coordinate characteristic
/// speeds `lambda_pm = (RmhdGr fast-magnetosonic BF speed) - beta^d` per grid direction,
/// materialized into wave_speed_l/r for the GR-UCT edge coefficients. UNLIKE the flat kernel
/// (the exact magnetosonic quartic) these are the algebraic fast bound `c_ms^2 = c_s^2 + v_A^2
/// - c_s^2 v_A^2` through the two-velocity BF transform — cheaper, and the ONLY consumer is the
/// UCT edge EMF (the GR flux computes its own inline). the shift makes them the induction
/// system's coordinate speeds, consistent with the transport velocity vtilde = alpha v - beta
/// the edge advection uses. metric at the cell centroid (the c2p point), ungridded polar slot
/// at pi/2. baked per (spacetime, spacing).
pub fn rmhd_wave_speeds_cell_gr_gv(
    spacetime: Spacetime,
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let ndim = axes.len();
    let rho = Gv::field("prim_rho", FieldRef::PrimRho);
    let vel: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8)));
    let pre = Gv::field("prim_pre", FieldRef::PrimPre);
    let mag: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_b{k}"), FieldRef::PrimMag(k as u8)));
    let gamma = Gv::scalar("gamma");
    let eos = IdealGas { gamma };
    let prim = MhdPrim::<Gv, 3> {
        hydro: Prim { rho, vel: Tensor::new(vel), pre },
        mag: Tensor::new(mag),
    };
    let geo = cell_geometry_gv(coords, spacing, axes, ndim);
    let x = Tensor::<Gv, 3>::new(std::array::from_fn(|c| {
        match axes.iter().position(|&a| a == c) {
            Some(d) => geo.centroid[d],
            None => gv_ungridded_slot(coords, c),
        }
    }));
    let mass = Gv::scalar("schwarzschild_mass");
    let (gm, gm_inv, alpha, beta) = match (spacetime, coords) {
        (Spacetime::Schwarzschild, _) => {
            let m = Schwarzschild { mass };
            (m.spatial_metric(x), m.spatial_metric_inv(x), m.lapse(x), <Schwarzschild<Gv> as Metric<Gv, 3>>::shift(&m, x))
        }
        (Spacetime::KerrSchild, Coords::Cartesian) => {
            let m = SchwarzschildKSCartesian { mass };
            (m.spatial_metric(x), m.spatial_metric_inv(x), m.lapse(x), <SchwarzschildKSCartesian<Gv> as Metric<Gv, 3>>::shift(&m, x))
        }
        (Spacetime::KerrSchild, Coords::Cylindrical) => {
            let m = SchwarzschildKSCylindrical { mass };
            (m.spatial_metric(x), m.spatial_metric_inv(x), m.lapse(x), <SchwarzschildKSCylindrical<Gv> as Metric<Gv, 3>>::shift(&m, x))
        }
        (Spacetime::KerrSchild, _) => {
            let m = SchwarzschildKS { mass };
            (m.spatial_metric(x), m.spatial_metric_inv(x), m.lapse(x), <SchwarzschildKS<Gv> as Metric<Gv, 3>>::shift(&m, x))
        }
        (Spacetime::Kerr, _) => {
            let m = KerrKS { mass, spin: Gv::scalar("kerr_spin") };
            (m.spatial_metric(x), m.spatial_metric_inv(x), m.lapse(x), <KerrKS<Gv> as Metric<Gv, 3>>::shift(&m, x))
        }
        (Spacetime::Minkowski, _) => unreachable!("the GR cell wave speeds are baked only for a curved spacetime"),
    };
    let regime = RmhdGr { metric: SpatialMetric::new(Gamma::new(gm), GammaInv::new(gm_inv)), alpha };
    let mut writes = Vec::with_capacity(2 * ndim);
    for d in 0..ndim {
        let nhat = Tensor::<Gv, 3>::unit(axes[d]);
        // the unshifted BF fast-bound speeds, then the shift for this coordinate direction.
        let (sl, sr) = regime.wave_speeds(&eos, &prim, &nhat);
        let bd = beta[axes[d]];
        writes.push((format!("ws_l_{d}"), format!("wave_speed_l[{d}]").into(), (sl - bd).node()));
        writes.push((format!("ws_r_{d}"), format!("wave_speed_r[{d}]").into(), (sr - bd).node()));
    }
    (end_trace(), writes)
}


/// the SOURCE-admissibility CFL limit for GR-RMHD — the wu 2017 (arXiv:1708.07267) lambda_S
/// mechanism. the geometric (gravity + covariant centrifugal + EM-tension) source
/// S = (S_mom, S_tau) advances the conserved state U -> U + dt S; for the light-cone LxF flux to
/// stay physical-constraint-preserving the timestep must ALSO keep U + dt S inside the admissible
/// cone q(U) = E - sqrt(D^2 + gamma^{ij} S_i S_j) >= 0, D > 0 (E = tau + D). q is concave along the
/// source ray and lipschitz-bounded below: q(U + dt S) >= q(U) - dt (|S_tau| + ||S_mom||_gamma),
/// so dt (|S_tau| + ||S_mom||_gamma)/q(U) <= 1 is SUFFICIENT for admissibility. this gives the
/// source characteristic rate lambda_S = (|S_tau| + ||S_mom||_gamma)/q(U), added into the CFL
/// scratch alongside the flux light-cone rate (wu eq 22: dt (lambda_flux + lambda_S) < 1).
/// lambda_S -> inf as q -> 0, collapsing dt at the cone boundary — the behavior that holds the
/// magnetized-torus cusp admissible with no floor. conservative: the lipschitz slope underestimates
/// the exact concave-quadratic root, so dt is at most that root. reads the flux rate already in the
/// scratch and writes their sum in place; the momentum source takes p = 0 and the energy source
/// the full p, matching the fused godunov source that actually advances U. metric at the cell
/// centroid, ungridded polar slot at pi/2.
pub fn rmhd_source_cfl_gr_gv(
    spacetime: Spacetime,
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    mag_from_bcell: bool,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let ndim = axes.len();
    let rho = Gv::field("prim_rho", FieldRef::PrimRho);
    let vel: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8)));
    let pre = Gv::field("prim_pre", FieldRef::PrimPre);
    let v = Tensor::<Gv, 3>::new(vel);
    let geo = cell_geometry_gv(coords, spacing, axes, ndim);
    let x = Tensor::<Gv, 3>::new(std::array::from_fn(|c| {
        match axes.iter().position(|&a| a == c) {
            Some(d) => geo.centroid[d],
            None => gv_ungridded_slot(coords, c),
        }
    }));
    // the metric-free rest enthalpy density rho_h = rho + Gamma/(Gamma-1) p (the source builds W
    // and b^mu from the harvested gamma internally); the cell magnetic field under the same key
    // convention as the fused godunov source.
    let gamma_eos = Gv::scalar("gamma");
    let rho_h = rho + gamma_eos / (gamma_eos - Gv::ONE) * pre;
    let b = Tensor::<Gv, 3>::new(std::array::from_fn(|k| {
        if mag_from_bcell {
            Gv::field(&format!("bc_{k}"), FieldRef::BCell(k as u8))
        } else {
            Gv::field(&format!("prim_b{k}"), FieldRef::PrimMag(k as u8))
        }
    }));
    let mass_gv = Gv::scalar("schwarzschild_mass");
    let mass = Dual::constant(mass_gv);
    // the inverse spatial metric (raises the covariant momentum for the admissible-cone norm) and
    // the covariant source (autodiff Dual metric), dispatched by (spacetime, chart). the momentum
    // source at p = 0, the energy source at the full p.
    let (gm_inv, s_mom, s_tau): (Matrix<Gv, 3>, Tensor<Gv, 3>, Gv) = match (spacetime, coords) {
        (Spacetime::Schwarzschild, _) => {
            let gi = Schwarzschild { mass: mass_gv }.spatial_metric_inv(x);
            let (sm, _) = grmhd_covariant_source(&Schwarzschild { mass }, x, rho_h, v, Gv::ZERO, b);
            let (_, st) = grmhd_covariant_source(&Schwarzschild { mass }, x, rho_h, v, pre, b);
            (gi, sm, st)
        }
        (Spacetime::KerrSchild, Coords::Cartesian) => {
            let gi = SchwarzschildKSCartesian { mass: mass_gv }.spatial_metric_inv(x);
            let (sm, _) = grmhd_covariant_source(&SchwarzschildKSCartesian { mass }, x, rho_h, v, Gv::ZERO, b);
            let (_, st) = grmhd_covariant_source(&SchwarzschildKSCartesian { mass }, x, rho_h, v, pre, b);
            (gi, sm, st)
        }
        (Spacetime::KerrSchild, Coords::Cylindrical) => {
            let gi = SchwarzschildKSCylindrical { mass: mass_gv }.spatial_metric_inv(x);
            let (sm, _) = grmhd_covariant_source(&SchwarzschildKSCylindrical { mass }, x, rho_h, v, Gv::ZERO, b);
            let (_, st) = grmhd_covariant_source(&SchwarzschildKSCylindrical { mass }, x, rho_h, v, pre, b);
            (gi, sm, st)
        }
        (Spacetime::KerrSchild, _) => {
            let gi = SchwarzschildKS { mass: mass_gv }.spatial_metric_inv(x);
            let (sm, _) = grmhd_covariant_source(&SchwarzschildKS { mass }, x, rho_h, v, Gv::ZERO, b);
            let (_, st) = grmhd_covariant_source(&SchwarzschildKS { mass }, x, rho_h, v, pre, b);
            (gi, sm, st)
        }
        (Spacetime::Kerr, _) => {
            let spin_gv = Gv::scalar("kerr_spin");
            let gi = KerrKS { mass: mass_gv, spin: spin_gv }.spatial_metric_inv(x);
            let spin = Dual::constant(spin_gv);
            let (sm, _) = grmhd_covariant_source(&KerrKS { mass, spin }, x, rho_h, v, Gv::ZERO, b);
            let (_, st) = grmhd_covariant_source(&KerrKS { mass, spin }, x, rho_h, v, pre, b);
            (gi, sm, st)
        }
        (Spacetime::Minkowski, _) => {
            unreachable!("the source-admissibility CFL is baked only for a curved spacetime")
        }
    };
    // the admissible cone at the current cell: E = tau + D, |S|^2 = gamma^{ij} S_i S_j.
    let d_cons = Gv::field("cons_den", FieldRef::cons_den());
    let mom: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8)));
    let tau = Gv::field("cons_nrg", FieldRef::cons_nrg());
    let e_cons = tau + d_cons;
    let gamma_norm = |a: &[Gv; 3]| {
        let mut acc = Gv::ZERO;
        for ii in 0..3 {
            for jj in 0..3 {
                acc = acc + gm_inv[(ii, jj)] * a[ii] * a[jj];
            }
        }
        acc
    };
    let q0 = e_cons - (d_cons * d_cons + gamma_norm(&mom)).sqrt();
    let sm_arr: [Gv; 3] = std::array::from_fn(|k| s_mom[k]);
    let sm_norm = gamma_norm(&sm_arr).sqrt();
    // lambda_S = (|S_tau| + ||S_mom||_gamma) / q; the max keeps a near-boundary cell driving
    // dt -> 0 through a positive denominator (a c2p-physical cell has q > 0).
    let q_safe = q0.max(Gv::from_f64(1e-12));
    let lam_s = (s_tau.abs() + sm_norm) / q_safe;
    let lam_flux = Gv::field("lambda", FieldRef::Scratch);
    let total = lam_flux + lam_s;
    (end_trace(), vec![("lambda".to_string(), FieldRef::Scratch.into(), total.node())])
}


/// the SOURCE-admissibility CFL for GR-HYDRO — the wu 2017 lambda_S mechanism, the perfect-fluid
/// analogue of [`rmhd_source_cfl_gr_gv`] (no magnetic field). the covariant geodesic source
/// S = (S_mom, S_tau) advances U -> U + dt S; the timestep must keep U + dt S inside the admissible
/// cone q(U) = E - sqrt(D^2 + gamma^{ij} S_i S_j) >= 0 (E = tau + D). lambda_S = (|S_tau| +
/// ||S_mom||_gamma)/q(U), added into the CFL scratch alongside the flux light-cone rate. `ncomp` is
/// the momentum DOF (1..3): momentum/velocity slots >= ncomp carry zero, so the metric-3D
/// contraction reduces to the actual gridded momenta. the energy input to the source is
/// e = rho + tau + p (the total energy density), matching the fused godunov hydro source.
pub fn rhd_source_cfl_gr_gv(
    spacetime: Spacetime,
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ncomp: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let ndim = axes.len();
    let rho = Gv::field("prim_rho", FieldRef::PrimRho);
    let vel: [Gv; 3] = std::array::from_fn(|k| {
        if k < ncomp {
            Gv::field(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8))
        } else {
            Gv::ZERO
        }
    });
    let pre = Gv::field("prim_pre", FieldRef::PrimPre);
    let v = Tensor::<Gv, 3>::new(vel);
    let geo = cell_geometry_gv(coords, spacing, axes, ndim);
    let x = Tensor::<Gv, 3>::new(std::array::from_fn(|c| {
        match axes.iter().position(|&a| a == c) {
            Some(d) => geo.centroid[d],
            None => gv_ungridded_slot(coords, c),
        }
    }));
    let d_cons = Gv::field("cons_den", FieldRef::cons_den());
    let tau = Gv::field("cons_nrg", FieldRef::cons_nrg());
    // the total energy density the covariant hydro source contracts (rest + internal + pressure).
    let e = rho + tau + pre;
    let mass_gv = Gv::scalar("schwarzschild_mass");
    let mass = Dual::constant(mass_gv);
    let (gm_inv, s_mom, s_tau): (Matrix<Gv, 3>, Tensor<Gv, 3>, Gv) = match (spacetime, coords) {
        (Spacetime::Schwarzschild, _) => {
            let gi = Schwarzschild { mass: mass_gv }.spatial_metric_inv(x);
            let (sm, _) = grhd_covariant_source(&Schwarzschild { mass }, x, e, v, Gv::ZERO);
            let (_, st) = grhd_covariant_source(&Schwarzschild { mass }, x, e, v, pre);
            (gi, sm, st)
        }
        (Spacetime::KerrSchild, Coords::Cartesian) => {
            let gi = SchwarzschildKSCartesian { mass: mass_gv }.spatial_metric_inv(x);
            let (sm, _) = grhd_covariant_source(&SchwarzschildKSCartesian { mass }, x, e, v, Gv::ZERO);
            let (_, st) = grhd_covariant_source(&SchwarzschildKSCartesian { mass }, x, e, v, pre);
            (gi, sm, st)
        }
        (Spacetime::KerrSchild, Coords::Cylindrical) => {
            let gi = SchwarzschildKSCylindrical { mass: mass_gv }.spatial_metric_inv(x);
            let (sm, _) = grhd_covariant_source(&SchwarzschildKSCylindrical { mass }, x, e, v, Gv::ZERO);
            let (_, st) = grhd_covariant_source(&SchwarzschildKSCylindrical { mass }, x, e, v, pre);
            (gi, sm, st)
        }
        (Spacetime::KerrSchild, _) => {
            let gi = SchwarzschildKS { mass: mass_gv }.spatial_metric_inv(x);
            let (sm, _) = grhd_covariant_source(&SchwarzschildKS { mass }, x, e, v, Gv::ZERO);
            let (_, st) = grhd_covariant_source(&SchwarzschildKS { mass }, x, e, v, pre);
            (gi, sm, st)
        }
        (Spacetime::Kerr, _) => {
            let spin_gv = Gv::scalar("kerr_spin");
            let gi = KerrKS { mass: mass_gv, spin: spin_gv }.spatial_metric_inv(x);
            let spin = Dual::constant(spin_gv);
            let (sm, _) = grhd_covariant_source(&KerrKS { mass, spin }, x, e, v, Gv::ZERO);
            let (_, st) = grhd_covariant_source(&KerrKS { mass, spin }, x, e, v, pre);
            (gi, sm, st)
        }
        (Spacetime::Minkowski, _) => {
            unreachable!("the source-admissibility CFL is baked only for a curved spacetime")
        }
    };
    let mom: [Gv; 3] = std::array::from_fn(|k| {
        if k < ncomp {
            Gv::field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8))
        } else {
            Gv::ZERO
        }
    });
    let e_cons = tau + d_cons;
    let gamma_norm = |a: &[Gv; 3]| {
        let mut acc = Gv::ZERO;
        for ii in 0..3 {
            for jj in 0..3 {
                acc = acc + gm_inv[(ii, jj)] * a[ii] * a[jj];
            }
        }
        acc
    };
    let q0 = e_cons - (d_cons * d_cons + gamma_norm(&mom)).sqrt();
    let sm_arr: [Gv; 3] = std::array::from_fn(|k| s_mom[k]);
    let sm_norm = gamma_norm(&sm_arr).sqrt();
    let q_safe = q0.max(Gv::from_f64(1e-12));
    let lam_s = (s_tau.abs() + sm_norm) / q_safe;
    let lam_flux = Gv::field("lambda", FieldRef::Scratch);
    let total = lam_flux + lam_s;
    (end_trace(), vec![("lambda".to_string(), FieldRef::Scratch.into(), total.node())])
}


/// RMHD is fixed 3D. reads the full 3-velocity + 3-magnetic-field prim + gamma.
pub fn rmhd_wave_speeds_cell_gv(ndim: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let rho = Gv::field("prim_rho", FieldRef::PrimRho);
    let vel: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8)));
    let pre = Gv::field("prim_pre", FieldRef::PrimPre);
    let mag: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_b{k}"), FieldRef::PrimMag(k as u8)));
    let gamma = Gv::scalar("gamma");
    let eos = IdealGas { gamma };
    let prim = MhdPrim::<Gv, 3> {
        hydro: Prim { rho, vel: Tensor::new(vel), pre },
        mag: Tensor::new(mag),
    };
    // one (lmin,lmax) pair per spatial sweep direction (ndim of them); the B is always
    // a 3-vector but the grid varies along ndim axes only.
    let mut writes = Vec::with_capacity(2 * ndim);
    for d in 0..ndim {
        let nhat = Tensor::<Gv, 3>::unit(d);
        // raw quartic min/max — NOT extremal_speeds (no zero-clamp); the flux clamps the fan.
        let (lmin, lmax) = Rmhd.wave_speeds(&eos, &prim, &nhat);
        writes.push((format!("ws_l_{d}"), format!("wave_speed_l[{d}]").into(), lmin.node()));
        writes.push((format!("ws_r_{d}"), format!("wave_speed_r[{d}]").into(), lmax.node()));
    }
    (end_trace(), writes)
}


/// the CLASSICAL (Newtonian ideal-gas) per-cell wave speeds (`NewtonianMhd::wave_speeds` = the fast
/// magnetosonic bound, lmin/lmax = v_n -/+ c_f). materializes `wave_speed_l/r[dir]` so UCT (which
/// reads them for the edge-EMF coefficients) works for NMHD — the classical regimes compute speeds
/// inline in the flux and otherwise do NOT store them. mirror of `rmhd_wave_speeds_cell_gv`.
pub fn nmhd_wave_speeds_cell_gv(ndim: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let rho = Gv::field("prim_rho", FieldRef::PrimRho);
    let vel: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8)));
    let pre = Gv::field("prim_pre", FieldRef::PrimPre);
    let mag: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_b{k}"), FieldRef::PrimMag(k as u8)));
    let gamma = Gv::scalar("gamma");
    let eos = IdealGas { gamma };
    let prim = MhdPrim::<Gv, 3> {
        hydro: Prim { rho, vel: Tensor::new(vel), pre },
        mag: Tensor::new(mag),
    };
    let mut writes = Vec::with_capacity(2 * ndim);
    for d in 0..ndim {
        let nhat = Tensor::<Gv, 3>::unit(d);
        let (lmin, lmax) = NewtonianMhd.wave_speeds(&eos, &prim, &nhat);
        writes.push((format!("ws_l_{d}"), format!("wave_speed_l[{d}]").into(), lmin.node()));
        writes.push((format!("ws_r_{d}"), format!("wave_speed_r[{d}]").into(), lmax.node()));
    }
    (end_trace(), writes)
}


/// isothermal-MHD per-cell wave speeds materialized for the UCT edge-EMF (`wave_speed_l/r[d]`).
/// mirror of `nmhd_wave_speeds_cell_gv` with `IsothermalMhd::wave_speeds` (fast magnetosonic at
/// a^2 = cs^2, NO pressure). lets isothermal MHD run UCT (the regime-generic HLL edge-EMF reads
/// these speeds); without it `--ct-method uct` silently falls back to Contact.
pub fn imhd_wave_speeds_cell_gv(ndim: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let rho = Gv::field("prim_rho", FieldRef::PrimRho);
    let vel: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8)));
    let mag: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_b{k}"), FieldRef::PrimMag(k as u8)));
    let cs = Gv::scalar("cs");
    let eos = Isothermal { cs };
    let prim = IsoMhdPrim::<Gv, 3> {
        hydro: PrimG { rho, vel: Tensor::new(vel), pre: Zero::default() },
        mag: Tensor::new(mag),
    };
    let mut writes = Vec::with_capacity(2 * ndim);
    for d in 0..ndim {
        let nhat = Tensor::<Gv, 3>::unit(d);
        let (lmin, lmax) = IsothermalMhd.wave_speeds(&eos, &prim, &nhat);
        writes.push((format!("ws_l_{d}"), format!("wave_speed_l[{d}]").into(), lmin.node()));
        writes.push((format!("ws_r_{d}"), format!("wave_speed_r[{d}]").into(), lmax.node()));
    }
    (end_trace(), writes)
}


#[cfg(test)]
mod m1_log_radius_tests {
    // M1 regression: the GR wave-speed maps must evaluate the metric at the SPACING-AWARE cell
    // center. on a log axis that is the geometric mean of the bounding faces (r_min * 10^((i+1/2)
    // slope)); the old uniform `x_lo + (i+1/2) dx` formula put every cell at ~r_min, suppressing
    // alpha^2 / beta^r / the scale factors and overestimating dt (a silent CFL violation).
    use super::*;
    use symbi_ir::gv::{begin_trace, end_trace, with_trace};
    use symbi_ir::graph::NodeId;

    fn eval(out: NodeId, values: &[(&str, f64)]) -> f64 {
        use symbi_ir::backends::interp::{Backend, Cpu};
        use symbi_ir::passes::scalarize::{scalarize_kernel, LoweredFn};
        // the runtime spacing map emits `map_kind`'s cond as an `Op::IfElse`, which the single-output
        // `scalarize` cannot lower (only `scalarize_kernel` handles it). lower the one output through
        // `scalarize_kernel` and wrap it as a `LoweredFn` for the elemental interpreter.
        let lowered = with_trace(|t| {
            let sc = scalarize_kernel(t.graph(), &[out]);
            let ty = t.graph().ty(out).clone();
            LoweredFn {
                name: "m1_center".to_string(),
                params: sc.params,
                body: sc.body,
                results: vec![sc.outputs[0].clone()],
                result_element: ty.element,
                result_shape: ty.shape,
            }
        });
        let inputs: Vec<f64> = lowered
            .params
            .iter()
            .map(|p| {
                values
                    .iter()
                    .find(|(n, _)| *n == p.name.as_str())
                    .map(|(_, v)| *v)
                    // an unbound spacing selector defaults to uniform (map_kind = 0).
                    .or_else(|| p.name.starts_with("map_kind_").then_some(0.0))
                    .unwrap_or_else(|| panic!("eval: missing param '{}'", p.name))
            })
            .collect();
        Cpu.eval_elemental(&lowered, &inputs)[0]
    }

    #[test]
    fn cell_center_is_geometric_mean_on_log_axis() {
        begin_trace();
        let node = gv_cell_center(0, &[Spacing::Log]).node();
        let (r_min, slope, i) = (3.0_f64, 0.02_f64, 10.0_f64);
        // spacing is now a runtime scalar: select the log map (map_kind_0 = 1); the `Spacing::Log`
        // builder arg is vestigial (the face map reads `map_kind`, not the codegen enum).
        let got = eval(node, &[("x_lo_0", r_min), ("dx_0", slope), ("_coord_0", i), ("map_kind_0", 1.0)]);
        end_trace();
        let geomean = r_min * 10f64.powf((i + 0.5) * slope); // sqrt(face_i * face_{i+1})
        let old_uniform_bug = r_min + (i + 0.5) * slope;
        assert!(
            (got - geomean).abs() < 1e-10,
            "log cell center {got} != geometric mean {geomean} (metric radius wrong on log grid)"
        );
        assert!(
            (got - old_uniform_bug).abs() > 1e-3,
            "M1 fix is a no-op: still the uniform x_lo + (i+1/2) dx radius"
        );
    }

    #[test]
    fn cell_center_is_bit_identical_on_uniform_axis() {
        begin_trace();
        let node = gv_cell_center(0, &[Spacing::Uniform]).node();
        let (x_lo, dx, i) = (2.0_f64, 0.5_f64, 7.0_f64);
        let got = eval(node, &[("x_lo_0", x_lo), ("dx_0", dx), ("_coord_0", i)]);
        end_trace();
        assert_eq!(got, x_lo + (i + 0.5) * dx, "uniform center must equal the old formula bit-for-bit");
    }
}
