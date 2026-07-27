// =============================================================================
// wavespeed.rs
//
// cfl wave-speed map + per-cell wave-speed kernel builders (the geometry-folded characteristic speed).
// =============================================================================

use super::*;
use symbi_algebra::Matrix;
use symbi_geometry::grhd_source::{grhd_covariant_source, grmhd_covariant_source};
use symbi_geometry::{
    KerrKS, KerrKSCartesian, KerrKSCylindrical, Metric, Schwarzschild, SchwarzschildKS,
    SchwarzschildKSCartesian, SchwarzschildKSCylindrical,
};
use symbi_ir::dual::Dual;

/// trace the newtonian-MHD CFL wave-speed map — `NewtonianMhd::wave_speeds` (the
/// EXACT closed-form fast magnetosonic; it is already cheap) folded
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
        hydro: Prim {
            rho,
            vel: Tensor::new(vel),
            pre,
        },
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
        hydro: PrimG {
            rho,
            vel: Tensor::new(vel),
            pre: Zero::default(),
        },
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
        (0..ndim)
            .map(|d| Gv::scalar(&format!("inv_dx_{d}")))
            .collect()
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
        Spacing::Uniform => {
            gv_axis_face_at(d, spacing[d], 0) + half * Gv::scalar(&format!("dx_{d}"))
        }
        Spacing::Log => {
            (gv_axis_face_at(d, spacing[d], 0) * gv_axis_face_at(d, spacing[d], 1)).sqrt()
        }
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
    // scale factors at the geometric-mean radius). `None` if `target` is not a grid axis.
    let coord_at = |target: usize| -> Option<Gv> {
        axes.iter()
            .position(|&c| c == target)
            .map(|d| gv_cell_center(d, spacing))
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
                2 => {
                    raw * r
                        * coord_at(1)
                            .expect("phi scale factor needs a polar axis")
                            .sin()
                }
                _ => raw,
            }
        };
    }
    let pre = Gv::field("prim_pre", FieldRef::PrimPre);
    let gamma = Gv::scalar("gamma");
    let eos = IdealGas { gamma };
    let prim = Prim::<Gv, 3> {
        rho,
        vel: Tensor::new(vel),
        pre,
    };
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
    // single ADM seam).
    let gr_radius: Option<Gv> = match spacetime {
        Spacetime::Minkowski => None,
        _ => {
            let d_r = axes
                .iter()
                .position(|&c| c == 0)
                .expect("GR wave-speed map needs a radial axis");
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
                match if is_radial {
                    gv_metric_shift_r_at(spacetime, r, None)
                } else {
                    None
                } {
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
            (
                g.lapse(x),
                g.spatial_metric_inv(x),
                <Schwarzschild<Gv> as Metric<Gv, 3>>::shift(&g, x),
            )
        }
        (Spacetime::SchwarzschildKS, Coords::Cartesian) => {
            let g = SchwarzschildKSCartesian { mass };
            (
                g.lapse(x),
                g.spatial_metric_inv(x),
                <SchwarzschildKSCartesian<Gv> as Metric<Gv, 3>>::shift(&g, x),
            )
        }
        (Spacetime::SchwarzschildKS, Coords::Cylindrical) => {
            let g = SchwarzschildKSCylindrical { mass };
            (
                g.lapse(x),
                g.spatial_metric_inv(x),
                <SchwarzschildKSCylindrical<Gv> as Metric<Gv, 3>>::shift(&g, x),
            )
        }
        (Spacetime::SchwarzschildKS, _) => {
            let g = SchwarzschildKS { mass };
            (
                g.lapse(x),
                g.spatial_metric_inv(x),
                <SchwarzschildKS<Gv> as Metric<Gv, 3>>::shift(&g, x),
            )
        }
        // spinning kerr on the CARTESIAN chart: the rank-1 kerr-schild update with the
        // oblate-spheroidal radius; non-diagonal gamma + shift on every axis.
        (Spacetime::KerrKS, Coords::Cartesian) => {
            let g = KerrKSCartesian {
                mass,
                spin: Gv::scalar("kerr_spin"),
            };
            (
                g.lapse(x),
                g.spatial_metric_inv(x),
                <KerrKSCartesian<Gv> as Metric<Gv, 3>>::shift(&g, x),
            )
        }
        (Spacetime::KerrKS, Coords::Cylindrical) => {
            let g = KerrKSCylindrical {
                mass,
                spin: Gv::scalar("kerr_spin"),
            };
            (
                g.lapse(x),
                g.spatial_metric_inv(x),
                <KerrKSCylindrical<Gv> as Metric<Gv, 3>>::shift(&g, x),
            )
        }
        (Spacetime::KerrKS, _) => {
            let g = KerrKS {
                mass,
                spin: Gv::scalar("kerr_spin"),
            };
            (
                g.lapse(x),
                g.spatial_metric_inv(x),
                <KerrKS<Gv> as Metric<Gv, 3>>::shift(&g, x),
            )
        }
        (Spacetime::Minkowski, _) => {
            unreachable!("the light-cone map is baked only for a curved spacetime")
        }
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
    let probe =
        Gv::field("cons_den", FieldRef::cons_den()) + Gv::field("cons_nrg", FieldRef::cons_nrg());
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
    // metric (lapse, shift, gamma^{cc}) must evaluate at the geometric-mean radius. theta is
    // uniform -> bit-identical to the old midpoint.
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
    euler_wave_speed_map_gv(
        &Newtonian,
        coords,
        Spacetime::Minkowski,
        spacing,
        axes,
        ndim,
    )
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
/// (`rmhd_magnetosonic_cfl_speeds`). the CFL needs
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
        hydro: Prim {
            rho,
            vel: Tensor::new(vel),
            pre,
        },
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
        hydro: Prim {
            rho,
            vel: Tensor::new(vel),
            pre,
        },
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
            (
                m.spatial_metric(x),
                m.spatial_metric_inv(x),
                m.lapse(x),
                <Schwarzschild<Gv> as Metric<Gv, 3>>::shift(&m, x),
            )
        }
        (Spacetime::SchwarzschildKS, Coords::Cartesian) => {
            let m = SchwarzschildKSCartesian { mass };
            (
                m.spatial_metric(x),
                m.spatial_metric_inv(x),
                m.lapse(x),
                <SchwarzschildKSCartesian<Gv> as Metric<Gv, 3>>::shift(&m, x),
            )
        }
        (Spacetime::SchwarzschildKS, Coords::Cylindrical) => {
            let m = SchwarzschildKSCylindrical { mass };
            (
                m.spatial_metric(x),
                m.spatial_metric_inv(x),
                m.lapse(x),
                <SchwarzschildKSCylindrical<Gv> as Metric<Gv, 3>>::shift(&m, x),
            )
        }
        (Spacetime::SchwarzschildKS, _) => {
            let m = SchwarzschildKS { mass };
            (
                m.spatial_metric(x),
                m.spatial_metric_inv(x),
                m.lapse(x),
                <SchwarzschildKS<Gv> as Metric<Gv, 3>>::shift(&m, x),
            )
        }
        // spinning kerr on the CARTESIAN chart: the rank-1 kerr-schild update with the
        // oblate-spheroidal radius; non-diagonal gamma + shift on every axis.
        (Spacetime::KerrKS, Coords::Cartesian) => {
            let m = KerrKSCartesian {
                mass,
                spin: Gv::scalar("kerr_spin"),
            };
            (
                m.spatial_metric(x),
                m.spatial_metric_inv(x),
                m.lapse(x),
                <KerrKSCartesian<Gv> as Metric<Gv, 3>>::shift(&m, x),
            )
        }
        (Spacetime::KerrKS, Coords::Cylindrical) => {
            let m = KerrKSCylindrical {
                mass,
                spin: Gv::scalar("kerr_spin"),
            };
            (
                m.spatial_metric(x),
                m.spatial_metric_inv(x),
                m.lapse(x),
                <KerrKSCylindrical<Gv> as Metric<Gv, 3>>::shift(&m, x),
            )
        }
        (Spacetime::KerrKS, _) => {
            let m = KerrKS {
                mass,
                spin: Gv::scalar("kerr_spin"),
            };
            (
                m.spatial_metric(x),
                m.spatial_metric_inv(x),
                m.lapse(x),
                <KerrKS<Gv> as Metric<Gv, 3>>::shift(&m, x),
            )
        }
        (Spacetime::Minkowski, _) => {
            unreachable!("the GR cell wave speeds are baked only for a curved spacetime")
        }
    };
    let regime = RmhdGr {
        metric: SpatialMetric::new(Gamma::new(gm), GammaInv::new(gm_inv)),
        alpha,
    };
    let mut writes = Vec::with_capacity(2 * ndim);
    for d in 0..ndim {
        let nhat = Tensor::<Gv, 3>::unit(axes[d]);
        // the unshifted BF fast-bound speeds, then the shift for this coordinate direction.
        let (sl, sr) = regime.wave_speeds(&eos, &prim, &nhat);
        let bd = beta[axes[d]];
        writes.push((
            format!("ws_l_{d}"),
            format!("wave_speed_l[{d}]").into(),
            (sl - bd).node(),
        ));
        writes.push((
            format!("ws_r_{d}"),
            format!("wave_speed_r[{d}]").into(),
            (sr - bd).node(),
        ));
    }
    (end_trace(), writes)
}

/// the source-admissibility CFL limit for GR-RMHD. the geometric source advances the conserved
/// state along a known ray `U(t) = U + t S`. the trial interval is the local state/source timescale;
/// its endpoint is tested against the full wu-tang `(D,q,psi)` admissible set. convexity proves the
/// interval safe when that endpoint is admissible; otherwise fixed-count bisection finds the largest
/// known-safe subinterval. its inverse is added to the flux rate. this directional construction
/// cannot collapse merely because `q` is small while the source points inward or tangent to the
/// admissible set, unlike the nondirectional `|S|/q` lipschitz bound. the momentum source takes
/// `p = 0` and the energy source the full pressure, matching the fused godunov update. metric at the
/// cell centroid, ungridded polar slot at pi/2.
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
    // also harvest the lapse + shift: the evolved energy slot is the covariant ehat, so the eulerian
    // admissibility energy E = tau + D is recovered as E = (ehat + D + beta^i S_i) / alpha.
    let (gm, gm_inv, alpha, beta, s_mom, _s_tau): (
        Matrix<Gv, 3>,
        Matrix<Gv, 3>,
        Gv,
        Tensor<Gv, 3>,
        Tensor<Gv, 3>,
        Gv,
    ) = match (spacetime, coords) {
        (Spacetime::Schwarzschild, _) => {
            let mg = Schwarzschild { mass: mass_gv };
            let (sm, _) = grmhd_covariant_source(&Schwarzschild { mass }, x, rho_h, v, Gv::ZERO, b);
            let (_, st) = grmhd_covariant_source(&Schwarzschild { mass }, x, rho_h, v, pre, b);
            (
                mg.spatial_metric(x),
                mg.spatial_metric_inv(x),
                mg.lapse(x),
                mg.shift(x),
                sm,
                st,
            )
        }
        (Spacetime::SchwarzschildKS, Coords::Cartesian) => {
            let mg = SchwarzschildKSCartesian { mass: mass_gv };
            let (sm, _) = grmhd_covariant_source(
                &SchwarzschildKSCartesian { mass },
                x,
                rho_h,
                v,
                Gv::ZERO,
                b,
            );
            let (_, st) =
                grmhd_covariant_source(&SchwarzschildKSCartesian { mass }, x, rho_h, v, pre, b);
            (
                mg.spatial_metric(x),
                mg.spatial_metric_inv(x),
                mg.lapse(x),
                mg.shift(x),
                sm,
                st,
            )
        }
        (Spacetime::SchwarzschildKS, Coords::Cylindrical) => {
            let mg = SchwarzschildKSCylindrical { mass: mass_gv };
            let (sm, _) = grmhd_covariant_source(
                &SchwarzschildKSCylindrical { mass },
                x,
                rho_h,
                v,
                Gv::ZERO,
                b,
            );
            let (_, st) =
                grmhd_covariant_source(&SchwarzschildKSCylindrical { mass }, x, rho_h, v, pre, b);
            (
                mg.spatial_metric(x),
                mg.spatial_metric_inv(x),
                mg.lapse(x),
                mg.shift(x),
                sm,
                st,
            )
        }
        (Spacetime::SchwarzschildKS, _) => {
            let mg = SchwarzschildKS { mass: mass_gv };
            let (sm, _) =
                grmhd_covariant_source(&SchwarzschildKS { mass }, x, rho_h, v, Gv::ZERO, b);
            let (_, st) = grmhd_covariant_source(&SchwarzschildKS { mass }, x, rho_h, v, pre, b);
            (
                mg.spatial_metric(x),
                mg.spatial_metric_inv(x),
                mg.lapse(x),
                mg.shift(x),
                sm,
                st,
            )
        }
        // spinning kerr on the CARTESIAN chart: the rank-1 kerr-schild update with the
        // oblate-spheroidal radius; non-diagonal gamma + shift on every axis.
        (Spacetime::KerrKS, Coords::Cartesian) => {
            let spin_gv = Gv::scalar("kerr_spin");
            let mg = KerrKSCartesian {
                mass: mass_gv,
                spin: spin_gv,
            };
            let spin = Dual::constant(spin_gv);
            let (sm, _) =
                grmhd_covariant_source(&KerrKSCartesian { mass, spin }, x, rho_h, v, Gv::ZERO, b);
            let (_, st) =
                grmhd_covariant_source(&KerrKSCartesian { mass, spin }, x, rho_h, v, pre, b);
            (
                mg.spatial_metric(x),
                mg.spatial_metric_inv(x),
                mg.lapse(x),
                mg.shift(x),
                sm,
                st,
            )
        }
        (Spacetime::KerrKS, Coords::Cylindrical) => {
            let spin_gv = Gv::scalar("kerr_spin");
            let mg = KerrKSCylindrical {
                mass: mass_gv,
                spin: spin_gv,
            };
            let spin = Dual::constant(spin_gv);
            let (sm, _) =
                grmhd_covariant_source(&KerrKSCylindrical { mass, spin }, x, rho_h, v, Gv::ZERO, b);
            let (_, st) =
                grmhd_covariant_source(&KerrKSCylindrical { mass, spin }, x, rho_h, v, pre, b);
            (
                mg.spatial_metric(x),
                mg.spatial_metric_inv(x),
                mg.lapse(x),
                mg.shift(x),
                sm,
                st,
            )
        }
        (Spacetime::KerrKS, _) => {
            let spin_gv = Gv::scalar("kerr_spin");
            let mg = KerrKS {
                mass: mass_gv,
                spin: spin_gv,
            };
            let spin = Dual::constant(spin_gv);
            let (sm, _) = grmhd_covariant_source(&KerrKS { mass, spin }, x, rho_h, v, Gv::ZERO, b);
            let (_, st) = grmhd_covariant_source(&KerrKS { mass, spin }, x, rho_h, v, pre, b);
            (
                mg.spatial_metric(x),
                mg.spatial_metric_inv(x),
                mg.lapse(x),
                mg.shift(x),
                sm,
                st,
            )
        }
        (Spacetime::Minkowski, _) => {
            unreachable!("the source-admissibility CFL is baked only for a curved spacetime")
        }
    };
    // the admissible cone at the current cell: E = tau + D, |S|^2 = gamma^{ij} S_i S_j. the stored
    // energy is the covariant ehat, so E = (ehat + D + beta^i S_i) / alpha (metric-free b^2 blocks a
    // direct e - p reconstruction; the invert is exact).
    let d_cons = Gv::field("cons_den", FieldRef::cons_den());
    let mom: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8)));
    let ehat = Gv::field("cons_nrg", FieldRef::cons_nrg());
    let beta_s = (0..3).fold(Gv::ZERO, |acc, k| acc + beta[k] * mom[k]);
    let e_cons = (ehat + d_cons + beta_s) / alpha;
    let gamma_norm = |a: &[Gv; 3]| {
        let mut acc = Gv::ZERO;
        for ii in 0..3 {
            for jj in 0..3 {
                acc = acc + gm_inv[(ii, jj)] * a[ii] * a[jj];
            }
        }
        acc
    };
    let sm_arr: [Gv; 3] = std::array::from_fn(|k| s_mom[k]);
    let sm_norm = gamma_norm(&sm_arr).sqrt();
    let state_scale = symbi_hydro::admissible::rmhd_state_scale(
        d_cons,
        &Tensor::new(mom),
        e_cons,
        &b,
        &gm_inv,
        &gm,
    );
    // follow the ACTUAL geometric-source ray through the full RMHD admissible set. the evolved
    // killing-energy slot has no geometric source on a stationary metric, while the momentum update
    // is dS/dt = alpha Smom. therefore dE/dt = beta.Smom after differentiating
    // E = (ehat + D + beta.S)/alpha. the local state/source ratio supplies a dimensionally natural
    // trial time. if its endpoint is admissible, convexity proves the whole segment safe; otherwise
    // fixed-count bisection returns the largest known-safe fraction. unlike the lipschitz |source|/q
    // bound, an inward or tangent source does not collapse dt merely because a previous projection
    // left q near its strict floor.
    let (source_mom, e_dot) =
        symbi_hydro::admissible::stationary_killing_source_ray(alpha, &beta, Tensor::new(sm_arr));
    let source_scale = e_dot.abs().max(alpha * sm_norm);
    let eps_d = Gv::from_f64(1e-12) * state_scale;
    let eps_q = Gv::from_f64(symbi_hydro::admissible::ADMISSIBLE_REL_FLOOR) * state_scale;
    let eps_psi = Gv::from_f64(symbi_hydro::admissible::ADMISSIBLE_REL_FLOOR)
        * state_scale
        * state_scale.sqrt();
    let safe_time = symbi_hydro::admissible::rmhd_source_admissible_time(
        d_cons,
        Tensor::new(mom),
        e_cons,
        source_mom,
        e_dot,
        &b,
        &gm_inv,
        &gm,
        state_scale,
        source_scale,
        eps_d,
        eps_q,
        eps_psi,
        16,
    );
    let lam_s = Gv::ONE / safe_time;
    // no cell inside the event horizon may throttle the global timestep. the outer horizon
    // r_+ = M + sqrt(M^2 - a^2) is a one-way causal boundary on a horizon-penetrating chart: nothing
    // interior reaches the exterior, so an interior cell's admissibility rate carries no information
    // about exterior stability. infalling gas drives the interior pressure toward zero, which sends
    // the cone margin q -> 0 and the source rate lambda_S = (|S_tau| + ||S_mom||_gamma)/q -> inf, so
    // without this mask a handful of sub-horizon cells set dt for the whole grid — measured on the
    // spinning magnetized torus, 24 cells lying between the excision surface and r_+ held lambda_S at
    // 3.3e9 while the exterior maximum was 17.9, a factor 1.8e8 in dt.
    //
    // the threshold is the LARGER of r_+ and the excision surface, so a run that excises further out
    // keeps masking everything it excises. an excised cell is additionally numerical padding whose
    // onion-filled state can sit near the cone boundary indefinitely, the frozen clamped-core metric
    // driving enormous geodesic sources over donor-copied gas. gating cell CENTERS on r_+ matches how
    // interior guard activations are counted. cartesian charts only (spherical charts never excise).
    let lam_s = if coords == Coords::Cartesian
        && matches!(spacetime, Spacetime::SchwarzschildKS | Spacetime::KerrKS)
    {
        let spin = if spacetime == Spacetime::KerrKS {
            Gv::scalar("kerr_spin")
        } else {
            Gv::ZERO
        };
        let xc: [Gv; 3] = std::array::from_fn(|c| x[c]);
        let r_plus = mass_gv + (mass_gv * mass_gv - spin * spin).max(Gv::ZERO).sqrt();
        let r_mask = r_plus.max(Gv::scalar("excision_radius"));
        let excised = symbi_ib::excise::ks_excised(&xc, spin, r_mask);
        Gv::select(excised, Gv::ZERO, lam_s)
    } else {
        lam_s
    };
    let lam_flux = Gv::field("lambda", FieldRef::Scratch);
    let total = lam_flux + lam_s;
    (
        end_trace(),
        vec![("lambda".to_string(), FieldRef::Scratch.into(), total.node())],
    )
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
    // the densitized law samples its metric coefficients at the cell's ARITHMETIC MIDPOINT, the
    // same point the c2p undensitizes at, so the cone reads the state the recovery produced.
    let mid = gv_cell_midpoints(spacing, ndim);
    let x = Tensor::<Gv, 3>::new(std::array::from_fn(|c| {
        match axes.iter().position(|&a| a == c) {
            Some(d) => mid[d],
            None => gv_ungridded_slot(coords, c),
        }
    }));
    // the admissible cone is a statement about the PHYSICAL state, so the densitized conserveds
    // are divided by the full-chart measure sqrt(det gamma) before entering it. the densitized
    // update adds `dt sqrt(-g) S` to `sqrt(det gamma) U`, so the undensitized source rate the cone
    // sees is `alpha S` — the lapse is applied to the connection source below.
    let inv_dens = Gv::ONE / gv_metric_volume_factor_at(spacetime, coords, x);
    let d_cons = Gv::field("cons_den", FieldRef::cons_den()) * inv_dens;
    // the effective inertia e = rho h W^2 = h D^2 / rho_prim (W = D/rho_prim), reconstructed
    // metric-free and independent of the energy variable: the RHD nrg slot stores the killing
    // energy, not the Valencia tau, so `rho + tau + pre` no longer names rho h W^2. the eulerian
    // energy for the admissibility cone follows as E = e - p = rho h W^2 - p = tau + D (below).
    let gamma_eos = Gv::scalar("gamma");
    let h_enth = Gv::ONE + gamma_eos / (gamma_eos - Gv::ONE) * pre / rho;
    let e = h_enth * d_cons * d_cons / rho;
    let mass_gv = Gv::scalar("schwarzschild_mass");
    let mass = Dual::constant(mass_gv);
    let (gm_inv, s_mom, _s_tau, alpha, beta): (
        Matrix<Gv, 3>,
        Tensor<Gv, 3>,
        Gv,
        Gv,
        Tensor<Gv, 3>,
    ) = match (spacetime, coords) {
        (Spacetime::Schwarzschild, _) => {
            let gi = Schwarzschild { mass: mass_gv }.spatial_metric_inv(x);
            let al = Schwarzschild { mass: mass_gv }.lapse(x);
            let bt = Schwarzschild { mass: mass_gv }.shift(x);
            let (sm, _) = grhd_covariant_source(&Schwarzschild { mass }, x, e, v, Gv::ZERO);
            let (_, st) = grhd_covariant_source(&Schwarzschild { mass }, x, e, v, pre);
            (gi, sm, st, al, bt)
        }
        (Spacetime::SchwarzschildKS, Coords::Cartesian) => {
            let gi = SchwarzschildKSCartesian { mass: mass_gv }.spatial_metric_inv(x);
            let al = SchwarzschildKSCartesian { mass: mass_gv }.lapse(x);
            let bt = SchwarzschildKSCartesian { mass: mass_gv }.shift(x);
            let (sm, _) =
                grhd_covariant_source(&SchwarzschildKSCartesian { mass }, x, e, v, Gv::ZERO);
            let (_, st) = grhd_covariant_source(&SchwarzschildKSCartesian { mass }, x, e, v, pre);
            (gi, sm, st, al, bt)
        }
        (Spacetime::SchwarzschildKS, Coords::Cylindrical) => {
            let gi = SchwarzschildKSCylindrical { mass: mass_gv }.spatial_metric_inv(x);
            let al = SchwarzschildKSCylindrical { mass: mass_gv }.lapse(x);
            let bt = SchwarzschildKSCylindrical { mass: mass_gv }.shift(x);
            let (sm, _) =
                grhd_covariant_source(&SchwarzschildKSCylindrical { mass }, x, e, v, Gv::ZERO);
            let (_, st) = grhd_covariant_source(&SchwarzschildKSCylindrical { mass }, x, e, v, pre);
            (gi, sm, st, al, bt)
        }
        (Spacetime::SchwarzschildKS, _) => {
            let gi = SchwarzschildKS { mass: mass_gv }.spatial_metric_inv(x);
            let al = SchwarzschildKS { mass: mass_gv }.lapse(x);
            let bt = SchwarzschildKS { mass: mass_gv }.shift(x);
            let (sm, _) = grhd_covariant_source(&SchwarzschildKS { mass }, x, e, v, Gv::ZERO);
            let (_, st) = grhd_covariant_source(&SchwarzschildKS { mass }, x, e, v, pre);
            (gi, sm, st, al, bt)
        }
        // spinning kerr on the CARTESIAN chart: the rank-1 kerr-schild update with the
        // oblate-spheroidal radius; non-diagonal gamma + shift on every axis.
        (Spacetime::KerrKS, Coords::Cartesian) => {
            let spin_gv = Gv::scalar("kerr_spin");
            let gi = KerrKSCartesian {
                mass: mass_gv,
                spin: spin_gv,
            }
            .spatial_metric_inv(x);
            let al = KerrKSCartesian {
                mass: mass_gv,
                spin: spin_gv,
            }
            .lapse(x);
            let bt = KerrKSCartesian {
                mass: mass_gv,
                spin: spin_gv,
            }
            .shift(x);
            let spin = Dual::constant(spin_gv);
            let (sm, _) = grhd_covariant_source(&KerrKSCartesian { mass, spin }, x, e, v, Gv::ZERO);
            let (_, st) = grhd_covariant_source(&KerrKSCartesian { mass, spin }, x, e, v, pre);
            (gi, sm, st, al, bt)
        }
        (Spacetime::KerrKS, Coords::Cylindrical) => {
            let spin_gv = Gv::scalar("kerr_spin");
            let gi = KerrKSCylindrical {
                mass: mass_gv,
                spin: spin_gv,
            }
            .spatial_metric_inv(x);
            let al = KerrKSCylindrical {
                mass: mass_gv,
                spin: spin_gv,
            }
            .lapse(x);
            let bt = KerrKSCylindrical {
                mass: mass_gv,
                spin: spin_gv,
            }
            .shift(x);
            let spin = Dual::constant(spin_gv);
            let (sm, _) =
                grhd_covariant_source(&KerrKSCylindrical { mass, spin }, x, e, v, Gv::ZERO);
            let (_, st) = grhd_covariant_source(&KerrKSCylindrical { mass, spin }, x, e, v, pre);
            (gi, sm, st, al, bt)
        }
        (Spacetime::KerrKS, _) => {
            let spin_gv = Gv::scalar("kerr_spin");
            let gi = KerrKS {
                mass: mass_gv,
                spin: spin_gv,
            }
            .spatial_metric_inv(x);
            let al = KerrKS {
                mass: mass_gv,
                spin: spin_gv,
            }
            .lapse(x);
            let bt = KerrKS {
                mass: mass_gv,
                spin: spin_gv,
            }
            .shift(x);
            let spin = Dual::constant(spin_gv);
            let (sm, _) = grhd_covariant_source(&KerrKS { mass, spin }, x, e, v, Gv::ZERO);
            let (_, st) = grhd_covariant_source(&KerrKS { mass, spin }, x, e, v, pre);
            (gi, sm, st, al, bt)
        }
        (Spacetime::Minkowski, _) => {
            unreachable!("the source-admissibility CFL is baked only for a curved spacetime")
        }
    };
    let mom: [Gv; 3] = std::array::from_fn(|k| {
        if k < ncomp {
            Gv::field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8)) * inv_dens
        } else {
            Gv::ZERO
        }
    });
    // the connection source is densitized by sqrt(-g) while the state carries only
    // sqrt(det gamma), so the undensitized state actually moves at rate alpha S — a lapse the rate
    // below deliberately does NOT apply. alpha <= 1, so charging the full |S| bounds the true
    // margin consumption from above and the admissible step from below; relaxing it by the lapse
    // is a sharpening that has to be justified against the cone it protects, not a free
    // simplification.
    // the eulerian total energy E = tau + D = rho h W^2 - p = e - p (metric-free; no killing-energy
    // inversion needed for the admissibility cone).
    let e_cons = e - pre;
    let gamma_norm = |a: &[Gv; 3]| {
        let mut acc = Gv::ZERO;
        for ii in 0..3 {
            for jj in 0..3 {
                acc = acc + gm_inv[(ii, jj)] * a[ii] * a[jj];
            }
        }
        acc
    };
    let root = (d_cons * d_cons + gamma_norm(&mom)).sqrt();
    let q0 = e_cons - root;
    let state_scale =
        symbi_hydro::admissible::rhd_state_scale(d_cons, &Tensor::new(mom), e_cons, &gm_inv);
    let q_safe = q0.max(Gv::from_f64(symbi_hydro::admissible::ADMISSIBLE_REL_FLOOR) * state_scale);
    // the rate at which the geometric source consumes the admissibility margin
    // q0 = E - sqrt(D^2 + |S|^2), evaluated for the COVARIANT (killing) energy variable.
    // over a source step the mass has no source and the killing energy has none either
    // (its source is identically zero on a stationary metric), so ONLY the momentum
    // source moves the state. writing the eulerian energy in the stored variables,
    //   E = (ehat + D + beta^i S_i) / alpha,
    // and differentiating along S -> S + dt Smom,
    //   dq0/dt = (beta^i Smom_i)/alpha - (gamma^{ij} S_i Smom_j)/sqrt(D^2 + |S|^2).
    // the second term carries the factor |S|/sqrt(D^2 + |S|^2) and vanishes at rest, so
    // cold gas at rest on a SHIFT-FREE chart consumes no margin at first order: the
    // gravitational work rides inside the conserved killing energy rather than being
    // charged against the vanishing internal energy. charging the valencia form
    // (|S_tau| + |Smom|)/q0 instead makes the admissible dt scale like the pressure and
    // drives dt to zero on cold atmospheres, a limit this variable does not actually
    // impose. the energy source is absent from the update and so absent here too.
    let sm_arr: [Gv; 3] = std::array::from_fn(|k| s_mom[k]);
    let beta_dot_sm = {
        let mut acc = Gv::ZERO;
        for ii in 0..3 {
            acc = acc + beta[ii] * sm_arr[ii];
        }
        acc
    };
    let s_dot_sm = {
        let mut acc = Gv::ZERO;
        for ii in 0..3 {
            for jj in 0..3 {
                acc = acc + gm_inv[(ii, jj)] * mom[ii] * sm_arr[jj];
            }
        }
        acc
    };
    // the FIRST-ORDER rate above degenerates exactly where cold gas sits: at rest S -> 0
    // kills the second term, and a shift-free chart kills the first, leaving no constraint
    // at all. the margin is still consumed at SECOND order, because |S| grows like
    // dt |Smom| and
    //   sqrt(D^2 + |S|^2) ~ root + (dt |Smom|)^2 / (2 root),
    // so admissibility needs dt < sqrt(2 root q0) / |Smom|. carried as the rate
    // |Smom| / sqrt(2 root q0), it makes the cold-gas limit scale like sqrt(p) instead of
    // the valencia form's p — far weaker, but NOT absent. omitting it lets a stationary
    // rotating equilibrium on a zero-shift chart run past its admissible step.
    let root_safe =
        root.max(Gv::from_f64(symbi_hydro::admissible::ADMISSIBLE_REL_FLOOR) * state_scale);
    let sm_norm = gamma_norm(&sm_arr).sqrt();
    let lam_first = ((beta_dot_sm / alpha).abs() + (s_dot_sm / root_safe).abs()) / q_safe;
    let lam_second = sm_norm / (Gv::from_f64(2.0) * root_safe * q_safe).sqrt();
    let lam_s = lam_first + lam_second;
    // no cell inside the event horizon may throttle the global timestep. the outer horizon
    // r_+ = M + sqrt(M^2 - a^2) is a one-way causal boundary on a horizon-penetrating chart: nothing
    // interior reaches the exterior, so an interior cell's admissibility rate carries no information
    // about exterior stability. infalling gas drives the interior pressure toward zero, which sends
    // the cone margin q -> 0 and the source rate lambda_S = (|S_tau| + ||S_mom||_gamma)/q -> inf, so
    // without this mask a handful of sub-horizon cells set dt for the whole grid — measured on the
    // spinning magnetized torus, 24 cells lying between the excision surface and r_+ held lambda_S at
    // 3.3e9 while the exterior maximum was 17.9, a factor 1.8e8 in dt.
    //
    // the threshold is the LARGER of r_+ and the excision surface, so a run that excises further out
    // keeps masking everything it excises. an excised cell is additionally numerical padding whose
    // onion-filled state can sit near the cone boundary indefinitely, the frozen clamped-core metric
    // driving enormous geodesic sources over donor-copied gas. gating cell CENTERS on r_+ matches how
    // interior guard activations are counted. cartesian charts only (spherical charts never excise).
    let lam_s = if coords == Coords::Cartesian
        && matches!(spacetime, Spacetime::SchwarzschildKS | Spacetime::KerrKS)
    {
        let spin = if spacetime == Spacetime::KerrKS {
            Gv::scalar("kerr_spin")
        } else {
            Gv::ZERO
        };
        let xc: [Gv; 3] = std::array::from_fn(|c| x[c]);
        let r_plus = mass_gv + (mass_gv * mass_gv - spin * spin).max(Gv::ZERO).sqrt();
        let r_mask = r_plus.max(Gv::scalar("excision_radius"));
        let excised = symbi_ib::excise::ks_excised(&xc, spin, r_mask);
        Gv::select(excised, Gv::ZERO, lam_s)
    } else {
        lam_s
    };
    let lam_flux = Gv::field("lambda", FieldRef::Scratch);
    let total = lam_flux + lam_s;
    (
        end_trace(),
        vec![("lambda".to_string(), FieldRef::Scratch.into(), total.node())],
    )
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
        hydro: Prim {
            rho,
            vel: Tensor::new(vel),
            pre,
        },
        mag: Tensor::new(mag),
    };
    // one (lmin,lmax) pair per spatial sweep direction (ndim of them); the B is always
    // a 3-vector but the grid varies along ndim axes only.
    let mut writes = Vec::with_capacity(2 * ndim);
    for d in 0..ndim {
        let nhat = Tensor::<Gv, 3>::unit(d);
        // raw quartic min/max — NOT extremal_speeds (no zero-clamp); the flux clamps the fan.
        let (lmin, lmax) = Rmhd.wave_speeds(&eos, &prim, &nhat);
        writes.push((
            format!("ws_l_{d}"),
            format!("wave_speed_l[{d}]").into(),
            lmin.node(),
        ));
        writes.push((
            format!("ws_r_{d}"),
            format!("wave_speed_r[{d}]").into(),
            lmax.node(),
        ));
    }
    (end_trace(), writes)
}

/// the FOFC ADMISSIBLE-BOUNDARY PROJECTION for GR-hydro (adiabatic) — the provable replacement for the
/// freeze parachute. where the spliced first-order conserved `x_*` is inadmissible, blend it toward the
/// stage-input anchor `us_*` (admissible from stage entry) exactly onto the boundary of the
/// relativistic admissible set `G = { D > 0, E^2 > D^2 + gamma^{ij} S_i S_j }` (Wu & Tang 2015). G is
/// CONVEX, so the segment from an admissible anchor to any candidate crosses partial-G at most once and
/// the projection ALWAYS yields an admissible state — an already-admissible cell passes through
/// untouched (theta = 1), so no cell is ever unrecoverable. reads + writes the densitized conserveds
/// `x_den`/`x_mom_k`/`x_nrg` in place, reads the anchor `us_*`; the metric at the cell midpoint supplies
/// gamma^{ij} (for |S|^2) and alpha/beta (to reconstruct the eulerian energy E = (ehat + D + beta^i
/// S_i)/alpha from the stored killing energy). densitization is a common positive factor that cancels
/// in the admissibility sign, so this works directly on the stored state. curved spacetime only.
/// the FOFC ADMISSIBLE-BOUNDARY PROJECTION for GR-MHD — the magnetized twin of
/// [`fofc_project_gr_gv`], and the provable replacement for the RMHD freeze parachute.
///
/// two facts make the hydro projection carry over unchanged:
/// - the RMHD c2p's OWN admissibility criterion is the B-FREE cone `E^2 > D^2 + gamma^{ij} S_i S_j`
///   (`relativistic_cone_residual`, the Wu 2017 bound shared with the hydro recovery) — the magnetic
///   terms do not enter it, so `admissible_theta` is the same function;
/// - because that cone is B-free it is CONVEX in `(D, S_i, tau)` at FIXED `B`, so blending only the
///   hydro slots is sound. that matters: `B` is CONSTRAINED-TRANSPORT-evolved and must NOT be
///   touched, or `div(B) = 0` breaks. the projection leaves the staggered field alone by
///   construction.
///
/// GRMHD is UNDENSITIZED (the Valencia state with the covariant killing energy in the `nrg` slot),
/// so there is no `sqrt(-g)` here; the eulerian energy is recovered as
/// `E = (ehat + D + beta^i S_i)/alpha`, the same inversion the GRMHD c2p performs.
///
/// this enforces the SUFFICIENT admissibility condition, not merely the B-free cone: a state is
/// admissible iff D > 0, q > 0 AND psi > 0, where psi carries the magnetic terms and rejects states
/// whose magnetic energy leaves no positive gas pressure to recover. see
/// `symbi_hydro::admissible::rmhd_admissible_residuals`.
///
/// constrained transport owns the candidate magnetic field, so the anchor is rebuilt from the
/// stage-input primitive gas state with that SAME field. converting this hybrid primitive state to
/// conserved form produces a guaranteed-admissible anchor in the affine slice B = B_candidate
/// without modifying a shared face field. the projection can therefore always recover the gas state
/// while preserving div(B) = 0.
pub fn fofc_project_gr_mhd_gv(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    axes: &[usize],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let ndim = axes.len();
    let mid = gv_cell_midpoints(spacing, ndim);
    let x = Tensor::<Gv, 3>::new(std::array::from_fn(|c| {
        match axes.iter().position(|&a| a == c) {
            Some(d) => mid[d],
            None => gv_ungridded_slot(coords, c),
        }
    }));
    let mass = Gv::scalar("schwarzschild_mass");
    let (gm_inv, gm, alpha, beta): (Matrix<Gv, 3>, Matrix<Gv, 3>, Gv, Tensor<Gv, 3>) =
        match (spacetime, coords) {
            (Spacetime::Schwarzschild, _) => {
                let m = Schwarzschild { mass };
                (
                    m.spatial_metric_inv(x),
                    m.spatial_metric(x),
                    m.lapse(x),
                    m.shift(x),
                )
            }
            (Spacetime::SchwarzschildKS, Coords::Cartesian) => {
                let m = SchwarzschildKSCartesian { mass };
                (
                    m.spatial_metric_inv(x),
                    m.spatial_metric(x),
                    m.lapse(x),
                    m.shift(x),
                )
            }
            (Spacetime::SchwarzschildKS, Coords::Cylindrical) => {
                let m = SchwarzschildKSCylindrical { mass };
                (
                    m.spatial_metric_inv(x),
                    m.spatial_metric(x),
                    m.lapse(x),
                    m.shift(x),
                )
            }
            (Spacetime::SchwarzschildKS, _) => {
                let m = SchwarzschildKS { mass };
                (
                    m.spatial_metric_inv(x),
                    m.spatial_metric(x),
                    m.lapse(x),
                    m.shift(x),
                )
            }
            (Spacetime::KerrKS, Coords::Cartesian) => {
                let m = KerrKSCartesian {
                    mass,
                    spin: Gv::scalar("kerr_spin"),
                };
                (
                    m.spatial_metric_inv(x),
                    m.spatial_metric(x),
                    m.lapse(x),
                    m.shift(x),
                )
            }
            (Spacetime::KerrKS, Coords::Cylindrical) => {
                let m = KerrKSCylindrical {
                    mass,
                    spin: Gv::scalar("kerr_spin"),
                };
                (
                    m.spatial_metric_inv(x),
                    m.spatial_metric(x),
                    m.lapse(x),
                    m.shift(x),
                )
            }
            (Spacetime::KerrKS, _) => {
                let m = KerrKS {
                    mass,
                    spin: Gv::scalar("kerr_spin"),
                };
                (
                    m.spatial_metric_inv(x),
                    m.spatial_metric(x),
                    m.lapse(x),
                    m.shift(x),
                )
            }
            (Spacetime::Minkowski, _) => {
                unreachable!("the GRMHD FOFC projection is baked only for a curved spacetime")
            }
        };
    let read = |k: &str| Gv::field(k, k);
    let x_den = read("x_den");
    let x_nrg = read("x_nrg");
    // RMHD momentum is ALWAYS a 3-vector (the physics is 3D; grid symmetry handles 1D/2D).
    let x_mom: Vec<Gv> = (0..3).map(|k| read(&format!("x_mom_{k}"))).collect();
    let s_c = Tensor::<Gv, 3>::new(std::array::from_fn(|k| x_mom[k]));
    let beta_dot = |s: &Tensor<Gv, 3>| (0..3).fold(Gv::ZERO, |a, k| a + beta[k] * s[k]);
    let inv_alpha = Gv::ONE / alpha;
    let e_c = (x_nrg + x_den + beta_dot(&s_c)) * inv_alpha;
    // the magnetic field is held FIXED at the candidate's cell-centered value: it is
    // constrained-transport-evolved on the staggered faces and shared between neighbors, so blending
    // it per cell would desynchronize the shared face value and break div(B) = 0.
    let b = Tensor::<Gv, 3>::new(std::array::from_fn(|k| {
        Gv::field(&format!("bcell_{k}"), &format!("mhd.bcell[{k}]"))
    }));
    // the stage-input primitives still occupy the primitive fields here: FOFC restored u_stage and
    // ran c2p before constructing the first-order flux, while the candidate c2p has not run yet.
    // combine that known-physical gas state with candidate B and convert it through the SAME GRMHD
    // physics used by initialization. this makes the anchor admissible in the candidate magnetic
    // slice by construction.
    let anchor_gas = Prim {
        rho: Gv::field("prim_rho", FieldRef::PrimRho),
        vel: Tensor::new(std::array::from_fn(|kk| {
            Gv::field(&format!("prim_vel_{kk}"), FieldRef::PrimVel(kk as u8))
        })),
        pre: Gv::field("prim_pre", FieldRef::PrimPre),
    };
    let anchor_regime = RmhdGr {
        metric: SpatialMetric::new(Gamma::new(gm), GammaInv::new(gm_inv)),
        alpha,
    };
    let anchor = anchor_regime.admissible_anchor(
        &IdealGas {
            gamma: Gv::scalar("gamma"),
        },
        anchor_gas,
        b,
    );
    let a_den = anchor.den;
    let s_a = anchor.mom;
    let e_a = anchor.nrg + a_den;
    let a_nrg = alpha * anchor.nrg + (alpha - Gv::ONE) * a_den - beta_dot(&s_a);
    // strict-interior floors use one shared local conserved-state scale. D, |S|, E, and |B|^2
    // carry one power of energy; psi carries three halves. including magnetic energy prevents a
    // magnetically dominated atmosphere from defining its numerical margin using gas energy alone.
    let state_scale = symbi_hydro::admissible::rmhd_state_scale(a_den, &s_a, e_a, &b, &gm_inv, &gm);
    let eps_d = Gv::from_f64(1e-12) * state_scale;
    let eps_q = Gv::from_f64(symbi_hydro::admissible::ADMISSIBLE_REL_FLOOR) * state_scale;
    let eps_psi = Gv::from_f64(symbi_hydro::admissible::ADMISSIBLE_REL_FLOOR)
        * state_scale
        * state_scale.sqrt();
    // 20 halvings resolve theta to ~1e-6. every iteration unrolls into the traced expression graph,
    // and a truncated bisection only returns a SMALLER (more conservative) blend, never an
    // inadmissible one, so the count trades kernel size against how sharply the projection hugs the
    // boundary.
    let theta = symbi_hydro::admissible::rmhd_admissible_theta(
        x_den, s_c, e_c, a_den, s_a, e_a, &b, &gm_inv, &gm, eps_d, eps_q, eps_psi, 20,
    );
    let proj = |xc: Gv, ua: Gv| ua + theta * (xc - ua);
    let mut writes: Vec<(String, FieldBind, NodeId)> = Vec::new();
    writes.push((
        "x_den".to_string(),
        "x_den".into(),
        proj(x_den, a_den).node(),
    ));
    for k in 0..3 {
        let key = format!("x_mom_{k}");
        writes.push((key.clone(), key.into(), proj(x_mom[k], s_a[k]).node()));
    }
    writes.push((
        "x_nrg".to_string(),
        "x_nrg".into(),
        proj(x_nrg, a_nrg).node(),
    ));
    (end_trace(), writes)
}

pub fn fofc_project_gr_gv(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    axes: &[usize],
    ncomp: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let ndim = axes.len();
    let mid = gv_cell_midpoints(spacing, ndim);
    let x = Tensor::<Gv, 3>::new(std::array::from_fn(|c| {
        match axes.iter().position(|&a| a == c) {
            Some(d) => mid[d],
            None => gv_ungridded_slot(coords, c),
        }
    }));
    let mass = Gv::scalar("schwarzschild_mass");
    let (gm_inv, alpha, beta): (Matrix<Gv, 3>, Gv, Tensor<Gv, 3>) = match (spacetime, coords) {
        (Spacetime::Schwarzschild, _) => {
            let m = Schwarzschild { mass };
            (m.spatial_metric_inv(x), m.lapse(x), m.shift(x))
        }
        (Spacetime::SchwarzschildKS, Coords::Cartesian) => {
            let m = SchwarzschildKSCartesian { mass };
            (m.spatial_metric_inv(x), m.lapse(x), m.shift(x))
        }
        (Spacetime::SchwarzschildKS, Coords::Cylindrical) => {
            let m = SchwarzschildKSCylindrical { mass };
            (m.spatial_metric_inv(x), m.lapse(x), m.shift(x))
        }
        (Spacetime::SchwarzschildKS, _) => {
            let m = SchwarzschildKS { mass };
            (m.spatial_metric_inv(x), m.lapse(x), m.shift(x))
        }
        (Spacetime::KerrKS, Coords::Cartesian) => {
            let m = KerrKSCartesian {
                mass,
                spin: Gv::scalar("kerr_spin"),
            };
            (m.spatial_metric_inv(x), m.lapse(x), m.shift(x))
        }
        (Spacetime::KerrKS, Coords::Cylindrical) => {
            let m = KerrKSCylindrical {
                mass,
                spin: Gv::scalar("kerr_spin"),
            };
            (m.spatial_metric_inv(x), m.lapse(x), m.shift(x))
        }
        (Spacetime::KerrKS, _) => {
            let m = KerrKS {
                mass,
                spin: Gv::scalar("kerr_spin"),
            };
            (m.spatial_metric_inv(x), m.lapse(x), m.shift(x))
        }
        (Spacetime::Minkowski, _) => {
            unreachable!("the FOFC projection is baked only for a curved spacetime")
        }
    };
    let read = |k: &str| Gv::field(k, k);
    let x_den = read("x_den");
    let x_nrg = read("x_nrg");
    let us_den = read("us_den");
    let us_nrg = read("us_nrg");
    let x_mom: Vec<Gv> = (0..ncomp).map(|k| read(&format!("x_mom_{k}"))).collect();
    let us_mom: Vec<Gv> = (0..ncomp).map(|k| read(&format!("us_mom_{k}"))).collect();
    // pad the momentum to the metric's 3 slots (suppressed axes carry zero).
    let pad = |m: &[Gv]| {
        Tensor::<Gv, 3>::new(std::array::from_fn(
            |k| if k < ncomp { m[k] } else { Gv::ZERO },
        ))
    };
    let s_c = pad(&x_mom);
    let s_a = pad(&us_mom);
    // E = (ehat + D + beta^i S_i)/alpha; beta^i contravariant, S_i covariant.
    let beta_dot = |s: &Tensor<Gv, 3>| (0..3).fold(Gv::ZERO, |a, k| a + beta[k] * s[k]);
    let inv_alpha = Gv::ONE / alpha;
    let e_c = (x_nrg + x_den + beta_dot(&s_c)) * inv_alpha;
    let e_a = (us_nrg + us_den + beta_dot(&s_a)) * inv_alpha;
    // strict-interior floors share the anchor's local one-power energy scale. the density threshold
    // carries one power and the quadratic cone residual carries two.
    let state_scale = symbi_hydro::admissible::rhd_state_scale(us_den, &s_a, e_a, &gm_inv);
    let eps_d = Gv::from_f64(1e-12) * state_scale;
    let eps_f =
        Gv::from_f64(symbi_hydro::admissible::ADMISSIBLE_REL_FLOOR) * state_scale * state_scale;
    let theta = symbi_hydro::admissible::admissible_theta(
        x_den, s_c, e_c, us_den, s_a, e_a, &gm_inv, eps_d, eps_f,
    );
    let proj = |xc: Gv, ua: Gv| ua + theta * (xc - ua);
    let mut writes: Vec<(String, FieldBind, NodeId)> = Vec::new();
    writes.push((
        "x_den".to_string(),
        "x_den".into(),
        proj(x_den, us_den).node(),
    ));
    for k in 0..ncomp {
        let key = format!("x_mom_{k}");
        writes.push((key.clone(), key.into(), proj(x_mom[k], us_mom[k]).node()));
    }
    writes.push((
        "x_nrg".to_string(),
        "x_nrg".into(),
        proj(x_nrg, us_nrg).node(),
    ));
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
        hydro: Prim {
            rho,
            vel: Tensor::new(vel),
            pre,
        },
        mag: Tensor::new(mag),
    };
    let mut writes = Vec::with_capacity(2 * ndim);
    for d in 0..ndim {
        let nhat = Tensor::<Gv, 3>::unit(d);
        let (lmin, lmax) = NewtonianMhd.wave_speeds(&eos, &prim, &nhat);
        writes.push((
            format!("ws_l_{d}"),
            format!("wave_speed_l[{d}]").into(),
            lmin.node(),
        ));
        writes.push((
            format!("ws_r_{d}"),
            format!("wave_speed_r[{d}]").into(),
            lmax.node(),
        ));
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
        hydro: PrimG {
            rho,
            vel: Tensor::new(vel),
            pre: Zero::default(),
        },
        mag: Tensor::new(mag),
    };
    let mut writes = Vec::with_capacity(2 * ndim);
    for d in 0..ndim {
        let nhat = Tensor::<Gv, 3>::unit(d);
        let (lmin, lmax) = IsothermalMhd.wave_speeds(&eos, &prim, &nhat);
        writes.push((
            format!("ws_l_{d}"),
            format!("wave_speed_l[{d}]").into(),
            lmin.node(),
        ));
        writes.push((
            format!("ws_r_{d}"),
            format!("wave_speed_r[{d}]").into(),
            lmax.node(),
        ));
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
    use symbi_ir::graph::NodeId;
    use symbi_ir::gv::{begin_trace, end_trace, with_trace};

    fn eval(out: NodeId, values: &[(&str, f64)]) -> f64 {
        use symbi_ir::backends::interp::{Backend, Cpu};
        use symbi_ir::passes::scalarize::{LoweredFn, scalarize_kernel};
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
        // builder arg is vestigial (the face map reads `map_kind`).
        let got = eval(
            node,
            &[
                ("x_lo_0", r_min),
                ("dx_0", slope),
                ("_coord_0", i),
                ("map_kind_0", 1.0),
                ("map_param_0", 0.0),
            ],
        );
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
        let got = eval(
            node,
            &[
                ("x_lo_0", x_lo),
                ("dx_0", dx),
                ("_coord_0", i),
                ("map_kind_0", 0.0),
                ("map_param_0", 1.0),
            ],
        );
        end_trace();
        assert_eq!(
            got,
            x_lo + (i + 0.5) * dx,
            "uniform center must equal the old formula bit-for-bit"
        );
    }
}
