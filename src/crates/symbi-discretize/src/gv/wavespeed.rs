// =============================================================================
// wavespeed.rs
//
// cfl wave-speed map + per-cell wave-speed kernel builders (the geometry-folded characteristic speed).
// =============================================================================

use super::*;


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
    let half_c = Gv::from_f64(0.5);
    // the cell-centroid coordinate on the grid axis carrying COORDINATE `target` (radial = 0, polar =
    // 1), for the physical-velocity scale factors below. `None` if that coordinate is not a grid axis.
    let coord_at = |target: usize| -> Option<Gv> {
        axes.iter().position(|&c| c == target).map(|d| {
            Gv::scalar(&format!("x_lo_{d}")) + (Gv::coord(d as u8) + half_c) * Gv::scalar(&format!("dx_{d}"))
        })
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
                0 => raw / gv_metric_lapse_at(spacetime, r),
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
            Some(Gv::scalar(&format!("x_lo_{d_r}"))
                + (Gv::coord(d_r as u8) + half) * Gv::scalar(&format!("dx_{d_r}")))
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
                    gv_metric_lapse_sq_at(spacetime, r)
                } else {
                    gv_metric_lapse_at(spacetime, r)
                };
                // the radial shift beta^r (kerr-schild only); zero-shift backgrounds -> None.
                match if is_radial { gv_metric_shift_r_at(spacetime, r) } else { None } {
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
