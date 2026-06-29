// =============================================================================
// sources.rs
//
// geometric + user momentum/energy source kernel builders and their probes.
// =============================================================================

use super::*;


// =============================================================================
// the conserved-update GODUNOV family in Gv — the finite-volume divergence (the Gv stencil
// `field_shifted(F_i, +e_i) - field(F_i)`, no `MorphismKind::Diff`) composed with the
// forward-Euler / RK2 time update over the conserved set (mass + one scalar law per momentum
// component + optional energy), and the snapshot copy. EOS- AND geometry-generic: snapshot is
// a pure copy (every coord); the CARTESIAN-uniform divergence is `(F_hi - F_lo)/dx_i`, the
// CURVILINEAR is the analytic area-weighted `(1/V)(F_hi*A_hi - F_lo*A_lo)` from the in-kernel
// `cell_geometry_gv` metric, plus the geometric momentum source `S^i = -Gamma^i_jk T^jk`
// (the gv christoffel) on the curvilinear momentum laws. ONE trace per (regime, geom).
// =============================================================================

/// which geometric momentum source the CURVILINEAR godunov adds to the momentum laws. the
/// flux divergence + the conserved-law structure are identical across regimes; only the source
/// expression differs (the "one operator, regime supplies its physics" rule). cartesian = none.
#[derive(Clone, Copy)]
pub enum GeoSource {
    /// hydro / SRHD: well-balanced pressure + (ndim>=2) velocity-quadratic centrifugal/coriolis,
    /// regime-agnostic via the CONSERVED momentum (Newtonian `mom=rho v`, SRHD `mom=rho h W^2 v`).
    Hydro { inertial: bool },
    /// RMHD: pressure + inertial + magnetic tension, from `rmhd_source_quantities` (cons.mom
    /// carries B-momentum so it can't serve the source).
    Rmhd,
    /// newtonian MHD: pressure (p + 1/2|B|^2) + gas inertial (cons.mom IS rho v — the Maxwell
    /// stress lives in the flux, not the momentum) + magnetic tension from the LAB-FRAME B (no
    /// relativistic four-vector). simpler than RMHD: cons.mom serves the inertial directly.
    NewtonianMhd,
    /// isothermal MHD: identical to `NewtonianMhd` but the gas pressure is `cs^2 rho` (no
    /// energy / `prim.pre`); the closure scalar `cs` is read in-kernel.
    IsothermalMhd,
}


/// the centrifugal/coriolis INERTIAL momentum source per component `S^i = -Gamma^i_jk mom^j
/// v^k` (the velocity-quadratic geometric terms), in Gv. REGIME-AGNOSTIC via the conserved `mom`
/// (Newtonian rho v -> rho v^2; relativistic rho h W^2 v -> wgam2 v^2). `centroid` is
/// COORDINATE-indexed (r at [0], theta at [1]); one source per carried component (`ncomp`).
fn inertial_momentum_sources_gv(
    ncomp: usize,
    coords: Coords,
    mom: &[Gv],
    vel: &[Gv],
    centroid: &[Gv],
) -> Vec<Gv> {
    let mut s = vec![Gv::ZERO; ncomp];
    if coords == Coords::Cartesian {
        return s; // flat space: no inertial source.
    }
    let inv_r = Gv::ONE / centroid[0]; // 1/r (centroid radial)
    match coords {
        // spherical (r, theta, phi): each `rho v_a v_b` term is `mom_a v_b`.
        Coords::Spherical => {
            // S_r = (sum_t mom_t v_t) / r — the carried transverse components.
            let mut sum: Option<Gv> = None;
            for t in 1..ncomp {
                let mvt = mom[t] * vel[t];
                sum = Some(match sum {
                    None => mvt,
                    Some(a) => a + mvt,
                });
            }
            if let Some(sum) = sum {
                s[0] = sum * inv_r;
            }
            if ncomp >= 2 {
                let cot = centroid[1].cos() / centroid[1].sin();
                // S_theta = (mom_phi v_phi cot - mom_r v_theta) / r.
                let mut num = Gv::ZERO - mom[0] * vel[1];
                if ncomp >= 3 {
                    num = num + mom[2] * vel[2] * cot;
                }
                s[1] = num * inv_r;
                if ncomp >= 3 {
                    // S_phi = -mom_phi (v_r + v_theta cot) / r.
                    let inner = vel[0] + vel[1] * cot;
                    s[2] = Gv::ZERO - mom[2] * inner * inv_r;
                }
            }
        }
        // cylindrical (r, phi, z): phi (component 1) is the swirl; z has no inertial source.
        Coords::Cylindrical => {
            if ncomp >= 2 {
                s[0] = mom[1] * vel[1] * inv_r; // S_r = mom_phi v_phi / r
                s[1] = Gv::ZERO - mom[0] * vel[1] * inv_r; // S_phi = -mom_r v_phi / r
            }
        }
        Coords::Cartesian => unreachable!(),
    }
    s
}


/// the FULL geometric momentum source per component `S^i = -Gamma^i_jk T^jk` in Gv, split
/// into the three pieces every
/// regime shares: well-balanced PRESSURE `ptot*(A_hi - A_lo)*inv_V`, INERTIAL `-Gamma(wgam2 v
/// v)`, and (RMHD) MAGNETIC `+Gamma(bmu bmu)`. `gas_mom`/`vel`/`bmu` are the regime quantities;
/// pass EMPTY `gas_mom` to skip the inertial (1D radial). `axes[d]` = the coord of grid axis d.
fn geometric_momentum_sources_gv(
    coords: Coords,
    axes: &[usize],
    ndim: usize,
    ncomp: usize,
    geo: &CellGeometryGv,
    ptot: Gv,
    gas_mom: &[Gv],
    vel: &[Gv],
    bmu: Option<&[Gv]>,
) -> Vec<Gv> {
    // COORDINATE-indexed centroid (r at [0], theta at [1]); symmetry slots stay 0 (never read).
    let mut coord_centroid = vec![Gv::ZERO; 3];
    for d in 0..ndim {
        coord_centroid[axes[d]] = geo.centroid[d];
    }
    let inertial = (!gas_mom.is_empty())
        .then(|| inertial_momentum_sources_gv(ncomp, coords, gas_mom, vel, &coord_centroid));
    // the magnetic tension is the SAME christoffel on the four-vector, negated.
    let mag = bmu.map(|b| inertial_momentum_sources_gv(ncomp, coords, b, b, &coord_centroid));
    (0..ncomp)
        .map(|coord| {
            // PRESSURE: only a GRIDDED coordinate has a pressure gradient; written in the
            // divergence's (ptot*A_hi - ptot*A_lo)*inv_V form so a v=0 uniform-ptot state
            // cancels the pressure flux divergence bit-exactly (well-balanced HSE).
            let mut s = if let Some(d) = axes.iter().position(|&c| c == coord) {
                (ptot * geo.area_hi[d] - ptot * geo.area_lo[d]) * geo.inv_volume
            } else {
                Gv::ZERO
            };
            if let Some(ref inert) = inertial {
                s = s + inert[coord];
            }
            if let Some(ref m) = mag {
                s = s - m[coord]; // - magnetic tension (= +Gamma bmu bmu)
            }
            s
        })
        .collect()
}


/// trace the regime's geometric momentum source quantities + form the per-component source.
/// HYDRO/SRHD: total pressure = `prim.pre`, gas momentum density * v = the CONSERVED momentum
/// (cons.mom IS rho v / rho h W^2 v), no magnetic term. RMHD: the gas + magnetic quantities
/// from symbi-hydro's `rmhd_source_quantities` (the SAME carrier-generic source the RMHD flux
/// uses), gas_mom = wgam2*v, bmu = the spatial four-vector. `cons_mom` is the already-read
/// in-place conserved momentum (shared so the gas-inertial reuses it — no duplicate buffer).
pub(crate) fn gv_geometric_source(
    coords: Coords,
    axes: &[usize],
    ndim: usize,
    ncomp: usize,
    geo: &CellGeometryGv,
    source: GeoSource,
    cons_mom: &[Gv],
    mag_from_bcell: bool,
) -> Vec<Gv> {
    // the cell-B the magnetic geo source reads. NORMALLY the primitive `prim.mag[k]`. but when
    // this stage is FUSED with the cell-B predictor (which binds `bc_k` in-place), reading mag
    // via the SAME `bc_k` key lets try_fuse merge the two cell-B reads into ONE binding — without
    // it, `prim.mag[k]` and `bc_k` are distinct manifest entries that both resolve to bcell[k] at
    // runtime, aliasing a read-only input to an in-place output (UB on CPU). both keys carry the
    // SAME old-bcell value (the predictor writes after the source evaluates), so it's bit-identical.
    let mag_field = |k: usize| -> Gv {
        if mag_from_bcell {
            Gv::field(&format!("bc_{k}"), FieldRef::BCell(k as u8))
        } else {
            Gv::field(&format!("prim_b{k}"), FieldRef::PrimMag(k as u8))
        }
    };
    match source {
        GeoSource::Hydro { inertial } => {
            let ptot = Gv::field("pre", FieldRef::PrimPre);
            // the velocity-quadratic inertial vanishes for 1D radial — skip it (+ its vel reads).
            let (gas_mom, vel): (Vec<Gv>, Vec<Gv>) = if inertial && ndim >= 2 {
                let v = (0..ndim)
                    .map(|d| Gv::field(&format!("prim_v{d}"), FieldRef::PrimVel(d as u8)))
                    .collect();
                (cons_mom[..ndim].to_vec(), v) // gas_mom = cons.mom (shared, no duplicate read)
            } else {
                (Vec::new(), Vec::new())
            };
            geometric_momentum_sources_gv(coords, axes, ndim, ncomp, geo, ptot, &gas_mom, &vel, None)
        }
        GeoSource::Rmhd => {
            // the RMHD stress = pressure + gas inertial + magnetic tension: read prim + gamma,
            // trace symbi-hydro's `rmhd_source_quantities` (wgam2, bmu, ptot) at S=Gv.
            let rho = Gv::field("prim_rho", FieldRef::PrimRho);
            let vel: [Gv; 3] =
                std::array::from_fn(|k| Gv::field(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8)));
            let pre = Gv::field("prim_pre", FieldRef::PrimPre);
            let mag: [Gv; 3] = std::array::from_fn(|k| mag_field(k));
            let eos = IdealGas { gamma: Gv::scalar("gamma") };
            let prim = MhdPrim::<Gv, 3> {
                hydro: Prim { rho, vel: Tensor::new(vel), pre },
                mag: Tensor::new(mag),
            };
            let (wgam2, bmu, ptot) = rmhd_source_quantities(&eos, &prim);
            // the inertial + magnetic geometric sources need ALL `ncomp` (DOF) components, not
            // just the `ndim` gridded ones: a 2.5D spherical (r,theta) grid has DOF=3 and the
            // out-of-plane phi momentum (mom[2]) drives the S_theta cot term + the S_phi source.
            let gas_mom: Vec<Gv> = (0..ncomp).map(|k| wgam2 * vel[k]).collect();
            let vel_n: Vec<Gv> = vel[..ncomp].to_vec();
            let bmu_n: Vec<Gv> = (0..ncomp).map(|k| bmu[k]).collect();
            geometric_momentum_sources_gv(
                coords, axes, ndim, ncomp, geo, ptot, &gas_mom, &vel_n, Some(&bmu_n),
            )
        }
        GeoSource::NewtonianMhd => {
            // newtonian MHD stress: ptot = p + 1/2|B|^2; gas inertial via cons.mom (= rho v,
            // pure gas); magnetic tension via the lab-frame B (the Maxwell stress -B_i B_j has
            // the SAME christoffel form as the inertial, so it reuses the inertial builder, then
            // is subtracted by geometric_momentum_sources_gv). no wgam2 / four-vector.
            // ALL `ncomp` (DOF) components: a 2.5D spherical grid (DOF=3 > ndim=2) needs the
            // out-of-plane phi velocity/momentum/B for the S_theta cot + S_phi geometric sources.
            let vel: Vec<Gv> = (0..ncomp)
                .map(|d| Gv::field(&format!("prim_v{d}"), FieldRef::PrimVel(d as u8)))
                .collect();
            let mag: [Gv; 3] = std::array::from_fn(|k| mag_field(k));
            let pre = Gv::field("prim_pre", FieldRef::PrimPre);
            let bsq = mag[0] * mag[0] + mag[1] * mag[1] + mag[2] * mag[2];
            let ptot = pre + Gv::from_f64(0.5) * bsq;
            let gas_mom: Vec<Gv> = cons_mom[..ncomp].to_vec();
            let mag_n: Vec<Gv> = (0..ncomp).map(|k| mag[k]).collect();
            geometric_momentum_sources_gv(
                coords, axes, ndim, ncomp, geo, ptot, &gas_mom, &vel, Some(&mag_n),
            )
        }
        GeoSource::IsothermalMhd => {
            // isothermal MHD stress: ptot = cs^2 rho + 1/2|B|^2 (no prim.pre; cs is a scalar).
            // otherwise identical to NewtonianMhd (gas inertial via cons.mom, lab-frame B tension).
            // ALL `ncomp` (DOF) components (see NewtonianMhd) for the spherical 2.5D out-of-plane source.
            let vel: Vec<Gv> = (0..ncomp)
                .map(|d| Gv::field(&format!("prim_v{d}"), FieldRef::PrimVel(d as u8)))
                .collect();
            let mag: [Gv; 3] = std::array::from_fn(|k| mag_field(k));
            let rho = Gv::field("prim_rho", FieldRef::PrimRho);
            let cs = Gv::scalar("cs");
            let bsq = mag[0] * mag[0] + mag[1] * mag[1] + mag[2] * mag[2];
            let ptot = cs * cs * rho + Gv::from_f64(0.5) * bsq;
            let gas_mom: Vec<Gv> = cons_mom[..ncomp].to_vec();
            let mag_n: Vec<Gv> = (0..ncomp).map(|k| mag[k]).collect();
            geometric_momentum_sources_gv(
                coords, axes, ndim, ncomp, geo, ptot, &gas_mom, &vel, Some(&mag_n),
            )
        }
    }
}


/// SPIKE probe: trace the carrier-generic `symbi_hydro::UniformAccel` source at S=Gv.
/// constructs the source with `g_ext_k` runtime scalars (the same names the splice path
/// declares), reads rho/vel as cell fields, and writes `s_mom_k` + `s_nrg`. a host test
/// renders + evaluates this and asserts the analytical `rho*g_ext` / `rho*(v.g_ext)` — the
/// SAME result `uniform_acceleration_*_source` produces via its hand-built graph, proving the
/// carrier-generic form is a drop-in for the splice path (and is f64==Gv by construction).
pub fn uniform_accel_probe_gv<const D: usize>() -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let rho = Gv::field("rho", FieldRef::cons_den());
    let vel: [Gv; D] =
        std::array::from_fn(|k| Gv::field(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8)));
    let g_ext: [Gv; D] = std::array::from_fn(|k| Gv::scalar(&format!("g_ext_{k}")));
    let src = symbi_hydro::UniformAccel::<Gv, D> { g_ext };
    let mom = src.momentum(rho);
    let nrg = src.energy(rho, &vel);
    let mut writes: Vec<(String, FieldBind, NodeId)> = (0..D)
        .map(|k| (format!("s_mom_{k}"), format!("s_mom_{k}").into(), mom[k].node()))
        .collect();
    writes.push(("s_nrg".to_string(), "s_nrg".into(), nrg.node()));
    (end_trace(), writes)
}


/// splice an externally-lowered user-expression `BuiltSource` (a parsed script, bridged into the
/// `symbi-ir` Graph via `symbi_hydro::expr_bridge`, optionally wrapped in a conservation law by
/// `source_spec::user_force_*` / `user_cooling_source`) into a Gv trace — binding each declared
/// param to a runtime Gv scalar of the same name — and write its outputs `s_k`. a user expression
/// FUSES into a kernel graph and RENDERS (CPU + CUDA) through the exact same `splice_built_source_into`
/// path a built-in source uses: the user script becomes compiled kernel code, not a per-cell
/// register-VM walk. carrier-equivalence + the work-energy coupling are
/// gated by `source_term_carrier.rs`.
pub fn splice_user_source_gv(
    built: &symbi_hydro::source_spec::BuiltSource,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    // bind every declared param (x_k, t, p_i, ...) to a runtime Gv scalar of the same name;
    // in production the position `x_k` binds to the in-kernel centroid instead.
    let mut name_to_node = std::collections::HashMap::new();
    for p in &built.params {
        name_to_node.insert(p.clone(), Gv::scalar(p).node());
    }
    let outs = with_trace(|t| {
        symbi_hydro::source_spec::splice_built_source_into(built, t.graph(), &name_to_node)
    });
    let writes = outs
        .iter()
        .enumerate()
        .map(|(k, &n)| (format!("s_{k}"), format!("s_{k}").into(), n))
        .collect();
    (end_trace(), writes)
}


/// SPIKE probe: trace the carrier-generic `symbi_hydro::PointMassGravity` source at S=Gv.
/// reads rho/vel as cell fields and the position `x_k`, mass position `xm_k`, and `gm` as
/// runtime scalars (the same names the splice path declares); writes `s_mom_k` + `s_nrg`.
/// a host test renders + evaluates it and asserts `-rho*GM*(x-xm)/|x-xm|^3` — the SAME form
/// `point_mass_{momentum,energy}_source` hand-builds, proving the carrier-generic form is a
/// drop-in (and f64==Gv by construction). the shared `1/|x-xm|^3` is emitted once (hash-cons).
pub fn point_mass_gravity_probe_gv<const D: usize>() -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let rho = Gv::field("rho", FieldRef::cons_den());
    let vel: [Gv; D] =
        std::array::from_fn(|k| Gv::field(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8)));
    let x: [Gv; D] = std::array::from_fn(|k| Gv::scalar(&format!("x_{k}")));
    let xm: [Gv; D] = std::array::from_fn(|k| Gv::scalar(&format!("xm_{k}")));
    let gm = Gv::scalar("gm");
    let eps = Gv::scalar("eps");
    let src = symbi_hydro::PointMassGravity::<Gv, D> { gm, xm, eps };
    let mom = src.momentum(rho, &x);
    let nrg = src.energy(rho, &vel, &x);
    let mut writes: Vec<(String, FieldBind, NodeId)> = (0..D)
        .map(|k| (format!("s_mom_{k}"), format!("s_mom_{k}").into(), mom[k].node()))
        .collect();
    writes.push(("s_nrg".to_string(), "s_nrg".into(), nrg.node()));
    (end_trace(), writes)
}


/// the gv inertial-source probe:
/// read the conserved momentum + primitive velocity, compute the centrifugal/coriolis source
/// `S^i = -Gamma^i_jk mom^j v^k` from the in-kernel volume-weighted centroid, write `s_d`. a
/// host test bit-diffs it against the analytic `mom_t v_t / r` forms. identity axes (natural).
pub fn inertial_momentum_probe_gv(
    coords: Coords,
    spacing: &[Spacing],
    ndim: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let axes: Vec<usize> = (0..ndim).collect();
    let mom: Vec<Gv> = (0..ndim)
        .map(|d| Gv::field(&format!("cons_mom_{d}"), FieldRef::cons_mom(d as u8)))
        .collect();
    let vel: Vec<Gv> = (0..ndim)
        .map(|d| Gv::field(&format!("prim_v{d}"), FieldRef::PrimVel(d as u8)))
        .collect();
    let geo = cell_geometry_gv(coords, spacing, &axes, ndim);
    let s = inertial_momentum_sources_gv(ndim, coords, &mom, &vel, &geo.centroid);
    let writes = (0..ndim).map(|d| (format!("s_{d}"), format!("s_{d}").into(), s[d].node())).collect();
    (end_trace(), writes)
}


/// the gv FULL geometric-momentum-source probe — the carrier mirror of the ctx
/// `geometric_momentum_sources` path (+ the rmhd adapter): build the cell geometry, form the
/// per-component source `S^i = -Gamma^i_jk T^jk` via [`gv_geometric_source`], write `s_k`. a host
/// test bit-diffs it against the analytic pressure + inertial (+ rmhd magnetic) forms. `axes`
/// carries the role map (the cyl r-z swirl `[0, 2]` with `ncomp > ndim`).
pub fn geometric_momentum_source_probe_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: usize,
    ncomp: usize,
    source: GeoSource,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let geo = cell_geometry_gv(coords, spacing, axes, ndim);
    // hydro shares the conserved momentum (the gas inertial reads cons.mom); rmhd computes its
    // gas momentum density from prim (cons.mom carries B-momentum), so it reads no cons.mom.
    let cons_mom: Vec<Gv> = match source {
        // hydro + newtonian MHD: cons.mom IS the gas momentum density (rho v), read directly.
        // read ALL `ncomp` (DOF) components, not just `ndim`: a 2.5D spherical (r,theta) MHD grid
        // has DOF=3 and the geometric S_theta/S_phi need the out-of-plane phi momentum (mom[2]).
        // hydro has ncomp==ndim so this is unchanged there.
        GeoSource::Hydro { .. } | GeoSource::NewtonianMhd | GeoSource::IsothermalMhd => (0..ncomp)
            .map(|k| Gv::field(&format!("mom_{k}"), FieldRef::cons_mom(k as u8)))
            .collect(),
        GeoSource::Rmhd => Vec::new(),
    };
    let s = gv_geometric_source(coords, axes, ndim, ncomp, &geo, source, &cons_mom, false);
    let writes = (0..ncomp).map(|k| (format!("s_{k}"), format!("s_{k}").into(), s[k].node())).collect();
    (end_trace(), writes)
}
