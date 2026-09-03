// =============================================================================
// sources.rs
//
// geometric + user momentum/energy source kernel builders and their probes.
// =============================================================================

use super::*;
use symbi_geometry::{Cylindrical, CylindricalRPhi, Metric, Spherical};
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::spatial_metric::SpatialMetric;
use symbi_ir::{KernelProgram, KernelWrite, KernelWrites, trace_kernel};

// =============================================================================
// the conserved-update godunov family in Gv — the finite-volume divergence (the Gv stencil
// `field_shifted(F_i, +e_i) - field(F_i)`, spelled as a plain subtraction of loads) composed with
// the forward-euler / RK2 time update over the conserved set (mass + one scalar law per momentum
// component + optional energy), and the snapshot copy. EOS- and geometry-generic: snapshot is
// a pure copy (every coord); the cartesian-uniform divergence is `(F_hi - F_lo)/dx_i`, the
// curvilinear is the analytic area-weighted `(1/V)(F_hi*A_hi - F_lo*A_lo)` from the in-kernel
// `cell_geometry_gv` metric, plus the geometric momentum source `S^i = -Gamma^i_jk T^jk`
// (the gv christoffel) on the curvilinear momentum laws. one trace per (regime, geom).
// =============================================================================

/// which geometric momentum source the curvilinear godunov adds to the momentum laws. the
/// flux divergence + the conserved-law structure are identical across regimes; the source
/// expression is what differs (the "one operator, regime supplies its physics" rule). on
/// cartesian the connection vanishes and the source is zero.
#[derive(Clone, Copy)]
pub enum GeoSource {
    /// hydro / RHD: well-balanced pressure + (ndim>=2) velocity-quadratic centrifugal/coriolis,
    /// regime-agnostic via the conserved momentum (newtonian `mom=rho v`, RHD `mom=rho h W^2 v`).
    Hydro { inertial: bool },
    /// RMHD: pressure + inertial + magnetic tension, from `rmhd_source_quantities` (cons.mom
    /// carries B-momentum, so the gas momentum comes from those quantities).
    Rmhd,
    /// newtonian MHD: pressure (p + 1/2|B|^2) + gas inertial (cons.mom is rho v — the maxwell
    /// stress lives in the flux) + magnetic tension from the lab-frame B (where RMHD uses the
    /// four-vector b^mu). simpler than RMHD: cons.mom serves the inertial directly.
    NewtonianMhd,
    /// isothermal MHD: identical to `NewtonianMhd`, with the gas pressure `cs^2 rho` supplied by
    /// the closure (the state carries {den, mom, mag}); the closure scalar `cs` is read in-kernel.
    IsothermalMhd,
}

/// the centrifugal/coriolis inertial momentum source per component `S^i = -Gamma^i_jk mom^j
/// v^k` (the velocity-quadratic geometric terms), in Gv. delegates to the single-source
/// carrier-generic `Metric<Gv, D>::momentum_source_inertial` (symbi-geometry) — one christoffel
/// derivation, shared by the whole codebase.
///
/// regime-agnostic via the conserved `mom` (newtonian rho v; relativistic rho h W^2 v): the
/// source is the bilinear `-Gamma(mom, v)`, so this same call also serves the magnetic tension
/// `-Gamma(b, b)` (caller passes `mom = vel = b`). `centroid` is coordinate-indexed (r at [0],
/// theta/phi at [1]). dispatch is on the number of provided momentum components `mom.len()` (the
/// Hydro branch supplies the gridded `ndim`; the MHD branches supply the full `ncomp`); the
/// momentum slot order is coordinate order, so cylindrical 2-component is the (r,phi) disk plane
/// (e.g. an (r,z) grid with swirl carries [r, phi]) -> `CylindricalRPhi` (the (r,z)
/// `Cylindrical<2>` axisymmetric reduction would zero the swirl). the result is padded to
/// the full `ncomp` DOF — suppressed trailing components (e.g. z) carry zero inertial.
fn inertial_momentum_sources_gv<'t>(
    ncomp: usize,
    coords: Coords,
    mom: &[Gv<'t>],
    vel: &[Gv<'t>],
    centroid: &[Gv<'t>],
) -> Vec<Gv<'t>> {
    // const-D bridge: build the fixed-rank tensors from the runtime slices, evaluate the metric's
    // inertial source, splat back to a per-component Vec of length `mom.len()`.
    fn run<'t, M, const D: usize>(
        metric: M,
        mom: &[Gv<'t>],
        vel: &[Gv<'t>],
        x: &[Gv<'t>],
    ) -> Vec<Gv<'t>>
    where
        M: Metric<Gv<'t>, D>,
    {
        let s = metric.momentum_source_inertial(
            Tensor::from_fn(|i| x[i]),
            Tensor::from_fn(|i| mom[i]),
            Tensor::from_fn(|i| vel[i]),
        );
        (0..D).map(|i| s[i]).collect()
    }
    let mut s = match (coords, mom.len()) {
        (Coords::Cartesian, _) => Vec::new(), // flat space: the connection vanishes.
        (Coords::Spherical, 1) => run::<_, 1>(Spherical, mom, vel, centroid),
        (Coords::Spherical, 2) => run::<_, 2>(Spherical, mom, vel, centroid),
        (Coords::Spherical, 3) => run::<_, 3>(Spherical, mom, vel, centroid),
        (Coords::Cylindrical, 1) => run::<_, 1>(Cylindrical, mom, vel, centroid),
        (Coords::Cylindrical, 2) => run::<_, 2>(CylindricalRPhi, mom, vel, centroid),
        (Coords::Cylindrical, 3) => run::<_, 3>(Cylindrical, mom, vel, centroid),
        (c, d) => panic!("inertial source: unsupported (coords {c:?}, components {d})"),
    };
    // suppressed trailing DOF (e.g. z in an (r,phi)+z layout) carry zero inertial.
    s.resize(ncomp, Gv::ZERO);
    s
}

/// the full geometric momentum source per component `S^i = -Gamma^i_jk T^jk` in Gv, split
/// into the three pieces every
/// regime shares: well-balanced pressure `ptot*(A_hi - A_lo)*inv_V`, inertial `-Gamma(wgam2 v
/// v)`, and (RMHD) magnetic `+Gamma(bmu bmu)`. `gas_mom`/`vel`/`bmu` are the regime quantities;
/// an empty `gas_mom` selects the pressure-only form (1D radial). `axes[d]` = the coord of grid
/// axis d.
fn geometric_momentum_sources_gv<'t>(
    coords: Coords,
    axes: &[usize],
    ndim: usize,
    ncomp: usize,
    geo: &CellGeometryGv<'t>,
    ptot: Gv<'t>,
    gas_mom: &[Gv<'t>],
    vel: &[Gv<'t>],
    bmu: Option<&[Gv<'t>]>,
) -> Vec<Gv<'t>> {
    // coordinate-indexed centroid (r at [0], theta at [1]). ungridded slots take the chart symmetry
    // default (spherical polar -> pi/2): a reduced-dimension spherical grid (a 1.5D radial
    // chart with ungridded theta, or a 2.5D r-phi chart) still evaluates the angular christoffels
    // cot(theta)/sin(theta) in the inertial source, and theta = 0 diverges cot(theta) and NaNs the
    // state. `gv_ungridded_slot` is the single chart authority for these fills (spherical theta ->
    // pi/2, every other suppressed axis -> 0, so cartesian / cylindrical are unchanged).
    let mut coord_centroid: Vec<Gv> = (0..3).map(|k| gv_ungridded_slot(coords, k)).collect();
    for d in 0..ndim {
        coord_centroid[axes[d]] = geo.centroid[d];
    }
    let inertial = (!gas_mom.is_empty())
        .then(|| inertial_momentum_sources_gv(ncomp, coords, gas_mom, vel, &coord_centroid));
    // the magnetic tension is the same christoffel on the four-vector, negated.
    let mag = bmu.map(|b| inertial_momentum_sources_gv(ncomp, coords, b, b, &coord_centroid));
    (0..ncomp)
        .map(|coord| {
            // pressure: a gridded coordinate is the one carrying a pressure gradient;
            // written in the divergence's (ptot*A_hi - ptot*A_lo)*inv_V form so a v=0
            // uniform-ptot state cancels the pressure flux divergence bit-exactly
            // (well-balanced hse).
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
/// hydro/RHD: total pressure = `prim.pre`, gas momentum density * v = the conserved momentum
/// (cons.mom is rho v / rho h W^2 v), the stress being purely hydrodynamic. RMHD: the gas +
/// magnetic quantities from symbi-hydro's `rmhd_source_quantities` (the carrier-generic source
/// the RMHD flux uses), gas_mom = wgam2*v, bmu = the spatial four-vector. `cons_mom` is the
/// already-read in-place conserved momentum, shared so the gas-inertial reuses that one buffer.
pub(crate) fn gv_geometric_source<'t>(
    cx: TraceCx<'t>,
    coords: Coords,
    axes: &[usize],
    ndim: usize,
    ncomp: usize,
    geo: &CellGeometryGv<'t>,
    source: GeoSource,
    cons_mom: &[Gv<'t>],
    mag_from_bcell: bool,
) -> Vec<Gv<'t>> {
    // the cell-B the magnetic geo source reads: the primitive `prim.mag[k]` in the general case.
    // when this stage is fused with the cell-B predictor (which binds `bc_k` in-place), reading mag
    // via that same `bc_k` key lets try_fuse merge the two cell-B reads into one binding; keeping
    // them apart leaves `prim.mag[k]` and `bc_k` as distinct manifest entries that both resolve to
    // bcell[k] at runtime, aliasing a read-only input to an in-place output (ub on CPU). both keys
    // carry the same old-bcell value (the predictor writes after the source evaluates), so it is
    // bit-identical.
    let mag_field = |k: usize| {
        if mag_from_bcell {
            cx.field(&format!("bc_{k}"), FieldRef::BCell(k as u8))
        } else {
            cx.field(&format!("prim_b{k}"), FieldRef::PrimMag(k as u8))
        }
    };
    match source {
        GeoSource::Hydro { inertial } => {
            let ptot = cx.field("pre", FieldRef::PrimPre);
            // the velocity-quadratic inertial vanishes for 1D radial, so that arm leaves it and
            // its vel reads out of the graph.
            let (gas_mom, vel): (Vec<Gv>, Vec<Gv>) = if inertial && ndim >= 2 {
                let v = (0..ndim)
                    .map(|d| cx.field(&format!("prim_v{d}"), FieldRef::PrimVel(d as u8)))
                    .collect();
                (cons_mom[..ndim].to_vec(), v) // gas_mom = cons.mom (shared, read once)
            } else {
                (Vec::new(), Vec::new())
            };
            geometric_momentum_sources_gv(
                coords, axes, ndim, ncomp, geo, ptot, &gas_mom, &vel, None,
            )
        }
        GeoSource::Rmhd => {
            // the RMHD stress = pressure + gas inertial + magnetic tension: read prim + gamma,
            // trace symbi-hydro's `rmhd_source_quantities` (wgam2, bmu, ptot) at S=Gv.
            let rho = cx.field("prim_rho", FieldRef::PrimRho);
            let vel: [Gv; 3] = std::array::from_fn(|k| {
                cx.field(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8))
            });
            let pre = cx.field("prim_pre", FieldRef::PrimPre);
            let mag: [Gv; 3] = std::array::from_fn(|k| mag_field(k));
            let eos = IdealGas {
                gamma: cx.scalar("gamma"),
            };
            let prim = MhdPrim::<Gv, 3>::new(
                Prim::adiabatic(Density(rho), Tensor::new(vel), Pressure(pre)),
                Tensor::new(mag),
            );
            // flat-frame metric = identity (constant-folds to euclidean norms; traced kernel bit-identical).
            let (wgam2, bmu, ptot) = rmhd_source_quantities(&eos, &prim, &SpatialMetric::flat());
            // the inertial + magnetic geometric sources need every `ncomp` (DOF) component,
            // past the `ndim` gridded ones: a 2.5D spherical (r,theta) grid has DOF=3 and the
            // out-of-plane phi momentum (mom[2]) drives the S_theta cot term + the S_phi source.
            let gas_mom: Vec<Gv> = (0..ncomp).map(|k| wgam2 * vel[k]).collect();
            let vel_n: Vec<Gv> = vel[..ncomp].to_vec();
            let bmu_n: Vec<Gv> = (0..ncomp).map(|k| bmu[k]).collect();
            geometric_momentum_sources_gv(
                coords,
                axes,
                ndim,
                ncomp,
                geo,
                ptot,
                &gas_mom,
                &vel_n,
                Some(&bmu_n),
            )
        }
        GeoSource::NewtonianMhd => {
            // newtonian MHD stress: ptot = p + 1/2|B|^2; gas inertial via cons.mom (= rho v,
            // pure gas); magnetic tension via the lab-frame B (the maxwell stress -B_i B_j has
            // the same christoffel form as the inertial, so it reuses the inertial builder, then
            // is subtracted by geometric_momentum_sources_gv). the newtonian limit works from
            // rho v and the lab-frame B, where RMHD carries wgam2 and the four-vector.
            // every `ncomp` (DOF) component: a 2.5D spherical grid (DOF=3 > ndim=2) needs the
            // out-of-plane phi velocity/momentum/B for the S_theta cot + S_phi geometric sources.
            let vel: Vec<Gv> = (0..ncomp)
                .map(|d| cx.field(&format!("prim_v{d}"), FieldRef::PrimVel(d as u8)))
                .collect();
            let mag: [Gv; 3] = std::array::from_fn(|k| mag_field(k));
            let pre = cx.field("prim_pre", FieldRef::PrimPre);
            let bsq = mag[0] * mag[0] + mag[1] * mag[1] + mag[2] * mag[2];
            let ptot = pre + Gv::from_f64(0.5) * bsq;
            let gas_mom: Vec<Gv> = cons_mom[..ncomp].to_vec();
            let mag_n: Vec<Gv> = (0..ncomp).map(|k| mag[k]).collect();
            geometric_momentum_sources_gv(
                coords,
                axes,
                ndim,
                ncomp,
                geo,
                ptot,
                &gas_mom,
                &vel,
                Some(&mag_n),
            )
        }
        GeoSource::IsothermalMhd => {
            // isothermal MHD stress: ptot = cs^2 rho + 1/2|B|^2, the gas pressure coming from the
            // `cs` scalar and rho. otherwise identical to NewtonianMhd (gas inertial via cons.mom,
            // lab-frame B tension). every `ncomp` (DOF) component (see NewtonianMhd) for the
            // spherical 2.5D out-of-plane source.
            let vel: Vec<Gv> = (0..ncomp)
                .map(|d| cx.field(&format!("prim_v{d}"), FieldRef::PrimVel(d as u8)))
                .collect();
            let mag: [Gv; 3] = std::array::from_fn(|k| mag_field(k));
            let rho = cx.field("prim_rho", FieldRef::PrimRho);
            let cs = cx.scalar("cs");
            let bsq = mag[0] * mag[0] + mag[1] * mag[1] + mag[2] * mag[2];
            let ptot = cs * cs * rho + Gv::from_f64(0.5) * bsq;
            let gas_mom: Vec<Gv> = cons_mom[..ncomp].to_vec();
            let mag_n: Vec<Gv> = (0..ncomp).map(|k| mag[k]).collect();
            geometric_momentum_sources_gv(
                coords,
                axes,
                ndim,
                ncomp,
                geo,
                ptot,
                &gas_mom,
                &vel,
                Some(&mag_n),
            )
        }
    }
}

/// spike probe: trace the carrier-generic `symbi_hydro::UniformAccel` source at S=Gv.
/// constructs the source with `g_ext_k` runtime scalars (the same names the splice path
/// declares), reads rho/vel as cell fields, and writes `s_mom_k` + `s_nrg`. a host test
/// renders + evaluates this and asserts the analytical `rho*g_ext` / `rho*(v.g_ext)` — the
/// same result `uniform_acceleration_*_source` produces via its hand-built graph, proving the
/// carrier-generic form is a drop-in for the splice path (and is f64==Gv by construction).
pub fn uniform_accel_probe_gv<const D: usize>() -> KernelProgram {
    trace_kernel(|cx| {
        let rho = cx.field("rho", FieldRef::cons_den());
        let vel: [Gv; D] =
            std::array::from_fn(|k| cx.field(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8)));
        let g_ext: [Gv; D] = std::array::from_fn(|k| cx.scalar(&format!("g_ext_{k}")));
        let src = symbi_hydro::UniformAccel::<Gv, D> { g_ext };
        let mom = src.momentum(rho);
        let nrg = src.energy(rho, &vel);
        let mut writes: KernelWrites = (0..D)
            .map(|k| KernelWrite::new(format!("s_mom_{k}"), format!("s_mom_{k}"), mom[k].node()))
            .collect();
        writes.push(KernelWrite::new("s_nrg", "s_nrg", nrg.node()));
        writes
    })
}

/// splice an externally-lowered user-expression `SourceProgram` (a parsed script, bridged into the
/// `symbi-ir` Graph via `symbi_source_compile::expr_bridge`, optionally wrapped in a conservation law by
/// `source_spec::user_force_*` / `user_cooling_source`) into a Gv trace — binding each declared
/// param to a runtime Gv scalar of the same name — and write its outputs `s_k`. a user expression
/// fuses into a kernel graph and renders (CPU + CUDA) through the exact same `SourceProgram::splice_into`
/// path a built-in source uses: the user script becomes compiled kernel code.
/// carrier-equivalence + the work-energy coupling are gated by `source_term_carrier.rs`.
pub fn splice_user_source_gv(
    built: &symbi_source_compile::source_spec::SourceProgram,
) -> KernelProgram {
    trace_kernel(|cx| {
        // bind every declared param (x_k, t, p_i, ...) to a runtime Gv scalar of the same name;
        // in production the position `x_k` binds to the in-kernel centroid instead.
        let mut name_to_node = std::collections::HashMap::new();
        for p in built.params() {
            name_to_node.insert(p.clone(), cx.scalar(p).node());
        }
        let outs = cx.with_trace(|t| built.splice_into(t.graph(), &name_to_node));
        let writes = outs
            .iter()
            .enumerate()
            .map(|(k, &n)| KernelWrite::new(format!("s_{k}"), format!("s_{k}"), n))
            .collect();
        writes
    })
}

/// spike probe: trace the carrier-generic `symbi_hydro::PointMassGravity` source at S=Gv.
/// reads rho/vel as cell fields and the position `x_k`, mass position `xm_k`, and `gm` as
/// runtime scalars (the same names the splice path declares); writes `s_mom_k` + `s_nrg`.
/// a host test renders + evaluates it and asserts `-rho*GM*(x-xm)/|x-xm|^3` — the same form
/// `point_mass_{momentum,energy}_source` hand-builds, proving the carrier-generic form is a
/// drop-in (and f64==Gv by construction). the shared `1/|x-xm|^3` is emitted once (hash-cons).
pub fn point_mass_gravity_probe_gv<const D: usize>() -> KernelProgram {
    trace_kernel(|cx| {
        let rho = cx.field("rho", FieldRef::cons_den());
        let vel: [Gv; D] =
            std::array::from_fn(|k| cx.field(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8)));
        let x: [Gv; D] = std::array::from_fn(|k| cx.scalar(&format!("x_{k}")));
        let xm: [Gv; D] = std::array::from_fn(|k| cx.scalar(&format!("xm_{k}")));
        let gm = cx.scalar("gm");
        let eps = cx.scalar("eps");
        let src = symbi_hydro::PointMassGravity::<Gv, D> { gm, xm, eps };
        let mom = src.momentum(rho, &x);
        let nrg = src.energy(rho, &vel, &x);
        let mut writes: KernelWrites = (0..D)
            .map(|k| KernelWrite::new(format!("s_mom_{k}"), format!("s_mom_{k}"), mom[k].node()))
            .collect();
        writes.push(KernelWrite::new("s_nrg", "s_nrg", nrg.node()));
        writes
    })
}

/// the gv inertial-source probe:
/// read the conserved momentum + primitive velocity, compute the centrifugal/coriolis source
/// `S^i = -Gamma^i_jk mom^j v^k` from the in-kernel volume-weighted centroid, write `s_d`. a
/// host test bit-diffs it against the analytic `mom_t v_t / r` forms. identity axes (natural).
pub fn inertial_momentum_probe_gv(
    coords: Coords,
    spacing: &[Spacing],
    ndim: usize,
) -> KernelProgram {
    trace_kernel(|cx| {
        let axes: Vec<usize> = (0..ndim).collect();
        let mom: Vec<Gv> = (0..ndim)
            .map(|d| cx.field(&format!("cons_mom_{d}"), FieldRef::cons_mom(d as u8)))
            .collect();
        let vel: Vec<Gv> = (0..ndim)
            .map(|d| cx.field(&format!("prim_v{d}"), FieldRef::PrimVel(d as u8)))
            .collect();
        let geo = cell_geometry_gv(cx, coords, spacing, &axes, ndim);
        let s = inertial_momentum_sources_gv(ndim, coords, &mom, &vel, &geo.centroid);
        let writes = (0..ndim)
            .map(|d| KernelWrite::new(format!("s_{d}"), format!("s_{d}"), s[d].node()))
            .collect();
        writes
    })
}

/// the gv full geometric-momentum-source probe — the carrier mirror of the ctx
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
) -> KernelProgram {
    trace_kernel(|cx| {
        let geo = cell_geometry_gv(cx, coords, spacing, axes, ndim);
        // hydro shares the conserved momentum (the gas inertial reads cons.mom); rmhd computes its
        // gas momentum density from prim (cons.mom carries B-momentum), so its cons_mom list is empty.
        let cons_mom: Vec<Gv> = match source {
            // hydro + newtonian MHD: cons.mom is the gas momentum density (rho v), read directly.
            // read every `ncomp` (DOF) component: a 2.5D spherical (r,theta) MHD grid
            // has DOF=3 and the geometric S_theta/S_phi need the out-of-plane phi momentum (mom[2]).
            // hydro has ncomp==ndim so this is unchanged there.
            GeoSource::Hydro { .. } | GeoSource::NewtonianMhd | GeoSource::IsothermalMhd => (0
                ..ncomp)
                .map(|k| cx.field(&format!("mom_{k}"), FieldRef::cons_mom(k as u8)))
                .collect(),
            GeoSource::Rmhd => Vec::new(),
        };
        let s = gv_geometric_source(
            cx, coords, axes, ndim, ncomp, &geo, source, &cons_mom, false,
        );
        let writes = (0..ncomp)
            .map(|k| KernelWrite::new(format!("s_{k}"), format!("s_{k}"), s[k].node()))
            .collect();
        writes
    })
}
