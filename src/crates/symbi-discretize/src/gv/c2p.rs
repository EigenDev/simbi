// =============================================================================
// c2p.rs
//
// cons->prim recovery kernel builders (adiabatic / iso / rhd / rmhd / nmhd / imhd).
// =============================================================================

use super::*;
use symbi_hydro::spatial_metric::{Gamma, GammaInv, SpatialMetric};
use symbi_geometry::{KerrKS, KerrKSCartesian, KerrKSCylindrical, Metric, Schwarzschild, SchwarzschildKS, SchwarzschildKSCartesian, SchwarzschildKSCylindrical};

/// trace the REAL adiabatic (ideal-gas) c2p — symbi-hydro's `Cons::to_primitive` at
/// `S = Gv` — into a dispatchable kernel. the carrier-generic physics IS the kernel
/// builder; this is what replaces the hand-written `adiabatic_c2p` Expr builder. returns
/// the `GvKernel` (graph + ABI manifest) and the `(write_key, runtime_path, root)` writes.
/// note: the `Regime::to_primitive` WRAPPER's native error-code branches are host-only
/// diagnostics — the kernel traces only the branch-free math `Cons::to_primitive`.
pub fn adiabatic_c2p_gv<const D: usize>() -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    // input binding: the conserved fields + the eos scalar, as Gv leaves.
    let den = Gv::field("cons_den", FieldRef::cons_den());
    let mom: Vec<Gv> = (0..D)
        .map(|k| Gv::field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8)))
        .collect();
    let nrg = Gv::field("cons_nrg", FieldRef::cons_nrg());
    let gamma = Gv::scalar("gamma");

    // the SINGLE-SOURCE physics, instantiated at the tracing carrier.
    let mom_arr: [Gv; D] = mom.try_into().expect("D momentum components");
    let cons = Cons::<Gv, D> { den, mom: Tensor::new(mom_arr), nrg };
    let prim: Prim<Gv, D> = cons.to_primitive(&IdealGas { gamma });

    // decompose the recovered primitive into field writes.
    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), prim.rho.node())];
    for k in 0..D {
        writes.push((format!("prim_vel_{k}"), FieldRef::PrimVel(k as u8).into(), prim.vel[k].node()));
    }
    writes.push(("prim_pre".to_string(), FieldRef::PrimPre.into(), prim.pre.node()));

    (end_trace(), writes)
}


/// trace the REAL isothermal c2p — symbi-hydro's `IsoNewtonian::to_primitive` (the pure
/// `rho = den`, `vel = mom / rho` kinematics) plus the `Isothermal::pressure` closure
/// `p = cs^2 * rho` — at `S = Gv`. replaces the hand-written `iso_c2p` Expr builder.
///
/// `IsoModel`'s `prim.pre` is a ZST: the HOST runtime elides pressure storage and
/// recomputes `cs^2 * rho` in the flux. the SUBSTRATE stores a real `prim.pre` field
/// (the iso face flux PLM-reconstructs it), so the materialized closure is traced
/// explicitly here — the value (`cs^2 * rho`) is the single source either way.
///
/// iso c2p is geometry-independent and ncomp == ndim (the cyl r-z swirl has no iso c2p),
/// so the `<D>` instance is a complete drop-in: no geom branch, no retained builder.
pub fn iso_c2p_gv<const D: usize>() -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    // input binding: the conserved fields + the prescribed per-cell sound-speed-squared
    // field `cs2` (the local temperature; global isothermal is a uniform cs2). NO scalar —
    // cs2 is a FIELD so the run can be LOCALLY isothermal (cs varies per cell).
    let den = Gv::field("cons_den", FieldRef::cons_den());
    let mom: Vec<Gv> = (0..D)
        .map(|k| Gv::field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8)))
        .collect();
    let cs2 = Gv::field("cs2", "cs2");

    // the SINGLE-SOURCE physics: symbi-hydro's LOCALLY-isothermal recovery (state.rs
    // `locally_isothermal_recover`) — `Cons::to_primitive` with the Isothermal eos reads
    // cs^2 from the nrg slot: rho = den, vel = mom/rho, p = recover_pressure = cs2 * rho.
    // the cs2 is the separate prescribed field, fed through the compute struct's nrg slot.
    let mom_arr: [Gv; D] = mom.try_into().expect("D momentum components");
    let cons = Cons::<Gv, D> { den, mom: Tensor::new(mom_arr), nrg: cs2 };
    let prim = cons.to_primitive(&Isothermal { cs: Gv::ONE }); // cs unused: recover reads nrg

    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), prim.rho.node())];
    for k in 0..D {
        writes.push((format!("prim_vel_{k}"), FieldRef::PrimVel(k as u8).into(), prim.vel[k].node()));
    }
    writes.push(("prim_pre".to_string(), FieldRef::PrimPre.into(), prim.pre.node()));

    (end_trace(), writes)
}


/// the isothermal eos law `p = cs^2(x) * rho` as a standalone pointwise kernel, from the
/// PRIMITIVE density. c2p derives the substrate pressure from the conserved state over
/// the interior only, but coarse-fine ghost cells receive prim rho by prolongation and
/// carry NO conserved state — the pressure there must be re-derived from the prolonged
/// rho or the face reconstruction sees a spurious vacuum at every level seam. pointwise
/// and dimension-independent (emitted per ndim like the snapshot family).
pub fn iso_pre_gv() -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let rho = Gv::field("prim_rho", FieldRef::PrimRho);
    let cs2 = Gv::field("cs2", "cs2");
    // the materialized `Isothermal::pressure` closure, same single source as iso c2p.
    let pre = cs2 * rho;
    (
        end_trace(),
        vec![("prim_pre".to_string(), FieldRef::PrimPre.into(), pre.node())],
    )
}


/// trace the REAL RHD c2p — symbi-hydro's branch-free `rhd_recover` (the iterative
/// relativistic cons->prim: a carrier-generic Newton on the pressure root, then the
/// algebraic velocity/Lorentz/density recovery) at `S = Gv`. the Newton lowers to one
/// `Op::IterateInline` (body traced once); `max_iters` bakes the fixed loop count. this
/// is the FIRST iterative gv kernel — replaces the hand-written `rhd_c2p` Expr builder.
///
/// numerically equivalent within ULP; the values differ in the last bits because the builder
/// hand-cancels rho in `c2`/`h` while the EOS-generic form keeps
/// `eos.pressure`/`sound_speed_sq`/explicit `h`.
/// the host wrapper's input guard + post-hoc diagnostics are host-only — the kernel
/// computes the raw recovery, exactly as the substrate already does.
pub fn rhd_c2p_gv<const D: usize>(max_iters: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    // input binding: the conserved fields + the eos scalar, as Gv leaves.
    let den = Gv::field("cons_den", FieldRef::cons_den());
    let mom: Vec<Gv> = (0..D)
        .map(|k| Gv::field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8)))
        .collect();
    let nrg = Gv::field("cons_nrg", FieldRef::cons_nrg());
    let gamma = Gv::scalar("gamma");

    // the SINGLE-SOURCE physics, instantiated at the tracing carrier.
    let mom_arr: [Gv; D] = mom.try_into().expect("D momentum components");
    let cons = Cons::<Gv, D> { den, mom: Tensor::new(mom_arr), nrg };
    // flat-frame spatial metric = identity (constant-folds to the euclidean norm, so the
    // traced/compiled kernel is bit-identical). the GR metric threads in here.
    let prim = rhd_recover(&IdealGas { gamma }, &cons, &SpatialMetric::flat(), max_iters);

    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), prim.rho.node())];
    for k in 0..D {
        writes.push((format!("prim_vel_{k}"), FieldRef::PrimVel(k as u8).into(), prim.vel[k].node()));
    }
    writes.push(("prim_pre".to_string(), FieldRef::PrimPre.into(), prim.pre.node()));

    (end_trace(), writes)
}


/// the RHD cons->prim on a curved SPATIAL metric — the `_schw`/`_ks` GR path. identical to
/// `rhd_c2p_gv` except the recovery contracts with the REAL spatial metric gamma(r) at the cell
/// (not identity): `|S|^2 = gamma^{ij} S_i S_j` and the recovered `v^i = gamma^{ij} S_j / (tau+D+p)`
/// is the CONTRAVARIANT velocity (Valencia). the metric is evaluated at the volume-weighted radial
/// centroid — the SAME cell radius the godunov densitization lapse uses — so the covariant conserved
/// `S_i` round-trips. reduces to `rhd_c2p_gv` bit-for-bit at identity gamma. `D` is the momentum
/// DOF (all D components contracted), the GRID dimension is `axes.len()` — they differ for the
/// spherical swirl (DOF = 3 azimuthal momentum on a 2D (r, theta) grid). reads
/// `schwarzschild_mass` + the grid scalars for the cell centroid.
pub fn rhd_c2p_gr_gv<const D: usize>(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    axes: &[usize],
    max_iters: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>)
where
    Schwarzschild<Gv>: Metric<Gv, D>,
    SchwarzschildKS<Gv>: Metric<Gv, D>,
    SchwarzschildKSCartesian<Gv>: Metric<Gv, D>,
    KerrKSCartesian<Gv>: Metric<Gv, D>,
    KerrKSCylindrical<Gv>: Metric<Gv, D>,
    SchwarzschildKSCylindrical<Gv>: Metric<Gv, D>,
    KerrKS<Gv>: Metric<Gv, D>,
{
    begin_trace();
    let den = Gv::field("cons_den", FieldRef::cons_den());
    let mom: [Gv; D] = std::array::from_fn(|k| Gv::field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8)));
    let nrg = Gv::field("cons_nrg", FieldRef::cons_nrg());
    let gamma = Gv::scalar("gamma");

    // the in-kernel spatial metric at the cell centroid — the SAME volume-weighted centroid the
    // godunov densitization lapse uses, so the covariant `S_i` stored under `to_conserved` inverts
    // exactly (well-balanced). gridded coordinate slots take the cell centroid; an ungridded
    // symmetry slot (the axisymmetric phi of the spherical swirl) takes zero — the spherical
    // metrics never read phi, and gamma_{phi phi} = r^2 sin^2(theta) needs only the GRIDDED
    // (r, theta). a suppressed POLAR slot would zero sin(theta) (singular gamma) — rejected.
    let ndim = axes.len();
    let geo = cell_geometry_gv(coords, spacing, axes, ndim);
    let x = Tensor::<Gv, D>::new(std::array::from_fn(|c| {
        match axes.iter().position(|&a| a == c) {
            Some(d) => geo.centroid[d],
            None => gv_ungridded_slot(coords, c),
        }
    }));
    let mass = Gv::scalar("schwarzschild_mass");
    // the covariant energy ehat = alpha tau + (alpha-1) D - beta^i S_i is what the godunov evolves,
    // so the recovery harvests the cell lapse + shift to invert it back to the Valencia tau the
    // newton consumes: tau = (ehat + (1-alpha) D + beta^i S_i) / alpha.
    let (gm, gm_inv, alpha, beta) = match (spacetime, coords) {
        (Spacetime::Schwarzschild, _) => {
            let m = Schwarzschild { mass };
            (m.spatial_metric(x), m.spatial_metric_inv(x), m.lapse(x), m.shift(x))
        }
        (Spacetime::KerrSchild, Coords::Cartesian) => {
            let m = SchwarzschildKSCartesian { mass };
            (m.spatial_metric(x), m.spatial_metric_inv(x), m.lapse(x), m.shift(x))
        }
        (Spacetime::KerrSchild, Coords::Cylindrical) => {
            let m = SchwarzschildKSCylindrical { mass };
            (m.spatial_metric(x), m.spatial_metric_inv(x), m.lapse(x), m.shift(x))
        }
        (Spacetime::KerrSchild, _) => {
            let m = SchwarzschildKS { mass };
            (m.spatial_metric(x), m.spatial_metric_inv(x), m.lapse(x), m.shift(x))
        }
        // spinning kerr on the CARTESIAN chart: the rank-1 kerr-schild update with the
        // oblate-spheroidal radius; non-diagonal gamma + shift on every axis.
        (Spacetime::Kerr, Coords::Cartesian) => {
            let m = KerrKSCartesian { mass, spin: Gv::scalar("kerr_spin") };
            (m.spatial_metric(x), m.spatial_metric_inv(x), m.lapse(x), m.shift(x))
        }
(Spacetime::Kerr, Coords::Cylindrical) => {
            let m = KerrKSCylindrical { mass, spin: Gv::scalar("kerr_spin") };
            (m.spatial_metric(x), m.spatial_metric_inv(x), m.lapse(x), m.shift(x))
        }
        (Spacetime::Kerr, _) => {
            // spinning kerr: non-diagonal gamma_{r phi} — only the azimuthal-momentum (swirl,
            // D = 3) instantiation carries the metric; the D = 1/2 arms are unreachable at bake.
            let m = KerrKS { mass, spin: Gv::scalar("kerr_spin") };
            (m.spatial_metric(x), m.spatial_metric_inv(x), m.lapse(x), m.shift(x))
        }
        (Spacetime::Minkowski, _) => unreachable!("the GR c2p is baked only for a curved spacetime"),
    };
    let metric = SpatialMetric::<Gv, D>::new(Gamma::new(gm), GammaInv::new(gm_inv));

    let mom_t = Tensor::new(mom);
    let tau = (nrg + (Gv::ONE - alpha) * den + beta.dot(&mom_t)) / alpha;
    let cons = Cons::<Gv, D> { den, mom: mom_t, nrg: tau };
    let prim = rhd_recover(&IdealGas { gamma }, &cons, &metric, max_iters);

    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), prim.rho.node())];
    for k in 0..D {
        writes.push((format!("prim_vel_{k}"), FieldRef::PrimVel(k as u8).into(), prim.vel[k].node()));
    }
    writes.push(("prim_pre".to_string(), FieldRef::PrimPre.into(), prim.pre.node()));
    (end_trace(), writes)
}


/// trace the REAL RMHD c2p — symbi-hydro's branch-free `rmhd_recover` (the KKC
/// false-position: a 6-state bracketed iterate over `kkc_fmu44` + `find_mu_plus`,
/// Illinois half-damp, sticky `done`) at `S = Gv`. the LAST + hardest c2p: the
/// bracketed solve lowers to a multi-accumulator `Op::IterateInline` via the new
/// `Scalar::iterate_vec`. replaces the hand-written `rmhd_c2p` Expr builder.
///
/// RMHD vectors are ALWAYS 3-component (the physics is 3D; grid symmetry handles the
/// 1D/2D cases), so this always traces `rmhd_recover::<Gv, 3>` — `ndim` only selects
/// the emit grid loop. reads the 8-field conserved (den, mom_{0,1,2}, nrg, mag_{0,1,2})
/// + gamma; writes (rho, vel_{0,1,2}, pre). B passes through (CT-evolved, so c2p does not recover it).
pub fn rmhd_c2p_gv(max_iters: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    // input binding, in the substrate's field-read order: den, mom, nrg (tau), mag, gamma.
    let den = Gv::field("cons_den", FieldRef::cons_den());
    let mom: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8)));
    let nrg = Gv::field("cons_nrg", FieldRef::cons_nrg());
    let mag: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("cons_mag_{k}"), &format!("cons.mag_{k}")));
    let gamma = Gv::scalar("gamma");

    // the SINGLE-SOURCE physics at the tracing carrier (3-component RMHD state).
    let cons = MhdCons::<Gv, 3> {
        hydro: Cons { den, mom: Tensor::new(mom), nrg },
        mag: Tensor::new(mag),
    };
    let prim = rmhd_recover(&IdealGas { gamma }, &cons, &SpatialMetric::flat(), max_iters);

    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), prim.rho.node())];
    for k in 0..3 {
        writes.push((format!("prim_vel_{k}"), FieldRef::PrimVel(k as u8).into(), prim.vel[k].node()));
    }
    writes.push(("prim_pre".to_string(), FieldRef::PrimPre.into(), prim.pre.node()));

    (end_trace(), writes)
}


/// the RMHD cons->prim on a curved SPATIAL metric — the metric-aware KKC recovery
/// (`|r|^2 = gamma^{ij} r_i r_j`, `B^2 = gamma_ij h^i h^j`, contravariant `v^i` raised) with
/// gamma at the cell's VOLUME-WEIGHTED centroid — the same point the covariant `to_conserved`
/// stored at, so the round-trip is exact (well-balanced). the metric evaluates at its FULL
/// three coordinates regardless of the grid dimension (RMHD vectors are always 3-component):
/// gridded slots take the centroid, the ungridded polar slot the exact equatorial pi/2, the
/// azimuthal slot zero. spinning kerr requires the gridded polar axis.
pub fn rmhd_c2p_gr_gv(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    axes: &[usize],
    max_iters: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let den = Gv::field("cons_den", FieldRef::cons_den());
    let mom: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8)));
    let nrg = Gv::field("cons_nrg", FieldRef::cons_nrg());
    let mag: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("cons_mag_{k}"), &format!("cons.mag_{k}")));
    let gamma_eos = Gv::scalar("gamma");

    let ndim = axes.len();
    let geo = cell_geometry_gv(coords, spacing, axes, ndim);
    let x = Tensor::<Gv, 3>::new(std::array::from_fn(|c| {
        match axes.iter().position(|&a| a == c) {
            Some(d) => geo.centroid[d],
            None => gv_ungridded_slot(coords, c),
        }
    }));
    let mass = Gv::scalar("schwarzschild_mass");
    // the covariant energy ehat = alpha tau + (alpha-1) D - beta^i S_i is what the godunov evolves,
    // so the recovery harvests the cell lapse + shift to invert it back to the Valencia tau the KKC
    // c2p consumes: tau = (ehat + (1-alpha) D + beta^i S_i) / alpha.
    let (gm, gm_inv, alpha, beta) = match (spacetime, coords) {
        (Spacetime::Schwarzschild, _) => {
            let m = Schwarzschild { mass };
            (m.spatial_metric(x), m.spatial_metric_inv(x), m.lapse(x), m.shift(x))
        }
        (Spacetime::KerrSchild, Coords::Cartesian) => {
            let m = SchwarzschildKSCartesian { mass };
            (m.spatial_metric(x), m.spatial_metric_inv(x), m.lapse(x), m.shift(x))
        }
        (Spacetime::KerrSchild, Coords::Cylindrical) => {
            let m = SchwarzschildKSCylindrical { mass };
            (m.spatial_metric(x), m.spatial_metric_inv(x), m.lapse(x), m.shift(x))
        }
        (Spacetime::KerrSchild, _) => {
            let m = SchwarzschildKS { mass };
            (m.spatial_metric(x), m.spatial_metric_inv(x), m.lapse(x), m.shift(x))
        }
        // spinning kerr on the CARTESIAN chart: the rank-1 kerr-schild update with the
        // oblate-spheroidal radius; non-diagonal gamma + shift on every axis.
        (Spacetime::Kerr, Coords::Cartesian) => {
            // spinning kerr: theta-dependent non-diagonal gamma (Sigma = r^2 + a^2 cos^2 theta), so
            // the polar axis must be GRIDDED — the swirl 2D (r, theta) bake grids it (the
            // equatorial-pi/2 fallback above would drop the a^2 cos^2 theta term). D = 3 swirl only.
            let m = KerrKSCartesian { mass, spin: Gv::scalar("kerr_spin") };
            (m.spatial_metric(x), m.spatial_metric_inv(x), m.lapse(x), m.shift(x))
        }
(Spacetime::Kerr, Coords::Cylindrical) => {
            // spinning kerr: theta-dependent non-diagonal gamma (Sigma = r^2 + a^2 cos^2 theta), so
            // the polar axis must be GRIDDED — the swirl 2D (r, theta) bake grids it (the
            // equatorial-pi/2 fallback above would drop the a^2 cos^2 theta term). D = 3 swirl only.
            let m = KerrKSCylindrical { mass, spin: Gv::scalar("kerr_spin") };
            (m.spatial_metric(x), m.spatial_metric_inv(x), m.lapse(x), m.shift(x))
        }
        (Spacetime::Kerr, _) => {
            // spinning kerr: theta-dependent non-diagonal gamma (Sigma = r^2 + a^2 cos^2 theta), so
            // the polar axis must be GRIDDED — the swirl 2D (r, theta) bake grids it (the
            // equatorial-pi/2 fallback above would drop the a^2 cos^2 theta term). D = 3 swirl only.
            let m = KerrKS { mass, spin: Gv::scalar("kerr_spin") };
            (m.spatial_metric(x), m.spatial_metric_inv(x), m.lapse(x), m.shift(x))
        }
        (Spacetime::Minkowski, _) => unreachable!("the GRMHD c2p is baked only for a curved spacetime"),
    };
    let metric = SpatialMetric::new(Gamma::new(gm), GammaInv::new(gm_inv));

    let mom_t = Tensor::new(mom);
    let tau = (nrg + (Gv::ONE - alpha) * den + beta.dot(&mom_t)) / alpha;
    let cons = MhdCons::<Gv, 3> {
        hydro: Cons { den, mom: mom_t, nrg: tau },
        mag: Tensor::new(mag),
    };
    let prim = rmhd_recover(&IdealGas { gamma: gamma_eos }, &cons, &metric, max_iters);

    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), prim.rho.node())];
    for k in 0..3 {
        writes.push((format!("prim_vel_{k}"), FieldRef::PrimVel(k as u8).into(), prim.vel[k].node()));
    }
    writes.push(("prim_pre".to_string(), FieldRef::PrimPre.into(), prim.pre.node()));

    (end_trace(), writes)
}


// =============================================================================
// newtonian MHD — the non-relativistic ideal-MHD regime. ALGEBRAIC c2p (no
// iteration -> no current-sheet failure mode), closed-form fast-magnetosonic
// wave speeds (cheap enough to compute inline in the flux from the reconstructed
// face states; NO per-cell materialization needed, unlike the RMHD quartic). all
// three builders trace the SAME `NewtonianMhd` carrier-generic physics validated
// at f64 in symbi-hydro. B passes through c2p unchanged (CT-evolved, so c2p does not recover it).
// =============================================================================

/// trace the newtonian-MHD c2p — the carrier-safe algebraic `nmhd_recover` at
/// `S = Gv`. binds cons (den, mom, nrg, mag) + gamma; writes the recovered hydro
/// (rho, vel, pre). the host-side `to_primitive` error codes are NOT traced (the
/// math is branch-free; comparisons stay on the host). reads `cons_mag_k` because
/// recovering the gas pressure requires stripping 1/2|B|^2 from the total energy.
pub fn nmhd_c2p_gv() -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let den = Gv::field("cons_den", FieldRef::cons_den());
    let mom: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8)));
    let nrg = Gv::field("cons_nrg", FieldRef::cons_nrg());
    let mag: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("cons_mag_{k}"), &format!("cons.mag_{k}")));
    let gamma = Gv::scalar("gamma");

    let cons = MhdCons::<Gv, 3> {
        hydro: Cons { den, mom: Tensor::new(mom), nrg },
        mag: Tensor::new(mag),
    };
    let prim = nmhd_recover(&IdealGas { gamma }, &cons);

    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), prim.rho.node())];
    for k in 0..3 {
        writes.push((format!("prim_vel_{k}"), FieldRef::PrimVel(k as u8).into(), prim.vel[k].node()));
    }
    writes.push(("prim_pre".to_string(), FieldRef::PrimPre.into(), prim.pre.node()));

    (end_trace(), writes)
}


// =============================================================================
// ISOTHERMAL MHD gv builders — the same shapes as the NMHD ones, but over the
// energy-model-generic state at E = IsoModel: the conserved vector is {den, mom,
// mag} (NO nrg), c2p is trivial (rho = den, v = mom/den, no pressure), and the
// closure is p = cs^2 rho (Isothermal EOS, scalar `cs` replaces `gamma`). the
// flux is `IsothermalMhd::to_flux` -> HLLE / the 3-state `hlld_isothermal`.
// =============================================================================

/// trace the isothermal-MHD c2p — trivial inversion (rho = den, v = mom/den); no
/// energy/pressure output. the single source the substrate c2p kernel renders.
pub fn imhd_c2p_gv() -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let den = Gv::field("cons_den", FieldRef::cons_den());
    let mom: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8)));
    let mag: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("cons_mag_{k}"), &format!("cons.mag_{k}")));

    let cons = IsoMhdCons::<Gv, 3> {
        hydro: ConsG { den, mom: Tensor::new(mom), nrg: Zero::default() },
        mag: Tensor::new(mag),
    };
    // imhd_recover ignores the EOS (pure kinematics); Gv::ZERO -> no `cs` param.
    let prim = imhd_recover(&Isothermal { cs: Gv::ZERO }, &cons);

    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), prim.rho.node())];
    for k in 0..3 {
        writes.push((format!("prim_vel_{k}"), FieldRef::PrimVel(k as u8).into(), prim.vel[k].node()));
    }
    // no prim.pre — the isothermal closure has no independent pressure.
    (end_trace(), writes)
}
