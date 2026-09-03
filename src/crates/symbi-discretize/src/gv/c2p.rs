// =============================================================================
// c2p.rs
//
// cons->prim recovery kernel builders (adiabatic / iso / rhd / rmhd / nmhd / imhd).
// =============================================================================

use super::*;
use symbi_algebra::Matrix;
use symbi_geometry::{
    KerrKS, KerrKSCartesian, KerrKSCylindrical, Metric, SchwarzschildKS, SchwarzschildKSCartesian,
    SchwarzschildKSCylindrical,
};
use symbi_hydro::eos::Eos as _;
use symbi_hydro::quantity::{Density, EnergyDensity, Pressure, SoundSpeedSquared, VelocitySquared};
use symbi_hydro::spatial_metric::{Gamma, GammaInv, SpatialMetric};
use symbi_hydro::{KernelC2pStatus, traced_recovery};
use symbi_ir::{GvMask, KernelWrite, KernelWrites};

/// the c2p status channel renderer: the typed accept/reject fact materializes
/// in the field vocabulary the diagnostics speak — zero accepted,
/// `ErrorCode::INVALID_PRIMITIVE` rejected. the write rides the recovery
/// kernel itself, so the candidate fields and the status channel leave one
/// trace together and a rejected pressure is data on its own channel.
fn c2p_status_write<'t>(status: KernelC2pStatus<GvMask<'t>>) -> KernelWrite {
    let code = Gv::select(
        status.accepted(),
        Gv::ZERO,
        Gv::from_f64(symbi_hydro::c2p_result::ErrorCode::INVALID_PRIMITIVE.0 as f64),
    );
    KernelWrite::new("c2p_status", FieldRef::Scratch, code.node())
}

/// trace the real adiabatic (ideal-gas) c2p — symbi-hydro's `Cons::to_primitive` at
/// `S = Gv` — into a dispatchable kernel. the carrier-generic physics is the kernel
/// builder; this is what replaces the hand-written `adiabatic_c2p` Expr builder. returns
/// the `GvKernel` (graph + ABI manifest) and its named write effects.
/// note: the `Regime::to_primitive` wrapper's native error-code branches are host-only
/// diagnostics — the kernel traces the branch-free math `Cons::to_primitive`.
pub fn adiabatic_c2p_gv<const D: usize>() -> (GvKernel, KernelWrites) {
    trace(|cx| {
        // input binding: the conserved fields + the eos scalar, as Gv leaves.
        let den = cx.field("cons_den", FieldRef::cons_den());
        let mom: Vec<Gv> = (0..D)
            .map(|k| cx.field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8)))
            .collect();
        let nrg = cx.field("cons_nrg", FieldRef::cons_nrg());
        let gamma = cx.scalar("gamma");

        // the single-source physics, instantiated at the tracing carrier.
        let mom_arr: [Gv; D] = mom.try_into().expect("D momentum components");
        let cons = Cons::<Gv, D>::adiabatic(Density(den), Tensor::new(mom_arr), EnergyDensity(nrg));
        let prim: Prim<Gv, D> = cons.to_primitive(&IdealGas { gamma });
        let (prim, c2p_status) = traced_recovery::newtonian(prim).into_parts();

        // decompose the recovered primitive into field writes.
        let mut writes = vec![KernelWrite::new(
            "prim_rho",
            FieldRef::PrimRho,
            prim.rho().node(),
        )];
        for k in 0..D {
            writes.push(KernelWrite::new(
                format!("prim_vel_{k}"),
                FieldRef::PrimVel(k as u8),
                prim.vel()[k].node(),
            ));
        }
        writes.push(KernelWrite::new(
            "prim_pre",
            FieldRef::PrimPre,
            prim.pre().node(),
        ));

        writes.push(c2p_status_write(c2p_status));

        writes
    })
}

/// trace the real isothermal c2p — symbi-hydro's `IsoNewtonian::to_primitive` (the pure
/// `rho = den`, `vel = mom / rho` kinematics) plus the `Isothermal::pressure` closure
/// `p = cs^2 * rho` — at `S = Gv`. replaces the hand-written `iso_c2p` Expr builder.
///
/// `IsoModel`'s `prim.pre` is a zst: the host runtime elides pressure storage and
/// recomputes `cs^2 * rho` in the flux. the substrate stores a real `prim.pre` field
/// (the iso face flux PLM-reconstructs it), so the materialized closure is traced
/// explicitly here — the value (`cs^2 * rho`) is the single source either way.
///
/// iso c2p is geometry-independent and ncomp == ndim (the cyl r-z swirl, with DOF > ndim,
/// falls outside its coverage), so the `<D>` instance is a complete drop-in: one
/// geometry-free builder serves every iso grid.
pub fn iso_c2p_gv<const D: usize>() -> (GvKernel, KernelWrites) {
    trace(|cx| {
        // input binding: the conserved fields + the prescribed per-cell sound-speed-squared
        // field `cs2` (the local temperature; global isothermal is a uniform cs2). cs2 is bound
        // as a field so the run can be locally isothermal (cs varies per cell).
        let den = cx.field("cons_den", FieldRef::cons_den());
        let mom: Vec<Gv> = (0..D)
            .map(|k| cx.field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8)))
            .collect();
        let cs2 = cx.field("cs2", "cs2");

        // the single-source physics: symbi-hydro's locally-isothermal recovery —
        // rho = den, vel = mom/rho (the `Cons::to_primitive` operation order), and the
        // pressure through the isothermal closure's recovery-quantity door,
        // p = recover_pressure(SoundSpeedSquared(cs2)) = cs2 * rho. the cs2 is the
        // separate prescribed field, entering as the sound-speed-squared quantity it is
        // (an adiabatic Cons cannot carry it: its nrg slot stores an energy density).
        let mom_arr: [Gv; D] = mom.try_into().expect("D momentum components");
        let mom_t = Tensor::new(mom_arr);
        let rho = den;
        let vel = mom_t.map(|m| m / rho);
        let v2 = vel.dot(&vel);
        let pre = Isothermal { cs: Gv::ONE }.recover_pressure(
            Density(rho),
            VelocitySquared(v2),
            SoundSpeedSquared(cs2),
        ); // cs unused: recovery consumes the prescribed cs2
        let prim = Prim::adiabatic(Density(rho), vel, Pressure(pre));
        let (prim, c2p_status) = traced_recovery::isothermal(prim).into_parts();

        let mut writes = vec![KernelWrite::new(
            "prim_rho",
            FieldRef::PrimRho,
            prim.rho().node(),
        )];
        for k in 0..D {
            writes.push(KernelWrite::new(
                format!("prim_vel_{k}"),
                FieldRef::PrimVel(k as u8),
                prim.vel()[k].node(),
            ));
        }
        writes.push(KernelWrite::new(
            "prim_pre",
            FieldRef::PrimPre,
            prim.pre().node(),
        ));

        writes.push(c2p_status_write(c2p_status));

        writes
    })
}

/// the isothermal eos law `p = cs^2(x) * rho` as a standalone pointwise kernel, from the
/// primitive density. c2p derives the substrate pressure from the conserved state over
/// the interior; coarse-fine ghost cells receive prim rho by prolongation and carry the
/// primitives alone, so the pressure there is re-derived from the prolonged rho — otherwise
/// the face reconstruction sees a spurious vacuum at every level seam. pointwise
/// and dimension-independent (emitted per ndim like the snapshot family).
pub fn iso_pre_gv() -> (GvKernel, KernelWrites) {
    trace(|cx| {
        let rho = cx.field("prim_rho", FieldRef::PrimRho);
        let cs2 = cx.field("cs2", "cs2");
        // the materialized `Isothermal::pressure` closure, same single source as iso c2p.
        let pre = cs2 * rho;
        vec![KernelWrite::new("prim_pre", FieldRef::PrimPre, pre.node())]
    })
}

/// trace the real RHD c2p — symbi-hydro's branch-free `rhd_recover` (the iterative
/// relativistic cons->prim: a carrier-generic newton on the pressure root, then the
/// algebraic velocity/lorentz/density recovery) at `S = Gv`. the newton lowers to one
/// `Op::IterateInline` (body traced once); `max_iters` bakes the fixed loop count. this
/// is the first iterative gv kernel — replaces the hand-written `rhd_c2p` Expr builder.
///
/// numerically equivalent within ULP; the values differ in the last bits because the builder
/// hand-cancels rho in `c2`/`h` while the EOS-generic form keeps
/// `eos.pressure`/`sound_speed_sq`/explicit `h`.
/// the host wrapper's input guard + post-hoc diagnostics are host-only — the kernel
/// computes the raw recovery, exactly as the substrate already does.
pub fn rhd_c2p_gv<const D: usize>(max_iters: usize, eos_arm: EosArm) -> (GvKernel, KernelWrites) {
    trace(|cx| {
        // input binding: the conserved fields + the eos scalar, as Gv leaves.
        let den = cx.field("cons_den", FieldRef::cons_den());
        let mom: Vec<Gv> = (0..D)
            .map(|k| cx.field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8)))
            .collect();
        let nrg = cx.field("cons_nrg", FieldRef::cons_nrg());
        let gamma = cx.scalar("gamma");

        // the single-source physics, instantiated at the tracing carrier.
        let mom_arr: [Gv; D] = mom.try_into().expect("D momentum components");
        let cons = Cons::<Gv, D>::adiabatic(Density(den), Tensor::new(mom_arr), EnergyDensity(nrg));
        // flat-frame spatial metric = identity (constant-folds to the euclidean norm, so the
        // traced/compiled kernel is bit-identical). the GR metric threads in here. the newton
        // is eos-generic — the closure arm selects gamma-law or taub-mathews at trace time
        // (gamma stays bound on both arms; under the synge closure it is bound-but-inert).
        let prim = rhd_recover(
            &super::gv_eos(eos_arm, gamma),
            &cons,
            &SpatialMetric::flat(),
            max_iters,
        );

        let (prim, c2p_status) =
            traced_recovery::relativistic(prim, &SpatialMetric::flat()).into_parts();

        let mut writes = vec![KernelWrite::new(
            "prim_rho",
            FieldRef::PrimRho,
            prim.rho().node(),
        )];
        for k in 0..D {
            writes.push(KernelWrite::new(
                format!("prim_vel_{k}"),
                FieldRef::PrimVel(k as u8),
                prim.vel()[k].node(),
            ));
        }
        writes.push(KernelWrite::new(
            "prim_pre",
            FieldRef::PrimPre,
            prim.pre().node(),
        ));

        writes.push(c2p_status_write(c2p_status));

        writes
    })
}

/// the RHD cons->prim on a curved spacetime — the `_schw`/`_ks` GR path. it undensitizes the
/// evolved state by the known measure `sqrt(-g)(x)` and then runs the recovery contracted with the
/// spatial metric gamma(r) carried at the cell: `|S|^2 = gamma^{ij} S_i S_j` and the
/// recovered `v^i = gamma^{ij} S_j / (tau+D+p)` is the contravariant velocity (valencia). the metric
/// is evaluated at the volume-weighted centroid — the point the seeding densitizes at — so the
/// state round-trips per cell. reduces to `rhd_c2p_gv` at identity gamma. `D` is the momentum
/// DOF (all D components contracted), the grid dimension is `axes.len()` — they differ for the
/// spherical swirl (DOF = 3 azimuthal momentum on a 2D (r, theta) grid). reads
/// `schwarzschild_mass` + the grid scalars for the cell centroid.
pub fn rhd_c2p_gr_gv<const D: usize>(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    axes: &[usize],
    max_iters: usize,
) -> (GvKernel, KernelWrites)
where
    for<'t> SchwarzschildKS<Gv<'t>>: Metric<Gv<'t>, D>,
    for<'t> SchwarzschildKSCartesian<Gv<'t>>: Metric<Gv<'t>, D>,
    for<'t> KerrKSCartesian<Gv<'t>>: Metric<Gv<'t>, D>,
    for<'t> KerrKSCylindrical<Gv<'t>>: Metric<Gv<'t>, D>,
    for<'t> SchwarzschildKSCylindrical<Gv<'t>>: Metric<Gv<'t>, D>,
    for<'t> KerrKS<Gv<'t>>: Metric<Gv<'t>, D>,
{
    trace(|cx| {
        let den = cx.field("cons_den", FieldRef::cons_den());
        let mom: [Gv; D] = std::array::from_fn(|k| {
            cx.field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8))
        });
        let nrg = cx.field("cons_nrg", FieldRef::cons_nrg());
        let gamma = cx.scalar("gamma");

        // the in-kernel spatial metric at the cell's arithmetic midpoint — the point the seeding
        // densitizes at and the godunov evaluates the connection source at, so the stored state
        // inverts exactly. the densitized law's cell average is over the plain coordinate volume, so
        // the midpoint is its second-order sampling point, where the area-weighted law would read the
        // chart's volume-weighted centroid. gridded coordinate slots take that midpoint; an ungridded
        // symmetry slot (the axisymmetric phi of the spherical swirl) takes zero — the spherical
        // metrics read the gridded (r, theta) alone, gamma_{phi phi} = r^2 sin^2(theta). a suppressed
        // polar slot would zero sin(theta) (singular gamma) — rejected.
        let ndim = axes.len();
        let mid = gv_cell_midpoints(cx, spacing, ndim);
        let x = Tensor::<Gv, D>::new(std::array::from_fn(|c| {
            match axes.iter().position(|&a| a == c) {
                Some(d) => mid[d],
                None => gv_ungridded_slot(coords, c),
            }
        }));
        // the evolved state is the densitized sqrt(-g)[rho u^t, T^t_i, -(T^t_t + rho u^t)], so the
        // recovery harvests the cell lapse, shift and full-chart measure `volume_factor`: undensitize
        // by the known sqrt(det gamma)(x), then invert the killing energy back to the valencia tau the
        // newton consumes, tau = (ehat + (1-alpha) D + beta^i S_i) / alpha.
        // spinning kerr on the cartesian chart: the rank-1 kerr-schild update with the
        // oblate-spheroidal radius; non-diagonal gamma + shift on every axis. on the spherical
        // chart the non-diagonal gamma_{r phi} means the azimuthal-momentum (swirl, D = 3)
        // instantiation is the one carrying the metric; the D = 1/2 arms are unreachable at bake.
        let (gm, gm_inv, alpha, beta, sqrt_gamma) = {
            fn adm<'t, const N: usize, M: Metric<Gv<'t>, N>>(
                m: &M,
                x: Tensor<Gv<'t>, N>,
            ) -> (
                Matrix<Gv<'t>, N>,
                Matrix<Gv<'t>, N>,
                Gv<'t>,
                Tensor<Gv<'t>, N>,
                Gv<'t>,
            ) {
                (
                    m.spatial_metric(x),
                    m.spatial_metric_inv(x),
                    m.lapse(x),
                    m.shift(x),
                    m.volume_factor(x),
                )
            }
            with_ks_metric!(cx, spacetime, coords, "the GR c2p", |m| adm(&m, x))
        };
        let metric = SpatialMetric::<Gv, D>::new(Gamma::new(gm), GammaInv::new(gm_inv));

        let inv_dens = Gv::ONE / sqrt_gamma;
        let den = den * inv_dens;
        let nrg = nrg * inv_dens;
        let mom_t = Tensor::new(mom).scale(inv_dens);
        let tau = (nrg + (Gv::ONE - alpha) * den + beta.dot(&mom_t)) / alpha;
        let cons = Cons::<Gv, D>::adiabatic(Density(den), mom_t, EnergyDensity(tau));
        let prim = rhd_recover(&IdealGas { gamma }, &cons, &metric, max_iters);
        let (prim, c2p_status) = traced_recovery::relativistic(prim, &metric).into_parts();

        let mut writes = vec![KernelWrite::new(
            "prim_rho",
            FieldRef::PrimRho,
            prim.rho().node(),
        )];
        for k in 0..D {
            writes.push(KernelWrite::new(
                format!("prim_vel_{k}"),
                FieldRef::PrimVel(k as u8),
                prim.vel()[k].node(),
            ));
        }
        writes.push(KernelWrite::new(
            "prim_pre",
            FieldRef::PrimPre,
            prim.pre().node(),
        ));
        writes.push(c2p_status_write(c2p_status));

        writes
    })
}

/// trace the real RMHD c2p — symbi-hydro's branch-free `rmhd_recover` (the KKC
/// false-position: a 6-state bracketed iterate over `kkc_fmu44` + `find_mu_plus`,
/// illinois half-damp, sticky `done`) at `S = Gv`. the last and hardest c2p: the
/// bracketed solve lowers to a multi-accumulator `Op::IterateInline` via the new
/// `Scalar::iterate_vec`. replaces the hand-written `rmhd_c2p` Expr builder.
///
/// RMHD vectors are 3-component on every grid (the physics is 3D; grid symmetry handles the
/// 1D/2D cases), so this always traces `rmhd_recover::<Gv, 3>` — `ndim` selects the emit grid
/// loop. reads the 8-field conserved (den, mom_{0,1,2}, nrg, mag_{0,1,2})
/// + gamma; writes (rho, vel_{0,1,2}, pre). B passes through, recovered by the CT evolution.
pub fn rmhd_c2p_gv(max_iters: usize) -> (GvKernel, KernelWrites) {
    trace(|cx| {
        // input binding, in the substrate's field-read order: den, mom, nrg (tau), mag, gamma.
        let den = cx.field("cons_den", FieldRef::cons_den());
        let mom: [Gv; 3] = std::array::from_fn(|k| {
            cx.field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8))
        });
        let nrg = cx.field("cons_nrg", FieldRef::cons_nrg());
        let mag: [Gv; 3] =
            std::array::from_fn(|k| cx.field(&format!("cons_mag_{k}"), &format!("cons.mag_{k}")));
        let gamma = cx.scalar("gamma");

        // the single-source physics at the tracing carrier (3-component RMHD state).
        let cons = MhdCons::<Gv, 3>::new(
            Cons::adiabatic(Density(den), Tensor::new(mom), EnergyDensity(nrg)),
            Tensor::new(mag),
        );
        let prim = rmhd_recover(
            &IdealGas { gamma },
            &cons,
            &SpatialMetric::flat(),
            max_iters,
        );

        let (prim, c2p_status) =
            traced_recovery::relativistic_mhd(prim, &SpatialMetric::flat()).into_parts();

        let mut writes = vec![KernelWrite::new(
            "prim_rho",
            FieldRef::PrimRho,
            prim.rho().node(),
        )];
        for k in 0..3 {
            writes.push(KernelWrite::new(
                format!("prim_vel_{k}"),
                FieldRef::PrimVel(k as u8),
                prim.vel()[k].node(),
            ));
        }
        writes.push(KernelWrite::new(
            "prim_pre",
            FieldRef::PrimPre,
            prim.pre().node(),
        ));

        writes.push(c2p_status_write(c2p_status));

        writes
    })
}

/// the RMHD cons->prim on a curved spatial metric — the metric-aware KKC recovery
/// (`|r|^2 = gamma^{ij} r_i r_j`, `B^2 = gamma_ij h^i h^j`, contravariant `v^i` raised) with
/// gamma at the cell's volume-weighted centroid — the same point the covariant `to_conserved`
/// stored at, so the round-trip is exact (well-balanced). the metric evaluates at its full
/// three coordinates regardless of the grid dimension (RMHD vectors are always 3-component):
/// gridded slots take the centroid, the ungridded polar slot the exact equatorial pi/2, the
/// azimuthal slot zero. spinning kerr requires the gridded polar axis.
pub fn rmhd_c2p_gr_gv(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    axes: &[usize],
    max_iters: usize,
) -> (GvKernel, KernelWrites) {
    trace(|cx| {
        let den = cx.field("cons_den", FieldRef::cons_den());
        let mom: [Gv; 3] = std::array::from_fn(|k| {
            cx.field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8))
        });
        let nrg = cx.field("cons_nrg", FieldRef::cons_nrg());
        let mag: [Gv; 3] =
            std::array::from_fn(|k| cx.field(&format!("cons_mag_{k}"), &format!("cons.mag_{k}")));
        let gamma_eos = cx.scalar("gamma");

        let ndim = axes.len();
        let geo = cell_geometry_gv(cx, coords, spacing, axes, ndim);
        let x = Tensor::<Gv, 3>::new(std::array::from_fn(|c| {
            match axes.iter().position(|&a| a == c) {
                Some(d) => geo.centroid[d],
                None => gv_ungridded_slot(coords, c),
            }
        }));
        // the covariant energy ehat = alpha tau + (alpha-1) D - beta^i S_i is what the godunov evolves,
        // so the recovery harvests the cell lapse + shift to invert it back to the valencia tau the KKC
        // c2p consumes: tau = (ehat + (1-alpha) D + beta^i S_i) / alpha.
        // spinning kerr on the cartesian chart: the rank-1 kerr-schild update with the
        // oblate-spheroidal radius; non-diagonal gamma + shift on every axis. every spinning
        // kerr chart has a theta-dependent non-diagonal gamma (Sigma = r^2 + a^2 cos^2 theta),
        // so the polar axis must be gridded — the swirl 2D (r, theta) bake grids it (the
        // equatorial-pi/2 fallback above would drop the a^2 cos^2 theta term). D = 3 swirl only.
        let (gm, gm_inv, alpha, beta) = {
            fn adm<'t, M: Metric<Gv<'t>, 3>>(
                m: &M,
                x: Tensor<Gv<'t>, 3>,
            ) -> (
                Matrix<Gv<'t>, 3>,
                Matrix<Gv<'t>, 3>,
                Gv<'t>,
                Tensor<Gv<'t>, 3>,
            ) {
                (
                    m.spatial_metric(x),
                    m.spatial_metric_inv(x),
                    m.lapse(x),
                    m.shift(x),
                )
            }
            with_ks_metric!(cx, spacetime, coords, "the GRMHD c2p", |m| adm(&m, x))
        };
        let metric = SpatialMetric::new(Gamma::new(gm), GammaInv::new(gm_inv));

        let mom_t = Tensor::new(mom);
        let tau = (nrg + (Gv::ONE - alpha) * den + beta.dot(&mom_t)) / alpha;
        let cons = MhdCons::<Gv, 3>::new(
            Cons::adiabatic(Density(den), mom_t, EnergyDensity(tau)),
            Tensor::new(mag),
        );
        let prim = rmhd_recover(&IdealGas { gamma: gamma_eos }, &cons, &metric, max_iters);
        let (prim, c2p_status) = traced_recovery::relativistic_mhd(prim, &metric).into_parts();

        let mut writes = vec![KernelWrite::new(
            "prim_rho",
            FieldRef::PrimRho,
            prim.rho().node(),
        )];
        for k in 0..3 {
            writes.push(KernelWrite::new(
                format!("prim_vel_{k}"),
                FieldRef::PrimVel(k as u8),
                prim.vel()[k].node(),
            ));
        }
        writes.push(KernelWrite::new(
            "prim_pre",
            FieldRef::PrimPre,
            prim.pre().node(),
        ));

        writes.push(c2p_status_write(c2p_status));

        writes
    })
}

// =============================================================================
// newtonian MHD — the non-relativistic ideal-MHD regime. algebraic c2p (closed-form,
// so it holds through the current sheets where an iterate fails), closed-form
// fast-magnetosonic wave speeds (cheap enough to compute inline in the flux from the
// reconstructed face states, where the RMHD quartic needs a per-cell materialization). all
// three builders trace the one `NewtonianMhd` carrier-generic physics validated
// at f64 in symbi-hydro. B passes through c2p unchanged; the CT evolution advances it.
// =============================================================================

/// trace the newtonian-MHD c2p — the carrier-safe algebraic `nmhd_recover` at
/// `S = Gv`. binds cons (den, mom, nrg, mag) + gamma; writes the recovered hydro
/// (rho, vel, pre). the host-side `to_primitive` error codes stay on the host (the
/// traced math is branch-free, comparisons living with the caller). reads `cons_mag_k` because
/// recovering the gas pressure requires stripping 1/2|B|^2 from the total energy.
pub fn nmhd_c2p_gv() -> (GvKernel, KernelWrites) {
    trace(|cx| {
        let den = cx.field("cons_den", FieldRef::cons_den());
        let mom: [Gv; 3] = std::array::from_fn(|k| {
            cx.field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8))
        });
        let nrg = cx.field("cons_nrg", FieldRef::cons_nrg());
        let mag: [Gv; 3] =
            std::array::from_fn(|k| cx.field(&format!("cons_mag_{k}"), &format!("cons.mag_{k}")));
        let gamma = cx.scalar("gamma");

        let cons = MhdCons::<Gv, 3>::new(
            Cons::adiabatic(Density(den), Tensor::new(mom), EnergyDensity(nrg)),
            Tensor::new(mag),
        );
        let prim = nmhd_recover(&IdealGas { gamma }, &cons);
        let (prim, c2p_status) = traced_recovery::newtonian_mhd(prim).into_parts();

        let mut writes = vec![KernelWrite::new(
            "prim_rho",
            FieldRef::PrimRho,
            prim.rho().node(),
        )];
        for k in 0..3 {
            writes.push(KernelWrite::new(
                format!("prim_vel_{k}"),
                FieldRef::PrimVel(k as u8),
                prim.vel()[k].node(),
            ));
        }
        writes.push(KernelWrite::new(
            "prim_pre",
            FieldRef::PrimPre,
            prim.pre().node(),
        ));

        writes.push(c2p_status_write(c2p_status));

        writes
    })
}

// =============================================================================
// isothermal MHD gv builders — the same shapes as the NMHD ones, over the
// energy-model-generic state at E = IsoModel: the conserved vector is {den, mom,
// mag}, c2p is trivial (rho = den, v = mom/den), and the
// closure supplies p = cs^2 rho (Isothermal EOS, scalar `cs` replaces `gamma`). the
// flux is `IsothermalMhd::to_flux` -> HLLE / the 3-state `hlld_isothermal`.
// =============================================================================

/// trace the isothermal-MHD c2p — trivial inversion (rho = den, v = mom/den), writing rho
/// and vel. the single source the substrate c2p kernel renders.
pub fn imhd_c2p_gv() -> (GvKernel, KernelWrites) {
    trace(|cx| {
        let den = cx.field("cons_den", FieldRef::cons_den());
        let mom: [Gv; 3] = std::array::from_fn(|k| {
            cx.field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8))
        });
        let mag: [Gv; 3] =
            std::array::from_fn(|k| cx.field(&format!("cons_mag_{k}"), &format!("cons.mag_{k}")));

        let cons = IsoMhdCons::<Gv, 3>::new(
            ConsG::isothermal(Density(den), Tensor::new(mom)),
            Tensor::new(mag),
        );
        // imhd_recover is pure kinematics, so the EOS argument is inert; Gv::zero keeps `cs`
        // out of the manifest.
        let prim = imhd_recover(&Isothermal { cs: Gv::ZERO }, &cons);
        let (prim, c2p_status) = traced_recovery::isothermal_mhd(prim).into_parts();

        let mut writes = vec![KernelWrite::new(
            "prim_rho",
            FieldRef::PrimRho,
            prim.rho().node(),
        )];
        for k in 0..3 {
            writes.push(KernelWrite::new(
                format!("prim_vel_{k}"),
                FieldRef::PrimVel(k as u8),
                prim.vel()[k].node(),
            ));
        }
        // the writes stop at vel — the isothermal closure sets the pressure from rho.
        writes.push(c2p_status_write(c2p_status));

        writes
    })
}
