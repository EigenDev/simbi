// =============================================================================
// gv_excise.rs
//
// the traced horizon-excision fill for the cartesian kerr-schild charts: three
// kernels that together freeze every excised cell at a cold vacuum floor
// (rho_floor, v = 0, p_floor), then rebuild the conserved state with the cell's
// own metric. the excised region is the sublevel set of the kerr-schild radius
// r_ks(x; a) < r_exc — the sphere about the chart origin at a = 0, the oblate
// spheroid (x^2 + y^2)/(r_exc^2 + a^2) + z^2/r_exc^2 < 1 at spin a about z (the
// r = const surfaces of the chart).
//
// the vacuum is a one-way absorbing accretion boundary: the exterior gas rarefies
// into the vacuum at the excision faces (material crosses in and stays), the
// physical horizon. a zero-gradient / outward-copy fill would instead be a
// transmissive outflow bc — on the staircased cartesian surface the per-axis sweep
// speeds carry both signs, so a transmissive interior leaks back into the
// exterior flux; accretion and outflow are different boundary conditions, and the
// absorber is the one a black hole obeys.
//
//   excise_fill      prim -> scratch = the vacuum-floor fill
//   excise_writeback scratch -> prim                    (the fill commit)
//   excise_p2c       prim -> cons, valencia to_conserved at the cell centroid,
//                    excised cells only (live cells pass their cons through)
//
// the fill carries the gas primitives alone (rho, v, p). the magnetized p2c
// reads the cell's own B (the face average the constrained transport owns) and
// rebuilds (D, S_i, tau) with it — the staggered faces keep the values transport
// gave them, so the densitized div(B) invariant survives excision by construction.
//
// the fill is pointwise and idempotent, so a single pass fills the region; the
// pair runs under a store-driven pass count and p2c runs once after it. the
// dispatch box is the excision region's index bbox — computed host-side, so the
// support lives in the runtime dispatch and the baked artifacts carry the arithmetic alone.
//
// usage (build.rs):
//   let (k, writes) = excise_fill_gv();
//   emit_gv(out, "excise_fill_2d", 2, &k, &writes);
// =============================================================================

use symbi_algebra::Tensor;
use symbi_algebra::algebra::Numeric;
use symbi_carrier::Scalar;
use symbi_geometry::{KerrKS, KerrKSCartesian, Metric};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::regime::Regime;
use symbi_hydro::spatial_metric::{Gamma, GammaInv, SpatialMetric};
use symbi_hydro::state::{Prim, Valencia};
use symbi_hydro::{MhdPrim, RhdGr, RmhdGr};
use symbi_ib::excise::ks_excised;
use symbi_ir::gv::{KernelWrite, KernelWrites};
use symbi_ir::{FieldRef, Gv, GvKernel, TraceCx, trace};

use crate::coords::{Coords, Spacetime, Spacing};
use crate::gv::cell_geometry_gv;
use crate::gv::geometry::{gv_cell_midpoints, gv_ungridded_slot};

/// the vacuum-floor primitive at storage slot `kk` of a `[rho, vel_0.., pre]` set of arity `nf`:
/// `rho_floor` at the density slot, `p_floor` at the pressure slot, zero velocity between.
fn vacuum_floor<'t>(cx: TraceCx<'t>, kk: usize, nf: usize) -> Gv<'t> {
    if kk == 0 {
        cx.scalar("excision_rho")
    } else if kk == nf - 1 {
        cx.scalar("excision_pre")
    } else {
        Gv::ZERO
    }
}

/// the gas-primitive FieldRefs in storage order: rho, vel_0..dof-1, pre.
fn prim_refs(dof: usize) -> Vec<FieldRef> {
    let mut refs = vec![FieldRef::PrimRho];
    for kk in 0..dof {
        refs.push(FieldRef::PrimVel(kk as u8));
    }
    refs.push(FieldRef::PrimPre);
    refs
}

fn prim_names(dof: usize) -> Vec<String> {
    let mut names = vec!["rho".to_string()];
    for kk in 0..dof {
        names.push(format!("vel_{kk}"));
    }
    names.push("pre".to_string());
    names
}

/// the cell centroid of the uniform cartesian grid, as traced coordinates.
fn centroid<'t>(cx: TraceCx<'t>, ndim: usize) -> Vec<Gv<'t>> {
    let spacing = vec![Spacing::Uniform; ndim];
    let axes: Vec<usize> = (0..ndim).collect();
    let geo = cell_geometry_gv(cx, Coords::Cartesian, &spacing, &axes, ndim);
    geo.centroid
}

/// the excision mask at the traced centroid: r_ks(x; a) < r_exc with the
/// host-filled `kerr_spin` / `excision_radius` scalars (spin = 0 for the
/// schwarzschild chart). one definition shared by the fill and the rebuild.
fn excised_mask_2d<'t>(cx: TraceCx<'t>, x: &[Gv<'t>; 2]) -> <Gv<'t> as Scalar>::Mask {
    ks_excised(x, cx.scalar("kerr_spin"), cx.scalar("excision_radius"))
}

fn excised_mask_3d<'t>(cx: TraceCx<'t>, x: &[Gv<'t>; 3]) -> <Gv<'t> as Scalar>::Mask {
    ks_excised(x, cx.scalar("kerr_spin"), cx.scalar("excision_radius"))
}

/// one 2d fill pass over `1 + dof + 1` gas primitives: every excised cell is
/// frozen at the cold vacuum floor; live cells copy their own state. writes the
/// filled state to the exc_0.. scratch (the commit is `excise_writeback`, so the
/// fill sees the pre-pass state at every read). `dof = 2` is the in-plane GR-hydro state; `dof = 3`
/// carries the out-of-plane momentum of the 2.5d MHD state.
fn excise_fill_2d_dof_gv(dof: usize) -> (GvKernel, KernelWrites) {
    trace(|cx| {
        let names = prim_names(dof);
        let refs = prim_refs(dof);
        let nf = refs.len();

        let own: Vec<Gv> = (0..nf).map(|kk| cx.field(&names[kk], refs[kk])).collect();

        let c = centroid(cx, 2);
        let x = [c[0], c[1]];
        let excised = excised_mask_2d(cx, &x);
        // the vacuum-floor sink: an excised (inside-horizon) cell is frozen at a cold c2p-safe vacuum;
        // live cells keep their own state. the boundary riemann then rarefies the exterior gas into the
        // vacuum -- a one-way absorbing accretion bc (material crosses in and stays), the physical horizon.
        let filled: Vec<Gv> = (0..nf)
            .map(|kk| Gv::select(excised, vacuum_floor(cx, kk, nf), own[kk]))
            .collect();

        let mut writes = KernelWrites::new();
        for (kk, val) in filled.iter().enumerate() {
            writes.push(KernelWrite::new(
                format!("exc_out_{kk}"),
                format!("exc_{kk}"),
                val.node(),
            ));
        }
        writes
    })
}

pub fn excise_fill_gv() -> (GvKernel, KernelWrites) {
    excise_fill_2d_dof_gv(2)
}

/// the 2.5d (dof = 3) gas fill: rho, vel_0..2, pre on the 2d grid — the
/// magnetized equatorial slice's momentum set.
pub fn excise_fill_dof3_gv() -> (GvKernel, KernelWrites) {
    excise_fill_2d_dof_gv(3)
}

/// the sweep commit: copy the exc scratch back into the primitive fields.
/// unmasked over the dispatch box — the fill wrote live cells' own values,
/// so the copy is the bitwise identity there.
fn excise_writeback_dof_gv(dof: usize) -> (GvKernel, KernelWrites) {
    trace(|cx| {
        let names = prim_names(dof);
        let refs = prim_refs(dof);
        let vals: Vec<Gv> = (0..refs.len())
            .map(|kk| cx.field(&format!("exc_{kk}"), format!("exc_{kk}")))
            .collect();
        let mut writes = KernelWrites::new();
        for kk in 0..refs.len() {
            writes.push(KernelWrite::new(
                format!("{}_out", names[kk]),
                refs[kk],
                vals[kk].node(),
            ));
        }
        writes
    })
}

pub fn excise_writeback_gv() -> (GvKernel, KernelWrites) {
    excise_writeback_dof_gv(2)
}

pub fn excise_writeback_dof3_gv() -> (GvKernel, KernelWrites) {
    excise_writeback_dof_gv(3)
}

/// the 1d radial commit (dof = 1): rho, vel_0, pre. the writeback is a chart-free scratch copy,
/// so this one serves the spherical row without a spacetime or geometry variant.
pub fn excise_writeback_dof1_gv() -> (GvKernel, KernelWrites) {
    excise_writeback_dof_gv(1)
}

/// rebuild the conserved state of every excised cell from its (just-filled)
/// primitives: the valencia `to_conserved` (covariant S_i = rho h W^2 gamma_ij v^j)
/// with the cartesian kerr-schild spatial metric at the cell's own centroid — a
/// conserved state copied from any other cell would carry that cell's metric
/// factors, so setting the primitives and rebuilding locally is the exact route. live cells pass their conserved
/// state through untouched (in-place select). the metric is the spinning-kerr
/// rank-1 form with the host-filled `kerr_spin` (zero for the a = 0 chart).
pub fn excise_p2c_gv() -> (GvKernel, KernelWrites) {
    trace(|cx| {
        let gamma = cx.scalar("gamma");
        let mass = cx.scalar("schwarzschild_mass");
        let spin = cx.scalar("kerr_spin");

        let rho = cx.field("rho", FieldRef::PrimRho);
        let vel: [Gv; 2] =
            std::array::from_fn(|kk| cx.field(&format!("vel_{kk}"), FieldRef::PrimVel(kk as u8)));
        let pre = cx.field("pre", FieldRef::PrimPre);
        let den = cx.field("den", FieldRef::cons_den());
        let mom: [Gv; 2] =
            std::array::from_fn(|kk| cx.field(&format!("mom_{kk}"), FieldRef::cons_mom(kk as u8)));
        let nrg = cx.field("nrg", FieldRef::cons_nrg());

        let c = centroid(cx, 2);
        let x = [c[0], c[1]];
        let excised = excised_mask_2d(cx, &x);

        let xt = Tensor::<Gv, 2>::new(x);
        let m = KerrKSCartesian { mass, spin };
        let metric = SpatialMetric::<Gv, 2>::new(
            Gamma::new(m.spatial_metric(xt)),
            GammaInv::new(m.spatial_metric_inv(xt)),
        );
        // the densitized storage sqrt(-g)[rho u^t, T^t_i, -(T^t_t + rho u^t)] reads the cell lapse,
        // shift and full-chart measure, so the excised fill carries the same block the flux/c2p use.
        let regime = RhdGr {
            metric,
            alpha: m.lapse(xt),
            shift: m.shift(xt),
            sqrt_gamma: m.volume_factor(xt),
        };
        let prim = Prim::<Gv, 2>::adiabatic(Density(rho), Tensor::new(vel), Pressure(pre));
        let cons = regime.to_conserved(&IdealGas { gamma }, &Valencia(prim)).0;

        let mut writes = KernelWrites::new();
        writes.push(KernelWrite::new(
            "den_out",
            FieldRef::cons_den(),
            Gv::select(excised, cons.den(), den).node(),
        ));
        for kk in 0..2 {
            writes.push(KernelWrite::new(
                format!("mom_out_{kk}"),
                FieldRef::cons_mom(kk as u8),
                Gv::select(excised, cons.mom()[kk], mom[kk]).node(),
            ));
        }
        writes.push(KernelWrite::new(
            "nrg_out",
            FieldRef::cons_nrg(),
            Gv::select(excised, cons.nrg(), nrg).node(),
        ));
        writes
    })
}

/// the traced metric-sampling position of a spherical cell: the per-axis midpoint on every
/// gridded axis, the symmetry value on every ungridded one (the polar slot of a 1d radial row is
/// pi/2 — suppressing it to zero would zero sin(theta) and make gamma_{phi phi} singular).
///
/// the midpoint is the sampling point, ahead of the chart's volume-weighted centroid: the excised
/// state is stored densitized, and a densitized cell average is taken over the plain coordinate
/// volume, whose second-order sampling point is the midpoint. the two differ by dr^2/(6r) on a
/// radial axis, and the recovery inverts at the midpoint — sampling the rebuild there is what
/// returns an excised cell's primitives as the floor it was frozen at.
///
/// the face positions this is built from are selected at runtime by `map_kind_{ax}`, so one
/// traced kernel serves a uniform and a log-radial grid alike.
fn sample_position_sph<'t, const D: usize>(cx: TraceCx<'t>, ndim: usize) -> Tensor<Gv<'t>, D> {
    let spacing = vec![Spacing::Uniform; ndim];
    let mid = gv_cell_midpoints(cx, &spacing, ndim);
    Tensor::<Gv, D>::new(std::array::from_fn(|c| {
        if c < ndim {
            mid[c]
        } else {
            gv_ungridded_slot(Coords::Spherical, c)
        }
    }))
}

/// the excision mask on a spherical chart: `r < r_exc` on the grid's own radial coordinate.
///
/// a plain radial comparison replaces the cartesian mask's quartic and its staircased surface. the
/// cartesian mask solves `r_ks(x; a)` because a sphere cut out of a cartesian lattice crosses every
/// axis obliquely; here `r` is a coordinate, so the excised region is an exact slab of the
/// innermost radial cells and its surface is a coordinate surface. the spin leaves that slab as it
/// stands: the oblate spheroid of the cartesian chart is the `r = const` surface of this one.
fn excised_mask_sph<'t>(cx: TraceCx<'t>, x_r: Gv<'t>) -> <Gv<'t> as Scalar>::Mask {
    x_r.cmp_lt(cx.scalar("excision_radius"))
}

/// the spherical-chart gas fill: every cell inside the horizon is frozen at the cold vacuum floor,
/// live cells keep their own state. the same absorbing boundary the cartesian charts use — the
/// exterior rarefies into the vacuum at the excision faces and stays there, which is the
/// physical content of a horizon: every characteristic points inward.
fn excise_fill_sph_dof_gv(ndim: usize, dof: usize) -> (GvKernel, KernelWrites) {
    trace(|cx| {
        let names = prim_names(dof);
        let refs = prim_refs(dof);
        let nf = refs.len();
        let own: Vec<Gv> = (0..nf).map(|kk| cx.field(&names[kk], refs[kk])).collect();

        let x = sample_position_sph::<3>(cx, ndim);
        let excised = excised_mask_sph(cx, x[0]);
        let filled: Vec<Gv> = (0..nf)
            .map(|kk| Gv::select(excised, vacuum_floor(cx, kk, nf), own[kk]))
            .collect();

        let mut writes = KernelWrites::new();
        for (kk, val) in filled.iter().enumerate() {
            writes.push(KernelWrite::new(
                format!("exc_out_{kk}"),
                format!("exc_{kk}"),
                val.node(),
            ));
        }
        writes
    })
}

/// the 1d radial gas fill (dof = 1): the michel / bondi row.
pub fn excise_fill_sph_1d_gv() -> (GvKernel, KernelWrites) {
    excise_fill_sph_dof_gv(1, 1)
}

/// the 2d (r, theta) gas fill with the azimuthal swirl momentum (dof = 3): the rotating GR flows.
pub fn excise_fill_sph_2d_gv() -> (GvKernel, KernelWrites) {
    excise_fill_sph_dof_gv(2, 3)
}

/// rebuild the conserved state of every excised spherical cell from its frozen primitives, with
/// the ingoing kerr-schild metric at the cell's own sampling position; live cells pass their
/// conserved state through untouched. the spinning form serves both charts — at `a = 0` it is the
/// schwarzschild kerr-schild metric, so one kernel covers the whole horizon-penetrating family.
fn excise_p2c_sph_ks_dof_gv(ndim: usize, dof: usize) -> (GvKernel, KernelWrites) {
    trace(|cx| {
        let gamma = cx.scalar("gamma");
        let mass = cx.scalar("schwarzschild_mass");
        let spin = cx.scalar("kerr_spin");

        let rho = cx.field("rho", FieldRef::PrimRho);
        let vel: Vec<Gv> = (0..dof)
            .map(|kk| cx.field(&format!("vel_{kk}"), FieldRef::PrimVel(kk as u8)))
            .collect();
        let pre = cx.field("pre", FieldRef::PrimPre);
        let den = cx.field("den", FieldRef::cons_den());
        let mom: Vec<Gv> = (0..dof)
            .map(|kk| cx.field(&format!("mom_{kk}"), FieldRef::cons_mom(kk as u8)))
            .collect();
        let nrg = cx.field("nrg", FieldRef::cons_nrg());

        let x = sample_position_sph::<3>(cx, ndim);
        let excised = excised_mask_sph(cx, x[0]);

        let m = KerrKS { mass, spin };
        let metric = SpatialMetric::<Gv, 3>::new(
            Gamma::new(m.spatial_metric(x)),
            GammaInv::new(m.spatial_metric_inv(x)),
        );
        let regime = RhdGr {
            metric,
            alpha: m.lapse(x),
            shift: m.shift(x),
            sqrt_gamma: m.volume_factor(x),
        };
        let prim = Prim::<Gv, 3>::adiabatic(
            Density(rho),
            Tensor::new(std::array::from_fn(|kk| {
                vel.get(kk).copied().unwrap_or(Gv::ZERO)
            })),
            Pressure(pre),
        );
        let cons = regime.to_conserved(&IdealGas { gamma }, &Valencia(prim)).0;

        let mut writes = vec![KernelWrite::new(
            "den_out",
            FieldRef::cons_den(),
            Gv::select(excised, cons.den(), den).node(),
        )];
        for kk in 0..dof {
            writes.push(KernelWrite::new(
                format!("mom_out_{kk}"),
                FieldRef::cons_mom(kk as u8),
                Gv::select(excised, cons.mom()[kk], mom[kk]).node(),
            ));
        }
        writes.push(KernelWrite::new(
            "nrg_out",
            FieldRef::cons_nrg(),
            Gv::select(excised, cons.nrg(), nrg).node(),
        ));
        writes
    })
}

pub fn excise_p2c_sph_ks_1d_gv() -> (GvKernel, KernelWrites) {
    excise_p2c_sph_ks_dof_gv(1, 1)
}

pub fn excise_p2c_sph_ks_2d_gv() -> (GvKernel, KernelWrites) {
    excise_p2c_sph_ks_dof_gv(2, 3)
}

/// the 3d gas fill: every excised cell takes the primitive state of its outward
/// corner-diagonal neighbor (sign-selected on all three axes); live cells copy
/// their own state. the rho/vel_0..2/pre set serves both the 3d GR-hydro state
/// and the 3d magnetized gas state (the field lives on the staggered faces,
/// outside this fill).
pub fn excise_fill_3d_gv() -> (GvKernel, KernelWrites) {
    trace(|cx| {
        let names = prim_names(3);
        let refs = prim_refs(3);

        let own: [Gv; 5] = std::array::from_fn(|kk| cx.field(&names[kk], refs[kk]));
        let nf = refs.len();

        let c = centroid(cx, 3);
        let x = [c[0], c[1], c[2]];
        let excised = excised_mask_3d(cx, &x);
        // the vacuum-floor sink: an excised (inside-horizon) cell is frozen at a cold c2p-safe vacuum;
        // live cells keep their own state. the boundary riemann rarefies the exterior gas into the
        // vacuum -- a one-way absorbing accretion bc (material crosses in and stays), the physical horizon.
        let filled: [Gv; 5] =
            std::array::from_fn(|kk| Gv::select(excised, vacuum_floor(cx, kk, nf), own[kk]));

        let mut writes = KernelWrites::new();
        for (kk, val) in filled.iter().enumerate() {
            writes.push(KernelWrite::new(
                format!("exc_out_{kk}"),
                format!("exc_{kk}"),
                val.node(),
            ));
        }
        writes
    })
}

/// the 3d sweep commit: copy the exc scratch back into the primitive fields.
pub fn excise_writeback_3d_gv() -> (GvKernel, KernelWrites) {
    excise_writeback_dof_gv(3)
}

/// the 3d conserved rebuild of every excised cell from its (just-filled) primitives:
/// the valencia `to_conserved` with the (spin-generic) cartesian kerr-schild spatial
/// metric at the cell's own centroid; live cells pass their conserved state through
/// untouched.
pub fn excise_p2c_3d_gv() -> (GvKernel, KernelWrites) {
    trace(|cx| {
        let gamma = cx.scalar("gamma");
        let mass = cx.scalar("schwarzschild_mass");
        let spin = cx.scalar("kerr_spin");

        let rho = cx.field("rho", FieldRef::PrimRho);
        let vel: [Gv; 3] =
            std::array::from_fn(|kk| cx.field(&format!("vel_{kk}"), FieldRef::PrimVel(kk as u8)));
        let pre = cx.field("pre", FieldRef::PrimPre);
        let den = cx.field("den", FieldRef::cons_den());
        let mom: [Gv; 3] =
            std::array::from_fn(|kk| cx.field(&format!("mom_{kk}"), FieldRef::cons_mom(kk as u8)));
        let nrg = cx.field("nrg", FieldRef::cons_nrg());

        let c = centroid(cx, 3);
        let x = [c[0], c[1], c[2]];
        let excised = excised_mask_3d(cx, &x);

        let xt = Tensor::<Gv, 3>::new(x);
        let m = KerrKSCartesian { mass, spin };
        let metric = SpatialMetric::<Gv, 3>::new(
            Gamma::new(m.spatial_metric(xt)),
            GammaInv::new(m.spatial_metric_inv(xt)),
        );
        // the densitized storage sqrt(-g)[rho u^t, T^t_i, -(T^t_t + rho u^t)] reads the cell lapse,
        // shift and full-chart measure, so the excised fill carries the same block the flux/c2p use.
        let regime = RhdGr {
            metric,
            alpha: m.lapse(xt),
            shift: m.shift(xt),
            sqrt_gamma: m.volume_factor(xt),
        };
        let prim = Prim::<Gv, 3>::adiabatic(Density(rho), Tensor::new(vel), Pressure(pre));
        let cons = regime.to_conserved(&IdealGas { gamma }, &Valencia(prim)).0;

        let mut writes = KernelWrites::new();
        writes.push(KernelWrite::new(
            "den_out",
            FieldRef::cons_den(),
            Gv::select(excised, cons.den(), den).node(),
        ));
        for kk in 0..3 {
            writes.push(KernelWrite::new(
                format!("mom_out_{kk}"),
                FieldRef::cons_mom(kk as u8),
                Gv::select(excised, cons.mom()[kk], mom[kk]).node(),
            ));
        }
        writes.push(KernelWrite::new(
            "nrg_out",
            FieldRef::cons_nrg(),
            Gv::select(excised, cons.nrg(), nrg).node(),
        ));
        writes
    })
}

/// the magnetized conserved rebuild of every excised cell: the ideal-GRMHD
/// valencia `to_conserved` — S_i = (rho h W^2 + B^2) v_i - (v.B) B_i and
/// tau = rho h W^2 + B^2 - (p + b^2/2) - D, all contractions through the
/// cell-centroid spatial metric — from the just-filled gas primitives and the
/// cell's own B (the face average the constrained transport owns). the
/// staggered faces keep the values transport gave them, so d_i(sqrt(gamma) B^i) = 0 survives
/// excision identically; the conserved B slots alias the cell B and pass
/// through untouched. MHD momentum/velocity vectors are always 3-component
/// (the 2d grid instance is the equatorial slice with z = 0 in the metric
/// position), so one builder serves both grid dimensions.
fn excise_p2c_mhd_dim_gv(ndim: usize) -> (GvKernel, KernelWrites) {
    trace(|cx| {
        let gamma = cx.scalar("gamma");
        let mass = cx.scalar("schwarzschild_mass");
        let spin = cx.scalar("kerr_spin");

        let rho = cx.field("rho", FieldRef::PrimRho);
        let vel: [Gv; 3] =
            std::array::from_fn(|kk| cx.field(&format!("vel_{kk}"), FieldRef::PrimVel(kk as u8)));
        let pre = cx.field("pre", FieldRef::PrimPre);
        let mag: [Gv; 3] =
            std::array::from_fn(|kk| cx.field(&format!("bc_{kk}"), FieldRef::BCell(kk as u8)));
        let den = cx.field("den", FieldRef::cons_den());
        let mom: [Gv; 3] =
            std::array::from_fn(|kk| cx.field(&format!("mom_{kk}"), FieldRef::cons_mom(kk as u8)));
        let nrg = cx.field("nrg", FieldRef::cons_nrg());

        let c = centroid(cx, ndim);
        // the metric position padded to 3 slots (the 2d grid is the z = 0 equatorial slice).
        let x3: [Gv; 3] = std::array::from_fn(|kk| c.get(kk).copied().unwrap_or(Gv::ZERO));
        let excised = if ndim == 2 {
            excised_mask_2d(cx, &[x3[0], x3[1]])
        } else {
            excised_mask_3d(cx, &x3)
        };

        let xt = Tensor::<Gv, 3>::new(x3);
        let m = KerrKSCartesian { mass, spin };
        let metric = SpatialMetric::<Gv, 3>::new(
            Gamma::new(m.spatial_metric(xt)),
            GammaInv::new(m.spatial_metric_inv(xt)),
        );
        // the covariant energy slot ehat = alpha tau + (alpha-1) D - beta^i S_i reads the cell lapse and
        // shift, so the excised-fill storage carries the same 3+1 block the flux/c2p use. RMHD keeps the
        // valencia solver (to_conserved gives tau), so re-split the energy slot here as the flux kernel does.
        let alpha = m.lapse(xt);
        let beta = m.shift(xt);
        let regime = RmhdGr { metric, alpha };
        let prim = MhdPrim::<Gv, 3>::new(
            Prim::<Gv, 3>::adiabatic(Density(rho), Tensor::new(vel), Pressure(pre)),
            Tensor::new(mag),
        );
        let cons = regime.to_conserved(&IdealGas { gamma }, &Valencia(prim)).0;
        let nrg_cov = alpha * cons.hydro().nrg() + (alpha - Gv::ONE) * cons.hydro().den()
            - beta.dot(cons.hydro().mom());
        let cons = cons.with_hydro(cons.hydro().with_nrg(nrg_cov));

        let mut writes = KernelWrites::new();
        writes.push(KernelWrite::new(
            "den_out",
            FieldRef::cons_den(),
            Gv::select(excised, cons.hydro().den(), den).node(),
        ));
        for kk in 0..3 {
            writes.push(KernelWrite::new(
                format!("mom_out_{kk}"),
                FieldRef::cons_mom(kk as u8),
                Gv::select(excised, cons.hydro().mom()[kk], mom[kk]).node(),
            ));
        }
        writes.push(KernelWrite::new(
            "nrg_out",
            FieldRef::cons_nrg(),
            Gv::select(excised, cons.hydro().nrg(), nrg).node(),
        ));
        writes
    })
}

pub fn excise_p2c_mhd_gv() -> (GvKernel, KernelWrites) {
    excise_p2c_mhd_dim_gv(2)
}

pub fn excise_p2c_mhd_3d_gv() -> (GvKernel, KernelWrites) {
    excise_p2c_mhd_dim_gv(3)
}

/// the per-cell outward boundary-flux contribution of the diagnostic region
/// `Omega = { r_ks(x) < diagnostic_radius }`, for the densitized numerical flux field
/// `flux_base` (`"mass_flux"` | `"nrg_flux"` | `"mom_flux_k"`). each cell owns its lo
/// faces: for gridded axis `d`, the lo face (between `c - e_d` and `c`) is a boundary
/// face of `Omega` iff the two cells straddle the shell, and its outward-normal
/// densitized flux `+/- F_lo * A_lo` is emitted (`+` when the interior cell is on the
/// `-d` side). `field_reduce(Add)` over the domain then telescopes to the net outward
/// flux through `d(Omega)`; the accretion rate is its negation. the `F_lo * A_lo` face
/// flux is the same quantity `gv_divergence` consumes, so the diagnostic is bit-consistent
/// with the finite-volume update and a steady flow is `diagnostic_radius`-invariant to
/// roundoff. cartesian kerr-schild only (the charts that excise).
pub fn shell_flux_map_gv(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: usize,
    flux_base: &str,
) -> (GvKernel, KernelWrites) {
    trace(|cx| {
        let geo = cell_geometry_gv(cx, coords, spacing, axes, ndim);
        let r_d = cx.scalar("diagnostic_radius");
        let spin = if matches!(spacetime, Spacetime::KerrKS) {
            cx.scalar("kerr_spin")
        } else {
            Gv::ZERO
        };
        // the 3d cell-center position (cartesian kerr-schild excision is cartesian): gridded axes at
        // their centroid, the suppressed axis (2d equatorial slice) at zero.
        let x_c: [Gv; 3] = std::array::from_fn(|c| match axes.iter().position(|&a| a == c) {
            Some(d) => geo.centroid[d],
            None => Gv::ZERO,
        });
        let inside_c = ks_excised(&x_c, spin, r_d);
        let mut contrib = Gv::ZERO;
        for d in 0..ndim {
            let a = axes[d];
            // the previous cell's center (one cell back along the swept axis) for its membership.
            let mut x_prev = x_c;
            x_prev[a] = x_c[a] - cx.scalar(&format!("dx_{a}"));
            let inside_prev = ks_excised(&x_prev, spin, r_d);
            // the densitized lo-face flux F_lo * A_lo (the field is face-centered at the lo faces).
            let densit = cx.field(&format!("{flux_base}_{d}"), &format!("{flux_base}[{d}]"))
                * geo.area_lo[d];
            // outward normal out of Omega: +d when the interior cell is on the -d side (prev inside),
            // -d when it is on the +d side (c inside). nonzero only on a boundary face (memberships differ).
            let from_prev_in = Gv::select(inside_c, Gv::ZERO, densit);
            let from_prev_out = Gv::select(inside_c, Gv::ZERO - densit, Gv::ZERO);
            contrib = contrib + Gv::select(inside_prev, from_prev_in, from_prev_out);
        }
        vec![KernelWrite::new(
            "shell_flux",
            FieldRef::Scratch,
            contrib.node(),
        )]
    })
}

#[cfg(test)]
mod shell_flux_tests {
    use super::*;

    #[test]
    fn excision_fill_reads_scale_derived_atmosphere() {
        let (kernel, _) = excise_fill_gv();
        assert!(
            kernel
                .scalar_params()
                .iter()
                .any(|name| name == "excision_rho")
        );
        assert!(
            kernel
                .scalar_params()
                .iter()
                .any(|name| name == "excision_pre")
        );
    }

    #[test]
    fn shell_flux_map_wires_the_diagnostic_radius_and_reads_the_flux_field() {
        // the cartesian kerr-schild shell reduction threads the diagnostic-radius level set + the
        // per-axis grid scalars, and reads the densitized mass flux; schwarzschild carries no spin.
        let (k, writes) = shell_flux_map_gv(
            Coords::Cartesian,
            Spacetime::SchwarzschildKS,
            &[Spacing::Uniform; 3],
            &[0, 1, 2],
            3,
            "mass_flux",
        );
        assert!(
            k.scalar_params().iter().any(|s| s == "diagnostic_radius"),
            "must wire diagnostic_radius: {:?}",
            k.scalar_params()
        );
        assert!(
            k.scalar_params().iter().any(|s| s == "dx_0"),
            "must wire the grid spacing: {:?}",
            k.scalar_params()
        );
        assert!(
            !k.scalar_params().iter().any(|s| s == "kerr_spin"),
            "schwarzschild ks carries no spin"
        );
        // the flux field is read at offset 0 (the cell's own lo face) — a direct read, bound at
        // dispatch, so the kernel's stencil reach stays zero. one scratch output per quantity pass.
        assert_eq!(writes.len(), 1, "one scratch output per quantity pass");

        // the spinning-kerr chart adds the spin scalar.
        let (kk, _) = shell_flux_map_gv(
            Coords::Cartesian,
            Spacetime::KerrKS,
            &[Spacing::Uniform; 3],
            &[0, 1, 2],
            3,
            "nrg_flux",
        );
        assert!(
            kk.scalar_params().iter().any(|s| s == "kerr_spin"),
            "spinning kerr carries the spin scalar"
        );
    }
}
