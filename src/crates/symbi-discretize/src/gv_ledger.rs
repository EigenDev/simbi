// =============================================================================
// gv_ledger.rs
//
// the integrands of the conservation ledger on the axisymmetric (r, theta) and (R, z) planes with
// the azimuthal velocity, one kernel per channel, each writing the scratch the executor then
// reduces where the data lives (an exact sum, or a minimum):
// - cell channels over the interior, from the conserved state and the cell's exact geometry:
//     Energy           E V
//     Kinetic          |m|^2 / (2 rho) V
//     AngularMomentum  m_phi R_c V, R_c the distance of the volume-weighted centroid from the axis
//     LowDensity       rho, or -inf where rho is not finite
//     LowSpecific      (E - |m|^2 / (2 rho)) / rho, or -inf where it or |m|^2 is not finite
//     Locate*          the cell code j * 2^16 + i where the channel's value equals the scalar
//                      `ledger_target`, else +inf: the minimum is the first such cell, j slowest
// - wall channels over one domain wall's face plane on grid axis `axis`, from the stage's stored
//   face fluxes and the face's exact area A, lever R_f and radius r_f:
//     WallMass         A F_rho
//     WallMassIn       max(-s A F_rho, 0), s = `outward_sign` (+1 on a high wall, -1 on a low one)
//     WallEnergy       A F_E
//     WallPotential    A Phi(r_f) F_rho, Phi the potential of the single static body (slot 0)
//     WallAngular      A R_f F_phi
// the geometry is the one the update kernels trace (`cell_geometry_gv`, the runtime axis maps), so
// the sums measure the discrete totals the scheme conserves.
//
// usage:
//   let program = ledger_gv(Coords::Spherical, &[0, 1], LedgerChannel::Kinetic);
// =============================================================================

use symbi_algebra::algebra::Numeric as _;
use symbi_carrier::Scalar as _;
use symbi_ir::gv::{KernelWrite, KernelWrites};
use symbi_ir::{FieldRef, Gv, KernelProgram, trace_kernel};

use crate::coords::{Coords, Spacing};
use crate::gv::{cell_geometry_gv, gv_axis_face_at};
use crate::gv_immersed::{body_cart_axes, body_vec3, to_cartesian_gv};

const NDIM: usize = 2;
const NCOMP: usize = 3;

/// what one kernel of the ledger family writes into the scratch
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LedgerChannel {
    Energy,
    Kinetic,
    AngularMomentum,
    LowDensity,
    LowSpecific,
    LocateDensity,
    LocateSpecific,
    WallMass { axis: usize },
    WallMassIn { axis: usize },
    WallEnergy { axis: usize },
    WallPotential { axis: usize },
    WallAngular { axis: usize },
}

impl LedgerChannel {
    /// the kernel name stem, `ledger_<channel>` with the wall's axis appended
    pub fn stem(self) -> String {
        match self {
            Self::Energy => "ledger_energy".into(),
            Self::Kinetic => "ledger_kinetic".into(),
            Self::AngularMomentum => "ledger_angular".into(),
            Self::LowDensity => "ledger_low_density".into(),
            Self::LowSpecific => "ledger_low_specific".into(),
            Self::LocateDensity => "ledger_locate_density".into(),
            Self::LocateSpecific => "ledger_locate_specific".into(),
            Self::WallMass { axis } => format!("ledger_wall_mass_{axis}"),
            Self::WallMassIn { axis } => format!("ledger_wall_mass_in_{axis}"),
            Self::WallEnergy { axis } => format!("ledger_wall_energy_{axis}"),
            Self::WallPotential { axis } => format!("ledger_wall_potential_{axis}"),
            Self::WallAngular { axis } => format!("ledger_wall_angular_{axis}"),
        }
    }

    /// every channel of the family
    pub fn all() -> Vec<LedgerChannel> {
        let mut out = vec![
            Self::Energy,
            Self::Kinetic,
            Self::AngularMomentum,
            Self::LowDensity,
            Self::LowSpecific,
            Self::LocateDensity,
            Self::LocateSpecific,
        ];
        for axis in 0..NDIM {
            out.extend([
                Self::WallMass { axis },
                Self::WallMassIn { axis },
                Self::WallEnergy { axis },
                Self::WallPotential { axis },
                Self::WallAngular { axis },
            ]);
        }
        out
    }
}

/// the azimuthal momentum slot of the plane: the coordinate the two grid axes leave out
fn azimuthal_slot(axes: &[usize]) -> u8 {
    (3 - axes[0] - axes[1]) as u8
}

/// a group of channels one kernel evaluates together, each into its own scratch slot `lb_{k}` in
/// the order `channels` lists them
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LedgerBatch {
    /// one wall's mass, gross inflow, energy and angular momentum fluxes, with the potential-energy
    /// flux of the single static body when `body`
    Wall { axis: usize, body: bool },
    /// the lowest density and specific internal energy, as their reducible values
    Extremes,
    /// the cell codes holding the two minima, against the scalars `ledger_target_density` and
    /// `ledger_target_specific`
    Locate,
    /// the volume integrands of total energy, kinetic energy and angular momentum
    Partition,
}

impl LedgerBatch {
    pub fn channels(self) -> Vec<LedgerChannel> {
        match self {
            Self::Wall { axis, body } => {
                let mut v = vec![
                    LedgerChannel::WallMass { axis },
                    LedgerChannel::WallMassIn { axis },
                    LedgerChannel::WallEnergy { axis },
                    LedgerChannel::WallAngular { axis },
                ];
                if body {
                    v.push(LedgerChannel::WallPotential { axis });
                }
                v
            }
            Self::Extremes => vec![LedgerChannel::LowDensity, LedgerChannel::LowSpecific],
            Self::Locate => vec![LedgerChannel::LocateDensity, LedgerChannel::LocateSpecific],
            Self::Partition => vec![LedgerChannel::Energy, LedgerChannel::Kinetic, LedgerChannel::AngularMomentum],
        }
    }

    /// the kernel name stem, `ledger_batch_<group>` with the wall's axis and body appended
    pub fn stem(self) -> String {
        match self {
            Self::Wall { axis, body } => format!("ledger_batch_wall{}_{axis}", if body { "_body" } else { "" }),
            Self::Extremes => "ledger_batch_extremes".into(),
            Self::Locate => "ledger_batch_locate".into(),
            Self::Partition => "ledger_batch_partition".into(),
        }
    }

    /// every batch of the family
    pub fn all() -> Vec<LedgerBatch> {
        let mut out = vec![Self::Extremes, Self::Locate, Self::Partition];
        for axis in 0..NDIM {
            out.extend([Self::Wall { axis, body: false }, Self::Wall { axis, body: true }]);
        }
        out
    }

    /// the most slots any batch writes
    pub const MAX_SLOTS: usize = 5;
}

/// the scalar a locate channel compares its value with: `ledger_target` in its own kernel, the
/// channel's named target inside a batch
fn target_scalar(channel: LedgerChannel, batched: bool) -> &'static str {
    match (channel, batched) {
        (_, false) => "ledger_target",
        (LedgerChannel::LocateDensity, true) => "ledger_target_density",
        _ => "ledger_target_specific",
    }
}

/// the integrand of each channel in one trace; the per-channel and the batched kernels both build
/// their values here, so a channel's arithmetic is the same in either
fn integrands<'t>(cx: symbi_ir::TraceCx<'t>, coords: Coords, axes: &[usize], channels: &[LedgerChannel], batched: bool) -> Vec<Gv<'t>> {
    let slot = azimuthal_slot(axes);
    let spacing = vec![Spacing::Uniform; NDIM];
    let geo = cell_geometry_gv(cx, coords, &spacing, axes, NDIM);
    let volume = Gv::ONE / geo.inv_volume;
    // the distance from the axis of a point given by its two grid coordinates
    fn lever<'t>(coords: Coords, x0: Gv<'t>, x1: Gv<'t>) -> Gv<'t> {
        match coords {
            Coords::Spherical => x0 * x1.sin(),
            _ => x0,
        }
    }
    let cell = || {
        let den = cx.field("den", FieldRef::cons_den());
        let nrg = cx.field("nrg", FieldRef::cons_nrg());
        let mom: Vec<Gv> = (0..NCOMP).map(|k| cx.field(&format!("mom_{k}"), FieldRef::cons_mom(k as u8))).collect();
        let m2 = mom.iter().fold(Gv::ZERO, |acc, m| acc + *m * *m);
        (den, nrg, mom, m2)
    };
    // fn items name the trace brand
    fn low_density<'t>(den: Gv<'t>) -> Gv<'t> {
        let infinity = Gv::from_f64(f64::INFINITY);
        Gv::select(den.abs().cmp_lt(infinity), den, -infinity)
    }
    fn low_specific<'t>(den: Gv<'t>, nrg: Gv<'t>, m2: Gv<'t>) -> Gv<'t> {
        let infinity = Gv::from_f64(f64::INFINITY);
        let e = (nrg - Gv::from_f64(0.5) * m2 / den) / den;
        Gv::select(e.abs().cmp_lt(infinity), Gv::select(m2.abs().cmp_lt(infinity), e, -infinity), -infinity)
    }
    fn located<'t>(value: Gv<'t>, target: Gv<'t>, code: Gv<'t>) -> Gv<'t> {
        Gv::select(value.cmp_eq(target), code, Gv::from_f64(f64::INFINITY))
    }
    let code = || cx.coord(1) * Gv::from_f64(65536.0) + cx.coord(0) * Gv::ONE;
    channels
        .iter()
        .map(|&channel| match channel {
            LedgerChannel::Energy => {
                let (_, nrg, _, _) = cell();
                nrg * volume
            }
            LedgerChannel::Kinetic => {
                let (den, _, _, m2) = cell();
                Gv::from_f64(0.5) * m2 / den * volume
            }
            LedgerChannel::AngularMomentum => {
                let (_, _, mom, _) = cell();
                mom[slot as usize] * lever(coords, geo.centroid[0], geo.centroid[1]) * volume
            }
            LedgerChannel::LowDensity => low_density(cell().0),
            LedgerChannel::LowSpecific => {
                let (den, nrg, _, m2) = cell();
                low_specific(den, nrg, m2)
            }
            LedgerChannel::LocateDensity => located(low_density(cell().0), cx.scalar(target_scalar(channel, batched)), code()),
            LedgerChannel::LocateSpecific => {
                let (den, nrg, _, m2) = cell();
                located(low_specific(den, nrg, m2), cx.scalar(target_scalar(channel, batched)), code())
            }
            LedgerChannel::WallMass { axis }
            | LedgerChannel::WallMassIn { axis }
            | LedgerChannel::WallEnergy { axis }
            | LedgerChannel::WallPotential { axis }
            | LedgerChannel::WallAngular { axis } => {
                let ax = axis as u8;
                // the face is this index's low face: the wall face of a low wall at its first index,
                // of a high wall at the index past its last
                let area = geo.area_lo[axis];
                let x_f = gv_axis_face_at(cx, axis, spacing[axis], 0);
                let other = 1 - axis;
                let (x0, x1) = if axis == 0 { (x_f, geo.centroid[1]) } else { (geo.centroid[0], x_f) };
                let mass = cx.field("f_rho", FieldRef::MassFlux(ax));
                match channel {
                    LedgerChannel::WallMass { .. } => area * mass,
                    LedgerChannel::WallMassIn { .. } => {
                        let inward = -cx.scalar("outward_sign") * area * mass;
                        Gv::select(inward.cmp_gt(Gv::ZERO), inward, Gv::ZERO)
                    }
                    LedgerChannel::WallEnergy { .. } => area * cx.field("f_nrg", FieldRef::NrgFlux(ax)),
                    LedgerChannel::WallAngular { .. } => {
                        let f_phi = cx.field("f_phi", FieldRef::MomFlux { comp: slot, axis: ax });
                        area * lever(coords, x0, x1) * f_phi
                    }
                    _ => {
                        // the potential at the face center, as the face gravity work evaluates it
                        let mut point = [Gv::ZERO; 3];
                        point[axes[axis]] = x_f;
                        point[axes[other]] = geo.centroid[other];
                        let cart = to_cartesian_gv(coords, &point);
                        let cart_axes = body_cart_axes(coords, NDIM, axes);
                        let pos = body_vec3(cx, 0, NDIM, &cart_axes, "pos");
                        let rvec: [Gv; 3] = std::array::from_fn(|ii| cart[ii] - pos[ii]);
                        let (m, soft, kind) = (cx.scalar("body_0_mass"), cx.scalar("body_0_soft"), cx.scalar("body_0_softkind"));
                        area * crate::ibm::body_potential(rvec, m, soft, kind) * mass
                    }
                }
            }
        })
        .collect()
}

pub fn ledger_gv(coords: Coords, axes: &[usize], channel: LedgerChannel) -> KernelProgram {
    assert!(matches!(coords, Coords::Spherical | Coords::Cylindrical), "the ledger kernels live on the axisymmetric planes");
    assert_eq!(axes.len(), NDIM, "the ledger kernels run on a two-axis plane");
    trace_kernel(|cx| {
        let value = integrands(cx, coords, axes, &[channel], false)[0];
        let mut writes = KernelWrites::new();
        writes.push(KernelWrite::new("scratch", FieldRef::Scratch, value.node()));
        writes
    })
}

/// the kernel of a batch: every channel's integrand into its scratch slot `lb_{k}`
pub fn ledger_batch_gv(coords: Coords, axes: &[usize], batch: LedgerBatch) -> KernelProgram {
    assert!(matches!(coords, Coords::Spherical | Coords::Cylindrical), "the ledger kernels live on the axisymmetric planes");
    assert_eq!(axes.len(), NDIM, "the ledger kernels run on a two-axis plane");
    trace_kernel(|cx| {
        let values = integrands(cx, coords, axes, &batch.channels(), true);
        let mut writes = KernelWrites::new();
        for (k, value) in values.into_iter().enumerate() {
            writes.push(KernelWrite::new(format!("lb_{k}_out"), format!("lb_{k}"), value.node()));
        }
        writes
    })
}
