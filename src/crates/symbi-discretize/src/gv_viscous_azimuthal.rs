// =============================================================================
// gv_viscous_azimuthal.rs
//
// the azimuthal shear on an axisymmetric plane: the geometric discretization on
// the actual mesh and the conservative update. the stress across a face is a
// difference of the angular velocity omega = v_phi / R_c of the two cells sharing
// it, over their actual center distance, so a graded axis needs no special case:
//
//   sphere, r face:     T = mu_f r_f sin(theta_c) d(omega) / dr
//   sphere, theta face: T = mu_f sin(theta_f) d(omega) / dtheta
//   (R, z), R face:     T = mu_f R_f d(omega) / dR
//   (R, z), z face:     T = mu_f R_c d(omega) / dz
//
// the face torque is G = A_f R_f T, with mu_f the mean of the two cells' dynamic
// viscosity. per cell, s = +1 on its high face and -1 on its low face:
//
//   V R_c dM_phi / dt = sum s G
//   V dE / dt         = sum s G omega_f,  omega_f = (omega_a + omega_b) / 2
//
// the torques and the work telescope to the domain faces, a uniform omega has no
// stress on any mesh, and the heating each cell takes is
// sum (1/2) K (omega_nb - omega_c)^2 >= 0, K = A_f R_f mu_f chi_f, chi_f the positive
// factor multiplying d(omega) in T. forward euler keeps a discrete maximum principle
// in omega for dt <= rho_c R_c^2 V / sum K.
//
// stencil: rho and v_phi one cell across each face. a face on a zero-torque domain
// edge drops from every sum, ghost values unread. the stress update reads primitives,
// writes the interior momentum and, on the energy closure, the total energy.
//
// usage:
//   let program = viscous_azimuthal_gv(coords, AzimuthalChannel::Update { energy: true });
// =============================================================================

use symbi_algebra::algebra::Numeric as _;
use symbi_carrier::Scalar as _;
use symbi_geometry::volume_weighted_centroid;
use symbi_hydro::viscous::dynamic_viscosity;
use symbi_ir::gv::{KernelWrite, KernelWrites};
use symbi_ir::{FieldRef, Gv, KernelProgram, trace_kernel};

use crate::coords::{Coords, Spacing};
use crate::gv::{cell_geometry_gv, gv_axis_face_at};

/// what one kernel of the family writes
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AzimuthalChannel {
    /// the momentum and, with `energy`, the total energy
    Update { energy: bool },
    /// `sum K / (rho R_c^2 V)` into the scratch: the inverse of the cell's timestep bound
    Rate,
    /// `sum s G` over this cell's continued domain faces into the scratch: the torque entering
    BoundaryTorque,
    /// `sum s G omega_f` over this cell's continued domain faces into the scratch
    BoundaryWork,
}

const NDIM: u8 = 2;

pub fn viscous_azimuthal_gv(coords: Coords, channel: AzimuthalChannel) -> KernelProgram {
    let (axes, slot): ([usize; 2], usize) = match coords {
        Coords::Spherical => ([0, 1], 2),
        Coords::Cylindrical => ([0, 2], 1),
        Coords::Cartesian => panic!("the azimuthal shear lives on an axisymmetric plane"),
    };
    trace_kernel(|cx| {
        let dt = cx.scalar("dt");
        let law = ["mu_scale", "mu_radius", "mu_radius_power", "mu_density", "mu_density_power"].map(|n| cx.scalar(n));
        // the local index of each domain face closed with zero torque, or an index no cell holds
        let zero: Vec<Gv> = ["zero_lo_0", "zero_hi_0", "zero_lo_1", "zero_hi_1"]
            .iter()
            .map(|n| {
                let node = cx.with_trace(|t| t.scalar_int(n));
                cx.gv(node)
            })
            .collect();
        let open: Option<Vec<Gv>> = matches!(channel, AzimuthalChannel::BoundaryTorque | AzimuthalChannel::BoundaryWork).then(|| {
            ["open_lo_0", "open_hi_0", "open_lo_1", "open_hi_1"]
                .iter()
                .map(|n| {
                    let node = cx.with_trace(|t| t.scalar_int(n));
                    cx.gv(node)
                })
                .collect()
        });

        let spacing = vec![Spacing::Uniform; NDIM as usize];
        let geo = cell_geometry_gv(cx, coords, &spacing, &axes, NDIM as usize);
        let face = |d: usize, offset: i64| gv_axis_face_at(cx, d, spacing[d], offset);
        let center = |d: usize, offset: i64| {
            volume_weighted_centroid(coords.to_geometry(), axes[d], face(d, offset), face(d, offset + 1))
        };
        let c = [center(0, 0), center(1, 0)];
        // fn items name the trace brand: the distances from the axis and from the origin
        fn axis_distance<'t>(coords: Coords, x0: Gv<'t>, x1: Gv<'t>) -> Gv<'t> {
            match coords {
                Coords::Spherical => x0 * x1.sin(),
                _ => x0,
            }
        }
        fn origin_distance<'t>(coords: Coords, x0: Gv<'t>, x1: Gv<'t>) -> Gv<'t> {
            match coords {
                Coords::Spherical => x0,
                _ => (x0 * x0 + x1 * x1).sqrt(),
            }
        }
        let cell_at = |di: i32, dj: i32, x0, x1| {
            let off = [di, dj];
            let rho = cx.field_offset("prim_rho", "prim.rho", NDIM, &off);
            let v = cx.field_offset("prim_v", FieldRef::PrimVel(slot as u8), NDIM, &off);
            let mu = dynamic_viscosity(law[0], law[1], law[2], law[3], law[4], origin_distance(coords, x0, x1), rho);
            (rho, mu, v / axis_distance(coords, x0, x1))
        };
        let (rho_c, mu_c, omega_c) = cell_at(0, 0, c[0], c[1]);
        let lever_c = axis_distance(coords, c[0], c[1]);

        let half = Gv::from_f64(0.5);
        let mut torque = Gv::ZERO;
        let mut work = Gv::ZERO;
        let mut conductance = Gv::ZERO;
        let mut ledger_torque = Gv::ZERO;
        let mut ledger_work = Gv::ZERO;
        for d in 0..2usize {
            for high in [false, true] {
                let step: i64 = if high { 1 } else { -1 };
                let nb = center(d, step);
                let (x0, x1) = if d == 0 { (nb, c[1]) } else { (c[0], nb) };
                let off = if d == 0 { (step as i32, 0) } else { (0, step as i32) };
                let (_, mu_nb, omega_nb) = cell_at(off.0, off.1, x0, x1);
                let f = face(d, i64::from(high));
                let area = if high { geo.area_hi[d] } else { geo.area_lo[d] };
                // the lever at the face center and the factor multiplying d(omega) in the stress
                let (lever_f, chi_metric) = match (coords, d) {
                    (Coords::Spherical, 0) => (f * c[1].sin(), f * c[1].sin()),
                    (Coords::Spherical, _) => (c[0] * f.sin(), f.sin()),
                    (_, 0) => (f, f),
                    _ => (c[0], c[0]),
                };
                let distance = if high { nb - c[d] } else { c[d] - nb };
                let mu_f = half * (mu_c + mu_nb);
                let k = area * lever_f * mu_f * chi_metric / distance;
                let index = cx.coord(d as u8);
                let side = 2 * d + usize::from(high);
                let closed = index.cmp_eq(zero[side]);
                let k = Gv::select(closed, Gv::ZERO, k);
                // the torque this cell takes through the face, positive when it gains
                let gain = k * (omega_nb - omega_c);
                let omega_f = half * (omega_c + omega_nb);
                torque = torque + gain;
                work = work + gain * omega_f;
                conductance = conductance + k;
                if let Some(open) = &open {
                    let held = index.cmp_eq(open[side]);
                    ledger_torque = ledger_torque + Gv::select(held, gain, Gv::ZERO);
                    ledger_work = ledger_work + Gv::select(held, gain * omega_f, Gv::ZERO);
                }
            }
        }
        let mut writes = KernelWrites::new();
        match channel {
            AzimuthalChannel::Update { energy } => {
                let mom = cx.field("mom", FieldRef::cons_mom(slot as u8));
                writes.push(KernelWrite::new(
                    "mom_out",
                    FieldRef::cons_mom(slot as u8),
                    (mom + dt * torque * geo.inv_volume / lever_c).node(),
                ));
                if energy {
                    let nrg = cx.field("nrg", FieldRef::cons_nrg());
                    writes.push(KernelWrite::new("nrg_out", FieldRef::cons_nrg(), (nrg + dt * work * geo.inv_volume).node()));
                }
            }
            AzimuthalChannel::Rate => {
                let rate = conductance * geo.inv_volume / (rho_c * lever_c * lever_c);
                writes.push(KernelWrite::new("scratch", FieldRef::Scratch, rate.node()));
            }
            AzimuthalChannel::BoundaryTorque => writes.push(KernelWrite::new("scratch", FieldRef::Scratch, ledger_torque.node())),
            AzimuthalChannel::BoundaryWork => writes.push(KernelWrite::new("scratch", FieldRef::Scratch, ledger_work.node())),
        }
        writes
    })
}
