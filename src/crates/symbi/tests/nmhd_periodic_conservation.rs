// =============================================================================
// nmhd_periodic_conservation.rs
//
// regression pin for the staggered-bface ghost fill (the CT wrap-drift fix):
// the bface transverse halo was never filled, so the transversely-extended
// flux sweep read zero normal-B at every boundary-adjacent ghost face, the
// boundary-edge EMFs were wrong from the first step, the two periodic wrap
// copies of every face drifted apart, and the flux telescoping leaked mass
// ~1e-9/step (hydro was exact; divB never noticed — CT preserves it for any
// EMF). with the fill, the wrap copies stay BIT-identical by induction and
// the periodic totals are conserved to machine precision.
//
// pins, on a single-level orszag-tang run with periodic walls:
//   (a) the bface transverse halo equals its periodic source exactly,
//   (b) the wrap copies of bface and efield are bit-identical,
//   (c) total mass and momentum are conserved to 1e-12.
// =============================================================================

use std::f64::consts::PI;

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet3D;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<NewtonianMhd, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = NewtonianMhdSubstrateKernelSet3D<HostMemory, f64>;

const N: usize = 16;
const NZ: usize = 2;
const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const V0: f64 = 0.5;
const B0: f64 = 1.0;

#[test]
fn nmhd_periodic_run_conserves_and_keeps_wrap_copies_locked() {
    let dx = 1.0 / N as f64;
    let dz = 1.0 / NZ as f64;
    let rho0 = GAMMA * GAMMA;
    let p0 = GAMMA;
    // staggered bface is the CELL-AVERAGED analytic field (integral of the OT sin over the face's
    // transverse cell width); seed_faces reconstructs the cell edges from the face midpoint + dx.
    // bcell is point-sampled from the cell-centered prim mag (via set_initial) — they intentionally
    // differ, as in the original IC.
    let mut sim = Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N, NZ])
        .spacing([dx, dx, dz])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .allocate()
        .unwrap()
        .set_initial(|[x, y, _z]| MhdPrim {
            hydro: Prim {
                rho: rho0,
                vel: Tensor::new([-V0 * (2.0 * PI * y).sin(), V0 * (2.0 * PI * x).sin(), 0.0]),
                pre: p0,
            },
            mag: Tensor::new([-B0 * (2.0 * PI * y).sin(), B0 * (4.0 * PI * x).sin(), 0.0]),
        })
        .seed_faces(|axis, [x, y, _z]| match axis {
            0 => {
                let (y0, y1) = (y - dx / 2.0, y + dx / 2.0);
                B0 * ((2.0 * PI * y1).cos() - (2.0 * PI * y0).cos()) / (2.0 * PI * dx)
            }
            1 => {
                let (x0, x1) = (x - dx / 2.0, x + dx / 2.0);
                B0 * ((4.0 * PI * x0).cos() - (4.0 * PI * x1).cos()) / (4.0 * PI * dx)
            }
            _ => 0.0,
        })
        .build();
    let k = Kset::new(GAMMA, CFL, 1.0, &sim.geom.allocated);
    let n = N as isize;
    let vol = dx * dx * dz;

    let totals = |s: &Sim| -> (f64, f64) {
        let mut mass = 0.0;
        let mut momx = 0.0;
        for c in s.geom.interior.iter() {
            mass += *s.fields.cons.den.view().at(c) * vol;
            momx += *s.fields.cons.mom[0].view().at(c) * vol;
        }
        (mass, momx)
    };
    let (m0, p0) = totals(&sim);

    evolve_with_callback(&mut sim, &k, 0.06, 1, |s| {
        let mhd = s.fields.mhd.as_ref().unwrap();
        // the wrap copies of the normal faces are the SAME physical face:
        // they must stay bit-identical under evolution.
        for j in 0..n {
            for kk in 0..NZ as isize {
                let a = *mhd.bface[0].view().at([0, j, kk]);
                let b = *mhd.bface[0].view().at([n, j, kk]);
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "bface wrap copies drifted at iter {} (j={j}, k={kk}): {a:e} vs {b:e}",
                    s.iteration
                );
            }
        }
        // the bface transverse halo equals its periodic source exactly.
        for i in 0..=n {
            for kk in 0..NZ as isize {
                let a = *mhd.bface[0].view().at([i, -1, kk]);
                let b = *mhd.bface[0].view().at([i, n - 1, kk]);
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "bface halo stale at iter {} (i={i}, k={kk}): {a:e} vs {b:e}",
                    s.iteration
                );
            }
        }
        // boundary-edge EMF wrap copies bit-identical (the quantity whose
        // inconsistency drove the drift).
        for j in 0..=n {
            for kk in 0..NZ as isize {
                let a = *mhd.efield[2].view().at([0, j, kk]);
                let b = *mhd.efield[2].view().at([n, j, kk]);
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "efield wrap copies differ at iter {} (j={j}, k={kk})",
                    s.iteration
                );
            }
        }
    })
    .unwrap();

    assert!(
        sim.iteration >= 5,
        "only {} steps — gate barely exercised",
        sim.iteration
    );
    let (m1, p1) = totals(&sim);
    let rel = |a: f64, b: f64, s: f64| ((a - b) / s).abs();
    assert!(rel(m1, m0, m0) < 1e-12, "mass drift {:e}", rel(m1, m0, m0));
    assert!(
        rel(p1, p0, m0) < 1e-12,
        "momentum drift {:e}",
        rel(p1, p0, m0)
    );
}
