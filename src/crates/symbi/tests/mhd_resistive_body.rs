// =============================================================================
// mhd_resistive_body.rs
//
// the immersed-body LOCALIZED Ohmic resistivity (`MagneticSpec::Resistive`): a body dissipates the
// magnetic field THREADING it (`eta*chi*J` added to the edge EMF, masked by the body indicator chi)
// while the exterior flux is left to ideal constrained transport. the kernel is exercised DIRECTLY
// (one masked-resistive EMF + one induction curl), pinning the two defining properties:
//   - localization: the added EMF is nonzero near the body (chi > 0) and exactly zero far away,
//     even though the field itself is nonzero everywhere. a `MagneticSpec::None` body leaves the
//     EMF untouched.
//   - dissipation: the magnetic-energy change `<B, curl(eta chi J B)>_F <= 0` — the body sheds
//     field monotonically (`-C diag(eta chi) C^T` is negative-definite for eta,chi >= 0).
// stability of the composed operator is proven to machine precision by the cyl/cartesian adjoint
// oracle; this test pins the mask (localization) and the production dispatch wiring.
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, MagneticSpec, SurfaceSpec};
use symbi_substrate::regimes::mhd_substrate::{body_resistive_emf, ct_curl};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;

const N: usize = 32;
const GAMMA: f64 = 5.0 / 3.0;
const BODY: [f64; 2] = [0.5, 0.5];
const R_BODY: f64 = 0.18;
const ETA: f64 = 0.1;
const PAD: isize = 3; // compact support: keep the random field off the domain boundary

// deterministic per-coordinate pseudo-random in [-0.5, 0.5].
fn rnd(i: isize, j: isize, salt: u64) -> f64 {
    let mut x = (i as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (j as i64 as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
        ^ salt.wrapping_mul(0x2545_F491_4F6C_DD1D);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x as f64 / u64::MAX as f64) - 0.5
}
fn in_window(c: [isize; 2]) -> bool {
    c[0] >= PAD && c[0] < N as isize - PAD && c[1] >= PAD && c[1] < N as isize - PAD
}
fn bx_seed(c: [isize; 2]) -> f64 {
    if in_window(c) {
        rnd(c[0], c[1], 1)
    } else {
        0.0
    }
}
fn by_seed(c: [isize; 2]) -> f64 {
    if in_window(c) {
        rnd(c[0], c[1], 2)
    } else {
        0.0
    }
}

fn make_sim(magnetic: MagneticSpec) -> Sim {
    let dx = 1.0 / N as f64;
    let sim = SimStateGeneric::<
        NewtonianMhd,
        2,
        3,
        Cartesian,
        IdealGas<f64>,
        CpuSpace,
        HostMemory,
        f64,
    >::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
    .cells([N, N])
    .origin([0.0, 0.0])
    .spacing([dx, dx])
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .cfl(0.3)
    .allocate()
    .expect("resistive body sim construction failed")
    .set_initial(|_| MhdPrim {
        hydro: Prim {
            rho: 1.0,
            vel: Tensor::new([0.0, 0.0, 0.0]),
            pre: 1.0,
        },
        mag: Tensor::new([0.0, 0.0, 0.0]),
    })
    .seed_faces(|_, _| 0.0)
    .build();
    // a hydrodynamically-transparent porous body (drain off, wall force off) that keeps a mask radius
    // for the magnetic SDF; the coupling under test is purely the magnetic one.
    sim.with_bodies(
        BodyCollection::new().add(
            Body::rigid_sphere(
                0,
                Tensor::new(BODY),
                Tensor::zeros(),
                1.0,
                R_BODY,
                1.0,
                false,
            )
            .with_surface(SurfaceSpec::Porous {
                porosity: 0.0,
                k_eta_n: 0.0,
                k_eta_t: 0.0,
            })
            .with_magnetic(magnetic),
        ),
    )
}

// distance from the body center to the E_z corner at edge coord c.
fn corner_dist(s: &Sim, c: [isize; 2]) -> f64 {
    let dx = s.geom.dx[0];
    let px = s.geom.x_lo[0] + c[0] as f64 * dx;
    let py = s.geom.x_lo[1] + c[1] as f64 * dx;
    ((px - BODY[0]).powi(2) + (py - BODY[1]).powi(2)).sqrt()
}

fn seed_field(s: &Sim) {
    let m = s.fields.mhd.as_ref().unwrap();
    for c in m.bface[0].domain().iter() {
        m.bface[0].set(c, bx_seed(c));
    }
    for c in m.bface[1].domain().iter() {
        m.bface[1].set(c, by_seed(c));
    }
    for c in m.efield[0].domain().iter() {
        m.efield[0].set(c, 0.0);
    }
}

#[test]
fn resistive_body_localizes_and_dissipates() {
    // the resistive body: efield[0] <- eta*chi*J*B (efield started at zero).
    let sim = make_sim(MagneticSpec::Resistive { eta: ETA });
    seed_field(&sim);
    body_resistive_emf::<2, 3, HostMemory, f64>(&sim);

    // localization: the added EMF is nonzero near the body and exactly zero beyond the masked region,
    // even though B is nonzero throughout the window.
    let mut near_max = 0.0_f64;
    let mut far_max = 0.0_f64;
    {
        let m = sim.fields.mhd.as_ref().unwrap();
        for c in m.efield[0].domain().iter() {
            let e = m.efield[0].at(c).abs();
            let r = corner_dist(&sim, c);
            if r < R_BODY {
                near_max = near_max.max(e);
            }
            // beyond ~6 cells the mollified tanh mask (width one cell) has decayed by >5 decades.
            if r > R_BODY + 6.0 * sim.geom.dx[0] {
                far_max = far_max.max(e);
            }
        }
    }
    assert!(
        near_max > 1e-6,
        "the resistive body added no EMF inside its mask (near_max = {near_max})"
    );
    assert!(
        far_max < 1e-3 * near_max,
        "the resistive EMF is not localized to the body mask: far_max = {far_max}, near_max = {near_max}"
    );

    // DISSIPATION: curl the masked EMF and confirm the magnetic-energy change is negative (cartesian
    // face weights are unity). dW = sum_f B_f*(bface_after - B_f); ct_curl does bface -= dt*curl(E).
    ct_curl::<2, 3, HostMemory, f64>(&sim, 1.0);
    let mut dw = 0.0_f64;
    {
        let m = sim.fields.mhd.as_ref().unwrap();
        for c in m.bface[0].domain().iter() {
            dw += bx_seed(c) * (*m.bface[0].at(c) - bx_seed(c));
        }
        for c in m.bface[1].domain().iter() {
            dw += by_seed(c) * (*m.bface[1].at(c) - by_seed(c));
        }
    }
    assert!(
        dw < -1e-9,
        "the resistive body did not dissipate magnetic energy (dW = {dw} >= 0): it is not negative-definite"
    );

    // a MagneticSpec::None body adds NOTHING: the EMF stays exactly zero.
    let sim_none = make_sim(MagneticSpec::None);
    seed_field(&sim_none);
    body_resistive_emf::<2, 3, HostMemory, f64>(&sim_none);
    let none_max = {
        let m = sim_none.fields.mhd.as_ref().unwrap();
        sim_none
            .geom
            .interior
            .iter()
            .map(|c| m.efield[0].at(c).abs())
            .fold(0.0_f64, f64::max)
    };
    assert_eq!(
        none_max, 0.0,
        "a non-magnetic body perturbed the edge EMF (max |E| = {none_max})"
    );
}

// the magnetic energy in a radial shell [lo, hi) from the body center, over the interior.
fn mag_energy_shell(s: &Sim, lo: f64, hi: f64) -> f64 {
    let m = s.fields.mhd.as_ref().unwrap();
    let dx = s.geom.dx[0];
    s.geom
        .interior
        .iter()
        .filter_map(|c| {
            let px = s.geom.x_lo[0] + (c[0] as f64 + 0.5) * dx;
            let py = s.geom.x_lo[1] + (c[1] as f64 + 0.5) * dx;
            let r = ((px - BODY[0]).powi(2) + (py - BODY[1]).powi(2)).sqrt();
            (r >= lo && r < hi).then(|| {
                let mut bsq = 0.0;
                for k in 0..3 {
                    let b = *m.bcell[k].view().at(c);
                    bsq += b * b;
                }
                0.5 * bsq
            })
        })
        .sum()
}

// seed a threading field B_x = B0 sin(k y) (div-free, nonzero current J_z) for an evolve run.
fn make_evolve_sim(magnetic: MagneticSpec) -> Sim {
    let dx = 1.0 / N as f64;
    let k = 2.0 * std::f64::consts::PI;
    let b0 = 1e-2;
    let sim = SimStateGeneric::<
        NewtonianMhd,
        2,
        3,
        Cartesian,
        IdealGas<f64>,
        CpuSpace,
        HostMemory,
        f64,
    >::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
    .cells([N, N])
    .origin([0.0, 0.0])
    .spacing([dx, dx])
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .cfl(0.3)
    .allocate()
    .expect("evolve sim construction failed")
    .set_initial(move |[_x, y]| MhdPrim {
        hydro: Prim {
            rho: 1.0,
            vel: Tensor::new([0.0, 0.0, 0.0]),
            pre: 1.0,
        },
        mag: Tensor::new([b0 * (k * y).sin(), 0.0, 0.0]),
    })
    .seed_faces(move |axis, [_x, y]| if axis == 0 { b0 * (k * y).sin() } else { 0.0 })
    .build();
    sim.with_bodies(
        BodyCollection::new().add(
            Body::rigid_sphere(
                0,
                Tensor::new(BODY),
                Tensor::zeros(),
                1.0,
                R_BODY,
                1.0,
                false,
            )
            .with_surface(SurfaceSpec::Porous {
                porosity: 0.0,
                k_eta_n: 0.0,
                k_eta_t: 0.0,
            })
            .with_magnetic(magnetic),
        ),
    )
}

#[test]
fn resistive_body_localizes_under_full_evolution() {
    // the full production path (godunov -> post_godunov body resistive EMF + curl -> penalize): the
    // resistive body preferentially decays the near-body field while leaving the far field identical
    // to the non-magnetic run. exercises the whole substrate.
    let run = |magnetic: MagneticSpec| -> (f64, f64) {
        let mut sim = make_evolve_sim(magnetic);
        let near0 = mag_energy_shell(&sim, 0.0, 0.30);
        let far0 = mag_energy_shell(&sim, 0.60, 1.0);
        let sub = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(
            GAMMA,
            0.3,
            1.0,
            &sim.geom.allocated,
        );
        evolve_with_callback(&mut sim, &sub, 0.1, u64::MAX, |_| {}).expect("evolve failed");
        (
            mag_energy_shell(&sim, 0.0, 0.30) / near0,
            mag_energy_shell(&sim, 0.60, 1.0) / far0,
        )
    };
    let (near_res, far_res) = run(MagneticSpec::Resistive { eta: 0.03 });
    let (near_none, far_none) = run(MagneticSpec::None);
    // the near-body field decays substantially more WITH the resistive body...
    assert!(
        near_res < near_none - 0.05,
        "resistive body did not locally dissipate under evolution: near ratios resistive={near_res:.4}, none={near_none:.4}"
    );
    // ...while the FAR field is left untouched (the coupling is local).
    assert!(
        (far_res - far_none).abs() < 0.02 * far_none,
        "the resistive body perturbed the far field under evolution: far resistive={far_res:.4}, none={far_none:.4}"
    );
}

// =============================================================================
// 3D cartesian: the same localization + dissipation properties for the full 3D
// body-mask resistive EMF (all three edge EMFs).
// =============================================================================
type Sim3 =
    SimStateGeneric<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;

const N3: usize = 16;
const BODY3: [f64; 3] = [0.5, 0.5, 0.5];

fn rnd3(c: [isize; 3], salt: u64) -> f64 {
    let mut x = (c[0] as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (c[1] as i64 as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
        ^ (c[2] as i64 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ salt.wrapping_mul(0x2545_F491_4F6C_DD1D);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x as f64 / u64::MAX as f64) - 0.5
}
fn in_window3(c: [isize; 3]) -> bool {
    (0..3).all(|a| c[a] >= PAD && c[a] < N3 as isize - PAD)
}
fn bseed3(axis: usize, c: [isize; 3]) -> f64 {
    if in_window3(c) {
        rnd3(c, axis as u64 + 1)
    } else {
        0.0
    }
}

fn make_sim3(magnetic: MagneticSpec) -> Sim3 {
    let dx = 1.0 / N3 as f64;
    let sim = Sim3::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N3, N3, N3])
        .origin([0.0, 0.0, 0.0])
        .spacing([dx, dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("3D resistive body sim construction failed")
        .set_initial(|_| MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.0, 0.0, 0.0]),
                pre: 1.0,
            },
            mag: Tensor::new([0.0, 0.0, 0.0]),
        })
        .seed_faces(|_, _| 0.0)
        .build();
    sim.with_bodies(
        BodyCollection::new().add(
            Body::rigid_sphere(
                0,
                Tensor::new(BODY3),
                Tensor::zeros(),
                1.0,
                R_BODY,
                1.0,
                false,
            )
            .with_surface(SurfaceSpec::Porous {
                porosity: 0.0,
                k_eta_n: 0.0,
                k_eta_t: 0.0,
            })
            .with_magnetic(magnetic),
        ),
    )
}

fn seed_field3(s: &Sim3) {
    let m = s.fields.mhd.as_ref().unwrap();
    for axis in 0..3 {
        for c in m.bface[axis].domain().iter() {
            m.bface[axis].set(c, bseed3(axis, c));
        }
    }
    for k in 0..3 {
        for c in m.efield[k].domain().iter() {
            m.efield[k].set(c, 0.0);
        }
    }
}

// cell-center distance from the body (a coarse locator for the near/far windows).
fn cell_dist3(s: &Sim3, c: [isize; 3]) -> f64 {
    let dx = s.geom.dx[0];
    (0..3)
        .map(|a| (s.geom.x_lo[a] + (c[a] as f64 + 0.5) * dx - BODY3[a]).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn resistive_body_3d_localizes_and_dissipates() {
    // the 3D body resistive EMF fills all three edge EMFs with eta*chi*J_dir. pin the same two
    // properties as the 2.5D case, now across every edge component.
    let sim = make_sim3(MagneticSpec::Resistive { eta: ETA });
    seed_field3(&sim);
    body_resistive_emf::<3, 3, HostMemory, f64>(&sim);

    // LOCALIZATION: some edge EMF is nonzero near the body and every edge EMF is ~zero far from it.
    let (mut near_max, mut far_max) = (0.0_f64, 0.0_f64);
    {
        let m = sim.fields.mhd.as_ref().unwrap();
        for k in 0..3 {
            for c in m.efield[k].domain().iter() {
                let e = m.efield[k].at(c).abs();
                let r = cell_dist3(&sim, c);
                if r < R_BODY {
                    near_max = near_max.max(e);
                }
                if r > R_BODY + 6.0 * sim.geom.dx[0] {
                    far_max = far_max.max(e);
                }
            }
        }
    }
    assert!(
        near_max > 1e-6,
        "the 3D resistive body added no EMF inside its mask (near_max = {near_max})"
    );
    assert!(
        far_max < 1e-3 * near_max,
        "the 3D resistive EMF is not localized: far_max = {far_max}, near_max = {near_max}"
    );

    // DISSIPATION: the div-B-clean 3D curl consumes the masked EMF and the magnetic-energy change is
    // negative (cartesian face weights are unity). dW = sum_{k,f} B_f*(bface_after - B_f).
    ct_curl::<3, 3, HostMemory, f64>(&sim, 1.0);
    let mut dw = 0.0_f64;
    {
        let m = sim.fields.mhd.as_ref().unwrap();
        for axis in 0..3 {
            for c in m.bface[axis].domain().iter() {
                dw += bseed3(axis, c) * (*m.bface[axis].at(c) - bseed3(axis, c));
            }
        }
    }
    assert!(
        dw < -1e-9,
        "the 3D resistive body did not dissipate magnetic energy (dW = {dw} >= 0)"
    );

    // a MagneticSpec::None body adds NOTHING to any edge EMF.
    let sim_none = make_sim3(MagneticSpec::None);
    seed_field3(&sim_none);
    body_resistive_emf::<3, 3, HostMemory, f64>(&sim_none);
    let none_max = {
        let m = sim_none.fields.mhd.as_ref().unwrap();
        (0..3)
            .flat_map(|k| sim_none.geom.interior.iter().map(move |c| (k, c)))
            .map(|(k, c)| m.efield[k].at(c).abs())
            .fold(0.0_f64, f64::max)
    };
    assert_eq!(
        none_max, 0.0,
        "a non-magnetic 3D body perturbed an edge EMF (max |E| = {none_max})"
    );
}

#[test]
fn drain_sink_in_2p5d_mhd_is_local_and_stable() {
    // the immersed-body penalize under 2.5D MHD (D=2, DOF=3). the kernel writes only the D=2
    // in-plane momentum components, so a dispatch that binds the full DOF=3 momentum shifts the nrg
    // write onto mom[2] and wipes the gas energy on every cell -> whole-domain NaN in a few steps.
    // a Drain sink evolves stably and removes gas inside its mask alone, leaving the far gas
    // exactly at the ambient state.
    let dx = 1.0 / N as f64;
    let k = 2.0 * std::f64::consts::PI;
    let b0 = 1e-2;
    let mut sim = SimStateGeneric::<
        NewtonianMhd,
        2,
        3,
        Cartesian,
        IdealGas<f64>,
        CpuSpace,
        HostMemory,
        f64,
    >::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
    .cells([N, N])
    .origin([0.0, 0.0])
    .spacing([dx, dx])
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .cfl(0.3)
    .allocate()
    .unwrap()
    .set_initial(move |[_x, y]| MhdPrim {
        hydro: Prim {
            rho: 1.0,
            vel: Tensor::new([0.0, 0.0, 0.0]),
            pre: 1.0,
        },
        mag: Tensor::new([b0 * (k * y).sin(), 0.0, 0.0]),
    })
    .seed_faces(move |axis, [_x, y]| if axis == 0 { b0 * (k * y).sin() } else { 0.0 })
    .build()
    .with_bodies(
        BodyCollection::new().add(
            Body::rigid_sphere(
                0,
                Tensor::new(BODY),
                Tensor::zeros(),
                1.0,
                R_BODY,
                1.0,
                false,
            )
            .with_surface(SurfaceSpec::Drain),
        ),
    );

    let far_den = |s: &Sim| -> (f64, f64) {
        // (min, max) interior density in the far shell r > 0.6.
        let dx = s.geom.dx[0];
        s.geom
            .interior
            .iter()
            .fold((f64::INFINITY, 0.0_f64), |(mn, mx), c| {
                let px = s.geom.x_lo[0] + (c[0] as f64 + 0.5) * dx;
                let py = s.geom.x_lo[1] + (c[1] as f64 + 0.5) * dx;
                let r = ((px - BODY[0]).powi(2) + (py - BODY[1]).powi(2)).sqrt();
                if r > 0.6 {
                    let d = *s.fields.cons.den.view().at(c);
                    (mn.min(d), mx.max(d))
                } else {
                    (mn, mx)
                }
            })
    };
    let near_mass = |s: &Sim| -> f64 {
        let dx = s.geom.dx[0];
        s.geom
            .interior
            .iter()
            .filter_map(|c| {
                let px = s.geom.x_lo[0] + (c[0] as f64 + 0.5) * dx;
                let py = s.geom.x_lo[1] + (c[1] as f64 + 0.5) * dx;
                let r = ((px - BODY[0]).powi(2) + (py - BODY[1]).powi(2)).sqrt();
                (r < R_BODY).then(|| *s.fields.cons.den.view().at(c))
            })
            .sum()
    };
    let near0 = near_mass(&sim);
    let sub = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(
        GAMMA,
        0.3,
        1.0,
        &sim.geom.allocated,
    );
    evolve_with_callback(&mut sim, &sub, 0.05, u64::MAX, |_| {})
        .expect("2.5D MHD drain evolve went unstable");

    // the sink removed mass inside its mask...
    assert!(
        near_mass(&sim) < near0 * 0.999,
        "the 2.5D MHD drain removed no mass inside the body"
    );
    // ...while the far gas stayed at the ambient density (the drain did not wipe the whole domain).
    let (fmn, fmx) = far_den(&sim);
    assert!(
        (fmn - 1.0).abs() < 1e-3 && (fmx - 1.0).abs() < 1e-3,
        "the 2.5D MHD drain corrupted the far gas density: far den in [{fmn:.5}, {fmx:.5}], expected ~1.0"
    );
}
