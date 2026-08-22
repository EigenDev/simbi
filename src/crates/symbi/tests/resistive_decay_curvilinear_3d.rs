// =============================================================================
// resistive_decay_curvilinear_3d.rs
//
// the evolution gate for 3D curvilinear (spherical, cylindrical) Ohmic resistivity: a smooth
// compact div-free field threading still gas, run through the entire production NMHD kernel set
// with eta > 0, must
//   (a) lose magnetic energy strictly monotonically step over step (the resistive edge EMF is the
//       mimetic adjoint of the induction curl, so -curl(eta J) is negative-definite in the
//       dec-weighted energy norm), and
//   (b) keep the h-weighted staggered div(B) at machine zero (the resistive EMF rides the same
//       constrained-transport curl, so it can diffuse but never create monopoles).
// bug-injections: eta = 0 loses only the scheme's numerical-diffusion floor (the resistive term
// must dominate it), and resistivity = 0 declared is bit-identical to resistivity never declared
// (the disabled resistive path leaves no trace in the arithmetic).
//
// the seed: bface = the production ct-curl of a smooth windowed edge potential A (dt = 1 on a
// zero field), so the initial div(B) is machine zero in exactly the discrete divergence the curl
// preserves — no analytic-vs-discrete mismatch enters. the window vanishes well inside the
// boundary, so the outflow boundaries stay quiescent for the whole run.
// =============================================================================

use std::f64::consts::PI;

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{Cylindrical, Spherical};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::regime::Regime;
use symbi_hydro::state::{Cons, Prim};
use symbi_substrate::regimes::mhd_substrate::ct_curl;
use symbi_xpu::{CpuSpace, HostMemory};

type Store = symbi_sim::state::FieldStore<3, 3, HostMemory, f64>;
type SimSph = SimState<NewtonianMhd, 3, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
type SimCyl = SimState<NewtonianMhd, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>;

const N: usize = 12;
const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const ETA: f64 = 0.1;
const T_FINAL: f64 = 0.1;
// edge-potential amplitudes: |B| ~ amp * pi / window-width ~ 5e-3, magnetic pressure ~1e-5 << p=1,
// so the ideal Alfven/tension dynamics are negligible against the resistive decay over the run.
const AMP: [f64; 3] = [3e-4, 1e-3, 7e-4];

#[derive(Clone, Copy)]
enum Chart {
    Sph,
    Cyl,
}
// lame scale factors (h0, h1, h2) at coordinate x. must match Metric::scale_factors of the chart.
fn h(chart: Chart, x: [f64; 3]) -> [f64; 3] {
    match chart {
        Chart::Sph => [1.0, x[0], x[0] * x[1].sin()], // (h_r, h_theta = r, h_phi = r sin theta)
        Chart::Cyl => [1.0, x[0], 1.0],               // (h_r, h_phi = r, h_z = 1)
    }
}
fn bounds(chart: Chart) -> ([f64; 3], [f64; 3]) {
    match chart {
        // r in [1,2], theta in [0.8, 2.3] (off the poles), phi in [0,1].
        Chart::Sph => ([1.0, 0.8, 0.0], [2.0, 2.3, 1.0]),
        // r in [1,2], phi in [0,1], z in [0,1].
        Chart::Cyl => ([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]),
    }
}

// c^1 compact bump over the middle 70% of [lo, hi]; zero (with zero slope) outside, so the seeded
// field never touches the outflow boundary band.
fn window(x: f64, lo: f64, hi: f64) -> f64 {
    let t = (x - (lo + 0.15 * (hi - lo))) / (0.7 * (hi - lo));
    if t <= 0.0 || t >= 1.0 {
        0.0
    } else {
        (PI * t).sin().powi(2)
    }
}

// the staggered face coordinate: `face_axis` on the (low) face, the others at cell centers.
fn pos_face(fs: &Store, c: [isize; 3], face_axis: usize) -> [f64; 3] {
    std::array::from_fn(|a| {
        let base = fs.geom.x_lo[a] + c[a] as f64 * fs.geom.dx[a];
        if a == face_axis {
            base
        } else {
            base + 0.5 * fs.geom.dx[a]
        }
    })
}
// the edge along axis k: axis k at the cell center, the two transverse axes on faces.
fn pos_edge(fs: &Store, c: [isize; 3], edge_axis: usize) -> [f64; 3] {
    std::array::from_fn(|a| {
        let base = fs.geom.x_lo[a] + c[a] as f64 * fs.geom.dx[a];
        if a == edge_axis {
            base + 0.5 * fs.geom.dx[a]
        } else {
            base
        }
    })
}

// seed bface = ct_curl(A) through the production curl (dt = 1 on a zero field): div-free by the
// discrete d-of-d in exactly the divergence the evolution preserves. bcell is the face average,
// and cons is rebuilt so the total energy carries the seeded magnetic contribution.
fn seed<M>(
    sim: &mut SimState<NewtonianMhd, 3, M, IdealGas<f64>, CpuSpace, HostMemory>,
    chart: Chart,
) where
    M: symbi_geometry::Metric<f64, 3> + Copy,
{
    let (lo, hi) = bounds(chart);
    {
        let m = sim.fields.mhd.as_ref().expect("mhd fields");
        for k in 0..3 {
            for c in m.bface[k].domain().iter() {
                m.bface[k].set(c, 0.0);
            }
            for c in m.efield[k].domain().iter() {
                let x = pos_edge(sim, c, k);
                let w: f64 = (0..3).map(|a| window(x[a], lo[a], hi[a])).product();
                m.efield[k].set(c, AMP[k] * w);
            }
        }
    }
    ct_curl::<3, 3, HostMemory, f64>(sim, 1.0);
    let m = sim.fields.mhd.as_ref().expect("mhd fields");
    for k in 0..3 {
        for c in m.efield[k].domain().iter() {
            m.efield[k].set(c, 0.0);
        }
        for c in m.bcell[k].domain().iter() {
            m.bcell[k].set(c, 0.0);
        }
    }
    for c in sim.geom.interior.iter() {
        let mag: [f64; 3] = std::array::from_fn(|k| {
            let mut cp = c;
            cp[k] += 1;
            0.5 * (*m.bface[k].at(c) + *m.bface[k].at(cp))
        });
        let prim = MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([0.0, 0.0, 0.0]),
                pre: 1.0,
            },
            mag: Tensor::new(mag),
        };
        for k in 0..3 {
            m.bcell[k].set(c, mag[k]);
        }
        let cons = sim.physics.regime.to_conserved(&sim.physics.eos, &prim);
        sim.fields.cons.scatter(
            c,
            Cons {
                chi: Default::default(),
                den: cons.den,
                mom: cons.mom,
                nrg: cons.nrg,
            },
        );
    }
}

// the dec-weighted magnetic energy: sum_k B_k^2 * w_{B_k} over the interior low faces, with
// w_{B_k} = product of the other two scale factors at the k-face (the norm the adjoint identity
// makes monotone under the resistive EMF). the seed is compact, so the uncounted high-boundary
// faces carry no field.
fn e_face(fs: &Store, chart: Chart) -> f64 {
    let m = fs.fields.mhd.as_ref().expect("mhd fields");
    let mut e = 0.0;
    for k in 0..3 {
        for c in fs.geom.interior.iter() {
            let b = *m.bface[k].at(c);
            let hk = h(chart, pos_face(fs, c, k));
            let w: f64 = (0..3).filter(|&a| a != k).map(|a| hk[a]).product();
            e += b * b * w;
        }
    }
    e
}

// the h-weighted staggered divergence the curvilinear ct curl preserves:
// div = sum_k [w_k B_k](c + e_k) - [w_k B_k](c)) / dx_k, w_k = prod of the other two scale
// factors at the k-face center — the exact reciprocal of the curl kernel's 1/(h_p1 h_p2)
// face-center prefactor, so the edge terms telescope to machine zero. returns
// (max |div|, max face |w B| / min dx) — the second is the natural magnitude scale.
fn divb_max(fs: &Store, chart: Chart) -> (f64, f64) {
    let m = fs.fields.mhd.as_ref().expect("mhd fields");
    let dx_min = fs.geom.dx.iter().copied().fold(f64::INFINITY, f64::min);
    let mut div_max = 0.0_f64;
    let mut scale = 0.0_f64;
    for c in fs.geom.interior.iter() {
        let mut div = 0.0;
        for k in 0..3 {
            let mut cp = c;
            cp[k] += 1;
            let wb = |cc: [isize; 3]| -> f64 {
                let hk = h(chart, pos_face(fs, cc, k));
                let w: f64 = (0..3).filter(|&a| a != k).map(|a| hk[a]).product();
                w * *m.bface[k].at(cc)
            };
            let (hi, lo) = (wb(cp), wb(c));
            div += (hi - lo) / fs.geom.dx[k];
            scale = scale.max(hi.abs().max(lo.abs()) / dx_min);
        }
        div_max = div_max.max(div.abs());
    }
    (div_max, scale)
}

fn substrate(sim: &Store, eta: Option<f64>) -> NewtonianMhdSubstrateKernelSet<HostMemory, f64, 3> {
    let sub = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 3>::new(
        GAMMA,
        CFL,
        1.0,
        &sim.geom.allocated,
    );
    match eta {
        Some(e) => sub.with_resistivity(e),
        None => sub,
    }
}

// evolve to T_FINAL recording (iteration, weighted face energy) once per step; returns the ledger.
fn evolve_ledger<M>(
    sim: &mut SimState<NewtonianMhd, 3, M, IdealGas<f64>, CpuSpace, HostMemory>,
    chart: Chart,
    eta: Option<f64>,
) -> Vec<(u64, f64)>
where
    M: symbi_geometry::Metric<f64, 3> + Copy + Send + Sync,
{
    let sub = substrate(sim, eta);
    let mut ledger: Vec<(u64, f64)> = vec![(0, e_face(sim, chart))];
    evolve_with_callback(&mut *sim, &sub, T_FINAL, 1, |s| {
        if ledger.last().map(|(it, _)| *it) != Some(s.iteration) {
            ledger.push((s.iteration, e_face(s, chart)));
        }
    })
    .expect("evolve failed");
    ledger
}

fn make_sph() -> SimSph {
    let (lo, hi) = bounds(Chart::Sph);
    let mut sim = SimSph::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Spherical)
        .cells([N, N, N])
        .bounds(lo, hi)
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("3d spherical sim")
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
    seed(&mut sim, Chart::Sph);
    sim
}
fn make_cyl() -> SimCyl {
    let (lo, hi) = bounds(Chart::Cyl);
    let mut sim = SimCyl::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cylindrical)
        .cells([N, N, N])
        .bounds(lo, hi)
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("3d cylindrical sim")
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
    seed(&mut sim, Chart::Cyl);
    sim
}

fn assert_decay_and_divb(sim: &Store, chart: Chart, ledger: &[(u64, f64)], label: &str) {
    let e0 = ledger[0].1;
    let e_end = ledger.last().unwrap().1;
    assert!(e0 > 0.0, "{label}: degenerate seed, zero magnetic energy");
    assert!(
        ledger.len() > 10,
        "{label}: only {} steps — the resistive rate never engaged the cfl",
        ledger.len()
    );
    for pair in ledger.windows(2) {
        assert!(
            pair[1].1 < pair[0].1,
            "{label}: magnetic energy rose across step {} -> {}: {:.9e} -> {:.9e}",
            pair[0].0,
            pair[1].0,
            pair[0].1,
            pair[1].1
        );
    }
    assert!(
        e_end < 0.9 * e0,
        "{label}: no substantial ohmic decay: E/E0 = {}",
        e_end / e0
    );
    let (div, scale) = divb_max(sim, chart);
    assert!(
        scale > 0.0,
        "{label}: degenerate div oracle, zero field scale"
    );
    assert!(
        div < 1e-11 * scale,
        "{label}: staggered div(B) off machine zero after the resistive evolve: \
         max |div| = {div:.3e} vs field scale {scale:.3e}"
    );
    println!(
        "{label}: {} steps, E/E0 = {:.4}, max |div B| = {:.3e} (field scale {:.3e})",
        ledger.len() - 1,
        e_end / e0,
        div,
        scale
    );
}

#[test]
fn spherical_3d_resistive_decay_and_divb() {
    let mut sim = make_sph();
    let (div0, scale0) = divb_max(&sim, Chart::Sph);
    assert!(
        div0 < 1e-12 * scale0,
        "seed not div-free: {div0:.3e} vs scale {scale0:.3e}"
    );
    let ledger = evolve_ledger(&mut sim, Chart::Sph, Some(ETA));
    assert_decay_and_divb(&sim, Chart::Sph, &ledger, "3D spherical");
}

#[test]
fn cylindrical_3d_resistive_decay_and_divb() {
    let mut sim = make_cyl();
    let (div0, scale0) = divb_max(&sim, Chart::Cyl);
    assert!(
        div0 < 1e-12 * scale0,
        "seed not div-free: {div0:.3e} vs scale {scale0:.3e}"
    );
    let ledger = evolve_ledger(&mut sim, Chart::Cyl, Some(ETA));
    assert_decay_and_divb(&sim, Chart::Cyl, &ledger, "3D cylindrical");
}

#[test]
fn spherical_3d_resistivity_dominates_the_ideal_numerical_diffusion() {
    // eta = 0 still loses a little field to the scheme's own finite-resolution
    // diffusion; the resistive term must cause substantially more loss than that floor, else the
    // companion decay could be a numerical artifact.
    let mut ideal = make_sph();
    let li = evolve_ledger(&mut ideal, Chart::Sph, None);
    let mut resistive = make_sph();
    let lr = evolve_ledger(&mut resistive, Chart::Sph, Some(ETA));
    let (ri, rr) = (
        li.last().unwrap().1 / li[0].1,
        lr.last().unwrap().1 / lr[0].1,
    );
    assert!(
        rr < 0.6 * ri,
        "resistivity did not dominate the numerical diffusion: ideal ratio {ri}, resistive {rr}"
    );
}

#[test]
fn eta_zero_is_bit_identical_to_never_enabled() {
    // resistivity declared as 0 must leave no trace: the disabled resistive path (the dispatch
    // gate, the cfl fold) is exactly the ideal path, bit for bit, on a curvilinear 3D chart.
    let mut never = make_sph();
    evolve_ledger(&mut never, Chart::Sph, None);
    let mut zero = make_sph();
    evolve_ledger(&mut zero, Chart::Sph, Some(0.0));
    let (mn, mz) = (
        never.fields.mhd.as_ref().unwrap(),
        zero.fields.mhd.as_ref().unwrap(),
    );
    for k in 0..3 {
        for c in mn.bface[k].domain().iter() {
            assert_eq!(
                mn.bface[k].at(c).to_bits(),
                mz.bface[k].at(c).to_bits(),
                "bface[{k}] differs at {c:?}"
            );
        }
        for c in never.geom.interior.iter() {
            assert_eq!(
                mn.bcell[k].at(c).to_bits(),
                mz.bcell[k].at(c).to_bits(),
                "bcell[{k}] differs at {c:?}"
            );
        }
    }
    for c in never.geom.interior.iter() {
        assert_eq!(
            never.fields.prim.rho.view().at(c).to_bits(),
            zero.fields.prim.rho.view().at(c).to_bits(),
            "rho differs at {c:?}"
        );
    }
}
