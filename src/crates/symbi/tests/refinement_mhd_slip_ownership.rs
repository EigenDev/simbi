// =============================================================================
// refinement_mhd_slip_ownership.rs
//
// the finest level is the sole owner of a magnetic-slip sink on a refined hierarchy: coarser
// levels carry gravity-only proxies with no accretion and no magnetic coupling, allocate no slip
// storage, and the coupled-step decision reads the finest owner. the sink's operator support (the
// drain mask and the slip shell to their f64 support, plus the stencil reach) must lie inside the
// finest patch, checked before any step. after a transfer-only fine-to-coarse synchronization the
// covered coarse cells are the restriction of the fine cells, the covered coarse faces the
// area-weighted fine faces, the coarse cell field the interpolation of those faces, the coarse
// divergence at roundoff, the restricted energy untouched by any patch, and every coarse cell
// outside the coverage and its one-cell shell bit-identical to before.
// =============================================================================

use std::sync::atomic::Ordering;

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_grid::Field;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, BodyKind, MagneticSpec, SurfaceSpec};
use symbi_refinement::refinement::transfer::{restrict_bface, restrict_cell_field};
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Kset = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 3>;
type Hier = Hierarchy<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

const N: usize = 32;
const GAMMA: f64 = 5.0 / 3.0;

fn slip_spec(shell_width: f64) -> MagneticSpec {
    MagneticSpec::Slip {
        diffusivity_ratio: 2.0,
        shell_width,
        slip_length_ratio: 1.0,
        field_regularization: 0.1,
        placement: 0.0,
    }
}

// the corner vector potential A_z of a discretely solenoidal in-plane field: on the staggered
// complex, B_x = d_y A_z on x-faces and B_y = -d_x A_z on y-faces by backward differences of the
// corner values, so the discrete divergence vanishes exactly on any grid. `n` faces per unit length.
fn potential(amp: f64, k: f64) -> impl Fn(f64, f64) -> f64 + Copy {
    move |x: f64, y: f64| amp * (k * x).sin() * (k * y).sin()
}
fn face_of_potential(az: impl Fn(f64, f64) -> f64 + Copy, dx: f64) -> impl Fn(usize, [f64; 3]) -> f64 + Copy {
    // an x-face at (x_i, y_c) reads the corners (x_i, y_c +- dx/2); a y-face at (x_c, y_j) the
    // corners (x_c +- dx/2, y_j).
    move |axis: usize, [x, y, _z]: [f64; 3]| match axis {
        0 => (az(x, y + 0.5 * dx) - az(x, y - 0.5 * dx)) / dx,
        1 => -(az(x + 0.5 * dx, y) - az(x - 0.5 * dx, y)) / dx,
        _ => 0.0,
    }
}

// a smooth discretely solenoidal field seeded through the two-representation contract on a 32^3
// root refined over [1/8, 7/8] (fine spacing 1/64, half-width 24 fine cells); the fine level is
// seeded by the hierarchy's own divergence-free prolongation.
fn two_level(body: Option<Body<f64, 3>>) -> Hier {
    let dx = 1.0 / N as f64;
    let k = 2.0 * std::f64::consts::PI;
    let face = face_of_potential(potential(0.3 / k, k), dx);
    let kset = |s: &Sim| Kset::new(GAMMA, 0.3, 1.0, &s.geom.allocated);
    let coarse = Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .origin([0.0; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("root construction")
        .set_initial(move |[x, y, z]| {
            let bx = 0.5 * (face(0, [x - 0.5 * dx, y, z]) + face(0, [x + 0.5 * dx, y, z]));
            let by = 0.5 * (face(1, [x, y - 0.5 * dx, z]) + face(1, [x, y + 0.5 * dx, z]));
            MhdPrim::new(
                Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0 + 0.1 * (k * x).sin())),
                Tensor::new([bx, by, 0.0]),
            )
        })
        .seed_faces(face)
        .build();
    let ck = kset(&coarse);
    let regions = [RefinementRegion {
        x_lo: [0.125; 3],
        x_hi: [0.875; 3],
    }];
    let hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, kset).unwrap();
    hier.seed_fine_from_coarse().expect("fine seed");
    match body {
        Some(b) => hier.with_bodies(BodyCollection::new().add(b)),
        None => hier,
    }
}

fn sink(r_acc: f64, magnetic: MagneticSpec) -> Body<f64, 3> {
    Body::black_hole(0, Tensor::new([0.5; 3]), Tensor::zeros(), 1.0, r_acc, 0.05, 1.0, 1.0, r_acc)
        .with_surface(SurfaceSpec::Drain)
        .with_magnetic(magnetic)
}

// the finest patch has a half-width of 24 fine cells. the drain mask's f64 support is about
// twenty fine cells, so a sink of half a cell with a slip shell of an eighth of a cell keeps its
// whole support (0.5 + 19.7 + 3 cells) inside; a shell of two cells (about forty cells of
// support) does not.
const DX_FINE: f64 = 1.0 / (2.0 * N as f64);

#[test]
fn the_finest_level_owns_the_slip_and_coarse_proxies_carry_neither_sink_nor_coupling() {
    let w = 0.125 * DX_FINE;
    let hier = two_level(Some(sink(0.5 * DX_FINE, slip_spec(w))));
    assert_eq!(hier.levels.len(), 2);
    let fine = &hier.levels[1];
    let coarse = &hier.levels[0];
    let fine_body = fine.state.immersed.as_ref().unwrap().bodies.get(0);
    let coarse_body = coarse.state.immersed.as_ref().unwrap().bodies.get(0);
    assert!(matches!(fine_body.spec.magnetic, MagneticSpec::Slip { .. }), "the finest body lost its coupling");
    assert!(matches!(fine_body.kind, BodyKind::BlackHole { .. }), "the finest body lost its sink");
    assert!(matches!(coarse_body.spec.magnetic, MagneticSpec::None), "the coarse proxy carries a magnetic coupling");
    assert!(matches!(coarse_body.kind, BodyKind::Gravitational { .. }), "the coarse proxy carries a sink");
    assert!(coarse_body.accretion_radius().is_none(), "the coarse proxy can drain");
    let fmhd = fine.state.fields.mhd.as_ref().unwrap();
    let cmhd = coarse.state.fields.mhd.as_ref().unwrap();
    assert!(fmhd.magnetic_slip.is_some() && fmhd.slip_quadrature.is_some(), "the finest level has no slip storage");
    assert!(cmhd.magnetic_slip.is_none() && cmhd.slip_quadrature.is_none(), "a coarse level allocated slip storage");
    assert!(hier.finest_has_magnetic_slip(), "the owner is not detected on the finest level");
    assert!(!coarse.kernels.has_magnetic_slip(&coarse.state), "the root claims the slip");
}

#[test]
#[should_panic(expected = "sink support sphere")]
fn a_sink_whose_operator_support_reaches_the_finest_boundary_is_refused_before_stepping() {
    // a shell of two fine cells: about twenty shell widths of support, far beyond the eight-cell
    // half-width of the finest patch.
    let _ = two_level(Some(sink(0.5 * DX_FINE, slip_spec(2.0 * DX_FINE))));
}

// a snapshot of every stored value of a level's conserved, cell, and face fields.
fn snapshot(sim: &Sim) -> Vec<f64> {
    let m = sim.fields.mhd.as_ref().unwrap();
    let mut v = Vec::new();
    for c in sim.geom.allocated.iter() {
        v.push(*sim.fields.cons.den.at(c));
        for k in 0..3 {
            v.push(*sim.fields.cons.mom[k].at(c));
        }
        v.push(*sim.fields.cons.nrg_field().unwrap().at(c));
        for d in 0..3 {
            v.push(*m.bcell[d].at(c));
        }
    }
    for d in 0..3 {
        for c in m.bface[d].domain().iter() {
            v.push(*m.bface[d].at(c));
        }
    }
    v
}

#[test]
fn the_split_synchronization_makes_the_covered_coarse_state_the_restriction_and_touches_nothing_else() {
    let mut hier = two_level(None);
    hier.prime();
    // move the fine state away from the prolonged coarse state: a smooth divergence-free
    // perturbation of the fine faces and a pressure bump, so the synchronization has work to do.
    {
        let fine = &hier.levels[1].state;
        let m = fine.fields.mhd.as_ref().unwrap();
        let k = 2.0 * std::f64::consts::PI;
        let dxf = fine.geom.dx[0];
        let bump = face_of_potential(potential(0.05 / k, 2.0 * k), dxf);
        for d in 0..2 {
            for c in m.bface[d].domain().iter() {
                // face d of cell c sits at the face position along d and the cell center across.
                let pos: [f64; 3] = std::array::from_fn(|a| {
                    fine.geom.x_lo[a] + (c[a] as f64 + if a == d { 0.0 } else { 0.5 }) * dxf
                });
                m.bface[d].set(c, *m.bface[d].at(c) + bump(d, pos));
            }
        }
        let nrg = fine.fields.cons.nrg_field().unwrap();
        for c in fine.geom.interior.iter() {
            nrg.set(c, *nrg.at(c) * 1.01);
        }
        m.bface_initialized.store(true, Ordering::Relaxed);
    }
    let coarse_before = snapshot(&hier.levels[0].state);
    hier.sync_all_fine_to_coarse();
    let coarse = &hier.levels[0].state;
    let fine = &hier.levels[1].state;
    let cov = hier.levels[0].coverage.as_ref().unwrap().clone();
    let cm = coarse.fields.mhd.as_ref().unwrap();
    let fm = fine.fields.mhd.as_ref().unwrap();

    // covered conserved cells are the restriction, bit for bit.
    let scratch: Field<f64, 3, HostMemory> = Field::zeros(&coarse.geom.allocated).unwrap();
    let cons_pairs: Vec<(&Field<f64, 3, HostMemory>, &Field<f64, 3, HostMemory>, &str)> = vec![
        (&fine.fields.cons.den, &coarse.fields.cons.den, "density"),
        (&fine.fields.cons.mom[0], &coarse.fields.cons.mom[0], "momentum"),
        (fine.fields.cons.nrg_field().unwrap(), coarse.fields.cons.nrg_field().unwrap(), "energy"),
    ];
    for (f, c, name) in cons_pairs {
        restrict_cell_field(f, &scratch, &cov);
        for cell in cov.iter() {
            assert!(*c.at(cell) == *scratch.at(cell), "{name} at {cell:?} is not the restriction");
        }
    }
    // covered faces are the area-weighted fine faces, interface faces included.
    let face_scratch = symbi_sim::state::BfaceFields {
        b: std::array::from_fn(|d| Field::<f64, 3, HostMemory>::zeros(cm.bface[d].domain()).unwrap()),
    };
    restrict_bface(&fm.bface, &face_scratch, &cov);
    for d in 0..3 {
        for c in cov.extend(d, 0, 1).iter() {
            assert!(*cm.bface[d].at(c) == *face_scratch.b[d].at(c), "face ({d}, {c:?}) is not the area average");
        }
    }
    // the coarse cell field over the coverage is the interpolation of the synchronized faces, and
    // the coarse divergence over the coverage is at roundoff.
    let inv_dx = 1.0 / coarse.geom.dx[0];
    for c in cov.iter() {
        let mut div = 0.0;
        for d in 0..3 {
            let mut up = c;
            up[d] += 1;
            let interp = 0.5 * (*cm.bface[d].at(c) + *cm.bface[d].at(up));
            assert!((*cm.bcell[d].at(c) - interp).abs() <= 1e-15, "cell field ({d}, {c:?}) is not the face interpolation");
            div += (*cm.bface[d].at(up) - *cm.bface[d].at(c)) * inv_dx;
        }
        assert!(div.abs() <= 1e-12, "coarse divergence {div:.3e} at {c:?}");
    }
    // everything outside the coverage and its one-cell shell is untouched.
    let shell = {
        let mut lo = [0isize; 3];
        let mut hi = [0isize; 3];
        for a in 0..3 {
            lo[a] = cov.spaces[a].lo - 1;
            hi[a] = cov.spaces[a].hi + 1;
        }
        (lo, hi)
    };
    let after = snapshot(coarse);
    let mut i = 0usize;
    for c in coarse.geom.allocated.iter() {
        let inside = (0..3).all(|a| c[a] >= shell.0[a] && c[a] < shell.1[a]);
        for _ in 0..8 {
            if !inside {
                assert!(coarse_before[i] == after[i], "a coarse cell outside the coverage shell moved at {c:?}");
            }
            i += 1;
        }
    }
    // the coarse ghosts were refilled: the periodic images of the interior hold.
    let n = coarse.geom.interior.spaces[0].size() as isize;
    for c in coarse.geom.interior.iter() {
        if c[0] != 0 {
            continue;
        }
        let image = [c[0] + n, c[1], c[2]];
        assert!(*coarse.fields.prim.rho.at(image) == *coarse.fields.prim.rho.at(c), "coarse ghost at {image:?} is not the periodic image");
    }
}
