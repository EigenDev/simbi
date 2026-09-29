// =============================================================================
// cyl_rz_uct_emf.rs
//
// the UCT corner EMF E_phi on the cylindrical r-z plane, against the cartesian
// UCT corner EMF E_z as the oracle. E_phi's cyclic plane is (p1, p2) = (z, r)
// on grid axes (g1, g2) = (1, 0); the cartesian E_z's is (x, y) on (0, 1). the
// map (r, phi, z) -> (y, z, x) is a cyclic permutation of the components, a
// proper rotation, so an r-z state carried over by it, with the grid transposed,
// is the same physical corner problem: v_z, B_z, the z-normal faces and the z
// fan become the x ones, v_r, B_r, the r-normal faces and the r fan the y ones,
// and v_phi, B_phi the out-of-plane z. every UCT input is an asymmetric
// pseudo-random field; for each regime and each UCT solver family the r-z E_phi
// at corner (i, j) equals the cartesian E_z at corner (j, i) bit for bit, and
// the same comparison without the transpose departs by O(1).
//
// a z-uniform magnetized equilibrium at rest, B_z = B0 sin(k (r - r_min))
// between reflecting radial walls with a periodic z, carries a 1e-14
// z-alternating radial velocity. under the UCT corner EMF the seed reaches the
// face field within the first steps and the face field stays z-uniform to
// 1e-12 B0 through t = 0.3: a grid-scale mode in the seed decays under a
// consistent corner stencil and grows under a transposed one.
//
// run: cargo test -p symbi --test cyl_rz_uct_emf
// =============================================================================

use symbi::sim::evolve::KernelSet;
use symbi::sim::state::*;
use symbi::sim::substrate_seam::Solver;
use symbi_algebra::Tensor;
use symbi_geometry::{Cartesian, Cylindrical};
use symbi_grid::Field;
use symbi_hydro::eos::{IdealGas, Isothermal};
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 8;
type F = Field<f64, 2, HostMemory>;

fn rnd(i: isize, j: isize, salt: u64) -> f64 {
    let mut x = (i as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (j as i64 as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
        ^ salt.wrapping_mul(0x2545_F491_4F6C_DD1D);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x as f64 / u64::MAX as f64) - 0.5
}

/// the value an r-z field holds at its index (a, b): `shape` of a salted pseudo-random number
fn value(a: isize, b: isize, salt: u64, shape: fn(f64) -> f64) -> f64 {
    shape(rnd(a, b, salt))
}

fn signed(u: f64) -> f64 {
    u
}
fn positive(u: f64) -> f64 {
    1.0 + 0.6 * u
}
fn left(u: f64) -> f64 {
    -(0.5 + 0.4 * u)
}
fn right(u: f64) -> f64 {
    0.5 + 0.4 * u
}

/// fills `rz` with the pattern at its own indices and `cart` with the same pattern at the
/// transposed indices, or at its own indices when `transpose` is off
fn fill_pair(rz: &F, cart: &F, salt: u64, shape: fn(f64) -> f64, transpose: bool) {
    for c in rz.domain().iter() {
        *rz.view_mut().at_mut(c) = value(c[0], c[1], salt, shape);
    }
    for c in cart.domain().iter() {
        let (a, b) = if transpose { (c[1], c[0]) } else { (c[0], c[1]) };
        *cart.view_mut().at_mut(c) = value(a, b, salt, shape);
    }
}

/// every UCT input of the two stores, carried over by (r, phi, z) -> (y, z, x): the r-z component
/// k lands on the cartesian component `COMP[k]`, the r-z grid axis d on the cartesian grid axis
/// `1 - d`
fn carry<const DOF: usize>(
    rz: &FieldStore<2, DOF, HostMemory, f64>,
    cart: &FieldStore<2, DOF, HostMemory, f64>,
    transpose: bool,
) {
    const COMP: [usize; 3] = [1, 2, 0];
    let (mr, mc) = (rz.fields.mhd.as_ref().unwrap(), cart.fields.mhd.as_ref().unwrap());
    fill_pair(&rz.fields.prim.rho, &cart.fields.prim.rho, 1, positive, transpose);
    if let (Some(pr), Some(pc)) = (rz.fields.prim.pre_field(), cart.fields.prim.pre_field()) {
        fill_pair(pr, pc, 2, positive, transpose);
    }
    if let (Some(cr), Some(cc)) = (rz.fields.cs2.as_ref(), cart.fields.cs2.as_ref()) {
        fill_pair(cr, cc, 3, positive, transpose);
    }
    for k in 0..3 {
        fill_pair(&rz.fields.prim.vel[k], &cart.fields.prim.vel[COMP[k]], 10 + k as u64, signed, transpose);
        fill_pair(&mr.bcell.b[k], &mc.bcell.b[COMP[k]], 20 + k as u64, signed, transpose);
    }
    for d in 0..2 {
        let e = if transpose { 1 - d } else { d };
        fill_pair(&mr.bface.b[d], &mc.bface.b[e], 30 + d as u64, signed, transpose);
        fill_pair(&mr.wave_speed_l[d], &mc.wave_speed_l[e], 40 + d as u64, left, transpose);
        fill_pair(&mr.wave_speed_r[d], &mc.wave_speed_r[e], 50 + d as u64, right, transpose);
    }
}

/// (largest |E_phi|, largest |E_phi(i, j) - E_z(j, i)|, largest |E_phi(i, j) - E_z(i, j)|) over
/// the interior corners
fn compare<const DOF: usize>(
    rz: &FieldStore<2, DOF, HostMemory, f64>,
    cart: &FieldStore<2, DOF, HostMemory, f64>,
) -> (f64, f64, f64) {
    let (er, ec) = (&rz.fields.mhd.as_ref().unwrap().efield.e[0], &cart.fields.mhd.as_ref().unwrap().efield.e[0]);
    let s = &rz.geom.interior.spaces;
    let (mut scale, mut transposed, mut direct) = (0.0f64, 0.0f64, 0.0f64);
    for i in s[0].lo..=s[0].hi {
        for j in s[1].lo..=s[1].hi {
            let got = *er.view().at([i, j]);
            scale = scale.max(got.abs());
            transposed = transposed.max((got - *ec.view().at([j, i])).abs());
            direct = direct.max((got - *ec.view().at([i, j])).abs());
        }
    }
    (scale, transposed, direct)
}

/// the orientation gates of one regime: its r-z and cartesian stores, and one test per UCT
/// solver family
macro_rules! uct_orientation {
    ($module:ident, $regime:expr, $regime_ty:ty, $eos:expr, $eos_ty:ty, $kern:ty, $param:expr, $prim:expr, [$($test:ident => $solver:expr),*]) => {
        mod $module {
            use super::*;
            type Cyl = SimStateGeneric<$regime_ty, 2, 3, Cylindrical, $eos_ty, CpuSpace, HostMemory, f64>;
            type Cart = SimStateGeneric<$regime_ty, 2, 3, Cartesian, $eos_ty, CpuSpace, HostMemory, f64>;

            fn cyl() -> Cyl {
                Cyl::build($regime, $eos, Cylindrical)
                    .cells([N, N])
                    .bounds([1.0, 0.0], [2.0, 1.0])
                    .boundaries(Boundaries::uniform(BoundaryType::Outflow))
                    .cfl(0.3)
                    .allocate()
                    .expect("cylindrical r-z sim")
                    .set_initial(|_| $prim)
                    .seed_faces(|_, _| 0.0)
                    .build()
            }

            fn cart() -> Cart {
                Cart::build($regime, $eos, Cartesian)
                    .cells([N, N])
                    .bounds([1.0, 0.0], [2.0, 1.0])
                    .boundaries(Boundaries::uniform(BoundaryType::Outflow))
                    .cfl(0.3)
                    .allocate()
                    .expect("cartesian sim")
                    .set_initial(|_| $prim)
                    .seed_faces(|_, _| 0.0)
                    .build()
            }

            fn kernels(alloc: &symbi_algebra::Domain<2>, solver: Solver) -> $kern {
                <$kern>::new($param, 0.3, 1.5, alloc)
                    .with_solver(solver)
                    .expect("solver")
                    .ct_method(CtMethod::Uct)
            }

            /// (scale, transposed departure, direct departure) of one solver family
            fn run(solver: Solver, transpose: bool) -> (f64, f64, f64) {
                let (rz, xy) = (cyl(), cart());
                assert_eq!(rz.geom.axes, [0, 2], "the r-z chart carries components 0 and 2");
                carry(&rz, &xy, transpose);
                KernelSet::efield(&kernels(&rz.geom.allocated, solver), &rz);
                KernelSet::efield(&kernels(&xy.geom.allocated, solver), &xy);
                compare(&rz, &xy)
            }

            $(
                #[test]
                fn $test() {
                    let (scale, transposed, _) = run($solver, true);
                    assert!(scale > 0.05, "degenerate inputs: max |E_phi| = {scale:e}");
                    assert_eq!(
                        transposed, 0.0,
                        "E_phi departs from the rotated cartesian E_z by {transposed:.3e} (max |E_phi| {scale:.3e})"
                    );
                    // the same fields laid on the untransposed grid: the r-z kernel must see a different
                    // corner problem, or the comparison cannot tell the axes apart
                    let (scale, _, direct) = run($solver, false);
                    assert!(
                        direct > 1e-2 * scale,
                        "the untransposed carry agrees to {:.3e} of max |E_phi|; the inputs do not distinguish the axes",
                        direct / scale
                    );
                }
            )*
        }
    };
}

uct_orientation!(
    nmhd,
    symbi_hydro::newtonian_mhd::NewtonianMhd,
    symbi_hydro::newtonian_mhd::NewtonianMhd,
    IdealGas { gamma: 5.0 / 3.0 },
    IdealGas<f64>,
    symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet<HostMemory, f64, 2>,
    5.0 / 3.0,
    MhdPrim::new(Prim::adiabatic(Density(1.0), Tensor::new([0.0; 3]), Pressure(1.0)), Tensor::new([0.0; 3])),
    [hll => Solver::Hlle, hlld => Solver::Hlld]
);

uct_orientation!(
    rmhd,
    symbi_hydro::rmhd::Rmhd,
    symbi_hydro::rmhd::Rmhd,
    IdealGas { gamma: 4.0 / 3.0 },
    IdealGas<f64>,
    symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet<HostMemory, f64, 2>,
    4.0 / 3.0,
    MhdPrim::new(Prim::adiabatic(Density(1.0), Tensor::new([0.0; 3]), Pressure(1.0)), Tensor::new([0.0; 3])),
    [hll => Solver::Hlle, hlld => Solver::Hlld]
);

uct_orientation!(
    imhd,
    symbi_hydro::isothermal_mhd::IsothermalMhd,
    symbi_hydro::isothermal_mhd::IsothermalMhd,
    Isothermal { cs: 1.0 },
    Isothermal<f64>,
    symbi::regimes::substrate_isothermal_mhd::IsothermalMhdSubstrateKernelSet<HostMemory, f64, 2>,
    1.0,
    symbi_hydro::mhd_state::MhdPrimG::<f64, 3, symbi_hydro::energy::IsoModel>::new(
        symbi_hydro::state::PrimG::isothermal(Density(1.0), Tensor::new([0.0; 3])),
        Tensor::new([0.0; 3])
    ),
    [hll => Solver::Hlle, hlld => Solver::Hlld]
);

mod equilibrium {
    use super::*;
    use std::f64::consts::PI;
    use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
    use symbi::sim::evolve::evolve_with_callback;
    use symbi_hydro::newtonian_mhd::NewtonianMhd;

    type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory, f64>;
    const GAMMA: f64 = 5.0 / 3.0;
    const R_MIN: f64 = 1.0;
    const R_MAX: f64 = 3.0;
    const Z_MAX: f64 = 1.0;
    const B0: f64 = 1e-2;
    const K: f64 = 4.0 * PI / (R_MAX - R_MIN);

    fn sim(n: usize, seed: f64) -> Sim {
        Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cylindrical)
            .cells([n, n])
            .bounds([R_MIN, 0.0], [R_MAX, Z_MAX])
            .boundaries(Boundaries::per_axis([
                [BoundaryType::Reflect, BoundaryType::Reflect],
                [BoundaryType::Periodic, BoundaryType::Periodic],
            ]))
            .cfl(0.3)
            .allocate()
            .expect("cylindrical r-z sim")
            .set_initial(move |[r, z]| {
                let j = (z / (Z_MAX / n as f64)).floor() as i64;
                let vr = if j % 2 == 0 { seed } else { -seed };
                MhdPrim::new(
                    Prim::adiabatic(Density(1.0), Tensor::new([vr, 0.0, 0.0]), Pressure(1.0)),
                    Tensor::new([0.0, 0.0, B0 * (K * (r - R_MIN)).sin()]),
                )
            })
            .seed_faces(|axis, [r, _]| if axis == 1 { B0 * (K * (r - R_MIN)).sin() } else { 0.0 })
            .build()
    }

    /// max over rows r of max_z F - min_z F on the interior
    fn z_departure(s: &Sim, f: &F) -> f64 {
        let i = &s.geom.interior.spaces;
        (i[0].lo..i[0].hi)
            .map(|ir| {
                let (lo, hi) = (i[1].lo..i[1].hi).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), iz| {
                    let v = *f.view().at([ir, iz]);
                    (lo.min(v), hi.max(v))
                });
                hi - lo
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn a_z_uniform_equilibrium_keeps_its_face_field_z_uniform_under_uct() {
        let mut s = sim(48, 1e-14);
        let k = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.3, 1.0, &s.geom.allocated)
            .ct_method(CtMethod::Uct);
        let steps = std::cell::RefCell::new(Vec::<(f64, f64)>::new());
        evolve_with_callback(&mut s, &k, 0.3, 1, |st| {
            let m = st.fields.mhd.as_ref().unwrap();
            let dep = z_departure(st, &m.bface.b[0]).max(z_departure(st, &m.bface.b[1])) / B0;
            steps.borrow_mut().push((st.time, dep));
        })
        .expect("cylindrical equilibrium evolve");
        let steps = steps.into_inner();
        assert!(steps.len() > 40, "only {} steps; the mode had no time to grow", steps.len());
        assert!(
            steps.iter().take(4).any(|(_, d)| *d > 0.0),
            "the seed never reached the face field in the first four steps; the gate is vacuous"
        );
        let (t, worst) = steps.iter().fold((0.0, 0.0f64), |acc, &(t, d)| if d > acc.1 { (t, d) } else { acc });
        assert!(worst < 1e-12, "the face field departed from z-uniformity by {worst:.3e} B0 at t = {t:.4}");
    }
}
