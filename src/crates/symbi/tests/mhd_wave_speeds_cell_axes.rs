// =============================================================================
// mhd_wave_speeds_cell_axes.rs
//
// the materialized per-cell wave speeds of the magnetized regimes follow the
// grid's coordinate axes: slot d of wave_speed_{l,r} holds the characteristic
// speeds along the physical component that grid axis d carries. on the
// cylindrical (r, z) chart the grid axes carry components (0, 2), so slot 1 is
// the z direction; the r-z face fluxes and the UCT edge EMF read that slot for
// the z faces. the state is anisotropic in both velocity and field, so the
// azimuthal and vertical fans differ in every cell, and a premise asserts they
// do. the oracle is the regime's own wave-speed function evaluated on the host
// along the normal of each grid axis.
//
// run: cargo test -p symbi --test mhd_wave_speeds_cell_axes
// =============================================================================

use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet;
use symbi::sim::evolve::KernelSet;
use symbi::sim::state::*;
use symbi_algebra::{FaceNormal, Normalized, Tensor};
use symbi_hydro::regime::Regime;
use symbi_geometry::Cylindrical;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 4.0 / 3.0;
const N: usize = 8;

type Sim = SimStateGeneric<Rmhd, 2, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory, f64>;

/// an anisotropic magnetized state varying across the grid: (v_r, v_phi, v_z) and (B_r, B_phi, B_z)
/// all distinct and nonzero
fn state([r, z]: [f64; 2]) -> MhdPrim<f64, 3> {
    MhdPrim::new(
        Prim::adiabatic(
            Density(1.0 + 0.3 * z),
            Tensor::new([0.1 * r, 0.45 - 0.1 * z, -0.3 + 0.2 * z]),
            Pressure(0.2 + 0.05 * r),
        ),
        Tensor::new([0.2, 0.9 - 0.2 * z, 0.1 + 0.1 * r]),
    )
}

#[test]
fn rmhd_speeds_on_the_cylindrical_rz_chart_follow_the_grid_axes() {
    let sim = Sim::build(Rmhd, IdealGas { gamma: GAMMA }, Cylindrical)
        .cells([N, N])
        .bounds([1.0, 0.0], [2.0, 1.0])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(0.3)
        .allocate()
        .expect("cylindrical r-z sim")
        .set_initial(state)
        .seed_faces(|axis, x| if axis == 0 { state(x).mag()[0] } else { state(x).mag()[2] })
        .build();
    assert_eq!(sim.geom.axes, [0, 2], "the (r, z) chart carries components 0 and 2");
    let k = RmhdSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.3, 1.0, &sim.geom.allocated);
    k.c2p(&sim);
    k.wave_speeds(&sim);
    let mhd = sim.fields.mhd.as_ref().expect("magnetized fields");
    let eos = IdealGas { gamma: GAMMA };
    let mut fans_differ = false;
    for c in sim.geom.interior.iter() {
        let prim = MhdPrim::<f64, 3>::new(
            Prim::adiabatic(
                Density(*sim.fields.prim.rho.view().at(c)),
                Tensor::new(std::array::from_fn(|k| *sim.fields.prim.vel[k].view().at(c))),
                Pressure(*sim.fields.prim.pre_field().unwrap().view().at(c)),
            ),
            Tensor::new(std::array::from_fn(|k| *mhd.bcell.b[k].view().at(c))),
        );
        for (d, &component) in sim.geom.axes.iter().enumerate() {
            let (l, r) = Rmhd.wave_speeds(&eos, &prim, &Normalized::axis(component));
            let got = (*mhd.wave_speed_l[d].view().at(c), *mhd.wave_speed_r[d].view().at(c));
            // the kernel and the oracle evaluate one function on one state: equal to the
            // rounding of a quartic root solve, a few ulp of the unit-bounded speed
            let tol = 64.0 * f64::EPSILON;
            assert!(
                (got.0 - l).abs() <= tol && (got.1 - r).abs() <= tol,
                "cell {c:?} slot {d} (component {component}): kernel ({:.15}, {:.15}) against the \
                 component-{component} fan ({l:.15}, {r:.15})",
                got.0,
                got.1
            );
        }
        let (lp, rp) = Rmhd.wave_speeds(&eos, &prim, &Normalized::axis(1));
        let (lz, rz) = Rmhd.wave_speeds(&eos, &prim, &Normalized::axis(2));
        fans_differ |= (lp - lz).abs() > 1e-3 || (rp - rz).abs() > 1e-3;
    }
    assert!(fans_differ, "the azimuthal and vertical fans coincide; the gate cannot see the axis");
}
