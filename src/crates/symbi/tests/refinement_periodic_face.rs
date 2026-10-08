// =============================================================================
// refinement_periodic_face.rs
//
// a refined patch against a periodic axis on the single hierarchy. a fine
// side flush with a periodic face inherits the periodic boundary and wraps
// onto the fine level's own opposite side, which is the domain's opposite
// side only when the patch spans the whole axis. a patch reaching one
// periodic face alone is refused by name, on either side; a patch spanning
// the axis and a patch clear of both faces conserve the mass of a density
// wave advected through the periodic domain to roundoff.
//
// run: cargo test -p symbi --test refinement_periodic_face
// =============================================================================
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_discretize::Recon;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const N: usize = 32;
const DX: f64 = 1.0 / N as f64;
// several crossings of the patch by the wave, about thirty root steps
const T_FINAL: f64 = 0.2;

/// a density wave carried along the periodic x axis at unit speed; y is closed by outflow walls
/// the uniform flow never crosses
fn root() -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .spacing([DX; 2])
        .origin([0.0; 2])
        .boundaries(Boundaries([
            [BoundaryType::Periodic, BoundaryType::Periodic],
            [BoundaryType::Outflow, BoundaryType::Outflow],
        ]))
        .cfl(CFL)
        .ghosts(2)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("root sim construction failed")
        .set_initial(|[x, _y]| {
            let rho = 1.0 + 0.5 * (2.0 * std::f64::consts::PI * x).sin();
            Prim::adiabatic(Density(rho), Tensor::new([1.0, 0.0]), Pressure(1.0))
        })
        .build()
}

fn kset(sim: &Sim) -> Kern {
    Kern::new(GAMMA, CFL, &sim.geom.allocated).reconstruction(Recon::Plm)
}

fn build(region: RefinementRegion<2>) -> Result<Hier, String> {
    let r = root();
    let k = kset(&r);
    Hier::with_refinement(r, k, &[region], ProlongOrder::Ppm, kset).map_err(|e| e.detail)
}

/// the root's interior mass: after the restriction the covered cells hold the fine average, so
/// this is the composite mass
fn mass(h: &Hier) -> f64 {
    let s = &h.levels[0].state;
    let v = s.fields.cons.den.view();
    s.geom.interior.iter().map(|c| *v.at(c)).sum::<f64>() * DX * DX
}

#[test]
fn a_patch_reaching_one_periodic_face_is_refused_by_name() {
    for (side, region) in [
        ("low", RefinementRegion { x_lo: [0.0, 0.25], x_hi: [0.25, 0.75] }),
        ("high", RefinementRegion { x_lo: [0.75, 0.25], x_hi: [1.0, 0.75] }),
    ] {
        let err = build(region).err().unwrap_or_else(|| panic!("a patch on the {side} periodic face was admitted"));
        assert!(err.contains("reaches the periodic face of axis 0 on one side only"), "{side}: the refusal names another reason: {err}");
    }
}

#[test]
fn a_patch_spanning_the_periodic_axis_or_clear_of_it_conserves_mass() {
    for (label, region) in [
        ("spanning the axis", RefinementRegion { x_lo: [0.0, 0.25], x_hi: [1.0, 0.75] }),
        ("clear of both faces", RefinementRegion { x_lo: [0.375, 0.25], x_hi: [0.625, 0.75] }),
    ] {
        let mut h = build(region).unwrap_or_else(|e| panic!("{label}: {e}"));
        h.seed_fine_from_coarse().expect("seed fine");
        h.prime();
        let m0 = mass(&h);
        h.evolve(T_FINAL).expect("evolve");
        assert!(h.levels[0].state.iteration > 10, "{label}: the wave barely moved; the gate is vacuous");
        // the reflux makes the composite update conservative: what remains is the roundoff of
        // summing ~1e3 cells of order one over ~30 steps, measured at 1.4e-14 unrefined
        let drift = ((mass(&h) - m0) / m0).abs();
        assert!(drift < 1e-13, "{label}: the composite mass drifted by {drift:.3e}");
    }
}
