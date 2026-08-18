// =============================================================================
// masked_wall_low_mach_dissipation.rs
//
// a sealed spherical accretor immersed in a gravitating gas, evolved with the
// published low-mach ramp. the ramp scales the acoustic dissipation by
// `sin(min(1, Ma/Ma_ref) pi/2)` on the face-normal mach number, reading a
// vanishing mach number as evidence of smooth subsonic flow. inside a
// penalization mask the velocity is relaxed onto the wall's, so the mach number
// there reports the boundary condition rather than the flow: the ramp switches
// the acoustic dissipation off in exactly the cells where the wall relaxation
// deposits its friction heat.
//
// the two quantities that fail when it does are measured here, both inside the
// mask: the entropy `K = p / rho^gamma`, which an adiabatic gas behind a sealed
// wall carries as the ambient isentrope plus the wall's own friction heat, and
// the sound speed, which sets the CFL timestep. a masked interior that decouples
// cell from cell supports large density contrasts at one pressure, so `K` climbs
// and `c_s ~ sqrt(gamma p / rho)` follows the evacuating cell down in density and
// up in speed.
//
// the miniature separates the two arms by a factor of four in the entropy excess
// the masked interior carries; the production configuration this reproduces in
// small separated by a factor of forty and ended in a timestep collapse.
//
// run: cargo test -p symbi --test masked_wall_low_mach_dissipation -- --nocapture
// =============================================================================

use symbi::prelude::Solver;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, SurfaceSpec};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.3;
const N: usize = 32;
const GM: f64 = 3.0;
/// the entropy of the ambient isentrope, and the reference the mask interior is measured
/// against: the wall may add heat, so `K/K_0` is bounded below by one and its growth is the
/// quantity under test.
const K0: f64 = 0.6;
/// four cells across the mask radius, so the mask carries an interior rather than a surface.
const R_ACC: f64 = 4.0 / N as f64;
/// plummer softening at a quarter of the mask radius: the body's field is bare Newtonian
/// across most of the masked interior and regular at its centre, so the penalization absorbs
/// a finite acceleration that still varies strongly from cell to cell.
const SOFT: f64 = R_ACC / 4.0;
const STEPS: u64 = 600;

type Sim = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 3>;
type Hier = Hierarchy<Newtonian, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

/// the ambient isentropic atmosphere in hydrostatic balance against the softened point mass,
/// from the bernoulli invariant `gamma K_0/(gamma-1) rho^(gamma-1) + phi = const`, normalized
/// to `rho = 1` at the domain corner. inside the mask the profile holds at its mask-radius
/// value: that region is the wall's, and its state is whatever the penalization makes of it.
///
/// the ambient is stagnant and the stratification subsonic, so every face in the run sits at
/// the low mach numbers the ramp acts on. what forces the mask interior is the mismatch
/// between the flat state the wall holds and the body's own field: gravity accelerates each
/// masked cell every step, the wall relaxation takes the momentum straight back out, and the
/// work of that exchange stays behind as heat.
fn atmosphere(x: [f64; 3]) -> Prim<f64, 3> {
    let r = x.iter().map(|c| c * c).sum::<f64>().sqrt().max(R_ACC);
    let r_ref = 3.0_f64.sqrt() * 0.5;
    let phi = |rr: f64| -GM / (rr * rr + SOFT * SOFT).sqrt();
    let a = (GAMMA - 1.0) / (GAMMA * K0);
    let rho = (1.0 + a * (phi(r_ref) - phi(r))).powf(1.0 / (GAMMA - 1.0));
    Prim {
        rho,
        vel: Tensor::new([0.0; 3]),
        pre: K0 * rho.powf(GAMMA),
    }
}

fn build() -> Hier {
    let dx = 1.0 / N as f64;
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N, N])
        .origin([-0.5, -0.5, -0.5])
        .spacing([dx, dx, dx])
        // a reflecting wall does no work on gas at rest, so the ambient hydrostatic state
        // is a fixed point of the boundary as well as of the interior.
        .boundaries(Boundaries::uniform(BoundaryType::Reflect))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim construction failed")
        .set_initial(atmosphere)
        .build();
    let kernels = Kset::new(GAMMA, CFL, &sim.geom.allocated)
        .with_solver(Solver::HllcPlus)
        .expect("solver/regime mismatch")
        // the production pairing: the ramp leaves a plain reconstruction's hydrostatic
        // truncation residual undamped, and the balanced reconstruction removes that
        // residual at its source, so the dissipation the mask restores acts on the
        // genuine structure alone.
        .well_balanced_reconstruction(true);
    Hierarchy::single(sim, kernels).with_bodies(
        BodyCollection::new().add(
            Body::black_hole(
                0,
                Tensor::new([0.0; 3]),
                Tensor::zeros(),
                GM,
                R_ACC,
                SOFT,
                0.0,
                1.0,
                R_ACC,
            )
            // porosity 0 is the sealed wall: the mask interior is relaxed onto the body's
            // own velocity on both the normal and the tangential channel, and no mass
            // leaves through it. the relaxation is what holds the interior stagnant, and
            // the work it does is what heats the interior.
            .with_surface(SurfaceSpec::Porous {
                porosity: 0.0,
                k_eta_n: 50.0,
                k_eta_t: 50.0,
            }),
        ),
    )
}

/// the largest `K/K_0` inside the mask and the smallest timestep the run took. one step at a
/// time, so the timestep is sampled at every step rather than at the end, where a collapse
/// has already ended the run.
fn run() -> (f64, f64, usize) {
    let mut hier = build();
    let mut dt_min = f64::INFINITY;
    for ii in 0..STEPS {
        hier.evolve_steps(1).unwrap();
        let dt = hier.levels[0].state.dt;
        assert!(dt.is_finite() && dt > 0.0, "timestep left the reals: {dt}");
        dt_min = dt_min.min(dt);
        if (ii + 1) % 100 == 0 {
            let s = scan(&hier);
            println!(
                "step {:5}  t {:.4}  dt {:.3e}  K/K0 {:.3}  rho [{:.3e}, {:.3e}]  |v|max {:.3e}",
                ii + 1,
                hier.levels[0].state.time,
                dt,
                s.0,
                s.2,
                s.3,
                s.4
            );
        }
    }
    let st = &hier.levels[hier.levels.len() - 1].state;
    let rho = st.fields.prim.rho.view();
    let pre = st.fields.prim.pre_field().expect("adiabatic pre").view();
    let dx = st.geom.dx[0];
    let x_lo = st.geom.x_lo[0];
    let lo: [isize; 3] = std::array::from_fn(|a| st.geom.interior.spaces[a].lo as isize);
    let mut k_max = 0.0_f64;
    let mut masked = 0usize;
    for c in st.geom.interior.iter() {
        let x: [f64; 3] =
            std::array::from_fn(|a| x_lo + ((c[a] as isize - lo[a]) as f64 + 0.5) * dx);
        let r = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if r >= R_ACC {
            continue;
        }
        masked += 1;
        let (r_c, p_c) = (*rho.at(c), *pre.at(c));
        assert!(
            r_c.is_finite() && p_c.is_finite() && r_c > 0.0 && p_c > 0.0,
            "non-positive state inside the mask at {c:?}: rho {r_c}, pre {p_c}"
        );
        k_max = k_max.max(p_c / r_c.powf(GAMMA) / K0);
    }
    (k_max, dt_min, masked)
}

/// mask-interior diagnostics: (max K/K0, masked cells, rho_min, rho_max, max |v|).
fn scan(hier: &Hier) -> (f64, usize, f64, f64, f64) {
    let st = &hier.levels[hier.levels.len() - 1].state;
    let rho = st.fields.prim.rho.view();
    let pre = st.fields.prim.pre_field().expect("adiabatic pre").view();
    let vel: Vec<_> = (0..3).map(|a| st.fields.prim.vel[a].view()).collect();
    let dx = st.geom.dx[0];
    let x_lo = st.geom.x_lo[0];
    let lo: [isize; 3] = std::array::from_fn(|a| st.geom.interior.spaces[a].lo as isize);
    let (mut k_max, mut n, mut rmin, mut rmax, mut vmax) =
        (0.0_f64, 0usize, f64::INFINITY, 0.0_f64, 0.0_f64);
    for c in st.geom.interior.iter() {
        let x: [f64; 3] =
            std::array::from_fn(|a| x_lo + ((c[a] as isize - lo[a]) as f64 + 0.5) * dx);
        if x.iter().map(|v| v * v).sum::<f64>().sqrt() >= R_ACC {
            continue;
        }
        n += 1;
        let (r_c, p_c) = (*rho.at(c), *pre.at(c));
        k_max = k_max.max(p_c / r_c.powf(GAMMA) / K0);
        rmin = rmin.min(r_c);
        rmax = rmax.max(r_c);
        vmax = vmax.max(
            (0..3)
                .map(|a| vel[a].at(c) * vel[a].at(c))
                .sum::<f64>()
                .sqrt(),
        );
    }
    (k_max, n, rmin, rmax, vmax)
}

/// the masked interior stays thermodynamically coupled and the timestep stays usable.
///
/// measured, 600 steps, N = 32, four cells across the mask radius, balanced reconstruction:
///
///   acoustic dissipation floored inside the mask   max K/K_0 = 1.0282, min dt = 1.1659e-3
///   published ramp alone                           max K/K_0 = 1.1334, min dt = 1.1514e-3
///
/// the entropy the masked interior accumulates above the ambient isentrope is 0.028 with the
/// floor and 0.133 without it — a factor of five, held steady over the run (both arms are
/// falling, so the separation is the converged difference rather than a transient). the entropy
/// bound below sits at 1.08, a factor of 2.9 above the floored measurement and a factor of 1.6
/// below the ramp-alone one.
///
/// the two cells nearest the body sit deep enough in the potential that the isentrope through
/// them reaches vacuum within the reconstruction footprint, so the balanced reconstruction fades
/// its profile out there and reconstructs the state itself; the floored arm reads 1.0282 with
/// that fade and 1.0362 following the isentrope past its own domain boundary.
///
/// the timestep bound guards the second production failure — a mask cell evacuating at fixed
/// pressure drives `c_s` up and the CFL step down — and at this amplitude the two arms sit
/// within 1.3 percent of each other, so it is a floor on collapse rather than the
/// discriminating measurement.
#[test]
fn a_sealed_wall_holds_its_masked_entropy_and_timestep_under_the_low_mach_ramp() {
    let (k_max, dt_min, masked) = run();
    println!(
        "\nsealed wall in a stratified atmosphere, {STEPS} steps, {masked} masked cells\n\
         max K/K_0 inside the mask: {k_max:.4}\n\
         min dt over the run:       {dt_min:.4e}"
    );

    // the premise: the mask has to contain cells, or both measurements are over an empty set.
    assert!(
        masked >= 200,
        "only {masked} cells sit inside the mask; the entropy and timestep measurements are \
         taken over too thin an interior to separate a decoupling one from a coupled one. \
         widen the mask or refine the grid"
    );
    assert!(
        k_max < 1.08,
        "the masked interior reached K/K_0 = {k_max:.4}. an adiabatic gas behind a sealed wall \
         carries the ambient isentrope plus the wall's own friction heat; the excess above that \
         measures how much of the interior's structure the scheme is grinding down without \
         redistributing, which is what a face whose acoustic dissipation has been scaled away \
         does inside a region the penalization holds stagnant"
    );
    assert!(
        dt_min > 9.0e-4,
        "the timestep fell to {dt_min:.4e}. the CFL condition reads the sound speed, and a mask \
         cell evacuating at fixed pressure drives c_s up without bound; a floor this low is \
         that collapse in progress"
    );
}
