// =============================================================================
// tracer_drivers.rs
//
// mass-transport tracers through both production drivers: the uni-grid evolve
// loop and the single-level hierarchy must transport an identically seeded
// population to identical discrete owners. also pins that tracers cross cell
// faces, mass-weighted seeding puts more tracers where the gas is denser, and
// an undyed run is untouched.
//
// run: cargo test -p symbi --test tracer_drivers
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_sim::mass_transport::ItoOrder;
use symbi_sim::tracers::{ContinuousTracerSet, seed_mass_weighted};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const N: usize = 48;
const L: f64 = 1.0;
const N_TRACERS: usize = 300;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

fn build() -> Sim {
    let dx = 2.0 * L / N as f64;
    // a dense shear band in a lighter ambient: nontrivial velocities AND a
    // density contrast for the seeding assertion.
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .allocate()
        .expect("sim")
        .set_initial(|[_, y]: [f64; 2]| {
            let band = y.abs() < 0.4;
            Prim {
                rho: if band { 3.0 } else { 1.0 },
                vel: Tensor::new([if band { 0.4 } else { -0.4 }, 0.0]),
                pre: 1.0,
            }
        })
        .build()
}

#[test]
fn both_drivers_produce_identical_mass_ownership() {
    const T: f64 = 0.3;

    let mut sim_a = build();
    sim_a.tracers = Some(seed_mass_weighted(&sim_a, N_TRACERS));
    let owner_0 = sim_a.tracers.as_ref().unwrap().owner.clone();
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sim_a.geom.allocated);
    evolve(&mut sim_a, &sub, T).expect("uni-grid tracer drive");
    let tr_a = sim_a.tracers.take().unwrap();

    let mut sim_b = build();
    sim_b.tracers = Some(seed_mass_weighted(&sim_b, N_TRACERS));
    let sub_b =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sim_b.geom.allocated);
    let mut hier = Hierarchy::single(sim_b, sub_b);
    hier.evolve(T).expect("hierarchy tracer drive");
    let tr_b = hier.levels[0].state.tracers.take().unwrap();

    assert_eq!(tr_a.len(), N_TRACERS);
    assert_eq!(tr_a.len(), tr_b.len());
    assert_eq!(tr_a.id, tr_b.id);
    assert_eq!(tr_a.owner, tr_b.owner);
    assert_eq!(tr_a.flags, tr_b.flags);
    let moved = tr_a
        .owner
        .iter()
        .zip(&owner_0)
        .filter(|(owner, initial)| owner != initial)
        .count();
    // the shear transports the population across cell faces, so a driver
    // missing the stage transport call cannot pass vacuously.
    assert!(
        moved > N_TRACERS / 2,
        "tracers did not cross cell faces: only {moved} of {} moved",
        tr_a.len()
    );
}

#[test]
fn seeding_follows_the_mass() {
    let sim = build();
    let tr = seed_mass_weighted(&sim, N_TRACERS);
    // the band carries rho = 3 over 40% of the area vs rho = 1 elsewhere:
    // mass fraction in the band = 1.2 / (1.2 + 0.6) = 2/3 of the tracers.
    // the expectation comes from the discrete density field itself (the
    // band edge lands between cell centers, so a continuum 2/3 is off by the
    // edge rows): band mass fraction over the interior, cell by cell.
    let dx = 2.0 * L / N as f64;
    let (mut m_band, mut m_all) = (0.0, 0.0);
    for c in sim.geom.interior.iter() {
        let lo = sim.geom.interior.spaces[1].lo;
        let y = sim.geom.x_lo[1] + ((c[1] - lo) as f64 + 0.5) * dx;
        let m = *sim.fields.cons.den.view().at(c);
        m_all += m;
        if y.abs() < 0.4 {
            m_band += m;
        }
    }
    let in_band = tr.x.iter().filter(|p| p[1].abs() < 0.4).count();
    let expect = (N_TRACERS as f64 * m_band / m_all).round() as isize;
    assert!(
        (in_band as isize - expect).unsigned_abs() <= 5,
        "band holds {in_band} tracers, field expects ~{expect}"
    );
    // weight times population books the total interior mass exactly.
    let m_total = tr.weight * N_TRACERS as f64;
    let m_field = m_all * dx * dx;
    assert!(
        (m_total - m_field).abs() < 1e-10 * m_field,
        "sampled mass {m_total} vs field mass {m_field}"
    );
}

#[test]
fn continuous_tracer_concentration_tracks_gas_mass() {
    const PARTICLES: usize = 32_768;
    const BINS: usize = 12;
    const T: f64 = 0.15;

    for order in [ItoOrder::Two, ItoOrder::Three] {
        let mut sim = build();
        let seed = seed_mass_weighted(&sim, PARTICLES);
        sim.continuous_tracers =
            Some(ContinuousTracerSet::from_discrete(&seed, order).unwrap());
        let kernels =
            AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(
                GAMMA,
                0.4,
                &sim.geom.allocated,
            );

        evolve(&mut sim, &kernels, T).unwrap();

        let mut observed = vec![0usize; BINS * BINS];
        let tracers = sim.continuous_tracers.as_ref().unwrap();
        unsafe {
            for ii in 0..tracers.len {
                assert_eq!(*tracers.escaped.as_ptr::<u8>().add(ii), 0);
                let x = *tracers.x[0].as_ptr::<f64>().add(ii);
                let y = *tracers.x[1].as_ptr::<f64>().add(ii);
                let bx = (((x + L) / (2.0 * L) * BINS as f64).floor() as isize)
                    .rem_euclid(BINS as isize) as usize;
                let by = (((y + L) / (2.0 * L) * BINS as f64).floor() as isize)
                    .rem_euclid(BINS as isize) as usize;
                observed[bx + BINS * by] += 1;
            }
        }
        let mut gas = vec![0.0; BINS * BINS];
        let cells_per_bin = N / BINS;
        for coord in sim.geom.interior.iter() {
            let ix = (coord[0] - sim.geom.interior.spaces[0].lo) as usize;
            let iy = (coord[1] - sim.geom.interior.spaces[1].lo) as usize;
            gas[ix / cells_per_bin + BINS * (iy / cells_per_bin)] +=
                *sim.fields.cons.den.view().at(coord);
        }
        let gas_total: f64 = gas.iter().sum();
        let expected: Vec<_> = gas
            .iter()
            .map(|mass| PARTICLES as f64 * mass / gas_total)
            .collect();
        assert!(
            expected.iter().all(|count| *count > 50.0),
            "gas-tracer chi-square gate has an underpopulated bin"
        );
        let chi_square: f64 = observed
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| {
                let residual = *actual as f64 - expected;
                residual * residual / expected
            })
            .sum();
        let reduced = chi_square / (BINS * BINS - 1) as f64;
        assert!(
            reduced < 4.0,
            "{order:?} tracer concentration disagrees with gas: reduced chi-square {reduced:.3}"
        );
    }
}

#[test]
fn tracing_is_numerically_inert_to_the_hydro_solution() {
    const T: f64 = 0.3;

    let mut untraced = build();
    let mut traced = build();
    traced.tracers = Some(seed_mass_weighted(&traced, N_TRACERS));
    let initial_owners = traced.tracers.as_ref().unwrap().owner.clone();

    let untraced_kernels = AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(
        GAMMA,
        0.4,
        &untraced.geom.allocated,
    );
    let traced_kernels = AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(
        GAMMA,
        0.4,
        &traced.geom.allocated,
    );
    evolve(&mut untraced, &untraced_kernels, T).expect("untraced drive");
    evolve(&mut traced, &traced_kernels, T).expect("traced drive");

    let moved = traced
        .tracers
        .as_ref()
        .unwrap()
        .owner
        .iter()
        .zip(initial_owners)
        .filter(|(owner, initial)| **owner != *initial)
        .count();
    assert!(
        moved > N_TRACERS / 2,
        "tracer transport was not exercised: only {moved} tracers moved"
    );
    assert!(untraced.tracers.is_none());
    assert_eq!(traced.time.to_bits(), untraced.time.to_bits());
    assert_eq!(traced.dt.to_bits(), untraced.dt.to_bits());
    assert_eq!(traced.iteration, untraced.iteration);

    let traced_nrg = traced.fields.cons.nrg_field().unwrap();
    let untraced_nrg = untraced.fields.cons.nrg_field().unwrap();
    for coord in traced.geom.allocated.iter() {
        assert_eq!(
            traced.fields.cons.den.view().at(coord).to_bits(),
            untraced.fields.cons.den.view().at(coord).to_bits(),
            "density changed at {coord:?}"
        );
        for dd in 0..2 {
            assert_eq!(
                traced.fields.cons.mom[dd].view().at(coord).to_bits(),
                untraced.fields.cons.mom[dd].view().at(coord).to_bits(),
                "momentum {dd} changed at {coord:?}"
            );
        }
        assert_eq!(
            traced_nrg.view().at(coord).to_bits(),
            untraced_nrg.view().at(coord).to_bits(),
            "energy changed at {coord:?}"
        );
    }
}
