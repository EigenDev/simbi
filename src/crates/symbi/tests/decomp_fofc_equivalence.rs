// =============================================================================
// decomp_fofc_equivalence.rs
//
// the tiling-invariance contract for the first-order flux correction on the
// decomposed path: an EXCISED cartesian kerr-schild box carrying a cold
// atmosphere in prescribed mach-17 radial infall trips the correction where the
// supersonic stream meets the excision rim's vacuum-floor cells, while staying
// recoverable — an unexcised clamped core is a PERSISTENT poison that (correctly)
// fires the freeze-streak fail-loud instead. the per-tile redo — driven from
// exchange-fresh stage-input halos with the failure reduction evaluated per tile —
// reproduces the monolithic correction bit-for-bit, THROUGH genuine correction
// events (the fallback counters assert non-vacuity).
//
// the infall is prescribed in the initial data, so the correction fires from the
// first step: a scheme accurate enough to keep the rim recoverable on its own
// would otherwise silence the correction and leave the equivalence vacuous, which
// the fallback-count assertions report as a failure.
// =============================================================================

use symbi::regimes::fofc::{fofc_reset_stats, fofc_stats};
use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, evolve_decomposed, flatten, unflatten};
use symbi::sim::state::*;
use symbi::sim::substrate_seam::WithExcision;
use symbi_algebra::Tensor;
use symbi_geometry::SchwarzschildKSCartesian;
use symbi_hydro::Rhd;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 4.0 / 3.0;
const CFL: f64 = 0.3;
const N: usize = 32;
const L: f64 = 1.2;
const DX: f64 = 2.0 * L / N as f64;
const MASS: f64 = 0.3;
// long enough that the prescribed infall reaches the excision rim and the correction
// fires there over many substages, short enough to stay a unit-test-scale run.
const T_FINAL: f64 = 5.0e-4;
// any dt below this on a 32^2 unit-scale grid means the source-admissibility rate has
// started scaling with the pressure again; fail fast and loud instead of running forever.
const DT_FLOOR: f64 = 1.0e-9;
const R_EXC: f64 = 0.35;
// prescribed radial infall speed. mach 173 against the cs = 1.155e-3 atmosphere, so the
// stream reaching the excision rim is firmly supersonic and the rim's vacuum-floor cells
// drive the high-order recovery out of the physical set.
const V_INFALL: f64 = 0.2;

type Sim = SimState<Rhd, 2, SchwarzschildKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = RhdSubstrateKernelSet<HostMemory, f64, 2>;

fn make(cells: [usize; 2], origin: [f64; 2], bnd: Boundaries<2>) -> (Sim, Kern) {
    let sim = Sim::build(
        Rhd,
        IdealGas { gamma: GAMMA },
        SchwarzschildKSCartesian { mass: MASS },
    )
    .cells(cells)
    .spacing([DX; 2])
    .origin(origin)
    .boundaries(bnd)
    .timestepping(Timestepping::Rk2)
    .allocate()
    .expect("sim construction failed")
    // a cold atmosphere (p/rho = 1e-6, so cs = sqrt(gamma p / rho) = 1.155e-3) falling
    // radially inward at 0.2c — mach 173. the infall is PRESCRIBED in the initial data, so
    // the correction fires from the first step at a rate set by the initial data alone,
    // independent of how fast the well steepens the flow. the supersonic stream meets the excision rim's
    // vacuum-floor cells and the high-order c2p there leaves the physical set
    // intermittently: the deliberate, RECOVERABLE FOFC trigger.
    //
    // p/rho = 1e-6 is deliberately colder than the correction needs: the admissibility
    // rate of the geometric source is charged against the covariant energy, whose
    // killing-energy source vanishes on a stationary metric, so the admissible dt is
    // independent of the pressure. charging the valencia form instead makes dt
    // proportional to p and collapses this atmosphere into the floor below — this
    // temperature is therefore also the regression guard on that rate.
    .set_initial(|x| {
        let r = (x[0] * x[0] + x[1] * x[1]).sqrt().max(1.0e-12);
        Prim {
            rho: 1.0,
            vel: Tensor::new([-V_INFALL * x[0] / r, -V_INFALL * x[1] / r]),
            pre: 1.0e-6,
        }
    })
    .build();
    let k = Kern::new(GAMMA, CFL, &sim.geom.allocated).with_excision(R_EXC, 1.0, 1.0);
    (sim, k)
}

fn grid_tiles(counts: [usize; 2]) -> Vec<(Sim, Kern)> {
    let m: [usize; 2] = std::array::from_fn(|a| {
        assert!(N % counts[a] == 0, "N must split evenly into counts[{a}]");
        N / counts[a]
    });
    let total: usize = counts.iter().product();
    (0..total)
        .map(|flat| {
            let tc = unflatten(flat, counts);
            let origin = std::array::from_fn(|a| -L + tc[a] as f64 * m[a] as f64 * DX);
            let bnd = Boundaries(std::array::from_fn(|a| {
                let lo = if tc[a] == 0 {
                    BoundaryType::Outflow
                } else {
                    BoundaryType::CoarseFine
                };
                let hi = if tc[a] == counts[a] - 1 {
                    BoundaryType::Outflow
                } else {
                    BoundaryType::CoarseFine
                };
                [lo, hi]
            }));
            make(m, origin, bnd)
        })
        .collect()
}

fn run(tiles: &mut [(Sim, Kern)], counts: [usize; 2]) -> u64 {
    fofc_reset_stats();
    let devices: Vec<i32> = vec![0; tiles.len()];
    let mut stores = Vec::new();
    let mut kernels = Vec::new();
    for (s, k) in tiles.iter_mut() {
        stores.push(&mut **s);
        kernels.push(&*k);
    }
    evolve_decomposed(
        &mut stores,
        &kernels,
        counts,
        &devices,
        Timestepping::Rk2,
        0.0,
        T_FINAL,
        1,
        &LocalCopy,
        |_, _, stores| {
            // dt-collapse guard: below the floor the run would spin to the time
            // limit, so it fails fast here with the collapsed dt named.
            let dt = stores[0].dt;
            assert!(
                dt > DT_FLOOR,
                "dt collapsed to {dt:.3e} at t = {:.6e}: the cold-rim collapse returned",
                stores[0].time,
            );
            std::ops::ControlFlow::Continue(())
        },
    );
    let (fallback, _freeze) = fofc_stats();
    fallback
}

fn global_den(tiles: &[(Sim, Kern)], counts: [usize; 2]) -> Vec<f64> {
    let m: [usize; 2] = std::array::from_fn(|a| N / counts[a]);
    let mut out = vec![f64::NAN; N * N];
    for (flat_tile, (sim, _)) in tiles.iter().enumerate() {
        let tc = unflatten(flat_tile, counts);
        let ilo: [isize; 2] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
        for c in sim.geom.interior.iter() {
            let g: [usize; 2] = std::array::from_fn(|a| tc[a] * m[a] + (c[a] - ilo[a]) as usize);
            out[flatten(g, [N; 2])] = *sim.fields.cons.den.view().at(c);
        }
    }
    out
}

#[test]
fn fofc_fires_and_tiled_matches_monolithic() {
    let mut mono = grid_tiles([1, 1]);
    let mono_fallbacks = run(&mut mono, [1, 1]);
    let mono_vals = global_den(&mono, [1, 1]);
    // non-vacuous: the correction genuinely fired on the monolithic run — this
    // setup exists to exercise FOFC, and a quiet run proves nothing.
    assert!(
        mono_fallbacks > 0,
        "the cold-core setup never tripped FOFC; the equivalence would be vacuous"
    );

    let mut dec = grid_tiles([2, 2]);
    let dec_fallbacks = run(&mut dec, [2, 2]);
    let dec_vals = global_den(&dec, [2, 2]);
    assert!(dec_fallbacks > 0, "the tiled run never tripped FOFC");

    assert!(
        mono_vals.iter().all(|v| v.is_finite()) && dec_vals.iter().all(|v| v.is_finite()),
        "some global cells were never written (gather bug) or the state broke"
    );
    let max_err = mono_vals
        .iter()
        .zip(&dec_vals)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_err < 1e-12,
        "decomposed FOFC diverged from monolithic through {mono_fallbacks} correction \
         events: max den err {max_err:e}"
    );
}
