// =============================================================================
// rhd_taub_mathews.rs
//
// the taub-mathews (synge-gas) eos through the real evolve() loop: the `_tm`
// kernel twins (c2p + wave-speed map + flux) dispatch by name, and the closure
// satisfies its two thermodynamic limits on the relativistic sod —
//   cold (theta = p/rho -> 0):  h -> 1 + gamma_c/(gamma_c - 1) theta, gamma_c = 5/3
//   hot  (theta -> infinity):   h -> 4 theta,                        gamma_h = 4/3
// so a cold run must reproduce the ideal-5/3 evolution and a hot run the
// ideal-4/3 one, to the accuracy the limit itself sets (the effective gamma sits
// O(theta) from 5/3 cold and O(1/theta) from 4/3 hot). at theta ~ 1 the closure
// is genuinely NEITHER — the non-vacuity of the sandwich.
//
// run: cargo test -p symbi --test rhd_taub_mathews
// =============================================================================

use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi::EosArm;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::rhd::Rhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 128;

/// evolve a v = 0 sod with left/right (rho, p) states to `t_end`; returns the
/// interior density profile. `eos` selects the closure; `gamma` feeds the
/// ideal-gas arm and stays inert on the taub-mathews one.
fn sod_density(
    eos: EosArm,
    gamma: f64,
    left: (f64, f64),
    right: (f64, f64),
    t_end: f64,
) -> Vec<f64> {
    type Sim = SimState<Rhd, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let dx = 1.0 / N as f64;
    let mut sim = Sim::build(Rhd, IdealGas { gamma }, Cartesian)
        .cells([N])
        .spacing([dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("rhd sim construction failed")
        .set_initial(|x| {
            let (rho, pre) = if x[0] < 0.5 { left } else { right };
            Prim {
                rho,
                vel: Tensor::new([0.0]),
                pre,
            }
        })
        .build();
    let sub = RhdSubstrateKernelSet::<HostMemory, f64, 1>::new(gamma, 0.4, &sim.geom.allocated)
        .with_eos(eos);
    evolve(&mut sim, &sub, t_end).expect("rhd evolution failed");
    sim.geom
        .interior
        .iter()
        .map(|c| *sim.fields.prim.rho.view().at(c))
        .collect()
}

fn rel_linf(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / x.abs().max(y.abs()))
        .fold(0.0_f64, f64::max)
}

/// the trans-relativistic sod (theta ~ 1) under taub-mathews: finite, positive,
/// subluminal, and genuinely a different scheme from BOTH ideal limits —
/// identical output would mean the eos dispatch silently fell through.
#[test]
fn taub_mathews_sod_evolves_and_is_neither_ideal_limit() {
    let l = (1.0, 1.0);
    let r = (0.125, 0.1);
    let tm = sod_density(EosArm::TaubMathews, 5.0 / 3.0, l, r, 0.25);
    assert!(
        tm.iter().all(|v| v.is_finite() && *v > 0.0),
        "taub-mathews sod produced a non-finite or non-positive density"
    );
    let cold = sod_density(EosArm::IdealGamma, 5.0 / 3.0, l, r, 0.25);
    let hot = sod_density(EosArm::IdealGamma, 4.0 / 3.0, l, r, 0.25);
    let d_cold = rel_linf(&tm, &cold);
    let d_hot = rel_linf(&tm, &hot);
    println!("theta ~ 1 sod: |tm - ideal53| = {d_cold:.3e}, |tm - ideal43| = {d_hot:.3e}");
    // at theta ~ 1 the effective gamma sits mid-walk (~1.5), well away from both
    // limits; a few percent of profile difference against each ideal run is the
    // physics, and anything at roundoff means the dispatch fell through.
    assert!(
        d_cold > 1.0e-3 && d_hot > 1.0e-3,
        "the taub-mathews sod is indistinguishable from an ideal-gamma run \
         (|tm - 5/3| = {d_cold:.3e}, |tm - 4/3| = {d_hot:.3e}); the _tm kernels \
         did not dispatch"
    );
}

/// the cold limit: at theta ~ 1e-5 the taub-mathews effective gamma sits
/// O(theta) from 5/3, so the evolved profiles must agree to a few times that
/// scale. pressures scale down by 1e-5 (sound crossing slows by ~sqrt of that;
/// t_end stretches to keep the wave displacement resolved).
#[test]
fn cold_taub_mathews_reproduces_the_ideal_five_thirds_sod() {
    let l = (1.0, 1.0e-5);
    let r = (0.125, 1.0e-6);
    let tm = sod_density(EosArm::TaubMathews, 5.0 / 3.0, l, r, 40.0);
    let ideal = sod_density(EosArm::IdealGamma, 5.0 / 3.0, l, r, 40.0);
    let d = rel_linf(&tm, &ideal);
    println!("cold limit: |tm - ideal53| = {d:.3e}");
    assert!(
        d < 1.0e-3,
        "cold (theta ~ 1e-5) taub-mathews deviates from ideal 5/3 by {d:.3e}; \
         the closure fails its non-relativistic limit"
    );
}

/// the hot limit: at theta ~ 1e3 the effective gamma sits O(1/theta) from 4/3,
/// so the evolved profiles must agree to a few times 1e-3.
#[test]
fn hot_taub_mathews_reproduces_the_ideal_four_thirds_sod() {
    let l = (1.0, 1.0e3);
    let r = (0.125, 1.0e2);
    let tm = sod_density(EosArm::TaubMathews, 4.0 / 3.0, l, r, 0.25);
    let ideal = sod_density(EosArm::IdealGamma, 4.0 / 3.0, l, r, 0.25);
    let d = rel_linf(&tm, &ideal);
    println!("hot limit: |tm - ideal43| = {d:.3e}");
    assert!(
        d < 1.0e-2,
        "hot (theta ~ 1e3) taub-mathews deviates from ideal 4/3 by {d:.3e}; \
         the closure fails its ultra-relativistic limit"
    );
}

/// the spherical chart — the decelerating-blast geometry the synge gas exists
/// for. the tm c2p and flux are chart-free (pointwise / no geometric factor);
/// only the wave-speed map carries the chart suffix, and its `_tm_sph` twin is
/// baked with the other curvilinear cells. an over-pressured sphere drives a
/// relativistic blast that must stay finite, positive, and subluminal.
#[test]
fn taub_mathews_runs_the_spherical_blast_chart() {
    type Sim = SimState<Rhd, 1, symbi_geometry::Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
    let dx = 1.0 / N as f64;
    let mut sim = Sim::build(Rhd, IdealGas { gamma: 5.0 / 3.0 }, symbi_geometry::Spherical)
        .cells([N])
        .origin([1.0])
        .spacing([dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("rhd sim construction failed")
        .set_initial(|x| {
            let (rho, pre) = if x[0] < 1.1 { (1.0, 100.0) } else { (1.0, 0.01) };
            Prim {
                rho,
                vel: Tensor::new([0.0]),
                pre,
            }
        })
        .build();
    let sub =
        RhdSubstrateKernelSet::<HostMemory, f64, 1>::new(5.0 / 3.0, 0.4, &sim.geom.allocated)
            .with_eos(EosArm::TaubMathews);
    evolve(&mut sim, &sub, 0.2).expect("spherical tm blast failed");
    let pre = sim.fields.prim.pre_field().expect("prim.pre");
    let mut max_v = 0.0_f64;
    for c in sim.geom.interior.iter() {
        let rho = *sim.fields.prim.rho.view().at(c);
        let p = *pre.view().at(c);
        let v = *sim.fields.prim.vel[0].view().at(c);
        assert!(rho.is_finite() && rho > 0.0, "bad density {rho} at {c:?}");
        assert!(p.is_finite() && p > 0.0, "bad pressure {p} at {c:?}");
        assert!(v.abs() < 1.0, "superluminal velocity {v} at {c:?}");
        max_v = max_v.max(v.abs());
    }
    assert!(max_v > 0.3, "the blast never became relativistic (max |v| = {max_v})");
}

/// the refusal: a curved spacetime has no `_tm` twins — the first cfl dispatch
/// refuses before any kernel name could miss.
#[test]
#[should_panic(expected = "flat rhd family")]
fn taub_mathews_refuses_a_curved_spacetime() {
    use symbi_geometry::SchwarzschildKSCartesian;
    type Sim =
        SimState<Rhd, 2, SchwarzschildKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory>;
    let n = 32usize;
    let dx = 12.0 / n as f64;
    let mut sim = Sim::build(
        Rhd,
        IdealGas { gamma: 5.0 / 3.0 },
        SchwarzschildKSCartesian { mass: 1.0 },
    )
    .cells([n, n])
    .origin([-6.0, -6.0])
    .spacing([dx, dx])
    .boundaries(Boundaries::uniform(BoundaryType::Outflow))
    .allocate()
    .expect("rhd sim construction failed")
    .set_initial(|_| Prim {
        rho: 1.0,
        vel: Tensor::new([0.0, 0.0]),
        pre: 0.1,
    })
    .build();
    let sub =
        RhdSubstrateKernelSet::<HostMemory, f64, 2>::new(5.0 / 3.0, 0.3, &sim.geom.allocated)
            .with_eos(EosArm::TaubMathews);
    evolve(&mut sim, &sub, 0.01).unwrap();
}
