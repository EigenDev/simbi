// =============================================================================
// expression_sampling_point.rs
//
// a cell-sampled declared expression reads one point per cell on every host and
// compiled evaluator: the volume-weighted centroid of the cell in chart
// coordinates. per grid axis, the first moment of the cell over that axis's
// coordinate under the chart's volume element (spherical r: r^2 dr; spherical
// theta: sin(theta) d theta; cylindrical R: R dR; every other axis: the
// arithmetic midpoint of its faces). the grids are nonuniform and curvilinear,
// where this point and the map's own cell center differ: a log-radial spherical
// (r, theta) grid, a log-radial cylindrical (R, z) grid and a log-spaced
// cartesian axis. the expected point is computed here by quadrature from the
// face positions, independent of the code's closed forms.
//   - the host point equals the quadrature centroid on every axis, and differs
//     from the map center (the premise);
//   - consumers: the census interpreter and the compiled census map, a user
//     density source through the host interpreter, the compiled source kernel
//     and the fused update-plus-source kernel, and a driven boundary's ghost
//     band on the host, each reads the quadrature centroid. on the log-radial
//     axis the compiled kernels read their faces through the map's decade
//     slope. the device path of these consumers is not run here.
//
// run: cargo test -p symbi --test expression_sampling_point
// =============================================================================

use symbi::prelude::SimSubstrate;
use symbi::sim::evolve::step_once;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::{AxisMap, Cartesian, Cylindrical, Spherical};
use symbi_hydro::NEWTONIAN_SPEC;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::census::CensusEvaluator;
use symbi_sim::substrate_seam::KernelSet;
use symbi_source_compile::expr_bridge::{build_boundary_dag, build_user_source};
use symbi_source_compile::{CensusConfig, SourceConfig};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const NR: usize = 12;
const NT: usize = 6;
const R_LO: f64 = 1.0;
const R_HI: f64 = 10.0;
const T_LO: f64 = 0.05;
const T_HI: f64 = std::f64::consts::PI - 0.05;

type SphSim = SimState<Newtonian, 2, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
type CylSim = SimState<Newtonian, 2, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>;
type CartSim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

fn log_r() -> AxisMap {
    AxisMap::Log { start: R_LO, log_slope: (R_HI / R_LO).log10() / NR as f64 }
}

/// the face position of a log axis at whole-grid index `i`, written out here
fn log_face(i: isize) -> f64 {
    R_LO * 10f64.powf(i as f64 * (R_HI / R_LO).log10() / NR as f64)
}

/// the first moment `int x w(x) dx / int w(x) dx` over [lo, hi] by composite simpson quadrature
fn moment(lo: f64, hi: f64, w: impl Fn(f64) -> f64) -> f64 {
    let n = 4000;
    let h = (hi - lo) / n as f64;
    let (mut num, mut den) = (0.0, 0.0);
    for k in 0..=n {
        let x = lo + k as f64 * h;
        let c = if k == 0 || k == n { 1.0 } else if k % 2 == 1 { 4.0 } else { 2.0 };
        num += c * x * w(x);
        den += c * w(x);
    }
    num / den
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

/// the spherical grid; `driven` makes its inner radial face a driven boundary
fn sph_with(driven: bool) -> SphSim {
    let dt = (T_HI - T_LO) / NT as f64;
    let inner = if driven { BoundaryType::Driven(0) } else { BoundaryType::Outflow };
    SphSim::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells([NR, NT])
        .origin([R_LO, T_LO])
        .spacing([(R_HI - R_LO) / NR as f64, dt])
        .coord_maps(Some([log_r(), AxisMap::Uniform { start: T_LO, dx: dt }]))
        .boundaries(Boundaries::per_axis([[inner, BoundaryType::Outflow], [BoundaryType::Outflow; 2]]))
        .allocate()
        .expect("spherical sim")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)))
        .build()
}

fn sph() -> SphSim {
    sph_with(false)
}

/// the independent expected point of spherical cell (i, j): quadrature moments of r^2 and sin(theta)
fn sph_expected(c: [isize; 2]) -> [f64; 2] {
    let dt = (T_HI - T_LO) / NT as f64;
    let (tl, th) = (T_LO + c[1] as f64 * dt, T_LO + (c[1] + 1) as f64 * dt);
    [moment(log_face(c[0]), log_face(c[0] + 1), |r| r * r), moment(tl, th, f64::sin)]
}

#[test]
fn the_host_point_is_the_volume_centroid_on_every_chart() {
    let s = sph();
    let mut separated = 0.0f64;
    for c in s.geom.interior.iter() {
        let (got, want, center) = (s.geom.expression_point(c), sph_expected(c), s.geom.cell_coord(c));
        for a in 0..2 {
            assert!(rel(got[a], want[a]) < 1e-11, "spherical {c:?} axis {a}: {} against {}", got[a], want[a]);
            separated = separated.max(rel(center[a], want[a]));
        }
    }
    assert!(separated > 1e-3, "the map center and the centroid agree to {separated:.1e} on this grid; the gate is vacuous");

    let dz = 0.5;
    let cyl = CylSim::build(Newtonian, IdealGas { gamma: GAMMA }, Cylindrical)
        .cells([NR, 4])
        .origin([R_LO, -1.0])
        .spacing([(R_HI - R_LO) / NR as f64, dz])
        .coord_maps(Some([log_r(), AxisMap::Uniform { start: -1.0, dx: dz }]))
        .allocate()
        .expect("cylindrical sim")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)))
        .build();
    for c in cyl.geom.interior.iter() {
        let got = cyl.geom.expression_point(c);
        let want = [moment(log_face(c[0]), log_face(c[0] + 1), |r| r), -1.0 + (c[1] as f64 + 0.5) * dz];
        for a in 0..2 {
            assert!(rel(got[a], want[a]) < 1e-11, "cylindrical {c:?} axis {a}: {} against {}", got[a], want[a]);
        }
    }

    let cart = CartSim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([NR])
        .origin([R_LO])
        .spacing([(R_HI - R_LO) / NR as f64])
        .coord_maps(Some([log_r()]))
        .allocate()
        .expect("cartesian sim")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0]), Pressure(1.0)))
        .build();
    for c in cart.geom.interior.iter() {
        let want = moment(log_face(c[0]), log_face(c[0] + 1), |_| 1.0);
        let got = cart.geom.expression_point(c)[0];
        assert!(rel(got, want) < 1e-11, "cartesian log {c:?}: {got} against {want}");
        assert!(rel(cart.geom.cell_coord(c)[0], want) > 1e-4, "the log map center is the face midpoint; the gate is vacuous");
    }
}

#[test]
fn the_census_interpreter_and_the_compiled_map_read_the_centroid() {
    let s = sph();
    let kernels = s.substrate();
    kernels.c2p(&s);
    let cfg = CensusConfig::from_json(
        r#"{"name": "position", "axes": [], "values": [0, 1], "value_names": ["r", "theta"], "op": "add",
            "params": [], "nodes": [{"op": "VARIABLE_X1"}, {"op": "VARIABLE_X2"}]}"#,
    )
    .expect("census config");
    let ev = CensusEvaluator::new(&cfg).expect("census compiles");
    ev.lower().expect("the census lowers to a compiled map");
    let interpreted = s.census_fields(&ev, None).expect("host sim");
    let compiled = s.store.census_scratch(&ev).expect("host sim");
    assert!(
        symbi::regimes::substrate_kernels::census_compiled::census_map_compiled(&s.store, &ev, &compiled.values, &compiled.segment, true),
        "the compiled map declined the census"
    );
    for c in s.geom.interior.iter() {
        let want = sph_expected(c);
        for a in 0..2 {
            let (i, k) = (*interpreted.values[a].view().at(c), *compiled.values[a].view().at(c));
            assert!(rel(i, want[a]) < 1e-11, "interpreted census {c:?} axis {a}: {i} against {}", want[a]);
            assert!(rel(k, want[a]) < 1e-11, "compiled census {c:?} axis {a}: {k} against {}", want[a]);
        }
    }
}

/// the density each cell of the spherical grid gains over one forward-euler step of `dt` under the
/// raw source `d(rho)/dt = x_1`, from gas at rest, where the density flux vanishes
#[derive(Clone, Copy, PartialEq)]
enum SourcePath {
    Interpreted,
    Compiled,
    Fused,
}

fn density_gain(path: SourcePath, dt: f64) -> Vec<([isize; 2], f64)> {
    let mut s = sph();
    s.timestepping = Timestepping::Euler;
    let cfg = SourceConfig::from_json(
        r#"{"kind": "raw", "dim": 2, "outputs": [0], "params": [], "vocabulary": {"reads": ["x_0"], "params": []},
            "target": "den", "nodes": [{"op": "VARIABLE_X1"}]}"#,
    )
    .expect("source config");
    let built = build_user_source(&cfg, &NEWTONIAN_SPEC).expect("source");
    let kernels = match path {
        SourcePath::Fused => s.substrate().with_fused_runtime_source(built, cfg.params),
        _ => s.substrate().with_runtime_source(built, cfg.params),
    };
    // withholding the standalone compiled kernel leaves the host interpreter on the two-pass path
    // and engages the fused update-plus-source kernel, the order production falls back in
    if path != SourcePath::Compiled {
        assert!(kernels.runtime_source.as_ref().unwrap().interpret_on_host(), "the compiled kernel was built first");
    }
    kernels.c2p(&s);
    kernels.ghost_fill(&s);
    let before: Vec<f64> = s.geom.interior.iter().map(|c| *s.fields.cons.den.view().at(c)).collect();
    step_once(&mut s, &kernels, dt);
    if path == SourcePath::Fused {
        assert_eq!(kernels.runtime_source.as_ref().unwrap().fused_cpu_state(), Some(true), "the fused kernel never ran");
    }
    s.geom.interior.iter().zip(before).map(|(c, b)| (c, (*s.fields.cons.den.view().at(c) - b) / dt)).collect()
}

#[test]
fn a_user_source_reads_the_centroid_through_the_interpreter_and_the_compiled_kernels() {
    let dt = 1e-6;
    let paths = [("interpreted", SourcePath::Interpreted), ("compiled", SourcePath::Compiled), ("fused", SourcePath::Fused)];
    for (label, path) in paths {
        for (c, gain) in density_gain(path, dt) {
            let want = sph_expected(c)[0];
            assert!(rel(gain, want) < 1e-8, "{label} source {c:?}: {gain} against {want}");
        }
    }
}

#[test]
fn a_driven_boundary_on_the_host_reads_the_centroid_of_its_ghost_cells() {
    let s = sph_with(true);
    let cfg = SourceConfig::from_json(
        r#"{"kind": "dirichlet", "dim": 2, "outputs": [0, 1, 2, 3], "params": [],
            "nodes": [{"op": "VARIABLE_X1"}, {"op": "CONSTANT", "value": 0.0}, {"op": "CONSTANT", "value": 0.0},
                      {"op": "VARIABLE_X2"}]}"#,
    )
    .expect("boundary config");
    let built = build_boundary_dag(&cfg, &NEWTONIAN_SPEC).expect("driven boundary");
    let (kernels, id) = s.substrate().with_driven_boundary(built, cfg.params.clone());
    assert_eq!(id, 0);
    kernels.c2p(&s);
    kernels.ghost_fill(&s);
    let int = &s.geom.interior;
    let mut checked = 0;
    for c in s.geom.allocated.iter() {
        if c[0] >= int.spaces[0].lo || c[1] < int.spaces[1].lo || c[1] >= int.spaces[1].hi {
            continue;
        }
        let want = sph_expected(c);
        let rho = *s.fields.prim.rho.view().at(c);
        let pre = *s.fields.prim.pre_field().unwrap().view().at(c);
        assert!(rel(rho, want[0]) < 1e-11, "ghost {c:?} r: {rho} against {}", want[0]);
        assert!(rel(pre, want[1]) < 1e-11, "ghost {c:?} theta: {pre} against {}", want[1]);
        checked += 1;
    }
    assert!(checked >= NT, "the driven band covered {checked} ghost cells; the gate is vacuous");
}




