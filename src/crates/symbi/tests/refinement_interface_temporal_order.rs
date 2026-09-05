// =============================================================================
// refinement_interface_temporal_order.rs
//
// the temporal order of a two-level hydro hierarchy, resolved by a cell's distance from the
// coarse-fine interface. the bulk of both levels carries the RK2 order in the timestep alone. the
// interface layer (two cells to either side) carries a residual of order dt times dx: the fine
// ghosts are interpolated from the coarse state of the step, while the reflux that follows the
// subcycle corrects the coarse cells at the interface by dt times the coarse-fine flux mismatch,
// itself of order dx^2 on smooth flow. at a fixed grid that residual is first order in dt with a
// coefficient proportional to dx; under joint refinement of dt and dx it is second order. the
// gates pin both readings, so a measurement of the composition's temporal order on a refined
// hierarchy reads the bulk.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 3>;
type Hier = Hierarchy<Newtonian, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

const N: usize = 32;
const GAMMA: f64 = 5.0 / 3.0;
const VEL: [f64; 3] = [0.3, 0.2, 0.1];

fn build() -> Hier {
    let dx = 1.0 / N as f64;
    let k = 2.0 * std::f64::consts::PI;
    let kset = |s: &Sim| Kset::new(GAMMA, 0.3, &s.geom.allocated);
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .origin([-0.5; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("root construction")
        .set_initial(move |[x, _y, _z]| Prim::adiabatic(Density(1.0), Tensor::new(VEL), Pressure(1.0 + 0.1 * (k * x).sin())))
        .build();
    let ck = kset(&coarse);
    let regions = [RefinementRegion {
        x_lo: [-0.375; 3],
        x_hi: [0.375; 3],
    }];
    let hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, kset).unwrap();
    hier.seed_fine_from_coarse().expect("fine seed");
    let mut hier = hier;
    hier.prime();
    hier
}

// the fine level's cells at least three from its boundary quarter their fixed-time difference when
// the timestep halves. the coarse leaf shell of this box is four cells wide, every cell of it
// within the reconstruction reach of a coverage face, so it carries the interface residual
// throughout and is reported without a pin; the 1D gate below reads the coarse side.
#[test]
fn the_fine_bulk_is_second_order_in_the_timestep_alone() {
    let run = |dt: f64, nsteps: usize| -> Hier {
        let mut hier = build();
        for _ in 0..nsteps {
            hier.step_root_with_dt(dt);
        }
        hier
    };
    let dt = 2.0e-3;
    let runs = [run(dt, 4), run(dt / 2.0, 8), run(dt / 4.0, 16), run(dt / 8.0, 32)];
    let interior = runs[0].levels[1].state.geom.interior.clone();
    let mut fine = vec![[0.0f64; 4]; 6];
    for cell in interior.iter() {
        let dist = (0..3).map(|k| (cell[k] - interior.spaces[k].lo).min(interior.spaces[k].hi - 1 - cell[k])).min().unwrap().min(5) as usize;
        for i in 0..3 {
            let x = *runs[i].levels[1].state.fields.cons.den.at(cell);
            let y = *runs[i + 1].levels[1].state.fields.cons.den.at(cell);
            fine[dist][i] += (x - y).powi(2);
        }
        fine[dist][3] += 1.0;
    }
    let cinterior = runs[0].levels[0].state.geom.interior.clone();
    let cov = runs[0].levels[0].coverage.as_ref().unwrap().clone();
    let mut coarse = vec![[0.0f64; 4]; 3];
    for cell in cinterior.iter() {
        if cov.contains(cell) {
            continue;
        }
        let dist = (0..3).map(|k| (cov.spaces[k].lo - 1 - cell[k]).max(cell[k] - cov.spaces[k].hi)).max().unwrap().max(0).min(2) as usize;
        for i in 0..3 {
            let x = *runs[i].levels[0].state.fields.cons.den.at(cell);
            let y = *runs[i + 1].levels[0].state.fields.cons.den.at(cell);
            coarse[dist][i] += (x - y).powi(2);
        }
        coarse[dist][3] += 1.0;
    }
    let report = |label: &str, k: usize, [a, b, c, n]: &[f64; 4]| -> [f64; 2] {
        let ratios = [(a / b.max(1e-300)).sqrt(), (b / c.max(1e-300)).sqrt()];
        println!("[hydro 3d] {label} distance {k}: abs diffs {:.3e} {:.3e} {:.3e} ratios {:.2} {:.2} over {n}", (a / n).sqrt(), (b / n).sqrt(), (c / n).sqrt(), ratios[0], ratios[1]);
        ratios
    };
    for (k, layer) in fine.iter().enumerate() {
        let r = report("fine density", k, layer);
        assert!(layer[2] > 0.0, "vacuous fine layer {k}");
        if k >= 3 {
            assert!(r[0] > 3.5 && r[1] > 3.5, "the fine bulk (distance {k}) is not second order in the timestep: ratios {:.2} {:.2}", r[0], r[1]);
        }
    }
    for (k, layer) in coarse.iter().enumerate() {
        report("coarse leaf density", k, layer);
    }
}

type Sim1 = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset1 = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;
type Hier1 = Hierarchy<Newtonian, 1, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset1>;

fn build_1d(n: usize) -> Hier1 {
    let dx = 1.0 / n as f64;
    let k = 2.0 * std::f64::consts::PI;
    let kset = |s: &Sim1| Kset1::new(GAMMA, 0.3, &s.geom.allocated);
    let coarse = Sim1::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n])
        .origin([-0.5])
        .spacing([dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("root construction")
        .set_initial(move |[x]| Prim::adiabatic(Density(1.0 + 0.1 * (k * x).sin()), Tensor::new([0.3]), Pressure(1.0)))
        .build();
    let ck = kset(&coarse);
    let regions = [RefinementRegion { x_lo: [-0.25], x_hi: [0.25] }];
    let hier = Hierarchy::with_refinement(coarse, ck, &regions, ProlongOrder::Ppm, kset).unwrap();
    hier.seed_fine_from_coarse().expect("fine seed");
    let mut hier = hier;
    hier.prime();
    hier
}

// the interface layer's fixed-time residual across three resolutions with the timestep sequence
// scaled with the spacing: the residual is first order in the timestep at every grid, and halving
// the spacing and the timestep together divides it by four, the signature of a term proportional
// to dt times dx. pooled, the fine cells at least three from the boundary and the coarse leaf
// cells at least five from the coverage quarter when the timestep alone halves.
#[test]
fn the_interface_residual_is_first_order_in_the_timestep_with_a_coefficient_proportional_to_the_spacing() {
    let mut interface_by_n: Vec<f64> = Vec::new();
    for n in [32usize, 64, 128] {
        let dt0 = 0.25e-3 * 64.0 / n as f64;
        let run = |dt: f64, nsteps: usize| -> Hier1 {
            let mut hier = build_1d(n);
            for _ in 0..nsteps {
                hier.step_root_with_dt(dt);
            }
            hier
        };
        let runs = [run(dt0, 8), run(dt0 / 2.0, 16), run(dt0 / 4.0, 32), run(dt0 / 8.0, 64)];
        let interior = runs[0].levels[1].state.geom.interior.clone();
        let cov = runs[0].levels[0].coverage.as_ref().unwrap().clone();
        let cinterior = runs[0].levels[0].state.geom.interior.clone();
        let mut fine = vec![[0.0f64; 4]; 9];
        for cell in interior.iter() {
            let dist = (cell[0] - interior.spaces[0].lo).min(interior.spaces[0].hi - 1 - cell[0]).min(8) as usize;
            for i in 0..3 {
                let x = *runs[i].levels[1].state.fields.cons.den.at(cell);
                let y = *runs[i + 1].levels[1].state.fields.cons.den.at(cell);
                fine[dist][i] += (x - y).powi(2);
            }
            fine[dist][3] += 1.0;
        }
        let mut coarse = vec![[0.0f64; 4]; 9];
        for cell in cinterior.iter() {
            if cov.contains(cell) {
                continue;
            }
            let dist = (cov.spaces[0].lo - 1 - cell[0]).max(cell[0] - cov.spaces[0].hi).max(0).min(8) as usize;
            for i in 0..3 {
                let x = *runs[i].levels[0].state.fields.cons.den.at(cell);
                let y = *runs[i + 1].levels[0].state.fields.cons.den.at(cell);
                coarse[dist][i] += (x - y).powi(2);
            }
            coarse[dist][3] += 1.0;
        }
        for (k, [a, b, c, m]) in fine.iter().enumerate() {
            println!("[1d N={n}] fine density distance {k}: abs diffs {:.3e} {:.3e} {:.3e} ratios {:.2} {:.2} over {m}", (a / m).sqrt(), (b / m).sqrt(), (c / m).sqrt(), (a / b.max(1e-300)).sqrt(), (b / c.max(1e-300)).sqrt());
        }
        for (k, [a, b, c, m]) in coarse.iter().enumerate() {
            println!("[1d N={n}] coarse leaf density distance {k}: abs diffs {:.3e} {:.3e} {:.3e} ratios {:.2} {:.2} over {m}", (a / m).sqrt(), (b / m).sqrt(), (c / m).sqrt(), (a / b.max(1e-300)).sqrt(), (b / c.max(1e-300)).sqrt());
        }
        let boundary = &fine[0];
        let r = [(boundary[0] / boundary[1]).sqrt(), (boundary[1] / boundary[2]).sqrt()];
        assert!(r[0] > 1.7 && r[0] < 2.6 && r[1] > 1.7 && r[1] < 2.6, "N={n}: the fine boundary layer is not first order in the timestep: ratios {:.2} {:.2}", r[0], r[1]);
        // the reflux corrects the coarse cell at the interface; the reconstruction reaches three
        // cells, so the coarse residual is read out to four cells from the coverage. the bulk bins
        // are pooled: a two-cell bin at a node of the wave sits at roundoff after 64 steps.
        for (label, layers, from) in [("fine", &fine, 3usize), ("coarse leaf", &coarse, 5)] {
            let pooled: [f64; 4] = std::array::from_fn(|i| layers.iter().skip(from).map(|l| l[i]).sum());
            let rb = [(pooled[0] / pooled[1]).sqrt(), (pooled[1] / pooled[2]).sqrt()];
            println!("[1d N={n}] pooled {label} bulk: abs diffs {:.3e} {:.3e} {:.3e} ratios {:.2} {:.2} over {}", (pooled[0] / pooled[3]).sqrt(), (pooled[1] / pooled[3]).sqrt(), (pooled[2] / pooled[3]).sqrt(), rb[0], rb[1], pooled[3]);
            assert!((pooled[2] / pooled[3]).sqrt() > 1e-13, "N={n}: the {label} bulk measurement sits at roundoff");
            assert!(rb[0] > 3.5 && rb[1] > 3.5, "N={n}: the {label} bulk is not second order in the timestep: ratios {:.2} {:.2}", rb[0], rb[1]);
        }
        interface_by_n.push((boundary[0] / boundary[3]).sqrt());
    }
    for w in interface_by_n.windows(2) {
        let q = w[0] / w[1];
        println!("[1d] interface residual, spacing and timestep halved together: divided by {q:.2}");
        assert!((q - 4.0).abs() < 0.4, "the interface residual is not proportional to dt times dx: joint halving divided it by {q:.2}");
    }
}
