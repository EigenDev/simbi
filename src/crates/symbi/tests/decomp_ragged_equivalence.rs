// =============================================================================
// decomp_ragged_equivalence.rs
//
// a deliberately non-uniform partition reproduces the monolithic run exactly:
// per-axis cuts at unequal positions split the grid into tiles of different
// sizes, and the decomposed march through the production evolve loop matches
// the one-tile run cell for cell. the cuts sit near the acoustic bump so the
// unequal seams carry real flux, and the exact match holds because every tile
// extent is read from the partition itself.
//
// run: cargo test -p symbi --test decomp_ragged_equivalence
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, Partition, evolve_decomposed, flatten, unflatten};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
// a power-of-two cell count keeps DX and every tile origin binary-exact, so the
// tiles seed bitwise-identical initial states; the cuts themselves stay ragged.
const N: usize = 64;
const DX: f64 = 1.0 / N as f64;
const T_FINAL: f64 = 0.05;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;

// a smooth, centered pressure+density bump along axis 0: sound waves cross the
// interior cuts, so every unequal seam is genuinely exercised, while the outer
// outflow boundaries stay inactive and mono == decomposed is exact.
fn bump(x: f64) -> f64 {
    0.2 * (-((x - 0.5) / 0.1).powi(2)).exp()
}

fn make(cells: [usize; 2], origin: [f64; 2], bnd: Boundaries<2>) -> (Sim, Kern) {
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin(origin)
        .boundaries(bnd)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|x| {
            let b = bump(x[0]);
            Prim::adiabatic(Density(1.0 + b), Tensor::new([0.0; 2]), Pressure(1.0 + b))
        })
        .build();
    let k = Kern::new(GAMMA, CFL, &sim.geom.allocated);
    (sim, k)
}

// build one tile per partition entry: cells and origin from the partition's own
// extents, a CoarseFine face on every interior seam, physical outflow outside.
fn partition_tiles(partition: &Partition<2>) -> Vec<(Sim, Kern)> {
    let counts = partition.counts();
    (0..partition.n_tiles())
        .map(|flat| {
            let tc = unflatten(flat, counts);
            let ext = partition.tile_extents(tc);
            let cells = [ext[0].1, ext[1].1];
            let origin = [ext[0].0 as f64 * DX, ext[1].0 as f64 * DX];
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
            make(cells, origin, bnd)
        })
        .collect()
}

fn run(tiles: &mut [(Sim, Kern)], counts: [usize; 2]) {
    let devices = vec![0i32; tiles.len()];
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
        u64::MAX,
        &LocalCopy,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );
}

// scatter every tile's interior density onto the global grid by the partition's
// own extents; works for any rectilinear tile topology.
fn global_den(tiles: &[(Sim, Kern)], partition: &Partition<2>) -> Vec<f64> {
    let counts = partition.counts();
    let mut out = vec![f64::NAN; N * N];
    for (flat_tile, (sim, _)) in tiles.iter().enumerate() {
        let tc = unflatten(flat_tile, counts);
        let ext = partition.tile_extents(tc);
        let ilo: [isize; 2] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
        for c in sim.geom.interior.iter() {
            let g: [usize; 2] = std::array::from_fn(|a| ext[a].0 + (c[a] - ilo[a]) as usize);
            out[flatten(g, [N; 2])] = *sim.fields.cons.den.view().at(c);
        }
    }
    out
}

/// six tiles of five different shapes (19/18/27 cells on x, 27/37 on y), with the
/// x-cuts straddling the bump so both unequal seams carry acoustic flux; the
/// decomposed run matches the monolithic one exactly.
#[test]
fn a_ragged_partition_reproduces_the_monolithic_run_exactly() {
    let ragged = Partition::explicit([N, N], [vec![19, 37], vec![27]]).unwrap();
    assert!(!ragged.is_uniform(), "the gate must exercise unequal tiles");
    let mono = Partition::explicit([N, N], [Vec::new(), Vec::new()]).unwrap();

    let mut tiles = partition_tiles(&ragged);
    run(&mut tiles, ragged.counts());
    let decomposed = global_den(&tiles, &ragged);

    let mut one = partition_tiles(&mono);
    run(&mut one, mono.counts());
    let reference = global_den(&one, &mono);

    let mut worst = 0.0f64;
    for (i, (a, b)) in decomposed.iter().zip(reference.iter()).enumerate() {
        assert!(
            a.is_finite(),
            "uncovered global cell {i}: the scatter missed it"
        );
        worst = worst.max((a - b).abs());
        assert!(
            a.to_bits() == b.to_bits(),
            "cell {i}: decomposed {a:e} != monolithic {b:e}"
        );
    }
    println!("ragged == monolithic exactly (worst |diff| = {worst:e})");
}
