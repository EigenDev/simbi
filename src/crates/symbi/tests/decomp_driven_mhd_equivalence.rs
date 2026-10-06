// =============================================================================
// decomp_driven_mhd_equivalence.rs
//
// the driven-boundary correctness contract for decomposed MHD: a 2d RMHD grid split into
// tiles, each registering the same boundary DAG, must reproduce the monolithic run bit
// for bit and keep div(B) at machine zero across the tile cuts. the prescription covers
// the hydro prims + the cell B; the staggered face B rides the CT ghost fill and the
// transverse halo exchange.
//
// the inflow is position-dependent (rho = 2 + 0.2*y, the cell's global y) and carries a uniform
// in-plane B_x and an out-of-plane B_z, div-free; the transverse flow the density gradient drives
// makes E_z = v_x B_y - v_y B_x nonzero, so the staggered faces evolve and cross the cuts. the run
// lasts until the inflow reaches the cut half a domain from the driven face. the
// cut-along-the-face tiling is the decisive case for the prescription: a tile evaluating it at
// its tile-local y injects the wrong profile on every non-origin tile.
// =============================================================================

use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, evolve_decomposed, flatten, unflatten};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::RMHD_SPEC;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_source_compile::SourceConfig;
use symbi_source_compile::expr_bridge::build_boundary_dag;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const N: usize = 32; // cells per axis
const DX: f64 = 1.0 / N as f64;
const T_FINAL: f64 = 0.02;
/// the relativistic arms run until the inflow crosses the cut half a domain from the driven face
const T_REACH: f64 = 0.6;
const BX0: f64 = 0.2;
const BZ0: f64 = 0.3;

type Sim = SimStateGeneric<Rmhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = RmhdSubstrateKernelSet<HostMemory, f64, 2>;

fn boundary_json() -> String {
    // the rmhd prescription [rho, v1, v2, v3, pre, B1, B2, B3]:
    // rho = 2 + 0.2*y (VARIABLE_X2 -> global y), v = (0.5, 0, 0), p = 3, B = (0.2, 0, 0.5).
    r#"{
        "kind": "dirichlet", "dim": 3, "outputs": [4, 5, 6, 6, 7, 1, 6, 8], "params": [],
        "nodes": [ {"op": "VARIABLE_X2"}, {"op": "CONSTANT", "value": 0.2},
                   {"op": "MULTIPLY", "left": 0, "right": 1}, {"op": "CONSTANT", "value": 2.0},
                   {"op": "ADD", "left": 3, "right": 2},
                   {"op": "CONSTANT", "value": 0.5}, {"op": "CONSTANT", "value": 0.0},
                   {"op": "CONSTANT", "value": 3.0}, {"op": "CONSTANT", "value": 0.5} ]
    }"#
    .to_string()
}

// quiescent interior with a uniform field (B_x on the faces, B_z out of plane; div-free),
// distinct from the inflow so the ghost state provably comes from the DAG. the tile sits at the
// whole grid's origin plus the index offset of its first cell, as production places it.
fn make(cells: [usize; 2], offset: [usize; 2], bnd: Boundaries<2>) -> (Sim, Kern) {
    let sim = Sim::build(Rmhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells)
        .spacing([DX; 2])
        .origin([0.0; 2])
        .index_offset(std::array::from_fn(|a| offset[a] as isize))
        .boundaries(bnd)
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("mhd sim construction failed")
        .set_initial(|_| {
            MhdPrim::new(
                Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
                Tensor::new([BX0, 0.0, BZ0]),
            )
        })
        .seed_faces_uniform([BX0, 0.0])
        .build();
    let cfg = SourceConfig::from_json(&boundary_json()).expect("parse boundary config");
    let built = build_boundary_dag(&cfg, &RMHD_SPEC).expect("lower rmhd boundary dag");
    let (k, id) = Kern::new(GAMMA, CFL, 1.0, &sim.geom.allocated)
        .with_driven_boundary(built, cfg.params.clone());
    assert_eq!(id, 0, "first registration is id 0 (matches Driven(0))");
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
            let offset: [usize; 2] = std::array::from_fn(|a| tc[a] * m[a]);
            let phys_lo = [BoundaryType::Driven(0), BoundaryType::Outflow];
            let bnd = Boundaries(std::array::from_fn(|a| {
                let lo = if tc[a] == 0 {
                    phys_lo[a]
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
            make(m, offset, bnd)
        })
        .collect()
}

fn run(tiles: &mut [(Sim, Kern)], counts: [usize; 2]) {
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
        T_REACH,
        u64::MAX,
        &LocalCopy,
        |_, _, _| std::ops::ControlFlow::Continue(()),
    );
}

fn global_field(
    tiles: &[(Sim, Kern)],
    counts: [usize; 2],
    pick: impl Fn(&Sim, [isize; 2]) -> f64,
) -> Vec<f64> {
    let m: [usize; 2] = std::array::from_fn(|a| N / counts[a]);
    let mut out = vec![f64::NAN; N * N];
    for (flat_tile, (sim, _)) in tiles.iter().enumerate() {
        let tc = unflatten(flat_tile, counts);
        let ilo: [isize; 2] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
        for c in sim.geom.interior.iter() {
            let g: [usize; 2] = std::array::from_fn(|a| tc[a] * m[a] + (c[a] - ilo[a]) as usize);
            out[flatten(g, [N; 2])] = pick(sim, c);
        }
    }
    out
}

// max |div(B)| over a tile's interior: the staggered CT update keeps this at machine zero; a
// broken exchange or a div-injecting boundary fill spikes it at the cut- or face-adjacent cells.
fn div_b_max(sim: &Sim) -> f64 {
    let mhd = sim.fields.mhd.as_ref().expect("rmhd has mhd fields");
    let mut worst = 0.0_f64;
    for c in sim.geom.interior.iter() {
        let mut div = 0.0;
        for d in 0..2 {
            let mut chi = c;
            chi[d] += 1;
            div += (*mhd.bface[d].view().at(chi) - *mhd.bface[d].view().at(c)) / DX;
        }
        worst = worst.max(div.abs());
    }
    worst
}

fn max_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn assert_driven_mhd_matches(counts: [usize; 2]) {
    let den = |s: &Sim, c: [isize; 2]| *s.fields.cons.den.view().at(c);
    let momx = |s: &Sim, c: [isize; 2]| *s.fields.cons.mom[0].view().at(c);
    let bz = |s: &Sim, c: [isize; 2]| *s.fields.mhd.as_ref().unwrap().bcell[2].view().at(c);
    let bx = |s: &Sim, c: [isize; 2]| *s.fields.mhd.as_ref().unwrap().bcell[0].view().at(c);

    let mut mono = grid_tiles([1, 1]);
    run(&mut mono, [1, 1]);
    let (mono_den, mono_momx, mono_bz) = (
        global_field(&mono, [1, 1], den),
        global_field(&mono, [1, 1], momx),
        global_field(&mono, [1, 1], bz),
    );

    let mut dec = grid_tiles(counts);
    run(&mut dec, counts);
    let (dec_den, dec_momx, dec_bz) = (
        global_field(&dec, counts, den),
        global_field(&dec, counts, momx),
        global_field(&dec, counts, bz),
    );

    assert!(
        mono_den.iter().all(|v| v.is_finite()) && dec_den.iter().all(|v| v.is_finite()),
        "some global cells were never written"
    );
    let total_momx: f64 = mono_momx.iter().map(|v| v.abs()).sum();
    assert!(
        total_momx > 1e-3,
        "driven inflow produced no momentum ({total_momx:e}); test is vacuous"
    );
    // the injected B_z must have entered too, else the bcell prescription is untested.
    let bz_shift: f64 = mono_bz.iter().map(|v| (v - BZ0).abs()).sum();
    assert!(
        bz_shift > 1e-6,
        "the ghost B_z never influenced the interior ({bz_shift:e}); test is vacuous"
    );
    // the in-plane field evolved under CT, and the inflow reached the cut columns at x = 1/2
    let (mono_bx, dec_bx) = (global_field(&mono, [1, 1], bx), global_field(&dec, counts, bx));
    let bx_shift: f64 = mono_bx.iter().map(|v| (v - BX0).abs()).sum();
    assert!(bx_shift > 1e-6, "the in-plane field never evolved ({bx_shift:e}); the CT path is untested");
    let cut_momx: f64 = (0..N).map(|j| mono_momx[flatten([N / 2 - 1, j], [N; 2])].abs() + mono_momx[flatten([N / 2, j], [N; 2])].abs()).sum();
    assert!(cut_momx > 1e-6, "the inflow never reached the cut columns ({cut_momx:e}); the cut carries no signal");
    assert!(max_err(&mono_bx, &dec_bx) == 0.0, "{counts:?} bcell_x diverged");

    assert!(
        max_err(&mono_den, &dec_den) == 0.0,
        "{counts:?} density diverged"
    );
    assert!(
        max_err(&mono_momx, &dec_momx) == 0.0,
        "{counts:?} mom_x diverged"
    );
    assert!(
        max_err(&mono_bz, &dec_bz) == 0.0,
        "{counts:?} bcell_z diverged"
    );

    let dbm = dec
        .iter()
        .map(|(s, _)| div_b_max(s))
        .fold(0.0_f64, f64::max);
    let dbm_mono = div_b_max(&mono[0].0);
    assert!(
        dbm < 1e-12,
        "{counts:?} div(B) = {dbm:e} across cuts under a driven boundary"
    );
    assert!(
        dbm_mono < 1e-12,
        "monolithic div(B) = {dbm_mono:e} under a driven boundary"
    );
}

#[test]
fn driven_mhd_inflow_cut_perpendicular_to_the_face() {
    assert_driven_mhd_matches([2, 1]);
}

#[test]
fn driven_mhd_inflow_cut_along_the_face() {
    assert_driven_mhd_matches([1, 2]);
}

#[test]
fn driven_mhd_inflow_four_tiles() {
    assert_driven_mhd_matches([2, 2]);
}

#[test]
fn iso_mhd_driven_inflow_decomposed_matches_monolithic() {
    // the isothermal-mhd instance of the same contract: prescription [rho, v.., B..] (no
    // pressure slot; p = cs^2 rho), the cut along the driven face (both tiles own a piece),
    // mono == decomposed bit for bit. pins the imhd decomposed registration.
    use symbi::regimes::substrate_isothermal_mhd::IsothermalMhdSubstrateKernelSet;
    use symbi_hydro::ISO_MHD_SPEC;
    use symbi_hydro::eos::Isothermal;
    use symbi_hydro::isothermal_mhd::IsothermalMhd;
    use symbi_hydro::mhd_state::MhdPrimG;
    use symbi_hydro::state::PrimG;

    type SimI =
        SimStateGeneric<IsothermalMhd, 2, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
    type KernI = IsothermalMhdSubstrateKernelSet<HostMemory, f64, 2>;
    const CS: f64 = 1.0;

    let json = r#"{
        "kind": "dirichlet", "dim": 3, "outputs": [4, 5, 6, 6, 6, 6, 7], "params": [],
        "nodes": [ {"op": "VARIABLE_X2"}, {"op": "CONSTANT", "value": 0.2},
                   {"op": "MULTIPLY", "left": 0, "right": 1}, {"op": "CONSTANT", "value": 2.0},
                   {"op": "ADD", "left": 3, "right": 2},
                   {"op": "CONSTANT", "value": 0.5}, {"op": "CONSTANT", "value": 0.0},
                   {"op": "CONSTANT", "value": 0.5} ]
    }"#;

    let make_i = |cells: [usize; 2], offset: [usize; 2], bnd: Boundaries<2>| -> (SimI, KernI) {
        let sim = SimI::build(IsothermalMhd, Isothermal { cs: CS }, Cartesian)
            .cells(cells)
            .spacing([DX; 2])
            .origin([0.0; 2])
            .index_offset(std::array::from_fn(|a| offset[a] as isize))
            .boundaries(bnd)
            .cfl(CFL)
            .timestepping(Timestepping::Rk2)
            .allocate()
            .expect("imhd sim construction failed")
            .set_initial(|_| {
                MhdPrimG::new(
                    PrimG::isothermal(Density(1.0), Tensor::new([0.0, 0.0, 0.0])),
                    Tensor::new([0.0, 0.0, BZ0]),
                )
            })
            .seed_faces_uniform([0.0, 0.0])
            .build();
        let cfg = SourceConfig::from_json(json).expect("parse");
        let built = build_boundary_dag(&cfg, &ISO_MHD_SPEC).expect("lower imhd boundary dag");
        let (k, id) = KernI::new(CS, CFL, 1.0, &sim.geom.allocated)
            .with_driven_boundary(built, cfg.params.clone());
        assert_eq!(id, 0);
        (sim, k)
    };

    let grid = |counts: [usize; 2]| -> Vec<(SimI, KernI)> {
        let m: [usize; 2] = std::array::from_fn(|a| N / counts[a]);
        (0..counts.iter().product::<usize>())
            .map(|flat| {
                let tc = unflatten(flat, counts);
                let offset: [usize; 2] = std::array::from_fn(|a| tc[a] * m[a]);
                let phys_lo = [BoundaryType::Driven(0), BoundaryType::Outflow];
                let bnd = Boundaries(std::array::from_fn(|a| {
                    let lo = if tc[a] == 0 {
                        phys_lo[a]
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
                make_i(m, offset, bnd)
            })
            .collect()
    };

    let run_i = |tiles: &mut [(SimI, KernI)], counts: [usize; 2]| {
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
            u64::MAX,
            &LocalCopy,
            |_, _, _| std::ops::ControlFlow::Continue(()),
        );
    };

    let gather = |tiles: &[(SimI, KernI)], counts: [usize; 2], comp: usize| -> Vec<f64> {
        let m: [usize; 2] = std::array::from_fn(|a| N / counts[a]);
        let mut out = vec![f64::NAN; N * N];
        for (flat_tile, (sim, _)) in tiles.iter().enumerate() {
            let tc = unflatten(flat_tile, counts);
            let ilo: [isize; 2] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
            for c in sim.geom.interior.iter() {
                let g: [usize; 2] =
                    std::array::from_fn(|a| tc[a] * m[a] + (c[a] - ilo[a]) as usize);
                out[flatten(g, [N; 2])] = match comp {
                    0 => *sim.fields.cons.den.view().at(c),
                    _ => *sim.fields.mhd.as_ref().unwrap().bcell[2].view().at(c),
                };
            }
        }
        out
    };

    let counts = [1, 2]; // the cut along the driven face — the global-coordinate-decisive tiling
    let mut mono = grid([1, 1]);
    run_i(&mut mono, [1, 1]);
    let mut dec = grid(counts);
    run_i(&mut dec, counts);
    for comp in 0..2 {
        let a = gather(&mono, [1, 1], comp);
        let b = gather(&dec, counts, comp);
        assert!(
            a.iter().all(|v| v.is_finite()),
            "monolithic imhd produced non-finite state"
        );
        let err = max_err(&a, &b);
        assert!(
            err == 0.0,
            "imhd decomposed diverged from monolithic (comp {comp}): {err:e}"
        );
    }
}
