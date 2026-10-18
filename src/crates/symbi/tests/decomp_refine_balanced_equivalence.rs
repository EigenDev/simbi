// =============================================================================
// decomp_refine_balanced_equivalence.rs
//
// the balanced reconstruction on refined host tiles. an isentropic atmosphere
// at rest in a point mass's potential under one refined box, on the balanced
// reconstruction, whose coarse-fine transfer chains each fine ghost from the
// patch cell nearest it and whose restriction band rewrites the covered coarse
// cells within four cells of each seam from the uncovered cell beyond it. the
// tiles equal the single hierarchy in every root and fine density over about
// seventy-five root steps on layouts keeping each seam band on one tile (one
// cut through the patch, cuts on the band edges, cuts crossing the seams
// transversely, ragged cuts off the bands) and on layouts cutting a band once
// and twice, whose far tiles resume the chains across each cut. with a cut on
// a patch face the fine ghosts past it prolong through the balanced transfer
// from a parent stencil gathered across the cut, and the tile across the cut
// refluxes and recovers the band's uncovered cell before the band chains from
// it; those layouts match too, and after a rejected fine stage. the balance
// changes the run.
//
// run: cargo test -p symbi --test decomp_refine_balanced_equivalence
// =============================================================================
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::decomp::{LocalCopy, Partition, unflatten};
use symbi::sim::refinement::launched::receiving_faces_of_tile;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion, evolve_hierarchy_decomposed, seed_decomposed_fine_from_coarse};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_discretize::Recon;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::atlas::TileId;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;
const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const N: usize = 32;
const DX: f64 = 1.0 / N as f64;
const GM: f64 = 100.0;
const C: [f64; 2] = [0.5, -1.0];
// about seventy-five root steps: the seam bands are rewritten every root step and the coarse-fine
// transfer every fine substep
const T_FINAL: f64 = 0.16;
const PATCH: [(usize, usize); 2] = [(8, 24), (8, 24)];

fn atmosphere([x, y]: [f64; 2]) -> Prim<f64, 2> {
    // isentropic, p = rho^gamma at rest: gamma/(gamma-1) rho^(gamma-1) - GM/r is uniform, rho = 1
    // at (0.5, 1)
    let r = ((x - C[0]).powi(2) + (y - C[1]).powi(2)).sqrt();
    let k = GAMMA / (GAMMA - 1.0);
    let cst = k - GM / 2.0;
    let rho = ((cst + GM / r) / k).powf(1.0 / (GAMMA - 1.0));
    Prim::adiabatic(Density(rho), Tensor::new([0.0, 0.0]), Pressure(rho.powf(GAMMA)))
}

fn kset(balanced: bool) -> impl Fn(&Sim) -> Kern + Copy {
    move |s: &Sim| Kern::new(GAMMA, CFL, &s.geom.allocated).reconstruction(Recon::Plm).well_balanced_reconstruction(balanced)
}

/// the production plm halo
const NG: usize = 2;
fn root(cells: [usize; 2], offset: [usize; 2], bnd: Boundaries<2>, ng: usize) -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells(cells).spacing([DX; 2]).origin([0.0; 2])
        .index_offset(std::array::from_fn(|a| offset[a] as isize))
        .boundaries(bnd).cfl(CFL).ghosts(ng).timestepping(Timestepping::Rk2)
        .allocate().unwrap().set_initial(atmosphere).build()
}

fn bodies() -> symbi_ib::BodyCollection<f64, 2> {
    symbi_ib::BodyCollection::new().add(symbi_ib::Body::gravitational(0, Tensor::new(C), Tensor::zeros(), GM, 1.0e-3, 0.0))
}

fn region() -> RefinementRegion<2> {
    RefinementRegion { x_lo: [0.25, 0.25], x_hi: [0.75, 0.75] }
}

fn mono(balanced: bool) -> Hier {
    let r = root([N, N], [0, 0], Boundaries::uniform(BoundaryType::Outflow), NG);
    let k = kset(balanced)(&r);
    let h = Hier::with_refinement(r, k, &[region()], ProlongOrder::Ppm, kset(balanced)).unwrap();
    h.seed_fine_from_coarse().unwrap();
    let mut h = h.with_bodies(bodies());
    h.prime();
    h
}

fn tiles(partition: &Partition<2>, ng: usize) -> Vec<Hier> {
    let counts = partition.counts();
    let mut tiles = Vec::new();
    for flat in 0..partition.n_tiles() {
        let tc = unflatten(flat, counts);
        let ext = partition.tile_extents(tc);
        let m = [ext[0].1, ext[1].1];
        let off = [ext[0].0, ext[1].0];
        let bnd = Boundaries(std::array::from_fn(|a| {
            let face = |edge: bool| if edge { BoundaryType::Outflow } else { BoundaryType::CoarseFine };
            [face(tc[a] == 0), face(tc[a] == counts[a] - 1)]
        }));
        let r = root(m, off, bnd, ng);
        let lo: [f64; 2] = std::array::from_fn(|a| off[a] as f64 * DX);
        let hi: [f64; 2] = std::array::from_fn(|a| (off[a] + m[a]) as f64 * DX);
        let w = region();
        let cx = [w.x_lo[0].max(lo[0]), w.x_hi[0].min(hi[0])];
        let cy = [w.x_lo[1].max(lo[1]), w.x_hi[1].min(hi[1])];
        let k = kset(true)(&r);
        let h = if cx[0] < cx[1] && cy[0] < cy[1] {
            let c = RefinementRegion { x_lo: [cx[0], cy[0]], x_hi: [cx[1], cy[1]] };
            Hier::with_refinement_of_donating(r, k, &[c], &[w], ProlongOrder::Ppm, kset(true)).unwrap()
        } else {
            match receiving_faces_of_tile(partition, PATCH, TileId(flat as u32)) {
                Some((p, f)) => Hier::single(r, k).with_receiving_register(&p, f).unwrap(),
                None => Hier::single(r, k),
            }
        };
        tiles.push(h);
    }
    seed_decomposed_fine_from_coarse(&tiles, counts, &vec![0; tiles.len()], &LocalCopy).unwrap();
    let mut tiles: Vec<Hier> = tiles.into_iter().map(|h| { let mut h = h.with_bodies(bodies()); h.prime(); h }).collect();
    symbi::sim::refinement::prime_decomposed_levels(&mut tiles, partition.counts(), &vec![0; partition.n_tiles()], &LocalCopy).expect("decomposed prime");
    tiles
}

/// every conserved component of every interior cell of `level`, keyed by global cell: the density,
/// the two momenta and the energy (the band rewrite rebuilds the energy from its pressure)
fn densities(h: &Hier, level: usize, origin: [isize; 2]) -> Vec<([isize; 2], [u64; 4])> {
    let s = &h.levels[level].state;
    let cons = &s.fields.cons;
    let (den, m0, m1, nrg) = (cons.den.view(), cons.mom[0].view(), cons.mom[1].view(), cons.nrg_field().expect("an energy").view());
    let lo = [s.geom.interior.spaces[0].lo, s.geom.interior.spaces[1].lo];
    s.geom
        .interior
        .iter()
        .map(|c| {
            let bits = [den.at(c).to_bits(), m0.at(c).to_bits(), m1.at(c).to_bits(), nrg.at(c).to_bits()];
            ([origin[0] + c[0] - lo[0], origin[1] + c[1] - lo[1]], bits)
        })
        .collect()
}

fn assert_tiles_match(cuts: [Vec<usize>; 2], want: &Hier) {
    assert_tiles_match_rejecting(cuts, want, None);
}

/// `assert_tiles_match` with the fine stage `(substep, stage)` of the first root step rejected once
/// on every refined tile, as on the single hierarchy `want`
fn assert_tiles_match_rejecting(cuts: [Vec<usize>; 2], want: &Hier, reject: Option<(u8, usize)>) {
    let partition = Partition::explicit([N, N], cuts.clone()).unwrap();
    let mut t = tiles(&partition, NG);
    if let Some((substep, stage)) = reject {
        for h in t.iter_mut().filter(|h| h.levels.len() > 1) {
            h.reject_fine_once_at(0, substep, stage);
        }
    }
    let devices = vec![0; t.len()];
    evolve_hierarchy_decomposed(&mut t, partition.counts(), &devices, &LocalCopy, Timestepping::Rk2, 0.0, T_FINAL, u64::MAX, |_, _, _| std::ops::ControlFlow::Continue(()));
    let root: std::collections::HashMap<_, _> = densities(want, 0, [0, 0]).into_iter().collect();
    let fine_lo = 2 * PATCH[0].0 as isize;
    let fine: std::collections::HashMap<_, _> = densities(want, 1, [fine_lo, 2 * PATCH[1].0 as isize]).into_iter().collect();
    let (mut roots, mut fines) = (0, 0);
    for (flat, h) in t.iter().enumerate() {
        assert!(h.crash.is_none(), "cuts {cuts:?}: tile {flat} crashed: {:?}", h.crash);
        assert!(!h.forced_fine_rejection_pending(), "cuts {cuts:?}: tile {flat} never reached the forced rejection");
        assert_eq!(h.levels[0].state.iteration, want.levels[0].state.iteration, "cuts {cuts:?}: tile {flat} stopped short");
        let ext = partition.tile_extents(unflatten(flat, partition.counts()));
        for (g, bits) in densities(h, 0, [ext[0].0 as isize, ext[1].0 as isize]) {
            assert_eq!(bits, root[&g], "cuts {cuts:?}: root cell {g:?}");
            roots += 1;
        }
        if h.levels.len() > 1 {
            let cov = h.levels[0].coverage.as_ref().unwrap();
            let rlo = [h.levels[0].state.geom.interior.spaces[0].lo, h.levels[0].state.geom.interior.spaces[1].lo];
            let origin = std::array::from_fn(|a| 2 * (ext[a].0 as isize + cov.spaces[a].lo - rlo[a]));
            for (g, bits) in densities(h, 1, origin) {
                assert_eq!(bits, fine[&g], "cuts {cuts:?}: fine cell {g:?}");
                fines += 1;
            }
        }
    }
    assert_eq!((roots, fines), (root.len(), fine.len()), "cuts {cuts:?}: every root and fine cell compared once");
}

#[test]
fn balanced_refined_tiles_match_the_single_hierarchy_with_every_seam_band_on_one_tile() {
    let plain = { let mut h = mono(false); h.evolve(T_FINAL).unwrap(); h };
    let want = { let mut h = mono(true); h.evolve(T_FINAL).unwrap(); h };
    assert!(want.levels[0].state.iteration > 50, "the run took {} root steps", want.levels[0].state.iteration);
    let differs = densities(&plain, 1, [0, 0]).iter().zip(densities(&want, 1, [0, 0])).any(|(a, b)| a.1 != b.1);
    assert!(differs, "the balance left the refined run unchanged; the gate is vacuous");
    // the patch spans root cells 8..24 on both axes, and each seam band four covered cells
    for cuts in [[vec![16], vec![]], [vec![12], vec![20]], [vec![16], vec![16]], [vec![14], vec![18]], [vec![19], vec![13]]] {
        assert_tiles_match(cuts, &want);
    }
}

/// a cut strictly inside a seam band: the band continues past the cut, its far part on a tile
/// without the uncovered cell. the near tile's chains reach the cell beside the cut and the far
/// tile resumes them there, the same steps in the same order, so the tiles equal the single
/// hierarchy bit for bit. the corner layouts cut both axes inside the intersecting bands, whose
/// corner cells take the x band's rewrite before the y band's on every tile
#[test]
fn balanced_refined_tiles_match_the_single_hierarchy_with_a_cut_inside_a_seam_band() {
    let want = { let mut h = mono(true); h.evolve(T_FINAL).unwrap(); h };
    // the patch spans root cells 8..24 on both axes, the bands 8..12 and 20..24
    for cuts in [
        [vec![9], vec![]],
        [vec![11], vec![]],
        [vec![21], vec![]],
        [vec![23], vec![]],
        [vec![10], vec![22]],
        [vec![10], vec![10]],
        [vec![22], vec![9]],
    ] {
        assert_tiles_match(cuts, &want);
    }
}

/// two cuts inside one seam band: the band crosses a tile two cells wide, the narrowest tile the
/// plm halo admits, and the far tile resumes the chains the middle tile carried to its cut, one hop
/// further from the seam, so the tiles equal the single hierarchy bit for bit. the layouts cut the
/// low and the high x band, and an x and a y band at once
#[test]
fn balanced_refined_tiles_match_the_single_hierarchy_with_two_cuts_inside_a_seam_band() {
    let want = { let mut h = mono(true); h.evolve(T_FINAL).unwrap(); h };
    // the patch spans root cells 8..24 on both axes, the bands 8..12 and 20..24
    for cuts in [[vec![9, 11], vec![]], [vec![21, 23], vec![]], [vec![9, 11], vec![21, 23]]] {
        assert_tiles_match(cuts, &want);
    }
}

/// a cut on a patch face under the production halo: the fine ghosts past the face prolong
/// through the balanced transfer from the parent stencil gathered across the cut, and the band's
/// uncovered cell lies on the tile across the cut, which refluxes and recovers it before the band
/// on the patch tile chains from it, as the single hierarchy's does; the tiles equal the single
/// hierarchy bit for bit. the layouts cut the low and the high face of each axis and a corner
#[test]
fn balanced_refined_tiles_match_the_single_hierarchy_with_a_cut_on_a_patch_face() {
    let want = { let mut h = mono(true); h.evolve(T_FINAL).unwrap(); h };
    // the patch spans root cells 8..24 on both axes
    for cuts in [[vec![8], vec![]], [vec![24], vec![]], [vec![], vec![8]], [vec![], vec![24]], [vec![8], vec![24]]] {
        let partition = Partition::explicit([N, N], cuts.clone()).unwrap();
        let t = tiles(&partition, NG);
        assert!(t.iter().any(|h| h.prim_stencil().is_some()), "cuts {cuts:?}: no tile reads a prim stencil; the gate is vacuous");
        assert!(t.iter().any(|h| h.band_starts_across_cut(0)), "cuts {cuts:?}: no band starts across the cut; the gate is vacuous");
        assert_tiles_match(cuts, &want);
    }
}

/// a fine stage rejected in the second substep of the first root step with the patch face on a
/// cut: the replay restores every tile's entry, regathers the prim stencil, refluxes and recovers
/// the uncovered cells again before the band, and the tiles equal the single hierarchy rejected
/// the same way, bit for bit
#[test]
fn balanced_refined_tiles_on_a_cut_patch_face_match_after_a_rejected_fine_stage() {
    let want = {
        let mut h = mono(true);
        h.reject_fine_once_at(0, 1, 1);
        h.evolve(T_FINAL).unwrap();
        assert!(!h.forced_fine_rejection_pending(), "the single hierarchy never reached the forced rejection");
        h
    };
    for cuts in [[vec![8], vec![]], [vec![8], vec![24]]] {
        assert_tiles_match_rejecting(cuts, &want, Some((1, 1)));
    }
}
