// =============================================================================
// closure_whole_patch.rs
//
// the whole-patch closure builder reproduces the single hierarchy. on a
// periodic isothermal MHD root whose locally isothermal cs^2 varies, refined by
// two nested patches:
// - the builder's levels have the single hierarchy's interior, allocated cells
//   and boundaries, level by level;
// - the cs^2 it builds for every level equals the single hierarchy's on every
//   allocated cell, bit for bit, ghosts included, and varies across the patch;
// - its peak temporary storage is the accounting formula's.
//
// run: cargo test -p symbi --test closure_whole_patch
// =============================================================================

use symbi::regimes::substrate_isothermal_mhd::IsothermalMhdSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_grid::Field;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::Isothermal;
use symbi_hydro::isothermal_mhd::IsothermalMhd;
use symbi_hydro::mhd_state::MhdPrimG;
use symbi_hydro::quantity::Density;
use symbi_hydro::state::PrimG;
use symbi_refinement::refinement::closure_patch::{ClosureRule, build_whole_patch_closure, closure_peak_bytes, same_box, whole_levels};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<IsothermalMhd, 2, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory, f64>;
type Kset = IsothermalMhdSubstrateKernelSet<HostMemory, f64, 2>;

const N: usize = 32;

fn regions() -> [RefinementRegion<2>; 2] {
    [
        RefinementRegion { x_lo: [-0.25, -0.1875], x_hi: [0.25, 0.3125] },
        RefinementRegion { x_lo: [-0.125, -0.0625], x_hi: [0.125, 0.1875] },
    ]
}

fn hierarchy() -> Hierarchy<IsothermalMhd, 2, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory, Kset> {
    let dx = 1.0 / N as f64;
    let root = Sim::build(IsothermalMhd, Isothermal { cs: 1.0 }, Cartesian)
        .cells([N; 2])
        .origin([-0.5; 2])
        .spacing([dx; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("root")
        .set_initial(|_| MhdPrimG::<f64, 3, IsoModel>::new(PrimG::isothermal(Density(1.0), Tensor::new([0.0; 3])), Tensor::new([0.1, 0.0, 0.0])))
        .seed_faces(|axis, _| if axis == 0 { 0.1 } else { 0.0 })
        .build();
    let pre: Field<f64, 2, HostMemory> = Field::zeros(&root.geom.allocated).unwrap();
    let k = 2.0 * std::f64::consts::PI;
    for c in root.geom.interior.iter() {
        let x = root.geom.x_lo[0] + (c[0] as f64 + 0.5) * dx;
        let y = root.geom.x_lo[1] + (c[1] as f64 + 0.5) * dx;
        pre.set(c, 1.0 + 0.5 * (k * x).sin() * (k * y).cos());
    }
    root.set_isothermal_cs2_from_pressure(&pre);
    let kset = |s: &Sim| Kset::new(1.0, 0.3, 1.0, &s.geom.allocated);
    let ck = kset(&root);
    let hier = Hierarchy::with_refinement(root, ck, &regions(), ProlongOrder::Ppm, kset).expect("hierarchy");
    hier.seed_fine_from_coarse().expect("seed");
    hier
}

#[test]
fn the_whole_patch_closure_is_the_single_hierarchys_on_every_level() {
    let hier = hierarchy();
    let root = &hier.levels[0].state;
    let levels = whole_levels([N; 2], root.geom.ng, root.boundaries, root.geom.x_lo, root.geom.dx, &regions());
    assert_eq!(levels.len(), hier.levels.len());
    for (l, (whole, level)) in levels.iter().zip(&hier.levels).enumerate() {
        let g = &level.state.geom;
        assert!(same_box(&whole.interior, &g.interior), "level {l}: interior {:?} against {:?}", whole.interior.spaces, g.interior.spaces);
        assert!(same_box(&whole.allocated, &g.allocated), "level {l}: allocated {:?} against {:?}", whole.allocated.spaces, g.allocated.spaces);
        assert_eq!(whole.boundaries.0, level.state.boundaries.0, "level {l}: boundaries");
    }
    let root_cs2: Field<f64, 2, HostMemory> = Field::zeros(&levels[0].allocated).unwrap();
    for c in levels[0].allocated.iter() {
        root_cs2.set(c, *root.fields.cs2.as_ref().unwrap().at(c));
    }
    let mut checked = 0;
    let peak = build_whole_patch_closure(&levels, root_cs2, ProlongOrder::Ppm, ClosureRule::Extend, 2, |l, cs2| {
        let want = hier.levels[l].state.fields.cs2.as_ref().unwrap();
        let mut varies = false;
        for c in levels[l].allocated.iter() {
            assert_eq!(cs2.at(c).to_bits(), want.at(c).to_bits(), "level {l}: cs^2 at {c:?}");
            varies |= (*cs2.at(c) - *cs2.at([levels[l].interior.spaces[0].lo, levels[l].interior.spaces[1].lo])).abs() > 1e-3;
        }
        assert!(varies, "level {l}: the closure is uniform; the comparison is vacuous");
        checked += 1;
        Ok(())
    })
    .expect("the whole-patch closure");
    assert_eq!(checked, 2, "the builder visited {checked} levels");
    assert_eq!(peak, closure_peak_bytes(&levels, 2), "the measured peak departs from the accounting");
}
