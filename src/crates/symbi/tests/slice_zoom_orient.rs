// =============================================================================
// slice_zoom_orient.rs
//
// the display-slice zoom + orientation contracts: the field encodes its own
// cell INDICES exactly (rho = 10000 ix + 100 iy + iz + 1, f32-representable),
// so where a sample came from is decodable off its value. gates:
// - zoom 1 (2x) samples ONLY the centered half-extent window on both display
//   axes, at full output resolution (crisper), while the
//   unzoomed view reaches both domain edges;
// - orientation 1 (the y mid-plane) pins every sample at iy = N/2 exactly
//   while spanning the full x and z ranges.
// =============================================================================

use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 32;
const L: f64 = 1.0;
const DX: f64 = 2.0 * L / N as f64;

type Sim = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

fn build() -> Sim {
    let sim = Sim::build(Newtonian, IdealGas { gamma: 1.4 }, Cartesian)
        .cells([N; 3])
        .origin([-L; 3])
        .spacing([DX; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim { rho: 1.0, vel: Tensor::new([0.0; 3]), pre: 1.0 })
        .build();
    // exact integer index codes (f32-representable): rho = 10000 ix + 100 iy + iz + 1,
    // so every sample's source cell is decodable exactly.
    let lo: [isize; 3] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo);
    for c in sim.geom.interior.iter() {
        let (ix, iy, iz) = ((c[0] - lo[0]) as f64, (c[1] - lo[1]) as f64, (c[2] - lo[2]) as f64);
        sim.fields.prim.rho.set(c, 10000.0 * ix + 100.0 * iy + iz + 1.0);
    }
    sim
}

fn decode(v: f32) -> (usize, usize, usize) {
    let v = v as f64 - 1.0;
    let ix = (v / 10000.0).floor();
    let iy = ((v - 10000.0 * ix) / 100.0).floor();
    let iz = v - 10000.0 * ix - 100.0 * iy;
    (ix as usize, iy as usize, iz.round() as usize)
}

#[test]
fn zoom_samples_the_centered_half_window_only() {
    let sim = build();
    let full = sim.field_slice_oriented(64, 0, 0, 0).expect("full slice");
    let zoomed = sim.field_slice_oriented(64, 0, 0, 1).expect("zoomed slice");
    // the full view spans the whole ix range (storage x = the display row axis at
    // orient 0; the z mid-plane fixes iz = N/2).
    let full_ix: Vec<usize> = full.data.iter().map(|v| decode(*v).0).collect();
    assert!(full_ix.iter().any(|&i| i == 0) && full_ix.iter().any(|&i| i == N - 1),
        "full-extent view missing the domain edges");
    // the 2x view samples ONLY the centered half window on BOTH display axes, at
    // full output resolution (crisper).
    let (wlo, whi) = (N / 4, N / 4 + N / 2);
    for v in &zoomed.data {
        let (ix, iy, _) = decode(*v);
        assert!(ix >= wlo && ix < whi, "zoomed sample outside the x window: ix = {ix}");
        assert!(iy >= wlo && iy < whi, "zoomed sample outside the y window: iy = {iy}");
    }
    assert_eq!(zoomed.width, N / 2, "the zoomed window is decimated at sx = 1: width = span");
}

#[test]
fn orientation_permutes_the_display_axes() {
    let sim = build();
    // the y mid-plane (orient 1): every sample carries iy = N/2 exactly, while
    // ix and iz both span their full ranges.
    let sl = sim.field_slice_oriented(64, 0, 1, 0).expect("y-slice");
    let mut ix_seen = [false; N];
    let mut iz_seen = [false; N];
    for v in &sl.data {
        let (ix, iy, iz) = decode(*v);
        assert_eq!(iy, N / 2, "y mid-plane sample not at the mid index: iy = {iy}");
        ix_seen[ix] = true;
        iz_seen[iz] = true;
    }
    assert!(ix_seen.iter().all(|&b| b), "the y-slice does not span x");
    assert!(iz_seen.iter().all(|&b| b), "the y-slice does not span z");
}
