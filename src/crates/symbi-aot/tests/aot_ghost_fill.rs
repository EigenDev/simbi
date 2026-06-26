// =============================================================================
// aot_ghost_fill.rs
//
// run the BUILD-TIME-GENERATED lattice-map pullback ghost fill (docs/design/11),
// compiled into this crate from the substrate, and check it against the lattice-
// map definition: read prim at the integer source coord (periodic / reflect /
// outflow), write at the cell, velocity picks up the per-axis Jacobian sign.
//
// proves the generated Rust COMPILES, RUNS, and IS the pullback — in particular
// that the source coord is integer (a periodic ghost reads exactly one period
// over, a reflecting ghost reads the mirror cell and flips the normal velocity).
//
// generated signature (see OUT_DIR/iso_ghost_fill_generated.rs header):
//   iso_ghost_fill_1d(
//       buf0: &mut [f64],  // prim.rho      (in place: read at source, write at cell)
//       buf1: &mut [f64],  // prim.vel[0]
//       grid_size_0: i32, dom_lo_0: i32,
//       buf_lo_0_0: i32, buf_lo_1_0: i32,
//       map_type_0: i32, arg_0: i32, vel_sign_0: f64,
//   )
// =============================================================================

use symbi_aot::NamedKernel;

const N: usize = 8; // interior cells 0..8
const NG: i32 = 2; // ghost cells per side
const LO: i32 = -NG; // first allocated cell index -> buffer index 0

// buffer index of an absolute cell index.
fn bi(cell: i32) -> usize {
    (cell - LO) as usize
}

// a fresh allocated array (interior cells filled, ghosts = NaN to prove the fill
// overwrites them). `f(cell)` defines the interior profile.
fn alloc(f: impl Fn(i32) -> f64) -> Vec<f64> {
    let ext = N + 2 * NG as usize;
    (0..ext)
        .map(|b| {
            let cell = b as i32 + LO;
            if (0..N as i32).contains(&cell) { f(cell) } else { f64::NAN }
        })
        .collect()
}

fn rho_profile(cell: i32) -> f64 {
    1.0 + 0.1 * cell as f64
}
fn vel_profile(cell: i32) -> f64 {
    0.5 - 0.05 * cell as f64
}
fn pre_profile(cell: i32) -> f64 {
    2.0 + 0.07 * cell as f64
}

// run the fill on the low-side ghost region (cells -NG .. 0) with one map.
// pressure is a grade-0 primitive, pulled back like density (no sign flip).
fn fill_low(rho: &mut [f64], vel: &mut [f64], pre: &mut [f64], map_type: i32, arg: i32, vel_sign: f64) {
    // in-place (read at source, write at cell) over the low-ghost window; the
    // allocated buffer starts at lo = LO (= -NG), so the field views need the
    // explicit layout. map_type / arg are INT scalars, vel_sign FLOAT — the
    // harness routes each by name into the right ABI tail.
    let lo_arr = [LO];
    let ext = [rho.len() as u32];
    NamedKernel::new("iso_ghost_fill_1d")
        .output_at("prim.rho", rho, &lo_arr, &ext)
        .output_at("prim.vel[0]", vel, &lo_arr, &ext)
        .output_at("prim.pre", pre, &lo_arr, &ext)
        .grid(&[NG as u32]).dom_lo(&[LO])
        .int("map_type_0", map_type).int("arg_0", arg)
        .scalar("vel_sign_0", vel_sign)
        .run();
}

#[test]
fn aot_periodic_reads_one_period_over() {
    let mut rho = alloc(rho_profile);
    let mut vel = alloc(vel_profile);
    let mut pre = alloc(pre_profile);
    // periodic: map_type 1, arg = +period (period = interior length N). low ghost
    // cell c reads c + N (the high interior); velocity unchanged (sign +1).
    fill_low(&mut rho, &mut vel, &mut pre, 1, N as i32, 1.0);
    for c in -NG..0 {
        let src = c + N as i32;
        assert_eq!(rho[bi(c)], rho_profile(src), "periodic rho ghost {c} <- {src}");
        assert_eq!(vel[bi(c)], vel_profile(src), "periodic vel ghost {c} <- {src}");
        assert_eq!(pre[bi(c)], pre_profile(src), "periodic pre ghost {c} <- {src}");
    }
}

#[test]
fn aot_reflect_mirrors_and_flips_the_normal_velocity() {
    let mut rho = alloc(rho_profile);
    let mut vel = alloc(vel_profile);
    let mut pre = alloc(pre_profile);
    // reflect about the low wall (face = interior lo = 0): pivot2 = 2*0 - 1 = -1.
    // ghost c reads pivot2 - c; the wall-normal velocity flips (vel_sign -1),
    // pressure does NOT (grade 0).
    fill_low(&mut rho, &mut vel, &mut pre, 2, -1, -1.0);
    for c in -NG..0 {
        let src = -1 - c;
        assert_eq!(rho[bi(c)], rho_profile(src), "reflect rho ghost {c} <- {src}");
        assert_eq!(vel[bi(c)], -vel_profile(src), "reflect vel ghost {c} flips <- {src}");
        assert_eq!(pre[bi(c)], pre_profile(src), "reflect pre ghost {c} (no flip) <- {src}");
    }
}

#[test]
fn aot_outflow_clamps_to_the_edge() {
    let mut rho = alloc(rho_profile);
    let mut vel = alloc(vel_profile);
    let mut pre = alloc(pre_profile);
    // outflow: map_type 3, arg = edge interior cell (lo = 0). every ghost copies
    // the edge cell; velocity unchanged.
    fill_low(&mut rho, &mut vel, &mut pre, 3, 0, 1.0);
    for c in -NG..0 {
        assert_eq!(rho[bi(c)], rho_profile(0), "outflow rho ghost {c} <- edge 0");
        assert_eq!(vel[bi(c)], vel_profile(0), "outflow vel ghost {c} <- edge 0");
        assert_eq!(pre[bi(c)], pre_profile(0), "outflow pre ghost {c} <- edge 0");
    }
}

#[test]
fn aot_in_place_leaves_the_interior_untouched() {
    // the fill writes only the ghost region; interior cells are the source, never
    // a destination, so they must be byte-unchanged after an in-place fill.
    let mut rho = alloc(rho_profile);
    let mut vel = alloc(vel_profile);
    let mut pre = alloc(pre_profile);
    fill_low(&mut rho, &mut vel, &mut pre, 1, N as i32, 1.0);
    for c in 0..N as i32 {
        assert_eq!(rho[bi(c)], rho_profile(c), "interior rho {c} changed");
        assert_eq!(vel[bi(c)], vel_profile(c), "interior vel {c} changed");
        assert_eq!(pre[bi(c)], pre_profile(c), "interior pre {c} changed");
    }
}
