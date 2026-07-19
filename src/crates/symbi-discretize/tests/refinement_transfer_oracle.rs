// =============================================================================
// refine_transfer_oracle.rs
//
// gate: the gv-traced amr transfer kernels
// (refine_restrict_gv / refine_prolong_gv) BIT-MATCH the recovered gen-1 f64
// reference (symbi-amr prolong_nd / restrict_nd at git 3bfc5b9, vendored
// below) on pseudo-random data, across 1d/2d/3d, all prolongation orders,
// and time-interpolation fractions — and lower to CPU + CUDA source.
//
// the kernels run in ABSOLUTE level-global indices, including negative ghost
// coordinates (floor-division parent maps) — the windows below deliberately
// straddle zero to pin that.
// =============================================================================

mod harness;

use harness::KernelRun;
use symbi_discretize::{
    refine_prolong_face_gv, refine_prolong_gv, refine_restrict_face_gv, refine_restrict_gv, ProlongOrder,
};

// deterministic per-cell noise from the ABSOLUTE coordinate (splitmix-style
// integer mix), mapped into [0.5, 2.5) — strictly positive, structureless.
fn noise(seed: u64, c: &[i64]) -> f64 {
    let mut h = seed.wrapping_mul(0x9e3779b97f4a7c15);
    for &x in c {
        h ^= (x as u64).wrapping_mul(0xbf58476d1ce4e5b9);
        h = h.rotate_left(27).wrapping_mul(0x94d049bb133111eb);
    }
    0.5 + 2.0 * ((h >> 11) as f64 / (1u64 << 53) as f64)
}

// =============================================================================
// the vendored gen-1 f64 reference (git 3bfc5b9)
// =============================================================================

mod reference {
    use symbi_algebra::Domain;

    pub fn van_leer(dl: f64, dr: f64) -> f64 {
        let prod = dl * dr;
        if prod <= 0.0 { 0.0 } else { 2.0 * prod / (dl + dr) }
    }

    pub fn prolong_pcm(coarse: &[f64], fine: &mut [f64], ratio: usize) {
        let nc = coarse.len();
        assert_eq!(fine.len(), nc * ratio);
        for jj in 0..nc {
            for kk in 0..ratio {
                fine[jj * ratio + kk] = coarse[jj];
            }
        }
    }

    pub fn prolong_plm(coarse: &[f64], fine: &mut [f64], ratio: usize) {
        let nc = coarse.len();
        assert!(nc >= 3);
        let n_active = nc - 2;
        assert_eq!(fine.len(), n_active * ratio);
        let inv_ratio = 1.0 / ratio as f64;
        let half = 0.5;
        for jj in 0..n_active {
            let cc = jj + 1;
            let dl = coarse[cc] - coarse[cc - 1];
            let dr = coarse[cc + 1] - coarse[cc];
            let slope = van_leer(dl, dr);
            for kk in 0..ratio {
                let frac = (kk as f64 + half) * inv_ratio - half;
                fine[jj * ratio + kk] = coarse[cc] + slope * frac;
            }
        }
    }

    pub fn prolong_ppm(coarse: &[f64], fine: &mut [f64], ratio: usize) {
        let nc = coarse.len();
        assert!(nc >= 5);
        let n_active = nc - 4;
        assert_eq!(fine.len(), n_active * ratio);
        let inv_ratio = 1.0 / ratio as f64;
        let half = 0.5;
        let r = ratio as f64;
        for jj in 0..n_active {
            let cc = jj + 2;
            let seven = 7.0;
            let twelve_inv = 1.0 / 12.0;
            let mut u_l =
                (seven * (coarse[cc - 1] + coarse[cc]) - (coarse[cc - 2] + coarse[cc + 1])) * twelve_inv;
            let mut u_r =
                (seven * (coarse[cc] + coarse[cc + 1]) - (coarse[cc - 1] + coarse[cc + 2])) * twelve_inv;
            let lo_l = coarse[cc - 1].min(coarse[cc]);
            let hi_l = coarse[cc - 1].max(coarse[cc]);
            u_l = u_l.max(lo_l).min(hi_l);
            let lo_r = coarse[cc].min(coarse[cc + 1]);
            let hi_r = coarse[cc].max(coarse[cc + 1]);
            u_r = u_r.max(lo_r).min(hi_r);
            let (a_l, a_r) = monotonize(coarse[cc], u_l, u_r);
            let u6 = 6.0 * (coarse[cc] - half * (a_l + a_r));
            let c1 = a_l;
            let c2 = (a_r - a_l + u6) * half;
            let c3 = u6 / 3.0;
            for kk in 0..ratio {
                let xi_lo = kk as f64 * inv_ratio;
                let xi_hi = (kk + 1) as f64 * inv_ratio;
                let aa_hi = xi_hi * (c1 + xi_hi * (c2 - xi_hi * c3));
                let aa_lo = xi_lo * (c1 + xi_lo * (c2 - xi_lo * c3));
                fine[jj * ratio + kk] = (aa_hi - aa_lo) * r;
            }
        }
    }

    fn monotonize(u_c: f64, u_l: f64, u_r: f64) -> (f64, f64) {
        let mut a_l = u_l;
        let mut a_r = u_r;
        if (a_r - u_c) * (u_c - a_l) <= 0.0 {
            a_l = u_c;
            a_r = u_c;
        } else {
            let diff = a_r - a_l;
            let curv = 6.0 * (u_c - (a_l + a_r) / 2.0);
            if diff * curv > diff * diff {
                a_l = 3.0 * u_c - 2.0 * a_r;
            }
            if diff * curv < -(diff * diff) {
                a_r = 3.0 * u_c - 2.0 * a_l;
            }
        }
        (a_l, a_r)
    }

    pub fn restrict_average(fine: &[f64], coarse: &mut [f64], ratio: usize) {
        let nc = coarse.len();
        assert_eq!(fine.len(), nc * ratio);
        let inv_r = 1.0 / ratio as f64;
        for jj in 0..nc {
            let mut sum = 0.0;
            for kk in 0..ratio {
                sum = sum + fine[jj * ratio + kk];
            }
            coarse[jj] = sum * inv_r;
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum ProlongOrder { Pcm, Plm, Ppm }

    impl ProlongOrder {
        pub fn ghost_width(self) -> usize {
            match self {
                ProlongOrder::Pcm => 0,
                ProlongOrder::Plm => 1,
                ProlongOrder::Ppm => 2,
            }
        }
    }

    fn compute_strides(dims: &[usize]) -> Vec<usize> {
        let nn = dims.len();
        let mut ss = vec![0usize; nn];
        if nn > 0 {
            ss[nn - 1] = 1;
            for ii in (0..nn.saturating_sub(1)).rev() {
                ss[ii] = ss[ii + 1] * dims[ii + 1];
            }
        }
        ss
    }

    pub fn prolong_nd<const D: usize>(
        coarse: &impl Fn([isize; D]) -> f64,
        fine: &mut impl FnMut([isize; D], f64),
        coarse_active: &Domain<D>,
        ratio: usize,
        order: ProlongOrder,
    ) {
        let ng = order.ghost_width();
        let n: [usize; D] = std::array::from_fn(|ax| coarse_active.spaces[ax].size());
        let lo: [isize; D] = std::array::from_fn(|ax| coarse_active.spaces[ax].lo);

        let mut dims: Vec<usize> = (0..D).map(|ax| n[ax] + 2 * ng).collect();
        let total: usize = dims.iter().product();
        let strides = compute_strides(&dims);
        let mut buf = vec![0.0f64; total];
        for flat in 0..total {
            let mut coord = [0isize; D];
            let mut rem = flat;
            for ax in 0..D {
                let local = rem / strides[ax];
                rem %= strides[ax];
                coord[ax] = lo[ax] - ng as isize + local as isize;
            }
            buf[flat] = coarse(coord);
        }

        for ax in 0..D {
            let mut new_dims = dims.clone();
            new_dims[ax] = n[ax] * ratio;
            let new_total: usize = new_dims.iter().product();
            let mut new_buf = vec![0.0f64; new_total];
            prolong_axis(&buf, &dims, &mut new_buf, &new_dims, ax, ratio, order);
            buf = new_buf;
            dims = new_dims;
        }

        let strides = compute_strides(&dims);
        let r = ratio as isize;
        for flat in 0..buf.len() {
            let mut coord = [0isize; D];
            let mut rem = flat;
            for ax in 0..D {
                let local = rem / strides[ax];
                rem %= strides[ax];
                coord[ax] = lo[ax] * r + local as isize;
            }
            fine(coord, buf[flat]);
        }
    }

    fn prolong_axis(
        src: &[f64], src_dims: &[usize],
        dst: &mut [f64], dst_dims: &[usize],
        ax: usize, ratio: usize, order: ProlongOrder,
    ) {
        let ng = order.ghost_width();
        let ndim = src_dims.len();
        let src_str = compute_strides(src_dims);
        let dst_str = compute_strides(dst_dims);
        let nc_with_ghost = src_dims[ax];
        let n_fine = dst_dims[ax];
        assert_eq!(n_fine, (nc_with_ghost - 2 * ng) * ratio);
        let mut pencil = vec![0.0f64; nc_with_ghost];
        let mut fine_pencil = vec![0.0f64; n_fine];
        let mut trans_pos = vec![0usize; ndim];
        loop {
            let mut src_base = 0usize;
            let mut dst_base = 0usize;
            for aa in 0..ndim {
                if aa != ax {
                    src_base += trans_pos[aa] * src_str[aa];
                    dst_base += trans_pos[aa] * dst_str[aa];
                }
            }
            for ii in 0..nc_with_ghost {
                pencil[ii] = src[src_base + ii * src_str[ax]];
            }
            match order {
                ProlongOrder::Pcm => prolong_pcm(&pencil, &mut fine_pencil, ratio),
                ProlongOrder::Plm => prolong_plm(&pencil, &mut fine_pencil, ratio),
                ProlongOrder::Ppm => prolong_ppm(&pencil, &mut fine_pencil, ratio),
            }
            for ii in 0..n_fine {
                dst[dst_base + ii * dst_str[ax]] = fine_pencil[ii];
            }
            let mut carry = true;
            for aa in (0..ndim).rev() {
                if aa == ax { continue; }
                if carry {
                    trans_pos[aa] += 1;
                    if trans_pos[aa] < src_dims[aa] { carry = false; } else { trans_pos[aa] = 0; }
                }
            }
            if carry { break; }
        }
    }

    pub fn restrict_nd<const D: usize>(
        fine: &impl Fn([isize; D]) -> f64,
        coarse: &mut impl FnMut([isize; D], f64),
        fine_region: &Domain<D>,
        ratio: usize,
    ) {
        let n_fine: [usize; D] = std::array::from_fn(|ax| {
            let sz = fine_region.spaces[ax].size();
            assert!(sz % ratio == 0);
            sz
        });
        let lo: [isize; D] = std::array::from_fn(|ax| fine_region.spaces[ax].lo);

        let mut dims: Vec<usize> = n_fine.to_vec();
        let total: usize = dims.iter().product();
        let strides = compute_strides(&dims);
        let mut buf = vec![0.0f64; total];
        for flat in 0..total {
            let mut coord = [0isize; D];
            let mut rem = flat;
            for ax in 0..D {
                let local = rem / strides[ax];
                rem %= strides[ax];
                coord[ax] = lo[ax] + local as isize;
            }
            buf[flat] = fine(coord);
        }

        for ax in 0..D {
            let mut new_dims = dims.clone();
            new_dims[ax] = dims[ax] / ratio;
            let new_total: usize = new_dims.iter().product();
            let mut new_buf = vec![0.0f64; new_total];
            restrict_axis(&buf, &dims, &mut new_buf, &new_dims, ax, ratio);
            buf = new_buf;
            dims = new_dims;
        }

        let strides = compute_strides(&dims);
        let r = ratio as isize;
        for flat in 0..buf.len() {
            let mut coord = [0isize; D];
            let mut rem = flat;
            for ax in 0..D {
                let local = rem / strides[ax];
                rem %= strides[ax];
                coord[ax] = lo[ax] / r + local as isize;
            }
            coarse(coord, buf[flat]);
        }
    }

    fn restrict_axis(
        src: &[f64], src_dims: &[usize],
        dst: &mut [f64], dst_dims: &[usize],
        ax: usize, ratio: usize,
    ) {
        let ndim = src_dims.len();
        let src_str = compute_strides(src_dims);
        let dst_str = compute_strides(dst_dims);
        let n_fine = src_dims[ax];
        let n_coarse = dst_dims[ax];
        assert_eq!(n_fine, n_coarse * ratio);
        let mut pencil = vec![0.0f64; n_fine];
        let mut coarse_pencil = vec![0.0f64; n_coarse];
        let mut trans_pos = vec![0usize; ndim];
        loop {
            let mut src_base = 0usize;
            let mut dst_base = 0usize;
            for aa in 0..ndim {
                if aa != ax {
                    src_base += trans_pos[aa] * src_str[aa];
                    dst_base += trans_pos[aa] * dst_str[aa];
                }
            }
            for ii in 0..n_fine {
                pencil[ii] = src[src_base + ii * src_str[ax]];
            }
            restrict_average(&pencil, &mut coarse_pencil, ratio);
            for ii in 0..n_coarse {
                dst[dst_base + ii * dst_str[ax]] = coarse_pencil[ii];
            }
            let mut carry = true;
            for aa in (0..ndim).rev() {
                if aa == ax { continue; }
                if carry {
                    trans_pos[aa] += 1;
                    if trans_pos[aa] < src_dims[aa] { carry = false; } else { trans_pos[aa] = 0; }
                }
            }
            if carry { break; }
        }
    }
}

// =============================================================================
// restriction bit-match
// =============================================================================

// shared restrict geometry (absolute indices straddling zero):
// buffer [-4, 8) per axis; coarse window [-2, 3) -> fine children [-4, 6).
const R_BUF_LO: i32 = -4;
const R_BUF_EXT: usize = 12;
const R_WIN_LO: i32 = -2;
const R_WIN_N: usize = 5;

fn run_restrict<const D: usize>() {
    let seed = 7 + D as u64;
    let out = KernelRun::new(refine_restrict_gv(D, 2))
        .grid([R_BUF_EXT; D])
        .buffer_lo([R_BUF_LO; D])
        .compute_window([R_WIN_LO; D], [R_WIN_N; D])
        .field_with("src", move |local| {
            let abs: Vec<i64> = local.iter().map(|&l| l as i64 + R_BUF_LO as i64).collect();
            noise(seed, &abs)
        })
        .run();

    // the reference: gen-1 restrict_nd over the aligned fine region [-4, 6)^D.
    let fine_region = symbi_algebra::Domain::new(std::array::from_fn(|ax| symbi_algebra::Space {
        name: ["i", "j", "k"][ax],
        lo: 2 * R_WIN_LO as isize,
        hi: 2 * (R_WIN_LO as isize + R_WIN_N as isize),
    }));
    let fine = |c: [isize; D]| {
        let abs: Vec<i64> = c.iter().map(|&x| x as i64).collect();
        noise(seed, &abs)
    };
    let mut checked = 0usize;
    reference::restrict_nd(
        &fine,
        &mut |c: [isize; D], want: f64| {
            let local: Vec<usize> =
                c.iter().map(|&x| (x - R_BUF_LO as isize) as usize).collect();
            let got = out.get(&local, "dst");
            assert_eq!(
                got.to_bits(), want.to_bits(),
                "restrict {}d at {c:?}: got {got:e}, want {want:e}", D
            );
            checked += 1;
        },
        &fine_region,
        2,
    );
    assert_eq!(checked, R_WIN_N.pow(D as u32), "restrict {}d: oracle coverage", D);
}

#[test]
fn restrict_bit_matches_reference_1d() { run_restrict::<1>(); }
#[test]
fn restrict_bit_matches_reference_2d() { run_restrict::<2>(); }
#[test]
fn restrict_bit_matches_reference_3d() { run_restrict::<3>(); }

// =============================================================================
// prolongation bit-match (orders x dims x time-interpolation fractions)
// =============================================================================

// shared prolong geometry: buffer [-6, 8); coarse active [-2, 2) -> fine
// window [-4, 4); ppm parent halo reaches [-4, 3], inside the buffer.
const P_BUF_LO: i32 = -6;
const P_BUF_EXT: usize = 14;
const P_ACT_LO: i32 = -2;
const P_ACT_N: usize = 4;

fn gv_order(o: reference::ProlongOrder) -> ProlongOrder {
    match o {
        reference::ProlongOrder::Pcm => ProlongOrder::Pcm,
        reference::ProlongOrder::Plm => ProlongOrder::Plm,
        reference::ProlongOrder::Ppm => ProlongOrder::Ppm,
    }
}

fn run_prolong<const D: usize>(order: reference::ProlongOrder, alpha: f64) {
    let seed_old = 100 + D as u64;
    let seed_new = 200 + D as u64;
    // 3d windows shrink to keep the interpreted ppm expression cheap.
    let (act_lo, act_n) = if D == 3 { (-1i32, 2usize) } else { (P_ACT_LO, P_ACT_N) };

    let out = KernelRun::new(refine_prolong_gv(D, 2, gv_order(order)))
        .grid([P_BUF_EXT; D])
        .buffer_lo([P_BUF_LO; D])
        .compute_window([2 * act_lo; D], [2 * act_n; D])
        .field_with("src_old", move |local| {
            let abs: Vec<i64> = local.iter().map(|&l| l as i64 + P_BUF_LO as i64).collect();
            noise(seed_old, &abs)
        })
        .field_with("src_new", move |local| {
            let abs: Vec<i64> = local.iter().map(|&l| l as i64 + P_BUF_LO as i64).collect();
            noise(seed_new, &abs)
        })
        .scalars(&[("alpha", alpha)])
        .run();

    // the reference: gen-1 prolong_nd over the coarse active window, reading the
    // SAME time-interpolated coarse state.
    let coarse_active = symbi_algebra::Domain::new(std::array::from_fn(|ax| symbi_algebra::Space {
        name: ["i", "j", "k"][ax],
        lo: act_lo as isize,
        hi: act_lo as isize + act_n as isize,
    }));
    let coarse = move |c: [isize; D]| {
        let abs: Vec<i64> = c.iter().map(|&x| x as i64).collect();
        (1.0 - alpha) * noise(seed_old, &abs) + alpha * noise(seed_new, &abs)
    };
    let mut checked = 0usize;
    reference::prolong_nd(
        &coarse,
        &mut |f: [isize; D], want: f64| {
            let local: Vec<usize> =
                f.iter().map(|&x| (x - P_BUF_LO as isize) as usize).collect();
            let got = out.get(&local, "dst");
            assert_eq!(
                got.to_bits(), want.to_bits(),
                "prolong {order:?} {}d alpha={alpha} at fine {f:?}: got {got:e}, want {want:e}", D
            );
            checked += 1;
        },
        &coarse_active,
        2,
        order,
    );
    assert_eq!(checked, (2 * act_n).pow(D as u32), "prolong {}d: oracle coverage", D);
}

#[test]
fn prolong_pcm_bit_matches_reference() {
    for alpha in [0.0, 0.37, 1.0] {
        run_prolong::<1>(reference::ProlongOrder::Pcm, alpha);
        run_prolong::<2>(reference::ProlongOrder::Pcm, alpha);
        run_prolong::<3>(reference::ProlongOrder::Pcm, alpha);
    }
}

#[test]
fn prolong_plm_bit_matches_reference() {
    for alpha in [0.0, 0.37, 1.0] {
        run_prolong::<1>(reference::ProlongOrder::Plm, alpha);
        run_prolong::<2>(reference::ProlongOrder::Plm, alpha);
        run_prolong::<3>(reference::ProlongOrder::Plm, alpha);
    }
}

#[test]
fn prolong_ppm_bit_matches_reference() {
    for alpha in [0.0, 0.37, 1.0] {
        run_prolong::<1>(reference::ProlongOrder::Ppm, alpha);
        run_prolong::<2>(reference::ProlongOrder::Ppm, alpha);
        run_prolong::<3>(reference::ProlongOrder::Ppm, alpha);
    }
}

// =============================================================================
// face restriction bit-match (staggered fields): the coarse face is
// the transverse sweep-average of its ratio^(D-1) fine faces; the normal index
// scales exactly. the reference mirrors the sweep inline (axis 0 innermost among
// transverse axes, (a + b) * 0.5 per pass).
// =============================================================================

fn face_reference<const D: usize>(
    fine: &impl Fn([isize; D]) -> f64,
    coarse_face: [isize; D],
    axis: usize,
) -> f64 {
    fn eval<const D: usize>(
        fine: &impl Fn([isize; D]) -> f64,
        base: [isize; D],
        axis: usize,
        ax: isize,
        off: &mut [isize; 3],
    ) -> f64 {
        if ax < 0 {
            let mut c = base;
            for kk in 0..D {
                c[kk] += off[kk];
            }
            return fine(c);
        }
        let aa = ax as usize;
        if aa == axis {
            off[aa] = 0;
            return eval(fine, base, axis, ax - 1, off);
        }
        off[aa] = 0;
        let a = eval(fine, base, axis, ax - 1, off);
        off[aa] = 1;
        let b = eval(fine, base, axis, ax - 1, off);
        off[aa] = 0;
        (a + b) * 0.5
    }
    let base: [isize; D] = std::array::from_fn(|kk| 2 * coarse_face[kk]);
    eval(fine, base, axis, D as isize - 1, &mut [0; 3])
}

fn run_restrict_face<const D: usize>(axis: usize) {
    let seed = 300 + (10 * D + axis) as u64;
    let out = KernelRun::new(refine_restrict_face_gv(D, 2, axis))
        .grid([R_BUF_EXT; D])
        .buffer_lo([R_BUF_LO; D])
        .compute_window([R_WIN_LO; D], [R_WIN_N; D])
        .field_with("src", move |local| {
            let abs: Vec<i64> = local.iter().map(|&l| l as i64 + R_BUF_LO as i64).collect();
            noise(seed, &abs)
        })
        .run();

    let fine = |c: [isize; D]| {
        let abs: Vec<i64> = c.iter().map(|&x| x as i64).collect();
        noise(seed, &abs)
    };
    let win = symbi_algebra::Domain::new(std::array::from_fn(|ax| symbi_algebra::Space {
        name: ["i", "j", "k"][ax],
        lo: R_WIN_LO as isize,
        hi: R_WIN_LO as isize + R_WIN_N as isize,
    }));
    for c in win.iter() {
        let want = face_reference(&fine, c, axis);
        let local: Vec<usize> = c.iter().map(|&x| (x - R_BUF_LO as isize) as usize).collect();
        let got = out.get(&local, "dst");
        assert_eq!(
            got.to_bits(), want.to_bits(),
            "restrict_face axis {axis} {}d at {c:?}: got {got:e}, want {want:e}", D
        );
    }
}

#[test]
fn restrict_face_bit_matches_reference() {
    run_restrict_face::<1>(0);
    run_restrict_face::<2>(0);
    run_restrict_face::<2>(1);
    run_restrict_face::<3>(0);
    run_restrict_face::<3>(1);
    run_restrict_face::<3>(2);
}

// =============================================================================
// face prolongation bit-match (staggered fields): the
// normal axis pair-averages the time-interpolated coarse face lattice (exact
// on coincident even faces), transverse axes apply the van-leer plm sweep
// (axis 0 innermost among them, operating on the pair-averaged leaf values).
// =============================================================================

fn face_prolong_reference<const D: usize>(
    coarse: &impl Fn([isize; D]) -> f64,
    fine_face: [isize; D],
    axis: usize,
) -> f64 {
    fn eval<const D: usize>(
        coarse: &impl Fn([isize; D]) -> f64,
        fine: &[isize; D],
        axis: usize,
        ax: isize,
        off: &mut [isize; 3],
    ) -> f64 {
        if ax < 0 {
            let mut lo = [0isize; D];
            let mut hi = [0isize; D];
            for kk in 0..D {
                if kk == axis {
                    lo[kk] = fine[kk].div_euclid(2);
                    hi[kk] = (fine[kk] + 1).div_euclid(2);
                } else {
                    lo[kk] = fine[kk].div_euclid(2) + off[kk];
                    hi[kk] = lo[kk];
                }
            }
            return 0.5 * (coarse(lo) + coarse(hi));
        }
        let aa = ax as usize;
        if aa == axis {
            return eval(coarse, fine, axis, ax - 1, off);
        }
        let mut v = [0.0f64; 3];
        for (ii, dd) in (-1..=1isize).enumerate() {
            off[aa] = dd;
            v[ii] = eval(coarse, fine, axis, ax - 1, off);
        }
        off[aa] = 0;
        let parity = fine[aa] - 2 * fine[aa].div_euclid(2);
        let frac = (parity as f64 + 0.5) * 0.5 - 0.5;
        v[1] + reference::van_leer(v[1] - v[0], v[2] - v[1]) * frac
    }
    eval(coarse, &fine_face, axis, D as isize - 1, &mut [0; 3])
}

fn run_prolong_face<const D: usize>(axis: usize, alpha: f64) {
    let seed_old = 400 + (10 * D + axis) as u64;
    let seed_new = 500 + (10 * D + axis) as u64;
    let out = KernelRun::new(refine_prolong_face_gv(D, 2, axis))
        .grid([P_BUF_EXT; D])
        .buffer_lo([P_BUF_LO; D])
        .compute_window([-4; D], [8; D])
        .field_with("src_old", move |local| {
            let abs: Vec<i64> = local.iter().map(|&l| l as i64 + P_BUF_LO as i64).collect();
            noise(seed_old, &abs)
        })
        .field_with("src_new", move |local| {
            let abs: Vec<i64> = local.iter().map(|&l| l as i64 + P_BUF_LO as i64).collect();
            noise(seed_new, &abs)
        })
        .scalars(&[("alpha", alpha)])
        .run();

    let coarse = move |c: [isize; D]| {
        let abs: Vec<i64> = c.iter().map(|&x| x as i64).collect();
        (1.0 - alpha) * noise(seed_old, &abs) + alpha * noise(seed_new, &abs)
    };
    let win = symbi_algebra::Domain::new(std::array::from_fn(|ax| symbi_algebra::Space {
        name: ["i", "j", "k"][ax],
        lo: -4,
        hi: 4,
    }));
    for f in win.iter() {
        let want = face_prolong_reference(&coarse, f, axis);
        let local: Vec<usize> = f.iter().map(|&x| (x - P_BUF_LO as isize) as usize).collect();
        let got = out.get(&local, "dst");
        assert_eq!(
            got.to_bits(), want.to_bits(),
            "prolong_face axis {axis} {}d alpha={alpha} at {f:?}: got {got:e}, want {want:e}", D
        );
    }
}

#[test]
fn prolong_face_bit_matches_reference() {
    for alpha in [0.0, 0.37, 1.0] {
        run_prolong_face::<1>(0, alpha);
        run_prolong_face::<2>(0, alpha);
        run_prolong_face::<2>(1, alpha);
        run_prolong_face::<3>(0, alpha);
        run_prolong_face::<3>(1, alpha);
        run_prolong_face::<3>(2, alpha);
    }
}

// =============================================================================
// the lowerability gate: every transfer kernel renders to CPU + CUDA source
// =============================================================================

#[test]
fn transfer_kernels_lower_to_cpu_and_cuda() {
    use symbi_discretize::{
        refine_acc_edge_gv, refine_acc_face_gv, field_axpy_shift_gv, field_copy_gv, field_fill_gv,
    };
    for d in 1..=3usize {
        let grid = vec![8usize; d];
        KernelRun::new(refine_restrict_gv(d, 2)).grid(&grid).assert_lowers();
        KernelRun::new(field_copy_gv(d)).grid(&grid).assert_lowers();
        KernelRun::new(field_fill_gv(d)).grid(&grid).assert_lowers();
        KernelRun::new(field_axpy_shift_gv(d)).grid(&grid).assert_lowers();
        for ax in 0..d {
            KernelRun::new(refine_restrict_face_gv(d, 2, ax)).grid(&grid).assert_lowers();
            KernelRun::new(refine_prolong_face_gv(d, 2, ax)).grid(&grid).assert_lowers();
            KernelRun::new(refine_acc_face_gv(d, 2, ax)).grid(&grid).assert_lowers();
            KernelRun::new(refine_acc_edge_gv(d, 2, ax)).grid(&grid).assert_lowers();
        }
        for order in [ProlongOrder::Pcm, ProlongOrder::Plm, ProlongOrder::Ppm] {
            KernelRun::new(refine_prolong_gv(d, 2, order)).grid(&grid).assert_lowers();
        }
    }
}
