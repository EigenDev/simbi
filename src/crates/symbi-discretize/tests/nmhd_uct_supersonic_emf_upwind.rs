// =============================================================================
// nmhd_uct_supersonic_emf_upwind.rs
//
// the deterministic rust regression gate for the supersonic-EMF upwind-pairing
// invariant of the UCT master edge-EMF (`uct_master_emf`, ct_emf.rs). a prior bug
// paired the advective weight a^L with the downwind face (anti-upwind):
//   adv_x = -vbar_x (a^L by_E + a^R by_W)   [the bug: a^L -> East/downwind]
// the correct upwind pairing is:
//   adv_x = -vbar_x (a^L by_W + a^R by_E)   [a^L -> West/upwind]
// for symmetric edge speeds (a^L == a^R, e.g. subsonic orszag-tang) the two forms
// agree, so only asymmetry exposes the bug. supersonically (a^L >> a^R), the wrong
// form advects the downwind state -> a growing odd-even instability (the field-loop
// blow-up: gas pressure 1 -> 29, dt collapse). div(B) tests stay green through it
// (the CT curl preserves div(B) for any edge EMF, advective coefficients included).
//
// the gate stands up the minimal physics that exposes it: a weak magnetic loop
// (gardiner & stone 2005 / simbi_configs/examples/field_loop.py) advected
// supersonically (|v| = sqrt(5), cs ~ 1.29 => mach ~ 1.7) on a small periodic
// grid, evolved purely by the CT induction loop (uniform rho/p/v -> the gas is
// frozen, only B advects). it chains the production kernels through the CPU
// interpreter harness:
//   nmhd_edge_emf_uct_hllc_gv (the master EMF) -> rmhd_ct_curl_2d_dir_gv (induction)
// over N steps, asserting B / magnetic energy stays bounded and finite. with the
// correct upwind pairing the loop translates and stays bounded; with the
// downwind-paired bug the magnetic energy diverges. flipping by_w<->by_e at
// ct_emf.rs:577 must make this test fail.
// =============================================================================

mod harness;
use harness::KernelRun;
use symbi_algebra::FaceNormal;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_ir::SweepAxis;

use symbi_algebra::Tensor;
use symbi_discretize::{nmhd_edge_emf_uct_hlld_gv, rmhd_ct_curl_2d_dir_gv};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::regime::Regime;
use symbi_hydro::state::Prim;

// small periodic grid (interior); fast — just enough to seed + grow the supersonic
// odd-even instability the downwind pairing produces. mirrors field_loop.py geometry
// scaled to a square cell so the discrete loop is isotropic.
const NX: usize = 12;
const NY: usize = 12;
// buffers carry a ghost halo wide enough for the EMF/curl stencils: the EMF reads
// transverse neighbors at -2..+1 (recon_face_to_edge needs the 2nd transverse
// neighbor), the curl reads +1. NG=3 covers both sides with margin.
const NG: usize = 3;
const BX: usize = NX + 2 * NG; // full buffer extent, x
const BY: usize = NY + 2 * NG; // full buffer extent, y
const N: usize = BX * BY;

// physics (field_loop.py defaults, NMHD regime).
const GAMMA: f64 = 5.0 / 3.0;
const RHO0: f64 = 1.0;
const PRE0: f64 = 1.0;
const A0: f64 = 1.0e-3; // vector-potential amplitude (weak / passive)
const RAD: f64 = 0.3; // loop radius
const SPEED: f64 = 2.2360679774997896; // sqrt(5): the paper's supersonic v=(2,1), cs ~ 1.29
// domain: a square box centered on the origin so the diagonal loop is well inside.
const L: f64 = 1.0; // half-width in x and y
const THETA: f64 = 1.5; // plm-theta (the kernel's reconstruction slope limiter)
// a small div-free grid-scale (odd-even) seed superposed on the loop: a checkerboard in
// the corner vector potential A_z. its discrete curl is exactly div-free (CT preserves
// it), so it injects a pure grid-scale magnetic mode at machine-zero div(B). supersonic
// upwind advection (the correct pairing) damps this mode; the downwind-paired bug is the
// anti-upwind scheme, which amplifies it -> the discriminating instability. the smooth
// loop carries resolved scales alone, where the two pairings agree (by_w ~ by_e); the
// seed is the minimal feature that makes the upwind/downwind choice bite.
const SEED: f64 = 5.0e-5; // << A0*rad (the loop A_z); the field stays weak/passive

// axis-0-fastest flat index over the full (ghosted) buffer, matching the harness /
// interpreter / canonical Field convention. (i, j) are buffer-local (ghosts included).
fn idx(i: usize, j: usize) -> usize {
    i + j * BX
}

// cell-center coordinate of buffer cell (i, j): interior cell (NG, NG) sits at the
// lower-left of the domain. dx = dy (square cells).
fn dx() -> f64 {
    2.0 * L / NX as f64
}
fn dy() -> f64 {
    2.0 * L / NY as f64
}
// corner coordinate (lower-left corner of cell (i,j)).
fn xcorner(i: usize) -> f64 {
    -L + (i as f64 - NG as f64) * dx()
}
fn ycorner(j: usize) -> f64 {
    -L + (j as f64 - NG as f64) * dy()
}

// the loop vector potential A_z(x,y) = A0 (R - r) inside r < R, else 0 (field_loop.py).
fn az_loop(x: f64, y: f64) -> f64 {
    let r = (x * x + y * y).sqrt();
    if r < RAD { A0 * (RAD - r) } else { 0.0 }
}

// the corner vector potential at corner (i, j) = loop + the div-free odd-even seed.
// the seed is a checkerboard (-1)^(i+j) on the corner lattice -> its discrete curl is a
// pure grid-scale div-free B mode. defined on the grid-index corner so the periodicity
// and the checkerboard phase are exact (period nx/ny must be even for the wrap to align;
// nx=ny=24 is even).
fn az_corner(i: usize, j: usize) -> f64 {
    let parity = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
    az_loop(xcorner(i), ycorner(j)) + SEED * parity
}

// fast-magnetosonic wave speeds (sl, sr) = (vn - cf, vn + cf) for the NMHD prim along
// axis `ax`, via the production host physics at S = f64 (one physics source).
fn wave_speeds_axis(prim: &MhdPrim<f64, 3>, ax: usize) -> (f64, f64) {
    let eos = IdealGas { gamma: GAMMA };
    let nhat = symbi_algebra::Normalized::axis(ax);
    NewtonianMhd.wave_speeds(&eos, prim, &nhat)
}

// the running simulation state: the staggered faces + the (uniform, frozen) gas.
struct Sim {
    bx: Vec<f64>, // B_x on x-faces (cell-indexed: bx[i,j] = B_x at the (i-1/2,j) face)
    by: Vec<f64>, // B_y on y-faces
    vx: f64,
    vy: f64,
}

impl Sim {
    // initialize the supersonically-advected field loop. velocity is diagonal v=(2,1)
    // scaled to |v| = speed; B is the discrete curl of A_z (div-free to machine zero).
    fn new() -> Sim {
        let norm = 5.0_f64.sqrt();
        let vx = SPEED * 2.0 / norm;
        let vy = SPEED * 1.0 / norm;
        let (dx, dy) = (dx(), dy());
        let mut bx = vec![0.0_f64; N];
        let mut by = vec![0.0_f64; N];
        for i in 0..BX {
            for j in 0..BY {
                // staggered B = discrete curl of the corner A_z (loop + div-free seed).
                // any corner-defined A_z gives an exactly div-free staggered B.
                // B_x on the x-face (spans the two y-corners): B_x = dA_z/dy.
                if j + 1 < BY {
                    bx[idx(i, j)] = (az_corner(i, j + 1) - az_corner(i, j)) / dy;
                }
                // B_y on the y-face (spans the two x-corners): B_y = -dA_z/dx.
                if i + 1 < BX {
                    by[idx(i, j)] = -(az_corner(i + 1, j) - az_corner(i, j)) / dx;
                }
            }
        }
        Sim { bx, by, vx, vy }
    }

    // cell-centered B (the average of the two bounding faces) for cell (i,j).
    fn bcell(&self, i: usize, j: usize) -> (f64, f64) {
        let bxc = 0.5 * (self.bx[idx(i, j)] + self.bx[idx(i + 1, j)]);
        let byc = 0.5 * (self.by[idx(i, j)] + self.by[idx(i, j + 1)]);
        (bxc, byc)
    }

    // periodic wrap of one staggered face buffer: copy the interior into the ghost
    // halo so the stencils read the wrapped neighbor. cell-indexed staggering with
    // nx/ny interior cells -> period nx (x), ny (y).
    fn wrap(buf: &mut [f64]) {
        // x ghosts
        for j in 0..BY {
            for g in 0..NG {
                buf[idx(g, j)] = buf[idx(g + NX, j)];
                buf[idx(NG + NX + g, j)] = buf[idx(NG + g, j)];
            }
        }
        // y ghosts (after x so corners are filled from the wrapped columns)
        for i in 0..BX {
            for g in 0..NG {
                buf[idx(i, g)] = buf[idx(i, g + NY)];
                buf[idx(i, NG + NY + g)] = buf[idx(i, NG + g)];
            }
        }
    }

    fn wrap_all(&mut self) {
        let mut bx = std::mem::take(&mut self.bx);
        let mut by = std::mem::take(&mut self.by);
        Self::wrap(&mut bx);
        Self::wrap(&mut by);
        self.bx = bx;
        self.by = by;
    }

    // max fast-magnetosonic signal speed over the interior (for the CFL dt). uniform
    // gas + weak B -> this is ~constant, but it is recomputed honestly each step.
    fn smax(&self) -> f64 {
        let mut s = 0.0_f64;
        for i in NG..NG + NX {
            for j in NG..NG + NY {
                let (bxc, byc) = self.bcell(i, j);
                let p = MhdPrim::new(
                    Prim::adiabatic(
                        Density(RHO0),
                        Tensor::new([self.vx, self.vy, 0.0]),
                        Pressure(PRE0),
                    ),
                    Tensor::new([bxc, byc, 0.0]),
                );
                for ax in 0..2 {
                    let (sl, sr) = wave_speeds_axis(&p, ax);
                    s = s.max(sl.abs().max(sr.abs()));
                }
            }
        }
        s
    }

    // total staggered magnetic energy 0.5 sum(B_x^2 + B_y^2) over the interior faces — the
    // boundedness diagnostic. it reads the staggered faces directly: the anti-upwind
    // instability is a grid-scale odd-even mode on the staggered field, which the
    // cell-average `bcell` smooths away, leaving the cell-centered energy blind to it.
    fn mag_energy(&self) -> f64 {
        let mut e = 0.0_f64;
        for i in NG..NG + NX {
            for j in NG..NG + NY {
                let bxf = self.bx[idx(i, j)];
                let byf = self.by[idx(i, j)];
                e += 0.5 * (bxf * bxf + byf * byf);
            }
        }
        e
    }

    fn all_finite(&self) -> bool {
        self.bx.iter().chain(self.by.iter()).all(|v| v.is_finite())
    }
}

// build the per-cell scalar field buffers the EMF kernel binds: rho, vel_p1/p2, pre,
// bcell_p1/p2/out, and the per-cell wave speeds wsr/wsl for each in-plane axis.
struct EmfInputs {
    rho: Vec<f64>,
    vp1: Vec<f64>,
    vp2: Vec<f64>,
    pre: Vec<f64>,
    bp1: Vec<f64>,
    bp2: Vec<f64>,
    bout: Vec<f64>,
    wsr1: Vec<f64>,
    wsl1: Vec<f64>,
    wsr2: Vec<f64>,
    wsl2: Vec<f64>,
}

fn build_emf_inputs(sim: &Sim) -> EmfInputs {
    let mut e = EmfInputs {
        rho: vec![RHO0; N],
        vp1: vec![sim.vx; N],
        vp2: vec![sim.vy; N],
        pre: vec![PRE0; N],
        bp1: vec![0.0; N],
        bp2: vec![0.0; N],
        bout: vec![0.0; N],
        wsr1: vec![0.0; N],
        wsl1: vec![0.0; N],
        wsr2: vec![0.0; N],
        wsl2: vec![0.0; N],
    };
    // fill over the whole buffer (the EMF stencil reaches into the ghosts). the wrapped
    // faces already populate the ghost cell-B; the outermost ring (where bcell reads
    // face[i+1]) sits outside every edge target, so leaving it at 0 is harmless.
    for i in 0..BX {
        for j in 0..BY {
            let in_bounds = i + 1 < BX && j + 1 < BY;
            let (bxc, byc) = if in_bounds {
                sim.bcell(i, j)
            } else {
                (0.0, 0.0)
            };
            let f = idx(i, j);
            e.bp1[f] = bxc;
            e.bp2[f] = byc;
            e.bout[f] = 0.0;
            let p = MhdPrim::new(
                Prim::adiabatic(
                    Density(RHO0),
                    Tensor::new([sim.vx, sim.vy, 0.0]),
                    Pressure(PRE0),
                ),
                Tensor::new([bxc, byc, 0.0]),
            );
            let (sl1, sr1) = wave_speeds_axis(&p, 0);
            let (sl2, sr2) = wave_speeds_axis(&p, 1);
            e.wsl1[f] = sl1;
            e.wsr1[f] = sr1;
            e.wsl2[f] = sl2;
            e.wsr2[f] = sr2;
        }
    }
    e
}

// run the production NMHD uct-hllc edge EMF kernel over the interior corners. returns
// the full Ez buffer (corner field, cell-indexed: ez[i,j] = E_z at corner (i-1/2,j-1/2)).
fn run_edge_emf(sim: &Sim) -> Vec<f64> {
    let inp = build_emf_inputs(sim);
    let bx = sim.bx.clone();
    let by = sim.by.clone();
    let (rho, vp1, vp2, pre) = (inp.rho, inp.vp1, inp.vp2, inp.pre);
    let (bp1, bp2, bout) = (inp.bp1, inp.bp2, inp.bout);
    let (wsr1, wsl1, wsr2, wsl2) = (inp.wsr1, inp.wsl1, inp.wsr2, inp.wsl2);
    let pull = |buf: Vec<f64>| move |c: &[usize]| buf[idx(c[0], c[1])];
    // g1 = x (axis 0), g2 = y (axis 1): the out-of-plane z-edge EMF. the gate needs the HLLD
    // edge EMF: HLLC's three-wave fan fixes the advective weights a^L=a^R=1/2, so the
    // by_w<->by_e swap leaves the HLLC EMF identical and hides the bug. HLLD's five-wave fan
    // gives asymmetric a^L != a^R supersonically, where the upwind/downwind pairing decides
    // which state is advected. (this is field_loop.py's default solver, Solver.HLLD.)
    let out = KernelRun::new(nmhd_edge_emf_uct_hlld_gv(2, 0, 1))
        .grid([BX, BY])
        // the EMF reads -2..+1 in the transverse directions; compute strictly inside.
        .compute_window([NG as i32 - 1, NG as i32 - 1], [NX + 2, NY + 2])
        .field_with("h_rho", pull(rho))
        .field_with("h_vp1", pull(vp1))
        .field_with("h_vp2", pull(vp2))
        .field_with("h_pre", pull(pre))
        .field_with("h_bp1", pull(bp1))
        .field_with("h_bp2", pull(bp2))
        .field_with("h_bout", pull(bout))
        .field_with("h_bface_a", pull(bx))
        .field_with("h_bface_b", pull(by))
        .field_with("h_wsr1", pull(wsr1))
        .field_with("h_wsl1", pull(wsl1))
        .field_with("h_wsr2", pull(wsr2))
        .field_with("h_wsl2", pull(wsl2))
        .scalars(&[("theta", THETA), ("gamma", GAMMA)])
        .run();
    out.values("emf").to_vec()
}

// advance the staggered faces one step by the CT curl of the edge EMF (induction).
// dir=0 updates B_x from d_y(Ez); dir=1 updates B_y from d_x(Ez). out-of-place.
// note: the EMF kernel writes the corner field as `emf`; the curl kernel reads it as `ez`.
fn ct_induction(sim: &Sim, ez: &[f64], dt: f64) -> (Vec<f64>, Vec<f64>) {
    let (idx_inv, idy_inv) = (1.0 / dx(), 1.0 / dy());
    let ez_a = ez.to_vec();
    let ez_b = ez.to_vec();
    let bx_in = sim.bx.clone();
    let by_in = sim.by.clone();
    let pull = |buf: Vec<f64>| move |c: &[usize]| buf[idx(c[0], c[1])];
    // dir=0: B_x_new = B_x - dt*idy*(Ez[i,j+1]-Ez[i,j]).
    let bx_out = KernelRun::new(rmhd_ct_curl_2d_dir_gv(SweepAxis::new(0, 2)))
        .grid([BX, BY])
        .compute_window([NG as i32, NG as i32], [NX, NY])
        .field_with("b", pull(bx_in))
        .field_with("ez", pull(ez_a))
        .scalars(&[("dt", dt), ("idy", idy_inv)])
        .run()
        .values("b_new")
        .to_vec();
    // dir=1: B_y_new = B_y + dt*idx*(Ez[i+1,j]-Ez[i,j]).
    let by_out = KernelRun::new(rmhd_ct_curl_2d_dir_gv(SweepAxis::new(1, 2)))
        .grid([BX, BY])
        .compute_window([NG as i32, NG as i32], [NX, NY])
        .field_with("b", pull(by_in))
        .field_with("ez", pull(ez_b))
        .scalars(&[("dt", dt), ("idx", idx_inv)])
        .run()
        .values("b_new")
        .to_vec();
    (bx_out, by_out)
}

#[test]
fn nmhd_uct_supersonic_emf_stays_bounded() {
    let mut sim = Sim::new();
    sim.wrap_all();

    // sanity: the supersonic regime is real (mach > 1); subsonically a^L == a^R and the
    // two pairings agree.
    let cs = (GAMMA * PRE0 / RHO0).sqrt();
    let speed = (sim.vx * sim.vx + sim.vy * sim.vy).sqrt();
    assert!(
        speed > cs,
        "advection must be supersonic: |v|={speed:.3} cs={cs:.3}"
    );

    let e0 = sim.mag_energy();
    assert!(
        e0 > 0.0 && e0.is_finite(),
        "init magnetic energy degenerate: {e0}"
    );

    // a CFL dt from the (constant) max signal speed. CFL=0.1 sits comfortably inside the
    // forward-euler CT stability limit, so the UCT diffusion makes the correct scheme
    // monotonically decay the (slightly over-resolved) loop energy. the downwind-paired
    // pairing breaks this: it anti-diffuses the supersonically-advected transverse field,
    // so the energy grows -> the discriminating signal.
    let cfl = 0.1;
    let dt = cfl * dx().min(dy()) / sim.smax();
    // the downwind pairing blows past 5x E0 by roughly the 13th update, so 20 updates keeps
    // margin past that. the upwind pairing damps the seed monotonically over the same window.
    let n_steps = 20;

    let mut e_max = e0;
    for step in 0..n_steps {
        // the gas is frozen (uniform rho/p/v); only the staggered B advects through the
        // CT induction loop driven by the production edge EMF.
        let ez = run_edge_emf(&sim);
        let (bx_new, by_new) = ct_induction(&sim, &ez, dt);
        sim.bx = bx_new;
        sim.by = by_new;
        sim.wrap_all();

        assert!(
            sim.all_finite(),
            "step {step}: B went non-finite (NaN/Inf) — EMF blew up"
        );
        let e = sim.mag_energy();
        e_max = e_max.max(e);
        // the correct upwind pairing keeps the loop coherent: magnetic energy stays
        // within a small band of its initial value (numerical diffusion lowers it; the
        // bound is a generous ceiling). the downwind-paired bug anti-diffuses
        // the supersonically-advected transverse field -> a runaway odd-even mode that
        // pushes the energy far past this band before NaN.
        assert!(
            e < 5.0 * e0,
            "step {step}: magnetic energy {e:.3e} blew past 5x init {e0:.3e} \
             (E_max/E0 = {:.2}) — the supersonic EMF is UNSTABLE (downwind upwind-pairing bug?)",
            e_max / e0,
        );
    }

    // report the final boundedness (visible with --nocapture).
    let e_final = sim.mag_energy();
    eprintln!(
        "[supersonic-emf] Mach={:.2}  steps={n_steps}  dt={dt:.3e}  E0={e0:.4e}  \
         E_max={e_max:.4e}  E_final={e_final:.4e}  (E_max/E0={:.3})",
        speed / cs,
        e_max / e0,
    );
    // tighter post-hoc bound: a stable advection diffuses the loop slightly, so the energy
    // stays near or below its initial value (the bug grows it past this bound). this is the
    // discriminating assert.
    assert!(
        e_max < 1.5 * e0,
        "magnetic energy GREW (E_max/E0 = {:.3}) — a stable advection only diffuses; \
         growth means the EMF advective term is anti-upwind",
        e_max / e0,
    );
}
