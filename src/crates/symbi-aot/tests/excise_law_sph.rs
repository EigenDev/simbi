// =============================================================================
// excise_law_sph.rs
//
// the spherical horizon-excision kernels' gates. the compiled fill/writeback/rebuild chain is
// bit-identical to an f64 host chain built from the same carrier-generic pieces (the vacuum floor,
// `RhdGr::to_conserved` on the ingoing kerr-schild metric), on both a uniform and a log radial
// axis.
//
// the log axis is the point of this file. on a chart whose radial coordinate is r, the excision
// mask is `r < r_exc` — but "r" has to be the cell's actual radius, and the face positions are
// selected at runtime by `map_kind_0` (0 = uniform, 1 = log). a kernel that instead read the
// affine `x_lo + i dx` would be reading an index coordinate on a log grid: at the parameters
// below that expression reaches 0.5 only at i = 10 while the true radius reaches it at i = 4, so a
// wrong reading excises six extra shells of live gas. the uniform case leaves that error
// undetected — the two formulas agree there — which is why both are run against the same kernel.
//
// the second law is the sampling point. the excised state is stored densitized, so its cell
// average is over the plain coordinate volume and its second-order sampling point is the per-axis
// midpoint, not the chart's volume-weighted centroid (they differ by dr^2/(6r) on a radial axis).
// the recovery inverts at the midpoint, so the reference below rebuilds there too: a rebuild
// sampled anywhere else leaves an excised cell whose recovered primitives diverge from the floor
// it was frozen at, and the bitwise comparison catches it.
// =============================================================================

use symbi_algebra::Tensor;
use symbi_aot::{CpuField, CpuFieldMut, kernel_by_name};
use symbi_geometry::{KerrKS, Metric};
use symbi_hydro::RhdGr;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::regime::Regime;
use symbi_hydro::spatial_metric::{Gamma, GammaInv, SpatialMetric};
use symbi_hydro::state::Prim;

const N: usize = 32;
const MASS: f64 = 1.0;
const GAMMA: f64 = 4.0 / 3.0;
// the excision surface sits strictly inside the outer horizon r_+ = 2M and above the M/2 metric
// guard, the same window the runtime preflight enforces.
const R_EXC: f64 = 1.4;

// the cold c2p-safe vacuum an excised cell is frozen at. handed to the kernel as scalars so the
// substrate can scale the floor to the problem's units; the law under test is that the compiled
// and f64 paths agree bitwise on a given floor, not what the floor's value is.
const RHO_VAC: f64 = 1e-10;
const PRE_VAC: f64 = 1e-12;

/// the radial axis under test. `Uniform` is the affine map; `Log` places faces at
/// `x_lo * 10^(i dx)`, which is what the runtime `map_kind_0 = 1` selects.
#[derive(Clone, Copy, PartialEq)]
enum Axis {
    Uniform,
    Log,
}

impl Axis {
    fn map_kind(self) -> f64 {
        match self {
            Axis::Uniform => 0.0,
            Axis::Log => 1.0,
        }
    }
    /// x_lo and the map parameter. the log axis spans [0.5, 50] over N cells, so its innermost
    /// shells are far finer than the affine map's — the regime where reading the index coordinate
    /// as a radius goes wrong.
    fn params(self) -> (f64, f64) {
        match self {
            Axis::Uniform => (0.5, 0.1),
            Axis::Log => (0.5, (50.0f64 / 0.5).log10() / N as f64),
        }
    }
    /// the lower face of cell `i` — the same map `gv_axis_face_at_index` evaluates in-kernel,
    /// spelled operation for operation. the log arm is the closed form's exponential,
    /// `exp(i dx ln 10)`, which the kernel carries so one exponential serves every grading;
    /// a base-10 power here would agree only to an ulp and a bitwise oracle would report
    /// the spelling difference as a chain divergence.
    fn face(self, i: usize) -> f64 {
        let (start, param) = self.params();
        match self {
            Axis::Uniform => start + i as f64 * param,
            Axis::Log => start * (i as f64 * param * std::f64::consts::LN_10).exp(),
        }
    }
    /// the densitized law's sampling point: the per-axis midpoint of the cell's own faces.
    fn midpoint(self, i: usize) -> f64 {
        (self.face(i) + self.face(i + 1)) * 0.5
    }
}

/// the conserved rebuild at one radius, from the same carrier the kernel traces. the polar slot is
/// the ungridded symmetry default theta = pi/2 (suppressing it to zero would zero sin(theta) and
/// make gamma_{phi phi} singular); the azimuth is 0. spin 0: the ingoing kerr-schild metric of a
/// non-rotating hole, evaluated through the spinning form's own operation order so the
/// bit-exactness law compares identical f64 sequences.
fn to_conserved_at(r: f64, prim: &Prim<f64, 3>) -> symbi_hydro::state::Cons<f64, 3> {
    let m = KerrKS {
        mass: MASS,
        spin: 0.0,
    };
    let x = Tensor::new([r, std::f64::consts::FRAC_PI_2, 0.0]);
    let metric = SpatialMetric::<f64, 3>::new(
        Gamma::new(m.spatial_metric(x)),
        GammaInv::new(m.spatial_metric_inv(x)),
    );
    RhdGr {
        metric,
        alpha: m.lapse(x),
        shift: m.shift(x),
        sqrt_gamma: m.volume_factor(x),
    }
    .to_conserved(&IdealGas { gamma: GAMMA }, &symbi_hydro::state::Valencia(*prim))
    .0
}

struct Row {
    rho: Vec<f64>,
    v0: Vec<f64>,
    pre: Vec<f64>,
    den: Vec<f64>,
    m0: Vec<f64>,
    nrg: Vec<f64>,
}

/// a smooth 1d radial state. the conserved fields carry arbitrary smooth values the rebuild must
/// overwrite inside the excision surface and pass through bit-untouched outside it.
fn smooth_row(axis: Axis) -> Row {
    let mut g = Row {
        rho: vec![0.0; N],
        v0: vec![0.0; N],
        pre: vec![0.0; N],
        den: vec![0.0; N],
        m0: vec![0.0; N],
        nrg: vec![0.0; N],
    };
    for ii in 0..N {
        let r = axis.midpoint(ii);
        g.rho[ii] = 1.0 + 0.3 * (2.0 * r).sin();
        g.v0[ii] = -0.15 * (0.5 * r).cos();
        g.pre[ii] = 0.05 + 0.02 * r.cos();
        g.den[ii] = 2.0 + 0.1 * (3.0 * r).cos();
        g.m0[ii] = 0.4 * r.sin();
        g.nrg[ii] = 0.9 + 0.05 * (2.0 * r).sin();
    }
    g
}

fn run_compiled(g: &mut Row, axis: Axis) {
    let (fill, fill_ir) = kernel_by_name::<f64>("excise_fill_sph_1d").expect("sph fill kernel");
    let (wb, wb_ir) = kernel_by_name::<f64>("excise_writeback_sph_1d").expect("sph writeback");
    let (p2c, p2c_ir) = kernel_by_name::<f64>("excise_p2c_sph_ks_1d").expect("sph p2c kernel");

    let (x_lo, dx) = axis.params();
    let scalar = |name: &str| -> f64 {
        match name {
            "gamma" => GAMMA,
            "schwarzschild_mass" => MASS,
            "kerr_spin" => 0.0,
            "excision_radius" => R_EXC,
            "excision_rho" => RHO_VAC,
            "excision_pre" => PRE_VAC,
            "x_lo_0" => x_lo,
            "dx_0" => dx,
            "map_kind_0" => axis.map_kind(),
            "map_param_0" => 0.0,
            other => panic!("unexpected scalar '{other}'"),
        }
    };
    let bind_scalars = |ir| -> (Vec<i32>, Vec<f64>) {
        let (mut ints, mut scalars) = (Vec::new(), Vec::new());
        for (bind, is_int) in symbi_ir::kernel_scalar_params_typed_from_ir(ir) {
            let v = scalar(&bind.name());
            if is_int {
                ints.push(v as i32)
            } else {
                scalars.push(v)
            }
        }
        (ints, scalars)
    };
    let (fill_ints, fill_scalars) = bind_scalars(fill_ir);
    let (wb_ints, wb_scalars) = bind_scalars(wb_ir);
    let (p2c_ints, p2c_scalars) = bind_scalars(p2c_ir);

    let lo = [0i32; 1];
    let ext = [N as u32; 1];
    // the fill is pointwise, so the whole row dispatches — no stencil margin to reserve.
    let mut exc = [vec![0.0f64; N], vec![0.0; N], vec![0.0; N]];
    {
        let inputs = [
            CpuField::from_layout(&g.rho, &lo, &ext),
            CpuField::from_layout(&g.v0, &lo, &ext),
            CpuField::from_layout(&g.pre, &lo, &ext),
        ];
        let [e0, e1, e2] = &mut exc;
        let mut outs = [
            CpuFieldMut::from_layout(e0, &lo, &ext),
            CpuFieldMut::from_layout(e1, &lo, &ext),
            CpuFieldMut::from_layout(e2, &lo, &ext),
        ];
        fill(&inputs, &mut outs, &ext, &lo, &fill_ints, &fill_scalars);
    }
    {
        let inputs = [
            CpuField::from_layout(&exc[0], &lo, &ext),
            CpuField::from_layout(&exc[1], &lo, &ext),
            CpuField::from_layout(&exc[2], &lo, &ext),
        ];
        let mut outs = [
            CpuFieldMut::from_layout(&mut g.rho, &lo, &ext),
            CpuFieldMut::from_layout(&mut g.v0, &lo, &ext),
            CpuFieldMut::from_layout(&mut g.pre, &lo, &ext),
        ];
        wb(&inputs, &mut outs, &ext, &lo, &wb_ints, &wb_scalars);
    }
    {
        let (den_in, m0_in, nrg_in) = (g.den.clone(), g.m0.clone(), g.nrg.clone());
        let inputs = [
            CpuField::from_layout(&g.rho, &lo, &ext),
            CpuField::from_layout(&g.v0, &lo, &ext),
            CpuField::from_layout(&g.pre, &lo, &ext),
            CpuField::from_layout(&den_in, &lo, &ext),
            CpuField::from_layout(&m0_in, &lo, &ext),
            CpuField::from_layout(&nrg_in, &lo, &ext),
        ];
        let mut outs = [
            CpuFieldMut::from_layout(&mut g.den, &lo, &ext),
            CpuFieldMut::from_layout(&mut g.m0, &lo, &ext),
            CpuFieldMut::from_layout(&mut g.nrg, &lo, &ext),
        ];
        p2c(&inputs, &mut outs, &ext, &lo, &p2c_ints, &p2c_scalars);
    }
}

/// the f64 chain: freeze the vacuum floor inside r_exc, then rebuild the conserved state there.
fn run_reference(g: &mut Row, axis: Axis) {
    for ii in 0..N {
        if axis.midpoint(ii) < R_EXC {
            g.rho[ii] = RHO_VAC;
            g.v0[ii] = 0.0;
            g.pre[ii] = PRE_VAC;
        }
    }
    for ii in 0..N {
        let r = axis.midpoint(ii);
        if r < R_EXC {
            let prim = Prim::<f64, 3> {
                rho: g.rho[ii],
                vel: Tensor::new([g.v0[ii], 0.0, 0.0]),
                pre: g.pre[ii],
            };
            let cons = to_conserved_at(r, &prim);
            g.den[ii] = cons.den;
            g.m0[ii] = cons.mom[0];
            g.nrg[ii] = cons.nrg;
        }
    }
}

fn excised_count(axis: Axis) -> usize {
    (0..N).filter(|&ii| axis.midpoint(ii) < R_EXC).count()
}

fn assert_matches(axis: Axis, what: &str) {
    // the premise: the surface must actually cut this grid, with live cells on both sides of the
    // comparison. an excision that covers everything or nothing tests no masking at all.
    let cut = excised_count(axis);
    assert!(
        cut > 0 && cut < N,
        "{what}: the excision surface does not cut the grid ({cut} of {N} cells inside); \
         the mask is not being exercised"
    );

    let mut compiled = smooth_row(axis);
    let mut reference = smooth_row(axis);
    run_compiled(&mut compiled, axis);
    run_reference(&mut reference, axis);

    for ii in 0..N {
        for (name, a, b) in [
            ("rho", compiled.rho[ii], reference.rho[ii]),
            ("v0", compiled.v0[ii], reference.v0[ii]),
            ("pre", compiled.pre[ii], reference.pre[ii]),
            ("den", compiled.den[ii], reference.den[ii]),
            ("m0", compiled.m0[ii], reference.m0[ii]),
            ("nrg", compiled.nrg[ii], reference.nrg[ii]),
        ] {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{what}: {name} differs at cell {ii} (r = {:.6}): compiled {a:e} vs f64 {b:e}",
                axis.midpoint(ii)
            );
        }
    }
}

#[test]
fn spherical_excise_chain_matches_the_f64_chain_bitwise_on_a_uniform_axis() {
    assert_matches(Axis::Uniform, "uniform radial axis");
}

#[test]
fn spherical_excise_chain_matches_the_f64_chain_bitwise_on_a_log_axis() {
    assert_matches(Axis::Log, "log radial axis");
}

#[test]
fn the_log_axis_would_expose_an_index_coordinate_read_as_a_radius() {
    // the log gate above is only meaningful if the two readings of "r" actually disagree on this
    // grid. pin that: the affine expression x_lo + i dx tracks an index coordinate, not a radius,
    // on a log axis, and at these parameters the two cross the excision surface many shells
    // apart. without this the log test could pass while the kernel read the wrong quantity,
    // simply because the parameters happened to make them agree.
    let axis = Axis::Log;
    let (x_lo, dx) = axis.params();
    let affine_cut = (0..N)
        .filter(|&ii| x_lo + (ii as f64 + 0.5) * dx < R_EXC)
        .count();
    let true_cut = excised_count(axis);
    assert!(
        true_cut + 4 < affine_cut,
        "the log parameters do not separate the true radius from the index coordinate \
         (true {true_cut} cells inside, affine {affine_cut}); the log gate would be vacuous"
    );
}
