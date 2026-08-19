// =============================================================================
// refinement_balanced_3d.rs
//
// the balanced coarse-fine transfer on a 3d hierarchy, where the prolongation
// of the encoded departures rides the axis-split sweep kernels: the seam
// theorem holds per line. the coarse-fine ghost slab of an x-normal seam is
// encoded as pressure departures from the mechanical equilibrium chained along x
// from the coarse cell under the nearest fine interior cell, the departures are
// prolonged, and each fine ghost decodes on the fine chain along x from that
// interior cell through the ghosts' own densities. the chain runs along the
// slab normal alone for a slab whose transverse extent is the interior, so a
// column seeded in the mechanical class along every (y, z) line — with the full
// three-dimensional potential sampled on that line — encodes to departures that
// vanish identically on every line, and the decoded ghosts sit on the fine
// x-recursion against the interior to roundoff. the plain prolongation of the
// raw state is the positive control: it leaves the ghosts off that recursion at
// truncation order.
//
// run: cargo test -p symbi --test refinement_balanced_3d -- --nocapture
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const N: usize = 16;
const CFL: f64 = 0.4;
const K0: f64 = 1.0;
const GM: f64 = 100.0;
/// the point mass sits one domain width below x = 0 on the domain's transverse
/// center line, so the column covers r in [1, 2] along the center line and the
/// potential carries genuine transverse structure.
const BODY: [f64; 3] = [-1.0, 0.5, 0.5];

type Sim = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 3>;
type Hier = Hierarchy<Newtonian, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

fn phi(x: [f64; 3]) -> f64 {
    let r2 = (0..3).map(|a| (x[a] - BODY[a]).powi(2)).sum::<f64>();
    -GM / r2.sqrt()
}

/// the density along a line: the isentrope of the line's own potential,
/// normalized to one at the outer wall.
fn line_density(x: [f64; 3]) -> f64 {
    let a = (GAMMA - 1.0) / (GAMMA * K0);
    let c = 1.0 / a + phi([1.0, x[1], x[2]]);
    (a * (c - phi(x))).powf(1.0 / (GAMMA - 1.0))
}

/// the class column along the x line through `(y, z)` on an `n`-cell lattice of
/// width `h`: pressures follow the piecewise-constant-density segment sums on
/// the lattice's own center/face ladder, marched inward from the outer wall.
fn class_line(n: usize, h: f64, y: f64, z: f64) -> Vec<(f64, f64)> {
    let center = |k: usize| [(k as f64 + 0.5) * h, y, z];
    let face = |k: usize| [k as f64 * h, y, z];
    let mut col = vec![(0.0_f64, 0.0_f64); n];
    let rho_out = line_density(center(n - 1));
    col[n - 1] = (rho_out, K0 * rho_out.powf(GAMMA));
    for k in (0..n - 1).rev() {
        let (ra, rb) = (line_density(center(k)), line_density(center(k + 1)));
        let pre = col[k + 1].1
            + rb * (phi(center(k + 1)) - phi(face(k + 1)))
            + ra * (phi(face(k + 1)) - phi(center(k)));
        col[k] = (ra, pre);
    }
    assert!(
        col.iter().all(|&(r, p)| r > 0.0 && p > 0.0),
        "the class line left the physical regime"
    );
    col
}

fn kset(balanced: bool) -> impl Fn(&Sim) -> Kset {
    move |s: &Sim| {
        Kset::new(GAMMA, CFL, &s.geom.allocated).well_balanced_reconstruction(balanced)
    }
}

/// the two-level hierarchy with the patch spanning the interior transversely, so
/// both seams are x-normal. root cells seed the class line of their own (y, z);
/// fine cells seed their parent segment's continuation — density copied, pressure
/// on the parent's linear-in-phi segment at the fine center — which lies in the
/// fine lattice's own class because the density switch lands on a shared face.
fn build(balanced: bool) -> Hier {
    let h = 1.0 / N as f64;
    let seed = move |x: [f64; 3], fine: bool| -> Prim<f64, 3> {
        let col = class_line(N, h, x[1], x[2]);
        let j = ((x[0] / h) as usize).min(N - 1);
        let (rho, pre_parent) = col[j];
        let pre = if fine {
            let xc = [(j as f64 + 0.5) * h, x[1], x[2]];
            pre_parent + rho * (phi(xc) - phi(x))
        } else {
            pre_parent
        };
        Prim {
            rho,
            vel: symbi_algebra::Tensor::zeros(),
            pre,
        }
    };
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N, N])
        .spacing([h, h, h])
        .boundaries(Boundaries::uniform(BoundaryType::Reflect))
        .cfl(CFL)
        .allocate()
        .expect("sim construction failed")
        .set_initial(move |x| seed(x, false))
        .build();
    let make = kset(balanced);
    let ck = make(&coarse);
    let region = RefinementRegion {
        x_lo: [0.25, 0.0, 0.0],
        x_hi: [0.75, 1.0, 1.0],
    };
    let hier = Hierarchy::with_refinement(coarse, ck, &[region], ProlongOrder::Ppm, make)
        .unwrap()
        .with_bodies(symbi_ib::BodyCollection::new().add(symbi_ib::Body::gravitational(
            0,
            symbi_algebra::Tensor::new(BODY),
            symbi_algebra::Tensor::zeros(),
            GM,
            1.0e-3,
            0.0,
        )));
    // the fine lattice carries its own (y, z) centers; the class line is
    // re-derived per fine cell from the full potential on that line.
    for lvl in 1..hier.levels.len() {
        hier.levels[lvl].state.seed_cells(move |x| seed(x, true));
    }
    hier
}

/// the largest relative fine x-recursion residual over every coarse-fine ghost
/// pair of the fine level, walking outward from the interior edge on each
/// (y, z) line: `p_b - [p_a + rho_a (phi(c_a) - phi(F)) + rho_b (phi(F) - phi(c_b))]`.
fn ghost_recursion_residual(hier: &Hier) -> f64 {
    let st = &hier.levels[1].state;
    let rho = st.fields.prim.rho.view();
    let pre = st.fields.prim.pre.as_ref().expect("adiabatic").view();
    let (ilo, ihi) = (
        st.geom.interior.spaces[0].lo,
        st.geom.interior.spaces[0].hi,
    );
    let (alo, ahi) = (
        st.geom.allocated.spaces[0].lo,
        st.geom.allocated.spaces[0].hi,
    );
    let mut worst = 0.0_f64;
    for jy in st.geom.interior.spaces[1].lo..st.geom.interior.spaces[1].hi {
        for jz in st.geom.interior.spaces[2].lo..st.geom.interior.spaces[2].hi {
            let residual = |a: isize, b: isize| -> f64 {
                let (ca, cb) = (st.geom.centroid([a, jy, jz]), st.geom.centroid([b, jy, jz]));
                let (ra, pa) = (*rho.at([a, jy, jz]), *pre.at([a, jy, jz]));
                let (rb, pb) = (*rho.at([b, jy, jz]), *pre.at([b, jy, jz]));
                assert!(
                    rb > 0.0 && pb > 0.0 && rb.is_finite(),
                    "cf ghost ({b}, {jy}, {jz}) holds (rho, pre) = ({rb}, {pb}); the prolong never \
                     wrote it"
                );
                let f = [0.5 * (ca[0] + cb[0]), ca[1], ca[2]];
                let chained = pa + ra * (phi(ca) - phi(f)) + rb * (phi(f) - phi(cb));
                ((pb - chained) / pb).abs()
            };
            for ii in (alo..ilo).rev() {
                worst = worst.max(residual(ii + 1, ii));
            }
            for ii in ihi..ahi {
                worst = worst.max(residual(ii - 1, ii));
            }
        }
    }
    worst
}

#[test]
fn the_swept_balanced_transfer_lands_3d_ghosts_on_the_fine_recursion() {
    let mut plain = build(false);
    plain.prime();
    let r_plain = ghost_recursion_residual(&plain);
    let mut balanced = build(true);
    balanced.prime();
    let r_wb = ghost_recursion_residual(&balanced);
    println!(
        "3d cf ghosts, max fine x-recursion residual: plain prolongation {r_plain:.3e}, \
         balanced transfer {r_wb:.3e}"
    );
    // the positive control: raw prolongation of a curved column leaves the ghosts
    // off the recursion at truncation order (measured 3.6e-2).
    assert!(
        r_plain > 1.0e-6,
        "the plain prolongation sits on the recursion to {r_plain:.3e}; the column is not \
         curved enough to exercise the transfer and the balanced arm proves nothing"
    );
    // the theorem: departures vanish line by line, so the decoded ghosts sit on the
    // fine recursion to roundoff (measured 5.5e-16). three orders of margin.
    assert!(
        r_wb < 1.0e-12,
        "the balanced transfer left the 3d ghosts {r_wb:.3e} off the fine recursion; the \
         swept prolongation of the departures or the chain decode is broken"
    );
}
