// =============================================================================
// refinement_restriction_low_beta.rs
//
// the magnetized restriction at very low plasma beta. the restriction averages
// the fine level's conserved state into the covered coarse cells, so their total
// energy is the fine average, and derives each coarse cell field from the
// restricted faces, which keeps div B at machine zero. the coarse cell field
// then differs from the average of the fine cell fields by the field's curvature
// across the coarse cell, and the recovered coarse pressure is the residual of the
// total energy after that field's energy. at beta ~ 1e-3 the residual is
// smaller than that difference: fine cells that are all admissible restrict to a
// covered coarse cell of negative pressure. the same fine cells under a uniform
// field restrict to admissible coarse cells.
//
// the gate pins the current restriction's behavior at low beta, the mechanism
// behind the halt of a three-level magnetized draining sink: a change to the
// restriction's thermodynamics changes this gate on purpose.
//
// run: cargo test -p symbi --test refinement_restriction_low_beta
// =============================================================================
use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;
type Kern = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

const GAMMA: f64 = 5.0 / 3.0;
const N: usize = 8;
const DX: f64 = 1.0 / N as f64;
// near-vacuum gas under a strong field, as inside a draining sink: beta ~ 1e-3
const RHO: f64 = 1.0e-4;
const PRE: f64 = 1.5e-4;
const B0: f64 = 0.44;
const BZ: f64 = 0.12;

fn kset(sim: &Sim) -> Kern {
    Kern::new(GAMMA, 0.4, 1.0, &sim.geom.allocated)
}

/// the center of covered coarse cell (3, 3), where the field has no gradient
const X0: [f64; 2] = [3.5 * DX, 3.5 * DX];

/// a divergence-free field curved along x at strength `a` about `X0`: B_x = B0 + a (x - x0)^2,
/// B_y = -2 a (x - x0)(y - y0); at `X0` the field is uniform to first order, so the variance of the
/// children's fields across that coarse cell is second order and the curvature dominates
fn field(a: f64, d: usize, x: [f64; 2]) -> f64 {
    let (u, v) = (x[0] - X0[0], x[1] - X0[1]);
    if d == 0 { B0 + a * u * u } else { -2.0 * a * u * v }
}

/// a two-level hierarchy whose fine level holds the low-beta gas under the field of curvature `a`,
/// sampled on its own faces; returns it with the smallest fine pressure
fn build(a: f64) -> (Hier, f64) {
    let root = Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .spacing([DX; 2])
        .origin([0.0; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(0.4)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("root")
        .set_initial(|_| MhdPrim::new(Prim::adiabatic(Density(RHO), Tensor::new([0.0, 0.0, 0.0]), Pressure(PRE)), Tensor::new([B0, 0.0, BZ])))
        .seed_faces(|d, _| if d == 0 { B0 } else { 0.0 })
        .build();
    let k = kset(&root);
    let region = RefinementRegion { x_lo: [0.25, 0.25], x_hi: [0.75, 0.75] };
    let h = Hier::with_refinement(root, k, &[region], ProlongOrder::Ppm, kset).expect("hierarchy");
    h.seed_fine_from_coarse().expect("seed fine");
    let fine = &h.levels[1].state;
    let mhd = fine.fields.mhd.as_ref().unwrap();
    for d in 0..2 {
        let face = &mhd.bface.b[d];
        let dom = face.domain().clone();
        let mut v = face.view_mut();
        for c in dom.iter() {
            *v.at_mut(c) = field(a, d, fine.geom.face_coord([c[0], c[1]], d));
        }
    }
    let mut min_pre = f64::INFINITY;
    let cons = &fine.fields.cons;
    for c in fine.geom.interior.iter() {
        // each cell field the average of its two faces, the energy the gas's plus that field's
        let b: [f64; 3] = std::array::from_fn(|d| {
            if d == 2 {
                return BZ;
            }
            let face = mhd.bface.b[d].view();
            let mut hi = c;
            hi[d] += 1;
            0.5 * (*face.at(c) + *face.at(hi))
        });
        for d in 0..3 {
            *mhd.bcell[d].view_mut().at_mut(c) = b[d];
        }
        let me = 0.5 * b.iter().map(|x| x * x).sum::<f64>();
        *cons.den.view_mut().at_mut(c) = RHO;
        for d in 0..3 {
            *cons.mom[d].view_mut().at_mut(c) = 0.0;
        }
        *cons.nrg_field().unwrap().view_mut().at_mut(c) = PRE / (GAMMA - 1.0) + me;
        let recovered = (GAMMA - 1.0) * (*cons.nrg_field().unwrap().view().at(c) - me);
        min_pre = min_pre.min(recovered);
    }
    (h, min_pre)
}

/// the smallest pressure recovered on the covered coarse cells after the restriction
fn restricted_min_pressure(a: f64) -> (f64, f64) {
    let (mut h, fine_min) = build(a);
    h.level_restrict_reflux(0, 0.0);
    let coarse = &h.levels[0].state;
    let cov = h.levels[0].coverage.clone().unwrap();
    let pre = coarse.fields.prim.pre_field().unwrap().view();
    let coarse_min = cov.iter().map(|c| *pre.at(c)).fold(f64::INFINITY, f64::min);
    (fine_min, coarse_min)
}

#[test]
fn admissible_low_beta_children_under_a_curved_field_restrict_to_a_negative_coarse_pressure() {
    // the curvature makes the face-derived coarse B_x exceed the fine cell average by a h^2 / 2,
    // which at beta ~ 1e-3 outweighs the thermal energy
    let (fine_min, coarse_min) = restricted_min_pressure(2.0);
    assert!(fine_min > 0.0, "a child is inadmissible ({fine_min:e}); the gate is vacuous");
    assert!(coarse_min < 0.0, "every covered coarse pressure stays positive ({coarse_min:e}) under admissible children");
}

#[test]
fn the_same_children_under_a_uniform_field_restrict_admissibly() {
    let (fine_min, coarse_min) = restricted_min_pressure(0.0);
    assert!(fine_min > 0.0, "a child is inadmissible ({fine_min:e})");
    assert!(coarse_min > 0.0, "a uniform field restricted to a covered coarse pressure of {coarse_min:e}");
}
