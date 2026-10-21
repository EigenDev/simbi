// =============================================================================
// dual_energy.rs
//
// the entropy companion of the newtonian adiabatic MHD energy: S = rho K with
// K = p rho^(-gamma), carried in the store's passive-scalar slot (cons.chi) and
// read where the conservative internal energy E - |m|^2/2rho - |B|^2/2 is too
// small against the kinetic and magnetic energies to hold its truncation error.
//
// - `advance_entropy`: the stage update S = a0 S_n + ac (S_stage - dt div F_S),
//   F_S = F_rho^+ K_upwind-left + F_rho^- K_upwind-right on the face mass flux the
//   gas update used. while each cell keeps w = rho_stage - dt sum(outgoing mass
//   flux / dx) >= 0, the new K is a convex combination of its upwind stencil's K;
//   a negative w or a K outside the stencil's range beyond round-off halts.
// - `synchronize`: after the recovery, per cell, either the conservative energy
//   sets S (E trusted) or S sets the thermal energy (entropy path), booking the
//   energy that adds; a cell in a detected shock with an inadmissible energy is
//   left to the correction ladder.
// - `drain_entropy`: the drain scales den, mom and the gas energy by f, so
//   p' = f p and S' = f^(2-gamma) S; the dyed drain applies f to S, this applies
//   the remaining f^(1-gamma).
// - `check_budgets`: the halting budgets on the booked corrections.
//
// host loops on a cartesian 2.5D or 3D grid with uniform spacing; the store is
// built with a passive scalar, which carries S.
//
// usage:
//  let cfg = DualEnergy::new(5.0 / 3.0);
//  seed_entropy(sim, cfg.gamma);
//  advance_entropy(sim, dt, a0, ac, &mut stats);   // in the godunov stage
//  let receipt = synchronize(sim, &cfg);           // after the recovery
// =============================================================================

use symbi_algebra::OrderedNumeric;
use symbi_carrier::Scalar;
use symbi_sim::dual_energy_ledger::{CorrectionReceipt, CorrectionTotals};
use symbi_sim::state::{BoundaryType, FieldStore};
use symbi_xpu::MemorySpace;

/// the switch and the halting budgets of the dual-energy update
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DualEnergy {
    /// the adiabatic index
    pub gamma: f64,
    /// the entropy path is taken where e_E < eta_1 (|m|^2/2rho + |B|^2/2)
    pub eta_1: f64,
    /// a compressing cell whose stencil's relative pressure jump exceeds eta_2 is in a shock
    pub eta_2: f64,
    /// cumulative added and removed corrections, each against the domain's thermal energy
    pub budget_global: f64,
    /// one stage's summed |corrections| against the thermal energy of its entropy-path cells
    pub budget_local: f64,
    /// the largest correction one cell may take against its own thermal energy
    pub relative_max: f64,
}

impl DualEnergy {
    /// the switch and budgets fixed before any measurement (eta_1 = 1e-3, eta_2 = 0.2,
    /// B_G = 1e-2, B_L = 1, R_max = 10)
    pub fn new(gamma: f64) -> Self {
        Self { gamma, eta_1: 1e-3, eta_2: 0.2, budget_global: 1e-2, budget_local: 1.0, relative_max: 10.0 }
    }
}

/// the transport's convexity evidence over the stages it ran: the smallest w / rho_stage and the
/// largest relative excursion of K outside its stencil's range
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransportStats {
    pub min_weight: f64,
    pub max_excursion: f64,
    pub stages: u64,
}

impl Default for TransportStats {
    fn default() -> Self {
        Self { min_weight: f64::INFINITY, max_excursion: 0.0, stages: 0 }
    }
}

/// the relative excursion a convex combination may show from rounding alone
const ROUNDING_EXCURSION: f64 = 1e-12;

fn shift<const D: usize>(c: [isize; D], d: usize, by: isize) -> [isize; D] {
    let mut out = c;
    out[d] += by;
    out
}

/// the interior cell whose value a ghost index reads: the periodic image on a periodic axis, the
/// nearest interior cell on an outflow axis
fn interior_image<const D: usize, const DOF: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    c: [isize; D],
) -> [isize; D] {
    let int = &sim.geom.interior;
    std::array::from_fn(|a| {
        let (lo, hi) = (int.spaces[a].lo, int.spaces[a].hi);
        let n = hi - lo;
        if c[a] >= lo && c[a] < hi {
            c[a]
        } else if sim.boundaries.0[a][usize::from(c[a] >= hi)] == BoundaryType::Periodic {
            lo + (c[a] - lo).rem_euclid(n)
        } else {
            c[a].clamp(lo, hi - 1)
        }
    })
}

/// the refusals the prototype's scope carries: a cartesian grid of uniform spacing, one store
/// with periodic or outflow faces, host memory, the entropy slot allocated
pub fn assert_scope<const D: usize, const DOF: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
) {
    assert!(
        sim.has_passive_scalar(),
        "dual energy carries the entropy in the passive-scalar slot; build the store with_passive_scalar"
    );
    assert!(!Mem::IS_DEVICE_ACCESSIBLE, "dual energy runs on host memory");
    assert_eq!(sim.geom.coords, symbi_geometry::Geometry::Cartesian, "dual energy runs on a cartesian chart");
    assert!(sim.geom.maps.is_none(), "dual energy runs on uniformly spaced axes");
    assert!(
        sim.geom.index_offset.iter().all(|&o| o == 0),
        "dual energy runs on a single store; a tile of a decomposed grid is refused"
    );
    for faces in sim.boundaries.0.iter() {
        for &b in faces {
            assert!(
                matches!(b, BoundaryType::Periodic | BoundaryType::Outflow),
                "dual energy runs with periodic or outflow faces; a {b:?} face is refused (refinement, tiles \
                 and walls included)"
            );
        }
    }
}

/// S = rho^(1-gamma) p from the conservative energy over the interior: the entropy of the
/// state the run starts from
pub fn seed_entropy<const D: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric>(
    sim: &FieldStore<D, 3, Mem, Sc>,
    gamma: f64,
) {
    assert_scope(sim);
    let s = sim.fields.cons.chi_field().expect("entropy slot");
    for c in sim.geom.interior.iter() {
        let (rho, p) = (cell_density(sim, c), conservative_pressure(sim, c, gamma));
        assert!(p > 0.0, "the initial state at {c:?} has pressure {p:e}; its entropy is undefined");
        s.set(c, Sc::from_f64(rho.powf(1.0 - gamma) * p));
    }
}

fn cell_density<const D: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric>(
    sim: &FieldStore<D, 3, Mem, Sc>,
    c: [isize; D],
) -> f64 {
    sim.fields.cons.den.at(c).to_f64()
}

/// (kinetic, magnetic) energy densities of the conserved cell
fn kinetic_magnetic<const D: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric>(
    sim: &FieldStore<D, 3, Mem, Sc>,
    c: [isize; D],
) -> (f64, f64) {
    let cons = &sim.fields.cons;
    let rho = cons.den.at(c).to_f64();
    let m2: f64 = (0..3).map(|k| cons.mom[k].at(c).to_f64().powi(2)).sum();
    let b = &sim.fields.mhd.as_ref().expect("magnetized store").bcell;
    let b2: f64 = (0..3).map(|k| b[k].at(c).to_f64().powi(2)).sum();
    (0.5 * m2 / rho, 0.5 * b2)
}

fn conservative_pressure<const D: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric>(
    sim: &FieldStore<D, 3, Mem, Sc>,
    c: [isize; D],
    gamma: f64,
) -> f64 {
    let e = sim.fields.cons.nrg_field().expect("adiabatic gas").at(c).to_f64();
    let (kin, mag) = kinetic_magnetic(sim, c);
    (gamma - 1.0) * (e - kin - mag)
}

/// one stage of the entropy: S = a0 S_n + ac (S_stage - dt div F_S) on the face mass fluxes the
/// gas update used, into cons.chi over the interior. reads the stage input from the store's
/// stage-input slot, so a first-order redo of the stage rebuilds it from the same input on the
/// spliced fluxes. halts on a negative outgoing-flux weight or a K outside its stencil's range.
pub fn advance_entropy<const D: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric>(
    sim: &FieldStore<D, 3, Mem, Sc>,
    dt: f64,
    a0: f64,
    ac: f64,
    stats: &mut TransportStats,
) {
    let st = sim.stage_input();
    let (st_den, st_s) = (&st.den, st.chi_field().expect("stage-input entropy slot"));
    let un = &sim.workspace.u_n;
    let un_s = un.chi_field().expect("step-start entropy slot");
    let k_stage = |c: [isize; D]| {
        let i = interior_image(sim, c);
        st_s.at(i).to_f64() / st_den.at(i).to_f64()
    };
    let s_out = sim.fields.cons.chi_field().expect("entropy slot");
    let mut updated = Vec::with_capacity(sim.geom.interior.volume());
    for c in sim.geom.interior.iter() {
        let rho = st_den.at(c).to_f64();
        let k_c = k_stage(c);
        let (mut div, mut outgoing) = (0.0, 0.0);
        let (mut k_lo, mut k_hi) = (k_c, k_c);
        for d in 0..D {
            let dx = sim.geom.dx[d];
            let flux = &sim.fields.flux[d].den;
            let f_lo = flux.at(c).to_f64();
            let f_hi = flux.at(shift(c, d, 1)).to_f64();
            let k_left = k_stage(shift(c, d, -1));
            let k_right = k_stage(shift(c, d, 1));
            let up_lo = if f_lo >= 0.0 { k_left } else { k_c };
            let up_hi = if f_hi >= 0.0 { k_c } else { k_right };
            div += (f_hi * up_hi - f_lo * up_lo) / dx;
            outgoing += (f_hi.max(0.0) - f_lo.min(0.0)) / dx;
            if f_lo > 0.0 {
                k_lo = k_lo.min(k_left);
                k_hi = k_hi.max(k_left);
            }
            if f_hi < 0.0 {
                k_lo = k_lo.min(k_right);
                k_hi = k_hi.max(k_right);
            }
        }
        let weight = (rho - dt * outgoing) / rho;
        stats.min_weight = stats.min_weight.min(weight);
        assert!(
            weight >= -ROUNDING_EXCURSION,
            "dual-energy entropy transport at {c:?}: the outgoing mass flux exceeds the cell's mass \
             (w / rho = {weight:e}); the convex-combination property fails at this timestep"
        );
        let s_n = un_s.at(c).to_f64();
        if a0 > 0.0 {
            let k_n = s_n / un.den.at(c).to_f64();
            k_lo = k_lo.min(k_n);
            k_hi = k_hi.max(k_n);
        }
        let s_new = a0 * s_n + ac * (st_s.at(c).to_f64() - dt * div);
        updated.push((c, s_new, k_lo, k_hi));
    }
    for (c, s_new, k_lo, k_hi) in updated {
        let k_new = s_new / cell_density(sim, c);
        let excursion = ((k_lo - k_new) / k_lo).max((k_new - k_hi) / k_hi).max(0.0);
        stats.max_excursion = stats.max_excursion.max(excursion);
        assert!(
            excursion <= ROUNDING_EXCURSION,
            "dual-energy entropy transport at {c:?}: K = {k_new:e} leaves its stencil's range \
             [{k_lo:e}, {k_hi:e}] by {excursion:e}"
        );
        s_out.set(c, Sc::from_f64(s_new));
    }
    stats.stages += 1;
}

/// whether `c` sits in a compression shock: the velocity divergence over its interior neighbors
/// is negative and the relative jump of the entropy pressure to a neighbor exceeds eta_2
fn in_shock<const D: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric>(
    sim: &FieldStore<D, 3, Mem, Sc>,
    c: [isize; D],
    p_entropy: &dyn Fn([isize; D]) -> f64,
    eta_2: f64,
) -> bool {
    let int = &sim.geom.interior;
    let vel = |x: [isize; D], d: usize| sim.fields.cons.mom[d].at(x).to_f64() / cell_density(sim, x);
    let (mut div, mut jump) = (0.0, 0.0f64);
    let p_c = p_entropy(c);
    for d in 0..D {
        let (lo, hi) = (shift(c, d, -1), shift(c, d, 1));
        let (in_lo, in_hi) = (int.contains(lo), int.contains(hi));
        let (a, b, span) = match (in_lo, in_hi) {
            (true, true) => (lo, hi, 2.0),
            (true, false) => (lo, c, 1.0),
            (false, true) => (c, hi, 1.0),
            (false, false) => continue,
        };
        div += (vel(b, d) - vel(a, d)) / (span * sim.geom.dx[d]);
        for n in [lo, hi] {
            if int.contains(n) {
                let p_n = p_entropy(n);
                jump = jump.max((p_n - p_c).abs() / p_n.min(p_c));
            }
        }
    }
    div < 0.0 && jump > eta_2
}

/// the synchronization after a stage's recovery. per interior cell, with e_E the conservative
/// internal energy and p_S = S rho^(gamma-1):
/// - e_E admissible and at least eta_1 of the kinetic plus magnetic energy, or a detected shock with
///   e_E admissible: S <- rho^(1-gamma) p_E;
/// - otherwise outside a shock: the thermal energy becomes p_S / (gamma - 1), the primitives follow,
///   the recovery status clears, and the change is booked;
/// - a detected shock with e_E inadmissible: left for the correction ladder, counted.
pub fn synchronize<const D: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric>(
    sim: &FieldStore<D, 3, Mem, Sc>,
    cfg: &DualEnergy,
) -> CorrectionReceipt {
    let g = cfg.gamma;
    let cons = &sim.fields.cons;
    let s_field = cons.chi_field().expect("entropy slot");
    let nrg = cons.nrg_field().expect("adiabatic gas");
    let prim = &sim.fields.prim;
    let pre = prim.pre_field().expect("adiabatic gas pressure");
    let dv: f64 = sim.geom.dx[..D].iter().product();
    let p_entropy = |c: [isize; D]| s_field.at(c).to_f64() * cell_density(sim, c).powf(g - 1.0);
    // decide on the state as recovered, before any cell is rewritten
    let decisions: Vec<([isize; D], f64, f64, bool, bool, bool)> = sim
        .geom
        .interior
        .iter()
        .map(|c| {
            let (kin, mag) = kinetic_magnetic(sim, c);
            let e_cons = nrg.at(c).to_f64() - kin - mag;
            let admissible = e_cons.is_finite() && e_cons > 0.0;
            let trusted = admissible && e_cons >= cfg.eta_1 * (kin + mag);
            let shock = !trusted && in_shock(sim, c, &p_entropy, cfg.eta_2);
            (c, e_cons, kin + mag, admissible, trusted, shock)
        })
        .collect();
    let mut r = CorrectionReceipt::default();
    for (c, e_cons, kin_mag, admissible, trusted, shock) in decisions {
        let rho = cell_density(sim, c);
        let p_e = (g - 1.0) * e_cons;
        let conservative = trusted || shock;
        if conservative && admissible {
            r.shock_cells += u64::from(shock);
            s_field.set(c, Sc::from_f64(rho.powf(1.0 - g) * p_e));
            r.thermal_domain += p_e / (g - 1.0) * dv;
            continue;
        }
        if conservative {
            r.invalid_shock_cells += 1;
            continue;
        }
        let p_s = p_entropy(c);
        assert!(
            p_s.is_finite() && p_s > 0.0,
            "dual-energy synchronization at {c:?}: the entropy pressure {p_s:e} is not admissible"
        );
        let delta = (p_s - p_e) / (g - 1.0) * dv;
        if delta > 0.0 {
            r.added += delta;
        } else {
            r.removed += delta;
        }
        let rel = (p_s - p_e).abs() / p_s;
        if rel > r.worst_relative {
            r.worst_relative = rel;
            r.worst_cell = c.to_vec();
        }
        r.above_tenth += u64::from(rel > 0.1);
        r.above_one += u64::from(rel > 1.0);
        r.entropy_cells += 1;
        r.thermal_entropy_cells += p_s / (g - 1.0) * dv;
        r.thermal_domain += p_s / (g - 1.0) * dv;
        nrg.set(c, Sc::from_f64(p_s / (g - 1.0) + kin_mag));
        prim.rho.set(c, Sc::from_f64(rho));
        for k in 0..3 {
            prim.vel[k].set(c, Sc::from_f64(cons.mom[k].at(c).to_f64() / rho));
        }
        pre.set(c, Sc::from_f64(p_s));
        sim.fields.c2p_error.set(c, Sc::from_f64(0.0));
    }
    r
}

/// the drain's entropy: the dyed drain multiplied S by f = rho_after / rho_before, and the drain's
/// p' = f p needs S' = f^(2-gamma) S, so the remaining factor f^(1-gamma) applies here
pub fn drain_entropy<const D: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric>(
    sim: &FieldStore<D, 3, Mem, Sc>,
    before: &[f64],
    gamma: f64,
) {
    let s = sim.fields.cons.chi_field().expect("entropy slot");
    for (c, rho_before) in sim.geom.interior.iter().zip(before) {
        let f = cell_density(sim, c) / rho_before;
        if f != 1.0 {
            s.set(c, Sc::from_f64(s.at(c).to_f64() * f.powf(1.0 - gamma)));
        }
    }
}

/// the interior density, the drain's reference for `drain_entropy`
pub fn densities<const D: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric>(
    sim: &FieldStore<D, 3, Mem, Sc>,
) -> Vec<f64> {
    sim.geom.interior.iter().map(|c| cell_density(sim, c)).collect()
}

/// the halting budgets on a stage's receipt, the attempt's pending stages and the accepted steps:
/// the cumulative added and removed corrections each within budget_global of the domain's thermal
/// energy, the stage's summed |corrections| within budget_local of its entropy-path thermal energy,
/// and no cell's correction above relative_max of its own thermal energy. returns the first
/// exceeded budget, named with its numbers
pub fn exceeded_budget(
    cfg: &DualEnergy,
    stage: &CorrectionReceipt,
    pending: &CorrectionTotals,
    accepted: &CorrectionTotals,
) -> Option<String> {
    let added = accepted.added + pending.added;
    let removed = accepted.removed + pending.removed;
    let thermal = stage.thermal_domain;
    if added > cfg.budget_global * thermal {
        return Some(format!(
            "cumulative added correction {added:e} exceeds {} of the domain thermal energy {thermal:e}",
            cfg.budget_global
        ));
    }
    if -removed > cfg.budget_global * thermal {
        return Some(format!(
            "cumulative removed correction {removed:e} exceeds {} of the domain thermal energy {thermal:e}",
            cfg.budget_global
        ));
    }
    let local = stage.added - stage.removed;
    if local > cfg.budget_local * stage.thermal_entropy_cells {
        return Some(format!(
            "the stage's corrections {local:e} exceed {} of its entropy-path thermal energy {:e} \
             ({} cells)",
            cfg.budget_local, stage.thermal_entropy_cells, stage.entropy_cells
        ));
    }
    if stage.worst_relative > cfg.relative_max {
        return Some(format!(
            "a correction of {:e} of its cell's thermal energy at {:?} exceeds {}",
            stage.worst_relative, stage.worst_cell, cfg.relative_max
        ));
    }
    None
}
