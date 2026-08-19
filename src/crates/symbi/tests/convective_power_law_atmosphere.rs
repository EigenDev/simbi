// =============================================================================
// convective_power_law_atmosphere.rs
//
// the entropy-stratified power-law atmosphere in a point-mass field, and the two
// statements that make it an initial condition whose physics is fixed before the
// run starts.
//
// the family. writing p = K rho^gamma with K = K_0 (r/r_0)^-s into hydrostatic
// balance dp/dr = -rho GM/r^2 admits a power law only at one density index:
//
//   rho = rho_0 (r/r_0)^-n,   p = p_0 (r/r_0)^-(n+1),   n = (1 - s)/(gamma - 1),
//   p_0 = rho_0 GM/(r_0 (n + 1)),
//
// the exponent following from matching the two sides of the balance and the
// coefficient from the same match. the entropy slope s is therefore the single free
// shape parameter, and everything the atmosphere does follows from it:
//
//   cs^2 = gamma GM/((n + 1) r)               virial at every radius,
//   N^2  = -(s/gamma) GM/r^3                  schwarzschild, uniform in Omega_K^2,
//
// so the column is convectively unstable for every s > 0, marginally stable at
// s = 0, and the linear growth rate is the analytic N = Omega_K sqrt(s/gamma).
// at gamma = 5/3: s = 0 gives n = 3/2 (isentropic), s = 1/6 gives n = 5/4,
// s = 1/3 gives n = 1, s = 0.47 gives n = 0.795.
//
// the two gates.
//
//   stationarity, on a radial column. convection needs a non-radial displacement, so
//   a one-dimensional radial run has no unstable mode at all and whatever motion it
//   develops is the scheme's own. the isentropic member s = 0 lies exactly on the
//   local-isentrope profile the balanced reconstruction transforms against and is
//   held at machine zero; a member with s > 0 leaves that profile at second order in
//   the potential difference across a cell, so its residual is a truncation error
//   which shrinks under refinement. the plain arm at the same clock is the positive
//   control. every such residual is spherically symmetric, which is what makes it
//   orthogonal to the modes the second gate measures.
//
//   growth rate, on a thin meridional wedge. a shell thin against its own radius
//   carries an almost uniform N, and a wedge walled on all four sides admits a
//   gravest mode that is a half-wave in each direction, whose rate the boussinesq
//   g-mode branch sigma = N k_perp/|k| gives in closed form. that mode is seeded as
//   a stream function, so it is divergence-free and launches no sound, together with
//   the displacement a growing mode carries alongside its velocity, at 1e-6 of the
//   local sound speed. the measured quantity is the kinetic energy of the
//   non-axisymmetric velocity, which discards the radial residual the first gate
//   bounds. what separates the measurement from the prediction is the scheme's
//   dissipation of the seeded mode, and refining the wedge removes it.
//
// run: cargo test -p symbi --test convective_power_law_atmosphere -- --nocapture
//      cargo test --release -p symbi --test convective_power_law_atmosphere \
//        wedge_rate_probe -- --ignored --nocapture
// =============================================================================

use std::ops::ControlFlow;
use std::time::Instant;

use symbi::prelude::Solver;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Spherical;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection, SofteningKind};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const GM: f64 = 1.0;
const CFL: f64 = 0.4;

/// the compact softening length of the central body. the compact family is exactly
/// newtonian outside it, so at half the innermost gas radius the field over the whole
/// domain is the bare -GM/r the power law balances against, to the last bit.
const SOFTENING: f64 = 0.5;

/// the entropy slopes the ladder spans. `0.0` is the isentropic, marginally stable member
/// and `0.47` is the slope a self-similar convective accretion flow settles on.
const SLOPES: [f64; 4] = [0.0, 1.0 / 6.0, 1.0 / 3.0, 0.47];

/// the riemann solver the whole ladder runs.
///
/// the full acoustic dissipation is what makes both measurements below statements about the
/// atmosphere. the low-mach ramp reduces that dissipation in proportion to the local mach
/// number, and on a sealed column with a reflecting inner wall the mode standing against
/// that wall then grows exponentially at a rate near 1.6 per unit time -- five times the
/// buoyant rate at s = 1/6, rising with resolution, seeded by roundoff on the isentropic
/// column and by the second-order residual on a stratified one, and saturating near the
/// mach number at which the ramp restores the damping. under `Hllc` the same column holds
/// flat at truncation over the same clock.
const SOLVER: Solver = Solver::Hllc;

/// the density index hydrostatic balance forces at entropy slope `s`.
fn density_index(s: f64) -> f64 {
    (1.0 - s) / (GAMMA - 1.0)
}

/// the column at radius `r`, anchored to unit density at unit radius. the pressure
/// coefficient is fixed by the balance rather than free.
fn column(s: f64, r: f64) -> (f64, f64) {
    let n = density_index(s);
    (r.powf(-n), GM / (n + 1.0) * r.powf(-(n + 1.0)))
}

/// the brunt-vaisala growth rate `N = Omega_K sqrt(s/gamma)` at radius `r`.
fn growth_rate(s: f64, r: f64) -> f64 {
    (s / GAMMA).sqrt() * (GM / (r * r * r)).sqrt()
}

/// the sound speed the column carries at radius `r`: `cs^2 = gamma GM/((n + 1) r)`.
fn sound_speed(s: f64, r: f64) -> f64 {
    (GAMMA * GM / ((density_index(s) + 1.0) * r)).sqrt()
}

/// the power-law column in the mechanical scheme's own discrete class (Kaeppeli &
/// Mishra, A&A 587, A94, 2016): densities are the power law at the cell centers,
/// pressures follow the piecewise-constant-density segment sums of `-rho dphi` on the
/// kernel's center/face ladder, marched inward from the outer wall where the pressure
/// only grows. the mechanical scheme holds this column exactly at every entropy slope,
/// so the balanced 1D arms measure the scheme against its own discrete fixed point.
/// the compact softening is bare newtonian outside its support radius, so `-GM/r` is
/// the potential the kernels evaluate everywhere in the domain.
fn class_column(s: f64, cells: usize) -> Vec<(f64, f64)> {
    let h = (R_OUT - R_IN) / cells as f64;
    let phi = |r: f64| -GM / r;
    let center = |k: usize| R_IN + (k as f64 + 0.5) * h;
    let face = |k: usize| R_IN + k as f64 * h;
    let mut col = vec![(0.0_f64, 0.0_f64); cells];
    let (rho_out, pre_out) = column(s, center(cells - 1));
    col[cells - 1] = (rho_out, pre_out);
    for k in (0..cells - 1).rev() {
        let ra = column(s, center(k)).0;
        let rb = column(s, center(k + 1)).0;
        let pre = col[k + 1].1
            + rb * (phi(center(k + 1)) - phi(face(k + 1)))
            + ra * (phi(face(k + 1)) - phi(center(k)));
        col[k] = (ra, pre);
    }
    assert!(
        col.iter().all(|&(r, p)| r > 0.0 && p > 0.0),
        "the class column left the physical regime; the fixed point is vacuous"
    );
    col
}

/// the central point mass, as a gravity-only body at the chart origin under the compact
/// softening family.
fn central_mass<const D: usize>() -> BodyCollection<f64, D> {
    BodyCollection::new().add(
        Body::gravitational(
            0,
            Tensor::zeros(),
            Tensor::zeros(),
            GM,
            // pointlike mask: the body exerts gravity and nothing else.
            1.0e-6,
            SOFTENING,
        )
        .with_softening_kind(SofteningKind::Compact),
    )
}

/// the least-squares slope of `y` against `x`.
fn slope(points: &[(f64, f64)]) -> f64 {
    let m = points.len() as f64;
    let (sx, sy) = points.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
    let (mx, my) = (sx / m, sy / m);
    let num: f64 = points.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let den: f64 = points.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum();
    num / den
}

// =============================================================================
// gate one: the radial column stands still
// =============================================================================

/// the radial domain [1, 2]: the central mass sits at the chart origin, one domain width
/// inside the inner wall, so the potential is genuinely curved across the column.
const R_IN: f64 = 1.0;
const R_OUT: f64 = 2.0;
/// the clock the column is held over: 2.2 keplerian times 1/Omega_K at the middle of the
/// window and about three sound crossings of the domain, so the certificate covers the
/// dynamical time of the atmosphere it describes and every wall signal has crossed and
/// returned. a fixed clock rather than a fixed step count is what makes the residual
/// comparable across resolutions, where dt moves with dx.
const T_1D: f64 = 4.0;
/// the measurement window in r, clear of both walls.
const WINDOW_1D: (f64, f64) = (1.2, 1.8);

type Sim1 = SimState<Newtonian, 1, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset1 = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;
type Hier1 = Hierarchy<Newtonian, 1, 1, Spherical, IdealGas<f64>, CpuSpace, HostMemory, Kset1>;

fn build_column(s: f64, cells: usize, balanced: bool, solver: Solver, in_class: bool) -> Hier1 {
    let sim = Sim1::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells([cells])
        .origin([R_IN])
        .spacing([(R_OUT - R_IN) / cells as f64])
        // a reflecting wall exerts no work on gas at rest, so the column is a fixed point
        // of the boundary as well as of the interior.
        .boundaries(Boundaries::uniform(BoundaryType::Reflect))
        .cfl(CFL)
        .allocate()
        .expect("sim construction failed")
        .set_initial({
            // `in_class` seeds the mechanical scheme's discrete fixed point; the analytic
            // power law (which sits O(h^2) off that class) seeds the truncation arms.
            let col = in_class.then(|| class_column(s, cells));
            let h = (R_OUT - R_IN) / cells as f64;
            move |[r]: [f64; 1]| {
                let (rho, pre) = match &col {
                    Some(col) => col[(((r - R_IN) / h - 0.5).round() as usize).min(col.len() - 1)],
                    None => column(s, r),
                };
                Prim {
                    rho,
                    vel: Tensor::new([0.0]),
                    pre,
                }
            }
        })
        .build();
    let kernels = Kset1::new(GAMMA, CFL, &sim.geom.allocated)
        .with_solver(solver)
        .expect("solver/regime mismatch")
        .well_balanced_reconstruction(balanced);
    Hierarchy::single(sim, kernels).with_bodies(central_mass())
}

/// the largest |v| and the largest relative density departure from the declared column,
/// away from the walls, at `T_1D`.
fn column_residual(
    s: f64,
    cells: usize,
    balanced: bool,
    solver: Solver,
    in_class: bool,
) -> (f64, f64) {
    let mut hier = build_column(s, cells, balanced, solver, in_class);
    hier.evolve(T_1D).unwrap();
    let st = &hier.levels[0].state;
    let rho = st.fields.prim.rho.view();
    let vel = st.fields.prim.vel[0].view();
    let (mut vmax, mut dmax, mut counted) = (0.0_f64, 0.0_f64, 0usize);
    for ii in st.geom.interior.spaces[0].lo..st.geom.interior.spaces[0].hi {
        let r = st.geom.cell_coord([ii])[0];
        if r < WINDOW_1D.0 || r > WINDOW_1D.1 {
            continue;
        }
        counted += 1;
        let c = [ii];
        vmax = vmax.max(vel.at(c).abs());
        dmax = dmax.max((*rho.at(c) / column(s, r).0 - 1.0).abs());
    }
    assert!(counted > 16, "window too narrow: {counted} cells");
    (vmax, dmax)
}

/// a low-mach correction on a stratified background must leave the column at rest as
/// resolution rises. an isentropic column carries no free energy — s = 0 sets N^2 = 0, so
/// there is no unstable mode to feed on — and a radial run admits no convective displacement
/// in any case, so every motion the column develops is the scheme's own. the discriminating
/// measurement is the trend with resolution: a truncation residual falls under refinement,
/// while an amplification of roundoff rises with it, because refinement shortens the timestep
/// and buys the growing mode more e-foldings over the same physical time.
///
/// both HLLC+ corrections carry a velocity jump as a factor and this column presents neither,
/// so the flux is the classical one with its pressure-jump dissipation whole — the dissipation
/// that damps a hydrostatic truncation residual. a scheme that scaled the acoustic signal
/// speeds attenuates that same dissipation, since the speeds multiply the pressure jump too,
/// and the residual then grows rather than sits: the family this solver replaced reached
/// max|v| of 7.6e-11 on 128 cells and 1.4e-3 on 256, amplifying by seven orders across one
/// refinement.
///
/// the plain reconstruction is the positive control. it leaves the second-order hydrostatic
/// face jump the balancing removes, so the column carries a genuine residual and the balanced
/// arm's stillness is a statement about the pairing rather than about a column too quiet to
/// test anything.
#[test]
fn the_low_mach_correction_leaves_the_isentropic_column_at_rest_under_refinement() {
    // s = 0: the isentropic member, marginally stable, and the one that lies exactly on the
    // profile the balanced reconstruction transforms against.
    const S: f64 = 0.0;
    const T_END: f64 = 8.0;
    let peak_speed = |balanced: bool, cells: usize| -> f64 {
        let mut hier = build_column(S, cells, balanced, Solver::HllcPlus, true);
        hier.evolve(T_END).unwrap();
        let st = &hier.levels[0].state;
        let vel = st.fields.prim.vel[0].view();
        let mut vmax = 0.0_f64;
        for ii in st.geom.interior.spaces[0].lo..st.geom.interior.spaces[0].hi {
            vmax = vmax.max(vel.at([ii]).abs());
        }
        vmax
    };

    let balanced = [128usize, 256].map(|n| (n, peak_speed(true, n)));
    let plain = [128usize, 256].map(|n| (n, peak_speed(false, n)));
    println!(
        "\nisentropic sealed column at t = {T_END}, HLLC+\n{:>14} {:>7} {:>12}",
        "reconstruction", "cells", "max|v|"
    );
    for (tag, arm) in [("balanced", &balanced), ("plain", &plain)] {
        for (n, v) in arm {
            println!("{tag:>14} {n:>7} {v:>12.3e}");
        }
    }

    // the precondition: the column carries a hydrostatic residual the balancing has to remove.
    assert!(
        plain[1].1 > 1.0e-9,
        "the plain-reconstruction control sits at max|v| = {:.3e} on {} cells; the column is \
         no longer producing a hydrostatic truncation residual and the balanced arm's \
         stillness is vacuous. deepen the potential or lengthen the run",
        plain[1].1,
        plain[1].0
    );

    // the balanced arm holds the column at the level classical dissipation gives it, at both
    // resolutions. measured 3.1e-15 on 128 cells and 5.0e-15 on 256.
    for (n, v) in balanced {
        assert!(
            v < 1.0e-12,
            "the balanced arm reaches max|v| = {v:.3e} on {n} cells; a column at rest presents \
             no velocity jump, so both corrections are inert and the classical flux holds it \
             at roundoff"
        );
    }
    assert!(
        balanced[1].1 < 1.0e3 * balanced[0].1.max(1.0e-16),
        "the balanced arm's residual grew from {:.3e} to {:.3e} across a factor of two in \
         resolution; a stationary column's residual must not amplify under refinement",
        balanced[0].1,
        balanced[1].1
    );
}

#[test]
fn the_power_law_column_stands_still() {
    let clock = Instant::now();
    println!(
        "\nradial power-law column, t = {T_1D} at 128 cells, r in [1, 2], {SOLVER:?} solver"
    );
    println!(
        "{:>6} {:>8} {:>12} {:>12} {:>12} {:>12}",
        "s", "n", "|v| plain", "|v| wb", "drho plain", "drho wb"
    );
    let measured: Vec<(f64, f64, f64)> = SLOPES
        .iter()
        .map(|&s| {
            let (v_plain, d_plain) = column_residual(s, 128, false, SOLVER, true);
            let (v_wb, d_wb) = column_residual(s, 128, true, SOLVER, true);
            println!(
                "{:6.4} {:8.4} {:12.3e} {:12.3e} {:12.3e} {:12.3e}",
                s,
                density_index(s),
                v_plain,
                v_wb,
                d_plain,
                d_wb
            );
            (s, v_plain, v_wb)
        })
        .collect();

    for &(s, v_plain, v_wb) in &measured {
        // the plain arm must move: its analytic rho*g source mismatches the discrete
        // pressure gradient at truncation order, so a stagnant balanced arm means
        // something only against a setup that is demonstrably live. the plain arm runs
        // 2.5e-5 to 2.8e-5 across the slopes, so the floor sits an order below it.
        assert!(
            v_plain > 2.5e-6,
            "the plain arm sits at |v| = {v_plain:.3e} at s = {s}; the column is not \
             exercising the imbalance and the balanced arm proves nothing"
        );
        // the balanced arm sits on its own discrete fixed point at every slope and is
        // held at machine zero below; the two-order separation is the cheap invariant
        // that survives even a bound retune.
        assert!(
            v_wb * 1.0e2 < v_plain,
            "the balanced arm ({v_wb:.3e}) is within two orders of the plain arm \
             ({v_plain:.3e}) at s = {s}; the hydrostatic transform is no longer holding \
             the discrete equilibrium"
        );
    }

    // every slope is the mechanical scheme's own discrete fixed point: the segment-sum
    // class carries arbitrary entropy stratification, so the balanced arm holds each
    // column at machine zero — the isentrope enjoys no special status. measured 4.9e-16
    // to 7.4e-16 across the four slopes at the full clock, three orders inside the bound.
    for &(s, _, v_wb) in &measured {
        assert!(
            v_wb < 1.0e-12,
            "the s = {s} column drifts at |v| = {v_wb:.3e}; a column seeded in the \
             mechanical scheme's own discrete class is a fixed point at every entropy \
             slope and must be held at machine zero"
        );
    }
    println!("elapsed {:.1} s", clock.elapsed().as_secs_f64());
}

#[test]
fn the_stratified_column_residual_converges_at_second_order() {
    // the residual the balanced reconstruction leaves on a stratified-entropy column is a
    // discretization error: the local isentrope through a cell matches the true profile in
    // value and in first derivative and departs from it at second order in the potential
    // difference across the stencil. that makes the residual converge as dx^2, which is
    // what separates it from a standing force, and what makes the seeded perturbation of
    // the growth gate the dominant non-axisymmetric signal there. the isentropic member
    // carries no such term and is excluded: it sits at roundoff, where a rate is
    // meaningless.
    let clock = Instant::now();
    let s = 1.0 / 6.0;
    println!("\nstratified column residual under refinement, s = {s:.4}, {SOLVER:?} solver");
    let mut previous: Option<(usize, f64)> = None;
    for cells in [64usize, 128, 256] {
        let (v, drho) = column_residual(s, cells, true, SOLVER, false);
        let order = previous.map(|(pc, pv)| (pv / v).log2() / ((cells / pc) as f64).log2());
        println!(
            "{cells:5} cells: |v| {v:.3e}, drho/rho {drho:.3e}{}",
            order.map_or(String::new(), |o| format!(", order {o:.2}"))
        );
        if let Some(o) = order {
            assert!(
                o > 1.7,
                "the residual converges at order {o:.2} from {cells} cells down; a \
                 second-order truncation error is what the local-isentrope transform \
                 leaves on a stratified-entropy profile"
            );
        }
        previous = Some((cells, v));
    }
    println!("elapsed {:.1} s", clock.elapsed().as_secs_f64());
}

// =============================================================================
// gate two: the growth rate on a meridional wedge
// =============================================================================

/// the shell's inner radius and its thickness. eight percent of the radius holds N to
/// within +/- 5.8 percent of its value at the shell's midpoint.
const R_SHELL: f64 = 1.0;
const D_SHELL: f64 = 0.08;
/// the wedge the slope scan runs on: `NR` cells across the shell, `NT` across a wedge
/// `ASPECT` times as wide, so the cells are square and the gravest mode is resolved by 24
/// cells per half-wave in each direction.
///
/// `ASPECT` at one costs 29 percent of N in geometry (the gravest mode's k_perp equals its
/// k_z there) and buys back far more in dissipation, since the damping of a smooth mode
/// falls steeply with its wavelength in cells. it also keeps the seeded mode the dominant
/// one: a wider wedge admits half-waves with k_perp above the seeded one, which grow faster
/// on the g-mode branch, and past `ASPECT` = 2 they overtake the seed inside the run and
/// the record stops being a single exponential.
const NR: usize = 24;
const NT: usize = 24;
const ASPECT: f64 = 1.0;
/// the seeded velocity as a fraction of the local sound speed. a superadiabatic column
/// grows its own motion, so this sets only the length of the linear phase.
const SEED_MACH: f64 = 1.0e-6;
/// root iterations between samples of the perturbation energy.
const SAMPLE_INTERVAL: u64 = 200;
/// e-foldings of the predicted rate the wedge is run for.
const FOLDS: f64 = 6.0;
/// the plm slope compression, on the monotonized-central end of the admissible range.
///
/// this is the single largest term in the measurement. the minmod slope clips the extremum
/// of a smooth mode every step, and on this wedge that clipping alone costs 60 percent of
/// the growth rate: the measured rate at `NR` = 24 runs at 0.40 of the prediction under
/// minmod and 0.98 under the compressive limiter. a hydrostatic column has no extremum to
/// clip, which is why the radial gates above are unaffected by the choice and run the
/// default.
const THETA: f64 = 2.0;

type Sim2 = SimState<Newtonian, 2, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset2 = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Hier2 = Hierarchy<Newtonian, 2, 2, Spherical, IdealGas<f64>, CpuSpace, HostMemory, Kset2>;

/// the shell midpoint, where the prediction is evaluated.
fn r_mid() -> f64 {
    R_SHELL + 0.5 * D_SHELL
}

/// the eigenvalue of the gravest wedge mode, as a fraction of N.
///
/// a wedge of width `aspect` x the shell thickness, walled on all four sides, admits a
/// gravest mode that is a half-wave in each direction: k_perp = pi/width and
/// k_z = pi/thickness. the boussinesq g-mode branch sigma = N k_perp/|k| then reduces to
/// 1/sqrt(1 + aspect^2), which is the whole geometric content of the comparison.
fn mode_factor(aspect: f64) -> f64 {
    1.0 / (1.0 + aspect * aspect).sqrt()
}

/// the wedge, seeded with the gravest convective eigenmode at `SEED_MACH` of the local
/// sound speed.
///
/// the mode is carried by the stream function psi = C sin(pi u) sin(pi q) on the shell
/// coordinate u and the wedge coordinate q, read as v_r = psi_theta/(r^2 sin theta) and
/// v_theta = -psi_r/(r sin theta). that pair is divergence-free, so the seed carries no
/// compression and launches no sound, and psi vanishes on all four walls, which puts v_r at
/// zero on the radial walls and v_theta at zero on the angular ones.
///
/// the density carries the matching displacement. a parcel lifted by xi_r and brought
/// adiabatically to the local pressure arrives at rho (1 - xi_r s/(gamma r)), lighter than
/// its surroundings for s > 0, which is the buoyancy the growth feeds on. a growing mode
/// holds v = sigma xi, so seeding the displacement alongside the velocity at that ratio
/// puts the state on the eigenfunction and the record enters its exponential phase at once.
/// the pressure is the column's, since a displacement at constant pressure is what
/// buoyancy acts on.
fn build_wedge(s: f64, nr: usize, nt: usize, aspect: f64) -> Hier2 {
    let span = aspect * D_SHELL / r_mid();
    let theta_lo = 0.5 * std::f64::consts::PI - 0.5 * span;
    let pi = std::f64::consts::PI;
    let peak = SEED_MACH * sound_speed(s, r_mid());
    let amplitude = peak * r_mid() * r_mid() * span / pi;
    let rate = growth_rate(s, r_mid()) * mode_factor(aspect);
    let ic = move |[r, th]: [f64; 2]| {
        let (rho, pre) = column(s, r);
        let (u, q) = ((r - R_SHELL) / D_SHELL, (th - theta_lo) / span);
        let (f, df) = ((pi * u).sin(), (pi / D_SHELL) * (pi * u).cos());
        let (g, dg) = ((pi * q).sin(), (pi / span) * (pi * q).cos());
        let sin_t = th.sin();
        let v_r = amplitude * f * dg / (r * r * sin_t);
        let v_t = -amplitude * df * g / (r * sin_t);
        // the rate vanishes with the schwarzschild discriminant, so the marginal column
        // starts from pure velocity; its density perturbation carries a factor s and is
        // zero there in any case.
        let displacement = if rate > 0.0 { v_r / rate } else { 0.0 };
        Prim {
            rho: rho * (1.0 - displacement * s / (GAMMA * r)),
            vel: Tensor::new([v_r, v_t]),
            pre,
        }
    };
    let sim = Sim2::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells([nr, nt])
        .origin([R_SHELL, theta_lo])
        .spacing([D_SHELL / nr as f64, span / nt as f64])
        // reflecting on both axes: the radial walls seal the shell and the angular walls are
        // the symmetry planes of the gravest wedge mode, so the seeded mode is an exact
        // solution of the boundary as well as of the interior.
        .boundaries(Boundaries::uniform(BoundaryType::Reflect))
        .cfl(CFL)
        .allocate()
        .expect("sim construction failed")
        .set_initial(ic)
        .build();
    let kernels = Kset2::new(GAMMA, CFL, &sim.geom.allocated)
        .with_solver(SOLVER)
        .expect("solver/regime mismatch")
        .theta(THETA)
        .well_balanced_reconstruction(true);
    Hierarchy::single(sim, kernels).with_bodies(central_mass())
}

/// the kinetic energy carried by the non-axisymmetric part of the velocity.
///
/// subtracting the angular mean shell by shell removes the spherically symmetric residual
/// the balanced reconstruction leaves on a stratified-entropy column, which shares no
/// component with a convective mode. what remains is the instability alone.
fn fluctuating_energy(st: &Sim2) -> f64 {
    let rho = st.fields.prim.rho.view();
    let v_r = st.fields.prim.vel[0].view();
    let v_t = st.fields.prim.vel[1].view();
    let (ilo, ihi) = (st.geom.interior.spaces[0].lo, st.geom.interior.spaces[0].hi);
    let (jlo, jhi) = (st.geom.interior.spaces[1].lo, st.geom.interior.spaces[1].hi);
    let count = (jhi - jlo) as f64;
    let mut energy = 0.0;
    for ii in ilo..ihi {
        let (mut mean_r, mut mean_t) = (0.0, 0.0);
        for jj in jlo..jhi {
            mean_r += *v_r.at([ii, jj]);
            mean_t += *v_t.at([ii, jj]);
        }
        mean_r /= count;
        mean_t /= count;
        for jj in jlo..jhi {
            let c = [ii, jj];
            let (dr, dt) = (*v_r.at(c) - mean_r, *v_t.at(c) - mean_t);
            energy += 0.5 * *rho.at(c) * (dr * dr + dt * dt);
        }
    }
    energy
}

/// evolve the wedge to `t_end`, sampling the perturbation energy. returns the samples as
/// `(time, ln(E/E_0))`.
fn energy_record(s: f64, nr: usize, nt: usize, aspect: f64, t_end: f64) -> Vec<(f64, f64)> {
    let mut hier = build_wedge(s, nr, nt, aspect);
    hier.prime();
    let seeded = fluctuating_energy(&hier.levels[0].state);
    assert!(
        seeded > 0.0,
        "the wedge carries no non-axisymmetric velocity; the seed never landed"
    );
    let mut record = vec![(hier.levels[0].state.time, 0.0)];
    hier.evolve_with_callback(t_end, SAMPLE_INTERVAL, |h| {
        let st = &h.levels[0].state;
        record.push((st.time, (fluctuating_energy(st) / seeded).ln()));
        ControlFlow::Continue(())
    })
    .unwrap();
    record
}

/// the exponential rate of the perturbation velocity, fitted over the record's second half,
/// together with the rate over its final quarter.
///
/// the pair is the record's own linearity check: a record still shedding the part of the
/// seed that lies off the eigenfunction, or already leaving the linear regime, gives two
/// different rates over the two spans, while a clean exponential gives one.
fn fitted_rate(record: &[(f64, f64)]) -> (f64, f64) {
    let half = &record[record.len() / 2..];
    let quarter = &record[3 * record.len() / 4..];
    assert!(
        quarter.len() >= 8,
        "the record holds {} samples in its final quarter; a rate fitted over fewer is not \
         a measurement",
        quarter.len()
    );
    // the energy carries twice the velocity's exponent.
    (0.5 * slope(half), 0.5 * slope(quarter))
}

/// the measured rate on one wedge, reported against the predicted one.
fn measure(s: f64, nr: usize, nt: usize, aspect: f64) -> (f64, f64) {
    let predicted = growth_rate(s, r_mid()) * mode_factor(aspect);
    let record = energy_record(s, nr, nt, aspect, FOLDS / predicted);
    let (half, quarter) = fitted_rate(&record);
    assert!(
        (half - quarter).abs() < 0.03 * half.abs().max(predicted),
        "the rate over the record's second half ({half:.5}) and over its final quarter \
         ({quarter:.5}) disagree; the record is not in a clean exponential phase"
    );
    (quarter, predicted)
}

#[test]
#[ignore = "resolution and wavelength probe; run explicitly in release"]
fn wedge_rate_probe() {
    // the evidence behind the tolerance the slope scan asserts. the deficit is the scheme's
    // dissipation of the seeded mode, so it falls with the cell width and rises with the
    // mode's wavenumber, and both dependences are visible here.
    let s = 1.0 / 6.0;
    println!("\nwedge probe at s = {s:.4}, {SOLVER:?}, theta {THETA}");
    println!(
        "{:>4} {:>4} {:>7} {:>11} {:>11} {:>8} {:>11}",
        "n_r", "n_th", "aspect", "predicted", "measured", "ratio", "deficit"
    );
    for (nr, nt, aspect) in [
        (24usize, 12usize, 0.5),
        (48, 24, 0.5),
        (24, 24, 1.0),
        (48, 48, 1.0),
    ] {
        let (measured, predicted) = measure(s, nr, nt, aspect);
        println!(
            "{nr:4} {nt:4} {aspect:7.2} {predicted:11.5} {measured:11.5} {:8.4} {:11.5}",
            measured / predicted,
            predicted - measured
        );
    }
}

#[test]
fn the_wedge_grows_at_the_brunt_vaisala_rate() {
    let clock = Instant::now();
    println!(
        "\nconvective wedge {NR} x {NT}, aspect {ASPECT}, r in [{R_SHELL}, {:.3}], \
         {SOLVER:?} theta {THETA}",
        R_SHELL + D_SHELL
    );
    println!(
        "{:>6} {:>8} {:>11} {:>11} {:>11} {:>8}",
        "s", "n", "N", "predicted", "measured", "ratio"
    );
    let mut ratios: Vec<f64> = Vec::new();
    for &s in SLOPES.iter().skip(1) {
        let (measured, predicted) = measure(s, NR, NT, ASPECT);
        let ratio = measured / predicted;
        ratios.push(ratio);
        println!(
            "{:6.4} {:8.4} {:11.5} {:11.5} {:11.5} {:8.4}",
            s,
            density_index(s),
            growth_rate(s, r_mid()),
            predicted,
            measured,
            ratio
        );
        // every systematic here reduces the measured rate below the prediction: the
        // scheme's dissipation of the seeded mode, and the fall of N across the shell. the
        // first is 2.0 percent of the rate at this grid and falls as the cell width cubed
        // (the probe above measures 0.00429 at 24 cells against 0.00056 at 48, on the same
        // mode), the second is bounded by the 5.8 percent N spans over the shell and enters
        // only through the eigenfunction's weighting of it. five percent covers both.
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "the s = {s} wedge grows at {measured:.5} against the predicted \
             {predicted:.5} (ratio {ratio:.4}); the atmosphere is not on the \
             brunt-vaisala branch"
        );
    }
    // the sqrt(s) law is the sharp statement. the dissipation and the geometry are
    // properties of the grid and the mode rather than of the stratification, so they leave
    // the ratio flat across the slopes; a rate tracking something other than s would not
    // hold one ratio over a factor 1.7 in N.
    let spread = ratios.iter().cloned().fold(f64::MIN, f64::max)
        - ratios.iter().cloned().fold(f64::MAX, f64::min);
    println!("ratio spread across the slopes: {spread:.4}");
    assert!(
        spread < 0.03,
        "the measured/predicted ratio spans {spread:.4} across the slopes; the rate is not \
         scaling as sqrt(s)"
    );
    println!("elapsed {:.1} s", clock.elapsed().as_secs_f64());
}

#[test]
fn the_measured_rate_converges_to_the_analytic_one() {
    // what remains between the measurement and the prediction is discretization, so
    // refining removes it. this is what licenses the tolerance the scan above carries: the
    // gap is not a property of the atmosphere.
    let clock = Instant::now();
    let s = 1.0 / 6.0;
    println!("\nwedge rate under refinement, s = {s:.4}, aspect 0.5, {SOLVER:?} theta {THETA}");
    let (coarse, predicted) = measure(s, 24, 12, 0.5);
    let (fine, _) = measure(s, 48, 24, 0.5);
    let (d_coarse, d_fine) = (predicted - coarse, predicted - fine);
    println!(
        "  24 x 12: {coarse:.5} (deficit {d_coarse:.5})\n\
           48 x 24: {fine:.5} (deficit {d_fine:.5})\n\
         predicted {predicted:.5}, deficit ratio {:.2}",
        d_coarse / d_fine
    );
    assert!(
        d_fine * 4.0 < d_coarse,
        "the deficit fell from {d_coarse:.5} to {d_fine:.5} on halving the cell width, \
         short of the second order a discretization error carries"
    );
    assert!(
        (fine / predicted - 1.0).abs() < 0.02,
        "the refined wedge grows at {fine:.5} against the predicted {predicted:.5}; \
         refinement is not closing on the analytic rate"
    );
    println!("elapsed {:.1} s", clock.elapsed().as_secs_f64());
}

#[test]
fn the_marginal_wedge_does_not_grow() {
    // the isentropic column is neutrally stable: the schwarzschild discriminant vanishes
    // with s, so the seeded displacement is neither buoyant nor restored. running it on the
    // same wedge over the clock in which the s = 1/6 column grows through the whole record
    // is what shows the measurement responds to the entropy slope rather than to the seed.
    let clock = Instant::now();
    let reference = growth_rate(SLOPES[1], r_mid()) * mode_factor(ASPECT);
    let record = energy_record(0.0, NR, NT, ASPECT, FOLDS / reference);
    let final_ratio = record.last().expect("control record").1;
    let drift = 0.5 * slope(&record);
    println!(
        "\nmarginal wedge s = 0 over t = {:.3}: ln(E/E_0) = {final_ratio:.4}, fitted rate \
         {drift:.3e} against the s = 1/6 prediction {reference:.5}",
        FOLDS / reference
    );
    assert!(
        final_ratio < 1.0,
        "the marginal column's perturbation energy grew by e^{final_ratio:.3} over a clock \
         in which the s = 1/6 column grows by e^{FOLDS}; a marginally stable atmosphere \
         carries no unstable mode"
    );
    // the mode is neutral, so what is left acting on it is the scheme's dissipation, and
    // the record decays at 9.7e-3 -- the same size as the 4.3e-3 that dissipation takes off
    // the s = 1/6 mode's growth on this grid. growth is the failure mode: any positive rate
    // at s = 0 would mean the measurement tracks the seed rather than the stratification. a
    // decay far past the dissipation scale would mean the mode is being destroyed instead
    // of merely damped, so the window is bounded on that side too.
    assert!(
        drift < 0.02 * reference,
        "the marginal column grows at {drift:.3e}; a vanishing schwarzschild discriminant \
         carries no unstable mode, so the measurement is responding to the seed rather \
         than to the entropy slope"
    );
    assert!(
        drift > -0.10 * reference,
        "the marginal column decays at {drift:.3e}, an order past the dissipation the same \
         grid takes off the unstable modes; the seeded mode is being destroyed rather than \
         held"
    );
    println!("elapsed {:.1} s", clock.elapsed().as_secs_f64());
}

