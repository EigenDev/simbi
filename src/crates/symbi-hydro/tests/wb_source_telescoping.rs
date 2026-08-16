// =============================================================================
// wb_source_telescoping.rs
//
// the three-way telescoping identity behind the chart-generic well-balanced
// gravity source. on a gridded axis the momentum update at rest is
//
//     - [A_hi p_face,hi - A_lo p_face,lo] / V      (pressure flux divergence)
//     + p_cell (A_hi - A_lo) / V                   (geometric pressure source)
//     + S_grav,
//
// and on a discretely balanced isentrope -- where the balanced reconstruction
// makes every face pressure the equilibrium pressure p_eq(phi_face) and the
// cell pressure is p_eq(phi_c) -- the sum vanishes identically when
//
//     S_grav = [A_hi (p_eq(phi_hi) - p_eq(phi_c)) - A_lo (p_eq(phi_lo) - p_eq(phi_c))] / V.
//
// checked per chart with the chart's own exact finite-volume factors (cartesian
// A = 1, V = dr; cylindrical A = r, V = (r_hi^2 - r_lo^2)/2; spherical A = r^2,
// V = (r_hi^3 - r_lo^3)/3) and the same `LocalEquilibrium` profile the kernels
// trace. the face pressures come from the neighbor cell's equilibrium, so the
// residual also carries the reconstruction-consistency statement (both anchors
// give the face the same value to roundoff) rather than assuming it. positive
// control: the analytic `rho g` source leaves a truncation-scale residual on
// the same column, so a vanishing sum is a property of the pairing, not of a
// quiet setup.
//
// run: cargo test -p symbi-hydro --test wb_source_telescoping -- --nocapture
// =============================================================================

use symbi_hydro::hydrostatic::LocalEquilibrium;

const GAMMA: f64 = 5.0 / 3.0;
const GM: f64 = 3.0;
const K0: f64 = 0.7;

fn phi(r: f64) -> f64 {
    -GM / r
}

/// gravitational acceleration -dphi/dr, for the analytic-source positive control.
fn grav(r: f64) -> f64 {
    -GM / (r * r)
}

/// the exact isentrope through `rho = rho0` at `r = r0`: bernoulli invariant
/// `gamma K0/(gamma-1) rho^(gamma-1) + phi = const`.
fn isentrope(r: f64) -> (f64, f64) {
    let a = (GAMMA - 1.0) / (GAMMA * K0);
    let c = 1.0 / a + phi(2.0);
    let rho = (a * (c - phi(r))).powf(1.0 / (GAMMA - 1.0));
    (rho, K0 * rho.powf(GAMMA))
}

/// per-chart face area and exact cell volume on [r_lo, r_hi] (per unit transverse
/// measure; the common angular factor cancels in every A/V ratio).
#[derive(Clone, Copy)]
enum Chart {
    Cartesian,
    Cylindrical,
    Spherical,
}

impl Chart {
    fn area(self, r: f64) -> f64 {
        match self {
            Chart::Cartesian => 1.0,
            Chart::Cylindrical => r,
            Chart::Spherical => r * r,
        }
    }
    fn volume(self, r_lo: f64, r_hi: f64) -> f64 {
        match self {
            Chart::Cartesian => r_hi - r_lo,
            Chart::Cylindrical => 0.5 * (r_hi * r_hi - r_lo * r_lo),
            Chart::Spherical => (r_hi * r_hi * r_hi - r_lo * r_lo * r_lo) / 3.0,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Chart::Cartesian => "cartesian",
            Chart::Cylindrical => "cylindrical",
            Chart::Spherical => "spherical",
        }
    }
}

#[test]
fn the_wb_source_telescopes_against_divergence_and_geometric_source_per_chart() {
    let n = 64usize;
    let h = 1.0 / n as f64;
    for chart in [Chart::Cartesian, Chart::Cylindrical, Chart::Spherical] {
        // the radial column r in [1, 2]; anchors at the arithmetic face midpoints,
        // the kernel ladder's cell positions on a uniform axis.
        let face = |ii: usize| 1.0 + ii as f64 * h;
        let anchor = |ii: usize| 0.5 * (face(ii) + face(ii + 1));

        let mut worst_wb = 0.0_f64;
        let mut worst_plain = 0.0_f64;
        let mut scale = 0.0_f64;
        for ii in 1..n - 1 {
            let (r_lo, r_hi) = (face(ii), face(ii + 1));
            let r_c = anchor(ii);
            let (a_lo, a_hi) = (chart.area(r_lo), chart.area(r_hi));
            let inv_v = 1.0 / chart.volume(r_lo, r_hi);

            let (rho_c, p_c) = isentrope(r_c);
            let eq = LocalEquilibrium::through(rho_c, p_c, phi(r_c), GAMMA);

            // face pressures as the balanced scheme produces them: the neighbor
            // cell's equilibrium evaluated at the shared face, so the identity is
            // exercised across the reconstruction-consistency seam rather than
            // against the cell's own profile alone.
            let eq_dn = {
                let (rho, p) = isentrope(anchor(ii - 1));
                LocalEquilibrium::through(rho, p, phi(anchor(ii - 1)), GAMMA)
            };
            let eq_up = {
                let (rho, p) = isentrope(anchor(ii + 1));
                LocalEquilibrium::through(rho, p, phi(anchor(ii + 1)), GAMMA)
            };
            let p_face_lo = eq_dn.pressure_at(phi(r_lo));
            let p_face_hi = eq_up.pressure_at(phi(r_hi));

            let flux_div = (a_hi * p_face_hi - a_lo * p_face_lo) * inv_v;
            let geo = p_c * (a_hi - a_lo) * inv_v;
            // p_eq(phi_c) is p_c bit-exactly (the profile's anchor point), the same
            // spelling the traced source uses.
            let s_wb = (a_hi * (eq.pressure_at(phi(r_hi)) - p_c)
                - a_lo * (eq.pressure_at(phi(r_lo)) - p_c))
                * inv_v;

            worst_wb = worst_wb.max((-flux_div + geo + s_wb).abs());
            // positive control: the analytic source rho*g on the same column leaves
            // the truncation-scale flux/source mismatch the wb source removes.
            worst_plain = worst_plain.max((-flux_div + geo + rho_c * grav(r_c)).abs());
            // the roundoff scale of the cancellation: the individual terms are O(p/h).
            scale = scale.max(flux_div.abs());
        }
        let rel = worst_wb / scale;
        println!(
            "{}: |flux_div - geo - S_wb| max {worst_wb:.3e} (rel {rel:.3e}), \
             analytic-source residual {worst_plain:.3e}",
            chart.name()
        );
        // measured mismatch: 1.0e-3 (cartesian), 3.3e-4 (cylindrical), 9.3e-5
        // (spherical) -- truncation-scale on every chart, eight orders above the
        // roundoff the identity must reach, so the floor sits well under the
        // weakest chart while still refusing a quiet setup.
        assert!(
            worst_plain > 1.0e-5,
            "{}: positive control failed -- the analytic rho*g source leaves only \
             {worst_plain:.3e} against the discrete pressure gradient, so this column \
             does not exercise the mismatch and the identity is vacuous",
            chart.name()
        );
        // the three terms are each O(p/h) and cancel algebraically; the float residual
        // is roundoff of that scale. measured max 2.2e-13 relative (spherical, the
        // largest area ratio); the bound sits more than an order above the measurement
        // and eight below the positive control.
        assert!(
            rel < 1.0e-11,
            "{}: the telescoping residual is {rel:.3e} of the flux-divergence scale; \
             the wb source is not the exact remainder of the divergence/geometric pair",
            chart.name()
        );
    }
}

#[test]
fn the_wb_source_telescopes_on_a_log_spaced_axis() {
    // the identity is position-blind: it holds at whatever face/anchor positions the
    // three terms share. what a log axis adds is the anchor convention -- the cell
    // position is the map's own center, the geometric mean sqrt(r_lo r_hi), not the
    // arithmetic midpoint -- and the graded faces make every A/V ratio cell-dependent.
    // the arithmetic-anchor control below shows the identity is anchor-consistent
    // rather than anchor-forgiving: telescoping needs the same anchor in all three
    // terms, and the seeded column supplies the map's.
    let n = 64usize;
    let slope = 2.0_f64.log10() / n as f64;
    for chart in [Chart::Cartesian, Chart::Cylindrical, Chart::Spherical] {
        // the radial column r in [1, 2] on log faces; anchors at the geometric mean,
        // the kernel ladder's cell position on a log axis.
        let face = |ii: usize| 10.0_f64.powf(ii as f64 * slope);
        let anchor = |ii: i64| (face(ii as usize) * face(ii as usize + 1)).sqrt();

        // the graded premise: arithmetic and geometric centers must genuinely separate,
        // or this arm degenerates to the uniform test.
        let sep = (0..n)
            .map(|ii| (0.5 * (face(ii) + face(ii + 1)) - anchor(ii as i64)).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            sep > 1.0e-7,
            "{}: log faces separate the two center definitions by only {sep:.3e}",
            chart.name()
        );

        let mut worst_wb = 0.0_f64;
        let mut worst_plain = 0.0_f64;
        let mut scale = 0.0_f64;
        for ii in 1..n - 1 {
            let (r_lo, r_hi) = (face(ii), face(ii + 1));
            let r_c = anchor(ii as i64);
            let (a_lo, a_hi) = (chart.area(r_lo), chart.area(r_hi));
            let inv_v = 1.0 / chart.volume(r_lo, r_hi);

            let (rho_c, p_c) = isentrope(r_c);
            let eq = LocalEquilibrium::through(rho_c, p_c, phi(r_c), GAMMA);

            // face pressures from the neighbor cells' equilibria, as in the uniform
            // test: the identity is exercised across the reconstruction-consistency
            // seam.
            let eq_dn = {
                let (rho, p) = isentrope(anchor(ii as i64 - 1));
                LocalEquilibrium::through(rho, p, phi(anchor(ii as i64 - 1)), GAMMA)
            };
            let eq_up = {
                let (rho, p) = isentrope(anchor(ii as i64 + 1));
                LocalEquilibrium::through(rho, p, phi(anchor(ii as i64 + 1)), GAMMA)
            };
            let p_face_lo = eq_dn.pressure_at(phi(r_lo));
            let p_face_hi = eq_up.pressure_at(phi(r_hi));

            let flux_div = (a_hi * p_face_hi - a_lo * p_face_lo) * inv_v;
            let geo = p_c * (a_hi - a_lo) * inv_v;
            let s_wb = (a_hi * (eq.pressure_at(phi(r_hi)) - p_c)
                - a_lo * (eq.pressure_at(phi(r_lo)) - p_c))
                * inv_v;

            worst_wb = worst_wb.max((-flux_div + geo + s_wb).abs());
            worst_plain = worst_plain.max((-flux_div + geo + rho_c * grav(r_c)).abs());
            scale = scale.max(flux_div.abs());
        }
        let rel = worst_wb / scale;
        println!(
            "log {}: |flux_div - geo - S_wb| max {worst_wb:.3e} (rel {rel:.3e}), \
             analytic-source residual {worst_plain:.3e}",
            chart.name()
        );
        // measured mismatch: 1.7e-4 (cartesian), 8.3e-5 (cylindrical), 1.9e-4
        // (spherical) -- truncation-scale, so the log column exercises the same
        // flux/source mismatch the uniform one does.
        assert!(
            worst_plain > 1.0e-5,
            "{}: positive control failed -- the analytic rho*g source leaves only \
             {worst_plain:.3e}; this log column does not exercise the mismatch",
            chart.name()
        );
        // measured max 1.5e-13 relative (spherical); same roundoff-of-O(p/h)
        // argument as the uniform arm, same bound.
        assert!(
            rel < 1.0e-11,
            "{}: the log-axis telescoping residual is {rel:.3e} of the flux-divergence \
             scale; the wb source is not the exact remainder on a graded axis",
            chart.name()
        );
    }
}

#[test]
fn the_general_form_reduces_to_the_cartesian_pressure_difference() {
    // on the cartesian chart A_hi = A_lo = 1 and V = h, so the area-weighted form
    // collapses to the landed `(p_eq(phi_hi) - p_eq(phi_lo))/h` spelling. value
    // equality to roundoff is what licenses keeping the two spellings (the baked
    // cartesian kernel keeps its graph byte-for-byte).
    let n = 64usize;
    let h = 1.0 / n as f64;
    let face = |ii: usize| 1.0 + ii as f64 * h;
    let anchor = |ii: usize| 0.5 * (face(ii) + face(ii + 1));
    for ii in 1..n - 1 {
        let r_c = anchor(ii);
        let (rho_c, p_c) = isentrope(r_c);
        let eq = LocalEquilibrium::through(rho_c, p_c, phi(r_c), GAMMA);
        let (p_lo, p_hi) = (eq.pressure_at(phi(face(ii))), eq.pressure_at(phi(face(ii + 1))));
        let general = ((p_hi - p_c) - (p_lo - p_c)) / h;
        let cartesian = (p_hi - p_lo) / h;
        let ulp_scale = (cartesian.abs() + p_c / h) * f64::EPSILON;
        assert!(
            (general - cartesian).abs() <= 4.0 * ulp_scale,
            "cell {ii}: the general form ({general:.17e}) drifts from the cartesian \
             spelling ({cartesian:.17e}) beyond re-association roundoff"
        );
    }
}
