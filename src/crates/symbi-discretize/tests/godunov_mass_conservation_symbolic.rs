// =============================================================================
// godunov_mass_conservation_symbolic.rs
//
// the symbolic proof that the cartesian godunov mass update conserves mass exactly,
// by showing the traced flux-divergence is a sum of per-direction discrete
// divergences `G_d[+e_d] - G_d`, which telescope globally. the structural
// counterpart to the numerical sod-conservation integral (1e-9).
//
// finite-volume conservation: a shared face flux enters cell i (as its high face)
// and cell i+1 (as its low face) with equal-and-opposite signed coefficient, so
// summing over cells the interior faces cancel and only the domain boundary remains.
// the godunov mass update is `rho_new = rho - dt*div(mass_flux)`, and for cartesian
//   div = sum_d (F_d[+e_d] - F_d[0]) / dx_d.
// so the flux part is exactly `sum_d (G_d[+e_d] - G_d)` with `G_d = (-dt/dx_d) F_d[0]`
// — a discrete divergence per direction. asserting the extracted flux form equals
// that telescoping form proves conservation by construction, for an arbitrary flux field.
//
// (curvilinear conservation reduces to the shared-face area consistency
// `area_hi(c) == area_lo(c+e)`; proving it symbolically needs the coeff ring extended
// for the angular measure's cos/pi factors — the analog of the sin extension that
// A1-sph(b) added. deferred.)
// =============================================================================

use symbi_discretize::{Coords, Spacing, godunov_mass_gv};
use symbi_ir::proof::{LinFormR, Poly, RatFun};

const NDIM: usize = 2;
// the per-direction mass-flux reads + the in-place conserved density `rho`.
const FIELDS: &[&str] = &["mass_flux_0", "mass_flux_1", "rho"];
const SCALARS: &[&str] = &["dt", "dx_0", "dx_1"];

// strip the old conserved `rho` leaf: it reproduces the unchanged conserved field,
// not part of the "the flux update telescopes" property.
fn flux_only(mut lf: LinFormR) -> LinFormR {
    lf.terms.retain(|(key, _), _| key != "rho");
    lf
}

// the per-direction face-flux function `G_d = (-dt/dx_d) * mass_flux_d`, as a
// single-term linear form at the cell's low face (offset 0).
fn g_face(d: usize) -> LinFormR {
    let key = format!("mass_flux_{d}");
    LinFormR::single_var((key, vec![0; NDIM]), "dt").scale_rat(&RatFun::new(
        Poly::constant(-1),
        Poly::var(&format!("dx_{d}")),
    ))
}

#[test]
fn godunov_mass_conservation_symbolic() {
    let (kernel, writes) = godunov_mass_gv(
        Coords::Cartesian,
        &[Spacing::Uniform; NDIM],
        &[0, 1],
        NDIM as u8,
    );
    assert_eq!(writes.len(), 1, "mass builder must write exactly rho_new");
    let flux = flux_only(LinFormR::extract_rat(
        &kernel.graph,
        writes[0].2,
        FIELDS,
        SCALARS,
    ));
    assert!(
        !flux.is_zero(),
        "flux update is empty — extractor saw no flux reads"
    );

    // the telescoping target: sum_d (G_d[+e_d] - G_d), a discrete divergence per direction.
    let mut telescoping = LinFormR::default();
    for d in 0..NDIM {
        let g = g_face(d);
        let mut e_d = [0i32; NDIM];
        e_d[d] = 1;
        telescoping.add(&g.shifted(&e_d));
        telescoping.add(&g.neg_form());
    }

    // the proof: the traced flux update equals that sum of discrete divergences -> conserves.
    let mut residual = flux;
    residual.add(&telescoping.neg_form());
    assert!(
        residual.is_zero(),
        "godunov mass flux is NOT a discrete divergence (does not conserve) — residual:\n{:#?}",
        residual.residual()
    );
}

// negative control: a non-telescoping pair (a flux that enters two cells with the
// same sign) leaves a residual, so the checker has real content.
#[test]
fn conservation_symbolic_detects_nonconservative() {
    let mut lf = LinFormR::default();
    lf.add(&LinFormR::single_var(
        ("mass_flux_0".into(), vec![1, 0]),
        "dt",
    ));
    lf.add(&LinFormR::single_var(
        ("mass_flux_0".into(), vec![0, 0]),
        "dt",
    )); // same sign: not -dt
    assert!(
        !lf.is_zero(),
        "a same-sign (non-conservative) flux pair must NOT cancel"
    );
    assert_eq!(lf.residual().len(), 2);
}
