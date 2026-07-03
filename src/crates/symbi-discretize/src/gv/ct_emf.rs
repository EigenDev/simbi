// =============================================================================
// ct_emf.rs
//
// rmhd constrained-transport stack: staggered curl, edge-emf (uct hll/hllc/hlld), face->cell b, cell-b predictors.
// =============================================================================

use super::*;


/// the Gardiner & Stone CT-contact edge EMF (the SOFT-SIGN blend), carrier-generic at S=Gv.
/// a pointwise function of the 4 face EMFs, 4 cell-corner
/// EMFs, and 4 density fluxes: `s = f/(|f|+eps)`; `0.5*((a+b) + s*(a-b))`, transitions
/// continuously through f=0 (= a hard 3-way sign in the |f|>>eps limit). div(B) unaffected.
fn ct_contact_emf_gv(face_e: [Gv; 4], cell_e: [Gv; 4], dflux: [Gv; 4]) -> Gv {
    let [en, es, ee, ew] = face_e;
    let [ene, enw, ese, esw] = cell_e;
    let [fnf, fs, fe, fw] = dflux;
    let two = Gv::from_f64(2.0);
    let eps = Gv::from_f64(1.0e-12);
    let eavg = Gv::from_f64(0.25) * (es + en + ew + ee);
    let soft = |f: Gv, a: Gv, b: Gv| {
        let s = f / (f.abs() + eps);
        Gv::from_f64(0.5) * ((a + b) + s * (a - b))
    };
    let de_jl = soft(fw, two * (es - esw), two * (en - enw)); // west
    let de_jr = soft(fe, two * (ese - es), two * (ene - en)); // east
    let de_kl = soft(fs, two * (ew - esw), two * (ee - ese)); // south
    let de_kr = soft(fnf, two * (enw - ew), two * (ene - ee)); // north
    eavg + Gv::from_f64(0.125) * (de_jl - de_jr + de_kl - de_kr)
}


/// the orthogonal-curl scale-factor weights for the
/// curvilinear induction curl (h_p edge weights + the 1/(h_p1c h_p2c) face-center prefactor + the
/// transverse inverse widths). all Gv, from the cell index via gv_axis_face_at / gv_scale_factor.
struct CtCurlMetricGv {
    h1_here: Gv,
    h1_p2: Gv,
    h2_here: Gv,
    h2_p1: Gv,
    inv_pref: Gv,
    inv_dx_p1: Gv,
    inv_dx_p2: Gv,
}


fn ct_curl_metric_gv(coords: Coords, spacing: &[Spacing], dir: usize) -> CtCurlMetricGv {
    let p1 = (dir + 1) % 3;
    let p2 = (dir + 2) % 3;
    let pos_at = |off: [i64; 3]| -> Vec<Gv> {
        (0..3).map(|ax| gv_axis_face_at(ax, spacing[ax], off[ax])).collect()
    };
    let pos_here = pos_at([0, 0, 0]);
    let mut op2 = [0, 0, 0];
    op2[p2] = 1;
    let pos_p2 = pos_at(op2);
    let mut op1 = [0, 0, 0];
    op1[p1] = 1;
    let pos_p1 = pos_at(op1);
    let h1_here = gv_scale_factor(coords, p1, &pos_here);
    let h1_p2 = gv_scale_factor(coords, p1, &pos_p2);
    let h2_here = gv_scale_factor(coords, p2, &pos_here);
    let h2_p1 = gv_scale_factor(coords, p2, &pos_p1);

    let half = Gv::from_f64(0.5);
    let center: Vec<Gv> = (0..3)
        .map(|ax| {
            if ax == dir {
                gv_axis_face_at(ax, spacing[ax], 0)
            } else {
                (gv_axis_face_at(ax, spacing[ax], 0) + gv_axis_face_at(ax, spacing[ax], 1)) * half
            }
        })
        .collect();
    let inv_pref = Gv::ONE / (gv_scale_factor(coords, p1, &center) * gv_scale_factor(coords, p2, &center));
    let inv_dx_p1 = Gv::ONE / (gv_axis_face_at(p1, spacing[p1], 1) - gv_axis_face_at(p1, spacing[p1], 0));
    let inv_dx_p2 = Gv::ONE / (gv_axis_face_at(p2, spacing[p2], 1) - gv_axis_face_at(p2, spacing[p2], 0));
    CtCurlMetricGv { h1_here, h1_p2, h2_here, h2_p1, inv_pref, inv_dx_p1, inv_dx_p2 }
}


/// the 2.5D in-plane CT curl B-update along ONE face axis `dir` from the single
/// out-of-plane corner EMF Ez (cartesian), in-place on `b` (bface[dir]). PER-DIR
/// (mirroring the 3D `rmhd_ct_curl_3d_dir`) because bx lives on x-faces and by on
/// y-faces — distinct staggered domains, each updated over its own face domain so
/// the high boundary face is covered. dir=0: dBx/dt = -dEz/dy -> b -= dt*idy*(Ez[j+1]-Ez);
/// dir=1: dBy/dt = +dEz/dx -> b += dt*idx*(Ez[i+1]-Ez). div(B)=0 preserved.
/// (the out-of-plane Bz is NOT CT-evolved — it rides the induction-flux divergence.)
pub fn rmhd_ct_curl_2d_dir_gv(dir: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let b = Gv::field("b", "b");
    let ez = Gv::field("ez", "ez");
    let dt = Gv::scalar("dt");
    let b_new = if dir == 0 {
        let idy = Gv::scalar("idy");
        let ez_jp = gv_field_at("ez", "ez", 2, &[0, 1]);
        b - dt * idy * (ez_jp - ez)
    } else {
        let idx = Gv::scalar("idx");
        let ez_ip = gv_field_at("ez", "ez", 2, &[1, 0]);
        b + dt * idx * (ez_ip - ez)
    };
    (end_trace(), vec![("b_new".to_string(), "b".into(), b_new.node())])
}


/// the 2.5D cylindrical r-z (axisymmetric) CT curl from the single out-of-plane edge EMF
/// E_phi (efield[0]), in-place on `b` (bface[dir]). DERIVED from the 3D cyl curl restricted
/// to E_phi with d/dphi = 0 (verified to reproduce the 3D-cyl ct_curl_metric formula):
///   dir=0 (B_r, r-face):  dB_r/dt = +d_z E_phi            (z = grid axis 1; flat, no metric)
///   dir=1 (B_z, z-face):  dB_z/dt = -(1/r) d_r(r E_phi)   (r = grid axis 0; cylindrical metric)
/// r is computed per-cell from gv_axis_face_at(0, ..) (the geom scalars x_lo_0/dx_0). E_phi is
/// the corner field at offsets [0,0]/[+grid]. div(B)=0 preserved by the discrete d∘d.
pub fn rmhd_ct_curl_cyl_rz_gv(dir: usize, spacing: &[Spacing]) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let b = Gv::field("b", "b");
    let ez = Gv::field("ez", "ez"); // the out-of-plane corner EMF E_phi
    let dt = Gv::scalar("dt");
    // POSITIONAL scalar ABI: the runtime curl dispatch pushes `[dt] ++ push_curvilinear_geom`
    // = [dt, x_lo_0, dx_0, x_lo_1, dx_1] (all grid axes, every dir). scalar_params is fixed at
    // registration order with NO liveness pruning, so BOTH dir branches must register the full
    // geom set in that order — else dir=0 (which only touches axis 1) would bind x_lo_1 to the
    // runtime's x_lo_0 slot. this prelude pins the canonical order; the body's reads dedupe in.
    for ax in 0..2 {
        let _ = gv_axis_face_at(ax, spacing[ax], 0);
        let _ = gv_axis_face_at(ax, spacing[ax], 1);
    }
    let b_new = if dir == 0 {
        // dB_r/dt = +d_z E_phi : finite difference along grid axis 1 (z). no metric (h_z = 1).
        let inv_dz = Gv::ONE / (gv_axis_face_at(1, spacing[1], 1) - gv_axis_face_at(1, spacing[1], 0));
        let ez_zp = gv_field_at("ez", "ez", 2, &[0, 1]);
        b + dt * inv_dz * (ez_zp - ez)
    } else {
        // dB_z/dt = -(1/r_c) d_r(r E_phi) : the cylindrical metric on the radial derivative.
        // r at the cell's two r-faces (= the corner radii bounding this z-face), cell-center r_c.
        // minus because (curl E)_z = +(1/r) d_r(r E_phi) and dB/dt = -curl(E) — OPPOSITE sign to
        // the spherical-poloidal B_theta update (curl_theta carries its own minus). the plus form
        // leaks div(B) secularly (d/dt div(B) != 0); the rotor div(B) blows to O(1) in one step.
        let inv_dr = Gv::ONE / (gv_axis_face_at(0, spacing[0], 1) - gv_axis_face_at(0, spacing[0], 0));
        let r_lo = gv_axis_face_at(0, spacing[0], 0);
        let r_hi = gv_axis_face_at(0, spacing[0], 1);
        let r_c = (r_lo + r_hi) * Gv::from_f64(0.5);
        let ez_rp = gv_field_at("ez", "ez", 2, &[1, 0]);
        b - dt * (Gv::ONE / r_c) * inv_dr * (r_hi * ez_rp - r_lo * ez)
    };
    (end_trace(), vec![("b_new".to_string(), "b".into(), b_new.node())])
}


/// the 2.5D cylindrical r-phi DISK CT curl from the single out-of-plane edge EMF E_z
/// (efield[0]), in-place on `b` (bface[dir]). DERIVED from the cyl curl restricted to E_z with
/// d/dz = 0 (verified to preserve the staggered cyl div(B) = (1/r)d_r(r B_r) + (1/r)d_phi B_phi):
///   dir=0 (B_r, r-face):   dB_r/dt   = -(1/r) d_phi E_z   (phi = grid axis 1; 1/r metric, r = the r-face radius)
///   dir=1 (B_phi, phi-face): dB_phi/dt = +d_r E_z         (r = grid axis 0; flat, NO metric — mirror of r-z)
/// r is the r-FACE radius (where B_r lives) via gv_axis_face_at(0, .., 0). E_z is the corner field
/// at offsets [0,0]/[+grid]. div(B)=0 preserved by the discrete d∘d (mixed partials cancel).
pub fn rmhd_ct_curl_cyl_rphi_gv(dir: usize, spacing: &[Spacing]) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let b = Gv::field("b", "b");
    let ez = Gv::field("ez", "ez"); // the out-of-plane corner EMF E_z
    let dt = Gv::scalar("dt");
    // POSITIONAL scalar ABI: the runtime curl dispatch pushes [dt, x_lo_0, dx_0, x_lo_1, dx_1]
    // every dir (see rmhd_ct_curl_cyl_rz_gv). pin the full geom set in canonical order up front.
    for ax in 0..2 {
        let _ = gv_axis_face_at(ax, spacing[ax], 0);
        let _ = gv_axis_face_at(ax, spacing[ax], 1);
    }
    let b_new = if dir == 0 {
        // dB_r/dt = -(1/r) d_phi E_z : the 1/r metric on the phi-derivative (grid axis 1). r is
        // the r-FACE radius (B_r lives on the r-face = the cell's low r-face, offset 0).
        let r_face = gv_axis_face_at(0, spacing[0], 0);
        let inv_dphi = Gv::ONE / (gv_axis_face_at(1, spacing[1], 1) - gv_axis_face_at(1, spacing[1], 0));
        let ez_phip = gv_field_at("ez", "ez", 2, &[0, 1]);
        b - dt * (Gv::ONE / r_face) * inv_dphi * (ez_phip - ez)
    } else {
        // dB_phi/dt = +d_r E_z : finite difference along grid axis 0 (r). NO metric (the phi-comp
        // of the cyl curl is metric-free; the discrete d∘d still cancels — proven).
        let inv_dr = Gv::ONE / (gv_axis_face_at(0, spacing[0], 1) - gv_axis_face_at(0, spacing[0], 0));
        let ez_rp = gv_field_at("ez", "ez", 2, &[1, 0]);
        b + dt * inv_dr * (ez_rp - ez)
    };
    (end_trace(), vec![("b_new".to_string(), "b".into(), b_new.node())])
}


/// the 2.5D SPHERICAL (r-theta plane, out-of-plane phi) CT curl from the single corner EMF
/// E_phi (efield[0]), in-place on `b` (bface[dir]). Faraday dB/dt = -curl E with E = E_phi phi-hat
/// (axisymmetric) gives the spherical-metric in-plane update:
///   dir=0 (B_r,   r-face):     dB_r/dt   = -(1/(r_f sin th_c)) d_th(sin th * E_phi)   (th = grid axis 1)
///   dir=1 (B_th, theta-face):  dB_th/dt  = +(1/r_c) d_r(r * E_phi)                     (r  = grid axis 0)
/// r_f is the r-FACE radius (where B_r lives); r_c / th_c are the staggered cell centers. mirrors
/// `rmhd_ct_curl_cyl_rz_gv` with the added sin(theta) area weight on the B_r update (and the
/// opposite B_theta sign vs the cylinder's B_z). div(B)=0 preservation for a nontrivial POLOIDAL
/// (B_r, B_theta) field is pinned by tests/rmhd_ct_curl_2d_sph_poloidal_divb.rs: B = curl(A_phi)
/// through this kernel, area-weighted div machine-zero before AND after a curl(E_phi) step.
pub fn rmhd_ct_curl_2d_sph_gv(dir: usize, spacing: &[Spacing]) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let b = Gv::field("b", "b");
    let ez = Gv::field("ez", "ez"); // the out-of-plane corner EMF E_phi
    let dt = Gv::scalar("dt");
    // POSITIONAL scalar ABI: the runtime curl dispatch pushes [dt, x_lo_0, dx_0, x_lo_1, dx_1]
    // every dir (see rmhd_ct_curl_cyl_rz_gv). pin the full geom set in canonical order up front.
    for ax in 0..2 {
        let _ = gv_axis_face_at(ax, spacing[ax], 0);
        let _ = gv_axis_face_at(ax, spacing[ax], 1);
    }
    let half = Gv::from_f64(0.5);
    let b_new = if dir == 0 {
        // dB_r/dt = -(1/(r_f sin th_c)) d_th(sin th E_phi). r_f = the low r-face (B_r lives there);
        // th_lo/th_hi are the corner thetas bounding this r-face, th_c the cell-center theta.
        let r_f = gv_axis_face_at(0, spacing[0], 0);
        let th_lo = gv_axis_face_at(1, spacing[1], 0);
        let th_hi = gv_axis_face_at(1, spacing[1], 1);
        let th_c = (th_lo + th_hi) * half;
        let inv_dth = Gv::ONE / (th_hi - th_lo);
        let ez_thp = gv_field_at("ez", "ez", 2, &[0, 1]);
        b - dt * (Gv::ONE / (r_f * th_c.sin())) * inv_dth * (th_hi.sin() * ez_thp - th_lo.sin() * ez)
    } else {
        // dB_th/dt = +(1/r_c) d_r(r E_phi). r_lo/r_hi are the corner radii bounding this theta-face,
        // r_c the cell-center radius (opposite sign to the cylinder's B_z update).
        let r_lo = gv_axis_face_at(0, spacing[0], 0);
        let r_hi = gv_axis_face_at(0, spacing[0], 1);
        let r_c = (r_lo + r_hi) * half;
        let inv_dr = Gv::ONE / (r_hi - r_lo);
        let ez_rp = gv_field_at("ez", "ez", 2, &[1, 0]);
        b + dt * (Gv::ONE / r_c) * inv_dr * (r_hi * ez_rp - r_lo * ez)
    };
    (end_trace(), vec![("b_new".to_string(), "b".into(), b_new.node())])
}


/// the 2.5D CURVED-SPACETIME (r-theta plane) CT curl from the single DENSITIZED corner EMF
/// `Etilde_phi` (efield[0]), in-place on the PHYSICAL `b` (bface[dir]). the update evolves the
/// densitized field Btilde^i = sqrt(gamma) B^i with the COORDINATE curl — the form whose
/// discrete divergence d_i Btilde^i telescopes to zero for ANY per-face constant weights —
/// then divides back by this face's own weight `w = sqrt(gamma)(face center) x coordinate
/// length` so the stored value stays the physical B every consumer reads:
///   dir=0 (B_r,  r-face):     B_r  -= dt (Etilde(th_hi) - Etilde(th_lo)) / (sqrtg(r_f, th_c) dth)
///   dir=1 (B_th, theta-face): B_th += dt (Etilde(r_hi)  - Etilde(r_lo))  / (sqrtg(r_c, th_f) dr)
/// the signs mirror the flat spherical kernel (the same edge orientation). sqrt(gamma) comes
/// from the metric trait at the face center, so one builder serves every KS-family chart.
/// div preservation for a nontrivial poloidal field is pinned by
/// tests/rmhd_ct_curl_2d_sph_gr_divb.rs (the w-weighted divergence, machine-zero).
pub fn rmhd_ct_curl_2d_sph_gr_gv(
    dir: usize,
    spacetime: Spacetime,
    spacing: &[Spacing],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    use symbi_geometry::{KerrKS, Metric, Schwarzschild, SchwarzschildKS};
    begin_trace();
    let b = Gv::field("b", "b");
    let ez = Gv::field("ez", "ez"); // the out-of-plane DENSITIZED corner EMF Etilde_phi
    let dt = Gv::scalar("dt");
    // positional scalar ABI mirror of the flat curl: pin [x_lo_0, dx_0, x_lo_1, dx_1] up front.
    for ax in 0..2 {
        let _ = gv_axis_face_at(ax, spacing[ax], 0);
        let _ = gv_axis_face_at(ax, spacing[ax], 1);
    }
    let half = Gv::from_f64(0.5);
    let mass = Gv::scalar("schwarzschild_mass");
    let sqrtg = |r: Gv, th: Gv| -> Gv {
        let x = Tensor::<Gv, 3>::new([r, th, Gv::ZERO]);
        match spacetime {
            Spacetime::Schwarzschild => {
                <Schwarzschild<Gv> as Metric<Gv, 3>>::sqrt_det_gamma(&Schwarzschild { mass }, x)
            }
            Spacetime::KerrSchild => {
                <SchwarzschildKS<Gv> as Metric<Gv, 3>>::sqrt_det_gamma(&SchwarzschildKS { mass }, x)
            }
            Spacetime::Kerr => <KerrKS<Gv> as Metric<Gv, 3>>::sqrt_det_gamma(
                &KerrKS { mass, spin: Gv::scalar("kerr_spin") },
                x,
            ),
            Spacetime::Minkowski => {
                unreachable!("the GR curl is baked only for a curved spacetime")
            }
        }
    };
    let b_new = if dir == 0 {
        // B_r lives on the low r-face; th_lo/th_hi are the bounding corner thetas.
        let r_f = gv_axis_face_at(0, spacing[0], 0);
        let th_lo = gv_axis_face_at(1, spacing[1], 0);
        let th_hi = gv_axis_face_at(1, spacing[1], 1);
        let th_c = (th_lo + th_hi) * half;
        let w = sqrtg(r_f, th_c) * (th_hi - th_lo);
        let ez_thp = gv_field_at("ez", "ez", 2, &[0, 1]);
        b - dt * (ez_thp - ez) / w
    } else {
        // B_theta lives on the low theta-face; r_lo/r_hi are the bounding corner radii.
        let r_lo = gv_axis_face_at(0, spacing[0], 0);
        let r_hi = gv_axis_face_at(0, spacing[0], 1);
        let th_f = gv_axis_face_at(1, spacing[1], 0);
        let r_c = (r_lo + r_hi) * half;
        let w = sqrtg(r_c, th_f) * (r_hi - r_lo);
        let ez_rp = gv_field_at("ez", "ez", 2, &[1, 0]);
        b + dt * (ez_rp - ez) / w
    };
    (end_trace(), vec![("b_new".to_string(), "b".into(), b_new.node())])
}


/// the 3D CT curl B-update along face axis `dir` (in-place on `b`), mirror of
/// `rmhd::rmhd_ct_curl_3d_dir`: `B_dir += dt*curl`, `curl = dE_p1/dx_p2 - dE_p2/dx_p1`
/// (cartesian, uniform `id_p1`/`id_p2`) or the orthogonal h-weighted curl (curvilinear, via
/// `ct_curl_metric_gv`). reads e_p1/e_p2 at the cell + `+e_p2`/`+e_p1`. div(B)=0 preserved.
pub fn rmhd_ct_curl_3d_dir_gv(
    coords: Coords,
    spacing: &[Spacing],
    dir: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let p1 = (dir + 1) % 3;
    let p2 = (dir + 2) % 3;
    let cartesian = coords == Coords::Cartesian;
    let b = Gv::field("b", "b");
    let dt = Gv::scalar("dt");
    let ids = cartesian.then(|| (Gv::scalar("id_p1"), Gv::scalar("id_p2")));
    let metric = (!cartesian).then(|| ct_curl_metric_gv(coords, spacing, dir));
    // unit offset on a single axis (the +e_p read).
    let off = |ax: usize| -> [i32; 3] {
        let mut o = [0, 0, 0];
        o[ax] = 1;
        o
    };
    let curl = if let Some(m) = metric {
        // (1/(h_p1c h_p2c)) [ d(h_p1 E_p1)/dx_p2 - d(h_p2 E_p2)/dx_p1 ], h-weighted edge EMFs.
        let de = |key: &str, runtime: &str, ax: usize, w_here: Gv, w_plus: Gv, inv_dx: Gv| {
            let e_h = gv_field_at(key, runtime, 3, &[0, 0, 0]);
            let e_p = gv_field_at(key, runtime, 3, &off(ax));
            (w_plus * e_p - w_here * e_h) * inv_dx
        };
        let de1 = de("e_p1", "e_p1", p2, m.h1_here, m.h1_p2, m.inv_dx_p2);
        let de2 = de("e_p2", "e_p2", p1, m.h2_here, m.h2_p1, m.inv_dx_p1);
        m.inv_pref * (de1 - de2)
    } else {
        let (id_p1, id_p2) = ids.expect("cartesian CT curl needs id scalars");
        let ddx = |key: &str, runtime: &str, ax: usize, inv: Gv| {
            let h = gv_field_at(key, runtime, 3, &[0, 0, 0]);
            let p = gv_field_at(key, runtime, 3, &off(ax));
            inv * (p - h)
        };
        let de1 = ddx("e_p1", "e_p1", p2, id_p2);
        let de2 = ddx("e_p2", "e_p2", p1, id_p1);
        de1 - de2
    };
    let b_new = b + dt * curl;
    (end_trace(), vec![("b_new".to_string(), "b".into(), b_new.node())])
}


/// the ISOTHERMAL CT face->cell B interpolation — `bcell_c = 0.5*(bface_c + bface_c[+e_c])`,
/// WITHOUT the 1/2|B|^2 energy correction (isothermal MHD has no energy to correct). reads
/// bface only, writes bcell only — no nrg, no bcell-old read.
pub fn imhd_bcell_from_bface_gv(ndim: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let half = Gv::from_f64(0.5);
    let off = |ax: usize| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        o[ax] = 1;
        o
    };
    // interpolate the ndim in-plane (face-staggered) components; out-of-plane components
    // (if any) are carried cell-centered and untouched here (2.5D / 1.5D — docs/design/30).
    let bf: Vec<Gv> = (0..ndim).map(|c| Gv::field(&format!("bf_{c}"), &format!("bf_{c}"))).collect();
    let writes = (0..ndim)
        .map(|c| {
            let bcc_n = (bf[c] + gv_field_at(&format!("bf_{c}"), &format!("bf_{c}"), ndim, &off(c))) * half;
            (format!("bc_{c}_new"), format!("bc_{c}").into(), bcc_n.node())
        })
        .collect();
    (end_trace(), writes)
}


/// the CT face->cell B interpolation + magnetic-energy correction, mirror of
/// `rmhd::rmhd_bcell_from_bface`: `bcell_c = 0.5*(bface_c + bface_c[+e_c])`,
/// `nrg += 0.5*(|bcell_new|^2 - |bcell_old|^2)`. in-place on bcell + nrg.
pub fn rmhd_bcell_from_bface_gv(ndim: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let half = Gv::from_f64(0.5);
    let off = |ax: usize| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        o[ax] = 1;
        o
    };
    // field order (positional dispatch): all ndim faces, then all ndim old cells, then nrg.
    let bf: Vec<Gv> = (0..ndim).map(|c| Gv::field(&format!("bf_{c}"), &format!("bf_{c}"))).collect();
    let bc: Vec<Gv> = (0..ndim).map(|c| Gv::field(&format!("bc_{c}"), FieldRef::BCell(c as u8))).collect();
    let nrg = Gv::field("nrg", "nrg");
    // interpolate the ndim in-plane components from their faces; out-of-plane components
    // (Bz in 2.5D) are untouched here, and their |B|^2 term cancels in the energy diff.
    let bc_n: Vec<Gv> = (0..ndim)
        .map(|c| (bf[c] + gv_field_at(&format!("bf_{c}"), &format!("bf_{c}"), ndim, &off(c))) * half)
        .collect();
    let sumsq = |v: &[Gv]| v.iter().fold(Gv::ZERO, |a, &x| a + x * x);
    let nrg_n = nrg + half * (sumsq(&bc_n) - sumsq(&bc));
    let mut writes: Vec<(String, FieldBind, NodeId)> = (0..ndim)
        .map(|c| (format!("bc_{c}_new"), format!("bc_{c}").into(), bc_n[c].node()))
        .collect();
    writes.push(("nrg_new".to_string(), "nrg".into(), nrg_n.node()));
    (end_trace(), writes)
}


/// the CT edge EMF along edge axis `dir`, mirror of `rmhd::rmhd_edge_emf`: gather the 12
/// contact-formula inputs by integer-offset `load_at` (corner cell EMFs v_p2*b_p1 - v_p1*b_p2
/// at coord / -e_p1 / -e_p2 / -e_p1-e_p2; face EMFs from -bflux_a / +bflux_b; density fluxes),
/// then the `ct_contact_emf_gv` soft blend. 8 generic inputs the dispatch binds per edge.
pub fn rmhd_edge_emf_gv(ndim: usize, g1: usize, g2: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    // g1/g2 are the two GRID offset axes the corner stencil walks (the edge's perpendicular
    // grid plane). they are DECOUPLED from the in-plane physical components p1/p2 the runtime
    // binds to vel_p1/bcell_p1/...: for identity geometries grid axis == component (3D: g1/g2 =
    // (dir+1)%3/(dir+2)%3), but for cyl r-z the grid axes are {0,1} while the components are
    // {r=0, z=2}. the kernel is component-agnostic — only the gather offsets are geometric.
    // pin the 8 inputs in the dispatch's order (vel_p1/p2, bcell_p1/p2, bflux_a/b, fden_p1/p2);
    // the actual values are read at the gather offsets below (gv_field_at, deduped).
    gv_register_field("edge_vp1", "vel_p1");
    gv_register_field("edge_vp2", "vel_p2");
    gv_register_field("edge_bp1", "bcell_p1");
    gv_register_field("edge_bp2", "bcell_p2");
    gv_register_field("edge_bflux_a", "bflux_a");
    gv_register_field("edge_bflux_b", "bflux_b");
    gv_register_field("edge_fden_p1", "fden_p1");
    gv_register_field("edge_fden_p2", "fden_p2");
    // -1 on the listed GRID axes (ndim-length offset; the 2.5D corner walks grid axes 0/1).
    let cm = |axes: &[usize]| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        for &ax in axes {
            o[ax] = -1;
        }
        o
    };
    let zero = vec![0i32; ndim];
    // cell edge-EMF E_dir = v_p2*b_p1 - v_p1*b_p2 at the given offset.
    let cell = |o: &[i32]| -> Gv {
        let vp1 = gv_field_at("edge_vp1", "vel_p1", ndim, o);
        let vp2 = gv_field_at("edge_vp2", "vel_p2", ndim, o);
        let bp1 = gv_field_at("edge_bp1", "bcell_p1", ndim, o);
        let bp2 = gv_field_at("edge_bp2", "bcell_p2", ndim, o);
        vp2 * bp1 - vp1 * bp2
    };
    let ene = cell(&zero);
    let enw = cell(&cm(&[g1]));
    let ese = cell(&cm(&[g2]));
    let esw = cell(&cm(&[g1, g2]));
    // face EMFs: en=-bflux_a[coord], es=-bflux_a[-e_g2], ee=+bflux_b[coord], ew=+bflux_b[-e_g1].
    let en = Gv::ZERO - gv_field_at("edge_bflux_a", "bflux_a", ndim, &zero);
    let es = Gv::ZERO - gv_field_at("edge_bflux_a", "bflux_a", ndim, &cm(&[g2]));
    let ee = gv_field_at("edge_bflux_b", "bflux_b", ndim, &zero);
    let ew = gv_field_at("edge_bflux_b", "bflux_b", ndim, &cm(&[g1]));
    // density fluxes: fn/fs = fden_p1 at coord / -e_g2; fe/fw = fden_p2 at coord / -e_g1.
    let fnf = gv_field_at("edge_fden_p1", "fden_p1", ndim, &zero);
    let fs = gv_field_at("edge_fden_p1", "fden_p1", ndim, &cm(&[g2]));
    let fe = gv_field_at("edge_fden_p2", "fden_p2", ndim, &zero);
    let fw = gv_field_at("edge_fden_p2", "fden_p2", ndim, &cm(&[g1]));
    let emf = ct_contact_emf_gv([en, es, ee, ew], [ene, enw, ese, esw], [fnf, fs, fe, fw]);
    (end_trace(), vec![("emf".to_string(), "emf".into(), emf.node())])
}


/// the per-direction UCT flux/diffusion coefficients at the edge — the (a^L, a^R, d^L, d^R) of the
/// master formula (Mignone & Del Zanna 2020, Eq. 30). `al`/`ar` are the advective flux weights of the
/// upwind/downwind states (a^L + a^R = 1); `dl`/`dr` the dissipative diffusion coefficients (equal
/// for HLL/HLLC's symmetric advection, distinct for HLLD). THIS is the only solver-specific piece:
/// HLL fills it from the fast speeds (regime-generic); HLLC/HLLD swap it for the contact/Alfvén-aware
/// coefficients (Eq. 38 / 44) — the SAME master EMF kernel consumes it.
struct UctDir {
    al: Gv,
    ar: Gv,
    dl: Gv,
    dr: Gv,
}


/// HLL coefficients (Eq. 32) from the edge signal speeds `ap = max(0, lambda_max)`,
/// `am = max(0, -lambda_min)`: a^L = ap/(ap+am), a^R = am/(ap+am), d^L = d^R = ap*am/(ap+am).
fn uct_hll_coeffs(ap: Gv, am: Gv) -> UctDir {
    let eps = Gv::from_f64(1.0e-30);
    let sum = ap + am + eps;
    let d = ap * am / sum;
    UctDir { al: ap / sum, ar: am / sum, dl: d, dr: d }
}


/// HLLC coefficients (Eq. 37-38). the three-wave fan (two fast `ll<=0<=lr` + the contact `lstar`)
/// gives a^L = a^R = 1/2 and the contact-aware diffusion
///   chi^s = -(vx^s - lambda^s)/(lambda^s - lstar),   d^s = ((|lstar|-|lambda^s|)/2) chi^s + |lambda^s|/2
/// (s = L,R). less dissipative than HLL because the transverse-field jump is resolved across the
/// contact, not the fast wave. `vxl`/`vxr` are the L/R normal velocities. classical & relativistic
/// share this algebra; only `lstar` (the contact speed) is regime-specific (computed upstream).
fn uct_hllc_coeffs(ll: Gv, lr: Gv, lstar: Gv, vxl: Gv, vxr: Gv) -> UctDir {
    let half = Gv::from_f64(0.5);
    let eps = Gv::from_f64(1.0e-30);
    // guard the (lambda^s - lstar) denominators away from zero (preserve sign).
    let den_l = ll - lstar;
    let den_r = lr - lstar;
    let den_l = den_l + eps * sign_gv(den_l);
    let den_r = den_r + eps * sign_gv(den_r);
    let chi_l = (Gv::ZERO - (vxl - ll)) / den_l;
    let chi_r = (Gv::ZERO - (vxr - lr)) / den_r;
    // Eq. 38: d^s = ((|lstar| - |lambda^s|)/2) chi^s + |lstar|/2  (the LAST term is |lstar|, the
    // contact speed, NOT |lambda^s|). this is the B_x = 0 DEGENERATE case (for B_x != 0 HLLC == HLL);
    // it is the building block for the HLLD singular limit (Eq. 46, v* = 0), not a standalone solver.
    let dl = ((lstar.abs() - ll.abs()) * half) * chi_l + lstar.abs() * half;
    let dr = ((lstar.abs() - lr.abs()) * half) * chi_r + lstar.abs() * half;
    // clamp to [0, d_HLL]: the HLL diffusion is the stable upper bound (so HLLC is never MORE
    // dissipative than HLL), and 0 the lower bound (no ANTI-diffusion). this also tames the fan
    // degeneracy lambda^s -> lstar where chi^s blows up (an approximate edge-level lstar can push
    // d^s hugely negative -> anti-diffusion -> blow-up; the proper per-face lstar would not, but
    // the clamp is a robust guard regardless).
    // FLOOR (d >= 0): the diffusion coefficient must be DISSIPATIVE. with the per-face lstar, d^s can
    // still dip slightly negative where lstar approaches lambda^s (chi^s grows); allowing it (no
    // floor) yields an unphysically "sharp" result from anti-diffusion (the checkerboard-prone
    // direction). flooring at 0 is the correct physical guard. NO upper cap — HLLC's d legitimately
    // differs from HLL's, and capping at d_HLL artificially over-diffuses.
    let floor = |d: Gv| Gv::ZERO.max(d);
    UctDir { al: half, ar: half, dl: floor(dl), dr: floor(dr) }
}


/// the UCT edge EMF in MASTER form (Mignone & Del Zanna 2020, Eq. 33) — the structure that
/// generalizes across Riemann solvers by swapping only the per-direction (a^L, a^R, d) coefficients
/// (`uct_*_coeffs`). for the out-of-plane (z) edge:
/// ```text
///   Ez = -vbar_x (a^L_x B_y^E + a^R_x B_y^W) + d_x (B_y^E - B_y^W)   [x: advect + diffuse B_y]
///       + vbar_y (a^L_y B_x^N + a^R_y B_x^S) - d_y (B_x^N - B_x^S)   [y: advect + diffuse B_x]
/// ```
/// `vbar_t` is the upwind transverse velocity (Eq. 29, from the edge speeds); the `B` are the
/// STAGGERED div-free FACE fields (B_y on y-faces E/W, B_x on x-faces N/S); the edge speeds are the
/// MAX over the 4 surrounding cells (the paper-sanctioned maximal-diffusion edge reconstruction).
/// reduces to `v_y B_x - v_x B_y` in the symmetric-speed limit; the diffusion matches the verified
/// compact Eq. 27. div(B)=0 preserved (a CT curl of one edge EMF, independent of the coefficients).
/// component-agnostic: only the gather offsets are geometric (g1/g2 = the perpendicular grid plane).
pub fn rmhd_edge_emf_uct_gv(ndim: usize, g1: usize, g2: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    gv_register_field("edge_vp1", "vel_p1");
    gv_register_field("edge_vp2", "vel_p2");
    gv_register_field("edge_bface_a", "bface_a");
    gv_register_field("edge_bface_b", "bface_b");
    gv_register_field("edge_wsr1", "wsr_p1");
    gv_register_field("edge_wsl1", "wsl_p1");
    gv_register_field("edge_wsr2", "wsr_p2");
    gv_register_field("edge_wsl2", "wsl_p2");
    let cm = |axes: &[usize]| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        for &ax in axes {
            o[ax] = -1;
        }
        o
    };
    let zero = vec![0i32; ndim];
    let half = Gv::from_f64(0.5);
    let zero_g = Gv::ZERO;
    // cell velocity gathers. corners about the edge (lower-left of cell NE): NE=0, NW=-g1, SE=-g2,
    // SW=-g1-g2. the side velocities are the 2-cell averages straddling the edge.
    let vp1 = |o: &[i32]| gv_field_at("edge_vp1", "vel_p1", ndim, o);
    let vp2 = |o: &[i32]| gv_field_at("edge_vp2", "vel_p2", ndim, o);
    let vx_e = (vp1(&zero) + vp1(&cm(&[g2]))) * half; // East cells (NE, SE)
    let vx_w = (vp1(&cm(&[g1])) + vp1(&cm(&[g1, g2]))) * half; // West (NW, SW)
    let vy_n = (vp2(&zero) + vp2(&cm(&[g1]))) * half; // North (NE, NW)
    let vy_s = (vp2(&cm(&[g2])) + vp2(&cm(&[g1, g2]))) * half; // South (SE, SW)
    // edge signal speeds: MAX over the 4 surrounding cells (maximal-diffusion edge reconstruction).
    let max4 = |key: &str, path: &str| -> Gv {
        let v0 = gv_field_at(key, path, ndim, &zero);
        let v1 = gv_field_at(key, path, ndim, &cm(&[g1]));
        let v2 = gv_field_at(key, path, ndim, &cm(&[g2]));
        let v3 = gv_field_at(key, path, ndim, &cm(&[g1, g2]));
        v0.max(v1).max(v2).max(v3)
    };
    let neg_min4 = |key: &str, path: &str| -> Gv {
        let v0 = zero_g - gv_field_at(key, path, ndim, &zero);
        let v1 = zero_g - gv_field_at(key, path, ndim, &cm(&[g1]));
        let v2 = zero_g - gv_field_at(key, path, ndim, &cm(&[g2]));
        let v3 = zero_g - gv_field_at(key, path, ndim, &cm(&[g1, g2]));
        v0.max(v1).max(v2).max(v3)
    };
    let apx = zero_g.max(max4("edge_wsr1", "wsr_p1"));
    let amx = zero_g.max(neg_min4("edge_wsl1", "wsl_p1"));
    let apy = zero_g.max(max4("edge_wsr2", "wsr_p2"));
    let amy = zero_g.max(neg_min4("edge_wsl2", "wsl_p2"));
    // SOLVER-SPECIFIC coefficients (HLL here; swap uct_hll_coeffs -> hllc/hlld later).
    let cx = uct_hll_coeffs(apx, amx);
    let cy = uct_hll_coeffs(apy, amy);
    // upwind transverse velocities (Eq. 29): vbar_x upwind in x (alpha^+ carries the West/left state),
    // vbar_y upwind in y (alpha^+ carries the South/lower state).
    let eps = Gv::from_f64(1.0e-30);
    let vbar_x = (apx * vx_w + amx * vx_e) / (apx + amx + eps);
    let vbar_y = (apy * vy_s + amy * vy_n) / (apy + amy + eps);
    // staggered face B PLM-reconstructed a half-cell to the EDGE (M&DZ: the staggered transverse
    // field reconstructed from the adjacent interface — the load-bearing 2nd-order piece). geometry
    // VERIFIED vs the CT curl: Ez[i,j] is the corner (i-1/2,j-1/2); B_y is at the corner's y but
    // offset +-1/2 in x (recon along x = its transverse), B_x at the corner's x offset +-1/2 in y.
    // one-sided minmod-theta extrapolation: +1/2 toward the edge from the lower face, -1/2 from the
    // upper. needs the 2nd transverse neighbour -> bface allocated with +-2 transverse halo.
    let theta = Gv::scalar("theta");
    let recon = |key: &str, rt: &str, base: &[i32], axis: usize, sign: f64| -> Gv {
        let off = |d: i32| -> Vec<i32> { let mut o = base.to_vec(); o[axis] += d; o };
        let q0 = gv_field_at(key, rt, ndim, base);
        let qm = gv_field_at(key, rt, ndim, &off(-1));
        let qp = gv_field_at(key, rt, ndim, &off(1));
        let slope = minmod3((q0 - qm) * theta, half * (qp - qm), (qp - q0) * theta);
        q0 + Gv::from_f64(0.5 * sign) * slope
    };
    let by_e = recon("edge_bface_b", "bface_b", &zero, g1, -1.0); // B_y[i,j],   recon -1/2 in x
    let by_w = recon("edge_bface_b", "bface_b", &cm(&[g1]), g1, 1.0); // B_y[i-1,j], recon +1/2 in x
    let bx_n = recon("edge_bface_a", "bface_a", &zero, g2, -1.0); // B_x[i,j],   recon -1/2 in y
    let bx_s = recon("edge_bface_a", "bface_a", &cm(&[g2]), g2, 1.0); // B_x[i,j-1], recon +1/2 in y
    let emf = uct_master_emf(&cx, &cy, vbar_x, vbar_y, by_e, by_w, bx_n, bx_s);
    (end_trace(), vec![("emf".to_string(), "emf".into(), emf.node())])
}


/// PLM-reconstruct a staggered face field a half-cell to the EDGE (M&DZ: the staggered transverse
/// field reconstructed from the adjacent interface — the 2nd-order piece that preserves smooth fields,
/// VERIFIED on the field-loop test). `base` the face offset; `axis` the reconstruction direction (the
/// face's TRANSVERSE: x for B_y on y-faces, y for B_x on x-faces); `sign` = +1 reconstructs +1/2
/// toward the edge from the lower face, -1 reconstructs -1/2 from the upper. minmod-theta slope;
/// needs the 2nd transverse neighbour, hence bface's +-2 transverse halo.
fn recon_face_to_edge(ndim: usize, theta: Gv, key: &str, rt: &str, base: &[i32], axis: usize, sign: f64) -> Gv {
    let half = Gv::from_f64(0.5);
    let off = |d: i32| -> Vec<i32> { let mut o = base.to_vec(); o[axis] += d; o };
    let q0 = gv_field_at(key, rt, ndim, base);
    let qm = gv_field_at(key, rt, ndim, &off(-1));
    let qp = gv_field_at(key, rt, ndim, &off(1));
    let a = q0 - qm;
    let b = qp - q0;
    let mm = minmod3(a * theta, half * (a + b), b * theta);
    let slope = Gv::select(theta.cmp_lt(Gv::ZERO), van_leer(a, b), mm);
    q0 + Gv::from_f64(0.5 * sign) * slope
}


/// the master-formula edge EMF combination (Eq. 33), shared by every UCT coefficient family. given
/// the per-direction coefficients + the upwind transverse velocities + the staggered face B at the
/// edge:
/// ```text
///   Ez = -vbar_x (a^L_x B_y^E + a^R_x B_y^W) + (d^R_x B_y^E - d^L_x B_y^W)
///       + vbar_y (a^L_y B_x^N + a^R_y B_x^S) - (d^R_y B_x^N - d^L_y B_x^S)
/// ```
/// (signs verified against the compact Eq. 27 diffusion + the symmetric-speed reduction v_y B_x - v_x B_y.)
fn uct_master_emf(cx: &UctDir, cy: &UctDir, vbar_x: Gv, vbar_y: Gv, by_e: Gv, by_w: Gv, bx_n: Gv, bx_s: Gv) -> Gv {
    let zero_g = Gv::ZERO;
    // a^L (= alpha^+/sum) weights the UPWIND face: West for +x (a^L -> by_w), South for +y (a^L -> bx_s)
    // — CONSISTENT with the diffusion's d^L->West/d^R->East pairing and with vbar (apx*vx_w). pairing
    // a^L to the downwind face is anti-upwind: invisible for symmetric speeds (a^L==a^R, subsonic OT)
    // but ADVECTS THE DOWNWIND state at supersonic Mach -> instability (the field-loop blow-up). the
    // single x-upwound vbar (NOT the literal W/E-distinct Eq:UCT_HLL2) is retained: the latter was
    // implemented and made OT noisier (the upwinding supplies smoothing the bare master form lacks).
    let adv_x = zero_g - vbar_x * (cx.al * by_w + cx.ar * by_e);
    let dif_x = cx.dr * by_e - cx.dl * by_w;
    let adv_y = vbar_y * (cy.al * bx_s + cy.ar * bx_n);
    let dif_y = zero_g - (cy.dr * bx_n - cy.dl * bx_s);
    adv_x + dif_x + adv_y + dif_y
}

/// proof entry point for the upwind-pairing invariant. traces `uct_master_emf` in ISOLATION with
/// symbolic param leaves: the four staggered face reads {by_w, by_e, bx_n, bx_s} as the "fields"
/// and the uct coefficients {vbar_x/y, al/ar/dl/dr per dir} as opaque scalars. because the master
/// composition is LINEAR in the faces (all the wave-speed nonlinearity lives UPSTREAM, in cx/cy),
/// `symbi_ir::proof::LinForm` reads off each face's exact coefficient polynomial, and the upwind
/// invariant — a^L weights the UPWIND face (by_w for +x, bx_s for +y) — becomes a coefficient
/// check at graph-build time. instant, and it covers ALL uct emf kernels at once: every one of
/// them composes through `uct_master_emf`. `swap` passes by_w/by_e in each other's argument slots
/// to inject the ct_emf.rs anti-upwind bug, for the negative control.
#[doc(hidden)]
pub fn uct_master_emf_proof_kernel(swap: bool) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let cx = UctDir {
        al: Gv::param("al_x"),
        ar: Gv::param("ar_x"),
        dl: Gv::param("dl_x"),
        dr: Gv::param("dr_x"),
    };
    let cy = UctDir {
        al: Gv::param("al_y"),
        ar: Gv::param("ar_y"),
        dl: Gv::param("dl_y"),
        dr: Gv::param("dr_y"),
    };
    let vbar_x = Gv::param("vbar_x");
    let vbar_y = Gv::param("vbar_y");
    let by_e = Gv::param("by_e");
    let by_w = Gv::param("by_w");
    let bx_n = Gv::param("bx_n");
    let bx_s = Gv::param("bx_s");
    // correct arg order is (.., by_e, by_w, bx_n, bx_s); swapping by_e<->by_w in the call
    // reproduces the anti-upwind pairing the gate must reject (a^L lands on the downwind face).
    let emf = if swap {
        uct_master_emf(&cx, &cy, vbar_x, vbar_y, by_w, by_e, bx_n, bx_s)
    } else {
        uct_master_emf(&cx, &cy, vbar_x, vbar_y, by_e, by_w, bx_n, bx_s)
    };
    (end_trace(), vec![("emf".to_string(), "emf".into(), emf.node())])
}



/// the UCT-HLLC edge EMF (master Eq. 33 + HLLC coefficients Eq. 37-38). same master formula as the
/// HLL kernel, but the diffusion uses the CONTACT speed `lstar` (the three-wave fan) -> less
/// dissipative. CLASSICAL ideal-gas (NMHD): `lstar = m_n^hll/rho^hll` is the HLL-average normal
/// velocity, computed in-kernel from the cell prims with the classical momentum flux
/// `F[m_n] = rho v_n^2 + p + |B|^2/2 - B_n^2`. edge speeds & per-side states use the MAX-over-4-cells
/// / 2-cell-average reconstruction. (IMHD: p = cs^2*rho; RMHD: relativistic conserved/flux.)
pub fn nmhd_edge_emf_uct_hllc_gv(ndim: usize, g1: usize, g2: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    gv_register_field("e_rho", "rho");
    gv_register_field("e_vp1", "vel_p1");
    gv_register_field("e_vp2", "vel_p2");
    gv_register_field("e_pre", "pre");
    gv_register_field("e_bp1", "bcell_p1");
    gv_register_field("e_bp2", "bcell_p2");
    gv_register_field("e_bout", "bcell_out");
    gv_register_field("e_bface_a", "bface_a");
    gv_register_field("e_bface_b", "bface_b");
    gv_register_field("e_wsr1", "wsr_p1");
    gv_register_field("e_wsl1", "wsl_p1");
    gv_register_field("e_wsr2", "wsr_p2");
    gv_register_field("e_wsl2", "wsl_p2");
    let cm = |axes: &[usize]| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        for &ax in axes {
            o[ax] = -1;
        }
        o
    };
    let zero = vec![0i32; ndim];
    let half = Gv::from_f64(0.5);
    let eps = Gv::from_f64(1.0e-30);
    let zero_g = Gv::ZERO;
    let rho = |o: &[i32]| gv_field_at("e_rho", "rho", ndim, o);
    let vp1 = |o: &[i32]| gv_field_at("e_vp1", "vel_p1", ndim, o);
    let vp2 = |o: &[i32]| gv_field_at("e_vp2", "vel_p2", ndim, o);
    let pre = |o: &[i32]| gv_field_at("e_pre", "pre", ndim, o);
    let bp1 = |o: &[i32]| gv_field_at("e_bp1", "bcell_p1", ndim, o);
    let bp2 = |o: &[i32]| gv_field_at("e_bp2", "bcell_p2", ndim, o);
    let bout = |o: &[i32]| gv_field_at("e_bout", "bcell_out", ndim, o);
    let bsq = |o: &[i32]| {
        let a = bp1(o);
        let b = bp2(o);
        let c = bout(o);
        a * a + b * b + c * c
    };
    let avg2 = |a: Gv, b: Gv| (a + b) * half;
    // corners about the edge: NE=0, NW=-g1, SE=-g2, SW=-g1-g2.
    let ne = zero.clone();
    let nw = cm(&[g1]);
    let se = cm(&[g2]);
    let sw = cm(&[g1, g2]);
    // edge signal speeds: MAX over the 4 cells.
    let max4 = |key: &str, path: &str| -> Gv {
        let v0 = gv_field_at(key, path, ndim, &ne);
        let v1 = gv_field_at(key, path, ndim, &nw);
        let v2 = gv_field_at(key, path, ndim, &se);
        let v3 = gv_field_at(key, path, ndim, &sw);
        v0.max(v1).max(v2).max(v3)
    };
    let neg_min4 = |key: &str, path: &str| -> Gv {
        let v0 = zero_g - gv_field_at(key, path, ndim, &ne);
        let v1 = zero_g - gv_field_at(key, path, ndim, &nw);
        let v2 = zero_g - gv_field_at(key, path, ndim, &se);
        let v3 = zero_g - gv_field_at(key, path, ndim, &sw);
        v0.max(v1).max(v2).max(v3)
    };
    let apx = zero_g.max(max4("e_wsr1", "wsr_p1"));
    let amx = zero_g.max(neg_min4("e_wsl1", "wsl_p1"));
    let apy = zero_g.max(max4("e_wsr2", "wsr_p2"));
    let amy = zero_g.max(neg_min4("e_wsl2", "wsl_p2"));
    // per-FACE HLLC coefficients: each face uses ITS OWN two cells (first-order L/R) + Davis face
    // speeds (s_r = max(0, ws_r^L, ws_r^R), s_l = min(0, ws_l^L, ws_l^R)) so lstar = m_n^hll/rho^hll
    // is CONSISTENT (the contact stays inside the fan -> no degeneracy blow-up). then the diffusion
    // is MAX-combined to the edge (the maximal-diffusion edge reconstruction). vn/bn read the normal
    // velocity / normal cell-B; wsr/wsl are the direction's per-cell speed fields.
    let face_d = |l: &[i32], r: &[i32], vn: &dyn Fn(&[i32]) -> Gv, bn: &dyn Fn(&[i32]) -> Gv,
                  wsr_k: &str, wsr_p: &str, wsl_k: &str, wsl_p: &str| -> (Gv, Gv) {
        let (rl, rr) = (rho(l), rho(r));
        let (vl, vr) = (vn(l), vn(r));
        let (pl, pr) = (pre(l), pre(r));
        let (bsl, bsr) = (bsq(l), bsq(r));
        let (bnl, bnr) = (bn(l), bn(r));
        let sr = zero_g.max(gv_field_at(wsr_k, wsr_p, ndim, l).max(gv_field_at(wsr_k, wsr_p, ndim, r)));
        let sl = zero_g.min(gv_field_at(wsl_k, wsl_p, ndim, l).min(gv_field_at(wsl_k, wsl_p, ndim, r)));
        let (mxl, mxr) = (rl * vl, rr * vr);
        let fl = rl * vl * vl + pl + half * bsl - bnl * bnl;
        let fr = rr * vr * vr + pr + half * bsr - bnr * bnr;
        let inv = Gv::ONE / (sr - sl + eps);
        let rho_hll = (sr * rr - sl * rl + mxl - mxr) * inv;
        let mx_hll = (sr * mxr - sl * mxl + fl - fr) * inv;
        let lstar = mx_hll / (rho_hll + eps * sign_gv(rho_hll));
        let c = uct_hllc_coeffs(sl, sr, lstar, vl, vr);
        (c.dl, c.dr)
    };
    // x-faces (normal p1): North NW->NE, South SW->SE; MAX-combine d to the edge.
    let (dln, drn) = face_d(&nw, &ne, &vp1, &bp1, "e_wsr1", "wsr_p1", "e_wsl1", "wsl_p1");
    let (dls, drs) = face_d(&sw, &se, &vp1, &bp1, "e_wsr1", "wsr_p1", "e_wsl1", "wsl_p1");
    let cx = UctDir { al: half, ar: half, dl: avg2(dln, dls), dr: avg2(drn, drs) };
    // y-faces (normal p2): West SW->NW, East SE->NE.
    let (dlw, drw) = face_d(&sw, &nw, &vp2, &bp2, "e_wsr2", "wsr_p2", "e_wsl2", "wsl_p2");
    let (dle, dre) = face_d(&se, &ne, &vp2, &bp2, "e_wsr2", "wsr_p2", "e_wsl2", "wsl_p2");
    let cy = UctDir { al: half, ar: half, dl: avg2(dlw, dle), dr: avg2(drw, dre) };
    // upwind transverse velocities (Eq. 29) for the advective part: alpha^+ carries West / South.
    let vx_w = avg2(vp1(&nw), vp1(&sw));
    let vx_e = avg2(vp1(&ne), vp1(&se));
    let vy_s = avg2(vp2(&sw), vp2(&se));
    let vy_n = avg2(vp2(&nw), vp2(&ne));
    let vbar_x = (apx * vx_w + amx * vx_e) / (apx + amx + eps);
    let vbar_y = (apy * vy_s + amy * vy_n) / (apy + amy + eps);
    // staggered face B PLM-reconstructed a half-cell to the edge (M&DZ transverse reconstruction).
    let theta = Gv::scalar("theta");
    let by_e = recon_face_to_edge(ndim, theta, "e_bface_b", "bface_b", &zero, g1, -1.0);
    let by_w = recon_face_to_edge(ndim, theta, "e_bface_b", "bface_b", &nw, g1, 1.0);
    let bx_n = recon_face_to_edge(ndim, theta, "e_bface_a", "bface_a", &zero, g2, -1.0);
    let bx_s = recon_face_to_edge(ndim, theta, "e_bface_a", "bface_a", &se, g2, 1.0);
    let emf = uct_master_emf(&cx, &cy, vbar_x, vbar_y, by_e, by_w, bx_n, bx_s);
    (end_trace(), vec![("emf".to_string(), "emf".into(), emf.node())])
}


/// the NMHD UCT-HLLD edge EMF — Mignone & Del Zanna (2020), reproduced VERBATIM. NO liberties, NO
/// floors. the composition is the solver-agnostic MASTER form (eq:emf2D, = `uct_master_emf`,
/// byte-identical structure to UCT-HLL); per the paper, the ONLY solver-specific part is the per-face
/// (a,d) coefficients — the five-wave HLLD fan (eq:UCT_HLLD_ad, eq:UCT_HLLD_nu, eq:By_chi). edge
/// combine: MAX on the diffusion `d` (paper-sanctioned, "maximizing the diffusion terms"), AVERAGE
/// on the advective `a` (eq:dEW). upwind transverse velocity `vbar` (eq:vt), shared with UCT-HLL.
/// NO floor on rho^{*s}: physical states give rho^{*s} > 0; the degenerate guard (nu* = 0 when the
/// rotational waves collapse, eps = 1e-9) is the ONLY safeguard. zeroth order = R± reconstruction
/// is identity (theta = 0). spec: literature/uct_algorithm.md §3.4 + mignone_delzanna/method2.tex.
pub fn nmhd_edge_emf_uct_hlld_gv(ndim: usize, g1: usize, g2: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    gv_register_field("h_rho", "rho");
    gv_register_field("h_pre", "pre");
    gv_register_field("h_vp1", "vel_p1");
    gv_register_field("h_vp2", "vel_p2");
    gv_register_field("h_bp1", "bcell_p1");
    gv_register_field("h_bp2", "bcell_p2");
    gv_register_field("h_bout", "bcell_out");
    gv_register_field("h_bface_a", "bface_a");
    gv_register_field("h_bface_b", "bface_b");
    gv_register_field("h_wsr1", "wsr_p1");
    gv_register_field("h_wsl1", "wsl_p1");
    gv_register_field("h_wsr2", "wsr_p2");
    gv_register_field("h_wsl2", "wsl_p2");
    let cm = |axes: &[usize]| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        for &ax in axes {
            o[ax] = -1;
        }
        o
    };
    let zero = vec![0i32; ndim];
    let half = Gv::from_f64(0.5);
    let one = Gv::ONE;
    let eps = Gv::from_f64(1.0e-30);
    let zero_g = Gv::ZERO;
    let ne = zero.clone();
    let nw = cm(&[g1]);
    let se = cm(&[g2]);
    let sw = cm(&[g1, g2]);
    // PLM reconstruction of a CELL field to a face — identical to the gas flux's plm_theta_gv (same
    // theta, same limiter: theta<0 selects van Leer). THE FIX: M&DZ feed the edge-EMF fan the SAME
    // reconstructed L/R face states the 1D gas Riemann solves (Eq. 29 + the per-face Riemann input),
    // NOT cell-centered values — cell states make the EMF fan inconsistent with the flux at sharp
    // reconstruction, which is the OT/field-loop checkerboard. L uses sign +1 (reconstruct toward the
    // +naxis face), R uses -1.
    let theta = Gv::scalar("theta");
    let recon_cell = |key: &str, rt: &str, base: &[i32], naxis: usize, sign: f64| -> Gv {
        let off = |d: i32| -> Vec<i32> {
            let mut o = base.to_vec();
            o[naxis] += d;
            o
        };
        let q0 = gv_field_at(key, rt, ndim, base);
        let qm = gv_field_at(key, rt, ndim, &off(-1));
        let qp = gv_field_at(key, rt, ndim, &off(1));
        let a = q0 - qm;
        let b = qp - q0;
        let mm = minmod3(a * theta, half * (a + b), b * theta);
        let slope = Gv::select(theta.cmp_lt(Gv::ZERO), van_leer(a, b), mm);
        q0 + Gv::from_f64(0.5 * sign) * slope
    };
    // reconstruct the 8-component MHD prim to ONE side of a face; override the NORMAL B with the
    // staggered face value (M&DZ guideline 2, identical to nmhd_reconstruct). the out-of-plane velocity
    // never enters the coefficients (they use only the normal velocity + rho/p/|B|), so leave it zero.
    let eos = IdealGas { gamma: Gv::scalar("gamma") };
    let face_prim = |base: &[i32], naxis: usize, nidx: usize, sign: f64, bn_face: Gv| -> MhdPrim<Gv, 3> {
        let mut mag = [
            recon_cell("h_bp1", "bcell_p1", base, naxis, sign),
            recon_cell("h_bp2", "bcell_p2", base, naxis, sign),
            recon_cell("h_bout", "bcell_out", base, naxis, sign),
        ];
        mag[nidx] = bn_face;
        MhdPrim {
            hydro: Prim {
                rho: recon_cell("h_rho", "rho", base, naxis, sign),
                vel: Tensor::new([
                    recon_cell("h_vp1", "vel_p1", base, naxis, sign),
                    recon_cell("h_vp2", "vel_p2", base, naxis, sign),
                    Gv::ZERO,
                ]),
                pre: recon_cell("h_pre", "pre", base, naxis, sign),
            },
            mag: Tensor::new(mag),
        }
    };
    let nhat_x = Tensor::<Gv, 3>::unit(0);
    let nhat_y = Tensor::<Gv, 3>::unit(1);
    // per-FACE UCT-HLLD coefficients (a^L, d^L, d^R) from the RECONSTRUCTED L/R states
    // (eq:UCT_HLLD_ad via hlld_newtonian_coeffs) — the EMF fan is now IDENTICAL to the gas flux's at
    // the same reconstructed face state (the CT-consistency M&DZ require). NO clamp.
    let hlld_face = |l: &[i32], r: &[i32], naxis: usize, nidx: usize, bn_face: Gv, nhat: &Tensor<Gv, 3>| -> (Gv, Gv, Gv) {
        let pl = face_prim(l, naxis, nidx, 1.0, bn_face);
        let pr = face_prim(r, naxis, nidx, -1.0, bn_face);
        hlld_newtonian_coeffs(&eos, &pl, &pr, nhat)
    };
    // advective velocity at each y/x-face: PLM-RECONSTRUCT the transverse velocity to the face (Eq. 29),
    // then average the L/R — closes D2 (was a 2-cell PCM average). x-velocities reconstruct in g2 to the
    // W/E y-faces; y-velocities in g1 to the N/S x-faces. matches the fan's recon_cell exactly.
    // cell velocity gathers + 2-cell-average face velocities, then the upwind transverse vbar (eq:vt),
    // shared with UCT-HLL. NOTE: the LITERAL W/E-distinct + PLM-reconstructed velocity form (Eq:UCT_HLL2
    // + line 59) was implemented and made OT noisier — the single-vbar x-upwinding supplies smoothing
    // the bare master form lacks. reverted to the validated form; the velocity stays a deviation (D2/D7).
    let vp1 = |o: &[i32]| gv_field_at("h_vp1", "vel_p1", ndim, o);
    let vp2 = |o: &[i32]| gv_field_at("h_vp2", "vel_p2", ndim, o);
    let vx_e = (vp1(&ne) + vp1(&se)) * half;
    let vx_w = (vp1(&nw) + vp1(&sw)) * half;
    let vy_n = (vp2(&ne) + vp2(&nw)) * half;
    let vy_s = (vp2(&se) + vp2(&sw)) * half;
    let max4 = |key: &str, path: &str| -> Gv {
        gv_field_at(key, path, ndim, &ne)
            .max(gv_field_at(key, path, ndim, &nw))
            .max(gv_field_at(key, path, ndim, &se))
            .max(gv_field_at(key, path, ndim, &sw))
    };
    let neg_min4 = |key: &str, path: &str| -> Gv {
        (zero_g - gv_field_at(key, path, ndim, &ne))
            .max(zero_g - gv_field_at(key, path, ndim, &nw))
            .max(zero_g - gv_field_at(key, path, ndim, &se))
            .max(zero_g - gv_field_at(key, path, ndim, &sw))
    };
    let apx = zero_g.max(max4("h_wsr1", "wsr_p1"));
    let amx = zero_g.max(neg_min4("h_wsl1", "wsl_p1"));
    let apy = zero_g.max(max4("h_wsr2", "wsr_p2"));
    let amy = zero_g.max(neg_min4("h_wsl2", "wsl_p2"));
    // upwind transverse velocity (eq:vt): alpha^+ carries the West/South state.
    let vbar_x = (apx * vx_w + amx * vx_e) / (apx + amx + eps);
    let vbar_y = (apy * vy_s + amy * vy_n) / (apy + amy + eps);
    // staggered face B reconstructed to the EDGE (R±; theta=0 => identity = zeroth order). these are the
    // DISSIPATED transverse fields in the master composition (Eq. 16).
    let by_e = recon_face_to_edge(ndim, theta, "h_bface_b", "bface_b", &zero, g1, -1.0);
    let by_w = recon_face_to_edge(ndim, theta, "h_bface_b", "bface_b", &nw, g1, 1.0);
    let bx_n = recon_face_to_edge(ndim, theta, "h_bface_a", "bface_a", &zero, g2, -1.0);
    let bx_s = recon_face_to_edge(ndim, theta, "h_bface_a", "bface_a", &se, g2, 1.0);
    // per-face coefficients at the 4 faces, combined to the edge: AVERAGE on a (eq:dEW), MAX on d
    // (paper-sanctioned "maximizing the diffusion terms"). the fan's NORMAL B is the staggered FACE
    // value (guideline 2, the un-reconstructed face field). x-faces (normal g1, prim comp 0).
    let bn_n = gv_field_at("h_bface_a", "bface_a", ndim, &ne);
    let bn_s = gv_field_at("h_bface_a", "bface_a", ndim, &se);
    let (aln, dln, drn) = hlld_face(&nw, &ne, g1, 0, bn_n, &nhat_x);
    let (als, dls, drs) = hlld_face(&sw, &se, g1, 0, bn_s, &nhat_x);
    let ax_l = (aln + als) * half;
    let cx = UctDir { al: ax_l, ar: one - ax_l, dl: dln.max(dls), dr: drn.max(drs) };
    // y-faces (normal g2, prim comp 1): West SW->NW, East SE->NE.
    let bn_w = gv_field_at("h_bface_b", "bface_b", ndim, &nw);
    let bn_e = gv_field_at("h_bface_b", "bface_b", ndim, &ne);
    let (alw, dlw, drw) = hlld_face(&sw, &nw, g2, 1, bn_w, &nhat_y);
    let (ale, dle, dre) = hlld_face(&se, &ne, g2, 1, bn_e, &nhat_y);
    let ay_l = (alw + ale) * half;
    let cy = UctDir { al: ay_l, ar: one - ay_l, dl: dlw.max(dle), dr: drw.max(dre) };
    // master composition (eq:emf2D), identical to UCT-HLL.
    let emf = uct_master_emf(&cx, &cy, vbar_x, vbar_y, by_e, by_w, bx_n, bx_s);
    (end_trace(), vec![("emf".to_string(), "emf".into(), emf.node())])
}


/// the ISOTHERMAL UCT-HLLD edge EMF (IMHD). twin of `nmhd_edge_emf_uct_hlld_gv` but the per-face fan
/// is `hlld_isothermal_coeffs` (M&DZ 2020 Appendix A: no contact mode -> chi~^s uses the HLL central
/// state, a/d/nu unchanged) and the prim has NO pressure (`Isothermal{cs}`, `IsoMhdPrim`). everything
/// else — staggered-B recon to the edge, the single-vbar advection, the master composition — matches
/// the NMHD kernel verbatim.
pub fn imhd_edge_emf_uct_hlld_gv(ndim: usize, g1: usize, g2: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    gv_register_field("h_rho", "rho");
    gv_register_field("h_vp1", "vel_p1");
    gv_register_field("h_vp2", "vel_p2");
    gv_register_field("h_bp1", "bcell_p1");
    gv_register_field("h_bp2", "bcell_p2");
    gv_register_field("h_bout", "bcell_out");
    gv_register_field("h_bface_a", "bface_a");
    gv_register_field("h_bface_b", "bface_b");
    gv_register_field("h_wsr1", "wsr_p1");
    gv_register_field("h_wsl1", "wsl_p1");
    gv_register_field("h_wsr2", "wsr_p2");
    gv_register_field("h_wsl2", "wsl_p2");
    let cm = |axes: &[usize]| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        for &ax in axes {
            o[ax] = -1;
        }
        o
    };
    let zero = vec![0i32; ndim];
    let half = Gv::from_f64(0.5);
    let one = Gv::ONE;
    let eps = Gv::from_f64(1.0e-30);
    let zero_g = Gv::ZERO;
    let ne = zero.clone();
    let nw = cm(&[g1]);
    let se = cm(&[g2]);
    let sw = cm(&[g1, g2]);
    let theta = Gv::scalar("theta");
    let recon_cell = |key: &str, rt: &str, base: &[i32], naxis: usize, sign: f64| -> Gv {
        let off = |d: i32| -> Vec<i32> {
            let mut o = base.to_vec();
            o[naxis] += d;
            o
        };
        let q0 = gv_field_at(key, rt, ndim, base);
        let qm = gv_field_at(key, rt, ndim, &off(-1));
        let qp = gv_field_at(key, rt, ndim, &off(1));
        let a = q0 - qm;
        let b = qp - q0;
        let mm = minmod3(a * theta, half * (a + b), b * theta);
        let slope = Gv::select(theta.cmp_lt(Gv::ZERO), van_leer(a, b), mm);
        q0 + Gv::from_f64(0.5 * sign) * slope
    };
    // isothermal prim reconstructed to a face; NORMAL B overridden with the staggered face value
    // (guideline 2). NO pressure slot (IsoModel ZST), out-velocity unused by the coefficients.
    let eos = Isothermal { cs: Gv::scalar("cs") };
    let face_prim = |base: &[i32], naxis: usize, nidx: usize, sign: f64, bn_face: Gv| -> IsoMhdPrim<Gv, 3> {
        let mut mag = [
            recon_cell("h_bp1", "bcell_p1", base, naxis, sign),
            recon_cell("h_bp2", "bcell_p2", base, naxis, sign),
            recon_cell("h_bout", "bcell_out", base, naxis, sign),
        ];
        mag[nidx] = bn_face;
        IsoMhdPrim {
            hydro: PrimG {
                rho: recon_cell("h_rho", "rho", base, naxis, sign),
                vel: Tensor::new([
                    recon_cell("h_vp1", "vel_p1", base, naxis, sign),
                    recon_cell("h_vp2", "vel_p2", base, naxis, sign),
                    Gv::ZERO,
                ]),
                pre: Zero::default(),
            },
            mag: Tensor::new(mag),
        }
    };
    let nhat_x = Tensor::<Gv, 3>::unit(0);
    let nhat_y = Tensor::<Gv, 3>::unit(1);
    let hlld_face = |l: &[i32], r: &[i32], naxis: usize, nidx: usize, bn_face: Gv, nhat: &Tensor<Gv, 3>| -> (Gv, Gv, Gv) {
        let pl = face_prim(l, naxis, nidx, 1.0, bn_face);
        let pr = face_prim(r, naxis, nidx, -1.0, bn_face);
        hlld_isothermal_coeffs(&eos, &pl, &pr, nhat)
    };
    let vp1 = |o: &[i32]| gv_field_at("h_vp1", "vel_p1", ndim, o);
    let vp2 = |o: &[i32]| gv_field_at("h_vp2", "vel_p2", ndim, o);
    let vx_e = (vp1(&ne) + vp1(&se)) * half;
    let vx_w = (vp1(&nw) + vp1(&sw)) * half;
    let vy_n = (vp2(&ne) + vp2(&nw)) * half;
    let vy_s = (vp2(&se) + vp2(&sw)) * half;
    let max4 = |key: &str, path: &str| -> Gv {
        gv_field_at(key, path, ndim, &ne)
            .max(gv_field_at(key, path, ndim, &nw))
            .max(gv_field_at(key, path, ndim, &se))
            .max(gv_field_at(key, path, ndim, &sw))
    };
    let neg_min4 = |key: &str, path: &str| -> Gv {
        (zero_g - gv_field_at(key, path, ndim, &ne))
            .max(zero_g - gv_field_at(key, path, ndim, &nw))
            .max(zero_g - gv_field_at(key, path, ndim, &se))
            .max(zero_g - gv_field_at(key, path, ndim, &sw))
    };
    let apx = zero_g.max(max4("h_wsr1", "wsr_p1"));
    let amx = zero_g.max(neg_min4("h_wsl1", "wsl_p1"));
    let apy = zero_g.max(max4("h_wsr2", "wsr_p2"));
    let amy = zero_g.max(neg_min4("h_wsl2", "wsl_p2"));
    let vbar_x = (apx * vx_w + amx * vx_e) / (apx + amx + eps);
    let vbar_y = (apy * vy_s + amy * vy_n) / (apy + amy + eps);
    let by_e = recon_face_to_edge(ndim, theta, "h_bface_b", "bface_b", &zero, g1, -1.0);
    let by_w = recon_face_to_edge(ndim, theta, "h_bface_b", "bface_b", &nw, g1, 1.0);
    let bx_n = recon_face_to_edge(ndim, theta, "h_bface_a", "bface_a", &zero, g2, -1.0);
    let bx_s = recon_face_to_edge(ndim, theta, "h_bface_a", "bface_a", &se, g2, 1.0);
    let bn_n = gv_field_at("h_bface_a", "bface_a", ndim, &ne);
    let bn_s = gv_field_at("h_bface_a", "bface_a", ndim, &se);
    let (aln, dln, drn) = hlld_face(&nw, &ne, g1, 0, bn_n, &nhat_x);
    let (als, dls, drs) = hlld_face(&sw, &se, g1, 0, bn_s, &nhat_x);
    let ax_l = (aln + als) * half;
    let cx = UctDir { al: ax_l, ar: one - ax_l, dl: dln.max(dls), dr: drn.max(drs) };
    let bn_w = gv_field_at("h_bface_b", "bface_b", ndim, &nw);
    let bn_e = gv_field_at("h_bface_b", "bface_b", ndim, &ne);
    let (alw, dlw, drw) = hlld_face(&sw, &nw, g2, 1, bn_w, &nhat_y);
    let (ale, dle, dre) = hlld_face(&se, &ne, g2, 1, bn_e, &nhat_y);
    let ay_l = (alw + ale) * half;
    let cy = UctDir { al: ay_l, ar: one - ay_l, dl: dlw.max(dle), dr: drw.max(dre) };
    let emf = uct_master_emf(&cx, &cy, vbar_x, vbar_y, by_e, by_w, bx_n, bx_s);
    (end_trace(), vec![("emf".to_string(), "emf".into(), emf.node())])
}


/// the RELATIVISTIC UCT-HLLD edge EMF (RMHD). built from the WAVE-SUM dissipative flux (Mignone &
/// Del Zanna 2020 Eq. 39 + MUB09 star states), NOT the classical coefficient form (Eq. 44) — that
/// bakes in a CLASSICAL velocity-chi that is invalid relativistically and was VERIFIED wrong
/// (telescoping test, 2026-06-24). derivation + paper proof in `literature/uct_algorithm.md` 3.5.
///
/// the EMF is the centered advection minus the per-direction dissipative flux Phi:
/// ```text
///   E_z = -1/2 (v_x^E B_y^E + v_x^W B_y^W) + 1/2 (v_y^N B_x^N + v_y^S B_x^S) + Phi_x - Phi_y
/// ```
/// where Phi is the EXACT HLLD induction-flux dissipation over the ACTUAL star fields (M&DZ Eq. 39):
/// ```text
///   Phi = 1/2 [ |lambda^L|(B_t^{sL}-B_t^L) + |lambda^{sL}|(B_c-B_t^{sL})
///             + |lambda^{sR}|(B_t^{sR}-B_c) + |lambda^R|(B_t^R-B_t^{sR}) ]
/// ```
/// `B_t^{sL,sR}` single-star (`hlld_rmhd_states.bstar`), `B_c` contact (`.bc`); `lambda` fast (`lam`),
/// `lambda^{s}` Alfven (`alf`). BOUNDED by construction (field differences times |speed| — no ratio,
/// no 1/B_t, no floor, no clamp). reduces EXACTLY to `-F_hlld_rmhd[B_t]` in 1D (verified to machine
/// precision). the STAGGERED transverse face fields are the Riemann L/R (CT consistency, M&DZ p.8) so
/// Phi damps the staggered checkerboard; cell velocities/rho/pre are the 2-cell edge average. gated on
/// `success`: where the secant fails, Phi -> the finite HLL dissipation (the lam are always finite).
pub fn rmhd_edge_emf_uct_hlld_gv(ndim: usize, g1: usize, g2: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    gv_register_field("e_rho", "rho");
    gv_register_field("e_vp1", "vel_p1");
    gv_register_field("e_vp2", "vel_p2");
    gv_register_field("e_vout", "vel_out");
    gv_register_field("e_pre", "pre");
    gv_register_field("e_bp1", "bcell_p1");
    gv_register_field("e_bp2", "bcell_p2");
    gv_register_field("e_bout", "bcell_out");
    gv_register_field("e_bface_a", "bface_a");
    gv_register_field("e_bface_b", "bface_b");
    let cm = |axes: &[usize]| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        for &ax in axes {
            o[ax] = -1;
        }
        o
    };
    let zero = vec![0i32; ndim];
    let half = Gv::from_f64(0.5);
    let eps = Gv::from_f64(1.0e-30);
    let zero_g = Gv::ZERO;
    let gamma = Gv::scalar("gamma");
    let eos = IdealGas { gamma };
    let avg2 = |a: Gv, b: Gv| (a + b) * half;
    let ne = zero.clone();
    let nw = cm(&[g1]);
    let se = cm(&[g2]);
    let sw = cm(&[g1, g2]);
    // PLM-reconstruct a CELL field to a face (same theta + limiter as the gas flux). THE FIX (M&DZ
    // §3.6 EXACT recipe): the wave-sum's per-face Riemann must use the SAME reconstructed L/R states
    // the gas flux solves, NOT a 2-cell-averaged single edge Riemann — cell states make the EMF fan
    // inconsistent with the flux at sharp reconstruction (the plm=2 checkerboard, the relativistic D1).
    let theta = Gv::scalar("theta");
    let recon_cell = |key: &str, rt: &str, base: &[i32], naxis: usize, sign: f64| -> Gv {
        let off = |d: i32| -> Vec<i32> {
            let mut o = base.to_vec();
            o[naxis] += d;
            o
        };
        let q0 = gv_field_at(key, rt, ndim, base);
        let qm = gv_field_at(key, rt, ndim, &off(-1));
        let qp = gv_field_at(key, rt, ndim, &off(1));
        let a = q0 - qm;
        let b = qp - q0;
        let mm = minmod3(a * theta, half * (a + b), b * theta);
        let slope = Gv::select(theta.cmp_lt(Gv::ZERO), van_leer(a, b), mm);
        q0 + Gv::from_f64(0.5 * sign) * slope
    };
    // RMHD prim reconstructed to ONE side of a face; the face-NORMAL and the dissipated TRANSVERSE B
    // are both overridden with the staggered div-free values (constant-B_n Riemann; the transverse is
    // the staggered face that gets dissipated). L uses sign +1 (toward +naxis face), R uses -1.
    let prim_face = |base: &[i32], naxis: usize, sign: f64, n_idx: usize, bn: Gv, t_idx: usize, bt: Gv| -> MhdPrim<Gv, 3> {
        let r = |key: &str, rt: &str| recon_cell(key, rt, base, naxis, sign);
        let rho = r("e_rho", "rho");
        let pre = r("e_pre", "pre");
        let v = [r("e_vp1", "vel_p1"), r("e_vp2", "vel_p2"), r("e_vout", "vel_out")];
        let mut b = [r("e_bp1", "bcell_p1"), r("e_bp2", "bcell_p2"), r("e_bout", "bcell_out")];
        b[n_idx] = bn;
        b[t_idx] = bt;
        MhdPrim::<Gv, 3> { hydro: Prim { rho, vel: Tensor::new(v), pre }, mag: Tensor::new(b) }
    };
    // staggered face B reconstructed to the EDGE — the DISSIPATED transverse fields in the wave-sum.
    let bx_n = recon_face_to_edge(ndim, theta, "e_bface_a", "bface_a", &zero, g2, -1.0);
    let bx_s = recon_face_to_edge(ndim, theta, "e_bface_a", "bface_a", &se, g2, 1.0);
    let by_w = recon_face_to_edge(ndim, theta, "e_bface_b", "bface_b", &nw, g1, 1.0);
    let by_e = recon_face_to_edge(ndim, theta, "e_bface_b", "bface_b", &zero, g1, -1.0);
    // the single div-free NORMAL B at each face (the un-reconstructed staggered FACE value).
    let bx_n_face = gv_field_at("e_bface_a", "bface_a", ndim, &ne);
    let bx_s_face = gv_field_at("e_bface_a", "bface_a", ndim, &se);
    let by_w_face = gv_field_at("e_bface_b", "bface_b", ndim, &nw);
    let by_e_face = gv_field_at("e_bface_b", "bface_b", ndim, &ne);
    // the wave-sum dissipative flux Phi (M&DZ Eq. 39) for a Riemann whose transverse component is `t`,
    // with staggered endpoints `bt_l`,`bt_r` and the single-/double-star fields from `st`. gated on
    // `success` -> HLL dissipation (NaN-safe true select; the HLL branch uses only the finite lam).
    let wave_sum = |st: &HlldStates<Gv, 3>, t: usize, bt_l: Gv, bt_r: Gv| -> Gv {
        let phi_hlld = half
            * (st.lam[0].abs() * (st.bstar[0][t] - bt_l)
                + st.alf[0].abs() * (st.bc[t] - st.bstar[0][t])
                + st.alf[1].abs() * (st.bstar[1][t] - st.bc[t])
                + st.lam[1].abs() * (bt_r - st.bstar[1][t]));
        let ap = zero_g.max(st.lam[1]);
        let am = zero_g.max(zero_g - st.lam[0]);
        let phi_hll = (ap * am / (ap + am + eps)) * (bt_r - bt_l);
        Gv::select(st.success.cmp_gt(half), phi_hlld, phi_hll)
    };
    // x-Riemann dissipates B_y (component 1), normal B_x (0). PER-FACE: North (NW->NE) and South
    // (SW->SE), each from reconstructed states (matching the gas flux); Phi_x = 1/2(Phi_N + Phi_S)
    // (M&DZ Eq. 34 edge average). normal B_x = the staggered FACE B_x; transverse B_y = by_w / by_e.
    let xn_l = prim_face(&nw, g1, 1.0, 0, bx_n_face, 1, by_w);
    let xn_r = prim_face(&ne, g1, -1.0, 0, bx_n_face, 1, by_e);
    let st_xn = hlld_rmhd_states(&Rmhd, &eos, &xn_l, &xn_r, &Tensor::<Gv, 3>::unit(0));
    let xs_l = prim_face(&sw, g1, 1.0, 0, bx_s_face, 1, by_w);
    let xs_r = prim_face(&se, g1, -1.0, 0, bx_s_face, 1, by_e);
    let st_xs = hlld_rmhd_states(&Rmhd, &eos, &xs_l, &xs_r, &Tensor::<Gv, 3>::unit(0));
    let phi_x = avg2(wave_sum(&st_xn, 1, by_w, by_e), wave_sum(&st_xs, 1, by_w, by_e));
    // y-Riemann dissipates B_x (component 0), normal B_y (1). PER-FACE: West (SW->NW) and East
    // (SE->NE). normal B_y = the staggered FACE B_y; transverse B_x = bx_s / bx_n.
    let yw_l = prim_face(&sw, g2, 1.0, 1, by_w_face, 0, bx_s);
    let yw_r = prim_face(&nw, g2, -1.0, 1, by_w_face, 0, bx_n);
    let st_yw = hlld_rmhd_states(&Rmhd, &eos, &yw_l, &yw_r, &Tensor::<Gv, 3>::unit(1));
    let ye_l = prim_face(&se, g2, 1.0, 1, by_e_face, 0, bx_s);
    let ye_r = prim_face(&ne, g2, -1.0, 1, by_e_face, 0, bx_n);
    let st_ye = hlld_rmhd_states(&Rmhd, &eos, &ye_l, &ye_r, &Tensor::<Gv, 3>::unit(1));
    let phi_y = avg2(wave_sum(&st_yw, 0, bx_s, bx_n), wave_sum(&st_ye, 0, bx_s, bx_n));
    // centered advective velocities (2-cell averages straddling the edge).
    let vp1 = |o: &[i32]| gv_field_at("e_vp1", "vel_p1", ndim, o);
    let vp2 = |o: &[i32]| gv_field_at("e_vp2", "vel_p2", ndim, o);
    let vx_w = avg2(vp1(&nw), vp1(&sw));
    let vx_e = avg2(vp1(&ne), vp1(&se));
    let vy_s = avg2(vp2(&sw), vp2(&se));
    let vy_n = avg2(vp2(&nw), vp2(&ne));
    // E_z = -1/2(v_x^E B_y^E + v_x^W B_y^W) + 1/2(v_y^N B_x^N + v_y^S B_x^S) + Phi_x - Phi_y.
    let emf = zero_g - half * (vx_e * by_e + vx_w * by_w)
        + half * (vy_n * bx_n + vy_s * bx_s)
        + phi_x
        - phi_y;
    (end_trace(), vec![("emf".to_string(), "emf".into(), emf.node())])
}


/// the RK2 edge-EMF save `e_n = e` (pointwise copy; the generic 2-buffer copy the runtime also
/// reuses for the bcell^n snapshot). write root == the read field node.
pub fn rmhd_save_efield_gv() -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let e = Gv::field("e", "e");
    (end_trace(), vec![("e_n".to_string(), "e_n".into(), e.node())])
}


/// the RK2 edge-EMF time-average `e = 0.5*(e + e_n)`, in-place on e. mirror of
/// `rmhd::rmhd_average_efield`.
pub fn rmhd_average_efield_gv() -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let e = Gv::field("e", "e");
    let en = Gv::field("e_n", "e_n");
    let e_new = Gv::from_f64(0.5) * (e + en);
    (end_trace(), vec![("e_new".to_string(), "e".into(), e_new.node())])
}
