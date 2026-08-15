// =============================================================================
// ct_emf.rs
//
// rmhd constrained-transport stack: staggered curl, edge-emf (uct hll/hllc/hlld), face->cell b, cell-b predictors.
// =============================================================================

use super::*;

/// chart-generic ADM data (lapse alpha, sqrt(det gamma), full shift beta) at a world position, for
/// the curved-spacetime CT stack. one (spacetime, coords) match selects the KS-family metric of the
/// active chart. the shift is the FULL 3-vector: a chart with a transverse in-plane shift (cartesian
/// carries it on every axis, cylindrical on r and z) contributes both plane components to the
/// transport EMF; the spherical polar chart has beta only along r, so its second plane component
/// vanishes identically and the generic form reduces to the single-shift spherical EMF.
fn gr_adm_at(spacetime: Spacetime, coords: Coords, x: Tensor<Gv, 3>) -> (Gv, Gv, Tensor<Gv, 3>) {
    use symbi_geometry::Metric;
    // spinning kerr on the CARTESIAN chart: the rank-1 kerr-schild update with the
    // oblate-spheroidal radius; non-diagonal gamma + shift on every axis.
    fn adm<M: Metric<Gv, 3>>(m: &M, x: Tensor<Gv, 3>) -> (Gv, Gv, Tensor<Gv, 3>) {
        (m.lapse(x), m.sqrt_det_gamma(x), m.shift(x))
    }
    with_ks_metric!(spacetime, coords, "the GR CT stack", |m| adm(&m, x))
}

/// the chart-generic spatial metric (gamma + gamma^{-1}) at a world position — the tetrad-frame
/// HLLD fan needs the full metric beyond the ADM scalars. same (spacetime, coords) selection as
/// [`gr_adm_at`]; the orthonormal_basis(dir) gram-schmidt of this gamma is the tetrad.
fn gr_spatial_metric_at(
    spacetime: Spacetime,
    coords: Coords,
    x: Tensor<Gv, 3>,
) -> SpatialMetric<Gv, 3> {
    use symbi_geometry::Metric;
    // spinning kerr on the CARTESIAN chart: the rank-1 kerr-schild update with the
    // oblate-spheroidal radius; non-diagonal gamma + shift on every axis.
    fn adm<M: Metric<Gv, 3>>(m: &M, x: Tensor<Gv, 3>) -> SpatialMetric<Gv, 3> {
        SpatialMetric::new(
            Gamma::new(m.spatial_metric(x)),
            GammaInv::new(m.spatial_metric_inv(x)),
        )
    }
    with_ks_metric!(spacetime, coords, "the GR CT stack", |m| adm(&m, x))
}

/// the poloidal CT plane's in-plane physical components `(p1, p2)` in the cyclic order that fixes
/// the corner-EMF sign, and the grid axes `(g1, g2) = (pos(p1), pos(p2))` carrying them. the
/// out-of-plane component is `k = 3 - a0 - a1`; `p1 = (k+1) % 3`, `p2 = (k+2) % 3`. contiguous
/// identity axes (spherical/cartesian/disk `[0,1]`) give `(0, 1, 0, 1)`, so the shift indexing and
/// the +/-2 face reconstruction address grid axes 0/1 directly; the GAPPED cylindrical `(R, z)` set
/// `[0,2]` gives `(2, 0, 1, 0)` — phi is out-of-plane, `p1 = z` on grid axis 1, `p2 = R` on axis 0 —
/// so the shift pairs with the right velocity component and the reconstruction reads the axis whose
/// transverse halo the staggered field actually carries. matches the runtime `ct_edges` mapping.
fn gr_ct_plane(axes: &[usize]) -> (usize, usize, usize, usize) {
    let k = 3 - axes[0] - axes[1];
    let (p1, p2) = ((k + 1) % 3, (k + 2) % 3);
    let pos = |c: usize| {
        axes.iter()
            .position(|&a| a == c)
            .expect("an in-plane component is a grid axis")
    };
    (p1, p2, pos(p1), pos(p2))
}

/// the world position (3-vector) of a 2D CT-plane point: the two in-plane grid axes `axes = [a0, a1]`
/// take the coordinate values `(p0, p1)`; the ungridded slot takes the chart default (the equatorial
/// pi/2 for the spherical polar angle, zero elsewhere). one builder for every corner / face / cell
/// metric point of the poloidal CT, so the metric reads the true world position of each chart.
fn gr_plane_pos(coords: Coords, axes: &[usize], p0: Gv, p1: Gv) -> Tensor<Gv, 3> {
    Tensor::<Gv, 3>::new(std::array::from_fn(|c| {
        if c == axes[0] {
            p0
        } else if c == axes[1] {
            p1
        } else {
            gv_ungridded_slot(coords, c)
        }
    }))
}

/// the gardiner & stone CT-contact edge EMF (the SOFT-SIGN blend), carrier-generic at S=Gv.
/// a pointwise function of the 4 face EMFs, 4 cell-corner
/// EMFs, and 4 density fluxes: `s = f/(|f|+eps)`; `0.5*((a+b) + s*(a-b))`, transitions
/// continuously through f=0 (= a hard 3-way sign in the |f|>>eps limit). div(B) unaffected.
fn ct_contact_emf_gv(face_e: [Gv; 4], cell_e: [Gv; 4], dflux: [Gv; 4]) -> Gv {
    let [en, es, ee, ew] = face_e;
    let [ene, enw, ese, esw] = cell_e;
    let [fnf, fs, fe, fw] = dflux;
    let two = Gv::from_f64(2.0);
    let flux_scale = fnf.abs().max(fs.abs()).max(fe.abs()).max(fw.abs());
    let eps = Gv::from_f64(32.0 * f64::EPSILON) * flux_scale;
    let eavg = Gv::from_f64(0.25) * (es + en + ew + ee);
    let soft = |f: Gv, a: Gv, b: Gv| {
        let denominator = f.abs() + eps;
        let nonzero = denominator.cmp_gt(Gv::ZERO);
        let divisor = Gv::select(nonzero, denominator, Gv::ONE);
        let s = Gv::select(nonzero, f / divisor, Gv::ZERO);
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
        (0..3)
            .map(|ax| gv_axis_face_at(ax, spacing[ax], off[ax]))
            .collect()
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
    let inv_pref =
        Gv::ONE / (gv_scale_factor(coords, p1, &center) * gv_scale_factor(coords, p2, &center));
    let inv_dx_p1 =
        Gv::ONE / (gv_axis_face_at(p1, spacing[p1], 1) - gv_axis_face_at(p1, spacing[p1], 0));
    let inv_dx_p2 =
        Gv::ONE / (gv_axis_face_at(p2, spacing[p2], 1) - gv_axis_face_at(p2, spacing[p2], 0));
    CtCurlMetricGv {
        h1_here,
        h1_p2,
        h2_here,
        h2_p1,
        inv_pref,
        inv_dx_p1,
        inv_dx_p2,
    }
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
    (
        end_trace(),
        vec![("b_new".to_string(), "b".into(), b_new.node())],
    )
}

/// the 2.5D CARTESIAN OHMIC RESISTIVE edge EMF: adds the anomalous/ohmic contribution `eta * J_z`
/// to the out-of-plane edge EMF `ez` (efield[0]) IN PLACE, where `J_z = dB_y/dx - dB_x/dy` is the
/// current on the z-edge from the staggered face field (`bx` = bface[0], `by` = bface[1]). the
/// difference stencil is the ADJOINT of the `E -> B` induction curl (`rmhd_ct_curl_2d_dir_gv`): its
/// `+1` neighbor offsets become the `-1` offsets here, so the composed `curl(eta * curl(B))` is a
/// NEGATIVE-definite discrete laplacian — the field DIFFUSES (decays as `exp(-eta k^2 t)`), never
/// anti-diffuses. the same div-B-clean curl then consumes `ez`, so the update carries no new
/// monopole (`div(curl) = 0` — the existing symbolic proof covers it). `eta = 0` is an exact no-op.
pub fn rmhd_resistive_emf_2d_gv() -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let ez = Gv::field("ez", "ez");
    let eta = Gv::scalar("eta");
    let idx = Gv::scalar("idx");
    let idy = Gv::scalar("idy");
    let bx = Gv::field("bx", "bx");
    let by = Gv::field("by", "by");
    let bx_jm = gv_field_at("bx", "bx", 2, &[0, -1]); // B_x at the neighbor below in y
    let by_im = gv_field_at("by", "by", 2, &[-1, 0]); // B_y at the neighbor behind in x
    let jz = idx * (by - by_im) - idy * (bx - bx_jm);
    let ez_new = ez + eta * jz;
    (
        end_trace(),
        vec![("ez_new".to_string(), "ez".into(), ez_new.node())],
    )
}

/// the 3D CARTESIAN OHMIC RESISTIVE edge EMF along edge `dir`: adds `eta * J_dir` to that edge's EMF
/// (`emf` = efield[slot]) IN PLACE, where `J_dir = dB_p2/dx_p1 - dB_p1/dx_p2` is the current on the
/// dir-edge from the two transverse face components (`b_p1` = bface[p1], `b_p2` = bface[p2],
/// p1=(dir+1)%3, p2=(dir+2)%3). the `-1` difference offsets are the ADJOINT of the per-dir induction
/// curl `rmhd_ct_curl_3d_dir_gv` (whose `+1` reads they mirror), so the composed
/// `curl(eta * curl(B))` is the negative-definite discrete laplacian — the field diffuses, never
/// anti-diffuses. the same div-B-clean 3D curl consumes the augmented EMF. `eta = 0` is a no-op.
pub fn rmhd_resistive_emf_3d_dir_gv(dir: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let p1 = (dir + 1) % 3;
    let p2 = (dir + 2) % 3;
    let emf = Gv::field("emf", "emf");
    let eta = Gv::scalar("eta");
    let id_p1 = Gv::scalar("id_p1");
    let id_p2 = Gv::scalar("id_p2");
    let b_p1 = Gv::field("b_p1", "b_p1");
    let b_p2 = Gv::field("b_p2", "b_p2");
    let back = |ax: usize| -> [i32; 3] {
        let mut o = [0, 0, 0];
        o[ax] = -1;
        o
    };
    let b_p1_m = gv_field_at("b_p1", "b_p1", 3, &back(p2)); // dB_p1/dx_p2, backward
    let b_p2_m = gv_field_at("b_p2", "b_p2", 3, &back(p1)); // dB_p2/dx_p1, backward
    let j = id_p1 * (b_p2 - b_p2_m) - id_p2 * (b_p1 - b_p1_m);
    let emf_new = emf + eta * j;
    (
        end_trace(),
        vec![("emf_new".to_string(), "emf".into(), emf_new.node())],
    )
}

/// the 2.5D cylindrical r-z (axisymmetric) CT curl from the single out-of-plane edge EMF
/// E_phi (efield[0]), in-place on `b` (bface[dir]). DERIVED from the 3D cyl curl restricted
/// to E_phi with d/dphi = 0 (verified to reproduce the 3D-cyl ct_curl_metric formula):
///   dir=0 (B_r, r-face):  dB_r/dt = +d_z E_phi            (z = grid axis 1; flat, no metric)
///   dir=1 (B_z, z-face):  dB_z/dt = -(1/r) d_r(r E_phi)   (r = grid axis 0; cylindrical metric)
/// r is computed per-cell from gv_axis_face_at(0, ..) (the geom scalars x_lo_0/dx_0). E_phi is
/// the corner field at offsets [0,0]/[+grid]. div(B)=0 preserved by the discrete d-of-d.
pub fn rmhd_ct_curl_cyl_rz_gv(
    dir: usize,
    spacing: &[Spacing],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
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
        let inv_dz =
            Gv::ONE / (gv_axis_face_at(1, spacing[1], 1) - gv_axis_face_at(1, spacing[1], 0));
        let ez_zp = gv_field_at("ez", "ez", 2, &[0, 1]);
        b + dt * inv_dz * (ez_zp - ez)
    } else {
        // dB_z/dt = -(1/r_c) d_r(r E_phi) : the cylindrical metric on the radial derivative.
        // r at the cell's two r-faces (= the corner radii bounding this z-face), cell-center r_c.
        // minus because (curl E)_z = +(1/r) d_r(r E_phi) and dB/dt = -curl(E) — OPPOSITE sign to
        // the spherical-poloidal B_theta update (curl_theta carries its own minus). the plus form
        // leaks div(B) secularly (d/dt div(B) != 0); the rotor div(B) blows to O(1) in one step.
        let inv_dr =
            Gv::ONE / (gv_axis_face_at(0, spacing[0], 1) - gv_axis_face_at(0, spacing[0], 0));
        let r_lo = gv_axis_face_at(0, spacing[0], 0);
        let r_hi = gv_axis_face_at(0, spacing[0], 1);
        let r_c = (r_lo + r_hi) * Gv::from_f64(0.5);
        let ez_rp = gv_field_at("ez", "ez", 2, &[1, 0]);
        b - dt * (Gv::ONE / r_c) * inv_dr * (r_hi * ez_rp - r_lo * ez)
    };
    (
        end_trace(),
        vec![("b_new".to_string(), "b".into(), b_new.node())],
    )
}

/// the 2.5D CYLINDRICAL r-z OHMIC RESISTIVE edge EMF: adds `eta * J_phi` to the out-of-plane edge
/// EMF `ephi` (efield[0]) IN PLACE, where `J_phi` is the MIMETIC ADJOINT of the cylindrical induction
/// curl `rmhd_ct_curl_cyl_rz_gv` — so `-curl(eta * J)` is a negative-definite laplacian (stable
/// diffusion). metric-free form, from `J = W_E^-1 C^T W_F` with the natural cyl weights
/// `w_r \propto r_edge, w_z \propto r_c, w_E \propto r_edge`, whose r-factors cancel; the discrete
/// adjoint identity `<C E, B>_F = <E, J B>_E` pins it to machine precision. `B_r` = bface[0],
/// `B_z` = bface[1]. `eta = 0` is an exact no-op.
pub fn rmhd_resistive_emf_cyl_rz_gv(
    spacing: &[Spacing],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let ephi = Gv::field("ephi", "ephi");
    let eta = Gv::scalar("eta");
    // register the geom scalars in the canonical order (both r and z faces), matching the curl ABI.
    for ax in 0..2 {
        let _ = gv_axis_face_at(ax, spacing[ax], 0);
        let _ = gv_axis_face_at(ax, spacing[ax], 1);
    }
    let inv_dr = Gv::ONE / (gv_axis_face_at(0, spacing[0], 1) - gv_axis_face_at(0, spacing[0], 0));
    let inv_dz = Gv::ONE / (gv_axis_face_at(1, spacing[1], 1) - gv_axis_face_at(1, spacing[1], 0));
    let br = Gv::field("br", "br");
    let bz = Gv::field("bz", "bz");
    let br_zm = gv_field_at("br", "br", 2, &[0, -1]); // B_r at the neighbor below in z (axis 1)
    let bz_rm = gv_field_at("bz", "bz", 2, &[-1, 0]); // B_z at the neighbor behind in r (axis 0)
    // J_phi = (curl B)_phi = dB_r/dz - dB_z/dr (backward differences). this sign makes -curl(eta J) a
    // NEGATIVE-definite laplacian (magnetic energy decays); the flipped sign would grow it. matches
    // the cartesian resistive convention J = +(curl B).
    let jphi = inv_dz * (br - br_zm) - inv_dr * (bz - bz_rm);
    let ephi_new = ephi + eta * jphi;
    (
        end_trace(),
        vec![("ephi_new".to_string(), "ephi".into(), ephi_new.node())],
    )
}

/// the COVARIANT 2.5D orthogonal-chart OHMIC RESISTIVE edge EMF: adds `eta * J_out` to the
/// out-of-plane corner EMF, where `J_out = (1/(h_i h_j)) [d_i(h_j B_j) - d_j(h_i B_i)]` is the physical
/// curl of the poloidal field in the running orthogonal chart, `h_i` the chart's lame scale factors
/// (`Metric::scale_factors`). this is the DEC codifferential — the mimetic ADJOINT of the induction
/// curl — written through the scale factors, so ONE kernel covers every 2.5D orthogonal chart:
/// cyl r-z (`h = (1, 1)`) recovers the metric-free `d_z B_r - d_r B_z`; cyl r-phi and spherical r-theta
/// (`h_2 = r`) grow the `(1/r) d_r(r .)` factor. the geometry-agnostic adjoint reference validates each
/// chart; `-curl(eta J)` is negative-definite (magnetic energy decays). `eta = 0` is an exact no-op.
/// `B_i` = bface[i]; scale factors are sampled at the staggered face/corner positions of each term.
pub fn rmhd_resistive_emf_ortho_gv(
    coords: Coords,
    spacing: &[Spacing],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let e = Gv::field("e", "e"); // the out-of-plane corner EMF
    let eta = Gv::scalar("eta");
    // pin the geom scalars in canonical order (both axes' faces), matching the curl ABI.
    for ax in 0..2 {
        let _ = gv_axis_face_at(ax, spacing[ax], 0);
        let _ = gv_axis_face_at(ax, spacing[ax], 1);
    }
    let dx0 = gv_axis_face_at(0, spacing[0], 1) - gv_axis_face_at(0, spacing[0], 0);
    let dx1 = gv_axis_face_at(1, spacing[1], 1) - gv_axis_face_at(1, spacing[1], 0);
    let inv_dx0 = Gv::ONE / dx0;
    let inv_dx1 = Gv::ONE / dx1;
    let b0 = Gv::field("b0", "b0"); // bface[0], normal to in-plane axis 0
    let b1 = Gv::field("b1", "b1"); // bface[1], normal to in-plane axis 1
    let b0_jm = gv_field_at("b0", "b0", 2, &[0, -1]); // B_0 one cell back in axis 1
    let b1_im = gv_field_at("b1", "b1", 2, &[-1, 0]); // B_1 one cell back in axis 0
    // the coordinate centroid; the corner (edge) sits half a cell below it on each axis, and the
    // staggered face positions are fixed coordinate offsets from it (scale factors read there).
    let geo = cell_geometry_gv(coords, spacing, &[0usize, 1usize], 2);
    let (c0, c1) = (geo.centroid[0], geo.centroid[1]);
    let half = Gv::from_f64(0.5);
    let sf = |x0: Gv, x1: Gv| crate::gv_viscous::scale_factors_at(coords, 2, &[x0, x1]);
    // prefactor 1/(h0 h1) at the corner (x0_face, x1_face).
    let hc = sf(c0 - half * dx0, c1 - half * dx1);
    let inv_h = Gv::ONE / (hc[0] * hc[1]);
    // d_0(h1 B1): B1 lives at (x0_center, x1_face); sample h1 there, difference along axis 0.
    let h1_ij = sf(c0, c1 - half * dx1)[1];
    let h1_im = sf(c0 - dx0, c1 - half * dx1)[1];
    let d0 = inv_dx0 * (h1_ij * b1 - h1_im * b1_im);
    // d_1(h0 B0): B0 lives at (x0_face, x1_center); sample h0 there, difference along axis 1.
    let h0_ij = sf(c0 - half * dx0, c1)[0];
    let h0_jm = sf(c0 - half * dx0, c1 - dx1)[0];
    let d1 = inv_dx1 * (h0_ij * b0 - h0_jm * b0_jm);
    let jout = inv_h * (d0 - d1);
    // E += eta*jout: the adjoint sign for a RIGHT-HANDED (axis0, axis1, out-of-plane) triple — cyl
    // r-phi `(r, phi, z)` and spherical r-theta `(r, theta, phi)`. (cyl r-z's `(r, z, phi)` triple is
    // left-handed and uses its own oppositely-signed kernel.) sign-pinned to make -curl(eta J)
    // negative-definite so the magnetic energy decays.
    let e_new = e + eta * jout;
    (
        end_trace(),
        vec![("e_new".to_string(), "e".into(), e_new.node())],
    )
}

/// the 2.5D cylindrical r-phi DISK CT curl from the single out-of-plane edge EMF E_z
/// (efield[0]), in-place on `b` (bface[dir]). DERIVED from the cyl curl restricted to E_z with
/// d/dz = 0 (verified to preserve the staggered cyl div(B) = (1/r)d_r(r B_r) + (1/r)d_phi B_phi):
///   dir=0 (B_r, r-face):   dB_r/dt   = -(1/r) d_phi E_z   (phi = grid axis 1; 1/r metric, r = the r-face radius)
///   dir=1 (B_phi, phi-face): dB_phi/dt = +d_r E_z         (r = grid axis 0; flat, NO metric — mirror of r-z)
/// r is the r-FACE radius (where B_r lives) via gv_axis_face_at(0, .., 0). E_z is the corner field
/// at offsets [0,0]/[+grid]. div(B)=0 preserved by the discrete d-of-d (mixed partials cancel).
pub fn rmhd_ct_curl_cyl_rphi_gv(
    dir: usize,
    spacing: &[Spacing],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
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
        let inv_dphi =
            Gv::ONE / (gv_axis_face_at(1, spacing[1], 1) - gv_axis_face_at(1, spacing[1], 0));
        let ez_phip = gv_field_at("ez", "ez", 2, &[0, 1]);
        b - dt * (Gv::ONE / r_face) * inv_dphi * (ez_phip - ez)
    } else {
        // dB_phi/dt = +d_r E_z : finite difference along grid axis 0 (r). NO metric (the phi-comp
        // of the cyl curl is metric-free; the discrete d-of-d still cancels — proven).
        let inv_dr =
            Gv::ONE / (gv_axis_face_at(0, spacing[0], 1) - gv_axis_face_at(0, spacing[0], 0));
        let ez_rp = gv_field_at("ez", "ez", 2, &[1, 0]);
        b + dt * inv_dr * (ez_rp - ez)
    };
    (
        end_trace(),
        vec![("b_new".to_string(), "b".into(), b_new.node())],
    )
}

/// the 2.5D SPHERICAL (r-theta plane, out-of-plane phi) CT curl from the single corner EMF
/// E_phi (efield[0]), in-place on `b` (bface[dir]). faraday dB/dt = -curl E with E = E_phi phi-hat
/// (axisymmetric) gives the spherical-metric in-plane update:
///   dir=0 (B_r,   r-face):     dB_r/dt   = -(1/(r_f sin th_c)) d_th(sin th * E_phi)   (th = grid axis 1)
///   dir=1 (B_th, theta-face):  dB_th/dt  = +(1/r_c) d_r(r * E_phi)                     (r  = grid axis 0)
/// r_f is the r-FACE radius (where B_r lives); r_c / th_c are the staggered cell centers. mirrors
/// `rmhd_ct_curl_cyl_rz_gv` with the added sin(theta) area weight on the B_r update (and the
/// opposite B_theta sign vs the cylinder's B_z). div(B)=0 preservation for a nontrivial POLOIDAL
/// (B_r, B_theta) field is pinned by tests/rmhd_ct_curl_2d_sph_poloidal_divb.rs: B = curl(A_phi)
/// through this kernel, area-weighted div machine-zero before AND after a curl(E_phi) step.
pub fn rmhd_ct_curl_2d_sph_gv(
    dir: usize,
    spacing: &[Spacing],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
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
        b - dt
            * (Gv::ONE / (r_f * th_c.sin()))
            * inv_dth
            * (th_hi.sin() * ez_thp - th_lo.sin() * ez)
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
    (
        end_trace(),
        vec![("b_new".to_string(), "b".into(), b_new.node())],
    )
}

/// the 2.5D CURVED-SPACETIME (r-theta plane) CT curl from the single DENSITIZED corner EMF
/// `Etilde_phi` (efield[0]), in-place on the PHYSICAL `b` (bface[dir]). the update evolves the
/// densitized field Btilde^i = sqrt(gamma) B^i with the COORDINATE curl — the form whose
/// discrete divergence d_i Btilde^i telescopes to zero for ANY per-face constant weights —
/// then divides back by this face's own weight `w = sqrt(gamma)(face center) x coordinate
/// length` so the stored value stays the physical B every consumer reads. with the two in-plane
/// grid axes `axes = [a0, a1]` (spherical [r, theta], cartesian [x, y], cylindrical r-z [R, z]):
///   dir=0 (B_a0, a0-face):  B_a0 -= dt (Etilde(a1_hi) - Etilde(a1_lo)) / (sqrtg(a0_f, a1_c) da1)
///   dir=1 (B_a1, a1-face):  B_a1 += dt (Etilde(a0_hi) - Etilde(a0_lo)) / (sqrtg(a0_c, a1_f) da0)
/// the signs are the 2D-curl antisymmetry (chart-independent). sqrt(gamma) comes from the metric
/// trait at the face center via [`gr_adm_at`] / [`gr_plane_pos`], so one builder serves every
/// KS-family chart. div preservation for a nontrivial poloidal field is pinned by
/// tests/rmhd_ct_curl_2d_sph_gr_divb.rs (the w-weighted divergence, machine-zero).
pub fn rmhd_ct_curl_2d_sph_gr_gv(
    dir: usize,
    spacetime: Spacetime,
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let b = Gv::field("b", "b");
    let ez = Gv::field("ez", "ez"); // the out-of-plane DENSITIZED corner EMF Etilde
    let dt = Gv::scalar("dt");
    // positional scalar ABI mirror of the flat curl: pin [x_lo_0, dx_0, x_lo_1, dx_1] up front.
    for ax in 0..2 {
        let _ = gv_axis_face_at(ax, spacing[ax], 0);
        let _ = gv_axis_face_at(ax, spacing[ax], 1);
    }
    let half = Gv::from_f64(0.5);
    let sqrtg = |p0: Gv, p1: Gv| -> Gv {
        gr_adm_at(spacetime, coords, gr_plane_pos(coords, axes, p0, p1)).1
    };
    let b_new = if dir == 0 {
        // B_a0 lives on the low a0-face; a1_lo/a1_hi are the bounding corners of the a1 axis.
        let a0_f = gv_axis_face_at(0, spacing[0], 0);
        let a1_lo = gv_axis_face_at(1, spacing[1], 0);
        let a1_hi = gv_axis_face_at(1, spacing[1], 1);
        let a1_c = (a1_lo + a1_hi) * half;
        let w = sqrtg(a0_f, a1_c) * (a1_hi - a1_lo);
        let ez_a1p = gv_field_at("ez", "ez", 2, &[0, 1]);
        b - dt * (ez_a1p - ez) / w
    } else {
        // B_a1 lives on the low a1-face; a0_lo/a0_hi are the bounding corners of the a0 axis.
        let a0_lo = gv_axis_face_at(0, spacing[0], 0);
        let a0_hi = gv_axis_face_at(0, spacing[0], 1);
        let a1_f = gv_axis_face_at(1, spacing[1], 0);
        let a0_c = (a0_lo + a0_hi) * half;
        let w = sqrtg(a0_c, a1_f) * (a0_hi - a0_lo);
        let ez_a0p = gv_field_at("ez", "ez", 2, &[1, 0]);
        b + dt * (ez_a0p - ez) / w
    };
    (
        end_trace(),
        vec![("b_new".to_string(), "b".into(), b_new.node())],
    )
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
    (
        end_trace(),
        vec![("b_new".to_string(), "b".into(), b_new.node())],
    )
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
    // (if any) are carried cell-centered and untouched here (2.5D / 1.5D).
    let bf: Vec<Gv> = (0..ndim)
        .map(|c| Gv::field(&format!("bf_{c}"), &format!("bf_{c}")))
        .collect();
    let writes = (0..ndim)
        .map(|c| {
            let bcc_n =
                (bf[c] + gv_field_at(&format!("bf_{c}"), &format!("bf_{c}"), ndim, &off(c))) * half;
            (
                format!("bc_{c}_new"),
                format!("bc_{c}").into(),
                bcc_n.node(),
            )
        })
        .collect();
    (end_trace(), writes)
}

/// the CT face->cell B interpolation `bcell_c = 0.5*(bface_c + bface_c[+e_c])`, in place on the
/// in-plane cell B (mag rows only). the cell field is a DERIVED quantity — the arithmetic average of
/// its two bounding faces — used for reconstruction + the c2p magnetic-energy subtraction. NO energy
/// correction: `cons.nrg` (tau) already carries the magnetic energy and is conserved by the godunov
/// flux (the poynting term), so a `nrg += 0.5 d|bcell|^2` patch would DOUBLE-ACCOUNT it and does
/// not telescope.
pub fn rmhd_bcell_from_bface_gv(ndim: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let half = Gv::from_f64(0.5);
    let off = |ax: usize| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        o[ax] = 1;
        o
    };
    let bf: Vec<Gv> = (0..ndim)
        .map(|c| Gv::field(&format!("bf_{c}"), &format!("bf_{c}")))
        .collect();
    // interpolate the ndim in-plane components from their faces; out-of-plane components
    // (Bz in 2.5D) are untouched here.
    let bc_n: Vec<Gv> = (0..ndim)
        .map(|c| {
            (bf[c] + gv_field_at(&format!("bf_{c}"), &format!("bf_{c}"), ndim, &off(c))) * half
        })
        .collect();
    let writes: Vec<(String, FieldBind, NodeId)> = (0..ndim)
        .map(|c| {
            (
                format!("bc_{c}_new"),
                format!("bc_{c}").into(),
                bc_n[c].node(),
            )
        })
        .collect();
    (end_trace(), writes)
}

/// the CT edge EMF along edge axis `dir`, mirror of `rmhd::rmhd_edge_emf`: gather the 12
/// contact-formula inputs by integer-offset `load_at` (corner cell EMFs v_p2*b_p1 - v_p1*b_p2
/// at coord / -e_p1 / -e_p2 / -e_p1-e_p2; face EMFs from -bflux_a / +bflux_b; density fluxes),
/// then the `ct_contact_emf_gv` soft blend. 8 generic inputs the dispatch binds per edge.
pub fn rmhd_edge_emf_gv(
    ndim: usize,
    g1: usize,
    g2: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
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
    (
        end_trace(),
        vec![("emf".to_string(), "emf".into(), emf.node())],
    )
}

/// the CURVED-SPACETIME CT edge EMF for the 2.5D (r, theta) poloidal plane (edge = phi) — the
/// contact assembly of [`rmhd_edge_emf_gv`] producing the DENSITIZED corner EMF
/// `Etilde_phi = vtilde_theta Btilde_r - vtilde_r Btilde_theta` (vtilde = alpha v - beta,
/// Btilde = sqrt(gamma) B) that the GR curl consumes:
///   cell terms:  sqrt(gamma)|cell x [(alpha v_p2 - beta_p2) b_p1 - (alpha v_p1 - beta_p1) b_p2]
///   face terms:  alpha sqrt(gamma)|face x (raw mag-row flux) — the raw flux times the face
///                measure IS the exact coordinate-form flux vtilde^n Btilde^i - vtilde^i Btilde^n,
///                so the edge EMF stays flux-consistent (the UCT-checkerboard lesson).
/// every gather point carries the metric factors at ITS OWN position (cells at the arithmetic
/// centers matching the curl's prefactor convention, faces at (r_f, th_c)/(r_c, th_f)). the
/// density fluxes stay RAW — the soft-sign blend uses only their signs, and the alpha
/// sqrt(gamma) measure is positive. baked per (spacetime, spacing) for the (r, theta) grid.
pub fn rmhd_edge_emf_gr_gv(
    spacetime: Spacetime,
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let ndim = 2usize;
    let (pc1, pc2, g1, g2) = gr_ct_plane(axes);
    gv_register_field("edge_vp1", "vel_p1");
    gv_register_field("edge_vp2", "vel_p2");
    gv_register_field("edge_bp1", "bcell_p1");
    gv_register_field("edge_bp2", "bcell_p2");
    gv_register_field("edge_bflux_a", "bflux_a");
    gv_register_field("edge_bflux_b", "bflux_b");
    gv_register_field("edge_fden_p1", "fden_p1");
    gv_register_field("edge_fden_p2", "fden_p2");
    let cm = |axes: &[usize]| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        for &ax in axes {
            o[ax] = -1;
        }
        o
    };
    let zero = vec![0i32; ndim];
    let half = Gv::from_f64(0.5);
    // the thread coord is the CORNER (a0_f, a1_f); cell centers/faces at integer offsets.
    let a0_f = gv_axis_face_at(0, spacing[0], 0);
    let a1_f = gv_axis_face_at(1, spacing[1], 0);
    let a0_c =
        |o: i64| (gv_axis_face_at(0, spacing[0], o) + gv_axis_face_at(0, spacing[0], o + 1)) * half;
    let a1_c =
        |o: i64| (gv_axis_face_at(1, spacing[1], o) + gv_axis_face_at(1, spacing[1], o + 1)) * half;
    // (alpha, sqrt(gamma), beta_p1, beta_p2) at an in-plane point: the transport velocity of the
    // densitized EMF is alpha v - beta on EACH in-plane axis, the shift taken at the SAME physical
    // component as the paired velocity (beta_p1 = shift of the p1 component carried by vel_p1, etc).
    // spherical r maps to p1 with beta_theta = 0 on p2; cartesian carries both; cylindrical (R, z)
    // carries both with (p1, p2) = (z, R) (phi out-of-plane), so the shift addresses the true axes.
    let adm = |c0: Gv, c1: Gv| -> (Gv, Gv, Gv, Gv) {
        let (alpha, sqrtg, beta) = gr_adm_at(spacetime, coords, gr_plane_pos(coords, axes, c0, c1));
        (alpha, sqrtg, beta[pc1], beta[pc2])
    };
    // densitized cell EMF at the cell whose LOW corner offset is (o_a0, o_a1). the transport EMF is
    // sqrt(gamma) [ (alpha v_p2 - beta_p2) B_p1 - (alpha v_p1 - beta_p1) B_p2 ].
    let cell = |o: &[i32]| -> Gv {
        let (alpha, sqrtg, beta_p1, beta_p2) = adm(a0_c(o[0] as i64), a1_c(o[1] as i64));
        let vp1 = gv_field_at("edge_vp1", "vel_p1", ndim, o);
        let vp2 = gv_field_at("edge_vp2", "vel_p2", ndim, o);
        let bp1 = gv_field_at("edge_bp1", "bcell_p1", ndim, o);
        let bp2 = gv_field_at("edge_bp2", "bcell_p2", ndim, o);
        sqrtg * ((alpha * vp2 - beta_p2) * bp1 - (alpha * vp1 - beta_p1) * bp2)
    };
    let ene = cell(&zero);
    let enw = cell(&cm(&[g1]));
    let ese = cell(&cm(&[g2]));
    let esw = cell(&cm(&[g1, g2]));
    // densitized face EMFs: the a0-face fluxes (bflux_a, at (a0_f, a1_c)) and the a1-face fluxes
    // (bflux_b, at (a0_c, a1_f)), each times alpha sqrt(gamma) at its own face point.
    let asg = |p0: Gv, p1: Gv| -> Gv {
        let (alpha, sqrtg, ..) = adm(p0, p1);
        alpha * sqrtg
    };
    let en = Gv::ZERO - asg(a0_f, a1_c(0)) * gv_field_at("edge_bflux_a", "bflux_a", ndim, &zero);
    let es =
        Gv::ZERO - asg(a0_f, a1_c(-1)) * gv_field_at("edge_bflux_a", "bflux_a", ndim, &cm(&[g2]));
    let ee = asg(a0_c(0), a1_f) * gv_field_at("edge_bflux_b", "bflux_b", ndim, &zero);
    let ew = asg(a0_c(-1), a1_f) * gv_field_at("edge_bflux_b", "bflux_b", ndim, &cm(&[g1]));
    let fnf = gv_field_at("edge_fden_p1", "fden_p1", ndim, &zero);
    let fs = gv_field_at("edge_fden_p1", "fden_p1", ndim, &cm(&[g2]));
    let fe = gv_field_at("edge_fden_p2", "fden_p2", ndim, &zero);
    let fw = gv_field_at("edge_fden_p2", "fden_p2", ndim, &cm(&[g1]));
    let emf = ct_contact_emf_gv([en, es, ee, ew], [ene, enw, ese, esw], [fnf, fs, fe, fw]);
    (
        end_trace(),
        vec![("emf".to_string(), "emf".into(), emf.node())],
    )
}

/// the CURVED-SPACETIME CT face->cell B interpolation: each in-plane cell component is the
/// arithmetic average of its two bounding faces, exactly as in the flat
/// [`rmhd_bcell_from_bface_gv`]. NO magnetic-energy correction — tau carries the magnetic
/// energy and is conserved by the godunov flux, so a `nrg += 1/2 d(gamma_ij B^i B^j)` patch
/// would double-account it and does not telescope.
pub fn rmhd_bcell_from_bface_gr_gv(
    // the interpolation is metric-FREE; the chart args are kept for call-site symmetry with the
    // other GR kernels.
    _spacetime: Spacetime,
    _coords: Coords,
    _spacing: &[Spacing],
    axes: &[usize],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let ndim = axes.len();
    let half = Gv::from_f64(0.5);
    let off = |ax: usize| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        o[ax] = 1;
        o
    };
    let bf: Vec<Gv> = (0..ndim)
        .map(|c| Gv::field(&format!("bf_{c}"), &format!("bf_{c}")))
        .collect();
    // interpolate each in-plane cell component from its two bounding faces (arithmetic average;
    // metric-free — the cell field is a derived reconstruction / c2p quantity holding no conserved status). NO
    // energy patch: tau carries the magnetic energy and is conserved by the godunov flux, so a
    // metric-weighted `nrg += 1/2 d(gamma_ij B^i B^j)` would double-account it non-conservatively.
    let writes: Vec<(String, FieldBind, NodeId)> = axes
        .iter()
        .enumerate()
        .map(|(d, &c)| {
            let interp =
                (bf[d] + gv_field_at(&format!("bf_{d}"), &format!("bf_{d}"), ndim, &off(d))) * half;
            (
                format!("bc_{c}_new"),
                format!("bc_{c}").into(),
                interp.node(),
            )
        })
        .collect();
    (end_trace(), writes)
}

/// the per-direction UCT flux/diffusion coefficients at the edge — the (a^L, a^R, d^L, d^R) of the
/// master formula (mignone & del zanna 2020, Eq. 30). `al`/`ar` are the advective flux weights of the
/// upwind/downwind states (a^L + a^R = 1); `dl`/`dr` the dissipative diffusion coefficients (equal
/// for HLL/HLLC's symmetric advection, distinct for HLLD). THIS is the only solver-specific piece:
/// HLL fills it from the fast speeds (regime-generic); HLLC/HLLD swap it for the contact/alfven-aware
/// coefficients (Eq. 38 / 44) — the SAME master EMF kernel consumes it.
struct UctDir {
    al: Gv,
    ar: Gv,
    dl: Gv,
    dr: Gv,
}

fn weighted_average(ap: Gv, vp: Gv, am: Gv, vm: Gv) -> Gv {
    let sum = ap + am;
    let nonzero = sum.cmp_gt(Gv::ZERO);
    let divisor = Gv::select(nonzero, sum, Gv::ONE);
    Gv::select(nonzero, (ap * vp + am * vm) / divisor, Gv::ZERO)
}

/// HLL coefficients (Eq. 32) from the edge signal speeds `ap = max(0, lambda_max)`,
/// `am = max(0, -lambda_min)`: a^L = ap/(ap+am), a^R = am/(ap+am), d^L = d^R = ap*am/(ap+am).
fn uct_hll_coeffs(ap: Gv, am: Gv) -> UctDir {
    let sum = ap + am;
    let nonzero = sum.cmp_gt(Gv::ZERO);
    let divisor = Gv::select(nonzero, sum, Gv::ONE);
    let d = Gv::select(nonzero, ap * am / divisor, Gv::ZERO);
    UctDir {
        al: Gv::select(nonzero, ap / divisor, Gv::from_f64(0.5)),
        ar: Gv::select(nonzero, am / divisor, Gv::from_f64(0.5)),
        dl: d,
        dr: d,
    }
}

/// HLLC coefficients (Eq. 37-38). the three-wave fan (two fast `ll<=0<=lr` + the contact `lstar`)
/// gives a^L = a^R = 1/2 and the contact-aware diffusion
///   chi^s = -(vx^s - lambda^s)/(lambda^s - lstar),   d^s = ((|lstar|-|lambda^s|)/2) chi^s + |lambda^s|/2
/// (s = L,R). less dissipative than HLL because the transverse-field jump is resolved across the
/// contact wave. `vxl`/`vxr` are the L/R normal velocities. classical & relativistic
/// share this algebra; only `lstar` (the contact speed) is regime-specific (computed upstream).
fn uct_hllc_coeffs(ll: Gv, lr: Gv, lstar: Gv, vxl: Gv, vxr: Gv) -> UctDir {
    let half = Gv::from_f64(0.5);
    let speed_scale = ll
        .abs()
        .max(lr.abs())
        .max(lstar.abs())
        .max(vxl.abs())
        .max(vxr.abs());
    let eps = Gv::from_f64(32.0 * f64::EPSILON) * speed_scale;
    // guard the (lambda^s - lstar) denominators away from zero (preserve sign).
    let den_l = ll - lstar;
    let den_r = lr - lstar;
    let den_l = guard_denominator(den_l, eps);
    let den_r = guard_denominator(den_r, eps);
    let chi_l = (Gv::ZERO - (vxl - ll)) / den_l;
    let chi_r = (Gv::ZERO - (vxr - lr)) / den_r;
    // Eq. 38: d^s = ((|lstar| - |lambda^s|)/2) chi^s + |lstar|/2  (the LAST term is |lstar|, the
    // contact speed; a common transcription slip writes |lambda^s| here). this is the B_x = 0 DEGENERATE case (for B_x != 0 HLLC == HLL);
    // it is the building block for the HLLD singular limit (Eq. 46, v* = 0), used only within that composition.
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
    UctDir {
        al: half,
        ar: half,
        dl: floor(dl),
        dr: floor(dr),
    }
}

/// the UCT edge EMF in MASTER form (mignone & del zanna 2020, Eq. 33) — the structure that
/// generalizes across riemann solvers by swapping only the per-direction (a^L, a^R, d) coefficients
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
pub fn rmhd_edge_emf_uct_gv(
    ndim: usize,
    g1: usize,
    g2: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
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
    // the SOLVER-SPECIFIC coefficients; HLL here, and the only piece an hllc/hlld variant replaces.
    let cx = uct_hll_coeffs(apx, amx);
    let cy = uct_hll_coeffs(apy, amy);
    // upwind transverse velocities (Eq. 29): vbar_x upwind in x (alpha^+ carries the West/left state),
    // vbar_y upwind in y (alpha^+ carries the South/lower state).
    let vbar_x = weighted_average(apx, vx_w, amx, vx_e);
    let vbar_y = weighted_average(apy, vy_s, amy, vy_n);
    // staggered face B PLM-reconstructed a half-cell to the EDGE (M&DZ: the staggered transverse
    // field reconstructed from the adjacent interface — the load-bearing 2nd-order piece). geometry
    // VERIFIED vs the CT curl: Ez[i,j] is the corner (i-1/2,j-1/2); B_y is at the corner's y but
    // offset +-1/2 in x (recon along x = its transverse), B_x at the corner's x offset +-1/2 in y.
    // one-sided minmod-theta extrapolation: +1/2 toward the edge from the lower face, -1/2 from the
    // upper. needs the 2nd transverse neighbor -> bface allocated with +-2 transverse halo.
    let theta = Gv::scalar("theta");
    let recon = |key: &str, rt: &str, base: &[i32], axis: usize, sign: f64| -> Gv {
        let off = |d: i32| -> Vec<i32> {
            let mut o = base.to_vec();
            o[axis] += d;
            o
        };
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
    (
        end_trace(),
        vec![("emf".to_string(), "emf".into(), emf.node())],
    )
}

/// the CURVED-SPACETIME UCT-HLL edge EMF for the 2.5D (r, theta) poloidal plane — the master
/// form [`rmhd_edge_emf_uct_gv`] producing the DENSITIZED corner EMF `Etilde_phi` the GR curl
/// consumes. GENERALIZATION (GR-UCT): (1) the edge signal speeds are the SHIFTED
/// coordinate speeds materialized by `rmhd_wave_speeds_cell_gr_gv` (the BF fast bound minus the
/// shift — quartic-free; the flux still computes its own inline); (2) the advective velocity is
/// the transport velocity `vtilde = alpha v - beta` at the corner (the upwound physical v
/// transformed there); (3) the whole master EMF is densitized by `sqrt(gamma)(corner)` — since
/// reconstruction co-locates all four staggered fields AT the corner, one metric factor scales
/// the entire (advective + dissipative) EMF, and div(B)=0 is preserved (a CT curl of one scalar
/// corner EMF, coefficient-independent). reduces to the flat master form at gamma = id, alpha =
/// 1, beta = 0. baked per (spacetime, spacing).
pub fn rmhd_edge_emf_uct_gr_gv(
    spacetime: Spacetime,
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let ndim = 2usize;
    let (pc1, pc2, g1, g2) = gr_ct_plane(axes);
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
    let vp1 = |o: &[i32]| gv_field_at("edge_vp1", "vel_p1", ndim, o);
    let vp2 = |o: &[i32]| gv_field_at("edge_vp2", "vel_p2", ndim, o);
    let vx_e = (vp1(&zero) + vp1(&cm(&[g2]))) * half;
    let vx_w = (vp1(&cm(&[g1])) + vp1(&cm(&[g1, g2]))) * half;
    let vy_n = (vp2(&zero) + vp2(&cm(&[g1]))) * half;
    let vy_s = (vp2(&cm(&[g2])) + vp2(&cm(&[g1, g2]))) * half;
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
    let cx = uct_hll_coeffs(apx, amx);
    let cy = uct_hll_coeffs(apy, amy);
    let vbar_x = weighted_average(apx, vx_w, amx, vx_e);
    let vbar_y = weighted_average(apy, vy_s, amy, vy_n);
    // the transport velocity at the corner: vtilde = alpha v - beta on EACH in-plane axis (beta_p1
    // vanishes on the spherical polar angle but not on cartesian y / cylindrical z). the metric at
    // the corner (a0_f, a1_f) also densitizes the whole EMF below.
    let a0_f = gv_axis_face_at(0, spacing[0], 0);
    let a1_f = gv_axis_face_at(1, spacing[1], 0);
    let (alpha_c, sqrtg_c, beta_c) =
        gr_adm_at(spacetime, coords, gr_plane_pos(coords, axes, a0_f, a1_f));
    let vtilde_r = alpha_c * vbar_x - beta_c[pc1];
    let vtilde_th = alpha_c * vbar_y - beta_c[pc2];
    let theta = Gv::scalar("theta");
    let recon = |key: &str, rt: &str, base: &[i32], axis: usize, sign: f64| -> Gv {
        let off = |d: i32| -> Vec<i32> {
            let mut o = base.to_vec();
            o[axis] += d;
            o
        };
        let q0 = gv_field_at(key, rt, ndim, base);
        let qm = gv_field_at(key, rt, ndim, &off(-1));
        let qp = gv_field_at(key, rt, ndim, &off(1));
        let slope = minmod3((q0 - qm) * theta, half * (qp - qm), (qp - q0) * theta);
        q0 + Gv::from_f64(0.5 * sign) * slope
    };
    let by_e = recon("edge_bface_b", "bface_b", &zero, g1, -1.0);
    let by_w = recon("edge_bface_b", "bface_b", &cm(&[g1]), g1, 1.0);
    let bx_n = recon("edge_bface_a", "bface_a", &zero, g2, -1.0);
    let bx_s = recon("edge_bface_a", "bface_a", &cm(&[g2]), g2, 1.0);
    // the physical master EMF with transport velocities, densitized at the corner -> Etilde_phi.
    let emf_phys = uct_master_emf(&cx, &cy, vtilde_r, vtilde_th, by_e, by_w, bx_n, bx_s);
    let emf = sqrtg_c * emf_phys;
    (
        end_trace(),
        vec![("emf".to_string(), "emf".into(), emf.node())],
    )
}

/// the FULL-3D densitized UCT-HLL corner EMF for the cartesian GR charts: the same
/// master-formula combination as the 2d builder, per EDGE direction `dir` — the edge
/// runs along `dir`, the transverse plane is (g1, g2) = (dir+1, dir+2) mod 3, and the
/// corner metric sits at (cell center on `dir`, faces on g1/g2). the dispatch binds
/// the per-edge slots (vel/bface/wave-speed pairs) exactly as it does the 2d family.
/// cartesian only: the physical components coincide with the grid axes, so the
/// transport velocity is vtilde = alpha v - beta on g1/g2 directly.
pub fn rmhd_edge_emf_uct_gr_3d_gv(
    dir: usize,
    spacetime: Spacetime,
    coords: Coords,
    spacing: &[Spacing],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    assert!(
        coords == Coords::Cartesian,
        "the 3d GR UCT EMF is baked for the cartesian charts"
    );
    let ndim = 3usize;
    let g1 = (dir + 1) % 3;
    let g2 = (dir + 2) % 3;
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
    let vp1 = |o: &[i32]| gv_field_at("edge_vp1", "vel_p1", ndim, o);
    let vp2 = |o: &[i32]| gv_field_at("edge_vp2", "vel_p2", ndim, o);
    let vx_e = (vp1(&zero) + vp1(&cm(&[g2]))) * half;
    let vx_w = (vp1(&cm(&[g1])) + vp1(&cm(&[g1, g2]))) * half;
    let vy_n = (vp2(&zero) + vp2(&cm(&[g1]))) * half;
    let vy_s = (vp2(&cm(&[g2])) + vp2(&cm(&[g1, g2]))) * half;
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
    let cx = uct_hll_coeffs(apx, amx);
    let cy = uct_hll_coeffs(apy, amy);
    let vbar_x = weighted_average(apx, vx_w, amx, vx_e);
    let vbar_y = weighted_average(apy, vy_s, amy, vy_n);
    // the corner metric: the edge midpoint — cell-centered along the edge, faces on
    // the transverse pair; the transport velocity is vtilde = alpha v - beta on each
    // transverse axis, and the corner sqrt(gamma) densitizes the whole EMF.
    let a_f = |ax: usize, o: i64| gv_axis_face_at(ax, spacing[ax], o);
    let a_c = |ax: usize, o: i64| (a_f(ax, o) + a_f(ax, o + 1)) * half;
    let mut xc = [Gv::ZERO; 3];
    xc[dir] = a_c(dir, 0);
    xc[g1] = a_f(g1, 0);
    xc[g2] = a_f(g2, 0);
    let (alpha_c, sqrtg_c, beta_c) = gr_adm_at(spacetime, coords, Tensor::new(xc));
    let vtilde_r = alpha_c * vbar_x - beta_c[g1];
    let vtilde_th = alpha_c * vbar_y - beta_c[g2];
    let theta = Gv::scalar("theta");
    let by_e = recon_face_to_edge(ndim, theta, "edge_bface_b", "bface_b", &zero, g1, -1.0);
    let by_w = recon_face_to_edge(ndim, theta, "edge_bface_b", "bface_b", &cm(&[g1]), g1, 1.0);
    let bx_n = recon_face_to_edge(ndim, theta, "edge_bface_a", "bface_a", &zero, g2, -1.0);
    let bx_s = recon_face_to_edge(ndim, theta, "edge_bface_a", "bface_a", &cm(&[g2]), g2, 1.0);
    let emf_phys = uct_master_emf(&cx, &cy, vtilde_r, vtilde_th, by_e, by_w, bx_n, bx_s);
    let emf = sqrtg_c * emf_phys;
    (
        end_trace(),
        vec![("emf".to_string(), "emf".into(), emf.node())],
    )
}

/// PLM-reconstruct a staggered face field a half-cell to the EDGE (M&DZ: the staggered transverse
/// field reconstructed from the adjacent interface — the 2nd-order piece that preserves smooth fields,
/// VERIFIED on the field-loop test). `base` the face offset; `axis` the reconstruction direction (the
/// face's TRANSVERSE: x for B_y on y-faces, y for B_x on x-faces); `sign` = +1 reconstructs +1/2
/// toward the edge from the lower face, -1 reconstructs -1/2 from the upper. minmod-theta slope;
/// needs the 2nd transverse neighbor, hence bface's +-2 transverse halo.
fn recon_face_to_edge(
    ndim: usize,
    theta: Gv,
    key: &str,
    rt: &str,
    base: &[i32],
    axis: usize,
    sign: f64,
) -> Gv {
    let half = Gv::from_f64(0.5);
    let off = |d: i32| -> Vec<i32> {
        let mut o = base.to_vec();
        o[axis] += d;
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
}

/// the master-formula edge EMF combination (Eq. 33), shared by every UCT coefficient family. given
/// the per-direction coefficients + the upwind transverse velocities + the staggered face B at the
/// edge:
/// ```text
///   Ez = -vbar_x (a^L_x B_y^E + a^R_x B_y^W) + (d^R_x B_y^E - d^L_x B_y^W)
///       + vbar_y (a^L_y B_x^N + a^R_y B_x^S) - (d^R_y B_x^N - d^L_y B_x^S)
/// ```
/// (signs verified against the compact Eq. 27 diffusion + the symmetric-speed reduction v_y B_x - v_x B_y.)
fn uct_master_emf(
    cx: &UctDir,
    cy: &UctDir,
    vbar_x: Gv,
    vbar_y: Gv,
    by_e: Gv,
    by_w: Gv,
    bx_n: Gv,
    bx_s: Gv,
) -> Gv {
    let zero_g = Gv::ZERO;
    // a^L (= alpha^+/sum) weights the UPWIND face: West for +x (a^L -> by_w), South for +y (a^L -> bx_s)
    // — CONSISTENT with the diffusion's d^L->West/d^R->East pairing and with vbar (apx*vx_w). pairing
    // a^L to the downwind face is anti-upwind: invisible for symmetric speeds (a^L==a^R, subsonic OT)
    // but ADVECTS THE DOWNWIND state at supersonic mach -> instability (the field-loop blow-up). the
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
/// check at graph-build time. instant, and it covers every emf kernel built on the MASTER form
/// (`uct_master_emf`: nmhd/imhd HLL + HLLD, rmhd HLL, GR ortho). the wave-sum HLLD EMF
/// (`rmhd_edge_emf_uct_hlld_gv` / `_gr_gv`, M&DZ Eq. 39) does not route through this form; its
/// dissipation-sign pairing is proven by `hlld_wave_sum_proof_kernel` (see `hlld_wave_sum_terms`).
/// `swap` passes by_w/by_e in each other's argument slots to inject the ct_emf.rs anti-upwind bug,
/// for the negative control.
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
    (
        end_trace(),
        vec![("emf".to_string(), "emf".into(), emf.node())],
    )
}

/// the UN-halved HLLD wave-sum dissipation for one transverse B component across the 4-wave fan
/// (M&DZ Eq. 39): `sum_k |wave speed_k| * (jump across wave k)`. the DISSIPATION-SIGN PAIRING is the
/// load-bearing invariant — the wave-sum analog of the upwind pairing: the LEFT staggered endpoint
/// `bt_l` is diffused by the LEFT fast wave `alam_l` (it enters as `- alam_l * bt_l`), the RIGHT
/// endpoint `bt_r` by the RIGHT fast wave `alam_r` (`+ alam_r * bt_r`); the star / central states
/// telescope. mispairing an endpoint with the OPPOSITE fast wave flips the dissipation sign
/// (anti-diffusion — the invisible-subsonically failure history). the 1/2 is applied by the CALLER
/// (kept out here so the pairing coefficients stay integral for the symbolic proof). the speeds are
/// the ABSOLUTE (>= 0) fan speeds; the states are the reconstructed transverse-field values.
pub(crate) fn hlld_wave_sum_terms(
    alam_l: Gv,
    aalf_l: Gv,
    aalf_r: Gv,
    alam_r: Gv,
    bt_l: Gv,
    bstar_l: Gv,
    bc: Gv,
    bstar_r: Gv,
    bt_r: Gv,
) -> Gv {
    alam_l * (bstar_l - bt_l)
        + aalf_l * (bc - bstar_l)
        + aalf_r * (bstar_r - bc)
        + alam_r * (bt_r - bstar_r)
}

/// proof entry point for the HLLD wave-sum dissipation-sign pairing. traces `hlld_wave_sum_terms`
/// in ISOLATION with symbolic leaves: the five staggered / star transverse fields
/// {bt_l, bstar_l, bc, bstar_r, bt_r} are the LinForm "fields", the four ABSOLUTE fan speeds
/// {alam_l, aalf_l, aalf_r, alam_r} the opaque scalars. because the wave-sum is LINEAR in the fields,
/// `LinForm` reads each field's coefficient polynomial and the pairing is a coefficient check.
/// `swap` mispairs the two fast-wave endpoints (`alam_l` <-> `alam_r`) to reproduce the anti-diffusive
/// bug for the negative control. shared by `rmhd_edge_emf_uct_hlld_gv` (flat) and `_gr_gv` (both build
/// the dissipative Phi through `hlld_wave_sum_terms`), so this one proof binds both.
#[doc(hidden)]
pub fn hlld_wave_sum_proof_kernel(swap: bool) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let alam_l = Gv::param("alam_l");
    let aalf_l = Gv::param("aalf_l");
    let aalf_r = Gv::param("aalf_r");
    let alam_r = Gv::param("alam_r");
    let bt_l = Gv::param("bt_l");
    let bstar_l = Gv::param("bstar_l");
    let bc = Gv::param("bc");
    let bstar_r = Gv::param("bstar_r");
    let bt_r = Gv::param("bt_r");
    let phi = if swap {
        hlld_wave_sum_terms(
            alam_r, aalf_l, aalf_r, alam_l, bt_l, bstar_l, bc, bstar_r, bt_r,
        )
    } else {
        hlld_wave_sum_terms(
            alam_l, aalf_l, aalf_r, alam_r, bt_l, bstar_l, bc, bstar_r, bt_r,
        )
    };
    (
        end_trace(),
        vec![("phi".to_string(), "phi".into(), phi.node())],
    )
}

/// the UCT-HLLC edge EMF (master Eq. 33 + HLLC coefficients Eq. 37-38). same master formula as the
/// HLL kernel, but the diffusion uses the CONTACT speed `lstar` (the three-wave fan) -> less
/// dissipative. CLASSICAL ideal-gas (NMHD): `lstar = m_n^hll/rho^hll` is the HLL-average normal
/// velocity, computed in-kernel from the cell prims with the classical momentum flux
/// `F[m_n] = rho v_n^2 + p + |B|^2/2 - B_n^2`. edge speeds & per-side states use the MAX-over-4-cells
/// / 2-cell-average reconstruction. (IMHD: p = cs^2*rho; RMHD: relativistic conserved/flux.)
pub fn nmhd_edge_emf_uct_hllc_gv(
    ndim: usize,
    g1: usize,
    g2: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
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
    // per-FACE HLLC coefficients: each face uses ITS OWN two cells (first-order L/R) + davis face
    // speeds (s_r = max(0, ws_r^L, ws_r^R), s_l = min(0, ws_l^L, ws_l^R)) so lstar = m_n^hll/rho^hll
    // is CONSISTENT (the contact stays inside the fan -> no degeneracy blow-up). then the diffusion
    // is MAX-combined to the edge (the maximal-diffusion edge reconstruction). vn/bn read the normal
    // velocity / normal cell-B; wsr/wsl are the direction's per-cell speed fields.
    let face_d = |l: &[i32],
                  r: &[i32],
                  vn: &dyn Fn(&[i32]) -> Gv,
                  bn: &dyn Fn(&[i32]) -> Gv,
                  wsr_k: &str,
                  wsr_p: &str,
                  wsl_k: &str,
                  wsl_p: &str|
     -> (Gv, Gv) {
        let (rl, rr) = (rho(l), rho(r));
        let (vl, vr) = (vn(l), vn(r));
        let (pl, pr) = (pre(l), pre(r));
        let (bsl, bsr) = (bsq(l), bsq(r));
        let (bnl, bnr) = (bn(l), bn(r));
        let sr =
            zero_g.max(gv_field_at(wsr_k, wsr_p, ndim, l).max(gv_field_at(wsr_k, wsr_p, ndim, r)));
        let sl =
            zero_g.min(gv_field_at(wsl_k, wsl_p, ndim, l).min(gv_field_at(wsl_k, wsl_p, ndim, r)));
        let (mxl, mxr) = (rl * vl, rr * vr);
        let fl = rl * vl * vl + pl + half * bsl - bnl * bnl;
        let fr = rr * vr * vr + pr + half * bsr - bnr * bnr;
        let wave_span = sr - sl;
        let wave_nonzero = wave_span.cmp_gt(Gv::ZERO);
        let wave_divisor = Gv::select(wave_nonzero, wave_span, Gv::ONE);
        let inv = Gv::select(wave_nonzero, Gv::ONE / wave_divisor, Gv::ZERO);
        let rho_hll = (sr * rr - sl * rl + mxl - mxr) * inv;
        let mx_hll = (sr * mxr - sl * mxl + fl - fr) * inv;
        let rho_scale = rl.abs().max(rr.abs());
        let rho_nonzero = rho_hll
            .abs()
            .cmp_gt(Gv::from_f64(32.0 * f64::EPSILON) * rho_scale);
        let rho_divisor = Gv::select(rho_nonzero, rho_hll, Gv::ONE);
        let lstar = Gv::select(rho_nonzero, mx_hll / rho_divisor, Gv::ZERO);
        let c = uct_hllc_coeffs(sl, sr, lstar, vl, vr);
        (c.dl, c.dr)
    };
    // x-faces (normal p1): North NW->NE, South SW->SE; MAX-combine d to the edge.
    let (dln, drn) = face_d(&nw, &ne, &vp1, &bp1, "e_wsr1", "wsr_p1", "e_wsl1", "wsl_p1");
    let (dls, drs) = face_d(&sw, &se, &vp1, &bp1, "e_wsr1", "wsr_p1", "e_wsl1", "wsl_p1");
    let cx = UctDir {
        al: half,
        ar: half,
        dl: avg2(dln, dls),
        dr: avg2(drn, drs),
    };
    // y-faces (normal p2): West SW->NW, East SE->NE.
    let (dlw, drw) = face_d(&sw, &nw, &vp2, &bp2, "e_wsr2", "wsr_p2", "e_wsl2", "wsl_p2");
    let (dle, dre) = face_d(&se, &ne, &vp2, &bp2, "e_wsr2", "wsr_p2", "e_wsl2", "wsl_p2");
    let cy = UctDir {
        al: half,
        ar: half,
        dl: avg2(dlw, dle),
        dr: avg2(drw, dre),
    };
    // upwind transverse velocities (Eq. 29) for the advective part: alpha^+ carries West / South.
    let vx_w = avg2(vp1(&nw), vp1(&sw));
    let vx_e = avg2(vp1(&ne), vp1(&se));
    let vy_s = avg2(vp2(&sw), vp2(&se));
    let vy_n = avg2(vp2(&nw), vp2(&ne));
    let vbar_x = weighted_average(apx, vx_w, amx, vx_e);
    let vbar_y = weighted_average(apy, vy_s, amy, vy_n);
    // staggered face B PLM-reconstructed a half-cell to the edge (M&DZ transverse reconstruction).
    let theta = Gv::scalar("theta");
    let by_e = recon_face_to_edge(ndim, theta, "e_bface_b", "bface_b", &zero, g1, -1.0);
    let by_w = recon_face_to_edge(ndim, theta, "e_bface_b", "bface_b", &nw, g1, 1.0);
    let bx_n = recon_face_to_edge(ndim, theta, "e_bface_a", "bface_a", &zero, g2, -1.0);
    let bx_s = recon_face_to_edge(ndim, theta, "e_bface_a", "bface_a", &se, g2, 1.0);
    let emf = uct_master_emf(&cx, &cy, vbar_x, vbar_y, by_e, by_w, bx_n, bx_s);
    (
        end_trace(),
        vec![("emf".to_string(), "emf".into(), emf.node())],
    )
}

/// the NMHD UCT-HLLD edge EMF — mignone & del zanna (2020), reproduced VERBATIM. NO liberties, NO
/// floors. the composition is the solver-agnostic MASTER form (eq:emf2D, = `uct_master_emf`,
/// byte-identical structure to UCT-HLL); per the paper, the ONLY solver-specific part is the per-face
/// (a,d) coefficients — the five-wave HLLD fan (eq:UCT_HLLD_ad, eq:UCT_HLLD_nu, eq:By_chi). edge
/// combine: MAX on the diffusion `d` (paper-sanctioned, "maximizing the diffusion terms"), AVERAGE
/// on the advective `a` (eq:dEW). upwind transverse velocity `vbar` (eq:vt), shared with UCT-HLL.
/// NO floor on rho^{*s}: physical states give rho^{*s} > 0; the degenerate guard (nu* = 0 when the
/// rotational waves collapse, eps = 1e-9) is the ONLY safeguard. zeroth order = R+/- reconstruction
/// is identity (theta = 0). mignone & del zanna method 2.
pub fn nmhd_edge_emf_uct_hlld_gv(
    ndim: usize,
    g1: usize,
    g2: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
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
    let zero_g = Gv::ZERO;
    let ne = zero.clone();
    let nw = cm(&[g1]);
    let se = cm(&[g2]);
    let sw = cm(&[g1, g2]);
    // PLM reconstruction of a CELL field to a face — identical to the gas flux's plm_theta_gv (same
    // theta, same limiter: theta<0 selects van leer). the edge-EMF fan is fed the SAME
    // reconstructed L/R face states the 1D gas riemann solves (Eq. 29 + the per-face riemann input),
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
    let eos = IdealGas {
        gamma: Gv::scalar("gamma"),
    };
    let face_prim =
        |base: &[i32], naxis: usize, nidx: usize, sign: f64, bn_face: Gv| -> MhdPrim<Gv, 3> {
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
    // (eq:UCT_HLLD_ad via hlld_newtonian_coeffs) — the EMF fan is IDENTICAL to the gas flux's at
    // the same reconstructed face state, which is the CT-consistency mignone & del zanna require.
    // NO clamp.
    let hlld_face = |l: &[i32],
                     r: &[i32],
                     naxis: usize,
                     nidx: usize,
                     bn_face: Gv,
                     nhat: &Tensor<Gv, 3>|
     -> (Gv, Gv, Gv) {
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
    let vbar_x = weighted_average(apx, vx_w, amx, vx_e);
    let vbar_y = weighted_average(apy, vy_s, amy, vy_n);
    // staggered face B reconstructed to the EDGE (R+/-; theta=0 => identity = zeroth order). these are the
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
    let cx = UctDir {
        al: ax_l,
        ar: one - ax_l,
        dl: dln.max(dls),
        dr: drn.max(drs),
    };
    // y-faces (normal g2, prim comp 1): West SW->NW, East SE->NE.
    let bn_w = gv_field_at("h_bface_b", "bface_b", ndim, &nw);
    let bn_e = gv_field_at("h_bface_b", "bface_b", ndim, &ne);
    let (alw, dlw, drw) = hlld_face(&sw, &nw, g2, 1, bn_w, &nhat_y);
    let (ale, dle, dre) = hlld_face(&se, &ne, g2, 1, bn_e, &nhat_y);
    let ay_l = (alw + ale) * half;
    let cy = UctDir {
        al: ay_l,
        ar: one - ay_l,
        dl: dlw.max(dle),
        dr: drw.max(dre),
    };
    // master composition (eq:emf2D), identical to UCT-HLL.
    let emf = uct_master_emf(&cx, &cy, vbar_x, vbar_y, by_e, by_w, bx_n, bx_s);
    (
        end_trace(),
        vec![("emf".to_string(), "emf".into(), emf.node())],
    )
}

/// the ISOTHERMAL UCT-HLLD edge EMF (IMHD). twin of `nmhd_edge_emf_uct_hlld_gv` but the per-face fan
/// is `hlld_isothermal_coeffs` (M&DZ 2020 appendix A: no contact mode -> chi~^s uses the HLL central
/// state, a/d/nu unchanged) and the prim has NO pressure (`Isothermal{cs}`, `IsoMhdPrim`). everything
/// else — staggered-B recon to the edge, the single-vbar advection, the master composition — matches
/// the NMHD kernel verbatim.
pub fn imhd_edge_emf_uct_hlld_gv(
    ndim: usize,
    g1: usize,
    g2: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
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
    let eos = Isothermal {
        cs: Gv::scalar("cs"),
    };
    let face_prim =
        |base: &[i32], naxis: usize, nidx: usize, sign: f64, bn_face: Gv| -> IsoMhdPrim<Gv, 3> {
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
    let hlld_face = |l: &[i32],
                     r: &[i32],
                     naxis: usize,
                     nidx: usize,
                     bn_face: Gv,
                     nhat: &Tensor<Gv, 3>|
     -> (Gv, Gv, Gv) {
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
    let vbar_x = weighted_average(apx, vx_w, amx, vx_e);
    let vbar_y = weighted_average(apy, vy_s, amy, vy_n);
    let by_e = recon_face_to_edge(ndim, theta, "h_bface_b", "bface_b", &zero, g1, -1.0);
    let by_w = recon_face_to_edge(ndim, theta, "h_bface_b", "bface_b", &nw, g1, 1.0);
    let bx_n = recon_face_to_edge(ndim, theta, "h_bface_a", "bface_a", &zero, g2, -1.0);
    let bx_s = recon_face_to_edge(ndim, theta, "h_bface_a", "bface_a", &se, g2, 1.0);
    let bn_n = gv_field_at("h_bface_a", "bface_a", ndim, &ne);
    let bn_s = gv_field_at("h_bface_a", "bface_a", ndim, &se);
    let (aln, dln, drn) = hlld_face(&nw, &ne, g1, 0, bn_n, &nhat_x);
    let (als, dls, drs) = hlld_face(&sw, &se, g1, 0, bn_s, &nhat_x);
    let ax_l = (aln + als) * half;
    let cx = UctDir {
        al: ax_l,
        ar: one - ax_l,
        dl: dln.max(dls),
        dr: drn.max(drs),
    };
    let bn_w = gv_field_at("h_bface_b", "bface_b", ndim, &nw);
    let bn_e = gv_field_at("h_bface_b", "bface_b", ndim, &ne);
    let (alw, dlw, drw) = hlld_face(&sw, &nw, g2, 1, bn_w, &nhat_y);
    let (ale, dle, dre) = hlld_face(&se, &ne, g2, 1, bn_e, &nhat_y);
    let ay_l = (alw + ale) * half;
    let cy = UctDir {
        al: ay_l,
        ar: one - ay_l,
        dl: dlw.max(dle),
        dr: drw.max(dre),
    };
    let emf = uct_master_emf(&cx, &cy, vbar_x, vbar_y, by_e, by_w, bx_n, bx_s);
    (
        end_trace(),
        vec![("emf".to_string(), "emf".into(), emf.node())],
    )
}

/// the RELATIVISTIC UCT-HLLD edge EMF (RMHD). built from the WAVE-SUM dissipative flux (mignone &
/// del zanna 2020 Eq. 39 + MUB09 star states). the classical coefficient form (Eq. 44) bakes in a
/// CLASSICAL velocity-chi that is invalid relativistically and fails the
/// energy-telescoping property, so it is unusable here.
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
/// `lambda^{s}` alfven (`alf`). BOUNDED by construction (field differences times |speed| — no ratio,
/// no 1/B_t, no floor, no clamp). reduces EXACTLY to `-F_hlld_rmhd[B_t]` in 1D (verified to machine
/// precision). the STAGGERED transverse face fields are the riemann L/R (CT consistency, M&DZ p.8) so
/// Phi damps the staggered checkerboard; cell velocities/rho/pre are the 2-cell edge average. gated on
/// `success`: where the secant fails, Phi -> the finite HLL dissipation (the lam are always finite).
pub fn rmhd_edge_emf_uct_hlld_gv(
    ndim: usize,
    g1: usize,
    g2: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
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
    let zero_g = Gv::ZERO;
    let gamma = Gv::scalar("gamma");
    let eos = IdealGas { gamma };
    let avg2 = |a: Gv, b: Gv| (a + b) * half;
    let ne = zero.clone();
    let nw = cm(&[g1]);
    let se = cm(&[g2]);
    let sw = cm(&[g1, g2]);
    // PLM-reconstruct a CELL field to a face (same theta + limiter as the gas flux). the mignone &
    // del zanna exact recipe: the wave-sum's per-face riemann must use the SAME reconstructed L/R states
    // the gas flux solves; a 2-cell-averaged single edge riemann uses cell states that make the EMF fan
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
    // are both overridden with the staggered div-free values (constant-B_n riemann; the transverse is
    // the staggered face that gets dissipated). L uses sign +1 (toward +naxis face), R uses -1.
    let prim_face = |base: &[i32],
                     naxis: usize,
                     sign: f64,
                     n_idx: usize,
                     bn: Gv,
                     t_idx: usize,
                     bt: Gv|
     -> MhdPrim<Gv, 3> {
        let r = |key: &str, rt: &str| recon_cell(key, rt, base, naxis, sign);
        let rho = r("e_rho", "rho");
        let pre = r("e_pre", "pre");
        let v = [
            r("e_vp1", "vel_p1"),
            r("e_vp2", "vel_p2"),
            r("e_vout", "vel_out"),
        ];
        let mut b = [
            r("e_bp1", "bcell_p1"),
            r("e_bp2", "bcell_p2"),
            r("e_bout", "bcell_out"),
        ];
        b[n_idx] = bn;
        b[t_idx] = bt;
        MhdPrim::<Gv, 3> {
            hydro: Prim {
                rho,
                vel: Tensor::new(v),
                pre,
            },
            mag: Tensor::new(b),
        }
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
    // the wave-sum dissipative flux Phi (M&DZ Eq. 39) for a riemann whose transverse component is `t`,
    // with staggered endpoints `bt_l`,`bt_r` and the single-/double-star fields from `st`. gated on
    // `success` -> HLL dissipation (NaN-safe true select; the HLL branch uses only the finite lam).
    let wave_sum = |st: &HlldStates<Gv, 3>, t: usize, bt_l: Gv, bt_r: Gv| -> Gv {
        let phi_hlld = half
            * hlld_wave_sum_terms(
                st.lam[0].abs(),
                st.alf[0].abs(),
                st.alf[1].abs(),
                st.lam[1].abs(),
                bt_l,
                st.bstar[0][t],
                st.bc[t],
                st.bstar[1][t],
                bt_r,
            );
        let ap = zero_g.max(st.lam[1]);
        let am = zero_g.max(zero_g - st.lam[0]);
        let phi_hll = uct_hll_coeffs(ap, am).dl * (bt_r - bt_l);
        Gv::select(st.success.cmp_gt(half), phi_hlld, phi_hll)
    };
    // x-riemann dissipates B_y (component 1), normal B_x (0). PER-FACE: North (NW->NE) and South
    // (SW->SE), each from reconstructed states (matching the gas flux); Phi_x = 1/2(Phi_N + Phi_S)
    // (M&DZ Eq. 34 edge average). normal B_x = the staggered FACE B_x; transverse B_y = by_w / by_e.
    let xn_l = prim_face(&nw, g1, 1.0, 0, bx_n_face, 1, by_w);
    let xn_r = prim_face(&ne, g1, -1.0, 0, bx_n_face, 1, by_e);
    let st_xn = hlld_rmhd_states(
        &Rmhd,
        &eos,
        &xn_l,
        &xn_r,
        &Tensor::<Gv, 3>::unit(0),
        &SpatialMetric::flat(),
    );
    let xs_l = prim_face(&sw, g1, 1.0, 0, bx_s_face, 1, by_w);
    let xs_r = prim_face(&se, g1, -1.0, 0, bx_s_face, 1, by_e);
    let st_xs = hlld_rmhd_states(
        &Rmhd,
        &eos,
        &xs_l,
        &xs_r,
        &Tensor::<Gv, 3>::unit(0),
        &SpatialMetric::flat(),
    );
    let phi_x = avg2(
        wave_sum(&st_xn, 1, by_w, by_e),
        wave_sum(&st_xs, 1, by_w, by_e),
    );
    // y-riemann dissipates B_x (component 0), normal B_y (1). PER-FACE: West (SW->NW) and East
    // (SE->NE). normal B_y = the staggered FACE B_y; transverse B_x = bx_s / bx_n.
    let yw_l = prim_face(&sw, g2, 1.0, 1, by_w_face, 0, bx_s);
    let yw_r = prim_face(&nw, g2, -1.0, 1, by_w_face, 0, bx_n);
    let st_yw = hlld_rmhd_states(
        &Rmhd,
        &eos,
        &yw_l,
        &yw_r,
        &Tensor::<Gv, 3>::unit(1),
        &SpatialMetric::flat(),
    );
    let ye_l = prim_face(&se, g2, 1.0, 1, by_e_face, 0, bx_s);
    let ye_r = prim_face(&ne, g2, -1.0, 1, by_e_face, 0, bx_n);
    let st_ye = hlld_rmhd_states(
        &Rmhd,
        &eos,
        &ye_l,
        &ye_r,
        &Tensor::<Gv, 3>::unit(1),
        &SpatialMetric::flat(),
    );
    let phi_y = avg2(
        wave_sum(&st_yw, 0, bx_s, bx_n),
        wave_sum(&st_ye, 0, bx_s, bx_n),
    );
    // centered advective velocities (2-cell averages straddling the edge).
    let vp1 = |o: &[i32]| gv_field_at("e_vp1", "vel_p1", ndim, o);
    let vp2 = |o: &[i32]| gv_field_at("e_vp2", "vel_p2", ndim, o);
    let vx_w = avg2(vp1(&nw), vp1(&sw));
    let vx_e = avg2(vp1(&ne), vp1(&se));
    let vy_s = avg2(vp2(&sw), vp2(&se));
    let vy_n = avg2(vp2(&nw), vp2(&ne));
    // E_z = -1/2(v_x^E B_y^E + v_x^W B_y^W) + 1/2(v_y^N B_x^N + v_y^S B_x^S) + Phi_x - Phi_y.
    let emf =
        zero_g - half * (vx_e * by_e + vx_w * by_w) + half * (vy_n * bx_n + vy_s * bx_s) + phi_x
            - phi_y;
    (
        end_trace(),
        vec![("emf".to_string(), "emf".into(), emf.node())],
    )
}

/// the CURVED-SPACETIME UCT-HLLD edge EMF (the wave-sum dissipative form, M&DZ Eq. 39) for the
/// 2.5D (r, theta) poloidal plane — the sharp, alfven-resolving GR-UCT EMF. mirrors the flat
/// `rmhd_edge_emf_uct_hlld_gv` with three GR generalizations (GR-HLLD): (1) the per-face
/// HLLD riemann uses the ORTHONORMAL-frame MUB09 solver `hlld_rmhd_states_gr_ortho(.., &face_metric)`
/// — the flat star fields + speeds map back to the coordinate frame (fields /sqrt(g_i), speeds
/// /sqrt(g_n)) so the wave-sum telescopes EXACTLY to the coordinate HLLD B_t flux (proven,
/// riemann/hlld.rs); (2) the advective velocity is the transport velocity vtilde = alpha v - beta
/// (spinning kerr carries a radial shift; schwarzschild is zero-shift), and the wave-sum fan speeds
/// enter RELATIVE to the moving interface vf = beta^r/alpha, matching the shifted gas HLLD flux;
/// (3) the whole coordinate EMF is densitized by sqrt(gamma)(corner) -> Etilde_phi (the same corner-
/// densitization as the GR-UCT-HLL EMF; the corner-averaged Phi (Eq. 34) is a corner quantity).
/// SCHWARZSCHILD + SPINNING KERR — the kerr-schild 2D MHD row is unbaked. the per-face metrics sit at
/// each riemann's face center, matching the gas HLLD flux. baked per spacing.
pub fn rmhd_edge_emf_uct_hlld_gr_gv(
    spacetime: Spacetime,
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    // the kernel feeds a full 3-vector prim + the WORLD spatial metric to the tetrad HLLD fan, so it
    // is indexed by the in-plane physical components (pc1, pc2) and the out-of-plane one (`out`): the
    // prim is assembled in WORLD order (v[pc]=..), the fan solves along the world normal (dir = pc),
    // and the shift/transverse fields address the physical axes. offsets/reconstructions stay in grid
    // space via (g1, g2). contiguous axes [0,1] give (pc1,pc2,g1,g2) = (0,1,0,1) (bit-identical to the
    // former hardcode); the gapped (R,z) set [0,2] gives (2,0,1,0) and is handled by the same code.
    begin_trace();
    let ndim = 2usize;
    let (pc1, pc2, g1, g2) = gr_ct_plane(axes);
    let out = 3 - pc1 - pc2;
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
    let zero_g = Gv::ZERO;
    let gamma = Gv::scalar("gamma");
    let eos = IdealGas { gamma };
    let avg2 = |a: Gv, b: Gv| (a + b) * half;
    let ne = zero.clone();
    let nw = cm(&[g1]);
    let se = cm(&[g2]);
    let sw = cm(&[g1, g2]);
    // the corner (grid-0 face, grid-1 face) — the advective-EMF densitization point.
    let r_f = gv_axis_face_at(0, spacing[0], 0);
    let th_f = gv_axis_face_at(1, spacing[1], 0);
    // the face spatial metric (for the tetrad fan) + the (alpha, sqrt(gamma), FULL shift) ADM triad at
    // a point, chart-generic via [`gr_spatial_metric_at`] / [`gr_adm_at`]: diagonal schwarzschild/disk,
    // the non-diagonal cartesian/kerr gamma, and the KS multi-axis shift alike.
    let metric_at = |c0: Gv, c1: Gv| -> SpatialMetric<Gv, 3> {
        gr_spatial_metric_at(spacetime, coords, gr_plane_pos(coords, axes, c0, c1))
    };
    let adm_at = |c0: Gv, c1: Gv| -> (Gv, Gv, Tensor<Gv, 3>) {
        gr_adm_at(spacetime, coords, gr_plane_pos(coords, axes, c0, c1))
    };
    // the world position (grid-0 coord, grid-1 coord) of a face point: grid axis `fa` sits on a FACE
    // (offset `fo`), the OTHER grid axis on a CELL center (offset `co`). feeds metric_at / adm_at at
    // each riemann's own face center.
    let face_cell = |fa: usize, fo: i64, co: i64| -> (Gv, Gv) {
        let ca = 1 - fa;
        let cell = |d: usize, o: i64| {
            (gv_axis_face_at(d, spacing[d], o) + gv_axis_face_at(d, spacing[d], o + 1)) * half
        };
        let mut c = [Gv::ZERO; 2];
        c[fa] = gv_axis_face_at(fa, spacing[fa], fo);
        c[ca] = cell(ca, co);
        (c[0], c[1])
    };
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
    // the reconstructed face prim in WORLD component order: vel_p1/p2/out carry the physical
    // components (pc1, pc2, out), so place each at its world index. the staggered normal face-B
    // `bn` overrides the physical normal component `n_phys`, the transverse edge-B `bt` the
    // transverse `t_phys` — both world component indices (pc for the active riemann).
    let prim_face = |base: &[i32],
                     naxis: usize,
                     sign: f64,
                     n_phys: usize,
                     bn: Gv,
                     t_phys: usize,
                     bt: Gv|
     -> MhdPrim<Gv, 3> {
        let r = |key: &str, rt: &str| recon_cell(key, rt, base, naxis, sign);
        let rho = r("e_rho", "rho");
        let pre = r("e_pre", "pre");
        let mut v = [Gv::ZERO; 3];
        v[pc1] = r("e_vp1", "vel_p1");
        v[pc2] = r("e_vp2", "vel_p2");
        v[out] = r("e_vout", "vel_out");
        let mut b = [Gv::ZERO; 3];
        b[pc1] = r("e_bp1", "bcell_p1");
        b[pc2] = r("e_bp2", "bcell_p2");
        b[out] = r("e_bout", "bcell_out");
        b[n_phys] = bn;
        b[t_phys] = bt;
        MhdPrim::<Gv, 3> {
            hydro: Prim {
                rho,
                vel: Tensor::new(v),
                pre,
            },
            mag: Tensor::new(b),
        }
    };
    let bx_n = recon_face_to_edge(ndim, theta, "e_bface_a", "bface_a", &zero, g2, -1.0);
    let bx_s = recon_face_to_edge(ndim, theta, "e_bface_a", "bface_a", &se, g2, 1.0);
    let by_w = recon_face_to_edge(ndim, theta, "e_bface_b", "bface_b", &nw, g1, 1.0);
    let by_e = recon_face_to_edge(ndim, theta, "e_bface_b", "bface_b", &zero, g1, -1.0);
    let bx_n_face = gv_field_at("e_bface_a", "bface_a", ndim, &ne);
    let bx_s_face = gv_field_at("e_bface_a", "bface_a", ndim, &se);
    let by_w_face = gv_field_at("e_bface_b", "bface_b", ndim, &nw);
    let by_e_face = gv_field_at("e_bface_b", "bface_b", ndim, &ne);
    // the wave-sum Phi (coordinate frame, contravariant B jumps), success -> HLL fallback. `vf` is
    // the MOVING-INTERFACE speed (beta^n/alpha): the shifted chart evaluates the fan at x/t = vf, so
    // every wave speed enters relative to it (lambda - vf) — matching the shifted gas HLLD flux
    // (hlld_rmhd_gr_ortho with vface), which is what keeps the edge EMF flux-consistent. vf = 0 for
    // the zero-shift schwarzschild chart and for the theta-direction fan (beta^theta = 0).
    let wave_sum = |st: &HlldStates<Gv, 3>, t: usize, bt_l: Gv, bt_r: Gv, vf: Gv| -> Gv {
        let (l0, l1) = (st.lam[0] - vf, st.lam[1] - vf);
        let (a0, a1) = (st.alf[0] - vf, st.alf[1] - vf);
        let phi_hlld = half
            * hlld_wave_sum_terms(
                l0.abs(),
                a0.abs(),
                a1.abs(),
                l1.abs(),
                bt_l,
                st.bstar[0][t],
                st.bc[t],
                st.bstar[1][t],
                bt_r,
            );
        let ap = zero_g.max(l1);
        let am = zero_g.max(zero_g - l0);
        let phi_hll = uct_hll_coeffs(ap, am).dl * (bt_r - bt_l);
        Gv::select(st.success.cmp_gt(half), phi_hlld, phi_hll)
    };
    // the grid-g1-face riemann (world normal pc1) at its two grid-g2 cell centers (N=0, S=-1): the
    // fan solves along the world axis pc1 and rides the moving-interface speed vf = beta^{pc1}/alpha.
    // its wave-sum feeds the pc2 (transverse) field flux, read at the world index pc2 of the states.
    let (an0, an1) = face_cell(g1, 0, 0);
    let (a_xn, _, b_xn) = adm_at(an0, an1);
    let vf_xn = b_xn[pc1] / a_xn;
    let xn_l = prim_face(&nw, g1, 1.0, pc1, bx_n_face, pc2, by_w);
    let xn_r = prim_face(&ne, g1, -1.0, pc1, bx_n_face, pc2, by_e);
    let st_xn = hlld_rmhd_states_gr_ortho(&eos, &xn_l, &xn_r, pc1, &metric_at(an0, an1));
    let (as0, as1) = face_cell(g1, 0, -1);
    let (a_xs, _, b_xs) = adm_at(as0, as1);
    let vf_xs = b_xs[pc1] / a_xs;
    let xs_l = prim_face(&sw, g1, 1.0, pc1, bx_s_face, pc2, by_w);
    let xs_r = prim_face(&se, g1, -1.0, pc1, bx_s_face, pc2, by_e);
    let st_xs = hlld_rmhd_states_gr_ortho(&eos, &xs_l, &xs_r, pc1, &metric_at(as0, as1));
    let phi_x = avg2(
        wave_sum(&st_xn, pc2, by_w, by_e, vf_xn),
        wave_sum(&st_xs, pc2, by_w, by_e, vf_xs),
    );
    // the grid-g2-face riemann (world normal pc2) at its two grid-g1 cell centers (W=-1, E=0): vf =
    // beta^{pc2}/alpha (zero on the spherical polar angle and the disk azimuth, nonzero on cartesian
    // y and cylindrical z). its wave-sum feeds the pc1 field flux (world index pc1 of the states).
    let (bw0, bw1) = face_cell(g2, 0, -1);
    let (a_yw, _, b_yw) = adm_at(bw0, bw1);
    let vf_yw = b_yw[pc2] / a_yw;
    let yw_l = prim_face(&sw, g2, 1.0, pc2, by_w_face, pc1, bx_s);
    let yw_r = prim_face(&nw, g2, -1.0, pc2, by_w_face, pc1, bx_n);
    let st_yw = hlld_rmhd_states_gr_ortho(&eos, &yw_l, &yw_r, pc2, &metric_at(bw0, bw1));
    let (be0, be1) = face_cell(g2, 0, 0);
    let (a_ye, _, b_ye) = adm_at(be0, be1);
    let vf_ye = b_ye[pc2] / a_ye;
    let ye_l = prim_face(&se, g2, 1.0, pc2, by_e_face, pc1, bx_s);
    let ye_r = prim_face(&ne, g2, -1.0, pc2, by_e_face, pc1, bx_n);
    let st_ye = hlld_rmhd_states_gr_ortho(&eos, &ye_l, &ye_r, pc2, &metric_at(be0, be1));
    let phi_y = avg2(
        wave_sum(&st_yw, pc1, bx_s, bx_n, vf_yw),
        wave_sum(&st_ye, pc1, bx_s, bx_n, vf_ye),
    );
    // advective transport velocity at the corner: vtilde = alpha v - beta on EACH in-plane axis.
    let (alpha_c, sqrtg_c, beta_c) = adm_at(r_f, th_f);
    let (beta_x_c, beta_y_c) = (beta_c[pc1], beta_c[pc2]);
    let vp1 = |o: &[i32]| gv_field_at("e_vp1", "vel_p1", ndim, o);
    let vp2 = |o: &[i32]| gv_field_at("e_vp2", "vel_p2", ndim, o);
    let vx_w = alpha_c * avg2(vp1(&nw), vp1(&sw)) - beta_x_c;
    let vx_e = alpha_c * avg2(vp1(&ne), vp1(&se)) - beta_x_c;
    let vy_s = alpha_c * avg2(vp2(&sw), vp2(&se)) - beta_y_c;
    let vy_n = alpha_c * avg2(vp2(&nw), vp2(&ne)) - beta_y_c;
    // the coordinate corner EMF, densitized to Etilde_phi = sqrt(gamma)(corner) * E_z.
    let ez =
        zero_g - half * (vx_e * by_e + vx_w * by_w) + half * (vy_n * bx_n + vy_s * bx_s) + phi_x
            - phi_y;
    let emf = sqrtg_c * ez;
    (
        end_trace(),
        vec![("emf".to_string(), "emf".into(), emf.node())],
    )
}

/// the CURVED-SPACETIME UCT-HLLD edge EMF for the FULL 3D grid, along edge axis `dir` — the
/// wave-sum dissipative form (M&DZ Eq. 39) with the three GR generalizations of the 2D builder
/// (`rmhd_edge_emf_uct_hlld_gr_gv`): per-face ORTHONORMAL-frame MUB09 fans at each riemann's own
/// face metric, the transport velocity vtilde = alpha v - beta with the fan speeds relative to the
/// moving interface vf = beta^n/alpha, and the corner sqrt(gamma) densitization. identity chart
/// only (physical component == grid axis): the edge is cell-centered along `dir` and cornered on
/// the transverse pair (g1, g2) = (dir+1, dir+2) mod 3; the out-of-plane prim component is `dir`.
pub fn rmhd_edge_emf_uct_hlld_gr_3d_gv(
    dir: usize,
    spacetime: Spacetime,
    coords: Coords,
    spacing: &[Spacing],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    assert!(
        coords == Coords::Cartesian,
        "the 3d GR UCT-HLLD EMF is baked for the cartesian charts"
    );
    let ndim = 3usize;
    let g1 = (dir + 1) % 3;
    let g2 = (dir + 2) % 3;
    let (pc1, pc2, out) = (g1, g2, dir);
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
    let zero_g = Gv::ZERO;
    let gamma = Gv::scalar("gamma");
    let eos = IdealGas { gamma };
    let avg2 = |a: Gv, b: Gv| (a + b) * half;
    let ne = zero.clone();
    let nw = cm(&[g1]);
    let se = cm(&[g2]);
    let sw = cm(&[g1, g2]);
    let a_f = |ax: usize, o: i64| gv_axis_face_at(ax, spacing[ax], o);
    let a_c = |ax: usize, o: i64| (a_f(ax, o) + a_f(ax, o + 1)) * half;
    // the edge midpoint: cell-centered along `dir`, faces on the transverse pair —
    // the advective-EMF densitization point.
    let corner = {
        let mut xc = [Gv::ZERO; 3];
        xc[dir] = a_c(dir, 0);
        xc[g1] = a_f(g1, 0);
        xc[g2] = a_f(g2, 0);
        Tensor::new(xc)
    };
    let metric_at =
        |x: Tensor<Gv, 3>| -> SpatialMetric<Gv, 3> { gr_spatial_metric_at(spacetime, coords, x) };
    // the 3d position of a face point: grid axis `fa` on a FACE (offset `fo`), the
    // OTHER transverse axis on a cell center (offset `co`), the edge axis cell-centered.
    let face_pos = |fa: usize, fo: i64, co: i64| -> Tensor<Gv, 3> {
        let ca = if fa == g1 { g2 } else { g1 };
        let mut c = [Gv::ZERO; 3];
        c[dir] = a_c(dir, 0);
        c[fa] = a_f(fa, fo);
        c[ca] = a_c(ca, co);
        Tensor::new(c)
    };
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
    // the reconstructed face prim in WORLD component order; the staggered normal
    // face-B overrides the normal component, the edge-reconstructed transverse B the
    // transverse one (identity chart: world index == grid axis).
    let prim_face = |base: &[i32],
                     naxis: usize,
                     sign: f64,
                     n_phys: usize,
                     bn: Gv,
                     t_phys: usize,
                     bt: Gv|
     -> MhdPrim<Gv, 3> {
        let r = |key: &str, rt: &str| recon_cell(key, rt, base, naxis, sign);
        let rho = r("e_rho", "rho");
        let pre = r("e_pre", "pre");
        let mut v = [Gv::ZERO; 3];
        v[pc1] = r("e_vp1", "vel_p1");
        v[pc2] = r("e_vp2", "vel_p2");
        v[out] = r("e_vout", "vel_out");
        let mut b = [Gv::ZERO; 3];
        b[pc1] = r("e_bp1", "bcell_p1");
        b[pc2] = r("e_bp2", "bcell_p2");
        b[out] = r("e_bout", "bcell_out");
        b[n_phys] = bn;
        b[t_phys] = bt;
        MhdPrim::<Gv, 3> {
            hydro: Prim {
                rho,
                vel: Tensor::new(v),
                pre,
            },
            mag: Tensor::new(b),
        }
    };
    let bx_n = recon_face_to_edge(ndim, theta, "e_bface_a", "bface_a", &zero, g2, -1.0);
    let bx_s = recon_face_to_edge(ndim, theta, "e_bface_a", "bface_a", &se, g2, 1.0);
    let by_w = recon_face_to_edge(ndim, theta, "e_bface_b", "bface_b", &nw, g1, 1.0);
    let by_e = recon_face_to_edge(ndim, theta, "e_bface_b", "bface_b", &zero, g1, -1.0);
    let bx_n_face = gv_field_at("e_bface_a", "bface_a", ndim, &ne);
    let bx_s_face = gv_field_at("e_bface_a", "bface_a", ndim, &se);
    let by_w_face = gv_field_at("e_bface_b", "bface_b", ndim, &nw);
    let by_e_face = gv_field_at("e_bface_b", "bface_b", ndim, &ne);
    // the wave-sum Phi (coordinate frame, contravariant B jumps), success -> HLL
    // fallback; every wave speed enters relative to the moving interface vf.
    let wave_sum = |st: &HlldStates<Gv, 3>, t: usize, bt_l: Gv, bt_r: Gv, vf: Gv| -> Gv {
        let (l0, l1) = (st.lam[0] - vf, st.lam[1] - vf);
        let (a0, a1) = (st.alf[0] - vf, st.alf[1] - vf);
        let phi_hlld = half
            * hlld_wave_sum_terms(
                l0.abs(),
                a0.abs(),
                a1.abs(),
                l1.abs(),
                bt_l,
                st.bstar[0][t],
                st.bc[t],
                st.bstar[1][t],
                bt_r,
            );
        let ap = zero_g.max(l1);
        let am = zero_g.max(zero_g - l0);
        let phi_hll = uct_hll_coeffs(ap, am).dl * (bt_r - bt_l);
        Gv::select(st.success.cmp_gt(half), phi_hlld, phi_hll)
    };
    // the grid-g1-face riemann (world normal pc1) at its two grid-g2 cell centers.
    let xn_pos = face_pos(g1, 0, 0);
    let (a_xn, _, b_xn) = gr_adm_at(spacetime, coords, xn_pos);
    let vf_xn = b_xn[pc1] / a_xn;
    let xn_l = prim_face(&nw, g1, 1.0, pc1, bx_n_face, pc2, by_w);
    let xn_r = prim_face(&ne, g1, -1.0, pc1, bx_n_face, pc2, by_e);
    let st_xn = hlld_rmhd_states_gr_ortho(&eos, &xn_l, &xn_r, pc1, &metric_at(xn_pos));
    let xs_pos = face_pos(g1, 0, -1);
    let (a_xs, _, b_xs) = gr_adm_at(spacetime, coords, xs_pos);
    let vf_xs = b_xs[pc1] / a_xs;
    let xs_l = prim_face(&sw, g1, 1.0, pc1, bx_s_face, pc2, by_w);
    let xs_r = prim_face(&se, g1, -1.0, pc1, bx_s_face, pc2, by_e);
    let st_xs = hlld_rmhd_states_gr_ortho(&eos, &xs_l, &xs_r, pc1, &metric_at(xs_pos));
    let phi_x = avg2(
        wave_sum(&st_xn, pc2, by_w, by_e, vf_xn),
        wave_sum(&st_xs, pc2, by_w, by_e, vf_xs),
    );
    // the grid-g2-face riemann (world normal pc2) at its two grid-g1 cell centers.
    let yw_pos = face_pos(g2, 0, -1);
    let (a_yw, _, b_yw) = gr_adm_at(spacetime, coords, yw_pos);
    let vf_yw = b_yw[pc2] / a_yw;
    let yw_l = prim_face(&sw, g2, 1.0, pc2, by_w_face, pc1, bx_s);
    let yw_r = prim_face(&nw, g2, -1.0, pc2, by_w_face, pc1, bx_n);
    let st_yw = hlld_rmhd_states_gr_ortho(&eos, &yw_l, &yw_r, pc2, &metric_at(yw_pos));
    let ye_pos = face_pos(g2, 0, 0);
    let (a_ye, _, b_ye) = gr_adm_at(spacetime, coords, ye_pos);
    let vf_ye = b_ye[pc2] / a_ye;
    let ye_l = prim_face(&se, g2, 1.0, pc2, by_e_face, pc1, bx_s);
    let ye_r = prim_face(&ne, g2, -1.0, pc2, by_e_face, pc1, bx_n);
    let st_ye = hlld_rmhd_states_gr_ortho(&eos, &ye_l, &ye_r, pc2, &metric_at(ye_pos));
    let phi_y = avg2(
        wave_sum(&st_yw, pc1, bx_s, bx_n, vf_yw),
        wave_sum(&st_ye, pc1, bx_s, bx_n, vf_ye),
    );
    // advective transport velocity at the corner: vtilde = alpha v - beta on each
    // transverse axis; the coordinate corner EMF densitized by sqrt(gamma)(corner).
    let (alpha_c, sqrtg_c, beta_c) = gr_adm_at(spacetime, coords, corner);
    let (beta_x_c, beta_y_c) = (beta_c[pc1], beta_c[pc2]);
    let vp1 = |o: &[i32]| gv_field_at("e_vp1", "vel_p1", ndim, o);
    let vp2 = |o: &[i32]| gv_field_at("e_vp2", "vel_p2", ndim, o);
    let vx_w = alpha_c * avg2(vp1(&nw), vp1(&sw)) - beta_x_c;
    let vx_e = alpha_c * avg2(vp1(&ne), vp1(&se)) - beta_x_c;
    let vy_s = alpha_c * avg2(vp2(&sw), vp2(&se)) - beta_y_c;
    let vy_n = alpha_c * avg2(vp2(&nw), vp2(&ne)) - beta_y_c;
    let e_dir =
        zero_g - half * (vx_e * by_e + vx_w * by_w) + half * (vy_n * bx_n + vy_s * bx_s) + phi_x
            - phi_y;
    let emf = sqrtg_c * e_dir;
    (
        end_trace(),
        vec![("emf".to_string(), "emf".into(), emf.node())],
    )
}

/// the RK2 edge-EMF save `e_n = e` (pointwise copy; the generic 2-buffer copy the runtime also
/// reuses for the bcell^n snapshot). write root == the read field node.
pub fn rmhd_save_efield_gv() -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let e = Gv::field("e", "e");
    (
        end_trace(),
        vec![("e_n".to_string(), "e_n".into(), e.node())],
    )
}

/// the RK2 edge-EMF time-average `e = 0.5*(e + e_n)`, in-place on e. mirror of
/// `rmhd::rmhd_average_efield`.
pub fn rmhd_average_efield_gv() -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let e = Gv::field("e", "e");
    let en = Gv::field("e_n", "e_n");
    let e_new = Gv::from_f64(0.5) * (e + en);
    (
        end_trace(),
        vec![("e_new".to_string(), "e".into(), e_new.node())],
    )
}

/// the FOFC EDGE-EMF SPLICE for an edge whose two transverse GRID axes are `g1`, `g2`: choose the
/// FIRST-ORDER edge EMF (`e_fo`, the live `efield` just recomputed by the Contact/HLL corner kernel)
/// on edges touching a flagged cell, else the saved HIGH-ORDER EMF (`e_ho`, `efield_ho`). the edge at
/// `coord` is incident to the four corner cells `coord`, `coord - e_g1`, `coord - e_g2`,
/// `coord - e_g1 - e_g2` — the EXACT gather of the edge-EMF kernel (`rmhd_edge_emf_gv`), so the flag
/// is read at the same offsets and the edge is first-order iff ANY of the four incident cells is
/// flagged. `e_fo` is read+write IN PLACE. after the splice the curl of this single-valued edge EMF
/// gives flagged cells first-order (diffused, recoverable) B while leaving non-flagged faces at the
/// high-order value and preserving div(B) = 0.
pub fn fofc_emf_splice_gv(
    ndim: usize,
    g1: usize,
    g2: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let zero = vec![0i32; ndim];
    let cm = |axes: &[usize]| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        for &ax in axes {
            o[ax] = -1;
        }
        o
    };
    let flag_gt0 = |o: &[i32]| gv_field_at("flag", "flag", ndim, o).cmp_gt(Gv::ZERO);
    let edge_fo =
        flag_gt0(&zero) | flag_gt0(&cm(&[g1])) | flag_gt0(&cm(&[g2])) | flag_gt0(&cm(&[g1, g2]));
    let e_fo = Gv::field("e_fo", "e_fo");
    let e_ho = Gv::field("e_ho", "e_ho");
    let chosen = Gv::select(edge_fo, e_fo, e_ho);
    (
        end_trace(),
        vec![("e_fo".to_string(), "e_fo".into(), chosen.node())],
    )
}

/// the CURVED-SPACETIME CT edge EMF for the FULL 3D grid, along edge axis `dir` — the contact
/// assembly of [`rmhd_edge_emf_gv`] producing the DENSITIZED corner EMF the GR curl consumes:
///   cell terms:  sqrt(gamma)|cell x [(alpha v_p2 - beta_p2) b_p1 - (alpha v_p1 - beta_p1) b_p2]
///   face terms:  alpha sqrt(gamma)|face x (raw mag-row flux)
/// every gather point carries the metric factors at ITS OWN 3d position: the edge is
/// cell-centered along `dir` and cornered on the transverse axes (g1, g2) = (dir+1, dir+2) mod 3;
/// cells sit at arithmetic centers, the bflux_a points on the g1-faces, the bflux_b points on the
/// g2-faces. the density fluxes stay RAW (the soft-sign blend uses only their signs, and the
/// alpha sqrt(gamma) measure is positive). identity chart only: physical component == grid axis,
/// so the shift components are addressed directly by (g1, g2).
pub fn rmhd_edge_emf_gr_3d_gv(
    dir: usize,
    spacetime: Spacetime,
    coords: Coords,
    spacing: &[Spacing],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let ndim = 3usize;
    let g1 = (dir + 1) % 3;
    let g2 = (dir + 2) % 3;
    gv_register_field("edge_vp1", "vel_p1");
    gv_register_field("edge_vp2", "vel_p2");
    gv_register_field("edge_bp1", "bcell_p1");
    gv_register_field("edge_bp2", "bcell_p2");
    gv_register_field("edge_bflux_a", "bflux_a");
    gv_register_field("edge_bflux_b", "bflux_b");
    gv_register_field("edge_fden_p1", "fden_p1");
    gv_register_field("edge_fden_p2", "fden_p2");
    let cm = |axes: &[usize]| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        for &ax in axes {
            o[ax] = -1;
        }
        o
    };
    let zero = vec![0i32; ndim];
    let half = Gv::from_f64(0.5);
    // per-axis coordinates: faces at integer offsets, arithmetic cell centers between them.
    let a_f = |ax: usize, o: i64| gv_axis_face_at(ax, spacing[ax], o);
    let a_c = |ax: usize, o: i64| (a_f(ax, o) + a_f(ax, o + 1)) * half;
    // a full 3d position: cell-centered along `dir`; the transverse coords supplied per point.
    let pos = |c_g1: Gv, c_g2: Gv| -> Tensor<Gv, 3> {
        let mut x = [Gv::ZERO; 3];
        x[dir] = a_c(dir, 0);
        x[g1] = c_g1;
        x[g2] = c_g2;
        Tensor::new(x)
    };
    let adm = |c_g1: Gv, c_g2: Gv| -> (Gv, Gv, Gv, Gv) {
        let (alpha, sqrtg, beta) = gr_adm_at(spacetime, coords, pos(c_g1, c_g2));
        (alpha, sqrtg, beta[g1], beta[g2])
    };
    // densitized cell EMF at the cell whose LOW corner offset (on g1, g2) is `o`.
    let cell = |o: &[i32]| -> Gv {
        let (alpha, sqrtg, beta_p1, beta_p2) = adm(a_c(g1, o[g1] as i64), a_c(g2, o[g2] as i64));
        let vp1 = gv_field_at("edge_vp1", "vel_p1", ndim, o);
        let vp2 = gv_field_at("edge_vp2", "vel_p2", ndim, o);
        let bp1 = gv_field_at("edge_bp1", "bcell_p1", ndim, o);
        let bp2 = gv_field_at("edge_bp2", "bcell_p2", ndim, o);
        sqrtg * ((alpha * vp2 - beta_p2) * bp1 - (alpha * vp1 - beta_p1) * bp2)
    };
    let ene = cell(&zero);
    let enw = cell(&cm(&[g1]));
    let ese = cell(&cm(&[g2]));
    let esw = cell(&cm(&[g1, g2]));
    // densitized face EMFs: alpha sqrt(gamma) at the face's own point times the raw flux.
    let asg = |c_g1: Gv, c_g2: Gv| -> Gv {
        let (alpha, sqrtg, ..) = adm(c_g1, c_g2);
        alpha * sqrtg
    };
    let en = Gv::ZERO
        - asg(a_f(g1, 0), a_c(g2, 0)) * gv_field_at("edge_bflux_a", "bflux_a", ndim, &zero);
    let es = Gv::ZERO
        - asg(a_f(g1, 0), a_c(g2, -1)) * gv_field_at("edge_bflux_a", "bflux_a", ndim, &cm(&[g2]));
    let ee = asg(a_c(g1, 0), a_f(g2, 0)) * gv_field_at("edge_bflux_b", "bflux_b", ndim, &zero);
    let ew =
        asg(a_c(g1, -1), a_f(g2, 0)) * gv_field_at("edge_bflux_b", "bflux_b", ndim, &cm(&[g1]));
    let fnf = gv_field_at("edge_fden_p1", "fden_p1", ndim, &zero);
    let fs = gv_field_at("edge_fden_p1", "fden_p1", ndim, &cm(&[g2]));
    let fe = gv_field_at("edge_fden_p2", "fden_p2", ndim, &zero);
    let fw = gv_field_at("edge_fden_p2", "fden_p2", ndim, &cm(&[g1]));
    let emf = ct_contact_emf_gv([en, es, ee, ew], [ene, enw, ese, esw], [fnf, fs, fe, fw]);
    (
        end_trace(),
        vec![("emf".to_string(), "emf".into(), emf.node())],
    )
}

/// the CURVED-SPACETIME 3D CT face-B update along face axis `dir` from the two incident
/// DENSITIZED edge EMFs: the coordinate curl of Etilde divided by the face's own
/// sqrt(gamma) — the densitized induction d(sqrt(gamma) B)/dt = -curl(Etilde) solved for
/// B at the face point, so div(sqrt(gamma) B) is preserved exactly (the coordinate curl
/// telescopes for any weight).
pub fn rmhd_ct_curl_3d_gr_dir_gv(
    dir: usize,
    spacetime: Spacetime,
    coords: Coords,
    spacing: &[Spacing],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let p1 = (dir + 1) % 3;
    let p2 = (dir + 2) % 3;
    let b = Gv::field("b", "b");
    let dt = Gv::scalar("dt");
    // positional scalar ABI: pin every axis's face coords up front (the runtime pushes the
    // full geom set in axis order; liveness pruning must not shift the binding).
    for ax in 0..3 {
        let _ = gv_axis_face_at(ax, spacing[ax], 0);
        let _ = gv_axis_face_at(ax, spacing[ax], 1);
    }
    let half = Gv::from_f64(0.5);
    let a_f = |ax: usize, o: i64| gv_axis_face_at(ax, spacing[ax], o);
    let a_c = |ax: usize| (a_f(ax, 0) + a_f(ax, 1)) * half;
    // sqrt(gamma) at THIS face's center: on the low dir-face, centered transversely.
    let face_pos = {
        let mut x = [Gv::ZERO; 3];
        x[dir] = a_f(dir, 0);
        x[p1] = a_c(p1);
        x[p2] = a_c(p2);
        Tensor::new(x)
    };
    let sqrtg = gr_adm_at(spacetime, coords, face_pos).1;
    let off = |ax: usize| -> [i32; 3] {
        let mut o = [0, 0, 0];
        o[ax] = 1;
        o
    };
    let inv_dx = |ax: usize| Gv::ONE / (a_f(ax, 1) - a_f(ax, 0));
    let de = |key: &str, runtime: &str, ax: usize| -> Gv {
        let e_h = gv_field_at(key, runtime, 3, &[0, 0, 0]);
        let e_p = gv_field_at(key, runtime, 3, &off(ax));
        (e_p - e_h) * inv_dx(ax)
    };
    let de1 = de("e_p1", "e_p1", p2);
    let de2 = de("e_p2", "e_p2", p1);
    let b_new = b + dt * (de1 - de2) / sqrtg;
    (
        end_trace(),
        vec![("b_new".to_string(), "b".into(), b_new.node())],
    )
}
