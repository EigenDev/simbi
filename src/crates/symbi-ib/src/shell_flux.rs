// =============================================================================
// shell_flux.rs
//
// the accretion diagnostic for a first-class immersed boundary: the net flux of
// (rest mass, momentum, covariant energy, angular momentum) INTO the region
// Omega = { phi(x) < 0 } through its boundary, summed from the code's OWN
// densitized numerical face fluxes. divergence-theorem consistent with the
// finite-volume update -- the diagnostic flux IS the flux the scheme applies
// across those faces, so a discretely divergence-free flow gives a phi-invariant
// (surface-independent) rate to roundoff. phi is a level set: r_ks - r_d for a
// horizon shell, the surface sdf for a material body later -- one reduction.
//
// the ledger reuses BodyDelta: mass_delta = Mdot*dt, energy_delta = Edot*dt
// (covariant/killing energy, exactly conserved), force_delta = the momentum flux
// (the accretion thrust on the hole), torque_delta = Ldot*dt = the moment
// r x F_S (the spin-up angular-momentum flux).
//
// usage:
//   let delta = shell_accretion(shape, |c| phi(c), |c,a| face_flux(c,a),
//                               |c,a| face_centroid(c,a), body_idx, dt);
// =============================================================================

use symbi_algebra::Tensor;

use crate::body_delta::BodyDelta;

/// the densitized numerical flux through one coordinate face (in the `+axis`
/// direction): `numerical_flux * sqrt(gamma) * dA_coordinate` -- the same quantity
/// the godunov divergence consumes, so the diagnostic is bit-consistent with the
/// evolution. `nrg` is the COVARIANT energy flux on a GR path.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FaceFlux<const D: usize> {
    pub den: f64,
    pub mom: Tensor<f64, D>,
    pub nrg: f64,
}

/// row-major unravel of a flat index into a `[i0, i1, ...]` multi-index.
fn unravel<const D: usize>(flat: usize, shape: [usize; D]) -> [usize; D] {
    let mut idx = [0usize; D];
    let mut r = flat;
    for a in (0..D).rev() {
        idx[a] = r % shape[a];
        r /= shape[a];
    }
    idx
}

/// pad a spatial vector to 3 slots (the suppressed axes carry zero) for the
/// angular-momentum cross product.
fn pad3<const D: usize>(v: &Tensor<f64, D>) -> Tensor<f64, 3> {
    Tensor::new(std::array::from_fn(|k| if k < D { v[k] } else { 0.0 }))
}

fn cross3(a: Tensor<f64, 3>, b: Tensor<f64, 3>) -> Tensor<f64, 3> {
    Tensor::new([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
}

/// accumulate the accretion ledger of the region `Omega = { phi < 0 }` from the
/// densitized face fluxes. a face between an inside cell and an outside cell is a
/// boundary face; the body accretes the INWARD flux (`-` the outward normal flux),
/// integrated over `dt`. interior faces (both sides inside, or both outside) cancel
/// by construction, so the result is the exact discrete boundary integral.
///
/// `face_flux(c, a)` / `face_centroid(c, a)` describe the `+a`-normal face of cell
/// `c` (between `c` and `c + e_a`); the domain-edge face (no `+a` neighbor) is
/// skipped -- `Omega` is expected strictly interior.
pub fn shell_accretion<const D: usize>(
    shape: [usize; D],
    phi: impl Fn([usize; D]) -> f64,
    face_flux: impl Fn([usize; D], usize) -> FaceFlux<D>,
    face_centroid: impl Fn([usize; D], usize) -> Tensor<f64, D>,
    body_idx: usize,
    dt: f64,
) -> BodyDelta<f64, D> {
    let mut d = BodyDelta::new(body_idx);
    let total: usize = shape.iter().product();
    for flat in 0..total {
        let c = unravel(flat, shape);
        let inside_c = phi(c) < 0.0;
        for a in 0..D {
            if c[a] + 1 >= shape[a] {
                continue; // domain-edge face: no +a neighbor
            }
            let mut cn = c;
            cn[a] += 1;
            let inside_cn = phi(cn) < 0.0;
            if inside_c == inside_cn {
                continue; // not a boundary face of Omega
            }
            // outward normal out of Omega points along +a when the inside cell is on the
            // -a side (c inside), along -a when the inside cell is on the +a side (cn inside).
            let outward_sign = if inside_c { 1.0 } else { -1.0 };
            // accreted (into Omega) = -(outward flux), integrated over dt.
            let w = -outward_sign * dt;
            let f = face_flux(c, a);
            d.mass_delta += w * f.den;
            d.energy_delta += w * f.nrg;
            d.force_delta = d.force_delta + f.mom.scale(w);
            let moment = cross3(pad3(&face_centroid(c, a)), pad3(&f.mom));
            d.torque_delta = d.torque_delta + moment.scale(w);
        }
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the discrete divergence theorem: the boundary integral the reduction computes MUST
    /// equal the volume sum of the per-cell net outward flux over Omega, to roundoff, for an
    /// ARBITRARY flux field. this is the exactness that makes a steady flow phi-invariant.
    #[test]
    fn boundary_integral_equals_the_volume_divergence() {
        let shape = [12usize, 12];
        // an arbitrary (not divergence-free) densitized flux field on the +a faces.
        let flux = |c: [usize; 2], a: usize| -> FaceFlux<2> {
            let (i, j) = (c[0] as f64, c[1] as f64);
            let s = 0.37 * i - 0.21 * j + 1.7 * (a as f64) + 0.9;
            FaceFlux { den: s, mom: Tensor::new([0.11 * s, -0.4 * s]), nrg: 2.3 * s }
        };
        let centroid = |c: [usize; 2], a: usize| {
            let mut x = [c[0] as f64 + 0.5, c[1] as f64 + 0.5];
            x[a] += 0.5; // face center sits on the +a boundary of the cell
            Tensor::new(x)
        };
        // Omega: a disk strictly interior to the grid.
        let (cx, cy, rd) = (6.0, 6.0, 3.2);
        let phi = |c: [usize; 2]| {
            let (dx, dy) = (c[0] as f64 + 0.5 - cx, c[1] as f64 + 0.5 - cy);
            (dx * dx + dy * dy).sqrt() - rd
        };
        let dt = 0.5;
        let d = shell_accretion(shape, phi, flux, centroid, 0, dt);
        // the reduction's outward mass flux = -mass_delta/dt.
        let boundary_outward = -d.mass_delta / dt;

        // independent volume sum: for each Omega cell, its net outward den flux across all 2D
        // faces (the -a face of c is the +a face of c - e_a). interior faces cancel pairwise.
        let mut volume_div = 0.0;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                if phi([i, j]) >= 0.0 {
                    continue;
                }
                for a in 0..2 {
                    volume_div += flux([i, j], a).den; // +a face outward
                    let mut cm = [i, j];
                    cm[a] -= 1; // -a face of c = +a face of c-e_a; outward normal is -a
                    volume_div -= flux(cm, a).den;
                }
            }
        }
        assert!(
            (boundary_outward - volume_div).abs() < 1e-11 * (1.0 + volume_div.abs()),
            "divergence theorem: boundary {boundary_outward} != volume {volume_div}"
        );
    }

    /// phi-invariance for a discretely divergence-free field: two nested shells enclosing the
    /// same (zero) net source give the SAME accretion rate to roundoff -- the certificate a
    /// steady accretion flow must satisfy.
    #[test]
    fn nested_shells_agree_for_a_divergence_free_field() {
        let shape = [16usize, 16];
        // a uniform flux F_x = k on every x-face, F_y = 0: each cell's net flux is k - k = 0.
        let flux = |_c: [usize; 2], a: usize| -> FaceFlux<2> {
            FaceFlux { den: if a == 0 { 1.3 } else { 0.0 }, mom: Tensor::zeros(), nrg: 0.0 }
        };
        let centroid = |c: [usize; 2], _a: usize| Tensor::new([c[0] as f64, c[1] as f64]);
        let disk = |rd: f64| move |c: [usize; 2]| {
            let (dx, dy) = (c[0] as f64 + 0.5 - 8.0, c[1] as f64 + 0.5 - 8.0);
            (dx * dx + dy * dy).sqrt() - rd
        };
        let m_small = shell_accretion(shape, disk(2.5), flux, centroid, 0, 1.0).mass_delta;
        let m_large = shell_accretion(shape, disk(5.5), flux, centroid, 0, 1.0).mass_delta;
        assert!(m_small.abs() < 1e-12, "closed shell of a div-free field is zero: {m_small}");
        assert!((m_small - m_large).abs() < 1e-12, "nested shells disagree: {m_small} vs {m_large}");
    }

    /// sign + angular momentum: a single-cell Omega with a purely azimuthal (circulating)
    /// inward flux accretes positive mass and a nonzero z-torque.
    #[test]
    fn single_cell_infall_signs() {
        // 3x3 grid, Omega = center cell [1,1] only.
        let shape = [3usize, 3];
        let phi = |c: [usize; 2]| if c == [1, 1] { -1.0 } else { 1.0 };
        // radial infall: mass flux points toward the center on every boundary face.
        // +x face of center (c=[1,1], a=0) -> flow in -x (into center): den < 0.
        // +x face of left cell (c=[0,1], a=0) is the -x boundary of center -> flow in +x: den > 0.
        let flux = |c: [usize; 2], a: usize| -> FaceFlux<2> {
            let den = match (c, a) {
                ([1, 1], 0) => -1.0, // center's +x face: inward is -x
                ([0, 1], 0) => 1.0,  // center's -x face: inward is +x
                ([1, 1], 1) => -1.0, // center's +y face
                ([1, 0], 1) => 1.0,  // center's -y face
                _ => 0.0,
            };
            FaceFlux { den, mom: Tensor::new([-0.5 * (a == 1) as u8 as f64, 0.5 * (a == 0) as u8 as f64]), nrg: 3.0 * den }
        };
        let centroid = |c: [usize; 2], a: usize| {
            let mut x = [c[0] as f64 + 0.5, c[1] as f64 + 0.5];
            x[a] += 0.5;
            Tensor::new(x)
        };
        let d = shell_accretion(shape, phi, flux, centroid, 7, 2.0);
        assert_eq!(d.idx, 7);
        // four inward faces, each |den| = 1, dt = 2 -> Mdot*dt = 4 * 1 * 2 = 8.
        assert!((d.mass_delta - 8.0).abs() < 1e-12, "accreted mass {} != 8", d.mass_delta);
        // energy flux is 3x the mass flux here.
        assert!((d.energy_delta - 24.0).abs() < 1e-12, "accreted energy {} != 24", d.energy_delta);
        // a circulating momentum flux gives a nonzero net z-torque (spin-up).
        assert!(d.torque_delta[2].abs() > 1e-9, "expected nonzero z-torque, got {}", d.torque_delta[2]);
    }
}
