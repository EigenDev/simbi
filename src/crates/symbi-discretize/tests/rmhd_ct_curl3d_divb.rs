// =============================================================================
// rmhd_ct_curl3d_divb.rs
//
// proves the 3D constrained-transport curl update (rmhd_ct_curl_3d_dir, run once
// per face axis) preserves div(B) = 0 to machine precision — the defining CT
// property, in three dimensions (the 2D result generalized). B is
// initialized divergence-free from a 3D discrete vector potential A (B = curl A,
// so the discrete div telescopes to 0), then evolved one step by the curl of an
// arbitrary edge EMF E = (Ex, Ey, Ez). the discrete curl + discrete divergence
// share the mixed E differences, so d(div B)/dt is identically 0.
//
// each per-dir kernel updates one bface component from its two transverse edge
// fields E_p1 / E_p2 (p1=(dir+1)%3, p2=(dir+2)%3); the interpreter run uses
// out-of-place writes so the before/after div comparison can read the originals.
//
// staggering (cell-indexed, axis-0-fastest flat = i + (j + k*M)*M, the canonical symbi
// `Field`/`View` convention the harness output buffer uses): Bx/By/Bz on faces, Ex/Ey/Ez on edges.
//   div(B)[i,j,k] = idx*(Bx[i+1]-Bx) + idy*(By[j+1]-By) + idz*(Bz[k+1]-Bz)
// =============================================================================

mod harness;
use harness::KernelRun;

use symbi_discretize::{Coords, Spacing, rmhd_ct_curl_3d_dir_gv};

const M: usize = 8; // buffer extent per axis
const IDX: f64 = 8.0;
const IDY: f64 = 8.0;
const IDZ: f64 = 8.0;
const DT: f64 = 0.011;

fn idx3(i: usize, j: usize, k: usize) -> usize {
    i + (j + k * M) * M // axis-0-fastest, matching the harness/interp/Field storage convention
}
fn at(b: &[f64], i: usize, j: usize, k: usize) -> f64 {
    b[idx3(i, j, k)]
}

// div(B) at cell (i,j,k) from the face-centered B (needs Bx[i+1], By[j+1], Bz[k+1]).
fn div_b(bx: &[f64], by: &[f64], bz: &[f64], i: usize, j: usize, k: usize) -> f64 {
    IDX * (at(bx, i + 1, j, k) - at(bx, i, j, k))
        + IDY * (at(by, i, j + 1, k) - at(by, i, j, k))
        + IDZ * (at(bz, i, j, k + 1) - at(bz, i, j, k))
}

#[test]
fn ct_curl3d_preserves_div_b() {
    // an arbitrary smooth 3D vector potential A and edge EMF E.
    let f = M * M * M;
    let ax = |i: usize, j: usize, k: usize| {
        (0.3 * i as f64).sin() * (0.2 * j as f64 + 0.1 * k as f64).cos()
    };
    let ay = |i: usize, j: usize, k: usize| {
        (0.25 * j as f64).cos() * (0.15 * k as f64 - 0.2 * i as f64).sin()
    };
    let az = |i: usize, j: usize, k: usize| {
        (0.2 * k as f64).sin() * (0.3 * i as f64 + 0.1 * j as f64).cos()
    };
    let efn = |s: f64, i: usize, j: usize, k: usize| {
        (0.4 * i as f64 + s).sin() * (0.3 * j as f64).cos() + 0.2 * (0.2 * k as f64 - s).sin()
    };

    // divergence-free init: B = curl(A), defined on i,j,k in [0, M-1).
    let (mut bx, mut by, mut bz) = (vec![0.0; f], vec![0.0; f], vec![0.0; f]);
    let (mut ex, mut ey, mut ez) = (vec![0.0; f], vec![0.0; f], vec![0.0; f]);
    for i in 0..M {
        for j in 0..M {
            for k in 0..M {
                ex[idx3(i, j, k)] = efn(0.0, i, j, k);
                ey[idx3(i, j, k)] = efn(1.0, i, j, k);
                ez[idx3(i, j, k)] = efn(2.0, i, j, k);
            }
        }
    }
    for i in 0..M - 1 {
        for j in 0..M - 1 {
            for k in 0..M - 1 {
                // Bx = dAz/dy - dAy/dz ; By = dAx/dz - dAz/dx ; Bz = dAy/dx - dAx/dy.
                bx[idx3(i, j, k)] =
                    IDY * (az(i, j + 1, k) - az(i, j, k)) - IDZ * (ay(i, j, k + 1) - ay(i, j, k));
                by[idx3(i, j, k)] =
                    IDZ * (ax(i, j, k + 1) - ax(i, j, k)) - IDX * (az(i + 1, j, k) - az(i, j, k));
                bz[idx3(i, j, k)] =
                    IDX * (ay(i + 1, j, k) - ay(i, j, k)) - IDY * (ax(i, j + 1, k) - ax(i, j, k));
            }
        }
    }

    // sanity: init is divergence-free to machine precision.
    for i in 0..M - 2 {
        for j in 0..M - 2 {
            for k in 0..M - 2 {
                assert!(
                    div_b(&bx, &by, &bz, i, j, k).abs() < 1e-11,
                    "init div(B) nonzero at {i},{j},{k}"
                );
            }
        }
    }

    let id = [IDX, IDY, IDZ];

    // per-dir CT curl: update bface[dir] from E_p1/E_p2 over [0, M-1)^3, into a fresh buffer.
    // out-of-place writes (the single write b_new) so the before/after div comparison reads
    // the originals. b = bface[dir]; e_p1 = efield[p1]; e_p2 = efield[p2].
    let pick = |a: usize| -> &[f64] {
        match a {
            0 => &ex,
            1 => &ey,
            _ => &ez,
        }
    };
    let mut new_b: [Vec<f64>; 3] = [vec![0.0; f], vec![0.0; f], vec![0.0; f]];
    for dir in 0..3usize {
        let p1 = (dir + 1) % 3;
        let p2 = (dir + 2) % 3;
        let built = rmhd_ct_curl_3d_dir_gv(Coords::Cartesian, &[Spacing::Uniform; 3], dir);
        assert_eq!(
            built.0.scalar_params(),
            vec!["dt".to_string(), "id_p1".to_string(), "id_p2".to_string()]
        );
        let (bvec, ep1, ep2) = (
            (match dir {
                0 => &bx,
                1 => &by,
                _ => &bz,
            })
            .clone(),
            pick(p1).to_vec(),
            pick(p2).to_vec(),
        );
        let out = KernelRun::new(built)
            .grid([M, M, M])
            .compute_window([0, 0, 0], [M - 1, M - 1, M - 1])
            .field_with("b", move |c| bvec[idx3(c[0], c[1], c[2])])
            .field_with("e_p1", move |c| ep1[idx3(c[0], c[1], c[2])])
            .field_with("e_p2", move |c| ep2[idx3(c[0], c[1], c[2])])
            .scalars(&[("dt", DT), ("id_p1", id[p1]), ("id_p2", id[p2])])
            .run();
        new_b[dir] = out.values("b_new").to_vec();
    }
    let (bxn, byn, bzn) = (&new_b[0], &new_b[1], &new_b[2]);

    // div(B) after the update must still be machine zero (and unchanged from before).
    for i in 0..M - 2 {
        for j in 0..M - 2 {
            for k in 0..M - 2 {
                let after = div_b(bxn, byn, bzn, i, j, k);
                let before = div_b(&bx, &by, &bz, i, j, k);
                assert!(
                    after.abs() < 1e-11,
                    "post-update div(B) nonzero at {i},{j},{k}: {after}"
                );
                assert!(
                    (after - before).abs() < 1e-12,
                    "div(B) changed at {i},{j},{k}: {before} -> {after}"
                );
            }
        }
    }
}
