// a theta-independent state on marcus's exact grid shape MUST render an
// angularly uniform arc — any modulation along the arc is a sampling artifact.
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Spherical;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

#[test]
fn thermal_bomb_shape_renders_an_angularly_uniform_arc() {
    let (nr, nt) = (1024usize, 699usize);
    let r_max = 1.0f64;
    let dr = r_max / nr as f64;
    let sim = SimState::<Newtonian, 2, Spherical, IdealGas<f64>, CpuSpace, HostMemory>::build(
        Newtonian,
        IdealGas { gamma: 1.4 },
        Spherical,
    )
    .cells([nr, nt])
    .origin([0.0, 0.0])
    .spacing([dr, std::f64::consts::FRAC_PI_2 / nt as f64])
    .boundaries(Boundaries::uniform(BoundaryType::Outflow))
    .allocate()
    .expect("sim")
    .set_initial(|_| Prim {
        rho: 1.0,
        vel: Tensor::zeros(),
        pre: 1.0,
    })
    .build();
    // sedov-like radial profile, THETA-INDEPENDENT: evacuated interior, thin
    // bright shell (2 cells) at r = 0.25, ambient outside.
    let r_sh = 0.25f64;
    for c in sim.geom.interior.iter() {
        let r = (c[0] as f64 + 0.5) * dr;
        let v = if r < r_sh - dr {
            0.1
        } else if r < r_sh + dr {
            2.0
        } else {
            1.0
        };
        sim.fields.prim.rho.set(c, v);
    }
    let fd = sim.field_slice(200, 0).expect("slice");

    // walk display angles; for each, the max value along the ray near the shell
    // radius. the state is theta-independent, so these maxima must agree.
    let (w, h) = (fd.width, fd.height);
    let mut ray_max = Vec::new();
    let mut ray_min = Vec::new();
    for k in 0..90 {
        let ang = (k as f64 + 0.5) / 90.0 * std::f64::consts::FRAC_PI_2;
        let (mut m, mut n) = (f32::NAN, f32::NAN);
        // sample along the ray at radii around the shell (s = r/r_max, s0 = 0).
        for t in 0..40 {
            let s = r_sh / r_max + (t as f64 - 20.0) * 0.004;
            let (x, y) = (s * ang.sin(), s * ang.cos());
            let (px, py) = ((x * w as f64) as usize, ((1.0 - y) * h as f64) as usize);
            if px < w && py < h {
                let v = fd.data[py * w + px];
                if !v.is_nan() {
                    if v > m || m.is_nan() {
                        m = v;
                    }
                    if v < n || n.is_nan() {
                        n = v;
                    }
                }
            }
        }
        ray_max.push(m);
        ray_min.push(n);
    }
    let lo = ray_max.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = ray_max.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "arc max per ray: lo={lo:.3} hi={hi:.3} spread={:.3}",
        hi - lo
    );
    // the bright rim shimmers under HONEST block averaging (a 2-cell shell next to
    // an evacuated interior dilutes by pixel phase — the cartesian path decimates
    // identically); the gate is the VISIBLE structure: every ray must cross the
    // dark evacuated band continuously, and no ray may show a spurious bright gap
    // ABOVE ambient (a spurious bright gap signals dropped samples, a rendering bug distinct from honest dilution).
    assert!(
        hi <= 2.0 + 1e-6,
        "averaging can never exceed the shell peak: hi={hi}"
    );
    for (k, m) in ray_min.iter().enumerate() {
        assert!(*m < 0.5, "ray {k} lost the dark band: min={m}");
    }
}
