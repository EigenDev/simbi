// =============================================================================
// bm_skymap.rs
//
// make a synthetic GRB afterglow sky image from a Blandford-McKee blast wave and
// write it as a PGM (portable graymap — a trivial text image format, no deps; open
// it with any image viewer or `convert bm_skymap.pgm bm_skymap.png`). also prints the
// equal-area radial surface-brightness profile, which shows the canonical
// limb-brightened ring for an on-axis viewer.
//
// flags:
//   --t-obs <day>      observer time (default 1.0)
//   --theta-obs <deg>     observer angle from the +z (sector) axis (default 0 = on-axis)
//   --theta-sector <deg>  emission sector half-angle (default 180 = full sphere; a smaller
//                         angle is a jet, a ring, or any angularly-bounded outflow)
//   --bolometric          use delta^4 (frequency-integrated) instead of delta^3 (band)
//   --n-pix <n>           image size (default 256)
//   --out <path>          output .pgm (default bm_skymap.pgm)
//
// usage:
//   cargo run --release -p symbi-afterglow --example bm_skymap -- --t-obs 1.0 --out bm.pgm
//   cargo run --release -p symbi-afterglow --example bm_skymap -- --theta-sector 6 --theta-obs 9
// =============================================================================

use std::io::Write;

use symbi_afterglow::observe::{DOPPLER_BAND, DOPPLER_BOLOMETRIC, compute_skymap};
use symbi_afterglow::synthesize_afterglow_events;

fn main() {
    let mut t_obs_day: f64 = 1.0;
    let mut theta_obs_deg: f64 = 0.0;
    let mut theta_sector_deg: f64 = 180.0;
    let mut bolometric = false;
    let mut n_pix = 256usize;
    let mut out_path = "bm_skymap.pgm".to_string();

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let next = |i: usize| args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--t-obs" => { t_obs_day = next(i).parse().unwrap(); i += 2 }
            "--theta-obs" => { theta_obs_deg = next(i).parse().unwrap(); i += 2 }
            "--theta-sector" => { theta_sector_deg = next(i).parse().unwrap(); i += 2 }
            "--n-pix" => { n_pix = next(i).parse().unwrap(); i += 2 }
            "--out" => { out_path = next(i); i += 2 }
            "--bolometric" => { bolometric = true; i += 1 }
            _ => i += 1,
        }
    }

    let theta_obs = theta_obs_deg.to_radians();
    let theta_sector = theta_sector_deg.to_radians();
    let doppler_power = if bolometric { DOPPLER_BOLOMETRIC } else { DOPPLER_BAND };
    // observer direction: angle theta_obs from the jet (+z) axis, in the x-z plane.
    let obs_dir = [theta_obs.sin(), 0.0, theta_obs.cos()];

    // E_iso = 1e53 erg, n0 = 1 cm^-3, p = 2.5, d_L = 1e26 cm. dense lab-time sampling (many
    // snapshots) smooths the radial profile — each snapshot's EATS slice lands in ~one ring.
    let events = synthesize_afterglow_events(
        1.0e53, 1.0, 2.5, 0.1, 0.01, 0.0, 1.0e26, t_obs_day, theta_sector, 0.5, 2.5, 400, 30.0, 6,
        64, 128, 1, 1, 60_000_000,
    );
    println!("synthesized {} photon packets", events.len());

    // a wider arrival window blurs the discrete EATS slices into a continuous image.
    let img = compute_skymap(
        &events, obs_dir, t_obs_day, 2.0 * t_obs_day, 0.0, 1.0e30, 0.0, doppler_power, n_pix,
    );

    // radial surface-brightness profile (equal-area annuli) — the limb-brightening diagnostic.
    let prof = img.radial_profile(16);
    let peak = prof.iter().cloned().fold(0.0_f64, f64::max).max(1e-300);
    println!("\nradial surface-brightness profile (center -> edge), normalized to peak:");
    for (i, v) in prof.iter().enumerate() {
        let bar = "#".repeat((40.0 * v / peak) as usize);
        println!("  ring {i:2}  {:6.3}  {bar}", v / peak);
    }

    // write the image as a PGM, normalized to the max pixel. the header carries the physical
    // scale and geometry as a comment (`# key=val ...`) so a plotting script can label the image.
    let maxv = img.intensity.iter().cloned().fold(0.0_f64, f64::max).max(1e-300);
    let mut f = std::fs::File::create(&out_path).expect("create pgm");
    writeln!(f, "P2").unwrap();
    writeln!(
        f,
        "# half_width_cm={:.6e} t_obs_day={} d_l_cm=1e26 e_iso_erg=1e53 n0_cm3=1 p=2.5 \
         theta_obs_deg={} theta_sector_deg={} doppler_power={}",
        img.half_width, t_obs_day, theta_obs_deg, theta_sector_deg, doppler_power
    )
    .unwrap();
    writeln!(f, "{n_pix} {n_pix}\n255").unwrap();
    for iy in 0..n_pix {
        let mut line = String::new();
        for ix in 0..n_pix {
            let g = (255.0 * img.pixel(ix, iy) / maxv).round() as u32;
            line.push_str(&g.to_string());
            line.push(' ');
        }
        writeln!(f, "{}", line.trim_end()).unwrap();
    }
    println!("\nwrote {out_path} ({n_pix}x{n_pix})  half-width = {:.3e} cm", img.half_width);
}
