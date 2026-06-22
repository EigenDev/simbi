// =============================================================================
// checkpoint_skymap.rs
//
// end-to-end: read a REAL symbi HDF5 checkpoint, synthesize its synchrotron photon
// catalog, and write a sky image as a PGM (the same format scripts/plot_bm_skymap.py
// renders). works for any geometry the hydro ran (the adapter maps the axis roles).
//
// flags:
//   --checkpoint <path>   the .h5 checkpoint (required)
//   --t-obs <day>         observer time to image at (default 1.0)
//   --theta-obs <deg>     observer angle from the +z axis (default 0)
//   --window <day>        EATS arrival window full width (default = t-obs)
//   --gamma <g>           adiabatic index (default 4/3, ultrarelativistic)
//   --p / --eps-e / --eps-b   electron index & microphysics (defaults 2.5 / 0.1 / 0.01)
//   --length / --density / --pressure / --time   code->CGS scales (default 1.0)
//   --bolometric          delta^4 weighting instead of delta^3
//   --n-pix <n>           image size (default 256)
//   --out <path>          output .pgm (default checkpoint_skymap.pgm)
//
// usage:
//   cargo run --release -p symbi-afterglow-io --example checkpoint_skymap -- \
//     --checkpoint output/jet_0042.h5 --theta-obs 6 --length 1e15 --time 1e5
// =============================================================================

use std::io::Write;
use std::path::Path;

use symbi_afterglow::observe::{DOPPLER_BAND, DOPPLER_BOLOMETRIC, compute_skymap};
use symbi_afterglow::{Microphysics, generate_events_from_cells};
use symbi_afterglow_io::{CgsScales, Synth, read_cells};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |key: &str, def: f64| -> f64 {
        args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(def)
    };
    let t_obs = get("--t-obs", 1.0);
    let theta_obs = get("--theta-obs", 0.0).to_radians();
    let window = get("--window", t_obs);
    let gamma = get("--gamma", 4.0 / 3.0);
    let p = get("--p", 2.5);
    let eps_e = get("--eps-e", 0.1);
    let eps_b = get("--eps-b", 0.01);
    let scales = CgsScales {
        length:   get("--length", 1.0),
        density:  get("--density", 1.0),
        pressure: get("--pressure", 1.0),
        time:     get("--time", 1.0),
    };
    let dt = get("--dt", 1.0e5);
    let n_pix = get("--n-pix", 256.0) as usize;
    let bolometric = args.iter().any(|a| a == "--bolometric");
    let doppler_power = if bolometric { DOPPLER_BOLOMETRIC } else { DOPPLER_BAND };

    let checkpoint = args
        .iter()
        .position(|a| a == "--checkpoint")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .expect("--checkpoint <path> is required");
    let out_path = args
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "checkpoint_skymap.pgm".to_string());

    let cells = read_cells(Path::new(&checkpoint), &scales, &Synth::default())
        .unwrap_or_else(|e| panic!("read {checkpoint}: {e}"));
    println!("read {} cells from {checkpoint}", cells.len());

    let micro = Microphysics { p, eps_e, eps_b, adiabatic_index: gamma, dt };
    let events = generate_events_from_cells(&cells, &micro, 1, 4, 60_000_000);
    println!("synthesized {} photon packets", events.len());

    let obs = [theta_obs.sin(), 0.0, theta_obs.cos()];
    let img = compute_skymap(&events, obs, t_obs, window, 0.0, 1.0e30, 0.0, doppler_power, n_pix);

    // write the PGM (same header convention as the BM example -> plot_bm_skymap.py renders it).
    let maxv = img.intensity.iter().cloned().fold(0.0_f64, f64::max).max(1e-300);
    let mut f = std::fs::File::create(&out_path).expect("create pgm");
    writeln!(f, "P2").unwrap();
    writeln!(
        f,
        "# half_width_cm={:.6e} t_obs_day={t_obs} d_l_cm=1e26 e_iso_erg=NA n0_cm3=NA p={p} \
         theta_obs_deg={} theta_sector_deg=NA doppler_power={doppler_power}",
        img.half_width, theta_obs.to_degrees()
    )
    .unwrap();
    writeln!(f, "{n_pix} {n_pix}\n255").unwrap();
    for iy in 0..n_pix {
        let mut line = String::new();
        for ix in 0..n_pix {
            line.push_str(&((255.0 * img.pixel(ix, iy) / maxv).round() as u32).to_string());
            line.push(' ');
        }
        writeln!(f, "{}", line.trim_end()).unwrap();
    }
    println!("wrote {out_path} ({n_pix}x{n_pix})  half-width = {:.3e} cm", img.half_width);
}
