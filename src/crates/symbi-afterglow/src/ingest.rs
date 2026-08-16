// =============================================================================
// ingest.rs
//
// the real-data ingestion path: turn hydro cells from an actual simulation — in any
// geometry (Cartesian / Spherical / Cylindrical) the hydro ran, with the full velocity
// vector (v1, v2, v3) — into synchrotron photon packets. this is what the afterglow
// module consumes to observe real spheres, jets, and rings, capturing lateral spreading
// (the velocity is not assumed radial) and letting the observer sit anywhere.
//
// a `Cell` is the geometry-neutral unit: a Cartesian position + Cartesian three-velocity +
// thermodynamics + proper volume + lab emission time. `Cell::from_coords` builds one from a
// hydro cell in its native coordinates via `coords` (which matches the hydro's own transforms).
// `generate_events_from_cells` emits packets, beaming about each cell's velocity vector — so a
// laterally-spreading sector images correctly and the observer-direction doppler is exact.
//
// the multi-checkpoint EATS is just calling this per checkpoint (with that checkpoint's
// `t_emission`) and concatenating; the spreading-over-time then falls out of the EATS.
//
// usage:
//  let cells: Vec<Cell> = grid.iter().map(|c| Cell::from_coords(
//      coords, c.x, c.v, c.rho, c.pre, c.volume, t_lab)).collect();
//  let events = generate_events_from_cells(&cells, &micro, seed, 4, max_events);
// =============================================================================

use crate::coords::Coords;
use crate::event::PhotonEvent;
use crate::rng::Rng;
use crate::transfer::{CellState, emit_packets};
use crate::units::{Energy, Time, Volume};

/// a geometry-neutral hydro cell, in the global Cartesian frame the afterglow works in.
#[derive(Clone, Copy, Debug)]
pub struct Cell {
    /// cell-center position [cm].
    pub position: [f64; 3],
    /// fluid three-velocity (units of c) as a Cartesian vector — direction captures spreading.
    pub beta_vec: [f64; 3],
    /// proper mass density [g/cm^3].
    pub rho: f64,
    /// pressure [erg/cm^3].
    pub pre: f64,
    /// proper cell volume [cm^3].
    pub volume: f64,
    /// lab-frame emission time [s] (the checkpoint's time).
    pub t_emission: f64,
}

impl Cell {
    /// build a Cartesian cell from a hydro cell in coordinate system `coords`: the coordinate
    /// position `x = (x1, x2, x3)` and the physical three-velocity `v = (v1, v2, v3)` (units of c)
    /// are converted to the global Cartesian frame via the canonical hydro transforms. positions
    /// must already be in cm, velocity in units of c (apply the sim's code-unit scales first).
    #[allow(clippy::too_many_arguments)]
    pub fn from_coords(
        coords: Coords,
        x: [f64; 3],
        v: [f64; 3],
        rho: f64,
        pre: f64,
        volume: f64,
        t_emission: f64,
    ) -> Cell {
        Cell {
            position: coords.position_to_cartesian(x),
            beta_vec: coords.velocity_to_cartesian(x, v),
            rho,
            pre,
            volume,
            t_emission,
        }
    }
}

/// the emission microphysics shared by every cell (the parts of `SimConditions` the cell path
/// needs). `dt` is the emission timestep [s] each cell radiates over (the checkpoint spacing).
#[derive(Clone, Copy, Debug)]
pub struct Microphysics {
    pub p: f64,
    pub eps_e: f64,
    pub eps_b: f64,
    pub adiabatic_index: f64,
    pub dt: f64,
}

/// generate synchrotron photon packets from a list of Cartesian hydro cells (any geometry, any
/// velocity direction). each cell radiates its synchrotron energy over `micro.dt` into
/// `photons_per_cell` equal-weight packets, emitted isotropically in the fluid frame and
/// aberrated about the cell's velocity vector. `seed` makes the catalog reproducible;
/// `max_events` caps the total. reduce the result with `compute_skymap` / the light-curve /
/// polarization functions for any observer direction.
pub fn generate_events_from_cells(
    cells: &[Cell],
    micro: &Microphysics,
    seed: u64,
    photons_per_cell: u64,
    max_events: u64,
) -> Vec<PhotonEvent> {
    let mut rng = Rng::seed(seed);
    let mut events = Vec::new();

    for (id, c) in cells.iter().enumerate() {
        if events.len() as u64 >= max_events {
            break;
        }
        let beta = (c.beta_vec[0] * c.beta_vec[0]
            + c.beta_vec[1] * c.beta_vec[1]
            + c.beta_vec[2] * c.beta_vec[2])
            .sqrt();
        let vhat = if beta > 1e-300 {
            [
                c.beta_vec[0] / beta,
                c.beta_vec[1] / beta,
                c.beta_vec[2] / beta,
            ]
        } else {
            [0.0, 0.0, 1.0] // at rest: no beaming, direction is irrelevant
        };
        let w = 1.0 / (1.0 - (beta * beta).min(1.0 - 1e-15)).sqrt();

        let cell = CellState::from_physical(
            c.rho,
            c.pre,
            beta,
            micro.adiabatic_index,
            micro.eps_e,
            micro.eps_b,
            micro.p,
            c.t_emission / w,
        );

        // emitted comoving energy in the sampled band over the represented lab interval:
        // (band-integrated SPN98 emissivity) x dV_lab x dt_lab — the same normalization the
        // radial generators and the deterministic deposit use. equal-weight packets.
        let total_energy: Energy =
            cell.band_power_density(micro.p) * Volume::new(c.volume) * Time::new(micro.dt);
        let packet_weight = (total_energy / photons_per_cell as f64).value();

        emit_packets(
            &mut events,
            &mut rng,
            &cell,
            micro.p,
            c.t_emission,
            c.position,
            vhat,
            packet_weight,
            photons_per_cell,
            max_events,
            id as u32,
        );
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::C_LIGHT;
    use crate::observe::compute_skymap;
    use std::f64::consts::PI;

    fn micro() -> Microphysics {
        Microphysics {
            p: 2.5,
            eps_e: 0.1,
            eps_b: 0.01,
            adiabatic_index: 4.0 / 3.0,
            dt: 1.0e5,
        }
    }

    fn norm(v: [f64; 3]) -> f64 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    // a spherical cell with purely radial velocity -> beta_vec along the position direction;
    // adding a polar component tilts it off-radial (the spreading signature).
    #[test]
    fn from_coords_radial_vs_spreading() {
        let x = [1.0e16, PI / 3.0, PI / 5.0];
        let pos = Coords::Spherical.position_to_cartesian(x);
        let rmag = norm(pos);
        let rhat = [pos[0] / rmag, pos[1] / rmag, pos[2] / rmag];

        let radial = Cell::from_coords(
            Coords::Spherical,
            x,
            [0.6, 0.0, 0.0],
            1e-24,
            1e-3,
            1e45,
            0.0,
        );
        // radial velocity is parallel to r-hat.
        let cos = (radial.beta_vec[0] * rhat[0]
            + radial.beta_vec[1] * rhat[1]
            + radial.beta_vec[2] * rhat[2])
            / norm(radial.beta_vec);
        assert!(
            (cos - 1.0).abs() < 1e-12,
            "radial velocity should be along r-hat"
        );

        let spreading = Cell::from_coords(
            Coords::Spherical,
            x,
            [0.6, 0.3, 0.0],
            1e-24,
            1e-3,
            1e45,
            0.0,
        );
        let cos2 = (spreading.beta_vec[0] * rhat[0]
            + spreading.beta_vec[1] * rhat[1]
            + spreading.beta_vec[2] * rhat[2])
            / norm(spreading.beta_vec);
        assert!(
            cos2 < 0.999,
            "a polar (spreading) component tilts beta off-radial: cos={cos2}"
        );
    }

    // generation from cells is reproducible and emits finite positive-weight packets carrying the
    // cell's velocity vector.
    #[test]
    fn cell_generation_is_reproducible() {
        let cells: Vec<Cell> = (0..20)
            .map(|i| {
                let theta = PI * (i as f64 + 0.5) / 20.0;
                Cell::from_coords(
                    Coords::Spherical,
                    [1.0e16, theta, 0.3],
                    [0.9, 0.1, 0.0],
                    1e-24,
                    1e-3,
                    1e45,
                    0.0,
                )
            })
            .collect();
        let a = generate_events_from_cells(&cells, &micro(), 1, 5, 1_000_000);
        let b = generate_events_from_cells(&cells, &micro(), 1, 5, 1_000_000);
        assert_eq!(a, b, "same seed -> identical catalog");
        assert!(!a.is_empty());
        assert!(
            a.iter()
                .all(|e| e.energy_weight.is_finite() && e.energy_weight > 0.0)
        );
    }

    // spreading is captured: a cell whose lateral velocity points toward the observer images
    // brighter (larger doppler) than the same cell whose lateral velocity points away — a purely
    // radial treatment (which ignores the lateral component) could not tell them apart.
    #[test]
    fn lateral_velocity_changes_the_image() {
        // cell on the +z axis; observer in the x-z plane, off-axis.
        let pos = [0.0, 0.0, 1.0e16];
        let obs = {
            let a = 30.0_f64.to_radians();
            [a.sin(), 0.0, a.cos()]
        };
        let t_emit = 1.0e6; // lab emission time [s] (>0: a real checkpoint past the explosion)
        // arrival = t_emit - r.n/c; depends only on position (same for both) -> same window.
        let t_arr_day = (t_emit - pos[2] * obs[2] / C_LIGHT.value()) / 86400.0;

        let mk = |vx: f64| {
            let cell = Cell {
                position: pos,
                beta_vec: [vx, 0.0, 0.7], // mostly radial (+z) with a lateral x component
                rho: 1e-24,
                pre: 1e-3,
                volume: 1e45,
                t_emission: t_emit,
            };
            let ev = generate_events_from_cells(&[cell], &micro(), 2, 200, 1_000_000);
            let img = compute_skymap(
                &ev, obs, t_arr_day, 4.0, 0.0, 1.0e30, 0.0, 3.0, 16, 0.0, 0.0, 0.1,
            );
            img.intensity.iter().sum::<f64>()
        };
        // lateral velocity toward the observer (+x) beams more flux at it than away (-x) — a
        // purely radial treatment (ignoring the lateral component) could not tell them apart.
        assert!(
            mk(0.3) > mk(-0.3),
            "lateral velocity toward the observer must brighten the image"
        );
    }
}
