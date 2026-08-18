# Afterglow

The afterglow tools turn hydrodynamic snapshots into synthetic light curves, spectra, sky maps, and
polarization. They use the Sari, Piran & Narayan (1998) synchrotron model with equal-arrival-time
surface (EATS) integration. The model is intended for GRB-like relativistic blast waves, but it can
also be applied to other simulated relativistic outflows.

The original implementation was a C++/pybind11 extension. The current Rust physics core has no
dependency on `simbi`; it operates on hydrodynamic arrays over a spherical mesh and can be reused
independently.

---

## Where things live

**`symbi-afterglow`** (the Rust core, `src/crates/symbi-afterglow/`) does the physics in CGS, with two
complementary paths:

| | modules | what it is |
|---|---|---|
| deterministic | `synchrotron` → `lightcurve` | per-cell broken-power-law spectrum fed into the equal-arrival-time flux integral. Noise-free. |
| monte carlo | `transfer` → `observe` | samples beamed photon packets, propagates them with self-absorption / Thomson scattering / optional pair production, then reduces the catalog for any line of sight. |

Supporting modules include `units` (compile-time M/L/T dimensional checking), `bm`
(Blandford-McKee), `coords`, `deposit` (the noise-free imager), `event`,
`ingest`, `rng`, `constants`. **`symbi-afterglow-io`** reads the checkpoints.

**Python** (this directory) is the frontend: `generate.py`, `lightcurve.py`, `spec.py`,
`postprocess.py`, `plotting.py`, plus `scales.py` (code units → CGS) and `spn98.py` (the analytic
break frequencies, handy for sanity-checking a numerical result against theory).

---

## Quick start

Generate a photon catalog from a set of checkpoints, then use it for one or more reductions:

```bash
# build the event catalog (this is the expensive step; do it once)
simbi afterglow generate data/*.h5 --output events.h5 --max-events 1000000

# light curve — the angle is in DEGREES
simbi afterglow lightcurve events.h5 --observer-angle 6.0

# sky map and spectrum — --time is in DAYS
simbi afterglow skymap events.h5 --time 1.0
simbi afterglow spectrum events.h5 --time 1.0

# polarization, and a sky-map movie sweeping observer time
simbi afterglow polarization events.h5 --observer-angle 6.0
simbi afterglow movie events.h5 --output skymap.mp4
```

Each reduction reads the same catalog, so generation only needs to run once for a given set of events.

`generate` also takes `--mcrt` to switch on the Monte-Carlo transfer (self-absorption, scattering),
`--no-scattering` to keep absorption but drop scattering, and `--photons-per-cell` to control
sampling density. For imaging, `--method deposit` is the noise-free reducer and is auto-selected for
hydro checkpoints; `--method mc` uses the photon catalog directly.

---

## The two YAMLs

Two files drive everything, and both are auto-discovered next to your data:

- **`system.yaml`** — the code-unit → CGS conversion. Without it you fall back to `--scale`.
- **`observer.yaml`** — redshift, luminosity distance, microphysics, **and the frequencies**.

There is no `--frequencies` flag; frequencies are configured in `observer.yaml`. Without an
observer file, the defaults are 10 pc and p = 2.5.

---

## Physics

**Electron distribution** is a power law `n(γ) ∝ γ^(-p)` between `γ_min` and `γ_max`, giving the
usual three-segment spectrum around the break frequencies `ν_m` (minimum, from the `γ_min`
electrons) and `ν_c` (cooling):

| range | slope |
|---|---|
| `ν < ν_m` | `F_ν ∝ ν^(1/3)` |
| `ν_m < ν < ν_c` | `F_ν ∝ ν^(-(p-1)/2)` |
| `ν > ν_c` | `F_ν ∝ ν^(-p/2)` |

Each Monte-Carlo packet draws its **comoving** frequency from that broken power law and carries an
equal share of the cell's emitted energy — frequency and energy are separate fields on purpose. If
you stored one energy and reconstructed a frequency from it, the Monte-Carlo spectrum could never
reproduce the analytic one.

**Transfer** (under `--mcrt`) applies synchrotron self-absorption, which matters at low frequency and
high density; Thomson scattering, which redirects and depolarizes; and optionally pair production.

**Polarization** comes from synchrotron emission in an ordered field — up to ~70% for a power-law
electron population, with the angle perpendicular to the projected B field. Needs an MHD run.

### Parameters worth knowing

- `eps_e`, the shock energy fraction in electrons — typically 0.01–0.5
- `eps_b`, the fraction in magnetic field — typically 0.001–0.1
- `p`, the electron index — typically 2.0–3.0
- `max_events`, total packets. Roughly 200 MB per million, so it's the memory dial.

---

## Units

Everything inside the core is CGS — gauss, Hz, erg, cm, s — and `QuantScales` does the conversion
from your simulation's code units. The `units` module encodes dimensions in the type system, so a
dimensionally inconsistent expression is a compile error rather than a silently wrong number.

On the Python side, `scales.py` carries the scale models (`Solar`, `BlandfordMckee`, or your own via
`user_scale`) and exposes `time_scale` [s], `length_scale` [cm], and `rho_scale` [g/cm³].

---

## Sanity checks

`spn98.py` provides the analytic SPN98 break frequencies. Use it to compute the breaks for your
parameters and compare the measured slopes with the table above. The deterministic and Monte Carlo
paths use the same per-Hz emissivity normalization, so differences larger than the expected shot
noise should be investigated.
