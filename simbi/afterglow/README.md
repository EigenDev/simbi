# Afterglow

Turn a hydro snapshot into something an observer would actually see: light curves, spectra, sky
maps, and polarization. The model is Sari, Piran & Narayan (1998) synchrotron plus equal-arrival-time
(EATS) integration, so it's aimed at GRB-like relativistic blast waves — though nothing stops you
pointing it at any relativistic outflow you've simulated.

This used to be a C++/pybind11 extension. It's Rust now, and the physics core has no `simbi`
dependency at all — it operates on plain hydro arrays over a spherical mesh, so it's reusable
outside this codebase if you ever want that.

---

## Where things live

**`symbi-afterglow`** (the Rust core, `src/crates/symbi-afterglow/`) does the physics in CGS, with two
complementary paths:

| | modules | what it is |
|---|---|---|
| deterministic | `synchrotron` → `lightcurve` | per-cell broken-power-law spectrum fed into the equal-arrival-time flux integral. Noise-free. |
| monte carlo | `transfer` → `observe` | samples beamed photon packets, propagates them with self-absorption / Thomson scattering / optional pair production, then reduces the catalog for any line of sight. |

Supporting cast: `units` (compile-time M/L/T dimensional checking — a dimensionally wrong expression
won't compile), `bm` (Blandford-McKee), `coords`, `deposit` (the noise-free imager), `event`,
`ingest`, `rng`, `constants`. **`symbi-afterglow-io`** reads the checkpoints.

**Python** (this directory) is the frontend: `generate.py`, `lightcurve.py`, `spec.py`,
`postprocess.py`, `plotting.py`, plus `scales.py` (code units → CGS) and `spn98.py` (the analytic
break frequencies, handy for sanity-checking a numerical result against theory).

---

## Quick start

Generate a photon catalog from a stack of checkpoints, then reduce it however you like:

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

Generate once, reduce many times — every reduction reads the same catalog.

`generate` also takes `--mcrt` to switch on the Monte-Carlo transfer (self-absorption, scattering),
`--no-scattering` to keep absorption but drop scattering, and `--photons-per-cell` to control
sampling density. For imaging, `--method deposit` is the noise-free reducer and is auto-selected for
hydro checkpoints; `--method mc` uses the photon catalog directly.

---

## The two YAMLs

Two files drive everything, and both are auto-discovered next to your data:

- **`system.yaml`** — the code-unit → CGS conversion. Without it you fall back to `--scale`.
- **`observer.yaml`** — redshift, luminosity distance, microphysics, **and the frequencies**.

That last point matters: there is no `--frequencies` flag. Frequencies live in `observer.yaml`,
which is the only place to set them. With no observer file you get defaults (10 pc, p = 2.5).

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

`spn98.py` gives you the analytic SPN98 break frequencies, which is the fastest way to tell whether a
numerical light curve is behaving: compute the breaks for your parameters and check the measured
slopes match the table above on either side of them. If the deterministic and Monte-Carlo paths
disagree by more than shot noise, that's a real signal — the two are normalized against the same
per-Hz emissivity precisely so they can be compared.
