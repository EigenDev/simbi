# Afterglow Module

Synchrotron radiation modeling suite for RMHD simulations. Generates photon events from hydrodynamic snapshots, computes lightcurves, sky maps, and polarization signatures for arbitrary observer geometries.

---

## Overview

This module provides two radiation calculation approaches:

1. **Event-Based System** (Modern, Recommended)
   - Monte Carlo photon event generation
   - Full 4D position/momentum tracking
   - Polarization from magnetic field geometry (SRMHD)
   - Arbitrary observer angles and times
   - Optional Monte Carlo Radiative Transfer (MCRT)

2. **Direct Flux Calculation** (Legacy, Fast)
   - On-axis flux calculation via `py_calc_fnu`
   - Time-binned flux densities
   - No polarization or off-axis viewing
   - Faster for simple lightcurves

**Use event-based for:** off-axis jets, polarimetry, sky maps  
**Use direct flux for:** quick on-axis lightcurves

---

## Architecture

```
afterglow/
├── src/                     # C++ core
│   ├── rad.cpp              # synchrotron physics calculations
│   ├── photon_event_io.cpp  # HDF5 serialization
│   └── units.hpp            # compile-time dimensional analysis
├── bindings/
│   └── binding.cpp          # pybind11 Python interface
├── generate.py              # workflow: hydro → events → HDF5
├── postprocess.py           # analysis: events → lightcurves/skymaps
├── radiation.py             # legacy direct flux calculation
└── README.md                # you are here
```

**Compiled extension:** `simbi.libs.rad_hydro` (built via meson)

---

## Quick Start

### Generate Photon Events

```python
from simbi.afterglow.generate import generate_from_files
import numpy as np

generate_from_files(
    files=["checkpoint_0050.h5", "checkpoint_0100.h5"],
    output="photon_events.h5",
    max_events=1000000,
    eps_e=0.1,           # electron energy fraction
    eps_b=0.01,          # magnetic energy fraction
    p=2.5,               # electron power-law index
    theta_obs=np.deg2rad(20),  # observer angle
    z=0.1,               # redshift
    apply_mcrt=True,     # enable radiative transfer
    hydro_type="SRMHD"   # or "SRHD"
)
```

### Compute Lightcurve

```python
from simbi.afterglow.postprocess import (
    read_photon_events,
    compute_lightcurve
)

# load events
events, meta = read_photon_events("photon_events.h5")

# compute multi-frequency lightcurve
lc = compute_lightcurve(
    events=events,
    meta=meta,
    observer_angle=np.deg2rad(20),
    frequencies=[1e9, 1e10, 1e15],  # Hz
    n_bins=50
)

# plot
import matplotlib.pyplot as plt
for nu in lc.frequencies:
    plt.plot(lc.times, lc.fluxes[nu], label=f"{nu:.1e} Hz")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Observer Time [day]')
plt.ylabel('Flux Density [mJy]')
plt.legend()
plt.show()
```

### Compute Sky Map

```python
from simbi.afterglow.postprocess import compute_skymap

skymap = compute_skymap(
    events=events,
    meta=meta,
    time=1.0,           # observer day
    energy_min=1e-12,   # erg
    energy_max=1e-10,
    n_theta=128,
    n_phi=256
)

# plot
import matplotlib.pyplot as plt
plt.pcolormesh(skymap.phi, skymap.theta, skymap.intensity)
plt.colorbar(label='Intensity')
plt.xlabel('Azimuth [rad]')
plt.ylabel('Polar Angle [rad]')
plt.show()
```

### Compute Polarization

```python
from simbi.afterglow.postprocess import compute_polarization

pol = compute_polarization(
    events=events,
    meta=meta,
    observer_angle=np.deg2rad(20),
    energy_min=1e-12,
    energy_max=1e-10
)

# plot polarization degree
plt.plot(pol.times, pol.polarization_degree * 100)
plt.xlabel('Observer Time [day]')
plt.ylabel('Polarization Degree [%]')
plt.xscale('log')
plt.show()
```

---

## Physics

### Synchrotron Radiation Model

**Electron Distribution:**
```
n(γ) ∝ γ^(-p)  for γ_min < γ < γ_max
```

**Key Frequencies:**
- `ν_g`: gyration frequency in magnetic field
- `ν_c`: critical frequency (cooling break)
- `ν_m`: minimum frequency (γ_min electrons)

**Power-Law Spectrum:**
- `ν < ν_m`: F_ν ∝ ν^(1/3)`
- `ν_m < ν < ν_c`: F_ν ∝ ν^(-(p-1)/2)`
- `ν > ν_c`: F_ν ∝ ν^(-p/2)`

### Polarization (SRMHD only)

- Linear polarization from synchrotron emission
- Degree: ~70% for power-law electrons in ordered field
- Angle: perpendicular to projected B-field
- Circular polarization (Stokes V): typically small

### Monte Carlo Radiative Transfer (MCRT)

Optional processes:
1. **Synchrotron Self-Absorption (SSA)**
   - Optical depth: τ_SSA ~ n_e σ_ν L
   - Important at low frequencies, high densities

2. **Thomson Scattering**
   - Optical depth: τ_T ~ n_e σ_T L
   - Depolarizes radiation, changes direction

3. **Pair Production** (optional, high energy)
   - γγ → e⁺e⁻ for E > m_e c²

---

## Physical Parameters

### Microphysics

- **`eps_e`**: Fraction of shock energy in electrons (typical: 0.01–0.5)
- **`eps_b`**: Fraction of shock energy in magnetic field (typical: 0.001–0.1)
- **`p`**: Electron power-law index (typical: 2.0–3.0)

### Observer Parameters

- **`theta_obs`**: Viewing angle from jet axis [radians]
- **`z`**: Cosmological redshift (affects flux and observed frequency)
- **`d_L`**: Luminosity distance [cm] (auto-computed from z if not provided)

### Numerical Parameters

- **`max_events`**: Total photons to generate (memory limit: ~1M events ≈ 200 MB)
- **`photons_per_cell`**: Sampling density (0 = auto, uniform distribution)

---

## Data Structures

### `photon_event_t` (C++)

```cpp
struct photon_event_t {
    // spacetime (lab frame)
    double t_emission;           // [s]
    double x, y, z;              // [cm]
    
    // 4-momentum
    double energy;               // [erg]
    double px, py, pz;           // direction (unit vector)
    
    // polarization (stokes parameters)
    double stokes_I, Q, U, V;
    
    // fluid properties
    double doppler_factor;
    double lorentz_factor;
    double optical_depth;
    
    // metadata
    uint32_t cell_id;
    bool absorbed;
    uint32_t n_scatter;
};
```

### HDF5 File Format

```
photon_events.h5
├── attributes (metadata)
│   ├── dt, theta_obs, adiabatic_index
│   ├── p, z, eps_e, eps_b, d_L
│   └── time_scale, rho_scale, ...
├── datasets (columnar)
│   ├── t_emission[n_events]
│   ├── x, y, z[n_events]
│   ├── energy, px, py, pz[n_events]
│   ├── stokes_I, Q, U, V[n_events]
│   └── ...
└── frequencies[n_freq] (optional)
```

**Compression:** gzip level 6 (default)  
**Chunking:** Automatic optimal size

---

## Units and Scales

### CGS Throughout

- Length: centimeters
- Mass: grams
- Time: seconds
- Energy: ergs
- Frequency: Hertz
- Flux: mJy (milliJansky = 10⁻²⁶ erg/cm²/s/Hz)

### Scale Models

Dimensionless hydro output is converted using physical scales:

```python
from simbi.afterglow.scales import get_scale_model

scales = get_scale_model("solar")  # or "kilonova", "collapsar", etc.
# scales.time_scale   → [s]
# scales.length_scale → [cm]
# scales.rho_scale    → [g/cm³]
```

---

## Performance

### Event Generation

- **Speed:** ~10⁴–10⁵ events/sec per core (OpenMP parallelized)
- **Memory:** ~200 bytes/event
- **Bottleneck:** Cell loop (scales as N_cells × photons_per_cell)

### MCRT

- **Speed:** ~10⁴ events/sec (slower if many scattering events)
- **Memory:** In-place modification (no extra allocation)

### Postprocessing

- **`compute_lightcurve`:** O(N_events × N_bins) — fast
- **`compute_skymap`:** O(N_events × N_theta × N_phi) — slow, consider subsampling
- **`compute_polarization`:** O(N_events × N_bins) — fast

---

## Troubleshooting

### Import Error: `cannot import name 'rad_hydro'`

**Problem:** Extension module not compiled  
**Solution:**
```bash
cd /path/to/simbi
meson setup build
ninja -C build
```

Check that `simbi/libs/rad_hydro.*.so` exists.

### Warning: `data_dim may be used uninitialized`

**Status:** Fixed in current version (default initialization added)  
**If persists:** Update to latest code

### No Events Generated

**Possible causes:**
1. `max_events` too small — increase it
2. Hydro data has no emitting regions — check density/pressure fields
3. `eps_e` or `eps_b` too small — use typical values (0.01–0.1)

### MCRT absorbs all photons

**Possible causes:**
1. Very optically thick medium — expected for high densities
2. Optical depth calculation bug — check `calc_ssa_optical_depth` logic
3. Thomson scattering too efficient — reduce medium density or disable scattering

---

## Advanced Usage

### Custom Mesh for Off-Axis

If computing off-axis afterglows from 1D/2D simulations:

```python
from simbi.afterglow.helpers import generate_pseudo_mesh

# creates 3D mesh from lower-dimensional data
generate_pseudo_mesh(
    args,
    mesh_dict,
    full_sphere=True,
    full_threed=True
)
```

### Filtered Event Reading

```python
# load only unabsorbed high-energy photons
events, meta = read_photon_events("photons.h5")

mask = (~events.absorbed) & (events.energy > 1e-10)
filtered = events.filter(mask)
```

### Parallel Processing

The C++ event generation uses OpenMP automatically. Control threads:

```bash
export OMP_NUM_THREADS=8
python your_script.py
```

---

## References

### Key Papers

- **Synchrotron Theory:** Rybicki & Lightman (1979), *Radiative Processes*
- **Afterglow Model:** Sari, Piran, & Narayan (1998), ApJ 497, L17
- **MCRT in Relativistic Flows:** Mimica et al. (2009), ApJ 696, 1142

### Related Codes

- **afterglowpy:** Semi-analytic afterglow lightcurves (Ryan+ 2020)
- **BoxFit:** Structured jet modeling (van Eerten+ 2012)
- **MCRTΡ:** Monte Carlo radiative transfer (Lundman+ 2018)

---

## API Reference

### `generate.py`

```python
generate_from_files(
    files: List[str],
    output: str,
    max_events: int = 1000000,
    photons_per_cell: int = 0,
    eps_e: float = 0.1,
    eps_b: float = 0.01,
    p: float = 2.5,
    theta_obs: float = 0.0,
    z: float = 0.0,
    d_L: Optional[float] = None,
    apply_mcrt: bool = False,
    include_scattering: bool = True,
    hydro_type: str = "SRHD",
    scale_model: str = "solar"
) -> None
```

### `postprocess.py`

```python
read_photon_events(filename: str) -> Tuple[photon_events_t, metadata_t]

compute_lightcurve(
    events: photon_events_t,
    meta: metadata_t,
    observer_angle: float,
    frequencies: List[float],
    time_bins: Optional[np.ndarray] = None,
    n_bins: int = 50,
    energy_cut: float = 0.0
) -> lightcurve_t

compute_skymap(
    events: photon_events_t,
    meta: metadata_t,
    time: float,
    energy_min: float = 0.0,
    energy_max: float = np.inf,
    n_theta: int = 128,
    n_phi: int = 256,
    time_window: float = 0.1
) -> skymap_t

compute_polarization(
    events: photon_events_t,
    meta: metadata_t,
    observer_angle: float,
    time_bins: Optional[np.ndarray] = None,
    n_bins: int = 50,
    energy_min: float = 0.0,
    energy_max: float = np.inf
) -> polarization_t

compute_spectrum(
    events: photon_events_t,
    meta: metadata_t,
    observer_angle: float,
    time: float,
    frequencies: np.ndarray,
    time_window: float = 0.1
) -> spectrum_t
```

---

## Contributing

### Code Style

- Follow `simbi/CLAUDE.md` guidelines
- All comments lowercase
- Class names: `snake_case_t` suffix
- Loop indices: `ii`, `jj`, `kk`
- No numbered comments, no "talking to user" in code

### Adding Features

1. Implement C++ core in `src/`
2. Add pybind11 bindings in `bindings/binding.cpp`
3. Expose Python interface in `generate.py` or `postprocess.py`
4. Update this README

---

## License

MIT License (see root `LICENSE` file)

---

## Contact

Part of **simbi**: https://github.com/EigenDev/simbi