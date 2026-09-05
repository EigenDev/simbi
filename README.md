# SIMBI

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/simbi-dither-dark.png">
  <img alt="SIMBI" src="docs/assets/simbi-dither-light.png" width="640">
</picture>

</div>

<div align="center">

**A high-performance 3D relativistic magneto-gas dynamics code for astrophysical fluid simulations**

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-backend-orange.svg?style=for-the-badge&logo=rust)](https://www.rust-lang.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-supported-76B900.svg?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![ROCm](https://img.shields.io/badge/ROCm-supported-ED1C24.svg?style=for-the-badge&logo=amd)](https://rocm.docs.amd.com/)

**[Quick Start](#quick-start) · [Installation](#installation) · [Usage](#usage) · [Publications](#publications)**

</div>

---
<img width="1200" height="760" alt="sedov" src="https://github.com/user-attachments/assets/13c35423-117d-4521-b479-2b047574d96c" />


## Overview

SIMBI is a finite-volume code for astrophysical fluid simulations. If you want to put relativistic jets, shock tubes, stellar explosions, accretion flows, or magnetized turbulence on a grid and see what happens, this is the tool. Results from SIMBI have appeared in *The Astrophysical Journal* and *The Astrophysical Journal Letters*.

SIMBI started life as a C++ code. I eventually rewrote the compute backend in Rust after getting interested in graph theory and the possibility of generating architecture-agnostic code from one description of the physics. [Sethi and Ullman (1970)](https://dl.acm.org/doi/10.1145/321607.321620), and the work that followed it, provided some of the inspiration. The physics stayed the same, but years of experience gave me a chance to make the code faster and much easier to reason about. You still drive the whole thing from Python; the Rust API is there when you want it.
<!--that the russt workspace is called `symbi` instead of `simbi`. Well, that's because I really love the idea of turning symbolic expressions into machine code, but since both "simbi" and "symbi" sound the same, I thought that shift was a cool nod to the new direction of the codebase. The Python package is still called `simbi`, so you can keep your scripts and configs the same.-->

**What you get:**
- Six fluid regimes in one code: Newtonian hydro, relativistic hydro (RHD), Newtonian and relativistic MHD, plus isothermal variants of both
- Spacetime as its own axis: hand the relativistic regimes a Minkowski or horizon-penetrating Kerr-Schild metric (nonspinning or Kerr) the same way you would pick a coordinate system
- GPU acceleration on NVIDIA CUDA and AMD ROCm/HIP devices, with kernels compiled on the fly for the active accelerator
- High-resolution shock capturing with HLLE, HLLC, HLLC+ (a low-Mach, shock-stable variant; see [Chen et al. 2020](https://doi.org/10.1137/18M119032X)), and HLLD Riemann solvers. First-order flux correction handles failed high-order updates and reports each affected cell. I am still working on making this safety mechanism stronger, possibly with a method in the spirit of [Zalesak et al.](https://apps.dtic.mil/sti/tr/pdf/ADA360122.pdf).
- Constrained-transport MHD (contact [Gardiner & Stone](https://arxiv.org/abs/0712.2634) or UCT ([Mignone & DelZanna (2021)](https://arxiv.org/abs/2004.10542)) edge EMFs) that keeps div B at machine zero by construction
- Physical transport when you want it: Navier-Stokes viscosity (constant or alpha-disk) and Ohmic resistivity, layered on top of the ideal solvers
- Immersed bodies with point-mass gravity, masked accretors for problems such as Bondi-Hoyle flow, and rigid walls built from constructive solid geometry (CSG). The surface coupling uses volume penalization in the spirit of [Angot, Bruneau & Fabrie (1999)](https://doi.org/10.1007/s002110050401). Bodies support prescribed motion or two-way coupling, including translation, rotation, gas–body energy exchange, and force, torque, and accretion diagnostics. This part of SIMBI grew out of a class I took with [Chuck Peskin](https://en.wikipedia.org/wiki/Charles_S._Peskin) as a graduate student at NYU; I loved the subject and wanted to bring some of those ideas into the code.
- Horizon excision for GR accretion: on a horizon-penetrating Kerr-Schild chart, cells inside the horizon are held at a cold vacuum floor while the exterior flow remains regular
- Block-based static mesh refinement with [Berger-Colella](https://www.sciencedirect.com/science/article/pii/0021999189900351) subcycling
- Single-node **multi-GPU domain decomposition** — set `gpus > 1` and the domain splits across the cards, halo-exchanged in lockstep and bit-identical to a monolithic run
- In-situ binned reductions (a "census") for shell profiles, scaling laws, and other summary quantities
- Flux-based [Monte Carlo tracers from Genel et al. (2013)](https://doi.org/10.1093/mnras/stt1383) and continuous [Itô tracers from Moseley, Teyssier & Abel (2026)](https://arxiv.org/abs/2604.23041), including transport across refinement levels and multi-GPU boundaries
- Afterglow radiation transport using the synchrotron model of [Sari, Piran & Narayan (1998)](https://doi.org/10.1086/311269), so you can turn a simulation into synthetic observables
- A live terminal dashboard with pause, single-step, on-demand checkpoints, and field heatmaps; `simbi attach` provides a read-only view of a headless run from another shell
- A type-safe Python configuration system that generates its own CLI

> CUDA and HIP use kernels generated from the same definitions. SIMBI currently
> supports multi-GPU decomposition within one node; multi-node decomposition is
> not yet implemented. Most of my current science problems fit on one node, so it
> remains further down the roadmap.
---

## Simulation Gallery

<div align="center">

| Relativistic Jet Evolution | Relativistic Shock Tube | Rayleigh-Taylor Instability |
|:---:|:---:|:---:|
| [Animation](https://user-images.githubusercontent.com/29236993/145315802-c8d7e8c5-7beb-488c-b496-b9edf404be2e.mp4) | [Animation](https://user-images.githubusercontent.com/29236993/212521070-0e2a7ced-cd5f-4006-9039-be67f174fb07.mp4) | [Animation](https://github.com/EigenDev/simbi/assets/29236993/818d930d-d993-4e5d-8ed4-47a9bae11a7f) |

| Moving Mesh Techniques | SRMHD Turbulence |
|:---:|:---:|
| [Animation](https://user-images.githubusercontent.com/29236993/205418982-943af187-8ae3-4401-92d5-e09a4ea821e2.mp4) | [Animation](https://github.com/user-attachments/assets/c3c636f9-60ca-4331-9600-3442970a6325)

</div>

---

## Quick Start

If you only read one section, read this one. I strongly recommend [uv](https://docs.astral.sh/uv/) for creating the Python environment and installing SIMBI. It handles the jobs I once split between pip, venv, and conda, and keeps the setup short. Other environment managers are fine too.

Need uv? Grab it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Now run the Marti and Muller relativistic shock tube test on CPU:

```bash
# create the environment and build the rust backend, all in one shot
uv venv
uv pip install .

# run the test problem
uv run simbi run marti-muller --mode cpu --resolution 400

# look at the result
uv run simbi plot data/400.chkpt.000_400.h5 --setup "Marti & Muller Problem 1" --fields rho v p
```
To work inside the environment without prefixing every command with `uv run`, activate it with `source .venv/bin/activate`.

Got an NVIDIA or AMD accelerator? Select its production backend through the
project helper:

```bash
uv sync --no-install-project   # puts maturin in the venv; dev.py needs it
./dev.py install --cuda        # or --hip for AMD; pick the one matching your card
uv run simbi run marti-muller --mode gpu --resolution 1024
```

Cargo and maturin handle the Rust and Python portions of the build. I still have a soft spot for CMake and Ninja—they taught me a lot while SIMBI was growing—but Cargo is a better fit for the range of architectures the project now covers.

---

## Installation

### What you need

- A Rust toolchain (cargo), the easy way is [rustup](https://rustup.rs)
- Python 3.10 or newer
- HDF5 (it gets linked into the extension)
- For NVIDIA GPU execution, a driver providing `libcuda` and `libnvrtc`
- For AMD GPU execution, ROCm providing `libamdhip64` and `libhiprtc`
- On Linux, `patchelf` (uv and maturin will tell you if it is missing)

Kernels compile at run time with NVRTC on NVIDIA devices and hipRTC on AMD
devices, so a separate CUDA or HIP compiler toolchain is not required.

### The uv way (recommended)

uv builds the Rust extension through maturin and installs it into an isolated environment.

```bash
# create an isolated environment
uv venv

# build and install the package
uv pip install .

# include the optional plotting and CLI dependencies
uv pip install ".[visual,cli]"
```

From here on, prefix commands with `uv run` and you are always using the right environment:

```bash
uv run simbi run sedov --mode cpu
```

To install the `simbi` command as a standalone uv-managed tool:

```bash
uv tool install .
```

The `simbi` command will then be available without activating the project environment.

### Working on SIMBI itself?

Keep dependency installation and native compilation separate during development.
`uv sync` normally installs the current project too; because SIMBI's build backend is
maturin, that implicitly compiles the Rust extension. Use `--no-install-project` so uv
installs maturin and the other development dependencies without building SIMBI:

```bash
uv sync --no-install-project
```

Build the desired backend explicitly through the project helper:

```bash
python dev.py install            # editable cpu build
python dev.py install --cuda     # editable nvidia cuda build
python dev.py install --hip      # editable amd rocm/hip build
```

After activating `.venv`, invoke Python tools directly:

```bash
python -m pytest
simbi run sedov --mode cpu
```

Outside an activated environment, suppress uv's automatic synchronization so it
leaves the built project alone:

```bash
uv run --no-sync pytest
uv run --no-sync simbi run sedov --mode cpu
```

Run `uv sync --no-install-project` again after dependency changes. Run
`python dev.py install` again after Rust, AOT, PyO3, or backend-feature changes.
Pure Python changes take effect immediately.

### GPU builds

GPU builds go through `dev.py`, a thin wrapper around maturin. Select CUDA for
NVIDIA or HIP for AMD:

```bash
./dev.py install --cuda
./dev.py install --hip
```

The CPU and GPU extensions coexist as separate modules (`cpu_ext` and `gpu_ext`), so you can keep
both installed and pick the backend at run time with `--mode cpu` or `--mode gpu`. The CPU path
threads with rayon, so size it with `RAYON_NUM_THREADS` if you want to pin it.

If you build a GPU backend into a fresh clone, pass `--with-cpu` to keep a CPU extension around too;
otherwise `--mode cpu` finds nothing and drops into demo mode. `--cuda` and `--hip` are alternatives:
whichever you build last owns `gpu_ext`.

### Cleaning up

```bash
./dev.py clean           # remove cargo artifacts; preserve installed extensions
./dev.py clean --all     # also remove repository python caches
./dev.py install --cuda  # rebuild the nvidia backend
./dev.py install --hip   # rebuild the amd backend
```

The AOT bake writes its generated kernel source, registry, and neutral
`*.ir.json` inspection artifacts to Cargo's `OUT_DIR`, normally beneath
`target/<profile>/build/symbi-aot-<hash>/out/`. Different profiles, feature
sets, and build hashes may retain more than one copy. These are disposable build
artifacts, not source data; `./dev.py clean` (or `cargo clean`) removes them.

---

## Usage

### The four commands

```bash
simbi run        # run simulations
simbi plot       # visualize checkpoint data
simbi afterglow  # radiation transport and observables
simbi attach     # watch a headless (cluster/batch) run from your own terminal
```

(Remember, with uv you write `uv run simbi ...`.)

### Running simulations

```bash
# the basic shape of it
simbi run marti-muller --mode gpu --resolution 400

# what knobs does this problem expose?
simbi run <problem> --info

# what problems ship with simbi?
simbi run --configs

# point it at your own config file
simbi run simbi_configs/examples/newtonian/kh.py --mode cpu --resolution 512

# pick up where a previous run left off
simbi run <problem> --checkpoint data/128x128.chkpt.000_400.h5
```

The CLI tries to be a good roommate: configuration names work with kebab-case or underscores,
typos get a “did you mean…?”, and if two directories contain the same name, it lists both and asks.

**Common options:**
- `--mode cpu|gpu` sets the execution backend
- `--resolution N`, `--resolution N,M`, or `--resolution N,M,K` sets the grid. Comma is required for 2D and 3D.
- `--end-time` is when to stop
- `--data-directory` is where the output goes
- `--validate` builds and checks the whole config without allocating a grid or writing anything — run this before you queue a job
- `--checkpoint-interval` and `--diagnostic-interval` set the output cadences (in natural units, independent of each other)
- `--live` writes a read-only snapshot each cadence so `simbi attach <data_dir>` can watch from elsewhere

**When you want to know where the time goes:**

```bash
SYMBI_PROFILE=1 simbi run <problem> ...
```

prints a per-phase wall-time breakdown at the end of the run — flux, godunov, c2p,
checkpoint I/O, even the JIT compile — in ns per zone-cycle. The exit summary reports
wall time and I/O time separately, and quotes throughput over pure integration time.

**Programmatic runs return their own diagnostics.** If you already have a
problem instance, `run` returns a frozen `RunResult` after a real execution:

```python
from simbi import run

result = run(problem, compute_mode="cpu")
if result is not None:  # None means no backend was installed, so SIMBI ran in demo mode
    print(result.data_directory)
    print(result.diagnostics.projection.projected_cells)
    print(result.diagnostics.projection.injected_nrg)
    print(result.diagnostics.guards.troubled_cells)
    print(result.diagnostics.guards.frozen_cells)
```

These are accepted, run-owned values: retries and rejected steps do not enter
the returned evidence, and concurrent runs do not share it. `troubled_cells`
counts cells flagged by recovery; `frozen_cells` counts cells the correcting
select actually held. A troubled-cell count is not a claim that a cell-local
fallback was applied—the flux splice itself acts on faces. Existing callers may
continue to ignore the return value.

### Visualization

```bash
# plot a few fields from a checkpoint
simbi plot data/128x128.chkpt.000_400.h5 --setup "Problem Name" --fields rho v p

# draw the immersed bodies on top (silhouette at the evolved pose; cartesian only)
simbi plot data/128x128.chkpt.000_400.h5 --fields rho --draw-bodies

# stitch a stack of checkpoints into an animation
simbi plot data/*.h5 --animate --fields rho

# get a starter config to customize
simbi plot data/*.h5 --generate-config
```

### Afterglow analysis

Turn hydro snapshots into synthetic observables:

```bash
# build a photon event catalog from the snapshots
simbi afterglow generate data/*.h5 --output events.h5 --max-events 1000000

# observer lightcurve (angle in DEGREES; the frequencies live in the observer yaml,
# which is auto-discovered next to the data or passed with --observer)
simbi afterglow lightcurve events.h5 --observer-angle 6.0

# sky intensity map (--time is in DAYS)
simbi afterglow skymap events.h5 --time 1.0

# polarization evolution
simbi afterglow polarization events.h5 --observer-angle 6.0

# spectrum
simbi afterglow spectrum events.h5 --time 1.0

# sweep the observer time into a sky-map movie
simbi afterglow movie events.h5 --output skymap.mp4
```

### Checkpoints, restarts, and the live view

Checkpoints are named `<res>.chkpt.<time>.h5` — resolution as the per-axis interior counts joined
with `x`, then the sim time with the decimal rendered as `_` and zero-padded so a directory listing
sorts chronologically. So `128x128.chkpt.000_400.h5`, `64x64x64.chkpt.009_000.h5`. Instead of a
time you may see `final` (clean finish), `interrupted` (Ctrl-C, a scheduler eviction, or `q` in the
TUI), or `crashed` (the solver stopped after an error). All three can be used to restart a run, so
a job that reaches its wall-clock limit still leaves a usable checkpoint.

> If you use globs, note that `*.chkpt.final*.h5` only exists after a clean finish. A job stopped at
> its wall-clock limit writes an `interrupted` checkpoint instead.

Resuming with `--checkpoint` picks up the sim clock and the checkpoint numbering, so you get
`031, 032, ...` alongside the earlier files. `end_time` becomes the larger of yours and
the checkpoint's, so a restart can only extend a run. Fields marked `checkpoint_safe=True`
you can override on the command line; pass a conflicting flag that lacks that mark and the run
stops with a `ConfigError`, leaving the choice to you.

One thing to know: the `body_diagnostics` and census series inside a checkpoint cover the current
run segment only and start empty again after a restart. Stitch the segments offline.

Each checkpoint carries `metadata` (time, dt, iteration, gamma, cfl, regime, spacetime, solver,
spacing, ...) plus the compiled backend's `build_git_sha`, `build_git_dirty`, and
`build_source_id`, and the defining Python file's `config_source` and `config_sha256`. A dirty
backend is labeled `<commit>-dirty` rather than claiming that the clean commit reproduces it.
The same short identities appear in the startup table. Each `level_<N>` group holds the primitives
and conserved state, and — when the run has them — the file also carries `bodies`,
`body_diagnostics`, `tracers`, and `census/<name>`.
Immersed-body runs also append a plain-text `diagnostics.dat` (time, position, velocity, force,
torque, mass, accreted mass, accretion rate per body) if you set `--diagnostic-interval`; it's off
by default.

**Live TUI keys:** `space` pause, `s` single-step, `w` checkpoint now, `q`/`Esc` quit (writes an
`interrupted` checkpoint), `Tab`/arrows to move between panels, `f` cycle field, `c` cycle colormap,
`l` toggle log color, `o` cycle the 3D slice plane, `+`/`-` zoom about the center.

`simbi attach <data_dir>` gives you the same view of a headless run over a shared filesystem — it
polls the snapshot `--live` writes, over the shared filesystem alone. The view is read-only by
design, so `f`/`c`/`l` and the panel keys work (the snapshot ships every field) while
pause/step/checkpoint stay with the owning process. It needs the run to still be going: the
snapshot is removed when the run ends.

### In-situ profiles (census)

Sometimes what you want from a run is a scaling law or radial profile, and writing a thousand full
snapshots to get one is wasteful. A census is a binned reduction: you declare the bin axes and what
to accumulate as expressions, and the run reduces them on the device at each cadence and stores the
result in the checkpoint.

```python
from simbi import expression as expr

@computed_field
@property
def census_expressions(self) -> list[ExpressionDict]:
    x1, x2, x3 = expr.coords(3)
    r = expr.sqrt(x1 * x1 + x2 * x2 + x3 * x3)
    vx, vy, vz = (expr.velocity(ii, x1) for ii in (0, 1, 2))
    v_r = (x1 * vx + x2 * vy + x3 * vz) / r
    rho, dv = expr.density(x1), expr.cell_volume(x1)
    m = rho * dv
    return [
        expr.Census(
            name="shells",
            axes=[expr.BinAxis("r", r, expr.log_edges(1e-3, 1.0, 64))],
            values={"volume": dv, "mass": m, "mass_vr": m * v_r},
            op=expr.ReductionOp.ADD,
            sample_interval=0.05,
            cadence=expr.Cadence.PER_LEVEL_STEP,
        ).serialize()
    ]
```

Census values are stored as sums because sums combine cleanly across refinement levels,
decomposed tiles, and restart segments. The reader computes means and variances from ratios of
those sums:

```python
from simbi.reader import census_names, read_census

census_names("run.chkpt.final.h5")     # ('shells',)
c = read_census("run.chkpt.final.h5", "shells")
c.bin_centers(0)                       # the radial axis
c.favre("mass_vr", "mass")             # mass-weighted <v_r> per shell
c.assert_fully_binned()                # raises if cells fell outside the bins
```

Every row carries its level, sample count, time span, and a `dropped` count for cells outside the
bins. Check this count before interpreting a profile because incomplete bin coverage can resemble
physical structure. `PER_LEVEL_STEP` samples each refinement level on its own clock. This gives the
inner shells more samples than root-step sampling, which is useful when fitting their slopes.

The history covers one run segment and starts empty after a restart. For a restart chain, read the
last checkpoint from each segment and stitch the histories offline. Accumulating rows combine as a
count-weighted sum through `n_samples`.

### Tracer particles

Set `n_tracers` to seed a mass-weighted tracer population. `tracer_scheme="discrete"` uses the
flux-based Monte Carlo method of [Genel et al. (2013)](https://doi.org/10.1093/mnras/stt1383): each
accepted finite-volume mass flux defines the probability of a tracer moving between cells. `ito2`
and `ito3` use the continuous-trajectory [Itô method of
Moseley, Teyssier & Abel (2026)](https://arxiv.org/abs/2604.23041), matching the drift, diffusion,
and, for `ito3`, dispersion of the numerical mass transport. The Itô schemes keep continuous
particle positions while accounting for the diffusion introduced by the Eulerian solver.

Tracer state is written into each checkpoint. The `traced-kh` example uses `ito3` and shows the
configuration fields in context.

### Offline analysis

Immersed-body runs write a `body_diagnostics` time series (Mdot, drag, torque) into every
checkpoint. `simbi.analysis` reads it back and, for an accretor, finds when the flow settled:

```python
from simbi.analysis import load_body_diagnostics, steady_state_time

diag = load_body_diagnostics("run.chkpt.final.h5")
t0 = steady_state_time(diag.time, diag.mdot[:, 0])   # steady-state onset for body 0
```

---

## Configuration System

Problems are Python classes derived from `SimbiProblem`. Parameters declared with `ProblemParam`
are type-checked and can be exposed through the generated CLI.

### Basic structure

```python
from pathlib import Path
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import (
    BoundaryCondition,
    CoordSystem,
    Regime,
    Solver,
)

class KelvinHelmholtz(SimbiProblem):
    """kelvin-helmholtz instability in a newtonian fluid."""

    # physics parameters
    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, description="adiabatic index")
    ]
    rhoL: Annotated[float, ProblemParam(2.0, description="density in central layer")]
    rhoR: Annotated[float, ProblemParam(1.0, description="density in outer regions")]

    # domain configuration
    resolution: Annotated[
        tuple[int, int],
        ProblemParam((256, 256), cli=True, description="grid resolution"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(-0.5, 0.5), (-0.5, 0.5)], description="domain boundaries"),
    ]
    coord_system: Annotated[
        CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="coordinate system")
    ]
    regime: Annotated[Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime")]

    # numerics
    boundary_conditions: Annotated[
        BoundaryCondition, ProblemParam(BoundaryCondition.PERIODIC)
    ]
    solver: Annotated[Solver, ProblemParam(Solver.HLLC, description="riemann solver")]

    # simulation control
    data_directory: Annotated[
        Path,
        ProblemParam(Path("data/kh"), cli=True, checkpoint_safe=True),
    ]
    end_time: Annotated[
        float, ProblemParam(20.0, cli=True, checkpoint_safe=True)
    ]

    def initial_primitive_state(self):
        """generate initial conditions."""
        def gas_state():
            nx, ny = self.resolution
            for jj in range(ny):
                for ii in range(nx):
                    # the y coordinate of this cell
                    y = self.bounds[1][0] + jj * (self.bounds[1][1] - self.bounds[1][0]) / ny
                    if abs(y) < 0.25:
                        yield (self.rhoL, 0.5, 0.0, 2.5)  # rho, vx, vy, p
                    else:
                        yield (self.rhoR, -0.5, 0.0, 2.5)
        return gas_state
```

### ProblemParam options

| Option | What it does |
|--------|--------------|
| `cli=True` | Expose the field as a CLI argument |
| `checkpoint_safe=True` | Allow overriding it when resuming from a checkpoint |
| `description="..."` | Help text for the CLI |
| `ge=`, `le=`, `gt=`, `lt=` | Validation bounds |
| `cli_name="..."` | Override the derived kebab-case flag name |
| `group="..."` | Section label in the live dashboard's setup panel |

`cli=True` is set per field, including common fields such as `resolution` and `adiabatic_index`.
Use `simbi run <problem> --info` to list the flags for a particular problem.

### Deriving fields from other fields

If a parameter depends on another one, compute it in `setup()`. Declare the derived field
`Optional[...] = None` and fill it in there — and call `super().setup()`:

```python
bondi_radius: Annotated[Optional[float], ProblemParam(None)] = None

def setup(self) -> None:
    super().setup()
    self.bondi_radius = self.central_mass / self.sound_speed**2
```

The optional `summary()` method returns `(group, label, value)` rows for the setup panel in the live
dashboard. It is a convenient place to report derived quantities.

### Source terms

Add gravity or custom hydro sources as expressions. Constant downward gravity is one line:

```python
from simbi import expression as expr

@computed_field
@property
def source_expressions(self) -> list[ExpressionDict]:
    return [expr.force([0.0, -0.1], dim=2)]
```

For position-dependent sources, `coords` returns the spatial variables. These expressions support
ordinary arithmetic with one another and with numeric constants:

```python
@computed_field
@property
def source_expressions(self) -> list[ExpressionDict]:
    x, y = expr.coords(2)
    r_sq = x * x + y * y
    kappa = expr.where(r_sq > 0.64, 2.0, 0.0)          # damp the outer annulus
    # [kappa, rho_ref, vel_ref_x, vel_ref_y, pre_ref], as primitives
    return [expr.sponge([kappa, 1.0, 0.0, 0.0, 1.0], dim=2)]
```

The available constructors are `force`, `rotating_frame`, `cooling`,
`velocity_relaxation`, `sponge`, `inject`, and `raw`, along with `boundary` for
a driven Dirichlet face and `equilibrium` for a stationary target.
`velocity_relaxation` changes momentum and its associated kinetic work without
relaxing density or internal energy; `sponge` relaxes a full primitive reference
state. The old `relax` constructor remains a deprecated input alias. Each
constructor validates its arguments in Python. All of them accept a `region=`
mask, and `raw` also accepts `target=`.

Reference states are given as primitive variables and converted by the conservation law for the
selected regime. The same sponge interface therefore works for Newtonian, relativistic, and curved
spacetime problems. See `newtonian/rt.py`, `newtonian/rotating_sponge.py`, and
`grhd/gr_bondi_cartesian.py` for examples.

Problem configurations that expose an outer sponge use the canonical names
`use_sponge`, `sponge_time_fraction`, `sponge_parameters`, and `sponge_terms`.
The former `buffer` spellings remain accepted temporarily as deprecated input;
serialized compatibility keys are unchanged.

### Immersed bodies

Each immersed body has one `capability`:

- `GRAVITATIONAL` — a fixed-potential (softened) point mass
- `ACCRETION` — an immersed-boundary accretor. By default, a mollified spherical mask applies the
  volume drain described above. `porosity` mixes that drain with no-penetration and no-slip wall
  channels. The optional `torque_free_xi` parameter changes the tangential momentum channel using
  the idea from [Dittmann & Ryan (2021)](https://arxiv.org/abs/2102.05684); at `xi = 1`, the masked
  drain removes mass without transferring its angular momentum to the body
- `RIGID` — a solid wall, spherical by default or described by a CSG `Shape` in the body frame.
  Available operations include boxes, spheres, unions, intersections, and rotations. `k_eta_n`
  controls the no-penetration penalty, while `k_eta_t` controls tangential drag when
  `apply_no_slip` is enabled

The simplest case, a gravitating mass:

```python
from simbi.types import ImmersedBodyConfig, BodyCapability, GravitationalProperties

@computed_field
@property
def immersed_bodies(self) -> list[ImmersedBodyConfig]:
    return [
        ImmersedBodyConfig(
            capability=BodyCapability.GRAVITATIONAL,
            mass=1.0,
            radius=0.05,
            position=(0.0, 0.0),
            velocity=(0.0, 0.0),
            gravitational=GravitationalProperties(softening_length=0.01),
        )
    ]
```

With `two_way_coupling=True`, the gas force and torque evolve the body's translation and rotation.
Rotation follows Euler's equations and supports an anisotropic inertia tensor, so off-axis spins
can precess and asymmetric bodies can tumble. Here is a card in a wind tunnel:

```python
from simbi.types import ImmersedBodyConfig, BodyCapability, RigidProperties, Shape

card = Shape.box((0.0, 0.0, 0.0), (0.45, 0.22, 0.15))   # half-extents, body frame
ImmersedBodyConfig(
    capability=BodyCapability.RIGID,
    mass=1.0,
    radius=1.0,                 # the mask-gate scale; the CSG defines the geometry
    position=(0.0, 0.0, 0.0),
    velocity=(0.0, 0.0, 0.0),
    two_way_coupling=True,      # evolve the body from the fluid force and torque
    rigid=RigidProperties(
        inertia=1.0,
        apply_no_slip=True,     # tangential drag on (free-slip if False)
        k_eta_n=50.0,           # no-penetration stiffness
        k_eta_t=50.0,           # no-slip stiffness
        shape=card,
        omega=2.0,              # spin rate
        spin_axis=(0.3, 1.0, 0.2),          # arbitrary axis
        inertia_principal=(1.0, 3.0, 3.8),  # unequal moments -> precession / nutation
    ),
)
```

Gas–body energy exchange is conservative. Drag heats the gas, while an isothermal wall transfers
that heat to its reservoir. Every body reports its force, torque, and accreted mass each step.
Bodies can also orbit as gravitational binaries, and MHD runs can give a body Ohmic `magnetic`
coupling.

### Dynamic mesh motion

For domains that expand or contract, expose the scale factor `a(t)` and its derivative as plain
`@property` hooks returning a callable of time (they return a closure, so they stay plain
properties):

```python
@property
def scale_factor(self) -> Optional[Callable[[float], float]]:
    return lambda time: 1.0 + 0.1 * time

@property
def scale_factor_derivative(self) -> Optional[Callable[[float], float]]:
    return lambda time: 0.1
```

---

## Physics

### Regimes

| Regime | Description | Use cases |
|--------|-------------|-----------|
| `NEWTONIAN` | Classical hydrodynamics | Stellar winds, ISM dynamics, classical turbulence |
| `ISOTHERMAL` | Classical hydro with a fixed (optionally position-dependent) sound speed | Disks, locally-isothermal setups |
| `RHD` | Relativistic hydrodynamics | Gamma-ray bursts, relativistic shocks, stellar explosions |
| `NMHD` | Newtonian magnetohydrodynamics | Classical MHD turbulence, blast waves |
| `IMHD` | Isothermal MHD | Magnetized disks |
| `RMHD` | Relativistic magnetohydrodynamics | AGN jets, pulsar wind nebulae, magnetic reconnection |

The relativistic regimes also take an equation-of-state choice. `eos="synge"` uses the
[Taub–Mathews closure of Mignone, Plewa & Bodo (2005)](https://doi.org/10.1086/430905) to the
Synge relativistic perfect gas in place of the constant-gamma law. Its
effective adiabatic index varies from 5/3 in cold gas to 4/3 in hot gas, which is useful for flows
that cross the transrelativistic temperature range, such as a blast wave decelerating into the
Newtonian regime. It has no free adiabatic index.

The `Spacetime` axis sets a run's relativity: `RHD`/`RMHD` on Minkowski are special-relativistic, on a curved spacetime general-relativistic. Checkpoints and configs written under the legacy `srhd`/`srmhd` slugs still load (mapped to `rhd`/`rmhd`).

### Spacetimes

Relativity in SIMBI lives in the *spacetime*: you select it by layering a `Spacetime` under a
relativistic regime, and the geometry supplies the metric the fluid evolves on:

- `MINKOWSKI` — flat, i.e. plain special relativity
- `SCHWARZSCHILD_KS` — nonspinning Schwarzschild in horizon-penetrating
  Kerr-Schild coordinates; gas can cross r = 2M
- `KERR_KS` — spinning Kerr in horizon-penetrating Kerr-Schild coordinates

For example, GR hydrodynamics around a Kerr black hole uses the `RHD` regime with the `KERR_KS`
spacetime.

### Horizon excision

Horizon-penetrating charts (`SCHWARZSCHILD_KS` and `KERR_KS`) support excision inside a black
hole. Set `excision_radius` inside the horizon and above the metric guard at M/2 (roughly 0.7 r_+).
At each step, cells inside that radius are set to a cold vacuum floor and their conserved state is
rebuilt using the local metric. The exterior flow can cross the horizon while the computational
domain remains regular. Excision works with hydro and MHD, including spinning `KERR_KS` horizons,
GPU runs, and multi-GPU decomposition. In MHD, constrained transport continues to own the
staggered magnetic faces.

### Coordinate systems

- `CARTESIAN`, the usual x, y, z
- `SPHERICAL`, r, theta, phi
- `CYLINDRICAL` and `AXIS_CYLINDRICAL`, which both use the r-z plane in 2D; r-z is the default plane
- `PLANAR_CYLINDRICAL`, the r-phi plane

### Numerical methods

**Riemann solvers:**
- `HLLE` uses a two-wave fan and a branch-free closed form that vectorizes well.
- `HLLC` adds a contact wave to HLL following [Toro, Spruce & Speares
  (1994)](https://doi.org/10.1007/BF01414629). Toro's adaptive pressure estimate is evaluated only in cells
  that need the shock estimate. The MHD regimes use HLLC fluxes with HLL edge EMFs. The isothermal
  regimes use `HLLE`, since their wave structure has no thermal contact.
- `HLLC_PLUS` implements the corrections from [Chen et al. (2020)](https://doi.org/10.1137/18M119032X).
  The corrections act on the normal and transverse velocity jumps without changing the signal
  speeds, contact speed, or star states. They depend on the local flow, vanish at the sonic point,
  and do not require a reference Mach number.
  - The **normal** correction reduces excessive dissipation at low Mach number. In the unmodified
    scheme, this term scales with the sound speed and produces `O(Ma)` pressure fluctuations rather
    than the `O(Ma^2)` behavior of the continuous Euler equations. [Guillard & Viozat
    (1999)](https://doi.org/10.1016/S0045-7930%2898%2900017-6) gives the asymptotic analysis, and
    [Chen et al. (2022)](https://doi.org/10.1016/j.jcp.2022.111027) presents the correction in a
    form that applies to several Godunov fluxes.
  - The **transverse** correction adds shear dissipation near grid-aligned shocks to suppress the
    carbuncle instability. It is activated by a characteristic-speed reversal between neighboring
    cells, which distinguishes a shock from a steep hydrostatic pressure gradient.

  Newtonian hydrodynamics uses both corrections. RHD uses the transverse correction with the
  relativistic enthalpy density `rho h W^2 = e + p` in place of mass density. For stratified
  problems, use `HLLC_PLUS` with `wb_reconstruction`; otherwise the reduced low-Mach dissipation
  allows the hydrostatic truncation residual to produce a non-convergent entropy deficit.
- `HLLD` uses [Miyoshi & Kusano (2005)](https://doi.org/10.1016/j.jcp.2005.02.017) for Newtonian
  MHD, [Mignone (2007)](https://doi.org/10.1016/j.jcp.2006.12.031) for isothermal MHD, and
  [Mignone, Ugliano & Bodo (2009)](https://doi.org/10.1111/j.1365-2966.2008.14221.x) for RMHD.

**Well-balanced reconstruction:**

With `wb_reconstruction=True`, the scheme reconstructs each cell's departure from its local
hydrostatic equilibrium ([Käppeli & Mishra 2016](https://doi.org/10.1051/0004-6361/201527815)),
including equilibria with arbitrary entropy stratification,
and writes the gravity source as an equilibrium-pressure difference at the cell faces. A sealed
stratified column then holds its discrete equilibrium to machine precision (velocity residual
around 1e-15 and entropy deficit around 1e-16). The same representation is used for reconstruction,
source terms, reflecting-wall ghosts, first-order flux correction, and coarse-fine transfer.

This is available for Newtonian gamma-law hydrodynamics on Cartesian, cylindrical, and spherical
grids with `HLLE`, `HLLC`, or `HLLC_PLUS`. It requires an immersed gravitating body. On curvilinear
grids, the area-weighted pressure difference balances the geometric pressure source and flux
divergence. The extra work is about 1.4x in the flux stage and less over a complete step.

**Grid spacing:**
- `LINEAR`, uniform spacing
- `LOG`, log spacing, handy for spherical setups
- `GEOMETRIC`, a graded mesh in which each cell is a fixed ratio larger than the previous one. This
  concentrates resolution near one or both boundaries. Set the
  growth per axis with `x1_spacing_ratio` / `x2_spacing_ratio` / `x3_spacing_ratio` (and pick the
  spacing itself with `x1_spacing` and friends); see `geometric_boundaries.py` for the shape of it.

**Boundary conditions:**
- `PERIODIC`, wrap around
- `REFLECTING`, mirror symmetry
- `OUTFLOW`, zero gradient
- `DYNAMIC`, user-defined expressions

There are also two dataclass boundaries you can drop into the per-face list alongside those:
`Neumann` (prescribed gradient) and `Robin` (mixed `a*U + b*dU/dn = c`), per primitive variable.

**Time integration:**
- `RK1`, forward Euler (`EULER` also parses, as an alias)
- `RK2`, second-order SSP Runge-Kutta ([Shu 1988](https://doi.org/10.1137/0909073)); refinement
  levels use Berger-Colella subcycling
- `RK3`, third-order SSP Runge-Kutta ([Shu 1988](https://doi.org/10.1137/0909073))

**Constrained transport (MHD):**
- `CONTACT`, the edge-EMF construction of [Gardiner & Stone
  (2005)](https://doi.org/10.1016/j.jcp.2004.11.016) (default)
- `UCT`, the upwind constrained-transport method of [Mignone & Del Zanna
  (2021)](https://doi.org/10.1016/j.jcp.2020.109748), which suppresses the checkerboard mode

Both methods keep div B at machine precision. The test suite checks the discrete identity
div(curl) = 0 symbolically and includes bug-injection tests for this property.

**A few extras:**
- `reconstruction` picks `PCM`, `PLM`, or [PPM from Colella & Woodward
  (1984)](https://doi.org/10.1016/0021-9991(84)90143-8). The shorthand `--order 1/2/3` pairs each one with
  its corresponding time integrator: PCM+RK1, PLM+RK2, or PPM+RK3. PPM includes a
  convergence-gated flattener for spurious entropy loss in smooth sustained compression, such as
  gravitational infall onto a sink, while retaining its formal order.
- `plm_theta`, the PLM limiter parameter (0 < theta <= 2, default 1.5; theta = 2 is the sharpest). `limiter` picks `MINMOD` or `VAN_LEER` (van Leer is the smooth harmonic one and ignores `plm_theta`); `Limiter` lives at `simbi.types.input`
- First-order flux correction (FOFC): if a high-order update drives a cell unphysical, that cell is redone at first order, and the run reports how often that happened — per window while it runs, and again in the exit summary
- Prolongation at refinement boundaries runs one order above the interior reconstruction, which preserves the scheme's accuracy across level edges

### What runs where

Feature coverage varies by chart and regime. SIMBI validates these combinations at startup:

| Feature | Where it works |
|---|---|
| `HLLC` / `HLLC_PLUS` | Newtonian hydro, RHD, and both MHD regimes (the ones carrying a contact wave). `HLLC_PLUS` is Newtonian (both corrections) + RHD (the shear half) |
| `HLLD` | the MHD regimes |
| `wb_reconstruction` | Newtonian gamma-law hydro on cartesian, cylindrical, and spherical charts with `LINEAR`, `LOG`, or geometrically graded spacing, with `HLLE`/`HLLC`/`HLLC_PLUS`; carries through refinement and needs a gravitating immersed body |
| viscosity | adiabatic and isothermal, on every chart: cartesian, cylindrical, and spherical, in 2D, 2.5D (3-component on a 2-axis grid), and 3D. Viscosity is not implemented for `RHD`; its coefficient is currently accepted but ignored |
| alpha-disk viscosity | the same charts as constant-nu viscosity, and it needs a central immersed body |
| resistivity | every MHD chart: cartesian, cylindrical, and spherical, in 2.5D (r-z, r-phi, r-theta) and 3D |
| refinement | cartesian with `LINEAR` spacing. MHD refinement is 3D cartesian only, and runs on its own — immersed bodies and mesh motion are separate paths |
| passive scalar | Newtonian and isothermal, cartesian. carries through refinement, immersed bodies, mesh motion, and multi-GPU |
| tracers | flat cartesian (refinement is fine) |
| horizon excision | 3D cartesian, or 1D/2D spherical. 2D cartesian is unsupported because the slice represents a black string and the staircased excision circle seeds a growing m = 4 mode |

For a GR run you'll also want `schwarzschild_mass` and, on Kerr, `kerr_spin` (with `|a| <= M`).
`excision_radius` comes from your own subclass — declare it there, and keep it between `M/2`
and `r_+`.

### Non-ideal transport

Two dissipative terms sit on top of the ideal solvers, both off by default (coefficient zero):

- `viscosity` — a Navier-Stokes shear viscosity. Give it a constant, or set `viscosity_alpha` for the alpha-disk law (nu ~ alpha c_s H) in accretion-disk setups. The coefficient is spelled `viscosity_alpha` rather than `alpha` so it cannot be confused with a problem's own quantity of that name
- `resistivity` — Ohmic resistivity for the MHD regimes; the field diffuses while constrained transport keeps div B at machine zero

### Static mesh refinement

SIMBI uses the conservative level-transfer, subcycling, and flux-register construction of
[Berger & Colella (1989)](https://doi.org/10.1016/0021-9991(89)90035-1). Refinement regions are
currently specified in the problem configuration rather than selected dynamically.

```python
# turn refinement on
refinement_enabled: Annotated[bool, ProblemParam(True)]
refinement_max_levels: Annotated[int, ProblemParam(3)]
# each region is a FLAT list of 2*ndim floats: [x_lo, x_hi, y_lo, y_hi, ...]
refinement_regions: Annotated[
    list[list[float]],
    ProblemParam([[-0.1, 0.1, -0.1, 0.1], [-0.05, 0.05, -0.05, 0.05]]),
]
refinement_ratios: Annotated[list[int], ProblemParam([2, 2])]
refinement_subcycling_mode: Annotated[
    SubCycleMode, ProblemParam(SubCycleMode.STANDARD)
]
```

**Subcycling modes:**
- `STANDARD` / `NONE`, the fixed-ratio schedule: level `l` takes `2^l` steps per root step, and the
  root step is picked so every level stays inside its own CFL. These two are the same thing.
- `ADAPTIVE` / `MANUAL` are still on the roadmap. Today they raise `NotImplementedError` at
  validation, so a config that asks for one says so up front.

---

## Architecture

Here is a quick tour of how the Rust side fits together.

The physics is written once, generically over a carrier type, and traced into an intermediate
representation (IR). I like this part of the design because it lets the code stay close to the
mathematics while serving several backends. The IR becomes native CPU code at build time, CUDA or
HIP source at run time, or, for Python source terms, machine code compiled at startup with
Cranelift. Evaluating the same kernel with `f64` also gives the tests a reference for comparing CPU,
GPU, and JIT results bit for bit.

The compiler performs common-subexpression elimination, constant-power strength
reduction (`r ** -2` becomes two multiplies), lazy scheduling for sufficiently
expensive conditional branches, and select vectorization for suitable
branch-free kernel bodies. A `KernelProgram` owns its graph and writes as one
value; its derived `Effects` record reads, writes, in-place access, and stencil
reach. Checked composition uses those effects to preserve ordering and reject
unlawful parallel combinations, while the baked manifest is the sole authority
for runtime buffer roles. Generated kernels compute only the values needed by
their outputs.

A few core pieces:

- **`symbi-ir`** holds the kernel IR, graph passes, and CPU/CUDA/HIP code generation
- **`symbi-hydro`** is the physics: regimes, equations of state, and the Riemann solvers
- **`symbi-jit`** is the Cranelift JIT for runtime-authored kernels (your Python source expressions)
- **`symbi`** is the user-facing Rust crate: the builder API, scientific `Problem` surface, prelude, and orchestration entry points
- **`symbi-sim`** holds simulation state, shared stage/step machinery,
  checkpoint I/O, census, tracers, the flat-decomposed driver, and run-owned
  diagnostic evidence. It sits below the concrete integrators in the dependency
  graph
- **`symbi-discretize`** traces the carrier-generic physics into the IR
- **`symbi-aot`** bakes those traced kernels at build time
- **`symbi-exec`** is the CPU executor and its cache-blocked cover
- **`symbi-expr`** compiles the source expressions you write in Python (this is where the strength reduction lives)
- **`symbi-geometry`** holds the metrics, coordinate charts, and the `Spacetime` implementations
- **`symbi-io`** does the HDF5 checkpoints, and **`symbi-display`** is the live TUI
- **`symbi-substrate`** assembles the per-regime kernel sets (flux, c2p, godunov, cfl, ghost fill)
- **`symbi-refinement`** is the fixed refinement hierarchy and its evolution driver: prolongation, restriction, flux registers, and subcycling
- **`symbi-ib`** is the immersed-body layer: body state, motion, and accretion ledgers
- **`symbi-xpu`** is the device layer: memory, streams, and kernel launches
- **`symbi-afterglow`** does the radiation transport and observables
- **`symbi-py`** is the thin pyo3 bridge that becomes the `cpu_ext` and `gpu_ext` Python modules

Fields use a structure-of-arrays layout so CPU accesses can vectorize and GPU accesses can
coalesce. On the CPU, kernels run over cache-blocked tiles whose contiguous axis spans a full grid
row, giving the vectorized kernel bodies long unit-stride ranges. Time stepping goes through the
`KernelSet` trait, which gives the driver one interface to the simulation state. A multi-GPU run
treats each subdomain as its own simulation state and exchanges halos between neighboring tiles in
lockstep. It uses the same halo machinery as the refinement hierarchy and gives bit-identical
results to a monolithic run.

The IR is precision-agnostic: the same graph renders to `f64` or `f32` at the target's launch
precision, and the Cranelift path is generic over the scalar type as well. CUDA and HIP share the
kernel definitions and have small target-specific implementations for the runtime and token
mapping.

For one concrete reference point, in double precision on my Apple M4 Pro laptop with eight performance cores, a second-order 3D Newtonian linear-wave problem using HLLE at 256³ sustains about 38 million zone-cycles per second (MZCS). A 2D Kelvin–Helmholtz problem with HLLE runs at about 70 MZCS. Your problems will have their own numbers; set `SYMBI_PROFILE=1` to see where the time goes in a particular run.

---

## Example Configurations

There are 68 ready-to-run configs under `simbi_configs/examples/`, sorted into `newtonian/`,
`srhd/`, `srmhd/`, `isothermal/`, `grhd/`, `grmhd/`, and `ibm/`. A sampler:

The `Run with` column is the slug you pass to `simbi run` (the file stem with underscores swapped
for dashes; underscores also work). The CLI finds it by name, wherever it lives.

| Example | Run with | What it is |
|---------|----------|------------|
| `newtonian/sod.py` | `sod` | Newtonian shock tube |
| `srhd/marti_muller.py` | `marti-muller` | SRHD shock tube (1D and 3D variants) |
| `newtonian/kh.py` | `kh` | Kelvin-Helmholtz instability |
| `newtonian/rt.py` | `rt` | Rayleigh-Taylor instability (with gravity) |
| `newtonian/sedov.py` | `sedov` | Sedov-Taylor explosion (spherical) |
| `newtonian/linear_wave.py` | `linear-wave` | Linear wave convergence — the benchmark behind the throughput number above |
| `srhd/thermal_bomb.py` | `thermal-bomb` | Thermal bomb (2D and 3D variants) |
| `srmhd/magnetic_blast.py` | `magnetic-blast` | MHD blast wave |
| `srmhd/magnetic_shock_tube.py` | `magnetic-shock-tube` | 1D MHD shock |
| `srmhd/rmhd_orszag_tang.py` | `rmhd-orszag-tang` | SRMHD Orszag-Tang vortex (Newtonian, isothermal, and resistive variants also ship) |
| `newtonian/field_loop.py` | `field-loop` | Advected field loop — the constrained-transport regression |
| `newtonian/quirk.py` | `quirk` | Odd-even decoupling — run it with `--solver hllc` and `hllc_plus` and diff |
| `isothermal/kepler.py` | `kepler` | Keplerian disk with a central mass |
| `newtonian/bondi.py` | `bondi` | 3D Bondi accretion onto a sink, with a sponge zone and optional refinement |
| `newtonian/refined_blast.py` | `refined-blast` | Static mesh refinement on a blast wave |
| `newtonian/traced_kh.py` | `traced-kh` | Lagrangian tracer particles riding a KH billow |
| `newtonian/dyed_kh.py` | `dyed-kh` | Passive scalar (dye) advection |
| `newtonian/viscous_shear.py` | `viscous-shear` | Navier-Stokes shear viscosity |
| `newtonian/resistive_orszag_tang.py` | `resistive-orszag-tang` | Ohmic resistivity in MHD |
| `ibm/tumbling_body.py` | `tumbling-body` | Two-way coupled rigid body: tumbles and precesses |
| `ibm/rubble_wind.py` | `rubble-wind` | Bonded rubble-pile fragments in a wind |
| `ibm/magnetized_sink.py` | `magnetized-sink` | Accreting sink with Ohmic magnetic coupling |
| `isothermal/dittmann_single_disk.py` | `dittmann-single-disk` | Torque-free sink in a disk |
| `newtonian/uniform_sphere.py` | `uniform-sphere` | Uniform sphere with homologous mesh expansion |
| `srhd/quad_shocktube.py` | `quad-shocktube` | 2D multi-region shock |
| `newtonian/ordered_sources.py` | `ordered-sources` | Ordered density, momentum, and energy source composition |
| `newtonian/rotating_sponge.py` | `rotating-sponge` | Rotating-frame forces composed with an outer sponge |
| `newtonian/tabulated_source_1d.py` | `tabulated-source-1d` | Piecewise-linear tabulated energy source |
| `newtonian/geometric_boundaries.py` | `geometric-boundaries` | Graded (geometric) mesh concentrating cells at a boundary |
| `newtonian/decomposed_tabulated_geometric.py` | `decomposed-tabulated-geometric` | Geometric mesh and tabulated source on one or several devices |
| `grhd/gr_fishbone_moncrief.py` | `gr-fishbone-moncrief` | GRHD Fishbone-Moncrief torus over nearly the full meridional domain |
| `grhd/gr_fishbone_moncrief_cartesian.py` | `gr-fishbone-moncrief-cartesian` | Pole-free 3D Cartesian Kerr-Schild torus |
| `grmhd/gr_fishbone_moncrief_mhd.py` | `gr-fishbone-moncrief-mhd` | Spinning-hole GRMHD torus with a divergence-free MRI seed field |
| `grmhd/gr_kerr_dragging.py` | `gr-kerr-dragging` | Kerr frame dragging of a weak magnetic field |

Run any of them:

```bash
uv run simbi run sedov --mode gpu
uv run simbi run kepler --mode cpu --resolution 128,128
```

---

## Publications

SIMBI has been used in the following papers:

| Year | Publication |
|------|-------------|
| **2026** | [DuPont, M. & Quataert E., "Self-Limited Accretion onto Embedded Binaries in a Uniform Medium"](https://iopscience.iop.org/article/10.3847/1538-4357/ae7431) |
| **2024** | [DuPont, M. et al., "Strong Bow Shocks: Turbulence and An Exact Self-Similar Asymptotic"](https://iopscience.iop.org/article/10.3847/1538-4357/ad5adc) |
| **2023** | [DuPont, M. et al., "Explosions in Roche-lobe Distorted Stars: Relativistic Bullets in Binaries"](https://iopscience.iop.org/article/10.3847/1538-4357/ad284e) |
| **2023** | [DuPont, M. & MacFadyen A., "Stars Bisected By Relativistic Blades"](https://iopscience.iop.org/article/10.3847/2041-8213/ad132c) |
| **2022** | [DuPont, M. et al., "Ellipsars: Ring-like Explosions from Flattened Stars"](https://iopscience.iop.org/article/10.3847/2041-8213/ac6ded) |

---

## Citation

```bibtex
@software{2023ascl.soft08003D,
       author = {{DuPont}, Marcus},
        title = "{SIMBI: 3D relativistic gas dynamics code}",
 howpublished = {Astrophysics Source Code Library, record ascl:2308.003},
         year = 2023,
        month = aug,
          eid = {ascl:2308.003},
archivePrefix = {ascl},
       eprint = {2308.003},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023ascl.soft08003D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

---

## Version History

| Version | Changes |
|---------|---------|
| **v0.9.0** (current) | Full rewrite of the compute backend from C++ to Rust (traced-IR kernels, LLVM + NVRTC + hipRTC + Cranelift); production NVIDIA CUDA and AMD HIP support; GR spacetimes + horizon excision; constrained-transport MHD; viscosity + resistivity; two-way rigid-body immersed walls (CSG shapes, spin, energy-conserving coupling); single-node multi-GPU domain decomposition; static mesh refinement; live TUI; the big performance campaign |
| **v0.8.0** | Minimized compiler warnings | 
| **v0.7.0** | Added mypy type checking and the immersed boundary method |
| **v0.6.0** | Fixed git tag ordering, general refactoring |
| **v0.5.0** | Performance optimizations |
| **v0.4.0** | Code restructuring |
| **v0.3.0** | Improved code organization |
| **v0.2.0** | Memory contiguity optimizations |
| **v0.1.0** | Initial release |

---

## Support

Found a bug or want a feature? Open an issue at [GitHub Issues](https://github.com/EigenDev/simbi/issues).

When an install fails, check the basics first:

```bash
python --version   # want 3.10 or newer
cargo --version    # the rust toolchain is present
nvidia-smi         # the GPU and driver are visible (NVIDIA)
rocminfo           # the GPU and ROCm runtime are visible (AMD)
```

When a run misbehaves:

```bash
simbi run <problem> --info  # see the options this problem takes
simbi run --configs         # list the problems you can run
```

When a run is *slow* and you want receipts:

```bash
SYMBI_PROFILE=1 simbi run <problem> ...   # per-phase wall-time breakdown at exit
```

---

## License

SIMBI is distributed under the [MIT License](https://opensource.org/licenses/MIT).

---

## Acknowledgements

SIMBI was developed at the Center for Cosmology and Particle Physics (CCPP) at New York University, and I thank the CCPP group for their support and feedback. I also thank the following people for their contributions to the project:

- **Andrew MacFadyen** (NYU) for his mentorship and guidance on the project.
- **Jonathan Zrake** (Clemson University) for his intellectual feedback on the project.
- **Jim Stone** (Institute for Advanced Study) for his feedback on the MHD implementation and for pointing me to the robust conserved-to-primitive formalism of [Kastaun et al. 2021](https://scixplorer.org/abs/2021PhRvD.103b3018K/abstract).
- **Romain Teyssier** (Princeton University) for his willingness to talk shop with me, especially as I was getting into mesh refinement.
---

## Further reading on physics and numerical methods

- **[Marti & Muller 1994](https://ui.adsabs.harvard.edu/abs/2003LRR.....6....7M/abstract)**: The relativistic shock-tube work that got me started. I found it while learning relativistic hydrodynamics as a graduate student, and it remains a good entry point.
- **[Font 2007](https://ui.adsabs.harvard.edu/abs/2008LRR....11....7F/abstract)**: A review of numerical methods for relativistic magnetohydrodynamics.
- **[Andersson & Comer 2021](https://link.springer.com/article/10.1007/s41114-021-00031-6)**: A modern review of relativistic fluid dynamics with a particularly good introduction.
- **[Moseley et al. 2026](https://arxiv.org/abs/2604.23041)**: A method for evolving Lagrangian tracer particles in an Eulerian fluid simulation. I learned about this work directly from Romain Teyssier and found it very useful for SIMBI.
- **[Berberich et al. 2021](https://www.sciencedirect.com/science/article/pii/S0045793021000244#section-cited-by)**: A modern and generic technique for well-balanced evolution that is equation of state independent.
- **[Guillard & Viozat 1999](https://doi.org/10.1016/S0045-7930%2898%2900017-6)**: An asymptotic analysis showing why numerical viscosity in upwind schemes scales poorly at low Mach number.
- **[Chen, Lin, Li & Yan 2020](https://doi.org/10.1137/18M119032X)**: The HLLC+ solver used by SIMBI for low-Mach flows. It treats the normal velocity jump and transverse carbuncle instability separately; Appendix A gives concise pseudocode.
- **[Chen et al. 2022](https://doi.org/10.1016/j.jcp.2022.111027)**: A general presentation of the low-Mach correction for Rusanov, HLL, Roe, HLLC, and AUSM+ solvers.
- **[Fleischmann, Adami & Adams 2020](https://www.sciencedirect.com/science/article/pii/S0021999120305362)**: The HLLC-LM solver, which SIMBI previously included. Its signal-speed scaling also damps the pressure jump, an important distinction for hydrostatic problems.
- **[Quirk 1994](https://doi.org/10.1002/fld.1650180603)**: The source of the odd-even decoupling test used by SIMBI as a carbuncle regression.
---

<div align="center">

**[Report a Bug](https://github.com/EigenDev/simbi/issues) · [Request a Feature](https://github.com/EigenDev/simbi/issues)**

</div>

---

The C++-to-Rust port benefited greatly from Claude Code. Et tu, Brute?
