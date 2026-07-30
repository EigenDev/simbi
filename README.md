# SIMBI

```
  ███████╗██╗███╗   ███╗██████╗ ██╗
  ██╔════╝██║████╗ ████║██╔══██╗██║
  ███████╗██║██╔████╔██║██████╔╝██║
  ╚════██║██║██║╚██╔╝██║██╔══██╗██║
  ███████║██║██║ ╚═╝ ██║██████╔╝██║
  ╚══════╝╚═╝╚═╝     ╚═╝╚═════╝ ╚═╝
```

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

SIMBI is a finite volume code for astrophysical fluid simulations. If you want to throw relativistic jets, shock tubes, stellar explosions, or magnetized turbulence at a grid and see what happens, this is the tool. Results from SIMBI have shown up in *The Astrophysical Journal* and *The Astrophysical Journal Letters*, covering relativistic jets, shock morphology, and stellar explosions.

A quick note on what this is these days: SIMBI started life as a C++ code and was rewritten from the ground up in Rust. I embarked on this because I got super interested in graph theory and how it can be used to help generate architecture-agnostic code, and I wanted to give it a shot from a clean slate in rust (see [Sethi & Ullman 1970](https://dl.acm.org/doi/10.1145/321607.321620) and works that cite them. Super interesting and deep stuff imo.) The physics is the same, the speed got better (because I became a better programmer over the years from the wee ol' monolithic-to-the-max days), and the codebase is a lot easier to reason about. You drive the whole thing from Python; the Rust is there when you want it.
<!--that the russt workspace is called `symbi` instead of `simbi`. Well, that's because I really love the idea of turning symbolic expressions into machine code, but since both "simbi" and "symbi" sound the same, I thought that shift was a cool nod to the new direction of the codebase. The Python package is still called `simbi`, so you can keep your scripts and configs the same.-->

**What you get:**
- Six fluid regimes in one code: Newtonian hydro, relativistic hydro (RHD), Newtonian and relativistic MHD, plus isothermal variants of both
- Spacetime as its own axis: hand the relativistic regimes a Minkowski or horizon-penetrating Kerr-Schild metric (nonspinning or Kerr) the same way you would pick a coordinate system
- GPU acceleration on NVIDIA CUDA and AMD ROCm/HIP devices, with kernels compiled on the fly for the active accelerator
- High-resolution shock capturing with HLLE, HLLC (plus a low-Mach variant, see [Fleischmann et al. 2020](https://www.sciencedirect.com/science/article/pii/S0021999120305362)), and HLLD Riemann solvers, backed by a first-order flux-correction safety net that logs every cell it touches (still working on strengthening this. Maybe I'll switch to [Zalesak and friends](https://apps.dtic.mil/sti/tr/pdf/ADA360122.pdf) at some point).
- Constrained-transport MHD (contact [Gardiner & Stone](https://arxiv.org/abs/0712.2634) or UCT ([Mignone & DelZanna (2021)](https://arxiv.org/abs/2004.10542)) edge EMFs) that keeps div B at machine zero by construction
- Physical transport when you want it: Navier-Stokes viscosity (constant or alpha-disk) and Ohmic resistivity, layered on top of the ideal solvers
- A rich immersed-boundary method: point-mass gravity, Bondi-Hoyle accretion sinks, and rigid walls of *any* constructive solid geometry (CSG) shape (spheres, boxes, unions/intersections) with no-penetration / no-slip porous surfaces. Bodies can be one-way (prescribed motion) or **two-way coupled** — the gas reaction force and torque then drive the body's full rigid-body dynamics: translation, plus rotation about an arbitrary axis via Euler's equations with a principal-moment inertia tensor, so an asymmetric body tumbles and precesses. Spinning walls drag the gas at omega x r, the gas <-> body energy exchange conserves total energy, and every body reports force / torque / accretion diagnostics. I took [Chuck Peskin](https://en.wikipedia.org/wiki/Charles_S._Peskin)'s class during my time as a graduate student at NYU,
and I loved it so much that I thought it'd be cool to introduce it into my code! 
- Horizon excision for GR accretion: on a horizon-penetrating Kerr-Schild chart the region inside the black hole is frozen at a cold vacuum, so you can swallow the singularity and still keep a well-posed accretion-rate certificate
- Block-based static mesh refinement with [Berger-Colella](https://www.sciencedirect.com/science/article/pii/0021999189900351) subcycling
- Single-node **multi-GPU domain decomposition** — set `gpus > 1` and the domain splits across the cards, halo-exchanged in lockstep and bit-identical to a monolithic run
- In-situ binned reductions (a "census"): declare shell profiles as expressions and the run reduces them on the device each cadence, straight into the checkpoint — handy when what you want out of a run is a scaling law
- Lagrangian tracer particles that ride along with the flow, across refinement levels and multi-GPU cuts too
- Afterglow radiation transport, so you can turn a simulation into synthetic observables
- A live terminal dashboard while you run (pause, single-step, checkpoint on demand, field heatmaps), and `simbi attach` to peek at a headless run from another shell
- A type-safe Python config system that generates its own CLI, so you stop hand-writing argument parsers

> CUDA and HIP are peer production backends generated from the same kernel
definitions. Multi-*node* decomposition is the next step on the roadmap;
single-node multi-GPU already runs today.
My science problems fit on one node these days, so multi-node sits far down the list.
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

If you only read one section, read this one. We lean hard on [uv](https://docs.astral.sh/uv/) here, and you should too — a strong recommendation, freely ignored. It is an absurdly fast Python package manager and environment tool, it replaces pip and venv and conda in one binary (though conda is objectively broader and more powerful; it's just a bit much for my use cases these days), and it makes the whole setup a two-line affair.

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
You can also save a bunch of time by doing `source .venv/bin/activate` and you'll remain in the 
simbi environment you created when you ran `uv venv`. 

Got an NVIDIA or AMD accelerator? Select its production backend through the
project helper:

```bash
uv sync --no-install-project   # puts maturin in the venv; dev.py needs it
./dev.py install --cuda        # or --hip for AMD; pick the one matching your card
uv run simbi run marti-muller --mode gpu --resolution 1024
```

That is the whole thing — cargo and maturin drive the build end to end.
(Though, I am a big fan of CMake and Ninja! I learned so much about programming and developing
a major project like this from utilizing those tools. It just becomes a bit much for me to
deal with as I explore more architectures and directions. Cargo is standard enough for my 
purposes these days. :D)

---

## Installation

### What you need

- A Rust toolchain (cargo), the easy way is [rustup](https://rustup.rs)
- Python 3.10 or newer
- HDF5 (it gets linked into the extension)
- For NVIDIA GPU execution, a driver providing `libcuda` and `libnvrtc`
- For AMD GPU execution, ROCm providing `libamdhip64` and `libhiprtc`
- On Linux, `patchelf` (uv and maturin will tell you if it is missing)

That is it. Kernels compile at run time with NVRTC on NVIDIA devices and hipRTC
on AMD devices, so the driver runtime is the whole GPU toolchain requirement.

### The uv way (recommended)

Seriously, just use uv. It builds the Rust extension through maturin behind the scenes and stays out of your way from there on.

```bash
# spin up an isolated environment
uv venv

# build and install the package
uv pip install .

# want the plotting and CLI niceties too?
uv pip install ".[visual,cli]"
```

From here on, prefix commands with `uv run` and you are always using the right environment:

```bash
uv run simbi run sedov --mode cpu
```

Just want the `simbi` command on your PATH without thinking about environments? Since it is a CLI tool, this is the slick option:

```bash
uv tool install .
```

Now `simbi` works from anywhere, on its own.

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

The CLI tries to be a good roommate: config names match with or without kebab-case, a
typo gets a "did you mean...?", and if two configs in different directories share a name,
it lists both and asks.

**Options you will reach for:**
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
TUI), or `crashed` (the solver gave up). All three are restartable, which is the point — a wall-clock
kill on a cluster still leaves you something to resume from.

> Heads up if you glob: `*.chkpt.final*.h5` only exists after a clean finish. A job killed at the
> wall-clock limit leaves `interrupted`, so a downstream script keyed on `final` quietly finds
> nothing.

Resuming with `--checkpoint` picks up the sim clock and the checkpoint numbering, so you get
`031, 032, ...` alongside the earlier files. `end_time` becomes the larger of yours and
the checkpoint's, so a restart can only extend a run. Fields marked `checkpoint_safe=True`
you can override on the command line; pass a conflicting flag that lacks that mark and the run
stops with a `ConfigError`, leaving the choice to you.

One thing to know: the `body_diagnostics` and census series inside a checkpoint cover the current
run segment only and start empty again after a restart. Stitch the segments offline.

Each checkpoint carries `metadata` (time, dt, iteration, gamma, cfl, regime, spacetime, solver,
spacing, ...) plus a `level_<N>` group per refinement level holding the primitives and conserved
state, and — when the run has them — `bodies`, `body_diagnostics`, `tracers`, and `census/<name>`.
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

Sometimes what you want out of a run is a scaling law, and dumping a thousand full snapshots to get
one is silly. A census is a binned reduction: you declare the bin axes and what to accumulate as
expressions, and the run reduces them on the device every cadence and tucks the result into each
checkpoint, ready to read straight back out.

```python
from simbi import expression as expr

@computed_field
@property
def census_expressions(self) -> list[ExpressionDict]:
    g = expr.ExprGraph()
    x1, x2, x3 = (expr.variable(v, g) for v in ("x1", "x2", "x3"))
    r = expr.sqrt(x1 * x1 + x2 * x2 + x3 * x3)
    vx, vy, vz = (expr.velocity(ii, g) for ii in (0, 1, 2))
    v_r = (x1 * vx + x2 * vy + x3 * vz) / r
    rho, dv = expr.density(g), expr.cell_volume(g)
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

Note that you only ever accumulate sums. That's deliberate: sums are the quantity that merges
cleanly across refinement levels, across decomposed tiles, and across restart segments. So the
reader forms means and variances as ratios of sums when you ask for them:

```python
from simbi.reader import census_names, read_census

census_names("run.chkpt.final.h5")     # ('shells',)
c = read_census("run.chkpt.final.h5", "shells")
c.bin_centers(0)                       # the radial axis
c.favre("mass_vr", "mass")             # mass-weighted <v_r> per shell
c.assert_fully_binned()                # loud if cells fell outside the bins
```

Every row carries its level, sample count, time span, and a `dropped` count — cells that fell
outside the bins. Worth watching, since a census that quietly under-covers its domain looks a lot
like real structure. `PER_LEVEL_STEP` samples each refinement level on its own clock, which is
usually what you want: root-step sampling under-resolves the innermost shells, and those are often
the ones you're fitting a slope to.

One gotcha: the history covers a single run segment and starts empty again on a restart. So for a
restart chain, grab the last checkpoint of each segment and stitch them offline (accumulating rows
combine as a count-weighted sum via `n_samples`).

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

Problems are plain Python classes. You inherit from `SimbiProblem`, declare your parameters with `ProblemParam`, and SIMBI builds the CLI for you from the type annotations. No argparse boilerplate, and the types are checked.

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

Note that `cli=True` is opt-in per field, and that includes `resolution` and `adiabatic_index` — a
config only gets `--resolution` if it declares it. `simbi run <problem> --info` tells you which flags
a given problem actually exposes.

### Deriving fields from other fields

If a parameter depends on another one, compute it in `setup()`. Declare the derived field
`Optional[...] = None` and fill it in there — and call `super().setup()`:

```python
bondi_radius: Annotated[Optional[float], ProblemParam(None)] = None

def setup(self) -> None:
    super().setup()
    self.bondi_radius = self.central_mass / self.sound_speed**2
```

There's also `summary()`, which returns `(group, label, value)` rows that show up in the live
dashboard's setup panel — handy for the derived numbers you'd otherwise print by hand.

### Source terms

Add gravity or custom hydro sources as expression graphs:

```python
from simbi import expression as expr

@computed_field
@property
def source_expressions(self) -> list[ExpressionDict]:
    graph = expr.ExprGraph()
    x_comp = expr.constant(0.0, graph)
    y_comp = expr.constant(-0.1, graph)      # constant downward gravity
    compiled = graph.compile([x_comp, y_comp])
    return [compiled.serialize_source(expr.SourceKind.FORCE, dim=2)]
```

`FORCE` is one of several kinds — there's also `ROTATING_FRAME`, `COOLING`, `RELAX`, `SPONGE`,
`INJECT`, and `RAW`, and `serialize_source` takes `params=` (runtime scalars), `region=` (a mask
folded in), and `target=`. See `newtonian/rt.py` and `newtonian/ordered_sources.py`.

### Immersed bodies

Drop objects into the domain. Each body carries one `capability`:

- `GRAVITATIONAL` — a fixed-potential (softened) point mass
- `ACCRETION` — a Bondi-Hoyle sink: it removes mass, and its `AccretionProperties` can layer on a porous surface (a `porosity` dial), a no-penetration/no-slip wall, or a torque-free (Dittmann) sink that swallows mass without angular momentum
- `RIGID` — a solid wall. Sphere by default, or *any* CSG `Shape` (boxes, spheres, `union`/`intersect`/`rotated`) authored in the body frame. The surface enforces no-penetration (`k_eta_n`) and, under `apply_no_slip`, no-slip tangential drag (`k_eta_t`)

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

Flip `two_way_coupling=True` and the body stops being scenery: the gas reaction force and torque
drive its full rigid-body motion — it translates, and rotates about an arbitrary axis via Euler's
equations with an anisotropic inertia tensor, so an off-axis spin precesses and an asymmetric shape
tumbles. A tumbling card in a wind tunnel:

```python
from simbi.types import ImmersedBodyConfig, BodyCapability, RigidProperties, Shape

card = Shape.box((0.0, 0.0, 0.0), (0.45, 0.22, 0.15))   # half-extents, body frame
ImmersedBodyConfig(
    capability=BodyCapability.RIGID,
    mass=1.0,
    radius=1.0,                 # the mask-gate scale; the CSG defines the geometry
    position=(0.0, 0.0, 0.0),
    velocity=(0.0, 0.0, 0.0),
    two_way_coupling=True,      # the flow moves AND spins the body; the reaction acts back
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

The gas <-> body energy exchange is conserved (drag heats the gas; an isothermal wall carries that
heat off to its reservoir), and every body reports its force, torque, and accreted mass each step.
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

The `Spacetime` axis sets a run's relativity: `RHD`/`RMHD` on Minkowski are special-relativistic, on a curved spacetime general-relativistic. Checkpoints and configs written under the legacy `srhd`/`srmhd` slugs still load (mapped to `rhd`/`rmhd`).

### Spacetimes

Relativity in SIMBI lives in the *spacetime*: you select it by layering a `Spacetime` under a
relativistic regime, and the geometry supplies the metric the fluid evolves on:

- `MINKOWSKI` — flat, i.e. plain special relativity
- `SCHWARZSCHILD_KS` — nonspinning Schwarzschild in horizon-penetrating
  Kerr-Schild coordinates; gas crosses r = 2M without drama
- `KERR_KS` — spinning Kerr in horizon-penetrating Kerr-Schild coordinates

So "GR hydro around a Kerr black hole" is the same `RHD` regime you already know,
handed a different metric.

### Horizon excision

Point a horizon-penetrating chart (`SCHWARZSCHILD_KS` or `KERR_KS`) at a black hole and you can actually
*swallow* it. Set `excision_radius` to a radius inside the horizon (above the metric-guard radius
M/2, so around 0.7 r_+) and every step the cells inside get frozen at a cold vacuum floor, with
their conserved state rebuilt from the local metric. The exterior gas rarefies in and nothing comes
back out, so the accretion-rate certificate stays well-posed and the chart stays regular straight
through the horizon. Works for hydro and MHD (the staggered magnetic faces stay constrained-transport-owned),
for spinning (`KERR_KS`) horizons, on the GPU, and across the multi-GPU decomposed path.

### Coordinate systems

- `CARTESIAN`, the usual x, y, z
- `SPHERICAL`, r, theta, phi
- `CYLINDRICAL` and `AXIS_CYLINDRICAL`, which in 2D both mean the r-z plane (they're the same metric; watch out, r-z is the default plane)
- `PLANAR_CYLINDRICAL`, the r-phi plane

### Numerical methods

**Riemann solvers:**
- `HLLE`, the two-wave workhorse, written in a branch-free closed form the compiler can vectorize
- `HLLC`, HLL with a contact wave, Toro's adaptive pressure estimates evaluated lazily — the shock estimate is paid for only in the cells that need it. Works on the MHD regimes too (HLLC flux + HLL edge EMF); the *isothermal* regimes stay on `HLLE`, whose two-wave fan matches their wave structure
- `HLLC_LM`, the Fleischmann (2020) low-Mach / low-dissipation HLLC, for both Newtonian and relativistic hydro
- `HLLD`, HLL with discontinuities (magnetohydrodynamics), faithful to Mignone & Del Zanna

**Grid spacing:**
- `LINEAR`, uniform spacing
- `LOG`, log spacing, handy for spherical setups
- `GEOMETRIC`, a graded mesh: each cell is a fixed ratio bigger than the last, so you can pack
  resolution against one boundary (or both) without paying for it across the whole domain. Set the
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
- `RK2`, second-order SSP Runge-Kutta (Berger-Colella subcycling under refinement)
- `RK3`, third-order SSP Runge-Kutta

**Constrained transport (MHD):**
- `CONTACT`, Gardiner & Stone (2005) edge EMFs (default)
- `UCT`, Del Zanna / Mignone & Del Zanna upwind CT (kills the checkerboard mode)

Either way, div B stays at machine zero by construction — the curl-of-EMF update carries
a symbolic proof of div(curl) = 0 in the test suite, and bug-injection tests keep the
proof valid.

**A few extras:**
- `plm_theta`, the PLM reconstruction parameter (must be > 0 and <= 2, default 1.5). For piecewise-constant use `reconstruction=PCM` or `--order 1`
- `reconstruction` picks `PCM` or `PLM`, and `limiter` picks `MINMOD` or `VAN_LEER` (van Leer is the smooth harmonic one and ignores `plm_theta`). `Limiter` lives at `simbi.types.input`
- First-order flux correction (FOFC): if a high-order update drives a cell unphysical, that cell is redone at first order, and the run reports how often that happened — per window while it runs, and again in the exit summary
- Prolongation at refinement boundaries runs one order above the interior reconstruction, which preserves the scheme's accuracy across level edges

### What runs where

Feature coverage varies by chart and regime. Every combination below is checked at startup and
refused loudly when it falls outside the table, though it saves you a round trip to know up front:

| Feature | Where it works |
|---|---|
| `HLLC` / `HLLC_LM` | Newtonian hydro, RHD, and both MHD regimes (the ones carrying a contact wave). `HLLC_LM` is Newtonian + RHD |
| `HLLD` | the MHD regimes |
| viscosity | adiabatic and isothermal, on every chart: cartesian, cylindrical, and spherical, in 2D, 2.5D (3-component on a 2-axis grid), and 3D. `RHD` accepts the coefficient and silently ignores it |
| alpha-disk viscosity | the same charts as constant-nu viscosity, and it needs a central immersed body. cartesian 3D is baked for the isothermal regime only |
| resistivity | cartesian 2.5D/3D, cylindrical r-z and r-phi, spherical r-theta |
| refinement | cartesian with `LINEAR` spacing. MHD refinement is 3D cartesian only, and runs on its own — immersed bodies and mesh motion are separate paths |
| passive scalar | Newtonian and isothermal, cartesian. carries through refinement, immersed bodies, mesh motion, and multi-GPU |
| tracers | flat cartesian (refinement is fine) |
| horizon excision | 3D cartesian, or 1D/2D spherical. 2D cartesian is refused on purpose — that slice is a black *string*, and the staircased excision circle seeds a growing m = 4 mode |

For a GR run you'll also want `schwarzschild_mass` and, on Kerr, `kerr_spin` (with `|a| <= M`).
`excision_radius` comes from your own subclass — declare it there, and keep it between `M/2`
and `r_+`.

### Non-ideal transport

Two dissipative terms sit on top of the ideal solvers, both off by default (coefficient zero):

- `viscosity` — a Navier-Stokes shear viscosity. Give it a constant, or switch on the alpha-disk law (nu ~ alpha c_s H) for accretion-disk setups
- `resistivity` — Ohmic resistivity for the MHD regimes; the field diffuses while constrained transport keeps div B at machine zero

### Static mesh refinement

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

Here is the quick tour of how the Rust side fits together, in case you want to hack on it.

The compute backend is a Cargo workspace of small, focused crates. The interesting idea at the center of it: the physics is written ONCE, generically over a "carrier" type, and traced into an intermediate representation. That IR gets lowered to native CPU code (compiled by LLVM at build time), CUDA source (compiled at run time with NVRTC), HIP source (compiled at run time with hipRTC), or — for the source terms you write in Python — machine code JIT-compiled at startup with Cranelift. One definition of the math serves every backend. The same trick powers the test suite: the f64 evaluation of a kernel doubles as its own oracle, so CPU, GPU, and JIT are checked against each other bit-for-bit.

The compiler layer earns its keep: common-subexpression elimination, constant-power strength reduction (`r ** -2` in your config compiles down to two multiplies), automatic lazy scheduling of expensive conditional branches (a `where(...)` in your source expressions becomes a real branch when the arms are worth skipping), and a cost-gated select-vectorization pass that turns branch-free kernel bodies into NEON/SIMD-friendly straight-line code. The guiding rule, enforced by the graph itself: only compute what you actually need.

A few core pieces:

- **`symbi-ir`** holds the kernel IR, graph passes, and CPU/CUDA/HIP code generation
- **`symbi-hydro`** is the physics: regimes, equations of state, and the Riemann solvers
- **`symbi-jit`** is the Cranelift JIT for runtime-authored kernels (your Python source expressions)
- **`symbi`** is the top crate: the builder API and the single-grid `evolve` driver
- **`symbi-sim`** is the hub everything orbits — simulation state, checkpoint I/O, the census, tracers, and the decomposed driver. It sits *below* the integrator on purpose, so nothing in it depends upward
- **`symbi-discretize`** is where the carrier-generic physics actually gets traced into the IR
- **`symbi-aot`** bakes those traced kernels at build time
- **`symbi-exec`** is the CPU executor and its cache-blocked cover
- **`symbi-expr`** compiles the source expressions you write in Python (this is where the strength reduction lives)
- **`symbi-geometry`** holds the metrics and charts — the whole `Spacetime` axis
- **`symbi-io`** does the HDF5 checkpoints, and **`symbi-display`** is the live TUI
- **`symbi-substrate`** assembles the per-regime kernel sets (flux, c2p, godunov, cfl, ghost fill)
- **`symbi-amr`** is the refinement hierarchy: prolongation, restriction, flux registers, and subcycling
- **`symbi-ib`** is the immersed-body layer: body state, motion, and accretion ledgers
- **`symbi-xpu`** is the device layer: memory, streams, and kernel launches
- **`symbi-afterglow`** does the radiation transport and observables
- **`symbi-py`** is the thin pyo3 bridge that becomes the `cpu_ext` and `gpu_ext` Python modules

A few design choices worth calling out. Fields are stored struct-of-arrays, which is what lets the CPU vectorize and the GPU coalesce its memory reads. The CPU executor fans serial kernels over a cache-blocked cover whose tiles run the full grid row along the contiguous axis, which gives the vectorized kernel bodies the long unit-stride runs they thrive on. And the time step is sequenced entirely through a `KernelSet` trait, so the driver touches state only through that one interface. That last part is what makes multi-GPU work: a subdomain is just a self-contained simulation state, so `gpus > 1` splits the domain into tiles that halo-exchange between neighbors in lockstep — bit-identical to a monolithic run, and riding the exact halo machinery the refinement hierarchy already uses. Multi-*node* is the natural next step from here.

The neutral IR is precision-agnostic too: the same traced graph renders to f64 or f32 at the target's launch precision (an f32 device run just halves the bandwidth bill), and the Cranelift runtime path is generic over the scalar the same way. The device backend is written against a backend-agnostic trait, so the production CUDA and HIP paths share one kernel definition and diverge only in the small target-specific runtime and token mapping at the bottom.

On speed (one machine, one problem class, double precision): the 3D Newtonian linear wave at 256^3 sustains ~38 million zone-cycles per second on an 8-performance-core Apple M4 Pro laptop, and a 2D Kelvin-Helmholtz with HLLE runs around 70. For a sense of scale, [AthenaK](https://github.com/IAS-Astrophysics/athenak) reports 34 Mzc/s for the same class of test on an M1 Pro. Your problems will have their own numbers — `SYMBI_PROFILE=1` will happily show you where every nanosecond goes.

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
| `newtonian/quirk.py` | `quirk` | Odd-even decoupling — run it with `--solver hllc` and `hllc_lm` and diff |
| `isothermal/kepler.py` | `kepler` | Keplerian disk with a central mass |
| `newtonian/bondi.py` | `bondi` | 3D Bondi accretion onto a sink, with a buffer zone and optional refinement |
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
SIMBI was developed at the Center for Cosmology and Paricle Physics (CCPP) at New York University, and I thank the CCPP group for their support and feedback. I also thanks
the following people for their contributions to the project:
- **Andrew Macfadyen** (NYU) for his mentorship and guidance on the project.
- **Jonathan Zrake** (Clemson University) for his intellectual feedback on the endeavor.
- **Jim Stone** (Institute for Advanced Study) for his feedback on the MHD implementation and pointing me towards the robust conserved to primitive formalism of [Kastaun et al. 2021](https://scixplorer.org/abs/2021PhRvD.103b3018K/abstract).
- **Romain Teyssier** (Princeton University) for his willingness to talk shop with me. Especially during my embarkment towards mesh refinement.
---

## Important Works For Folks Wanting to Learn More About the Physics and Numerics
- **[Marti & Muller 1994](https://ui.adsabs.harvard.edu/abs/2003LRR.....6....7M/abstract)**: The original relativistic shock tube paper. This is the paper that started it all for me. I was a grad student at the time and I was trying to understand how to do relativistic hydrodynamics. This paper was a great starting point!
- **[Font 2007](https://ui.adsabs.harvard.edu/abs/2008LRR....11....7F/abstract)**: A review of numerical methods for relativistic (magento)-hydrodynamics. This paper is a great resource for understanding the different numerical methods that can be used to solve the equations of GENERAL relativistic hydrodynamics.
- **[Andersson & Comer 2021](https://link.springer.com/article/10.1007/s41114-021-00031-6)**: A modern review on the general field of relavistic fluid dynamics. Super nice stuff with a great into. Please check it out! 
- **[Moseley et al. 2026](https://arxiv.org/abs/2604.23041)**: A modern technique for evolving Lagrangian tracer particles in an Eulerian fluid simulation. This is a very nice paper that I found to be very useful for my own work. Plus, I got to learna about it live from Romain Teyssier himself!
---

<div align="center">

**[Report a Bug](https://github.com/EigenDev/simbi/issues) · [Request a Feature](https://github.com/EigenDev/simbi/issues)**

</div>

---

> Porting this to rust from c++ benefitted greatly from the use of the Claude Code tool.
On a finite postdoc clock, the developmental speed boost was plain. Sigh, drinking the koolaid sparingly...
