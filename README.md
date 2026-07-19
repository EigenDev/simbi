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

**[Quick Start](#quick-start) · [Installation](#installation) · [Usage](#usage) · [Publications](#publications)**

</div>

---

## Overview

SIMBI is a finite volume code for astrophysical fluid simulations. If you want to throw relativistic jets, shock tubes, stellar explosions, or magnetized turbulence at a grid and see what happens, this is the tool. Results from SIMBI have shown up in *The Astrophysical Journal* and *The Astrophysical Journal Letters*, covering relativistic jets, shock morphology, and stellar explosions.

A quick note on what this is these days: SIMBI started life as a C++ code and was rewritten from the ground up in Rust. I emabarked on this because I got super interested in graph theory and how it can be used to help generate architecture-agnostic code, and I wanted to give it a shot from a clean slate in rust(see [Sethi & Ulman 1970](https://dl.acm.org/doi/10.1145/321607.321620) and works that cite them. Super interesting and deep stuff imo.) The physics is the same, the speed got better (because I became a better progrtammer over the years from the wee ol' monolithic-to-the-max days), and the codebase is a lot easier to live in. You drive the whole thing from Python, so you never have to touch the Rust unless you want to.

**What you get:**
- Six fluid regimes in one code: Newtonian hydro, relativistic hydro (RHD), Newtonian and relativistic MHD, plus isothermal variants of both
- Spacetime as its own axis: hand the relativistic regimes a Minkowski, Schwarzschild, or horizon-penetrating Kerr-Schild metric the same way you would pick a coordinate system
- GPU acceleration on NVIDIA cards, with kernels compiled on the fly so there is no separate build step and no architecture flag to remember
- High-resolution shock capturing with HLLE, HLLC (plus a low-Mach variant, see [Fleischmann et al. 2020](https://www.sciencedirect.com/science/article/pii/S0021999120305362)), and HLLD Riemann solvers, backed by a first-order flux-correction safety net that logs every cell it touches (still working on strengthening this. Maybe I'll switch to [Zalesak and friends](https://apps.dtic.mil/sti/tr/pdf/ADA360122.pdf) at some point).
- Constrained-transport MHD (contact [Gardiner & Stone](https://arxiv.org/abs/0712.2634) or UCT ([Mignone & DelZanna (2021)](https://arxiv.org/abs/2004.10542)) edge EMFs) that keeps div B at machine zero by construction
- Physical transport when you want it: Navier-Stokes viscosity (constant or alpha-disk) and Ohmic resistivity, layered on top of the ideal solvers
- A rich immersed-boundary method: point-mass gravity, Bondi-Hoyle accretion sinks, and rigid walls of *any* constructive solid geometry (CSG) shape (spheres, boxes, unions/intersections) with no-penetration / no-slip porous surfaces. Bodies can be one-way (prescribed motion) or **two-way coupled** — the gas reaction force and torque then drive the body's full rigid-body dynamics: translation, plus rotation about an arbitrary axis via Euler's equations with a principal-moment inertia tensor, so an asymmetric body tumbles and precesses. Spinning walls drag the gas at omega x r, the gas <-> body energy exchange conserves total energy, and every body reports force / torque / accretion diagnostics. I took [Chuck Peskin](https://en.wikipedia.org/wiki/Charles_S._Peskin)'s class during my time as a graduate student at NYU,
and I loved it so much that I thought it'd be cool to introduce it into my code! 
- Horizon excision for GR accretion: on a horizon-penetrating Kerr-Schild chart the region inside the black hole is excised and refilled by a causal one-way onion sweep, so you can swallow the singularity and still keep a well-posed accretion-rate certificate
- Block-based static mesh refinement with [Berger-Colella](https://www.sciencedirect.com/science/article/pii/0021999189900351) subcycling
- Single-node **multi-GPU domain decomposition** — set `gpus > 1` and the domain splits across the cards, halo-exchanged in lockstep and bit-identical to a monolithic run
- Afterglow radiation transport, so you can turn a simulation into synthetic observables
- A live terminal dashboard while you run (pause, single-step, checkpoint on demand, field heatmaps), and `simbi attach` to peek at a headless run from another shell
- A type-safe Python config system that generates its own CLI, so you stop hand-writing argument parsers

An AMD/HIP backend is in the tree as an experimental build feature (the compute layer is backend-agnostic, so the CUDA and HIP paths share one kernel definition). Multi-*node* decomposition is the next step on the roadmap; single-node multi-GPU already runs today.
I don't have a science problem that needs it at this time, so it prob won't be added for 
quite a while.
---

## Simulation Gallery

<div align="center">

| Relativistic Jet Evolution | Relativistic Shock Tube | Rayleigh-Taylor Instability |
|:---:|:---:|:---:|
| [Animation](https://user-images.githubusercontent.com/29236993/145315802-c8d7e8c5-7beb-488c-b496-b9edf404be2e.mp4) | [Animation](https://user-images.githubusercontent.com/29236993/212521070-0e2a7ced-cd5f-4006-9039-be67f174fb07.mp4) | [Animation](https://github.com/EigenDev/simbi/assets/29236993/818d930d-d993-4e5d-8ed4-47a9bae11a7f) |

| Moving Mesh Techniques | Magnetic Turbulence |
|:---:|:---:|
| [Animation](https://user-images.githubusercontent.com/29236993/205418982-943af187-8ae3-4401-92d5-e09a4ea821e2.mp4) | [Animation](https://github.com/user-attachments/assets/9e5b8c42-ce3e-4c23-a380-7903eec52b92) |

</div>

---

## Quick Start

If you only read one section, read this one. We lean hard on [uv](https://docs.astral.sh/uv/) here, and you should too (Not really. Just my strong recommendation is all). It is an absurdly fast Python package manager and environment tool, it replaces pip and venv and conda in one binary (though conda is objectively broader and more powerful. It's just too ""powerful" for my use cases these days), and it makes the whole setup a two-line affair.

Do not have uv yet? Grab it:

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
uv run simbi plot data/1000.chkpt.000_400.h5 --setup "Marti & Muller Problem 1" --field rho v p
```
You can also save a bunch of time by doing `source .venv/bin/activate` and you'll remain in the 
simbi environment you created when you ran `uv venv`. 

Got an NVIDIA card? The GPU build adds the CUDA feature, so it goes through the project helper:

```bash
./dev.py install --gpu
uv run simbi run marti-muller --mode gpu --resolution 1024
```

That is the whole thing. No CMake, no Ninja, no compiler environment variables to babysit.
(Though, I am a big fan of those tools! I learned so much about programming and developing
a major project like this from utilizing those tools. It just becomes a bit much for me to
deal with as I explore more architectures and directions. Cargo is standard enough for my 
purposes these days. :D)

---

## Installation

### What you need

- A Rust toolchain (cargo), the easy way is [rustup](https://rustup.rs)
- Python 3.10 or newer
- HDF5 (it gets linked into the extension)
- For GPU, the NVIDIA driver, which gives you `libcuda` and `libnvrtc`
- On Linux, `patchelf` (uv and maturin will tell you if it is missing)

That is it. You do not need `nvcc`. Kernels are compiled at runtime with NVRTC, so the GPU build figures out your card's architecture on its own.

### The uv way (recommended)

Seriously, just use uv. It builds the Rust extension through maturin behind the scenes and you never think about it again.

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
uv run simbi run sedov --mode cpu --resolution 256
```

Just want the `simbi` command on your PATH without thinking about environments? Since it is a CLI tool, this is the slick option:

```bash
uv tool install .
```

Now `simbi` works from anywhere, no `uv run` prefix needed.

### Working on SIMBI itself?

If you are hacking on the code rather than just running it, that is when you want an editable install. Use the project helper, which runs `maturin develop` so the Rust extension gets rebuilt in place:

```bash
./dev.py install            # editable, rebuilds the rust backend
./dev.py install --gpu      # same, with the cuda feature
```

Plain `uv pip install -e .` works too for the Python side, but it will not recompile the Rust on its own, so `./dev.py install` is the better contributor loop.

### GPU builds

The GPU path needs the cargo `cuda` feature turned on, so it goes through `dev.py`, which is a thin wrapper around maturin:

```bash
./dev.py install --gpu
```

The CPU and GPU extensions coexist as separate modules (`cpu_ext` and `gpu_ext`), so you can keep both installed and pick the backend at run time with `--mode cpu`, `--mode omp` (OpenMP-threaded CPU, size it with `--nthreads N`), or `--mode gpu`.

### Cleaning up

```bash
./dev.py clean --all     # drop the built extensions and run cargo clean
./dev.py install --gpu   # rebuild from scratch
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
simbi run simbi_configs/examples/kh.py --mode cpu --resolution 512

# pick up where a previous run left off
simbi run <problem> --checkpoint data/checkpoint.h5
```

The CLI tries to be a good roommate: config names match with or without kebab-case, a
typo gets a "did you mean...?", and if two configs in different directories share a name,
it lists both and asks.

**Options you will reach for:**
- `--mode cpu|gpu` sets the execution backend
- `--resolution N`, `--resolution N,M`, or `--resolution N,M,K` sets the grid. Comma is required for 2D and 3D.
- `--adiabatic-index` is the ratio of specific heats
- `--end-time` is when to stop
- `--data-directory` is where the output goes
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
simbi plot data/checkpoint.h5 --setup "Problem Name" --field rho v p

# include immersed body diagnostics
simbi plot data/checkpoint.h5 --bodies

# stitch a stack of checkpoints into an animation
simbi plot data/*.h5 --animate --field rho

# get a starter config to customize
simbi plot --generate-config
```

### Afterglow analysis

Turn hydro snapshots into synthetic observables:

```bash
# build a photon event catalog from the snapshots
simbi afterglow generate data/*.h5 --output events.h5 --max-events 1000000

# observer lightcurve
simbi afterglow lightcurve events.h5 --observer-angle 0.1 --frequencies 1e9 1e14 1e18

# sky intensity map
simbi afterglow skymap events.h5 --observer-time 1e5

# polarization evolution
simbi afterglow polarization events.h5 --observer-angle 0.1

# spectrum
simbi afterglow spectrum events.h5 --observer-time 1e5

# sweep the observer time into a sky-map movie
simbi afterglow movie events.h5 --output skymap.mp4
```

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

### Source terms

Add gravity or custom hydro sources as expression graphs:

```python
import simbi

@computed_field
@property
def gravity_source_expressions(self):
    graph = simbi.Expr.Graph()
    x_comp = simbi.Expr.constant(0.0, graph)
    y_comp = simbi.Expr.constant(-0.1, graph)  # constant downward gravity
    terms = graph.compile([x_comp, y_comp])
    return terms.serialize()
```

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
| `SRMHD` | Relativistic magnetohydrodynamics | AGN jets, pulsar wind nebulae, magnetic reconnection |

(`SRHD` still works as an alias for `RHD` — old configs keep running.)

### Spacetimes

Relativity in SIMBI lives in the *spacetime*: you select it by layering a `Spacetime` under a
relativistic regime, and the geometry supplies the metric the fluid evolves on:

- `MINKOWSKI` — flat, i.e. plain special relativity
- `SCHWARZSCHILD` — a static central mass
- `KERR_SCHILD` — horizon-penetrating coordinates; gas crosses r = 2M without drama
- `KERR` — a spinning black hole

So "GR hydro around a Kerr black hole" is the same `RHD` regime you already know,
handed a different metric.

### Horizon excision

Point a horizon-penetrating chart (`KERR_SCHILD` or `KERR`) at a black hole and you can actually
*swallow* it. Set `excision_radius` to a radius inside the horizon (above the metric-guard radius
M/2, so around 0.7 r_+) and the cells inside are excised every step: their primitives are
overwritten by a causal, outward one-way "onion" sweep and their conserved state is rebuilt from
the local metric, so nothing unphysical leaks back out across the horizon and the accretion-rate
certificate stays well-posed — no coordinate singularity to babysit. It works for hydro and MHD
(the staggered magnetic faces stay constrained-transport-owned), for spinning (`KERR`) horizons,
on the GPU, and across the multi-GPU decomposed path.

### Coordinate systems

- `CARTESIAN`, the usual x, y, z
- `SPHERICAL`, r, theta, phi
- `CYLINDRICAL`, r, phi, z
- `AXIS_CYLINDRICAL`, cylindrical with axis symmetry
- `PLANAR_CYLINDRICAL`, 2D cylindrical in the r-phi plane

### Numerical methods

**Riemann solvers:**
- `HLLE`, the two-wave workhorse, written in a branch-free closed form the compiler can vectorize
- `HLLC`, HLL with a contact wave (hydrodynamics), Toro's adaptive pressure estimates evaluated lazily — a smooth cell never pays for the shock estimate
- `HLLC_LM`, the Fleischmann (2020) low-Mach / low-dissipation HLLC
- `HLLD`, HLL with discontinuities (magnetohydrodynamics), faithful to Mignone & Del Zanna

**Grid spacing:**
- `LINEAR`, uniform spacing
- `LOGARITHMIC`, log spacing, handy for spherical setups

**Boundary conditions:**
- `PERIODIC`, wrap around
- `REFLECTING`, mirror symmetry
- `OUTFLOW`, zero gradient
- `DYNAMIC`, user-defined expressions

**Time integration:**
- `EULER`, forward Euler
- `RK2`, second-order SSP Runge-Kutta (Berger-Colella subcycling under refinement)
- `RK3`, third-order SSP Runge-Kutta

**Constrained transport (MHD):**
- `CONTACT`, Gardiner & Stone (2005) edge EMFs (default)
- `UCT`, Del Zanna / Mignone & Del Zanna upwind CT (kills the checkerboard mode)

Either way, div B stays at machine zero by construction — the curl-of-EMF update carries
a symbolic proof of div(curl) = 0 in the test suite, and bug-injection tests keep the
proof honest.

**A few extras:**
- `plm_theta`, the PLM reconstruction parameter (0 to 2, default 1.5; 0 gives you piecewise-constant)
- `use_quirk_smoothing`, [Quirk 1994](https://onlinelibrary.wiley.com/doi/abs/10.1002/fld.1650180603)'s carbuncle fix
- First-order flux correction (FOFC): if a high-order update drives a cell unphysical, that cell is redone at first order, and the run reports how often that happened — per window while it runs, and again in the exit summary
- Prolongation at refinement boundaries runs one order above the interior reconstruction, which preserves the scheme's accuracy across level edges

### Non-ideal transport

Two dissipative terms sit on top of the ideal solvers, both off by default (coefficient zero):

- `viscosity` — a Navier-Stokes shear viscosity. Give it a constant, or switch on the alpha-disk law (nu ~ alpha c_s H) for accretion-disk setups
- `resistivity` — Ohmic resistivity for the MHD regimes; the field diffuses while constrained transport keeps div B at machine zero

### Static mesh refinement

```python
# turn refinement on
refinement_enabled: Annotated[bool, ProblemParam(True)]
refinement_max_levels: Annotated[int, ProblemParam(3)]
refinement_regions: Annotated[
    list[list[tuple[float, float]]],
    ProblemParam([[(-0.1, 0.1), (-0.1, 0.1)], [(-0.05, 0.05), (-0.05, 0.05)]]),
]
refinement_ratios: Annotated[list[int], ProblemParam([2, 2])]
refinement_subcycling_mode: Annotated[
    SubCycleMode, ProblemParam(SubCycleMode.STANDARD)
]
```

**Subcycling modes:**
- `NONE`, every level advances on the same timestep
- `STANDARD`, subcycle by the refinement ratio
- `MANUAL`, you specify substeps per level

---

## Architecture

Here is the quick tour of how the Rust side fits together, in case you want to hack on it.

The compute backend is a Cargo workspace of small, focused crates. The interesting idea at the center of it: the physics is written ONCE, generically over a "carrier" type, and traced into an intermediate representation. That IR gets lowered to native CPU code (compiled by LLVM at build time), CUDA source (compiled at run time with NVRTC), or — for the source terms you write in Python — machine code JIT-compiled at startup with Cranelift. One definition of the math serves every backend. The same trick powers the test suite: the f64 evaluation of a kernel doubles as its own oracle, so CPU, GPU, and JIT are checked against each other bit-for-bit.

The compiler layer earns its keep: common-subexpression elimination, constant-power strength reduction (`r ** -2` in your config compiles down to two multiplies), automatic lazy scheduling of expensive conditional branches (a `where(...)` in your source expressions becomes a real branch when the arms are worth skipping), and a cost-gated select-vectorization pass that turns branch-free kernel bodies into NEON/SIMD-friendly straight-line code. The guiding rule, enforced by the graph itself: only compute what you actually need.

A few load-bearing pieces:

- **`symbi-ir`** holds the kernel IR, the graph passes, and the code generators (CPU and CUDA share one renderer)
- **`symbi-hydro`** is the physics: regimes, equations of state, and the Riemann solvers
- **`symbi-jit`** is the Cranelift JIT for runtime-authored kernels (your Python source expressions)
- **`symbi-sim`** owns the simulation state and the kernel-native evolution driver
- **`symbi-substrate`** assembles the per-regime kernel sets (flux, c2p, godunov, cfl, ghost fill)
- **`symbi-amr`** is the refinement hierarchy: prolongation, restriction, flux registers, and subcycling
- **`symbi-ib`** is the immersed-body layer: body state, motion, and accretion ledgers
- **`symbi-xpu`** is the device layer: memory, streams, and kernel launches
- **`symbi-afterglow`** does the radiation transport and observables
- **`symbi-py`** is the thin pyo3 bridge that becomes the `cpu_ext` and `gpu_ext` Python modules

A few design choices worth calling out. Fields are stored struct-of-arrays, which is what lets the CPU vectorize and the GPU coalesce its memory reads. The CPU executor fans serial kernels over a cache-blocked cover whose tiles run the full grid row along the contiguous axis, which gives the vectorized kernel bodies the long unit-stride runs they thrive on. And the time step is sequenced entirely through a `KernelSet` trait, so the driver never reaches into the fields directly. That last part is what makes multi-GPU work: a subdomain is just a self-contained simulation state, so `gpus > 1` splits the domain into tiles that halo-exchange between neighbors in lockstep — bit-identical to a monolithic run, and riding the exact halo machinery the refinement hierarchy already uses. Multi-*node* is the natural next step from here.

The neutral IR is precision-agnostic too: the same traced graph renders to f64 or f32 at the target's launch precision (an f32 device run just halves the bandwidth bill), and the Cranelift runtime path is generic over the scalar the same way. The device backend is written against a backend-agnostic trait, so the CUDA and HIP (AMD, experimental) paths share one kernel definition and diverge only in the tiny token-map at the very bottom.

On speed (one machine, one problem class, double precision): the 3D Newtonian linear wave at 256^3 sustains ~38 million zone-cycles per second on an 8-performance-core Apple M4 Pro laptop, and a 2D Kelvin-Helmholtz with HLLE runs around 70. For a sense of scale, [AthenaK](https://github.com/IAS-Astrophysics/athenak) reports 34 Mzc/s for the same class of test on an M1 Pro. Your problems will have their own numbers — `SYMBI_PROFILE=1` will happily show you where every nanosecond goes.

---

## Example Configurations

There are 60-odd ready-to-run configs in `simbi_configs/examples/`. A sampler:

The `Run with` column is the slug you pass to `simbi run` (the file stem with underscores swapped
for dashes; underscores also work).

| Example | Run with | What it is |
|---------|----------|------------|
| `sod.py` | `sod` | Newtonian shock tube |
| `marti_muller.py` | `marti-muller` | SRHD shock tube (1D and 3D variants) |
| `kh.py` | `kh` | Kelvin-Helmholtz instability |
| `rt.py` | `rt` | Rayleigh-Taylor instability (with gravity) |
| `sedov.py` | `sedov` | Sedov-Taylor explosion (spherical) |
| `thermal_bomb.py` | `thermal-bomb` | Thermal bomb (2D and 3D variants) |
| `magnetic_blast.py` | `magnetic-blast` | MHD blast wave |
| `magnetic_shock_tube.py` | `magnetic-shock-tube` | 1D MHD shock |
| `rmhd_orszag_tang.py` | `rmhd-orszag-tang` | SRMHD Orszag-Tang vortex (Newtonian, isothermal, and resistive variants also ship) |
| `kepler.py` | `kepler` | Keplerian disk with a central mass |
| `bondi.py` | `bondi` | 3D Bondi accretion onto a sink, with a buffer zone and optional refinement |
| `uniform_sphere.py` | `uniform-sphere` | Uniform sphere with homologous mesh expansion |
| `quad_shocktube.py` | `quad-shocktube` | 2D multi-region shock |

Run any of them:

```bash
uv run simbi run sedov --mode gpu --resolution 256
uv run simbi run kepler --mode cpu --resolution 128 128
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
| **v0.9.0** (current) | Full rewrite of the compute backend from C++ to Rust (traced-IR kernels, LLVM + NVRTC + Cranelift); GR spacetimes + horizon excision; constrained-transport MHD; viscosity + resistivity; two-way rigid-body immersed walls (CSG shapes, spin, energy-conserving coupling); single-node multi-GPU domain decomposition; static mesh refinement; live TUI; the big performance campaign |
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

When something will not install, check the basics first:

```bash
python --version   # want 3.10 or newer
cargo --version    # the rust toolchain is present
nvidia-smi         # the GPU and driver are visible (NVIDIA)
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

<div align="center">

**[Report a Bug](https://github.com/EigenDev/simbi/issues) · [Request a Feature](https://github.com/EigenDev/simbi/issues)**

</div>

---

> Porting this to rust from c++ benefitted greatly from the use of the Claude Code tool.
The speed boost was just undeniable versus if I had taken the time to do it myself (during my finite postdoc). Sigh, drinking the koolaid sparingly...
