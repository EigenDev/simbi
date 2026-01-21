# marassa restructure complete

## what happened

renamed `rusti` → `marassa` and reorganized from ad-hoc rust conventions to professional structure mirroring the battle-tested c++ codebase.

## directory structure

```
marassa/
├── base/                   # pure numerical methods (no device awareness)
│   └── src/
│       ├── lib.rs
│       └── stencil.rs      # compile-time pattern generation
│
├── xpu/                    # hardware abstraction
│   ├── core/               # Device, Buffer, Kernel traits
│   ├── host/               # cpu implementation
│   ├── cuda/               # cuda implementation (stub)
│   └── metal/              # metal implementation (stub)
│
├── compute/                # functional computation layer
│   ├── src/
│   │   ├── domain.rs       # index spaces
│   │   ├── field.rs        # data containers
│   │   ├── computation.rs  # lazy expression graphs
│   │   └── execution.rs    # binding to devices
│   └── examples/           # 1d euler, godunov workflow
│
├── physics/                # physics solvers
│   └── src/
│       └── hydro/
│           ├── eos.rs      # equations of state
│           └── riemann.rs  # riemann solvers (stub)
│
└── cpp_src/                # reference c++ implementation
```

## dependency graph

```
physics → compute → xpu/core
       → base

compute → base
       → xpu/core
       → xpu/host

xpu/host  → xpu/core
xpu/cuda  → xpu/core
xpu/metal → xpu/core

base → (nothing - pure math)
```

## key changes

1. **dropped verbose prefixes**: `rusti-xpu` → `xpu`, `rusti-math` → `compute`
2. **mirrors c++ structure**: `cpp_src/base` ↔ `base/`, `cpp_src/xpu` ↔ `xpu/`, etc.
3. **workspace standardization**: unified versioning, authors, edition across all crates
4. **clean separation**: hardware (xpu) / math (base) / computation (compute) / physics

## what works

```bash
$ cargo test --workspace --lib
```

all tests pass:
- `base`: 13 tests (stencil pattern generation)
- `xpu-core`: 44 tests (view, reduce, device traits)
- `xpu-host`: 21 tests (cpu device implementation)
- `compute`: 41 tests (domain, field, lazy computation)
- `physics`: 6 tests (eos functions)

## what's new

### base crate (stencil foundation)

compile-time stencil pattern generation for finite-volume methods:

```rust
use base::{Reconstruction, left_pattern, right_pattern, stencil_size};

// compile-time pattern for plm reconstruction in x-direction
const PATTERN: [[i64; 2]; 3] = left_pattern(Reconstruction::PLM, 0);
// produces: [[-2, 0], [-1, 0], [0, 0]]
```

supports:
- **pcm**: piecewise constant (first-order, stencil size 1)
- **plm**: piecewise linear (second-order, stencil size 3)

patterns are `const fn` - **zero runtime cost**, computed at compile time.

### physics crate (equations of state)

ideal gas eos implementation:

```rust
use physics::hydro::{ideal_gas_pressure, ideal_gas_sound_speed};

let p = ideal_gas_pressure(rho, epsilon, gamma);
let c = ideal_gas_sound_speed(rho, p, gamma);
```

## next steps

1. **stencil view**: neighbor gathering using patterns from `base`
2. **reconstruction**: pcm/plm reconstruction functions
3. **riemann solvers**: hlle, hllc implementations
4. **stencil operators**: `Field::stencil_map()` for cfd kernels
5. **multi-gpu**: partition, halo exchange

## build commands

```bash
# build everything
cargo build --workspace

# test everything
cargo test --workspace

# build specific crate
cargo build -p base
cargo build -p xpu-core
cargo build -p compute
cargo build -p physics

# run example
cargo run --example euler_1d_hlle
```

## crate naming convention

**package name** (kebab-case): `xpu-core`, `xpu-host`  
**lib name** (snake_case): `xpu_core`, `xpu_host`

use in code: `use xpu_core::Device;`  
specify in deps: `xpu-core = { workspace = true }`

## performance profile

```toml
[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
panic = "abort"
```

thin lto for fast compile times with good optimization.  
single codegen unit for maximum cross-crate inlining.

## production grade

- workspace-level version management
- consistent edition (2021)
- unified dependency versions
- proper feature flags
- no dead code warnings (allowed in dev)
- all tests passing
- mirrors proven c++ architecture

ready to implement stencil operators and riemann solvers.