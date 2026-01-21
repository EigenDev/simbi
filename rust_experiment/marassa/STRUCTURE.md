# marassa workspace structure

mirrors the c++ `src/` directory structure exactly.

## workspace layout

```
marassa/
├── Cargo.toml              # workspace root
├── Cargo.lock
│
├── base/                   # numerical methods (stencils, reconstruction)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── stencil.rs
│       └── reconstruct.rs
│
├── grid/                   # data structures (field, domain, mesh)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── field.rs
│       ├── domain.rs
│       ├── mesh_config.rs
│       ├── decomposition.rs
│       └── skeleton.rs
│
├── compute/                # operations (computation graph, cfd)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── computation.rs
│       ├── cfd.rs
│       └── numerics.rs
│
├── xpu/                    # device abstraction
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── core/
│       ├── execution/
│       └── device/
│
├── physics/                # riemann solvers, eos
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── hydro/
│       ├── eos/
│       └── ib/
│
├── containers/             # vector_t, state structs
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       └── vector.rs
│
├── geometry/               # coordinate systems, metrics
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       └── metrics.rs
│
├── ecs/                    # simulation, components, systems
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── simulation.rs
│       └── components.rs
│
├── functional/             # fp utilities (compose, zip, etc)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       └── fp.rs
│
├── context/                # timing, progress, checkpoint
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       └── timing.rs
│
├── io/                     # checkpoint, serialization
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       └── checkpoint.rs
│
├── traits/                 # common trait definitions
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs
│
├── utility/                # enums, helpers
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       └── enums.rs
│
├── dispatch/               # compile-time type routing
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       └── dispatcher.rs
│
├── config/                 # json schema, validation
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       └── schema.rs
│
└── cli/                    # binary entry point
    ├── Cargo.toml
    └── src/
        └── main.rs
```

## dependency graph

```
cli → dispatch → config
    → ecs → grid → xpu
          → physics → base
          → compute → grid
                    → functional
```

## status

- ✓ base/ - implemented (stencils)
- ✓ xpu/ - partially implemented (traits, cpu)
- ✓ physics/ - stub (eos only)
- ✓ compute/ - partial (computation graph)
- → grid/ - next to implement
- → containers/ - needed for vector_t
- → dispatch/ - needed for type routing
- → config/ - needed for json input
- → cli/ - final binary

## next steps

1. create Cargo.toml for each crate
2. create lib.rs stubs
3. move existing code to correct locations
4. fill in missing implementations