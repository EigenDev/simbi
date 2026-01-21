# marassa development roadmap

## mission
build a mathematically sophisticated, game-engine-inspired computational physics framework in rust. multi-gpu, multi-node, zero-cost abstractions. no fluff, straight performance and safety.

---

## architecture philosophy

**partition = actor**
- each gpu gets partitions (not individual cells)
- systems operate on partitions
- stencils gather neighbors within partition
- halo exchange between partitions

**functional core**
- lazy computation graphs (no execution until .with(executor))
- pure functions compose into complex operations
- domain algebra for safe slicing

**compile-time dispatch**
- regime × geometry × solver × reconstruction → concrete type
- no vtables in hot path
- JSON config routes to monomorphized simulation

---

## current status (january 2026)

### ✓ completed

**base/** - numerical methods (no device awareness)
- stencil pattern generation (PCM, PLM)
- compile-time const fn patterns
- 13 passing tests

**xpu/core/** - device abstraction traits
- Device, Buffer, Kernel traits
- View system (1D, 2D, 3D)
- Reduce operations
- 44 passing tests

**xpu/host/** - cpu implementation
- CpuDevice, ParCpuDevice (rayon)
- 21 passing tests

**xpu/cuda/** - stub implementation
- ready for cuda kernel integration

**xpu/metal/** - stub implementation  
- ready for metal compute shaders

**physics/hydro/eos.rs** - ideal gas equation of state
- pressure, sound speed, specific energy
- 6 passing tests

**compute/** - partial implementation
- Computation<T, N, F> lazy graph
- map, zip, remap combinators
- domain tracking

---

## phase 1: foundation (4 weeks)

**goal: 1d newtonian euler working end-to-end**

### week 1: compute layer enhancement
- [ ] Field<T, D, N> with view/slice
- [ ] FieldView for zero-copy access
- [ ] Executor binding: comp.with(&executor)
- [ ] Materialization: lazy → eager
- [ ] Tests: field operations, slicing

### week 2: stencil operators
- [ ] StencilView<T, N, Rec> for neighbor access
- [ ] reconstruct_left/right functions (PCM, PLM)
- [ ] Field::stencil_map() combinator
- [ ] Tests: 1d stencil access, plm reconstruction

### week 3: riemann solvers
- [ ] physics/hydro/hlle.rs - HLLE flux
- [ ] physics/hydro/state.rs - Primitive/Conserved structs
- [ ] physics/hydro/conversion.rs - prim ↔ cons
- [ ] Tests: sod shock tube exact solution comparison

### week 4: config + dispatch
- [ ] config/ crate - JSON schema (serde)
- [ ] dispatch/ crate - compile-time type routing
- [ ] cli/ binary - standalone executable
- [ ] Example: sod shock tube from JSON

**deliverable:**
```bash
cat > sod_1d.json << EOF
{
  "regime": "newtonian",
  "geometry": "cartesian",
  "solver": "hlle",
  "reconstruction": "plm",
  "dims": 1,
  "mesh": {"nx": [1000], "x_min": [0.0], "x_max": [1.0]},
  "physics": {"gamma": 1.4},
  "numerics": {"cfl": 0.4, "t_final": 0.2},
  "initial_conditions": {"type": "sod"}
}
EOF

cargo run --release --bin marassa -- --config sod_1d.json
# outputs: output/sod_1d/density_final.dat
```

---

## phase 2: geometry + 2d/3d (3 weeks)

**goal: spherical coordinates, 2d/3d problems**

### week 5: geometry traits
- [ ] GeometryTrait for coordinate systems
- [ ] CartesianGeometry
- [ ] SphericalGeometry
- [ ] Metric factors, face areas, volumes
- [ ] Tests: coordinate transformations

### week 6: multi-dimensional
- [ ] 2D stencil operations
- [ ] 3D stencil operations
- [ ] Flux divergence in multiple dimensions
- [ ] Conservative update
- [ ] Tests: 2d kelvin-helmholtz, 3d bondi

### week 7: time integration
- [ ] EulerIntegrator (first-order)
- [ ] RK2Integrator (second-order)
- [ ] CFL timestep calculation
- [ ] Tests: convergence order verification

**deliverable:**
```bash
marassa --config bondi_accretion_3d.json
# 3d spherical bondi inflow, steady-state validation
```

---

## phase 3: multi-gpu (4 weeks)

**goal: partition system, halo exchange, scaling**

### week 8-9: partition system
- [ ] Partition<D, N> - executor + domains
- [ ] LevelDecomposition - partition collection
- [ ] Domain decomposition strategies (cartesian split)
- [ ] PartitionTopology - neighbor graph
- [ ] Tests: domain splitting correctness

### week 10-11: halo exchange
- [ ] HaloLink - send/recv pair specification
- [ ] HaloGraph - all exchanges for a level
- [ ] Device-to-device copy (CUDA peer, metal shared)
- [ ] Async overlap: compute while transferring
- [ ] Tests: correctness, bandwidth measurement

**deliverable:**
```bash
marassa --config euler_2d.json --gpus 4
# weak scaling test: 512³ per gpu
```

---

## phase 4: advanced numerics (3 weeks)

**goal: srhd, mhd, higher-order**

### week 12: special relativity
- [ ] SRHD primitive recovery (newton-raphson)
- [ ] Lorentz factors
- [ ] HLLC solver for SRHD
- [ ] Tests: relativistic blast wave

### week 13: magnetohydrodynamics
- [ ] MHD state structs (magnetic field)
- [ ] Staggered B-field layout
- [ ] HLLD solver
- [ ] Constrained transport
- [ ] Tests: alfven wave, orszag-tang vortex

### week 14: higher-order
- [ ] WENO5 reconstruction
- [ ] RK3 time integration
- [ ] Limiters (minmod, van leer)
- [ ] Tests: accuracy order verification

---

## phase 5: i/o + validation (2 weeks)

**goal: checkpoint/restart, output, validation suite**

### week 15: i/o infrastructure
- [ ] HDF5 output (or custom binary format)
- [ ] Checkpoint/restart
- [ ] Diagnostic output (every N steps)
- [ ] Field slicing for analysis
- [ ] Tests: round-trip checkpoint

### week 16: validation suite
- [ ] Sod shock tube (1d)
- [ ] Noh problem (spherical symmetry)
- [ ] Sedov blast wave (self-similar)
- [ ] Kelvin-helmholtz (2d)
- [ ] Bondi accretion (3d spherical)
- [ ] Orszag-Tang vortex (mhd)
- [ ] All pass ✓

**deliverable:**
```bash
./scripts/run_validation_suite.sh
# all tests pass within tolerance
```

---

## phase 6: amr (future - 6 weeks)

**goal: adaptive mesh refinement**

- [ ] Level hierarchy
- [ ] Refinement criteria
- [ ] Flux correction at level boundaries
- [ ] Regridding
- [ ] Time subcycling

---

## phase 7: distributed (future - 4 weeks)

**goal: multi-node via mpi**

- [ ] MPI communicator
- [ ] Rank topology
- [ ] Inter-node halo exchange
- [ ] Load balancing
- [ ] Weak scaling to 1000+ gpus

---

## phase 8: python integration (1 week)

**goal: drop-in replacement for c++ backend**

- [ ] PyO3 thin wrapper
- [ ] runner.py uses rust backend
- [ ] Feature parity with c++ version
- [ ] Performance validation

---

## testing strategy

**unit tests**
- every module has tests/
- property-based testing with proptest
- 90%+ coverage

**integration tests**
- examples/ directory
- each example is a test case
- compare against known solutions

**performance tests**
- benchmarks/ with criterion
- track regression
- profile hot paths

**validation tests**
- scripts/validation/
- classic test problems
- compare against published results

---

## design principles (always)

1. **kiss** - simple beats clever
2. **srp** - one function, one job
3. **functional** - pure where reasonable
4. **zero-cost** - abstractions compile away
5. **soa** - struct of arrays for gpu
6. **explicit** - no magic

---

## success criteria

**correctness**
- all validation tests pass
- bit-identical across devices (where deterministic)
- conservation properties preserved

**performance**
- match or beat c++ version
- 90%+ gpu utilization
- weak scaling efficiency > 0.8

**usability**
- single json config
- sensible defaults
- clear error messages

**maintainability**
- < 20k loc rust (vs 60k c++)
- clear module boundaries
- comprehensive tests

---

## resources needed

**hardware**
- 4x nvidia gpus (testing multi-gpu)
- 1x apple silicon (testing metal)
- hpc cluster access (distributed testing)

**time**
- ~6 months part-time for phases 1-5
- full-time would be ~3 months

**dependencies**
- rayon (cpu parallelism)
- serde (config parsing)
- hdf5-rust or custom binary format
- criterion (benchmarking)
- proptest (property testing)

---

## current focus

**immediate next steps:**
1. finish compute/field.rs
2. implement stencil operators
3. complete hlle solver
4. build dispatch system
5. run first 1d euler simulation

**this week's goal:**
get sod shock tube working, even if slow. correctness first, optimization later.

---

*last updated: january 2026*
*by: marcus + claude*