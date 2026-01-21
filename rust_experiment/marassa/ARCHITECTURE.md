# marassa architecture

pure functional, device-agnostic godunov code built on mathematical foundations.

---

## philosophy

**simplicity over complexity**
- delete code before adding code
- one memory layout (soa), not two
- pure functions, explicit state
- mathematical clarity: Φ: state → state

**compile-time over runtime**
- monomorphization for zero overhead
- type-level physics selection
- device type as generic parameter
- all variants compiled ahead of time

**functional over imperative**
- lazy computation graphs
- composable operations
- no hidden mutations
- category-theoretic abstractions

---

## architecture layers

```
┌─────────────────────────────────────────┐
│  dispatch (compile-time variant gen)    │  ← python config → rust binary selection
├─────────────────────────────────────────┤
│  sim (world state)                      │  ← the god object
│    - WorldState<R, S, D, RANK>          │
│    - PartitionState (soa fields)        │
│    - HaloGraph (communication)          │
├─────────────────────────────────────────┤
│  physics (regime-specific logic)        │  ← mathematical operators
│    - Primitive<R, RANK> (point types)   │
│    - riemann solvers (trait Solver)     │
│    - eos (equation of state)            │
├─────────────────────────────────────────┤
│  compute (lazy expressions)             │  ← functional programming layer
│    - Computation<T, N, F>               │
│    - Domain<N> (pure topology)          │
│    - Field<'d, T, D, N>                 │
│    - stencil operations                 │
├─────────────────────────────────────────┤
│  xpu (device abstraction)               │  ← hardware backends
│    - trait Device                       │
│    - CpuDevice (host, rayon)            │
│    - MetalDevice (apple silicon)        │
│    - CudaDevice (nvidia)                │
└─────────────────────────────────────────┘
```

---

## mathematical structure

### godunov operator

```
Φ: State → State

Φ = boundary ∘ update ∘ flux ∘ reconstruct

where:
  reconstruct: U → (U_L, U_R)  (spatial operator)
  flux:        (U_L, U_R) → F  (riemann solver)
  update:      (U, F) → U'     (hyperbolic pde)
  boundary:    U → U           (ghost zone fill)
```

### state manifold

```
State = ⊕ᵢ Partition(Dᵢ)  (direct sum over devices)

Partition = {
  den: Field<f64, D, RANK>,  (density)
  mom: [Field<f64, D, RANK>; RANK],  (momentum)
  nrg: Field<f64, D, RANK>,  (energy)
}
```

fields are functors: `Domain<N> → DeviceMemory<T>`

### field operations (category theory)

```rust
// natural transformations
field.map(f: T → U): Field<T> → Field<U>
field.zip(other): Field<T> × Field<U> → Field<(T,U)>

// stencil = neighborhood functor
stencil.apply(field): Field<T> → Field<U>

// reduction = fold
field.fold(init, op): Field<T> → T
```

---

## key types

### world state (the god object)

```rust
pub struct WorldState<'d, R, S, D: Device, const RANK: usize> {
    partitions: Vec<PartitionState<'d, R, D, RANK>>,
    halo_graph: HaloGraph<RANK>,
    config: PhysicsConfig<RANK>,
    time: f64,
    _solver: PhantomData<S>,
}

impl WorldState<'_, R, S, D, RANK> {
    pub fn step(&mut self, dt: f64) -> Result<(), D::Error> {
        self.halo_graph.exchange(&mut self.partitions)?;
        R::step::<S, D>(&mut self.partitions, dt, &self.config)?;
        self.time += dt;
        Ok(())
    }
}
```

### partition state (soa on device)

```rust
pub struct PartitionState<'d, R, D: Device, const RANK: usize> {
    den: Field<'d, f64, D, RANK>,
    mom: [Field<'d, f64, D, RANK>; RANK],
    nrg: Field<'d, f64, D, RANK>,
    domain: Domain<RANK>,
    device: &'d D,
    _regime: PhantomData<R>,
}
```

### point types (kernel-local)

```rust
pub struct Primitive<R: Regime, const RANK: usize> {
    rho: f64,
    vel: [f64; RANK],
    p: f64,
    _regime: PhantomData<R>,
}
```

point types used only in kernel logic, never for storage.

---

## memory layout

### soa (structure of arrays) - always

```
storage:
  den: [ρ₀, ρ₁, ρ₂, ρ₃, ...]
  mom: [[vₓ₀, vₓ₁, vₓ₂, ...],  // momentum components
        [vᵧ₀, vᵧ₁, vᵧ₂, ...]]
  nrg: [E₀, E₁, E₂, E₃, ...]

kernel access:
  for each cell i:
    prim = load_point(den[i], mom[*][i], nrg[i])  // aos → local
    cons = riemann_solve(prim_L, prim_R)
    store_point(i, cons)  // local → aos
```

**why soa?**
- gpu memory coalescing (adjacent threads access adjacent memory)
- simd vectorization on cpu
- cache-friendly on modern hardware
- simpler than managing both layouts

---

## compile-time dispatch

### problem

user configures simulation via python:
```json
{
  "regime": "srhd",
  "solver": "hlle", 
  "reconstruction": "plm",
  "dimension": 2
}
```

need runtime selection but want compile-time optimization.

### solution

pre-compile all variants, select at runtime:

```rust
pub fn dispatch(config: Config) -> Result<(), Error> {
    match (config.regime, config.solver, config.dim) {
        (Newtonian, Hlle, 1) => run::<Newtonian, Hlle, 1>(config),
        (Newtonian, Hlle, 2) => run::<Newtonian, Hlle, 2>(config),
        (Srhd, Hlle, 1) => run::<Srhd, Hlle, 1>(config),
        // ... all combinations
    }
}

fn run<R: Regime, S: Solver, const RANK: usize>(config: Config) {
    // fully monomorphized - no vtables, all inlined
    let device = MetalDevice::new(0)?;
    let mut world = WorldState::<R, S, _, RANK>::new(&device, config)?;
    
    while world.time < config.t_final {
        world.step(world.compute_dt(config.cfl)?)?;
    }
}
```

binary contains ~72 variants (4 regimes × 3 solvers × 3 dims × 2 devices).
python selects which one to execute.

---

## device abstraction

### trait Device

```rust
pub trait Device: Sized {
    type Buffer<T>: DeviceBuffer<T>;
    type Error: Debug + Display;
    
    fn alloc<T>(&self, n: usize) -> Result<Self::Buffer<T>, Self::Error>;
    fn launch<K, Args>(&self, kernel: K, config: LaunchConfig, args: Args) 
        -> Result<(), Self::Error>;
    fn copy_to_host<T>(&self, device_buf: &Self::Buffer<T>, host: &mut [T]) 
        -> Result<(), Self::Error>;
    // ...
}
```

### implementations

- **CpuDevice**: rayon for parallelism, host memory
- **MetalDevice**: metal compute pipeline, unified memory on apple silicon
- **CudaDevice**: cuda runtime, explicit transfers (future)

user code is device-agnostic:

```rust
fn algorithm<D: Device>(device: &D, data: &Field<f64, D, 1>) {
    // same code works on cpu, metal, cuda
}
```

---

## multi-gpu strategy

### domain decomposition

```
global domain [0, 1000] split across 4 gpus:

GPU 0: [0, 250]     + ghosts
GPU 1: [250, 500]   + ghosts  
GPU 2: [500, 750]   + ghosts
GPU 3: [750, 1000]  + ghosts
```

### halo exchange

communication graph (hypergraph):

```
nodes = partitions
edges = halo dependencies

GPU 0 → GPU 1: copy [248:250] → [0:2]  (ghost zones)
GPU 1 → GPU 0: copy [250:252] → [248:250]
GPU 1 → GPU 2: copy [498:500] → [0:2]
...
```

execute graph traversal in parallel:
- same-device: device-to-device copy (fast)
- different-device: peer-to-peer transfer (gpu-direct)
- multi-node: mpi (future)

---

## status

### ✓ complete

- [x] xpu-core (device trait)
- [x] xpu-host (cpu backend with rayon)
- [x] xpu-metal (metal backend, device trait impl)
- [x] compute (lazy computation graphs, stencils, reconstruction)
- [x] physics/hydro/state (point types: Primitive, Conserved)
- [x] sim/world (WorldState, PartitionState, HaloGraph)

### → next (phase c: metal kernels)

- [ ] flux computation kernel (hlle riemann solver)
- [ ] reconstruction kernel (plm with limiters)
- [ ] update kernel (finite volume update)
- [ ] boundary kernel (ghost zone fill)
- [ ] metal shader code (.metal files)
- [ ] kernel launchers (rust wrappers)

### → future

- [ ] dispatch system (macro to generate all variants)
- [ ] initial conditions (set_ic implementation)
- [ ] timestep computation (cfl reduction)
- [ ] multi-gpu halo exchange
- [ ] cuda backend
- [ ] mpi for multi-node
- [ ] amr (adaptive mesh refinement)
- [ ] python bindings

---

## design decisions

### why no aos?

deleted. soa is superior for:
- gpu memory coalescing
- simd vectorization
- cache locality
- simpler codebase (one layout, not two)

point types (Primitive, Conserved) used only in kernel logic, never storage.

### why no ecs?

ecs adds complexity for no gain in this use case:
- simulations have fixed schema (den, mom, nrg)
- no dynamic component addition/removal
- domain decomposition is explicit, not query-based

replaced with simple functional state.

### why pure functions?

- easier to reason about (no hidden mutations)
- testable (deterministic)
- parallelizable (no race conditions)
- composable (unix philosophy)

state transitions are explicit: `step(state, dt) → state'`

### why lazy computation?

separation of concerns:
- **what** to compute (Computation graph)
- **where** to compute (Device selection)
- **when** to compute (evaluation trigger)

enables optimization:
- fusion (combine multiple ops into one kernel)
- pruning (eliminate unused branches)
- caching (memoize expensive ops)

---

## performance targets

### baseline (c++ simbi)

- 1d sod shock tube (1000 cells): ~0.5 ms/step (cpu)
- 2d blast wave (512²): ~50 ms/step (cpu)

### goals (rust marassa)

- match or exceed c++ performance
- scale linearly across gpus
- zero overhead abstractions (same as hand-written kernels)

### measurement strategy

- micro-benchmarks (criterion)
- end-to-end comparison (sod, blast wave, etc)
- profiling (instruments on metal, nsight on cuda)

---

## future: distributed

### multi-node with mpi

```rust
struct DistributedWorld<R, S, D: Device, const RANK: usize> {
    local_world: WorldState<R, S, D, RANK>,
    mpi_comm: Communicator,
    neighbor_ranks: Vec<usize>,
}

impl DistributedWorld {
    fn step(&mut self, dt: f64) {
        // 1. local computation
        self.local_world.step(dt)?;
        
        // 2. pack halo data
        let send_buffers = self.pack_halos()?;
        
        // 3. mpi exchange
        mpi::all_to_all(&send_buffers, &mut recv_buffers)?;
        
        // 4. unpack halo data
        self.unpack_halos(recv_buffers)?;
    }
}
```

---

## contact

marcus dupont (md9952@nyu.edu)

built with ❤️ and category theory.