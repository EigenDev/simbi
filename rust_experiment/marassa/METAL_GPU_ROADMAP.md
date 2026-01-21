# Metal GPU Backend Implementation Roadmap

## Current Status

✅ **Serial CPU**: Complete and validated (3.4e7 zone-cycles/sec peak)
✅ **Parallel CPU (Rayon)**: Complete but **not beneficial** (0.1-1.1x, memory-bandwidth limited)
⏳ **Metal GPU**: Infrastructure exists, needs compute kernels

## Why Metal GPU Will Succeed Where Parallel CPU Failed

**Problem**: 1D Euler solver is **memory-bandwidth bound**
- Arithmetic intensity < 1 flop/byte
- Serial CPU: ~50 GB/s memory bandwidth
- Parallel CPU: Same 50 GB/s shared by all threads → contention
- **Metal GPU**: 400-600 GB/s HBM → **8-12x bandwidth advantage**

**Expected Speedup**: 10-50x for grids > 10,000 cells

---

## Phase 1: Metal Compute Kernels (1-2 days)

### Kernel 1: Primitive ↔ Conserved Conversions
**File**: `xpu/metal/kernels/euler_conversions.metal`

```metal
kernel void prim_to_conserved(
    device const float* rho,
    device const float* vx,
    device const float* p,
    device float* cons_rho,
    device float* cons_mom,
    device float* cons_energy,
    constant float& gamma,
    uint gid [[thread_position_in_grid]]
)
```

**Input**: SoA primitive fields (rho, vx, p)
**Output**: SoA conserved fields (rho, mom, energy)
**Complexity**: O(N), trivial parallelism

### Kernel 2: PLM Reconstruction
**File**: `xpu/metal/kernels/plm_reconstruction.metal`

```metal
kernel void plm_reconstruct_left(
    device const float* u,
    device float* u_left,
    constant int& nghost,
    uint gid [[thread_position_in_grid]]
)
```

**Input**: Cell-centered values
**Output**: Reconstructed left/right states at interfaces
**Complexity**: O(N), stencil access (i-2, i-1, i)
**Challenge**: Shared memory for stencil access optimization

### Kernel 3: HLLE Riemann Solver
**File**: `xpu/metal/kernels/hlle_solver.metal`

```metal
kernel void compute_hlle_flux(
    device const float* rho_left,
    device const float* vx_left,
    device const float* p_left,
    device const float* rho_right,
    device const float* vx_right,
    device const float* p_right,
    device float* flux_mass,
    device float* flux_mom,
    device float* flux_energy,
    constant float& gamma,
    uint gid [[thread_position_in_grid]]
)
```

**Input**: Left/right primitive states
**Output**: Numerical fluxes
**Complexity**: O(N), fully parallel
**Core**: Wave speed estimation + HLL formula

### Kernel 4: Flux Update
**File**: `xpu/metal/kernels/flux_update.metal`

```metal
kernel void update_conserved(
    device float* u,
    device const float* flux_left,
    device const float* flux_right,
    constant float& dt,
    constant float& dx,
    constant int& nghost,
    uint gid [[thread_position_in_grid]]
)
```

**Input**: Current state + fluxes
**Output**: Updated state
**Complexity**: O(N), du/dt = -(f[i+1] - f[i])/dx

---

## Phase 2: Rust-Metal Integration (1 day)

### Task 1: Create Metal Device Wrapper
**File**: `physics/src/hydro/metal_backend.rs`

```rust
pub struct MetalEuler1DSolver {
    device: MetalDevice,
    buffers: MetalFieldBuffers,
    kernels: MetalKernels,
    ncells: usize,
    // ...
}
```

**Responsibilities**:
- Allocate GPU buffers for fields
- Compile and cache Metal kernels
- Manage host ↔ device transfers
- Launch compute kernels

### Task 2: Field Buffers (SoA Layout)
**File**: `physics/src/hydro/metal_backend.rs`

```rust
struct MetalFieldBuffers {
    // primitive
    rho: MetalBuffer<f32>,
    vx: MetalBuffer<f32>,
    p: MetalBuffer<f32>,
    
    // conserved
    cons_rho: MetalBuffer<f32>,
    cons_mom: MetalBuffer<f32>,
    cons_energy: MetalBuffer<f32>,
    
    // fluxes
    flux_mass: MetalBuffer<f32>,
    flux_mom: MetalBuffer<f32>,
    flux_energy: MetalBuffer<f32>,
}
```

**Note**: Already optimal for GPU (SoA = coalesced memory access)

### Task 3: Kernel Compilation
**File**: `physics/src/hydro/metal_backend.rs`

```rust
struct MetalKernels {
    prim_to_conserved: metal::Function,
    conserved_to_prim: metal::Function,
    plm_reconstruct: metal::Function,
    hlle_flux: metal::Function,
    flux_update: metal::Function,
}

impl MetalKernels {
    fn compile(device: &MetalDevice) -> Result<Self, MetalError> {
        let library = device.new_library_with_source(KERNEL_SOURCE)?;
        // ...
    }
}
```

---

## Phase 3: Solver Integration (1 day)

### Add Metal Execution Mode
**File**: `physics/src/hydro/solver1d.rs`

```rust
pub enum ExecutionMode {
    Serial,
    ParallelCpu,
    MetalGpu,  // ← NEW
}
```

### Implement Metal Spatial Operator
**File**: `physics/src/hydro/solver1d.rs`

```rust
fn compute_spatial_operator_metal(
    device: &MetalDevice,
    buffers: &MetalFieldBuffers,
    kernels: &MetalKernels,
    gamma: f64,
    dx: f64,
    ncells: usize,
) -> Vec<Conserved1D> {
    // 1. Launch PLM reconstruction kernel
    // 2. Launch HLLE flux kernel
    // 3. Launch flux update kernel
    // 4. Copy results back to host
    // 5. Return updated state
}
```

**Pipeline**:
1. Prim → Cons (GPU)
2. PLM Reconstruct (GPU)
3. HLLE Flux (GPU)
4. Flux Update (GPU)
5. Cons → Prim (GPU)
6. Copy to host (only when needed)

---

## Phase 4: Optimization (1-2 days)

### Optimization 1: Minimize Host ↔ Device Transfers
**Strategy**: Keep all data on GPU, only transfer for I/O
- Initial condition: CPU → GPU (once)
- Time stepping: entirely on GPU
- Output: GPU → CPU (periodic)

**Benefit**: Avoid PCIe bottleneck (16 GB/s vs 400 GB/s HBM)

### Optimization 2: Kernel Fusion
**Current**: 5 separate kernel launches per time step
**Optimized**: 1-2 kernel launches

```metal
kernel void fused_euler_step(
    // All buffers...
) {
    // Reconstruction + Flux + Update in one kernel
    // Reduces kernel launch overhead
    // Better register usage
}
```

**Benefit**: Reduce launch overhead (~5 µs per launch)

### Optimization 3: Shared Memory for Stencils
**Current**: Each thread reads from global memory
**Optimized**: Threadgroup loads tile to shared memory

```metal
kernel void plm_reconstruct_tiled(
    threadgroup float* shared_mem [[threadgroup(0)]],
    // ...
) {
    // Load tile to shared memory
    // Reconstruct from shared memory (faster)
}
```

**Benefit**: 10-100x faster memory access for stencils

### Optimization 4: Half-Precision (f16)
**Current**: f32 everywhere
**Optimized**: f16 for intermediate values

**Trade-off**: 2x throughput vs precision
**Use case**: Large grids where bandwidth dominates

---

## Phase 5: Validation & Benchmarking (1 day)

### Validation Tests
1. **Correctness**: Metal vs Serial (< 1e-5 error)
2. **Conservation**: Mass/momentum/energy conserved
3. **Sod shock tube**: Matches exact solution
4. **Performance**: Zone-cycles/sec measurement

### Performance Targets

| Grid Size | Serial CPU | Expected Metal GPU | Target Speedup |
|-----------|------------|-------------------|----------------|
| 1,000     | 3.2e7      | 1.0e8             | 3x             |
| 10,000    | 2.9e7      | 5.0e8             | 17x            |
| 100,000   | N/A        | 1.0e9             | 30x+           |
| 1,000,000 | N/A        | 2.0e9             | 50x+           |

**Rationale**: Larger grids → better GPU utilization → higher speedup

---

## Implementation Checklist

### Week 1: Kernels
- [ ] Write Metal compute kernels
- [ ] Test kernels individually
- [ ] Validate correctness vs serial
- [ ] Profile kernel performance

### Week 2: Integration
- [ ] Create Metal backend wrapper
- [ ] Integrate with solver
- [ ] Add Metal execution mode
- [ ] End-to-end testing

### Week 3: Optimization
- [ ] Minimize transfers
- [ ] Kernel fusion
- [ ] Shared memory optimization
- [ ] Performance tuning

### Week 4: Production
- [ ] Comprehensive testing
- [ ] Performance benchmarking
- [ ] Documentation
- [ ] Example codes

---

## Expected Outcomes

### Performance
- **10-50x speedup** over serial CPU
- **20-100x speedup** over parallel CPU
- **Scalability** to millions of cells

### Architecture Benefits
1. **Already optimal for GPU**: SoA layout, zero-cost abstractions
2. **Clean separation**: Kernels independent of host code
3. **Type-safe**: Compile-time device selection
4. **Extensible**: Easy to add CUDA/ROCm/Vulkan backends

### Real-World Impact
- Enable **real-time** simulations for interactive exploration
- Scale to **production-size** grids (millions of cells)
- Foundation for **2D/3D** solvers with even larger speedups

---

## Alternative: Start with 2D Before Metal?

**Pros**:
- Dimensional splitting is easier than GPU programming
- 2D shows solver generality
- More science, less engineering

**Cons**:
- Still memory-bandwidth limited on CPU
- Won't see performance breakthrough
- 2D on GPU is just scaling up 1D GPU

**Recommendation**: **Metal first**, then 2D on both CPU and GPU.
GPU investment pays dividends for all future work.

---

## Files to Create

```
xpu/metal/kernels/
├── euler_conversions.metal      # Prim ↔ Cons
├── plm_reconstruction.metal     # Stencil-based reconstruction
├── hlle_solver.metal            # Riemann solver
├── flux_update.metal            # Conservative update
└── shared.metal                 # Common functions/constants

physics/src/hydro/
├── metal_backend.rs             # Metal device integration
└── solver1d.rs                  # Add MetalGpu execution mode

physics/examples/
└── metal_performance.rs         # Metal vs CPU comparison
```

---

**Status**: Ready to implement
**Estimated Effort**: 1-2 weeks full-time
**Expected Result**: 10-50x speedup, production-ready GPU solver

*Let's build it.* 🚀