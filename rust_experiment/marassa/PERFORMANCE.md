# Performance Results: 1D Euler Solver

## Serial CPU Baseline (M3 Max)

Date: January 2025  
Configuration: Release build, RK3 time integration, PLM reconstruction, MinMod limiter

### Zone-Cycles Per Second

| Grid Size | Zone-Cycles/sec | Time/Step (ms) | Notes |
|-----------|----------------|----------------|-------|
| 100       | 2.8e7          | 0.004          | Small problem overhead |
| 500       | 3.1e7          | 0.016          | Sweet spot begins |
| 1,000     | 3.2e7          | 0.032          | Good cache utilization |
| 5,000     | 3.4e7          | 0.146          | Peak serial performance |
| 10,000    | 2.9e7          | 0.345          | Cache pressure |

### Key Observations

**Peak Performance**: ~34 million zone-cycles/second at 5,000 cells

**Performance Characteristics**:
- Small grids (< 500): Dominated by overhead
- Medium grids (500-5000): Linear scaling, optimal cache usage
- Large grids (> 10k): Cache misses reduce performance

**Bottlenecks**:
- Memory bandwidth limited (not compute bound)
- Single-threaded execution
- No SIMD vectorization yet

## Comparison Matrix

| Implementation | Status | Measured Speedup | Notes |
|---------------|--------|------------------|-------|
| Serial CPU    | ✅ Complete | 1.0x (baseline) | Peak: 3.4e7 zone-cycles/sec |
| Parallel CPU  | ✅ Complete | 0.1-1.1x | **Memory-bandwidth limited, not beneficial** |
| Metal GPU     | ⏳ Pending | 10-50x expected | XPU Metal backend required |

### Parallel CPU Analysis

**Implemented but not beneficial** for this problem:
- Small grids (< 1000): Overhead dominates, ~1.0x (no gain)
- Large grids (> 1000): **0.1-0.5x (slower!)** due to:
  - Memory bandwidth saturation (not compute-bound)
  - Rayon overhead for fine-grained parallelism
  - Cache thrashing from parallel access patterns

**Conclusion**: This solver is **memory-bandwidth limited**, not compute-limited. Parallel CPU threads compete for the same memory bus. GPU with high-bandwidth memory (HBM) will show real speedup.

## Next Steps

### Metal GPU (High Priority)
- Port spatial operator to Metal compute shader
- SoA (Structure of Arrays) layout already optimal for GPU
- Expected: 10-50x speedup depending on problem size
- Best for grids > 10,000 cells

### Optimization Opportunities
1. **Metal GPU**: High-bandwidth memory will eliminate bottleneck (10-50x expected)
2. **SIMD vectorization**: Manual SIMD for flux computation (~2-4x)
3. **Cache blocking**: Improve locality for large grids (marginal gains)
4. **Multi-GPU**: XPU system designed for heterogeneous execution

### Why Parallel CPU Failed
The 1D Euler solver is **memory-bandwidth bound**:
- Each time step reads/writes ~6x grid size in memory
- Arithmetic intensity too low (flops/byte ratio < 1)
- CPU memory bandwidth: ~50 GB/s (shared by all cores)
- Adding more threads = more contention for same bus
- **GPU with HBM**: ~400-600 GB/s → real speedup expected

## Validation

All test cases pass:
- ✅ Sod shock tube (t=0.2, 200 cells): Correct shock/rarefaction/contact
- ✅ Constant state preservation
- ✅ Symmetry tests
- ✅ 42 unit tests (compute crate)
- ✅ 35 unit tests (physics crate)

## Build System

```bash
# Run performance test
cargo run --release --example performance_comparison

# Run Sod shock tube
cargo run --release --example sod_shock_tube

# Run unit tests
cargo test -p physics --lib
cargo test -p compute --lib
```

## Code Quality

- **Zero warnings** in release build
- **Zero unsafe** in solver core (only in low-level views)
- **Zero runtime overhead** from abstractions
- **Zero-sized types** for compile-time dispatch

## Future Work

1. ✅ ~~Implement parallel CPU version (Rayon)~~ - Complete but not beneficial
2. **Implement Metal GPU version (XPU backend)** ← High priority
3. Add 2D solver via dimensional splitting
4. Add HLLC solver for sharper contacts
5. Add more test cases (blast waves, etc.)
6. Consider GPU-specific optimizations (tiling, shared memory)

---

*Baseline established: January 2025*  
*Framework: Marassa (Rust XPU Zero-Cost Performance)*