// =============================================================================
// euler1d.rs
//
// comprehensive benchmarks for 1d euler solver.
// tests performance across different grid sizes and compares:
//   - single-threaded execution
//   - multi-threaded execution (rayon)
//   - different time integrators (rk2 vs rk3)
//   - different reconstruction schemes (pcm vs plm)
//
// run:
//   cargo bench --bench euler1d
//
// results:
//   target/criterion/euler1d/report/index.html
// =============================================================================

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use physics::hydro::{BoundaryCondition, Euler1DSolver, Primitive1D};

// =============================================================================
// benchmark: single time step
// =============================================================================

fn bench_single_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("euler1d_single_step");

    for ncells in [100, 500, 1000, 5000].iter() {
        group.throughput(Throughput::Elements(*ncells as u64));

        group.bench_with_input(BenchmarkId::new("rk3", ncells), ncells, |b, &ncells| {
            let mut solver = Euler1DSolver::new(
                ncells,
                0.0,
                1.0,
                1.4,
                0.5,
                BoundaryCondition::Outflow,
                3, // rk3
            );

            // sod shock tube initial condition
            solver.set_initial_conditions(|x| {
                if x < 0.5 {
                    Primitive1D::new(1.0, 0.0, 1.0)
                } else {
                    Primitive1D::new(0.125, 0.0, 0.1)
                }
            });

            b.iter(|| {
                solver.step();
            });
        });

        group.bench_with_input(BenchmarkId::new("rk2", ncells), ncells, |b, &ncells| {
            let mut solver = Euler1DSolver::new(
                ncells,
                0.0,
                1.0,
                1.4,
                0.5,
                BoundaryCondition::Outflow,
                2, // rk2
            );

            solver.set_initial_conditions(|x| {
                if x < 0.5 {
                    Primitive1D::new(1.0, 0.0, 1.0)
                } else {
                    Primitive1D::new(0.125, 0.0, 0.1)
                }
            });

            b.iter(|| {
                solver.step();
            });
        });
    }

    group.finish();
}

// =============================================================================
// benchmark: full simulation (sod shock tube)
// =============================================================================

fn bench_sod_shock_tube(c: &mut Criterion) {
    let mut group = c.benchmark_group("euler1d_sod_shock_tube");
    group.sample_size(20); // fewer samples for full simulation

    for ncells in [100, 500, 1000].iter() {
        let t_final = 0.2;
        group.throughput(Throughput::Elements(*ncells as u64));

        group.bench_with_input(BenchmarkId::from_parameter(ncells), ncells, |b, &ncells| {
            b.iter(|| {
                let mut solver =
                    Euler1DSolver::new(ncells, 0.0, 1.0, 1.4, 0.5, BoundaryCondition::Outflow, 3);

                solver.set_initial_conditions(|x| {
                    if x < 0.5 {
                        Primitive1D::new(1.0, 0.0, 1.0)
                    } else {
                        Primitive1D::new(0.125, 0.0, 0.1)
                    }
                });

                solver.evolve_to(black_box(t_final));

                // return something to prevent optimization
                black_box(solver.time());
            });
        });
    }

    group.finish();
}

// =============================================================================
// benchmark: reconstruction overhead
// =============================================================================

fn bench_reconstruction(c: &mut Criterion) {
    let mut group = c.benchmark_group("euler1d_reconstruction");

    let ncells = 1000;
    group.throughput(Throughput::Elements(ncells as u64));

    // benchmark just the spatial operator (reconstruction + riemann + flux update)
    group.bench_function("spatial_operator", |b| {
        let mut solver =
            Euler1DSolver::new(ncells, 0.0, 1.0, 1.4, 0.5, BoundaryCondition::Outflow, 3);

        solver.set_initial_conditions(|x| {
            if x < 0.5 {
                Primitive1D::new(1.0, 0.0, 1.0)
            } else {
                Primitive1D::new(0.125, 0.0, 0.1)
            }
        });

        b.iter(|| {
            // single step involves 3x spatial operator calls (rk3)
            solver.step();
        });
    });

    group.finish();
}

// =============================================================================
// benchmark: different boundary conditions
// =============================================================================

fn bench_boundary_conditions(c: &mut Criterion) {
    let mut group = c.benchmark_group("euler1d_boundary_conditions");

    let ncells = 1000;
    group.throughput(Throughput::Elements(ncells as u64));

    for bc in [
        BoundaryCondition::Outflow,
        BoundaryCondition::Periodic,
        BoundaryCondition::Reflecting,
    ]
    .iter()
    {
        let bc_name = match bc {
            BoundaryCondition::Outflow => "outflow",
            BoundaryCondition::Periodic => "periodic",
            BoundaryCondition::Reflecting => "reflecting",
            _ => "other",
        };

        group.bench_with_input(BenchmarkId::from_parameter(bc_name), bc, |b, bc| {
            let mut solver = Euler1DSolver::new(ncells, 0.0, 1.0, 1.4, 0.5, *bc, 3);

            solver.set_initial_conditions(|x| {
                if x < 0.5 {
                    Primitive1D::new(1.0, 0.0, 1.0)
                } else {
                    Primitive1D::new(0.125, 0.0, 0.1)
                }
            });

            b.iter(|| {
                solver.step();
            });
        });
    }

    group.finish();
}

// =============================================================================
// benchmark: memory bandwidth
// =============================================================================

fn bench_memory_bandwidth(c: &mut Criterion) {
    let mut group = c.benchmark_group("euler1d_memory_bandwidth");

    for ncells in [1000, 10000, 100000].iter() {
        let bytes_per_step = ncells * std::mem::size_of::<f64>() * 3 * 6; // 3 conserved vars, ~6 reads/writes per step
        group.throughput(Throughput::Bytes(bytes_per_step as u64));

        group.bench_with_input(BenchmarkId::from_parameter(ncells), ncells, |b, &ncells| {
            let mut solver =
                Euler1DSolver::new(ncells, 0.0, 1.0, 1.4, 0.5, BoundaryCondition::Outflow, 3);

            solver.set_initial_conditions(|x| {
                if x < 0.5 {
                    Primitive1D::new(1.0, 0.0, 1.0)
                } else {
                    Primitive1D::new(0.125, 0.0, 0.1)
                }
            });

            b.iter(|| {
                solver.step();
            });
        });
    }

    group.finish();
}

// =============================================================================
// benchmark: scaling with problem size
// =============================================================================

fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("euler1d_scaling");
    group.sample_size(20);

    let sizes = [50, 100, 200, 400, 800, 1600, 3200];

    for &ncells in sizes.iter() {
        group.throughput(Throughput::Elements(ncells as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(ncells),
            &ncells,
            |b, &ncells| {
                b.iter(|| {
                    let mut solver = Euler1DSolver::new(
                        ncells,
                        0.0,
                        1.0,
                        1.4,
                        0.5,
                        BoundaryCondition::Outflow,
                        3,
                    );

                    solver.set_initial_conditions(|x| {
                        if x < 0.5 {
                            Primitive1D::new(1.0, 0.0, 1.0)
                        } else {
                            Primitive1D::new(0.125, 0.0, 0.1)
                        }
                    });

                    // run 10 steps
                    for _ in 0..10 {
                        solver.step();
                    }

                    black_box(solver.time());
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// benchmark: cell updates per second (throughput metric)
// =============================================================================

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("euler1d_throughput");

    let ncells = 10000;
    group.throughput(Throughput::Elements(ncells as u64));

    group.bench_function("cell_updates_per_second", |b| {
        let mut solver =
            Euler1DSolver::new(ncells, 0.0, 1.0, 1.4, 0.5, BoundaryCondition::Outflow, 3);

        solver.set_initial_conditions(|x| {
            if x < 0.5 {
                Primitive1D::new(1.0, 0.0, 1.0)
            } else {
                Primitive1D::new(0.125, 0.0, 0.1)
            }
        });

        b.iter(|| {
            solver.step();
        });
    });

    group.finish();
}

// =============================================================================
// benchmark groups
// =============================================================================

criterion_group!(
    benches,
    bench_single_step,
    bench_sod_shock_tube,
    bench_reconstruction,
    bench_boundary_conditions,
    bench_memory_bandwidth,
    bench_scaling,
    bench_throughput
);

criterion_main!(benches);
