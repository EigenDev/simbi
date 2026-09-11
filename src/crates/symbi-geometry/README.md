# symbi-geometry

Coordinate maps, metric tensors, and finite-volume geometry: cell volumes, face
areas, centroids, and the connection terms that appear as sources in curvilinear
coordinates.

The metrics use the 3+1 ADM decomposition. Cartesian, spherical, and cylindrical
coordinates in flat spacetime use the same machinery as Schwarzschild and Kerr.
To add a spacetime, implement `Metric`; the discretization can then use that
metric definition.

## Dependencies

Uses `symbi-algebra` and `symbi-ir`. The physics crates build on this one.

## Start here

`metric.rs` defines the trait and its implementations. `coord_map.rs` maps grid
indices to positions, including logarithmic and geometric spacing. `centroid.rs`
computes volume-weighted cell centers.

## Notes

Christoffel symbols are computed by automatic differentiation with `Dual`. Keep
that calculation tied to the metric definition when adding a chart, so there's
no separate set of hand-derived connection coefficients to maintain.
