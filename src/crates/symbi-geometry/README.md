# symbi-geometry

Coordinate maps, metric tensors, and the finite-volume geometry that follows from
them. Cell volumes, face areas, centroids, and the connection terms that appear as
sources when the coordinates are curvilinear.

The design is built around the 3+1 ADM decomposition, which has a pleasant
consequence. Flat charts such as Cartesian, spherical, and cylindrical are special
cases of the same machinery that carries Schwarzschild and Kerr, so extending to a
new spacetime means implementing the `Metric` trait rather than opening up the
discretization.

## Where it sits

Above `symbi-algebra` and `symbi-ir`, below the physics.

## Where to start reading

`metric.rs` for the trait and its implementations, `coord_map.rs` for the
index-to-position maps including the logarithmic and geometric spacings, and
`centroid.rs` for the volume-weighted cell centers.

## Things worth knowing before you change it

Christoffel symbols are obtained by automatic differentiation through the `Dual`
carrier rather than being written out by hand. This is load-bearing. Hand-derived
connection coefficients for a new chart are a reliable source of quiet errors, and
letting the derivative fall out of the metric definition removes that whole class.
