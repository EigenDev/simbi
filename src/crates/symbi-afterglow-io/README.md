# symbi-afterglow-io

The adapter that lets the afterglow module read real simulation output. It takes
an HDF5 checkpoint written in whatever geometry the hydrodynamics ran, in one, two,
or three dimensions, and produces the neutral Cartesian cell list the afterglow
core expects.

Its particular job is knowing which axis means what, since that is the part that
differs from run to run. Spherical three-dimensional data has axes for radius,
polar angle, and azimuth, while a two-dimensional axisymmetric run has only the
first two and needs the third synthesized. Cylindrical two-dimensional data is the
one to watch, because its second axis carries the third role.

## Where it sits

Above `symbi-afterglow` and `symbi-io`.
