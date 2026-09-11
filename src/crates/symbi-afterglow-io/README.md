# symbi-afterglow-io

This reads hydrodynamic HDF5 checkpoints and converts them to the Cartesian cell
list used by `symbi-afterglow`. It handles simulation output in one, two, or three
dimensions and accounts for the geometry of the original grid.

Most of the bookkeeping is about axes. A 3D spherical grid has radius, polar
angle, and azimuth. A 2D axisymmetric grid needs the azimuthal direction filled
in. The 2D cylindrical case needs extra care because its second axis plays the
role of the third coordinate.

## Dependencies

`symbi-afterglow` and `symbi-io`.
