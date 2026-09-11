# symbi-refinement

Fixed mesh refinement. Each level has its own simulation state and kernel set.
Transfer operators move data between levels, and flux and EMF registers keep the
coarse-fine interface conservative.

Restriction needs to conserve the transferred quantities, and prolongation needs
to avoid introducing spurious structure. The registers reconcile fluxes at shared
faces so the coarse and fine levels agree.

## Dependencies

Uses stepping routines from `symbi-sim` and per-level kernel sets from
`symbi-substrate`. The `symbi` crate drives it. The refined and single-grid
drivers share those lower-level routines without depending on each other.

## Start here

`refinement/hierarchy.rs` defines `Hierarchy`.

## Notes

Coarse-fine ghost transfer accounts for hydrostatic balance. On a stratified
background, transferring the raw state leaves an entropy signature at the
interface. Instead, the transfer uses the departure from the local hydrostatic
isentrope. The generated device kernels are bit-identical to the host path.

If a new fine level spans a domain decomposition cut, use the conserved exchange
and decomposed seeding path before priming the level.
