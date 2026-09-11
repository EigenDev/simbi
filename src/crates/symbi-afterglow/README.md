# symbi-afterglow

Synchrotron afterglow calculations for relativistic blast waves, such as those in
gamma-ray bursts. This takes hydrodynamic snapshots and computes the emission a
distant observer would receive, using the Sari, Piran and Narayan spectral model
and equal-arrival-time surface integration.

There are two ways to do the calculation. The deterministic path computes each
cell's emission and integrates it into a light curve. The Monte Carlo path
samples photon events and transports them with self-absorption, scattering, and
pair production. Its event catalog can be reduced along different lines of sight
to get light curves, sky maps, and polarization. That costs more, but lets you
explore geometry and transport in more detail.

## Dependencies

The physics uses CGS units and works on arrays, with no dependencies on other
workspace crates. It can be tested against analytic solutions on its own.

## Start here

`synchrotron.rs` has the emission model, `lightcurve.rs` handles arrival-time
integration, and `transfer.rs` has the Monte Carlo calculation.

## Notes

The `units` module checks dimensions at compile time: passing a mass where a
length is expected is a type error.

Observed flux needs the laboratory radius. On a homologously expanding mesh,
divide the comoving radius by the scale factor before the arrival-time calculation.
