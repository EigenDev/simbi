# symbi-afterglow

Synchrotron afterglow post-processing, for gamma-ray-burst-like relativistic blast
waves. It turns a hydrodynamic snapshot into what a distant observer would
actually see, using the Sari, Piran and Narayan spectral model together with
equal-arrival-time surface integration.

There are two complementary paths through it. The deterministic one computes
per-cell emission and integrates over the surface of equal arrival time to give a
light curve. The Monte Carlo one generates photon events and transports them with
self-absorption, scattering, and pair production, producing a catalog that can then
be reduced along any line of sight into light curves, sky maps, and polarization.
The second path costs more and answers questions about geometry that the first
cannot.

## Where it sits

A leaf. The physics is a pure CGS core over neutral arrays with no dependency on
the rest of the workspace, which is what makes it testable against analytic
solutions on its own.

## Where to start reading

`synchrotron.rs` for the emission, `lightcurve.rs` for the arrival-time
integration, and `transfer.rs` if you need the Monte Carlo path.

## Things worth knowing before you change it

Dimensional correctness is enforced at compile time through the `units` module, so
a mass where a length belongs is a type error. Observed flux needs the laboratory
radius, which on a homologously expanding mesh means dividing the comoving radius
by the scale factor before it enters the arrival-time calculation.
