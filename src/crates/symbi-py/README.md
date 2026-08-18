# symbi-py

The Python extension module. It parses the configuration dictionary the Python
frontend produces, drains the initial-condition generator into a typed buffer,
releases the interpreter lock, dispatches on the run's regime, dimensionality,
geometry, and equation of state, and runs.

Checkpoints are written by `symbi_sim::checkpoint`, in the layout the existing
Python reader expects, so results come back through the unchanged `simbi.reader`
and `simbi.viz` stack.

## Where it sits

At the very top. It depends on `symbi` and on several crates directly for the
configuration and post-processing surfaces it exposes.

## Where to start reading

`lib.rs`, following one configuration field from the dictionary through to the
value the solver receives.

## Things worth knowing before you change it

This crate is where a mistaken configuration should be caught, since a
misinterpreted field here becomes wrong physics with no other warning. The
pre-flight validation exists for that reason, and a new configuration surface
deserves a check there alongside the wiring.

Retired option names raise an informative error rather than being ignored, which
matters because a silently dropped solver name would leave a run quietly using the
default.
