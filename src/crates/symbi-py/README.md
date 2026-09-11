# symbi-py

The Python extension module. It reads the frontend's configuration dictionary,
collects the initial-condition generator into a typed buffer, releases the Python
interpreter lock, and starts the run with the selected regime, dimension,
geometry, and equation of state.

`symbi_sim::checkpoint` writes checkpoints in the layout expected by the Python
reader, so results can be loaded with `simbi.reader` and plotted with `simbi.viz`.

## Dependencies

Uses `symbi` and several other crates directly for configuration and
post-processing.

## Start here

`lib.rs`. Following one configuration field from the dictionary to the solver is
a useful way to see how the pieces connect.

## Notes

Validate configuration here, before it reaches the solver. A misread field can
change the physics without causing a runtime error. Add validation when exposing
a new option.

Retired option names raise an explanatory error. Silently ignoring one could
leave a run using a default solver the user didn't ask for.
