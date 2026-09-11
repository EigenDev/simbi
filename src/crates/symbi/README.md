# symbi

The Rust interface for setting up and running a simulation. It re-exports the
core crates and provides the builder and evolution driver. A program looks
roughly like this:

    let mut sim = SimState::build(Newtonian, eos, Cartesian)
        .cells([n]).spacing([dx]).boundaries(BoundaryType::Outflow)
        .allocate()?.set_initial(|x| prim_at(x)).build();
    let sub = sim.substrate();
    evolve(&mut sim, &sub, t_final)?;

## Dependencies

This brings together most of the Rust workspace. `symbi-py` exposes it to Python.

## Start here

`prelude.rs` lists the main imports. `sim/` has the builder and evolution driver.
