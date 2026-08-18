# symbi

The user-facing crate. It re-exports the core crates and provides the builder and
the evolution driver, so that a program written against it reads roughly as

    let mut sim = SimState::build(Newtonian, eos, Cartesian)
        .cells([n]).spacing([dx]).boundaries(BoundaryType::Outflow)
        .allocate()?.set_initial(|x| prim_at(x)).build();
    let sub = sim.substrate();
    evolve(&mut sim, &sub, t_final)?;

## Where it sits

At the top of the Rust side, depending on nearly everything. `symbi-py` sits above
it and exposes the same capability to Python.

## Where to start reading

`prelude.rs` to see what a user is given, then `sim/` for the builder and the
driver.
