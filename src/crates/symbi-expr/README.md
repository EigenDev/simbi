# symbi-expr

The expression language used in simulation configurations. Initial conditions,
source terms, boundary conditions, and mesh motion laws come through here.

Expressions start as directed acyclic graphs. They're compiled into a flat
instruction stream and evaluated by a register machine, with no stack or
recursion. That form works on a GPU and can also be inserted into a larger
computation graph.

## Dependencies

Uses serde for serialization and has no workspace dependencies. `symbi-hydro`
reads configuration expressions through it, and `symbi-discretize` uses it in
tests.

## Start here

`dag.rs` builds expressions. `linearize.rs` handles topological sorting and
register allocation, and `eval.rs` runs the instructions. `load.rs` defines the
JSON format sent by the Python frontend.

## Notes

Expressions are scheduled in index order, and a register can be reused after its
value's last use. Register pressure depends on how many values are live at once,
not just the number of nodes. When looking at a large expression, check how long
intermediate results have to stick around.
