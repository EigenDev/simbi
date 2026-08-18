# symbi-expr

The expression language a user's configuration writes in. You build a directed
acyclic graph of operations, compile it into a flat instruction stream, and
evaluate that stream with a register machine that uses no stack and no recursion.
Initial conditions, source terms, boundary conditions, and mesh motion laws all
arrive through here.

The flatness is the point. A linearized instruction stream with a fixed register
bank is something a GPU can execute, and it is also something the rest of the
workspace can splice into a larger computation graph.

## Where it sits

A leaf in the workspace, carrying serde for the wire format. `symbi-hydro`
reads configurations through it, and `symbi-discretize` uses
it in tests.

## Where to start reading

`dag.rs` to see how expressions are built, then `linearize.rs` for the topological
sort and the register allocation, then `eval.rs` for the machine itself.
`load.rs` holds the JSON wire format the Python frontend sends.

## Things worth knowing before you change it

Registers are recycled at their last use, and expressions are scheduled in index
order. The quantity under pressure is how many values are simultaneously live,
which is a rather different thing from how many nodes the graph has. A wide
expression with a shallow dependence structure costs very little; a narrow one that
keeps early results alive until the end costs a great deal.
