# symbi-abi

A vocabulary crate. It holds the names that a traced kernel and the dispatch code
that launches it have to agree on, so that agreement is arranged by construction
rather than by two places spelling the same string and hoping. `FieldRef` and
`FieldBind` name field buffers, `ScalarRef` and `ScalarBind` name scalar
parameters, and `MeshScalar` names the mesh quantities.

Each name is minted in exactly one function, `name()`, and read back by `parse()`.
When those two are the only doors, a rename is a compile error somewhere rather
than a silent mismatch at launch.

## Where it sits

A leaf, depending only on serde. `symbi-ir` builds on it so the IR can carry typed
containers while every hydro field name is spelled once, here.

## Where to start reading

`field_ref.rs` is representative. The other two modules follow the same pattern for
scalars.
