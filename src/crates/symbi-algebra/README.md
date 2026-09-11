# symbi-algebra

The basic mathematical types: tensors and their variance, grid domains, memory
layout, boundary kinds, and marker traits used by the other crates. This crate
has no dependencies, including outside the workspace. Keep additions small since
these types are shared widely.

## Dependencies

None. This is one of the starting points of the workspace dependency graph.

## Start here

`tensor.rs` and `variance.rs` cover tensor indices. `domain.rs` describes a grid
before memory is allocated, and `layout.rs` defines the traversal order.

Pay attention to `layout.rs` when changing indexing. It defines which axis is
contiguous, and a traversal that uses a different convention can give plausible
but incorrect results. Keep the layout tests in mind when making changes.

## Notes

`Scalar` and `Selectable` live in `symbi_ir::algebra`. The types here don't need
to know about tracing or code generation, which keeps this crate dependency-free.
