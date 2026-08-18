# symbi-algebra

The mathematical floor of the workspace. Tensors and their variance, the domain
description a field is laid out over, the memory layout rules, boundary kinds, and
the marker traits everything else is generic over. It has no dependencies at all,
inside the workspace or outside it, and that is deliberate. Anything placed here is
available everywhere, so the bar for adding to it is high.

## Where it sits

At the bottom. Every other crate depends on this one, and this one depends on
nothing.

## Where to start reading

`tensor.rs` and `variance.rs` for the index machinery, `domain.rs` for how a grid
is described before any memory is allocated, and `layout.rs` for the traversal
order. That last one deserves a moment of attention. It owns the single definition
of which axis is contiguous, and a traversal that disagrees with it produces
answers that look physically reasonable while being wrong, so the tests that pin it
are worth more than they appear.

## Things worth knowing before you change it

The production `Scalar` and `Selectable` traits live in `symbi_ir::algebra` rather
than here. This crate carries the mathematics that needs no notion of tracing or
code generation, and the split is what keeps it dependency-free.
