# Writing physics here

The physics in this crate is generic over a carrier `S: Scalar`. Run it at
`S = f64` and it computes numbers. Run it at `S = Gv` and the same text records a
computation graph, which becomes the GPU kernel. Run it at `S = Dual` and it
carries derivatives alongside values.

That one arrangement buys a great deal, and it costs you four things. This page is
the whole list.

## One: you cannot branch on a value

At trace time there is no value to branch on, so an ordinary `if` on something of
type `S` cannot work. Comparisons return `S::Mask` rather than `bool`, and you
choose with `S::select`:

```rust
let bm = S::select(s_l.cmp_lt(S::ZERO), s_l, S::ZERO);
```

The comparison vocabulary is `cmp_lt`, `cmp_gt`, `cmp_le`, `cmp_ge`, `cmp_eq`.
Native `<` and `==` are deliberately absent from the trait, so reaching for them is
a compile error rather than a comparison of node indices.

**Both arms of a select are evaluated.** A division that is safe only in the taken
arm still has to be made safe in the other one, which is why you see denominators
guarded before the select rather than inside it:

```rust
let inv = S::ONE / S::select(ok, dn, S::ONE);
```

`S::branch` exists for the rarer case where the two arms have different shapes.

## Two: constants come from the carrier

`S::ZERO`, `S::ONE`, `S::HALF`, `S::TWO`, `S::THREE`, `S::FOUR`, plus
`S::INFINITY`, `S::NEG_INFINITY` and `S::NAN`. Anything else goes through
`S::from_f64`, which is what you want for a named constant:

```rust
let eps = S::from_f64(REL_EPS);
```

A bare `0.5` cannot be used directly, since `f32` also implements `Scalar` and
mixed `f64` arithmetic cannot be implemented for it under the orphan rule.

## Three: the trace is ambient

Building a graph opens a thread-local trace, and `Gv` operations append to
whichever trace is open. A builder can therefore run standalone or partway through
a larger trace, and it must behave the same either way. `in_isolated_trace` saves
and restores any open outer trace, and it is the sanctioned way to build a graph
from inside another one.

## Four: keep the physics in one place

Anything written here serves the CPU, CUDA and HIP paths at once. That is the
reason to resist writing a second copy of a formula for a special case, and the
reason a change here deserves a diff of the emitted kernels when it is meant to
preserve behavior.

## What this does not cost you

Signatures stay small. Nearly every function here is `<S: Scalar>` or
`<S: Scalar, const D: usize>`, and the body reads as arithmetic:

```rust
fn specific_enthalpy<S: Scalar>(rho: S, pre: S, gamma: S) -> S {
    S::ONE + gamma / (gamma - S::ONE) * pre / rho
}
```

You can also call any of it at `f64` from a unit test and step through it in a
debugger, which is the compensation for everything above.
