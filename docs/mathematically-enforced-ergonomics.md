# A mathematically enforced ergonomics program

## Status and purpose

This document is an architectural north star, not a claim that every proposed
type or API should land unchanged. Its purpose is to keep the compiler powerful
without making compiler machinery the language in which physics is written.

The governing idea is:

> Physics describes lawful mathematics. Discretization describes locality.
> Compilation interprets those descriptions. Execution manages machines.

Each layer should expose concepts from its own domain and make invalid
cross-layer states unrepresentable. The graph is an implementation of one
interpretation, not the ontology of the whole system.

The desired result is a system that remains:

- mathematically well-posed;
- pleasant for a physicist to extend;
- explicit about effects and numerical semantics;
- scalable to new equations, discretizations, and targets;
- portable across CPU, CUDA, HIP, and future architectures;
- guided by single responsibility, simple interfaces, and functional design.

## 1. Four semantic languages

The system contains four distinct languages:

```text
Physics<S>
    |
    | interpretation
    v
DiscreteProgram<S, Domain>
    |
    | tracing
    v
KernelProgram
    |
    | lowering
    v
Executable<Target>
```

They should have separate vocabularies and responsibilities.

| Layer | Vocabulary | Must not know |
| --- | --- | --- |
| Physics | states, fluxes, sources, metrics | nodes, kernels, devices |
| Discretization | cells, faces, stencils, boundaries | graph indices, CUDA |
| Kernel IR | values, control flow, loads, stores | Riemann solvers, bodies |
| Execution | buffers, devices, launches | symbolic mathematics |

This yields a mechanically checkable dependency rule:

> A crate may depend on the interface of the language immediately below it,
> but it may not manipulate that language's representation.

Physics may depend on a carrier algebra. It must not depend on `Graph` or
`NodeId`. Discretization may request neighbor values. It must not manufacture
raw load nodes. Execution may launch a compiled kernel. It must not inspect the
physics that produced it.

## 2. A foundational carrier algebra

`Scalar`, `Mask`, selection, and lawful iteration are mathematical interfaces.
They should live in a small foundation crate, such as `symbi-carrier`, rather
than in the IR implementation.

An illustrative core is:

```rust
pub trait Scalar:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    type Predicate: Predicate;

    const ZERO: Self;
    const ONE: Self;

    fn from_f64(value: f64) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn cmp_lt(self, rhs: Self) -> Self::Predicate;

    fn select(
        condition: Self::Predicate,
        then_value: Self,
        else_value: Self,
    ) -> Self;
}
```

Interpretations then point inward:

```text
symbi-carrier
 |- f64 interpretation
 |- f32 interpretation
 |- Dual<S> interpretation
 `- Gv interpretation       supplied by symbi-ir
```

The compiler implements the mathematical interface; it does not own the
interface. This makes the claim that physics is compiler-independent true in
both source code and the crate dependency graph.

The carrier surface should remain deliberately small. Adding a primitive is a
one-way architectural decision because every interpretation must implement it.
Derived operations belong in ordinary generic functions whenever possible.

## 3. Branded, scoped tracing

Tracing should be a scoped interpretation, not ambient mutable state.

The desired use is approximately:

```rust
let kernel = trace(|cx| {
    let rho = cx.field(Field::Density);
    let pressure = cx.field(Field::Pressure);
    let radius = cx.coordinate(Axis::R);

    let source = radial_source(rho, pressure, radius);

    cx.outputs([Output::momentum(Axis::R, source)])
});
```

A symbolic value is branded with the lifetime of the trace that owns it:

```rust
pub struct Symbolic<'trace> {
    id: ValueId,
    context: &'trace TraceContext<'trace>,
}
```

The trace closure should be generative, conceptually using `for<'trace>`, so
each invocation creates a distinct logical universe. Rust can then prevent a
symbolic value from escaping its graph or being combined with a value from a
different graph.

```rust,compile_fail
let leaked = trace(|cx| cx.constant(1.0));
// the trace-branded value cannot escape
```

Operator syntax can remain concise because a `Symbolic<'trace>` can retain
scoped access to its recorder. The mutation is then local, graph-affiliated,
panic-safe, and impossible to use after the trace closes.

This design should guarantee:

- no manual `begin_trace` and `end_trace` pairing;
- no silent replacement of an active trace;
- restoration during panic unwinding;
- no values belonging to an unknown graph;
- no cross-graph arithmetic;
- no thread-local state in the contributor-facing API.

An RAII trace guard is a useful transitional step, but the end state should be
a closure that cannot return trace-owned values.

## 4. Three classes of laws

The numerical constitution must distinguish structural facts, exact
computational identities, and analytic mathematics. Conflating them risks
turning a useful mathematical statement into an unsound floating-point rewrite.

### 4.1 Structural laws

These are enforced by traits and types:

- every carrier implements every primitive;
- predicates cannot be converted into host booleans;
- values cannot cross trace ownership boundaries;
- stores accept values with the required element type;
- tensor shape, frame, and variance agree;
- iteration has an explicit and carrier-independent convergence contract.

### 4.2 Exact computational laws

These may justify compiler rewrites under a named numerical theory:

- selecting on a known Boolean constant;
- double Boolean negation;
- projection from a freshly constructed product;
- exact integer index normalization, with overflow semantics specified;
- identities proven safe for the element type and floating-point policy.

Even `x + 0 = x` requires a policy decision in the presence of signed zero and
NaN. No rewrite should inherit permission merely because it is true over the
real numbers.

### 4.3 Analytic laws

These express continuous mathematics but do not automatically authorize
floating-point rewrites:

- associativity and distributivity;
- transcendental inverse identities;
- differential identities;
- conservation and covariance properties;
- convergence to a continuum solution.

They belong in symbolic proofs, property tests, convergence studies, or tests
with explicitly stated tolerances and domains.

A pass should declare the theory under which it operates:

```rust
pub trait RewriteTheory {
    fn permits(&self, law: Law) -> bool;
}

pub enum FloatingPointTheory {
    StrictIeee,
    FiniteOnly,
    FastMath,
}
```

An optimization is thereby parameterized by semantics rather than by an
implicit collection of assumptions.

## 5. Semantic physical types

Important physical distinctions should appear in types at subsystem
boundaries:

```rust
struct Density<S>(S);
struct Pressure<S>(S);
struct Velocity<S, const D: usize>(Vector<S, D>);
struct Momentum<S, const D: usize>(Covector<S, D>);
struct EnergyDensity<S>(S);
```

Not every intermediate needs a wrapper. Types earn their place where confusing
two quantities would be physically meaningful and dangerous:

- primitive and conserved states;
- coordinate and orthonormal components;
- covariant and contravariant vectors;
- cell-, face-, and edge-centered fields;
- coordinate and proper time;
- densities and integrated quantities;
- dimensional and nondimensional values.

Signatures should explain the physics:

```rust
fn pressure<S, E>(
    eos: &E,
    primitive: &PrimitiveState<S>,
) -> Pressure<S>
where
    S: Scalar,
    E: EquationOfState<S>;
```

Within a tightly scoped formula, destructuring wrappers can recover concise
arithmetic. Semantic types are most valuable at ownership and phase boundaries,
not as ceremony around every multiplication.

## 6. Variance and geometric frame

In relativistic and curvilinear physics, a bare array of components is often
underspecified.

```rust
struct Vector<S, Frame, Variance, const D: usize> {
    components: [S; D],
    _marker: PhantomData<(Frame, Variance)>,
}

enum CoordinateFrame {}
enum OrthonormalFrame {}
enum Covariant {}
enum Contravariant {}
```

Contraction can then be lawful by signature:

```rust
fn contract<S, F, const D: usize>(
    covector: Vector<S, F, Covariant, D>,
    vector: Vector<S, F, Contravariant, D>,
) -> S;
```

The type system should reject:

- addition across different frames;
- contraction of two contravariant vectors without a metric;
- coordinate velocity passed to an orthonormal solver;
- a face-normal flux treated as a cell-centered vector.

Frame transformations, raising and lowering, and changes in centering should
be named operations. They are mathematically meaningful events, not incidental
array manipulation.

## 7. A stencil algebra instead of raw graph loads

Discretization should have a small language of its own:

```rust
pub trait Stencil<S: Scalar, const D: usize> {
    fn cell(&self, field: Field) -> S;
    fn neighbor(&self, field: Field, offset: Offset<D>) -> S;
    fn position(&self) -> Point<S, D>;
    fn spacing(&self) -> CoordinateVector<S, D>;
}
```

A discrete operator should read like numerical analysis:

```rust
fn centered_gradient<S, C, const D: usize>(
    cx: &C,
    field: Field,
    axis: Axis<D>,
) -> S
where
    S: Scalar,
    C: Stencil<S, D>,
{
    let left = cx.neighbor(field, Offset::backward(axis));
    let right = cx.neighbor(field, Offset::forward(axis));
    (right - left) / (S::TWO * cx.spacing()[axis])
}
```

The `Stencil<Symbolic<'trace>, D>` interpretation generates the appropriate IR
loads. The formula never sees `LoadAt`, coordinate nodes, or graph indices.

Because this language already expresses locality, it can derive:

- halo reach;
- field-read manifests;
- boundary requirements;
- shared-memory tiling candidates;
- dependency and support information.

Those properties should not need to be rediscovered from loosely structured
graph operations when the discretization description already knows them.

## 8. Opaque kernel programs

A compiled kernel description should be an immutable, opaque value:

```rust
pub struct KernelProgram {
    graph: Graph,
    signature: KernelSignature,
    effects: KernelEffects,
}

pub struct KernelSignature {
    inputs: Vec<KernelInput>,
    parameters: Vec<KernelParameter>,
    outputs: Vec<KernelOutput>,
    domain: LaunchDomain,
}

pub struct KernelEffects {
    reads: FieldSet,
    writes: FieldSet,
    stencil_reach: StencilReach,
    support: Support,
}
```

Its representation remains private. Public compiler-client APIs should not
mention `NodeId`.

A source builder should return a domain object:

```rust
pub struct SourceProgram {
    kernel: KernelProgram,
    target: SourceTarget,
}
```

rather than a public graph paired with a vector of output node indices.

### 8.1 Strings are presentation, not identity

Raw strings are appropriate at the shell of the compiler: parsing user input,
serialization, diagnostics, and rendering source code. They are not an
appropriate internal identity for fields, parameters, effects, or ABI slots.
The governing rule is:

> Strings are presentation. Typed values are identity.

At present, spellings such as `"prim_v0"`, `"prim.vel[0]"`, and
`FieldRef::PrimVel(0)` can denote related parts of one physical quantity while
inhabiting unrelated string conventions. A typo remains valid Rust, renaming is
non-local, collisions are detected late, and algorithms may accidentally
compare names from different namespaces. This is stringly typed compiler state.

The internal vocabulary should distinguish those namespaces:

```rust
enum FieldSlot {
    PrimRho,
    PrimVel(Axis),
    PrimPre,
    ConsDen,
    ConsMom(Axis),
    FluxMom(Axis),
    WaveSpeed { side: Side, axis: Axis },
    Scratch(ScratchId),
    User(UserFieldId),
}

enum ScalarParam {
    Gamma,
    Dt,
    GridSpacing(Axis),
    BodyMass(BodyId),
    User(UserParamId),
}

struct InputKey(SymbolId);
struct OutputKey(SymbolId);

struct KernelWrite {
    key: OutputKey,
    destination: FieldSlot,
    value: NodeId,
}
```

`InputKey`, `OutputKey`, `FieldSlot`, and `ScalarParam` are different types
because they answer different questions. Even if two values eventually render
to the same bytes, they must not compare equal merely because their textual
spellings happen to coincide.

User-defined names are the legitimate open world. Parse or intern them once at
the boundary into `UserFieldId`, `UserParamId`, or `SymbolId`; thereafter,
equality and hashing use the typed identifier. Built-in identities should be
closed enums or validated constructors. Formatting into names such as
`"prim.vel[0]"` belongs in one ABI renderer:

```rust
impl FieldSlot {
    fn render_abi(&self, out: &mut String) { /* one naming authority */ }
}
```

This yields several enforceable laws:

1. A field destination cannot be passed where a scalar parameter is expected.
2. Component and axis indices are range-checked when their typed values are
   constructed, not when a generated kernel runs.
3. Fusion, support inference, and dependency analysis compare semantic
   identities rather than formatting conventions.
4. Renaming an emitted ABI spelling changes one renderer and no physical code.
5. Every remaining internal `format!(...)` that creates identity is treated as
   a migration site, while formatting diagnostics remains ordinary and benign.

The objective is not to ban `String`. Kernel display names, error messages,
generated source, serialized configuration, and user-authored symbols remain
textual. The objective is to prevent text from serving as the proof that two
compiler objects are the same object.

## 9. Reads and writes as an effect system

Kernel fusion is an effect-composition problem. It should be modeled as one.

```rust
struct Effects {
    reads: FieldSet,
    writes: FieldSet,
    locality: Locality,
}

enum Locality {
    Pointwise,
    Stencil(StencilReach),
    Reduction,
}
```

Two programs can compose in parallel only when their effects commute. At a
minimum, for programs `A` and `B`, this requires

\[
W_A \cap W_B = \varnothing.
\]

Depending on stage semantics, it may also require

\[
W_A \cap R_B = \varnothing,
\qquad
W_B \cap R_A = \varnothing.
\]

Expose the distinction directly:

```rust
impl KernelProgram {
    pub fn parallel(self, other: Self) -> Result<Self, Conflict>;
    pub fn sequential(self, other: Self) -> Pipeline;
}
```

The vocabulary carries meaning:

- `parallel` means the effects commute and fusion may be legal;
- `sequential` preserves an ordering dependency;
- `map` transforms outputs;
- `zip` combines independent products;
- `then` constructs a pipeline.

Fusion is then an optimization of a lawful composition, not the semantic
operation exposed to physics code.

## 10. Physical composition before compiler composition

A sum of source terms is mathematical:

\[
S_{\mathrm{total}}(U, x, t) = \sum_i S_i(U, x, t).
\]

It should be represented as such:

```rust
let source =
    gravity(point_mass)
        + geometry(metric)
        + immersed_boundary(body);
```

The compiler may lower this into one graph, several fused kernels, or an
ordered pipeline. The source composition itself should not call graph splicing
or remap node indices.

The mathematical expression is the specification. Graph splicing is one
implementation strategy.

## 11. Interpretable domain programs

The most scalable functional pattern is a small immutable program interpreted
in several algebras:

```rust
pub trait SourceLaw<const D: usize> {
    fn evaluate<S: Scalar>(
        &self,
        state: PrimitiveState<S, D>,
        point: Point<S, D>,
        time: S,
    ) -> ConservedSource<S, D>;
}
```

The same source text can run at:

- `S = f64` for reference evaluation;
- `S = f32` for reduced-precision conformance;
- `S = Dual<f64>` for derivatives and Jacobians;
- `S = Symbolic<'trace>` for compilation;
- an interval carrier for enclosure checks;
- a dimensional carrier for unit validation;
- a polynomial carrier for perturbative analysis.

The scalable abstraction is not that everything is a graph. It is that each
mathematical program admits multiple lawful interpretations.

## 12. Dimensional validation as an interpretation

Units need not infect every production type to be checked. A verification
carrier can transport dimensions alongside values:

```rust
struct Dimensional<S> {
    value: S,
    dimension: Dimension,
}
```

Its algebra enforces:

- addition only between equal dimensions;
- multiplication by adding dimension exponents;
- division by subtracting dimension exponents;
- transcendental functions only on dimensionless arguments;
- comparisons only between compatible quantities.

Representative physical programs can be interpreted through this carrier in
tests or validation builds. This checks the same formula text that produces
host answers and device kernels without adding runtime unit metadata.

## 13. Valid physical phases as types

Primitive and conserved states are different physical phases, not similarly
shaped bags of arrays.

```rust
struct PrimitiveState<S, R: Regime> { /* private */ }
struct ConservedState<S, R: Regime> { /* private */ }

fn recover<R, S>(
    conserved: ConservedState<S, R>,
    eos: &R::Eos,
) -> Result<PrimitiveState<S, R>, RecoveryFailure<S>>;
```

Other phase markers can be useful at carefully chosen boundaries:

```rust
struct Admissible<T>(T);
struct Normalized<T>(T);
struct CellCentered<T>(T);
struct FaceCentered<T, Axis>(T);
```

They should be introduced where they remove real ambiguity. The goal is to
make important state transitions explicit, not to maximize the number of type
parameters.

## 14. A radically simple public path

The contributor-facing experience should fit on one page:

```rust
impl<const D: usize> SourceLaw<D> for PointMass {
    fn evaluate<S: Scalar>(
        &self,
        primitive: PrimitiveState<S, D>,
        point: Point<S, D>,
        _time: S,
    ) -> ConservedSource<S, D> {
        let displacement = point - self.center.cast();
        let radius_squared =
            displacement.norm_squared() + S::from_f64(self.softening_squared);
        let inv_radius_cubed = radius_squared.recip_sqrt() / radius_squared;
        let acceleration =
            -S::from_f64(self.gm) * displacement * inv_radius_cubed;

        ConservedSource::from_acceleration(primitive, acceleration)
    }
}
```

Everything below this level is infrastructure:

- trace ownership;
- graph construction and interning;
- support inference;
- scalarization;
- register pressure;
- target emission;
- launch policy.

A physicist should usually encounter only the carrier algebra, physical state
types, tensors and geometry, stencil access where necessary, and source or flux
traits.

## 15. The design test

Every public abstraction should answer five questions:

1. What mathematical or computational object does this represent?
2. What invalid states does its type exclude?
3. Is mutation absent or locally scoped?
4. Can it be interpreted independently of a backend?
5. Can a domain expert understand its signature without compiler vocabulary?

If the answers are unclear, the abstraction probably belongs one layer lower
or combines more than one responsibility.

Three additional constraints keep the type system from becoming its own form
of complexity:

- introduce a type only when it removes a demonstrated class of mistakes;
- keep generic parameters at subsystem boundaries rather than on every local;
- prefer one compositional abstraction over several partially overlapping
  convenience APIs.

## 16. Incremental migration

This program does not require a rewrite. The following sequence preserves
working behavior while tightening one boundary at a time.

### Phase 1: name the existing contracts

Implementation status: complete. Kernel writes
are named records; field inputs, outputs, and scalar parameters inhabit distinct
interned nominal types (`InputKey`, `OutputKey`, and `ScalarParam`);
destinations are `FieldBind` values over the closed `FieldRef` vocabulary plus
separate compiler-owned `Scratch` and externally-owned `User` namespaces; fusion and support inference compare typed identities;
and text is recovered only at graph/code-generation, serialization,
configuration, diagnostic, and test-presentation boundaries. A field input and
scalar parameter may not share one emitted graph spelling: tracing rejects that
collision immediately. The prepared-IR wire version was advanced to 3 so cached
artifacts cannot silently cross the identity-schema boundary. `GvKernel` and
`BuiltSource` representations are private outside their owning crates and expose
read-only compiler views. Every traced kernel carries an explicit
`NumericalPolicy` (`StrictIeee`, `FiniteOnly`, or `FastMath`); fusion rejects
mixed policies, and serialized prepared IR preserves the policy at wire version 3.

1. Replace positional manifest tuples with named records.
2. Inventory every string used as identity and classify its namespace: field
   slot, parameter, input key, output key, user symbol, or presentation only.
3. Introduce typed field, parameter, input, and output identifiers; intern the
   genuinely open user-defined namespaces.
4. Centralize conversion from typed identifiers to emitted ABI spellings.
5. Change fusion, support inference, and runtime binding to compare typed
   identities exclusively.
6. Make graph-bearing source and kernel fields private.
7. State strict-IEEE and relaxed numerical policies explicitly.

Success means ordinary consumers no longer destructure compiler representation
or depend on tuple position and string spelling. Outside parsers, serializers,
diagnostics, and renderers, creating an identity with `String` or `format!`
should be impossible or mechanically rejected.

### Phase 2: contain tracing effects

Implementation status: complete. Tracing is a scoped interpretation:
`trace(|cx| ...)`, `trace_for_domain(d, |cx| ...)`, and `trace_with(grade,
|cx| ...)` open a fresh graph, run the closure with a `TraceCx<'t>` capability
token, and return the finished `GvKernel`; an enclosing trace is restored on
both normal return and panic. The closure is generative over the invariant
brand lifetime `'t`, and every symbolic value (`Gv<'t>`, `GvMask<'t>`) carries
the brand, so escaping the trace, smuggling through a captured slot, and
cross-trace arithmetic are compile errors (pinned by `compile_fail` doctests
on `trace` and `Gv`). A per-trace generation stamp on every node handle backs
the brand at runtime: a value minted in an enclosing trace and recorded inside
a nested one panics at the recording site. The manual `begin_trace` /
`end_trace` family is deleted, minting constructors live on `TraceCx`, raw
graph access requires the token (`cx.with_trace`), and a workspace-wide
structural gate (`trace_containment`) keeps the ambient protocol from
returning. Launch grades, builder-declared support balls with derived output
support, and NodeId-based `KernelWrite` output ownership are carried through
the scoped API unchanged.

1. Add a panic-safe `trace(|cx| ...)` API.
2. Reject accidental nesting unless it is explicitly scoped.
3. Migrate manual trace pairs behind the scoped API.
4. Introduce branded symbolic values once the closure API is stable.

Success means trace lifetime and graph ownership are enforced by construction,
with no ambient protocol visible to physics authors.

### Phase 3: remove graph construction from physics

Implementation status: complete. Production `symbi-hydro` contains no
references to `Graph`, `NodeId`, or IR operation variants (gated by
`physics_graph_boundary`; test modules may still inspect compiler
artifacts). The opaque `symbi_ir::SourceProgram` carries a traced source
expression, its scalar param names, and its outputs; physics constructs one by
tracing carrier code (`SourceProgram::trace`) and composes programs inside a
later trace via `TraceCx::splice_source` / `splice_source_as_scalars`, while
the graph-level `splice_into` and accessor views remain the compiler-facing
door. The user-expression bridge interprets the `symbi-expr` DAG directly in
the carrier algebra (comparisons produce masks, conditionals select on them,
`atan` joined the carrier surface), so tracing the interpretation is what
builds the graph; additive source composition, region masking, and overlay
summation are trace-and-splice compositions with carrier arithmetic. The
geometric, gravity, immersed-body, and user source builders are all carrier
formulas. Neighbor access never appears in physics — stencils live in the
discretization layer (`cx.field_shifted` / `field_offset`), so the stencil
algebra of section 7 remains a discretization-layer concern.

1. Express source formulas over `S: Scalar`.
2. Add a stencil interface for neighbor access.
3. Move splicing and node remapping behind `KernelProgram`.
4. Make physical composition produce domain programs rather than graphs.

Success means `symbi-hydro` contains no production references to `Graph`,
`NodeId`, or IR operation variants.

### Phase 4: invert the carrier dependency

Implementation status: complete. The carrier constitution (`Scalar`, `Mask`,
`Selectable`, the algebraic laws, `SourceLoc`, and the `Dual` derivative
carrier, with the zero-panic contract and the executable carrier-law suite)
lives in `symbi-carrier`, which depends only on `symbi-algebra`; `symbi-ir`
depends on the foundation and supplies the tracing interpretation
(`Scalar for Gv`). The compiler re-export facade is deleted: every crate
imports the constitution from `symbi-carrier` directly, and the
`physics_independence` gate scans for the old spellings so the facade cannot
return. Physics is compiler-free in the dependency graph: `symbi-geometry`,
`symbi-ib`, and `symbi-hydro` depend on the foundation alone (compiler crates
appear only under dev-dependencies, for tests that inspect traced artifacts).
The program-construction and program-evaluation machinery moved out of hydro
into `symbi-source-compile` — expression interpretation, `SourceProgram`
tracing/splicing and overlay composition, the traced `StateLaw` conversion
(the `StateLawGv` extension over the physical descriptor), host evaluation of
lowered programs, and the GPU source emission/launch wiring — sitting between
`symbi-hydro` (what a physical source means) and `symbi-ir` (generic
representation and lowering), beside `symbi-discretize` (where continuous
laws acquire mesh locality). `RenderPolicy` was deleted rather than
relocated: it had no consumer.

1. Extract `Scalar`, predicates, and their structural laws into a foundation
   crate.
2. Let `symbi-ir` implement the symbolic interpretation.
3. Let geometry, hydro, and immersed-boundary physics depend only on the
   foundation.

Success means compiler independence is visible in the crate dependency graph.

### Phase 5: enforce physical structure

1. Add semantic types at primitive/conserved and centering boundaries.
2. Make frame and variance explicit in geometric interfaces.
3. Add dimensional and interval verification carriers where useful.
4. Encode admissibility and normalization phases when they remove real error
   paths.

Success means important physical category errors fail to compile or fail at a
single validated constructor.

### Phase 6: make effects compositional

1. Derive read, write, support, and stencil effects from domain descriptions.
2. Express parallel and sequential composition separately.
3. Treat fusion as a target-dependent optimization of parallel composition.
4. Verify effect inference against emitted-kernel behavior.

Success means launch transformations preserve semantics because the relevant
preconditions are explicit values or proofs, not conventions distributed among
callers.

## 17. Non-goals

This program does not seek to:

- eliminate the IR or disguise compiler work inside vague abstractions;
- encode all mathematics in Rust's type system;
- wrap every scalar temporary in a newtype;
- force CPU and accelerator execution to have identical operational behavior;
- replace measured numerical validation with type-level claims;
- build a universal compiler framework before a concrete consumer needs it.

The compiler should remain explicit and sophisticated inside its boundary. The
goal is to ensure that its complexity is paid once by compiler infrastructure,
not repeatedly by every physics contributor.

## North star

> Pure formulas at the top, explicit effects at the boundary, opaque
> representations underneath, and laws enforced at the lowest layer capable
> of enforcing them.

That arrangement preserves the power of a miniature compiler while keeping the
physics legible, the numerical assumptions honest, and the architecture open to
new equations and machines.
