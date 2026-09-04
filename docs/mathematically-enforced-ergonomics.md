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

Implementation status: in progress — the category-erasure audit is complete
and the existing frame/variance laws are pinned by compile-fail doctests
(`Indexed` in symbi-algebra: cross-variance arithmetic, orthonormal/embedded
mixing, metric-free double-contravariant contraction; `Metric` in
symbi-geometry: coordinate values at the Euclidean door, orthonormal values
at the raising/lowering doors). The audit extends the existing structures —
`PrimG`/`ConsG`, the `Indexed` frame family, the `Metric` morphisms — and
honors the recorded interior-stays-bare decision on `ConsG::mom`: frames are
typed at crossings, and formula interiors remain carrier arithmetic.

| Boundary | Current erased type | Error permitted | Enforced replacement |
| --- | --- | --- | --- |
| regime velocity/momentum frame | bare `Tensor<S, D>` in `PrimG::vel` / `ConsG::mom`; orthonormal `V_a` for flat regimes, Valencia contravariant `v^i` for GR, distinguished in prose only | a flat-regime state reaches a GR conversion (or the reverse) with its components silently reinterpreted | done: the GR regimes' `Prim` is `Valencia<..>`, wrapped where a state enters a conversion or flux door; interiors stay bare |
| face normal in `to_flux` / wave speeds | `nhat: &Tensor<S, D>`, unit length and frame/variance role by convention | a non-unit or wrong-frame vector projects fluxes | done: `Regime::Normal: FaceNormal` — flat solvers `Normalized<Physical>`, valencia regimes `Normalized<Covariant>`; `axis(k)` is the one-hot honest constructor (gated by `frame_boundaries`) |
| EOS scalar arguments | positional bare `S`: `sound_speed(rho, pre)`, `pressure(rho, e_int)` — two different second-argument meanings on one trait | swapped or confused arguments compile | done: the `Eos` doors take `Density` / `Pressure` / `SpecificInternalEnergy` / `VelocitySquared`. the `EosFor<S, E>` bound relates a state's `EnergyModel` to its lawful closure by `ClosureKind` (`EnergyEvolving` recovers from the evolved `nrg` slot; `IsothermalClosure` recovers from an externally prescribed temperature — the isothermal state truthfully has no energy slot), with `Eos::RecoveryQuantity` naming what recovery consumes: `EnergyDensity` for the gamma-law closures and `SoundSpeedSquared` for isothermal. every regime door, the adiabatic convenience conversions, and the sim/amr pairing points require the lawful relation, so an adiabatic state cannot round-trip through an isothermal closure. bare scalar storage and field boundaries are crossed only through the named `StoredQuantity` doors; the old adiabatic-layout isothermal fixtures were replaced by an explicitly typed isothermal input lane. arguments carry identity; returns stay bare scalars except the recovery quantity, whose meaning varies per closure and therefore lives in the associated type. swapped, e_int/nrg-confused, cs^2-as-energy, and unlawful-pairing calls are compile-fail-pinned |
| frame minting | `From<Tensor>` / `From<[S; D]>` on `Indexed` coerce a bare tensor into any frame implicitly | `.into()` claims a frame nobody stated | done: `From` deleted (it had zero call sites); the compile-fail doctest pins the absence |
| state construction | `PrimG` / `ConsG` / `MhdPrimG` / `MhdConsG` fields public: any literal mints a state, slots filled positionally by memory | rho and pre swapped, cs^2 written into an energy slot, an iso state minted with a fabricated pressure | done: fields are `pub(crate)`; the constructors name the closure family (`adiabatic` takes `Density`/`Pressure`/`EnergyDensity`, `isothermal` omits the absent slot), so the two arities are distinct methods rather than an overload. no model-generic public assembly door exists: generic code rebuilds from an existing state (`with_den`/`with_nrg`/... functional updates, `into_parts`) or crosses the untyped-wire door (`from_slots`, the SoA gather / trace-leaf boundary) by name. consumers read through field-named accessors; `_mut` doors remain for component-wise in-place updates, with whole-slot writes expressed as functional updates. five compile-fail pins run as doctests: the external literal, a bare pressure, a bare energy, the absent generic door, and a pressure offered to the isothermal constructor |
| immersed-body chart bridges | duplicated per-file `[Gv; 3]` cartesian/chart converters in the penalize and immersed builders, outside the `Metric` morphisms | chart-local components pass as cartesian; the two copies drift | claims typed: the immersed bridge takes/returns `Embedded`, and the body gravity crossing carries the witness (arithmetic untouched). folding the immersed trig onto the `Metric` morphisms changes float association, so that unification is deferred behind an A/B bake-diff gate |
| MHD centering | cell B is `FieldRef::PrimMag`; face B and edge EMF are scratch string keys (`bface_a`, `emf`), centering by naming convention | a face field consumed as a cell field, an EMF as a cell scalar, cross-axis face mixing | `CellCentered` / `FaceCentered` / `EdgeCentered` handles at the CT builder boundaries |
| recovered-state validity | `C2pResult { value, error }` carries a value even when recovery failed; floors regularize without a phase | `.value` read without consulting `.error` | `Admissible` / `Regularized` phases behind honest constructors |
| sweep-normal axis | integer axis convention (`vel[axes[d]]`, `dir: usize`) in kernel builders | off-axis component reads compile | typed axis where the centering work already introduces one |
| `Prim` / `Cons` aliases | second spelling of `PrimG` / `ConsG`, whose energy parameter already defaults | one type under two names (~114 alias sites) | delete the aliases |

Recorded no-action rows: the chart point `x: Tensor<S, D>` (a position/vector
confusion has no observed site), density versus deposited amount (the source
bridge already rejects the cell-volume leaf at build time), and coordinate
versus proper time (naming suffices at the current usage sites).

The resolved CT/UCT scratch inventory — every wire name whose string equality
carries identity across the discretize/substrate seam, classified by the buffer
the dispatch binder actually attaches (not by the name's suggestion). the
relative roles (`A`/`B` transverse grid axes, `P1`/`P2`/`Out` physical
components) resolve to absolute indices only through the validated `CtEdge`
descriptor (`g1`/`g2` grid axes, `p1`/`p2`/`name_k` physical components). this
table is the source for the typed scratch vocabulary and the structural gate.

| Wire name | Semantic role | Centering | Relative axis/component | Bound buffer |
| --- | --- | --- | --- | --- |
| `bface_a` / `bface_b` | staggered face B, transverse | Face | normal = A / B | `mhd.bface[g1]` / `[g2]` |
| `bface_n` | staggered face B, sweep-normal | Face | sweep axis (out-of-band `dir`) | `mhd.bface[dir]` (already `FieldRef::BFaceNormal`) |
| `b`, `bx`/`by`, `br`/`bz`, `b0`/`b1` | curl-written / chart-spelled face B | Face | write = sweep; chart pairs = axes 0/1 | `mhd.bface[dir]`, `mhd.bface[0..1]` |
| `bcell_p1` / `bcell_p2` / `bcell_out` | cell B, in-plane / out-of-plane | Cell | P1 / P2 / Out | `mhd.bcell[p1/p2/p_out]` |
| `vel_p1` / `vel_p2` / `vel_out` | cell velocity | Cell | P1 / P2 / Out | `prim.vel[p1/p2/p_out]` |
| `rho`, `pre` | cell primitives | Cell | -- | `prim.rho`, `prim.pre` |
| `flag` | FOFC troubled-cell flag | Cell | -- | the FOFC flag field |
| `fden_p1` / `fden_p2` | gas mass flux through transverse faces | Face | normal = A / B | `fields.flux[g1].den` / `[g2].den` |
| `bflux_a` / `bflux_b` | induction flux through transverse faces | Face | normal = A / B (comp p2 / p1) | `mhd.bflux[g1][p2]` / `[g2][p1]` |
| `bf_{c}` | staggered face B addressed by its own component (the face-to-cell interpolation) | Face | comp c, normal = c's grid axis | `mhd.bface[c]` |
| `fo_bflux_{c}` / `ho_bflux_{c}` | FOFC induction-flux splice: live first-order / saved high-order | Face | comp c, normal = sweep | `mhd.bflux[dir][c]` / `bflux_ho[dir][c]` |
| `bc_{c}`, `nrg` | cell B write of the interpolation; a defensive conserved-energy arm | Cell | comp c | `mhd.bcell[c]` (via `FieldRef::BCell`); `cons.nrg` |
| `wsl_p1` `wsr_p1` / `wsl_p2` `wsr_p2` | left/right signal speeds at transverse faces | Face | normal = A / B | `mhd.wave_speed_l/r[g1]` / `[g2]` |
| `emf`, `ez`, `e`, `ephi` | the edge EMF (one identity, four ABI spellings) | Edge | dual comp `name_k` | `mhd.efield[edge.slot]` (2D charts: `[0]`) |
| `e_p1` / `e_p2` | incident edge EMFs read by a face curl | Edge | incident edge A / B | `mhd.efield[edge_slots[0/1]]` |
| `e_fo` / `e_ho` | FOFC splice: live first-order EMF / saved high-order EMF | Edge | dual comp | `mhd.efield[slot]` / `mhd.efield_ho[slot]` |
| `e_n` | RK2 edge-EMF stage snapshot | Edge | dual comp | `mhd.efield_n[slot]` |

`idx`/`idy`/`id_p1`/`id_p2`/`eta` are `ScalarBind` entries (inverse widths, the
resistivity), outside field identity. `cons_mag_{k}`, `prim.mag[{k}]`, and
`bcell_{k}` are cell-centered B spellings already in the closed `FieldRef`
vocabulary. the derived `edge_*`/`h_*`/`e_*`-prefixed spellings are trace-local
SSA keys; identity lives in the wire path alone.

The recovered-state validity inventory — every representation that carries
recovery success, admissibility membership, a fallback-tier decision, or a
regularization firing, classified by what actually holds the fact. This table
is the source for the admissibility/regularization phase work; each row names
the erasure and the honest form it earns.

| Fact | Current representation | Error permitted | Honest form |
| --- | --- | --- | --- |
| host recovery outcome | `C2pResult { pub value, pub error }` (`c2p_result.rs:121`): the header promises a value "always safe to use" while `C2P_FAILURE_SENTINEL` (`:156`) forbids the failed value from entering evolution — two failure contracts in one struct. `C2pResult::err(v, ErrorCode::NONE)` mints an ok-that-claims-failure; `ErrorCode(pub u8)` mints arbitrary codes | a sentinel primitive read as physics; a fabricated or vacuous code | success mints a `Recovered` primitive (or `Admissible<_, RecoveryInterior>` only when the named recovery-interior predicate is the constructor); failure carries the code and no usable primitive, or a `Regularized` value carrying its intervention record; code minting closed |
| kernel recovery outcome | the in-band pressure channel: cone failure writes `pre = -\|D\|` (`c2p_cone_fail_pressure`) into the physical pressure field; every kernel decode is a sign-and-finiteness test on `prim.pre` (`primitive_physical_gv`, the fofc probe/select/freeze family, the wb target decode); the `p = 0` cold boundary shares the failure branch by design; a NaN sound speed is a third spelling of the same fact, halting through a non-finite dt | a genuinely negative pressure from a bad update and a cone-failure sentinel are one signal; a decode that misses one spelling (the 2-select ghost-kill bug was this) | the outcome rides its own channel at the recovery boundary (the flag field already crosses the same seam); pressure stays physics |
| the c2p error field | `sim.fields.c2p_error`: a float buffer holding `u8` bitflags. the kernel producer emits only `{0.0, 64.0}` (`c2p_status_gv`) while the host path emits the full seven-flag vocabulary; `scan_c2p_errors` merges both with no provenance, so the freeze-streak panic reports a vocabulary the kernel path can never have written. the same buffer is temporarily a 0/1 exterior mask (`substrate_mhd.rs` fofc), defended by lifetime discipline alone | a code read as a mask, a mask read as a code, a diagnostic that names flags nothing produced | one producer vocabulary per channel; mask scratch under its own name |
| troubled-cell flag | `ws.fofc_flag`, a float field written 0/1, decoded three ways: `> 0` in the splice kernels, `> 0.5` on the host, and `Add`-reduced as a cardinality; ghost-filled through the physical scalar BC machinery | the boolean, the count, and the boundary fill each reinterpret the same floats under a different convention | one decode convention behind a named accessor; the count as its own reduction |
| freeze decision | decided implicitly inside `fofc_select_gv` and independently recomputed by `fofc_freeze_probe_gv` at a different pipeline point; halt/retry keys off the recomputed count. the physicality predicate is hand-copied in five kernel builders | the two evaluations drift; the count diverges from the act | one predicate source (partially done: probe and status share `primitive_physical_gv`); the act reports itself |
| fallback-tier taxonomy | `SourceReplay` is the one typed tier; the `KernelSet::fofc` seam returns a bare `bool` meaning "replay the step", erasing the outcome taxonomy (clean pass, corrected in place, froze, inactive), which the census then reconstructs from process-global counters | the caller distinguishes outcomes by side channels | a typed stage outcome at the seam (`StageOutcome` already exists one layer up) |
| admissibility membership | membership has no value-level form: `rmhd_admissible_residuals` returns the raw `(q, psi)` pair, consumed only as masked selects inside theta computations. `rmhd_admissible_theta = 0` conflates an infeasible anchor with a binding constraint — the distinction `constraints::anchor_feasibility` was built to carry | an infeasible anchor reads as a hard-binding floor; no site can assert membership; ordinary C2P success is confused with membership in the broader Wu–Tang/temperature/magnetization family | the residual pair behind a witness-producing, law-parameterized `Admissible<T, L>` predicate at the few host boundaries that ask; `Recovered<T>` remains a distinct phase unless that named law is actually checked |
| projection magnitude | the live GR projections (`fofc_project_gr_{mhd,}_gv`) blend `(D, S_i, tau)` by a theta the kernel computes and discards; the anchor-energy raise (`rmhd_anchor_energy_with_margin`) books nothing; the census counts probe hits, carrying neither movement nor magnitude | a run reports "N cells flagged" while the injected mass/energy is unknowable | the theta/binding outputs and the `ConstraintLedger` that already exist, wired |
| multiplexed scratch | `cfl_scratch` holds, at different times, wave-speed rates, the finiteness mask, the GRMHD source theta, and the freeze mask — meanings separated by lifetime discipline and one comment; `state_finite` failure is re-encoded a third way as `lambda_max = inf` | a stale meaning read across a lifetime boundary | named scratch roles, the pattern the CT vocabulary set |

The silent-regularization inventory — sites that mutate state with no record,
distinguished from the deliberate absorbing boundaries:

- the GR admissible projections: per-cell theta computed and discarded, anchor
  energy raised unbooked (`gv/wavespeed.rs`, `admissible.rs`);
- the source-limiter theta: materialized into `cfl_scratch` for one replay,
  then overwritten, with no count (`substrate_mhd.rs`);
- the RHD `rho = D / W(clamped v^2)` sanitize: the velocity stays exact so
  `SUPERLUMINAL` still fires, and the returned density is floored with no flag
  of its own (`rhd/cons.rs`);
- the HLLD pressure-guess floor (`riemann/hlld.rs`), the HLLE fan clamp, and
  the CT diffusion-coefficient floor: iterate/coefficient guards, benign,
  recorded here as the complete set;
- the excision vacuum fill and the `r >= M/2` metric clamp: deliberate
  absorbing-region semantics, masked out of the census, with no accounting of
  the absorbed mass/energy.

The live GRMHD admissible-boundary projection now books its evidence: the
production kernel emits a typed `ProjectionReceipt` (theta and the (D, tau)
state deltas), the substrate returns it through the `FofcReport`, and the sim
accumulates it over the accepted step into a `ProjectionLedger` (`symbi-sim`,
thread-local, discard-on-retry). The ledger separates the signed net injection
from the gross absolute intervention and is read through
`projection_ledger_report`. It is evidence about production physics, not a
correction — the receipts record what the projection already did, and the
candidate writes are byte-identical whether the diagnostics are on or off.

The broader constraint algebra (`symbi-hydro/src/constraints.rs`: the concave
`c(U) >= 0` family with the projection axioms, `WuTangAdmissibility`,
`TemperatureFloor`, `MagnetizationCeiling`, `DensityFloor`, per-member theta
attribution, `anchor_feasibility`) stays unwired: `constraint_projection_gv`
is baked and oracle-tested, its dispatch wrapper has zero callers,
`ConstraintParams` holds inert defaults with no config path, and the
`constraint_theta` field is allocated and untouched. Activating the floors
changes physics and is a separate decision; the projection ledger books only
the interventions the live admissibility projection already performs. The
anchor convention that blocked wiring was resolved by
a controlled A/B measurement of the two candidate anchors on the same firing
projection. The Eulerian rebuild — the p2c reconstruction from the stage-input
primitives paired with the candidate cell B — is admissible by construction:
its anchor-energy raise is a millionth of the energy the projection itself
moves (raise/segment-energy of 1.2e-6 to 4.2e-6, stable across 32^3/48^3/64^3
and across plasma beta). The stage-input anchor pairs the stage conserved state
with the constrained-transport-advanced B, a hybrid that sits on the
admissibility boundary: its raise is a percent-level fraction of the projection
energy (1.2% to 2.2%) and grows with resolution. Both survive in the accessible
regime because the raise repairs the marginal stage anchor, so the anchor is a
numerically rescued hybrid while the Eulerian rebuild supplies the lawful
anchor. The ledger therefore books behind the Eulerian rebuild — the convention
production already runs. The projection is a truncation effect — it fires only
in the turbulent near-horizon flow of a 3D torus and vanishes as the grid
refines; a 2D axisymmetric torus never trips it, and there the two anchors agree
to machine precision. The module's usage
example also names `joint_projection_theta`, which does not exist; the live
name is `joint_theta`. There is currently no live magnetization ceiling,
temperature floor, or evolved-state density floor anywhere in production —
the only floors that run are the projections, the excision fill, and the
iterate guards above.

The recovery outcome algebra follows from that inventory. There are two
interpretations of the same validity law, because host control flow and a
traced accelerator program do not have the same carrier. They share the
named acceptance predicate and its meaning; they do not pretend to share an
operational shape or a diagnostic vocabulary they cannot both compute.

On the host, recovery uses Rust's existing closed sum rather than inventing a
second one:

```rust
pub type Recovery<T> = Result<Recovered<T>, RecoveryFailure<T>>;

#[repr(transparent)]
pub struct Recovered<T>(T);

pub struct RecoveryFailure<T> {
    issues: RecoveryIssues,
    candidate: DiagnosticOnly<T>,
}
```

`Recovered<T>` has a private field and is minted only after the named C2P
interior predicate succeeds. It is deliberately not called `Admissible<T>`:
convergence plus positive density/pressure and subluminal velocity does not
prove the wider Wu--Tang, temperature, magnetization, or application-specific
constraint family. Expected numerical rejection is ordinary `Result` control
flow, not a panic. `Recovered::into_inner` is the single transition back to an
ordinary primitive after the caller has handled the outcome.

`RecoveryFailure` carries a candidate only for diagnostics. The candidate is
not exposed as `T` or `&T`, does not dereference, and has no `into_inner`; named
diagnostic projections may reveal scalar residuals or a formatted snapshot,
never a primitive that can re-enter evolution. This preserves useful
Newton/cone evidence without letting a failed primitive silently become
physics. Early failures that have no meaningful candidate carry a dedicated
diagnostic placeholder rather than a fabricated physical state.

`RecoveryIssues` is a private-bit, non-empty set. Its public vocabulary is the
seven existing named issues; construction occurs through those constants and
`merge`, so neither an unknown bit pattern nor an empty failure can be minted.
The successful branch carries no issue set at all. Serialization to the
diagnostic `u8` representation is an explicit boundary operation, not the
host's semantic type.

Regularization is a policy applied to a rejection, not a third spelling of
success:

```rust
pub struct Regularized<T> {
    recovered: Recovered<T>,
    receipt: RegularizationReceipt,
}
```

Only a named `RegularizationPolicy` may consume a `RecoveryFailure<T>`. It
records the original issues and every intervention, then reruns the recovery
interior predicate before producing `Regularized<T>`. A policy that cannot
prove that predicate returns the rejection. `Regularized::into_recovered`
forgets provenance explicitly; no blanket conversion does. Iterate guards
(bracket clamps and coefficient floors), absorbing-region fills, and physical
state regularizations are separate receipt kinds rather than one misleading
"floor" category. The iterate guards on the carrier path — the HLLD
pressure-guess floor and the extremal fan clamps — are permanently
unreceipted: they live inside branch-free solver interiors where a receipt
would demand a status output per guard and change the graph. They are benign
by construction (they perturb an iterate or a coefficient, never the state),
they are enumerated in the silent-regularization inventory above, and that
enumeration is their record.

For a symbolic carrier, the same law is a product because a Rust enum would
branch while tracing:

```rust
pub struct TracedRecovery<T, M> {
    candidate: T,
    status: KernelC2pStatus<M>,
}

pub struct KernelC2pStatus<M> {
    accepted: M,
}
```

`M` is the carrier predicate (`GvMask` when tracing). Candidate formulas remain
branch-free. The one named predicate builder produces `KernelC2pStatus`; at
materialization, candidate fields and a dedicated status channel are written
separately. A failed pressure is never itself the control signal. The kernel
does not counterfeit the host's seven-cause diagnosis: it carries only the
accept/reject fact it actually computes. Backend lowering may represent that
fact as a scalar 0/1 field, but only the typed channel renderer knows that
representation.

This gives the fallback ladder three distinct channels:

- `C2pStatus`: the recovery producer's typed accept/reject fact;
- `TroubledCell`: the one boolean convention consumed by first-order splicing;
- `FreezeApplied`: written by the select that actually freezes a cell.

They do not share storage. Counts are named reductions of masks, not alternate
interpretations of the mask field. The select reports its act through
`FreezeApplied`; the independent freeze-probe kernel and its copied predicate
can then be deleted. Multiple evaluations of the shared predicate remain
lawful when they classify genuinely different candidate states, but the act is
never inferred by recomputing it later. Whether `TroubledCell` is a fresh
classification or a materialization of `C2pStatus` is a fact about the
pipeline, audited once during the channel work: if nothing mutates the
primitives between the recovery and the probe within a substage, the two
channels carry one fact and the probe is a rename; if a ghost fill or source
intervenes, the probe classifies a later state and is legitimately its own
evaluation. The answer is recorded on the channel's documentation, so the
channel work knows whether it deletes an evaluation or preserves one. The
audit ran with the channel work: the stage driver runs c2p then fofc directly,
nothing mutates the primitives between the recovery's writes and the probe's
reads, so today `TroubledCell` is a re-encoding of `C2pStatus`. The channels
stay distinct because their consumers and encodings differ and the ladder may
one day flag cells the recovery accepted; the probe's re-evaluation is
therefore deletable in favor of a status-channel decode when the mask
unification lands.

Moving the freeze count from the recomputing probe to the act was audited
against the select kernels themselves: both selects gate on the same
physicality predicate over the same spliced candidate the probe classified,
and both parachute flavors — the bare stage input and its body-evolved twin —
replace the candidate and waive the cell's conservation, so each is the freeze
act. The act count therefore equals the old prediction cell for cell; the
parachute chooses the value a frozen cell holds, never whether it froze. No
census gate re-baselines; the number's producer moved, its meaning held.

At the host seam the ladder returns a typed report rather than a boolean:

```rust
pub struct FofcReport {
    troubled: CellCount,
    frozen: CellCount,
    replay: SourceReplayOutcome,
    decision: FofcDecision,
}

pub enum SourceReplayOutcome { SharedRedo, ConservativeReplay }
pub enum FofcDecision { Accept, RetryStep }
```

The existing `StageOutcome::{Accepted, RetryStep}` remains the timestep-level
decision. Folding `FofcReport::decision` into it is an explicit policy in the
stage driver; the counts report what the actual select did, and process-global
counters are observations of the report, not a second source of truth. The
source-replay branch records whether the normal shared redo or conservative
replay actually ran; it is not mislabeled as a request made after the fact.
A regime whose ladder is inactive constructs its report through the one
inactive-pass constructor (zero counts, the shared-redo path, `Accept`), so an
inactive pass cannot fabricate a replay claim or a count.

General admissibility is a separate, law-indexed proof:

```rust
pub struct Admissible<T, L> {
    value: T,
    witness: L::Witness,
}
```

Only `L::validate` constructs it. `RecoveryInterior` may be one law; the
Wu--Tang family and any configured temperature/magnetization family are other
laws. A witness-producing residual evaluation distinguishes membership, a
binding constraint, and an infeasible anchor. The existing constraint family
must not be wired merely to obtain this type: its stage-input versus eulerian
anchor convention remains a physical decision and blocks projection, not host
recovery safety.

The implementation sequence is intentionally asymmetric:

1. Close `RecoveryIssues`; replace `C2pResult` at host regime doors with
   `Recovery<T>` and migrate every `.value`/`.error` consumer. This is a type
   migration with no kernel or floating-point change.
2. Introduce the typed kernel product and dedicated `C2pStatus`,
   `TroubledCell`, and `FreezeApplied` channels. Preserve candidate arithmetic
   and prove prepared-IR changes are limited to the new status plumbing — the
   prepared-graph golden diff at this step touches only the added status
   output. The cone-failure sentinel is still written into the pressure
   candidate (that write is part of the preserved arithmetic); only the decode
   moves to the status channel. Deleting the write would change the graph.
3. Make the correcting select emit `FreezeApplied`, delete the recomputing
   freeze probe, unify mask decoding, and return `FofcReport` from the seam.
4. Add law-indexed host admissibility witnesses for the residual queries that
   already exist. Do not activate new floors or the dormant constraint family.
5. Resolve the anchor convention by a separate physical experiment; only then
   may projection receipts and `ConstraintLedger` become production state.

Each step has its own stop condition. Host compile-fail pins reject direct
failed-candidate extraction, empty/unknown failure codes, and use of
`Recovered<T>` as a broader `Admissible<T, L>`. Carrier tests pin one predicate
source, the kernel's deliberately smaller status vocabulary, and distinct
channel identities; prepared-IR and AOT equivalence pin unchanged candidate
arithmetic. Stage tests pin that reported freeze counts equal the cells
actually selected and that retry decisions consume the typed report exactly
once.

1. Add semantic types at primitive/conserved and centering boundaries.
2. Make frame and variance explicit in geometric interfaces.
3. Add dimensional and interval verification carriers where useful.
4. Encode admissibility and normalization phases when they remove real error
   paths.

Success means important physical category errors fail to compile or fail at a
single validated constructor.

### Phase 6: make effects compositional

The objective is to make the program's existing read/write/locality contracts
truthful, composable, and compiler-enforced, while preserving generated
arithmetic, ABI names, buffer order, and execution behavior. A kernel program
must have one authoritative effect description derived from its actual IR — not
a second handwritten account that can drift — and composition (sequential
ordering, parallel independence) must reason over that description with
structured evidence, never a boolean "compatible".

#### Stage 1 audit — the effect truth today

The single most important finding: a kernel's read-set and write-set exist
conceptually as `GvKernel.field_inputs` + `KernelWrites` (the latter is already
named "the named write-effect manifest", `symbi-ir/src/gv.rs:224`), but that
notion is **re-derived independently three times**, in two different identity
namespaces, and the write half is not even owned by the kernel:

1. at bake, `render.rs:271-313` recomputes the input/output split and buffer
   order as an `is_output: bool` per `FieldBinding`;
2. at execution, `symbi-exec/src/policy.rs:150-170` (and again inline at
   `:241-254` for the parallel cover path) enforces aliasing over runtime
   pointer identity (`as_ptr()`);
3. at fusion, `gv.rs:856-903` computes write-conflict and inter-dependence over
   typed `FieldBind` identity.

`KernelWrites` is a free-standing `Vec` threaded alongside `GvKernel` by caller
convention (`try_fuse(a, a_writes, b, b_writes)`, `noop() -> (GvKernel,
KernelWrites)`); nothing binds a kernel to its writes, and nothing checks that a
`KernelWrite.value` NodeId belongs to the trace's dataflow or that its
`destination` is actually produced. This is the largest drift surface and the
keystone the phase must fix.

The drift and the authority are at two different times, and both findings hold:
the `is_out` split at `render.rs:271-313` makes read/write **direction**
authoritative, but only *after* the sibling `KernelWrites` has been assembled
into the manifest — it is downstream of the construction-time pairing, so it
inherits whatever writes the builder handed back. Dispatch therefore generally
consumes authoritative manifest direction, while kernel construction can still
pair a graph with missing, foreign, or incorrectly ordered writes with nothing
to object. Folding writes into the kernel closes the construction-time gap the
manifest cannot see.

The inventory, one row per effect fact:

| effect fact | authoritative source | duplicates / re-derivations | exact or conservative | identity namespace | drift risk | proposed Phase-6 owner |
|---|---|---|---|---|---|---|
| pointwise reads | `field_inputs: Vec<(InputKey, FieldBind)>` (`gv.rs:45`, deduped first-seen) | graph `Op::Load`/param nodes | exact | typed `FieldBind` (but `InputKey`/`LoadAt Symbol` are interned strings, rejoined by string equality `gv.rs:680`) | low | kept on trace, projected into the effect set |
| shifted reads | `Op::LoadAt(Symbol, Vec<NodeId>)` with exact integer offsets (`graph.rs:498`; builders `gv.rs:493,529`) | `stencil_reach` coarsens to per-axis `|offset|` (`passes/stencil_reach.rs`); `TileSpec.halo` hand-declared | graph exact; reach conservative (`Unbounded` on scaled/select/runtime index) | offsets are ints; field key is a `Symbol` | halo/tile declaration can silently understate real reach — no gate links them | `Read(Resource, ReadFootprint)`: exact offset where the index statically resolves, else bounded/unbounded reach |
| writes | `KernelWrite { key, destination: FieldBind, value: NodeId }` (`gv.rs:204`), returned as a sibling `Vec` | `is_output` at bake (`render.rs:273`); scalarizer keeps only `value` | exact destination, **unchecked value** | typed `FieldBind` | **high** — not owned by the kernel, not validated against the graph | fold `KernelWrites` **into** `GvKernel`; the write-set |
| in-place (RMW) | implicit: a field in both `field_inputs` and `field_writes` | re-derived by set-intersection at `try_fuse` (`gv.rs:882`) and `support_infer` (`:363`); folded to `is_output=true` at bake | exact intersection | typed `FieldBind` | recomputed per consumer; no marker on `KernelWrite`/`FieldBinding` | a first-class RMW access, so no site re-derives it |
| locality / support | `output_support` declared (`gv.rs:604`) then derived (`support_infer.rs`); `stencil_reach` for reach | `TileSpec` (hand-declared, `infer_tile_spec` returns it unchanged — a misnomer, `gv.rs:670`) | derived exact where affine, else `Everywhere`/`Unbounded` (fail-safe) | `ParamExpr` over scalar manifest | `TileSpec` halo vs actual reach | the locality projection of the effect set |
| reductions | none at `GvKernel` level; three unrelated notions — `graph::ReduceOp` (in-kernel axis, `graph.rs:425`), `emit::ReductionOp` (launch whole-field, `emit.rs:272`), `engine.rs` segmented (`:304`) | census "map" traces as ordinary pointwise writes; the fold is separate untraced machinery | host exact; device segmented uses order-nondeterministic atomics (`engine.rs:300`, self-reports `ReductionOrder`) | `ReductionOp` / bucket index | three enums, "never convert" (`emit.rs:266`) | deferred launch-level work — outside the initial kernel-access algebra unless a common semantic carrier is proven |
| aliasing rule | prose + asserts at `policy.rs:150-170`; re-implemented at `:241-254`; fusion analog `gv.rs:856-903` | three independent derivations | conservative (pointer HashSet; can't see partial overlap) | **two namespaces**: runtime pointer at exec, `FieldBind` at fusion | high — same law spelled three times | one alias/dependence checker consuming the effect set; the pointer check stays a runtime backstop |
| dependence (fusion) | `WriteConflict` (Waw) + `InterDep` (Raw and War lumped together) + in-place exemption (`gv.rs:856-903`) | the only true read/write-set dependence analysis in the tree | exact over declared paths; order-agnostic within a stage | `FieldBind` | War and Waw are not distinguished | refine `InterDep` into `Raw`/`War`/`Waw` |
| resource identity | `FieldRef` (closed physical vocab), `ScratchKey`/`CtScratchKey` (typed CT scratch from Phase 5), `ScalarRef` | scalar params degrade to interned strings **inside `GvKernel`** (`scalar_params: Vec<ScalarParam>`, `gv.rs:47`); census/source scratch mint `format!` strings → `ScratchKey::Free` | — | typed core with open `Free`/`User`/`Spec` tails | `FieldBind::from_path` classifies an unknown (misspelled) path as `Scratch` with no error (`field_ref.rs:427`) | reuse `FieldBind`/`CtScratchKey` as the effect's `Resource`; the string tails are the seam to close |
| source composition | `build_total_source` `Add`-fold (`simulation_laws.rs:378`); order = `overlays()` `Vec` chain (`:326`) | `SourceProgram.params` = "first mention" order | semantically order-free (`Add`), structurally order-dependent | param **strings**, `target_field: &str` | ABI-visible param order = `Vec` order; the fused path already consumes only the first family (`:155`) | Stage 4 validates composition using effects while preserving current source/param order; canonical source ordering is a separate future ABI/bake change |
| stage order | `STAGE_PIPELINE` + `fold_stage` (`stage.rs:144,287`), folded by all drivers | none for the stage sequence; the per-**step** scaffolding is hand-copied per driver (`evolve.rs:225`, `hierarchy.rs`, `decomp.rs`) | exact | — | per-step extras drift, guarded only by `stage_pipeline_law.rs` | the per-step scaffolding is the remaining un-unified sequence |
| launch ordering | implicit caller call-order; **no scheduler, DAG, token, or barrier in `symbi-exec`** | — | sequential program order | none | — | new: sequential/parallel composition as a value, ordering unchanged |
| host/device transfer | `MemorySpace` models allocation only; `SharedHandle` dirty-bit API exists (`handle.rs:57`) but is **dead code** (no consumers) | implicit unified-memory coherence + manual `ctx_sync` (`engine.rs:547`) | implicit/none | `SharedHandle` Arc (unused) | coherency is aspirational, unenforced | out of scope for the kernel effect algebra; the observable one is the device reduction's reported non-reproducibility |
| host-side observable state | three inconsistent representations: projection ledger (thread-local `Cell<Book>`, typed, `projection_ledger.rs`), FOFC/guard census (process-global `AtomicU64`, untyped, **not run-scoped**, `fofc.rs:44`), horizon ledger (owned per-body, typed, `driver.rs:260`) | — | — | typed / untyped / typed | the FOFC globals are shared across concurrent runs | a typed, run-scoped observable channel; the FOFC atomics are the reclassification target |

Direct answers to the investigation questions:

- **`GvKernel` exposes enough IR to derive effects exactly?** Not by itself.
  Reads and their exact offsets are recoverable from `graph()` + `field_inputs()`;
  writes are a sibling `Vec`; reductions have no marker; `infer_tile_spec`
  does not infer. Extraction is exact where the graph is intact and conservative
  (fail-safe wide) where a bound cannot be proven.
- **Scratch alias rules — IR, executor, or handwritten dispatch?** All three
  (`is_output` at bake, pointer asserts at exec, `FieldBind` sets at fusion), no
  single authoritative object.
- **Reductions vs pointwise?** Indistinguishable at the IR level; three
  unrelated reduce notions live in emit/exec, and whole-field vs segmented are
  already separate machinery with separate result types.
- **Source composition depends on incidental vector order?** Semantically no
  (`Add`), structurally yes (concatenation → first-mention param manifest), and
  the fused path already leaks order.
- **Shifted reads — exact vectors or coarse reach?** Exact integer offsets in
  the graph; the derived reach coarsens to per-axis unsigned magnitude.
- **In-place needs an explicit RMW effect?** It has none today; it is
  re-derived by intersection at every consumer.
- **Any parallel execution today?** None. The exec layer is sequential
  caller-ordered; the only parallelism is rayon fork-join over a proven-disjoint
  cell partition within one launch. Phase 6 establishes the proof algebra, not a
  scheduler.
- **Phase 5 CT scratch supplies scratch identity?** Yes — `CtScratchKey` is a
  complete typed identity already used in `FieldBind`/`try_fuse`, the correct
  substrate to adopt; the gap is census/source scratch minting raw strings.

#### The proposed constitution

The hypothesis model holds, with repository-driven refinements. The trace graph
type is `GvKernel`; the audit found no `KernelProgram`. The construction audit
settles the ownership question: `trace<R>(...) -> (GvKernel, R)` (`gv.rs:389`)
is generic over the closure's return, so a `GvKernel` is produced paired with
whatever `R` the trace yields — `KernelWrites` for a kernel, but a
`SourceProgram` or other value for a source or boundary trace. Writes are
therefore **not intrinsic to every `GvKernel`**, so the owner is a new opaque
`KernelProgram { kernel: GvKernel, writes: KernelWrites }` rather than an owned
`GvKernel` field. The invariant the phase establishes is that **no public
compilation or fusion door accepts a graph and an independently supplied write
vector**. `Resource` is the existing typed
`FieldBind` (physical `FieldRef` ∪ `ScratchKey`/`CtScratchKey` ∪ the
`User`/`Free` tails), never a string; scalar parameters are a distinct resource
class. A read's displacement is exact only where the `LoadAt` offset statically
resolves to a constant; otherwise stencil analysis yields a bounded reach or
`Unbounded`, so a read carries a `ReadFootprint`, never an assumed exact offset.

```rust
struct KernelProgram {     // the opaque owner: a graph and its writes, paired once
    kernel: GvKernel,
    writes: KernelWrites,
}

struct Effects {
    accesses: AccessSet,   // canonical, normalized, deduped
    locality: Locality,    // derived from read footprints + support
}

enum Access {
    Read(Resource, ReadFootprint),
    Write(Resource),
    ReadWrite(Resource),   // a normalized VIEW; its Read and Write facts stay queryable
}

enum ReadFootprint {
    Point,                 // pointwise
    ExactOffset(Offset),   // a statically resolved constant displacement
    Bounded(Reach),        // affine but not constant: a per-axis reach
    Unbounded,             // scaled / select / runtime index — analysis could not bound it
}

enum Dependence {
    Raw { resource: Resource }, // refines fusion's InterDep
    War { resource: Resource },
    Waw { resource: Resource },
    Unknown { reason: UnknownReason },  // genuinely unresolved footprint or identity
}

enum Composition { Sequential, Parallel }
```

The initial kernel-access algebra models only the per-kernel field footprint —
reads, shifted reads, writes, and in-place — of a pointwise-or-stencil kernel.
Reductions stay out of it. The three meanings of "reduce" the audit found are
genuinely distinct and must not collapse into one enum: `graph::ReduceOp` is
kernel-internal compute (an axis fold that lowers to loop code, whose access
footprint is still fully described by its `ReadAt` reads and its `Write`);
`emit::ReductionOp` whole-field and the `engine.rs` segmented fold are separate
launch-level morphisms that never trace as `GvKernel`s at all. A reduction
effect kind — and the segmented atomic-add non-reproducibility as an observable
effect — enters the algebra only if the final audit proves a common semantic
carrier for them, and even then whole-field and segmented stay distinct.
Forcing all three into one `Reduce` variant now would manufacture the false
unity Phase 6 exists to remove.

Four refinements the audit forces:

1. **Writes pair with the graph once, behind an opaque owner.** Introduce
   `KernelProgram { kernel, writes }` (or make writes an owned `GvKernel` field
   only if the construction audit proves outputs are intrinsic to every graph).
   The invariant is that no public compilation or fusion door accepts a graph
   and an independently supplied write vector, so the three re-derivations
   become projections of one value and no caller can pair a graph with missing,
   foreign, or misordered writes.
2. **In-place is a first-class `ReadWrite` — a normalized view, not lost
   information.** The underlying `Read` and `Write` facts stay queryable, so
   dependence classification still distinguishes RAW, WAR, and WAW involving an
   in-place kernel (two in-place accesses on one resource carry multiple
   dependencies, not one). `try_fuse` and `support_infer` stop reconstructing
   in-place by set-intersection; the fusion "same-stage reader wants the
   pre-state" exemption is expressed once, on the access.
3. **`Unknown` is reserved for genuinely unresolved footprint or identity, not
   for an open vocabulary.** `ScratchKey::Free("a")` and `User("a")` have exact
   equality identities and compose by exact match; treating them as
   conflict-with-everything would cripple lawful fusion. `Unknown` covers only
   an unbounded/unresolved read footprint or a resource whose identity cannot be
   resolved. The real hazard — `FieldBind::from_path` silently classifying a
   misspelled physical field as `Scratch` — is closed at the parsing boundary:
   Stage 2 makes `from_path` fail loud on an unrecognized physical-looking path,
   rather than making the algebra pessimistic.
4. **The identity namespace unifies on `FieldBind`.** Fusion already uses it;
   exec's pointer check remains as the final runtime backstop, not a second
   source of truth. The host-side observable transactions (projection ledger,
   FOFC census, horizon ledger) are a related concern at the driver layer, not
   the kernel effect algebra — noted for a run-scoped typed channel, with the
   process-global FOFC atomics as the reclassification target, but not folded
   into the kernel `Effects`.

Physics keeps describing physics; the effect set is derived, immutable, and read
through `program.effects()`; users never enumerate read/write sets.

#### Backend portability as a checked boundary

Maximal portability means that physics, tracing, effect analysis, and prepared
IR do not know whether the eventual target is CPU, CUDA, HIP, Metal, SYCL, or a
future backend. A backend declares its capabilities and lowers only programs it
can represent. An unsupported program must fail before code generation with a
structured capability error; it must never reach a renderer crash, acquire a
silent fallback, or change numerical policy implicitly.

The intended boundary is:

`physics laws -> carrier-generic trace -> KernelProgram/Effects -> prepared IR
-> capability validation -> backend lowering and runtime`

A backend capability description should cover at least scalar types, address
spaces and memory/coherence rules, atomics, whole-field and segmented reduction
modes, dynamic iteration and control-flow limits, workgroup/local memory,
launch limits, and numerical policy. Adding a backend should normally require a
capability declaration, renderer/compiler adapter, runtime/memory adapter, and
backend-specific conformance tests — not changes to physics, hydro, geometry,
or discretization crates. Metal may lower to MSL with its explicit address-space
and threadgroup limits; SYCL may lower through generated C++/SYCL and its
external toolchain. Neither model belongs in the shared IR unless it expresses
a genuinely backend-independent semantic fact.

The seven principal portability risks split into removable leaks and
irreducible target differences:

1. **CUDA/HIP syntax and launch assumptions:** remove them completely from the
   shared layers; isolate them in backend lowering and build integration, with
   structural gates against their reintroduction.
2. **Unified-memory assumptions:** hardware differences cannot be removed, but
   they can be completely contained behind an explicit allocation, transfer,
   coherence, and ownership contract.
3. **Implicit host/device synchronization:** remove the implicitness by making
   transfers, dependencies, dirty state, and synchronization explicit runtime
   effects. The backend-specific mechanism remains internal to the adapter.
4. **Reduction and atomic semantics:** ordering, reproducibility, and available
   atomics genuinely differ. Keep whole-field and segmented modes distinct,
   declare their guarantees, and reject unsupported combinations rather than
   inventing false common semantics.
5. **Dynamic loops, stack arrays, and control-flow limits:** target limits are
   irreducible; validate the prepared program against declared capabilities and
   return a structured error before lowering.
6. **Floating-point and fast-math differences:** cross-architecture bit identity
   is not generally available. Encode the numerical policy and required
   guarantees explicitly, test the promised laws/tolerances, and reject a
   backend that cannot honor them.
7. **Scattered `cfg(cuda)` / `cfg(hip)` decisions:** remove them from upper
   layers in favor of capability queries and backend traits. Target-specific
   conditionals remain only inside backend modules and toolchain integration.

Thus the physical and compiler-facing risks can be removed as silent failure
modes. The remaining hardware and numerical differences cannot honestly be
abolished, but they can be confined to one typed, capability-checked boundary.
The acceptance test for a new backend is that it either executes the declared
semantics or refuses the program explicitly, while the prepared program,
effects, binding identity, and physics remain unchanged.

#### Recommended stage boundaries

- **Stage 2 — canonical effect extraction (lowest crate: `symbi-ir`).** Pair
  writes with the graph behind `KernelProgram`; derive `Effects` from `graph()`
  + `field_inputs()` + the now-owned writes, recording each read's `ReadFootprint`
  (point / exact offset / bounded / unbounded) truthfully rather than assuming an
  exact offset. Delete the redundant handwritten summaries where derivation is
  exact — the bake `is_output` split and the fusion in-place recomputation become
  projections. Pin exact extraction for pointwise, stencil, in-place, and scratch
  kernels. Extraction must not alter prepared IR or generated code (verify by
  prepared-IR byte identity). Make `from_path` fail loud on an unrecognized
  physical-looking path.
- **Stage 3 — dependence algebra.** Pure `Sequential`/`Parallel` composition;
  `Raw`/`War`/`Waw`/`Unknown` classification; shifted-read footprint union;
  identity and associativity laws; structured evidence naming the resource and
  dependence kind. Refine fusion's `InterDep` into the typed kinds. Schedule
  nothing differently.
- **Stage 4 — migrate the composition doors.** Move `try_fuse`, the exec
  aliasing checks (`policy.rs`), and the handwritten positional buffer `Vec`s
  (`dispatch.rs:1718/2265/2601/2707`) onto the one effect set, and validate
  source-program composition using effects while preserving the current source
  and parameter ordering. Delete the parallel handwritten dependency logic only
  after equivalence. Keep execution
  order; do not activate the dormant intra-stage parallelism the algebra could
  now justify.
- **Stage 5 — enforcement and closure.** Structural gates forbidding new
  stringly resource identities or parallel metadata outside the constitution;
  prepared-IR byte identity, AOT candidate/kernel equivalence, unchanged
  manifests/binding order, checks for every enabled backend (CPU execution plus
  CUDA and HIP compilation/emission today), single-grid/decomposed/refined
  ordering equivalence, aliasing-rejection and reduction and
  source-composition tests, workspace all-targets, focused batteries. Where
  generated text differs only from the known `unswitch` bake nondeterminism,
  compare prepared IR and explain the downstream noise rather than fixing
  unswitch here.

Success means launch transformations preserve semantics because the relevant
preconditions are explicit values or proofs, not conventions distributed among
callers.

#### Stage 2 — as implemented

The graph/write pairing and the effect algebra landed together in `symbi-ir`,
with the builder and door migration reaching every consuming crate.

- **`KernelProgram { kernel, writes }`** is the opaque owner. Construction
  validates that every write root is a node in bounds for the graph (a bound, not
  a provenance proof — a single trace produces the graph and its writes together)
  and that both the output keys and the write destinations are unique, so output
  aliasing is rejected at construction. A program with no writes is lawful;
  `has_no_outputs` reports that count, while the proven structural identity of
  fusion is the empty-graph `KernelProgram::noop(grade)`. `trace_kernel` /
  `trace_kernel_for_domain` are the chokepoints a builder returns through, so a
  graph and an independently supplied write vector never travel together in a
  public signature.
- **Door migration.** Every kernel builder in `symbi-discretize` (209 functions)
  returns `KernelProgram`. `try_fuse` takes and returns `KernelProgram`. The
  compilation and emission doors — `symbi-jit::compile_gv_kernel` /
  `compile_gv_kernel_prec` and the substrate `gv_kernel_to_ir` — take a
  `&KernelProgram`; the AOT bake's `emit_gv` takes a `&KernelProgram` and rebinds
  the graph and writes at its top, leaving the emission body byte-identical.
  Where a consumer compiles a subset of a kernel's writes (the census value-only
  map), it constructs an explicit sub-program. The `GvKernel`/`KernelWrites`
  tuple no longer appears in any public signature across the workspace.
- **`Effects`** is the normalized, deduped access set. `Access` distinguishes
  `Read(Resource, ReadFootprint)`, `Write(Resource)`, and the in-place
  `ReadWrite(Resource, ReadFootprint)` view whose read and write facts stay
  separately queryable. `Resource` is the born-typed `FieldBind` (typed `Ref`,
  compiler `Scratch`, external `User`), each with exact equality — resource
  identity round-trips through the value, never through the spelling.
  `Dependence` names `Raw`/`War`/`Waw` per shared resource; two in-place kernels
  over one resource carry all three. The access set is sorted over structural
  identity: an outer tag for the `FieldBind` variant, an inner tag splitting
  `Scratch` into its typed `Ct` and open `Free` arms, and a per-arm structural
  discriminant (the derived `Debug` of a `FieldRef` or `CtScratchKey`, or the raw
  string of an open name) — never the rendered `name()`, which a `Ct` wire and a
  `Free` of the same spelling share. So distinct identities never collide in the
  key and equal semantic sets compare equal regardless of construction order. When
  several input keys alias one resource, their footprints join by lattice
  (`Point` is the identity, bounded reaches join componentwise by axis, a
  dimensionality mismatch or any unbounded axis gives `Unbounded`), never
  collapsing to `Unbounded` on any disagreement.
- **Footprint precision follows analysis state.** `KernelProgram::effects()`
  reports every read as `Unbounded` — the sound conservative element — because
  the per-axis reach lives in the `FieldLoadAt` index expressions the scalarizer
  produces. `AnalyzedKernelProgram::analyze(program)` scalarizes and measures
  each read through the single authoritative `stencil_reach`, sharpening the
  footprint to `Point` (a center-only read), `Bounded(reach)`, or `Unbounded`,
  and retains the program with its measured effects (the scalarized artifact is
  consumed by the measurement — it is an analysis of the program, not a prepared
  executable). The unprepared program never claims a precision it has not
  measured.
- **Fusion projects from the effect set.** `try_fuse` derives its write-write
  conflict and its read-after-write independence hazard from
  `a.effects().dependences_into(&b.effects())` and `effects().in_place()`,
  retiring the hand-recomputed write-path and input-path sets. The nine fusion
  laws hold unchanged.

The pairing preserves the emitted artifacts. Baking the full AOT kernel set
(5369 files) at this stage and at the pre-stage baseline and comparing them:
the 2684 prepared-IR blobs (`*.ir.json`, carrying the serialized manifests and
the input/output binding order) are byte-identical, with no kernel added or
removed. The 269 generated CPU sources (`*_generated.rs`) that differ are all
body-source-fused kernels, and the difference is the coordinate-map log-spacing
select's unswitch collapse — which axis's guard the renderer folds first, a
`HashMap` tie-break seeded per process. Re-baking this same stage twice differs
in 279 such files, more than the 269 across the stage boundary, so the render
noise is bake-to-bake and independent of the change; the byte-identical prepared
IR is the semantic proof, and the generated-source difference is that known
unswitch nondeterminism, not a change in arithmetic or ordering.

Two decisions departed from the constitution above and want review:

1. **`ReadFootprint` omits `ExactOffset`.** The single authoritative reach
   analysis (`stencil_reach`) coarsens each read to a per-axis `|offset|`
   window; it never yields a signed exact displacement. A footprint variant that
   nothing populates is a promise the code cannot keep, and populating it would
   require a second reach analysis over the raw graph — the drift-prone duplicate
   the first law forbids. The derivable footprints are `Point` / `Bounded` /
   `Unbounded`.
2. **`support_infer::derive_output_support` stays as it is.** It propagates the
   tagged support balls to a kernel-level output support — a support-region
   question, distinct from the read/write dependence the effect set answers — so
   it is not a re-derivation of the effect algebra and gains nothing from
   projecting through it.

#### Stage 3 — as implemented

The composition algebra lives in `symbi-ir/src/composition.rs`, additive over
Stage 2's `Effects`. It represents ordered and independent composition as
values with structured proof; fusion, scheduling, launch order, builders, and
emitters are untouched, and nothing executes differently.

- **`Composition`** is the owner: a private tree whose leaf holds one
  `KernelProgram` with its effects (conservative from `KernelProgram::effects`,
  or measured when built `From<AnalyzedKernelProgram>`), whose `Sequential`
  group holds parts in program order, and whose `Parallel` group holds parts
  proven pairwise independent. The tree is readable through `view()` and the
  retained programs come back in semantic order through `programs()`; a
  parallel group cannot be constructed except through the proof, so holding the
  value is holding the proof.
- **`then`** is total. `parallel` returns `Result<Composition, ConflictSet>`.
  Both live on `Composition` and on `KernelProgram` (`first.then(second)`,
  `left.parallel(right)?`). Lawfulness is `Effects::dependences_into` over the
  two operands' aggregate effects, in both serializations; read/read sharing
  yields nothing, and an `Unbounded` footprint means unknown locality, so it
  never conflicts with an unrelated resource (identity is exact `FieldBind`
  equality).
- **`ConflictSet`** is the complete, canonical evidence: the union of the
  dependences of both serializations, sorted by resource then hazard, deduped,
  nonempty by construction. A parallel group has no order, so a resource one
  side writes and the other reads obstructs both orders and carries both the
  `Raw` of the write-first serialization and the `War` of the read-first one;
  a resource both sides write carries `Waw`; two in-place programs carry all
  three. This is why swapping the operands yields an equal `ConflictSet`, while
  RAW, WAR, and WAW stay distinct and each names a serialization. There is no
  boolean compatibility surface, and the structural gate
  `tests/no_public_boolean_compatibility.rs` scans every public signature in
  the workspace for one (a bool-returning function named like a compatibility
  predicate or taking an effect-algebra type).
- **Normalization.** Nested groups of one kind flatten (a sequence into its
  parent sequence, preserving order; a parallel group into its parent parallel
  group, which is lawful because the aggregate check over a group covers every
  leaf), the canonical noop `KernelProgram::noop(grade)` drops out of either
  composition, and a single survivor stands alone. This gives left and right
  noop identity, associativity of `then`, and deterministic normalization at
  the semantic level. A parallel group keeps its construction order; lawful
  commutativity holds on the aggregate effects and the leaf set.
- **Aggregate versus structure.** `aggregate_effects()` is what a group touches
  against the outside — `Effects::normalized` over every leaf's reads and
  writes, so aliased footprints join by the Stage 2 lattice (`Point` identity,
  bounded componentwise max, unbounded dominance) and a measured leaf's sharp
  footprint survives into the union. `ordered_dependences()` is the order the
  structure carries — each earlier part's aggregate into each later part's, for
  every sequential group, recursively. A write-then-read sequence shows `x` in
  place in the aggregate and `Raw{x}` in the ordered dependences; the two are
  never conflated.
- **Nested conflicts surface.** A group's aggregate covers every nested leaf, so
  wrapping a conflicting program in a sequence, or in a lawful parallel group,
  cannot launder its conflict against a later `parallel` operand — the gate
  `a_sequence_cannot_hide_a_nested_conflict` pins both wrappings from both
  sides.

The one deviation from the stage brief: the brief described the one-direction
`a.dependences_into(&b)` vector as the parallel conflict set, and also asked
that reversed construction order produce the same evidence. For a write/read
pair those two requirements cannot both hold — one direction yields `Raw`, the
other `War` — so the conflict set is the union over both serializations, which
is the complete obstruction of an unordered pair and is reversal-invariant. The
`Unknown` dependence variant of the constitution sketch stays absent for the
Stage 2 reason: identity is always resolved and an unbounded footprint is a
locality fact, so there is no genuinely unresolved dependence to name.

### Phase 7: normalize the language and geography

After the architectural phases are complete, run a deliberately semantics-free
aesthetic sweep. Its purpose is not brevity for its own sake: the names and the
source tree should describe the architecture that survived the refactor rather
than preserve the vocabulary and module boundaries of its migration history.

The governing test is:

> Can a new reader predict where a concept lives and what a name means without
> knowing how the project used to be implemented?

Proceed in three passes:

1. **Canonical vocabulary.** Establish one term for each concept and remove its
   historical synonyms. In particular, audit program versus graph, receipt
   versus report versus ledger, recovery versus admissibility, field versus
   slot versus binding, and effect versus access versus dependence. Preserve
   distinctions that carry different laws; rename only false multiplicity.
2. **Structural geography.** Rename files, modules, and directories so the tree
   narrates the dependency direction — physics describes laws, tracing builds
   programs, compiler layers analyze them, execution interprets them, and
   diagnostics record interventions. Split files that still own unrelated
   responsibilities and merge tiny modules whose boundary is purely
   historical.
3. **Local expression.** Tighten type, function, variable, and test names.
   Remove suffixes such as `gv_` or `built_` once they no longer distinguish a
   live alternative, and replace vague local names (`ctx`, `spec`, `data`,
   `impl`) where a precise physical or compiler role is available. Do not
   expand obvious mathematical notation into prose or erase established domain
   language merely to make names longer.

This phase has strict discipline:

- rename one vocabulary family or crate at a time in reviewable commits;
- do not retain aliases or compatibility facades for deleted internal names;
- do not combine renames with arithmetic, scheduling, ABI, serialization, or
  ownership changes;
- preserve typed identities and keep strings at presentation boundaries;
- update documentation, tests, examples, and Python names as one atomic
  vocabulary migration where they express the same public concept;
- prove prepared-IR and serialized-manifest identity, unchanged binding order,
  AOT/host equivalence, and workspace all-targets cleanliness after every
  structural pass.

Success means the filesystem, public API, and local expressions tell the same
story as the enforced architecture, with no knowledge of the refactor required
to understand them.

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
