# TODO — axiomatic foundation toward GRHD/GRMHD

north star: make the metric a value the carrier homomorphism transports, so GRHD/GRMHD
is a parameter change rather than a rewrite; and turn the invariants the code maintains
(div(B)=0, conservation, positivity) from runtime hopes into theorems checked on the
traced IR DAG.

the one real, lawful, load-bearing abstraction today is the `Scalar`/`Gv` carrier
(`symbi-ir/src/algebra.rs:146`). it is zero-cost monomorphized to native f64 AND traces
the same source to an inspectable DAG. everything below either connects that asset to the
`Metric` trait, or cashes the checks the categorical vocabulary currently writes.

## status legend

- `[ ]` not started   `[~]` in progress   `[x]` done   `[!]` blocked

---

## workstream A — provability (the DAG is a proof object)  *keystone*

the traced IR DAG lowers every CT curl / flux to reads `gv_field_at(key, offset)` with
known integer stencil offsets. invariants become static polynomial-cancellation checks at
graph-build time — no evolve loop, no FP tolerance, all geometries at once.

- [x] **A1. symbolic `div(curl B) = 0` checker.** DONE for ALL geometries + EMF upwind (see
      the CAPSTONE entry in follow-ups). every CT curl is a graph-build-time theorem now.
- [x] **A2. conservation telescoping checker (cartesian).** DONE: the godunov mass update's
      flux part IS a sum of per-direction DISCRETE DIVERGENCES `G_d[+e_d]-G_d`
      (`G_d=(-dt/dx_d)F_d`), which telescope globally -> conserves for ANY flux field.
      `godunov_mass_conservation_symbolic.rs`; negative control + bug-injection verified
      (breaking `f_hi-f_lo` -> `f_hi+f_lo` makes it FAIL "NOT a discrete divergence").
      CURVILINEAR follow-up: reduces to shared-face area consistency `area_hi(c)==area_lo(c+e)`,
      but the angular measure carries cos/pi factors the integer-poly/sin ring can't represent —
      needs a cos + non-integer-constant ring extension (the analog of A1-sph(b)'s sin extension).
- [x] **A2-curvilinear DONE.** CYLINDRICAL + SPHERICAL mass conservation proven symbolically via
      the shared-face area consistency `area_hi_0(c) == area_lo_0(c+e_r)` (single-valued shared
      face -> volume-weighted flux div telescopes -> conserves). machinery added: `Op::Cos` as an
      opaque symbol (mirrors sin; keyed identically so the same theta-edge -> same symbol),
      `Poly::cos_sym`, pub `extract_scalar` (field-free geometry node -> RatFun), pub
      `RatFun::equals`/`shift_coords`. cylindrical is transcendental-free (r-face = r*dx_1*dx_2);
      spherical's solid-angle Omega = (cos th_lo - cos th_hi)*dphi is c_0-independent + a common
      factor -> cancels structurally under the r-shift (no pi for 3D). `godunov_mass_conservation_
      {cyl,sph}_symbolic.rs` + negative controls. independently verified: my own low-face
      perturbations (cyl rl->rh, sph r^2->r) make each FAIL; reverted clean.
- [x] **A3. EMF upwind-pairing structural assert.** DONE (`nmhd_uct_emf_upwind_symbolic.rs`,
      see follow-ups). covers all 5 uct kernels; flipping ct_emf.rs:577 makes it FAIL.
- [ ] **A4. weighted-curl invariance.** once A1 lands, assert the α/√(-g)-weighted curl
      (GR step B3) still telescopes — makes the GR weighting a checkable transformation.

## workstream B — the GR seam (connect carrier algebra <-> Metric)

root cause: the carrier traces SR-Minkowski algebra (`vel.dot(&vel)` is flat `v^i v^i`,
`srhd/cons.rs:38,70`); the `Metric` trait is GR-shaped but `raise`/`lower`/`lapse`/`shift`
have ZERO real callers and `symbi-discretize` re-derives Christoffels by hand. they don't
talk to each other.

- [x] **B1. wire `symbi-geometry` into `symbi-discretize`; delete hand-rolled
      `Coords`/Christoffel** in `symbi-discretize/src/gv/sources.rs`. NOT pure dedup — the
      inertial Christoffel dedups against `Metric`, but the well-balanced discrete pressure
      source STAYS (D2). progress:
      - [x] **B1a** — `Metric::momentum_source_inertial` made REGIME-AGNOSTIC (`rho`->conserved
        `mom`; bilinear `-Gamma(mom,v)`, also serves magnetic tension via `(b,b)`). all impls +
        block.rs wrapper + 8 tests updated. symbi-geometry 94 green.
      - [x] **B1a.2** — added `CylindricalRPhi` (the (r,phi) 2D DISK metric) — the existing
        `Cylindrical<2>` is (r,z)-axisymmetric/zero-inertial, which would silently zero the disk
        swirl. now metric.rs covers ALL discretize cases. 95 green (incl. the contrast test).
      - [x] **B1b DONE** — `sources.rs::inertial_momentum_sources_gv` now dispatches on the
        PROVIDED component count `mom.len()` (coordinate-order: cyl-2D = (r,phi) DISK ->
        CylindricalRPhi) to the const-D `Metric<Gv,D>`, padding to ncomp. added
        `Tensor::from_fn` + the `symbi-discretize -> symbi-geometry` dep; HAND-ROLLED CHRISTOFFEL
        DELETED. one geometry source for the codebase. VERIFIED regression-free: D2 stays
        bit-exact 0e0 (inertial vanishes at v=0), spherical sod, cylindrical disk/swirl/regime,
        nmhd_rotor_cyl_rphi, spherical NMHD/RMHD, decomp/additive source-equivalence all green.
        also FIXED + validated the pre-existing `cylindrical_rz_swirl_source` test (bound the
        probe-created mom_2) — proves CylindricalRPhi swirl matches the analytic.

## pre-existing failures (found by running the FULL suites during B1 — NOT B1 regressions) — FIXED

the committed sprints only ran targeted gate tests, never `cargo test -p symbi-discretize` /
`-p symbi` in full. doing so for B1's verification surfaced two committed-broken tests, now fixed:
- [x] **`unsupported_target_field_panics_loudly`** — STALE TEST: it used "bcell" as the
      "unsupported" example, but `bcell` became a VALID godunov target (cell-B prescription,
      godunov.rs:217). the panic discipline (godunov.rs:223) was intact. fixed the test to use a
      genuinely-bogus target_field. 9/9 green.
- [x] **`nmhd_rotor_cyl_rz_preserves_divb...`** — REAL SIGN BUG: the (r,z) CT curl `dir=1`
      (`rmhd_ct_curl_cyl_rz_gv`, ct_emf.rs:141) was `b + dt*(1/r_c)d_r(r E_phi)` but must be MINUS
      (`(curl E)_z = +(1/r)d_r(r E_phi)`, `dB/dt = -curl E`; OPPOSITE sign to spherical-poloidal
      B_theta). copied the poloidal form without flipping. div(B) blew to O(1) in ONE step. fixed;
      rotor + full discretize CT suite green. *this is exactly what a SYMBOLIC proof of the (r,z)
      curl would have caught — the 2D curls (rz/rphi/sph-poloidal) have only NUMERICAL guards.*
- [x] **B2. thread the metric as a carrier-generic value into physics.** replace
      `vel.dot(&vel)` / `mom.dot(&mom)` with `g_{ij} v^i v^j` / `gamma^{ij} S_i S_j`. SR->GR pivot.
      threading mechanism: a `Matrix<S,D>` (the spatial metric / its inverse) passed on the `eos`
      rails; `Matrix::quadratic`/`contract`/`identity` already exist; flat/orthonormal = identity =
      bit-identical (optimizer folds it). Regime trait NOT touched (host wrapper absorbs it).
      - [x] **B2.P1** — proof of concept: threaded `gamma_inv` into `srhd_recover`, converted the
        `s_mag = sqrt(gamma^{ij} S_i S_j)` site, identity at all 4 callers. VERIFIED bit-identical
        at all 3 levels: f64 (38), interpreter oracle, and the COMPILED-kernel gate
        (`aot_carrier_equivalence::srhd_c2p_kernel_equals_host` green). de-risks the whole arc.
      - [x] **B2.P2** — fan out the remaining Tier-1 `.dot()` sites. introduced `SpatialMetric<S,D>`
        carrier (symbi-hydro/src/spatial_metric.rs: gamma + gamma_inv + `flat()` + variance-NAMED
        norm helpers `norm_sq_cov`/`norm_sq_contra` — encodes the contra/cov trap). done:
        - [x] SRHD c2p complete: s_mag (norm_sq_cov, S covariant) + v^2 (norm_sq_contra, v contra).
          all callers pass `SpatialMetric::flat()`. fast-verified (f64 + interpreter oracle).
        - [x] RMHD c2p (`rmhd_recover`): r_sq -> norm_sq_cov (r covariant), bee_sq -> norm_sq_contra
          (B contra). rdb = r_i B^i is a cov*contra pairing -> METRIC-FREE, stays `.dot()`. rperp.dot
          mixes variance (KKC is SR-flat) -> bit-identical for flat, GR reformulation deferred. all 3
          levels green incl compiled-kernel (`rmhd_c2p_kernel_equals_host`). CON2PRIM BATCH DONE.
        - [x] **flux-path batch part 1 — rmhd algebra + flux**: rmhd/algebra (magnetic_pressure /
          total_pressure / four_vector / source_quantities) all take `metric`; bsq/vsq -> norm_sq_contra,
          vdb=gamma_{ij}v^i B^j -> `contract_contra` (added). cascaded into rmhd.rs to_conserved/to_flux
          (construct flat() at the boundary) + the Gv source builder (sources.rs:196). `cons.mom.dot(nhat)`
          left metric-free (ambiguous conserved-momentum variance, settled by C1). added `project_transverse`
          helper for the riemann. all 3 levels green (28 hydro + 27 oracle + 3 immersed, f64+interp).
        - [x] **flux-path batch part 2 — the Riemann solvers**: triaged the relativistic bodies ONLY
          (hllc_srhd/hllc_rmhd + hlld_vdiff/hlld_rmhd_converge/hlld_rmhd/hlld_rmhd_states). converted the
          contravariant velocity/B contractions (v.n, B.n, |v|^2, |B|^2, v.B, the K-vector |K|^2 / K.B / K.n,
          the transverse-B projector `B - n(B.n)` -> `project_transverse`); LEFT the momentum-class
          (conserved S_i / flux-of-momentum / r-vector `.mom` . n^i, and the mixed B(contra).mom(cov)
          pairings c, rdv) as metric-free `.dot()` pending C1. the Newtonian/iso solvers (hlld_newtonian,
          *_coeffs, hlld_isothermal) are orthonormal-by-domain -> left Euclidean. metric = flat() at each
          solver entry, threaded through vdiff/converge. all 3 levels green: 273 hydro f64 + 27 interp oracle
          (+ AOT gate to stamp). B2.P2 COMPLETE (con2prim + full flux path metric-aware).
        dev loop = `cargo test -p symbi-discretize --test carrier_oracle` (~2s); AOT gate per batch (~22-47s).
      - [x] **B2.P3 — RESCOPED (the structural Tier-2 was mis-framed, now dissolved/redirected).**
        the original scope (Banyuls-Font wave speeds `alpha/beta/gamma^nn` + a "GR" face normal) ASSUMED
        the Valencia coordinate-basis formulation. but the geometry layer already DECIDED the formulation
        via the `DiagonalMetric` type-gate (`metric.rs:55-64`): diagonal metrics (flat, curvilinear,
        Schwarzschild, FLRW) use the existing ORTHONORMAL / physical-component convention (`V_â = h_a v^a`,
        spatial metric = identity in physical components); non-diagonal (Kerr) impls `Metric` but NOT
        `DiagonalMetric`, so the genuine-`gamma_ij` path is COMPILE-GATED off until written. consequences:
        - face normal `Tensor::unit(coord_n)` is ALREADY CORRECT in an orthonormal frame (no `gamma^nn`
          normalization for diagonal metrics) -> the `project_transverse` gamma^nn caveat was a Valencia
          worry. NO-OP for the realized (diagonal) physics.
        - wave-speed EIGENVALUES (the MUB09 quartic) stay SR-verbatim in the local orthonormal frame;
          alpha/beta enter at the DIVERGENCE (B3), not in the eigenvalue formula. the raw `.dot()`s left in
          `rmhd/wave_speeds.rs` are Tier-1 leftovers (identity under the orthonormal convention) -> OPTIONAL
          cosmetic tidy (thread `SpatialMetric` for carrier-story uniformity), NOT structural. [ ] if desired.
        - C1 (variance) is VALIDATED, not deferred: in an orthonormal frame `S_â n^â` is genuinely
          metric-free, so the Tier-1 decision to leave the momentum-class as `.dot()` is CORRECT (a typed
          covariant `Tensor` is still worth it for safety, but it is not a correctness gap here).
        - the `SpatialMetric` carrier's genuine NON-identity job is the Kerr frontier (B5) -> YAGNI until a
          Kerr metric exists. the real GR structure for diagonal targets is alpha/beta/sqrt(gamma) -> B3.
        B2 COMPLETE: every contravariant contraction routes through the carrier; the SR->GR seam is closed
        at Tier-1, and the type system (not a comment) marks where Tier-2 genuinely begins.
- [~] **B3. densitize the conserved state + 3+1 transport** (`sqrt(gamma) D, sqrt(gamma) S_i,
      sqrt(gamma) tau`) and add alpha-scaling + beta^i advection to the update (`godunov.rs:317`)
      + CT (`ct_emf.rs:233`). activates the dead `lapse`/`shift`/`sqrt_det_gamma` surface (all SR-default
      no-ops, so invisible to existing runs). PLUMBING DESIGN (user-confirmed): **(C) Metric-trait-generic**
      — dispatch `Coords -> concrete Metric -> metric.lapse(x)` as a traced Gv expression (the B1
      source-dispatch pattern), NOT a codegen `Spacetime` enum (A) nor runtime metric fields (B). keeps
      geometry single-sourced from the trait; alpha is a traced expr, not a stored field; flat -> ONE ->
      bit-identical. (B) is the escape hatch if the trait churn through AOT proves invasive.
      KEY PHYSICS (Schwarzschild coordinate gift): `sqrt(-g) = alpha sqrt(gamma) = r^2 sin(theta)` =
      FLAT spherical area -> flux face areas UNCHANGED; `1/sqrt(gamma) = alpha/sqrt(gamma_flat)` -> the
      whole spatial RHS (flux div + geometric source) collapses to a single `x alpha(r)` weight. beta = 0.
      - [x] **B3.0 — the lapse seam (inert, de-risk)**: `gv_lapse_weight(coords) -> Option<Gv>`
        (geometry.rs) returns `None` for every flat metric today; threaded into the `fe` combine
        (godunov.rs) weighting `div` + the geometric `src` (NOT the `u` / mesh-dilution terms). flat ->
        `None` -> DAG UNTOUCHED -> bit-identical (no `xONE` node — `Gv::binop` does not fold). proves the
        seam threads through the fused stage + AOT WITHOUT perturbing. verified: full symbi-discretize
        suite green (carrier oracle 27 + all godunov/flux). NO public-signature churn (derived internally
        from `coords`). only the PRIMARY gas update (`godunov_stage_gv_with_fused_built`, mass+mom+nrg);
        the mass-demo (`godunov_mass_gv`), bcell euler, and CT (`ct_emf.rs`) are B3.1 follow-ups.
      - [~] **B3.1 — Schwarzschild goes live**:
        - [x] **the `Spacetime` SELECTOR skeleton (inert, flat-identical)**: a `Spacetime` enum
          ORTHOGONAL to `Coords` AND to the regime (GR is a spacetime, NOT a regime — there is NO Grhd
          regime and there shouldn't be; Srhd/Rmhd compose with any spacetime). added in BOTH layers
          (`symbi_geometry::Spacetime` + `Metric::spacetime()` default Minkowski, mirroring
          `Geometry`/`geometry()`; `symbi_discretize::Spacetime` codegen mirror, like `Coords`). threaded
          `spacetime` through the 3 stage builders (`godunov_stage_gv{,_with_fused_sources,_with_fused_built}`)
          + ALL callers (gv/mod.rs, runtime_source.rs runtime path, build.rs AOT bake x3, emit_iso_cuda
          example); `gv_lapse_weight(coords, spacetime)` dispatches on it. carried in `PartitionGeometry`
          beside `coords` (`spacetime: metric.spacetime()`). every realized run -> `Minkowski` -> `None` ->
          DAG UNTOUCHED. verified flat-identical: geometry+discretize+substrate+sim all build clean,
          carrier oracle 27 + discretize suite + 273 hydro f64 green. (AOT bake = the only heavy rebuild.)
        - [x] **the `Schwarzschild: Metric` impl (geometry + lapse), unit-tested**: `Schwarzschild { mass }`
          in symbi-geometry — DIAGONAL (impls `DiagonalMetric`), `f(r) = 1-2M/r`, lapse = sqrt(f),
          gamma = diag(1/f, r^2, r^2 sin^2), sqrt(gamma) = r^2 sin/sqrt(f), for D=1/2/3 (mirrors Spherical
          + the radial stretch). geometry() = Spherical, spacetime() = Schwarzschild. `Spacetime::Schwarzschild`
          tag added to BOTH enums (geometry `=1`, discretize mirror). 4 unit tests green (analytic lapse/gamma,
          gamma*gamma^-1=I, the sqrt(-g)=r^2 sin GIFT, M=0 -> Spherical exactly, 1D radial) + 99 existing
          geometry green. gravity (the geodesic momentum source) left at the trait default (zero) -> B4.
          `gv_lapse_weight` Schwarzschild arm STUBBED `None` (exhaustive match; unreachable — no sim selects
          it yet); discretize oracle 27 still flat-identical.
        - [ ] **step B — make Schwarzschild FLOW through the kernel** (the live wiring): (1) refactor the
          struct to `Schwarzschild<S> { mass: S }` so M can be a `Gv::scalar` at codegen (host-filled),
          keeping `metric.lapse` the single source (vs baking a literal M); (2) thread the cell centroid
          into `gv_lapse_weight` + the `(Spherical, Schwarzschild) -> run::<Schwarzschild<Gv>,D>` dispatch
          returning `Some(metric.lapse(centroid))`; (3) the mass-scalar HOST BINDING (substrate manifest +
          driver fills `schwarzschild_mass`); (4) bake-loop spacetime enumeration + slug tag + runtime
          kernel-select keys on `PartitionGeometry.spacetime` (+ the geometry->discretize Spacetime bridge);
          (5) PIN the exact sqrt(-g)-on-faces vs sqrt(gamma)-on-volume factor placement AGAINST a known
          solution. then the gravity source (B4) for actual infall; extend the seam to CT + bcell + mass-demo.
      CONCRETE DRIVER: **Schwarzschild (standard coords) — DIAGONAL, beta = 0.** SR Riemann/c2p VERBATIM
      + alpha-scaling + sqrt(gamma)-densitization + the B1 Christoffel gravity source. first genuine GR run.
- [ ] **B4. `Christoffel`/covariant-derivative as first-class `Metric` methods**, derived
      from g(x), replacing the diagonal-only hand-rolled source. slots in after B1-B3.
- [ ] **B5. non-diagonal `Kerr: Metric`** (NOT `DiagonalMetric`) — the genuine non-diagonal exerciser
      (off-diagonal gamma + frame-dragging shift beta != 0). this is where the `SpatialMetric` carrier
      earns its keep (real gamma_ij contractions) AND where the one genuine open fork lives: do the
      SR-orthonormal solvers get a tetrad transform at the face, or a coordinate-basis (Valencia) solver?
      PREMATURE to decide now (no Kerr metric exists). the type gate already exists (`metric.rs:55-64`).
      (NOTE: standard Schwarzschild is DIAGONAL -> it is the B3 driver, not this. a non-diagonal exerciser
      needs Kerr, or Schwarzschild in Kerr-Schild/Eddington form with a shift.)

## workstream C — type-level invariants (cash the categorical checks)

template to copy: `EnergyModel` ZST slot (`energy.rs:102`) — isothermal energy access
doesn't compile. illegal-states-unrepresentable done right.

- [ ] **C1. fix the variance contraction rule.** IR currently accepts `Upper·Upper` /
      rejects `Upper·Lower` — the INVERSE of the tensor rule — while the correct predicate
      `VarianceTag::contracts_with` is dead code. promote variance to per-axis; make
      Upper·Lower the only legal contraction (`graph.rs:1351/1561/1732`). *blocks B4 sanity.*
- [ ] **C2. `Validated<Prim>` newtype.** `C2pResult{value,error}` is parse-don't-validate
      backwards — an unphysical `Prim` is fully readable with the error in an ignorable side
      channel (`c2p_result.rs:81`). make the only way out of c2p a checked constructor.
- [ ] **C3. centering typestate: wire or delete.** `Cell/Face/Edge` has ZERO construction
      sites — a face flux CAN be stored at a cell (`symbi-sim/src/state.rs:513`). either
      annotate the ~10 sites + add the one `curl: Edge->Face` signature, or delete
      `symbi-grid/src/centering.rs` (currently fails the rent test).
- [ ] **C4. (deferred) real `Pass` trait** so passes are composable `Graph->Graph` instead
      of `&mut` mutators (`passes/mod.rs:6` claims this but signatures contradict). only when
      reordering/fusion forces it. otherwise fix the doc comment.

## workstream D — regression gates (protect the refactor)  *do alongside A*

- [ ] **D1. supersonic field-loop EMF gate (rust).** deterministic test that the advective
      EMF stays stable supersonically. closes the highest-severity UNTESTED invariant; only
      guard today is the python `field_loop.py`, invisible to `cargo test` and to subsonic OT.
- [ ] **D2. static-HSE well-balanced gate.** seed a hydrostatic atmosphere (p(r) balancing
      the geometric source, v=0), assert `|v| < eps` for N steps. `sources.rs:130` is
      well-balanced by CONSTRUCTION but NOTHING pins it; the spherical tests assert the
      OPPOSITE (motion).
- [ ] **D3. NaN cell-locator.** the NaN->dt-guard chain is excellent but reports only
      "invalid dt". add which-cell/which-field first went non-finite. the dual of
      NaN-propagation is NaN-localization.
- [ ] **D4. RMHD c2p bracket property test.** randomized physical (rho,v,p,B) asserting
      `f_lo0 < 0 <= f_hi0` for `find_mu_plus` (`rmhd/cons.rs:84,151-162`). the one con2prim
      path where the no-silent-floors guarantee can leak (defensive poison was removed).
- [ ] **D5. audit `PRESSURE_FLOOR_EPS=1e-3`** (`rmhd/cons.rs:25,175`) — a real silent floor
      in the kernel path with no ErrorCode, contradicting the SRHD "NO floors" discipline.
      either flag it via `ErrorCode` or document why it is exempt.

## workstream E — HPC guardrails (protect before GR lands)

layering is sound and zero-cost (no dyn/Box/per-cell alloc; SoA coalescing-optimal;
`Gv::iterate` traces the Newton body once). GR's first failure mode is GPU register spilling
on the fused flux+metric kernel.

- [ ] **E1. make the `S::scope` register-pressure discipline load-bearing NOW.** partition
      the existing RMHD kernel phases into nested `S::scope(||...)` blocks + gate the build
      with `assert_peak_pressure!`, while a working baseline exists to A/B against. GR's
      metric+inverse+Christoffel+Jacobian live-set will otherwise overflow the register file
      (RMHD wave-speed already hit 154 regs/thread; CSE is scope-local, `pressure.rs` is
      analysis-only).
- [ ] **E2. convert hot HLLC/HLLD star-states from `S::branch` (both arms) to `S::cond`**
      (one arm). under GR each star-state carries metric contractions — `branch` doubles the
      most expensive per-face block (`riemann/hllc.rs:225-241`).
- [ ] **E3. free wins:** `policy.rs:211` per-block `Vec<Buf>` -> `SmallVec`; drop the
      release-active aliasing `HashSet` at `policy.rs:159-167`. hottest scheduler path.

## workstream F — build / dev-speed (structural — HURTS development NOW)

- [x] **F0. the AOT "slow compile" was the DWARF SIGBUS HANG, not volume.** root cause: the
      createAndAddScopeChildren stack-overflow SIGBUS crashed symbi-aot's test crate; in a sandboxed
      agent shell the crash HANGS (masquerading as an endless compile), in an interactive shell it
      dies in ~20s. FIXED: workspace-wide `[profile.dev] debug = false` (was line-tables-only) — no
      DWARF scope tree -> no overflow. measured REAL aot compile = 47s (tolerable per-batch gate).
      [[debug-build-dwarf-sigbus]]. LESSON: an agent reporting a multi-minute "compile hang" on a
      heavy crate is probably THIS crash — have the user run it for the fast crash/backtrace.
- [~] **F1. (downgraded) AOT codegen recompile VOLUME.** `symbi-aot/build.rs` emits ~820K
      lines (~69MB) of fully-scalarized kernel Rust into OUT_DIR (one straight-line `pub fn` per
      regime x geometry x dim x solver; the `rmhd_face_flux_cyl_rz_hlld_2d` kernel alone is 1168
      lines), ALL `include!`d into ONE crate. ANY `symbi-hydro` change (e.g. every B2 `.dot()`
      site) invalidates build.rs -> rustc recompiles the ENTIRE pile. observed: `cargo test -p
      symbi-aot --test aot_carrier_equivalence` was STILL compiling at 7 min when killed — actual
      duration UNKNOWN (could be far worse). 7+ min for one test crate smells PATHOLOGICAL, not
      merely "big": suspect a single huge codegen unit OR rustc/LLVM choking on one massive
      monomorphized fn (the 1168-line scalarized kernel is the prime suspect).
      - INVESTIGATE FIRST: `cargo build -p symbi-aot --timings` (or `-Z time-passes`) to find WHICH
        unit/fn is the time sink, and whether it's rustc front-end, LLVM, or codegen-units serialization.
      - quick mitigations (do not fix the disease): the dev loop is `cargo test -p symbi-discretize`
        (interpreter oracle — NEVER compiles the generated kernels, ~2s) + `-p symbi-hydro` f64;
        `SYMBI_GEN_SERIAL=off --features debug-emit-knobs` drops the serial twins (~1.5x leaner gate).
      - structural fix: split generated kernels so a physics change invalidates ONLY the affected
        ones — per-kernel compilation units / finer `cargo:rerun-if-changed` scoping, or raise
        codegen-units / break the giant kernels. makes the AOT gate INCREMENTAL instead of all-or-nothing.

---

## sequencing / dependencies

```
NOW (parallel, additive, low-risk):
  A1 div-checker prototype  ──┐
  D1 supersonic EMF gate    ──┼─ protect + prove BEFORE touching physics
  D2 HSE well-balanced gate ──┘

THEN (the GR seam, regression-safe because gates are in place):
  B1 wire geometry  ->  B2 metric-contract  ->  B3 densitize (+Schwarzschild, diagonal)  ->  B4 Christoffel  ->  B5 Kerr (non-diagonal)
  C1 variance fix runs alongside B (needed for B4 to be sane)
  A2/A3/A4 provability checks land as each physics piece stabilizes

PARALLEL ANYTIME (independent):
  C2 Validated<Prim>,  C3 centering wire/delete,  D3 NaN locator,  D4 bracket test,
  D5 floor audit,  E1 register discipline,  E2 cond,  E3 free wins
```

## current sprint (NOW)

- [x] A1 — symbolic div(curl)=0 checker. cartesian 3D proven by zero-polynomial
      cancellation; `symbi-ir/src/proof.rs` + `rmhd_ct_curl3d_divb_symbolic.rs`. runs in
      0.00s. spherical needs a rational-function coeff ring (deferred, see A1-sph below).
- [x] D1 — supersonic field-loop EMF gate. `nmhd_uct_supersonic_emf_upwind.rs`; HLLD only
      (HLLC fixes a^L=a^R=1/2 so the swap is a no-op), div-free checkerboard seed,
      staggered-face energy. trimmed to 12x12/20 steps = 3.11s. flipping ct_emf.rs:577
      makes E blow past 5x by step 13 (gates).
- [x] D2 — static-HSE well-balanced gate. `symbi/tests/substrate_well_balanced.rs`; uniform
      rho/p, v=0 on spherical(adiabatic/iso/srhd)+cylindrical; max |v_r| = 0e0 bit-exact,
      N=16. MUST run `--release` (debug crashes rustc, see below).

## follow-ups opened by the sprint

- [x] **A3 done** — instant structural upwind-pairing proof. `uct_master_emf_proof_kernel`
      traces the master emf with symbolic params (linear in the 4 face reads); LinForm reads
      each coefficient; `Poly::coefficient_of` asserts a^L weights the upwind face. covers ALL
      5 uct kernels (they all compose through uct_master_emf). 0.00s. negative control +
      verified: flipping the real ct_emf.rs:577 makes it FAIL. `nmhd_uct_emf_upwind_symbolic.rs`.
      D1 (the 3.11s numerical blow-up) kept as the heavier end-to-end integration check;
      A3 is now the PRIMARY guard.
- [x] **debug-build fix** — `[profile.dev.package.symbi-substrate] debug = false` (the DWARF
      scope-tree recursion overflowed rustc's stack -> SIGBUS). full symbi stack now builds in
      debug (22s vs 90s release); leaf-crate tests already fast. see [[debug-build-dwarf-sigbus]].
- [x] **D2 sensitivity check** — verified: collapsing the geometric pressure source (area_hi
      -> area_lo at sources.rs:132) makes all 4 regimes FAIL with |v_r| ~ O(0.1) (vs bit-exact
      0e0), ~14 orders above the 1e-12 bound. the gate gates hard. reverted.
- [x] **A1-sph (a)** — numerical div(B) test for the 2D spherical POLOIDAL curl
      (`rmhd_ct_curl_2d_sph_gv`), the gap its own doc flagged. B=curl(A_phi), nontrivial |B|=3.25,
      area-weighted div = 2.9e-16 before AND after a curl(E_phi) step. derived the consistent
      area weighting (r_f^2 sin(th_c) dth / r_c sin(th_f) dr). verified: a B_theta sign flip
      drives div to 7.8e-2. `rmhd_ct_curl_2d_sph_poloidal_divb.rs`; builder doc updated.
- [x] **A1-sph (b)** — DONE: full 3D spherical symbolic div(curl B)=0 proof. added a
      rational-function ring (`RatFun` = num/den Poly) + `LinFormR` with a COVARIANT shift
      (subst_shift on c_N + sin-symbol remap) in proof.rs (~360 lines); `eval_rat` handles
      Op::Div/Sin/Cast; r is affine-poly, sin(theta@offset) an opaque symbol (half-unit keys for
      cell centers). all 3 dirs telescope to the zero rational function, EXACT (no FP tolerance).
      `rmhd_ct_curl_sph_divb_symbolic.rs` + negative control. independently verified: a
      spherical-only curl sign-flip makes it FAIL while cartesian stays green; reverted clean.
      key finding: covariant coord-shift of the coefficients is the crux (cartesian coeffs are
      translation-invariant; spherical aren't); canonicalization is SIMPLER (only field keys,
      since geometry uses absolute x_lo_N/dx_N not per-dir id_pN).
- [x] **A1-cyl** — DONE: 3D CYLINDRICAL symbolic div(curl B)=0 (`rmhd_ct_curl_cyl_divb_symbolic.rs`).
      reuses the RatFun ring — cylindrical has NO sin (h_r=1, h_phi=r, h_z=1), so coeffs are pure
      affine-r rationals (the no-sin subset of spherical). area weights A_r=r(0)dphi dz,
      A_phi=dr dz, A_z=r_c dr dphi; telescopes exact. verified: a cyl-specific h_phi (r->1) bug
      makes it FAIL while spherical stays green; reverted clean. all 3 geometries now symbolic.
- [x] **A1-2D + proof.rs split (CAPSTONE)** — symbolic div(curl B)=0 now covers EVERY CT curl.
      added the 3 remaining 2D builders: `rmhd_ct_curl_cyl_rz_divb_symbolic.rs`,
      `..._cyl_rphi...`, `..._2d_sph...` (single `ez` field, per-geometry area weights). each
      telescopes exact + has a negative control; bug-injection verified each CATCHES a sign error
      — the rz proof FAILS on the exact `+`-sign bug that was committed-broken. ALSO split
      `proof.rs` (1049 lines) by responsibility -> `proof/{poly,linform,extract,mod}.rs` (SRP;
      pub(crate) at the new boundaries; public API + all 7 proof tests unchanged/green).
