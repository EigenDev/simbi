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

- [ ] **A1. symbolic `div(curl B) = 0` checker.** compose the face-update DAGs for one
      cell, assert the edge-EMF coefficients cancel to the zero polynomial. proves div(B)=0
      for ANY input field. converts `rmhd_divb_under_evolve.rs:147` (empirical 1e-12) into a
      theorem. *prototype first on one geometry, then generalize.*
- [ ] **A2. conservation telescoping checker.** assert each shared face flux enters the two
      neighbor updates with equal-and-opposite signed coefficient. proves global conservation
      symbolically, replacing the 1e-9 sod integral as the primary guard.
- [ ] **A3. EMF upwind-pairing structural assert.** on the `uct_master_emf` DAG, assert the
      advective term pairs `a^L` with the UPWIND face (`ct_emf.rs:577`). this is the only
      guard for the supersonic-EMF regression that currently has NO rust test.
- [ ] **A4. weighted-curl invariance.** once A1 lands, assert the α/√(-g)-weighted curl
      (GR step B3) still telescopes — makes the GR weighting a checkable transformation.

## workstream B — the GR seam (connect carrier algebra <-> Metric)

root cause: the carrier traces SR-Minkowski algebra (`vel.dot(&vel)` is flat `v^i v^i`,
`srhd/cons.rs:38,70`); the `Metric` trait is GR-shaped but `raise`/`lower`/`lapse`/`shift`
have ZERO real callers and `symbi-discretize` re-derives Christoffels by hand. they don't
talk to each other.

- [~] **B1. wire `symbi-geometry` into `symbi-discretize`; delete hand-rolled
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
- [ ] **B2. thread the metric as a carrier-generic value into physics.** replace
      `vel.dot(&vel)` / `mom.dot(&mom)` with `g_{ij} v^i v^j` / `gamma^{ij} S_i S_j` in
      `srhd/cons.rs`, `srhd/algebra.rs`, `riemann/`. for a diagonal-flat metric this must
      compile BIT-IDENTICALLY to today (regression-safe). this is the SR->GR pivot.
- [ ] **B3. densitize the conserved state** (`sqrt(gamma) D, sqrt(gamma) S_i, sqrt(gamma)
      tau`) and add α/β^i transport to the update (`godunov.rs:317`) + CT (`ct_emf.rs:233`).
      activates the dead `lapse`/`shift`/`sqrt_det_gamma` surface.
- [ ] **B4. `Christoffel`/covariant-derivative as first-class `Metric` methods**, derived
      from g(x), replacing the diagonal-only hand-rolled source. slots in after B1-B3.
- [ ] **B5. non-diagonal `Schwarzschild: Metric`** (NOT `DiagonalMetric`) to exercise and
      prove the general-metric path. the type gate already exists (`metric.rs:250`).

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

---

## sequencing / dependencies

```
NOW (parallel, additive, low-risk):
  A1 div-checker prototype  ──┐
  D1 supersonic EMF gate    ──┼─ protect + prove BEFORE touching physics
  D2 HSE well-balanced gate ──┘

THEN (the GR seam, regression-safe because gates are in place):
  B1 wire geometry  ->  B2 metric-contract  ->  B3 densitize  ->  B4 Christoffel  ->  B5 Schwarzschild
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
