# =============================================================================
# uct_algorithm.md
#
# the core of the Upwind Constrained Transport (UCT) edge-EMF algorithm, from
# Mignone & Del Zanna (2020), "Systematic construction of upwind constrained
# transport schemes for MHD", J. Comput. Phys. (arXiv:2004.10542v2). the PDF is
# in this directory (mignone_delzanna.pdf). equation numbers below refer to it.
#
# this is the authoritative spec for the simbi UCT implementation. when in doubt
# about an intricate coefficient (HLLC/HLLD intermediate states), OPEN THE PDF
# and verify — the subscript-heavy equations are easy to mis-transcribe.
# =============================================================================

## 0. why UCT (the problem it solves)

constrained transport (CT) keeps `div(B) = 0` to machine precision by evolving
face-staggered B from edge-centered electric fields (EMFs) via Stokes' theorem.
the EMF at a cell EDGE (a corner in 2D) must be reconstructed from the 1D Riemann
data at the surrounding FACES. the simplest reconstructions (arithmetic average,
Gardiner-Stone "contact") under-dissipate the grid-aligned odd-even mode — the
**checkerboard instability** — because they upwind on the contact (mass-flux)
direction only, not on the true MHD characteristic speeds.

UCT fixes this by building the edge EMF as a genuinely multi-dimensional upwind
flux: the corner EMF is weighted by the Riemann signal speeds in BOTH transverse
directions, plus a `(speed) * (transverse-B jump)` dissipation that couples the
odd/even modes and damps the checkerboard. **the dissipation coefficient is set
by the base Riemann solver — that is the whole point: HLL / HLLC / HLLD give
progressively smaller diffusion (fast → +contact → +Alfvén speeds), hence sharper
results.** div(B)=0 is preserved by ANY edge EMF (the discrete curl telescopes);
the EMF choice only affects accuracy/stability.

## 1. staggering & notation (Fig. 1, Eq. 5-9)

- flow conserved vars `U = (rho, rho*v, E)` are CELL-centered (`U_c`, c=(i,j,k)).
- magnetic field is FACE-staggered: `Bx` on x-faces `(i+1/2,j,k)`, etc. (`B_f`).
- EMF `E` is EDGE-staggered: `Ez` on z-edges `(i+1/2,j+1/2,k)`, etc. (`E_e`).
- the EMF IS the induction flux: for `Bx`, `F^[Bx] = (0, Ez, -Ey)`; the z-edge
  EMF `Ez = v_x B_y - v_y B_x` is the y-flux of `Bx` and the (-)x-flux of `By`.

cardinal directions around the z-edge `(i+1/2,j+1/2)` (Fig. 1b):
- S / N : the x-faces below/above the edge — `xf=(i+1/2,j)` and `(i+1/2,j+1)`.
- W / E : the y-faces left/right of the edge — `yf=(i,j+1/2)` and `(i+1,j+1/2)`.
the four CELLS around the edge: SW=(i,j), SE=(i+1,j), NW=(i,j+1), NE=(i+1,j+1).

semi-discrete CT update (Eq. 9), div-free by construction (Eq. 10):
```
dBx_f/dt = -(Dy Ez_e)/dy + (Dz Ey_e)/dz      (D = backward difference, Eq. 6)
dBy_f/dt = -(Dz Ex_e)/dz + (Dx Ez_e)/dx
dBz_f/dt = -(Dx Ey_e)/dx + (Dy Ex_e)/dy
```

## 2. THE MASTER FORMULA (Eq. 30, 33) — solver-agnostic

every solver writes its 1D induction flux at a face in the SAME form (Eq. 30):
```
F^[Bt]_xf = a^L_x F^L + a^R_x F^R  -  (d^R_x B^R_t - d^L_x B^L_t)
            \___ advective ___/      \____ dissipative ____/
```
where `F = v_x B_t - v_t B_x` (the induction flux, t = transverse component),
`a^L + a^R = 1`, and `(a^{L,R}, d^{L,R})` are the SOLVER's coefficients (§3).
`B^{L,R}_t` are the transverse field states reconstructed to the face.

the edge EMF is then composed from the per-face coefficients (Eq. 33):
```
Ez_e = - [ (a_x vbar_x B_y)^W + (a_x vbar_x B_y)^E ]      # x-advection of By
       + [ (a_y vbar_y B_x)^N + (a_y vbar_y B_x)^S ]      # y-advection of Bx
       + [ (d_x B_y)^E - (d_x B_y)^W ]                     # x-diffusion of By
       - [ (d_y B_x)^N - (d_y B_x)^S ]                     # y-diffusion of Bx
```
- `vbar_t` = the UPWIND transverse velocity at the face (Eq. 29, below).
- `B_x`, `B_y` are the STAGGERED face fields reconstructed to the edge.
- `(.)^{W,E,N,S}` = the quantity at that cardinal face brought to the edge.
- the coefficients are combined to the edge from the two faces straddling it.

### 2a. upwind transverse velocity (Eq. 29)
at an x-face, the upwind transverse velocity (HLL-averaged) is
```
vbar_{t,xf} = (alpha^+_x v^L_xf,t + alpha^-_x v^R_xf,t) / (alpha^+_x + alpha^-_x)
```
for t = y,z, with `alpha^+ = max(0, lambda^R)`, `alpha^- = -min(0, lambda^L)`.
(the scaffolded `v_upwind[dir][t]` fields hold these.)

### 2b. edge combination of the coefficients — USE MAX (not the paper default)
the paper's default (Eq. 34-35) AVERAGES the two face coefficients to the edge:
```
d_x^W = (d^L_xf + d^L_{xf+ey})/2,   d_x^E = (d^R_xf + d^R_{xf+ey})/2   (Eq. 34)
d_y^S = (d^L_yf + d^L_{yf+ex})/2,   d_y^N = (d^R_yf + d^R_{yf+ex})/2   (Eq. 35)
```
**simbi uses MAX instead of the average** for the edge speed/diffusion
reconstruction (a touch more diffusive, more robust). the paper EXPLICITLY
sanctions this: "Other forms of averaging for these coefficients - based on the
upwind direction or by maximizing the diffusion terms - are of course possible."
keep MAX. apply it to the edge speeds `alpha^{+,-}` (max over the surrounding
cells/faces) from which `d` is then formed, OR to `d` directly — both fine.

## 3. THE COEFFICIENT SETS (one per solver) — the only solver-specific part

specialize to an x-interface; drop the `xf` subscript. `lambda^{L,R}` are the
outermost (fast magnetosonic) speeds; `lambda*` the contact; `lambda^{sL,sR}` the
Alfvén/rotational speeds. `alpha^R = max(0,lambda^R) >= 0`,
`alpha^L = -min(0,lambda^L) >= 0`.

### 3.1 Rusanov / local Lax-Friedrichs (Eq. 31)
```
a^L = a^R = 1/2
d^L = d^R = |lambda^max| / 2          # lambda^max = max(|lambda^L|,|lambda^R|)
```

### 3.2 HLL (Eq. 32) — REGIME-GENERIC (classical AND relativistic)
```
a^L = alpha^R/(alpha^R+alpha^L) = 1/2 + (1/2)(|lambda^R|-|lambda^L|)/(lambda^R-lambda^L)
a^R = alpha^L/(alpha^R+alpha^L) = 1/2 - (1/2)(|lambda^R|-|lambda^L|)/(lambda^R-lambda^L)
d^L = d^R = alpha^R alpha^L / (alpha^R + alpha^L)
```
needs ONLY the two fast speeds → works for every regime (uses `wave_speed_l/r`).
this is the most diffusive: `d ~ lambda_fast/2`.

NOTE: the compact 2D form of HLL-UCT actually implemented first (matching this
paper's Eq. 27 / Del Zanna 2007) is algebraically equivalent for HLL but does NOT
generalize to HLLC/HLLD (which need a^L != a^R). use the MASTER form (Eq. 33) so
all three solvers share one EMF kernel.

### 3.3 HLLC (Eq. 36-38) — CLASSICAL (Miyoshi-Kusano contact wave)
**CRITICAL (p.11): for the MAGNETIC field, HLLC == HLL when `B_x != 0`.** the contact
wave does NOT resolve the transverse field — `B_t` is CONTINUOUS across the contact
(`B_t^{sL} = B_t^{sR}`) when `B_x != 0`, so the contact dissipation term vanishes and
the HLLC EMF coefficients reduce EXACTLY to HLL (Eq. 32). a standalone "UCT-HLLC" thus
adds NOTHING over UCT-HLL except in the measure-zero `B_x = 0` layer. (this is why the
EMF-diffusion win comes from HLLD's Alfvén waves, NOT HLLC's contact.)

ONLY in the degenerate `B_x = 0` case does `B_t` jump across the contact (Eq. 37):
```
B*^s - B^s = B^s chi^s,    chi^s = -(v_x^s - lambda^s)/(lambda^s - lambda*)   (s=L,R)
```
giving the coefficients (Eq. 38) — NOTE the last term is `|lambda*|/2`, NOT `|lambda^s|/2`:
```
a^L = a^R = 1/2
d^s = ((|lambda*| - |lambda^s|)/2) chi^s + |lambda*|/2          (s=L,R)
```
`lambda* = m_x^hll / rho^hll` (the contact = x-momentum/density of the HLL avg). these
Eq.-38 coefficients are NOT a robust standalone solver — they are the BUILDING BLOCK for
HLLD's singular `B_x -> 0` limit (Eq. 46, recovered by `v* = 0`). do NOT apply them for
`B_x != 0` (there `d^s` can go negative -> anti-diffusion; that is the wrong regime).

### 3.4 HLLD (Eq. 39-46) — CLASSICAL (Miyoshi-Kusano five-wave)
five-wave fan: `lambda^L, lambda^{sL}, lambda*, lambda^{sR}, lambda^R` (two fast,
two Alfvén/rotational, one contact). the star-state densities and contact:
```
rho^{*s} = rho^s (lambda^s - v_x^s)/(lambda^s - lambda*)        (s=L,R)
lambda*  = m_x^hll / rho^hll
lambda^{sL} = lambda* - |Bx|/sqrt(rho^{*L}),  lambda^{sR} = lambda* + |Bx|/sqrt(rho^{*R})   (Eq. 40)
```
HLL average state (Eq. 41): `U^hll = (lambda^R U^R - lambda^L U^L + F^L - F^R)/(lambda^R - lambda^L)`.
transverse-field jumps across the fast waves (Eq. 42):
```
B_t^{ss} - B_t^s = B_t^s chi^s,
chi^s = (v_x^s - lambda^s)(lambda^s - lambda*) / [ (lambda^{ss} - lambda^s)(lambda^{*s} + lambda^s - 2 lambda^s) ]
```
  ^^^ VERIFY THIS DENOMINATOR AGAINST THE PDF before coding — OCR-fragile.
coefficients (Eq. 44-45), with `chi~^s = (lambda^{ss} - lambda^s) chi^s`:
```
a^L = (1 + v*)/2,   a^R = (1 - v*)/2
d^s = (1/2)(v^s - v*) chi~^s + (1/2)(|lambda^{ss}| - v* lambda^{ss})
v^s = (|lambda^{sR}| - |lambda^{sL}|)/(lambda^{sR} - lambda^{sL})
v*  = (|lambda^{*R}| - |lambda^{*L}|)/(lambda^{*R} - lambda^{*L})
```
**degenerate guard (Eq. 46):** when `Bx -> 0` the two Alfvén waves collapse onto
the contact (`lambda^{sL},lambda^{sR} -> lambda*`) and `v*` becomes ill-defined at
stagnation points. switch to the three-wave (HLLC) limit by setting `v* = 0`
whenever `|lambda^{sR} - lambda^{sL}| <= eps |lambda^R - lambda^L|`, eps = 1e-9.
least diffusive: the Alfvén speeds bound the transverse-field diffusion, far
below the fast speed.

### 3.5 RMHD HLLD — the BOUNDED DISSIPATIVE-FLUX form (derived 2026-06-24)

**Why the classical coefficients can't be ported.** the classical `d^s` (Eq. 44)
contains `chi~^s = (lambda^{ss}-lambda^s) chi^s`, with `chi^s = (B_t^{ss}-B_t^s)/B_t^s`
the FRACTIONAL transverse-field jump. classically this is SPEED-ONLY because the
single-star transverse field is a SCALAR multiple of the upstream:
```
classical:   B_t^s* = B_t^s · f(speeds, rho)      ->   chi^s = f - 1   (B_t cancels; finite)
```
relativistically it is NOT. expanding MUB09 Eq. (21) with the wave-bracket
`R_{By} = lambda B_y - F_{By} = B_y(lambda - v_x) + v_y B^x`:
```
relativistic:  B_y^s* = B_y^s  +  B^x (v_y^s - v_y^{s*}) / (lambda^s - v_x^{s*})
                              \________ additive; does NOT scale with B_y ________/
->  chi^s = B^x (v_y^s - v_y^{s*}) / [ B_y^s (lambda^s - v_x^{s*}) ]
```
the magnetic-velocity coupling `B^x (v_y - v_y*)` (the Lorentz-factor remnant; it
vanishes in MUB09's non-rel limit Section 3.4.2, and at `B^x=0`) breaks the
cancellation: as `B_y^s -> 0` the numerator stays finite (it is `prop B^x`, nonzero
exactly where HLLD beats HLL), so `chi^s -> infinity`. the coefficient `d^s` is
GENUINELY SINGULAR at every transverse-field zero (OT current sheets, the wind
equator). a closed form for `d^s` would need a B_FLOOR regularization — masking,
not robustness. **the singularity is in the RATIO, and the EMF never needs the
ratio.**

**The fix: the WAVE-SUM dissipative flux (M&DZ Eq. 39), not the coefficient form.**
the `d^s` COEFFICIENT form (Eq. 44) bakes in `chi~^s` as the VELOCITY closed-form
(Eq. 42) — derived from the CLASSICAL Miyoshi-Kusano jump conditions, NOT valid
relativistically. substituting a relativistic B-ratio `chi` into Eq. 44 does NOT
reproduce the RMHD flux (VERIFIED: telescoping test gave F_hat=1.03 vs 0.144). the
ROBUST, relativistically-EXACT object is the wave-sum form, M&DZ Eq. 39 — the HLLD
flux written as central minus the per-wave dissipation over the ACTUAL star fields:
```
F^[By]  =  (1/2)[ F^L + F^R
                  - |lambda^L|  (B_y^{sL} - B_y^L)
                  - |lambda^{sL}|(B_c^y    - B_y^{sL})
                  - |lambda^{sR}|(B_y^{sR} - B_c^y)
                  - |lambda^R|  (B_y^R    - B_y^{sR}) ]
   F^s = v_x^s B_y^s - v_y^s B^x   (per-side induction flux)
```
`B_y^{sL},B_y^{sR}` = single-star fields (MUB09 Eq. 21, `hlld_rmhd_states.bstar`);
`B_c^y` = double-star == contact field (MUB09 Eq. 45, `hlld_rmhd_states.bc`);
`lambda^{L,R}` fast (`lam`), `lambda^{sL,sR}` Alfvén (`alf`). this is BOUNDED (all
FIELD DIFFERENCES weighted by `|speed|`; no ratio, no `1/B_y`), and the wave-fan
telescoping identity `F = (1/2)(F^L+F^R) - (1/2)sum|lambda_k| dU_k` makes it the
EXACT Godunov flux for ANY consistent fan. the dissipative part is just
`Phi^[By] = (1/2)(F^L+F^R) - F^[By] = (1/2) sum_k |lambda_k| (dB_y)_k` (M&DZ Eq. 18).

**Three checks (the gate before any rebuild):**

1. **finite as `B_y -> 0`** ✓ — `Phi` is `sum |lambda_k| (B_y star-field jumps)`; every
   term is a bounded field difference. no ratio anywhere.

2. **exact HLLD flux** ✓ — the telescoping identity holds for any consistent wave fan,
   so `F^[By]` (Eq. 39) IS the HLLD flux by construction (not an approximation).

3. **grid-aligned reduction** ✓ VERIFIED (test `hlld_rmhd_uct_telescopes_to_flux`):
   `F^[By]` (Eq. 39, from `bstar` + `bc`) == `hlld_rmhd().mag[1]` to MACHINE PRECISION
   (diff 0.0e0). the contact identity `hlld_rmhd.mag[1] == lambda* B_c^y - v_c^y B^x`
   also holds to 1e-10 (test `hlld_rmhd_emf_reduces_to_by_flux`). the star-field
   extraction (`bstar`, `bc`, `vc`) is fully validated.

**2D edge EMF (the implementation target).** in 1D the EMF is `E_z = -F^[By]`. for the
2D corner, follow the UCT framework (M&DZ Eq. 23-24, the CT-Flux composition): the
dissipative flux `Phi` is BOUNDED, reconstruct it to the edge and add to the centered
advective:
```
E_z = -[centered advective]  +  (1/2)(Phi_x^N + Phi_x^S)  -  (1/2)(Phi_y^W + Phi_y^E)
```
where each face `Phi` is the Eq.-39 wave-sum dissipation computed from `hlld_rmhd_states`
called with the STAGGERED transverse face fields as the Riemann L/R (so `Phi` damps the
staggered checkerboard, M&DZ point 1-2 p.8) + the cell velocities/rho/pre. reduces to
`-F^[By]` in 1D; less diffusive than HLL because the Alfven waves split the jump;
bounded by construction. div(B)=0 preserved (single-valued edge EMF).

**Why this is robust, not a hack:** `Phi` is the EXACT HLLD numerical dissipation (Eq.
39 verified == the solver flux). no coefficient back-calculation, no `1/B_y`, no floor,
no clamp. the only modelling choice is the standard UCT one: use the STAGGERED transverse
field in the EMF Riemann (a CT-consistency requirement, not an approximation).

**Implementation delta** (one rebuild): `hlld_rmhd_states` now returns `bc, vc` [DONE].
rewrite `rmhd_edge_emf_uct_hlld_gv`: per face, call `hlld_rmhd_states` with staggered
transverse B as L/R, form `Phi` (Eq. 39 wave-sum), compose per the 2D formula above.
DROP the entire `side_d`/`d^s`/B-ratio/d-clamp/straddle/FORCE_HLL scaffolding — the
wave-sum is bounded and correct by construction. gate `Phi` on `success` (HLL `Phi`
where the secant fails).

## 3.6 THE FULL RECIPE — EXACT, no corner cuts (M&DZ Eq. 16-35)

the EMF FORMULA (Eq. 27/28) we implemented is CORRECT (verified algebraically for
uniform v). the loop dies because we cut TWO corners on the INPUTS to that formula.
follow this to the letter.

**geometry (z-edge at corner `(i+1/2, j+1/2)`):**
- `B_y` lives on Y-faces: `B_y^W` at `(i, j+1/2)`, `B_y^E` at `(i+1, j+1/2)` (flank edge in x).
- `B_x` lives on X-faces: `B_x^S` at `(i+1/2, j)`, `B_x^N` at `(i+1/2, j+1)` (flank edge in y).
- the X-faces carrying the `B_y`-flux are `x_f=(i+1/2,j)` and `x_f+e_y=(i+1/2,j+1)`
  (S and N of the edge). the Y-faces carrying the `B_x`-flux are `y_f`/`y_f+e_x` (W,E).

**CORNER 1 — per-face signal speeds (Eq. 27), NOT max-over-4-cells:**
```
alpha_x^+ = max(0, lambda^R_{x_f}, lambda^R_{x_f+e_y})   # max over the 2 ADJACENT X-FACES
alpha_x^- = -min(0, lambda^L_{x_f}, lambda^L_{x_f+e_y})
```
the `lambda` are the per-FACE 1D-Riemann signal speeds (the same the gas flux used),
NOT cell wave speeds maxed over 4 neighbours.

**CORNER 2 — transverse R± reconstruction of EVERY input (Eq. 16-18):**
the centered fluxes, the dissipations, AND the transverse velocities are each
PLM-reconstructed FROM THE FACE TO THE EDGE in the transverse direction:
```
E_z^S = R_y^+( -F^[B_y]_{x_f} ),   E_z^N = R_y^-( -F^[B_y]_{x_f+e_y} )   # x-face flux, reconstruct in Y
E_z^W = R_x^+(  F^[B_x]_{y_f} ),   E_z^E = R_x^-(  F^[B_x]_{y_f+e_x} )   # y-face flux, reconstruct in X
phi_x^{S,N} = R_y^{+,-}( Phi^[B_y]_{x_f, x_f+e_y} ),  phi_y^{W,E} = R_x^{+,-}( Phi^[B_x]_{y_f,y_f+e_x} )
Phi^[B_y]_{x_f} = F^[B_y]_{x_f}(centered, Eq.22) - Fhat^[B_y]_{x_f}(Riemann numerical flux = bflux)   # Eq.18
```
`R_y^+` reconstructs the S-face value UP to the edge (`+1/2 slope_y`), `R_y^-` the
N-face DOWN; minmod/plm_theta slope. for UCT-HLL only the field+velocity need
reconstruction (Eq. 28), reducing cost.

**UCT-HLL EMF, to the letter (Eq. 28):**
```
Ez = -[ alpha_x^+ (vbar_x B_y)^W + alpha_x^- (vbar_x B_y)^E - alpha_x^+ alpha_x^- (B_y^E - B_y^W) ] / (alpha_x^+ + alpha_x^-)
     +[ alpha_y^+ (vbar_y B_x)^S + alpha_y^- (vbar_y B_x)^N - alpha_y^+ alpha_y^- (B_x^N - B_x^S) ] / (alpha_y^+ + alpha_y^-)
vbar_{t,x_f} = (alpha_x^+ v_t^L + alpha_x^- v_t^R)/(alpha_x^+ + alpha_x^-)   # Eq.29, then R-reconstructed transversely
```
`(vbar_x B_y)^W`, `B_y^W`, `B_y^E` are the staggered Y-faces + the Eq.29 velocity,
ALL transversely reconstructed to the edge.

**UCT-HLLD:** same composition, swap `alpha^±` advective weights for the HLLD
`a^L,a^R,d^L,d^R` (Eq. 44-46 classical / the wave-sum Phi 3.5 for RMHD), coefficients
AVERAGED from the 2 adjacent faces (Eq. 34-35: `d_x^W=(d^L_{x_f}+d^L_{x_f+e_y})/2`).
the SAME transverse R± reconstruction applies.

**what simbi has vs needs:** the `plm_theta` minmod machinery EXISTS (gas reconstruction)
— it must be applied to the EMF inputs (the staggered fields, the per-face fluxes/Phi,
the velocities) in the transverse direction. that is the whole fix.

## 4. REGIME NOTES (what works where)

- **HLL is regime-generic** — only the two fast speeds; identical for NMHD / IMHD
  (isothermal) / RMHD. ship it everywhere.
- **HLLC, HLLD as written above are CLASSICAL** (Miyoshi-Kusano). they are correct
  for NMHD and IMHD (isothermal HLLD: see the paper's Appendix A — the chi^s
  coefficients differ for the isothermal EOS).
- **RMHD HLLC / HLLD need re-derived coefficients.** the STRUCTURE (master Eq. 30/33
  + the per-solver a,d) is unchanged, but the intermediate-state algebra is the
  relativistic one: the relativistic HLLC (Mignone & Bodo 2006) and relativistic
  HLLD (Mignone, Ukwatta-style / Mignone, Mattia, Bodo) fans. derive `chi^s`,
  `lambda*`, `lambda^{s,L/R}`, `rho^{*s}` from the relativistic jump conditions,
  following the SAME logic. the paper notes this generalization is straightforward
  in principle but does not carry it out.

## 5. IMPLEMENTATION MAPPING (simbi)

current state (as of this writing):
- `CtMethod::{Contact, Uct}` in `symbi-sim/src/state.rs`; selectable per-config
  (`ct_method` field) -> binding -> `MhdSubstrateKernelSet.ct_method` ->
  `mhd_substrate::efield(sim, ct_method)`.
- `rmhd_edge_emf_uct_gv` (gv.rs) implements the **MASTER form (Eq. 33)** [DONE],
  geometry-agnostic, baked as `rmhd_edge_emf_uct_2d_2`. solver coefficients via a
  pluggable `UctDir { al, ar, d }` from `uct_hll_coeffs(ap, am)` (Eq. 32). the
  advective part uses the upwind `vbar` (Eq. 29, computed in-kernel from the edge
  speeds + cell velocities) and the staggered face B; edge speeds are MAX over the
  4 cells. verified stable on OT (UCT-HLL diffusive vs contact, finite, div-free).
- per-cell Riemann speeds `wave_speed_l/r[d]` materialized over `geom.allocated`
  by `rmhd_wave_speeds_cell` (RMHD only; `R::SPEC.materializes_wave_speeds`).
- `v_upwind[dir][t]` fields ALLOCATED but still UNUSED (HLL computes vbar in-kernel;
  HLLC/HLLD may use them once the intermediate states need a stored pass).
- CLI `--ct-method` plumbing is BROKEN (pass-through arg not applied); setting
  `ct_method` in the config (default or kwarg) works. TODO: fix the CLI path.

to reach the full HLL/HLLC/HLLD set:
1. [DONE] master EMF kernel (Eq. 33) with the pluggable `UctDir { al, ar, dl, dr }`.
   `uct_hll_coeffs` (Eq. 32) + `uct_hllc_coeffs` (Eq. 37-38) implemented.
2. [DONE] **classical wave-speed materialization**: `nmhd_wave_speeds_cell_gv`
   (mirrors the RMHD one via `NewtonianMhd::wave_speeds`); the substrate runs it for
   NMHD when `ct_method == Uct` (RMHD always does). WITHOUT this, classical UCT
   silently falls back to contact (NMHD/IMHD don't otherwise store wave_speed_l/r).
3. [CORRECTED] **UCT-HLLC == UCT-HLL for the EMF.** an earlier attempt applied the Eq.-38
   `B_x=0` degenerate coefficients EVERYWHERE (+ a `|lambda^s|` vs `|lambda*|` typo), which
   produced negative `d` (needing a floor) and a SPURIOUS "3x less diffusive" result. that
   was wrong: HLLC's contact wave does not resolve `B_t` for `B_x != 0` (p.11), so the
   correct UCT-HLLC is just UCT-HLL. the dispatch now sends `CtMethod::Uct` -> the HLL
   master kernel for all regimes. the `nmhd_edge_emf_uct_hllc_gv` kernel + `uct_hllc_coeffs`
   (with the corrected `|lambda*|/2`, per-face consistent `lstar`) remain as DORMANT building
   blocks for HLLD's `B_x -> 0` limit; they are NOT dispatched.
   VERIFIED: NMHD OT UCT now == HLL (magE 0.034), stable, no floor.
4. [DONE] **UCT-HLLD** (`nmhd_edge_emf_uct_hlld_gv`): the five-wave fan, classical ideal-gas.
   per-face intermediate states (Eq. 40-46): `lambda*` (contact), `rho*^{L,R}`, `lambda*^{L,R}`
   (rotational, via the STAGGERED face `B_n`), `chitilde^s` (Eq. 42 with the singular
   `(lambda*^s-lambda^s)` factor CANCELLED -> `(v_n^s-lambda*)(lambda^s-lambda*)/(lambda*^s+lambda^s-2lambda*)`,
   verified vs both stated limits), per-side `v^s`, `v*` + the `B_x->0` degenerate guard (Eq. 46).
   `d^s` MAX-combined; `a^L=(1+v*)/2` averaged. NO floor. uses `uct_master_emf_perside` — the
   PER-SIDE conservative advective form, REQUIRED when `a^L != a^R` (a single vbar makes the
   `v*` term anti-diffusive and BLOWS UP; that was the only bug). dispatched via NMHD+HLLD gas.
   VERIFIED on NMHD OT: stable, finite, magE 0.088 (~2.7x less diffusive than HLL 0.032; the
   least-diffusive checkerboard-safe EMF).
5. TODO: IMHD (isothermal `p = cs^2 rho`); RMHD HLLC/HLLD (re-derive `lstar`/rotational from the
   relativistic jump conditions, same `UctDir` + per-side master); a proper EMF-family selector.

geometry: the EMF is METRIC-FREE (only the curl carries the metric), so ONE bake
per dimensionality serves cartesian + spherical + cylindrical (as the contact and
compact-HLL kernels already do).

div(B): unchanged — still the curl of one edge EMF, telescopes to zero regardless
of the coefficient set. only accuracy/dissipation changes.

## 6. SANITY CHECKS

- symmetric speeds (`alpha^+ = alpha^-`): HLL `d -> alpha/2`, the EMF reduces to
  arithmetic-average CT + LLF diffusion. good smoke test.
- Orszag-Tang: contact vs UCT-HLL must DIFFER (full-domain dynamics); UCT must
  stay bounded. anti-diffusive sign error blows it to |B|~1e3 (seen & fixed).
- ordering: `wave_speeds -> flux -> efield` (the phase table) so the speeds /
  per-face coefficients are materialized before the edge EMF reads them.
- HLLD -> HLLC limit (`v* = 0`, Eq. 46) and HLLC -> HLL limit are continuous;
  test the degenerate `Bx -> 0` path.
