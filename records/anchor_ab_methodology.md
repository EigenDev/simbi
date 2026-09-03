# Projection-anchor A/B: methodology and verdict

This records why the GRMHD admissible-boundary projection wires behind the
**Eulerian-rebuild** anchor, so the decision is reproducible after the
two-arm experiment apparatus was retired into production.

## The question

When the fallback ladder's admissible-boundary projection fires, it blends an
inadmissible candidate cell toward an *anchor* state. Two anchors were
candidates:

- **stage_input**: the stage-input conserved slots directly, paired with the
  constrained-transport-advanced cell B — a hybrid whose admissibility in that
  magnetic slice is unproven;
- **eulerian_rebuilt**: p2c from the stage-input gas primitives with the
  candidate cell B, admissible in the candidate slice by construction.

## The measurement

Both anchors were run on the identical firing projection, holding everything
else fixed, and compared on their booked receipts. The controlled setup:

- config: `simbi_configs/examples/grmhd/gr_fishbone_moncrief_mhd_cartesian.py`
  (3D Cartesian Kerr-Schild — 3D develops the near-horizon MRI turbulence that
  fires the projection; a 2D axisymmetric torus never trips it and the two
  anchors are then identical to machine precision);
- `--kerr-spin 0` (Schwarzschild-Kerr-Schild), `--kappa 1.05` (a compact torus
  that fits the box at reduced resolution);
- resolutions 32³, 48³, 64³ to `--end-time 5`, and a 48³ pair to `--end-time 8`
  at `--target-beta 1`;
- the arms selected by the (then-live) `SIMBI_ANCHOR_EXPERIMENT` toggle, one arm
  per process, receipts read from the anchor-experiment report after each run.

## The result

The **anchor-energy raise** — the non-physical energy the projection adds to
lift the anchor to the admissibility margin — is the resolution-stable
discriminator, normalized by the projection's own segment-energy activity:

| resolution | eulerian_rebuilt raise/seg | stage_input raise/seg |
|---|---|---|
| 32³ | 1.4e-6 | 1.2% |
| 48³ | 1.2e-6 | 1.8% |
| 64³ | 4.2e-6 | 2.2% |

The Eulerian-rebuild anchor's raise is a millionth of the energy the projection
itself moves, essentially constant across resolution and plasma-beta: its anchor
is admissible by construction. The stage-input anchor's raise is a percent-level
fraction that grows with resolution: its anchor sits on the admissibility
boundary and only survives by a non-physical energy injection. Both arms survive
in the accessible regime because the raise repairs the marginal stage anchor, so
the historical dt-underflow collapse was not reproduced here; the injection
magnitudes are non-monotone across resolution (turbulent-realization noise), so
no order of convergence is claimed.

Raw receipts: `records/anchor_ab_3d_results.json`.

## The verdict

Stage-input anchoring is a numerically rescued hybrid; the Eulerian rebuild
supplies the lawful anchor. Production uses the Eulerian rebuild, so the
projection's receipts now book into the production `ProjectionLedger` behind that
convention — evidence about production physics, not a second correction. The
comparison analyzer that produced the tables above is preserved at
`simbi/analysis/anchor_ab.py`. Reproducing the live two-arm sweep requires
restoring the retired apparatus (git history: the projection-anchor measurement
apparatus and its capture producer).
