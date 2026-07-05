// =============================================================================
// godunov.rs
//
// the conserved-update godunov family: snapshot, ssp stage, fused sources, and the unified dag-application operator.
// =============================================================================

use super::*;
use symbi_geometry::{KerrKS, Schwarzschild, SchwarzschildKS, SchwarzschildKSCartesian, SchwarzschildKSCylindrical};
use symbi_geometry::grhd_source::{grhd_covariant_source, grmhd_covariant_source};
use symbi_algebra::Tensor;
use symbi_ir::dual::Dual;


/// snapshot `u_n = cons` — a pure pointwise copy (the RK2 stage-0 hold), geometry-INDEPENDENT
/// (works for every coord system). copies the energy too when `has_energy`. write root == the
/// read field node (a direct buffer copy).
pub fn snapshot_gv(ncomp: usize, has_energy: bool) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let den = Gv::field("cons_den", FieldRef::cons_den());
    let mom: Vec<Gv> = (0..ncomp)
        .map(|k| Gv::field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8)))
        .collect();
    let nrg = has_energy.then(|| Gv::field("cons_nrg", FieldRef::cons_nrg()));
    let mut writes = vec![("u_n_den".to_string(), FieldRef::un_den().into(), den.node())];
    for (k, m) in mom.iter().enumerate() {
        writes.push((format!("u_n_mom_{k}"), FieldRef::un_mom(k as u8).into(), m.node()));
    }
    if let Some(n) = nrg {
        writes.push(("u_n_nrg".to_string(), FieldRef::un_nrg().into(), n.node()));
    }
    (end_trace(), writes)
}


/// a componentwise field copy for the FOFC scratch: `dst = src` over the gas conserved
/// (den, mom[k], nrg?) and, when `include_prim`, the primitive (rho, vel[k], pre?). used to (a)
/// snapshot the high-order cons+prim into `u_fofc`/`prim_fofc` before the substage is redone at
/// first order, and (b) restore `cons <- u_stage` (cons only) so the redo reconstructs from the
/// physical stage-input state. explicit-field dispatch: slots `s_*` (source) -> `d_*` (dest).
pub fn fofc_copy_gv(ncomp: usize, has_energy: bool, include_prim: bool) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let mut writes: Vec<(String, FieldBind, NodeId)> = Vec::new();
    let mut cp = |name: &str| {
        let v = Gv::field(&format!("s_{name}"), &format!("s_{name}"));
        writes.push((format!("d_{name}"), format!("d_{name}").into(), v.node()));
    };
    cp("den");
    for k in 0..ncomp {
        cp(&format!("mom_{k}"));
    }
    if has_energy {
        cp("nrg");
    }
    if include_prim {
        cp("rho");
        for k in 0..ncomp {
            cp(&format!("vel_{k}"));
        }
        if has_energy {
            cp("pre");
        }
    }
    (end_trace(), writes)
}


/// the FIRST-ORDER FLUX-CORRECTION select: `out = physical(ho_prim) ? ho : fo`, componentwise over
/// the gas conserved (den, mom[k], nrg?) and primitive (rho, vel[k], pre?). the high-order state
/// `ho_*` is the snapshot taken before the substage was redone at first order; `fo_*` is the redone
/// (PCM + HLLE) result, aliased to the live cons/prim `out_*` (in-place read+write). the failure
/// test is metric-free: the c2p velocity ceiling keeps |v| < 1, so an unphysical recovery shows up
/// as rho <= 0, pre <= 0, or NaN (all of which fail `> 0`), never as superluminal. so a cell whose
/// HIGH-ORDER c2p is physical keeps its sharp state; only the failed cells take the diffusive
/// first-order result. carrier-generic, regime-generic (has_energy toggles the pressure law).
/// the FOFC HOST GATE probe: write 1 to the scratch where the high-order c2p is unphysical (density
/// or, for an energy regime, pressure non-finite or non-positive), else 0. a max-reduce over the
/// interior is > 0 exactly when some zone needs correcting; a clean substage reduces to 0 and skips
/// the whole FOFC pass (which would keep the high-order everywhere anyway — bit-identical to skip).
pub fn fofc_probe_gv(has_energy: bool) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let finite_pos = |v: Gv| (v - v).cmp_eq(Gv::ZERO) & v.cmp_gt(Gv::ZERO);
    let rho = Gv::field("prim_rho", FieldRef::PrimRho);
    let physical = if has_energy {
        let pre = Gv::field("prim_pre", FieldRef::PrimPre);
        finite_pos(rho) & finite_pos(pre)
    } else {
        finite_pos(rho)
    };
    let flag = Gv::select(physical, Gv::ZERO, Gv::ONE);
    (end_trace(), vec![("flag".to_string(), FieldRef::Scratch.into(), flag.node())])
}


pub fn fofc_select_gv(ncomp: usize, has_energy: bool) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    // finite AND positive: (v - v) is 0 for a finite value and NaN for NaN OR +-inf (inf - inf =
    // NaN), so cmp_eq(0) rejects every non-finite value; the > 0 rejects a vacuum/negative one. a
    // "physical" tier is one whose density (and pressure, when modelled) passes both.
    let finite_pos = |v: Gv| (v - v).cmp_eq(Gv::ZERO) & v.cmp_gt(Gv::ZERO);
    let ho_rho = Gv::field("ho_rho", "ho_rho");
    let x_rho = Gv::field("x_rho", "x_rho");
    let (physical_ho, physical_fo) = if has_energy {
        let ho_pre = Gv::field("ho_pre", "ho_pre");
        let x_pre = Gv::field("x_pre", "x_pre");
        (finite_pos(ho_rho) & finite_pos(ho_pre), finite_pos(x_rho) & finite_pos(x_pre))
    } else {
        (finite_pos(ho_rho), finite_pos(x_rho))
    };
    // THREE-TIER conserved select: the high-order zone if it is physical, else the first-order redo
    // if THAT is physical, else FREEZE to the stage-input state u_stage (`us_*`) — the pre-godunov
    // conserved, which already recovered its primitive at stage entry, so it is admissible and the
    // final c2p converges on it. the finiteness guard makes the frozen tier unconditional: a zone no
    // flux can update admissibly holds its stage-input value rather than propagating a NaN. only the
    // conserved is chosen here; the primitive is re-derived by the c2p that follows the select.
    let mut writes: Vec<(String, FieldBind, NodeId)> = Vec::new();
    // the live cons (`x_*`) is read+write IN PLACE: it holds the first-order result and is
    // overwritten with the chosen tier. one slot per component (read path == write path) so the IR
    // dedups it to a single in-place binding (the CT-`b` pattern) — no input/output aliasing.
    let mut sel_inplace = |comp: &str, ho: Gv, us: Gv| {
        let path = format!("x_{comp}");
        let x = Gv::field(&path, &path);
        let chosen = Gv::select(physical_ho, ho, Gv::select(physical_fo, x, us));
        writes.push((path.clone(), path.into(), chosen.node()));
    };
    let ho_den = Gv::field("ho_den", "ho_den");
    let us_den = Gv::field("us_den", "us_den");
    sel_inplace("den", ho_den, us_den);
    for k in 0..ncomp {
        let ho = Gv::field(&format!("ho_mom_{k}"), &format!("ho_mom_{k}"));
        let us = Gv::field(&format!("us_mom_{k}"), &format!("us_mom_{k}"));
        sel_inplace(&format!("mom_{k}"), ho, us);
    }
    if has_energy {
        let ho = Gv::field("ho_nrg", "ho_nrg");
        let us = Gv::field("us_nrg", "us_nrg");
        sel_inplace("nrg", ho, us);
    }
    (end_trace(), writes)
}


/// the single mass-law godunov step to a SEPARATE output buffer (the P2.2 demo):
/// `rho_new = rho - dt*div(mass_flux)`. cartesian-uniform OR curvilinear (area-weighted).
/// write -> `cons.den_new`.
pub fn godunov_mass_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let dt = Gv::scalar("dt");
    let geo = (!is_cartesian_uniform(coords, spacing))
        .then(|| cell_geometry_gv(coords, spacing, axes, ndim as usize));
    let rho = Gv::field("rho", FieldRef::cons_den());
    let rho_new = rho - dt * gv_divergence("mass_flux", ndim, &geo);
    let writes = vec![("rho_new".to_string(), "cons.den_new".into(), rho_new.node())];
    (end_trace(), writes)
}


/// the in-place SSP Shu-Osher stage update `cons = a0*u_n + ac*fe(cons)`, where the
/// forward-Euler operator is `fe(u) = u - dt*div(F) (+ dt*S_geom)`. ONE builder for every
/// explicit SSP scheme: the per-stage convex coefficients `(a0, ac)` arrive as RUNTIME
/// scalars, so a SINGLE compiled kernel serves forward-Euler `[(0,1)]`, SSP-RK2
/// `[(0,1),(1/2,1/2)]`, and SSP-RK3 `[(0,1),(3/4,1/4),(1/3,2/3)]` — the integrator is data,
/// not codegen. forward-Euler is the `(a0,ac)=(0,1)` instantiation (the `a0*u_n` term reads
/// the snapshot held by `snapshot_gv` and multiplies it by 0).
///
/// mass + one scalar law per momentum component (+ energy when `has_energy`). cartesian =
/// unweighted divergence, no source; curvilinear = area-weighted divergence + the geometric
/// momentum `source` carried inside the forward-Euler stage. write path == input path (in
/// place). EOS- AND geom-generic.
///
/// this is the no-overlay case of [`godunov_stage_gv_with_fused_sources`] — the full stage
/// body lives there, and the empty source slice traces exactly the plain SSP stage (the splice
/// helper short-circuits on no overlays, so there are no dead vocabulary nodes). kept as a named
/// entry point for the common no-source case.
pub fn godunov_stage_gv(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    source: GeoSource,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    godunov_stage_gv_with_fused_sources(coords, spacetime, spacing, axes, ndim, ncomp, has_energy, source, &[], false)
}


/// per-field NodeId contributions from a list of spec sources, bucketed by
/// `target_field`. consumed by `godunov_stage_gv_with_fused_sources` — the spec
/// vocabulary is spliced once, then each conserved law adds its bucket inside the
/// forward-Euler stage.
///
/// **structural shape contract**: spliced outputs MUST have the expected per-target
/// arity (1 for den/nrg, D for mom); spec authors that violate this get a panic, not a
/// silent wrong-component write.
struct FusedContribs {
    /// each entry is a `S_den` NodeId to add to `rho_new`.
    den: Vec<NodeId>,
    /// `mom[k]` is the list of `S_mom_k` NodeIds for momentum component k.
    mom: Vec<Vec<NodeId>>,
    /// each entry is a `S_nrg` NodeId to add to `nrg_new`.
    nrg: Vec<NodeId>,
    /// `mag[k]` is the per-component cell-B prescription, ONLY for a driven-boundary
    /// (`WriteMode::Assign`) MHD `bcell` slot. unused (empty) for hydro and for the
    /// accumulate (godunov source) path — the conservation-law lifts never target B.
    mag: Vec<Vec<NodeId>>,
}


/// **B6-iv Phase 4c/2c — fused-source splice helper**. requires an ACTIVE Gv trace
/// (the caller holds `begin_trace` / `end_trace`). builds the shared primitive
/// vocabulary (`rho`, `vel_k`, lazy `x_k` ↔ centroid), then splices every
/// spec into the trace and buckets the outputs by `target_field`. with no overlays it
/// returns empty buckets WITHOUT touching the trace — so the no-source `godunov_stage_gv`
/// wrapper traces exactly the plain SSP stage, no dead `mom/rho` vocabulary nodes.
fn splice_fused_sources_to_contribs(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    geo: &Option<CellGeometryGv>,
    // the STATE vocabulary the DAG reads `rho`/`vel_k` from (docs/design/33 `StateEnv`). `Some((rho,
    // mom))` binds them (sources read the stage/conserved state); `None` binds NOTHING from state — a
    // pure coordinate prescription (a driven boundary, whose DAG outputs the state rather than reading
    // it). `x_k` (centroid) + scalar params are bound regardless.
    state: Option<(Gv, &[Gv])>,
    // (target_field, built) pairs — the BuiltSource VALUES, so this serves both the AOT path
    // (SourceSpec.build_source(ndim)) and the RUNTIME path (build_user_source's loaded values).
    sources: &[(&str, &symbi_hydro::source_spec::BuiltSource)],
) -> FusedContribs {
    use std::collections::HashMap;

    if sources.is_empty() {
        return FusedContribs { den: Vec::new(), mom: vec![Vec::new(); ncomp], nrg: Vec::new(), mag: vec![Vec::new(); ncomp] };
    }

    // ----- shared primitive vocabulary, declared ONCE; CSE collapses the
    // `mom_k / rho` divisions across every overlay that uses them. bound only when the DAG reads
    // state (`StateEnv::Stage`); a coordinate prescription (`StateEnv::Coord`) skips it.
    let mut shared_params: HashMap<String, NodeId> = HashMap::new();
    if let Some((rho, mom)) = state {
        shared_params.insert("rho".to_string(), rho.node());
        for k in 0..ncomp {
            let v_k = mom[k] / rho;
            shared_params.insert(format!("vel_{k}"), v_k.node());
        }
        // pressure-reading sources (e.g., radiative cooling Lambda(rho, T), T = pre/rho): bind `pre`
        // to the c2p-computed `prim.pre` field. at source-apply / fused-godunov time prim is the SSP
        // stage input (not yet recomputed), so this is consistent with rho/vel above. energy-bearing
        // regimes only — iso has no pressure field. bound ONLY when a source actually references
        // `pre` (mirrors `needs_position`): an unconditional bind adds a manifest `prim.pre` read that
        // DUPLICATES the adiabatic godunov's own flux-reconstruction read -> input/output aliasing.
        let needs_pre = sources.iter().any(|(_, b)| b.params.iter().any(|p| p == "pre"));
        if has_energy && needs_pre {
            shared_params.insert("pre".to_string(), Gv::field("pre", FieldRef::PrimPre).node());
        }
    }
    // **Phase 2c — LAZY centroid binding**. `x_k` ↔ cell centroid for specs
    // that declare position params (gravity, immersed bodies). walk the
    // spec params FIRST to detect which axes are needed, then call
    // `cell_geometry_gv` (which declares `x_lo_k` / `dx_k` scalars in the
    // trace) ONLY if at least one axis is referenced. specs without
    // position dependence keep the prior scalar manifest unchanged.
    let needs_position = sources.iter().any(|(_, built)| {
        built.params.iter().any(|p| (0..(ndim as usize))
            .any(|k| *p == format!("x_{k}")))
    });
    if needs_position {
        let centroid_geo = geo.clone().unwrap_or_else(
            || cell_geometry_gv(coords, spacing, axes, ndim as usize),
        );
        for k in 0..(ndim as usize) {
            shared_params.insert(format!("x_{k}"), centroid_geo.centroid[k].node());
        }
    }

    // scalar-leaf cache so the SAME spec param across multiple overlays
    // (e.g., `g_ext_0` in the mom + nrg specs of uniform_acceleration)
    // resolves to ONE Gv leaf — runtime fills one scalar, CSE collapses.
    let mut scalar_leaves: HashMap<String, NodeId> = HashMap::new();
    let mut out = FusedContribs {
        den: Vec::new(),
        mom: vec![Vec::new(); ncomp],
        nrg: Vec::new(),
        mag: vec![Vec::new(); ncomp],
    };
    for (target_field, built) in sources {
        let mut name_to_node = shared_params.clone();
        for pname in &built.params {
            if name_to_node.contains_key(pname) { continue; }
            let nid = *scalar_leaves.entry(pname.clone())
                .or_insert_with(|| Gv::scalar(pname).node());
            name_to_node.insert(pname.clone(), nid);
        }
        let spliced = with_trace(|t| {
            symbi_hydro::source_spec::splice_built_source_into(
                built, t.graph(), &name_to_node,
            )
        });
        match *target_field {
            "den" => {
                assert_eq!(spliced.len(), 1,
                    "splice_fused_sources: den overlay must emit 1 scalar, got {}", spliced.len());
                out.den.push(spliced[0]);
            }
            "mom" => {
                assert_eq!(spliced.len(), ncomp,
                    "splice_fused_sources: mom overlay must emit {ncomp} components, got {}",
                    spliced.len());
                for k in 0..ncomp { out.mom[k].push(spliced[k]); }
            }
            "nrg" => {
                assert!(has_energy,
                    "splice_fused_sources: nrg overlay requires has_energy=true");
                assert_eq!(spliced.len(), 1,
                    "splice_fused_sources: nrg overlay must emit 1 scalar, got {}", spliced.len());
                out.nrg.push(spliced[0]);
            }
            // cell-B prescription (MHD driven boundary): the ncomp-component bcell vector.
            // only valid in the Assign (prescription) mode — the conservation-law source lifts
            // never target B, so the accumulate path asserts mag stays empty.
            "bcell" => {
                assert_eq!(spliced.len(), ncomp,
                    "splice_fused_sources: bcell overlay must emit {ncomp} components, got {}",
                    spliced.len());
                for k in 0..ncomp { out.mag[k].push(spliced[k]); }
            }
            other => panic!("splice_fused_sources: unsupported target_field {other:?}"),
        }
    }
    out
}


/// the SSP Shu-Osher stage update WITH a fused list of spec sources — the
/// `godunov_stage_gv` body (runtime `(a0, ac)` convex coefficients, `cons = a0*u_n + ac*fe`)
/// with the spec contributions spliced into the forward-Euler operator:
/// `fe(u, div, src) = u - dt*div + dt*(geo_src + Σ spec_src)`. one launch folds flux
/// divergence + geometric source + every user overlay + the integrator combine. the dispatch
/// `{prefix}_godunov_stage_with_{slug}_{D}d` resolves here.
///
/// the spec contributions live inside `fe`, so the stage's `ac` weight multiplies them — the
/// same convex coefficient that weights the flux divergence — which is exactly the SSP
/// source treatment (`ac*dt*S` per stage). pass an empty slice for the no-overlay variant.
///
/// this is the COMPILE-TIME entry: it materializes each `SourceSpec`'s `BuiltSource`
/// (`build_source(ndim)`) then delegates to [`godunov_stage_gv_with_fused_built`], the core
/// over `BuiltSource` VALUES that the AOT bake and the RUNTIME user-source path share. the
/// godunov+source trace lives ONCE, in the core.
pub fn godunov_stage_gv_with_fused_sources(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    source: GeoSource,
    user_sources: &[&symbi_hydro::source_spec::SourceSpec],
    // when this stage is FUSED with the cell-B predictor, the magnetic geo source reads cell-B
    // via the predictor's `bc_k` key so try_fuse merges the two reads (no input/output alias).
    // the plain (unfused) stage passes false -> reads `prim.mag[k]`.
    mag_from_bcell: bool,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    let builts: Vec<(&str, symbi_hydro::source_spec::BuiltSource)> = user_sources.iter()
        .map(|s| (s.target_field, (s.build_source)(ndim as usize)))
        .collect();
    let src_refs: Vec<(&str, &symbi_hydro::source_spec::BuiltSource)> =
        builts.iter().map(|(t, b)| (*t, b)).collect();
    godunov_stage_gv_with_fused_built(
        coords, spacetime, spacing, axes, ndim, ncomp, has_energy, source, &src_refs, mag_from_bcell,
    )
}


/// the SSP stage core over PRE-BUILT sources — `BuiltSource` VALUES paired with their target
/// field, the shape `splice_fused_sources_to_contribs` consumes. the SourceSpec entry
/// [`godunov_stage_gv_with_fused_sources`] feeds AOT specs (`build_source(ndim)`); the runtime
/// user-source CPU fusion feeds `RuntimeSource`'s loaded `BuiltSource`s directly. ONE trace, both
/// paths — no duplicated godunov+source lowering. `sources` is `(target_field, built)` pairs.
#[allow(clippy::too_many_arguments)]
pub fn godunov_stage_gv_with_fused_built(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    source: GeoSource,
    sources: &[(&str, &symbi_hydro::source_spec::BuiltSource)],
    mag_from_bcell: bool,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let dt = Gv::scalar("dt");
    let a0 = Gv::scalar("a0");
    let ac = Gv::scalar("ac");
    // the SSP source weight. computed as `ac*dt` so it is BIT-IDENTICAL to the standalone
    // `source_apply_gv` pass's `dt` scalar (the driver fills that with `ac*sim.dt` — the same IEEE
    // f64 product). this is what makes `fused == plain godunov + source_apply` exact, not just
    // ULP-close: the user source is added as a SEPARATE post-combine term with this weight, never
    // folded into the `ac*fe` multiply (which would distribute the rounding differently).
    let ac_dt = ac * dt;
    // flat spacetime: the physical (orthonormal) finite-volume geometry. curved (GR): the
    // COVARIANT geometry — coordinate-form angular face weights (the alpha sqrt(gamma) measure),
    // matching the covariant momentum S_i and the contravariant fluxes v^i; the orthonormal
    // angular weights would leave every theta-direction force on S_theta short by a factor r.
    // radial faces and the volume coincide, so 1D radial GR is bit-identical.
    // a curved spacetime ALWAYS needs the geometry (the metric position + the alpha sqrt(gamma)
    // densitization measure), even on a cartesian-uniform grid where flat hydro skips it.
    let geo = (!is_cartesian_uniform(coords, spacing) || spacetime != Spacetime::Minkowski)
        .then(|| match spacetime {
        Spacetime::Minkowski => cell_geometry_gv(coords, spacing, axes, ndim as usize),
        // spinning kerr: the densitized measure is Sigma sin(theta) — the spin rides the
        // `kerr_spin` kernel scalar into the face/volume moments.
        Spacetime::Kerr => cell_geometry_covariant_gv(
            coords, spacing, axes, ndim as usize, Some(Gv::scalar("kerr_spin")),
        ),
        _ => cell_geometry_covariant_gv(coords, spacing, axes, ndim as usize, None),
    });
    let rho = Gv::field("rho", FieldRef::cons_den());
    let mom: Vec<Gv> = (0..ncomp)
        .map(|k| Gv::field(&format!("mom_{k}"), FieldRef::cons_mom(k as u8)))
        .collect();
    // on a curved background the flat velocity-quadratic inertial is the WRONG contraction for the
    // covariant momentum S_i (it treats the components as flat); the covariant stress-energy
    // contraction below carries those blocks instead, so the hydro geometric source keeps ONLY its
    // discrete well-balanced pressure form `p (A_hi - A_lo) / V` — which cancels the pressure flux
    // divergence bit-exactly at a uniform-p hydrostatic state, unlike the analytic pressure block
    // `p d_j ln(alpha sqrt(gamma))` of the contraction.
    // the ideal-MHD stress moves to the covariant contraction on the GR path too — the flat
    // Rmhd curvilinear source would double-count the inertia/tension with the WRONG (flat)
    // contraction for covariant S_i; only the GAS-pressure discrete block stays.
    let source_discrete = match (spacetime, source) {
        (Spacetime::Minkowski, s) => s,
        (_, GeoSource::Hydro { .. }) | (_, GeoSource::Rmhd) => GeoSource::Hydro { inertial: false },
        (_, s) => s,
    };
    let src = geo
        .as_ref()
        .map(|g| gv_geometric_source(coords, axes, ndim as usize, ncomp, g, source_discrete, &mom, mag_from_bcell));

    let contribs = splice_fused_sources_to_contribs(
        coords, spacing, axes, ndim, ncomp, has_energy, &geo, Some((rho, &mom)), sources,
    );

    // the plain forward-Euler stage carries ONLY the flux divergence + the (well-balanced)
    // geometric source — NOT the user sources. `cons_new = a0*u_n + ac*fe`, identical to
    // `godunov_stage_gv`. the homologous-mesh dilution `-mesh_hdil * u` (with
    // `mesh_hdil = ndim * a_dot / a`, the comoving volume-growth rate) rides every
    // conserved law; the static binding mesh_hdil = 0 subtracts an exact zero.
    let h_dil = Gv::scalar("mesh_hdil");
    // GR densitization (Valencia 3+1, static diagonal background): the spatial RHS — the flux
    // divergence + the geometric momentum source — is weighted by the lapse `alpha(x)`. NOT the
    // `u` snapshot or the mesh-dilution term (those are the time / comoving parts, not the
    // densitized flux). flat spacetime -> `None` -> untouched, bit-identical (see `gv_lapse_weight`).
    // the coordinate-indexed cell centroid (r at slot 0) for the lapse alpha(x); only the
    // curvilinear path carries one (cartesian-uniform geo = None is always Minkowski -> unused).
    let coord_centroid: Vec<Gv> = match &geo {
        Some(g) => {
            let mut c = vec![Gv::ZERO; 3];
            for d in 0..(ndim as usize) {
                c[axes[d]] = g.centroid[d];
            }
            c
        }
        None => Vec::new(),
    };
    assert!(
        spacetime == Spacetime::Minkowski
            || matches!(source, GeoSource::Rmhd | GeoSource::Hydro { .. }),
        "the GR godunov source carries the perfect-fluid or ideal-MHD stress only"
    );
    let lapse = gv_lapse_weight(coords, spacetime, &coord_centroid);
    // the GR geodesic sources from the FULL covariant contraction `grhd_covariant_source`: the
    // per-coordinate momentum source S_j = (1/2) T^{mu nu} d_j g_{mu nu} and the energy source
    // S_tau, one forward-autodiff pass per axis at the metric's full spherical D = 3 (the metric
    // supplies only its ADM line element — no hand-derived christoffels). the MOMENTUM call takes
    // p = 0: the E-part only (gravity + covariant centrifugal), because the pressure block
    // `p d_j ln(alpha sqrt(gamma))` rides the DISCRETE well-balanced form in gv_geometric_source
    // above. the ENERGY call takes the full p — S_tau needs no discrete balance (it vanishes
    // identically at a zero-shift hydrostatic state). the polar angle is the cell centroid when
    // gridded, else pi/2 (exact: with no polar grid every theta-dependence cancels). flat -> None.
    // GRMHD-ready: the EM stress just changes T^{mu nu}.
    let geodesic: Option<(Tensor<Gv, 3>, Gv)> = match spacetime {
        Spacetime::Minkowski => None,
        _ => {
            let mass = Dual::constant(Gv::scalar("schwarzschild_mass")); // constant w.r.t. position
            // coordinate-indexed metric position: each gridded coordinate at its centroid, each
            // ungridded coordinate at its chart symmetry default (spherical polar -> pi/2, else 0).
            let x = Tensor::<Gv, 3>::new(std::array::from_fn(|c| {
                if axes.contains(&c) { coord_centroid[c] } else { gv_ungridded_slot(coords, c) }
            }));
            let e = rho + Gv::field("nrg", FieldRef::cons_nrg()) + Gv::field("pre", FieldRef::PrimPre);
            let p = Gv::field("pre", FieldRef::PrimPre);
            // the CONTRAVARIANT velocity in coordinate slots (the metric-aware c2p output);
            // spherical GR momentum slots are coordinate-ordered, so slot k == coordinate k.
            // coordinates without a momentum slot carry zero.
            let v = Tensor::<Gv, 3>::new(std::array::from_fn(|c| {
                if c < ncomp {
                    Gv::field(&format!("prim_v{c}"), FieldRef::PrimVel(c as u8))
                } else {
                    Gv::ZERO
                }
            }));
            if matches!(source, GeoSource::Rmhd) {
                // GRMHD: the ideal-MHD stress in the same contraction. the source takes the
                // METRIC-FREE rest enthalpy density rho_h = rho + Gamma/(Gamma-1) p (it builds
                // W and b^mu from the harvested gamma internally); B reads the cell field under
                // the same key convention as the discrete magnetic geo source. the momentum call
                // takes p = 0 (the gas-pressure block rides the discrete well-balanced form) but
                // keeps the FULL magnetic stress — the b^2/2 isotropic block is analytic; the
                // one-step-residual instrument adjudicates its balance. spinning kerr is design-44
                // phase C (the dragging-consistent reconstruction does not yet extend to B).
                let gamma_eos = Gv::scalar("gamma");
                let prim_rho = Gv::field("prim_rho", FieldRef::PrimRho);
                let rho_h = prim_rho + gamma_eos / (gamma_eos - Gv::ONE) * p;
                let b = Tensor::<Gv, 3>::new(std::array::from_fn(|k| {
                    if mag_from_bcell {
                        Gv::field(&format!("bc_{k}"), FieldRef::BCell(k as u8))
                    } else {
                        Gv::field(&format!("prim_b{k}"), &format!("prim.mag[{k}]"))
                    }
                }));
                let src_at = |pp: Gv| match spacetime {
                    Spacetime::Schwarzschild => {
                        grmhd_covariant_source(&Schwarzschild { mass }, x, rho_h, v, pp, b)
                    }
                    Spacetime::KerrSchild if coords == Coords::Cartesian => {
                        grmhd_covariant_source(&SchwarzschildKSCartesian { mass }, x, rho_h, v, pp, b)
                    }
                    Spacetime::KerrSchild if coords == Coords::Cylindrical => {
                        grmhd_covariant_source(&SchwarzschildKSCylindrical { mass }, x, rho_h, v, pp, b)
                    }
                    Spacetime::KerrSchild => {
                        grmhd_covariant_source(&SchwarzschildKS { mass }, x, rho_h, v, pp, b)
                    }
                    Spacetime::Kerr => {
                        // the generic covariant stress contraction S_j = (1/2) T^{mu nu} d_j g_{mu nu}
                        // with the EM stress; the non-diagonal kerr metric enters only through the
                        // autodiff Dual pass, no per-block closed form.
                        let spin = Dual::constant(Gv::scalar("kerr_spin"));
                        grmhd_covariant_source(&KerrKS { mass, spin }, x, rho_h, v, pp, b)
                    }
                    Spacetime::Minkowski => unreachable!("flat handled above"),
                };
                let (s_mom, _) = src_at(Gv::ZERO);
                let (_, s_tau) = src_at(p);
                Some((s_mom, s_tau))
            } else {
                let src_at = |pp: Gv| match spacetime {
                    Spacetime::Schwarzschild => grhd_covariant_source(&Schwarzschild { mass }, x, e, v, pp),
                    Spacetime::KerrSchild if coords == Coords::Cartesian => {
                        grhd_covariant_source(&SchwarzschildKSCartesian { mass }, x, e, v, pp)
                    }
                    Spacetime::KerrSchild if coords == Coords::Cylindrical => {
                        grhd_covariant_source(&SchwarzschildKSCylindrical { mass }, x, e, v, pp)
                    }
                    Spacetime::KerrSchild => grhd_covariant_source(&SchwarzschildKS { mass }, x, e, v, pp),
                    Spacetime::Kerr => {
                        let spin = Dual::constant(Gv::scalar("kerr_spin"));
                        grhd_covariant_source(&KerrKS { mass, spin }, x, e, v, pp)
                    }
                    Spacetime::Minkowski => unreachable!("flat handled above"),
                };
                let (s_mom, _) = src_at(Gv::ZERO);
                let (_, s_tau) = src_at(p);
                Some((s_mom, s_tau))
            }
        }
    };
    let mom_gravity: Option<Tensor<Gv, 3>> = geodesic.map(|(s_mom, _)| s_mom);
    // the GR geodesic ENERGY source S_tau — the second output of the contraction (gravity's rate
    // of work on the infalling gas). zero on a flat background.
    let nrg_gravity: Option<Gv> = geodesic.map(|(_, s_tau)| s_tau);
    let fe = |u: Gv, div: Gv, geo_src: Option<Gv>| {
        let div = match lapse { Some(a) => a * div, None => div };
        let mut r = u - dt * div - dt * (h_dil * u);
        if let Some(s) = geo_src {
            let s = match lapse { Some(a) => a * s, None => s };
            r = r + dt * s;
        }
        r
    };
    let combine = |un: Gv, fe: Gv| a0 * un + ac * fe;
    // the USER sources ride as a SEPARATE additive term after the combine: `+ Σ ac*dt*contrib`,
    // accumulated exactly as `source_apply_gv` accumulates it (start from the combine result,
    // `+= ac_dt*contrib` per spec). so the fused kernel IS `plain godunov + the additive pass`,
    // bit-for-bit, fused into one launch (proven by the fused-equivalence test).
    let with_sources = |base: Gv, srcs: &[NodeId]| {
        let mut r = base;
        for c in srcs {
            r = r + ac_dt * Gv::of(*c);
        }
        r
    };

    let u_n_rho = Gv::field("u_n_rho", FieldRef::un_den());
    let rho_new = with_sources(
        combine(u_n_rho, fe(rho, gv_divergence("mass_flux", ndim, &geo), None)),
        &contribs.den,
    );
    let mut writes = vec![("rho".to_string(), FieldRef::cons_den().into(), rho_new.node())];
    for k in 0..ncomp {
        let u_n_mom = Gv::field(&format!("u_n_mom_{k}"), FieldRef::un_mom(k as u8));
        let div = gv_divergence(&format!("mom_flux_{k}"), ndim, &geo);
        let geo_src = src.as_ref().map(|s| s[k]);
        // every momentum slot carries its covariant geodesic block (gravity + covariant
        // centrifugal, coordinate k of the contraction) on top of the discrete pressure form in
        // geo_src; a suppressed axisymmetric slot's block is identically zero (the metric never
        // reads phi, so its autodiff tangent vanishes — angular-momentum conservation).
        let mom_src = match mom_gravity {
            Some(g) => Some(geo_src.map_or(g[k], |s| s + g[k])),
            None => geo_src,
        };
        // Valencia covariant storage: the conserved momentum is the COVARIANT S_i = rho h W^2
        // gamma_ij v^j (the metric-aware c2p + flux), and the geodesic source is written for that
        // covariant S_i, so d_t S_i = -alpha div(F) + alpha S — a SINGLE, uniform lapse on every
        // conserved law, supplied by the `fe` weight. no orthonormal alpha^2 asymmetry: the flux
        // kernel already carries the contravariant v^n (no V_rhat), and the metric coefficient
        // gamma_ij rides inside S_i, not the densitization.
        let mom_new = with_sources(
            combine(u_n_mom, fe(mom[k], div, mom_src)),
            &contribs.mom[k],
        );
        writes.push((format!("mom_{k}"), FieldRef::cons_mom(k as u8).into(), mom_new.node()));
    }
    if has_energy {
        let nrg = Gv::field("nrg", FieldRef::cons_nrg());
        let u_n_nrg = Gv::field("u_n_nrg", FieldRef::un_nrg());
        let nrg_new = with_sources(
            combine(u_n_nrg, fe(nrg, gv_divergence("nrg_flux", ndim, &geo), nrg_gravity)),
            &contribs.nrg,
        );
        writes.push(("nrg".to_string(), FieldRef::cons_nrg().into(), nrg_new.node()));
    }
    (end_trace(), writes)
}


/// the standalone ADDITIVE source pass: `cons += dt * Σ S(prim, x; params)`, in place, per
/// conserved slot, for a list of spec sources. the GENERAL source executor — it runs ANY composed
/// source as a SEPARATE per-stage kernel (the `body_source_gv` mechanism, generalized to
/// `SourceSpec`s), as opposed to FUSING the source into the godunov stage.
///
/// it splices the SAME `splice_fused_sources_to_contribs` the fused godunov uses, so a plain
/// `godunov_stage_gv` (flux + geometric source, no user sources) followed by this pass is the
/// proven-equivalent DECOMPOSITION of `godunov_stage_gv_with_fused_sources`. the driver passes
/// `dt = ac*dt` (the SSP Shu-Osher stage weight — identical to how `body_source` is invoked), so
/// `S` lands with the same `ac*dt` weight the fused stage applies inside its `ac*fe` combine.
///
// =============================================================================
// THE UNIFIED DAG-APPLICATION OPERATOR (docs/design/33 section 7).
//
// `apply_dag_core_gv` is the ONE kernel builder behind BOTH the interior source pass and
// (docs/design/33) the driven-boundary pass. it factors out the decisions a source/boundary
// makes: WHERE the DAG reads state (`StateEnv`), and HOW its result lands in
// the target field (`WriteMode`). the iteration domain + target-field binding are the dispatch's job
// (the same `dispatch_runtime_ir` + `resolve_path` serve cons.* and prim.*), so this builder is the
// whole difference between a source and a boundary prescription. doc 32's user `combine` projects
// onto `WriteMode`: add/relax -> Accumulate (differ only in the constructed expression), overwrite ->
// Assign.
// =============================================================================

/// the state vocabulary the DAG reads `rho`/`vel_k` from. `Stage` binds them from the SSP stage
/// snapshot `u_stage` (an interior source evaluates at its stage input — the S2 invariant); `Coord`
/// binds NOTHING from state (a pure coordinate prescription — a driven boundary, whose DAG OUTPUTS
/// the state). `x_k` (centroid) + scalar params bind regardless of this.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StateEnv {
    Stage,
    Coord,
}


/// how the DAG result lands in the target field. `Accumulate` is the RHS form `target = read(target)
/// + dt * Σ contrib` (in place; the `dt` scalar is the SSP stage weight) — sources. `Assign` is the
/// prescription `target = expr` (write-only, no base, no weight) — driven boundaries. doc 32's
/// `combine`: add + relax both map to `Accumulate`, overwrite to `Assign`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WriteMode {
    Accumulate,
    Assign,
}


/// the unified core: trace a kernel that evaluates each `(slot, BuiltSource)` DAG per cell and writes
/// it to the slot's field under `mode`. `slot` names the STRUCTURAL conserved slot (`"den"` mass /
/// `"mom"` momentum-vector / `"nrg"` energy); `mode` + the slot pick the runtime path (Accumulate ->
/// `cons.{den,mom_k,nrg}`; Assign -> `prim.{rho,vel_k,pre}`). shared: trace, geometry, the
/// `splice_fused_sources_to_contribs` primitive (leaf binding + per-DAG lowering).
fn apply_dag_core_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    state: StateEnv,
    sources: &[(&str, &symbi_hydro::source_spec::BuiltSource)],
    mode: WriteMode,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let geo = (!is_cartesian_uniform(coords, spacing))
        .then(|| cell_geometry_gv(coords, spacing, axes, ndim as usize));

    // bind the state vocabulary the DAG reads from. `Stage` reads the stage-input snapshot `u_stage`
    // (NOT post-godunov `cons`): the fused stage evaluates at its stage input, so this standalone
    // pass must too, for `plain + this == fused` bit-for-bit. `Coord` reads no state.
    let state_vocab: Option<(Gv, Vec<Gv>)> = match state {
        StateEnv::Stage => {
            let rho = Gv::field("rho", FieldRef::ustage_den());
            let mom = (0..ncomp)
                .map(|k| Gv::field(&format!("mom_{k}"), FieldRef::ustage_mom(k as u8)))
                .collect();
            Some((rho, mom))
        }
        StateEnv::Coord => None,
    };
    let state_ref = state_vocab.as_ref().map(|(r, m)| (*r, m.as_slice()));

    let contribs = splice_fused_sources_to_contribs(
        coords, spacing, axes, ndim, ncomp, has_energy, &geo, state_ref, sources,
    );

    let writes = match mode {
        WriteMode::Accumulate => {
            // RHS in place: `cons_slot = cons_slot + Σ dt*contrib`, accumulated exactly as the fused
            // stage's `with_sources` — so fused and (plain godunov + this pass) agree bit-for-bit.
            let dt = Gv::scalar("dt"); // the driver fills this with ac*dt (the SSP stage weight)
            let cons_den = Gv::field("cons_den", FieldRef::cons_den());
            let mut rho_new = cons_den;
            for c in &contribs.den {
                rho_new = rho_new + dt * Gv::of(*c);
            }
            let mut writes = vec![("rho".to_string(), FieldRef::cons_den().into(), rho_new.node())];
            for k in 0..ncomp {
                let cons_mom = Gv::field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8));
                let mut mom_new = cons_mom;
                for c in &contribs.mom[k] {
                    mom_new = mom_new + dt * Gv::of(*c);
                }
                writes.push((format!("mom_{k}"), FieldRef::cons_mom(k as u8).into(), mom_new.node()));
            }
            if has_energy {
                let cons_nrg = Gv::field("cons_nrg", FieldRef::cons_nrg());
                let mut nrg_new = cons_nrg;
                for c in &contribs.nrg {
                    nrg_new = nrg_new + dt * Gv::of(*c);
                }
                writes.push(("nrg".to_string(), FieldRef::cons_nrg().into(), nrg_new.node()));
            }
            // the godunov-source (accumulate) path never targets B — the safe conservation-law
            // lifts touch only den/mom/nrg, and `raw` is gated to those slots. a bcell contrib
            // here means a mis-routed source; fail loud rather than silently drop it.
            debug_assert!(
                contribs.mag.iter().all(|m| m.is_empty()),
                "accumulate (godunov source) path does not support a `bcell` target",
            );
            writes
        }
        WriteMode::Assign => {
            // prescription: `prim_slot = expr` (write-only, no base, no weight). a prescription is a
            // COMPLETE state — exactly ONE DAG per slot (not summed overlays).
            assert_eq!(contribs.den.len(), 1, "Assign: prim.rho needs exactly one source DAG");
            let mut writes = vec![("rho".to_string(), FieldRef::PrimRho.into(), contribs.den[0])];
            for k in 0..ncomp {
                assert_eq!(contribs.mom[k].len(), 1,
                    "Assign: prim.vel_{k} needs exactly one source DAG");
                writes.push((format!("vel_{k}"), FieldRef::PrimVel(k as u8).into(), contribs.mom[k][0]));
            }
            if has_energy {
                assert_eq!(contribs.nrg.len(), 1, "Assign: prim.pre needs exactly one source DAG");
                writes.push(("pre".to_string(), FieldRef::PrimPre.into(), contribs.nrg[0]));
            }
            // MHD driven boundary: prescribe the cell-B vector (prim.mag). out-of-plane B_phi
            // (cell-centered, flux-evolved) is the SAFE toroidal case; in-plane components are
            // the user's responsibility to keep div-compatible (=0 for a purely toroidal field).
            // absent for a hydro prescription (no bcell slot -> empty mag buckets).
            if contribs.mag.iter().any(|m| !m.is_empty()) {
                for k in 0..ncomp {
                    assert_eq!(contribs.mag[k].len(), 1,
                        "Assign: prim.mag_{k} needs exactly one source DAG");
                    writes.push((format!("mag_{k}"), FieldRef::PrimMag(k as u8).into(), contribs.mag[k][0]));
                }
            }
            writes
        }
    };
    (end_trace(), writes)
}


/// AOT entry: the in-place source-apply kernel from declarative `SourceSpec`s (each `build_source`d
/// at dimension `ndim`). build.rs bakes this per (regime, ndim). the `(Stage, Accumulate)` instance
/// of [`apply_dag_core_gv`].
pub fn source_apply_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    user_sources: &[&symbi_hydro::source_spec::SourceSpec],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    let builts: Vec<(&str, symbi_hydro::source_spec::BuiltSource)> = user_sources.iter()
        .map(|s| (s.target_field, (s.build_source)(ndim as usize)))
        .collect();
    let src_refs: Vec<(&str, &symbi_hydro::source_spec::BuiltSource)> =
        builts.iter().map(|(t, b)| (*t, b)).collect();
    apply_dag_core_gv(coords, spacing, axes, ndim, ncomp, has_energy, StateEnv::Stage, &src_refs, WriteMode::Accumulate)
}


/// RUNTIME entry (Path B): the SAME in-place source-apply kernel, but from already-lowered
/// `(target_field, BuiltSource)` values — e.g., `expr_bridge::build_user_source`'s output from a
/// SourceConfig loaded at sim startup. the `(Stage, Accumulate)` instance of [`apply_dag_core_gv`].
pub fn source_apply_from_built_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    sources: &[(&str, &symbi_hydro::source_spec::BuiltSource)],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    apply_dag_core_gv(coords, spacing, axes, ndim, ncomp, has_energy, StateEnv::Stage, sources, WriteMode::Accumulate)
}


/// DRIVEN-BOUNDARY entry (docs/design/33): prescribe the primitive state from coordinate DAGs — the
/// `(Coord, Assign)` instance of [`apply_dag_core_gv`]. `sources` are `(slot, BuiltSource)` with slot
/// `"den"`/`"mom"`/`"nrg"` mapping to `prim.rho`/`prim.vel_k`/`prim.pre`; each DAG reads only
/// `x_k`/`t`/`p_i` and OUTPUTS the prescribed value. dispatched over a face's ghost band (task 2).
pub fn boundary_fill_from_built_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    sources: &[(&str, &symbi_hydro::source_spec::BuiltSource)],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    apply_dag_core_gv(coords, spacing, axes, ndim, ncomp, has_energy, StateEnv::Coord, sources, WriteMode::Assign)
}


/// the cell-B induction-flux divergence for component `c` (mirror of `rmhd::bcell_flux_div`):
/// cartesian `sum_d (bf_d_c[+e_d] - bf_d_c)/dx_d`; curvilinear the area-weighted `inv_V sum_d
/// (A_hi_d bf_d_c[+e_d] - A_lo_d bf_d_c)` from `geo` — the SAME divergence the gas godunov uses.
fn bcell_flux_div_gv(c: usize, ndim: usize, geo: &Option<CellGeometryGv>, dx: &[Gv]) -> Gv {
    let off = |d: usize| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        o[d] = 1;
        o
    };
    let zero = vec![0i32; ndim];
    let mut div: Option<Gv> = None;
    for d in 0..ndim {
        let key = format!("bf_{d}_{c}");
        let here = gv_field_at(&key, &key, ndim, &zero);
        let plus = gv_field_at(&key, &key, ndim, &off(d));
        let term = match geo {
            None => (plus - here) / dx[d],
            Some(g) => g.area_hi[d] * plus - g.area_lo[d] * here,
        };
        div = Some(match div {
            None => term,
            Some(a) => a + term,
        });
    }
    let div = div.unwrap();
    match geo {
        Some(g) => g.inv_volume * div,
        None => div,
    }
}


/// the PLAIN (metric-free) cell-B induction-flux divergence `sum_d (bf_d_c[+e_d] - bf_d_c)/width_d`
/// with the per-axis COORDINATE width read in-kernel (gv_axis_face_at). used for the OUT-OF-PLANE
/// B component whose curl carries no Lame factor — see `metric_free_oop_component`.
fn bcell_flux_div_plain_gv(c: usize, ndim: usize, spacing: &[Spacing]) -> Gv {
    let off = |d: usize| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        o[d] = 1;
        o
    };
    let zero = vec![0i32; ndim];
    let mut div: Option<Gv> = None;
    for d in 0..ndim {
        let key = format!("bf_{d}_{c}");
        let here = gv_field_at(&key, &key, ndim, &zero);
        let plus = gv_field_at(&key, &key, ndim, &off(d));
        let width = gv_axis_face_at(d, spacing[d], 1) - gv_axis_face_at(d, spacing[d], 0);
        let term = (plus - here) / width;
        div = Some(match div {
            None => term,
            Some(a) => a + term,
        });
    }
    div.unwrap()
}


/// the OUT-OF-PLANE B component (the one not in `axes`) whose induction curl is METRIC-FREE on the
/// gridded plane, so it must use the PLAIN divergence instead of the gas area-weighted one. the
/// out-of-plane component's curl carries the prefactor 1/(h_g1 h_g2) over the gridded axes — for
/// cylindrical the only non-unit Lame factor is h_phi = r, so this is the AZIMUTHAL component (phi
/// = 1) gridded as r-z (axes [0,2]): (curl E)_phi = d_z E_r - d_r E_z has NO 1/r, yet the gas FV
/// divergence carries h_phi=r in the cell volume (a spurious F_r/r source if reused). the r-phi
/// disk (out-of-plane z, h_z=1) and cartesian are unaffected — their out-of-plane curl IS the
/// area-weighted divergence. returns Some(phi=1) only for the cyl r-z plane.
fn metric_free_oop_component(coords: Coords, axes: &[usize], ncomp: usize) -> Option<usize> {
    if coords != Coords::Cylindrical {
        return None;
    }
    (0..ncomp).find(|c| !axes.contains(c) && *c == 1)
}


/// the RMHD cell-B FLUX PREDICTOR (Euler): `bcell[c] -= dt*div(bflux_c)`, in-place. flux-evolves
/// the cell B as a conserved component (so the energy correction reads the flux-implied b_old).
/// mirror of `rmhd::rmhd_bcell_godunov_euler`; reuses `cell_geometry_gv` on curvilinear grids.
pub fn rmhd_bcell_godunov_euler_gv(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let bc: Vec<Gv> = (0..ncomp).map(|c| Gv::field(&format!("bc_{c}"), FieldRef::BCell(c as u8))).collect();
    // pin the ndim*ncomp induction-flux inputs in d-outer/c-inner order (the positional dispatch
    // order [bf_0_0, bf_0_1, .., bf_1_0, ..]) before bcell_flux_div_gv reads them (it loops d).
    for d in 0..ndim {
        for c in 0..ncomp {
            gv_register_field(&format!("bf_{d}_{c}"), &format!("bf_{d}_{c}"));
        }
    }
    let dt = Gv::scalar("dt");
    let (geo, dx) = bcell_godunov_geom(coords, spacetime, spacing, ndim, axes);
    let oop = metric_free_oop_component(coords, axes, ncomp);
    let writes = (0..ncomp)
        .map(|c| {
            // the metric-free out-of-plane component (cyl r-z B_phi) uses the PLAIN divergence;
            // every other component the gas area-weighted (cartesian dx or the geo metric).
            let div = if Some(c) == oop {
                bcell_flux_div_plain_gv(c, ndim, spacing)
            } else {
                bcell_flux_div_gv(c, ndim, &geo, &dx)
            };
            let bnew = bc[c] - dt * div;
            (format!("bc_{c}_new"), format!("bc_{c}").into(), bnew.node())
        })
        .collect();
    (end_trace(), writes)
}


/// the RMHD cell-B FLUX PREDICTOR (RK2 stage 2): `bcell[c] = 0.5*(bcell_n[c] + (bcell[c] -
/// dt*div(bflux_c)))`, in-place. mirror of `rmhd::rmhd_bcell_godunov_rk2`.
pub fn rmhd_bcell_godunov_rk2_gv(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let bcn: Vec<Gv> = (0..ncomp).map(|c| Gv::field(&format!("bcn_{c}"), FieldRef::BCellN(c as u8))).collect();
    let bc: Vec<Gv> = (0..ncomp).map(|c| Gv::field(&format!("bc_{c}"), FieldRef::BCell(c as u8))).collect();
    for d in 0..ndim {
        for c in 0..ncomp {
            gv_register_field(&format!("bf_{d}_{c}"), &format!("bf_{d}_{c}"));
        }
    }
    let dt = Gv::scalar("dt");
    let half = Gv::from_f64(0.5);
    let (geo, dx) = bcell_godunov_geom(coords, spacetime, spacing, ndim, axes);
    let oop = metric_free_oop_component(coords, axes, ncomp);
    let writes = (0..ncomp)
        .map(|c| {
            let div = if Some(c) == oop {
                bcell_flux_div_plain_gv(c, ndim, spacing)
            } else {
                bcell_flux_div_gv(c, ndim, &geo, &dx)
            };
            let bc_star = bc[c] - dt * div;
            let bnew = half * (bcn[c] + bc_star);
            (format!("bc_{c}_new"), format!("bc_{c}").into(), bnew.node())
        })
        .collect();
    (end_trace(), writes)
}


/// the cell-B godunov geometry: curvilinear -> the gv cell geometry (area-weighted div);
/// cartesian -> the uniform `dx_d` scalars. registered in the order the bcell godunov needs
/// (the bf_d_c fields are read later by `bcell_flux_div_gv`). 3D, identity axes.
fn bcell_godunov_geom(coords: Coords, spacetime: Spacetime, spacing: &[Spacing], ndim: usize, axes: &[usize]) -> (Option<CellGeometryGv>, Vec<Gv>) {
    if coords == Coords::Cartesian {
        (None, (0..ndim).map(|d| Gv::scalar(&format!("dx_{d}"))).collect())
    } else {
        // axes maps grid axis -> coordinate (identity for sph/3d-cyl; [0,2] for cyl r-z) so the
        // area-weighted divergence uses the right radial axis for the cylindrical metric. a
        // curved spacetime takes the COVARIANT (alpha sqrt(gamma)) measure — the mag rows are
        // densitized conserved laws of the same form as the gas (d_t(sqrt(g) B) + coordinate
        // divergence), exactly like the gas godunov's geometry selection.
        let g = match spacetime {
            Spacetime::Minkowski => cell_geometry_gv(coords, spacing, axes, ndim),
            Spacetime::Kerr => cell_geometry_covariant_gv(
                coords, spacing, axes, ndim, Some(Gv::scalar("kerr_spin")),
            ),
            _ => cell_geometry_covariant_gv(coords, spacing, axes, ndim, None),
        };
        (Some(g), Vec::new())
    }
}

