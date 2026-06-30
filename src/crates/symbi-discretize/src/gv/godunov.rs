// =============================================================================
// godunov.rs
//
// the conserved-update godunov family: snapshot, ssp stage, fused sources, and the unified dag-application operator.
// =============================================================================

use super::*;


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
    let geo = (!is_cartesian_uniform(coords, spacing))
        .then(|| cell_geometry_gv(coords, spacing, axes, ndim as usize));
    let rho = Gv::field("rho", FieldRef::cons_den());
    let mom: Vec<Gv> = (0..ncomp)
        .map(|k| Gv::field(&format!("mom_{k}"), FieldRef::cons_mom(k as u8)))
        .collect();
    let src = geo
        .as_ref()
        .map(|g| gv_geometric_source(coords, axes, ndim as usize, ncomp, g, source, &mom, mag_from_bcell));

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
    let lapse = gv_lapse_weight(coords, spacetime);
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
        let mom_new = with_sources(
            combine(u_n_mom, fe(mom[k], div, src.as_ref().map(|s| s[k]))),
            &contribs.mom[k],
        );
        writes.push((format!("mom_{k}"), FieldRef::cons_mom(k as u8).into(), mom_new.node()));
    }
    if has_energy {
        let nrg = Gv::field("nrg", FieldRef::cons_nrg());
        let u_n_nrg = Gv::field("u_n_nrg", FieldRef::un_nrg());
        let nrg_new = with_sources(
            combine(u_n_nrg, fe(nrg, gv_divergence("nrg_flux", ndim, &geo), None)),
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
    let (geo, dx) = bcell_godunov_geom(coords, spacing, ndim, axes);
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
    let (geo, dx) = bcell_godunov_geom(coords, spacing, ndim, axes);
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
fn bcell_godunov_geom(coords: Coords, spacing: &[Spacing], ndim: usize, axes: &[usize]) -> (Option<CellGeometryGv>, Vec<Gv>) {
    if coords == Coords::Cartesian {
        (None, (0..ndim).map(|d| Gv::scalar(&format!("dx_{d}"))).collect())
    } else {
        // axes maps grid axis -> coordinate (identity for sph/3d-cyl; [0,2] for cyl r-z) so the
        // area-weighted divergence uses the right radial axis for the cylindrical metric.
        (Some(cell_geometry_gv(coords, spacing, axes, ndim)), Vec::new())
    }
}

