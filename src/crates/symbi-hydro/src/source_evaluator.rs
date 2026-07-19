// =============================================================================
// source_evaluator.rs
//
// `SourceEvaluator` — the runtime-usable artifact. takes a `SimulationLaws`
// + dimension, pre-scalarizes each field's source kernel into one
// `LoweredFn` per output component, and provides per-cell evaluation via
// the Cpu interpreter (the same path the unit tests in `regime_spec` /
// `source_spec` use).
//
// **what this layer is and isn't:**
//   - IS: the runtime evaluation hook an evolve loop calls per cell to get
//     the source-term contribution at that cell. one cache-hit per cell;
//     scalarize runs ONCE per field at construction.
//   - IS: the symmetry between the data layer and the runtime layer
//     made explicit — every component of every law / overlay produces
//     an evaluatable `LoweredFn`, and there is exactly one path from spec
//     data to runtime numbers.
//   - ISN'T: the GPU path. CUDA NVRTC compilation needs hardware (covered
//     structurally by the emit tests). the CPU interpreter path
//     here is the analogous workflow at `S = f64` — same `BuiltSource`
//     graph, different lowering target.
//   - ISN'T: the substrate codegen-driver. AOT-baking spec compositions
//     into the binary (extending `symbi-aot/build.rs`) is a separate
//     layer that needs compile-time spec definitions.
//
// usage:
//   use symbi_hydro::{NEWTONIAN_SPEC, spherical_geometric_sources,
//                     point_mass_gravity_sources, SimulationLaws,
//                     SourceEvaluator};
//
//   let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
//       .with_geometric(spherical_geometric_sources(3))
//       .with_gravity(point_mass_gravity_sources(3, true));
//   let evaluator = SourceEvaluator::new(&sim, 3)?;
//
//   // in the evolve loop, per cell:
//   let s_mom = evaluator.eval("mom", &cell_values)?;
//   // s_mom is Vec<f64> of length 3, the total momentum-source contribution.
// =============================================================================

use std::collections::HashMap;

use symbi_ir::backends::interp::{Backend, Cpu};
use symbi_ir::passes::scalarize::{scalarize, LoweredFn};

use crate::simulation_laws::SimulationLaws;

/// the per-field cache: one `LoweredFn` per output component (1 for
/// scalar-field laws like `den`/`nrg`; D for `mom`; 3 for `mag`). param
/// ordering is stable across components — the runtime fills the same
/// `inputs` vec for every component evaluation.
struct FieldKernel {
    /// declared param names in the order `LoweredFn` expects them.
    /// callers pass values by name; the evaluator routes by position.
    params: Vec<String>,
    /// pre-scalarized per-component kernels. `eval` calls each with the
    /// same `inputs` vec and collects the results. KEPT as the interpreter
    /// ORACLE / fallback even when the JIT path is available.
    components: Vec<LoweredFn>,
    /// the native-compiled twin (the CPU's NVRTC): one `CompiledFn` per
    /// component, in the SAME order as `components`. `Some` only when EVERY
    /// component compiled (an out-of-subset node makes the whole field fall
    /// back to the interpreter). a `CompiledFn` is allocation-free + `Send +
    /// Sync`, so the per-cell source pass runs native + block-parallel.
    jit: Option<Vec<symbi_jit::CompiledFn>>,
}

impl FieldKernel {
    /// build a field kernel from its per-component lowered fns, JIT-compiling each
    /// (the interpreter `components` stay as oracle/fallback). compilation happens
    /// ONCE here at construction; the string→register resolution it does is what
    /// deletes the interpreter's per-cell `HashMap`.
    fn build(params: Vec<String>, components: Vec<LoweredFn>) -> Self {
        let jit: Option<Vec<symbi_jit::CompiledFn>> = components
            .iter()
            .map(|lf| symbi_jit::compile(lf).ok())
            .collect();
        Self { params, components, jit }
    }
}

/// the runtime evaluator. constructed once per simulation from a
/// `SimulationLaws` + spatial dimension; subsequent `eval` calls hit
/// the cache and run the interpreter on the supplied cell state.
pub struct SourceEvaluator {
    field_kernels: HashMap<String, FieldKernel>,
}

impl std::fmt::Debug for SourceEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut entries: Vec<(&String, usize)> = self.field_kernels.iter()
            .map(|(name, k)| (name, k.components.len()))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        f.debug_struct("SourceEvaluator")
            .field("fields", &entries)
            .finish()
    }
}

impl SourceEvaluator {
    /// build the evaluator from spec data. validates the composition
    /// (clause-2 cross-checks via `SimulationLaws::validate`) before
    /// scalarizing. returns `Err` if validation fails — the runtime
    /// catches malformed compositions BEFORE the time-loop starts.
    pub fn new(
        laws: &SimulationLaws,
        d: usize,
    ) -> Result<Self, crate::simulation_laws::CompositionError> {
        laws.validate()?;

        let mut field_kernels: HashMap<String, FieldKernel> = HashMap::new();
        // walk every field any overlay targets — only those need source
        // kernels. fields with pure-divergence RHS (no overlays) are
        // skipped, matching the runtime's expectation that an empty
        // entry means "no source-term computation for this field."
        for field_name in laws.fields_with_overlays() {
            if let Some(built) = laws.build_total_source(field_name, d) {
                let components: Vec<LoweredFn> = built
                    .outputs
                    .iter()
                    .enumerate()
                    .map(|(k, &out)| {
                        let name = format!("{}_source_{k}", field_name);
                        scalarize(&built.graph, out, &name)
                    })
                    .collect();
                field_kernels.insert(
                    field_name.to_string(),
                    FieldKernel::build(built.params, components),
                );
            }
        }

        Ok(Self { field_kernels })
    }

    /// build the evaluator directly from already-lowered `(target_field, BuiltSource)` pairs — the
    /// RUNTIME path. unlike [`Self::new`] (which composes a `SimulationLaws` of compile-time
    /// fn-builders, AOT), this takes `BuiltSource` VALUES — e.g., `expr_bridge::build_user_source`'s
    /// output from a `SourceConfig` loaded at sim startup (python -> json, no recompile). each
    /// field's `BuiltSource` is scalarized into per-component `LoweredFn`s, exactly as `new` does.
    /// panics on a duplicate target field (the caller should pre-merge same-field sources).
    pub fn from_built(sources: &[(String, crate::source_spec::BuiltSource)]) -> Self {
        let mut field_kernels: HashMap<String, FieldKernel> = HashMap::new();
        for (field, built) in sources {
            let components: Vec<LoweredFn> = built
                .outputs
                .iter()
                .enumerate()
                .map(|(k, &out)| scalarize(&built.graph, out, &format!("{field}_source_{k}")))
                .collect();
            if field_kernels
                .insert(field.clone(), FieldKernel::build(built.params.clone(), components))
                .is_some()
            {
                panic!("SourceEvaluator::from_built: duplicate target field '{field}'");
            }
        }
        Self { field_kernels }
    }

    /// evaluate the total source contribution for `field` at one cell.
    /// `values` is a list of `(param_name, value)` pairs covering every
    /// declared param — `params_for(field)` enumerates them.
    ///
    /// returns `None` when the field has no overlay sources (the runtime
    /// should skip the additive-RHS source step for that field).
    pub fn eval(&self, field: &str, values: &[(&str, f64)]) -> Option<Vec<f64>> {
        let kernel = self.field_kernels.get(field)?;
        let inputs: Vec<f64> = kernel
            .params
            .iter()
            .map(|pname| {
                values
                    .iter()
                    .find(|(n, _)| *n == pname.as_str())
                    .map(|(_, v)| *v)
                    .unwrap_or_else(|| panic!(
                        "SourceEvaluator::eval: missing param '{pname}' for field '{field}'"
                    ))
            })
            .collect();

        let out: Vec<f64> = kernel
            .components
            .iter()
            .map(|lowered| Cpu.eval_elemental(lowered, &inputs)[0])
            .collect();

        Some(out)
    }

    /// the ordered list of param names this evaluator expects for `field`.
    /// callers use it to build the `values` slice. returns `None` for
    /// fields with no overlays.
    pub fn params_for(&self, field: &str) -> Option<&[String]> {
        self.field_kernels.get(field).map(|k| k.params.as_slice())
    }

    /// the native-compiled per-component kernels for `field`, in `params_for` input order.
    /// `Some` only when the WHOLE field JIT-compiled — the caller takes the allocation-free
    /// native path; `None` means use the interpreter (`eval`) for this field.
    pub fn jit_components(&self, field: &str) -> Option<&[symbi_jit::CompiledFn]> {
        self.field_kernels.get(field).and_then(|k| k.jit.as_deref())
    }

    /// the set of fields this evaluator handles — the runtime walks this
    /// list to know which fields need source-term computation per step.
    pub fn fields(&self) -> impl Iterator<Item = &str> {
        self.field_kernels.keys().map(|k| k.as_str())
    }

    /// the number of source-term output components for `field`. matches
    /// `RegimeSpec.fields[i].kind.components_at(d)` for any field whose
    /// overlays produce a complete source. returns `None` for fields
    /// with no overlays.
    pub fn component_count(&self, field: &str) -> Option<usize> {
        self.field_kernels.get(field).map(|k| k.components.len())
    }
}

// =============================================================================
// tests — the runtime evaluator computes the correct sum at known cells.
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regime_spec::{law_params, NEWTONIAN_SPEC, ISO_NEWTONIAN_SPEC};
    use crate::source_spec::{
        cylindrical_geometric_sources, gravity_params, ib_params,
        point_mass_gravity_sources, rigid_body_penalty_sources, source_params,
    };

    #[test]
    fn evaluator_for_empty_overlays_has_no_fields() {
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC);
        let evaluator = SourceEvaluator::new(&sim, 3).expect("empty validates");
        assert_eq!(evaluator.fields().count(), 0);
        assert!(evaluator.eval("mom", &[]).is_none());
        assert!(evaluator.params_for("mom").is_none());
    }

    #[test]
    fn evaluator_rejects_malformed_composition_at_construction() {
        // iso + adiabatic gravity = CompositionError. the evaluator
        // surfaces the error at construction (validate is called before
        // scalarize), so the time-loop never sees the bug.
        let sim = SimulationLaws::new(&ISO_NEWTONIAN_SPEC)
            .with_gravity(point_mass_gravity_sources(3, true));
        let err = SourceEvaluator::new(&sim, 3).unwrap_err();
        assert!(matches!(
            err,
            crate::simulation_laws::CompositionError::EnergyOverlayOnIsothermal { .. }
        ));
    }

    #[test]
    fn evaluator_for_gravity_mom_matches_analytical_3d() {
        // sanity: the evaluator's per-cell value equals the analytical
        // gravity formula. proves the runtime hook produces the right
        // numbers for the simplest non-trivial case.
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_gravity(point_mass_gravity_sources(3, false));
        let evaluator = SourceEvaluator::new(&sim, 3).expect("gravity validates");

        // single body at origin; cell at (1, 2, 3); gm = 1, rho = 1.5.
        // S_mom_k = -ρ * GM * x_k / |x|^3.
        let x = [1.0_f64, 2.0, 3.0];
        let rho = 1.5;
        let gm = 1.0;
        let r3 = (x.iter().map(|v| v * v).sum::<f64>()).sqrt().powi(3);

        let v0 = law_params::vel(0); let v1 = law_params::vel(1); let v2 = law_params::vel(2);
        let x0 = source_params::x(0); let x1 = source_params::x(1); let x2 = source_params::x(2);
        let xm0 = gravity_params::xm(0); let xm1 = gravity_params::xm(1); let xm2 = gravity_params::xm(2);
        let s = evaluator.eval("mom", &[
            (law_params::RHO, rho),
            (v0.as_str(), 0.0), (v1.as_str(), 0.0), (v2.as_str(), 0.0),
            (x0.as_str(), x[0]), (x1.as_str(), x[1]), (x2.as_str(), x[2]),
            (xm0.as_str(), 0.0), (xm1.as_str(), 0.0), (xm2.as_str(), 0.0),
            (gravity_params::GM, gm),
            (gravity_params::EPS, 0.0),
        ]).expect("mom has overlay");

        assert_eq!(s.len(), 3);
        for k in 0..3 {
            let expected = -rho * gm * x[k] / r3;
            assert!(
                (s[k] - expected).abs() < 1e-12,
                "evaluator gravity mom k={k}: {} != {expected}", s[k],
            );
        }
    }

    #[test]
    fn evaluator_composes_geometric_plus_gravity_additively() {
        // the load-bearing run: a Kepler-disk overlay stack (cylindrical
        // geometric source + point-mass gravity) evaluated at one cell
        // returns the SUM of the two contributions. proves that the
        // runtime hook delivers the composed result of both
        // overlays.
        use crate::source_spec::cylindrical_geometric_sources;

        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_geometric(cylindrical_geometric_sources(3))
            .with_gravity(point_mass_gravity_sources(3, false));
        let evaluator = SourceEvaluator::new(&sim, 3).expect("composes");

        let r = 2.0_f64;
        let phi = 0.3;
        let z = 0.5;
        let rho = 1.1;
        let vr = 0.2;
        let vp = 0.4;
        let vz = 0.1;
        let p = 0.9;
        let xm = [0.0_f64, 0.0, 0.0];
        let gm = 1.0;

        let v0 = law_params::vel(0); let v1 = law_params::vel(1); let v2 = law_params::vel(2);
        let x0 = source_params::x(0); let x1 = source_params::x(1); let x2 = source_params::x(2);
        let xm0 = gravity_params::xm(0); let xm1 = gravity_params::xm(1); let xm2 = gravity_params::xm(2);
        let s = evaluator.eval("mom", &[
            (law_params::RHO, rho),
            (v0.as_str(), vr), (v1.as_str(), vp), (v2.as_str(), vz),
            (law_params::PRE, p),
            (x0.as_str(), r), (x1.as_str(), phi), (x2.as_str(), z),
            (xm0.as_str(), xm[0]), (xm1.as_str(), xm[1]), (xm2.as_str(), xm[2]),
            (gravity_params::GM, gm),
            (gravity_params::EPS, 0.0),
        ]).expect("mom has overlays");

        // analytical sum: cylindrical S_r = (ρ*vp² + p)/r,  S_p = -ρ*vr*vp/r,  S_z = 0
        // gravity:        -ρ*GM*x_k/|x|³ (note: x here is the (r, phi, z) tuple
        //                 but gravity uses cartesian-style distance — the
        //                 spec treats `x` as the field-point components in
        //                 the regime's coord system. the evaluator agrees
        //                 with the data; this test asserts CONSISTENCY
        //                 between evaluator and data, making no claim about
        //                 physical meaningfulness in a curvilinear mix.)
        let dx_sq: f64 = r * r + phi * phi + z * z;
        let dx_cubed = dx_sq.sqrt().powi(3);
        let geom_sr = (rho * vp * vp + p) / r;
        let geom_sp = -rho * vr * vp / r;
        let geom_sz = 0.0;
        let grav_sr = -rho * gm * r / dx_cubed;
        let grav_sp = -rho * gm * phi / dx_cubed;
        let grav_sz = -rho * gm * z / dx_cubed;
        let expected = [geom_sr + grav_sr, geom_sp + grav_sp, geom_sz + grav_sz];

        for k in 0..3 {
            assert!(
                (s[k] - expected[k]).abs() < 1e-12,
                "composed mom k={k}: {} != {} = {} (geom) + {} (grav)",
                s[k], expected[k],
                [geom_sr, geom_sp, geom_sz][k],
                [grav_sr, grav_sp, grav_sz][k],
            );
        }
    }

    #[test]
    fn evaluator_fields_iterator_matches_overlay_targets() {
        // structural canary: the evaluator exposes EXACTLY the fields with
        // overlays — no extras (no fake fields), no missing (no skipped
        // overlays). the runtime walks this list to know which kernels
        // to call per step.
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_geometric(cylindrical_geometric_sources(3))     // mom
            .with_gravity(point_mass_gravity_sources(3, true))    // mom + nrg
            .with_ib(rigid_body_penalty_sources(3));              // mom
        let evaluator = SourceEvaluator::new(&sim, 3).expect("composes");

        let fields: std::collections::HashSet<&str> = evaluator.fields().collect();
        assert!(fields.contains("mom"));
        assert!(fields.contains("nrg"));
        assert!(!fields.contains("den"), "no overlay targets den");
        assert_eq!(fields.len(), 2);
    }

    #[test]
    fn evaluator_component_count_matches_field_kind() {
        // scalar fields ('nrg', 'den') -> 1 component; vector fields
        // ('mom') -> D components at D=3.
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_gravity(point_mass_gravity_sources(3, true));
        let evaluator = SourceEvaluator::new(&sim, 3).expect("composes");

        assert_eq!(evaluator.component_count("mom"), Some(3));
        assert_eq!(evaluator.component_count("nrg"), Some(1));
        assert_eq!(evaluator.component_count("den"), None); // no overlay
    }

    #[test]
    fn evaluator_ib_region_localization_holds_in_runtime() {
        // **the runtime-side clause-3 canary**: the IB penalty's mask
        // discipline (S::select on a carrier-generic mask) survives
        // through scalarize + interp. outside the body, the runtime
        // evaluator returns EXACTLY 0.0 per component.
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_ib(rigid_body_penalty_sources(3));
        let evaluator = SourceEvaluator::new(&sim, 3).expect("composes");

        // body at origin, radius 1.0; cell at (3, 0, 0) - outside.
        let v0 = law_params::vel(0); let v1 = law_params::vel(1); let v2 = law_params::vel(2);
        let x0 = source_params::x(0); let x1 = source_params::x(1); let x2 = source_params::x(2);
        let bxm0 = ib_params::body_xm(0); let bxm1 = ib_params::body_xm(1); let bxm2 = ib_params::body_xm(2);
        let vb0 = ib_params::vbody(0); let vb1 = ib_params::vbody(1); let vb2 = ib_params::vbody(2);
        let s = evaluator.eval("mom", &[
            (law_params::RHO, 1.0),
            (v0.as_str(), 0.5), (v1.as_str(), 0.0), (v2.as_str(), 0.0),
            (x0.as_str(), 3.0), (x1.as_str(), 0.0), (x2.as_str(), 0.0),
            (bxm0.as_str(), 0.0), (bxm1.as_str(), 0.0), (bxm2.as_str(), 0.0),
            (ib_params::BODY_RADIUS, 1.0),
            (vb0.as_str(), 0.0), (vb1.as_str(), 0.0), (vb2.as_str(), 0.0),
            (ib_params::PENALTY_STRENGTH, 100.0),
        ]).expect("mom has overlay");

        for k in 0..3 {
            assert_eq!(
                s[k], 0.0,
                "outside body: evaluator mom k={k} must be EXACTLY 0.0; got {}", s[k],
            );
        }
    }
}
