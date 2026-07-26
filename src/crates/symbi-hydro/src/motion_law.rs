// =============================================================================
// motion_law.rs
//
// expression-driven mesh-motion scale factor. a(t) and a_dot(t) arrive as a TWO-output traced
// expression (python `CompiledExpr.serialize_motion` -> the SourceConfig wire), lowered to the
// symbi-ir Graph and scalarized ONCE; evaluated EXACTLY in the time loop per (sub)stage — no
// linearization, no python in the loop. a_dot is autodiff'd from a in python (`a.diff(t)`); at
// construction a_dot is finite-difference-checked against a over the run's time window, so an
// inconsistent OR non-smooth derivative fails loudly before any stepping.
//
// usage:
//  let law = MotionLaw::from_json(&json, t0, t_end)?;
//  let a    = law.a_at(t);
//  let adot = law.adot_at(t);
// =============================================================================
use crate::expr_bridge::lower_dag_to_builtsource;
use symbi_expr::load::{SourceConfig, nodes_from_descs};
use symbi_ir::backends::interp::{Backend, Cpu};
use symbi_ir::passes::scalarize::{LoweredFn, scalarize};

/// a traced scale-factor law a(t) with its analytic derivative a_dot(t). both are scalar functions
/// of time only (the single param `t`).
pub struct MotionLaw {
    a: LoweredFn,
    adot: LoweredFn,
    /// number of declared params (each is the time variable `t`); 0 for a constant a.
    n_params: usize,
}

impl MotionLaw {
    /// build from the `serialize_motion` wire (`kind="motion"`, `outputs=[a, a_dot]`, `nodes`).
    /// lowers + scalarizes both outputs and finite-difference-checks `a_dot == da/dt` across
    /// `[t0, t_end]`. errors (never panics) on any malformed / inconsistent input.
    pub fn from_json(json: &str, t0: f64, t_end: f64) -> Result<Self, String> {
        let cfg = SourceConfig::from_json(json).map_err(|e| format!("motion json parse: {e}"))?;
        if cfg.kind != "motion" {
            return Err(format!("expected a motion law, got kind='{}'", cfg.kind));
        }
        if cfg.outputs.len() != 2 {
            return Err(format!(
                "motion needs exactly 2 outputs [a, a_dot], got {}",
                cfg.outputs.len()
            ));
        }
        let nodes = nodes_from_descs(&cfg.nodes).map_err(|e| format!("motion nodes: {e}"))?;
        let built = lower_dag_to_builtsource(&nodes, &cfg.outputs)
            .map_err(|e| format!("motion lower: {e:?}"))?;
        // a(t) is a function of TIME ONLY: every declared param must be `t` (no spatial/field leaves).
        for p in &built.params {
            if p != "t" {
                return Err(format!(
                    "scale factor a(t) may depend only on time `t`, found '{p}'"
                ));
            }
        }
        let a = scalarize(&built.graph, built.outputs[0], "motion_a");
        let adot = scalarize(&built.graph, built.outputs[1], "motion_adot");
        let law = MotionLaw {
            a,
            adot,
            n_params: built.params.len(),
        };
        law.fd_check(t0, t_end)?;
        Ok(law)
    }

    #[inline]
    fn eval(&self, f: &LoweredFn, t: f64) -> f64 {
        // every param is `t`; bind them all to the current time (n_params == 0 for a constant a).
        let inputs = vec![t; self.n_params];
        Cpu.eval_elemental(f, &inputs)[0]
    }

    #[inline]
    pub fn a_at(&self, t: f64) -> f64 {
        self.eval(&self.a, t)
    }

    #[inline]
    pub fn adot_at(&self, t: f64) -> f64 {
        self.eval(&self.adot, t)
    }

    /// the strict-correctness guard: a_dot MUST equal da/dt. central finite difference at five times
    /// spanning the run; a wrong autodiff rule OR a non-smooth a(t) (a kink, a t-dependent branch)
    /// makes AD and FD disagree past a relative tolerance and the run is refused.
    fn fd_check(&self, t0: f64, t_end: f64) -> Result<(), String> {
        let (lo, hi) = if t_end > t0 {
            (t0, t_end)
        } else {
            (t0, t0 + 1.0)
        };
        for k in 0..5 {
            let t = lo + (hi - lo) * (k as f64 + 0.5) / 5.0;
            let h = 1.0e-6 * t.abs().max(1.0);
            let fd = (self.a_at(t + h) - self.a_at(t - h)) / (2.0 * h);
            let ad = self.adot_at(t);
            let scale = ad.abs().max(fd.abs()).max(1.0e-8);
            if (ad - fd).abs() / scale > 1.0e-3 {
                return Err(format!(
                    "mesh motion: a_dot is inconsistent with da/dt at t={t:.4e} (a_dot={ad:.6e}, \
                     finite-difference={fd:.6e}). a_dot must be exactly da/dt — declare it as \
                     `a.diff(variable('t'))`. a non-smooth a(t) (a kink or a t-dependent branch) \
                     also trips this guard."
                ));
            }
        }
        Ok(())
    }
}
