// =============================================================================
// run_args.rs
//
// shared CLI argument parser for examples. each example calls ArgPool::env(),
// extracts whatever flags it cares about with `take_*`, then `finish()` to
// surface any unrecognized args. CliConfig bundles the universal flags
// (--tfinal, --cfl, --recon, --ckpt-interval, --data-dir, --prefix) so
// examples don't repeat the same field set.
//
// usage:
//   let mut p = ArgPool::env();
//   let n: usize = p.take_or("--n", 256);
//   let cli = CliConfig::extract(&mut p, "orszag_tang");
//   p.finish();
//   ...
//   SimRunner::new(&mut sim, &kernels, &cli.solver, &cli.recon_name())
//       .data_dir(&cli.data_dir)
//       .prefix(&cli.prefix)
//       .checkpoint_interval(cli.checkpoint_interval())
//       .run(cli.t_final)?;
// =============================================================================

use std::str::FromStr;

/// owned pool of CLI tokens. `take_*` consumes matched flags; `finish` panics
/// on any unrecognized leftover so typos surface immediately.
pub struct ArgPool {
    args: Vec<String>,
}

impl ArgPool {
    /// build from `std::env::args()`, dropping the program name.
    pub fn env() -> Self {
        let mut args: Vec<String> = std::env::args().collect();
        if !args.is_empty() { args.remove(0); }
        ArgPool { args }
    }

    /// take a `--flag VALUE` pair, parsing VALUE as T. returns Some(value) on
    /// match, None if absent. panics if the flag is present but the value is
    /// missing or unparseable.
    pub fn take<T: FromStr>(&mut self, flag: &str) -> Option<T>
    where T::Err: std::fmt::Display
    {
        let pos = self.args.iter().position(|a| a == flag)?;
        if pos + 1 >= self.args.len() {
            panic!("flag {} requires a value", flag);
        }
        let raw = self.args.remove(pos + 1);
        self.args.remove(pos);
        match raw.parse::<T>() {
            Ok(v) => Some(v),
            Err(e) => panic!("flag {} expects a value: {}", flag, e),
        }
    }

    /// `take` with a default fallback.
    pub fn take_or<T: FromStr>(&mut self, flag: &str, default: T) -> T
    where T::Err: std::fmt::Display
    {
        self.take(flag).unwrap_or(default)
    }

    /// take a boolean switch (`--flag` with no value).
    pub fn take_flag(&mut self, flag: &str) -> bool {
        if let Some(pos) = self.args.iter().position(|a| a == flag) {
            self.args.remove(pos);
            true
        } else {
            false
        }
    }

    /// panic if any unrecognized args remain. call after all `take_*`.
    pub fn finish(self) {
        if !self.args.is_empty() {
            panic!("unrecognized args: {:?}", self.args);
        }
    }
}

/// CLI flags every example honours: --tfinal, --cfl, --recon, --solver,
/// --theta, --timestepping, --ckpt-interval, --data-dir, --prefix.
///
/// `--timestepping` is for physics validation: run euler (first-order in time)
/// alongside `--recon pcm` (first-order in space) to sanity-check the raw
/// physics primitives without any reconstruction or multi-stage logic.
pub struct CliConfig {
    pub t_final: f64,
    pub cfl: f64,
    pub recon: Reconstruction,
    pub solver: Solver,
    /// theta-MC limiter compression in [1,2] (--theta): 1 = minmod (most
    /// diffusive), 2 = monotonized-central (sharpest TVD). only consumed by
    /// PLM-reconstructing kernels that thread it (the RMHD substrate flux).
    pub theta: f64,
    pub timestepping: crate::state::Timestepping,
    pub data_dir: String,
    pub prefix: String,
    pub ckpt_interval: Option<f64>,
}

#[derive(Copy, Clone)]
pub enum Reconstruction { Pcm, Plm }

impl Reconstruction {
    /// kernel-set encoding: 0.0 = PCM, 1.0 = PLM.
    pub fn id(self) -> f64 {
        match self { Reconstruction::Pcm => 0.0, Reconstruction::Plm => 1.0 }
    }
    pub fn name(self) -> &'static str {
        match self { Reconstruction::Pcm => "pcm", Reconstruction::Plm => "plm" }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
// only HLLE is wired to the codegen path. the host hllc/hlld Riemann solvers
// (symbi-hydro) exist but are not yet traced to a kernel, so the CLI does not
// offer them — a flag that silently ran HLLE would be a lie. add the variant
// here when the corresponding gv flux kernel is traced.
pub enum Solver { Hlle }

impl Solver {
    pub fn name(self) -> &'static str {
        match self { Solver::Hlle => "hlle" }
    }
}

impl CliConfig {
    /// shorthand: extract with all defaults.
    pub fn extract(p: &mut ArgPool, default_prefix: &str) -> Self {
        Self::builder().extract(p, default_prefix)
    }

    /// builder for examples that need non-default t_final / cfl / data_dir.
    pub fn builder() -> CliConfigBuilder {
        CliConfigBuilder::default()
    }

    /// effective checkpoint interval: --ckpt-interval if set, else t_final/5.
    pub fn checkpoint_interval(&self) -> f64 {
        self.ckpt_interval.unwrap_or(self.t_final / 5.0)
    }
}

pub struct CliConfigBuilder {
    t_final: f64,
    cfl: f64,
    recon: Reconstruction,
    solver: Solver,
    theta: f64,
    timestepping: crate::state::Timestepping,
    data_dir: String,
}

impl Default for CliConfigBuilder {
    fn default() -> Self {
        CliConfigBuilder {
            t_final: 0.5,
            cfl: 0.4,
            recon: Reconstruction::Plm,
            solver: Solver::Hlle,
            theta: 1.5, // C++ default plm_theta (stencil_view.hpp)
            timestepping: crate::state::Timestepping::Rk2,
            data_dir: "output".into(),
        }
    }
}

impl CliConfigBuilder {
    pub fn t_final(mut self, v: f64) -> Self { self.t_final = v; self }
    pub fn cfl(mut self, v: f64) -> Self { self.cfl = v; self }
    pub fn recon(mut self, v: Reconstruction) -> Self { self.recon = v; self }
    pub fn solver(mut self, v: Solver) -> Self { self.solver = v; self }
    pub fn theta(mut self, v: f64) -> Self { self.theta = v; self }
    pub fn timestepping(mut self, v: crate::state::Timestepping) -> Self {
        self.timestepping = v; self
    }
    pub fn data_dir(mut self, v: impl Into<String>) -> Self { self.data_dir = v.into(); self }

    pub fn extract(self, p: &mut ArgPool, default_prefix: &str) -> CliConfig {
        use crate::state::Timestepping;
        let recon_name: String = p.take_or("--recon", self.recon.name().into());
        let recon = match recon_name.as_str() {
            "pcm" => Reconstruction::Pcm,
            "plm" => Reconstruction::Plm,
            other => panic!("--recon must be one of {{pcm, plm}}, got {}", other),
        };
        let solver_name: String = p.take_or("--solver", self.solver.name().into());
        let solver = match solver_name.as_str() {
            "hlle" => Solver::Hlle,
            other => panic!("--solver must be hlle (hllc/hlld are not wired to the codegen path), got {}", other),
        };
        let ts_default = match self.timestepping {
            Timestepping::Euler => "euler",
            Timestepping::Rk2 => "rk2",
            Timestepping::Rk3 => "rk3",
        };
        let ts_name: String = p.take_or("--timestepping", ts_default.into());
        let timestepping = match ts_name.as_str() {
            "euler" => Timestepping::Euler,
            "rk2" => Timestepping::Rk2,
            "rk3" => Timestepping::Rk3,
            other => panic!("--timestepping must be one of {{euler, rk2, rk3}}, got {}", other),
        };
        CliConfig {
            t_final: p.take_or("--tfinal", self.t_final),
            cfl: p.take_or("--cfl", self.cfl),
            recon,
            solver,
            theta: p.take_or("--theta", self.theta),
            timestepping,
            data_dir: p.take_or("--data-dir", self.data_dir),
            prefix: p.take_or("--prefix", default_prefix.into()),
            ckpt_interval: p.take("--ckpt-interval"),
        }
    }
}
