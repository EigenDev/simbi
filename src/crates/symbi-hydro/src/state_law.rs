// =============================================================================
// state_law.rs
//
// the conserved state a regime builds from primitives, as a runtime descriptor.
//
// a source that relaxes toward a reference state has to know what "the conserved
// state" means: `rho v` on a newtonian gas, `rho h W^2 v` on a relativistic one,
// and `sqrt(gamma) rho h W^2 v` once a curved background densitizes it. that
// knowledge lives in `Regime::to_conserved` / `to_conserved_covariant`, which are
// generic over the concrete regime, while a user source is lowered from a runtime
// config that carries only a `RegimeSpec`.
//
// `StateLaw` closes that gap without restating a conservation law. it names the
// background and the closure, and dispatches to the regime's own conversion, so
// a source relaxes toward exactly the state the evolution stores. the metric is
// built from the cell position rather than passed in: `x_0/x_1/x_2` bind by name
// at splice, the same leaves a user expression reads, so a lift can evaluate the
// spacetime at the cell it is acting on.
//
// usage:
//   let law = StateLaw::newtonian(gamma);
//   let law = StateLaw::relativistic(gamma, Background::SchwarzschildKs { mass });
//   let u   = law.to_conserved_gv(rho, &vel, pre, dim);
// =============================================================================
use symbi_geometry::metric::{KerrKSCartesian, Metric, SchwarzschildKSCartesian};
use symbi_ir::Gv;

use crate::eos::IdealGas;
use crate::newtonian::Newtonian;
use crate::regime::Regime;
use crate::rhd::{Rhd, RhdGr};
use crate::spatial_metric::{Gamma, GammaInv, SpatialMetric};
use crate::state::Prim;
use symbi_algebra::Tensor;
use symbi_algebra::algebra::Numeric;

/// the background a state is built on. a curved entry carries the parameters its
/// metric needs; the metric itself is evaluated per cell from the position leaves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Background {
    Minkowski,
    /// ingoing kerr-schild for a non-rotating hole, in cartesian coordinates.
    SchwarzschildKsCartesian { mass: f64 },
    /// ingoing kerr-schild for a spinning hole, in cartesian coordinates. the spin is the
    /// specific angular momentum `a = J/M` about +z; the shift carries an azimuthal
    /// component, so treating a spinning hole as its non-rotating twin would build the
    /// conserved state against the wrong frame-dragging.
    KerrKsCartesian { mass: f64, spin: f64 },
}

/// the conservation law a source relaxes against: which background densitizes the
/// state, and which closure supplies the enthalpy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StateLaw {
    pub background: Background,
    pub relativistic: bool,
    /// the adiabatic index. an isothermal regime carries no energy slot and its
    /// conserved state is closure-free, so the value is unread there.
    pub gamma: f64,
}

impl StateLaw {
    /// a newtonian gas on a flat background.
    pub fn newtonian(gamma: f64) -> Self {
        Self {
            background: Background::Minkowski,
            relativistic: false,
            gamma,
        }
    }

    /// a relativistic gas on `background`.
    pub fn relativistic(gamma: f64, background: Background) -> Self {
        Self {
            background,
            relativistic: true,
            gamma,
        }
    }

    /// the cell position, from the leaves a spliced source binds by name. the
    /// godunov splice binds `x_0/x_1/x_2` to the cell centroid, so a lift reads the
    /// position of the cell it is acting on exactly as a user expression does.
    fn position<const D: usize>() -> Tensor<Gv, D> {
        Tensor::new(std::array::from_fn(|k| Gv::scalar(&format!("x_{k}"))))
    }

    /// the densitized conserved state on a curved background: assemble the regime from the
    /// metric evaluated at the cell, then let it convert. one body serves every chart, so a
    /// new background is a metric type rather than a second copy of this construction.
    fn curved<const D: usize, M: Metric<Gv, D>>(
        m: &M,
        eos: &IdealGas<Gv>,
        prim: &Prim<Gv, D>,
    ) -> Vec<Gv> {
        let x = Self::position::<D>();
        let regime = RhdGr {
            metric: SpatialMetric::<Gv, D>::new(
                Gamma::new(m.spatial_metric(x)),
                GammaInv::new(m.spatial_metric_inv(x)),
            ),
            alpha: m.lapse(x),
            shift: m.shift(x),
            sqrt_gamma: m.volume_factor(x),
        };
        let cons = regime.to_conserved(eos, prim);
        let mut out = vec![cons.den];
        out.extend((0..D).map(|k| cons.mom[k]));
        out.push(cons.nrg);
        out
    }

    /// the conserved state of a law on a FLAT background, at any spatial dimension.
    ///
    /// the curved charts carry metric impls from two dimensions upward — a one-dimensional
    /// kerr-schild slice is not a chart anyone writes — so a dimension-generic entry point
    /// that admitted them would put every flat one-dimensional source out of reach for a
    /// reason belonging to general relativity. this door takes the flat regimes at every
    /// dimension and refuses the curved ones by name.
    pub fn to_conserved_gv_flat<const D: usize>(
        &self,
        rho: Gv,
        vel: &[Gv],
        pre: Gv,
    ) -> Result<Vec<Gv>, String> {
        match self.background {
            Background::Minkowski => Ok(self.flat::<D>(rho, vel, pre)),
            other => Err(format!(
                "{other:?} has no {D}-dimensional chart; a curved background carries a \
                 conserved state from two spatial dimensions upward"
            )),
        }
    }

    /// the flat conserved state, newtonian or relativistic per the law.
    fn flat<const D: usize>(&self, rho: Gv, vel: &[Gv], pre: Gv) -> Vec<Gv> {
        let (prim, eos) = self.inputs::<D>(rho, vel, pre);
        let cons = if self.relativistic {
            Rhd.to_conserved(&eos, &prim)
        } else {
            Newtonian.to_conserved(&eos, &prim)
        };
        let mut out = vec![cons.den];
        out.extend((0..D).map(|k| cons.mom[k]));
        out.push(cons.nrg);
        out
    }

    /// the primitive state and closure every arm converts from.
    fn inputs<const D: usize>(
        &self,
        rho: Gv,
        vel: &[Gv],
        pre: Gv,
    ) -> (Prim<Gv, D>, IdealGas<Gv>) {
        (
            Prim::<Gv, D> {
                rho,
                vel: Tensor::new(std::array::from_fn(|k| {
                    vel.get(k).copied().unwrap_or(Gv::ZERO)
                })),
                pre,
            },
            IdealGas {
                gamma: Gv::from_f64(self.gamma),
            },
        )
    }

    /// the conserved state `[den, mom_0..mom_{D-1}, nrg]` this law builds from the
    /// primitives, in the same densitization the evolved state carries.
    ///
    /// every arm calls the regime's own conversion. an isothermal regime has no
    /// energy slot and returns `1 + D` components; every other arm returns `2 + D`.
    pub fn to_conserved_gv<const D: usize>(&self, rho: Gv, vel: &[Gv], pre: Gv) -> Vec<Gv>
    where
        // the metric impls are generated per spatial dimension, so the caller's `D`
        // carries the bounds; every instantiated dimension satisfies them.
        SchwarzschildKSCartesian<Gv>: Metric<Gv, D>,
        KerrKSCartesian<Gv>: Metric<Gv, D>,
    {
        let (prim, eos) = self.inputs::<D>(rho, vel, pre);
        match (self.relativistic, self.background) {
            (false, _) | (true, Background::Minkowski) => self.flat::<D>(rho, vel, pre),
            (true, Background::SchwarzschildKsCartesian { mass }) => {
                let m = SchwarzschildKSCartesian {
                    mass: Gv::from_f64(mass),
                };
                Self::curved::<D, _>(&m, &eos, &prim)
            }
            (true, Background::KerrKsCartesian { mass, spin }) => {
                let m = KerrKSCartesian {
                    mass: Gv::from_f64(mass),
                    spin: Gv::from_f64(spin),
                };
                Self::curved::<D, _>(&m, &eos, &prim)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_ir::{begin_trace, end_trace};

    /// trace a law's conversion and return the number of components it emits.
    fn arity<const D: usize>(law: StateLaw) -> usize
    where
        SchwarzschildKSCartesian<Gv>: Metric<Gv, D>,
        KerrKSCartesian<Gv>: Metric<Gv, D>,
    {
        begin_trace();
        let vel: Vec<Gv> = (0..D).map(|k| Gv::scalar(&format!("v{k}"))).collect();
        let out = law.to_conserved_gv::<D>(Gv::scalar("r"), &vel, Gv::scalar("p"));
        let _ = end_trace();
        out.len()
    }

    #[test]
    fn every_background_emits_one_slot_per_conservation_law() {
        // den + D momenta + nrg. a law that emitted a different count would splice
        // into the wrong conserved slots silently.
        for d in [2usize, 3] {
            let n = match d {
                2 => arity::<2>(StateLaw::newtonian(5.0 / 3.0)),
                _ => arity::<3>(StateLaw::newtonian(5.0 / 3.0)),
            };
            assert_eq!(n, 2 + d, "newtonian at D = {d}");
        }
    }

    #[test]
    fn a_flat_law_converts_at_one_dimension() {
        // the curved charts start at two dimensions, and a dimension-generic door that
        // admitted them would take every flat 1d source down with them. a 1d isothermal
        // sponge has nothing to do with kerr-schild and must keep working.
        begin_trace();
        let out = StateLaw::newtonian(5.0 / 3.0)
            .to_conserved_gv_flat::<1>(Gv::scalar("r"), &[Gv::scalar("v")], Gv::scalar("p"))
            .expect("a flat law converts at D = 1");
        let _ = end_trace();
        assert_eq!(out.len(), 3, "den + one momentum + nrg");
    }

    #[test]
    fn a_curved_law_is_refused_at_one_dimension_by_name() {
        // and the refusal says which background has no such chart, rather than failing
        // somewhere downstream with a trait-bound error the user cannot act on.
        begin_trace();
        let err = StateLaw::relativistic(
            4.0 / 3.0,
            Background::SchwarzschildKsCartesian { mass: 1.0 },
        )
        .to_conserved_gv_flat::<1>(Gv::scalar("r"), &[Gv::scalar("v")], Gv::scalar("p"))
        .unwrap_err();
        let _ = end_trace();
        assert!(err.contains("Schwarzschild"), "unhelpful: {err}");
        assert!(err.contains("1-dimensional"), "unhelpful: {err}");
    }

    #[test]
    fn a_spinning_hole_builds_a_different_state_than_a_static_one() {
        // the kerr shift carries an azimuthal component the schwarzschild one does not, so
        // the two backgrounds must not produce the same graph. mapping kerr onto the static
        // chart would drop the frame dragging silently, and the momentum slots would be
        // built against a frame the evolution does not use.
        let node_count = |law: StateLaw| {
            begin_trace();
            let vel: Vec<Gv> = (0..3).map(|k| Gv::scalar(&format!("v{k}"))).collect();
            let _ = law.to_conserved_gv::<3>(Gv::scalar("r"), &vel, Gv::scalar("p"));
            end_trace().graph.len()
        };
        let stat = node_count(StateLaw::relativistic(
            4.0 / 3.0,
            Background::SchwarzschildKsCartesian { mass: 1.0 },
        ));
        let spun = node_count(StateLaw::relativistic(
            4.0 / 3.0,
            Background::KerrKsCartesian {
                mass: 1.0,
                spin: 0.9,
            },
        ));
        assert_ne!(
            stat, spun,
            "the spinning and static charts traced identical graphs ({stat} nodes); the spin \
             is not reaching the metric"
        );
    }

    #[test]
    fn the_relativistic_arms_trace_at_every_supported_background() {
        // the point of the descriptor: one call site, several conservation laws. a
        // background whose metric failed to build would panic inside the trace.
        assert_eq!(arity::<3>(StateLaw::relativistic(4.0 / 3.0, Background::Minkowski)), 5);
        assert_eq!(
            arity::<3>(StateLaw::relativistic(
                4.0 / 3.0,
                Background::SchwarzschildKsCartesian { mass: 1.0 }
            )),
            5
        );
        assert_eq!(
            arity::<3>(StateLaw::relativistic(
                4.0 / 3.0,
                Background::KerrKsCartesian { mass: 1.0, spin: 0.9 }
            )),
            5
        );
    }

    #[test]
    fn a_curved_background_reads_the_cell_position() {
        // the metric is evaluated at the cell rather than passed in, so the curved
        // conversion must depend on the position leaves; a graph that ignored them
        // would be building the flat state under a curved label.
        begin_trace();
        let vel: Vec<Gv> = (0..3).map(|k| Gv::scalar(&format!("v{k}"))).collect();
        let _ = StateLaw::relativistic(
            4.0 / 3.0,
            Background::SchwarzschildKsCartesian { mass: 1.0 },
        )
        .to_conserved_gv::<3>(Gv::scalar("r"), &vel, Gv::scalar("p"));
        let k = end_trace();
        let names: Vec<String> = k.scalar_params.iter().map(|s| s.to_string()).collect();
        for axis in 0..3 {
            assert!(
                names.iter().any(|n| n == &format!("x_{axis}")),
                "the curved conversion never read x_{axis}; params were {names:?}"
            );
        }
    }

    #[test]
    fn a_flat_background_reads_no_position() {
        // and the flat arms must NOT bind position leaves, or every newtonian source
        // would acquire a spurious dependence on where its cell sits.
        begin_trace();
        let vel: Vec<Gv> = (0..3).map(|k| Gv::scalar(&format!("v{k}"))).collect();
        let _ = StateLaw::newtonian(5.0 / 3.0).to_conserved_gv::<3>(
            Gv::scalar("r"),
            &vel,
            Gv::scalar("p"),
        );
        let k = end_trace();
        let names: Vec<String> = k.scalar_params.iter().map(|s| s.to_string()).collect();
        assert!(
            !names.iter().any(|n| n.starts_with("x_")),
            "the flat conversion bound a position leaf: {names:?}"
        );
    }
}
