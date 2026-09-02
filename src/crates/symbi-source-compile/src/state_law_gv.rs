// =============================================================================
// state_law_gv.rs
//
// the traced conserved-state conversion for a `StateLaw`: dispatches to the
// regime's own `to_conserved` at `S = Gv`, evaluating a curved background's
// metric from the cell-position leaves (`x_0/x_1/x_2`, bound at splice), so a
// sponge or user source relaxes toward exactly the state the evolution
// stores. the physical descriptor (`StateLaw`, `Background`) lives with the
// physics in symbi-hydro; this extension supplies its compilation.
//
// usage:
//  use symbi_source_compile::StateLawGv;
//  let u = law.to_conserved_gv::<D>(cx, rho, &vel, pre);
// =============================================================================

use symbi_algebra::Tensor;
use symbi_algebra::algebra::Numeric;
use symbi_geometry::metric::{KerrKSCartesian, Metric, SchwarzschildKSCartesian};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::regime::Regime;
use symbi_hydro::rhd::{Rhd, RhdGr};
use symbi_hydro::spatial_metric::{Gamma, GammaInv, SpatialMetric};
use symbi_hydro::state::Prim;
use symbi_hydro::state::Valencia;
use symbi_hydro::state_law::{Background, StateLaw};
use symbi_ir::{Gv, TraceCx};

/// the cell position, from the leaves a spliced source binds by name. the
/// godunov splice binds `x_0/x_1/x_2` to the cell centroid, so a lift reads the
/// position of the cell it is acting on exactly as a user expression does.
fn position<'t, const D: usize>(cx: TraceCx<'t>) -> Tensor<Gv<'t>, D> {
    Tensor::new(std::array::from_fn(|k| cx.scalar(&format!("x_{k}"))))
}

/// the densitized conserved state on a curved background: assemble the regime from the
/// metric evaluated at the cell, then let it convert. one body serves every chart, so a
/// new background is a metric type rather than a second copy of this construction.
fn curved<'t, const D: usize, M: Metric<Gv<'t>, D>>(
    cx: TraceCx<'t>,
    m: &M,
    eos: &IdealGas<Gv<'t>>,
    prim: &Prim<Gv<'t>, D>,
) -> Vec<Gv<'t>> {
    let x = position::<D>(cx);
    let regime = RhdGr {
        metric: SpatialMetric::<Gv, D>::new(
            Gamma::new(m.spatial_metric(x)),
            GammaInv::new(m.spatial_metric_inv(x)),
        ),
        alpha: m.lapse(x),
        shift: m.shift(x),
        sqrt_gamma: m.volume_factor(x),
    };
    // the sponge reference enters the valencia door with its frame stated:
    // on a curved background the stored velocity components are v^i.
    let cons = regime.to_conserved(eos, &Valencia(*prim));
    let mut out = vec![cons.0.den];
    out.extend((0..D).map(|k| cons.0.mom[k]));
    out.push(cons.0.nrg);
    out
}

/// the flat conserved state, newtonian or relativistic per the law.
fn flat<'t, const D: usize>(
    law: &StateLaw,
    rho: Gv<'t>,
    vel: &[Gv<'t>],
    pre: Gv<'t>,
) -> Vec<Gv<'t>> {
    let (prim, eos) = inputs::<D>(law, rho, vel, pre);
    let cons = if law.relativistic {
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
fn inputs<'t, const D: usize>(
    law: &StateLaw,
    rho: Gv<'t>,
    vel: &[Gv<'t>],
    pre: Gv<'t>,
) -> (Prim<Gv<'t>, D>, IdealGas<Gv<'t>>) {
    (
        Prim::<Gv, D> {
            rho,
            vel: Tensor::new(std::array::from_fn(|k| {
                vel.get(k).copied().unwrap_or(Gv::ZERO)
            })),
            pre,
        },
        IdealGas {
            gamma: Gv::from_f64(law.gamma),
        },
    )
}

/// traced conserved-state conversion, as an extension of the physical
/// descriptor.
pub trait StateLawGv {
    /// the conserved state of a law on a FLAT background, at any spatial dimension.
    ///
    /// the curved charts carry metric impls from two dimensions upward — a one-dimensional
    /// kerr-schild slice is not a chart anyone writes — so a dimension-generic entry point
    /// that admitted them would put every flat one-dimensional source out of reach for a
    /// reason belonging to general relativity. this door takes the flat regimes at every
    /// dimension and refuses the curved ones by name.
    fn to_conserved_gv_flat<'t, const D: usize>(
        &self,
        rho: Gv<'t>,
        vel: &[Gv<'t>],
        pre: Gv<'t>,
    ) -> Result<Vec<Gv<'t>>, String>;

    /// the conserved state `[den, mom_0..mom_{D-1}, nrg]` this law builds from the
    /// primitives, in the same densitization the evolved state carries.
    ///
    /// every arm calls the regime's own conversion. an isothermal regime has no
    /// energy slot and returns `1 + D` components; every other arm returns `2 + D`.
    fn to_conserved_gv<'t, const D: usize>(
        &self,
        cx: TraceCx<'t>,
        rho: Gv<'t>,
        vel: &[Gv<'t>],
        pre: Gv<'t>,
    ) -> Vec<Gv<'t>>
    where
        SchwarzschildKSCartesian<Gv<'t>>: Metric<Gv<'t>, D>,
        KerrKSCartesian<Gv<'t>>: Metric<Gv<'t>, D>;
}

impl StateLawGv for StateLaw {
    fn to_conserved_gv_flat<'t, const D: usize>(
        &self,
        rho: Gv<'t>,
        vel: &[Gv<'t>],
        pre: Gv<'t>,
    ) -> Result<Vec<Gv<'t>>, String> {
        match self.background {
            Background::Minkowski => Ok(flat::<D>(self, rho, vel, pre)),
            other => Err(format!(
                "{other:?} has no {D}-dimensional chart; a curved background carries a \
                 conserved state from two spatial dimensions upward"
            )),
        }
    }

    fn to_conserved_gv<'t, const D: usize>(
        &self,
        cx: TraceCx<'t>,
        rho: Gv<'t>,
        vel: &[Gv<'t>],
        pre: Gv<'t>,
    ) -> Vec<Gv<'t>>
    where
        SchwarzschildKSCartesian<Gv<'t>>: Metric<Gv<'t>, D>,
        KerrKSCartesian<Gv<'t>>: Metric<Gv<'t>, D>,
    {
        let (prim, eos) = inputs::<D>(self, rho, vel, pre);
        match (self.relativistic, self.background) {
            (false, _) | (true, Background::Minkowski) => flat::<D>(self, rho, vel, pre),
            (true, Background::SchwarzschildKsCartesian { mass }) => {
                let m = SchwarzschildKSCartesian {
                    mass: Gv::from_f64(mass),
                };
                curved::<D, _>(cx, &m, &eos, &prim)
            }
            (true, Background::KerrKsCartesian { mass, spin }) => {
                let m = KerrKSCartesian {
                    mass: Gv::from_f64(mass),
                    spin: Gv::from_f64(spin),
                };
                curved::<D, _>(cx, &m, &eos, &prim)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_hydro::state_law::StateLaw;
    use symbi_ir::trace;

    /// trace a law's conversion and return the number of components it emits.
    fn arity<const D: usize>(law: StateLaw) -> usize
    where
        for<'t> SchwarzschildKSCartesian<Gv<'t>>: Metric<Gv<'t>, D>,
        for<'t> KerrKSCartesian<Gv<'t>>: Metric<Gv<'t>, D>,
    {
        let (_, len) = trace(|cx| {
            let vel: Vec<Gv> = (0..D).map(|k| cx.scalar(&format!("v{k}"))).collect();
            law.to_conserved_gv::<D>(cx, cx.scalar("r"), &vel, cx.scalar("p"))
                .len()
        });
        len
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
        let (_, len) = trace(|cx| {
            StateLaw::newtonian(5.0 / 3.0)
                .to_conserved_gv_flat::<1>(cx.scalar("r"), &[cx.scalar("v")], cx.scalar("p"))
                .expect("a flat law converts at D = 1")
                .len()
        });
        assert_eq!(len, 3, "den + one momentum + nrg");
    }

    #[test]
    fn a_curved_law_is_refused_at_one_dimension_by_name() {
        // and the refusal says which background has no such chart, rather than failing
        // somewhere downstream with a trait-bound error the user cannot act on.
        let (_, err) = trace(|cx| {
            StateLaw::relativistic(
                4.0 / 3.0,
                Background::SchwarzschildKsCartesian { mass: 1.0 },
            )
            .to_conserved_gv_flat::<1>(cx.scalar("r"), &[cx.scalar("v")], cx.scalar("p"))
            .unwrap_err()
        });
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
            let (kernel, _) = trace(|cx| {
                let vel: Vec<Gv> = (0..3).map(|k| cx.scalar(&format!("v{k}"))).collect();
                let _ = law.to_conserved_gv::<3>(cx, cx.scalar("r"), &vel, cx.scalar("p"));
            });
            kernel.graph().len()
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
        assert_eq!(
            arity::<3>(StateLaw::relativistic(4.0 / 3.0, Background::Minkowski)),
            5
        );
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
                Background::KerrKsCartesian {
                    mass: 1.0,
                    spin: 0.9
                }
            )),
            5
        );
    }

    #[test]
    fn a_curved_background_reads_the_cell_position() {
        // the metric is evaluated at the cell rather than passed in, so the curved
        // conversion must depend on the position leaves; a graph that ignored them
        // would be building the flat state under a curved label.
        let (k, _) = trace(|cx| {
            let vel: Vec<Gv> = (0..3).map(|kk| cx.scalar(&format!("v{kk}"))).collect();
            let _ = StateLaw::relativistic(
                4.0 / 3.0,
                Background::SchwarzschildKsCartesian { mass: 1.0 },
            )
            .to_conserved_gv::<3>(cx, cx.scalar("r"), &vel, cx.scalar("p"));
        });
        let names: Vec<String> = k.scalar_params().iter().map(|s| s.to_string()).collect();
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
        let (k, _) = trace(|cx| {
            let vel: Vec<Gv> = (0..3).map(|kk| cx.scalar(&format!("v{kk}"))).collect();
            let _ = StateLaw::newtonian(5.0 / 3.0).to_conserved_gv::<3>(
                cx,
                cx.scalar("r"),
                &vel,
                cx.scalar("p"),
            );
        });
        let names: Vec<String> = k.scalar_params().iter().map(|s| s.to_string()).collect();
        assert!(
            !names.iter().any(|n| n.starts_with("x_")),
            "the flat conversion bound a position leaf: {names:?}"
        );
    }
}
