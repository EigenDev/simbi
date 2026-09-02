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
//   // the traced conversion lives in symbi-source-compile (`StateLawGv`).
// =============================================================================

/// the background a state is built on. a curved entry carries the parameters its
/// metric needs; the metric itself is evaluated per cell from the position leaves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Background {
    Minkowski,
    /// ingoing kerr-schild for a non-rotating hole, in cartesian coordinates.
    SchwarzschildKsCartesian {
        mass: f64,
    },
    /// ingoing kerr-schild for a spinning hole, in cartesian coordinates. the spin is the
    /// specific angular momentum `a = J/M` about +z; the shift carries an azimuthal
    /// component, so treating a spinning hole as its non-rotating twin would build the
    /// conserved state against the wrong frame-dragging.
    KerrKsCartesian {
        mass: f64,
        spin: f64,
    },
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
}
