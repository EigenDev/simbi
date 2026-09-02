// =============================================================================
// regime_spec.rs
//
// `RegimeSpec` — physics regime metadata as a first-class data value, built
// against the `symbi-hydro::Regime` spine.
//
// the invariant this module encodes:
//
//   **`Rhd` and `Rmhd` collapse to consts-plus-c2p-hook.**
//
// every regime is identical at the metadata layer except for:
//   - a small set of `bool` / `u8` flag constants (is_relativistic, is_mhd,
//     has_energy, field count for the magnetic field),
//   - the conservative-to-primitive recovery function — algebraic for
//     newtonian, Newton-iterate for rhd, KKC false-position for rmhd
//     (carrier-generic but each with distinct internal physics).
//
// laws-as-OpNode encodes conservation laws as `algebra::Op` graphs; this
// module ships the metadata skeleton + the proof that the per-regime
// divergence is consts + one hook.
//
// usage:
//   use symbi_hydro::{Regime, Newtonian, RegimeSpec};
//   const SPEC: &'static RegimeSpec = Newtonian::SPEC;
//   assert!(!SPEC.is_relativistic);
//   assert_eq!(SPEC.fields.len(), 3); // den, mom, nrg
// =============================================================================

/// the shape of a single field a regime declares. layout primitive used by
/// `RegimeSpec.fields`. carries no carrier or dimension information — those
/// are supplied at the `Regime<S, D>` trait level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FieldKind {
    /// a rank-0 scalar (e.g., density, energy).
    Scalar,
    /// a rank-1 vector whose component count is the kernel's `D`
    /// (e.g., momentum: D components in the active spatial dimensionality).
    DimVector,
    /// a rank-1 vector with a fixed component count independent of `D`
    /// (e.g., the magnetic field is always 3-component, even in 2D RMHD).
    FixedVector { components: u8 },
}

/// declarative description of one conserved/primitive field. the bridge
/// between physics intent ("this regime has a momentum field") and runtime
/// layout (buffer count, stride). `FieldSpec` is shared by conserved and
/// primitive forms — `Cons.den` and `Prim.rho` are both
/// `FieldSpec { name: "den" / "rho", kind: Scalar }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FieldSpec {
    /// canonical field name; used for kernel-binding manifests and
    /// diagnostic display. matches the existing buffer-name convention in
    /// `symbi-discretize` / `symbi-aot` (e.g., "den", "mom_0", "nrg", "mag").
    pub name: &'static str,
    /// the field's structural kind (scalar vs vector, dim-sized vs fixed).
    pub kind: FieldKind,
}

/// the equation-of-state kind a regime is constrained to. cross-checked
/// against the `Eos<S>` impl chosen by the simulation at construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EosKind {
    /// ideal-gas adiabatic (`p = (gamma - 1) rho eps`); the regime evolves
    /// an energy field.
    Adiabatic,
    /// isothermal (`p = cs^2 rho`); no energy field, no c2p iteration.
    Isothermal,
}

/// the divergence of one regime from "the prototype" (newtonian).
/// `Rhd` and `Rmhd` are claims-to-collapse against this: at the metadata
/// layer their `RegimeSpec` differs from `NEWTONIAN_SPEC` by exactly:
///   - the flag bits in this struct (filled by `RegimeSpec::diff_flags`);
///   - the value of `c2p_kind` (algebraic vs iterative);
///   - (for RMHD) an extra `mag` field in the field list.
///
/// the collapse test (`regime_specs_collapse_to_consts_plus_c2p`)
/// asserts every non-flag, non-c2p-kind field is identical across regimes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RegimeSpec {
    /// canonical short name — used in kernel name suffixes / diagnostics.
    /// matches the existing crate naming ("newtonian", "rhd", "rmhd",
    /// "iso_newtonian").
    pub name: &'static str,
    /// conservative-side field manifest in declaration order. the runtime
    /// kernel-binding manifest preserves this order; tests assert it
    /// matches what `Cons` / `MhdCons` actually expose.
    pub fields: &'static [FieldSpec],
    /// **primitive-side field manifest** (rho / vel / pre / bcell ...).
    /// drives the I/O writer and reader's primitive-group iteration:
    /// `symbi-io::iter_components` consumes it to generate the on-disk dataset
    /// names, so `write_level_fields` and `read_level_fields` share one naming
    /// source of truth per regime.
    pub primitive_fields: &'static [FieldSpec],
    /// the EOS the regime is parametric on. checked against the runtime EOS.
    pub eos: EosKind,
    /// special relativity: `Cons.den = rho * W`, momentum carries `rho h W^2`.
    pub is_relativistic: bool,
    /// magnetohydrodynamics: extra `mag` field, induction term in flux,
    /// CT-friendly face-staggered representation in production.
    pub is_mhd: bool,
    /// false for isothermal (no energy equation, `nrg` field absent).
    pub has_energy: bool,
    /// MHD: whether the flux reads pre-materialized per-cell wave speeds (RMHD: the
    /// Mignone-Del Zanna quartic is too costly to inline per face, so a `wave_speeds`
    /// pass writes `wave_speed_l/r[d]` for the HLLE flux to read) vs computing the
    /// magnetosonic speed inline from the reconstructed L/R states (NMHD/iMHD closed
    /// form). drives the `wave_speeds` pass + the flux's `ws_l/ws_r` binding. false for
    /// non-MHD regimes.
    pub materializes_wave_speeds: bool,
    /// the c2p flavor — the one piece of physics that doesn't collapse to
    /// data because the recovery iteration is regime-specific.
    pub c2p_kind: C2pKind,
    /// the conservation laws this regime evolves — one `LawSpec` per
    /// conserved field, declaring `(field, kind)` only. the flux equations
    /// are the carrier-generic `Regime::to_flux` (single source of truth);
    /// this list is metadata consumed by `simulation_laws::validate`.
    pub laws: &'static [LawSpec],
}

/// the algorithmic flavor of conservative-to-primitive recovery. there are
/// exactly three across the supported regimes; the runtime dispatches the
/// matching iteration kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum C2pKind {
    /// `prim = f(cons)` is algebraic — no iteration. newtonian + isothermal.
    Algebraic,
    /// Newton-Raphson root-find on a 1-D pressure equation. rhd (mignone-
    /// bodo do-while transcribed as a sticky-done `Scalar::iterate`).
    NewtonOnPressure,
    /// Kastaun-Kalinani-Ciolfi false-position with Illinois half-damp on a
    /// 6-state bracket. rmhd.
    KkcFalsePosition,
}

// =============================================================================
// section 1.5 — laws as metadata.
//
// each conservation law evolves one conserved field U by
//   \partial_t U = -div(F(U)) + S(U)
// the flux F(U) is the carrier-generic `Regime::to_flux` (the single source of
// truth, run at `S = f64` and traced at `S = Gv` by the carrier gate); it is not
// encoded here as data. a `LawSpec` records only which field a
// regime evolves and the archetypal kind of that law.
// =============================================================================

/// canonical parameter naming convention shared by the graph-building source
/// builders (`source_spec.rs`) and `build_dot`. these names match the
/// runtime kernel-binding manifest so a higher layer doesn't need per-regime
/// routing tables.
pub mod law_params {
    /// scalar primitive density.
    pub const RHO: &str = "rho";
    /// scalar primitive pressure.
    pub const PRE: &str = "pre";
    /// per-axis primitive velocity component `vel_<k>`.
    pub fn vel(k: usize) -> String {
        format!("vel_{k}")
    }
}

/// kind of conservation law — the physical quantity being evolved. one
/// variant per archetypal field in classical hydro/MHD; new regimes (dust,
/// two-fluid, etc.) extend this enum as they land.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LawKind {
    Mass,
    Momentum,
    Energy,
    /// MHD induction equation — evolves B (RMHD's `mag` field).
    Induction,
}

/// the declarative description of one conservation law — the conserved
/// field name plus its law kind. the flux equation is deliberately not
/// carried here: the single source of truth for every regime's flux is the
/// carrier-generic `Regime::to_flux` (run at `S = f64`, traced at `S = Gv`
/// by the carrier gate). carrying a second hand-transcribed flux-as-`Graph`
/// layer here would let it drift from `to_flux` (e.g., iso-MHD losing its
/// magnetic stress); omitting it makes drift structurally impossible. this
/// struct is pure metadata, consumed by `simulation_laws::validate` (clause 2 —
/// every overlay/law targets a field the regime declares).
///
/// **identity by physics**: the `(field, kind)` pair
/// is the physical declaration; textual form carries no identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LawSpec {
    /// the conserved field this law evolves. matches a `FieldSpec.name`
    /// in the parent regime's `fields` array.
    pub field: &'static str,
    /// the law kind — the archetypal quantity (Mass / Momentum / etc.).
    pub kind: LawKind,
}

// =============================================================================
// section 1.6 — the per-regime law manifests (metadata only).
//
// each `*_LAWS` array declares which conserved fields a regime evolves and
// the archetypal kind of each. the flux equations live in the
// carrier-generic `Regime::to_flux` (single source of truth, traced at
// `S = Gv` by the carrier gate). `simulation_laws::validate` consumes these
// arrays to check that every law/overlay targets a declared field.
// =============================================================================

/// the three classical newtonian conservation laws.
pub const NEWTONIAN_LAWS: &[LawSpec] = &[
    LawSpec {
        field: "den",
        kind: LawKind::Mass,
    },
    LawSpec {
        field: "mom",
        kind: LawKind::Momentum,
    },
    LawSpec {
        field: "nrg",
        kind: LawKind::Energy,
    },
];

/// the isothermal-newtonian conservation laws — mass + momentum, no energy.
pub const ISO_NEWTONIAN_LAWS: &[LawSpec] = &[
    LawSpec {
        field: "den",
        kind: LawKind::Mass,
    },
    LawSpec {
        field: "mom",
        kind: LawKind::Momentum,
    },
];

/// the three RHD conservation laws — same shape as newtonian but with
/// relativistic semantics (`D = rho*W`, `M = rho*h*W^2*v`, `tau = H - p - D`).
pub const RHD_LAWS: &[LawSpec] = NEWTONIAN_LAWS;

/// the four RMHD conservation laws — mass + momentum + energy + induction.
pub const RMHD_LAWS: &[LawSpec] = &[
    LawSpec {
        field: "den",
        kind: LawKind::Mass,
    },
    LawSpec {
        field: "mom",
        kind: LawKind::Momentum,
    },
    LawSpec {
        field: "nrg",
        kind: LawKind::Energy,
    },
    LawSpec {
        field: "mag",
        kind: LawKind::Induction,
    },
];

/// isothermal MHD laws — mass + momentum + induction (no energy).
pub const ISO_MHD_LAWS: &[LawSpec] = &[
    LawSpec {
        field: "den",
        kind: LawKind::Mass,
    },
    LawSpec {
        field: "mom",
        kind: LawKind::Momentum,
    },
    LawSpec {
        field: "mag",
        kind: LawKind::Induction,
    },
];

impl FieldKind {
    /// the component count this field expands to at runtime given the
    /// kernel's spatial dimension `D`. `Scalar -> 1`, `DimVector -> D`,
    /// `FixedVector { components: n } -> n`. used by the layout dispatcher.
    #[inline]
    pub const fn components_at(self, d: usize) -> usize {
        match self {
            FieldKind::Scalar => 1,
            FieldKind::DimVector => d,
            FieldKind::FixedVector { components } => components as usize,
        }
    }
}

impl RegimeSpec {
    /// total scalar buffer count for the conservative state at dimension `D`.
    /// matches the buffer count the substrate codegen emits.
    pub const fn total_components_at(&self, d: usize) -> usize {
        let mut sum = 0usize;
        let mut i = 0usize;
        while i < self.fields.len() {
            sum += self.fields[i].kind.components_at(d);
            i += 1;
        }
        sum
    }
}

// =============================================================================
// section 2 — the per-regime specs. these are the data values the `Regime`
// trait exposes via its `SPEC` associated const. compare against each other
// in tests to validate the consts-plus-c2p-hook claim.
// =============================================================================

/// the newtonian conserved field set, in kernel-binding order.
/// {den (scalar), mom (D-vector), nrg (scalar)}.
const NEWTONIAN_FIELDS: &[FieldSpec] = &[
    FieldSpec {
        name: "den",
        kind: FieldKind::Scalar,
    },
    FieldSpec {
        name: "mom",
        kind: FieldKind::DimVector,
    },
    FieldSpec {
        name: "nrg",
        kind: FieldKind::Scalar,
    },
];

/// the **primitive** field sets driving the I/O writer. (rho, vel, pre,
/// bcell) — the canonical short names live in `symbi-io::field_layout`.
const NEWTONIAN_PRIMS: &[FieldSpec] = &[
    FieldSpec {
        name: "rho",
        kind: FieldKind::Scalar,
    },
    FieldSpec {
        name: "vel",
        kind: FieldKind::DimVector,
    },
    FieldSpec {
        name: "pre",
        kind: FieldKind::Scalar,
    },
];
const ISO_NEWTONIAN_PRIMS: &[FieldSpec] = &[
    FieldSpec {
        name: "rho",
        kind: FieldKind::Scalar,
    },
    FieldSpec {
        name: "vel",
        kind: FieldKind::DimVector,
    },
    // iso has no energy law; `pre` is substrate-owned (Option<...>),
    // written conditionally by the I/O writer when present.
];
// byte-aliased to the newtonian consts, exactly as RMHD/NMHD already share theirs:
// `primitive_fields` drives the HDF5 dataset names, so an independent copy that drifted
// would silently give RHD checkpoints a different on-disk schema. the collapse test
// asserts prims and laws equality alongside the conserved fields.
const RHD_PRIMS: &[FieldSpec] = NEWTONIAN_PRIMS;
const RMHD_PRIMS: &[FieldSpec] = &[
    FieldSpec {
        name: "rho",
        kind: FieldKind::Scalar,
    },
    // MHD velocity is a 3-vector (DOF=3), always — like B — even on a 1D/2D grid.
    FieldSpec {
        name: "vel",
        kind: FieldKind::FixedVector { components: 3 },
    },
    FieldSpec {
        name: "pre",
        kind: FieldKind::Scalar,
    },
    FieldSpec {
        name: "bcell",
        kind: FieldKind::FixedVector { components: 3 },
    },
];

/// isothermal: newtonian minus the energy field. matches `IsoCons<S, D>`'s
/// `Zero<S>` energy slot (the zst elision).
const ISO_NEWTONIAN_FIELDS: &[FieldSpec] = &[
    FieldSpec {
        name: "den",
        kind: FieldKind::Scalar,
    },
    FieldSpec {
        name: "mom",
        kind: FieldKind::DimVector,
    },
];

/// RHD: same field shape as newtonian (den = rho*W, mom = rho*h*W^2*v,
/// nrg = tau). the names and structural kinds match — only the *semantics*
/// of each conserved component differs.
const RHD_FIELDS: &[FieldSpec] = &[
    FieldSpec {
        name: "den",
        kind: FieldKind::Scalar,
    },
    FieldSpec {
        name: "mom",
        kind: FieldKind::DimVector,
    },
    FieldSpec {
        name: "nrg",
        kind: FieldKind::Scalar,
    },
];

/// RMHD: RHD plus the magnetic field. the magnetic field is always
/// 3-component, even when the simulation is 2D (the dropped axis sees zero
/// flux but the field still has 3 spatial components).
const RMHD_FIELDS: &[FieldSpec] = &[
    FieldSpec {
        name: "den",
        kind: FieldKind::Scalar,
    },
    // MHD momentum is a 3-vector (DOF=3), always — like B — even on a 1D/2D grid.
    FieldSpec {
        name: "mom",
        kind: FieldKind::FixedVector { components: 3 },
    },
    FieldSpec {
        name: "nrg",
        kind: FieldKind::Scalar,
    },
    FieldSpec {
        name: "mag",
        kind: FieldKind::FixedVector { components: 3 },
    },
];

/// isothermal MHD: RMHD fields minus the energy slot (den, mom, mag). matches
/// `IsoMhdCons<S, D>`'s `Zero<S>` energy slot (zst elision).
const ISO_MHD_FIELDS: &[FieldSpec] = &[
    FieldSpec {
        name: "den",
        kind: FieldKind::Scalar,
    },
    FieldSpec {
        name: "mom",
        kind: FieldKind::FixedVector { components: 3 },
    },
    FieldSpec {
        name: "mag",
        kind: FieldKind::FixedVector { components: 3 },
    },
];

/// isothermal MHD primitives: rho, vel, bcell — no `pre` (substrate-owned,
/// written conditionally by the I/O writer, mirroring ISO_NEWTONIAN_PRIMS).
const ISO_MHD_PRIMS: &[FieldSpec] = &[
    FieldSpec {
        name: "rho",
        kind: FieldKind::Scalar,
    },
    FieldSpec {
        name: "vel",
        kind: FieldKind::FixedVector { components: 3 },
    },
    FieldSpec {
        name: "bcell",
        kind: FieldKind::FixedVector { components: 3 },
    },
];

/// newtonian compressible Euler — the prototype every other spec is
/// compared against.
pub const NEWTONIAN_SPEC: RegimeSpec = RegimeSpec {
    name: "newtonian",
    fields: NEWTONIAN_FIELDS,
    primitive_fields: NEWTONIAN_PRIMS,
    eos: EosKind::Adiabatic,
    is_relativistic: false,
    is_mhd: false,
    has_energy: true,
    materializes_wave_speeds: false,
    c2p_kind: C2pKind::Algebraic,
    laws: NEWTONIAN_LAWS,
};

/// isothermal Euler — newtonian minus the energy equation.
pub const ISO_NEWTONIAN_SPEC: RegimeSpec = RegimeSpec {
    name: "iso_newtonian",
    fields: ISO_NEWTONIAN_FIELDS,
    primitive_fields: ISO_NEWTONIAN_PRIMS,
    eos: EosKind::Isothermal,
    is_relativistic: false,
    is_mhd: false,
    has_energy: false,
    materializes_wave_speeds: false,
    c2p_kind: C2pKind::Algebraic,
    laws: ISO_NEWTONIAN_LAWS,
};

/// special-relativistic hydrodynamics — newtonian + relativistic
/// + Newton-iterate c2p. **the collapse claim:** this spec differs from
/// `NEWTONIAN_SPEC` only in `name`, `is_relativistic`, and `c2p_kind`.
pub const RHD_SPEC: RegimeSpec = RegimeSpec {
    name: "rhd",
    fields: RHD_FIELDS,
    primitive_fields: RHD_PRIMS,
    eos: EosKind::Adiabatic,
    is_relativistic: true,
    is_mhd: false,
    has_energy: true,
    materializes_wave_speeds: false,
    c2p_kind: C2pKind::NewtonOnPressure,
    laws: RHD_LAWS,
};

/// relativistic MHD — RHD + magnetic field + KKC c2p. **the collapse claim:**
/// this spec differs from `RHD_SPEC` only in `name`, `is_mhd`, `c2p_kind`,
/// and the addition of the `mag` field (the one structural extension; every
/// other piece collapses).
pub const RMHD_SPEC: RegimeSpec = RegimeSpec {
    name: "rmhd",
    fields: RMHD_FIELDS,
    primitive_fields: RMHD_PRIMS,
    eos: EosKind::Adiabatic,
    is_relativistic: true,
    is_mhd: true,
    has_energy: true,
    materializes_wave_speeds: true,
    c2p_kind: C2pKind::KkcFalsePosition,
    laws: RMHD_LAWS,
};

/// newtonian ideal MHD. collapses to `RMHD_SPEC` except `name`,
/// `is_relativistic` (false), and `c2p_kind` (algebraic, a closed-form inversion where RMHD iterates) — the
/// conserved/primitive layout and the conservation laws are identical (MHD is
/// MHD; only the c2p inversion and the lorentz factors differ).
pub const NEWTONIAN_MHD_SPEC: RegimeSpec = RegimeSpec {
    name: "newtonian_mhd",
    fields: RMHD_FIELDS,
    primitive_fields: RMHD_PRIMS,
    eos: EosKind::Adiabatic,
    is_relativistic: false,
    is_mhd: true,
    has_energy: true,
    materializes_wave_speeds: false,
    c2p_kind: C2pKind::Algebraic,
    laws: RMHD_LAWS,
};

/// isothermal ideal MHD — newtonian MHD minus the energy equation, closed by
/// p = a^2 rho (Mignone 2007). collapses to `NEWTONIAN_MHD_SPEC` except `name`,
/// `eos` (Isothermal), `has_energy` (false), and the field/law lists (no energy
/// field/law). conserved = {den, mom, mag}; primitive = {rho, vel, bcell}.
pub const ISO_MHD_SPEC: RegimeSpec = RegimeSpec {
    name: "iso_mhd",
    fields: ISO_MHD_FIELDS,
    primitive_fields: ISO_MHD_PRIMS,
    eos: EosKind::Isothermal,
    is_relativistic: false,
    is_mhd: true,
    has_energy: false,
    materializes_wave_speeds: false,
    c2p_kind: C2pKind::Algebraic,
    laws: ISO_MHD_LAWS,
};

// =============================================================================
// section 3 — collapse proof tests. these are the load-bearing assertions
// against the invariant that `RegimeSpec` is consts + c2p hook. when
// future regimes land (e.g., iso-rhd, dust, two-fluid), they need to extend
// these tests with their own delta against the prototype.
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// the RHD vs newtonian collapse: every spec field must match except
    /// `name`, `is_relativistic`, and `c2p_kind`. this is the load-bearing
    /// assertion of the "consts-plus-c2p-hook" claim.
    #[test]
    fn rhd_collapses_to_newtonian_plus_relativistic_plus_c2p() {
        let n = &NEWTONIAN_SPEC;
        let s = &RHD_SPEC;

        // structurally identical fields (same names, same kinds, same order).
        // the semantics of each conserved field differ (newtonian den = rho;
        // rhd den = rho * W) but the layout collapses.
        assert_eq!(n.fields, s.fields);
        assert_eq!(
            n.primitive_fields, s.primitive_fields,
            "rhd primitive schema drifted from newtonian: HDF5 dataset names diverge"
        );
        assert_eq!(n.laws, s.laws);

        // shared metadata.
        assert_eq!(n.eos, s.eos);
        assert_eq!(n.is_mhd, s.is_mhd);
        assert_eq!(n.has_energy, s.has_energy);

        // the divergence — three fields, exactly.
        assert_ne!(n.name, s.name);
        assert_ne!(n.is_relativistic, s.is_relativistic);
        assert_ne!(n.c2p_kind, s.c2p_kind);
    }

    /// the RMHD vs RHD collapse: every spec field must match except
    /// `name`, `is_mhd`, `c2p_kind`, and the addition of the `mag` field.
    /// proves "MHD is RHD + magnetic-field-as-data".
    #[test]
    fn rmhd_collapses_to_rhd_plus_mhd_plus_mag_field_plus_c2p() {
        let s = &RHD_SPEC;
        let r = &RMHD_SPEC;

        // shared metadata.
        assert_eq!(s.eos, r.eos);
        assert_eq!(s.is_relativistic, r.is_relativistic);
        assert_eq!(s.has_energy, r.has_energy);

        // the divergence — name + flag + c2p + one extra field, exactly.
        assert_ne!(s.name, r.name);
        assert_ne!(s.is_mhd, r.is_mhd);
        assert_ne!(s.c2p_kind, r.c2p_kind);

        // structural extension: RMHD has the same field names as RHD in order (den, mom,
        // nrg) plus `mag` appended. the hydro prefix names match, but RMHD promotes mom to a
        // fixed 3-vector (vs RHD's D-vector) — MHD vectors are always 3-component.
        let s_count = s.fields.len();
        let r_count = r.fields.len();
        assert_eq!(r_count, s_count + 1);
        for (rf, sf) in r.fields[..s_count].iter().zip(s.fields.iter()) {
            assert_eq!(rf.name, sf.name, "hydro field names match in order");
        }
        assert_eq!(r.fields[1].name, "mom");
        assert_eq!(
            r.fields[1].kind,
            FieldKind::FixedVector { components: 3 },
            "MHD mom is a 3-vector"
        );
        assert_eq!(r.fields[s_count].name, "mag");
        assert_eq!(
            r.fields[s_count].kind,
            FieldKind::FixedVector { components: 3 }
        );
    }

    /// the newtonian-MHD vs RMHD collapse: every spec field must match except
    /// `name`, `is_relativistic` (false), and `c2p_kind` (algebraic, a
    /// closed-form inversion). proves "newtonian MHD is RMHD minus relativity minus the
    /// iterative inversion" — identical layout and laws, simpler physics.
    #[test]
    fn newtonian_mhd_collapses_to_rmhd_minus_relativistic_minus_c2p() {
        let r = &RMHD_SPEC;
        let nm = &NEWTONIAN_MHD_SPEC;

        // identical layout, laws, and MHD/energy metadata.
        assert_eq!(r.fields, nm.fields);
        assert_eq!(r.primitive_fields, nm.primitive_fields);
        assert_eq!(r.laws, nm.laws);
        assert_eq!(r.eos, nm.eos);
        assert_eq!(r.is_mhd, nm.is_mhd);
        assert_eq!(r.has_energy, nm.has_energy);

        // the divergence — name + relativistic flag + c2p, exactly.
        assert_ne!(r.name, nm.name);
        assert_ne!(r.is_relativistic, nm.is_relativistic);
        assert_ne!(r.c2p_kind, nm.c2p_kind);
        assert!(!nm.is_relativistic);
        assert_eq!(nm.c2p_kind, C2pKind::Algebraic);
    }

    /// isothermal collapse: newtonian minus the energy field. the
    /// prefix (den, mom) is bit-identical; `nrg` is gone.
    #[test]
    fn iso_collapses_to_newtonian_minus_energy() {
        let n = &NEWTONIAN_SPEC;
        let i = &ISO_NEWTONIAN_SPEC;

        // shared metadata.
        assert!(!n.is_relativistic && !i.is_relativistic);
        assert!(!n.is_mhd && !i.is_mhd);
        assert_eq!(n.c2p_kind, i.c2p_kind); // both Algebraic

        // structural reduction: iso drops the trailing `nrg` field.
        let i_count = i.fields.len();
        assert_eq!(i_count, n.fields.len() - 1);
        assert_eq!(i.fields, &n.fields[..i_count]);

        // the divergences.
        assert_ne!(n.eos, i.eos); // Adiabatic vs Isothermal
        assert_ne!(n.has_energy, i.has_energy);
    }

    // ----- field layout: components_at + total_components_at -----

    #[test]
    fn field_kind_components_at_resolves_correctly() {
        assert_eq!(FieldKind::Scalar.components_at(1), 1);
        assert_eq!(FieldKind::Scalar.components_at(3), 1);
        assert_eq!(FieldKind::DimVector.components_at(1), 1);
        assert_eq!(FieldKind::DimVector.components_at(2), 2);
        assert_eq!(FieldKind::DimVector.components_at(3), 3);
        assert_eq!(FieldKind::FixedVector { components: 3 }.components_at(1), 3);
        assert_eq!(FieldKind::FixedVector { components: 3 }.components_at(3), 3);
    }

    #[test]
    fn newtonian_total_components_grow_with_dimension() {
        // newtonian = den (1) + mom (D) + nrg (1) = D + 2
        assert_eq!(NEWTONIAN_SPEC.total_components_at(1), 3);
        assert_eq!(NEWTONIAN_SPEC.total_components_at(2), 4);
        assert_eq!(NEWTONIAN_SPEC.total_components_at(3), 5);
    }

    #[test]
    fn rhd_total_components_match_newtonian_at_every_dimension() {
        // the collapse claim extends to runtime layout: RHD's component
        // count equals newtonian's at every D (because the field list is
        // bit-identical). only the semantics of each component differs.
        for d in [1usize, 2, 3] {
            assert_eq!(
                NEWTONIAN_SPEC.total_components_at(d),
                RHD_SPEC.total_components_at(d),
                "newtonian and rhd must have identical buffer counts at D={d}",
            );
        }
    }

    #[test]
    fn rmhd_adds_three_components_for_the_magnetic_field() {
        // RMHD field count is D-independent: den(1) + mom(3) + nrg(1) + mag(3) = 8 at every
        // D, because the MHD momentum and magnetic field are fixed 3-vectors (DOF=3), unlike
        // RHD's D-vector momentum. (the spatial dimension is independent of the vector DOF count.)
        for d in [1usize, 2, 3] {
            assert_eq!(
                RMHD_SPEC.total_components_at(d),
                8,
                "rmhd is always 8 components (D={d})"
            );
        }
        // at D=3 the RHD momentum is itself 3, so RMHD = RHD + the 3 mag components.
        assert_eq!(
            RMHD_SPEC.total_components_at(3),
            RHD_SPEC.total_components_at(3) + 3
        );
    }

    #[test]
    fn iso_drops_one_scalar_from_newtonian() {
        for d in [1usize, 2, 3] {
            assert_eq!(
                ISO_NEWTONIAN_SPEC.total_components_at(d),
                NEWTONIAN_SPEC.total_components_at(d) - 1,
                "iso must drop exactly the `nrg` scalar at D={d}",
            );
        }
    }

    // ----- name uniqueness — sanity that the registry's identity key holds.

    // ----- trait derivation: Regime::is_relativistic / is_mhd / has_energy
    //                          read from SPEC (no per-regime override).

    #[test]
    fn trait_bool_methods_route_through_spec() {
        // proves the wiring: a regime's trait-method bool must equal its
        // SPEC's flag. closes the "consts collapse" claim end-to-end —
        // future regimes that forget to wire SPEC fail this test.
        use crate::{IsoNewtonian, Newtonian, Regime, Rhd, Rmhd};
        use symbi_ir::algebra::Scalar;

        fn check<R, S, const D: usize>(r: R, spec: &RegimeSpec)
        where
            R: Regime<S, D>,
            S: Scalar,
        {
            assert_eq!(r.is_relativistic(), spec.is_relativistic);
            assert_eq!(r.is_mhd(), spec.is_mhd);
            assert_eq!(r.has_energy(), spec.has_energy);
            // value equality: `pub const` synthesizes a fresh `&` per use site
            // (no stable memory location), so compare by value.
            assert_eq!(<R as Regime<S, D>>::SPEC, spec);
            assert_eq!(<R as Regime<S, D>>::SPEC.name, spec.name);
        }

        check::<_, f64, 1>(Newtonian, &NEWTONIAN_SPEC);
        check::<_, f64, 2>(Rhd, &RHD_SPEC);
        check::<_, f64, 3>(Rmhd, &RMHD_SPEC);
        check::<_, f64, 1>(IsoNewtonian, &ISO_NEWTONIAN_SPEC);
    }

    // ----- laws as data --------------------------------------------------

    #[test]
    fn newtonian_laws_match_the_field_layout() {
        // structural check: one law per conserved field, in the same order,
        // each with the expected kind. proves the laws table is a
        // structural declaration of "which physical quantities this regime
        // evolves," fixed in order and kind.
        let spec = &NEWTONIAN_SPEC;
        assert_eq!(spec.laws.len(), spec.fields.len());
        assert_eq!(spec.laws[0].field, "den");
        assert_eq!(spec.laws[0].kind, LawKind::Mass);
        assert_eq!(spec.laws[1].field, "mom");
        assert_eq!(spec.laws[1].kind, LawKind::Momentum);
        assert_eq!(spec.laws[2].field, "nrg");
        assert_eq!(spec.laws[2].kind, LawKind::Energy);
    }

    #[test]
    fn rhd_iso_laws_unchanged_by_b4iv() {
        // sanity check that adding RMHD laws didn't touch
        // the RHD / iso law tables.
        assert_eq!(RHD_LAWS.len(), 3);
        assert_eq!(ISO_NEWTONIAN_LAWS.len(), 2);
    }

    #[test]
    fn all_regime_names_are_distinct() {
        let names = [
            NEWTONIAN_SPEC.name,
            ISO_NEWTONIAN_SPEC.name,
            RHD_SPEC.name,
            RMHD_SPEC.name,
        ];
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                assert_ne!(
                    names[i], names[j],
                    "regime names must be distinct (kernel-name suffix collision)",
                );
            }
        }
    }
}
