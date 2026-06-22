// =============================================================================
// scalar_ref.rs
//
// typed names for the kernel SCALAR parameters that cross the trace -> dispatch
// ABI — the scalar analog of `FieldRef`, mirroring `MeshScalar`'s minted-once
// discipline over the WHOLE closed scalar vocabulary.
//
// a kernel scalar is the link between the traced graph (which DECLARES it via
// `Gv::scalar(name)`) and the host dispatch (which SUPPLIES its value by name).
// that link is a bare string, minted independently on each side, so it drifts:
// the same per-axis convention is spelled `inv_dx_0` / `x_lo_0` / `body_2_pos_0`
// across the trace builders and the dispatch resolvers with nothing forcing the
// two sides to agree.
//
// `ScalarRef` mints each CLOSED-vocabulary wire name in EXACTLY one place
// (`name()`), and `parse()` recovers the typed ref. a dispatch resolver parses
// once at manifest load (off the per-dispatch hot path) and then matches
// EXHAUSTIVELY — adding a scalar is a compile error until every match covers it,
// and the name can no longer be wrong.
//
// the vocabulary is NOT total over every kernel scalar: spec-source kernels
// declare arbitrary user-named knobs (`gm`, `g_ext_0`, `xm_1`, ...) that are not
// a fixed vocabulary. `parse()` returns `None` for those — the caller then looks
// them up in the spec's string-keyed scalar map. this is the SAME split FieldRef
// leaves: a closed typed core + an open spec tail.
//
// usage:
//  // producer (trace): Gv::scalar(&ScalarRef::XLo(d).name())
//  // consumer (dispatch): match ScalarRef::parse(name) {
//  //     Some(ScalarRef::XLo(ax)) => x_lo[ax],
//  //     None => spec_scalars[name],  // open user/spec knob
//  // }
// =============================================================================

use crate::scalar_param::MeshScalar;

/// a body field within the immersed-body scalar block: the gravity/accretion
/// knobs (`mass`/`soft`/`racc`/`sink`/`delta`) and the per-axis state
/// (`pos`/`vel`). addressed under a body index in `ScalarRef::Body`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum BodyScalar {
    Mass,
    Soft,
    Racc,
    Sink,
    Delta,
    Pos(u8),
    Vel(u8),
}

impl BodyScalar {
    /// the field-name tail (after `body_{idx}_`) on the wire.
    fn name(self) -> String {
        match self {
            BodyScalar::Mass => "mass".to_string(),
            BodyScalar::Soft => "soft".to_string(),
            BodyScalar::Racc => "racc".to_string(),
            BodyScalar::Sink => "sink".to_string(),
            BodyScalar::Delta => "delta".to_string(),
            BodyScalar::Pos(ax) => format!("pos_{ax}"),
            BodyScalar::Vel(ax) => format!("vel_{ax}"),
        }
    }

    fn parse(field: &str) -> Option<Self> {
        if let Some(ax) = field.strip_prefix("pos_") {
            return ax.parse().ok().map(BodyScalar::Pos);
        }
        if let Some(ax) = field.strip_prefix("vel_") {
            return ax.parse().ok().map(BodyScalar::Vel);
        }
        match field {
            "mass" => Some(BodyScalar::Mass),
            "soft" => Some(BodyScalar::Soft),
            "racc" => Some(BodyScalar::Racc),
            "sink" => Some(BodyScalar::Sink),
            "delta" => Some(BodyScalar::Delta),
            _ => None,
        }
    }
}

/// a typed kernel scalar-parameter name over the CLOSED dispatch vocabulary.
/// every variant round-trips through `name()`/`parse()`; `parse()` returns `None`
/// for an OPEN spec/user-source knob (the caller resolves those by string).
///
/// `is_int` is NOT carried here — the int/float sort lives in the kernel manifest
/// alongside the ref; the ghost-fill `MapType`/`Arg` variants are the int-lane
/// members and the manifest tags them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum ScalarRef {
    /// ideal-gas adiabatic index `gamma` (the energy-regime EOS param).
    Gamma,
    /// isothermal sound speed `cs` (the iso EOS param — the dual of `Gamma`).
    Cs,
    /// theta-MC limiter free parameter `theta`.
    Theta,
    /// the time step `dt` (the godunov/source stage weight feeds through this name).
    Dt,
    /// the SSP convex coefficient on the prior state `a0`.
    A0,
    /// the SSP convex coefficient on the flux-evolved state `ac`.
    Ac,
    /// the simulation time `t`.
    Time,
    /// the cartesian CFL width `inv_dx_{ax}` = `1/dx[ax]`.
    InvDx(u8),
    /// the axis origin `x_lo_{ax}`.
    XLo(u8),
    /// the axis step / log-slope `dx_{ax}`.
    Dx(u8),
    /// a moving-mesh rate (`mesh_hdil`, `mesh_adot_{ax}`, `mesh_vtrans_{ax}`).
    Mesh(MeshScalar),
    /// a user-source tunable knob `p{i}`.
    UserParam(u32),
    /// an immersed-body scalar `body_{idx}_{field}`.
    Body { idx: u8, field: BodyScalar },
    /// the ghost-fill per-axis lattice-map kind `map_type_{ax}` (INT lane).
    MapType(u8),
    /// the ghost-fill per-axis lattice-map argument `arg_{ax}` (INT lane).
    Arg(u8),
    /// the ghost-fill per-axis velocity-flip sign `vel_sign_{ax}`.
    VelSign(u8),
}

impl ScalarRef {
    /// the ONE place a closed-vocabulary scalar wire name is minted. holds
    /// `parse(x.name()) == Some(x)` for every variant.
    pub fn name(self) -> String {
        match self {
            ScalarRef::Gamma => "gamma".to_string(),
            ScalarRef::Cs => "cs".to_string(),
            ScalarRef::Theta => "theta".to_string(),
            ScalarRef::Dt => "dt".to_string(),
            ScalarRef::A0 => "a0".to_string(),
            ScalarRef::Ac => "ac".to_string(),
            ScalarRef::Time => "t".to_string(),
            ScalarRef::InvDx(ax) => format!("inv_dx_{ax}"),
            ScalarRef::XLo(ax) => format!("x_lo_{ax}"),
            ScalarRef::Dx(ax) => format!("dx_{ax}"),
            ScalarRef::Mesh(m) => m.name(),
            ScalarRef::UserParam(i) => format!("p{i}"),
            ScalarRef::Body { idx, field } => format!("body_{idx}_{}", field.name()),
            ScalarRef::MapType(ax) => format!("map_type_{ax}"),
            ScalarRef::Arg(ax) => format!("arg_{ax}"),
            ScalarRef::VelSign(ax) => format!("vel_sign_{ax}"),
        }
    }

    /// the inverse of `name`: recover the typed scalar from a wire name, or `None`
    /// when the name is an OPEN spec/user-source knob (resolved by string at the
    /// call site). holds `parse(x.name()) == Some(x)` for every variant.
    pub fn parse(name: &str) -> Option<Self> {
        // bare-name singletons.
        match name {
            "gamma" => return Some(ScalarRef::Gamma),
            "cs" => return Some(ScalarRef::Cs),
            "theta" => return Some(ScalarRef::Theta),
            "dt" => return Some(ScalarRef::Dt),
            "a0" => return Some(ScalarRef::A0),
            "ac" => return Some(ScalarRef::Ac),
            "t" => return Some(ScalarRef::Time),
            _ => {}
        }

        // the mesh family owns its own `mesh_*` namespace (tried before the bare
        // geom prefixes so the convention stays in `MeshScalar`).
        if let Some(m) = MeshScalar::parse(name) {
            return Some(ScalarRef::Mesh(m));
        }

        // per-axis geometry. `inv_dx_` must be tried before `dx_` (prefix overlap).
        if let Some(ax) = name.strip_prefix("inv_dx_") {
            return ax.parse().ok().map(ScalarRef::InvDx);
        }
        if let Some(ax) = name.strip_prefix("x_lo_") {
            return ax.parse().ok().map(ScalarRef::XLo);
        }
        if let Some(ax) = name.strip_prefix("dx_") {
            return ax.parse().ok().map(ScalarRef::Dx);
        }

        // ghost-fill lattice-map family.
        if let Some(ax) = name.strip_prefix("map_type_") {
            return ax.parse().ok().map(ScalarRef::MapType);
        }
        if let Some(ax) = name.strip_prefix("arg_") {
            return ax.parse().ok().map(ScalarRef::Arg);
        }
        if let Some(ax) = name.strip_prefix("vel_sign_") {
            return ax.parse().ok().map(ScalarRef::VelSign);
        }

        // immersed-body block: `body_{idx}_{field}`.
        if let Some(rest) = name.strip_prefix("body_") {
            let (idx_str, field) = rest.split_once('_')?;
            let idx: u8 = idx_str.parse().ok()?;
            return BodyScalar::parse(field).map(|field| ScalarRef::Body { idx, field });
        }

        // user-source knob `p{i}` (tried last — a bare `p` followed by digits, NOT
        // a prefix, so it does not shadow `pre`/`pos` which never reach here).
        if let Some(i) = name.strip_prefix('p') {
            if !i.is_empty() && i.bytes().all(|b| b.is_ascii_digit()) {
                return i.parse().ok().map(ScalarRef::UserParam);
            }
        }

        None
    }
}

/// a SERIALIZED kernel scalar binding: the typed core (`Ref`) over the closed
/// dispatch vocabulary, plus an open tail (`Spec`) for the spec/user-source knobs
/// (`gm`, `g_ext_0`, `xm_1`, ...) that are NOT a fixed vocabulary and resolve
/// against the spec's string-keyed scalar map at the call site. this is the scalar
/// analog of `FieldBind` (typed core + open tail): the manifest is born typed for
/// the closed dispatch scalars, and the spec knobs round-trip losslessly as the
/// raw string. the closed dispatch resolvers match `Ref` exhaustively; a spec
/// kernel's resolver also handles the open `Spec` knob.
#[derive(Clone, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum ScalarBind {
    Ref(ScalarRef),
    Spec(Box<str>),
}

impl ScalarBind {
    /// classify a wire name: a known closed-vocabulary scalar becomes `Ref`, any
    /// other name is held verbatim as `Spec`. the inverse of `name()`.
    pub fn from_name(s: &str) -> Self {
        ScalarRef::parse(s).map(ScalarBind::Ref).unwrap_or_else(|| ScalarBind::Spec(s.into()))
    }

    /// the wire name this bind names. `Ref` mints its canonical spelling through
    /// `ScalarRef::name`; `Spec` returns its stored string unchanged.
    pub fn name(&self) -> String {
        match self {
            ScalarBind::Ref(r) => r.name(),
            ScalarBind::Spec(s) => s.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_variants() -> Vec<ScalarRef> {
        let mut v = vec![
            ScalarRef::Gamma,
            ScalarRef::Cs,
            ScalarRef::Theta,
            ScalarRef::Dt,
            ScalarRef::A0,
            ScalarRef::Ac,
            ScalarRef::Time,
        ];
        for ax in 0..4u8 {
            v.push(ScalarRef::InvDx(ax));
            v.push(ScalarRef::XLo(ax));
            v.push(ScalarRef::Dx(ax));
            v.push(ScalarRef::MapType(ax));
            v.push(ScalarRef::Arg(ax));
            v.push(ScalarRef::VelSign(ax));
            v.push(ScalarRef::Mesh(MeshScalar::Adot(ax)));
            v.push(ScalarRef::Mesh(MeshScalar::Vtrans(ax)));
        }
        v.push(ScalarRef::Mesh(MeshScalar::Hdil));
        for i in 0..6u32 {
            v.push(ScalarRef::UserParam(i));
        }
        for idx in 0..4u8 {
            for field in [
                BodyScalar::Mass,
                BodyScalar::Soft,
                BodyScalar::Racc,
                BodyScalar::Sink,
                BodyScalar::Delta,
            ] {
                v.push(ScalarRef::Body { idx, field });
            }
            for ax in 0..3u8 {
                v.push(ScalarRef::Body { idx, field: BodyScalar::Pos(ax) });
                v.push(ScalarRef::Body { idx, field: BodyScalar::Vel(ax) });
            }
        }
        v
    }

    // the invariant the whole module exists for: name() and parse() are exact
    // inverses over every representable variant, so a producer and a consumer that
    // both route through ScalarRef can never diverge on a name.
    #[test]
    fn name_parse_round_trips() {
        for r in all_variants() {
            assert_eq!(ScalarRef::parse(&r.name()), Some(r), "round-trip failed for {r:?}");
        }
    }

    // the mesh family re-uses MeshScalar — confirm the bridge holds in both directions.
    #[test]
    fn mesh_family_bridges() {
        assert_eq!(ScalarRef::parse("mesh_hdil"), Some(ScalarRef::Mesh(MeshScalar::Hdil)));
        assert_eq!(ScalarRef::parse("mesh_adot_1"), Some(ScalarRef::Mesh(MeshScalar::Adot(1))));
        assert_eq!(ScalarRef::parse("mesh_vtrans_0"), Some(ScalarRef::Mesh(MeshScalar::Vtrans(0))));
    }

    // OPEN spec/user-source knobs are NOT in the closed vocabulary: parse must
    // return None so the caller falls through to the spec string map. this is the
    // documented boundary, not a bug.
    #[test]
    fn rejects_open_spec_scalars() {
        for n in ["gm", "g_ext_0", "xm_1", "body_radius", "value", "scale", "alpha", "cs2"] {
            assert_eq!(ScalarRef::parse(n), None, "'{n}' must stay an open spec scalar");
        }
    }

    // per-cell interpreter vars (resolve_runtime_param) are a SEPARATE path; they
    // must not be mistaken for closed kernel scalars (except `t` / `p{i}`, which
    // ARE shared names — the call site decides which path).
    #[test]
    fn rejects_per_cell_vars() {
        for n in ["rho", "pre", "vel_0", "x_2", "pos", "p"] {
            assert_eq!(ScalarRef::parse(n), None, "'{n}' is not a closed kernel scalar");
        }
    }

    // a closed-vocabulary name binds typed (`Ref`); an open spec/user knob is held
    // verbatim (`Spec`). both round-trip through name() exactly — the scalar analog
    // of the FieldBind classification gate.
    #[test]
    fn scalar_bind_classifies_and_round_trips() {
        for r in all_variants() {
            let bind = ScalarBind::from_name(&r.name());
            assert_eq!(bind, ScalarBind::Ref(r), "closed name should bind Ref for {r:?}");
            assert_eq!(bind.name(), r.name(), "Ref round-trip failed for {r:?}");
        }
        // open spec/user-source knobs are not in the closed vocabulary — they round-trip as Spec.
        for spec in ["gm", "g_ext_0", "xm_1", "value", "scale", "alpha", "cs2"] {
            let bind = ScalarBind::from_name(spec);
            assert_eq!(bind, ScalarBind::Spec(spec.into()), "'{spec}' should bind Spec");
            assert_eq!(bind.name(), spec, "Spec round-trip failed for '{spec}'");
        }
    }
}
