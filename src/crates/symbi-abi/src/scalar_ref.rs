// =============================================================================
// scalar_ref.rs
//
// typed names for the kernel scalar parameters that cross the trace -> dispatch
// ABI — the scalar analog of `FieldRef`, mirroring `MeshScalar`'s minted-once
// discipline across the closed scalar vocabulary.
//
// a kernel scalar is the link between the traced graph (which declares it via
// `Gv::scalar(name)`) and the host dispatch (which supplies its value by name).
// that link is a bare string, minted independently on each side, so it drifts:
// the same per-axis convention is spelled `inv_dx_0` / `x_lo_0` / `body_2_pos_0`
// across the trace builders and the dispatch resolvers with nothing forcing the
// two sides to agree.
//
// `ScalarRef` mints each closed-vocabulary wire name in exactly one place
// (`name()`), and `parse()` recovers the typed ref. a dispatch resolver parses
// once at manifest load (off the per-dispatch hot path) and then matches
// exhaustively — adding a scalar is a compile error until every match covers it.
//
// the vocabulary covers a fixed set of kernel scalars; spec-source kernels
// declare arbitrary user-named knobs (`gm`, `g_ext_0`, `xm_1`, ...) outside that
// set. `parse()` returns `None` for those — the caller then looks them up in the
// spec's string-keyed scalar map. this is the same split FieldRef leaves: a
// closed typed core + an open spec tail.
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
    /// which softened-gravity family `soft` parameterizes: `0` = Plummer (a real extended
    /// profile that approaches the Newtonian potential only asymptotically), `1` = compact
    /// (exactly Newtonian outside `soft`, regular within). carried as a scalar so one baked
    /// kernel serves both.
    SoftKind,
    Racc,
    /// the radius of the surface this body penalizes, and zero for a body that penalizes none.
    /// the zero absent-value is what makes a max over slots equal the penalized region exactly:
    /// a purely gravitational slot then carries an indicator that vanishes at every point,
    /// while `Racc` carries a placeholder radius its consumers screen off by other means.
    Rmask,
    Sink,
    Delta,
    Pos(u8),
    Vel(u8),
    /// the `k`-th component of the body's angular-velocity vector (world frame).
    Omega(u8),
    /// the `k`-th (row-major, 0..9) component of the body's orientation rotation matrix.
    Rot(u8),
}

impl BodyScalar {
    /// the field-name tail (after `body_{idx}_`) on the wire.
    fn name(self) -> String {
        match self {
            BodyScalar::Mass => "mass".to_string(),
            BodyScalar::Soft => "soft".to_string(),
            BodyScalar::SoftKind => "softkind".to_string(),
            BodyScalar::Racc => "racc".to_string(),
            BodyScalar::Rmask => "rmask".to_string(),
            BodyScalar::Sink => "sink".to_string(),
            BodyScalar::Delta => "delta".to_string(),
            BodyScalar::Pos(ax) => format!("pos_{ax}"),
            BodyScalar::Vel(ax) => format!("vel_{ax}"),
            BodyScalar::Omega(k) => format!("omega_{k}"),
            BodyScalar::Rot(k) => format!("rot_{k}"),
        }
    }

    fn parse(field: &str) -> Option<Self> {
        if let Some(ax) = field.strip_prefix("pos_") {
            return ax.parse().ok().map(BodyScalar::Pos);
        }
        if let Some(ax) = field.strip_prefix("vel_") {
            return ax.parse().ok().map(BodyScalar::Vel);
        }
        if let Some(k) = field.strip_prefix("omega_") {
            return k.parse().ok().map(BodyScalar::Omega);
        }
        if let Some(k) = field.strip_prefix("rot_") {
            return k.parse().ok().map(BodyScalar::Rot);
        }
        match field {
            "mass" => Some(BodyScalar::Mass),
            "soft" => Some(BodyScalar::Soft),
            "softkind" => Some(BodyScalar::SoftKind),
            "racc" => Some(BodyScalar::Racc),
            "rmask" => Some(BodyScalar::Rmask),
            "sink" => Some(BodyScalar::Sink),
            "delta" => Some(BodyScalar::Delta),
            _ => None,
        }
    }
}

/// a typed kernel scalar-parameter name over the closed dispatch vocabulary.
/// every variant round-trips through `name()`/`parse()`; `parse()` returns `None`
/// for an open spec/user-source knob (the caller resolves those by string).
///
/// the kernel manifest carries the int/float sort (`is_int`) alongside the ref;
/// the ghost-fill `MapType`/`Arg` variants are the int-lane members and the
/// manifest tags them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum ScalarRef {
    /// ideal-gas adiabatic index `gamma` (the energy-regime EOS param).
    Gamma,
    /// isothermal sound speed `cs` (the iso EOS param — the dual of `Gamma`).
    Cs,
    /// theta-MC limiter free parameter `theta`.
    Theta,
    /// the Schwarzschild geometric mass `M` (the lapse `alpha = sqrt(1 - 2M/r)` parameter). a
    /// spacetime dispatch scalar, resolved from the metric at the call site (like `gamma`).
    SchwarzschildMass,
    /// the Kerr specific angular momentum `a = J/M` (the spinning kerr-schild metric parameter,
    /// Sigma = r^2 + a^2 cos^2(theta)). a spacetime dispatch scalar, resolved from the metric.
    KerrSpin,
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
    /// the per-axis spacing kind `map_kind_{ax}` — the runtime selector the in-kernel face-position
    /// map branches on: 0 = uniform (`face = x_lo + i*dx`), 1 = log (`face = x_lo * 10^(i*dx)`).
    /// makes spacing a per-axis runtime value (log-r, log-theta, ...),
    /// so one kernel per (regime, geometry) serves every spacing; a moving mesh updates the
    /// `x_lo`/`dx` scalars on the fly while `map_kind` stays fixed.
    MapKind(u8),
    /// the per-axis coordinate-map parameter `map_param_{ax}`. geometric grading uses the
    /// adjacent-cell width ratio; uniform and logarithmic maps bind zero.
    MapParam(u8),
    /// a moving-mesh rate (`mesh_hdil`, `mesh_adot_{ax}`, `mesh_vtrans_{ax}`).
    Mesh(MeshScalar),
    /// a user-source tunable knob `p{i}`.
    UserParam(u32),
    /// an immersed-body scalar `body_{idx}_{field}`.
    Body { idx: u8, field: BodyScalar },
    /// the ghost-fill per-axis lattice-map kind `map_type_{ax}` (int lane).
    MapType(u8),
    /// the ghost-fill per-axis lattice-map argument `arg_{ax}` (int lane).
    Arg(u8),
    /// the ghost-fill per-axis velocity-flip sign `vel_sign_{ax}`.
    VelSign(u8),
}

impl ScalarRef {
    /// the sole place a closed-vocabulary scalar wire name is minted. holds
    /// `parse(x.name()) == Some(x)` for every variant.
    pub fn name(self) -> String {
        match self {
            ScalarRef::Gamma => "gamma".to_string(),
            ScalarRef::Cs => "cs".to_string(),
            ScalarRef::Theta => "theta".to_string(),
            ScalarRef::SchwarzschildMass => "schwarzschild_mass".to_string(),
            ScalarRef::KerrSpin => "kerr_spin".to_string(),
            ScalarRef::Dt => "dt".to_string(),
            ScalarRef::A0 => "a0".to_string(),
            ScalarRef::Ac => "ac".to_string(),
            ScalarRef::Time => "t".to_string(),
            ScalarRef::InvDx(ax) => format!("inv_dx_{ax}"),
            ScalarRef::XLo(ax) => format!("x_lo_{ax}"),
            ScalarRef::Dx(ax) => format!("dx_{ax}"),
            ScalarRef::MapKind(ax) => format!("map_kind_{ax}"),
            ScalarRef::MapParam(ax) => format!("map_param_{ax}"),
            ScalarRef::Mesh(m) => m.name(),
            ScalarRef::UserParam(i) => format!("p{i}"),
            ScalarRef::Body { idx, field } => format!("body_{idx}_{}", field.name()),
            ScalarRef::MapType(ax) => format!("map_type_{ax}"),
            ScalarRef::Arg(ax) => format!("arg_{ax}"),
            ScalarRef::VelSign(ax) => format!("vel_sign_{ax}"),
        }
    }

    /// the inverse of `name`: recover the typed scalar from a wire name, or `None`
    /// when the name is an open spec/user-source knob (resolved by string at the
    /// call site). holds `parse(x.name()) == Some(x)` for every variant.
    pub fn parse(name: &str) -> Option<Self> {
        // bare-name singletons.
        match name {
            "gamma" => return Some(ScalarRef::Gamma),
            "cs" => return Some(ScalarRef::Cs),
            "theta" => return Some(ScalarRef::Theta),
            "schwarzschild_mass" => return Some(ScalarRef::SchwarzschildMass),
            "kerr_spin" => return Some(ScalarRef::KerrSpin),
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

        // per-axis geometry. `inv_dx_` must be tried before `dx_` (prefix overlap); `map_kind_`
        // before `map_type_` share only the `map_` stem (exact strip, no overlap).
        if let Some(ax) = name.strip_prefix("inv_dx_") {
            return ax.parse().ok().map(ScalarRef::InvDx);
        }
        if let Some(ax) = name.strip_prefix("x_lo_") {
            return ax.parse().ok().map(ScalarRef::XLo);
        }
        if let Some(ax) = name.strip_prefix("map_kind_") {
            return ax.parse().ok().map(ScalarRef::MapKind);
        }
        if let Some(ax) = name.strip_prefix("map_param_") {
            return ax.parse().ok().map(ScalarRef::MapParam);
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

        // user-source knob `p{i}`, tried last: only an all-digit suffix after `p`
        // binds here, so `pre`/`pos` (non-digit suffixes) parse to `None`.
        if let Some(i) = name.strip_prefix('p')
            && !i.is_empty()
            && i.bytes().all(|b| b.is_ascii_digit())
        {
            return i.parse().ok().map(ScalarRef::UserParam);
        }

        None
    }
}

/// a serialized kernel scalar binding: the typed core (`Ref`) over the closed
/// dispatch vocabulary, plus an open tail (`Spec`) for the spec/user-source knobs
/// (`gm`, `g_ext_0`, `xm_1`, ...) that fall outside the fixed vocabulary and resolve
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
        ScalarRef::parse(s)
            .map(ScalarBind::Ref)
            .unwrap_or_else(|| ScalarBind::Spec(s.into()))
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
                BodyScalar::SoftKind,
                BodyScalar::Racc,
                BodyScalar::Rmask,
                BodyScalar::Sink,
                BodyScalar::Delta,
            ] {
                v.push(ScalarRef::Body { idx, field });
            }
            for ax in 0..3u8 {
                v.push(ScalarRef::Body {
                    idx,
                    field: BodyScalar::Pos(ax),
                });
                v.push(ScalarRef::Body {
                    idx,
                    field: BodyScalar::Vel(ax),
                });
            }
        }
        v
    }

    // the invariant the whole module exists for: name() and parse() are exact
    // inverses over every representable variant, so a producer and a consumer that
    // both route through ScalarRef agree on a name.
    #[test]
    fn name_parse_round_trips() {
        for r in all_variants() {
            assert_eq!(
                ScalarRef::parse(&r.name()),
                Some(r),
                "round-trip failed for {r:?}"
            );
        }
    }

    // the mesh family re-uses MeshScalar — confirm the bridge holds in both directions.
    #[test]
    fn mesh_family_bridges() {
        assert_eq!(
            ScalarRef::parse("mesh_hdil"),
            Some(ScalarRef::Mesh(MeshScalar::Hdil))
        );
        assert_eq!(
            ScalarRef::parse("mesh_adot_1"),
            Some(ScalarRef::Mesh(MeshScalar::Adot(1)))
        );
        assert_eq!(
            ScalarRef::parse("mesh_vtrans_0"),
            Some(ScalarRef::Mesh(MeshScalar::Vtrans(0)))
        );
    }

    // spec/user-source knobs live outside the closed vocabulary: parse returns
    // None so the caller falls through to the spec string map. this is the
    // documented boundary.
    #[test]
    fn rejects_open_spec_scalars() {
        for n in [
            "gm",
            "g_ext_0",
            "xm_1",
            "body_radius",
            "value",
            "scale",
            "alpha",
            "cs2",
        ] {
            assert_eq!(
                ScalarRef::parse(n),
                None,
                "'{n}' must stay an open spec scalar"
            );
        }
    }

    // per-cell interpreter vars (resolve_runtime_param) resolve through a
    // separate path from closed kernel scalars, except `t` / `p{i}`, which are
    // shared names — the call site decides which path.
    #[test]
    fn rejects_per_cell_vars() {
        for n in ["rho", "pre", "vel_0", "x_2", "pos", "p"] {
            assert_eq!(
                ScalarRef::parse(n),
                None,
                "'{n}' is not a closed kernel scalar"
            );
        }
    }

    // a closed-vocabulary name binds typed (`Ref`); an open spec/user knob is held
    // verbatim (`Spec`). both round-trip through name() exactly — the scalar analog
    // of the FieldBind classification gate.
    #[test]
    fn scalar_bind_classifies_and_round_trips() {
        for r in all_variants() {
            let bind = ScalarBind::from_name(&r.name());
            assert_eq!(
                bind,
                ScalarBind::Ref(r),
                "closed name should bind Ref for {r:?}"
            );
            assert_eq!(bind.name(), r.name(), "Ref round-trip failed for {r:?}");
        }
        // spec/user-source knobs live outside the closed vocabulary — they round-trip as Spec.
        for spec in ["gm", "g_ext_0", "xm_1", "value", "scale", "alpha", "cs2"] {
            let bind = ScalarBind::from_name(spec);
            assert_eq!(
                bind,
                ScalarBind::Spec(spec.into()),
                "'{spec}' should bind Spec"
            );
            assert_eq!(bind.name(), spec, "Spec round-trip failed for '{spec}'");
        }
    }
}
