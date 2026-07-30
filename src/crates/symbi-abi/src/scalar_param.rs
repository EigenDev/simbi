// =============================================================================
// scalar_param.rs
//
// typed names for kernel scalar parameters that cross the trace -> dispatch ABI.
//
// a kernel scalar is the link between the traced graph (which DECLARES it via
// `Gv::scalar(name)`) and the host dispatch (which SUPPLIES its value by name).
// that link is a bare string — and a string minted independently on each side
// drifts: the moving-mesh rate was `mesh_adot` in the flux trace but
// `mesh_adot_{axis}` in the wave-speed trace and the dispatch resolver, which
// silently dark-failed the flux carrier oracle.
//
// these types mint each wire name in EXACTLY one place. `name()` is the only
// `format!`; `parse()` is its inverse; the round-trip is property-tested. a
// producer and a consumer that both go through `MeshScalar` cannot disagree on
// a name, and adding a variant is a COMPILE error until every `match` covers it.
//
// usage:
//  // producer (trace): Gv::scalar(&MeshScalar::Adot(dir).name())
//  // consumer (dispatch): match MeshScalar::parse(name) { Some(MeshScalar::Adot(ax)) => .. }
// =============================================================================

/// a moving-mesh grid-velocity scalar parameter, addressed per axis. the grid
/// velocity a kernel applies is `vface = adot * x + vtrans`; `Hdil` is the
/// physical volume-dilution rate (axis-independent). every kernel that moves
/// with the mesh — flux, wave-speed/cfl, godunov — declares these, and the
/// host `motion_scalar` resolver supplies them; both sides name them through
/// this type so the per-axis convention is the ONLY convention.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum MeshScalar {
    /// homologous expansion (hubble) rate on `axis` — `a_dot / a` on expanding
    /// axes, zero otherwise.
    Adot(u8),
    /// uniform translation rate on `axis` — `a_dot` on axis 0, zero otherwise.
    Vtrans(u8),
    /// physical volume dilution rate `p * H` (zero for translation).
    Hdil,
}

impl MeshScalar {
    /// the ONE place a mesh-scalar wire name is minted.
    pub fn name(self) -> String {
        match self {
            MeshScalar::Adot(a) => format!("mesh_adot_{a}"),
            MeshScalar::Vtrans(a) => format!("mesh_vtrans_{a}"),
            MeshScalar::Hdil => "mesh_hdil".to_string(),
        }
    }

    /// the inverse of `name`: recover the typed scalar from a wire name, or
    /// `None` when `name` is not a mesh scalar. holds `parse(x.name()) == Some(x)`.
    pub fn parse(name: &str) -> Option<Self> {
        if name == "mesh_hdil" {
            return Some(MeshScalar::Hdil);
        }
        if let Some(a) = name.strip_prefix("mesh_adot_") {
            return a.parse().ok().map(MeshScalar::Adot);
        }
        if let Some(a) = name.strip_prefix("mesh_vtrans_") {
            return a.parse().ok().map(MeshScalar::Vtrans);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // the single invariant the whole module exists to guarantee: name() and
    // parse() are exact inverses over every representable variant, so a producer
    // and a consumer that both route through MeshScalar can never diverge.
    #[test]
    fn name_parse_round_trips() {
        let cases = (0..8u8)
            .flat_map(|a| [MeshScalar::Adot(a), MeshScalar::Vtrans(a)])
            .chain([MeshScalar::Hdil]);
        for s in cases {
            assert_eq!(
                MeshScalar::parse(&s.name()),
                Some(s),
                "round-trip failed for {s:?}"
            );
        }
    }

    #[test]
    fn parse_rejects_foreign_names() {
        for n in [
            "gamma",
            "theta",
            "inv_dx_0",
            "x_lo_1",
            "mesh_adot",
            "mesh_adotx",
            "",
        ] {
            assert_eq!(
                MeshScalar::parse(n),
                None,
                "'{n}' should not parse as a mesh scalar"
            );
        }
    }
}
