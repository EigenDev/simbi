// =============================================================================
// field_ref.rs
//
// typed names for the kernel field buffers that cross the trace -> dispatch ABI.
//
// a kernel field is the link between the traced graph (which declares it via
// `Gv::field(key, runtime_path)`) and the host dispatch (which binds the backing
// buffer by that runtime path). that link is a bare dotted string — and the same
// buffer is spelled three ways across five layers (`ir_key` `cons_mom_k`,
// `runtime_path` `cons.mom_k`, source short name `mom`) with nothing forcing them
// to agree. aliases make it worse: `flux.den` and `mass_flux[d]` name the same
// buffer under two spellings. a string minted on one side and matched on the
// other drifts silently — the exact failure mode `MeshScalar` was built to kill.
//
// `FieldRef` mints each field's canonical wire name in exactly one place
// (`name()`), and `parse()` recovers the typed ref from a runtime path. the
// consumer (`resolve_path`) parses once at the decode boundary and matches
// exhaustively — adding a field is then a compile error until every match covers
// it. the `GvKernel` manifest stays string-typed; only the decode chokepoint
// goes typed.
//
// usage:
//  // consumer (dispatch): match FieldRef::parse(path)? {
//  //     FieldRef::State { slot: StateSlot::Cons, comp: StateComp::Den } => ..,
//  //     FieldRef::MassFlux(ax) => ..,  // alias of flux[ax].den
//  // }
// =============================================================================

/// one of the four conserved/flux buffer families that share the same component
/// shape `{den, nrg, mom_k}`. `Cons` is the live conserved state, `UN`/`UStage`
/// the rk stage snapshots, `Flux` the per-direction interface flux (the active
/// direction is supplied out-of-band as `dir`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum StateSlot {
    Cons,
    UN,
    UStage,
    Flux,
}

impl StateSlot {
    /// the dotted prefix that names this slot on the wire (`cons` / `u_n` / ...).
    fn prefix(self) -> &'static str {
        match self {
            StateSlot::Cons => "cons",
            StateSlot::UN => "u_n",
            StateSlot::UStage => "u_stage",
            StateSlot::Flux => "flux",
        }
    }

    fn parse_prefix(s: &str) -> Option<Self> {
        match s {
            "cons" => Some(StateSlot::Cons),
            "u_n" => Some(StateSlot::UN),
            "u_stage" => Some(StateSlot::UStage),
            "flux" => Some(StateSlot::Flux),
            _ => None,
        }
    }
}

/// a component of a conserved/flux slot: scalar density, scalar energy, or a
/// momentum component addressed per axis. the momentum axis is read off the path
/// so `DOF != NDIM` needs no special-casing.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum StateComp {
    Den,
    Nrg,
    Mom(u8),
    /// the conserved passive scalar `D_chi = rho * chi` (dye). optional per run:
    /// the backing field is `None` when the config declares no passive scalar.
    Chi,
}

/// a typed kernel field-buffer name. every variant round-trips through
/// `name()`/`parse()`; `parse()` also accepts the secondary index spellings the
/// producers emit (`_k` vs `[k]`) so a single ref absorbs both — `name()` mints
/// the one canonical spelling, the round-trip is pinned on it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum FieldRef {
    /// primitive density `prim.rho`.
    PrimRho,
    /// primitive pressure `prim.pre` (supplied as the regime pressure override at
    /// the dispatch site, its value coming from the regime).
    PrimPre,
    /// primitive velocity component `prim.vel[k]`.
    PrimVel(u8),
    /// primitive (cell-centered) magnetic field component `prim.mag[k]` == the mhd
    /// `bcell[k]`.
    PrimMag(u8),
    /// primitive passive-scalar concentration `prim.chi` = `cons.chi / cons.den`.
    PrimChi,
    /// a conserved/flux slot component (`cons.den`, `flux.mom_2`, ...).
    State {
        slot: StateSlot,
        comp: StateComp,
    },
    /// flux-divergence alias for the mass flux in grid direction `ax`
    /// (`mass_flux[ax]` == `flux[ax].den`).
    MassFlux(u8),
    /// flux-divergence alias for the energy flux in grid direction `ax`
    /// (`nrg_flux[ax]` == `flux[ax].nrg`).
    NrgFlux(u8),
    /// flux-divergence alias for the dye flux in grid direction `ax`
    /// (`chi_flux[ax]` == `flux[ax].chi`). the dye flux is `mass_flux * chi` with the
    /// concentration upwinded on the sign of that mass flux; storing it at the interface is what
    /// lets a coarse-fine reflux correct the dye the same way it corrects mass.
    ChiFlux(u8),
    /// flux-divergence alias for momentum component `comp` swept in grid direction
    /// `axis` (`mom_flux_{comp}[axis]` == `flux[axis].mom[comp]`).
    MomFlux {
        comp: u8,
        axis: u8,
    },
    /// cell-centered B component `bc_{c}` (the fused god+bcell kernel's in-place
    /// cell-B i/o) == the mhd `bcell[c]`.
    BCell(u8),
    /// rk2 stage-1 cell-B snapshot `bcn_{c}` == the mhd `bcell_n[c]`.
    BCellN(u8),
    /// induction flux `bf_{d}_{c}` = the cell-B flux in grid direction `d` of
    /// B-component `c` == the mhd `bflux[d][c]`.
    BFlux {
        dir: u8,
        comp: u8,
    },
    /// the cfl wave-speed scratch buffer, spelled `scratch`.
    Scratch,
    /// the cfl wave-speed scratch buffer, spelled `c` (an alias of `Scratch` —
    /// distinct wire name, same backing buffer).
    ScratchC,
    /// the staggered sweep-normal face B (`bface_n`) == the mhd `bface[dir]` for the dispatch's
    /// sweep direction. its allocated domain differs from the cell fields (Gardiner-Stone CT
    /// coupling); the per-buffer dispatch layout (`Field::domain()`) binds it like any cell field.
    BFaceNormal,
    /// per-axis left/right wave-speed scratch (`wave_speed_l[k]` / `wave_speed_r[k]`) == the mhd
    /// `wave_speed_l[k]` / `wave_speed_r[k]` — materialized by the RMHD quartic pass, read by its
    /// HLLE flux (the Davis fan).
    WaveSpeedL(u8),
    WaveSpeedR(u8),
    /// the conserved magnetic component `cons.mag_{k}` — in ideal MHD the magnetic field is its
    /// own (cell-centered) conserved variable, so this is the same buffer as the mhd `bcell[k]`
    /// (the c2p reads it as the conserved input). a distinct wire name for the same field, like
    /// `PrimMag`/`BCell`.
    ConsMag(u8),
    /// the magnetic flux component `flux.mag_{k}` in the sweep direction — the induction flux ==
    /// the mhd `bflux[dir][k]` for the dispatch's `dir` (the standalone face-flux spelling of the
    /// fused god+bcell's `bf_{d}_{c}`).
    FluxMag(u8),
}

impl FieldRef {
    /// true for a field derived from the current primitives within a stage, whose meaning is
    /// confined to that stage: the intercell fluxes (gas, induction, passive scalar) and the
    /// per-cell Riemann wave speeds. every such field is produced by an earlier pass of the same
    /// stage and consumed by a later one, so a read that finds no producer is a defect regardless
    /// of what the buffer happens to hold.
    ///
    /// the distinction matters because it separates the two ways a buffer can be read. the
    /// conserved state, the primitives, the staggered face field and the stage snapshots all
    /// persist — a pass may legitimately read what an earlier stage or the initial condition left
    /// there, so whether this stage wrote it says nothing about their validity. these fields hold
    /// meaning only within the stage that writes them: an unwritten one holds its zero
    /// initialization, and zero wave speeds silently collapse an HLL fan onto the shift, leaving
    /// a one-sided sweep with no dissipation on any axis whose shift component vanishes.
    pub fn is_stage_local(&self) -> bool {
        matches!(
            self,
            FieldRef::State {
                slot: StateSlot::Flux,
                ..
            } | FieldRef::MassFlux(_)
                | FieldRef::NrgFlux(_)
                | FieldRef::ChiFlux(_)
                | FieldRef::MomFlux { .. }
                | FieldRef::BFlux { .. }
                | FieldRef::FluxMag(_)
                | FieldRef::WaveSpeedL(_)
                | FieldRef::WaveSpeedR(_)
        )
    }

    // concise constructors for the conserved/flux slot family — the producers mint these
    // as typed values, so a mistyped binding is a compile
    // error caught at build time. they wrap `State { slot, comp }` so the call sites stay readable
    // and need no `StateSlot`/`StateComp` in scope.
    pub const fn cons_den() -> Self {
        Self::State {
            slot: StateSlot::Cons,
            comp: StateComp::Den,
        }
    }
    pub const fn cons_nrg() -> Self {
        Self::State {
            slot: StateSlot::Cons,
            comp: StateComp::Nrg,
        }
    }
    pub const fn cons_mom(k: u8) -> Self {
        Self::State {
            slot: StateSlot::Cons,
            comp: StateComp::Mom(k),
        }
    }
    pub const fn un_den() -> Self {
        Self::State {
            slot: StateSlot::UN,
            comp: StateComp::Den,
        }
    }
    pub const fn un_nrg() -> Self {
        Self::State {
            slot: StateSlot::UN,
            comp: StateComp::Nrg,
        }
    }
    pub const fn un_mom(k: u8) -> Self {
        Self::State {
            slot: StateSlot::UN,
            comp: StateComp::Mom(k),
        }
    }
    pub const fn ustage_den() -> Self {
        Self::State {
            slot: StateSlot::UStage,
            comp: StateComp::Den,
        }
    }
    pub const fn ustage_nrg() -> Self {
        Self::State {
            slot: StateSlot::UStage,
            comp: StateComp::Nrg,
        }
    }
    pub const fn ustage_mom(k: u8) -> Self {
        Self::State {
            slot: StateSlot::UStage,
            comp: StateComp::Mom(k),
        }
    }
    pub const fn cons_chi() -> Self {
        Self::State {
            slot: StateSlot::Cons,
            comp: StateComp::Chi,
        }
    }
    pub const fn un_chi() -> Self {
        Self::State {
            slot: StateSlot::UN,
            comp: StateComp::Chi,
        }
    }
    pub const fn flux_den() -> Self {
        Self::State {
            slot: StateSlot::Flux,
            comp: StateComp::Den,
        }
    }
    pub const fn flux_nrg() -> Self {
        Self::State {
            slot: StateSlot::Flux,
            comp: StateComp::Nrg,
        }
    }
    pub const fn flux_mom(k: u8) -> Self {
        Self::State {
            slot: StateSlot::Flux,
            comp: StateComp::Mom(k),
        }
    }

    /// the sole place a field runtime-path is minted. holds `parse(x.name()) ==
    /// Some(x)` for every variant.
    pub fn name(self) -> String {
        match self {
            FieldRef::PrimRho => "prim.rho".to_string(),
            FieldRef::PrimPre => "prim.pre".to_string(),
            FieldRef::PrimVel(k) => format!("prim.vel[{k}]"),
            FieldRef::PrimMag(k) => format!("prim.mag[{k}]"),
            FieldRef::PrimChi => "prim.chi".to_string(),
            FieldRef::State { slot, comp } => match comp {
                StateComp::Den => format!("{}.den", slot.prefix()),
                StateComp::Nrg => format!("{}.nrg", slot.prefix()),
                StateComp::Mom(k) => format!("{}.mom_{k}", slot.prefix()),
                StateComp::Chi => format!("{}.chi", slot.prefix()),
            },
            FieldRef::MassFlux(ax) => format!("mass_flux[{ax}]"),
            FieldRef::NrgFlux(ax) => format!("nrg_flux[{ax}]"),
            FieldRef::ChiFlux(ax) => format!("chi_flux[{ax}]"),
            FieldRef::MomFlux { comp, axis } => format!("mom_flux_{comp}[{axis}]"),
            FieldRef::BCell(c) => format!("bc_{c}"),
            FieldRef::BCellN(c) => format!("bcn_{c}"),
            FieldRef::BFlux { dir, comp } => format!("bf_{dir}_{comp}"),
            FieldRef::Scratch => "scratch".to_string(),
            FieldRef::ScratchC => "c".to_string(),
            FieldRef::BFaceNormal => "bface_n".to_string(),
            FieldRef::WaveSpeedL(k) => format!("wave_speed_l[{k}]"),
            FieldRef::WaveSpeedR(k) => format!("wave_speed_r[{k}]"),
            FieldRef::ConsMag(k) => format!("cons.mag_{k}"),
            FieldRef::FluxMag(k) => format!("flux.mag_{k}"),
        }
    }

    /// the inverse of `name`: recover the typed field from a runtime path, returning
    /// `None` for a path outside the known field vocabulary. accepts both index
    /// spellings (`prim.vel_2` and `prim.vel[2]`) so the c2p-write and
    /// reconstruction-read forms of the same buffer parse to the same ref.
    pub fn parse(path: &str) -> Option<Self> {
        match path {
            "prim.rho" => return Some(FieldRef::PrimRho),
            "prim.pre" => return Some(FieldRef::PrimPre),
            "prim.chi" => return Some(FieldRef::PrimChi),
            "scratch" => return Some(FieldRef::Scratch),
            "c" => return Some(FieldRef::ScratchC),
            "bface_n" => return Some(FieldRef::BFaceNormal),
            _ => {}
        }

        // staggered per-axis wave-speed scratch (the RMHD quartic materialization).
        if let Some(r) = path.strip_prefix("wave_speed_l") {
            return parse_idx(r).map(FieldRef::WaveSpeedL);
        }
        if let Some(r) = path.strip_prefix("wave_speed_r") {
            return parse_idx(r).map(FieldRef::WaveSpeedR);
        }

        // the magnetic conserved / flux components (mhd `bcell` / `bflux`) — intercepted before
        // the generic `{slot}.{comp}` parse, which has no `mag` component.
        if let Some(r) = path.strip_prefix("cons.mag_") {
            return r.parse().ok().map(FieldRef::ConsMag);
        }
        if let Some(r) = path.strip_prefix("flux.mag_") {
            return r.parse().ok().map(FieldRef::FluxMag);
        }
        // `mhd.bcell[k]` — the iso lattice-map ghost-fill's spelling of the cell B; an alias of
        // the canonical `BCell(k)` (which mints `bc_{k}`), like `prim.vel_k` aliases `prim.vel[k]`.
        if let Some(r) = path.strip_prefix("mhd.bcell") {
            return parse_idx(r).map(FieldRef::BCell);
        }

        if let Some(r) = path.strip_prefix("prim.vel") {
            return parse_idx(r).map(FieldRef::PrimVel);
        }
        if let Some(r) = path.strip_prefix("prim.mag") {
            return parse_idx(r).map(FieldRef::PrimMag);
        }

        // flux-divergence aliases. distinct prefixes from the `flux.` slot mean any
        // check order here is correct; these stay grouped for readability.
        if let Some(r) = path.strip_prefix("mass_flux") {
            return parse_idx(r).map(FieldRef::MassFlux);
        }
        if let Some(r) = path.strip_prefix("nrg_flux") {
            return parse_idx(r).map(FieldRef::NrgFlux);
        }
        if let Some(r) = path.strip_prefix("chi_flux") {
            return parse_idx(r).map(FieldRef::ChiFlux);
        }
        if let Some(r) = path.strip_prefix("mom_flux_") {
            // "{comp}[{axis}]".
            let (comp, axis) = r.split_once('[')?;
            return Some(FieldRef::MomFlux {
                comp: comp.parse().ok()?,
                axis: axis.strip_suffix(']')?.parse().ok()?,
            });
        }

        // cell-B family. `bcn_` must be tried before `bc_` (prefix overlap).
        if let Some(r) = path.strip_prefix("bcn_") {
            return r.parse().ok().map(FieldRef::BCellN);
        }
        if let Some(r) = path.strip_prefix("bf_") {
            let (dir, comp) = r.split_once('_')?;
            return Some(FieldRef::BFlux {
                dir: dir.parse().ok()?,
                comp: comp.parse().ok()?,
            });
        }
        if let Some(r) = path.strip_prefix("bc_") {
            return r.parse().ok().map(FieldRef::BCell);
        }

        // the conserved/flux slot family: "{slot}.{comp}".
        let (slot_str, comp_str) = path.split_once('.')?;
        let slot = StateSlot::parse_prefix(slot_str)?;
        let comp = match comp_str {
            "den" => StateComp::Den,
            "nrg" => StateComp::Nrg,
            "chi" => StateComp::Chi,
            _ => StateComp::Mom(comp_str.strip_prefix("mom_")?.parse().ok()?),
        };
        Some(FieldRef::State { slot, comp })
    }
}

/// parse a trailing index fragment in either spelling: `[2]`, `_2`, or `2`.
fn parse_idx(s: &str) -> Option<u8> {
    s.trim_matches(|c| c == '[' || c == ']' || c == '_')
        .parse()
        .ok()
}

/// a serialized kernel field binding: the typed core (`Ref`) over the closed
/// cell-centered vocabulary, plus an open tail (`Raw`) for the hand-built
/// staggered/ct/geom/refinement kernels whose paths (`area_hi_0`, `bcell_p1`,
/// `bface*`, edge fields, the reduction scratch `buf0`, ...) fall outside
/// `FieldRef` and bind positionally by buffer index — they round-trip losslessly
/// as the raw string. this is the field analog of `ScalarBind` (typed core + open
/// spec tail): the manifest is born typed for the dispatched cell-centered
/// kernels, and hand-built kernels keep their raw spelling, outside `FieldRef`'s
/// scope. the metadata-driven typed dispatch only ever sees `Ref`; a `Raw`
/// reaching that path is a loud bug — hand-built kernels route through their own
/// raw-string path only.
#[derive(Clone, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum FieldBind {
    Ref(FieldRef),
    Raw(Box<str>),
}

impl FieldBind {
    /// classify a runtime path: a known closed-vocabulary field becomes `Ref`, any
    /// other path is held verbatim as `Raw`. the inverse of `name()`.
    pub fn from_path(s: &str) -> Self {
        FieldRef::parse(s)
            .map(FieldBind::Ref)
            .unwrap_or_else(|| FieldBind::Raw(s.into()))
    }

    /// the runtime path this bind names. `Ref` mints its canonical spelling through
    /// `FieldRef::name`; `Raw` returns its stored string unchanged.
    pub fn name(&self) -> String {
        match self {
            FieldBind::Ref(f) => f.name(),
            FieldBind::Raw(s) => s.to_string(),
        }
    }
}

/// a typed field ref is a `FieldBind::Ref` — the producer's born-typed binding.
impl From<FieldRef> for FieldBind {
    fn from(r: FieldRef) -> Self {
        FieldBind::Ref(r)
    }
}

/// a raw runtime path classifies through `from_path` (closed vocab -> `Ref`, else `Raw`).
/// a producer passing a `&str` path is classified here at construction, so the manifest is
/// born-typed straight from the bare string the call site already has. an unknown path lands
/// in `Raw` (the open-vocabulary hand-built kernels).
impl From<&str> for FieldBind {
    fn from(s: &str) -> Self {
        FieldBind::from_path(s)
    }
}

impl From<String> for FieldBind {
    fn from(s: String) -> Self {
        FieldBind::from_path(&s)
    }
}

impl From<&String> for FieldBind {
    fn from(s: &String) -> Self {
        FieldBind::from_path(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_variants() -> Vec<FieldRef> {
        let mut v = vec![
            FieldRef::PrimRho,
            FieldRef::PrimPre,
            FieldRef::PrimChi,
            FieldRef::Scratch,
            FieldRef::ScratchC,
            FieldRef::BFaceNormal,
        ];
        for k in 0..3u8 {
            v.push(FieldRef::PrimVel(k));
            v.push(FieldRef::PrimMag(k));
            v.push(FieldRef::MassFlux(k));
            v.push(FieldRef::NrgFlux(k));
            v.push(FieldRef::ChiFlux(k));
            v.push(FieldRef::BCell(k));
            v.push(FieldRef::BCellN(k));
            v.push(FieldRef::WaveSpeedL(k));
            v.push(FieldRef::WaveSpeedR(k));
            v.push(FieldRef::ConsMag(k));
            v.push(FieldRef::FluxMag(k));
        }
        for slot in [
            StateSlot::Cons,
            StateSlot::UN,
            StateSlot::UStage,
            StateSlot::Flux,
        ] {
            v.push(FieldRef::State {
                slot,
                comp: StateComp::Den,
            });
            v.push(FieldRef::State {
                slot,
                comp: StateComp::Nrg,
            });
            v.push(FieldRef::State {
                slot,
                comp: StateComp::Chi,
            });
            for k in 0..3u8 {
                v.push(FieldRef::State {
                    slot,
                    comp: StateComp::Mom(k),
                });
            }
        }
        for comp in 0..3u8 {
            for axis in 0..3u8 {
                v.push(FieldRef::MomFlux { comp, axis });
                v.push(FieldRef::BFlux { dir: axis, comp });
            }
        }
        v
    }

    // the invariant the whole module exists for: name() and parse() are exact
    // inverses over every representable variant, so a producer and a consumer
    // both routed through FieldRef agree on a name.
    #[test]
    fn name_parse_round_trips() {
        for r in all_variants() {
            assert_eq!(
                FieldRef::parse(&r.name()),
                Some(r),
                "round-trip failed for {r:?}"
            );
        }
    }

    // the producers emit two index spellings for the same buffer (`_k` c2p-write
    // form vs `[k]` reconstruction-read form); both must parse to the same ref.
    #[test]
    fn accepts_both_index_spellings() {
        assert_eq!(FieldRef::parse("prim.vel_2"), Some(FieldRef::PrimVel(2)));
        assert_eq!(FieldRef::parse("prim.vel[2]"), Some(FieldRef::PrimVel(2)));
        assert_eq!(FieldRef::parse("prim.mag_1"), Some(FieldRef::PrimMag(1)));
        assert_eq!(FieldRef::parse("prim.mag[1]"), Some(FieldRef::PrimMag(1)));
        assert_eq!(FieldRef::parse("mass_flux_0"), Some(FieldRef::MassFlux(0)));
        assert_eq!(FieldRef::parse("mass_flux[0]"), Some(FieldRef::MassFlux(0)));
    }

    #[test]
    fn parse_rejects_foreign_names() {
        for n in [
            "gamma",
            "prim.foo",
            "cons.bork",
            "mesh_adot_0",
            "flux",
            "",
            "bc_",
            "bf_0",
            "mom_flux_0",
        ] {
            assert_eq!(
                FieldRef::parse(n),
                None,
                "'{n}' should not parse as a field"
            );
        }
    }

    // a closed-vocabulary path binds typed (`Ref`); a hand-built / foreign path is
    // held verbatim (`Raw`). both round-trip through name() exactly.
    #[test]
    fn field_bind_classifies_and_round_trips() {
        for r in all_variants() {
            let bind = FieldBind::from_path(&r.name());
            assert_eq!(
                bind,
                FieldBind::Ref(r),
                "typed path should bind Ref for {r:?}"
            );
            assert_eq!(bind.name(), r.name(), "Ref round-trip failed for {r:?}");
        }
        // hand-built / staggered / reduction paths live outside FieldRef — they round-trip as Raw.
        for raw in [
            "area_hi_0",
            "bcell_p1",
            "bface_0_1",
            "edge_e2",
            "buf0",
            "prim.foo",
        ] {
            let bind = FieldBind::from_path(raw);
            assert_eq!(bind, FieldBind::Raw(raw.into()), "'{raw}' should bind Raw");
            assert_eq!(bind.name(), raw, "Raw round-trip failed for '{raw}'");
        }
    }
}
