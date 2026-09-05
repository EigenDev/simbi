// =============================================================================
// ct_scratch.rs
//
// typed identity for the constrained-transport scratch vocabulary: the staggered
// face/edge fields and edge-relative cell reads the CT/UCT kernels declare, and
// the substrate dispatch binds by name. each name's identity has three layers:
// - a semantic role (`CtScratch`): what the buffer is and where it lives on the
//   staggered mesh (cell / face / edge, with its edge-relative axis or
//   component role);
// - an ABI wire spelling (`CtWireName`): the exact reserved string a kernel
//   port renders as — several spellings can share one role (the edge EMF is
//   `emf`, `ez`, `e`, or `ephi` by kernel family);
// - the complete binding key (`CtScratchKey`): role + wire. equality includes
//   the wire, so distinct ports stay distinct in a manifest; centering laws and
//   the dispatch binder inspect the role alone.
//
// the relative roles (`Transverse::{A,B}` grid axes, `PlaneComp::{P1,P2,Out}`
// physical components) resolve to absolute indices only through the validated
// edge descriptor in the dispatch layer. `GridAxis<D>` and `PhysComp` keep the
// two index spaces distinct at that boundary: a grid axis is < D, a physical
// component is < 3, and they coincide only when the axis map is the identity.
//
// usage:
//  let key = CtScratchKey::canonical(CtFace::BFace(Transverse::A).into());
//  assert_eq!(key.render(), "bface_a");
//  let k2 = CtScratchKey::spelled(CtWireName::Ez);
//  assert_eq!(k2.role(), CtScratch::Edge(CtEdgeCt::Emf));
// =============================================================================

use serde::{Deserialize, Serialize};

/// one of the two transverse grid axes of a CT edge or face pair, in the
/// edge-relative frame (`a` = the first in-plane grid axis `g1`, `b` = `g2`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Transverse {
    A,
    B,
}

/// an edge-relative physical component: the two in-plane components of the
/// edge's dual plane (cyclic order fixes the EMF sign) and the out-of-plane
/// component.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum PlaneComp {
    P1,
    P2,
    Out,
}

/// a validated physical vector component of the MHD 3-space (`< 3`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PhysComp(u8);

impl PhysComp {
    /// the component index, validated against the 3-component vector space.
    pub fn new(c: usize) -> Self {
        assert!(c < 3, "physical component {c} out of the 3-vector space");
        Self(c as u8)
    }
    pub fn try_new(c: usize) -> Option<Self> {
        (c < 3).then(|| Self(c as u8))
    }
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// a validated grid axis of a `D`-dimensional mesh (`< D`). distinct from
/// `PhysComp`: on a 1.5D/2.5D grid the momentum/B vector space outruns the
/// grid, and conflating the two index spaces is the audited category error.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct GridAxis<const D: usize>(u8);

impl<const D: usize> GridAxis<D> {
    /// the axis, validated against the grid dimension; the panicking form for
    /// sites where validity is structurally guaranteed.
    pub fn new(k: usize) -> Self {
        assert!(k < D, "grid axis {k} out of the {D}-dimensional mesh");
        Self(k as u8)
    }
    pub fn try_new(k: usize) -> Option<Self> {
        (k < D).then(|| Self(k as u8))
    }
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// a validated sweep axis for the dimension-runtime kernel builders: the
/// dispatch direction of a CT curl / EMF kernel, checked against the mesh
/// dimension at construction. the const-generic `GridAxis<D>` types the same
/// space where the dimension is static; this form serves the builders whose
/// `ndim` arrives as a value.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SweepAxis(u8);

impl SweepAxis {
    pub fn new(dir: usize, ndim: usize) -> Self {
        assert!(
            dir < ndim,
            "sweep axis {dir} out of the {ndim}-dimensional mesh"
        );
        Self(dir as u8)
    }
    pub fn try_new(dir: usize, ndim: usize) -> Option<Self> {
        (dir < ndim).then(|| Self(dir as u8))
    }
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// cell-centered CT scratch roles: the edge-relative primitive/cell-B reads and
/// the FOFC troubled-cell flag.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum CtCellCt {
    BCell(PlaneComp),
    Vel(PlaneComp),
    Rho,
    Pre,
    FofcFlag,
    /// the magnetic-slip cell-quadrature vector `F_q = A(B_q)(R J)_q`, one entry per
    /// physical component: the per-cell dyad applied to the gathered current, written by
    /// the slip cell pass and scattered to the oriented edge EMF by the edge pass.
    SlipQuadrature(PhysComp),
    /// the frozen predicted midpoint gas internal energy density `e_g*` the slip coefficient's
    /// sound speed reads: a per-cell scalar the solve fills once from the predictor state, so the
    /// frozen operator's drain clock never consults the endpoint-reconciled total energy.
    SlipGasEnergy,
    /// the per-cell magnetic dissipation rate `qdot_c = (R J)_c . F_q,c >= 0` of the slip
    /// operator at the state the quadrature was formed on: the predicted heat that lifts the
    /// midpoint gas energy, and the heat the commit deposits into the total energy.
    SlipDissipation,
    /// the out-of-plane magnetic component of the operand the 2.5D slip operator acts on: a
    /// cell-centered `B_z` read by the current gather `(D_y B_z, -D_x B_z)` and written by the
    /// flux-form update `B_z -= dt (D_x F_y - D_y F_x)`. bound to the production cell field for the
    /// explicit operator and to a workspace vector when the frozen operator acts on a Krylov
    /// iterate, so the coefficient's `B_z` (`BCell`) and the operand's are distinct slots.
    SlipOperandBz,
}

/// face-centered CT scratch roles: the staggered face B (edge-relative
/// transverse, sweep-normal, or addressed by its own physical component for
/// the face-to-cell interpolation), the interface gas and induction fluxes
/// (the live first-order and saved high-order FOFC pair), and the per-face
/// left/right signal speeds. the transverse roles live on the faces normal to
/// the edge's two in-plane grid axes; the sweep roles live on the dispatch
/// direction's faces.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum CtFaceCt {
    BFace(Transverse),
    BFaceSweep,
    BFaceComp(PhysComp),
    FDen(Transverse),
    BFlux(Transverse),
    BFluxFirstOrder(PhysComp),
    BFluxHighOrder(PhysComp),
    WaveL(Transverse),
    WaveR(Transverse),
}

/// edge-centered CT scratch roles: the edge EMF (one semantic identity across
/// its ABI spellings), the two incident-edge reads of a face curl, the saved
/// high-order EMF the FOFC splice restores, and the RK2 stage snapshot.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum CtEdgeCt {
    Emf,
    EmfIncident(Transverse),
    EmfHighOrder,
    EmfSaved,
}

/// the CT scratch role: what a reserved wire name means and where its buffer
/// lives on the staggered mesh. the dispatch binder and the centering laws
/// match on this; the wire spelling is presentation.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum CtScratch {
    Cell(CtCellCt),
    Face(CtFaceCt),
    Edge(CtEdgeCt),
}

impl From<CtCellCt> for CtScratch {
    fn from(c: CtCellCt) -> Self {
        CtScratch::Cell(c)
    }
}
impl From<CtFaceCt> for CtScratch {
    fn from(f: CtFaceCt) -> Self {
        CtScratch::Face(f)
    }
}
impl From<CtEdgeCt> for CtScratch {
    fn from(e: CtEdgeCt) -> Self {
        CtScratch::Edge(e)
    }
}

/// a reserved CT wire name: the exact string a kernel port renders as on the
/// manifest. the full set is closed; `parse` recognizes these spellings alone,
/// so a derived SSA key or a typo stays outside the typed vocabulary.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum CtWireName {
    BFaceA,
    BFaceB,
    BP1,
    BP2,
    B,
    Bx,
    By,
    Br,
    Bz,
    B0,
    B1,
    BCellP1,
    BCellP2,
    BCellOut,
    VelP1,
    VelP2,
    VelOut,
    Rho,
    Pre,
    Flag,
    FDenP1,
    FDenP2,
    BFluxA,
    BFluxB,
    Bf(PhysComp),
    FoBFlux(PhysComp),
    HoBFlux(PhysComp),
    Fq(PhysComp),
    SlipGe,
    SlipQdot,
    SlipBz,
    WslP1,
    WslP2,
    WsrP1,
    WsrP2,
    Emf,
    Ez,
    E,
    Ephi,
    EP1,
    EP2,
    EFo,
    EHo,
    EN,
}

impl CtWireName {
    /// the exact ABI spelling this wire renders as. the single string mint for
    /// the CT vocabulary.
    pub fn render(self) -> String {
        match self {
            CtWireName::BFaceA => "bface_a".into(),
            CtWireName::BFaceB => "bface_b".into(),
            CtWireName::BP1 => "b_p1".into(),
            CtWireName::BP2 => "b_p2".into(),
            CtWireName::B => "b".into(),
            CtWireName::Bx => "bx".into(),
            CtWireName::By => "by".into(),
            CtWireName::Br => "br".into(),
            CtWireName::Bz => "bz".into(),
            CtWireName::B0 => "b0".into(),
            CtWireName::B1 => "b1".into(),
            CtWireName::BCellP1 => "bcell_p1".into(),
            CtWireName::BCellP2 => "bcell_p2".into(),
            CtWireName::BCellOut => "bcell_out".into(),
            CtWireName::VelP1 => "vel_p1".into(),
            CtWireName::VelP2 => "vel_p2".into(),
            CtWireName::VelOut => "vel_out".into(),
            CtWireName::Rho => "rho".into(),
            CtWireName::Pre => "pre".into(),
            CtWireName::Flag => "flag".into(),
            CtWireName::FDenP1 => "fden_p1".into(),
            CtWireName::FDenP2 => "fden_p2".into(),
            CtWireName::BFluxA => "bflux_a".into(),
            CtWireName::BFluxB => "bflux_b".into(),
            CtWireName::Bf(c) => format!("bf_{}", c.index()),
            CtWireName::FoBFlux(c) => format!("fo_bflux_{}", c.index()),
            CtWireName::HoBFlux(c) => format!("ho_bflux_{}", c.index()),
            CtWireName::Fq(c) => format!("fq_{}", c.index()),
            CtWireName::SlipGe => "slip_ge".into(),
            CtWireName::SlipQdot => "slip_qdot".into(),
            CtWireName::SlipBz => "slip_bz".into(),
            CtWireName::WslP1 => "wsl_p1".into(),
            CtWireName::WslP2 => "wsl_p2".into(),
            CtWireName::WsrP1 => "wsr_p1".into(),
            CtWireName::WsrP2 => "wsr_p2".into(),
            CtWireName::Emf => "emf".into(),
            CtWireName::Ez => "ez".into(),
            CtWireName::E => "e".into(),
            CtWireName::Ephi => "ephi".into(),
            CtWireName::EP1 => "e_p1".into(),
            CtWireName::EP2 => "e_p2".into(),
            CtWireName::EFo => "e_fo".into(),
            CtWireName::EHo => "e_ho".into(),
            CtWireName::EN => "e_n".into(),
        }
    }

    /// exact-match recognition of the reserved spellings. everything else —
    /// derived SSA keys, prefixed recon names, genuinely free scratch — returns
    /// `None` and stays untyped.
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "bface_a" => CtWireName::BFaceA,
            "bface_b" => CtWireName::BFaceB,
            "b_p1" => CtWireName::BP1,
            "b_p2" => CtWireName::BP2,
            "b" => CtWireName::B,
            "bx" => CtWireName::Bx,
            "by" => CtWireName::By,
            "br" => CtWireName::Br,
            "bz" => CtWireName::Bz,
            "b0" => CtWireName::B0,
            "b1" => CtWireName::B1,
            "bcell_p1" => CtWireName::BCellP1,
            "bcell_p2" => CtWireName::BCellP2,
            "bcell_out" => CtWireName::BCellOut,
            "vel_p1" => CtWireName::VelP1,
            "vel_p2" => CtWireName::VelP2,
            "vel_out" => CtWireName::VelOut,
            "rho" => CtWireName::Rho,
            "pre" => CtWireName::Pre,
            "flag" => CtWireName::Flag,
            "fden_p1" => CtWireName::FDenP1,
            "fden_p2" => CtWireName::FDenP2,
            "bflux_a" => CtWireName::BFluxA,
            "bflux_b" => CtWireName::BFluxB,
            "bf_0" => CtWireName::Bf(PhysComp(0)),
            "bf_1" => CtWireName::Bf(PhysComp(1)),
            "bf_2" => CtWireName::Bf(PhysComp(2)),
            "fo_bflux_0" => CtWireName::FoBFlux(PhysComp(0)),
            "fo_bflux_1" => CtWireName::FoBFlux(PhysComp(1)),
            "fo_bflux_2" => CtWireName::FoBFlux(PhysComp(2)),
            "ho_bflux_0" => CtWireName::HoBFlux(PhysComp(0)),
            "ho_bflux_1" => CtWireName::HoBFlux(PhysComp(1)),
            "ho_bflux_2" => CtWireName::HoBFlux(PhysComp(2)),
            "fq_0" => CtWireName::Fq(PhysComp(0)),
            "fq_1" => CtWireName::Fq(PhysComp(1)),
            "fq_2" => CtWireName::Fq(PhysComp(2)),
            "slip_ge" => CtWireName::SlipGe,
            "slip_qdot" => CtWireName::SlipQdot,
            "slip_bz" => CtWireName::SlipBz,
            "wsl_p1" => CtWireName::WslP1,
            "wsl_p2" => CtWireName::WslP2,
            "wsr_p1" => CtWireName::WsrP1,
            "wsr_p2" => CtWireName::WsrP2,
            "emf" => CtWireName::Emf,
            "ez" => CtWireName::Ez,
            "e" => CtWireName::E,
            "ephi" => CtWireName::Ephi,
            "e_p1" => CtWireName::EP1,
            "e_p2" => CtWireName::EP2,
            "e_fo" => CtWireName::EFo,
            "e_ho" => CtWireName::EHo,
            "e_n" => CtWireName::EN,
            _ => return None,
        })
    }

    /// the semantic role this spelling names. the 2D chart pairs (`bx`/`b0`/
    /// `br` and `by`/`b1`/`bz`) are the transverse faces at the identity axis
    /// map the 2D charts fix; the four EMF spellings are one identity.
    pub fn role(self) -> CtScratch {
        use CtWireName as W;
        match self {
            W::BFaceA | W::Bx | W::B0 | W::Br | W::BP1 => CtFaceCt::BFace(Transverse::A).into(),
            W::BFaceB | W::By | W::B1 | W::Bz | W::BP2 => CtFaceCt::BFace(Transverse::B).into(),
            W::B => CtFaceCt::BFaceSweep.into(),
            W::BCellP1 => CtCellCt::BCell(PlaneComp::P1).into(),
            W::BCellP2 => CtCellCt::BCell(PlaneComp::P2).into(),
            W::BCellOut => CtCellCt::BCell(PlaneComp::Out).into(),
            W::VelP1 => CtCellCt::Vel(PlaneComp::P1).into(),
            W::VelP2 => CtCellCt::Vel(PlaneComp::P2).into(),
            W::VelOut => CtCellCt::Vel(PlaneComp::Out).into(),
            W::Rho => CtCellCt::Rho.into(),
            W::Pre => CtCellCt::Pre.into(),
            W::Flag => CtCellCt::FofcFlag.into(),
            W::FDenP1 => CtFaceCt::FDen(Transverse::A).into(),
            W::FDenP2 => CtFaceCt::FDen(Transverse::B).into(),
            W::BFluxA => CtFaceCt::BFlux(Transverse::A).into(),
            W::BFluxB => CtFaceCt::BFlux(Transverse::B).into(),
            W::Bf(c) => CtFaceCt::BFaceComp(c).into(),
            W::FoBFlux(c) => CtFaceCt::BFluxFirstOrder(c).into(),
            W::HoBFlux(c) => CtFaceCt::BFluxHighOrder(c).into(),
            W::Fq(c) => CtCellCt::SlipQuadrature(c).into(),
            W::SlipGe => CtCellCt::SlipGasEnergy.into(),
            W::SlipQdot => CtCellCt::SlipDissipation.into(),
            W::SlipBz => CtCellCt::SlipOperandBz.into(),
            W::WslP1 => CtFaceCt::WaveL(Transverse::A).into(),
            W::WslP2 => CtFaceCt::WaveL(Transverse::B).into(),
            W::WsrP1 => CtFaceCt::WaveR(Transverse::A).into(),
            W::WsrP2 => CtFaceCt::WaveR(Transverse::B).into(),
            W::Emf | W::Ez | W::E | W::Ephi | W::EFo => CtEdgeCt::Emf.into(),
            W::EP1 => CtEdgeCt::EmfIncident(Transverse::A).into(),
            W::EP2 => CtEdgeCt::EmfIncident(Transverse::B).into(),
            W::EHo => CtEdgeCt::EmfHighOrder.into(),
            W::EN => CtEdgeCt::EmfSaved.into(),
        }
    }

    /// every reserved wire name, for the golden vocabulary pin and the
    /// structural gate.
    pub fn all() -> Vec<Self> {
        use CtWireName as W;
        let mut v = vec![
            W::BFaceA,
            W::BFaceB,
            W::BP1,
            W::BP2,
            W::B,
            W::Bx,
            W::By,
            W::Br,
            W::Bz,
            W::B0,
            W::B1,
            W::BCellP1,
            W::BCellP2,
            W::BCellOut,
            W::VelP1,
            W::VelP2,
            W::VelOut,
            W::Rho,
            W::Pre,
            W::Flag,
            W::FDenP1,
            W::FDenP2,
            W::BFluxA,
            W::BFluxB,
            W::WslP1,
            W::WslP2,
            W::WsrP1,
            W::WsrP2,
            W::Emf,
            W::Ez,
            W::E,
            W::Ephi,
            W::EP1,
            W::EP2,
            W::EFo,
            W::EHo,
            W::EN,
        ];
        for c in 0..3 {
            v.push(W::Bf(PhysComp(c)));
            v.push(W::FoBFlux(PhysComp(c)));
            v.push(W::HoBFlux(PhysComp(c)));
            v.push(W::Fq(PhysComp(c)));
        }
        v.push(W::SlipGe);
        v.push(W::SlipQdot);
        v.push(W::SlipBz);
        v
    }
}

/// the complete CT scratch binding key: the semantic role plus the ABI
/// spelling it renders as. equality includes the wire, so two ports of one
/// identity stay distinct entries in a manifest; the role is what the
/// centering laws and the dispatch binder consume.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct CtScratchKey {
    role: CtScratch,
    wire: CtWireName,
}

impl CtScratchKey {
    /// the key at a role's canonical spelling — the one the edge-EMF kernel
    /// family uses; chart-spelled ports go through `spelled`.
    pub fn canonical(role: CtScratch) -> Self {
        use CtWireName as W;
        let wire = match role {
            CtScratch::Face(CtFaceCt::BFace(Transverse::A)) => W::BFaceA,
            CtScratch::Face(CtFaceCt::BFace(Transverse::B)) => W::BFaceB,
            CtScratch::Face(CtFaceCt::BFaceSweep) => W::B,
            CtScratch::Face(CtFaceCt::FDen(Transverse::A)) => W::FDenP1,
            CtScratch::Face(CtFaceCt::FDen(Transverse::B)) => W::FDenP2,
            CtScratch::Face(CtFaceCt::BFlux(Transverse::A)) => W::BFluxA,
            CtScratch::Face(CtFaceCt::BFlux(Transverse::B)) => W::BFluxB,
            CtScratch::Face(CtFaceCt::BFaceComp(c)) => W::Bf(c),
            CtScratch::Face(CtFaceCt::BFluxFirstOrder(c)) => W::FoBFlux(c),
            CtScratch::Face(CtFaceCt::BFluxHighOrder(c)) => W::HoBFlux(c),
            CtScratch::Face(CtFaceCt::WaveL(Transverse::A)) => W::WslP1,
            CtScratch::Face(CtFaceCt::WaveL(Transverse::B)) => W::WslP2,
            CtScratch::Face(CtFaceCt::WaveR(Transverse::A)) => W::WsrP1,
            CtScratch::Face(CtFaceCt::WaveR(Transverse::B)) => W::WsrP2,
            CtScratch::Cell(CtCellCt::BCell(PlaneComp::P1)) => W::BCellP1,
            CtScratch::Cell(CtCellCt::BCell(PlaneComp::P2)) => W::BCellP2,
            CtScratch::Cell(CtCellCt::BCell(PlaneComp::Out)) => W::BCellOut,
            CtScratch::Cell(CtCellCt::Vel(PlaneComp::P1)) => W::VelP1,
            CtScratch::Cell(CtCellCt::Vel(PlaneComp::P2)) => W::VelP2,
            CtScratch::Cell(CtCellCt::Vel(PlaneComp::Out)) => W::VelOut,
            CtScratch::Cell(CtCellCt::Rho) => W::Rho,
            CtScratch::Cell(CtCellCt::Pre) => W::Pre,
            CtScratch::Cell(CtCellCt::FofcFlag) => W::Flag,
            CtScratch::Cell(CtCellCt::SlipQuadrature(c)) => W::Fq(c),
            CtScratch::Cell(CtCellCt::SlipGasEnergy) => W::SlipGe,
            CtScratch::Cell(CtCellCt::SlipDissipation) => W::SlipQdot,
            CtScratch::Cell(CtCellCt::SlipOperandBz) => W::SlipBz,
            CtScratch::Edge(CtEdgeCt::Emf) => W::Emf,
            CtScratch::Edge(CtEdgeCt::EmfIncident(Transverse::A)) => W::EP1,
            CtScratch::Edge(CtEdgeCt::EmfIncident(Transverse::B)) => W::EP2,
            CtScratch::Edge(CtEdgeCt::EmfHighOrder) => W::EHo,
            CtScratch::Edge(CtEdgeCt::EmfSaved) => W::EN,
        };
        Self { role, wire }
    }

    /// the key at an explicit ABI spelling; the role comes from the wire.
    pub fn spelled(wire: CtWireName) -> Self {
        Self {
            role: wire.role(),
            wire,
        }
    }

    pub fn role(&self) -> CtScratch {
        self.role
    }
    pub fn wire(&self) -> CtWireName {
        self.wire
    }
    pub fn render(&self) -> String {
        self.wire.render()
    }
}

impl CtScratchKey {
    /// a face key at an explicit spelling; the one validated constructor for
    /// chart-spelled face ports (`bx`/`br`/`b0`...). the wire must name the
    /// stated face role. a cell-centered role cannot enter the face door —
    ///
    /// ```compile_fail
    /// use symbi_abi::{CtCellCt, CtScratchKey, CtWireName};
    /// let _ = CtScratchKey::face(CtCellCt::Rho, CtWireName::Rho);
    /// ```
    ///
    /// and an edge EMF cannot enter the cell door —
    ///
    /// ```compile_fail
    /// use symbi_abi::{CtEdgeCt, CtScratchKey, CtWireName};
    /// let _ = CtScratchKey::cell(CtEdgeCt::Emf, CtWireName::Emf);
    /// ```
    pub fn face(f: CtFaceCt, wire: CtWireName) -> Self {
        assert_eq!(
            wire.role(),
            CtScratch::Face(f),
            "wire {wire:?} does not spell the face role {f:?}"
        );
        Self {
            role: CtScratch::Face(f),
            wire,
        }
    }
    /// a cell key at an explicit spelling; validated against the cell role.
    pub fn cell(c: CtCellCt, wire: CtWireName) -> Self {
        assert_eq!(
            wire.role(),
            CtScratch::Cell(c),
            "wire {wire:?} does not spell the cell role {c:?}"
        );
        Self {
            role: CtScratch::Cell(c),
            wire,
        }
    }
    /// an edge key at an explicit spelling; validated against the edge role.
    /// the four EMF spellings all satisfy `CtEdgeCt::Emf`.
    pub fn edge(e: CtEdgeCt, wire: CtWireName) -> Self {
        assert_eq!(
            wire.role(),
            CtScratch::Edge(e),
            "wire {wire:?} does not spell the edge role {e:?}"
        );
        Self {
            role: CtScratch::Edge(e),
            wire,
        }
    }
}

impl From<CtFaceCt> for CtScratchKey {
    fn from(f: CtFaceCt) -> Self {
        CtScratchKey::canonical(f.into())
    }
}
impl From<CtCellCt> for CtScratchKey {
    fn from(c: CtCellCt) -> Self {
        CtScratchKey::canonical(c.into())
    }
}
impl From<CtEdgeCt> for CtScratchKey {
    fn from(e: CtEdgeCt) -> Self {
        CtScratchKey::canonical(e.into())
    }
}

/// the compiler-owned scratch namespace: typed CT scratch (the reserved
/// vocabulary above) or a genuinely free scratch spelling. serializes as the
/// bare rendered string, so the manifest bytes are those of the plain string
/// the arm carried before the vocabulary was typed; deserialization
/// re-classifies at the same chokepoint.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
#[serde(into = "String", from = "String")]
pub enum ScratchKey {
    Ct(CtScratchKey),
    Free(Box<str>),
}

impl ScratchKey {
    /// the rendered wire spelling — the manifest string.
    pub fn name(&self) -> String {
        match self {
            ScratchKey::Ct(k) => k.render(),
            ScratchKey::Free(s) => s.to_string(),
        }
    }

    /// the CT role when this key is in the reserved vocabulary.
    pub fn ct_role(&self) -> Option<CtScratch> {
        match self {
            ScratchKey::Ct(k) => Some(k.role()),
            ScratchKey::Free(_) => None,
        }
    }
}

/// classification at the string boundary: a reserved spelling normalizes into
/// the typed arm, so a string-built and a typed-built key agree under `Eq`.
impl From<&str> for ScratchKey {
    fn from(s: &str) -> Self {
        match CtWireName::parse(s) {
            Some(w) => ScratchKey::Ct(CtScratchKey::spelled(w)),
            None => ScratchKey::Free(s.into()),
        }
    }
}

impl From<String> for ScratchKey {
    fn from(s: String) -> Self {
        ScratchKey::from(s.as_str())
    }
}

impl From<Box<str>> for ScratchKey {
    fn from(s: Box<str>) -> Self {
        ScratchKey::from(&*s)
    }
}

impl From<ScratchKey> for String {
    fn from(k: ScratchKey) -> String {
        k.name()
    }
}

impl From<CtScratchKey> for ScratchKey {
    fn from(k: CtScratchKey) -> Self {
        ScratchKey::Ct(k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the golden vocabulary pin: every reserved wire renders to its exact ABI
    /// string, parses back to itself, and its role round-trips through the key.
    #[test]
    fn reserved_vocabulary_round_trips() {
        let expected: &[(&str, &str)] = &[
            ("bface_a", "Face"),
            ("bface_b", "Face"),
            ("b_p1", "Face"),
            ("b_p2", "Face"),
            ("b", "Face"),
            ("bx", "Face"),
            ("by", "Face"),
            ("br", "Face"),
            ("bz", "Face"),
            ("b0", "Face"),
            ("b1", "Face"),
            ("bcell_p1", "Cell"),
            ("bcell_p2", "Cell"),
            ("bcell_out", "Cell"),
            ("vel_p1", "Cell"),
            ("vel_p2", "Cell"),
            ("vel_out", "Cell"),
            ("rho", "Cell"),
            ("pre", "Cell"),
            ("flag", "Cell"),
            ("fden_p1", "Face"),
            ("fden_p2", "Face"),
            ("bflux_a", "Face"),
            ("bflux_b", "Face"),
            ("wsl_p1", "Face"),
            ("wsl_p2", "Face"),
            ("wsr_p1", "Face"),
            ("wsr_p2", "Face"),
            ("emf", "Edge"),
            ("ez", "Edge"),
            ("e", "Edge"),
            ("ephi", "Edge"),
            ("e_p1", "Edge"),
            ("e_p2", "Edge"),
            ("e_fo", "Edge"),
            ("e_ho", "Edge"),
            ("e_n", "Edge"),
            ("bf_0", "Face"),
            ("fo_bflux_0", "Face"),
            ("ho_bflux_0", "Face"),
            ("bf_1", "Face"),
            ("fo_bflux_1", "Face"),
            ("ho_bflux_1", "Face"),
            ("bf_2", "Face"),
            ("fo_bflux_2", "Face"),
            ("ho_bflux_2", "Face"),
            ("fq_0", "Cell"),
            ("fq_1", "Cell"),
            ("fq_2", "Cell"),
            ("slip_ge", "Cell"),
            ("slip_qdot", "Cell"),
            ("slip_bz", "Cell"),
        ];
        let all = CtWireName::all();
        assert_eq!(all.len(), expected.len(), "vocabulary size drifted");
        for w in all {
            let s = w.render();
            assert_eq!(CtWireName::parse(&s), Some(w), "round-trip of {s}");
            let centering = match w.role() {
                CtScratch::Cell(_) => "Cell",
                CtScratch::Face(_) => "Face",
                CtScratch::Edge(_) => "Edge",
            };
            let exp = expected
                .iter()
                .find(|(name, _)| *name == s)
                .unwrap_or_else(|| panic!("{s} missing from the golden table"));
            assert_eq!(centering, exp.1, "centering of {s}");
        }
    }

    /// derived SSA keys and near-miss typos stay outside the vocabulary.
    #[test]
    fn derived_keys_are_not_reserved() {
        for s in [
            "edge_bface_a",
            "h_bp1",
            "e_bp2",
            "edge_wsr1",
            "emff",
            "bface_c",
            "bf_3",
            "fo_bflux_3",
            "ho_bflux_3",
            "prim.rho",
            "cons_den",
        ] {
            assert_eq!(CtWireName::parse(s), None, "{s} must stay untyped");
            assert!(matches!(ScratchKey::from(s), ScratchKey::Free(_)));
        }
    }

    /// the four EMF spellings are one semantic identity; the spelling stays
    /// visible on the key for the manifest.
    #[test]
    fn emf_spellings_share_one_role() {
        for w in [
            CtWireName::Emf,
            CtWireName::Ez,
            CtWireName::E,
            CtWireName::Ephi,
            CtWireName::EFo,
        ] {
            assert_eq!(w.role(), CtScratch::Edge(CtEdgeCt::Emf));
        }
        let a = CtScratchKey::spelled(CtWireName::Emf);
        let b = CtScratchKey::spelled(CtWireName::Ez);
        assert_eq!(a.role(), b.role());
        assert_ne!(a, b, "distinct ports stay distinct binding keys");
    }

    /// a string-built and a typed-built key agree under Eq.
    #[test]
    fn classification_normalizes() {
        let typed: ScratchKey =
            CtScratchKey::canonical(CtFaceCt::BFace(Transverse::A).into()).into();
        let stringly = ScratchKey::from("bface_a");
        assert_eq!(typed, stringly);
    }

    /// a face role spelled with a foreign wire fails at the validated
    /// constructor — the cross-axis/cross-family arm of the centering law.
    #[test]
    #[should_panic]
    fn face_key_rejects_a_wire_of_another_role() {
        let _ = CtScratchKey::face(CtFaceCt::BFace(Transverse::A), CtWireName::BFaceB);
    }

    #[test]
    #[should_panic]
    fn phys_comp_rejects_out_of_space() {
        let _ = PhysComp::new(3);
    }

    #[test]
    fn grid_axis_validates_against_d() {
        assert!(GridAxis::<2>::try_new(2).is_none());
        assert_eq!(GridAxis::<3>::new(2).index(), 2);
    }
}
