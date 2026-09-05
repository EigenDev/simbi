from __future__ import annotations

# =============================================================================
# bodies.py
#
# type definitions for immersed bodies and gravitational systems: the immutable
# config dataclasses (ImmersedBodyConfig, GravitationalSystemConfig, BinaryConfig,
# and the property blocks) plus `to_backend` / `body_payload` -- the single pure
# serialization the rust body factory reads.
# =============================================================================
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import IntFlag
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .shape import Shape

Array = NDArray[np.floating]


class BodyCapability(IntFlag):
    NONE = 0
    GRAVITATIONAL = 1 << 0
    ACCRETION = 1 << 1
    ELASTIC = 1 << 2
    DEFORMABLE = 1 << 3
    RIGID = 1 << 4


def has_capability(
    body_capability: BodyCapability, capability: BodyCapability
) -> bool:
    return bool(body_capability & capability)


# capability bits the rust binding actually honors: gravitational builds a
# fixed-potential mass, accretion a black-hole sink, rigid a drain-off wall
# (the porous surface with porosity 0). elastic / deformable have no backend
# path, so a config declaring them is rejected at load. running them would
# silently do nothing while the config claimed the capability was active.
_WIRED_CAPABILITIES = (
    BodyCapability.GRAVITATIONAL | BodyCapability.ACCRETION | BodyCapability.RIGID
)


def _config_error(message: str) -> Exception:
    """the house user-facing config error. imported lazily: simbi.simulation.
    problem imports this module at load time, so a top-level import would cycle."""
    from simbi.simulation.problem import ConfigError

    return ConfigError(message)


# the softening family vocabulary, shared by every config that carries a softening length so the
# two spellings cannot drift apart between body kinds.
#
# "plummer" is a genuine extended profile, and its field sits below newtonian at every radius
# (0.354 of it at r = h, reaching 0.99 only past r = 5h). a length chosen to keep the field finite
# near a body therefore biases gravity across the entire domain, which a measurement fitting a
# power law in radius reads as a shifted exponent.
#
# "compact" truncates the source at the softening length: the enclosed mass is complete outside it,
# so the field there is the bare point mass to the last bit and only the interior is regularized.
SOFTENING_KINDS = ("plummer", "compact")


def _validate_softening_kind(kind: str) -> None:
    """reject an unrecognized softening family at the config layer. the backend resolves anything
    other than "compact" to plummer, so an unchecked spelling runs silently on the other field."""
    if kind not in SOFTENING_KINDS:
        raise ValueError(
            f"softening_kind {kind!r} is not one of {SOFTENING_KINDS}. "
            "'plummer' is an extended profile whose field is below newtonian at every "
            "radius; 'compact' is exactly newtonian outside the softening length and "
            "regularized only within it."
        )


@dataclass(frozen=True)
class BinaryComponentConfig:
    mass: float
    radius: float
    is_an_accretor: bool
    softening_length: float
    two_way_coupling: bool
    accretion_radius: float
    sink_rate: float = 0.0
    position: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    velocity: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    force: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    total_accreted_mass: float = 0.0
    # which family `softening_length` parameterizes; see SOFTENING_KINDS.
    softening_kind: str = "plummer"
    # The same mutually-exclusive magnetic coupling carried by a standalone
    # immersed body.  Keeping it on the component is essential: the backend
    # turns prescribed binary components into ordinary Body values before it
    # attaches the Keplerian orbit, so dropping the coupling here would make a
    # moving magnetic sink impossible through the public API.
    magnetic: MagneticProperties | MagneticSlipProperties | None = None

    def __post_init__(self) -> None:
        _validate_softening_kind(self.softening_kind)
        if not isinstance(
            self.magnetic,
            (MagneticProperties, MagneticSlipProperties, type(None)),
        ):
            raise _config_error(
                "binary-component magnetic coupling must be MagneticProperties, "
                "MagneticSlipProperties, or None"
            )
        if (
            isinstance(self.magnetic, MagneticSlipProperties)
            and not self.is_an_accretor
        ):
            raise _config_error(
                "magnetic slip closes on a drain time, so a binary component carrying "
                "it must be an accretor"
            )

    def to_body_config(self) -> dict:
        """the backend wire for one binary component: top-level mass / radius / kinematics plus
        the nested `gravitational` and `accretion` property groups the rust body parser reads."""
        config = {
            "mass": self.mass,
            "radius": self.radius,
            "position": self.position,
            "velocity": self.velocity,
            "force": self.force,
            "two_way_coupling": self.two_way_coupling,
        }

        # all binary components have gravitational properties. the key set here must match the one
        # `GravitationalProperties` emits for a standalone body: both are read by the same backend
        # lookup, so a key present on one path and absent on the other resolves to a silent default
        # rather than an error.
        config["gravitational"] = {
            "softening_length": self.softening_length,
            "softening_kind": self.softening_kind,
        }

        # add accretion properties if this component is an accretor
        if self.is_an_accretor:
            config["accretion"] = {
                "accretion_radius": self.accretion_radius,
                "sink_rate": self.sink_rate,
                "total_accreted_mass": self.total_accreted_mass,
            }

        if isinstance(self.magnetic, MagneticSlipProperties):
            config["magnetic"] = {"slip": asdict(self.magnetic)}
        elif isinstance(self.magnetic, MagneticProperties):
            config["magnetic"] = asdict(self.magnetic)

        return config


@dataclass(frozen=True)
class BinaryConfig:
    semi_major: float
    eccentricity: float
    mass_ratio: float
    total_mass: float
    components: Sequence[BinaryComponentConfig]


@dataclass(frozen=True)
class BodySystemConfig:
    """base body-system config. `to_backend` is the serialization SSOT the backend
    body-system factory reads; the fieldless base contributes no keys."""

    def to_backend(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GravitationalSystemConfig(BodySystemConfig):
    """Configuration for gravitational system."""

    # general gravitational config
    prescribed_motion: bool
    reference_frame: str
    system_type: str
    # only used if system_type="binary"
    binary_config: BinaryConfig

    def to_backend(self) -> dict[str, Any]:
        """the backend body-system wire: the nested `binary_config` the gravitational
        factory reads. one of the two `to_backend` serializers `body_payload` composes."""
        result = {
            "prescribed_motion": self.prescribed_motion,
            "reference_frame": self.reference_frame,
            "system_type": self.system_type,
        }

        if self.binary_config:
            result["binary_config"] = {
                "semi_major": self.binary_config.semi_major,
                "eccentricity": self.binary_config.eccentricity,
                "mass_ratio": self.binary_config.mass_ratio,
                "total_mass": self.binary_config.total_mass,
                "components": [
                    comp.to_body_config()
                    for comp in self.binary_config.components
                ],
            }

        return result


@dataclass(frozen=True)
class GravitationalProperties:
    softening_length: float
    # which family `softening_length` parameterizes; see SOFTENING_KINDS. "compact" is the one to
    # use when the softening exists solely to keep the field finite where a sink has thinned the
    # gas -- set the length to the accretion radius and the flow outside it feels the exact point
    # mass.
    softening_kind: str = "plummer"

    def __post_init__(self) -> None:
        _validate_softening_kind(self.softening_kind)


@dataclass(frozen=True)
class AccretionProperties:
    accretion_radius: float
    sink_rate: float = 0.0
    total_accreted_mass: float = 0.0
    accretion_rate: float = 0.0
    # the porous surface dial: None keeps the pure drain.
    # porosity p scales the drain channel, (1 - p) the wall channels; the
    # wall rates are k_eta_* c_s / dx (multiplicative dials, zero = channel
    # off exactly, so k_eta_t = 0 is a free-slip surface). p = 0 is a sealed
    # wall (zero mass receipts, exactly); p = 1 reduces to the pure drain.
    porosity: float | None = None
    k_eta_n: float = 0.0
    k_eta_t: float = 0.0
    # the torque-free dial: None selects the pure drain kernel. a value in
    # [0, 1] selects the dittmann torque-free kernel for either adiabatic or
    # isothermal hydrodynamics, where xi = 1 removes mass but no angular
    # momentum. xi = 0 still selects that kernel and is not numerically
    # equivalent to None near evacuation. mutually exclusive with porosity.
    torque_free_xi: float | None = None

    def __post_init__(self) -> None:
        if self.accretion_radius <= 0.0:
            raise _config_error(
                f"accretion_radius must be > 0: a sink of zero radius drains no "
                f"gas. got {self.accretion_radius}."
            )
        # every immersed-boundary surface drains through the penalization stack. the spherical
        # accretor uses the faster of one acoustic cell crossing and free fall through the mask;
        # the shaped porous wall uses the acoustic rate. the pure drain scales every conserved
        # component uniformly, so its state-dependent sound speed is invariant under the drain
        # itself. the legacy in-godunov sink that `sink_rate` once set is
        # retired, and the scalar is bound to zero wherever a surface exists, which is everywhere.
        #
        # refused rather than ignored: a run that sets it is asserting control over the accretion
        # rate that it does not have, and the resulting Mdot looks entirely reasonable — it is
        # simply the penalization drain, whatever the dial said.
        if self.sink_rate != 0.0:
            raise _config_error(
                f"sink_rate={self.sink_rate} is not a live parameter: the immersed-boundary "
                "penalization surface owns accretion; the scalar is bound to zero. drop the "
                "argument."
            )
        if self.porosity is not None and not (0.0 <= self.porosity <= 1.0):
            raise _config_error(
                f"porosity must be in [0, 1]: it weights the drain channel by p "
                f"and the wall channels by (1 - p), so outside [0, 1] a channel "
                f"weight is negative (anti-relaxation). got {self.porosity}."
            )
        if self.k_eta_n < 0.0 or self.k_eta_t < 0.0:
            raise _config_error(
                f"k_eta_n and k_eta_t must be >= 0: they are surface-friction "
                f"rate dials; negative is anti-friction. got k_eta_n="
                f"{self.k_eta_n}, k_eta_t={self.k_eta_t}."
            )
        if self.torque_free_xi is not None and not (
            0.0 <= self.torque_free_xi <= 1.0
        ):
            raise _config_error(
                f"torque_free_xi must be in [0, 1]: it is the torque-free "
                f"strength (0 = standard drain, 1 = fully torque-free). got "
                f"{self.torque_free_xi}."
            )
        if self.torque_free_xi is not None and self.porosity is not None:
            raise _config_error(
                "an accretor cannot be both porous and torque-free: porosity and "
                "torque_free_xi are different surface physics on the same "
                "tangential channel. declare one."
            )


@dataclass(frozen=True)
class RigidProperties:
    # a rigid immersed wall: the drain-off porous surface (porosity 0). the wall
    # relaxes the gas velocity toward the body velocity on two channels at rates
    # k_eta_* c_s / dx: k_eta_n (normal, no-penetration) always acts; k_eta_t
    # (tangential) acts only under no-slip (apply_no_slip False = free slip, so
    # the tangential channel is switched off exactly). inertia carries the body's
    # moment of inertia for the two-way rotational coupling. `shape` is an
    # optional signed-distance CSG (body-local frame): None is the analytic sphere of
    # radius `body.radius`; a Shape gives an arbitrary rigid wall whose penalization
    # kernel is runtime-built + JIT-compiled per distinct geometry.
    inertia: float
    apply_no_slip: bool
    k_eta_n: float = 1.0
    k_eta_t: float = 1.0
    shape: Optional[Shape] = None
    # optional principal moments of inertia (I1, I2, I3) in the body frame. None = isotropic
    # (`inertia` on all three axes). unequal moments make an asymmetric body precess/tumble under
    # Euler's gyroscopic term (a torque-free wobble).
    inertia_principal: Optional[tuple[float, float, float]] = None
    # prescribed spin rate (radians/time) about `spin_axis`. nonzero makes a shaped wall spin: its
    # mask rotates as Rodrigues(axis, omega*t) and its no-slip surface drags the gas at
    # (omega * axis) x r. the axis is normalized here; default z (the 2D in-plane spin).
    omega: float = 0.0
    spin_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        norm = math.sqrt(sum(x * x for x in self.spin_axis))
        if self.omega != 0.0 and norm == 0.0:
            raise _config_error("a spinning wall needs a nonzero spin_axis.")
        if norm > 0.0:
            object.__setattr__(self, "spin_axis", tuple(x / norm for x in self.spin_axis))
        if self.shape is not None and not isinstance(self.shape, Shape):
            raise _config_error(
                f"rigid shape must be a Shape (Shape.sphere/box/union/...), got "
                f"{type(self.shape).__name__}."
            )
        if self.omega != 0.0 and self.shape is None:
            raise _config_error(
                "a spinning rigid wall (omega != 0) needs a `shape`: a rotationally symmetric "
                "sphere's mask does not change under rotation (spinning spheres are a follow-on)."
            )
        if self.k_eta_n <= 0.0:
            raise _config_error(
                f"rigid k_eta_n must be > 0: it is the normal (no-penetration) "
                f"wall-relaxation rate dial; zero leaves the wall permeable. got "
                f"{self.k_eta_n}."
            )
        if self.k_eta_t < 0.0:
            raise _config_error(
                f"rigid k_eta_t must be >= 0: it is the tangential (no-slip) "
                f"wall-relaxation rate dial; negative is anti-friction. got "
                f"{self.k_eta_t}."
            )
        if self.inertia_principal is not None:
            if len(self.inertia_principal) != 3 or any(
                m <= 0.0 for m in self.inertia_principal
            ):
                raise _config_error(
                    f"inertia_principal must be three positive moments (I1, I2, I3); got "
                    f"{self.inertia_principal}."
                )


@dataclass(frozen=True)
class MagneticProperties:
    """the body's magnetic coupling (MHD runs): a localized Ohmic resistivity `eta` that dissipates the
    magnetic field THREADING the body (`MagneticSpec::Resistive`). a no-op on B for a hydro run. the
    backend reads `magnetic.resistivity`; supported for cartesian 2.5D MHD."""

    resistivity: float

    def __post_init__(self) -> None:
        if self.resistivity < 0.0:
            raise _config_error(
                f"magnetic resistivity must be >= 0: it is a diffusivity, not a source. "
                f"got {self.resistivity}."
            )


@dataclass(frozen=True)
class MagneticSlipProperties:
    """force-selective magnetic slip at a mass-removing sink (`MagneticSpec::Slip`). the field
    threading a shell around the sink is transported relative to the draining gas by the
    Lorentz-driven slip velocity, so compressed flux is released while a force-free field is left
    exactly untouched; the magnetic energy released heats the gas. the coefficient closes on the
    sink's own drain time tau_rho through a magnetic Damkohler number, so the model carries no free
    resistivity:

        a_B = ell_B^2 / ((|B|^2 + field_regularization^2) * diffusivity_ratio * tau_rho)
        ell_B = slip_length_ratio * shell_width

    parameters (code units; magnetic energy |B|^2 / 2):
      diffusivity_ratio    D_B = tau_B / tau_rho > 0, how much slower the field decouples than the
                           gas drains. D_B near 1 releases flux on the accretion time; larger holds
                           the field longer.
      shell_width          w > 0, the mollification width of the slip shell (tanh ramp), in length
                           units; a few cells at the sink's mask edge.
      field_regularization B_0 > 0, bounds the slip speed at magnetic nulls; a small fraction of
                           the ambient field strength.
      slip_length_ratio    ell_B / w > 0, the transport length in shell widths. 1 is the sharp-
                           interface scaling.
      placement            shell center in shell widths relative to the mass surface: negative
                           inside, 0 centered on it, positive outside. finite, signed.

    runs in adiabatic Newtonian MHD (`Regime.NMHD`) on a cartesian grid, 3D or 2.5D (an x-y grid
    with three vector components, where the sink is a cylinder along the missing axis and the
    out-of-plane field takes part in the slip), on a body whose surface removes mass (a plain
    accretion drain, a torque-free drain, or a porous surface with porosity > 0), since tau_rho is
    that drain's timescale. the backend reads `magnetic.slip`."""

    diffusivity_ratio: float
    shell_width: float
    field_regularization: float
    slip_length_ratio: float = 1.0
    placement: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("diffusivity_ratio", self.diffusivity_ratio),
            ("shell_width", self.shell_width),
            ("field_regularization", self.field_regularization),
            ("slip_length_ratio", self.slip_length_ratio),
        ):
            if not (math.isfinite(value) and value > 0.0):
                raise _config_error(
                    f"magnetic slip {name} must be finite and positive; got {value}."
                )
        if not math.isfinite(self.placement):
            raise _config_error(
                f"magnetic slip placement must be finite (signed shell widths); got "
                f"{self.placement}."
            )


@dataclass(frozen=True)
class ImmersedBodyConfig:
    capability: BodyCapability
    mass: float
    velocity: Sequence[float]
    position: Sequence[float]
    radius: float
    two_way_coupling: bool = field(default=False)
    force: Sequence[float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    gravitational: Optional[GravitationalProperties] = None
    accretion: Optional[AccretionProperties] = None
    rigid: Optional[RigidProperties] = None
    # MHD magnetic coupling: an Ohmic resistive sink or the force-selective magnetic slip;
    # None = magnetically transparent. one coupling per body.
    magnetic: Optional[MagneticProperties | MagneticSlipProperties] = None

    def __post_init__(self) -> None:
        unsupported = self.capability & ~_WIRED_CAPABILITIES
        if unsupported:
            raise _config_error(
                f"immersed-body capability {unsupported!r} is not wired to the "
                f"backend; only GRAVITATIONAL, ACCRETION, and RIGID are honored. a "
                f"body declaring it would run as a passive gravitating mass."
            )
        if (
            has_capability(self.capability, BodyCapability.RIGID)
            and self.rigid is None
        ):
            raise _config_error(
                "capability RIGID requires a `rigid` property block (the wall "
                "no-slip flag and stiffness dials); without it the wall is undefined."
            )
        # a two-way rigid wall spins up from the gas reaction torque (I domega = torque). that
        # needs a shape (a sphere's mask is rotation-invariant, so it never reaches the spinning
        # kernel) and a positive moment of inertia; otherwise the coupling is a silent no-op.
        if (
            has_capability(self.capability, BodyCapability.RIGID)
            and self.two_way_coupling
            and self.rigid is not None
        ):
            if self.rigid.shape is None:
                raise _config_error(
                    "a two-way rigid wall spins from the gas reaction torque; it needs a `shape` "
                    "(a rotationally symmetric sphere never spins up)."
                )
            if self.rigid.inertia <= 0.0:
                raise _config_error(
                    f"a two-way rigid wall needs inertia > 0 (I domega = torque); got "
                    f"{self.rigid.inertia}."
                )
            if self.mass <= 0.0:
                raise _config_error(
                    f"a two-way rigid wall needs mass > 0 (mass dv = drag, the flow pushes it "
                    f"downstream); got {self.mass}."
                )
        if (
            has_capability(self.capability, BodyCapability.ACCRETION)
            and self.accretion is None
        ):
            raise _config_error(
                "capability ACCRETION requires an `accretion` property block; "
                "without it the sink has zero accretion radius and drains nothing."
            )
        if (
            has_capability(self.capability, BodyCapability.GRAVITATIONAL)
            and self.gravitational is None
        ):
            raise _config_error(
                "capability GRAVITATIONAL requires a `gravitational` property "
                "block (the softening length)."
            )
        if isinstance(self.magnetic, MagneticSlipProperties):
            # the slip coefficient closes on the drain time tau_rho, so the surface must remove mass:
            # a plain accretion drain, a torque-free drain, or a porous surface with porosity > 0.
            acc = self.accretion
            drains = (
                has_capability(self.capability, BodyCapability.ACCRETION)
                and acc is not None
                and (
                    acc.torque_free_xi is not None
                    or acc.porosity is None
                    or acc.porosity > 0.0
                )
            )
            if not drains:
                raise _config_error(
                    "magnetic slip closes its coefficient on the sink's drain time, so the body "
                    "must remove mass: give it capability ACCRETION with a draining surface (a "
                    "plain drain, torque_free_xi, or porosity > 0)."
                )

    def to_backend(self) -> dict[str, Any]:
        """the backend body wire: the nested dict the rust `BodyParams` reads --
        top-level capability / mass / radius / position / velocity / two_way_coupling,
        and the `gravitational` / `accretion` / `rigid` property groups (with
        `rigid.shape.wire` for a shaped wall). the field tree IS the contract, so the
        serialization is the identity asdict; the key set is pinned by test so a field
        rename cannot silently become a backend unwrap_or default. the magnetic coupling crosses
        as `magnetic.resistivity` for the Ohmic sink and as the `magnetic.slip` group for the
        magnetic slip, so the backend reads exactly one of the two."""
        wire = asdict(self)
        if isinstance(self.magnetic, MagneticSlipProperties):
            wire["magnetic"] = {"slip": asdict(self.magnetic)}
        return wire


@dataclass(frozen=True)
class BondMaterial:
    """elastic-bond parameters: normal/tangential stiffness, damping, and the
    strength envelope (tensile stress `sigma_t`, shear stress `tau_s`, over
    cross-section `area`). infinite strengths never break."""

    k_n: float
    k_t: float = 0.0
    gamma: float = 0.0
    area: float = 1.0
    sigma_t: float = math.inf
    tau_s: float = math.inf

    def __post_init__(self) -> None:
        if self.k_n <= 0.0:
            raise _config_error("bond k_n must be positive (the bond has no spring).")
        if self.area <= 0.0 or self.sigma_t <= 0.0 or self.tau_s <= 0.0:
            raise _config_error("bond area / sigma_t / tau_s must be positive.")


@dataclass(frozen=True)
class ContactMaterial:
    """soft-sphere contact parameters: normal spring `k_n`, tangential spring
    `k_t`, normal dashpot `gamma_n`, friction coefficient `mu`."""

    k_n: float
    k_t: float = 0.0
    gamma_n: float = 0.0
    mu: float = 0.0

    def __post_init__(self) -> None:
        if self.k_n <= 0.0:
            raise _config_error("contact k_n must be positive.")
        if self.mu < 0.0:
            raise _config_error("contact mu must be non-negative.")


@dataclass(frozen=True)
class MutualGravity:
    """pairwise gravity between all bodies (plummer-softened direct sum)."""

    g: float = 1.0
    softening: float = 0.0


@dataclass(frozen=True)
class BondedAssembly:
    """a cluster of wall-only rigid spherical fragments joined by breakable
    elastic bonds, with optional contact and mutual self-gravity. fragments
    never gravitate on the gas and never accrete; the gas sees each one as a
    sealed rigid wall and its motion integrates in the bonded subcycle.
    `mobile` marks per-fragment mobility (False = a clamp / prescribed drift);
    omitted means every fragment is mobile."""

    positions: Sequence[Sequence[float]]
    masses: Sequence[float]
    radii: Sequence[float]
    bonds: Sequence[Sequence[int]]
    bond_material: BondMaterial
    velocities: Optional[Sequence[Sequence[float]]] = None
    inertias: Optional[Sequence[float]] = None
    mobile: Optional[Sequence[bool]] = None
    contact: Optional[ContactMaterial] = None
    gravity: Optional[MutualGravity] = None
    k_eta_n: float = 50.0
    k_eta_t: float = 50.0

    def __post_init__(self) -> None:
        n = len(self.positions)
        if n == 0:
            raise _config_error("a bonded assembly needs at least one fragment.")
        for name in ("masses", "radii"):
            if len(getattr(self, name)) != n:
                raise _config_error(
                    f"bonded assembly `{name}` has {len(getattr(self, name))} entries "
                    f"for {n} fragments."
                )
        for name in ("velocities", "inertias", "mobile"):
            val = getattr(self, name)
            if val is not None and len(val) != n:
                raise _config_error(
                    f"bonded assembly `{name}` has {len(val)} entries for {n} fragments."
                )
        if any(m <= 0.0 for m in self.masses):
            raise _config_error("every fragment mass must be positive.")
        if any(r <= 0.0 for r in self.radii):
            raise _config_error("every fragment radius must be positive.")
        seen: set[tuple[int, int]] = set()
        for pair in self.bonds:
            if len(pair) != 2:
                raise _config_error(f"a bond is a pair (i, j); got {tuple(pair)}.")
            i, j = int(pair[0]), int(pair[1])
            if i == j or not (0 <= i < n) or not (0 <= j < n):
                raise _config_error(
                    f"bond ({i}, {j}) must join two distinct fragments in [0, {n})."
                )
            key = (min(i, j), max(i, j))
            if key in seen:
                raise _config_error(f"bond {key} is declared twice.")
            seen.add(key)
        if not isinstance(self.bond_material, BondMaterial):
            raise _config_error(
                "bond_material must be a BondMaterial (validated), not "
                f"{type(self.bond_material).__name__}."
            )

    @classmethod
    def pack(
        cls,
        shape: Shape,
        bounds: Sequence[Sequence[float]],
        spacing: float,
        fragment_mass: float,
        bond_material: BondMaterial,
        neighbor_cutoff: float = 1.5,
        jitter: float = 0.0,
        seed: int = 7,
        **kwargs: Any,
    ) -> "BondedAssembly":
        """fill `shape` (body-local frame) with a square/cubic lattice of
        spherical fragments and bond every pair closer than
        `neighbor_cutoff * spacing` (1.5 catches the 2d/3d diagonals). a
        lattice point survives when its whole fragment fits inside the shape:
        signed_distance <= -radius. `jitter` displaces each point by a
        deterministic (seeded) uniform offset of up to `jitter * spacing` per
        axis — an irregular pile instead of graph paper — and the fragment
        radius shrinks to `(0.5 - jitter) * spacing` so jittered neighbors can
        never start overlapped (the gaps close under self-gravity). pure
        geometry; extra keyword arguments flow to the assembly (contact,
        gravity, mobile...)."""
        if not 0.0 <= jitter < 0.5:
            raise _config_error(f"pack jitter must be in [0, 0.5); got {jitter}.")
        radius = (0.5 - jitter) * spacing
        axes = [np.arange(lo + 0.5 * spacing, hi, spacing) for lo, hi in bounds]
        grids = np.meshgrid(*axes, indexing="ij")
        points = np.stack([g.ravel() for g in grids], axis=1)
        if jitter > 0.0:
            rng = np.random.default_rng(seed)
            points = points + rng.uniform(
                -jitter * spacing, jitter * spacing, size=points.shape
            )
        kept: list[list[float]] = []
        for p in points:
            local = [float(p[0]), float(p[1]) if len(p) > 1 else 0.0, float(p[2]) if len(p) > 2 else 0.0]
            if shape.signed_distance(local) <= -radius:
                kept.append([float(c) for c in p])
        if not kept:
            raise _config_error(
                f"packing produced zero fragments: no lattice point of spacing "
                f"{spacing} fits radius {radius} inside the shape over {bounds}."
            )
        cutoff2 = (neighbor_cutoff * spacing) ** 2
        bonds: list[tuple[int, int]] = []
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                d2 = sum((a - b) ** 2 for a, b in zip(kept[i], kept[j]))
                if d2 <= cutoff2:
                    bonds.append((i, j))
        n = len(kept)
        return cls(
            positions=kept,
            masses=[fragment_mass] * n,
            radii=[radius] * n,
            bonds=bonds,
            bond_material=bond_material,
            **kwargs,
        )

    def to_backend(self) -> dict[str, Any]:
        """the backend wire: plain lists + nested material dicts. per-fragment
        inertia defaults to the solid sphere 0.4 m r^2; velocities default to
        rest; mobility defaults to every fragment mobile."""
        n = len(self.positions)
        inertias = (
            [float(v) for v in self.inertias]
            if self.inertias is not None
            else [0.4 * float(m) * float(r) ** 2 for m, r in zip(self.masses, self.radii)]
        )
        velocities = (
            [[float(c) for c in v] for v in self.velocities]
            if self.velocities is not None
            else [[0.0] * len(self.positions[0]) for _ in range(n)]
        )
        mobile = [bool(b) for b in self.mobile] if self.mobile is not None else [True] * n
        return {
            "positions": [[float(c) for c in p] for p in self.positions],
            "masses": [float(m) for m in self.masses],
            "radii": [float(r) for r in self.radii],
            "inertias": inertias,
            "velocities": velocities,
            "mobile": mobile,
            "bonds": [[int(p[0]), int(p[1])] for p in self.bonds],
            "bond_material": asdict(self.bond_material),
            "contact": asdict(self.contact) if self.contact is not None else None,
            "gravity": asdict(self.gravity) if self.gravity is not None else None,
            "k_eta_n": float(self.k_eta_n),
            "k_eta_t": float(self.k_eta_t),
        }


def body_payload(
    body_system: Optional[BodySystemConfig],
    immersed_bodies: Sequence[ImmersedBodyConfig],
    bonded_assembly: Optional[BondedAssembly] = None,
) -> dict[str, Any]:
    """the backend body fragment -- `{body_system?, immersed_bodies?,
    bonded_assembly?}`, the keys the rust body factory reads. pure: config values
    in, a plain dict out. an absent body kind contributes no key (a missing key is
    read as body-free). each immersed body must be a validated ImmersedBodyConfig:
    a raw dict would let the backend read every field through a silent unwrap_or
    default, turning a typo into a wrong-physics run rather than an error."""
    payload: dict[str, Any] = {}
    if body_system is not None and is_dataclass(body_system):
        payload["body_system"] = body_system.to_backend()
    serialized: list[dict[str, Any]] = []
    for idx, body in enumerate(immersed_bodies):
        if not isinstance(body, ImmersedBodyConfig):
            raise _config_error(
                f"immersed_bodies[{idx}] is a {type(body).__name__}, not an "
                f"ImmersedBodyConfig. construct an ImmersedBodyConfig so its fields "
                f"are validated before the backend, which silently defaults any key "
                f"it cannot read."
            )
        serialized.append(body.to_backend())
    if serialized:
        payload["immersed_bodies"] = serialized
    if bonded_assembly is not None:
        if not isinstance(bonded_assembly, BondedAssembly):
            raise _config_error(
                f"bonded_assembly is a {type(bonded_assembly).__name__}, not a "
                f"BondedAssembly."
            )
        payload["bonded_assembly"] = bonded_assembly.to_backend()
    return payload


__all__ = [
    "body_payload",
    "BondedAssembly",
    "BondMaterial",
    "ContactMaterial",
    "MutualGravity",
    "ImmersedBodyConfig",
    "GravitationalSystemConfig",
    "BinaryConfig",
    "BinaryComponentConfig",
    "BodySystemConfig",
    "BodyCapability",
    "has_capability",
    "MagneticProperties",
    "MagneticSlipProperties",
]
