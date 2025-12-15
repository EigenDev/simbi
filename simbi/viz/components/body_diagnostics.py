# =============================================================================
# body_diagnostics.py
#
# component for body diagnostics time series visualization.
# follows component architecture with props/component pattern.
#
# usage:
#   props = BodyDiagnosticsProps(plot_type="torque-z", normalize="canonical")
#   component = BodyDiagnosticsComponent(props)
#   result = component.render(data, style)
# =============================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import Field, field_validator

from simbi.reader import read_checkpoint
from simbi.viz.utility import get_field_str

from ..config import FigureConfig
from ..types import RenderResult
from .interface import Component, ComponentProps

Array = NDArray[np.floating]


# =============================================================================
# data structures
# =============================================================================


@dataclass
class BodyTimeSeries:
    """time series data for body diagnostics."""

    times: Array
    positions: Array  # (n_times, n_bodies, ndim)
    velocities: Array
    forces: Array
    torques: Array
    masses: Array
    accreted_masses: Array
    accretion_rates: Array
    n_bodies: int
    ndim: int

    def body(self, idx: int) -> "SingleBodyTimeSeries":
        """extract time series for one body."""
        return SingleBodyTimeSeries(
            times=self.times,
            position=self.positions[:, idx, :],
            velocity=self.velocities[:, idx, :],
            force=self.forces[:, idx, :],
            torque=self.torques[:, idx, :],
            mass=self.masses[:, idx],
            accreted_mass=self.accreted_masses[:, idx],
            accretion_rate=self.accretion_rates[:, idx],
            ndim=self.ndim,
        )


@dataclass
class SingleBodyTimeSeries:
    """time series for a single body."""

    times: Array
    position: Array  # shape: (n_times, ndim)
    velocity: Array
    force: Array
    torque: Array  # always 3D
    mass: Array  # shape: (n_times,)
    accreted_mass: Array
    accretion_rate: Array
    ndim: int


@dataclass
class BinaryTimeSeries:
    """derived time series for binary system."""

    times: Array
    separation: Array
    separation_velocity: Array
    relative_acceleration: Array
    radial_acceleration: Array
    orbital_frequency: Array
    specific_angular_momentum: Array
    specific_energy: Array
    radial_force: Array
    tangential_force: Array
    drag_force: Array
    torque_z: Array
    power: Array
    angular_momentum_transfer: Array
    migration_timescale: Array
    decay_rate: Array


@dataclass
class Normalizations:
    """canonical normalizations for binary diagnostics."""

    force_scale: float
    torque_scale: float
    power_scale: float
    time_scale: float
    length_scale: float
    mass_scale: float

    @classmethod
    def from_binary(cls, m1: float, m2: float, a: float, G: float = 1.0):
        """create canonical normalizations from binary parameters."""
        m_total = m1 + m2
        return cls(
            force_scale=G * m1 * m2 / (a * a),
            torque_scale=G * m1 * m2 * a,
            power_scale=G * m1 * m2 / a,
            time_scale=2.0 * np.pi * np.sqrt(a**3 / (G * m_total)),
            length_scale=a,
            mass_scale=m_total,
        )

    @classmethod
    def custom(
        cls,
        force: float = 1.0,
        torque: float = 1.0,
        power: float = 1.0,
        time: float = 1.0,
        length: float = 1.0,
        mass: float = 1.0,
    ):
        """create custom normalizations."""
        return cls(
            force_scale=force,
            torque_scale=torque,
            power_scale=power,
            time_scale=time,
            length_scale=length,
            mass_scale=mass,
        )


# =============================================================================
# component props
# =============================================================================


class BodyDiagnosticsProps(ComponentProps):
    """properties for body diagnostics component."""

    # plot selection
    plot_type: Literal[
        "radial-force",
        "tangential-force",
        "drag-force",
        "torque-z",
        "power",
        "migration-time",
        "decay-rate",
        "forces",
        "torques",
        "accretion",
        "separation",
        "orbital-elements",
        "radial-accel",
        "summary",
        "binary-summary",
    ] = "torque-z"

    # body selection
    body_idx: int = Field(default=0, ge=0)
    show_components: bool = False  # for vector quantities

    # normalization
    normalize: Optional[Literal["canonical", "custom"]] = None
    norm_force: Optional[float] = Field(default=None, gt=0)
    norm_torque: Optional[float] = Field(default=None, gt=0)
    norm_power: Optional[float] = Field(default=None, gt=0)
    norm_time: Optional[float] = Field(default=None, gt=0)
    norm_length: Optional[float] = Field(default=None, gt=0)
    norm_mass: Optional[float] = Field(default=None, gt=0)

    # plot options
    cumulative: bool = False  # for accretion
    with_velocity: bool = False  # for separation
    show_both_bodies: bool = False  # for force/torque plots

    # styling
    marker: str = "o"
    linestyle: str = "-"
    linewidth: float = 1.5
    alpha: float = 0.8

    @field_validator("body_idx")
    @classmethod
    def validate_body_idx(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"body_idx must be non-negative, got {v}")
        return v


# =============================================================================
# physics calculations
# =============================================================================


def load_body_timeseries(checkpoint_files: list[Path]) -> BodyTimeSeries:
    """load body diagnostics from checkpoint files."""
    times = []
    positions = []
    velocities = []
    forces = []
    torques = []
    masses = []
    accreted_masses = []
    accretion_rates = []

    for filepath in checkpoint_files:
        checkpoint = read_checkpoint(str(filepath)).unwrap()

        if checkpoint.bodies is None:
            raise ValueError(f"no bodies found in {filepath}")

        times.append(checkpoint.metadata.time)

        n_bodies = checkpoint.bodies.count
        ndim = len(checkpoint.bodies.bodies[0].position)

        pos_t = []
        vel_t = []
        force_t = []
        torque_t = []
        mass_t = []
        accr_mass_t = []
        accr_rate_t = []

        for body in checkpoint.bodies.bodies:
            pos_t.append(body.position[:ndim])
            vel_t.append(body.velocity[:ndim])
            force_t.append(body.force[:ndim])
            torque_t.append(body.torque[:3])
            mass_t.append(body.mass)

            if body.accretion is not None:
                accr_mass_t.append(body.accretion.total_accreted_mass)
                accr_rate_t.append(body.accretion.accretion_rate)
            else:
                accr_mass_t.append(0.0)
                accr_rate_t.append(0.0)

        positions.append(pos_t)
        velocities.append(vel_t)
        forces.append(force_t)
        torques.append(torque_t)
        masses.append(mass_t)
        accreted_masses.append(accr_mass_t)
        accretion_rates.append(accr_rate_t)

    return BodyTimeSeries(
        times=np.array(times),
        positions=np.array(positions),
        velocities=np.array(velocities),
        forces=np.array(forces),
        torques=np.array(torques),
        masses=np.array(masses),
        accreted_masses=np.array(accreted_masses),
        accretion_rates=np.array(accretion_rates),
        n_bodies=n_bodies,
        ndim=ndim,
    )


def compute_binary_dynamics(ts: BodyTimeSeries) -> BinaryTimeSeries:
    """compute binary separation, relative velocity, and derived quantities."""
    if ts.n_bodies != 2:
        raise ValueError(f"expected 2 bodies, got {ts.n_bodies}")

    r1 = ts.positions[:, 0, :]
    r2 = ts.positions[:, 1, :]
    v1 = ts.velocities[:, 0, :]
    v2 = ts.velocities[:, 1, :]
    f1 = ts.forces[:, 0, :]
    f2 = ts.forces[:, 1, :]
    m1 = ts.masses[:, 0]
    m2 = ts.masses[:, 1]

    delta_r = r1 - r2
    separation = np.linalg.norm(delta_r, axis=1)
    delta_v = v1 - v2
    r_hat = delta_r / separation[:, None]
    sep_velocity = np.sum(delta_v * r_hat, axis=1)

    a1 = f1 / m1[:, None]
    a2 = f2 / m2[:, None]
    a_rel = a1 - a2
    radial_accel = np.sum(a_rel * r_hat, axis=1)

    v_perp_sq = np.sum(delta_v**2, axis=1) - sep_velocity**2
    v_perp = np.sqrt(np.maximum(v_perp_sq, 0))
    orbital_freq = v_perp / separation

    mu = m1 * m2 / (m1 + m2)
    m_total = m1 + m2

    if ts.ndim == 2:
        L_specific = (
            delta_r[:, 0] * delta_v[:, 1] - delta_r[:, 1] * delta_v[:, 0]
        )
    else:
        L_vec = np.cross(delta_r, delta_v)
        L_specific = np.linalg.norm(L_vec, axis=1)

    v_sq = np.sum(delta_v**2, axis=1)
    E_specific = 0.5 * v_sq - m_total / separation

    # scalar force projections
    f1_radial = np.sum(f1 * r_hat, axis=1)
    f2_radial = np.sum(f2 * r_hat, axis=1)
    f1_tangential = np.linalg.norm(f1 - f1_radial[:, None] * r_hat, axis=1)
    f2_tangential = np.linalg.norm(f2 - f2_radial[:, None] * r_hat, axis=1)

    # drag force
    v1_mag = np.linalg.norm(v1, axis=1)
    v2_mag = np.linalg.norm(v2, axis=1)
    v1_hat = np.where(v1_mag[:, None] > 1e-15, v1 / v1_mag[:, None], 0)
    v2_hat = np.where(v2_mag[:, None] > 1e-15, v2 / v2_mag[:, None], 0)
    f1_drag = -np.sum(f1 * v1_hat, axis=1)
    f2_drag = -np.sum(f2 * v2_hat, axis=1)

    # torque z-component
    torque1 = ts.torques[:, 0, :]
    torque2 = ts.torques[:, 1, :]
    torque_z = torque1[:, 2] + torque2[:, 2]

    # power
    power1 = np.sum(f1 * v1, axis=1)
    power2 = np.sum(f2 * v2, axis=1)

    # angular momentum transfer
    dL_dt = np.linalg.norm(torque1 + torque2, axis=1)

    # migration timescale
    da_dt = np.gradient(separation, ts.times)
    migration_time = np.where(
        np.abs(da_dt) > 1e-15, separation / (np.abs(da_dt) + 1e-10), np.inf
    )

    # decay rate
    total_power = power1 + power2
    decay_rate = -2.0 * separation**2 / (m1 * m2) * total_power

    return BinaryTimeSeries(
        times=ts.times,
        separation=separation,
        separation_velocity=sep_velocity,
        relative_acceleration=np.linalg.norm(a_rel, axis=1),
        radial_acceleration=radial_accel,
        orbital_frequency=orbital_freq,
        specific_angular_momentum=L_specific,
        specific_energy=E_specific,
        radial_force=np.column_stack([f1_radial, f2_radial]),
        tangential_force=np.column_stack([f1_tangential, f2_tangential]),
        drag_force=np.column_stack([f1_drag, f2_drag]),
        torque_z=torque_z,
        power=np.column_stack([power1, power2]),
        angular_momentum_transfer=dL_dt,
        migration_timescale=migration_time,
        decay_rate=decay_rate,
    )


# =============================================================================
# component implementation
# =============================================================================


class BodyDiagnosticsComponent(Component[BodyDiagnosticsProps, list[Path]]):
    """
    component for body diagnostics visualization.

    expects data: checkpoint directory path or file list
    renders: body diagnostic time series plots
    """

    def __init__(self, props: BodyDiagnosticsProps):
        self.props = props
        self._initialized = False
        self._ts: Optional[BodyTimeSeries] = None
        self._binary: Optional[BinaryTimeSeries] = None
        self._norm: Optional[Normalizations] = None

    def initialize(self, fig: Figure, ax: Axes) -> None:
        self.fig = fig
        self.ax = ax
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def update(self, props: BodyDiagnosticsProps) -> None:
        self.props = props

    def cleanup(self) -> None:
        self._ts = None
        self._binary = None
        self._norm = None

    def _load_data(self, files: list[Path]) -> None:
        """load time series from checkpoint path."""
        self._ts = load_body_timeseries(files)
        if self._ts.n_bodies == 2:
            self._binary = compute_binary_dynamics(self._ts)

    def _compute_normalization(self) -> None:
        """compute normalization factors."""
        if self.props.normalize == "canonical":
            if self._ts is None:
                raise RuntimeError("no timeseries loaded")
            if self._ts.n_bodies == 2:
                m1 = self._ts.masses[0, 0]
                m2 = self._ts.masses[0, 1]
                a = self._binary.separation.mean()
                self._norm = Normalizations.from_binary(m1, m2, a, G=1.0)
            else:
                print("warning: canonical normalization requires 2 bodies")
        elif self.props.normalize == "custom":
            self._norm = Normalizations.custom(
                force=self.props.norm_force or 1.0,
                torque=self.props.norm_torque or 1.0,
                power=self.props.norm_power or 1.0,
                time=self.props.norm_time or 1.0,
                length=self.props.norm_length or 1.0,
                mass=self.props.norm_mass or 1.0,
            )

    def _plot_scalar_force(self, force_data: Array, ylabel: str) -> None:
        """plot scalar force quantity (radial, tangential, drag)."""
        scale = self._norm.force_scale if self._norm else 1.0
        time_scale = self._norm.time_scale if self._norm else 1.0
        ylabel_full = get_field_str(ylabel)

        body_idx = self.props.body_idx
        if self.props.show_both_bodies and self._ts and self._ts.n_bodies == 2:
            self.ax.plot(
                self._binary.times / time_scale,
                force_data[:, 0] / scale,
                marker=self.props.marker,
                label="body 0",
            )
            self.ax.plot(
                self._binary.times / time_scale,
                force_data[:, 1] / scale,
                marker=self.props.marker,
                label="body 1",
            )
            self.ax.legend()
        else:
            self.ax.plot(
                self._binary.times / time_scale,
                force_data[:, body_idx] / scale,
                marker=self.props.marker,
                color="C0",
            )
        xlabel = "$t$"
        if abs(time_scale - 2.0 * np.pi) < 1e-1:
            xlabel += " [orbit(s)]"
        self.ax.axhline(0, color="black", linestyle=":", alpha=0.5)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel_full)
        self.ax.grid(True, alpha=0.3)

    def render(self, data: list[Path], style: FigureConfig) -> RenderResult:
        """render body diagnostics plot."""
        if not self._initialized:
            raise RuntimeError("component not initialized")

        # load data
        self._load_data(data)
        self._compute_normalization()

        # dispatch to appropriate plot type
        plot_type = self.props.plot_type

        if plot_type == "radial-force":
            self._plot_scalar_force(
                self._binary.radial_force, "radial force F_r"
            )
        elif plot_type == "tangential-force":
            self._plot_scalar_force(
                self._binary.tangential_force, "tangential force F_t"
            )
        elif plot_type == "drag-force":
            self._plot_scalar_force(
                self._binary.drag_force, "drag force F_drag"
            )
        elif plot_type == "torque-z":
            self._plot_torque_z()
        elif plot_type == "power":
            self._plot_power()
        elif plot_type == "decay-rate":
            self._plot_decay_rate()
        elif plot_type == "accretion":
            # provide a simple mapping to module helper behavior
            plot_accretion_rate(
                self._ts,
                body_idx=self.props.body_idx,
                cumulative=self.props.cumulative,
                ax=self.ax,
            )
        else:
            raise NotImplementedError(
                f"plot type '{plot_type}' not yet implemented"
            )

        # collect artists from axes
        artists = {
            "lines": self.ax.get_lines(),
            "collections": self.ax.collections,
        }

        return RenderResult(
            artists=artists,
            metadata={
                "field_name": f"body_{plot_type}",
                "label": plot_type.replace("-", " "),
            },
        )

    def _plot_torque_z(self) -> None:
        """plot z-component of torque."""
        scale = self._norm.torque_scale if self._norm else 1.0
        time_scale = self._norm.time_scale if self._norm else 1.0
        ylabel = get_field_str("torque z")

        self.ax.plot(
            self._binary.times / time_scale,
            self._binary.torque_z / scale,
            marker=self.props.marker,
            color="C2",
        )
        xlabel = "$t$"
        if abs(time_scale - 2.0 * np.pi) < 1e-1:
            xlabel += " [orbit(s)]"
        self.ax.axhline(0, color="black", linestyle=":", alpha=0.5)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True, alpha=0.3)

        # annotate trend
        mean_torque = self._binary.torque_z.mean()
        if abs(mean_torque / scale) > 1e-10:
            trend = "outspiral" if mean_torque > 0 else "inspiral"
            self.ax.text(
                0.98,
                0.98,
                f"mean: ${mean_torque / scale:.2f}$\n({trend})",
                transform=self.ax.transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

    def _plot_power(self) -> None:
        """plot power dE/dt."""
        power_scale = self._norm.power_scale if self._norm else 1.0
        time_scale = self._norm.time_scale if self._norm else 1.0

        if self.props.show_both_bodies:
            self.ax.plot(
                self._binary.times / time_scale,
                self._binary.power[:, 0] / power_scale,
                marker=self.props.marker,
                label="body 0",
            )
            self.ax.plot(
                self._binary.times / time_scale,
                self._binary.power[:, 1] / power_scale,
                marker=self.props.marker,
                label="body 1",
            )
            total = (
                self._binary.power[:, 0] + self._binary.power[:, 1]
            ) / power_scale
            self.ax.plot(
                self._binary.times / time_scale,
                total,
                marker="^",
                label="total",
                linestyle="--",
                color="black",
            )
            self.ax.legend()
        else:
            self.ax.plot(
                self._binary.times / time_scale,
                self._binary.power[:, self.props.body_idx] / power_scale,
                marker=self.props.marker,
                color="C3",
            )
        xlabel = "$t$"
        if abs(time_scale - 2.0 * np.pi) < 1e-1:
            xlabel += " [orbit(s)]"
        self.ax.axhline(0, color="black", linestyle=":", alpha=0.5)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(get_field_str("power"))
        self.ax.grid(True, alpha=0.3)

    def _plot_decay_rate(self) -> None:
        """plot orbital decay rate da/dt."""
        time_scale = 1.0
        if self._norm:
            decay_norm = self._binary.decay_rate / (
                self._norm.length_scale / self._norm.time_scale
            )
            time_scale = self._norm.time_scale
        else:
            decay_norm = self._binary.decay_rate

        self.ax.plot(
            self._binary.times / time_scale,
            decay_norm,
            marker=self.props.marker,
            color="C6",
        )
        self.ax.axhline(0, color="black", linestyle=":", alpha=0.5)
        self.ax.set_xlabel("$t$")
        self.ax.set_ylabel(get_field_str("decay rate"))
        self.ax.grid(True, alpha=0.3)

        mean_decay = decay_norm.mean()
        if abs(mean_decay) > 1e-10:
            trend = "outspiral" if mean_decay > 0 else "inspiral"
            self.ax.text(
                0.98,
                0.98,
                f"mean: {mean_decay:.2e}\n({trend})",
                transform=self.ax.transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

    # convenience instance wrappers that call module-level helpers --------------------------------

    def plot_forces_instance(
        self,
        ax: Optional[Axes] = None,
        body_idx: Optional[int] = None,
        components: Optional[bool] = None,
    ) -> Axes:
        if self._ts is None:
            raise RuntimeError(
                "no timeseries loaded; call _load_data(files) first"
            )
        bi = self.props.body_idx if body_idx is None else body_idx
        comp = self.props.show_components if components is None else components
        return plot_forces(self._ts, body_idx=bi, components=comp, ax=ax)

    def plot_torques_instance(
        self,
        ax: Optional[Axes] = None,
        body_idx: Optional[int] = None,
        components: Optional[bool] = None,
    ) -> Axes:
        if self._ts is None:
            raise RuntimeError(
                "no timeseries loaded; call _load_data(files) first"
            )
        bi = self.props.body_idx if body_idx is None else body_idx
        comp = self.props.show_components if components is None else components
        return plot_torques(self._ts, body_idx=bi, components=comp, ax=ax)

    def plot_separation_instance(
        self, ax: Optional[Axes] = None, with_velocity: Optional[bool] = None
    ):
        if self._ts is None:
            raise RuntimeError(
                "no timeseries loaded; call _load_data(files) first"
            )
        wv = (
            self.props.with_velocity if with_velocity is None else with_velocity
        )
        return plot_separation(self._ts, with_velocity=wv, ax=ax)

    def plot_accretion_rate_instance(
        self,
        ax: Optional[Axes] = None,
        body_idx: Optional[int] = None,
        cumulative: Optional[bool] = None,
    ) -> Axes:
        if self._ts is None:
            raise RuntimeError(
                "no timeseries loaded; call _load_data(files) first"
            )
        bi = self.props.body_idx if body_idx is None else body_idx
        cum = self.props.cumulative if cumulative is None else cumulative
        return plot_accretion_rate(self._ts, body_idx=bi, cumulative=cum, ax=ax)

    def plot_orbital_elements_instance(self, ax: Optional[Axes] = None):
        if self._ts is None:
            raise RuntimeError(
                "no timeseries loaded; call _load_data(files) first"
            )
        return plot_orbital_elements(self._ts, ax=ax)

    def plot_radial_acceleration_instance(self, ax: Optional[Axes] = None):
        if self._ts is None:
            raise RuntimeError(
                "no timeseries loaded; call _load_data(files) first"
            )
        return plot_radial_acceleration(self._ts, ax=ax)

    def plot_body_diagnostics_summary_instance(
        self,
        body_idx: Optional[int] = None,
        figsize: tuple[float, float] = (14, 10),
    ):
        if self._ts is None:
            raise RuntimeError(
                "no timeseries loaded; call _load_data(files) first"
            )
        bi = self.props.body_idx if body_idx is None else body_idx
        return plot_body_diagnostics_summary(
            self._ts, body_idx=bi, figsize=figsize
        )

    def plot_binary_diagnostics_summary_instance(
        self, figsize: tuple[float, float] = (14, 10)
    ):
        if self._ts is None:
            raise RuntimeError(
                "no timeseries loaded; call _load_data(files) first"
            )
        if self._ts.n_bodies != 2:
            raise ValueError(f"expected 2 bodies, got {self._ts.n_bodies}")
        return plot_binary_diagnostics_summary(self._ts, figsize=figsize)

    def plot_radial_force_instance(
        self,
        ax: Optional[Axes] = None,
        body_idx: Optional[int] = None,
        normalize: Optional[Normalizations] = None,
    ) -> Axes:
        if self._ts is None:
            raise RuntimeError(
                "no timeseries loaded; call _load_data(files) first"
            )
        bi = body_idx
        norm = normalize if normalize is not None else self._norm
        return plot_radial_force(self._ts, body_idx=bi, normalize=norm, ax=ax)

    def plot_tangential_force_instance(
        self,
        ax: Optional[Axes] = None,
        body_idx: Optional[int] = None,
        normalize: Optional[Normalizations] = None,
    ) -> Axes:
        if self._ts is None:
            raise RuntimeError(
                "no timeseries loaded; call _load_data(files) first"
            )
        bi = body_idx
        norm = normalize if normalize is not None else self._norm
        return plot_tangential_force(
            self._ts, body_idx=bi, normalize=norm, ax=ax
        )

    def plot_torque_z_instance(self, ax: Optional[Axes] = None) -> Axes:
        if self._binary is None:
            if self._ts is None:
                raise RuntimeError(
                    "no timeseries loaded; call _load_data(files) first"
                )
            self._binary = compute_binary_dynamics(self._ts)
        return plot_torque_z(self._ts, ax=ax)

    def plot_power_instance(
        self,
        ax: Optional[Axes] = None,
        body_idx: Optional[int] = None,
        normalize: Optional[Normalizations] = None,
    ) -> Axes:
        if self._binary is None:
            if self._ts is None:
                raise RuntimeError(
                    "no timeseries loaded; call _load_data(files) first"
                )
            self._binary = compute_binary_dynamics(self._ts)
        bi = body_idx
        norm = normalize if normalize is not None else self._norm
        return plot_power(self._ts, body_idx=bi, normalize=norm, ax=ax)

    def plot_migration_timescale_instance(
        self, ax: Optional[Axes] = None
    ) -> Axes:
        if self._binary is None:
            if self._ts is None:
                raise RuntimeError(
                    "no timeseries loaded; call _load_data(files) first"
                )
            self._binary = compute_binary_dynamics(self._ts)
        return plot_migration_timescale(self._ts, ax=ax)

    def plot_drag_force_instance(
        self,
        ax: Optional[Axes] = None,
        body_idx: Optional[int] = None,
        normalize: Optional[Normalizations] = None,
    ) -> Axes:
        if self._binary is None:
            if self._ts is None:
                raise RuntimeError(
                    "no timeseries loaded; call _load_data(files) first"
                )
            self._binary = compute_binary_dynamics(self._ts)
        bi = body_idx
        norm = normalize if normalize is not None else self._norm
        return plot_drag_force(self._ts, body_idx=bi, normalize=norm, ax=ax)

    def plot_decay_rate_instance(
        self,
        ax: Optional[Axes] = None,
        normalize: Optional[Normalizations] = None,
    ) -> Axes:
        if self._binary is None:
            if self._ts is None:
                raise RuntimeError(
                    "no timeseries loaded; call _load_data(files) first"
                )
            self._binary = compute_binary_dynamics(self._ts)
        norm = normalize if normalize is not None else self._norm
        return plot_decay_rate(self._ts, normalize=norm, ax=ax)

    def apply_theme(self, theme_name: Optional[str] = None) -> None:
        """Apply named theme to the component's axes."""
        apply_theme_to_axes(self.ax, theme_name)


# =============================================================================
# plotting helpers (pure functions)
# =============================================================================


def filter_checkpoint_files(files: Sequence[str | Path]) -> list[Path]:
    """
    filter out interrupted/crashed checkpoints.

    excludes files with 'interrupted' or 'crashed' in filename.
    these represent incomplete simulation states and should not be
    included in time series analysis.
    """
    valid_files: list[Path] = []
    excluded_count = 0

    for filepath in files:
        path = Path(filepath)
        name_lower = path.name.lower()

        if "interrupted" in name_lower or "crashed" in name_lower:
            excluded_count += 1
            continue

        valid_files.append(path)

    if excluded_count > 0:
        print(f"excluded {excluded_count} interrupted/crashed checkpoint(s)")

    return valid_files


def plot_forces(
    ts: BodyTimeSeries,
    body_idx: int = 0,
    components: bool = True,
    ax: Optional[Axes] = None,
) -> Axes:
    """plot force components over time for one body."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    forces = ts.forces[:, body_idx, :]

    if components:
        labels = ["x", "y", "z"][: ts.ndim]
        for ii, label in enumerate(labels):
            ax.plot(ts.times, forces[:, ii], label=f"F_{label}", marker="o")
        ax.set_ylabel("force components")
        ax.legend()
    else:
        force_mag = np.linalg.norm(forces, axis=1)
        ax.plot(ts.times, force_mag, marker="o", color="black")
        ax.set_ylabel("force magnitude")

    ax.set_xlabel("time")
    ax.set_title(f"force on body {body_idx}")
    ax.grid(True, alpha=0.3)
    return ax


def plot_torques(
    ts: BodyTimeSeries,
    body_idx: int = 0,
    components: bool = True,
    ax: Optional[Axes] = None,
) -> Axes:
    """plot torque components over time for one body."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    torques = ts.torques[:, body_idx, :]

    if components:
        labels = ["x", "y", "z"]
        for ii, label in enumerate(labels):
            ax.plot(ts.times, torques[:, ii], label=f"τ_{label}", marker="o")
        ax.set_ylabel("torque components")
        ax.legend()
    else:
        torque_mag = np.linalg.norm(torques, axis=1)
        ax.plot(ts.times, torque_mag, marker="o", color="black")
        ax.set_ylabel("torque magnitude")

    ax.set_xlabel("time")
    ax.set_title(f"torque on body {body_idx}")
    ax.grid(True, alpha=0.3)
    return ax


def plot_separation(
    ts: BodyTimeSeries,
    with_velocity: bool = False,
    ax: Optional[Axes | Tuple[Axes, Axes]] = None,
) -> Any:
    """plot binary separation over time."""
    binary = compute_binary_dynamics(ts)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        binary.times, binary.separation, marker="o", color="C0", label="r(t)"
    )
    ax.set_xlabel("time")
    ax.set_ylabel("separation", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.grid(True, alpha=0.3)

    if with_velocity:
        ax2 = ax.twinx()
        ax2.plot(
            binary.times,
            binary.separation_velocity,
            marker="s",
            color="C1",
            linestyle="--",
            label="dr/dt",
        )
        ax2.set_ylabel("dr/dt", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
        ax.set_title("binary separation and rate of change")
        return ax, ax2
    else:
        ax.set_title("binary separation")
        return ax


def plot_accretion_rate(
    ts: BodyTimeSeries,
    body_idx: int = 0,
    cumulative: bool = False,
    ax: Optional[Axes] = None,
) -> Axes:
    """plot accretion rate or cumulative accreted mass."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    if cumulative:
        data = ts.accreted_masses[:, body_idx]
        ax.plot(ts.times, data, marker="o", color="C2")
        ax.set_ylabel("cumulative accreted mass")
        ax.set_title(f"total mass accreted by body {body_idx}")
    else:
        data = ts.accretion_rates[:, body_idx]
        ax.plot(ts.times, data, marker="o", color="C3")
        ax.set_ylabel("accretion rate (dm/dt)")
        ax.set_title(f"accretion rate for body {body_idx}")

    ax.set_xlabel("time")
    ax.grid(True, alpha=0.3)
    return ax


def plot_orbital_elements(
    ts: BodyTimeSeries, ax: Optional[Axes] = None
) -> Tuple[Axes, Axes]:
    """plot specific orbital energy and angular momentum over time."""
    binary = compute_binary_dynamics(ts)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        binary.times,
        binary.specific_energy,
        marker="o",
        color="C4",
        label="E/μ",
    )
    ax.set_xlabel("time")
    ax.set_ylabel("specific energy (E/μ)", color="C4")
    ax.tick_params(axis="y", labelcolor="C4")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(
        binary.times,
        binary.specific_angular_momentum,
        marker="s",
        color="C5",
        linestyle="--",
        label="L/μ",
    )
    ax2.set_ylabel("specific angular momentum (L/μ)", color="C5")
    ax2.tick_params(axis="y", labelcolor="C5")
    ax.set_title("orbital elements")
    return ax, ax2


def plot_radial_acceleration(
    ts: BodyTimeSeries, ax: Optional[Axes] = None
) -> Axes:
    """plot radial acceleration (d²r/dt²) over time."""
    binary = compute_binary_dynamics(ts)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.plot(binary.times, binary.radial_acceleration, marker="o", color="C6")
    ax.set_xlabel("time")
    ax.set_ylabel("radial acceleration (d²r/dt²)")
    ax.set_title("radial component of relative acceleration")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linestyle=":", alpha=0.5)
    return ax


def plot_body_diagnostics_summary(
    ts: BodyTimeSeries,
    body_idx: int = 0,
    figsize: tuple[float, float] = (14, 10),
) -> Figure:
    """create multi-panel summary plot for one body."""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f"body {body_idx} diagnostics summary", fontsize=14)

    plot_forces(ts, body_idx, components=True, ax=axes[0, 0])
    plot_forces(ts, body_idx, components=False, ax=axes[0, 1])
    plot_torques(ts, body_idx, components=True, ax=axes[0, 2])
    plot_accretion_rate(ts, body_idx, cumulative=False, ax=axes[1, 0])
    plot_accretion_rate(ts, body_idx, cumulative=True, ax=axes[1, 1])

    axes[1, 2].plot(ts.times, ts.masses[:, body_idx], marker="o", color="C7")
    axes[1, 2].set_xlabel("time")
    axes[1, 2].set_ylabel("mass")
    axes[1, 2].set_title("total mass")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_binary_diagnostics_summary(
    ts: BodyTimeSeries, figsize: tuple[float, float] = (14, 10)
) -> Figure:
    """create multi-panel summary plot for binary system."""
    if ts.n_bodies != 2:
        raise ValueError(f"expected 2 bodies, got {ts.n_bodies}")

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("binary diagnostics summary", fontsize=14)

    plot_separation(ts, with_velocity=False, ax=axes[0, 0])

    binary = compute_binary_dynamics(ts)
    axes[0, 1].plot(
        binary.times, binary.separation_velocity, marker="o", color="C1"
    )
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel("dr/dt")
    axes[0, 1].set_title("separation velocity")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(0, color="black", linestyle=":", alpha=0.5)

    plot_radial_acceleration(ts, ax=axes[0, 2])

    axes[1, 0].plot(
        binary.times, binary.orbital_frequency, marker="o", color="C8"
    )
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel("ω")
    axes[1, 0].set_title("orbital frequency")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(
        binary.times, binary.specific_energy, marker="o", color="C4"
    )
    axes[1, 1].set_xlabel("time")
    axes[1, 1].set_ylabel("E/μ")
    axes[1, 1].set_title("specific energy")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(
        binary.times, binary.specific_angular_momentum, marker="o", color="C5"
    )
    axes[1, 2].set_xlabel("time")
    axes[1, 2].set_ylabel("L/μ")
    axes[1, 2].set_title("specific angular momentum")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_radial_force(
    ts: BodyTimeSeries,
    body_idx: Optional[int] = None,
    normalize: Optional[Normalizations] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """plot radial force component F_r = F·r̂."""
    binary = compute_binary_dynamics(ts)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    scale = normalize.force_scale if normalize else 1.0
    ylabel = (
        "radial force F_r" if not normalize else "normalized radial force F_r"
    )

    if body_idx is None:
        ax.plot(
            binary.times,
            binary.radial_force[:, 0] / scale,
            marker="o",
            label="body 0",
        )
        ax.plot(
            binary.times,
            binary.radial_force[:, 1] / scale,
            marker="s",
            label="body 1",
        )
        ax.legend()
    else:
        ax.plot(
            binary.times,
            binary.radial_force[:, body_idx] / scale,
            marker="o",
            color="C0",
        )

    ax.axhline(0, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"radial force component {'(both bodies)' if body_idx is None else f'(body {body_idx})'}"
    )
    ax.grid(True, alpha=0.3)
    return ax


def plot_tangential_force(
    ts: BodyTimeSeries,
    body_idx: Optional[int] = None,
    normalize: Optional[Normalizations] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """plot tangential force component F_t (perpendicular to separation)."""
    binary = compute_binary_dynamics(ts)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    scale = normalize.force_scale if normalize else 1.0
    ylabel = (
        "tangential force F_t"
        if not normalize
        else "normalized tangential force F_t"
    )

    if body_idx is None:
        ax.plot(
            binary.times,
            binary.tangential_force[:, 0] / scale,
            marker="o",
            label="body 0",
        )
        ax.plot(
            binary.times,
            binary.tangential_force[:, 1] / scale,
            marker="s",
            label="body 1",
        )
        ax.legend()
    else:
        ax.plot(
            binary.times,
            binary.tangential_force[:, body_idx] / scale,
            marker="o",
            color="C1",
        )

    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"tangential force component {'(both bodies)' if body_idx is None else f'(body {body_idx})'}"
    )
    ax.grid(True, alpha=0.3)
    return ax


def plot_torque_z(ts: BodyTimeSeries, ax: Optional[Axes] = None) -> Axes:
    """plot z-component of total torque on binary system."""
    binary = compute_binary_dynamics(ts)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.plot(binary.times, binary.torque_z, marker="o", color="C2")
    ax.axhline(0, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("time")
    ax.set_ylabel("torque z-component τ_z")
    ax.set_title("total z-torque on binary system")
    ax.grid(True, alpha=0.3)

    mean_torque = binary.torque_z.mean()
    if abs(mean_torque) > 1e-10:
        trend = "outspiral" if mean_torque > 0 else "inspiral"
        ax.text(
            0.98,
            0.98,
            f"mean τ_z: {mean_torque:.2e}\n({trend})",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    return ax


def plot_power(
    ts: BodyTimeSeries,
    body_idx: Optional[int] = None,
    normalize: Optional[Normalizations] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """plot power dE/dt = F·v (energy dissipation rate)."""
    binary = compute_binary_dynamics(ts)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    power_scale = normalize.power_scale if normalize else 1.0
    time_scale = normalize.time_scale if normalize else 1.0
    ylabel = "power dE/dt" if not normalize else "normalized power dE/dt"

    if body_idx is None:
        ax.plot(
            binary.times / time_scale,
            binary.power[:, 0] / power_scale,
            marker="o",
            label="body 0",
        )
        ax.plot(
            binary.times / time_scale,
            binary.power[:, 1] / power_scale,
            marker="s",
            label="body 1",
        )
        total_power = (binary.power[:, 0] + binary.power[:, 1]) / power_scale
        ax.plot(
            binary.times / time_scale,
            total_power,
            marker="^",
            label="total",
            linestyle="--",
            color="black",
        )
        ax.legend()
    else:
        ax.plot(
            binary.times / time_scale,
            binary.power[:, body_idx] / power_scale,
            marker="o",
            color="C3",
        )

    ax.axhline(0, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"energy dissipation rate {'(both bodies)' if body_idx is None else f'(body {body_idx})'}"
    )
    ax.grid(True, alpha=0.3)
    return ax


def plot_migration_timescale(
    ts: BodyTimeSeries, ax: Optional[Axes] = None
) -> Axes:
    """plot migration timescale τ_mig = a / |da/dt|."""
    binary = compute_binary_dynamics(ts)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    tau_mig_clipped = np.clip(
        binary.migration_timescale,
        0,
        np.percentile(
            binary.migration_timescale[np.isfinite(binary.migration_timescale)],
            95,
        )
        * 2,
    )
    ax.plot(binary.times, tau_mig_clipped, marker="o", color="C4")
    ax.set_xlabel("time")
    ax.set_ylabel("migration timescale τ_mig")
    ax.set_title("time to merger at current inspiral rate")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")

    finite_vals = binary.migration_timescale[
        np.isfinite(binary.migration_timescale)
    ]
    if len(finite_vals) > 0:
        mean_tau = finite_vals.mean()
        ax.axhline(
            mean_tau,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"mean: {mean_tau:.2e}",
        )
        ax.legend()
    return ax


def plot_drag_force(
    ts: BodyTimeSeries,
    body_idx: Optional[int] = None,
    normalize: Optional[Normalizations] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """plot drag force (antiparallel to velocity): F_drag = -F·v̂."""
    binary = compute_binary_dynamics(ts)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    scale = normalize.force_scale if normalize else 1.0
    ylabel = "drag force F_drag" if not normalize else "normalized drag force"

    if body_idx is None:
        ax.plot(
            binary.times,
            binary.drag_force[:, 0] / scale,
            marker="o",
            label="body 0",
        )
        ax.plot(
            binary.times,
            binary.drag_force[:, 1] / scale,
            marker="s",
            label="body 1",
        )
        ax.legend()
    else:
        ax.plot(
            binary.times,
            binary.drag_force[:, body_idx] / scale,
            marker="o",
            color="C5",
        )

    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"drag force {'(both bodies)' if body_idx is None else f'(body {body_idx})'}"
    )
    ax.grid(True, alpha=0.3)
    return ax


def plot_decay_rate(
    ts: BodyTimeSeries,
    normalize: Optional[Normalizations] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """plot orbital decay rate da/dt from energy dissipation."""
    binary = compute_binary_dynamics(ts)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    if normalize:
        decay_normalized = binary.decay_rate / (
            normalize.length_scale / normalize.time_scale
        )
        ylabel = "normalized decay rate da/dt"
    else:
        decay_normalized = binary.decay_rate
        ylabel = "decay rate da/dt"

    ax.plot(binary.times, decay_normalized, marker="o", color="C6")
    ax.axhline(0, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.set_title("orbital decay rate (from energy loss)")
    ax.grid(True, alpha=0.3)

    mean_decay = decay_normalized.mean()
    if abs(mean_decay) > 1e-10:
        trend = "outspiral" if mean_decay > 0 else "inspiral"
        ax.text(
            0.98,
            0.98,
            f"mean da/dt: {mean_decay:.2e}\n({trend})",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    return ax


def apply_theme_to_axes(ax: Axes, theme: Optional[str] = None) -> None:
    """apply visualization theme to axes."""
    if theme is None:
        return

    from simbi.viz.styling.theme import ThemeConfig

    # load theme
    if theme == "scientific":
        from simbi.viz.styling.themes.scientific import scientific_theme

        theme_config = scientific_theme
    elif theme == "dark":
        from simbi.viz.styling.themes.dark import dark_theme

        theme_config = dark_theme
    else:
        theme_config = ThemeConfig()

    # apply theme styling
    theme_config.style_axis(ax)


__all__ = [
    "BodyDiagnosticsProps",
    "BodyDiagnosticsComponent",
    "BodyTimeSeries",
    "SingleBodyTimeSeries",
    "BinaryTimeSeries",
    "Normalizations",
    "filter_checkpoint_files",
    "compute_binary_dynamics",
    "load_body_timeseries",
    "plot_forces",
    "plot_torques",
    "plot_separation",
    "plot_accretion_rate",
    "plot_orbital_elements",
    "plot_radial_acceleration",
    "plot_body_diagnostics_summary",
    "plot_binary_diagnostics_summary",
    "plot_radial_force",
    "plot_tangential_force",
    "plot_torque_z",
    "plot_power",
    "plot_migration_timescale",
    "plot_drag_force",
    "plot_decay_rate",
    "apply_theme_to_axes",
]
