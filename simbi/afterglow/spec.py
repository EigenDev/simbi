# =============================================================================
# spec.py
#
# the dimensionful + observation spec for afterglow post-processing. the sim is
# SCALE-FREE; this module is the single source of dimensionful truth, split in two:
# - SystemManifest: the physical system (code->cgs scales, regime), auto-discovered
#   as `system.yaml` next to the checkpoints (config-emitted) or hand-written.
# - ObserverParams: the OBSERVATION choices (redshift, distance, microphysics,
#   frequencies) — a separate yaml so nobody types them on the command line.
# usage:
#  manifest = SystemManifest.resolve(checkpoint_path, scale_fallback="blandford-mckee")
#  observer = ObserverParams.resolve(observer_yaml)
#  qscales = manifest.to_qscales()
#  d_a_cm = observer.angular_diameter_distance_cm()
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from astropy import constants as _const

from .scales import get_scale_model

_C_CGS = _const.c.cgs.value
_RAD_TO_MAS = 206_264_806.247_096_36  # arcsec/rad * 1000
_JY_CGS = 1.0e-23  # jansky [erg/cm^2/s/Hz]

# yaml filenames auto-discovered next to the checkpoints/catalogs.
SYSTEM_MANIFEST_NAME = "system.yaml"
OBSERVER_PARAMS_NAME = "observer.yaml"


def calibrate_skymap(
    image,
    half_width_cm: float,
    observer: "ObserverParams",
    time_window_day: float,
    frequency_hz: float,
    frac_bandwidth: float = 0.1,
):
    """convert the raw skymap (beamed energy per cm^2 of sky plane) to physical units.

    the standard afterglow flux density F_nu = E / (4 pi d_L^2 dt dnu) [erg/cm^2/s/Hz];
    this mirrors `postprocess.compute_lightcurve` EXACTLY (same monochromatic, 10%-band
    approximation it already uses — not a spectrum convolution). returns:
      - surface brightness [mJy/mas^2], resolution-independent (the colorbar), and
      - the integrated flux density F_nu [mJy] over the image.
    """
    import numpy as np

    n_pix = image.shape[0]
    d_l = observer.luminosity_distance_cm()
    d_a = observer.angular_diameter_distance_cm()
    dt_s = time_window_day * 86400.0
    dnu = max(frequency_hz * frac_bandwidth, 1.0e-30)
    px_cm = 2.0 * half_width_cm / n_pix if half_width_cm > 0.0 else 1.0
    denom = 4.0 * np.pi * d_l * d_l * dt_s * dnu

    # per-pixel energy = intensity * pixel_area; integrate -> total flux density [mJy].
    flux_total_mjy = float(image.sum()) * px_cm * px_cm / denom / _JY_CGS * 1.0e3
    # surface brightness [mJy/mas^2]: F_nu,pixel / solid-angle; px / n_pix cancels.
    surface_brightness = (
        image * (d_a * d_a) / (_RAD_TO_MAS**2 * denom) / _JY_CGS * 1.0e3
    )
    return surface_brightness, flux_total_mjy


# =============================================================================
# system manifest (the physical system; scales are code->cgs)
# =============================================================================


@dataclass(frozen=True)
class SystemManifest:
    """the dimensionful description of a scale-free run: code->cgs scales plus a
    little physics provenance. `length_scale` [cm] and `density_scale` [g/cm^3] are
    the two independent anchors; velocity := c (relativistic code units), time :=
    length/c, pressure := density * c^2 are derived."""

    length_scale: float  # cm per code length
    density_scale: float  # g/cm^3 per code density
    regime: str = "srhd"
    physics: dict[str, Any] = field(default_factory=dict)

    @property
    def velocity_scale(self) -> float:
        return _C_CGS

    @property
    def time_scale(self) -> float:
        return self.length_scale / _C_CGS

    @property
    def pressure_scale(self) -> float:
        return self.density_scale * _C_CGS * _C_CGS

    def to_qscales(self) -> dict[str, float]:
        """the rust-binding `qscales` contract (code->cgs multipliers)."""
        return {
            "time": float(self.time_scale),
            "pre": float(self.pressure_scale),
            "rho": float(self.density_scale),
            "velocity": float(self.velocity_scale),
            "length": float(self.length_scale),
        }

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_named_model(cls, name: str) -> "SystemManifest":
        """build from a registered USER_SCALES model (e.g., 'blandford-mckee')."""
        m = get_scale_model(name)
        return cls(
            length_scale=float(m.length_scale.cgs.value),
            density_scale=float(m.rho_scale.cgs.value),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SystemManifest":
        """build from a parsed `system.yaml`. the `scales` block is either a named
        `model:` or explicit `length:`/`density:` [cgs]."""
        scales = data.get("scales", {}) or {}
        if "model" in scales:
            base = cls.from_named_model(str(scales["model"]))
            length = float(scales.get("length", base.length_scale))
            density = float(scales.get("density", base.density_scale))
        else:
            if "length" not in scales or "density" not in scales:
                raise ValueError(
                    "system manifest `scales` needs either `model:` or both "
                    "`length:` and `density:` (cgs)"
                )
            length = float(scales["length"])
            density = float(scales["density"])
        return cls(
            length_scale=length,
            density_scale=density,
            regime=str(data.get("regime", "srhd")),
            physics=dict(data.get("physics", {}) or {}),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SystemManifest":
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    @classmethod
    def discover(cls, near: str | Path) -> Optional["SystemManifest"]:
        """look for `system.yaml` in the directory holding the checkpoint/catalog."""
        manifest = Path(near).resolve().parent / SYSTEM_MANIFEST_NAME
        if manifest.is_file():
            return cls.from_yaml(manifest)
        return None

    @classmethod
    def resolve(
        cls, near: str | Path, scale_fallback: Optional[str] = None
    ) -> "SystemManifest":
        """auto-discover the manifest next to `near`; else fall back to a named scale
        model. raises if neither is available (no silent wrong units)."""
        found = cls.discover(near)
        if found is not None:
            return found
        if scale_fallback is not None:
            return cls.from_named_model(scale_fallback)
        raise FileNotFoundError(
            f"no {SYSTEM_MANIFEST_NAME} next to {near} and no scale fallback given; "
            "afterglow units are undefined"
        )


# =============================================================================
# observer params (how you OBSERVE the system — not a property of it)
# =============================================================================


@dataclass(frozen=True)
class ObserverParams:
    """the observation choices. redshift + distance set the angular/flux mapping;
    p/eps_e/eps_b are the synchrotron microphysics (used only when generating a
    catalog in-process); frequencies set the band."""

    redshift: float = 0.0
    luminosity_distance: Optional[float] = None  # cm; else derived from z
    p: float = 2.5
    eps_e: float = 0.1
    eps_b: float = 0.01
    frequencies: tuple[float, ...] = (1.0e9,)

    def luminosity_distance_cm(self) -> float:
        """explicit d_L if given; else from z via Planck18; z=0 -> a 10 pc reference."""
        if self.luminosity_distance is not None:
            return float(self.luminosity_distance)
        if self.redshift <= 0.0:
            return 3.085_677_581e19  # 10 pc in cm
        from astropy.cosmology import Planck18

        return float(Planck18.luminosity_distance(self.redshift).to("cm").value)

    def angular_diameter_distance_cm(self) -> float:
        """d_A = d_L / (1 + z)^2 — converts a projected length [cm] to an angle."""
        return self.luminosity_distance_cm() / (1.0 + self.redshift) ** 2

    def length_to_mas(self, length_cm: float) -> float:
        """a transverse proper length [cm] -> apparent angular size [mas]."""
        return length_cm / self.angular_diameter_distance_cm() * _RAD_TO_MAS

    def mas_to_length(self, mas: float) -> float:
        """apparent angular size [mas] -> transverse proper length [cm] (inverse of above)."""
        return mas / _RAD_TO_MAS * self.angular_diameter_distance_cm()

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObserverParams":
        obs = dict(data.get("observer", {}) or {})
        micro = dict(data.get("microphysics", {}) or {})
        observation = dict(data.get("observation", {}) or {})
        freqs = observation.get("frequencies", [1.0e9])
        return cls(
            redshift=float(obs.get("redshift", 0.0)),
            luminosity_distance=(
                float(obs["luminosity_distance"])
                if obs.get("luminosity_distance") is not None
                else None
            ),
            p=float(micro.get("p", 2.5)),
            eps_e=float(micro.get("eps_e", 0.1)),
            eps_b=float(micro.get("eps_b", 0.01)),
            frequencies=tuple(float(f) for f in freqs),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ObserverParams":
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    @classmethod
    def discover(cls, near: str | Path) -> Optional["ObserverParams"]:
        """look for `observer.yaml` in the directory holding the checkpoint/catalog."""
        path = Path(near).resolve().parent / OBSERVER_PARAMS_NAME
        if path.is_file():
            return cls.from_yaml(path)
        return None

    @classmethod
    def resolve(
        cls, path: Optional[str | Path] = None, near: Optional[str | Path] = None
    ) -> "ObserverParams":
        """an explicit `--observer` path wins; else auto-discover `observer.yaml` next to
        the data; else sane defaults (10 pc, p=2.5). symmetric with the system manifest."""
        if path is not None:
            return cls.from_yaml(path)
        if near is not None:
            found = cls.discover(near)
            if found is not None:
                return found
        return cls()
