#!/usr/bin/env python3
# =============================================================================
# fabric_campaign.py
#
# the validation campaign for `simbi launch`: one case table, one driver, one
# record per case. every case names a public example config, a resolution, a
# step count, and the tile layouts it runs; every arm is a subprocess of the
# public command line, so the campaign exercises exactly what an operator runs.
# arms of a case:
# - single      the single-grid run (the runner, fresh interpreter)
# - one         `simbi launch` on one worker
# - many        `simbi launch` on the case's uneven multi-worker layout
# - restart     `many` to half the steps, resumed onto a different layout
# - correction  a troubled cell against a cut, on one worker and on many
# - reject      a stage rejection reported by one worker, against one worker
#               driven through the same rejection (gas regimes and refined
#               magnetized cases); an unrefined magnetized kernel set carries no
#               complete rollback snapshot, so there the forced rejection is
#               refused by name and the arm records that
# - bad_cfl     an invalid timestep candidate: every worker exits nonzero and
#               no final checkpoint appears
# - lost        one worker ends as a lost node would: the survivors exit
#               nonzero and no final checkpoint appears
# - face        (magnetized) a perturbed shared face: the audit exits every
#               worker and publishes no checkpoint past the perturbation
# checks: every exit code, one backend hash across every arm, the final
# checkpoints equal interior cell for interior cell (bit for bit) across
# single, one, many and restart, the corrected many-worker run equal to the
# corrected one-worker run, the rejection count per worker, the conserved
# totals of a periodic case within roundoff of the initial checkpoint, the
# edge EMF the resumed magnetized run records, a non-default edge EMF changing
# the run against contact, and div B on a magnetized case. per case the driver writes result.json; the
# campaign writes provenance.txt (commit, dirty state, host, backend hash,
# scheduler job) and summary.json / summary.md.
#
# modes: local (workers are processes on this host) or scheduler (each arm is
# one srun inside the calling allocation, one task per worker, tasks record
# their host, cpus and exit). tiers: smoke (two cases) or full.
#
# usage:
#   python scripts/fabric_campaign.py /fresh/dir --tier smoke
#   sbatch --nodes=2 --ntasks-per-node=2 --cpus-per-task=16 --wrap \
#       "python scripts/fabric_campaign.py /shared/fresh-dir --mode scheduler --tier full"
# =============================================================================
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from math import prod
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = re.compile(r"backend=([0-9a-f]{40}(?:-dirty)?) config=")
EXIT = re.compile(r"launch: worker (\d+) exit (-?\d+)")
# a refined launch reports its counts and `refined=1` in place of the timing breakdown
SUMMARY = re.compile(
    r"SIMBI launch: worker=(\d+)/(\d+) steps=(\d+) rejections=(\d+) (?:compute=([\d.]+)s "
    r"collectives=([\d.]+)s exchange=([\d.]+)s final_checkpoint=([\d.]+)s|refined=1)"
)
AUDIT_ALL = (("SIMBI_FABRIC_AUDIT", "all"),)


@dataclass(frozen=True)
class Case:
    name: str
    module: str
    cls: str
    resolution: tuple[int, ...]
    steps: int
    tiers: tuple[str, ...]
    # the many-worker layout and the layout the restart resumes onto: interior cuts per axis
    cuts: tuple[tuple[int, ...], ...]
    restart_cuts: tuple[tuple[int, ...], ...]
    flags: tuple[str, ...] = ()
    # extra constructor arguments of the single-grid reference, `nu=0.05`
    fields: str = ""
    env: tuple[tuple[str, str], ...] = ()
    magnetized: bool = False
    # a cartesian magnetized case on a unit square: div B from the staggered faces
    div_b: bool = False
    # the domain wraps on every axis, so the conserved totals hold to roundoff
    periodic: bool = False
    # a troubled cell against a cut, as SIMBI_FABRIC_INJECT spells it
    trouble: str | None = None
    faults: tuple[str, ...] = ("reject",)
    # a refined case: the refined worker rolls a magnetized step back with its faces, so its
    # rejection arm compares against one worker in place of expecting a refusal
    refined: bool = False

    @property
    def script(self) -> str:
        return str(ROOT / (self.module.replace(".", "/") + ".py"))

    @property
    def ct_method(self) -> str:
        """the edge EMF a magnetized case runs under: the `--ct-method` flag, else contact"""
        if not self.magnetized:
            return ""
        return self.flags[self.flags.index("--ct-method") + 1] if "--ct-method" in self.flags else "contact"

    def under_contact(self) -> Case:
        """the same case under the contact edge EMF: the `--ct-method` flag and its field dropped"""
        at = self.flags.index("--ct-method")
        fields = ", ".join(f for f in self.fields.split(", ") if f and not f.startswith("ct_method="))
        return replace(self, flags=self.flags[:at] + self.flags[at + 2:], fields=fields)

    @property
    def dims(self) -> int:
        return len(self.cuts)

    def workers(self, cuts: tuple[tuple[int, ...], ...]) -> int:
        return prod(len(axis) + 1 for axis in cuts)


N = "simbi_configs.examples.newtonian"
CASES: tuple[Case, ...] = (
    Case("kh", f"{N}.kh", "KelvinHelmholtz", (32, 32), 8, ("smoke", "full"),
         cuts=((10, 20), ()), restart_cuts=((16,), (16,)), periodic=True,
         trouble="trouble:3:0:15:10", faults=("reject", "bad_cfl", "lost")),
    Case("nmhd_orszag_tang", f"{N}.nmhd_orszag_tang", "NewtonianOrszagTang", (32, 32, 1), 8, ("smoke", "full"),
         cuts=((16,), (16,)), restart_cuts=((10, 21), ()), env=AUDIT_ALL, magnetized=True, div_b=True, periodic=True,
         trouble="trouble:3:0:15:10", faults=("reject", "face")),
    Case("resistive_orszag_tang", f"{N}.resistive_orszag_tang", "ResistiveOrszagTang", (32, 32, 1), 8, ("full",),
         cuts=((16,), ()), restart_cuts=((10, 21), ()), env=AUDIT_ALL, magnetized=True, div_b=True, periodic=True,
         trouble="trouble:3:0:15:10", faults=("reject", "face")),
    Case("isothermal_kh", "simbi_configs.examples.isothermal.isothermal_kh", "IsothermalKelvinHelmholtz", (32, 32), 8, ("full",),
         cuts=((16,), ()), restart_cuts=((10, 21), ()), periodic=True, trouble="trouble:3:0:15:10"),
    Case("isothermal_mhd_orszag_tang", "simbi_configs.examples.isothermal.imhd_orszag_tang", "IsothermalOrszagTang", (32, 32, 1), 8, ("full",),
         cuts=((16,), ()), restart_cuts=((10, 21), ()), env=AUDIT_ALL, magnetized=True, div_b=True, periodic=True,
         trouble="trouble:3:0:15:10", faults=("reject", "face")),
    Case("viscous_shear", f"{N}.viscous_shear", "ViscousShear", (32, 32, 1), 8, ("full",),
         cuts=((16,), ()), restart_cuts=((10, 21), ()), flags=("--nu", "0.05"), fields="nu=0.05"),
    Case("gradient_box", f"{N}.gradient_box", "GradientBox", (32, 32), 8, ("full",),
         cuts=((16,), (16,)), restart_cuts=((10, 21), ()), trouble="trouble:3:1:16:31"),
    Case("rayleigh_taylor", f"{N}.rt", "RayleighTaylor", (16, 48), 8, ("full",),
         cuts=((8,), (24,)), restart_cuts=((), (16, 32)), trouble="trouble:3:0:7:0"),
    Case("nmhd_rotor", f"{N}.nmhd_rotor", "NewtonianRotor", (32, 32, 1), 8, ("full",),
         cuts=((16,), (16,)), restart_cuts=((10, 21), ()), env=AUDIT_ALL, magnetized=True, div_b=True,
         trouble="trouble:3:0:15:0", faults=("reject", "face")),
    Case("spherical_pulse", f"{N}.spherical_pulse", "SphericalPulse", (32, 32), 8, ("full",),
         cuts=((16,), (16,)), restart_cuts=((10, 21), ()), trouble="trouble:3:1:15:31"),
    Case("nmhd_rotor_cyl", f"{N}.nmhd_rotor_cyl", "NewtonianRotorCyl", (32, 32, 1), 8, ("full",),
         cuts=((16,), (16,)), restart_cuts=((7,), (16,)), env=AUDIT_ALL, magnetized=True,
         trouble="trouble:3:0:6:0", faults=("reject", "face")),
    # the rotor with an in-plane field, under contact and uct: the edge EMF acts on the faces
    Case("nmhd_rotor_cyl_inplane", f"{N}.nmhd_rotor_cyl", "NewtonianRotorCyl", (32, 32, 1), 8, ("full",),
         cuts=((7,), (16,)), restart_cuts=((10, 21), ()), env=AUDIT_ALL, magnetized=True,
         flags=("--br", "0.1", "--bphi", "0.05"), fields="br=0.1, bphi=0.05", trouble="trouble:3:0:6:0",
         faults=("reject", "bad_cfl", "lost")),
    Case("nmhd_rotor_cyl_uct", f"{N}.nmhd_rotor_cyl", "NewtonianRotorCyl", (32, 32, 1), 8, ("full",),
         cuts=((7,), (16,)), restart_cuts=((10, 21), ()), env=AUDIT_ALL, magnetized=True,
         flags=("--br", "0.1", "--bphi", "0.05", "--ct-method", "uct"), fields="br=0.1, bphi=0.05, ct_method='uct'",
         trouble="trouble:3:0:6:0", faults=("reject", "bad_cfl", "lost")),
    Case("marti_muller_3d", "simbi_configs.examples.srhd.marti_muller_3d", "MartiMuller3D", (16, 16, 16), 8, ("full",),
         cuts=((8,), (), (8,)), restart_cuts=((8,), (), ()), trouble="trouble:3:0:7:3:8"),
    Case("tabulated_source_3d", f"{N}.tabulated_source_3d", "TabulatedSource3D", (12, 10, 8), 8, ("full",),
         cuts=((6,), (), (4,)), restart_cuts=((6,), (), ()), trouble="trouble:3:1:5:4:3"),
    Case("spherical_pulse_3d", f"{N}.spherical_pulse_3d", "SphericalPulse3D", (16, 16, 16), 8, ("full",),
         cuts=((), (8,), (8,)), restart_cuts=((), (), (6, 10)), trouble="trouble:3:0:5:0:7"),
    Case("ram44", "simbi_configs.examples.srhd.ram44", "Ram44", (32, 32), 8, ("full",),
         cuts=((16,), ()), restart_cuts=((10, 21), ())),
    # the isothermal regimes behind walls and the isothermal vortex under resistivity
    Case("isothermal_channel", "simbi_configs.examples.isothermal.isothermal_channel", "IsothermalChannel",
         (32, 32), 8, ("full",), cuts=((16,), (16,)), restart_cuts=((10, 21), ()), trouble="trouble:3:0:15:0",
         faults=("reject", "bad_cfl", "lost")),
    Case("isothermal_gradient_box", "simbi_configs.examples.isothermal.isothermal_gradient_box",
         "IsothermalGradientBox", (32, 32), 8, ("full",), cuts=((16,), (16,)), restart_cuts=((10, 21), ()),
         trouble="trouble:3:1:16:31", faults=("reject", "bad_cfl", "lost")),
    Case("imhd_orszag_tang_walls", "simbi_configs.examples.isothermal.imhd_orszag_tang_walls",
         "IsothermalOrszagTangWalls", (32, 32, 1), 8, ("full",), cuts=((16,), (16,)), restart_cuts=((10, 21), ()),
         trouble="trouble:3:0:15:0", magnetized=True, faults=("reject", "bad_cfl", "lost")),
    Case("resistive_imhd_orszag_tang", "simbi_configs.examples.isothermal.resistive_imhd_orszag_tang",
         "ResistiveIsothermalOrszagTang", (32, 32, 1), 8, ("full",), cuts=((16,), (16,)), restart_cuts=((10, 21), ()),
         trouble="trouble:3:0:15:15", magnetized=True, periodic=True, div_b=True, faults=("reject", "bad_cfl", "lost")),
    # the constant-nu viscosity in the isothermal and magnetized regimes: a shear with a cross shear
    Case("viscous_shear_isothermal", "simbi_configs.examples.isothermal.viscous_shear_isothermal",
         "ViscousShearIsothermal", (32, 32), 8, ("full",), cuts=((16,), (16,)), restart_cuts=((10, 21), ()),
         trouble="trouble:3:0:15:15", periodic=True, faults=("reject", "bad_cfl", "lost")),
    Case("viscous_shear_mhd", f"{N}.viscous_shear_mhd", "ViscousShearMhd", (32, 32, 1), 8, ("full",),
         cuts=((16,), (16,)), restart_cuts=((10, 21), ()), trouble="trouble:3:0:15:15", magnetized=True,
         periodic=True, faults=("reject", "bad_cfl", "lost")),
    Case("viscous_shear_imhd", "simbi_configs.examples.isothermal.viscous_shear_imhd", "ViscousShearImhd",
         (32, 32, 1), 8, ("full",), cuts=((16,), (16,)), restart_cuts=((10, 21), ()),
         trouble="trouble:3:0:15:15", magnetized=True, periodic=True, faults=("reject", "bad_cfl", "lost")),
    # the curvilinear grids for newtonian and relativistic hydro and newtonian MHD
    Case("rz_pulse", f"{N}.rz_pulse", "RzPulse", (32, 32), 8, ("full",), cuts=((16,), (16,)),
         restart_cuts=((10, 21), ()), trouble="trouble:3:0:15:15", faults=("reject", "bad_cfl", "lost")),
    Case("rhd_spherical_pulse_3d", "simbi_configs.examples.srhd.rhd_spherical_pulse_3d", "RhdSphericalPulse3D",
         (16, 16, 16), 8, ("full",), cuts=((), (8,), (8,)), restart_cuts=((), (), (6, 10)),
         trouble="trouble:3:0:5:0:7", faults=("reject", "bad_cfl", "lost")),
    Case("rhd_cylindrical_pulse_3d", "simbi_configs.examples.srhd.rhd_cylindrical_pulse_3d", "RhdCylindricalPulse3D",
         (16, 16, 16), 8, ("full",), cuts=((8,), (), (8,)), restart_cuts=((5, 11), (), ()),
         trouble="trouble:3:0:7:3:7", faults=("reject", "bad_cfl", "lost")),
    Case("nmhd_jet_rz", f"{N}.nmhd_jet_rz", "NewtonianJetRz", (32, 32, 1), 8, ("full",), cuts=((7,), (16,)),
         restart_cuts=((10, 21), ()), env=AUDIT_ALL, magnetized=True, trouble="trouble:3:0:6:0",
         faults=("reject", "bad_cfl", "lost")),
    Case("nmhd_jet_rz_uct", f"{N}.nmhd_jet_rz", "NewtonianJetRz", (32, 32, 1), 8, ("full",), cuts=((7,), (16,)),
         restart_cuts=((10, 21), ()), env=AUDIT_ALL, magnetized=True, flags=("--ct-method", "uct"),
         fields="ct_method='uct'", trouble="trouble:3:0:6:0", faults=("reject", "bad_cfl", "lost")),
    Case("nmhd_spherical_field", f"{N}.nmhd_spherical_field", "NewtonianSphericalField", (32, 32, 1), 8, ("full",),
         cuts=((16,), (16,)), restart_cuts=((10, 21), ()), env=AUDIT_ALL, magnetized=True,
         trouble="trouble:3:1:15:31", faults=("reject", "bad_cfl", "lost")),
    Case("nmhd_spherical_field_uct", f"{N}.nmhd_spherical_field", "NewtonianSphericalField", (32, 32, 1), 8,
         ("full",), cuts=((16,), (16,)), restart_cuts=((10, 21), ()), env=AUDIT_ALL, magnetized=True,
         flags=("--ct-method", "uct"), fields="ct_method='uct'", trouble="trouble:3:1:15:31",
         faults=("reject", "bad_cfl", "lost")),
    # the 3d shell through both poles, its azimuth cut, so every axis ring spans two tiles
    Case("nmhd_polar_shell_3d", f"{N}.nmhd_polar_shell_3d", "NewtonianPolarShell3D", (8, 8, 16), 8, ("full",),
         cuts=((), (4,), (8,)), restart_cuts=((), (), (5, 11)), env=AUDIT_ALL, magnetized=True,
         trouble="trouble:3:0:3:0:7", faults=("reject", "bad_cfl", "lost")),
    Case("nmhd_polar_shell_3d_uct", f"{N}.nmhd_polar_shell_3d", "NewtonianPolarShell3D", (8, 8, 16), 8, ("full",),
         cuts=((), (4,), (8,)), restart_cuts=((), (), (5, 11)), env=AUDIT_ALL, magnetized=True,
         flags=("--ct-method", "uct"), fields="ct_method='uct'", trouble="trouble:3:0:3:0:7",
         faults=("reject", "bad_cfl", "lost")),
    Case("imhd_polar_shell_3d", "simbi_configs.examples.isothermal.imhd_polar_shell_3d", "IsothermalPolarShell3D",
         (8, 8, 16), 8, ("full",), cuts=((), (4,), (8,)), restart_cuts=((), (), (5, 11)), env=AUDIT_ALL,
         magnetized=True, trouble="trouble:3:0:3:0:7", faults=("reject", "bad_cfl", "lost")),
    Case("imhd_polar_shell_3d_uct", "simbi_configs.examples.isothermal.imhd_polar_shell_3d", "IsothermalPolarShell3D",
         (8, 8, 16), 8, ("full",), cuts=((), (4,), (8,)), restart_cuts=((), (), (5, 11)), env=AUDIT_ALL,
         magnetized=True, flags=("--ct-method", "uct"), fields="ct_method='uct'", trouble="trouble:3:0:3:0:7",
         faults=("reject", "bad_cfl", "lost")),
    # special-relativistic MHD, the relativistic orszag-tang vortex
    Case("rmhd_orszag_tang", "simbi_configs.examples.srmhd.rmhd_orszag_tang", "OrszagTang", (32, 32, 1), 8,
         ("full",), cuts=((16,), (16,)), restart_cuts=((10, 21), ()), env=AUDIT_ALL, magnetized=True,
         div_b=True, periodic=True, trouble="trouble:3:0:15:15", faults=("reject", "bad_cfl", "lost")),
    Case("rmhd_orszag_tang_uct", "simbi_configs.examples.srmhd.rmhd_orszag_tang", "OrszagTang", (32, 32, 1), 8,
         ("full",), cuts=((16,), (16,)), restart_cuts=((10, 21), ()), env=AUDIT_ALL, magnetized=True,
         div_b=True, periodic=True, flags=("--ct-method", "uct"), fields="ct_method='uct'", trouble="trouble:3:0:15:15",
         faults=("reject", "bad_cfl", "lost")),
    Case("isothermal_spherical_pulse_3d", "simbi_configs.examples.isothermal.isothermal_spherical_pulse_3d",
         "IsothermalSphericalPulse3D", (16, 16, 16), 8, ("full",), cuts=((), (8,), (8,)),
         restart_cuts=((), (), (6, 10)), trouble="trouble:3:0:5:0:7", faults=("reject", "bad_cfl", "lost")),
    Case("imhd_jet_rz_uct", "simbi_configs.examples.isothermal.imhd_jet_rz", "IsothermalJetRz", (32, 32, 1), 8,
         ("full",), cuts=((7,), (16,)), restart_cuts=((10, 21), ()), env=AUDIT_ALL, magnetized=True,
         flags=("--ct-method", "uct"), fields="ct_method='uct'", trouble="trouble:3:0:6:0",
         faults=("reject", "bad_cfl", "lost")),
    Case("imhd_spherical_field", "simbi_configs.examples.isothermal.imhd_spherical_field", "IsothermalSphericalField",
         (32, 32, 1), 8, ("full",), cuts=((16,), (16,)), restart_cuts=((10, 21), ()), env=AUDIT_ALL,
         magnetized=True, trouble="trouble:3:1:15:31", faults=("reject", "bad_cfl", "lost")),
    # first-order space (the plm kernel at a zero slope) under newtonian MHD; a first-order redo
    # equals the pcm update, so the case carries no correction arm
    Case("nmhd_orszag_tang_pcm", f"{N}.nmhd_orszag_tang", "NewtonianOrszagTang", (32, 32, 1), 8, ("full",),
         cuts=((16,), (16,)), restart_cuts=((10, 21), ()), env=AUDIT_ALL, magnetized=True, div_b=True,
         periodic=True, flags=("--reconstruction", "pcm"), fields="reconstruction='pcm'",
         faults=("reject", "bad_cfl", "lost")),
    # the constant-nu viscosity on the curvilinear grids and in 3d
    Case("viscous_spherical_pulse", f"{N}.viscous_spherical_pulse", "ViscousSphericalPulse", (32, 32), 8, ("full",),
         cuts=((16,), (16,)), restart_cuts=((10, 21), ()), trouble="trouble:3:1:15:31",
         faults=("reject", "bad_cfl", "lost")),
    Case("viscous_imhd_rotor_cyl", "simbi_configs.examples.isothermal.viscous_imhd_rotor_cyl",
         "ViscousIsothermalRotorCyl", (32, 32, 1), 8, ("full",), cuts=((7,), (16,)), restart_cuts=((10, 21), ()),
         env=AUDIT_ALL, magnetized=True, trouble="trouble:3:0:6:0", faults=("reject", "bad_cfl", "lost")),
    Case("viscous_rz_pulse", f"{N}.viscous_rz_pulse", "ViscousRzPulse", (32, 32), 8, ("full",),
         cuts=((16,), (16,)), restart_cuts=((10, 21), ()), trouble="trouble:3:0:15:15",
         faults=("reject", "bad_cfl", "lost")),
    Case("viscous_isothermal_cylindrical_pulse_3d",
         "simbi_configs.examples.isothermal.viscous_isothermal_cylindrical_pulse_3d",
         "ViscousIsothermalCylindricalPulse3D", (16, 16, 16), 8, ("full",), cuts=((8,), (), (8,)),
         restart_cuts=((5, 11), (), ()), trouble="trouble:3:0:7:3:7", faults=("reject", "bad_cfl", "lost")),
    Case("viscous_nmhd_spherical_field", f"{N}.viscous_nmhd_spherical_field", "ViscousNewtonianSphericalField",
         (32, 32, 1), 8, ("full",), cuts=((16,), (16,)), restart_cuts=((10, 21), ()), env=AUDIT_ALL,
         magnetized=True, trouble="trouble:3:1:15:31", faults=("reject", "bad_cfl", "lost")),
    Case("viscous_nmhd_orszag_tang_3d", f"{N}.viscous_nmhd_orszag_tang_3d", "ViscousNewtonianOrszagTang3D",
         (16, 16, 16), 8, ("full",), cuts=((8,), (), (8,)), restart_cuts=((5, 11), (), ()),
         trouble="trouble:3:0:7:3:7", magnetized=True, periodic=True, faults=("reject", "bad_cfl", "lost")),
    # the isothermal regimes in 3d: a three-axis shear, and the vortex with a z flow
    Case("isothermal_shear_3d", "simbi_configs.examples.isothermal.isothermal_shear_3d", "IsothermalShear3D",
         (16, 16, 16), 8, ("full",), cuts=((8,), (), (8,)), restart_cuts=((5, 11), (), ()),
         trouble="trouble:3:0:7:3:7", periodic=True, faults=("reject", "bad_cfl", "lost")),
    Case("imhd_orszag_tang_3d", "simbi_configs.examples.isothermal.imhd_orszag_tang_3d", "IsothermalOrszagTang3D",
         (16, 16, 16), 8, ("full",), cuts=((8,), (), (8,)), restart_cuts=((5, 11), (), ()),
         trouble="trouble:3:0:7:3:7", magnetized=True, periodic=True, faults=("reject", "bad_cfl", "lost")),
    # the isothermal regimes on the curvilinear charts: the log-radial shell spanning both poles,
    # and the rotating core on the (r, phi) disk plane with a periodic azimuth
    Case("isothermal_spherical_pulse", "simbi_configs.examples.isothermal.isothermal_spherical_pulse",
         "IsothermalSphericalPulse", (32, 32), 8, ("full",), cuts=((16,), (16,)), restart_cuts=((10, 21), ()),
         trouble="trouble:3:1:15:31", faults=("reject", "bad_cfl", "lost")),
    Case("isothermal_disk_cyl", "simbi_configs.examples.isothermal.isothermal_disk_cyl", "IsothermalDiskCyl",
         (32, 32), 8, ("full",), cuts=((7,), (16,)), restart_cuts=((10, 21), ()), trouble="trouble:3:0:6:0",
         faults=("reject", "bad_cfl", "lost")),
    Case("imhd_rotor_cyl", "simbi_configs.examples.isothermal.imhd_rotor_cyl", "IsothermalRotorCyl",
         (32, 32, 1), 8, ("full",), cuts=((7,), (16,)), restart_cuts=((10, 21), ()), env=AUDIT_ALL,
         magnetized=True, trouble="trouble:3:0:6:0", faults=("reject", "bad_cfl", "lost")),
    Case("imhd_rotor_cyl_uct", "simbi_configs.examples.isothermal.imhd_rotor_cyl", "IsothermalRotorCyl",
         (32, 32, 1), 8, ("full",), cuts=((7,), (16,)), restart_cuts=((10, 21), ()), env=AUDIT_ALL,
         flags=("--ct-method", "uct"), fields="ct_method='uct'", magnetized=True, trouble="trouble:3:0:6:0",
         faults=("reject", "bad_cfl", "lost")),
    # geometric widths on both axes under a tabulated source, and the synge gas blast
    Case("geometric_tabulated", f"{N}.decomposed_tabulated_geometric", "DecomposedTabulatedGeometric", (32, 16), 8,
         ("full",), cuts=((16,), (8,)), restart_cuts=((10, 21), ()), trouble="trouble:3:0:15:7",
         faults=("reject", "bad_cfl", "lost")),
    Case("synge_blast", "simbi_configs.examples.srhd.synge_blast", "SyngeBlast", (32, 32), 8, ("full",),
         cuts=((16,), (16,)), restart_cuts=((10, 21), ()), trouble="trouble:3:0:15:15",
         faults=("reject", "bad_cfl", "lost")),
    # the 3d vortex with a z flow on a periodic box: cuts along x and z, the correction against both
    Case("nmhd_orszag_tang_3d", f"{N}.nmhd_orszag_tang_3d", "NewtonianOrszagTang3D", (16, 16, 16), 8, ("full",),
         cuts=((8,), (), (8,)), restart_cuts=((5, 11), (), ()), trouble="trouble:3:0:7:3:7", magnetized=True,
         periodic=True, faults=("reject", "bad_cfl", "lost")),
    # one refined box (root cells 8..24 at ratio 2) that the cuts cross; the correction arm flags
    # a root cell against the x = 13 cut just below the box
    Case("refined_blast", f"{N}.refined_blast", "RefinedBlast", (32, 32, 1), 8, ("full",),
         cuts=((13,), (18,)), restart_cuts=((16,), ()), trouble="trouble:3:0:12:7",
         faults=("reject", "bad_cfl", "lost"), refined=True),
    Case("refined_mhd_blast", f"{N}.refined_mhd_blast", "RefinedMhdBlast", (32, 32, 1), 8, ("full",),
         cuts=((13,), (18,)), restart_cuts=((16,), ()), trouble="trouble:7:0:12:7", magnetized=True,
         faults=("reject", "bad_cfl", "lost"), refined=True),
    Case("refined_mhd_blast_uct", f"{N}.refined_mhd_blast", "RefinedMhdBlast", (32, 32, 1), 8, ("full",),
         cuts=((13,), (18,)), restart_cuts=((16,), ()), flags=("--ct-method", "uct"), fields="ct_method='uct'",
         trouble="trouble:7:0:12:7", magnetized=True, faults=("reject", "bad_cfl", "lost"), refined=True),
)


# ---- one arm: a command, its streams, its records ----------------------------
@dataclass
class Arm:
    name: str
    code: int | None
    stdout: str
    stderr: str
    exits: dict[int, int]
    backends: set[str]
    summaries: dict[int, dict]
    hosts: dict[int, str]
    seconds: float
    data: Path


class Driver:
    def __init__(self, out: Path, mode: str, limit: float, keep_data: bool) -> None:
        self.out = out
        self.mode = mode
        self.limit = limit
        self.keep_data = keep_data

    def layout(self, path: Path, cuts: tuple[tuple[int, ...], ...], directory: Path) -> Path:
        lines = ["[execution]", f"workers = {prod(len(a) + 1 for a in cuts)}"]
        if self.mode == "scheduler":
            lines += ['mode = "scheduler"', 'bind = "0.0.0.0"']
        lines += ["", "[partition]", "cuts = [" + ", ".join("[" + ", ".join(map(str, a)) + "]" for a in cuts) + "]"]
        lines += ["", "[checkpoint]", "staging_mb = 4", f'directory = "{directory}"']
        path.write_text("\n".join(lines) + "\n")
        return path

    def reference(self, case: Case, where: Path) -> Arm:
        """the single-grid run in a fresh interpreter: an example that seeds a perturbation
        once per process must draw the same realization as each worker process does"""
        where.mkdir(parents=True)
        extra = f", {case.fields}" if case.fields else ""
        snippet = (
            "import sys\n"
            "from simbi.simulation import runner\n"
            f"from {case.module} import {case.cls}\n"
            f"runner.run({case.cls}(resolution={case.resolution!r}, data_directory=sys.argv[1], "
            f"checkpoint_interval=1e9{extra}), compute_mode='cpu', max_steps={case.steps})\n"
        )
        env = dict(os.environ, **dict(case.env))
        start = time.time()
        done = subprocess.run([sys.executable, "-c", snippet, str(where / "data")], capture_output=True, text=True,
                              timeout=self.limit, env=env, cwd=ROOT)
        seconds = time.time() - start
        (where / "stdout").write_text(done.stdout)
        (where / "stderr").write_text(done.stderr)
        return Arm("single", done.returncode, done.stdout, done.stderr, {0: done.returncode},
                   set(PROVENANCE.findall(done.stderr)), _summaries(done.stderr), {0: socket.gethostname()},
                   seconds, where / "data")

    def launch(self, case: Case, name: str, where: Path, cuts: tuple[tuple[int, ...], ...], steps: int,
               checkpoint: Path | None = None, env: dict[str, str] | None = None) -> Arm:
        where.mkdir(parents=True)
        data = where / "data"
        layout = self.layout(where / "layout.toml", cuts, data)
        workers = case.workers(cuts)
        cmd = [sys.executable, "-m", "simbi.cli", "launch", case.script, "--layout", str(layout),
               "--max-steps", str(steps), "--resolution", ",".join(map(str, case.resolution)),
               "--data-directory", str(data), "--checkpoint-interval", "1e9", *case.flags]
        if checkpoint is not None:
            cmd += ["--checkpoint", str(checkpoint)]
        full_env = dict(os.environ, **dict(case.env), **(env or {}))
        (where / "command").write_text(" ".join(cmd) + "\n" + "".join(f"{k}={v}\n" for k, v in (env or {}).items()))
        start = time.time()
        if self.mode == "local":
            try:
                done = subprocess.run(cmd, capture_output=True, text=True, timeout=self.limit, env=full_env, cwd=ROOT)
                code, stdout, stderr = done.returncode, done.stdout, done.stderr
            except subprocess.TimeoutExpired as expired:
                code, stdout, stderr = None, str(expired.stdout or ""), str(expired.stderr or "") + "\ncampaign: timed out"
            exits = {int(w): int(c) for w, c in EXIT.findall(stdout)}
            hosts = {w: socket.gethostname() for w in exits}
        else:
            code, stdout, stderr, exits, hosts = self._srun(where, cmd, workers, full_env)
        seconds = time.time() - start
        (where / "stdout").write_text(stdout)
        (where / "stderr").write_text(stderr)
        return Arm(name, code, stdout, stderr, exits, set(PROVENANCE.findall(stderr)), _summaries(stderr), hosts, seconds, data)

    def _srun(self, where: Path, cmd: list[str], workers: int, env: dict[str, str]):
        """one task per worker inside the calling allocation; each records its host, cpus,
        streams and exit under records/"""
        records = where / "records"
        records.mkdir()
        exported = "".join(f"export {k}={_quote(v)}\n" for k, v in env.items() if k.startswith("SIMBI_"))
        task = where / "task.sh"
        task.write_text(
            "#!/usr/bin/env bash\nset -uo pipefail\n"
            f"cd {_quote(str(ROOT))}\n"
            'rank="${SLURM_PROCID:?the task step runs under srun}"\n'
            f"{exported}"
            f"hostname > {_quote(str(records))}/worker-$rank.host\n"
            "{ echo \"nproc: $(nproc)\"; grep -i cpus_allowed_list /proc/self/status 2>/dev/null; "
            "echo \"SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}\"; } "
            f"> {_quote(str(records))}/worker-$rank.cpus\n"
            + " ".join(_quote(c) for c in cmd)
            + f" > {_quote(str(records))}/worker-$rank.out 2> {_quote(str(records))}/worker-$rank.err\n"
            "code=$?\n"
            f"echo $code > {_quote(str(records))}/worker-$rank.exit\n"
            "exit $code\n"
        )
        task.chmod(0o755)
        srun = ["srun", f"--ntasks={workers}", f"--cpus-per-task={os.environ.get('SLURM_CPUS_PER_TASK', '1')}",
                "--kill-on-bad-exit=0", str(task)]
        try:
            done = subprocess.run(srun, capture_output=True, text=True, timeout=self.limit, env=env, cwd=ROOT)
            code, stdout, stderr = done.returncode, done.stdout, done.stderr
        except subprocess.TimeoutExpired as expired:
            code, stdout, stderr = None, str(expired.stdout or ""), str(expired.stderr or "") + "\ncampaign: timed out"
        exits, hosts = {}, {}
        for w in range(workers):
            exit_file, host_file, err_file = (records / f"worker-{w}.{s}" for s in ("exit", "host", "err"))
            if exit_file.exists():
                exits[w] = int(exit_file.read_text().strip() or -1)
            if host_file.exists():
                hosts[w] = host_file.read_text().strip()
            if err_file.exists():
                stderr += f"\n--- worker {w} stderr\n" + err_file.read_text(errors="replace")
        return code, stdout, stderr, exits, hosts


def _quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def _seconds(group: str | None) -> float | None:
    return None if group is None else float(group)


def _summaries(stderr: str) -> dict[int, dict]:
    out = {}
    for m in SUMMARY.finditer(stderr):
        out[int(m.group(1))] = {
            "workers": int(m.group(2)), "steps": int(m.group(3)), "rejections": int(m.group(4)),
            "compute_s": _seconds(m.group(5)), "collectives_s": _seconds(m.group(6)), "exchange_s": _seconds(m.group(7)),
            "final_checkpoint_s": _seconds(m.group(8)),
        }
    return out


# ---- checkpoints -------------------------------------------------------------
def final_of(data: Path) -> Path | None:
    found = sorted(data.glob("*final*.h5")) if data.exists() else []
    return found[0] if len(found) == 1 else None


def initial_of(data: Path) -> Path | None:
    found = [p for p in sorted(data.glob("*.h5")) if "final" not in p.name] if data.exists() else []
    for p in found:
        with h5py.File(p, "r") as f:
            if int(f["level_0"].attrs["iteration"]) == 0:
                return p
    return None


def interior(path: Path) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """every cell dataset of every level trimmed to its level's interior (a face field over
    its interior faces), keyed by level, and the root level's attributes"""
    with h5py.File(path, "r") as f:
        fields = {}
        levels = sorted(k for k in f.keys() if k.startswith("level_"))
        for level_name in levels:
            level = f[level_name]
            ng = int(level["mesh"].attrs["halo_width"])
            names: list[str] = []
            level.visit(lambda n, names=names, level=level: names.append(n) if isinstance(level[n], h5py.Dataset) else None)
            for n in names:
                if n.startswith("mesh/") or n.startswith("partition_0/owned") or "/domain/" in n:
                    continue
                data = level[n][()]
                if "/magnetic/" in n:
                    fields[f"{level_name}/{n}"] = data
                elif data.ndim >= 2:
                    fields[f"{level_name}/{n}"] = data[tuple(slice(ng, -ng) for _ in range(data.ndim))]
        attrs = {k: float(f["level_0"].attrs[k]) for k in ("time", "iteration")}
    return fields, attrs


def recorded_ct_method(path: Path) -> str | None:
    """the edge EMF a checkpoint records it was evolved under, wherever the file keeps it"""
    found: list[str] = []
    with h5py.File(path, "r") as f:
        for obj in [f, *(f[k] for k in f.keys())]:
            if "ct_method" in obj.attrs:
                value = obj.attrs["ct_method"]
                found.append(value.decode() if isinstance(value, bytes) else str(value))
    return found[0] if found else None


def totals(path: Path) -> dict[str, tuple[float, float]]:
    """the sum and the absolute sum of every conserved dataset over the interior"""
    fields, _ = interior(path)
    # the root level alone: a refined level's cells cover root cells the restriction already holds
    return {n: (float(np.sum(x, dtype=np.float64)), float(np.sum(np.abs(x), dtype=np.float64)))
            for n, x in fields.items() if "cons" in n and n.startswith("level_0/")}


def _vector_scale(start: dict[str, tuple[float, float]], name: str) -> float:
    """the scale a conserved total's drift is measured against: its own initial absolute sum, or
    for a vector component (`m1`, `b3`) the largest initial absolute sum among the components of
    its vector, so a component that starts at zero is weighed against the field it belongs to"""
    stem = name.rstrip("0123456789")
    if stem == name:
        return start[name][1]
    return max(v[1] for n, v in start.items() if n.rstrip("0123456789") == stem)


def max_div_b(path: Path) -> float:
    """the largest |div B| over the cells of a checkpoint from its staggered faces, on a
    unit square with nx cells across"""
    with h5py.File(path, "r") as f:
        b1 = f["level_0/partition_0/hydro/magnetic/B1/data"][()]
        b2 = f["level_0/partition_0/hydro/magnetic/B2/data"][()]
    ny, nx1 = b1.shape
    nx = nx1 - 1
    if b2.shape != (ny + 1, nx):
        raise ValueError(f"face shapes {b1.shape}, {b2.shape}")
    h = 1.0 / nx
    div = (b1[:, 1:] - b1[:, :-1]) / h + (b2[1:, :] - b2[:-1, :]) / h
    return float(np.abs(div).max())


def same_state(a: Path, b: Path) -> str | None:
    """the first difference between two final checkpoints, or None when they are equal bit
    for bit over every interior dataset"""
    xs, ax = interior(a)
    ys, bx = interior(b)
    for k in ("time", "iteration"):
        if ax[k] != bx[k]:
            return f"level attribute {k}: {ax[k]} against {bx[k]}"
    if xs.keys() != ys.keys():
        return f"dataset sets differ: {sorted(xs.keys() ^ ys.keys())}"
    for n, x in xs.items():
        if x.shape != ys[n].shape:
            return f"{n}: shape {x.shape} against {ys[n].shape}"
        if not np.array_equal(x, ys[n]):
            return f"{n} differs by {np.abs(x - ys[n]).max():.3e}"
    return None


# ---- one case ----------------------------------------------------------------
def run_case(driver: Driver, case: Case) -> dict:
    where = driver.out / case.name
    where.mkdir()
    checks: dict[str, str | None] = {}
    arms: dict[str, Arm] = {}
    record: dict = {"case": case.name, "resolution": list(case.resolution), "steps": case.steps,
                    "cuts": [list(a) for a in case.cuts], "restart_cuts": [list(a) for a in case.restart_cuts],
                    "checks": checks, "arms": {}, "totals": {}, "conservation": None, "div_b": {}}

    def ok(arm: Arm, expected_workers: int) -> str | None:
        if arm.code != 0:
            return f"{arm.name}: exit {arm.code}"
        if sorted(arm.exits) != list(range(expected_workers)) or any(c != 0 for c in arm.exits.values()):
            return f"{arm.name}: worker exits {arm.exits}"
        if len(arm.backends) != 1:
            return f"{arm.name}: backends {sorted(arm.backends)}"
        if final_of(arm.data) is None:
            return f"{arm.name}: no single final checkpoint under {arm.data}"
        return None

    def failed(arm: Arm, expected_workers: int, marker: str | None = None) -> str | None:
        """a fault arm: the launch exits nonzero, every worker that reported exited nonzero,
        no final checkpoint was published, and the marker (if any) appears in the streams"""
        if arm.code == 0:
            return f"{arm.name}: the launch succeeded"
        if any(c == 0 for c in arm.exits.values()):
            return f"{arm.name}: a worker exited zero: {arm.exits}"
        if len(arm.exits) != expected_workers and driver.mode == "local":
            return f"{arm.name}: exits reported for {sorted(arm.exits)}"
        if final_of(arm.data) is not None:
            return f"{arm.name}: a final checkpoint was published"
        # the streams are line-wrapped by the traceback printer, so the marker is matched
        # with its whitespace collapsed
        if marker and marker not in " ".join((arm.stderr + arm.stdout).split()):
            return f"{arm.name}: {marker!r} missing from the streams"
        return None

    # the three-way comparison and the restart
    one_cuts = tuple(() for _ in case.cuts)
    arms["single"] = driver.reference(case, where / "single")
    checks["single"] = ok(arms["single"], 1)
    # a case under another edge EMF exercises it only where the in-plane faces carry field: the
    # contact run of the same problem differs, or the method's arms test nothing
    if case.ct_method not in ("", "contact"):
        arms["single_contact"] = driver.reference(case.under_contact(), where / "single_contact")
        checks["single_contact"] = ok(arms["single_contact"], 1)
        if checks["single_contact"] is None and checks["single"] is None:
            checks["ct_method changes the run"] = (
                None if same_state(final_of(arms["single_contact"].data), final_of(arms["single"].data)) else
                f"{case.ct_method} and contact evolve the same state; the ct_method arms are vacuous")
    arms["one"] = driver.launch(case, "one", where / "one", one_cuts, case.steps)
    checks["one"] = ok(arms["one"], 1)
    arms["many"] = driver.launch(case, "many", where / "many", case.cuts, case.steps)
    checks["many"] = ok(arms["many"], case.workers(case.cuts))
    half = driver.launch(case, "half", where / "half", case.cuts, case.steps // 2)
    arms["half"] = half
    checks["half"] = ok(half, case.workers(case.cuts))
    if checks["half"] is None:
        # the step cap is an absolute root iteration: the resumed launch runs from written_at to the
        # uninterrupted run's cap
        with h5py.File(final_of(half.data), "r") as f:
            written_at = int(f["level_0"].attrs["iteration"])
        arms["restart"] = driver.launch(case, "restart", where / "restart", case.restart_cuts, case.steps,
                                        checkpoint=final_of(half.data))
        checks["restart"] = ok(arms["restart"], case.workers(case.restart_cuts))
        record["restart_written_at"] = written_at
        # the resumed run continues the faces under the edge EMF the case asks for
        if case.magnetized and checks["restart"] is None:
            resumed_ct = recorded_ct_method(final_of(arms["restart"].data))
            record["restart_ct_method"] = resumed_ct
            checks["restart ct_method"] = (
                None if resumed_ct == case.ct_method else
                f"the resumed run records ct_method={resumed_ct!r}; the case runs {case.ct_method!r}")
    else:
        checks["restart"] = "half: skipped after the partial run failed"
    for name in ("one", "many", "restart"):
        if checks.get(name) is None and checks["single"] is None:
            checks[f"{name} == single"] = same_state(final_of(arms[name].data), final_of(arms["single"].data))
    # the correction against a cut: the flagged one-worker run differs from the clean one and
    # equals the flagged many-worker run
    if case.trouble:
        inject = {"SIMBI_FABRIC_INJECT": case.trouble}
        arms["flagged_one"] = driver.launch(case, "flagged_one", where / "flagged_one", one_cuts, case.steps, env=inject)
        arms["flagged_many"] = driver.launch(case, "flagged_many", where / "flagged_many", case.cuts, case.steps, env=inject)
        checks["flagged_one"] = ok(arms["flagged_one"], 1)
        checks["flagged_many"] = ok(arms["flagged_many"], case.workers(case.cuts))
        if checks["flagged_one"] is None and checks["one"] is None:
            checks["flag changes the run"] = (
                None if same_state(final_of(arms["flagged_one"].data), final_of(arms["one"].data)) else
                "the troubled cell left the one-worker run unchanged; the correction gate is vacuous")
        if checks["flagged_one"] is None and checks["flagged_many"] is None:
            checks["flagged_many == flagged_one"] = same_state(final_of(arms["flagged_many"].data), final_of(arms["flagged_one"].data))
    # failure handling
    if "reject" in case.faults and case.magnetized and not case.refined:
        arms["reject_refused"] = driver.launch(case, "reject_refused", where / "reject_refused", case.cuts, case.steps,
                                               env={"SIMBI_FABRIC_INJECT": "reject:1:3:1"})
        checks["reject refused by name"] = failed(
            arms["reject_refused"], case.workers(case.cuts),
            marker="a forced rejection on a kernel set without a complete rollback snapshot")
    elif "reject" in case.faults:
        arms["reject_one"] = driver.launch(case, "reject_one", where / "reject_one", one_cuts, case.steps,
                                           env={"SIMBI_FABRIC_INJECT": "reject:0:3:1"})
        arms["reject_many"] = driver.launch(case, "reject_many", where / "reject_many", case.cuts, case.steps,
                                            env={"SIMBI_FABRIC_INJECT": "reject:1:3:1"})
        checks["reject_one"] = ok(arms["reject_one"], 1)
        checks["reject_many"] = ok(arms["reject_many"], case.workers(case.cuts))
        if checks["reject_one"] is None and checks["reject_many"] is None:
            checks["reject_many == reject_one"] = same_state(final_of(arms["reject_many"].data), final_of(arms["reject_one"].data))
            counts = {w: s["rejections"] for w, s in arms["reject_many"].summaries.items()}
            expected = {w: 1 for w in range(case.workers(case.cuts))}
            checks["every worker reports one rejection"] = None if counts == expected else f"rejections {counts}"
    if "bad_cfl" in case.faults:
        arms["bad_cfl"] = driver.launch(case, "bad_cfl", where / "bad_cfl", case.cuts, case.steps,
                                        env={"SIMBI_FABRIC_INJECT": "bad_cfl:1:3"})
        checks["bad_cfl"] = failed(arms["bad_cfl"], case.workers(case.cuts))
    if "lost" in case.faults:
        arms["lost"] = driver.launch(case, "lost", where / "lost", case.cuts, case.steps,
                                     env={"SIMBI_FABRIC_INJECT": "exit:1:3"})
        checks["lost"] = failed(arms["lost"], case.workers(case.cuts))
    if "face" in case.faults:
        # the injection perturbs the worker's lower x face, which is a shared face on the
        # last tile of the grid alone when the walls are open
        last = case.workers(case.cuts) - 1
        arms["face"] = driver.launch(case, "face", where / "face", case.cuts, case.steps,
                                     env={"SIMBI_FABRIC_INJECT": f"face:{last}:3"})
        checks["face"] = failed(arms["face"], case.workers(case.cuts), marker="shared-face mismatch")
    # one backend hash across every arm that reported one
    reported = {b for arm in arms.values() for b in arm.backends}
    checks["one backend hash"] = None if len(reported) == 1 else f"backends {sorted(reported)}"
    record["backend"] = sorted(reported)
    # the conserved totals and their drift over the run on a periodic domain
    if checks["many"] is None:
        end = totals(final_of(arms["many"].data))
        record["totals"] = end
        start_path = initial_of(arms["many"].data)
        if start_path is not None and end:
            start = totals(start_path)
            drift = {n: abs(end[n][0] - start[n][0]) / max(_vector_scale(start, n), 1e-300) for n in end if n in start}
            record["conservation"] = drift
            if case.periodic:
                worst = max(drift.values(), default=0.0)
                checks["conserved totals within roundoff"] = None if worst < 1e-10 else f"relative drift {worst:.3e}: {drift}"
    if case.div_b:
        for name in ("many", "restart", "flagged_many"):
            if checks.get(name) is None and name in arms:
                d = max_div_b(final_of(arms[name].data))
                record["div_b"][name] = d
                checks[f"div B on {name}"] = None if d < 1e-10 else f"max |div B| {d:.3e}"
    for name, arm in arms.items():
        record["arms"][name] = {"code": arm.code, "exits": arm.exits, "hosts": arm.hosts, "seconds": round(arm.seconds, 3),
                                "summaries": arm.summaries}
        if not driver.keep_data and arm.data.exists():
            shutil.rmtree(arm.data, ignore_errors=True)
    record["passed"] = all(v is None for v in checks.values())
    (where / "result.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    return record


# ---- the campaign ------------------------------------------------------------
def provenance(out: Path, args: argparse.Namespace) -> None:
    def git(*a: str) -> str:
        done = subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT)
        return done.stdout if done.returncode == 0 else f"(git {' '.join(a)} failed: {done.stderr.strip()})\n"
    lines = [
        f"date: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"host: {socket.gethostname()} platform: {platform.platform()} python: {sys.version.split()[0]}",
        f"mode: {args.mode} tier: {args.tier} cases: {args.case or 'every case of the tier'}",
        f"slurm job: {os.environ.get('SLURM_JOB_ID', 'none')} nodes: {os.environ.get('SLURM_JOB_NODELIST', 'none')} "
        f"cpus per task: {os.environ.get('SLURM_CPUS_PER_TASK', 'unset')}",
        "--- git rev-parse HEAD", git("rev-parse", "HEAD").strip(),
        "--- git status --short", git("status", "--short").rstrip(),
        "--- git show --no-patch --format=fuller HEAD", git("show", "--no-patch", "--format=fuller", "HEAD").rstrip(),
    ]
    (out / "provenance.txt").write_text("\n".join(lines) + "\n")


def summarize(out: Path, records: list[dict]) -> bool:
    backends = sorted({b for r in records for b in r.get("backend", [])})
    summary = {"passed": all(r["passed"] for r in records), "backends": backends,
               "cases": {r["case"]: {"passed": r["passed"], "failed_checks": {k: v for k, v in r["checks"].items() if v}}
                         for r in records}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# fabric campaign", "", f"backend: {', '.join(backends) or 'none reported'}", "",
             "| case | result | failed checks |", "|---|---|---|"]
    for r in records:
        failed = "; ".join(f"{k}: {v}" for k, v in r["checks"].items() if v) or ""
        lines.append(f"| {r['case']} | {'PASSED' if r['passed'] else 'FAILED'} | {failed} |")
    lines += ["", "the rows above are the cases this installation ran; a combination outside the table is unverified here."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    with (out / "provenance.txt").open("a") as prov:
        prov.write(f"--- reported backends\n{chr(10).join(backends) or 'none'}\n")
    return summary["passed"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="the validation campaign for simbi launch")
    parser.add_argument("out", type=Path, help="a fresh output directory (shared filesystem under a scheduler)")
    parser.add_argument("--mode", choices=("local", "scheduler"), default="local")
    parser.add_argument("--tier", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--case", action="append", help="run this case alone (repeatable)")
    parser.add_argument("--limit-seconds", type=float, default=900.0, help="the wall limit of one arm")
    parser.add_argument("--keep-data", action="store_true", help="keep every arm's checkpoints")
    parser.add_argument("--list", action="store_true", help="print the case table and exit")
    args = parser.parse_args(argv)
    chosen = [c for c in CASES if (args.tier in c.tiers) and (not args.case or c.name in args.case)]
    if args.case:
        unknown = set(args.case) - {c.name for c in CASES}
        if unknown:
            parser.error(f"unknown case(s) {sorted(unknown)}; known: {[c.name for c in CASES]}")
        chosen = [c for c in CASES if c.name in args.case]
    if args.list:
        for c in chosen:
            print(f"{c.name}: {c.cls} {c.resolution} steps {c.steps} cuts {c.cuts} restart {c.restart_cuts} tiers {c.tiers}")
        return 0
    if args.mode == "scheduler" and not os.environ.get("SLURM_JOB_ID"):
        parser.error("scheduler mode runs inside a slurm allocation")
    out = args.out
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} exists and is not empty; give a fresh directory")
    out.mkdir(parents=True, exist_ok=True)
    provenance(out, args)
    driver = Driver(out, args.mode, args.limit_seconds, args.keep_data)
    records = []
    for case in chosen:
        started = time.time()
        record = run_case(driver, case)
        records.append(record)
        state = "PASSED" if record["passed"] else "FAILED"
        print(f"{case.name}: {state} in {time.time() - started:.0f}s", flush=True)
        for k, v in record["checks"].items():
            if v:
                print(f"  - {k}: {v}", flush=True)
    passed = summarize(out, records)
    print(f"campaign: {'PASSED' if passed else 'FAILED'} ({sum(r['passed'] for r in records)}/{len(records)} cases); "
          f"records under {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
