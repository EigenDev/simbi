# =============================================================================
# test_fofc_hydro.py
#
# first-order flux correction (FOFC) for the hydro regimes (SRHD / Newtonian /
# isothermal), 1D and 3D. the c2p is fail-loud floor-free, so a strong shock or
# blast whose high-order reconstruction drives a zone unphysical would halt the
# run — unless FOFC redoes that zone at first order (PCM + HLLE) and, failing
# that, freezes it to the admissible stage input. each case is a canonical
# strong-shock / blast problem run to completion; the gate asserts it finishes
# with a strictly positive, finite state and no crash checkpoint.
# =============================================================================
import glob
import os
import tempfile

import h5py
import numpy as np
import pytest

from simbi.simulation import runner


def _run(cls, args, steps: int = 400):
    # `steps` is a cost bound, not a physical one: the first-order redo engages within
    # the first few steps a shock exists, so what is asserted (no crash, a state
    # that stayed admissible) is visible early. the 3D case carries a hundred times the
    # cells of the 1D one, so it gets proportionally fewer.
    d = tempfile.mkdtemp() + "/"
    p = cls.from_cli(args)
    p.data_directory = d
    p.checkpoint_interval = 1.0e30
    runner.run(p, compute_mode="cpu", max_steps=steps)
    assert not glob.glob(os.path.join(d, "*crashed*")), "run crashed (c2p halt not recovered by FOFC)"
    final = glob.glob(os.path.join(d, "*final*.h5"))
    assert final, "no final checkpoint written"
    with h5py.File(final[0]) as h:
        prim = h["level_0/partition_0/hydro/primitives"]
        rho = prim["rho"][:]
        pre = prim["pre"][:] if "pre" in prim else None
    assert np.isfinite(rho).all(), "non-finite density survived"
    assert float(rho.min()) > 0.0, "density went non-positive"
    if pre is not None:
        assert np.isfinite(pre).all(), "non-finite pressure survived"
        assert float(pre.min()) > 0.0, "pressure went non-positive"


def test_fofc_srhd_marti_muller_1d() -> None:
    # marti & muller relativistic shock tube (1000:1 pressure jump, mildly-to-strongly
    # relativistic): the canonical SRHD c2p stress in 1D.
    from simbi_configs.examples.srhd.marti_muller import MartiMuller

    _run(MartiMuller, ["--resolution", "400"])


def test_fofc_srhd_blast_3d() -> None:
    # the marti & muller blast in 3D — the same relativistic shock structure with the
    # first-order redo exercised on every gridded axis.
    from simbi_configs.examples.srhd.marti_muller_3d import MartiMuller3D

    _run(MartiMuller3D, [], steps=40)


def test_fofc_newtonian_sod_1d() -> None:
    from simbi_configs.examples.newtonian.sod import SodProblem

    _run(SodProblem, [])


def test_fofc_isothermal_sod_1d() -> None:
    # isothermal has no energy law; the physicality test is density-only (pressure is
    # cs^2*rho, positive whenever rho is).
    from simbi_configs.examples.isothermal.isothermal_sod import IsothermalSod

    _run(IsothermalSod, [])
