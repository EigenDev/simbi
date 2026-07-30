# =============================================================================
# test_body_tint.py
#
# the per-body gas tint overlay: a field re-colored near each immersed body with a
# per-body palette, for e.g. red/blue minidisks around a black-hole binary. the
# tests pin the region masking (only cells within `radius`, or a caller's custom
# region, are tinted) and the one-artist-per-body contract.
# =============================================================================
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simbi.viz.bodies import tint_bodies


def _write_two_bodies(path, positions) -> None:
    with h5py.File(path, "w") as f:
        g = f.create_group("bodies")
        g.attrs["n_bodies"] = len(positions)
        g.create_dataset("position", data=np.asarray(positions, dtype=float))
        g.create_dataset("orientation", data=np.asarray([np.eye(3)] * len(positions)))
        g.create_dataset("omega", data=np.zeros((len(positions), 3), dtype=float))
        g.create_dataset("mass", data=np.ones(len(positions), dtype=float))


def _grid(n=101, span=2.0):
    us = np.linspace(-span, span, n)
    vs = np.linspace(-span, span, n)
    field = np.ones((n, n), dtype=float)
    return us, vs, field


def test_tint_one_artist_per_body_masked_to_the_radius(tmp_path) -> None:
    path = str(tmp_path / "binary.h5")
    _write_two_bodies(path, [[0.5, 0.5], [-0.5, -0.5]])
    us, vs, field = _grid()

    fig, ax = plt.subplots()
    artists = tint_bodies(ax, field, us, vs, path, ["red", "blue"], radius=0.4)

    assert len(artists) == 2  # one tint layer per body
    for qm in artists:
        arr = qm.get_array()
        # a disk of radius 0.4 covers some cells but not the whole 4x4 domain:
        # both tinted (unmasked) and background (masked) cells must be present.
        assert np.ma.count(arr) > 0
        assert np.ma.count_masked(arr) > 0
    plt.close(fig)


def test_tint_region_is_local_to_each_body(tmp_path) -> None:
    # the two bodies are far apart; each tint's unmasked cells sit around its own center.
    path = str(tmp_path / "binary.h5")
    _write_two_bodies(path, [[1.5, 1.5], [-1.5, -1.5]])
    us, vs, field = _grid()
    ug, vg = np.meshgrid(us, vs)

    fig, ax = plt.subplots()
    artists = tint_bodies(ax, field, us, vs, path, ["red", "blue"], radius=0.3)

    # body 0 at (1.5, 1.5): its unmasked cells must all be near (1.5, 1.5), none near (-1.5, -1.5).
    mask0 = ~np.ma.getmaskarray(artists[0].get_array()).reshape(ug.shape)
    assert mask0[np.argmin(np.abs(vs - 1.5)), np.argmin(np.abs(us - 1.5))]
    assert not mask0[np.argmin(np.abs(vs + 1.5)), np.argmin(np.abs(us + 1.5))]
    plt.close(fig)


def test_tint_accepts_a_custom_region(tmp_path) -> None:
    # a caller-supplied region overrides the radial default: here, tint everything.
    path = str(tmp_path / "one.h5")
    _write_two_bodies(path, [[0.0, 0.0]])
    us, vs, field = _grid()

    def whole_domain(_body, ug, _vg, _ua, _va):
        return np.ones_like(ug, dtype=bool)

    fig, ax = plt.subplots()
    artists = tint_bodies(
        ax, field, us, vs, path, ["red"], radius=0.1, region=whole_domain
    )
    assert np.ma.count_masked(artists[0].get_array()) == 0  # nothing masked out
    plt.close(fig)
