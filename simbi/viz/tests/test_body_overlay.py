# =============================================================================
# test_body_overlay.py
#
# the immersed-body viz loader + silhouette mask: a synthetic checkpoint `bodies`
# group (the writer's format) round-trips into BodyPose, and body_mask marks the
# posed shape's interior. the numpy SDF must agree with simbi.types.shape's.
# =============================================================================
import json
import math

import h5py
import numpy as np

from simbi.types.shape import Shape
from simbi.viz.bodies import body_mask, load_bodies


def _write_bodies(path, position, orientation, omega, shape_wire) -> None:
    with h5py.File(path, "w") as f:
        g = f.create_group("bodies")
        g.attrs["n_bodies"] = 1
        g.create_dataset("position", data=np.asarray([position], dtype=float))
        g.create_dataset("orientation", data=np.asarray([orientation], dtype=float))
        g.create_dataset("omega", data=np.asarray([omega], dtype=float))
        g.create_dataset("mass", data=np.asarray([1.0], dtype=float))
        if shape_wire is not None:
            g.attrs["shape_0"] = json.dumps(shape_wire)


def test_load_bodies_and_mask_a_posed_box(tmp_path) -> None:
    box = Shape.box((0.0, 0.0, 0.0), (0.5, 0.2, 1.0))
    path = str(tmp_path / "chk.h5")
    _write_bodies(path, [1.0, 0.5], np.eye(3), [0.0, 0.0, 2.0], box.to_wire())

    bodies = load_bodies(path)
    assert len(bodies) == 1
    b = bodies[0]
    assert b.shape is not None
    assert np.allclose(b.position, [1.0, 0.5])
    assert np.allclose(b.omega, [0.0, 0.0, 2.0])

    xs = np.linspace(0.0, 2.0, 201)
    ys = np.linspace(-0.5, 1.5, 201)
    m = body_mask(b, xs, ys)

    def at(x, y):
        return m[int(np.argmin(np.abs(ys - y))), int(np.argmin(np.abs(xs - x)))]

    assert at(1.0, 0.5)  # box center (position) is inside
    assert not at(1.6, 0.5)  # 0.6 > 0.5 half-extent in x -> outside
    assert not at(1.0, 0.8)  # 0.3 > 0.2 half-extent in y -> outside


def test_body_mask_tracks_orientation(tmp_path) -> None:
    # a box rotated 90deg about z: its long (0.5) extent maps onto y, the short (0.2) onto x.
    box = Shape.box((0.0, 0.0, 0.0), (0.5, 0.2, 1.0))
    c, s = math.cos(math.pi / 2), math.sin(math.pi / 2)
    rot = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    path = str(tmp_path / "chk_rot.h5")
    _write_bodies(path, [0.0, 0.0], rot, [0.0, 0.0, 1.0], box.to_wire())
    b = load_bodies(path)[0]

    xs = np.linspace(-1.0, 1.0, 201)
    ys = np.linspace(-1.0, 1.0, 201)
    m = body_mask(b, xs, ys)

    def at(x, y):
        return m[int(np.argmin(np.abs(ys - y))), int(np.argmin(np.abs(xs - x)))]

    assert at(0.0, 0.4)  # the 0.5 extent now runs along y
    assert not at(0.4, 0.0)  # the 0.2 extent now runs along x
