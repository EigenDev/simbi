# =============================================================================
# test_bonded_assembly.py
#
# validation + serialization of the bonded-fragment assembly config:
# - constructor rejects mismatched arrays, bad bond indices, duplicate bonds
# - to_backend emits plain lists with solid-sphere inertia defaults
# - pack() fills a shape with touching fragments and bonds lattice neighbors
# - body_payload carries the assembly under its own key
# =============================================================================
import math

import pytest

from simbi.simulation.problem import ConfigError
from simbi.types.bodies import (
    body_payload,
    BondedAssembly,
    BondMaterial,
    ContactMaterial,
    MutualGravity,
)
from simbi.types.shape import Shape


def pair_assembly(**kwargs):
    return BondedAssembly(
        positions=[[0.0, 0.0], [1.0, 0.0]],
        masses=[1.0, 1.0],
        radii=[0.5, 0.5],
        bonds=[(0, 1)],
        bond_material=BondMaterial(k_n=100.0),
        **kwargs,
    )


class TestValidation:
    def test_valid_pair_constructs(self):
        pair_assembly()

    def test_mismatched_masses_rejected(self):
        with pytest.raises(ConfigError, match="masses"):
            BondedAssembly(
                positions=[[0.0, 0.0], [1.0, 0.0]],
                masses=[1.0],
                radii=[0.5, 0.5],
                bonds=[],
                bond_material=BondMaterial(k_n=1.0),
            )

    def test_out_of_range_bond_rejected(self):
        with pytest.raises(ConfigError, match="bond"):
            BondedAssembly(
                positions=[[0.0, 0.0], [1.0, 0.0]],
                masses=[1.0, 1.0],
                radii=[0.5, 0.5],
                bonds=[(0, 2)],
                bond_material=BondMaterial(k_n=1.0),
            )

    def test_self_bond_rejected(self):
        with pytest.raises(ConfigError, match="distinct"):
            BondedAssembly(
                positions=[[0.0, 0.0], [1.0, 0.0]],
                masses=[1.0, 1.0],
                radii=[0.5, 0.5],
                bonds=[(1, 1)],
                bond_material=BondMaterial(k_n=1.0),
            )

    def test_duplicate_bond_rejected(self):
        with pytest.raises(ConfigError, match="twice"):
            BondedAssembly(
                positions=[[0.0, 0.0], [1.0, 0.0]],
                masses=[1.0, 1.0],
                radii=[0.5, 0.5],
                bonds=[(0, 1), (1, 0)],
                bond_material=BondMaterial(k_n=1.0),
            )

    def test_nonpositive_bond_stiffness_rejected(self):
        with pytest.raises(ConfigError, match="k_n"):
            BondMaterial(k_n=0.0)

    def test_wrong_length_mobile_rejected(self):
        with pytest.raises(ConfigError, match="mobile"):
            pair_assembly(mobile=[True])


class TestSerialization:
    def test_backend_wire_shape(self):
        wire = pair_assembly(
            contact=ContactMaterial(k_n=1e3, mu=0.3),
            gravity=MutualGravity(g=1.0, softening=0.05),
        ).to_backend()
        assert wire["positions"] == [[0.0, 0.0], [1.0, 0.0]]
        assert wire["bonds"] == [[0, 1]]
        assert wire["bond_material"]["k_n"] == 100.0
        assert math.isinf(wire["bond_material"]["sigma_t"])
        assert wire["contact"]["mu"] == 0.3
        assert wire["gravity"]["softening"] == 0.05
        # solid-sphere inertia default 0.4 m r^2
        assert wire["inertias"] == [pytest.approx(0.4 * 1.0 * 0.25)] * 2
        assert wire["velocities"] == [[0.0, 0.0], [0.0, 0.0]]
        assert wire["mobile"] == [True, True]

    def test_payload_key(self):
        payload = body_payload(None, [], pair_assembly())
        assert set(payload.keys()) == {"bonded_assembly"}

    def test_absent_assembly_contributes_no_key(self):
        assert body_payload(None, [], None) == {}


class TestPacking:
    def test_packed_box_fragments_fit_and_bond(self):
        # a 2 x 1 box packed at spacing 0.5 -> a 4 x 2 lattice of radius-0.25
        # fragments, each fully inside, bonded to axis + diagonal neighbors.
        box = Shape.box([0.0, 0.0, 0.0], [1.0, 0.5, 10.0])
        asm = BondedAssembly.pack(
            box,
            bounds=[(-1.0, 1.0), (-0.5, 0.5)],
            spacing=0.5,
            fragment_mass=0.1,
            bond_material=BondMaterial(k_n=50.0),
        )
        assert len(asm.positions) == 8
        for p in asm.positions:
            assert box.signed_distance([p[0], p[1], 0.0]) <= -0.25 + 1e-12
        # interior lattice connectivity: 10 axis bonds + 6 diagonals.
        assert len(asm.bonds) == 16
        assert all(r == 0.25 for r in asm.radii)

    def test_jittered_packing_is_disjoint_and_deterministic(self):
        disk = Shape.sphere([0.0, 0.0, 0.0], 0.6)
        kwargs = dict(
            bounds=[(-0.6, 0.6), (-0.6, 0.6)],
            spacing=0.2,
            fragment_mass=0.1,
            bond_material=BondMaterial(k_n=50.0),
            jitter=0.12,
            seed=3,
        )
        a = BondedAssembly.pack(disk, **kwargs)
        b = BondedAssembly.pack(disk, **kwargs)
        assert a.positions == b.positions, "seeded jitter must be deterministic"
        # the shrunk radius guarantees jittered neighbors never start overlapped
        r = a.radii[0]
        assert r == pytest.approx((0.5 - 0.12) * 0.2)
        n = len(a.positions)
        for i in range(n):
            for j in range(i + 1, n):
                d = math.dist(a.positions[i], a.positions[j])
                assert d >= 2.0 * r - 1e-12, f"fragments {i},{j} overlap: {d} < {2 * r}"
        # jitter actually moved points off the lattice
        assert any(
            abs((p[0] / 0.1) - round(p[0] / 0.1)) > 1e-6 for p in a.positions
        )

    def test_empty_packing_fails_loud(self):
        tiny = Shape.sphere([0.0, 0.0, 0.0], 0.05)
        with pytest.raises(ConfigError, match="zero fragments"):
            BondedAssembly.pack(
                tiny,
                bounds=[(-1.0, 1.0), (-1.0, 1.0)],
                spacing=0.5,
                fragment_mass=0.1,
                bond_material=BondMaterial(k_n=50.0),
            )
