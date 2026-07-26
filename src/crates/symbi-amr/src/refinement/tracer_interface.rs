// =============================================================================
// tracer_interface.rs
//
// topology of mass-transfer faces between one refined box and the active
// coarse cells surrounding it. each fine face retains its own transverse child
// address so tracer receipts follow the accepted fine flux without a posterior
// density redistribution.
//
// usage:
//  let faces = interface_faces(&coverage, root_cells, coarse_level);
// =============================================================================

use symbi_algebra::Domain;
use symbi_geometry::{BlockGeometry, Metric};
use symbi_sim::mass_transport::ContainerId;
use symbi_sim::state::ConsFieldsGeneric;
use symbi_sim::tracers::cell_container_id;
use symbi_xpu::MemorySpace;

const RATIO: isize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InterfaceFace<const D: usize> {
    pub axis: usize,
    pub high: bool,
    pub fine_face: [isize; D],
    pub coarse_cell: ContainerId,
    pub fine_cell: ContainerId,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterfaceTransfer {
    pub source: ContainerId,
    pub destination: ContainerId,
    pub mass: f64,
}

fn linear<const D: usize>(coord: [isize; D], cells: [usize; D]) -> usize {
    let mut result = 0usize;
    let mut stride = 1usize;
    for aa in 0..D {
        assert!(coord[aa] >= 0, "negative global tracer cell index");
        result += coord[aa] as usize * stride;
        stride *= cells[aa];
    }
    result
}

/// enumerate every fine face on a coarse-fine interface. `coverage` is in
/// absolute coarse indices and `coarse_level` is the level it covers.
pub fn interface_faces<const D: usize>(
    coverage: &Domain<D>,
    root_cells: [usize; D],
    coarse_level: u8,
) -> Vec<InterfaceFace<D>> {
    assert!(coarse_level < 63, "tracer refinement level exceeds 63");
    let coarse_scale = 1usize << coarse_level;
    let coarse_cells: [usize; D] =
        std::array::from_fn(|aa| root_cells[aa] * coarse_scale);
    let fine_cells: [usize; D] =
        std::array::from_fn(|aa| coarse_cells[aa] * RATIO as usize);
    let mut result = Vec::new();

    for axis in 0..D {
        for high in [false, true] {
            if (!high && coverage.spaces[axis].lo == 0)
                || (high && coverage.spaces[axis].hi == coarse_cells[axis] as isize)
            {
                continue;
            }
            let face_axis = if high {
                coverage.spaces[axis].hi * RATIO
            } else {
                coverage.spaces[axis].lo * RATIO
            };
            let mut coord = std::array::from_fn(|aa| {
                if aa == axis {
                    face_axis
                } else {
                    coverage.spaces[aa].lo * RATIO
                }
            });
            loop {
                let fine_cell_coord: [isize; D] = std::array::from_fn(|aa| {
                    if aa == axis && high {
                        coord[aa] - 1
                    } else {
                        coord[aa]
                    }
                });
                let coarse_cell_coord: [isize; D] = std::array::from_fn(|aa| {
                    if aa == axis {
                        if high {
                            coverage.spaces[aa].hi
                        } else {
                            coverage.spaces[aa].lo - 1
                        }
                    } else {
                        coord[aa] / RATIO
                    }
                });
                result.push(InterfaceFace {
                    axis,
                    high,
                    fine_face: coord,
                    coarse_cell: cell_container_id(
                        linear(coarse_cell_coord, coarse_cells),
                        coarse_level,
                    ),
                    fine_cell: cell_container_id(
                        linear(fine_cell_coord, fine_cells),
                        coarse_level + 1,
                    ),
                });

                let mut advanced = false;
                for aa in 0..D {
                    if aa == axis {
                        continue;
                    }
                    coord[aa] += 1;
                    if coord[aa] < coverage.spaces[aa].hi * RATIO {
                        advanced = true;
                        break;
                    }
                    coord[aa] = coverage.spaces[aa].lo * RATIO;
                }
                if !advanced {
                    break;
                }
            }
        }
    }
    result
}

/// integrate the directed density flux on every fine interface face for one
/// accepted stage contribution. `weight` includes the stage coefficient and
/// timestep; face area is supplied by the fine-level geometry.
pub fn interface_mass_transfers<const D: usize, const DOF: usize, M, Mem>(
    faces: &[InterfaceFace<D>],
    flux: &[ConsFieldsGeneric<D, DOF, Mem>; D],
    geometry: &BlockGeometry<M, f64, D>,
    weight: f64,
) -> Vec<InterfaceTransfer>
where
    M: Metric<f64, D> + Copy,
    Mem: MemorySpace,
{
    let mut result = Vec::new();
    for face in faces {
        let signed_mass = *flux[face.axis].den.view().at(face.fine_face)
            * geometry.face_area(face.fine_face, face.axis)
            * weight;
        if signed_mass == 0.0 {
            continue;
        }
        let outward_mass = if face.high {
            signed_mass
        } else {
            -signed_mass
        };
        let (source, destination) = if outward_mass > 0.0 {
            (face.fine_cell, face.coarse_cell)
        } else {
            (face.coarse_cell, face.fine_cell)
        };
        result.push(InterfaceTransfer {
            source,
            destination,
            mass: outward_mass.abs(),
        });
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_algebra::{Domain, Space};
    use symbi_geometry::{BlockGeometry, Cartesian};
    use symbi_sim::state::ConsFieldsGeneric;
    use symbi_sim::tracers::cell_container_address;
    use symbi_xpu::HostMemory;

    #[test]
    fn two_dimensional_interface_keeps_each_transverse_child_face() {
        let coverage = Domain::new([
            Space {
                name: "i",
                lo: 2,
                hi: 4,
            },
            Space {
                name: "j",
                lo: 1,
                hi: 3,
            },
        ]);
        let faces = interface_faces(&coverage, [8, 6], 0);

        assert_eq!(faces.len(), 16);
        let low_x: Vec<_> = faces
            .iter()
            .filter(|face| face.axis == 0 && !face.high)
            .collect();
        assert_eq!(low_x.len(), 4);
        assert_eq!(
            low_x.iter().map(|face| face.fine_face[1]).collect::<Vec<_>>(),
            [2, 3, 4, 5]
        );
        assert!(
            low_x
                .iter()
                .all(|face| cell_container_address(face.coarse_cell).unwrap().0 == 0)
        );
        assert!(
            low_x
                .iter()
                .all(|face| cell_container_address(face.fine_cell).unwrap().0 == 1)
        );
        assert_eq!(low_x[0].coarse_cell, low_x[1].coarse_cell);
        assert_ne!(low_x[0].fine_cell, low_x[1].fine_cell);
    }

    #[test]
    fn fine_flux_direction_selects_the_material_source() {
        let coverage = Domain::new([Space {
            name: "i",
            lo: 2,
            hi: 4,
        }]);
        let faces = interface_faces(&coverage, [8], 0);
        let domain = Domain::new([Space {
            name: "i",
            lo: 3,
            hi: 9,
        }]);
        let flux = [ConsFieldsGeneric::<1, 1, HostMemory>::zeros(&domain).unwrap()];
        flux[0].den.view_mut().set(faces[0].fine_face, 2.0);
        flux[0].den.view_mut().set(faces[1].fine_face, 3.0);
        let geometry = BlockGeometry::uniform(Cartesian, [0.0], [0.5]);

        let transfers = interface_mass_transfers(&faces, &flux, &geometry, 0.25);

        assert_eq!(transfers.len(), 2);
        assert_eq!(transfers[0].source, faces[0].coarse_cell);
        assert_eq!(transfers[0].destination, faces[0].fine_cell);
        assert_eq!(transfers[0].mass, 0.5);
        assert_eq!(transfers[1].source, faces[1].fine_cell);
        assert_eq!(transfers[1].destination, faces[1].coarse_cell);
        assert_eq!(transfers[1].mass, 0.75);
    }
}
