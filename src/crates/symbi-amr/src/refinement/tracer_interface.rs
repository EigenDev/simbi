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

use std::collections::BTreeMap;
use symbi_algebra::Domain;
use symbi_geometry::{BlockGeometry, Metric};
use symbi_sim::mass_transport::{ContainerId, MassTransfer, TransportKernel};
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

/// the container address of a cell already held in absolute global indices.
fn global_address<const D: usize>(coord: [isize; D], cells: [usize; D]) -> usize {
    let cell: [usize; D] = std::array::from_fn(|aa| {
        assert!(coord[aa] >= 0, "negative global tracer cell index");
        coord[aa] as usize
    });
    symbi_sim::tracers::cell_address(cell, cells)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterfaceTransfer {
    pub source: ContainerId,
    pub destination: ContainerId,
    pub mass: f64,
}

/// enumerate every fine face on a coarse-fine interface. `coverage` is in
/// absolute coarse indices and `coarse_level` is the level it covers.
pub fn interface_faces<const D: usize>(
    coverage: &Domain<D>,
    root_cells: [usize; D],
    coarse_level: u8,
) -> Vec<InterfaceFace<D>> {
    interface_faces_with_layout(coverage, root_cells, [0; D], root_cells, coarse_level)
}

pub fn interface_faces_with_layout<const D: usize>(
    coverage: &Domain<D>,
    parent_cells: [usize; D],
    root_offset: [usize; D],
    root_global_cells: [usize; D],
    coarse_level: u8,
) -> Vec<InterfaceFace<D>> {
    assert!(coarse_level < 63, "tracer refinement level exceeds 63");
    let coarse_scale = 1usize << coarse_level;
    let coarse_cells: [usize; D] = std::array::from_fn(|aa| root_global_cells[aa] * coarse_scale);
    let fine_cells: [usize; D] = std::array::from_fn(|aa| coarse_cells[aa] * RATIO as usize);
    let mut result = Vec::new();

    for axis in 0..D {
        for high in [false, true] {
            if (!high && coverage.spaces[axis].lo == 0)
                || (high
                    && coverage.spaces[axis].hi
                        == parent_cells[axis] as isize * coarse_scale as isize)
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
                let coarse_global: [isize; D] = std::array::from_fn(|aa| {
                    coarse_cell_coord[aa] + (root_offset[aa] * coarse_scale) as isize
                });
                let fine_global: [isize; D] = std::array::from_fn(|aa| {
                    fine_cell_coord[aa] + (root_offset[aa] * coarse_scale * RATIO as usize) as isize
                });
                result.push(InterfaceFace {
                    axis,
                    high,
                    fine_face: coord,
                    coarse_cell: cell_container_id(
                        global_address(coarse_global, coarse_cells),
                        coarse_level,
                    ),
                    fine_cell: cell_container_id(global_address(fine_global, fine_cells), coarse_level + 1),
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
        let outward_mass = if face.high { signed_mass } else { -signed_mass };
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

/// build one simultaneous interface event from post-event cell masses and
/// directed receipts. conservation gives `pre = post + outbound - inbound`.
pub fn interface_transport_kernels(
    transfers: &[InterfaceTransfer],
    post_mass: &BTreeMap<ContainerId, f64>,
) -> Result<Vec<TransportKernel>, String> {
    let mut outgoing = BTreeMap::<ContainerId, Vec<MassTransfer>>::new();
    let mut incoming = BTreeMap::<ContainerId, f64>::new();
    for transfer in transfers {
        outgoing
            .entry(transfer.source)
            .or_default()
            .push(MassTransfer {
                destination: transfer.destination,
                mass: transfer.mass,
            });
        *incoming.entry(transfer.destination).or_insert(0.0) += transfer.mass;
    }
    outgoing
        .into_iter()
        .map(|(source, transfers)| {
            let outbound: f64 = transfers.iter().map(|transfer| transfer.mass).sum();
            let post = post_mass.get(&source).copied().ok_or_else(|| {
                format!("missing post-event mass for interface source {}", source.0)
            })?;
            let pre = post + outbound - incoming.get(&source).copied().unwrap_or(0.0);
            TransportKernel::new(source, pre, transfers)
        })
        .collect()
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
            low_x
                .iter()
                .map(|face| face.fine_face[1])
                .collect::<Vec<_>>(),
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
        let geometry = BlockGeometry::uniform(Cartesian, [0.0], [0.5], std::array::from_fn(|d| d));

        let transfers = interface_mass_transfers(&faces, &flux, &geometry, 0.25);

        assert_eq!(transfers.len(), 2);
        assert_eq!(transfers[0].source, faces[0].coarse_cell);
        assert_eq!(transfers[0].destination, faces[0].fine_cell);
        assert_eq!(transfers[0].mass, 0.5);
        assert_eq!(transfers[1].source, faces[1].fine_cell);
        assert_eq!(transfers[1].destination, faces[1].coarse_cell);
        assert_eq!(transfers[1].mass, 0.75);
    }

    #[test]
    fn interface_event_recovers_pre_event_mass_from_conservation() {
        let coarse = cell_container_id(3, 0);
        let fine = cell_container_id(6, 1);
        let transfers = [
            InterfaceTransfer {
                source: coarse,
                destination: fine,
                mass: 2.0,
            },
            InterfaceTransfer {
                source: fine,
                destination: coarse,
                mass: 1.0,
            },
        ];
        let post_mass = BTreeMap::from([(coarse, 9.0), (fine, 6.0)]);

        let kernels = interface_transport_kernels(&transfers, &post_mass).unwrap();

        let coarse_kernel = kernels
            .iter()
            .find(|kernel| kernel.source() == coarse)
            .unwrap();
        let fine_kernel = kernels
            .iter()
            .find(|kernel| kernel.source() == fine)
            .unwrap();
        assert_eq!(coarse_kernel.source_mass(), 10.0);
        assert_eq!(fine_kernel.source_mass(), 5.0);
        assert_eq!(coarse_kernel.destinations(), &[(coarse, 0.8), (fine, 0.2)]);
        assert_eq!(fine_kernel.destinations(), &[(coarse, 0.2), (fine, 0.8)]);
    }
}
