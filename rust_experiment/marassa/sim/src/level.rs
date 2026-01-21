// =============================================================================
// level.rs
//
// level state container for amr hierarchy.
// contains all partitions at a given refinement level, halo communication
// graph, mesh configuration, and optional flux registers for amr.
//
// design:
//   - matches c++ level_decomposition_t, level_info_t, level_mesh_t
//   - contains vector of partitions
//   - halo graph for inter-partition communication
//   - flux registers for amr reflux correction
//   - timestep and subcycling info
//
// usage:
//   let level = LevelState::single_partition(device, config, level_id)?;
//   level.advance_euler(dt)?;
// =============================================================================

use crate::partition::{BoundaryInfo, DomainSet, PartitionState};
use compute::Domain;
use std::marker::PhantomData;
use xpu_core::Device;

// =============================================================================
// mesh configuration
// =============================================================================

#[derive(Debug, Clone)]
pub struct MeshConfig<const RANK: usize> {
    pub x_min: [f64; RANK],
    pub x_max: [f64; RANK],
    pub n_cells: [usize; RANK],
    pub dx: [f64; RANK],
    pub halo_width: usize,
}

impl<const RANK: usize> MeshConfig<RANK> {
    pub fn new(
        x_min: [f64; RANK],
        x_max: [f64; RANK],
        n_cells: [usize; RANK],
        halo_width: usize,
    ) -> Self {
        let mut dx = [0.0; RANK];
        for d in 0..RANK {
            dx[d] = (x_max[d] - x_min[d]) / n_cells[d] as f64;
        }

        Self {
            x_min,
            x_max,
            n_cells,
            dx,
            halo_width,
        }
    }

    pub fn global_domain(&self) -> Domain<RANK> {
        let start = [0; RANK];
        let end = std::array::from_fn(|d| self.n_cells[d] as i64);
        Domain::new(start, end)
    }

    pub fn cell_center(&self, indices: [i64; RANK]) -> [f64; RANK] {
        let mut pos = [0.0; RANK];
        for d in 0..RANK {
            pos[d] = self.x_min[d] + (indices[d] as f64 + 0.5) * self.dx[d];
        }
        pos
    }
}

// =============================================================================
// halo communication
// =============================================================================

#[derive(Debug, Clone, Copy)]
pub enum Side {
    Left,
    Right,
}

#[derive(Debug, Clone)]
pub struct HaloLink<const RANK: usize> {
    pub src_partition: usize,
    pub dst_partition: usize,
    pub src_region: Domain<RANK>,
    pub dst_region: Domain<RANK>,
    pub dimension: usize,
    pub direction: Side,
}

#[derive(Debug, Clone)]
pub struct HaloGraph<const RANK: usize> {
    pub links: Vec<HaloLink<RANK>>,
}

impl<const RANK: usize> HaloGraph<RANK> {
    pub fn empty() -> Self {
        Self { links: Vec::new() }
    }

    pub fn add_link(&mut self, link: HaloLink<RANK>) {
        self.links.push(link);
    }

    pub fn build_1d(_n_partitions: usize, _axis: usize, _halo_width: usize) -> Self {
        let graph = Self::empty();

        // todo: build actual halo links
        /*
        for _i in 0.._n_partitions - 1 {
            // link from partition i to partition i+1 (right ghost zone)
            // todo: create actual halo links with proper domains
            // for now: placeholder
        }
        */

        graph
    }

    pub fn exchange<R, D: Device>(
        &self,
        _partitions: &mut [PartitionState<R, D, RANK>],
    ) -> Result<(), D::Error> {
        // todo: implement halo exchange
        // for each link:
        //   copy src_partition[src_region] -> dst_partition[dst_region]
        Ok(())
    }
}

// =============================================================================
// flux register (for amr reflux correction)
// =============================================================================

pub struct FluxRegister<'d, D: Device, const RANK: usize> {
    _registers: Vec<compute::Field<'d, f64, D, RANK>>,
    _coarse_domain: Domain<RANK>,
    _refinement_ratio: [usize; RANK],
}

impl<'d, D: Device, const RANK: usize> FluxRegister<'d, D, RANK> {
    pub fn allocate(
        _device: &'d D,
        _coarse_domain: Domain<RANK>,
        _refinement_ratio: [usize; RANK],
    ) -> Result<Self, D::Error> {
        // todo: implement flux register allocation
        Ok(Self {
            _registers: Vec::new(),
            _coarse_domain,
            _refinement_ratio,
        })
    }
}

// =============================================================================
// level state
// =============================================================================

pub struct LevelState<'d, R, D: Device, const RANK: usize> {
    pub level_id: usize,
    pub refinement_ratio: usize,

    pub partitions: Vec<PartitionState<'d, R, D, RANK>>,
    pub halo_graph: HaloGraph<RANK>,
    pub mesh_config: MeshConfig<RANK>,

    pub flux_registers: Option<Vec<FluxRegister<'d, D, RANK>>>,

    pub dt: f64,
    pub substeps: usize,

    _marker: PhantomData<R>,
}

impl<'d, R, D: Device, const RANK: usize> LevelState<'d, R, D, RANK> {
    pub fn single_partition(
        device: &'d D,
        mesh_config: MeshConfig<RANK>,
        level_id: usize,
        is_mhd: bool,
    ) -> Result<Self, D::Error> {
        let owned = mesh_config.global_domain();
        let nghosts = std::array::from_fn(|_| mesh_config.halo_width);
        let domains = DomainSet::new(owned, nghosts);

        let partition = PartitionState::allocate(device, 0, level_id, domains, is_mhd)?;

        Ok(Self {
            level_id,
            refinement_ratio: 1,
            partitions: vec![partition],
            halo_graph: HaloGraph::empty(),
            mesh_config,
            flux_registers: None,
            dt: 0.0,
            substeps: 1,
            _marker: PhantomData,
        })
    }

    pub fn decomposed(
        device: &'d D,
        mesh_config: MeshConfig<RANK>,
        level_id: usize,
        n_parts: usize,
        axis: usize,
        is_mhd: bool,
    ) -> Result<Self, D::Error> {
        let global = mesh_config.global_domain();

        let mut partitions = Vec::new();
        let part_size = (global.end[axis] - global.start[axis]) / n_parts as i64;

        for i in 0..n_parts {
            let mut owned = global;
            owned.start[axis] = global.start[axis] + (i as i64 * part_size);
            owned.end[axis] = if i == n_parts - 1 {
                global.end[axis]
            } else {
                owned.start[axis] + part_size
            };

            let nghosts = std::array::from_fn(|_| mesh_config.halo_width);
            let domains = DomainSet::new(owned, nghosts);

            let partition = PartitionState::allocate(device, i, level_id, domains, is_mhd)?;

            partitions.push(partition);
        }

        let halo_graph = HaloGraph::build_1d(n_parts, axis, mesh_config.halo_width);

        Ok(Self {
            level_id,
            refinement_ratio: 1,
            partitions,
            halo_graph,
            mesh_config,
            flux_registers: None,
            dt: 0.0,
            substeps: 1,
            _marker: PhantomData,
        })
    }

    pub fn num_partitions(&self) -> usize {
        self.partitions.len()
    }

    pub fn total_owned_cells(&self) -> usize {
        self.partitions.iter().map(|p| p.owned_size()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xpu_host::CpuDevice;

    struct TestRegime;

    #[test]
    fn test_mesh_config() {
        let config = MeshConfig::new([0.0, 0.0], [1.0, 1.0], [100, 100], 2);

        assert_eq!(config.dx[0], 0.01);
        assert_eq!(config.dx[1], 0.01);

        let domain = config.global_domain();
        assert_eq!(domain.start, [0, 0]);
        assert_eq!(domain.end, [100, 100]);

        let center = config.cell_center([0, 0]);
        assert!((center[0] - 0.005).abs() < 1e-10);
        assert!((center[1] - 0.005).abs() < 1e-10);
    }

    #[test]
    fn test_halo_graph() {
        let graph = HaloGraph::<2>::empty();
        assert_eq!(graph.links.len(), 0);
    }

    #[test]
    fn test_level_single_partition() {
        let device = CpuDevice::new(0).unwrap();
        let mesh = MeshConfig::new([0.0, 0.0], [1.0, 1.0], [100, 100], 2);

        let level =
            LevelState::<TestRegime, CpuDevice, 2>::single_partition(&device, mesh, 0, false)
                .unwrap();

        assert_eq!(level.level_id, 0);
        assert_eq!(level.num_partitions(), 1);
        assert_eq!(level.total_owned_cells(), 10000);
    }

    #[test]
    fn test_level_decomposed() {
        let device = CpuDevice::new(0).unwrap();
        let mesh = MeshConfig::new([0.0, 0.0], [1.0, 1.0], [99, 100], 2);

        let level =
            LevelState::<TestRegime, CpuDevice, 2>::decomposed(&device, mesh, 0, 3, 0, false)
                .unwrap();

        assert_eq!(level.num_partitions(), 3);
        assert_eq!(level.total_owned_cells(), 9900);
    }
}
