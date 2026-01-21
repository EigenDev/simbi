// =============================================================================
// world.rs
//
// LEGACY FILE - being replaced with new architecture
// temporarily disabled during transition
// =============================================================================

/*
// legacy code commented out during transition

use compute::Domain;
use physics::BoundaryCondition;
use std::marker::PhantomData;
use xpu_core::Device;

// =============================================================================
// physics configuration
// =============================================================================

#[derive(Debug, Clone, Copy)]
pub struct PhysicsConfig<const RANK: usize> {
    /// adiabatic index
    pub gamma: f64,
    /// physical domain bounds
    pub x_min: [f64; RANK],
    pub x_max: [f64; RANK],
    /// grid resolution
    pub n_cells: [usize; RANK],
    /// cell spacing
    pub dx: [f64; RANK],
    /// number of ghost zones per side
    pub nghosts: [usize; RANK],
}

impl<const RANK: usize> PhysicsConfig<RANK> {
    pub fn new(x_min: [f64; RANK], x_max: [f64; RANK], n_cells: [usize; RANK], gamma: f64) -> Self {
        let mut dx = [0.0; RANK];
        for ii in 0..RANK {
            dx[ii] = (x_max[ii] - x_min[ii]) / n_cells[ii] as f64;
        }

        Self {
            gamma,
            x_min,
            x_max,
            n_cells,
            dx,
            nghosts: [2; RANK], // default: 2 ghost zones for plm
        }
    }

    pub fn global_domain(&self) -> Domain<RANK> {
        let mut end = [0i64; RANK];
        for ii in 0..RANK {
            end[ii] = self.n_cells[ii] as i64;
        }
        Domain::new([0; RANK], end)
    }

    pub fn cell_center(&self, indices: [i64; RANK]) -> [f64; RANK] {
        let mut pos = [0.0; RANK];
        for ii in 0..RANK {
            pos[ii] = self.x_min[ii] + (indices[ii] as f64 + 0.5) * self.dx[ii];
        }
        pos
    }
}

// =============================================================================
// partition state (soa fields on device)
// =============================================================================

pub struct PartitionState<'d, R, D: Device, const RANK: usize> {
    /// conserved density (mass)
    pub den: compute::Field<'d, f64, D, RANK>,
    /// conserved momentum (per direction)
    pub mom: [compute::Field<'d, f64, D, RANK>; RANK],
    /// conserved energy (total)
    pub nrg: compute::Field<'d, f64, D, RANK>,

    /// domain (owned region + ghosts)
    pub domain: Domain<RANK>,
    /// device reference
    pub device: &'d D,
    /// which partition id
    pub id: usize,

    _regime: PhantomData<R>,
}

impl<'d, R, D: Device, const RANK: usize> PartitionState<'d, R, D, RANK> {
    /// create partition with zero-initialized fields
    pub fn zeros(device: &'d D, domain: Domain<RANK>, id: usize) -> Result<Self, D::Error> {
        let den = compute::Field::zeros(device, domain)?;
        let mom = std::array::from_fn(|_| compute::Field::zeros(device, domain).unwrap());
        let nrg = compute::Field::zeros(device, domain)?;

        Ok(Self {
            den,
            mom,
            nrg,
            domain,
            device,
            id,
            _regime: PhantomData,
        })
    }

    /// total number of cells (including ghosts)
    pub fn size(&self) -> usize {
        self.domain.size()
    }

    /// applies boundary condition to all fields in partition.
    pub fn apply_boundary<BC: BoundaryCondition>(
        &mut self,
        bc: &BC,
        nghosts: [usize; RANK],
    ) -> Result<(), D::Error> {
        bc.apply(
            &mut self.den,
            &mut self.mom,
            &mut self.nrg,
            self.domain,
            nghosts,
        )
    }
}

// =============================================================================
// halo communication graph
// =============================================================================

#[derive(Debug, Clone)]
pub struct HaloEdge<const RANK: usize> {
    pub src_partition: usize,
    pub dst_partition: usize,
    pub src_region: Domain<RANK>,
    pub dst_region: Domain<RANK>,
}

#[derive(Debug, Clone)]
pub struct HaloGraph<const RANK: usize> {
    edges: Vec<HaloEdge<RANK>>,
}

impl<const RANK: usize> HaloGraph<RANK> {
    pub fn empty() -> Self {
        Self { edges: Vec::new() }
    }

    pub fn add_edge(&mut self, edge: HaloEdge<RANK>) {
        self.edges.push(edge);
    }

    /// execute all halo exchanges
    /// for now: synchronous cpu copy (will be gpu-direct p2p later)
    pub fn exchange<R, D: Device>(
        &self,
        _partitions: &mut [PartitionState<R, D, RANK>],
    ) -> Result<(), D::Error> {
        // todo: implement halo exchange
        // for edge in &self.edges:
        //   copy src_partition[src_region] -> dst_partition[dst_region]
        Ok(())
    }
}

// =============================================================================
// world state
// =============================================================================

pub struct WorldState<'d, R, S, D: Device, const RANK: usize> {
    /// all partitions (one per device)
    pub partitions: Vec<PartitionState<'d, R, D, RANK>>,
    /// communication graph
    pub halo_graph: HaloGraph<RANK>,
    /// physics parameters
    pub config: PhysicsConfig<RANK>,
    /// current simulation time
    pub time: f64,
    /// solver type (zero-sized marker)
    _solver: PhantomData<S>,
}

impl<'d, R, S, D: Device, const RANK: usize> WorldState<'d, R, S, D, RANK> {
    /// create world state with single partition (no domain decomposition)
    pub fn single_device(device: &'d D, config: PhysicsConfig<RANK>) -> Result<Self, D::Error> {
        let global = config.global_domain();
        let partition = PartitionState::zeros(device, global, 0)?;

        Ok(Self {
            partitions: vec![partition],
            halo_graph: HaloGraph::empty(),
            config,
            time: 0.0,
            _solver: PhantomData,
        })
    }

    /// number of partitions
    pub fn num_partitions(&self) -> usize {
        self.partitions.len()
    }

    /// total cells across all partitions
    pub fn total_cells(&self) -> usize {
        self.partitions.iter().map(|p| p.size()).sum()
    }

    /// advance simulation by one timestep
    /// mathematical pipeline: Φ = boundary ∘ update ∘ flux ∘ reconstruct
    pub fn step(&mut self, _dt: f64) -> Result<(), D::Error> {
        // 1. exchange ghost zones (halo communication)
        self.halo_graph.exchange(&mut self.partitions)?;

        // 2. apply boundary conditions
        self.apply_boundaries()?;

        // 3. reconstruct at interfaces
        // todo: call reconstruction kernel

        // 4. compute fluxes (riemann solver)
        // todo: call flux kernel

        // 5. update conserved variables
        // todo: call update kernel

        // 6. advance time
        self.time += _dt;

        Ok(())
    }

    /// applies boundary conditions to all partitions.
    /// uses outflow bc by default (zero-gradient extrapolation).
    pub fn apply_boundaries(&mut self) -> Result<(), D::Error> {
        let bc = physics::OutflowBC;
        for partition in &mut self.partitions {
            partition.apply_boundary(&bc, self.config.nghosts)?;
        }
        Ok(())
    }

    /// compute timestep from cfl condition
    pub fn compute_dt(&self, _cfl: f64) -> Result<f64, D::Error> {
        // todo: reduce max wave speed across all partitions
        // dt = cfl * dx / max(|λ|)
        Ok(0.001) // placeholder
    }

    /// set initial conditions via function
    pub fn set_initial_conditions<F>(&mut self, _ic_func: F) -> Result<(), D::Error>
    where
        F: Fn([f64; RANK]) -> (f64, [f64; RANK], f64), // (rho, vel, p)
    {
        // todo: evaluate ic_func at each cell center, convert to conserved, copy to device
        Ok(())
    }

    /// copy solution to host for analysis
    pub fn to_host(&self) -> Result<HostSolution<RANK>, D::Error> {
        let partition = &self.partitions[0]; // single partition for now

        let size = partition.domain.size();
        let mut den_host = vec![0.0; size];
        let mut nrg_host = vec![0.0; size];

        partition
            .device
            .copy_to_host(partition.den.buffer(), &mut den_host)?;
        partition
            .device
            .copy_to_host(partition.nrg.buffer(), &mut nrg_host)?;

        let mom_host = std::array::from_fn(|d| {
            let mut buf = vec![0.0; size];
            partition
                .device
                .copy_to_host(partition.mom[d].buffer(), &mut buf)
                .unwrap();
            buf
        });

        Ok(HostSolution {
            den: den_host,
            mom: mom_host,
            nrg: nrg_host,
            domain: partition.domain,
        })
    }
}

// =============================================================================
// host solution (for diagnostics)
// =============================================================================

pub struct HostSolution<const RANK: usize> {
    pub den: Vec<f64>,
    pub mom: [Vec<f64>; RANK],
    pub nrg: Vec<f64>,
    pub domain: Domain<RANK>,
}

impl<const RANK: usize> HostSolution<RANK> {
    /// convert conserved to primitive at index
    pub fn primitive_at(&self, i: usize, gamma: f64) -> (f64, [f64; RANK], f64) {
        let rho = self.den[i];
        let vel = std::array::from_fn(|d| self.mom[d][i] / rho);
        let mom_sq: f64 = vel.iter().map(|v| v * v).sum();
        let ke = 0.5 * rho * mom_sq;
        let ie = self.nrg[i] - ke;
        let p = (gamma - 1.0) * ie;

        (rho, vel, p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xpu_host::CpuDevice;

    // dummy regime and solver markers for tests
    struct TestRegime;
    struct TestSolver;

    #[test]
    fn test_world_creation() {
        let device = CpuDevice::new(0).unwrap();
        let config = PhysicsConfig::new([0.0], [1.0], [100], 1.4);

        let world = WorldState::<TestRegime, TestSolver, _, 1>::single_device(&device, config);

        assert!(world.is_ok());
        let world = world.unwrap();
        assert_eq!(world.num_partitions(), 1);
        assert_eq!(world.time, 0.0);
    }

    #[test]
    fn test_config_domain() {
        let config = PhysicsConfig::new([0.0, 0.0], [1.0, 2.0], [100, 200], 1.4);

        let domain = config.global_domain();
        assert_eq!(domain.start, [0, 0]);
        assert_eq!(domain.end, [100, 200]);
        assert_eq!(domain.size(), 20000);
    }

    #[test]
    fn test_cell_centers() {
        let config = PhysicsConfig::new([0.0], [1.0], [10], 1.4);

        let center = config.cell_center([0]);
        assert!((center[0] - 0.05).abs() < 1e-10);

        let center = config.cell_center([5]);
        assert!((center[0] - 0.55).abs() < 1e-10);
    }

    #[test]
    fn test_partition_creation() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([100]);

        let partition = PartitionState::<TestRegime, _, 1>::zeros(&device, domain, 0).unwrap();

        assert_eq!(partition.size(), 100);
        assert_eq!(partition.id, 0);
    }

    #[test]
    fn test_halo_graph() {
        let mut graph = HaloGraph::<1>::empty();

        let edge = HaloEdge {
            src_partition: 0,
            dst_partition: 1,
            src_region: Domain::new([98], [100]),
            dst_region: Domain::new([0], [2]),
        };

        graph.add_edge(edge);
        assert_eq!(graph.edges.len(), 1);
    }

    #[test]
    fn test_step_placeholder() {
        let device = CpuDevice::new(0).unwrap();
        let config = PhysicsConfig::new([0.0], [1.0], [100], 1.4);

        let mut world =
            WorldState::<TestRegime, TestSolver, _, 1>::single_device(&device, config).unwrap();

        let dt = 0.001;
        world.step(dt).unwrap();

        assert_eq!(world.time, dt);
    }

    #[test]
    fn test_host_solution_conversion() {
        let solution = HostSolution {
            den: vec![1.0; 10],
            mom: [vec![0.5; 10]],
            nrg: vec![2.625; 10],
            domain: Domain::from_shape([10]),
        };

        let (rho, vel, p) = solution.primitive_at(0, 1.4);

        assert!((rho - 1.0).abs() < 1e-10);
        assert!((vel[0] - 0.5).abs() < 1e-10);
        assert!((p - 1.0).abs() < 1e-10);
    }
}
*/

// legacy module disabled
