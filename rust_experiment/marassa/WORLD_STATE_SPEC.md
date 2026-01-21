# world state specification

complete design document for production-grade worldstate matching c++ ecs exactly.

---

## c++ structure analysis

### partition components (from components.hpp)

```cpp
// partition topology and connectivity
partition_t<Rank> {
    block_info_t<Rank> block;              // id, geometry, face connectivity
    domain_t<Rank> owned_domain;           // cells to compute (no ghosts)
    domain_t<Rank> allocated_domain;       // owned + ghost padding
    vector_t<domain_t<Rank>, Rank> face_domains;   // n+1 in normal direction
    vector_t<domain_t<Rank>, Rank> edge_domains;   // +1 in transverse dirs
    executor_t executor;                   // device context
    rank_id_t rank_id;                     // mpi rank info
}

// partition fields
partition_fields_t<Conserved, Primitive, Rank> {
    // cell-centered (allocated_domain)
    field_t<Conserved, Rank> cons;         // (den, mom, nrg, chi)
    field_t<Primitive, Rank> prim;         // (rho, vel, pre, chi)
    
    // face-centered fluxes (owned + 1 in normal direction)
    vector_t<field_t<Conserved, Rank>, Rank> flux;
    
    // mhd fields (optional)
    vector_t<field_t<real, Rank>, Rank> bfield;   // face-centered B_d
    vector_t<field_t<real, Rank>, Rank> efield;   // edge-centered E_d
}

// workspace for rk time integration
partition_workspace_t<Conserved, Primitive, Rank> {
    field_t<Conserved, Rank> u_n;          // state at t^n
    field_t<Primitive, Rank> prim_n;       // primitives at t^n
    vector_t<field_t<real, Rank>, Rank> e_n;     // efield at t^n
    field_t<Conserved, Rank> u_star;       // intermediate rk stage
}

// partition geometry
partition_geometry_t<Rank> {
    mesh_config_t<Rank> config;           // grid spacing, bounds, etc
}

// level decomposition (container for partitions)
level_decomposition_t<Rank> {
    skeleton_t<Rank> skeleton;             // block topology
    vector<partition_t<Rank>> partitions;
    vector<halo_link_t<Rank>> halo_graph;
    vector<entity_t> partition_entities;   // ecs handles
    topology_t topology;                   // global connectivity
}

// level metadata
level_info_t {
    uint64_t level_id;                     // 0 = coarsest
    uint64_t refinement_ratio;             // relative to parent
}

// level mesh
level_mesh_t<Rank> {
    mesh_config_t<Rank> config;
}

// amr flux registers
flux_register_component_t<Conserved, Rank> {
    vector<flux_register_t<Conserved, Rank>> registers;
    iarray<Rank> ratio;                    // refinement ratio
    bool initialized;
}

// simulation metadata
simulation_metadata_t<Rank> {
    // physics
    real gamma;
    real plm_theta;
    real viscosity;
    real cfl;
    real ambient_sound_speed;
    
    // time
    real time;
    real tend;
    real global_dt;
    vector<real> level_dts;                // dt per level
    real initial_time;
    
    // iteration
    uint64_t iteration;
    uint64_t checkpoint_index;
    
    // checkpointing
    real checkpoint_interval;
    real checkpoint_time;
    real prev_checkpoint_time;
    real dlogt;                            // log-spaced checkpoints
    
    // grid
    uint64_t dimensions;
    uint64_t halo_radius;
    iarray<3> resolution;
    
    // enums
    regime_t regime;                       // newtonian, srhd, rmhd, mhd
    solver_t solver;                       // hlle, hllc, hlld
    geometry_t coord_system;               // cartesian, spherical, etc
    reconstruction_t reconstruction;       // pcm, plm
    timestepping_t timestepping;          // euler, rk2
    shockwave_limiter_t shock_smoother;
    cellspacing_t x1_spacing, x2_spacing, x3_spacing;
    
    // boundaries
    vector_t<boundary_type_t, 2*Rank> boundary_conditions;
    
    // flags
    bool is_mhd;
    bool is_relativistic;
    
    // subcycling (amr)
    subcycling_mode_t subcycling_mode;    // none, standard, manual, adaptive
    vector<uint64_t> level_substeps;
    
    // io
    string data_dir;
    uint64_t checkpoint_zones;
}

// sources (user-defined functions)
sources_t<Rank> {
    expression_t<Rank> hydro_source;      // source term
    expression_t<Rank> gravity_source;    // gravity
    vector_t<expression_t<Rank>, 2*Rank> bc_sources;  // dynamic bcs
}

// immersed bodies (rigid body dynamics)
immersed_bodies_t<Rank> {
    body_collection_t<Rank> bodies;
}

// mesh motion (cosmology, moving frames)
mesh_motion_config_t {
    function<real(real)> scale_factor;           // a(t)
    function<real(real)> scale_factor_derivative; // adot(t)
    bool homologous;
}
```

---

## rust translation

### type hierarchy

```rust
// zero-sized regime markers
pub struct Newtonian;
pub struct Srhd;
pub struct Rmhd;
pub struct Mhd;

pub trait Regime: Copy + Clone + Send + Sync + 'static {
    const IS_RELATIVISTIC: bool;
    const IS_MHD: bool;
}

impl Regime for Newtonian {
    const IS_RELATIVISTIC: bool = false;
    const IS_MHD: bool = false;
}
// ... similarly for other regimes
```

### domain types

```rust
// cell-centered, face-centered, edge-centered domains
pub enum DomainType {
    CellCentered,
    FaceCentered { axis: usize },
    EdgeCentered { axis: usize },
}

pub struct DomainSet<const RANK: usize> {
    pub owned: Domain<RANK>,
    pub allocated: Domain<RANK>,              // owned + ghosts
    pub face: [Domain<RANK>; RANK],           // +1 in normal dir
    pub edge: [Domain<RANK>; RANK],           // +1 in transverse
}

impl<const RANK: usize> DomainSet<RANK> {
    pub fn new(owned: Domain<RANK>, nghosts: [usize; RANK]) -> Self {
        // allocated = expand owned by nghosts
        let mut alloc_start = owned.start;
        let mut alloc_end = owned.end;
        for d in 0..RANK {
            alloc_start[d] -= nghosts[d] as i64;
            alloc_end[d] += nghosts[d] as i64;
        }
        let allocated = Domain::new(alloc_start, alloc_end);
        
        // face domains: +1 in normal direction
        let face = std::array::from_fn(|axis| {
            let mut face_dom = owned;
            face_dom.end[axis] += 1;
            face_dom
        });
        
        // edge domains: +1 in transverse directions
        let edge = std::array::from_fn(|axis| {
            let mut edge_dom = owned;
            for d in 0..RANK {
                if d != axis {
                    edge_dom.end[d] += 1;
                }
            }
            edge_dom
        });
        
        Self { owned, allocated, face, edge }
    }
}
```

### field bundles

```rust
// conserved variables (soa)
pub struct ConservedFields<'d, D: Device, const RANK: usize> {
    pub den: Field<'d, f64, D, RANK>,
    pub mom: [Field<'d, f64, D, RANK>; RANK],
    pub nrg: Field<'d, f64, D, RANK>,
    pub chi: Field<'d, f64, D, RANK>,  // passive scalar
}

// primitive variables (soa)
pub struct PrimitiveFields<'d, D: Device, const RANK: usize> {
    pub rho: Field<'d, f64, D, RANK>,
    pub vel: [Field<'d, f64, D, RANK>; RANK],
    pub pre: Field<'d, f64, D, RANK>,
    pub chi: Field<'d, f64, D, RANK>,  // passive scalar (same name as conserved)
}

// flux fields (face-centered, per direction)
pub struct FluxFields<'d, D: Device, const RANK: usize> {
    pub den: [Field<'d, f64, D, RANK>; RANK],  // flux[axis]
    pub mom: [[Field<'d, f64, D, RANK>; RANK]; RANK],  // flux[axis][component]
    pub nrg: [Field<'d, f64, D, RANK>; RANK],
    pub chi: [Field<'d, f64, D, RANK>; RANK],
}

// mhd fields (optional)
pub struct MhdFields<'d, D: Device, const RANK: usize> {
    pub bfield: [Field<'d, f64, D, RANK>; RANK],  // face-centered B_d
    pub efield: [Field<'d, f64, D, RANK>; RANK],  // edge-centered E_d
}

// rk workspace
pub struct RkWorkspace<'d, D: Device, const RANK: usize> {
    pub u_n: ConservedFields<'d, D, RANK>,     // state at t^n
    pub prim_n: PrimitiveFields<'d, D, RANK>,  // primitives at t^n
    pub e_n: Option<MhdFields<'d, D, RANK>>,   // efield at t^n (mhd only)
    pub u_star: ConservedFields<'d, D, RANK>,  // intermediate stage
}
```

### partition state (complete)

```rust
pub struct PartitionState<'d, R: Regime, D: Device, const RANK: usize> {
    // identity
    pub id: usize,
    pub level: usize,
    
    // topology
    pub domains: DomainSet<RANK>,
    
    // device reference
    pub device: &'d D,
    
    // cell-centered state (allocated domain)
    pub conserved: ConservedFields<'d, D, RANK>,
    pub primitive: PrimitiveFields<'d, D, RANK>,
    
    // face-centered fluxes (owned + 1 in normal direction)
    pub fluxes: FluxFields<'d, D, RANK>,
    
    // mhd fields (optional, compile-time)
    pub mhd: Option<MhdFields<'d, D, RANK>>,
    
    // rk workspace (optional, allocated on demand)
    pub workspace: Option<Box<RkWorkspace<'d, D, RANK>>>,
    
    // boundary connectivity (for halo exchange)
    pub boundary_info: BoundaryInfo<RANK>,
    
    _regime: PhantomData<R>,
}

pub struct BoundaryInfo<const RANK: usize> {
    pub faces: [FaceConnection; 2 * RANK],
}

pub enum FaceConnection {
    Physical(BoundaryType),
    Internal { neighbor_id: usize },
    Periodic,
}

pub enum BoundaryType {
    Outflow,
    Reflecting,
    Dynamic,
}
```

### level state

```rust
pub struct LevelState<'d, R: Regime, D: Device, const RANK: usize> {
    pub level_id: usize,
    pub refinement_ratio: usize,  // relative to parent
    
    // all partitions at this level
    pub partitions: Vec<PartitionState<'d, R, D, RANK>>,
    
    // halo communication graph
    pub halo_graph: HaloGraph<RANK>,
    
    // mesh configuration
    pub mesh_config: MeshConfig<RANK>,
    
    // amr flux registers (if child level exists)
    pub flux_registers: Option<Vec<FluxRegister<'d, D, RANK>>>,
    
    // timestep for this level
    pub dt: f64,
    pub substeps: usize,  // for subcycling
}
```

### world state (top level)

```rust
pub struct WorldState<'d, R: Regime, S: Solver, D: Device, const RANK: usize> {
    // amr level hierarchy (index 0 = coarsest)
    pub levels: Vec<LevelState<'d, R, D, RANK>>,
    
    // global simulation metadata
    pub metadata: SimulationMetadata<RANK>,
    
    // source terms (user-defined functions)
    pub sources: SourceTerms<RANK>,
    
    // immersed bodies (optional)
    pub bodies: Option<ImmersedBodies<RANK>>,
    
    // mesh motion (cosmology)
    pub mesh_motion: Option<MeshMotion>,
    
    // current simulation time
    pub time: f64,
    pub iteration: usize,
    
    _solver: PhantomData<S>,
}

pub struct SimulationMetadata<const RANK: usize> {
    // physics parameters
    pub gamma: f64,
    pub plm_theta: f64,
    pub cfl: f64,
    
    // time control
    pub tend: f64,
    pub global_dt: f64,
    
    // checkpointing
    pub checkpoint_interval: f64,
    pub checkpoint_index: usize,
    
    // grid
    pub halo_radius: usize,
    pub resolution: [usize; 3],
    
    // enum configurations
    pub regime: RegimeType,
    pub solver: SolverType,
    pub reconstruction: ReconstructionType,
    pub timestepping: TimesteppingType,
    pub coord_system: GeometryType,
    
    // boundaries
    pub boundary_conditions: [BoundaryType; 2 * RANK],
    
    // subcycling
    pub subcycling_mode: SubcyclingMode,
    
    // io
    pub data_dir: String,
}
```

### halo communication

```rust
pub struct HaloGraph<const RANK: usize> {
    pub links: Vec<HaloLink<RANK>>,
}

pub struct HaloLink<const RANK: usize> {
    pub src_partition: usize,
    pub dst_partition: usize,
    pub src_region: Domain<RANK>,
    pub dst_region: Domain<RANK>,
    pub dimension: usize,
    pub direction: Side,
}

pub enum Side {
    Left,
    Right,
}
```

### amr flux correction

```rust
pub struct FluxRegister<'d, D: Device, const RANK: usize> {
    // one register per face direction
    registers: Vec<Field<'d, f64, D, RANK>>,
    coarse_domain: Domain<RANK>,
    refinement_ratio: [usize; RANK],
}

impl<'d, D: Device, const RANK: usize> FluxRegister<'d, D, RANK> {
    pub fn accumulate_coarse(&mut self, flux: &FluxFields<D, RANK>, dt: f64);
    pub fn accumulate_fine(&mut self, flux: &FluxFields<D, RANK>, dt: f64);
    pub fn apply_correction(&self, coarse: &mut ConservedFields<D, RANK>);
    pub fn zero(&mut self);
}
```

---

## allocation strategy

### single-level (no amr)

```rust
impl WorldState {
    pub fn single_level(
        device: &'d D,
        config: PhysicsConfig<RANK>,
    ) -> Result<Self, D::Error> {
        let owned = config.global_domain();
        let domains = DomainSet::new(owned, config.nghosts);
        
        // allocate fields on allocated domain (includes ghosts)
        let conserved = ConservedFields::zeros(device, domains.allocated)?;
        let primitive = PrimitiveFields::zeros(device, domains.allocated)?;
        
        // allocate flux fields on face domains
        let fluxes = FluxFields::zeros_on_faces(device, &domains.face)?;
        
        // mhd fields if needed
        let mhd = if R::IS_MHD {
            Some(MhdFields::zeros(device, &domains)?);
        } else {
            None
        };
        
        let partition = PartitionState {
            id: 0,
            level: 0,
            domains,
            device,
            conserved,
            primitive,
            fluxes,
            mhd,
            workspace: None,
            boundary_info: BoundaryInfo::all_physical(),
            _regime: PhantomData,
        };
        
        let level = LevelState {
            level_id: 0,
            refinement_ratio: 1,
            partitions: vec![partition],
            halo_graph: HaloGraph::empty(),
            mesh_config: config.mesh_config(),
            flux_registers: None,
            dt: 0.0,
            substeps: 1,
        };
        
        Ok(WorldState {
            levels: vec![level],
            metadata: config.metadata(),
            sources: SourceTerms::none(),
            bodies: None,
            mesh_motion: None,
            time: 0.0,
            iteration: 0,
            _solver: PhantomData,
        })
    }
}
```

### multi-partition decomposition

```rust
impl WorldState {
    pub fn decomposed(
        devices: &'d [D],
        config: PhysicsConfig<RANK>,
        axis: usize,  // decompose along this axis
    ) -> Result<Self, D::Error> {
        let global = config.global_domain();
        let n_parts = devices.len();
        
        // split domain along axis
        let mut partitions = Vec::new();
        let part_size = (global.end[axis] - global.start[axis]) / n_parts as i64;
        
        for (i, device) in devices.iter().enumerate() {
            let mut owned = global;
            owned.start[axis] = global.start[axis] + (i as i64 * part_size);
            owned.end[axis] = if i == n_parts - 1 {
                global.end[axis]
            } else {
                owned.start[axis] + part_size
            };
            
            let domains = DomainSet::new(owned, config.nghosts);
            
            // allocate fields for this partition
            let partition = PartitionState::allocate(
                i, 0, domains, device, R::IS_MHD
            )?;
            
            partitions.push(partition);
        }
        
        // build halo graph
        let halo_graph = HaloGraph::build_1d(&partitions, axis, config.nghosts[axis]);
        
        let level = LevelState {
            level_id: 0,
            refinement_ratio: 1,
            partitions,
            halo_graph,
            mesh_config: config.mesh_config(),
            flux_registers: None,
            dt: 0.0,
            substeps: 1,
        };
        
        Ok(WorldState {
            levels: vec![level],
            metadata: config.metadata(),
            sources: SourceTerms::none(),
            bodies: None,
            mesh_motion: None,
            time: 0.0,
            iteration: 0,
            _solver: PhantomData,
        })
    }
}
```

---

## evolution operators

### single step (euler timestepping)

```rust
impl WorldState {
    pub fn step(&mut self, dt: f64) -> Result<(), D::Error> {
        // for each level (coarsest first)
        for level_id in 0..self.levels.len() {
            self.advance_level_euler(level_id, dt)?;
        }
        
        self.time += dt;
        self.iteration += 1;
        
        Ok(())
    }
    
    fn advance_level_euler(&mut self, level_id: usize, dt: f64) 
        -> Result<(), D::Error> 
    {
        let level = &mut self.levels[level_id];
        
        // 1. cons2prim
        for partition in &mut level.partitions {
            partition.cons_to_prim(self.metadata.gamma)?;
        }
        
        // 2. halo exchange
        level.halo_graph.exchange(&mut level.partitions)?;
        
        // 3. apply boundary conditions
        for partition in &mut level.partitions {
            partition.apply_boundaries(&self.metadata.boundary_conditions)?;
        }
        
        // 4. reconstruct at interfaces
        for partition in &mut level.partitions {
            partition.reconstruct(self.metadata.plm_theta)?;
        }
        
        // 5. compute fluxes (riemann solver)
        for partition in &mut level.partitions {
            partition.compute_fluxes::<R, S>(self.metadata.gamma)?;
        }
        
        // 6. accumulate flux into parent level (if amr child exists)
        if level_id > 0 {
            self.accumulate_fine_flux(level_id, dt)?;
        }
        
        // 7. update conserved variables
        for partition in &mut level.partitions {
            partition.update_conserved(dt, level.mesh_config.dx)?;
        }
        
        // 8. if has child level, recurse with subcycling
        if level_id < self.levels.len() - 1 {
            let nsteps = level.substeps;
            for _ in 0..nsteps {
                self.advance_level_euler(level_id + 1, dt / nsteps as f64)?;
            }
            
            // restriction: inject fine interior back to coarse
            self.restrict_from_fine(level_id + 1)?;
            
            // reflux: apply flux correction
            self.apply_flux_correction(level_id)?;
        }
        
        Ok(())
    }
}
```

---

## implementation phases

### phase 1: single-level hydro (no amr, no mhd)
- worldstate with single level
- single partition
- conserved + primitive fields
- flux fields
- no workspace
- euler timestepping only

### phase 2: multi-partition (no amr, no mhd)
- domain decomposition
- halo exchange
- multi-device support

### phase 3: rk timestepping
- workspace allocation
- rk2/rk3 integrators
- u_n, u_star staging

### phase 4: amr
- multiple levels
- flux registers
- restriction/prolongation
- subcycling
- reflux

### phase 5: mhd
- bfield, efield
- constrained transport
- divergence cleaning

---

## key design decisions

### 1. field allocation
**decision:** allocate cell-centered fields on `allocated_domain` (owned + ghosts), flux fields on `face_domains`.

**rationale:** matches c++ exactly. ghost zones included in allocation from the start.

### 2. soa layout everywhere
**decision:** separate field per scalar component. no aos point types in storage.

**rationale:** gpu memory coalescing, cache-friendly, simpler than dual layout.

### 3. chi field naming
**decision:** both conserved and primitive have field named `chi`. conserved chi is `den * concentration`, primitive chi is `concentration`.

**rationale:** matches c++ naming. user thinks in terms of concentration, not density-weighted concentration.

### 4. optional mhd fields
**decision:** `Option<MhdFields>` at runtime, but regime marker determines at compile time.

**rationale:** if R::IS_MHD is false, compiler eliminates mhd branches entirely. if true, fields are always Some.

### 5. workspace allocation
**decision:** `Option<Box<RkWorkspace>>`. allocated lazily when rk2/rk3 selected.

**rationale:** euler doesn't need staging. saves memory for simple cases.

### 6. halo graph as explicit data structure
**decision:** `HaloGraph` separate from partitions, contains all communication edges.

**rationale:** enables optimization (overlap compute/comm), clear dependency graph, matches c++ topology.

### 7. level hierarchy as vec
**decision:** `Vec<LevelState>` where index = level id.

**rationale:** amr naturally hierarchical. coarsest at 0. enables recursive traversal.

---

## memory layout summary

```
partition memory (1d example, 100 cells, 2 ghosts):

allocated_domain: [-2, 102]  (104 elements)
  den: [ghost ghost | owned 0..100 | ghost ghost]
  mom: [ghost ghost | owned 0..100 | ghost ghost]
  nrg: [ghost ghost | owned 0..100 | ghost ghost]
  chi: [ghost ghost | owned 0..100 | ghost ghost]
  rho: [ghost ghost | owned 0..100 | ghost ghost]
  vel: [ghost ghost | owned 0..100 | ghost ghost]
  pre: [ghost ghost | owned 0..100 | ghost ghost]

face_domain[0]: [0, 101]  (101 elements)
  flux_den[0]: [interface 0 | ... | interface 100]
  flux_mom[0]: [interface 0 | ... | interface 100]
  flux_nrg[0]: [interface 0 | ... | interface 100]
  flux_chi[0]: [interface 0 | ... | interface 100]

total memory per partition (hydro only):
  cell-centered: 7 fields * 104 elements * 8 bytes = 5.8 kb
  flux-centered: 4 fields * 101 elements * 8 bytes = 3.2 kb
  total: ~9 kb for 100 cells

scales linearly with resolution.
```
