// =============================================================================
// partition.rs
//
// production-grade partition state matching c++ ecs exactly.
// contains complete field structure: conserved, primitive, fluxes, mhd fields,
// and rk workspace.
//
// design:
//   - matches c++ partition_fields_t, partition_workspace_t exactly
//   - soa layout for all fields
//   - cell-centered: allocated domain (owned + ghosts)
//   - face-centered: owned + 1 in normal direction
//   - edge-centered: owned + 1 in transverse directions
//   - chi field in both conserved and primitive (same name)
//
// usage:
//   let partition = PartitionState::allocate(
//       device, domains, id, level, is_mhd
//   )?;
// =============================================================================

use compute::{Domain, Field};
use std::marker::PhantomData;
use xpu_core::Device;

// =============================================================================
// domain set (owned, allocated, face, edge)
// =============================================================================

#[derive(Debug, Clone, Copy)]
pub struct DomainSet<const RANK: usize> {
    pub owned: Domain<RANK>,
    pub allocated: Domain<RANK>,
    pub face: [Domain<RANK>; RANK],
    pub edge: [Domain<RANK>; RANK],
}

impl<const RANK: usize> DomainSet<RANK> {
    pub fn new(owned: Domain<RANK>, nghosts: [usize; RANK]) -> Self {
        let mut alloc_start = owned.start;
        let mut alloc_end = owned.end;

        for d in 0..RANK {
            alloc_start[d] -= nghosts[d] as i64;
            alloc_end[d] += nghosts[d] as i64;
        }

        let allocated = Domain::new(alloc_start, alloc_end);

        let face = std::array::from_fn(|axis| {
            let mut face_dom = owned;
            face_dom.end[axis] += 1;
            face_dom
        });

        let edge = std::array::from_fn(|axis| {
            let mut edge_dom = owned;
            for d in 0..RANK {
                if d != axis {
                    edge_dom.end[d] += 1;
                }
            }
            edge_dom
        });

        Self {
            owned,
            allocated,
            face,
            edge,
        }
    }
}

// =============================================================================
// conserved fields (soa)
// =============================================================================

pub struct ConservedFields<'d, D: Device, const RANK: usize> {
    pub den: Field<'d, f64, D, RANK>,
    pub mom: [Field<'d, f64, D, RANK>; RANK],
    pub nrg: Field<'d, f64, D, RANK>,
    pub chi: Field<'d, f64, D, RANK>,
}

impl<'d, D: Device, const RANK: usize> ConservedFields<'d, D, RANK> {
    pub fn zeros(device: &'d D, domain: Domain<RANK>) -> Result<Self, D::Error> {
        Ok(Self {
            den: Field::zeros(device, domain)?,
            mom: std::array::from_fn(|_| Field::zeros(device, domain).unwrap()),
            nrg: Field::zeros(device, domain)?,
            chi: Field::zeros(device, domain)?,
        })
    }
}

// =============================================================================
// primitive fields (soa)
// =============================================================================

pub struct PrimitiveFields<'d, D: Device, const RANK: usize> {
    pub rho: Field<'d, f64, D, RANK>,
    pub vel: [Field<'d, f64, D, RANK>; RANK],
    pub pre: Field<'d, f64, D, RANK>,
    pub chi: Field<'d, f64, D, RANK>,
}

impl<'d, D: Device, const RANK: usize> PrimitiveFields<'d, D, RANK> {
    pub fn zeros(device: &'d D, domain: Domain<RANK>) -> Result<Self, D::Error> {
        Ok(Self {
            rho: Field::zeros(device, domain)?,
            vel: std::array::from_fn(|_| Field::zeros(device, domain).unwrap()),
            pre: Field::zeros(device, domain)?,
            chi: Field::zeros(device, domain)?,
        })
    }
}

// =============================================================================
// flux fields (face-centered, per direction)
// =============================================================================

pub struct FluxFields<'d, D: Device, const RANK: usize> {
    pub den: [Field<'d, f64, D, RANK>; RANK],
    pub mom: [[Field<'d, f64, D, RANK>; RANK]; RANK],
    pub nrg: [Field<'d, f64, D, RANK>; RANK],
    pub chi: [Field<'d, f64, D, RANK>; RANK],
}

impl<'d, D: Device, const RANK: usize> FluxFields<'d, D, RANK> {
    pub fn zeros_on_faces(
        device: &'d D,
        face_domains: &[Domain<RANK>; RANK],
    ) -> Result<Self, D::Error> {
        Ok(Self {
            den: std::array::from_fn(|d| Field::zeros(device, face_domains[d]).unwrap()),
            mom: std::array::from_fn(|d| {
                std::array::from_fn(|_c| Field::zeros(device, face_domains[d]).unwrap())
            }),
            nrg: std::array::from_fn(|d| Field::zeros(device, face_domains[d]).unwrap()),
            chi: std::array::from_fn(|d| Field::zeros(device, face_domains[d]).unwrap()),
        })
    }
}

// =============================================================================
// mhd fields (optional)
// =============================================================================

pub struct MhdFields<'d, D: Device, const RANK: usize> {
    pub bfield: [Field<'d, f64, D, RANK>; RANK],
    pub efield: [Field<'d, f64, D, RANK>; RANK],
}

impl<'d, D: Device, const RANK: usize> MhdFields<'d, D, RANK> {
    pub fn zeros(
        device: &'d D,
        face_domains: &[Domain<RANK>; RANK],
        edge_domains: &[Domain<RANK>; RANK],
    ) -> Result<Self, D::Error> {
        Ok(Self {
            bfield: std::array::from_fn(|d| Field::zeros(device, face_domains[d]).unwrap()),
            efield: std::array::from_fn(|d| Field::zeros(device, edge_domains[d]).unwrap()),
        })
    }
}

// =============================================================================
// rk workspace
// =============================================================================

pub struct RkWorkspace<'d, D: Device, const RANK: usize> {
    pub u_n: ConservedFields<'d, D, RANK>,
    pub prim_n: PrimitiveFields<'d, D, RANK>,
    pub e_n: Option<[Field<'d, f64, D, RANK>; RANK]>,
    pub u_star: ConservedFields<'d, D, RANK>,
}

impl<'d, D: Device, const RANK: usize> RkWorkspace<'d, D, RANK> {
    pub fn allocate(device: &'d D, domain: Domain<RANK>, is_mhd: bool) -> Result<Self, D::Error> {
        let e_n = if is_mhd {
            Some(std::array::from_fn(|_| {
                Field::zeros(device, domain).unwrap()
            }))
        } else {
            None
        };

        Ok(Self {
            u_n: ConservedFields::zeros(device, domain)?,
            prim_n: PrimitiveFields::zeros(device, domain)?,
            e_n,
            u_star: ConservedFields::zeros(device, domain)?,
        })
    }
}

// =============================================================================
// boundary connectivity
// =============================================================================

#[derive(Debug, Clone, Copy)]
pub enum FaceConnection {
    Physical(BoundaryType),
    Internal { neighbor_id: usize },
    Periodic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryType {
    Outflow,
    Reflecting,
    Dynamic,
}

#[derive(Debug, Clone)]
pub struct BoundaryInfo {
    pub faces: Vec<FaceConnection>,
}

impl BoundaryInfo {
    pub fn all_physical<const RANK: usize>(bc_type: BoundaryType) -> Self {
        Self {
            faces: vec![FaceConnection::Physical(bc_type); 2 * RANK],
        }
    }

    pub fn all_outflow<const RANK: usize>() -> Self {
        Self::all_physical::<RANK>(BoundaryType::Outflow)
    }
}

// =============================================================================
// partition state (complete)
// =============================================================================

pub struct PartitionState<'d, R, D: Device, const RANK: usize> {
    pub id: usize,
    pub level: usize,
    pub domains: DomainSet<RANK>,
    pub device: &'d D,

    pub conserved: ConservedFields<'d, D, RANK>,
    pub primitive: PrimitiveFields<'d, D, RANK>,
    pub fluxes: FluxFields<'d, D, RANK>,

    pub mhd: Option<MhdFields<'d, D, RANK>>,
    pub workspace: Option<Box<RkWorkspace<'d, D, RANK>>>,

    pub boundary_info: BoundaryInfo,

    _regime: PhantomData<R>,
}

impl<'d, R, D: Device, const RANK: usize> PartitionState<'d, R, D, RANK> {
    pub fn allocate(
        device: &'d D,
        id: usize,
        level: usize,
        domains: DomainSet<RANK>,
        is_mhd: bool,
    ) -> Result<Self, D::Error> {
        let conserved = ConservedFields::zeros(device, domains.allocated)?;
        let primitive = PrimitiveFields::zeros(device, domains.allocated)?;
        let fluxes = FluxFields::zeros_on_faces(device, &domains.face)?;

        let mhd = if is_mhd {
            Some(MhdFields::zeros(device, &domains.face, &domains.edge)?)
        } else {
            None
        };

        Ok(Self {
            id,
            level,
            domains,
            device,
            conserved,
            primitive,
            fluxes,
            mhd,
            workspace: None,
            boundary_info: BoundaryInfo::all_outflow::<RANK>(),
            _regime: PhantomData,
        })
    }

    pub fn allocate_workspace(&mut self) -> Result<(), D::Error> {
        if self.workspace.is_none() {
            let ws =
                RkWorkspace::allocate(self.device, self.domains.allocated, self.mhd.is_some())?;
            self.workspace = Some(Box::new(ws));
        }
        Ok(())
    }

    pub fn owned_size(&self) -> usize {
        self.domains.owned.size()
    }

    pub fn allocated_size(&self) -> usize {
        self.domains.allocated.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xpu_host::CpuDevice;

    struct TestRegime;

    #[test]
    fn test_domain_set() {
        let owned = Domain::new([10, 10], [20, 20]);
        let nghosts = [2, 2];

        let domains = DomainSet::new(owned, nghosts);

        assert_eq!(domains.owned.start, [10, 10]);
        assert_eq!(domains.owned.end, [20, 20]);

        assert_eq!(domains.allocated.start, [8, 8]);
        assert_eq!(domains.allocated.end, [22, 22]);

        assert_eq!(domains.face[0].start, [10, 10]);
        assert_eq!(domains.face[0].end, [21, 20]);

        assert_eq!(domains.face[1].start, [10, 10]);
        assert_eq!(domains.face[1].end, [20, 21]);

        assert_eq!(domains.edge[0].start, [10, 10]);
        assert_eq!(domains.edge[0].end, [20, 21]);

        assert_eq!(domains.edge[1].start, [10, 10]);
        assert_eq!(domains.edge[1].end, [21, 20]);
    }

    #[test]
    fn test_conserved_fields_allocation() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([10, 10]);

        let fields = ConservedFields::<_, 2>::zeros(&device, domain).unwrap();

        assert_eq!(fields.den.domain(), domain);
        assert_eq!(fields.mom[0].domain(), domain);
        assert_eq!(fields.mom[1].domain(), domain);
        assert_eq!(fields.nrg.domain(), domain);
        assert_eq!(fields.chi.domain(), domain);
    }

    #[test]
    fn test_primitive_fields_allocation() {
        let device = CpuDevice::new(0).unwrap();
        let domain = Domain::from_shape([10, 10]);

        let fields = PrimitiveFields::<_, 2>::zeros(&device, domain).unwrap();

        assert_eq!(fields.rho.domain(), domain);
        assert_eq!(fields.vel[0].domain(), domain);
        assert_eq!(fields.vel[1].domain(), domain);
        assert_eq!(fields.pre.domain(), domain);
        assert_eq!(fields.chi.domain(), domain);
    }

    #[test]
    fn test_flux_fields_allocation() {
        let device = CpuDevice::new(0).unwrap();

        let face_domains = [Domain::new([0, 0], [11, 10]), Domain::new([0, 0], [10, 11])];

        let fluxes = FluxFields::<_, 2>::zeros_on_faces(&device, &face_domains).unwrap();

        assert_eq!(fluxes.den[0].domain(), face_domains[0]);
        assert_eq!(fluxes.den[1].domain(), face_domains[1]);
    }

    #[test]
    fn test_partition_allocation() {
        let device = CpuDevice::new(0).unwrap();
        let owned = Domain::new([0, 0], [10, 10]);
        let nghosts = [2, 2];
        let domains = DomainSet::new(owned, nghosts);

        let partition =
            PartitionState::<TestRegime, _, 2>::allocate(&device, 0, 0, domains, false).unwrap();

        assert_eq!(partition.id, 0);
        assert_eq!(partition.level, 0);
        assert_eq!(partition.owned_size(), 100);
        assert_eq!(partition.allocated_size(), 196);
        assert!(partition.mhd.is_none());
        assert!(partition.workspace.is_none());
    }

    #[test]
    fn test_partition_with_mhd() {
        let device = CpuDevice::new(0).unwrap();
        let owned = Domain::new([0, 0], [10, 10]);
        let nghosts = [2, 2];
        let domains = DomainSet::new(owned, nghosts);

        let partition =
            PartitionState::<TestRegime, _, 2>::allocate(&device, 0, 0, domains, true).unwrap();

        assert!(partition.mhd.is_some());

        let mhd = partition.mhd.as_ref().unwrap();
        assert_eq!(mhd.bfield.len(), 2);
        assert_eq!(mhd.efield.len(), 2);
    }

    #[test]
    fn test_workspace_allocation() {
        let device = CpuDevice::new(0).unwrap();
        let owned = Domain::new([0, 0], [10, 10]);
        let nghosts = [2, 2];
        let domains = DomainSet::new(owned, nghosts);

        let mut partition =
            PartitionState::<TestRegime, _, 2>::allocate(&device, 0, 0, domains, false).unwrap();

        assert!(partition.workspace.is_none());

        partition.allocate_workspace().unwrap();

        assert!(partition.workspace.is_some());

        let ws = partition.workspace.as_ref().unwrap();
        assert_eq!(ws.u_n.den.domain(), domains.allocated);
        assert!(ws.e_n.is_none());
    }

    #[test]
    fn test_boundary_info() {
        let info = BoundaryInfo::all_outflow::<2>();

        assert_eq!(info.faces.len(), 4);

        for face in &info.faces {
            match face {
                FaceConnection::Physical(BoundaryType::Outflow) => {}
                _ => panic!("expected outflow boundary"),
            }
        }
    }
}
