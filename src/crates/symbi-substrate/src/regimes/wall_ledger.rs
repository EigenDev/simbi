// =============================================================================
// regimes/wall_ledger.rs
//
// the host sums a conservation ledger books for the newtonian gas on a two-dimensional plane:
// - the outward rates of mass, angular momentum and E + rho Phi through each domain wall the
//   store's interior touches, from the stage's stored face fluxes and the walls' exact areas
// - the volume integrals of total energy, kinetic energy and angular momentum
// - the lowest density and specific internal energy of the conserved state, with their cells
// the geometry is the store's own: its maps or uniform spacing, its index offset and its
// coordinates, with cell centroids volume-weighted as the kernels weight them. angular momentum is
// about the symmetry axis on the spherical (r, theta) and cylindrical (R, z) planes and zero on a
// cartesian plane. the potential is a single static point mass at the origin, the one the face
// gravity work admits; a store with any other body gives no wall energy.
//
// the sums read host memory after the device queue drains, so a run on host-inaccessible device
// memory books no receipts; these are diagnostics, run only while a ledger scope is open.
// =============================================================================
use std::f64::consts::PI;

use symbi_algebra::OrderedNumeric;
use symbi_carrier::Scalar;
use symbi_geometry::Geometry;
use symbi_sim::conservation_ledger::{Extreme, Transport};
use symbi_sim::state::FieldStore;
use symbi_sim::substrate_seam::EnergyPartition;
use symbi_xpu::{MemorySpace, MemoryStorage};

/// the plane's cells and walls, on the whole-grid index frame
struct Plane {
    coords: Geometry,
    /// grid axis -> coordinate index: (r, theta) on the sphere, (R, z) on the cylinder
    axes: [usize; 2],
    lo: [isize; 2],
    hi: [isize; 2],
    offset: [isize; 2],
    whole: [(isize, isize); 2],
    faces: [Vec<f64>; 2],
}

impl Plane {
    fn of<const D: usize, const DOF: usize, Mem: MemoryStorage, Sc: Scalar + OrderedNumeric>(store: &FieldStore<D, DOF, Mem, Sc>) -> Option<Plane> {
        if D != 2 {
            return None;
        }
        let g = &store.geom;
        let (lo, hi) = ([g.interior.spaces[0].lo, g.interior.spaces[1].lo], [g.interior.spaces[0].hi, g.interior.spaces[1].hi]);
        let offset = [g.index_offset[0], g.index_offset[1]];
        let face = |ax: usize, whole: isize| match &g.maps {
            Some(maps) => maps[ax].face(whole),
            None => g.x_lo[ax] + whole as f64 * g.dx[ax],
        };
        let faces = std::array::from_fn(|ax| (lo[ax]..=hi[ax]).map(|i| face(ax, i + offset[ax])).collect());
        Some(Plane { coords: g.coords, axes: [g.axes[0], g.axes[1]], lo, hi, offset, whole: [g.whole_interior(0), g.whole_interior(1)], faces })
    }

    fn face(&self, ax: usize, local: isize) -> f64 {
        self.faces[ax][(local - self.lo[ax]) as usize]
    }

    /// the volume-weighted centroid of cell `local` along `ax`
    fn centroid(&self, ax: usize, local: isize) -> f64 {
        let (a, b) = (self.face(ax, local), self.face(ax, local + 1));
        match (self.coords, ax) {
            (Geometry::Spherical, 0) => (b.powi(4) - a.powi(4)) / 4.0 / ((b.powi(3) - a.powi(3)) / 3.0),
            (Geometry::Spherical, _) => ((b.sin() - b * b.cos()) - (a.sin() - a * a.cos())) / (a.cos() - b.cos()),
            (Geometry::Cylindrical, 0) => (b.powi(3) - a.powi(3)) / 3.0 / ((b * b - a * a) / 2.0),
            _ => 0.5 * (a + b),
        }
    }

    fn volume(&self, i: isize, j: isize) -> f64 {
        let ((r0, r1), (t0, t1)) = ((self.face(0, i), self.face(0, i + 1)), (self.face(1, j), self.face(1, j + 1)));
        match self.coords {
            Geometry::Spherical => 2.0 * PI * (r1.powi(3) - r0.powi(3)) / 3.0 * (t0.cos() - t1.cos()),
            Geometry::Cylindrical => PI * (r1 * r1 - r0 * r0) * (t1 - t0),
            _ => (r1 - r0) * (t1 - t0),
        }
    }

    /// the area of the face at local face index `f` along `ax`, crossed by transverse cell `t`
    fn area(&self, ax: usize, f: isize, t: isize) -> f64 {
        let other = 1 - ax;
        let (c0, c1) = (self.face(other, t), self.face(other, t + 1));
        let x = self.face(ax, f);
        match (self.coords, ax) {
            (Geometry::Spherical, 0) => 2.0 * PI * x * x * (c0.cos() - c1.cos()),
            (Geometry::Spherical, _) => PI * (c1 * c1 - c0 * c0) * x.sin(),
            (Geometry::Cylindrical, 0) => 2.0 * PI * x * (c1 - c0),
            (Geometry::Cylindrical, _) => PI * (c1 * c1 - c0 * c0),
            _ => c1 - c0,
        }
    }

    /// the distance from the symmetry axis and from the origin of a point given by grid coordinates
    fn lever_and_radius(&self, x0: f64, x1: f64) -> (f64, f64) {
        match self.coords {
            Geometry::Spherical => (x0 * x1.sin(), x0),
            Geometry::Cylindrical => (x0, (x0 * x0 + x1 * x1).sqrt()),
            _ => (0.0, (x0 * x0 + x1 * x1).sqrt()),
        }
    }

    /// the momentum slot of the azimuthal velocity, where the plane carries one
    fn azimuthal_slot(&self, dof: usize) -> Option<usize> {
        if dof != 3 || self.coords == Geometry::Cartesian {
            return None;
        }
        Some(3 - self.axes[0] - self.axes[1])
    }
}

/// the potential of the store's body, when it holds a single static point mass at the origin
fn potential<const D: usize, const DOF: usize, Mem: MemoryStorage, Sc: Scalar + OrderedNumeric>(
    store: &FieldStore<D, DOF, Mem, Sc>,
) -> Option<Box<dyn Fn(f64) -> f64>> {
    let Some(im) = store.immersed.as_ref() else {
        return Some(Box::new(|_| 0.0));
    };
    if !crate::regimes::substrate_newton::admits_face_gravity_work(store.geom.coords, DOF, im.bodies.bodies()) {
        return None;
    }
    let body = im.bodies.get(0);
    let (mass, soft) = (body.mass, body.softening().unwrap_or(0.0));
    let kind = body.softening_kind.as_scalar();
    Some(Box::new(move |r: f64| symbi_discretize::ibm::body_potential([r, 0.0, 0.0], mass, soft, kind)))
}

/// the outward wall rates, from the stage's stored fluxes
pub fn wall_transport<const D: usize, const DOF: usize, Mem: MemoryStorage, Sc: Scalar + OrderedNumeric>(
    store: &FieldStore<D, DOF, Mem, Sc>,
) -> Option<[[Transport; 2]; 3]> {
    symbi_exec::engine::device_sync::<Mem>();
    wall_transport_host(store.host()?)
}

fn wall_transport_host<const D: usize, const DOF: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric>(
    store: &FieldStore<D, DOF, Mem, Sc>,
) -> Option<[[Transport; 2]; 3]> {
    let plane = Plane::of(store)?;
    let phi = potential(store)?;
    let slot = plane.azimuthal_slot(DOF);
    let mut out = [[Transport::default(); 2]; 3];
    for ax in 0..2 {
        let flux = &store.fields.flux[ax];
        let (den, nrg) = (flux.den.view(), flux.nrg_field()?.view());
        let phi_flux = slot.map(|k| flux.mom[k].view());
        let other = 1 - ax;
        for side in 0..2 {
            let local = if side == 0 { plane.lo[ax] } else { plane.hi[ax] };
            let on_wall = if side == 0 { local + plane.offset[ax] == plane.whole[ax].0 } else { local + plane.offset[ax] == plane.whole[ax].1 };
            if !on_wall {
                continue;
            }
            let sign = if side == 0 { -1.0 } else { 1.0 };
            let x = plane.face(ax, local);
            let mut sum = Transport::default();
            for t in plane.lo[other]..plane.hi[other] {
                let mut cell = [0isize; D];
                cell[ax] = local;
                cell[other] = t;
                let a = plane.area(ax, local, t);
                let transverse = plane.centroid(other, t);
                let (lever, radius) = if ax == 0 { plane.lever_and_radius(x, transverse) } else { plane.lever_and_radius(transverse, x) };
                let mass = den.at(cell).to_f64();
                sum.mass += sign * a * mass;
                sum.mass_in += (-sign * a * mass).max(0.0);
                sum.energy += sign * a * (nrg.at(cell).to_f64() + phi(radius) * mass);
                if let Some(view) = &phi_flux {
                    sum.angular_momentum += sign * a * lever * view.at(cell).to_f64();
                }
            }
            out[ax][side] = sum;
        }
    }
    Some(out)
}

/// the volume integrals of total energy, kinetic energy and angular momentum
pub fn energy_partition<const D: usize, const DOF: usize, Mem: MemoryStorage, Sc: Scalar + OrderedNumeric>(
    store: &FieldStore<D, DOF, Mem, Sc>,
) -> Option<EnergyPartition> {
    symbi_exec::engine::device_sync::<Mem>();
    energy_partition_host(store.host()?)
}

fn energy_partition_host<const D: usize, const DOF: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric>(
    store: &FieldStore<D, DOF, Mem, Sc>,
) -> Option<EnergyPartition> {
    let plane = Plane::of(store)?;
    let c = &store.fields.cons;
    let (den, nrg) = (c.den.view(), c.nrg_field()?.view());
    let mom: Vec<_> = (0..DOF).map(|k| c.mom[k].view()).collect();
    let slot = plane.azimuthal_slot(DOF);
    let mut out = EnergyPartition::default();
    for j in plane.lo[1]..plane.hi[1] {
        for i in plane.lo[0]..plane.hi[0] {
            let mut cell = [0isize; D];
            cell[0] = i;
            cell[1] = j;
            let v = plane.volume(i, j);
            let rho = den.at(cell).to_f64();
            let m2: f64 = mom.iter().map(|m| m.at(cell).to_f64().powi(2)).sum();
            out.energy += nrg.at(cell).to_f64() * v;
            out.kinetic += 0.5 * m2 / rho * v;
            if let Some(k) = slot {
                let (lever, _) = plane.lever_and_radius(plane.centroid(0, i), plane.centroid(1, j));
                out.angular_momentum += mom[k].at(cell).to_f64() * lever * v;
            }
        }
    }
    Some(out)
}

/// the lowest density and specific internal energy of the conserved state. a cell with a
/// non-finite value reports NaN at that cell
pub fn stage_extremes<const D: usize, const DOF: usize, Mem: MemoryStorage, Sc: Scalar + OrderedNumeric>(
    store: &FieldStore<D, DOF, Mem, Sc>,
) -> Option<(Extreme, Extreme)> {
    symbi_exec::engine::device_sync::<Mem>();
    stage_extremes_host(store.host()?)
}

fn stage_extremes_host<const D: usize, const DOF: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric>(
    store: &FieldStore<D, DOF, Mem, Sc>,
) -> Option<(Extreme, Extreme)> {
    let plane = Plane::of(store)?;
    let c = &store.fields.cons;
    let (den, nrg) = (c.den.view(), c.nrg_field()?.view());
    let mom: Vec<_> = (0..DOF).map(|k| c.mom[k].view()).collect();
    let (mut density, mut specific) = (Extreme::default(), Extreme::default());
    for j in plane.lo[1]..plane.hi[1] {
        for i in plane.lo[0]..plane.hi[0] {
            let mut cell = [0isize; D];
            cell[0] = i;
            cell[1] = j;
            let at = [i + plane.offset[0], j + plane.offset[1], 0];
            let rho = den.at(cell).to_f64();
            let m2: f64 = mom.iter().map(|m| m.at(cell).to_f64().powi(2)).sum();
            let e = (nrg.at(cell).to_f64() - 0.5 * m2 / rho) / rho;
            let rho_value = if rho.is_finite() { rho } else { f64::NAN };
            let e_value = if e.is_finite() && m2.is_finite() { e } else { f64::NAN };
            if rho_value.is_nan() || rho_value < density.value {
                if !density.value.is_nan() {
                    density = Extreme { value: rho_value, cell: at };
                }
            }
            if e_value.is_nan() || e_value < specific.value {
                if !specific.value.is_nan() {
                    specific = Extreme { value: e_value, cell: at };
                }
            }
        }
    }
    Some((density, specific))
}
