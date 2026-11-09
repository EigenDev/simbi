// =============================================================================
// regimes/wall_ledger.rs
//
// the host sums a conservation ledger books for the newtonian gas on a two-dimensional plane:
// - the outward rates of mass, angular momentum and E + rho Phi through each domain wall the
//   store's interior touches, from the stage's stored face fluxes and the walls' exact areas; an
//   energy-free closure books mass and angular momentum
// - the volume integrals of total energy, kinetic energy and angular momentum
// - the lowest density and specific internal energy of the conserved state, with their cells
// the geometry is the store's own: its maps or uniform spacing, its index offset and its
// coordinates, with cell centroids volume-weighted as the kernels weight them. angular momentum is
// about the symmetry axis on the spherical (r, theta) and cylindrical (R, z) planes and zero on a
// cartesian plane. the potential is a single static point mass at the origin, the one the face
// gravity work admits; a store with any other body gives no wall energy.
//
// three implementations share these definitions. on the axisymmetric planes with the azimuthal
// velocity the production path evaluates the integrands in kernels and reduces them where the data
// lives, an exact sum or a minimum, so device memory needs no host access: batched, several
// channels per launch and one synchronization per group (`ledger_batch_gv`), or one kernel and one
// reduction per channel (`ledger_gv`), the reference the batches equal bit for bit. the host sums
// (`*_host`) read host memory after the device queue drains; they serve the other planes and stand
// as the independent reference both kernel paths are gated against. all run only while a ledger
// scope is open.
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

pub fn wall_transport_host<const D: usize, const DOF: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric>(
    store: &FieldStore<D, DOF, Mem, Sc>,
) -> Option<[[Transport; 2]; 3]> {
    let plane = Plane::of(store)?;
    let phi = potential(store)?;
    let slot = plane.azimuthal_slot(DOF);
    let mut out = [[Transport::default(); 2]; 3];
    for ax in 0..2 {
        let flux = &store.fields.flux[ax];
        // an energy-free closure moves no energy: its wall receipt carries mass and angular momentum
        let (den, nrg) = (flux.den.view(), flux.nrg_field().map(|f| f.view()));
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
                if let Some(nrg) = &nrg {
                    sum.energy += sign * a * (nrg.at(cell).to_f64() + phi(radius) * mass);
                }
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

pub fn energy_partition_host<const D: usize, const DOF: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric>(
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

pub fn stage_extremes_host<const D: usize, const DOF: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric>(
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

// ------------------------------- the kernel path -------------------------------

use crate::regimes::substrate_kernels::{LedgerReduce, ledger_reduce};
use symbi_discretize::LedgerChannel;
use symbi_grid::Field;

/// the process's choice of receipt path, read once from `SIMBI_LEDGER_REFERENCE`: `host` selects
/// the host sums, `channels` the per-channel kernels, and any other value or none the batched
/// kernels. the selection reads the receipts alone; the evolved state is the same under each
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LedgerPath {
    Host,
    Channels,
    Batched,
}

pub fn ledger_path() -> LedgerPath {
    static PATH: std::sync::OnceLock<LedgerPath> = std::sync::OnceLock::new();
    *PATH.get_or_init(|| match std::env::var("SIMBI_LEDGER_REFERENCE").as_deref() {
        Ok("host") => LedgerPath::Host,
        Ok("channels") => LedgerPath::Channels,
        _ => LedgerPath::Batched,
    })
}

/// whether the receipts take a kernel path for this store: the two-axis spherical and cylindrical
/// planes carrying the azimuthal velocity, whose ledger kernels are baked, unless the process
/// selects the host reference
pub fn kernels_baked<const D: usize, const DOF: usize, Mem: MemoryStorage, Sc: Scalar + OrderedNumeric>(store: &FieldStore<D, DOF, Mem, Sc>) -> bool {
    ledger_path() != LedgerPath::Host && D == 2 && DOF == 3 && matches!(store.geom.coords, Geometry::Spherical | Geometry::Cylindrical)
}

/// the outward wall rates from the stage's stored fluxes, each integrand reduced by the kernels
pub fn wall_transport_kernels<const D: usize, const DOF: usize, Mem: MemoryStorage + Sync, Sc: Scalar + OrderedNumeric>(
    store: &FieldStore<D, DOF, Mem, Sc>,
    scratch: &Field<Sc, D, Mem>,
) -> Option<[[Transport; 2]; 3]> {
    let body = match store.immersed.as_ref() {
        None => false,
        Some(im) if crate::regimes::substrate_newton::admits_face_gravity_work(store.geom.coords, DOF, im.bodies.bodies()) => true,
        Some(_) => return None,
    };
    let g = &store.geom;
    let mut out = [[Transport::default(); 2]; 3];
    for ax in 0..2 {
        let (lo, hi) = (g.interior.spaces[ax].lo, g.interior.spaces[ax].hi);
        let whole = g.whole_interior(ax);
        for side in 0..2 {
            let face = if side == 0 { lo } else { hi };
            let on_wall = if side == 0 { lo + g.index_offset[ax] == whole.0 } else { hi + g.index_offset[ax] == whole.1 };
            if !on_wall {
                continue;
            }
            let mut plane = g.interior.clone();
            plane.spaces[ax].lo = face;
            plane.spaces[ax].hi = face + 1;
            let sign = if side == 0 { -1.0 } else { 1.0 };
            let sum = |channel| ledger_reduce(store, scratch, channel, &plane, 0.0, sign, LedgerReduce::Sum);
            let potential = if body { sum(LedgerChannel::WallPotential { axis: ax }) } else { 0.0 };
            out[ax][side] = Transport {
                mass: sign * sum(LedgerChannel::WallMass { axis: ax }),
                angular_momentum: sign * sum(LedgerChannel::WallAngular { axis: ax }),
                energy: sign * (sum(LedgerChannel::WallEnergy { axis: ax }) + potential),
                mass_in: sum(LedgerChannel::WallMassIn { axis: ax }),
            };
        }
    }
    Some(out)
}

/// the volume integrals of total energy, kinetic energy and angular momentum, reduced by the kernels
pub fn energy_partition_kernels<const D: usize, const DOF: usize, Mem: MemoryStorage + Sync, Sc: Scalar + OrderedNumeric>(
    store: &FieldStore<D, DOF, Mem, Sc>,
    scratch: &Field<Sc, D, Mem>,
) -> EnergyPartition {
    let sum = |channel| ledger_reduce(store, scratch, channel, &store.geom.interior, 0.0, 0.0, LedgerReduce::Sum);
    EnergyPartition {
        energy: sum(LedgerChannel::Energy),
        kinetic: sum(LedgerChannel::Kinetic),
        angular_momentum: sum(LedgerChannel::AngularMomentum),
    }
}

/// the lowest density and specific internal energy and their cells, reduced by the kernels: the
/// minimum, then the first cell holding it with the row index slowest, the order the host scan
/// visits. a non-finite value reports NaN at the first non-finite cell
pub fn stage_extremes_kernels<const D: usize, const DOF: usize, Mem: MemoryStorage + Sync, Sc: Scalar + OrderedNumeric>(
    store: &FieldStore<D, DOF, Mem, Sc>,
    scratch: &Field<Sc, D, Mem>,
) -> (Extreme, Extreme) {
    let g = &store.geom;
    let extreme = |low: LedgerChannel, locate: LedgerChannel| {
        let value = ledger_reduce(store, scratch, low, &g.interior, 0.0, 0.0, LedgerReduce::Min);
        if value == f64::INFINITY {
            return Extreme::default();
        }
        let code = ledger_reduce(store, scratch, locate, &g.interior, value, 0.0, LedgerReduce::Min);
        assert!(code.is_finite(), "ledger: no cell holds the reduced minimum {value:e}; the locate channel left its value");
        let (j, i) = ((code / 65536.0).floor() as isize, (code % 65536.0) as isize);
        let reported = if value == f64::NEG_INFINITY { f64::NAN } else { value };
        Extreme { value: reported, cell: [i + g.index_offset[0], j + g.index_offset[1], 0] }
    };
    (
        extreme(LedgerChannel::LowDensity, LedgerChannel::LocateDensity),
        extreme(LedgerChannel::LowSpecific, LedgerChannel::LocateSpecific),
    )
}

// ------------------------------- the batched kernel path -------------------------------
//
// each batch evaluates several channels in one launch, each into its own slot, and the channels'
// reductions run together with one synchronization: the wall sums of a stage as one group, the
// partition as one, the extremes as two (the minima, then the cells holding them). every channel
// shares its integrand with its per-channel kernel and reduces by the same exact sum or minimum,
// so the receipts equal the per-channel path's bit for bit.

use crate::regimes::substrate_kernels::ledger_batch_dispatch;
use symbi_discretize::LedgerBatch;

/// the receipt slots on the store's workspace: one set of `LedgerBatch::MAX_SLOTS` full-domain
/// fields per plane axis, allocated on the first batched receipt and held for the store's life.
/// a wall of axis `ax` writes set `ax`, so the walls of both axes evaluate before one reduction;
/// the two walls of one axis write disjoint planes of the same set
fn ledger_slots<const D: usize, const DOF: usize, Mem: MemoryStorage + Sync, Sc: Scalar + OrderedNumeric>(store: &FieldStore<D, DOF, Mem, Sc>) -> &[Field<Sc, D, Mem>] {
    store.workspace.ledger_scratch.get_or_init(|| {
        (0..2 * LedgerBatch::MAX_SLOTS)
            .map(|_| Field::<Sc, D, Mem>::zeros(&store.geom.allocated).expect("receipt scratch allocation"))
            .collect()
    })
}

/// the outward wall rates from the stage's stored fluxes: every wall's channels in one launch per
/// wall, all walls' sums in one reduction
pub fn wall_transport_batched<const D: usize, const DOF: usize, Mem: MemoryStorage + Sync, Sc: Scalar + OrderedNumeric>(
    store: &FieldStore<D, DOF, Mem, Sc>,
) -> Option<[[Transport; 2]; 3]> {
    let body = match store.immersed.as_ref() {
        None => false,
        Some(im) if crate::regimes::substrate_newton::admits_face_gravity_work(store.geom.coords, DOF, im.bodies.bodies()) => true,
        Some(_) => return None,
    };
    let g = &store.geom;
    let slots = ledger_slots(store);
    let width = if body { 5 } else { 4 };
    let mut walls = Vec::with_capacity(4);
    for ax in 0..2 {
        let (lo, hi) = (g.interior.spaces[ax].lo, g.interior.spaces[ax].hi);
        let whole = g.whole_interior(ax);
        for side in 0..2 {
            let face = if side == 0 { lo } else { hi };
            let on_wall = if side == 0 { lo + g.index_offset[ax] == whole.0 } else { hi + g.index_offset[ax] == whole.1 };
            if !on_wall {
                continue;
            }
            let mut plane = g.interior.clone();
            plane.spaces[ax].lo = face;
            plane.spaces[ax].hi = face + 1;
            let sign = if side == 0 { -1.0 } else { 1.0 };
            let set = &slots[ax * LedgerBatch::MAX_SLOTS..(ax + 1) * LedgerBatch::MAX_SLOTS];
            ledger_batch_dispatch(store, set, LedgerBatch::Wall { axis: ax, body }, &plane, [0.0, 0.0], sign);
            walls.push((ax, side, plane, sign));
        }
    }
    let items: Vec<(&Field<Sc, D, Mem>, &symbi_algebra::Domain<D>)> = walls
        .iter()
        .flat_map(|(ax, _, plane, _)| (0..width).map(move |k| (&slots[ax * LedgerBatch::MAX_SLOTS + k], plane)))
        .collect();
    let sums: Vec<f64> = symbi_exec::engine::field_sums_exact(&items).iter().map(|s| s.value()).collect();
    let mut out = [[Transport::default(); 2]; 3];
    for (ww, (ax, side, _, sign)) in walls.iter().enumerate() {
        // the slots in `LedgerBatch::Wall` channel order: mass, inflow, energy, angular, potential
        let s = &sums[ww * width..(ww + 1) * width];
        let potential = if body { s[4] } else { 0.0 };
        out[*ax][*side] = Transport { mass: sign * s[0], angular_momentum: sign * s[3], energy: sign * (s[2] + potential), mass_in: s[1] };
    }
    Some(out)
}

/// the volume integrals of total energy, kinetic energy and angular momentum: one launch, one
/// reduction
pub fn energy_partition_batched<const D: usize, const DOF: usize, Mem: MemoryStorage + Sync, Sc: Scalar + OrderedNumeric>(store: &FieldStore<D, DOF, Mem, Sc>) -> EnergyPartition {
    let slots = ledger_slots(store);
    let interior = &store.geom.interior;
    ledger_batch_dispatch(store, slots, LedgerBatch::Partition, interior, [0.0, 0.0], 0.0);
    let sums = symbi_exec::engine::field_sums_exact(&[(&slots[0], interior), (&slots[1], interior), (&slots[2], interior)]);
    EnergyPartition { energy: sums[0].value(), kinetic: sums[1].value(), angular_momentum: sums[2].value() }
}

/// the lowest density and specific internal energy and their cells, as [`stage_extremes_kernels`]
/// reports them: both minima in one launch and one reduction, then both cells in one launch and
/// one reduction
pub fn stage_extremes_batched<const D: usize, const DOF: usize, Mem: MemoryStorage + Sync, Sc: Scalar + OrderedNumeric>(store: &FieldStore<D, DOF, Mem, Sc>) -> (Extreme, Extreme) {
    use symbi_ir::emit::ReductionOp;
    let g = &store.geom;
    let slots = ledger_slots(store);
    ledger_batch_dispatch(store, slots, LedgerBatch::Extremes, &g.interior, [0.0, 0.0], 0.0);
    let values = symbi_exec::engine::field_reduces(&[(&slots[0], &g.interior), (&slots[1], &g.interior)], ReductionOp::Min);
    if values.iter().all(|v| *v == f64::INFINITY) {
        return (Extreme::default(), Extreme::default());
    }
    ledger_batch_dispatch(store, slots, LedgerBatch::Locate, &g.interior, [values[0], values[1]], 0.0);
    let codes = symbi_exec::engine::field_reduces(&[(&slots[0], &g.interior), (&slots[1], &g.interior)], ReductionOp::Min);
    let extreme = |value: f64, code: f64| {
        if value == f64::INFINITY {
            return Extreme::default();
        }
        assert!(code.is_finite(), "ledger: no cell holds the reduced minimum {value:e}; the locate channel left its value");
        let (j, i) = ((code / 65536.0).floor() as isize, (code % 65536.0) as isize);
        let reported = if value == f64::NEG_INFINITY { f64::NAN } else { value };
        Extreme { value: reported, cell: [i + g.index_offset[0], j + g.index_offset[1], 0] }
    };
    (extreme(values[0], codes[0]), extreme(values[1], codes[1]))
}

/// the bytes of receipt scratch the store holds: the workspace's ledger fields once allocated
pub fn scratch_bytes<const D: usize, const DOF: usize, Mem: MemoryStorage, Sc: Scalar + OrderedNumeric>(store: &FieldStore<D, DOF, Mem, Sc>) -> usize {
    let cells: usize = (0..D).map(|ax| store.geom.allocated.spaces[ax].size() as usize).product();
    store.workspace.ledger_scratch.get().map_or(0, |fields| fields.len() * cells * std::mem::size_of::<Sc>())
}
