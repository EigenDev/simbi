// =============================================================================
// reflux_captured_operands.rs
//
// the reflux register operands a gpu run recorded where a two-worker refined
// launch first departed from the single device (x-momentum at the first cell of
// a refined box's low x face, the first fine accumulation of a step), driven
// through the register code on either side of a cut: the single grid's local
// face, and the contributing face whose unit crosses a host copy into the
// receiving face's inbox and is folded there. the default memory is the gpu
// build's device memory and the host memory otherwise.
// - the folded register equals the local register, bit for bit
// - on the host both equal the separately rounded result; on the device, where
//   both register kernels compile `dst + scale * x` to one fused multiply-add,
//   both equal the fused result, the one the single gpu recorded
// - the unit carries the unscaled sum and the coefficient bits exactly
// - the old protocol (the sender scaled, the receiver added with 1.0)
//   reproduces the separately rounded result the launch recorded, which differs
//   from the fused one
//
// run: cargo test -p symbi --test reflux_captured_operands
//      cargo test -p symbi --features hip --test reflux_captured_operands   (on a gpu)
// =============================================================================
use symbi::sim::refinement::FluxRegister;
use symbi::sim::state::{ConsFieldsGeneric, axis_name};
use symbi_algebra::{Domain, Space};
use symbi_xpu::{DefaultMemory, MemoryStorage};

type Cons = ConsFieldsGeneric<3, 3, DefaultMemory>;
type Register = FluxRegister<3, 3, DefaultMemory>;

/// the register after the step's two coarse accumulations, the fine coefficient `area * w`,
/// each of the four fine x-momentum fluxes, and the results the single gpu and the launch
/// recorded
const REGISTER: u64 = 0xbe87c83d621ea656;
const COEFFICIENT: u64 = 0x3e7dba4cbaa64fec;
const FINE: u64 = 0x3fb9999999999999;
const FUSED: u64 = 0xbe81d62e0996fcc1;
const SEPARATE: u64 = 0xbe81d62e0996fcc0;
/// the x-momentum's position in a register face's component list
const MOMENTUM_X: usize = 1;

fn domain(lo: [isize; 3], hi: [isize; 3]) -> Domain<3> {
    Domain::new(std::array::from_fn(|a| Space { name: axis_name(a), lo: lo[a], hi: hi[a] }))
}

fn sync() {
    #[cfg(feature = "gpu")]
    if <DefaultMemory as MemoryStorage>::IS_DEVICE_ACCESSIBLE {
        symbi_xpu::ctx_sync();
    }
}

/// the fine fluxes over the fine cells under the box: the captured x-momentum flux on every
/// face, and distinct values in the other components
fn fine_fluxes(alloc: &Domain<3>) -> [Cons; 3] {
    std::array::from_fn(|dir| {
        let f = Cons::zeros_with_energy(alloc, true).expect("fine fluxes");
        let fields = [&f.den, &f.mom[0], &f.mom[1], &f.mom[2], f.nrg_field().unwrap()];
        for (comp, field) in fields.iter().enumerate() {
            let value = if comp == MOMENTUM_X { f64::from_bits(FINE) } else { 0.25 + 0.125 * comp as f64 + 0.0625 * dir as f64 };
            for c in alloc.iter() {
                field.view_mut().set(c, value);
            }
        }
        f
    })
}

/// write the captured register value into every cell of the momentum component of face (0, 0)
fn set_register(reg: &Register) {
    let (fields, dom) = reg.face(0, 0).expect("the low x register face");
    for c in dom.iter() {
        fields[MOMENTUM_X].view_mut().set(c, f64::from_bits(REGISTER));
    }
}

fn register_bits(reg: &Register) -> Vec<u64> {
    let (fields, dom) = reg.face(0, 0).expect("the low x register face");
    dom.iter().map(|c| fields[MOMENTUM_X].view().at(c).to_bits()).collect()
}

#[test]
fn the_captured_operands_fold_on_the_receiver_as_on_the_single_grid() {
    // the arithmetic of the two protocols on the captured operands, with the fused multiply-add
    // both register kernels compile to
    let (r, s, f) = (f64::from_bits(REGISTER), f64::from_bits(COEFFICIENT), f64::from_bits(FINE));
    let sum = (f + f) + (f + f);
    assert_eq!(s.mul_add(sum, r).to_bits(), FUSED, "the single device's fused accumulation");
    assert_eq!(1.0f64.mul_add(s.mul_add(sum, 0.0), r).to_bits(), SEPARATE, "the old transported fold");
    assert_ne!(FUSED, SEPARATE, "the operands separate the two protocols; the gate is vacuous");

    // the root [0, 4)^3, the box [1, 3)^3: its low x face lies on the cut at x = 1, the fine faces
    // on the right tile, the corrected coarse cells on the left
    let root = domain([0, 0, 0], [4, 4, 4]);
    let right_root = domain([1, 0, 0], [4, 4, 4]);
    let coverage = domain([1, 1, 1], [3, 3, 3]);
    let mono = Register::new(&coverage, &root, true, false, 0).expect("single-grid register");
    let sender = Register::new(&coverage, &right_root, true, false, 0)
        .expect("sender register")
        .with_contributing_faces(&coverage, [[true, false], [false, false], [false, false]], true, false, 0)
        .expect("contributing face");
    let owner = Register::receiving(&[(0, 0, domain([1, 1, 1], [2, 3, 3]))], true, false, 0).expect("owner register");
    // a transverse fine spacing of 1 makes the face area 1, so the coefficient is w exactly
    let fine_dx = [0.5, 1.0, 1.0];
    let w = f64::from_bits(COEFFICIENT);
    let fine = fine_fluxes(&domain([0, 0, 0], [8, 8, 8]));

    mono.zero_uniform();
    owner.zero_uniform();
    sender.zero_uniform();
    sender.zero_contributions();
    sync();
    set_register(&mono);
    set_register(&owner);

    mono.accumulate_fine_uniform(&fine, &fine_dx, 0, w);
    sender.accumulate_fine_uniform(&fine, &fine_dx, 0, w);
    sync();
    // the unit: the unscaled sum in every gas slot's momentum cell, the coefficient bits last
    let contributing = sender.contributing();
    assert_eq!(contributing.len(), 1, "one contributing face");
    let (slots, face) = (&contributing[0].1, contributing[0].2);
    for c in face.iter() {
        assert_eq!(slots[MOMENTUM_X].view().at(c).to_bits(), sum.to_bits(), "the unit carries the unscaled sum at {c:?}");
        assert_eq!(slots.last().unwrap().view().at(c).to_bits(), COEFFICIENT, "the unit carries the coefficient bits at {c:?}");
    }
    // the wire: every slot copied cell for cell into the owner's inbox
    let inboxes = owner.inboxes();
    assert_eq!(inboxes[0].1.len(), slots.len(), "the inbox lands every slot");
    for (src, dst) in slots.iter().zip(&inboxes[0].1) {
        for c in face.iter() {
            dst.view_mut().set(c, *src.view().at(c));
        }
    }
    owner.fold_inboxes();
    sync();

    let local = register_bits(&mono);
    let folded = register_bits(&owner);
    let expected = if <DefaultMemory as MemoryStorage>::IS_DEVICE_ACCESSIBLE { FUSED } else { SEPARATE };
    assert_eq!(local.len(), 4, "four face cells");
    for (k, (l, g)) in local.iter().zip(&folded).enumerate() {
        assert_eq!(*l, *g, "cell {k}: the receiver's fold {g:016x} differs from the single grid's {l:016x}");
        assert_eq!(*l, expected, "cell {k}: {l:016x}, expected {expected:016x} on this backend");
    }
}
