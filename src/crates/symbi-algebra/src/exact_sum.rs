// =============================================================================
// exact_sum.rs
//
// an exact, order-independent sum of f64 terms: the long fixed-point
// accumulator of Kulisch & Miranker (1981), in the carry-save form ExBLAS uses
// (Collange, Defour, Graillat & Iakymchuk 2015). every finite f64 is an integer
// multiple of 2^-1074 below 2^1024, so a 2112-bit integer in units of 2^-1074
// holds any sum of them exactly. terms add into signed 32-bit digits held in
// i64 limbs; carries propagate lazily. two accumulators merge by adding their
// digits, so the total is the same integer for every partition of the terms
// and every order of the additions, and `value` rounds it once, to nearest with
// ties to even, as the correctly rounded sum of the terms.
//
// non-finite terms follow ieee addition: a NaN or a pair of opposite infinities
// gives NaN, otherwise an infinity of one sign gives that infinity. a finite
// total past the f64 range rounds to the infinity of its sign.
//
// usage:
//  let mut acc = ExactSum::new();
//  for x in terms { acc.add(x); }
//  acc.merge(&other_partial);
//  let total = acc.value();
// =============================================================================

/// the digit count: 66 digits of 32 bits span the 2098 bits of a finite f64 term's position and
/// mantissa (2^-1074 to 2^1024), and the rest absorb the carries of up to 2^60 terms
pub const DIGITS: usize = 70;

/// the flag bits of a term and of a sum's flag word: a NaN term, a positive infinity, a negative
/// infinity, a term other than negative zero, and at least one term
pub const FLAG_NAN: u64 = 1;
pub const FLAG_POS_INF: u64 = 2;
pub const FLAG_NEG_INF: u64 = 4;
pub const FLAG_NOT_NEGATIVE_ZERO: u64 = 8;
pub const FLAG_NONEMPTY: u64 = 16;

/// one term's exact contribution: three signed 32-bit slices added at digits `k`, `k + 1` and
/// `k + 2`, and the term's flags. a zero or non-finite term carries zero slices at digit 0
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TermDigits {
    pub k: usize,
    pub slices: [i64; 3],
    pub flags: u64,
}

/// the decomposition every exact accumulation shares: a finite term is mantissa * 2^(position -
/// 1074) with position = biased exponent - 1 (0 for a subnormal), its mantissa shifted by
/// position mod 32 and cut into three 32-bit slices from digit position / 32 upward, signed by
/// the term's sign
pub fn term_digits(x: f64) -> TermDigits {
    let bits = x.to_bits();
    let negative = bits >> 63 == 1;
    if !x.is_finite() {
        let flags = if x.is_nan() {
            FLAG_NAN
        } else if negative {
            FLAG_NEG_INF
        } else {
            FLAG_POS_INF
        };
        return TermDigits { k: 0, slices: [0; 3], flags: flags | FLAG_NOT_NEGATIVE_ZERO | FLAG_NONEMPTY };
    }
    let mut flags = FLAG_NONEMPTY;
    if !(x == 0.0 && negative) {
        flags |= FLAG_NOT_NEGATIVE_ZERO;
    }
    let exponent = ((bits >> 52) & 0x7ff) as i64;
    let fraction = bits & ((1u64 << 52) - 1);
    let (mantissa, position) = if exponent == 0 { (fraction, 0) } else { (fraction | (1u64 << 52), exponent - 1) };
    if mantissa == 0 {
        return TermDigits { k: 0, slices: [0; 3], flags };
    }
    let (k, r) = ((position / 32) as usize, (position % 32) as u32);
    let wide = (mantissa as u128) << r;
    let sign = if negative { -1 } else { 1 };
    let slices = std::array::from_fn(|j| sign * ((wide >> (32 * j)) as u64 & 0xffff_ffff) as i64);
    TermDigits { k, slices, flags }
}
/// digit additions between carry propagations, far below the 2^31 that fill an i64 limb
const CARRY_INTERVAL: u32 = 1 << 24;

/// an exact sum of f64 terms, mergeable across partitions
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactSum {
    /// signed digits in base 2^32, the digit `k` weighing 2^(32 k - 1074)
    digits: [i64; DIGITS],
    /// additions since the last carry propagation
    pending: u32,
    nan: bool,
    pos_inf: bool,
    neg_inf: bool,
    /// at least one term was added and every term was a negative zero (ieee keeps -0 only then)
    all_negative_zero: bool,
    empty: bool,
}

impl Default for ExactSum {
    fn default() -> Self {
        Self::new()
    }
}

impl ExactSum {
    /// the empty sum, zero
    pub fn new() -> Self {
        Self { digits: [0; DIGITS], pending: 0, nan: false, pos_inf: false, neg_inf: false, all_negative_zero: true, empty: true }
    }

    /// add one term exactly
    pub fn add(&mut self, x: f64) {
        let term = term_digits(x);
        self.apply_flags(term.flags);
        if term.slices == [0; 3] {
            return;
        }
        for (digit, slice) in self.digits[term.k..term.k + 3].iter_mut().zip(term.slices) {
            *digit += slice;
        }
        self.pending += 1;
        if self.pending >= CARRY_INTERVAL {
            self.carry();
        }
    }

    /// fold a flag word into the sum's flags
    fn apply_flags(&mut self, flags: u64) {
        self.nan |= flags & FLAG_NAN != 0;
        self.pos_inf |= flags & FLAG_POS_INF != 0;
        self.neg_inf |= flags & FLAG_NEG_INF != 0;
        if flags & FLAG_NOT_NEGATIVE_ZERO != 0 {
            self.all_negative_zero = false;
        }
        if flags & FLAG_NONEMPTY != 0 {
            self.empty = false;
        }
    }

    /// the sum of terms whose slices were added into `digits` without carries (each digit the
    /// plain integer sum of the slices `term_digits` gives at it) and whose flags were or-ed into
    /// `flags`: the accumulation a device performs. every digit sum must stay inside an i64, which
    /// holds for fewer than 2^31 terms
    pub fn from_uncarried(digits: &[i64; DIGITS], flags: u64) -> Result<Self, String> {
        if flags > 0x1f {
            return Err(format!("an exact sum's flag word {flags:#x} names no flag set"));
        }
        let mut acc = Self::new();
        acc.digits = *digits;
        acc.apply_flags(flags);
        acc.carry();
        Ok(acc)
    }

    /// add every term of `other`, as if its terms were added here
    pub fn merge(&mut self, other: &ExactSum) {
        let mut other = other.clone();
        other.carry();
        self.carry();
        for (digit, theirs) in self.digits.iter_mut().zip(other.digits.iter()) {
            *digit += theirs;
        }
        self.carry();
        self.nan |= other.nan;
        self.pos_inf |= other.pos_inf;
        self.neg_inf |= other.neg_inf;
        self.all_negative_zero &= other.all_negative_zero;
        self.empty &= other.empty;
    }

    /// the length of the wire form `to_words` writes
    pub const WORDS: usize = DIGITS + 1;

    /// the sum's wire form: its carried digits as 32-bit two's-complement words, the top one
    /// carrying the sign, then a word of flags. every word lies below 2^32, so none takes the bits
    /// of a word reserved for an absent value; `from_words` reads it back exactly
    pub fn to_words(&self) -> [u64; Self::WORDS] {
        let mut acc = self.clone();
        acc.carry();
        let top = acc.digits[DIGITS - 1];
        assert!(
            (-(1i64 << 31)..(1i64 << 31)).contains(&top),
            "an exact sum's top digit {top} leaves its 32-bit word; the total exceeds the accumulator"
        );
        let mut words = [0u64; Self::WORDS];
        for (word, digit) in words.iter_mut().zip(acc.digits.iter()) {
            *word = (*digit as u64) & 0xffff_ffff;
        }
        words[DIGITS] = u64::from(self.nan)
            | u64::from(self.pos_inf) << 1
            | u64::from(self.neg_inf) << 2
            | u64::from(self.all_negative_zero) << 3
            | u64::from(self.empty) << 4;
        words
    }

    /// the sum `to_words` wrote; a word out of its range is an error
    pub fn from_words(words: &[u64]) -> Result<Self, String> {
        if words.len() != Self::WORDS {
            return Err(format!("an exact sum travels as {} words, not {}", Self::WORDS, words.len()));
        }
        if let Some(k) = words[..DIGITS].iter().position(|&w| w > 0xffff_ffff) {
            return Err(format!("word {k} of an exact sum leaves its 32 bits: {:#x}", words[k]));
        }
        let flags = words[DIGITS];
        if flags > 0x1f {
            return Err(format!("an exact sum's flag word {flags:#x} names no flag set"));
        }
        let mut digits = [0i64; DIGITS];
        for (digit, word) in digits.iter_mut().zip(words.iter()).take(DIGITS - 1) {
            *digit = *word as i64;
        }
        digits[DIGITS - 1] = i64::from(words[DIGITS - 1] as u32 as i32);
        Ok(Self {
            digits,
            pending: 0,
            nan: flags & 1 != 0,
            pos_inf: flags & 2 != 0,
            neg_inf: flags & 4 != 0,
            all_negative_zero: flags & 8 != 0,
            empty: flags & 16 != 0,
        })
    }

    /// propagate carries: every digit below the top in [0, 2^32), the top digit carrying the sign
    fn carry(&mut self) {
        for k in 0..DIGITS - 1 {
            let carry = self.digits[k].div_euclid(1 << 32);
            self.digits[k] -= carry << 32;
            self.digits[k + 1] += carry;
        }
        self.pending = 0;
    }

    /// the sum, rounded once to the nearest f64, ties to even
    pub fn value(&self) -> f64 {
        if self.nan || (self.pos_inf && self.neg_inf) {
            return f64::NAN;
        }
        if self.pos_inf {
            return f64::INFINITY;
        }
        if self.neg_inf {
            return f64::NEG_INFINITY;
        }
        let mut acc = self.clone();
        acc.carry();
        let negative = acc.digits[DIGITS - 1] < 0;
        // the magnitude in unsigned digits: the two's complement negation of a negative total
        let mut magnitude = [0u64; DIGITS];
        if negative {
            let mut borrow = 0i64;
            for k in 0..DIGITS {
                let d = -acc.digits[k] - borrow;
                let r = d.rem_euclid(1 << 32);
                borrow = (r - d) >> 32;
                magnitude[k] = r as u64;
            }
        } else {
            for k in 0..DIGITS {
                magnitude[k] = acc.digits[k] as u64;
            }
        }
        let bit = |i: usize| (magnitude[i / 32] >> (i % 32)) & 1 == 1;
        let Some(top) = (0..DIGITS).rev().find(|&k| magnitude[k] != 0) else {
            return if self.all_negative_zero && !self.empty { -0.0 } else { 0.0 };
        };
        let length = 32 * top + (64 - magnitude[top].leading_zeros() as usize);
        let signed = |v: f64| if negative { -v } else { v };
        if length <= 53 {
            // below 2^53 units of 2^-1074 the total is an f64 as it stands
            let units = (0..length).fold(0u64, |q, i| q | (u64::from(bit(i)) << i));
            return signed(units as f64 * f64::from_bits(1));
        }
        let shift = length - 53;
        let mut q = (shift..length).fold(0u64, |q, i| q | (u64::from(bit(i)) << (i - shift)));
        let half = bit(shift - 1);
        let sticky = (0..shift - 1).any(bit);
        let mut exponent = shift as i64 - 1074;
        if half && (sticky || q & 1 == 1) {
            q += 1;
            if q == 1 << 53 {
                q >>= 1;
                exponent += 1;
            }
        }
        // q in [2^52, 2^53) at a scale of at least 2^-1073 is a normal number, exact when finite
        if exponent + 52 > 1023 {
            return signed(f64::INFINITY);
        }
        signed(q as f64 * pow2(exponent))
    }
}

/// 2^e for e in [-1074, 1023], exactly
fn pow2(e: i64) -> f64 {
    if e >= -1022 { f64::from_bits(((e + 1023) as u64) << 52) } else { f64::from_bits(1u64 << (e + 1074)) }
}

#[cfg(test)]
mod tests {
    use super::ExactSum;

    /// the exact sum by shewchuk's nonoverlapping expansion (the algorithm of python's
    /// `math.fsum`), rounded once: an independent reference for finite terms
    fn expansion_sum(terms: &[f64]) -> f64 {
        let mut partials: Vec<f64> = Vec::new();
        for &x in terms {
            let mut x = x;
            let mut kept = Vec::new();
            for &y in &partials {
                let (big, small) = if x.abs() < y.abs() { (y, x) } else { (x, y) };
                let hi = big + small;
                let lo = small - (hi - big);
                if lo != 0.0 {
                    kept.push(lo);
                }
                x = hi;
            }
            kept.push(x);
            partials = kept;
        }
        // round the expansion once, as fsum does: sum from the top, correcting the half-way case
        let mut hi = 0.0;
        let mut n = partials.len();
        if n > 0 {
            n -= 1;
            hi = partials[n];
            let mut lo = 0.0;
            while n > 0 {
                let x = hi;
                n -= 1;
                let y = partials[n];
                hi = x + y;
                let yr = hi - x;
                lo = y - yr;
                if lo != 0.0 {
                    break;
                }
            }
            if n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) || (lo > 0.0 && partials[n - 1] > 0.0)) {
                let y = lo * 2.0;
                let x = hi + y;
                let yr = x - hi;
                if y == yr {
                    hi = x;
                }
            }
        }
        hi
    }

    fn exact(terms: &[f64]) -> f64 {
        let mut acc = ExactSum::new();
        for &x in terms {
            acc.add(x);
        }
        acc.value()
    }

    /// a deterministic stream of terms spanning many binades and both signs
    fn terms(seed: u64, n: usize, spread: i32) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let mantissa = (s >> 11) as f64 / (1u64 << 53) as f64;
                let exponent = ((s >> 3) % (2 * spread as u64 + 1)) as i32 - spread;
                let sign = if s & 1 == 0 { 1.0 } else { -1.0 };
                sign * mantissa * 2f64.powi(exponent)
            })
            .collect()
    }

    #[test]
    fn cancellation_that_the_naive_sum_loses_is_kept() {
        assert_eq!(exact(&[1e100, 1.0, -1e100]), 1.0);
        assert_eq!(exact(&[1.0, 1e-300, -1.0]), 1e-300);
        assert_eq!(exact(&[0.1, 0.2, 0.3, -0.6]), expansion_sum(&[0.1, 0.2, 0.3, -0.6]));
    }

    #[test]
    fn the_sum_is_the_correctly_rounded_exact_sum() {
        for (seed, spread) in [(1, 4), (2, 60), (3, 300), (4, 1000)] {
            let t = terms(seed, 2000, spread);
            assert_eq!(exact(&t).to_bits(), expansion_sum(&t).to_bits(), "seed {seed} spread {spread}");
        }
    }

    #[test]
    fn every_order_and_partition_gives_the_same_bits() {
        let t = terms(7, 5000, 200);
        let whole = exact(&t);
        let mut reversed = t.clone();
        reversed.reverse();
        assert_eq!(exact(&reversed).to_bits(), whole.to_bits());
        for parts in [2, 3, 7, 64] {
            let mut total = ExactSum::new();
            for chunk in t.chunks(t.len().div_ceil(parts)).rev() {
                let mut partial = ExactSum::new();
                for &x in chunk {
                    partial.add(x);
                }
                total.merge(&partial);
            }
            assert_eq!(total.value().to_bits(), whole.to_bits(), "{parts} partitions");
        }
    }

    #[test]
    fn ties_round_to_even() {
        // 1 + 2^-53 lies halfway between 1 and its successor: even is 1
        assert_eq!(exact(&[1.0, 2f64.powi(-53)]), 1.0);
        // (1 + 2^-52) + 2^-53 lies halfway: even is 1 + 2^-51
        assert_eq!(exact(&[1.0 + 2f64.powi(-52), 2f64.powi(-53)]), 1.0 + 2f64.powi(-51));
        // a sticky bit below the half breaks the tie upward
        assert_eq!(exact(&[1.0, 2f64.powi(-53), 2f64.powi(-80)]), 1.0 + 2f64.powi(-52));
    }

    #[test]
    fn subnormal_and_extreme_terms_are_exact() {
        let tiny = f64::from_bits(1);
        assert_eq!(exact(&[tiny, tiny, tiny]), 3.0 * tiny);
        assert_eq!(exact(&[f64::MAX, -f64::MAX, tiny]), tiny);
        assert_eq!(exact(&[f64::MAX, f64::MAX]), f64::INFINITY);
        assert_eq!(exact(&[-f64::MAX, -f64::MAX, f64::MAX]), -f64::MAX);
    }

    #[test]
    fn non_finite_terms_follow_ieee_addition() {
        assert!(exact(&[1.0, f64::NAN]).is_nan());
        assert!(exact(&[f64::INFINITY, f64::NEG_INFINITY]).is_nan());
        assert_eq!(exact(&[1.0, f64::INFINITY]), f64::INFINITY);
        assert_eq!(exact(&[f64::NEG_INFINITY, 5.0]), f64::NEG_INFINITY);
        assert_eq!(exact(&[]).to_bits(), 0.0f64.to_bits());
        assert_eq!(exact(&[-0.0, -0.0]).to_bits(), (-0.0f64).to_bits());
        assert_eq!(exact(&[-0.0, 0.0]).to_bits(), 0.0f64.to_bits());
        assert_eq!(exact(&[1.0, -1.0]).to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn the_wire_form_carries_every_partition_exactly() {
        let t = terms(11, 3000, 600);
        let whole = exact(&t);
        let mut total = ExactSum::new();
        for chunk in t.chunks(700) {
            let mut partial = ExactSum::new();
            for &x in chunk {
                partial.add(x);
            }
            total.merge(&ExactSum::from_words(&partial.to_words()).expect("round trip"));
        }
        assert_eq!(total.value().to_bits(), whole.to_bits());
        let mut negative = ExactSum::new();
        negative.add(-3.5e200);
        negative.add(-f64::from_bits(1));
        let back = ExactSum::from_words(&negative.to_words()).unwrap();
        assert_eq!(back.to_words(), negative.to_words());
        assert_eq!(back.value(), negative.value());
    }

    #[test]
    fn a_negative_total_never_takes_the_absent_bits() {
        let mut negative = ExactSum::new();
        negative.add(-1.0);
        assert!(negative.to_words().iter().all(|&w| w <= 0xffff_ffff && w != u64::MAX));
    }

    #[test]
    fn the_flags_round_trip_and_malformed_words_are_refused() {
        for terms in [vec![f64::NAN], vec![f64::INFINITY], vec![f64::NEG_INFINITY], vec![-0.0], vec![]] {
            let mut acc = ExactSum::new();
            for x in &terms {
                acc.add(*x);
            }
            let back = ExactSum::from_words(&acc.to_words()).unwrap();
            assert_eq!(back.value().to_bits(), acc.value().to_bits(), "{terms:?}");
        }
        let mut words = ExactSum::new().to_words();
        words[3] = u64::MAX;
        assert!(ExactSum::from_words(&words).is_err());
        assert!(ExactSum::from_words(&words[1..]).is_err());
        let mut flags = ExactSum::new().to_words();
        flags[ExactSum::WORDS - 1] = 0x20;
        assert!(ExactSum::from_words(&flags).is_err());
    }

    #[test]
    fn uncarried_digit_sums_rebuild_the_sum_add_gives() {
        // the device accumulation: every term's slices summed into plain digits, flags or-ed
        let mut t = terms(17, 4000, 900);
        t.extend([0.0, -0.0, f64::from_bits(1), -f64::MAX, f64::MAX, 1e300, -1e-300]);
        let mut digits = [0i64; super::DIGITS];
        let mut flags = 0;
        for &x in &t {
            let term = super::term_digits(x);
            for (j, slice) in term.slices.iter().enumerate() {
                digits[term.k + j] += slice;
            }
            flags |= term.flags;
        }
        let device = ExactSum::from_uncarried(&digits, flags).unwrap();
        let mut host = ExactSum::new();
        for &x in &t {
            host.add(x);
        }
        assert_eq!(device.to_words(), host.to_words());
        assert_eq!(device.value().to_bits(), host.value().to_bits());
        // the flag cases: empty, all negative zero, non-finite
        for case in [vec![], vec![-0.0], vec![-0.0, -0.0], vec![-0.0, 0.0], vec![f64::NAN], vec![f64::INFINITY, 1.0],
                     vec![f64::NEG_INFINITY], vec![f64::INFINITY, f64::NEG_INFINITY]] {
            let mut digits = [0i64; super::DIGITS];
            let mut flags = 0;
            let mut host = ExactSum::new();
            for &x in &case {
                let term = super::term_digits(x);
                for (j, slice) in term.slices.iter().enumerate() {
                    digits[term.k + j] += slice;
                }
                flags |= term.flags;
                host.add(x);
            }
            let device = ExactSum::from_uncarried(&digits, flags).unwrap();
            assert_eq!(device.value().to_bits(), host.value().to_bits(), "{case:?}");
            assert_eq!(device.to_words(), host.to_words(), "{case:?}");
        }
        assert!(ExactSum::from_uncarried(&[0; super::DIGITS], 0x20).is_err());
    }

    #[test]
    fn the_largest_uncarried_digits_a_window_holds_carry_inside_an_i64() {
        // fewer than 2^31 terms, one slice of magnitude at most 2^32 - 1 per digit each
        let bound = ((1i64 << 31) - 1) * ((1i64 << 32) - 1);
        for sign in [1i64, -1] {
            // every digit at the bound: the chained carry (an overflow panics in a debug build)
            let mut digits = [sign * bound; super::DIGITS];
            digits[super::DIGITS - 1] = 0;
            let chained = ExactSum::from_uncarried(&digits, super::FLAG_NONEMPTY | super::FLAG_NOT_NEGATIVE_ZERO).unwrap();
            assert_eq!(chained.value().signum(), sign as f64);
            // one digit at the bound equals its carried form
            let mut one = [0i64; super::DIGITS];
            one[10] = sign * bound;
            let mut carried = [0i64; super::DIGITS];
            carried[10] = (sign * bound).rem_euclid(1 << 32);
            carried[11] = (sign * bound).div_euclid(1 << 32);
            let flags = super::FLAG_NONEMPTY | super::FLAG_NOT_NEGATIVE_ZERO;
            assert_eq!(
                ExactSum::from_uncarried(&one, flags).unwrap().to_words(),
                ExactSum::from_uncarried(&carried, flags).unwrap().to_words()
            );
        }
    }

    #[test]
    fn many_carries_stay_exact() {
        // past the carry interval the digits renormalize without loss
        let mut acc = ExactSum::new();
        for _ in 0..(1u32 << 25) {
            acc.add(0.5);
        }
        assert_eq!(acc.value(), (1u64 << 24) as f64);
    }
}
