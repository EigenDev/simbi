// =============================================================================
// rng.rs
//
// a small, seeded, deterministic pseudo-random generator for the monte-carlo
// photon transfer (xoshiro256** seeded via splitmix64). zero dependencies.
//
// a non-reproducible entropy source would give two runs of the same snapshot
// different photon catalogs, and the monte-carlo paths could not be regression-tested.
// a seeded generator makes science runs reproducible and the tests deterministic —
// the same (snapshot, seed) always yields the same events.
//
// usage:
//  let mut rng = Rng::seed(0xC0FFEE);
//  let u = rng.uniform(); // f64 in [0, 1)
// =============================================================================

/// xoshiro256** state. construct with `Rng::seed`; draw with `uniform`.
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// seed the four-word state from a single u64 via splitmix64 (the recommended
    /// initialization for xoshiro), so even a poor seed gives a well-mixed state.
    pub fn seed(seed: u64) -> Self {
        let mut z = seed;
        let mut splitmix = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng {
            s: [splitmix(), splitmix(), splitmix(), splitmix()],
        }
    }

    /// the next 64 raw random bits (xoshiro256** scrambler).
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// a uniform double in [0, 1) using the top 53 bits (the f64 mantissa width).
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // the same seed reproduces the same stream; different seeds diverge.
    #[test]
    fn seed_is_deterministic() {
        let mut a = Rng::seed(42);
        let mut b = Rng::seed(42);
        let mut c = Rng::seed(43);
        let sa: Vec<f64> = (0..8).map(|_| a.uniform()).collect();
        let sb: Vec<f64> = (0..8).map(|_| b.uniform()).collect();
        let sc: Vec<f64> = (0..8).map(|_| c.uniform()).collect();
        assert_eq!(sa, sb, "same seed -> same stream");
        assert_ne!(sa, sc, "different seed -> different stream");
    }

    // draws stay in [0, 1) and average near 0.5 over many samples.
    #[test]
    fn uniform_is_in_unit_interval() {
        let mut rng = Rng::seed(7);
        let n = 100_000;
        let mut sum = 0.0;
        for _ in 0..n {
            let u = rng.uniform();
            assert!((0.0..1.0).contains(&u), "u={u}");
            sum += u;
        }
        let mean = sum / n as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean {mean} should be ~0.5");
    }
}
