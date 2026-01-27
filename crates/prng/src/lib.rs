//! Stateless, splittable PRNG utilities compatible with JAX's Threefry keys.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use rand::{CryptoRng, RngCore};

/// A JAX-compatible PRNG key (two u32 words).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JaxKey {
    /// First 32-bit key word.
    pub k1: u32,
    /// Second 32-bit key word.
    pub k2: u32,
}

impl JaxKey {
    /// Create a new key from a 64-bit seed.
    pub fn new(seed: u64) -> Self {
        let (k1, k2) = threefry_seed(seed);
        Self { k1, k2 }
    }

    /// Create a key directly from raw u32 words.
    pub fn from_u32s(k1: u32, k2: u32) -> Self {
        Self { k1, k2 }
    }

    /// Split into `n` independent-looking keys.
    pub fn split(self, n: usize) -> Vec<Self> {
        if n == 0 {
            return Vec::new();
        }
        let total = n.saturating_mul(2);
        let mut counts = Vec::with_capacity(total);
        for i in 0..total {
            counts.push(i as u32);
        }
        let out = threefry_2x32(self, &counts);
        out.chunks(2)
            .map(|chunk| Self {
                k1: chunk[0],
                k2: chunk[1],
            })
            .collect()
    }

    /// Deterministically derive a subkey from additional data.
    pub fn fold_in(self, data: u64) -> Self {
        let (d1, d2) = threefry_seed(data);
        let (k1, k2) = threefry2x32_pair(self, d1, d2);
        Self { k1, k2 }
    }

    /// Convert the key into a concrete RNG backed by Threefry.
    pub fn to_rng(self) -> JaxRng {
        JaxRng::new(self)
    }
}

/// Stateless Threefry RNG stream derived from a key and an internal counter.
#[derive(Clone, Debug)]
pub struct JaxRng {
    key: JaxKey,
    counter: u64,
    buffer: [u32; 2],
    index: usize,
}

impl JaxRng {
    /// Create a new RNG stream from a key.
    pub fn new(key: JaxKey) -> Self {
        Self {
            key,
            counter: 0,
            buffer: [0; 2],
            index: 2,
        }
    }

    fn refill(&mut self) {
        let c0 = self.counter as u32;
        let c1 = self.counter.wrapping_add(1) as u32;
        let (y0, y1) = threefry2x32_pair(self.key, c0, c1);
        self.buffer = [y0, y1];
        self.index = 0;
        self.counter = self.counter.wrapping_add(2);
    }
}

impl RngCore for JaxRng {
    fn next_u32(&mut self) -> u32 {
        if self.index >= 2 {
            self.refill();
        }
        let out = self.buffer[self.index];
        self.index += 1;
        out
    }

    fn next_u64(&mut self) -> u64 {
        let lo = self.next_u32() as u64;
        let hi = self.next_u32() as u64;
        (hi << 32) | lo
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut remaining = dest;
        while !remaining.is_empty() {
            let chunk = self.next_u64().to_le_bytes();
            let take = remaining.len().min(chunk.len());
            remaining[..take].copy_from_slice(&chunk[..take]);
            remaining = &mut remaining[take..];
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl CryptoRng for JaxRng {}

/// SplitMix-style seed conversion mirroring JAX `threefry_seed`.
fn threefry_seed(seed: u64) -> (u32, u32) {
    let k1 = (seed >> 32) as u32;
    let k2 = (seed & 0xFFFF_FFFF) as u32;
    (k1, k2)
}

/// Apply the Threefry 2x32 hash to a single pair.
fn threefry2x32_pair(key: JaxKey, x0: u32, x1: u32) -> (u32, u32) {
    let ks0 = key.k1;
    let ks1 = key.k2;
    let ks2 = key.k1 ^ key.k2 ^ 0x1BD1_1BDA;

    let mut x0 = x0.wrapping_add(ks0);
    let mut x1 = x1.wrapping_add(ks1);

    // Round block 0.
    round(&mut x0, &mut x1, 13);
    round(&mut x0, &mut x1, 15);
    round(&mut x0, &mut x1, 26);
    round(&mut x0, &mut x1, 6);
    x0 = x0.wrapping_add(ks1);
    x1 = x1.wrapping_add(ks2).wrapping_add(1);

    // Round block 1.
    round(&mut x0, &mut x1, 17);
    round(&mut x0, &mut x1, 29);
    round(&mut x0, &mut x1, 16);
    round(&mut x0, &mut x1, 24);
    x0 = x0.wrapping_add(ks2);
    x1 = x1.wrapping_add(ks0).wrapping_add(2);

    // Round block 2.
    round(&mut x0, &mut x1, 13);
    round(&mut x0, &mut x1, 15);
    round(&mut x0, &mut x1, 26);
    round(&mut x0, &mut x1, 6);
    x0 = x0.wrapping_add(ks0);
    x1 = x1.wrapping_add(ks1).wrapping_add(3);

    // Round block 3.
    round(&mut x0, &mut x1, 17);
    round(&mut x0, &mut x1, 29);
    round(&mut x0, &mut x1, 16);
    round(&mut x0, &mut x1, 24);
    x0 = x0.wrapping_add(ks1);
    x1 = x1.wrapping_add(ks2).wrapping_add(4);

    // Round block 4.
    round(&mut x0, &mut x1, 13);
    round(&mut x0, &mut x1, 15);
    round(&mut x0, &mut x1, 26);
    round(&mut x0, &mut x1, 6);
    x0 = x0.wrapping_add(ks2);
    x1 = x1.wrapping_add(ks0).wrapping_add(5);

    (x0, x1)
}

#[inline]
fn round(x0: &mut u32, x1: &mut u32, rot: u32) {
    *x0 = x0.wrapping_add(*x1);
    *x1 = x1.rotate_left(rot);
    *x1 ^= *x0;
}

/// Apply the Threefry 2x32 hash to an array of counts.
fn threefry_2x32(key: JaxKey, counts: &[u32]) -> Vec<u32> {
    if counts.is_empty() {
        return Vec::new();
    }
    let mut flat: Vec<u32> = counts.to_vec();
    let odd = flat.len() % 2 == 1;
    if odd {
        flat.push(0);
    }
    let half = flat.len() / 2;
    let (left, right) = flat.split_at(half);

    let mut out0 = Vec::with_capacity(half);
    let mut out1 = Vec::with_capacity(half);
    for i in 0..half {
        let (y0, y1) = threefry2x32_pair(key, left[i], right[i]);
        out0.push(y0);
        out1.push(y1);
    }

    let mut out = Vec::with_capacity(flat.len());
    out.extend(out0);
    out.extend(out1);
    if odd {
        out.pop();
    }
    out
}

/// Common imports for PRNG utilities.
pub mod prelude {
    pub use crate::{JaxKey, JaxRng};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_deterministic() {
        let k = JaxKey::new(0);
        let a = k.split(4);
        let b = k.split(4);
        assert_eq!(a, b);
    }

    #[test]
    fn test_fold_in_changes_key() {
        let k = JaxKey::new(123);
        assert_ne!(k.fold_in(1), k.fold_in(2));
    }

    #[test]
    fn rng_is_deterministic() {
        let mut r1 = JaxKey::new(7).to_rng();
        let mut r2 = JaxKey::new(7).to_rng();
        for _ in 0..10 {
            assert_eq!(r1.next_u32(), r2.next_u32());
        }
    }
}
