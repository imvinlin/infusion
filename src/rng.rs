// using the xoshiro256++ algo with paper here: https://arxiv.org/pdf/1805.01407

pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut x = seed;
        let mut s = [0u64; 4];
        for word in &mut s {
            *word = splitmix64(&mut x);
        }
        Rng { s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    // standard normal by box-muller: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    pub fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_reference_implementation() {
        let mut r = Rng { s: [1, 2, 3, 4] };
        assert_eq!(r.next_u64(), 41_943_041);
        assert_eq!(r.next_u64(), 58_720_359);
        assert_eq!(r.next_u64(), 3_588_806_011_781_223);
        assert_eq!(r.next_u64(), 3_591_011_842_654_386);
    }

    #[test]
    fn seeding_matches_the_reference_splitmix64() {
        let mut r = Rng::new(42);
        assert_eq!(r.next_u64(), 15_021_278_609_987_233_951);
    }

    #[test]
    fn a_seed_is_reproducible_and_seeds_differ() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        let mut c = Rng::new(43);
        let x = a.next_u64();
        assert_eq!(x, b.next_u64());
        assert_ne!(x, c.next_u64());
    }

    const N: usize = 1_000_000;

    fn mean_var(xs: &[f64]) -> (f64, f64) {
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
        (mean, var)
    }

    #[test]
    fn uniforms_have_the_right_moments_and_range() {
        let mut r = Rng::new(7);
        let xs: Vec<f64> = (0..N).map(|_| r.uniform()).collect();
        assert!(xs.iter().all(|&x| (0.0..1.0).contains(&x)));
        let (mean, var) = mean_var(&xs);
        assert!((mean - 0.5).abs() < 2e-3, "mean {mean}");
        assert!((var - 1.0 / 12.0).abs() < 1e-3, "var {var}");
    }

    #[test]
    fn normals_have_the_right_moments() {
        let mut r = Rng::new(7);
        let xs: Vec<f64> = (0..N).map(|_| r.normal()).collect();
        let (mean, var) = mean_var(&xs);
        assert!(mean.abs() < 1e-2, "mean {mean}");
        assert!((var - 1.0).abs() < 1e-2, "var {var}");
    }
        assert!((var - 1.0 / 12.0).abs() < 1e-3, "var {var}");
    }

    #[test]
    fn normals_have_the_right_moments() {
        let mut r = Rng::new(7);
        let xs: Vec<f64> = (0..N).map(|_| r.normal()).collect();
        let (mean, var) = mean_var(&xs);
        assert!(mean.abs() < 1e-2, "mean {mean}");
        assert!((var - 1.0).abs() < 1e-2, "var {var}");
    }
}
