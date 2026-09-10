pub const MAX_RANK: usize = 4;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    dims: [usize; MAX_RANK],
    rank: usize,
}

impl Shape {
    pub fn new(dims: &[usize]) -> Shape {
        assert!(
            dims.len() <= MAX_RANK,
            "rank {} exceeds MAX_RANK {}",
            dims.len(),
            MAX_RANK
        );
        let mut d = [0usize; MAX_RANK];
        d[..dims.len()].copy_from_slice(dims);
        Shape {
            dims: d,
            rank: dims.len(),
        }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims[..self.rank]
    }

    pub fn dim(&self, i: usize) -> usize {
        self.dims()[i]
    }

    pub fn numel(&self) -> usize {
        self.dims().iter().product()
    }

    pub fn contiguous_strides(&self) -> [usize; MAX_RANK] {
        let mut s = [0usize; MAX_RANK];
        let mut acc = 1usize;
        for i in (0..self.rank).rev() {
            s[i] = acc;
            acc *= self.dims[i];
        }
        s
    }
}

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.dims())
    }
}
