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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_are_row_major() {
        assert_eq!(Shape::new(&[2, 3, 4]).contiguous_strides(), [12, 4, 1, 0]);
        assert_eq!(Shape::new(&[5]).contiguous_strides(), [1, 0, 0, 0]);
        assert_eq!(Shape::new(&[]).contiguous_strides(), [0, 0, 0, 0]);
    }

    #[test]
    fn numel_is_the_product_of_dims() {
        assert_eq!(Shape::new(&[2, 3, 4]).numel(), 24);
        assert_eq!(Shape::new(&[7]).numel(), 7);
        assert_eq!(Shape::new(&[]).numel(), 1);
    }

    #[test]
    fn size_one_axes_carry_an_unused_stride() {
        assert_eq!(Shape::new(&[1, 1, 7]).contiguous_strides(), [7, 7, 1, 0]);
    }

    #[test]
    fn zero_sized_dim_has_no_elements_and_does_not_panic() {
        let s = Shape::new(&[2, 0, 3]);
        assert_eq!(s.numel(), 0);
        assert_eq!(s.contiguous_strides(), [0, 3, 1, 0]);
    }

    #[test]
    fn dims_hides_the_padding() {
        let s = Shape::new(&[2, 3]);
        assert_eq!(s.rank(), 2);
        assert_eq!(s.dims(), &[2, 3]);
        assert_eq!(format!("{:?}", s), "[2, 3]");
    }

    #[test]
    #[should_panic(expected = "exceeds MAX_RANK")]
    fn rank_above_max_panics() {
        Shape::new(&[1, 2, 3, 4, 5]);
    }
}
