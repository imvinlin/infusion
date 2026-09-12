use crate::shape::{MAX_RANK, Shape};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TensorId(u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StorageId(u32);

#[derive(Clone, Copy, Debug)]
struct Tensor {
    shape: Shape,
    strides: [usize; MAX_RANK],
    offset: usize,
    storage: StorageId,
}

#[derive(Default)]
pub struct Graph<T> {
    tensors: Vec<Tensor>,
    storages: Vec<Vec<T>>,
}

impl<T: Copy + Default> Graph<T> {
    pub fn new() -> Graph<T> {
        Graph {
            tensors: Vec::new(),
            storages: Vec::new(),
        }
    }

    pub fn push(&mut self, shape: Shape, data: Vec<T>) -> TensorId {
        assert_eq!(
            shape.numel(),
            data.len(),
            "shape {:?} needs {} elements got {}",
            shape,
            shape.numel(),
            data.len()
        );
        let storage = StorageId(self.storages.len() as u32);
        self.storages.push(data);
        let id = TensorId(self.tensors.len() as u32);
        self.tensors.push(Tensor {
            shape,
            strides: shape.contiguous_strides(),
            offset: 0,
            storage,
        });
        id
    }

    pub fn from_slice(&mut self, data: &[T], dims: &[usize]) -> TensorId {
        self.push(Shape::new(dims), data.to_vec())
    }

    pub fn zeros(&mut self, dims: &[usize]) -> TensorId {
        let shape = Shape::new(dims);
        self.push(shape, vec![T::default(); shape.numel()])
    }

    fn tensor(&self, id: TensorId) -> &Tensor {
        &self.tensors[id.0 as usize]
    }

    pub fn shape(&self, id: TensorId) -> Shape {
        self.tensor(id).shape
    }

    pub fn strides(&self, id: TensorId) -> [usize; MAX_RANK] {
        self.tensor(id).strides
    }

    pub fn numel(&self, id: TensorId) -> usize {
        self.tensor(id).shape.numel()
    }

    // Tensor records
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn storage_count(&self) -> usize {
        self.storages.len()
    }

    pub fn is_contiguous(&self, id: TensorId) -> bool {
        let t = self.tensor(id);
        let mut expect = 1usize;
        for i in (0..t.shape.rank()).rev() {
            let d = t.shape.dim(i);
            if d == 1 {
                continue;
            }
            if t.strides[i] != expect {
                return false;
            }
            expect *= d;
        }
        true
    }

    // Host view
    pub fn cpu(&self, id: TensorId) -> &[T] {
        assert!(self.is_contiguous(id), "cpu() needs a contiguous tensor");
        let t = self.tensor(id);
        &self.storages[t.storage.0 as usize][t.offset..t.offset + t.shape.numel()]
    }

    // second tensor record -> no data move
    fn push_view(&mut self, base: TensorId, shape: Shape, strides: [usize; MAX_RANK]) -> TensorId {
        let b = *self.tensor(base);
        let id = TensorId(self.tensors.len() as u32);
        self.tensors.push(Tensor {
            shape,
            strides,
            offset: b.offset,
            storage: b.storage,
        });
        id
    }

    // reinterpret dims of the continguous tensor -> no data move
    pub fn reshape(&mut self, id: TensorId, dims: &[usize]) -> TensorId {
        assert!(
            self.is_contiguous(id),
            "reshape needs a contiguous tensor; call contiguous() first"
        );
        let shape = Shape::new(dims);
        assert_eq!(
            shape.numel(),
            self.numel(id),
            "reshape {:?} -> {:?} changes the elements count",
            self.shape(id),
            shape
        );
        self.push_view(id, shape, shape.contiguous_strides())
    }

    pub fn transpose(&mut self, id: TensorId, i: usize, j: usize) -> TensorId {
        let t = *self.tensor(id);
        let rank = t.shape.rank();
        assert!(
            i < rank && j < rank,
            "transpose axes {i},{j} out of range for rank {rank}"
        );
        let mut dims = [0usize; MAX_RANK];
        dims[..rank].copy_from_slice(t.shape.dims());
        dims.swap(i, j);
        let mut strides = t.strides;
        strides.swap(i, j);
        self.push_view(id, Shape::new(&dims[..rank]), strides)
    }

    // moves data (IMPORTANT, review again later)
    pub fn contiguous(&mut self, id: TensorId) -> TensorId {
        if self.is_contiguous(id) {
            return id;
        }
        let t = *self.tensor(id);
        let rank = t.shape.rank();
        let n = t.shape.numel();
        let src = &self.storages[t.storage.0 as usize];
        let mut out = Vec::with_capacity(n);
        let mut idx = [0usize; MAX_RANK];
        for _ in 0..n {
            let mut off = t.offset;
            // offset = dot prod of index and stride
            for (&i, &s) in idx.iter().zip(&t.strides).take(rank) {
                off += i * s;
            }
            out.push(src[off]);
            // odometer style last axis and carry left
            for k in (0..rank).rev() {
                idx[k] += 1;
                if idx[k] < t.shape.dim(k) {
                    break;
                }
                idx[k] = 0;
            }
        }
        self.push(t.shape, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> (Graph<f64>, TensorId) {
        let mut g = Graph::new();
        let a = g.from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        (g, a)
    }

    #[test]
    fn a_fresh_tensor_is_row_major_and_contiguous() {
        let (g, a) = table();
        assert_eq!(g.shape(a).dims(), &[2, 3]);
        assert_eq!(g.strides(a), [3, 1, 0, 0]);
        assert_eq!(g.numel(a), 6);
        assert!(g.is_contiguous(a));
        assert_eq!(g.cpu(a), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn strides_address_the_hand_written_table() {
        let (g, a) = table();
        let s = g.strides(a);
        let at = |i: usize, j: usize| g.cpu(a)[i * s[0] + j * s[1]];
        assert_eq!(at(0, 0), 1.0);
        assert_eq!(at(0, 2), 3.0);
        assert_eq!(at(1, 0), 4.0);
        assert_eq!(at(1, 2), 6.0);
    }

    #[test]
    fn zeros_fills_with_the_default() {
        let mut g: Graph<f64> = Graph::new();
        let z = g.zeros(&[2, 2]);
        assert_eq!(g.cpu(z), &[0.0; 4]);
    }

    #[test]
    fn every_push_allocates_one_storage() {
        let (mut g, _) = table();
        g.zeros(&[4]);
        assert_eq!(g.tensor_count(), 2);
        assert_eq!(g.storage_count(), 2);
    }

    #[test]
    #[should_panic(expected = "needs 6 elements got 5")]
    fn push_rejects_a_length_that_does_not_match_the_shape() {
        let mut g: Graph<f64> = Graph::new();
        g.from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
    }

    #[test]
    fn a_transposed_size_one_axis_is_still_contiguous() {
        let mut g: Graph<f64> = Graph::new();
        let a = g.from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[1, 7]);
        let t = g.transpose(a, 0, 1);
        assert_eq!(g.shape(t).dims(), &[7, 1]);
        assert_eq!(g.strides(t), [1, 7, 0, 0]);
        assert!(g.is_contiguous(t));
        assert_eq!(g.cpu(t), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn transpose_swaps_metadata_and_shares_storage() {
        let (mut g, a) = table();
        let t = g.transpose(a, 0, 1);
        assert_eq!(g.shape(t).dims(), &[3, 2]);
        assert_eq!(g.strides(t), [1, 3, 0, 0]);
        assert!(!g.is_contiguous(t));
        assert_eq!(g.tensor_count(), 2);
        assert_eq!(g.storage_count(), 1);
    }

    #[test]
    fn contiguous_materialises_in_row_major_order() {
        let (mut g, a) = table();
        let t = g.transpose(a, 0, 1);
        let c = g.contiguous(t);
        assert_eq!(g.cpu(c), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(g.strides(c), [2, 1, 0, 0]);
        assert_eq!(g.storage_count(), 2);
    }

    #[test]
    fn reshape_is_free_and_differs_from_transpose() {
        let (mut g, a) = table();
        let r = g.reshape(a, &[3, 2]);
        assert_eq!(g.cpu(r), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(g.storage_count(), 1);

        let t = g.transpose(a, 0, 1);
        let c = g.contiguous(t);
        assert_eq!(g.shape(r).dims(), g.shape(c).dims());
        assert_ne!(g.cpu(r), g.cpu(c));

        assert_eq!(g.tensor_count(), 4);
        assert_eq!(g.storage_count(), 2);
    }

    #[test]
    fn contiguous_is_a_no_op_on_packed_data() {
        let (mut g, a) = table();
        let c = g.contiguous(a);
        assert_eq!(c, a);
        assert_eq!(g.storage_count(), 1);
    }

    #[test]
    fn summing_does_not_depend_on_layout() {
        let mut g: Graph<f64> = Graph::new();
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let a = g.from_slice(&data, &[2, 3, 4]);
        let direct: f64 = g.cpu(a).iter().sum();
        assert_eq!(direct, 276.0);

        let t = g.transpose(a, 0, 2);
        let c = g.contiguous(t);
        let walked: f64 = g.cpu(c).iter().sum();
        assert_eq!(walked, 276.0);
        assert_ne!(g.cpu(c), g.cpu(a));
    }

    #[test]
    #[should_panic(expected = "reshape needs a contiguous tensor")]
    fn reshape_rejects_a_strided_view() {
        let (mut g, a) = table();
        let t = g.transpose(a, 0, 1);
        g.reshape(t, &[6]);
    }
}
