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
}
