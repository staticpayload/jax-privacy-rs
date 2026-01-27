use jax_privacy::core::{PyTree, Tensor};
use jax_privacy_derive::PyTree as DerivePyTree;

fn tensor(v: &[f64]) -> Tensor {
    ndarray::Array1::from_vec(v.to_vec()).into_dyn()
}

#[derive(Clone, Debug, PartialEq, DerivePyTree)]
struct Model {
    w: Tensor,
    b: Tensor,
}

#[derive(Clone, Debug, PartialEq, DerivePyTree)]
struct Wrapper {
    inner: Vec<Tensor>,
}

#[derive(Clone, Debug, PartialEq, DerivePyTree)]
struct Tuple(Tensor, Tensor);

#[derive(Clone, Debug, PartialEq, DerivePyTree)]
struct Generic<T> {
    inner: T,
}

#[test]
fn derive_pytree_round_trip_named_struct() {
    let m = Model {
        w: tensor(&[1.0, 2.0]),
        b: tensor(&[0.5, -0.5]),
    };

    let (leaves, spec) = m.flatten();
    assert_eq!(leaves.len(), 2);
    let m2 = Model::unflatten(&spec, leaves);
    assert_eq!(m, m2);
}

#[test]
fn derive_pytree_round_trip_nested_vec() {
    let w = Wrapper {
        inner: vec![tensor(&[1.0]), tensor(&[2.0, 3.0])],
    };

    let (leaves, spec) = w.flatten();
    assert_eq!(leaves.len(), 2);
    let w2 = Wrapper::unflatten(&spec, leaves);
    assert_eq!(w, w2);
}

#[test]
fn derive_pytree_round_trip_tuple_struct() {
    let t = Tuple(tensor(&[1.0]), tensor(&[2.0, 3.0]));
    let (leaves, spec) = t.flatten();
    assert_eq!(leaves.len(), 2);
    let t2 = Tuple::unflatten(&spec, leaves);
    assert_eq!(t, t2);
}

#[test]
fn derive_pytree_round_trip_generic_struct() {
    let g = Generic {
        inner: tensor(&[4.0, 5.0]),
    };
    let (leaves, spec) = g.flatten();
    assert_eq!(leaves.len(), 1);
    let g2 = Generic::<Tensor>::unflatten(&spec, leaves);
    assert_eq!(g, g2);
}
