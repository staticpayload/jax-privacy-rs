use jax_privacy::core::GradientTransform;
use jax_privacy::{
    gaussian_privatizer, matrix_factorization_privatizer_dense, JaxKey, NoiseStrategy,
};

#[test]
fn gaussian_privatizer_is_deterministic() {
    let grad = ndarray::array![0.0, 0.0].into_dyn();
    let p1 = gaussian_privatizer(1.0, JaxKey::new(7), NoiseStrategy::Default);
    let p2 = gaussian_privatizer(1.0, JaxKey::new(7), NoiseStrategy::Default);
    let (g1, _) = p1.update(&grad, p1.init(&grad));
    let (g2, _) = p2.update(&grad, p2.init(&grad));
    assert_eq!(g1, g2);
}

#[test]
fn dense_identity_matches_gaussian_first_step() {
    let grad = ndarray::array![0.0, 0.0].into_dyn();
    let dense = ndarray::array![[1.0, 0.0], [0.0, 1.0]];
    let pd =
        matrix_factorization_privatizer_dense(dense, 1.0, JaxKey::new(11), NoiseStrategy::Default);
    let pg = gaussian_privatizer(1.0, JaxKey::new(11), NoiseStrategy::Default);
    let (gd, _) = pd.update(&grad, pd.init(&grad));
    let (gg, _) = pg.update(&grad, pg.init(&grad));
    assert_eq!(gd, gg);
}
