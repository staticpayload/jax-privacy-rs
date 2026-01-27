//! Tensor types and low-level operations.

use ndarray::ArrayD;

/// Scalar type (f64 by default, f32 with feature flag).
#[cfg(not(feature = "f32"))]
pub type Scalar = f64;

/// Scalar type (f64 by default, f32 with feature flag).
#[cfg(feature = "f32")]
pub type Scalar = f32;

/// Dynamic-dimensional tensor.
pub type Tensor = ArrayD<Scalar>;

/// Compute L2 norm of a tensor, handling NaN/inf.
pub fn l2_norm(t: &Tensor) -> f64 {
    let mut sum_sq: f64 = 0.0;
    for &v in t.iter() {
        let v = v as f64;
        if v.is_nan() {
            return f64::NAN;
        }
        if !v.is_finite() {
            return f64::INFINITY;
        }
        sum_sq += v * v;
    }
    sum_sq.sqrt()
}

/// Replace NaN and infinite values in-place.
pub fn sanitize(t: &mut Tensor, nan_val: Scalar, inf_val: Scalar) {
    t.mapv_inplace(|x| {
        if x.is_nan() {
            nan_val
        } else if x.is_infinite() {
            if x.is_sign_negative() {
                -inf_val
            } else {
                inf_val
            }
        } else {
            x
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_l2_norm() {
        let t = array![3.0, 4.0].into_dyn();
        assert!((l2_norm(&t) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_l2_norm_nan() {
        let t = array![1.0, f64::NAN, 2.0].into_dyn();
        assert!(l2_norm(&t).is_nan());
    }

    #[test]
    fn test_sanitize() {
        let mut t = array![1.0, f64::NAN, f64::INFINITY].into_dyn();
        sanitize(&mut t, 0.0, 999.0);
        assert_eq!(t[[0]], 1.0);
        assert_eq!(t[[1]], 0.0);
        assert_eq!(t[[2]], 999.0);
    }
}
