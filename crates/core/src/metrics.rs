use core::iter::Sum;
use num_traits::Float;

pub fn mse<T>(y_true: &[T], y_pred: &[T]) -> T
where
    T: Float + Sum<T>,
{
    y_true
        .iter()
        .zip(y_pred)
        .map(|(&t, &p)| (t - p).powi(2))
        .sum::<T>()
        / T::from(y_true.len()).unwrap()
}

pub fn rmse<T>(y_true: &[T], y_pred: &[T]) -> T
where
    T: Float + Sum<T>,
{
    mse(y_true, y_pred).sqrt()
}

pub fn nrmse<T>(y_true: &[T], y_pred: &[T]) -> T
where
    T: Float + Sum<T>,
{
    let rmse = rmse(y_true, y_pred);
    let min = y_true.iter().copied().fold(T::max_value(), T::min);
    let max = y_true.iter().copied().fold(T::min_value(), T::max);
    let range = max - min;
    rmse / range
}

pub fn rsquare<T>(y_true: &[T], y_pred: &[T]) -> T
where
    T: Float + Sum<T>,
{
    let mean = y_true.iter().copied().sum::<T>() / T::from(y_true.len()).unwrap();
    let ss_tot: T = y_true.iter().map(|&v| (v - mean).powi(2)).sum();
    let ss_res: T = y_true
        .iter()
        .zip(y_pred)
        .map(|(&t, &p)| (t - p).powi(2))
        .sum();
    T::one() - ss_res / ss_tot
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    #[test]
    fn test_mse() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.1, 2.1, 2.9, 4.1];
        let result = mse(&y_true, &y_pred);
        assert!((result - 0.01).abs() < EPS);
    }

    #[test]
    fn test_rmse() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.1, 2.1, 2.9, 4.1];
        let result = rmse(&y_true, &y_pred);
        assert!((result - 0.1).abs() < EPS);
    }

    #[test]
    fn test_nrmse() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.1, 2.1, 2.9, 4.1];
        let result = nrmse(&y_true, &y_pred);
        assert!((result - 0.03333333333333333).abs() < EPS);
    }

    #[test]
    fn test_rsquare() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred_good = vec![1.1, 2.1, 2.9, 4.1];
        let y_pred_bad = vec![3.0, 3.0, 3.0, 3.0];

        let r2_good = rsquare(&y_true, &y_pred_good);
        assert!((r2_good - 0.992).abs() < EPS);

        let r2_bad = rsquare(&y_true, &y_pred_bad);
        assert!((r2_bad - (-0.2)).abs() < EPS);
    }

    #[test]
    fn test_metrics_perfect_prediction_f64() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.0, 2.0, 3.0, 4.0];

        assert!((mse(&y_true, &y_pred) - 0.0).abs() < EPS);
        assert!((rmse(&y_true, &y_pred) - 0.0).abs() < EPS);
        assert!((rsquare(&y_true, &y_pred) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_metrics_generic_f32() {
        let y_true: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred: Vec<f32> = vec![1.1, 2.1, 2.9, 4.1];

        let m = mse(&y_true, &y_pred);
        let r = rmse(&y_true, &y_pred);
        let n = nrmse(&y_true, &y_pred);
        let r2 = rsquare(&y_true, &y_pred);

        assert!((m - 0.01).abs() < 1e-5);
        assert!((r - 0.1).abs() < 1e-5);
        assert!((n - (0.1 / 3.0)).abs() < 1e-5);
        assert!((r2 - 0.992).abs() < 1e-3);
    }
}
