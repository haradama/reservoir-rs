use num_traits::Float;
use std::iter::Sum;

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
    let ss_res: T = y_true.iter().zip(y_pred).map(|(&t, &p)| (t - p).powi(2)).sum();
    T::one() - ss_res / ss_tot
}
