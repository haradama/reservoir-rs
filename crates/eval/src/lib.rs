pub fn mse(y_true: &[f32], y_pred: &[f32]) -> f32 {
    y_true
        .iter()
        .zip(y_pred)
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f32>()
        / y_true.len() as f32
}

pub fn nrmse(y_true: &[f32], y_pred: &[f32]) -> f32 {
    let rmse = mse(y_true, y_pred).sqrt();
    let range = y_true.iter().fold(f32::MIN, |a, &b| a.max(b))
        - y_true.iter().fold(f32::MAX, |a, &b| a.min(b));
    rmse / range
}

pub fn rmse(y_true: &[f32], y_pred: &[f32]) -> f32 {
    mse(y_true, y_pred).sqrt()
}

pub fn rsquare(y_true: &[f32], y_pred: &[f32]) -> f32 {
    let mean = y_true.iter().copied().sum::<f32>() / y_true.len() as f32;
    let ss_tot: f32 = y_true.iter().map(|v| (v - mean).powi(2)).sum();
    let ss_res: f32 = y_true.iter().zip(y_pred).map(|(t, p)| (t - p).powi(2)).sum();
    1.0 - ss_res / ss_tot
}
