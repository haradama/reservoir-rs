use ndarray::{Array2, s};

pub struct SlidingWindow {
    pub window: usize,
    pub horizon: usize,
}

impl SlidingWindow {
    pub fn split(&self, series: &Array2<f32>) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let rows = series.nrows();
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        for i in 0..rows - self.window - self.horizon {
            let x = series.slice(s![i..i + self.window, 0]).to_vec();
            let y_val = series[[i + self.window + self.horizon, 0]];
            inputs.push(x);
            targets.push(vec![y_val]);
        }
        (inputs, targets)
    }
}
