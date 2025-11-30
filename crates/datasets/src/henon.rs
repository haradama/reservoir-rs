extern crate alloc;
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use num_traits::Float;

#[derive(Clone, Debug)]
pub struct HenonParams {
    pub a: f64,
    pub b: f64,
    pub x0: f64,
    pub y0: f64,
    pub steps: usize,
}

impl Default for HenonParams {
    fn default() -> Self {
        Self {
            a: 1.4,
            b: 0.3,
            x0: 0.0,
            y0: 0.0,
            steps: 2000,
        }
    }
}

pub struct HenonMap {
    params: HenonParams,
    current_x: f64,
    current_y: f64,
}

impl HenonMap {
    pub fn new(params: HenonParams) -> Self {
        Self {
            current_x: params.x0,
            current_y: params.y0,
            params,
        }
    }

    pub fn step(&mut self) -> f64 {
        let x = self.current_x;
        let y = self.current_y;

        let next_x = 1.0 - self.params.a * x.powi(2) + y;
        let next_y = self.params.b * x;

        self.current_x = next_x;
        self.current_y = next_y;

        next_x
    }

    pub fn generate(&mut self) -> Vec<f64> {
        let mut data = Vec::with_capacity(self.params.steps);
        
        for _ in 0..100 {
            self.step();
        }

        for _ in 0..self.params.steps {
            data.push(self.step());
        }

        data
    }
}