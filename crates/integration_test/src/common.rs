use crate::weights::*;
use heapless::Vec as HVec;
use nalgebra::{SMatrix, SVector};
use reservoir_core::metrics::{mse, rmse, rsquare};
use reservoir_infer::readout::static_readout::StaticReadout as DenseStaticReadout;
use reservoir_infer::reservoir::static_sparse_reservoir::{StaticCsrMatrix, StaticSparseReservoir};
use reservoir_infer::esn::{StaticReadout as StaticReadoutTrait, StaticReservoir as StaticReservoirTrait};

pub trait TestLogger {
    fn log_info(&mut self, msg: &str);
    fn log_step(&mut self, step: usize, input: f32, target: f32, pred: f32);
    fn log_metrics(&mut self, mse: f32, rmse: f32, r2: f32);
}

pub fn run_inference_test<L: TestLogger>(logger: &mut L) {
    logger.log_info("Initializing Static Sparse Reservoir (CSR)...");

    let w_in = StaticCsrMatrix::<f32, W_IN_NROWS, W_IN_NCOLS>::new(
        &W_IN_ROW_PTR,
        &W_IN_COL_IDX,
        &W_IN_VALUES,
    );

    let w_res = StaticCsrMatrix::<f32, W_RES_NROWS, W_RES_NCOLS>::new(
        &W_RES_ROW_PTR,
        &W_RES_COL_IDX,
        &W_RES_VALUES,
    );

    let mut reservoir =
        StaticSparseReservoir::<f32, INPUT_DIM, RESERVOIR_SIZE, EXTENDED_SIZE>::create(
            w_in,
            w_res,
            LEAKING_RATE,
        );

    reservoir.res_state =
        SVector::<f32, RESERVOIR_SIZE>::from_column_slice(&INITIAL_STATE_DATA);

    let w_out =
        SMatrix::<f32, OUTPUT_DIM, EXTENDED_SIZE>::from_column_slice(&W_OUT_DATA);
    let readout = DenseStaticReadout::<f32, EXTENDED_SIZE, OUTPUT_DIM>::create(w_out);

    logger.log_info("Starting Inference Loop...");

    let mut predictions: HVec<f32, TEST_LEN> = HVec::new();

    for i in 0..TEST_LEN {
        let input_val = TEST_INPUTS[i];
        let target_val = TEST_TARGETS[i];

        let input = SVector::<f32, INPUT_DIM>::new(input_val);

        let state = reservoir.step_static(&input);
        let output = readout.predict_static(state);
        let pred_val = output[0];

        let _ = predictions.push(pred_val);

        logger.log_step(i, input_val, target_val, pred_val);
    }

    let mse_val = mse(&TEST_TARGETS, &predictions);
    let rmse_val = rmse(&TEST_TARGETS, &predictions);
    let r2_val = rsquare(&TEST_TARGETS, &predictions);

    logger.log_metrics(mse_val, rmse_val, r2_val);
}
