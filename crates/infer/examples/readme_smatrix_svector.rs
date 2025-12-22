use nalgebra::{SMatrix, SVector};
use reservoir_core::{mse, rmse, rsquare};
use reservoir_infer::esn::StaticESN;
use reservoir_infer::readout::static_readout::StaticReadout as DenseStaticReadout;
use reservoir_infer::reservoir::static_reservoir::StaticReservoir as DenseStaticReservoir;

type S = f32;

const IN: usize = 1;
const OUT: usize = 1;
const N: usize = 8;
const EXT: usize = 1 + IN + N; // = 10

const LEAKING_RATE: S = 0.3;

// ---- Weights (pre-trained) ----
const W_IN_DATA: [S; N * IN] = [
    -0.18329513,
    0.013278663,
    -0.12563086,
    0.021362603,
    0.18421328,
    0.06823254,
    0.24504232,
    -0.047049165,
];

const W_DATA: [S; N * N] = [
    0.46901882,
    0.12934124,
    0.36650026,
    0.1368283,
    -0.23916698,
    -0.4070009,
    0.019198537,
    0.43125474,
    -0.46565723,
    -0.36872113,
    -0.1093508,
    -0.4874071,
    0.3457904,
    -0.1489892,
    -0.09828675,
    0.42196798,
    0.11742401,
    -0.020595908,
    0.21007144,
    -0.3396176,
    -0.0363822,
    -0.19802523,
    -0.33582056,
    -0.077880025,
    -0.08504319,
    -0.49674797,
    -0.35912,
    0.019248724,
    -0.0022733212,
    -0.052440524,
    0.1546005,
    0.36930323,
    -0.15221453,
    0.32195556,
    0.3021648,
    -0.037684083,
    0.478917,
    -0.3601141,
    0.36539114,
    0.14560473,
    0.23742437,
    0.43214512,
    0.023039103,
    -0.44972634,
    0.0059776306,
    0.29438484,
    -0.07601333,
    0.14889145,
    -0.31422865,
    0.09885669,
    0.31836128,
    0.16216016,
    -0.08578122,
    0.082585216,
    -0.30444062,
    0.30872536,
    0.3492515,
    0.006149292,
    -0.28129447,
    0.1464504,
    -0.4415053,
    -0.29646504,
    0.38848472,
    -0.27922785,
];

const W_OUT_DATA: [S; OUT * EXT] = [
    -0.09186051,
    1.5557092,
    0.5148108,
    -3.0077815,
    1.6692747,
    -1.1821239,
    -1.7794359,
    -9.195823,
    2.237872,
    -6.3940053,
];

fn main() {
    // --- build static ESN (inference only; no training) ---
    let w_in: SMatrix<S, N, IN> = SMatrix::from_row_slice(&W_IN_DATA);
    let w: SMatrix<S, N, N> = SMatrix::from_row_slice(&W_DATA);

    let reservoir: DenseStaticReservoir<S, IN, N, EXT> =
        DenseStaticReservoir::create(w_in, w, LEAKING_RATE);

    let w_out: SMatrix<S, OUT, EXT> = SMatrix::from_row_slice(&W_OUT_DATA);
    let readout: DenseStaticReadout<S, EXT, OUT> = DenseStaticReadout::create(w_out);

    let mut esn: StaticESN<S, _, _> = StaticESN::new(reservoir, readout);

    let steps = 2000usize;
    let series: Vec<S> = (0..=steps).map(|t| ((t as f32) * 0.02).sin()).collect();

    let inputs: Vec<S> = series[..steps].to_vec();
    let targets: Vec<S> = series[1..].to_vec();

    let train_len = 1500usize;

    for &u in &inputs[..train_len] {
        let x: SVector<S, IN> = SVector::<S, IN>::from_row_slice(&[u]);
        let _ = esn.predict::<IN, OUT, EXT>(&x);
    }

    // --- evaluate  ---
    let mut preds: Vec<S> = Vec::with_capacity(steps - train_len);
    let mut y_true: Vec<S> = Vec::with_capacity(steps - train_len);

    for i in train_len..steps {
        let u = inputs[i];
        let t = targets[i];

        let x: SVector<S, IN> = SVector::<S, IN>::from_row_slice(&[u]);
        let y: SVector<S, OUT> = esn.predict::<IN, OUT, EXT>(&x);

        preds.push(y[0]);
        y_true.push(t);
    }

    println!("--------------------------------------------------");
    println!("MSE  : {:.6}", mse(&y_true, &preds));
    println!("RMSE : {:.6}", rmse(&y_true, &preds));
    println!("R²   : {:.6}", rsquare(&y_true, &preds));
    println!("--------------------------------------------------");
}
