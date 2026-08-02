//! Benchmarks for 0/1/2/3/N-dimensional linear interpolation
//! Run these with `cargo bench`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use ndarray::prelude::*;
use ninterp::prelude::*;

use ndarray_rand::rand::{prelude::StdRng, Rng, SeedableRng};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

const RANDOM_SEED: u64 = 1234567890;

#[allow(non_snake_case)]
/// 0-D interpolation (hardcoded)
fn benchmark_0D() {
    let interp_0d = Interp0D(0.5);
    interp_0d.interpolate(&[]).unwrap();
}

#[allow(non_snake_case)]
/// 0-D interpolation (multilinear interpolator)
fn benchmark_0D_multi() {
    let interp_0d_multi = InterpND::new(
        vec![array![]],
        array![0.5].into_dyn(),
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    interp_0d_multi.interpolate(black_box(&[])).unwrap();
}

#[allow(non_snake_case)]
/// 1-D interpolation (hardcoded)
fn benchmark_1D() {
    let mut rng = StdRng::seed_from_u64(RANDOM_SEED);
    let grid_data: Array1<f64> = (0..100).map(|x| x as f64).collect();
    // Generate interpolator data (same as N-D benchmark)
    let values_data = Array1::random_using(100, Uniform::new(0., 1.).unwrap(), &mut rng);
    // Create a 1-D interpolator with 100 data points
    let interp_1d =
        Interp1D::new(grid_data, values_data, strategy::Linear, Extrapolate::Error).unwrap();
    // Sample 1,000 points
    let points: Vec<f64> = (0..1_000).map(|_| rng.random::<f64>() * 99.).collect();
    for point in points {
        interp_1d.interpolate(black_box(&[point])).unwrap();
    }
}

#[allow(non_snake_case)]
/// 1-D interpolation (multilinear interpolator)
fn benchmark_1D_multi() {
    let mut rng = StdRng::seed_from_u64(RANDOM_SEED);
    // Generate interpolator data (same as hardcoded benchmark)
    let grid_data: Array1<f64> = (0..100).map(|x| x as f64).collect();
    let values_data = Array1::random_using(100, Uniform::new(0., 1.).unwrap(), &mut rng).into_dyn();
    // Create an N-D interpolator with 100x100 data (10,000 points)
    let interp_1d_multi = InterpND::new(
        vec![grid_data],
        values_data,
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    // Sample 1,000 points
    let points: Vec<f64> = (0..1_000).map(|_| rng.random::<f64>() * 99.).collect();
    for point in points {
        interp_1d_multi.interpolate(black_box(&[point])).unwrap();
    }
}

#[allow(non_snake_case)]
/// 2-D interpolation (hardcoded)
fn benchmark_2D() {
    let mut rng = StdRng::seed_from_u64(RANDOM_SEED);
    let grid_data: Array1<f64> = (0..100).map(|x| x as f64).collect();
    let values_data = Array2::random_using((100, 100), Uniform::new(0., 1.).unwrap(), &mut rng);
    // Create a 2-D interpolator with 100x100 data (10,000 points)
    let interp_2d = Interp2D::new(
        grid_data.view(),
        grid_data.view(),
        values_data.view(),
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    // Sample 1,000 points
    let points: Vec<Vec<f64>> = (0..1_000)
        .map(|_| vec![rng.random::<f64>() * 99., rng.random::<f64>() * 99.])
        .collect();
    for point in points {
        interp_2d.interpolate(black_box(&point)).unwrap();
    }
}

#[allow(non_snake_case)]
/// 2-D interpolation (multilinear interpolator)
fn benchmark_2D_multi() {
    let mut rng = StdRng::seed_from_u64(RANDOM_SEED);
    // Generate interpolator data (same as hardcoded benchmark)
    let grid_data: Array1<f64> = (0..100).map(|x| x as f64).collect();
    let values_data = Array2::random_using((100, 100), Uniform::new(0., 1.).unwrap(), &mut rng).into_dyn();
    // Create an N-D interpolator with 100x100 data (10,000 points)
    let interp_2d_multi = InterpND::new(
        vec![grid_data.view(), grid_data.view()],
        values_data.view(),
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    // Sample 1,000 points
    let points: Vec<Vec<f64>> = (0..1_000)
        .map(|_| vec![rng.random::<f64>() * 99., rng.random::<f64>() * 99.])
        .collect();
    for point in points {
        interp_2d_multi.interpolate(black_box(&point)).unwrap();
    }
}

#[allow(non_snake_case)]
/// 3-D interpolation (hardcoded)
fn benchmark_3D() {
    let mut rng = StdRng::seed_from_u64(RANDOM_SEED);
    let grid_data: Array1<f64> = (0..100).map(|x| x as f64).collect();
    // Generate interpolator data (same as N-D benchmark) and arrange into `Vec<Vec<Vec<f64>>>`
    let values_data = Array3::random_using((100, 100, 100), Uniform::new(0., 1.).unwrap(), &mut rng);
    // Create a 3-D interpolator with 100x100x100 data (1,000,000 points)
    let interp_3d = Interp3D::new(
        grid_data.view(),
        grid_data.view(),
        grid_data.view(),
        values_data.view(),
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    // Sample 1,000 points
    let points: Vec<Vec<f64>> = (0..1_000)
        .map(|_| {
            vec![
                rng.random::<f64>() * 99.,
                rng.random::<f64>() * 99.,
                rng.random::<f64>() * 99.,
            ]
        })
        .collect();
    for point in points {
        interp_3d.interpolate(black_box(&point)).unwrap();
    }
}

#[allow(non_snake_case)]
/// 3-D interpolation (multilinear interpolator)
fn benchmark_3D_multi() {
    let mut rng = StdRng::seed_from_u64(RANDOM_SEED);
    // Generate interpolator data (same as hardcoded benchmark)
    let grid_data: Array1<f64> = (0..100).map(|x| x as f64).collect();
    let values_data =
        Array3::random_using((100, 100, 100), Uniform::new(0., 1.).unwrap(), &mut rng).into_dyn();
    // Create an N-D interpolator with 100x100x100 data (1,000,000 points)
    let interp_3d_multi = InterpND::new(
        vec![grid_data.view(), grid_data.view(), grid_data.view()],
        values_data.view(),
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    // Sample 1,000 points
    let points: Vec<Vec<f64>> = (0..1_000)
        .map(|_| {
            vec![
                rng.random::<f64>() * 99.,
                rng.random::<f64>() * 99.,
                rng.random::<f64>() * 99.,
            ]
        })
        .collect();
    for point in points {
        interp_3d_multi.interpolate(black_box(&point)).unwrap();
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("0-D hardcoded", |b| b.iter(benchmark_0D));
    c.bench_function("0-D multilinear", |b| b.iter(benchmark_0D_multi));
    c.bench_function("1-D hardcoded", |b| b.iter(benchmark_1D));
    c.bench_function("1-D multilinear", |b| b.iter(benchmark_1D_multi));
    c.bench_function("2-D hardcoded", |b| b.iter(benchmark_2D));
    c.bench_function("2-D multilinear", |b| b.iter(benchmark_2D_multi));
    c.bench_function("3-D hardcoded", |b| b.iter(benchmark_3D));
    c.bench_function("3-D multilinear", |b| b.iter(benchmark_3D_multi));
}

criterion_group!(benchmarks, criterion_benchmark);
criterion_main!(benchmarks);
