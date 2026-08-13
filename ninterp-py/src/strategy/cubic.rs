use ninterp::strategy;
use pyo3::prelude::*;

#[pyclass]
pub struct CubicC2(pub strategy::CubicC2<f64>);
#[pymethods]
impl CubicC2 {
    #[staticmethod]
    fn not_a_knot() -> Self {
        CubicC2(strategy::CubicC2::not_a_knot())
    }
    #[staticmethod]
    fn natural() -> Self {
        CubicC2(strategy::CubicC2::natural())
    }
    #[staticmethod]
    fn clamped(lower: f64, upper: f64) -> Self {
        CubicC2(strategy::CubicC2::clamped(lower, upper))
    }
    #[staticmethod]
    fn periodic() -> Self {
        CubicC2(strategy::CubicC2::periodic())
    }
}
