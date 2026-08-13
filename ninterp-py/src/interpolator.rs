use crate::strategy::Strategy1D;
use ninterp::interpolator::Interpolator as _;
use ninterp::prelude::*;
use numpy::{AllowTypeChange, IntoPyArray, PyArrayLike1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

#[pyclass]
pub struct Interpolator(InterpolatorEnum<f64>);

#[pymethods]
impl Interpolator {
    #[staticmethod]
    #[pyo3(signature = (
        x: "numpy.typing.ArrayLike",
        f_x: "numpy.typing.ArrayLike",
        strategy: "Linear | Nearest | Step | CubicC2",
    ))]
    fn new_1d<'py>(
        x: PyArrayLike1<'py, f64, AllowTypeChange>,
        f_x: PyArrayLike1<'py, f64, AllowTypeChange>,
        strategy: Strategy1D,
    ) -> PyResult<Self> {
        InterpolatorEnum::new_1d(
            x.as_array().to_owned(),
            f_x.as_array().to_owned(),
            strategy.0,
            Extrapolate::Error,
        )
        .map(Interpolator)
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (strategy: "Linear | Nearest | Step | CubicC2"))]
    fn set_strategy(&mut self, strategy: Strategy1D) -> PyResult<()> {
        match &mut self.0 {
            InterpolatorEnumBase::Interp1D(interp) => interp
                .set_strategy(strategy.0)
                .map_err(|e| PyValueError::new_err(e.to_string())),
            _ => Err(PyValueError::new_err(
                "strategy dimensionality does not match interpolator",
            )),
        }
    }

    #[pyo3(signature = (x: "float | numpy.typing.ArrayLike") -> "float | numpy.ndarray")]
    fn interpolate<'py>(&self, py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(scalar) = x.extract::<f64>() {
            let result = self
                .0
                .interpolate(&[scalar])
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(result.into_pyobject(py)?.into_any().unbind());
        }

        let arr: PyArrayLike1<'py, f64, AllowTypeChange> = x.extract()?;
        let points: Vec<[f64; 1]> = arr.as_array().iter().map(|&v| [v]).collect();
        let point_refs: Vec<&[f64]> = points.iter().map(|p| p.as_slice()).collect();

        let mut out = vec![0.0; point_refs.len()];
        self.0
            .batch_interpolate_into(&point_refs, &mut out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(out.into_pyarray(py).into_any().unbind())
    }
}
