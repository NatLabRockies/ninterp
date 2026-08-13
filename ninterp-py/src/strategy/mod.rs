pub mod cubic;
pub mod linear;
pub mod linear_uniform;
pub mod nearest;
pub mod step;

pub use nearest::Nearest;

pub use step::{Step, StepDirection};

pub use linear::Linear;

pub use linear_uniform::LinearUniform;

pub use cubic::CubicC2;

use ninterp::strategy;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

pub struct Strategy1D(pub ninterp::strategy::enums::Strategy1DEnum<f64>);

impl<'a, 'py> FromPyObject<'a, 'py> for Strategy1D {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if obj.cast::<Nearest>().is_ok() {
            return Ok(Strategy1D(strategy::Nearest.into()));
        }
        if let Ok(v) = obj.cast::<Step>() {
            return Ok(Strategy1D(v.borrow().0.clone().into()));
        }
        if obj.cast::<Linear>().is_ok() {
            return Ok(Strategy1D(strategy::Linear.into()));
        }
        if obj.cast::<LinearUniform>().is_ok() {
            return Ok(Strategy1D(strategy::LinearUniform.into()));
        }
        if let Ok(v) = obj.cast::<CubicC2>() {
            return Ok(Strategy1D(v.borrow().0.clone().into()));
        }
        Err(PyTypeError::new_err(
            "expected a strategy instance (Linear, Nearest, Step, CubicC2, ...)",
        ))
    }
}
