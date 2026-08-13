use pyo3::prelude::*;

mod interpolator;
mod strategy;

/// A Python module implemented in Rust.
#[pymodule(name = "ninterp")]
mod ninterp_py {
    #[pymodule_export]
    use super::interpolator::Interpolator;

    #[pymodule_export]
    use super::strategy::{CubicC2, Linear, Nearest, Step, StepDirection};
}
