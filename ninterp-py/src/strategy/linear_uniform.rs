use pyo3::prelude::*;

#[pyclass]
pub struct LinearUniform;
#[pymethods]
impl LinearUniform {
    #[new]
    fn new() -> Self {
        LinearUniform
    }
}
