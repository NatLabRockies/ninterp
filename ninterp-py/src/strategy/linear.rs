use pyo3::prelude::*;

#[pyclass]
pub struct Linear;
#[pymethods]
impl Linear {
    #[new]
    fn new() -> Self {
        Linear
    }
}
