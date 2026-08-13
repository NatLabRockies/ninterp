use pyo3::prelude::*;

#[pyclass]
pub struct Nearest;
#[pymethods]
impl Nearest {
    #[new]
    fn new() -> Self {
        Nearest
    }
}
