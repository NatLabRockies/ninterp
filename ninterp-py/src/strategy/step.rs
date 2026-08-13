use ninterp::strategy;
use pyo3::prelude::*;

#[pyclass(eq, eq_int, from_py_object)]
#[derive(PartialEq, Clone, Copy)]
pub enum StepDirection {
    Lower,
    Upper,
}

impl From<StepDirection> for strategy::step::StepDirection {
    fn from(d: StepDirection) -> Self {
        match d {
            StepDirection::Lower => strategy::step::StepDirection::Lower,
            StepDirection::Upper => strategy::step::StepDirection::Upper,
        }
    }
}

#[pyclass]
pub struct Step(pub strategy::Step);
#[pymethods]
impl Step {
    #[staticmethod]
    fn lower() -> Self {
        Step(strategy::Step::lower())
    }
    #[staticmethod]
    fn upper() -> Self {
        Step(strategy::Step::upper())
    }
    #[new]
    fn new(directions: Vec<StepDirection>) -> Self {
        Step(strategy::Step::new(
            directions.into_iter().map(Into::into).collect(),
        ))
    }
}
