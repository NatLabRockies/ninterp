//! Crate error types

use std::fmt;
use thiserror::Error;

/// Error in interpolator data validation
#[allow(missing_docs)]
#[derive(Error, Clone, PartialEq)]
#[non_exhaustive]
pub enum ValidateError {
    #[error("`Extrapolate::Enable` is not supported by this strategy")]
    ExtrapolateUnsupported,
    #[error("at least 2 grid points are required per dimension: dim {0}")]
    InsufficientGridPoints(usize),
    #[error("supplied coordinates must be monotonically increasing: dim {0}")]
    NonMonotonic(usize),
    #[error("supplied grid and values are not compatible shapes: dim {0}")]
    IncompatibleShapes(usize),
    #[error("{0}")]
    Other(String),
}

impl fmt::Debug for ValidateError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Error in interpolation call
#[allow(missing_docs)]
#[derive(Error, Clone, PartialEq)]
#[non_exhaustive]
pub enum InterpolateError {
    #[error("point(s) out of bounds with `Extrapolate::Error` set: {0}")]
    OutOfBounds(String),
    #[error("point has length {found}, expected {expected} for {expected}-D interpolation")]
    PointLength { expected: usize, found: usize },
    #[error("output slice has length {found}, expected {expected}")]
    OutputLength { expected: usize, found: usize },
    #[error("{0}")]
    Other(String),
}

impl fmt::Debug for InterpolateError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
