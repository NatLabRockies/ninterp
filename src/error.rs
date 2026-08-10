//! Crate error types

use std::fmt;
use thiserror::Error;

/// Error in interpolator data validation
#[allow(missing_docs)]
#[derive(Error, Clone, PartialEq)]
#[non_exhaustive]
pub enum ValidateError {
    #[error("selected `Extrapolate` variant ({0}) is unimplemented/inapplicable for interpolator")]
    InvalidExtrapolate(String),
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
    #[error("attempted to interpolate at point(s) beyond grid data: {0}")]
    ExtrapolateError(String),
    #[error("supplied point slice should have length {0} for {0}-D interpolation")]
    PointLength(usize),
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
