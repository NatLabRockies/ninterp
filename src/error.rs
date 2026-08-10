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
    /// `(point index, dimension)` for every coordinate that fell outside the grid. One
    /// point out of bounds in two dimensions yields two entries. Index the `points` and
    /// `grid` you supplied to recover the offending coordinate and the bounds it missed.
    #[error("{}", fmt_out_of_bounds(.0))]
    OutOfBounds(Vec<(usize, usize)>),
    /// `(point index, actual length)` for every offending point.
    #[error("{}", fmt_point_length(*expected, failures))]
    PointLength {
        expected: usize,
        failures: Vec<(usize, usize)>,
    },
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

/// Renders `point`, or `point[i]` when there is anything to disambiguate. Every
/// single-point call reports index 0 and only index 0, so the index is noise there.
fn point_at(index: usize, show_index: bool) -> String {
    if show_index {
        format!("point[{index}]")
    } else {
        String::from("point")
    }
}

/// Whether point indices are worth printing: only once some failure is at a nonzero
/// index, which never happens for a single-point call.
fn show_index(failures: &[(usize, usize)]) -> bool {
    failures.iter().any(|(index, _)| *index != 0)
}

/// A lone failure reads as one sentence; several are listed under a summary line.
/// Shared with [`fmt_point_length`] so both aggregating variants render alike.
fn fmt_out_of_bounds(failures: &[(usize, usize)]) -> String {
    let show = show_index(failures);
    match failures {
        [(index, dim)] => format!(
            "{} is out of bounds in dim {dim}, with `Extrapolate::Error` set",
            point_at(*index, show)
        ),
        many => {
            // One point can be out of bounds in several dimensions, so the number of
            // failures doesn't decide the plural here; the number of distinct points does.
            let subject = if many.iter().any(|(index, _)| *index != many[0].0) {
                "points"
            } else {
                "point"
            };
            let mut s = format!("{subject} out of bounds with `Extrapolate::Error` set:");
            for (index, dim) in many {
                s.push_str(&format!("\n    {} in dim {dim}", point_at(*index, show)));
            }
            s
        }
    }
}

/// See [`fmt_out_of_bounds`].
fn fmt_point_length(expected: usize, failures: &[(usize, usize)]) -> String {
    let show = show_index(failures);
    match failures {
        [(index, found)] => format!(
            "{} has length {found}, expected {expected} for {expected}-D interpolation",
            point_at(*index, show)
        ),
        many => {
            // Unlike out-of-bounds, each failure here is a distinct point, so reaching
            // this arm always means more than one.
            let mut s = format!("points have the wrong length for {expected}-D interpolation:");
            for (index, found) in many {
                s.push_str(&format!(
                    "\n    {} has length {found}",
                    point_at(*index, show)
                ));
            }
            s
        }
    }
}
