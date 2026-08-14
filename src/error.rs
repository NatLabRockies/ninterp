//! Crate error types

use std::fmt;
use thiserror::Error;

use crate::strategy::Transform;

/// Error in interpolator data validation
#[allow(missing_docs)]
#[derive(Error, Clone, PartialEq)]
#[non_exhaustive]
pub enum ValidateError {
    #[error("`Extrapolate::Enable` is not supported by this strategy")]
    ExtrapolateUnsupported,
    #[error("at least 2 grid points are required per dimension: dim {0}")]
    InsufficientGridPoints(usize),
    #[error("supplied coordinates must be strictly increasing: dim {0}")]
    NotStrictlyIncreasing(usize),
    /// Raised by [`crate::strategy::utils::validate_uniform_grid`] and
    /// [`crate::strategy::utils::validate_uniform_grid_epsilon`], so any strategy
    /// requiring uniform spacing reports it the same way, not just `LinearUniform`.
    ///
    /// `index` is the first coordinate whose following interval, `grid[index + 1] -
    /// grid[index]`, differs from the grid's first interval, in either direction.
    #[error("grid[{dim}] is not uniformly spaced (spacing changes at index {index})")]
    NonUniform { dim: usize, index: usize },
    #[error("supplied grid and values are not compatible shapes: dim {0}")]
    IncompatibleShapes(usize),
    /// Number of grid axes doesn't match the dimensionality of `values`. Only reachable
    /// for `InterpDataND`, whose axis count isn't fixed by the type. Distinct from
    /// [`ValidateError::IncompatibleShapes`], which compares extents within one axis.
    #[error("grid has {found} axes, expected {expected} to match the values")]
    GridAxisCount { expected: usize, found: usize },
    /// Raised by [`crate::strategy::broadcast::Broadcastable::validate_len`], so any
    /// strategy built on [`crate::strategy::broadcast::Broadcastable`] reports a
    /// mismatched per-axis count
    /// the same way. `label` names the strategy (e.g. `"Step"`); `noun` names what's being
    /// counted (e.g. `"directions"`).
    #[error("{label} has {found} {noun} but interpolator is {ndim}-D (expected {ndim})")]
    PerAxisLen {
        label: &'static str,
        noun: &'static str,
        ndim: usize,
        found: usize,
    },
    /// A grid coordinate ([`crate::strategy::GridTransform`]) or data value
    /// ([`crate::strategy::ValuesTransform`]) lies outside `transform`'s valid
    /// domain (see [`Transform::in_domain`]). `label` is `"GridTransform"` or
    /// `"ValuesTransform"`.
    #[error("{label}: value is outside {transform:?}'s domain")]
    TransformDomain {
        label: &'static str,
        transform: Transform,
    },
    /// Escape hatch for conditions this crate doesn't model, chiefly custom strategies
    /// validating their own configuration.
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
    /// One entry per coordinate that fell outside the grid. A single point out of bounds
    /// in two dimensions yields two entries.
    #[error("{}", fmt_out_of_bounds(.0))]
    OutOfBounds(Vec<OutOfBoundsAt>),
    /// One entry per point whose length didn't match the interpolator's dimensionality.
    #[error("{}", fmt_point_length(*expected, failures))]
    PointLength {
        expected: usize,
        failures: Vec<WrongLengthAt>,
    },
    #[error("output slice has length {found}, expected {expected}")]
    OutputLength { expected: usize, found: usize },
    /// A query point coordinate lies outside `transform`'s valid domain (see
    /// [`Transform::in_domain`]), checked by [`crate::strategy::GridTransform`]
    /// before its own [`Transform::forward`] call: without this,
    /// `Extrapolate::Enable` pushing a query below e.g. [`Transform::Log`]'s lower
    /// bound would silently produce `NaN` instead of a clear error. `label` is
    /// `"GridTransform"`.
    #[error("{label}: value is outside {transform:?}'s domain")]
    TransformDomain {
        label: &'static str,
        transform: Transform,
    },
    #[error("{0}")]
    Other(String),
}

impl fmt::Debug for InterpolateError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Where a query point left the grid, in [`InterpolateError::OutOfBounds`].
///
/// Index the `points` and `grid` you supplied to recover the offending coordinate and
/// the bounds it missed: those are `D::Elem` values, which the error is not generic over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct OutOfBoundsAt {
    /// Position of the point within the batch, or 0 for a single-point call.
    pub index: usize,
    /// Dimension whose bounds the point exceeded.
    pub dim: usize,
}

/// A misshapen query point, in [`InterpolateError::PointLength`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct WrongLengthAt {
    /// Position of the point within the batch, or 0 for a single-point call.
    pub index: usize,
    /// The length it actually had. The expected length is on the variant, shared by
    /// every entry.
    pub found: usize,
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
fn show_index(mut indices: impl Iterator<Item = usize>) -> bool {
    indices.any(|index| index != 0)
}

/// A lone failure reads as one sentence; several are listed under a summary line.
/// Shared with [`fmt_point_length`] so both aggregating variants render alike.
fn fmt_out_of_bounds(failures: &[OutOfBoundsAt]) -> String {
    let show = show_index(failures.iter().map(|at| at.index));
    match failures {
        [at] => format!(
            "{} is out of bounds in dim {} with `Extrapolate::Error` set",
            point_at(at.index, show),
            at.dim
        ),
        many => {
            // One point can be out of bounds in several dimensions, so the number of
            // failures doesn't decide the plural here; the number of distinct points does.
            let subject = if many.iter().any(|at| at.index != many[0].index) {
                "points"
            } else {
                "point"
            };
            let mut s = format!("{subject} out of bounds with `Extrapolate::Error` set:");
            for at in many {
                s.push_str(&format!(
                    "\n    {} in dim {}",
                    point_at(at.index, show),
                    at.dim
                ));
            }
            s
        }
    }
}

/// See [`fmt_out_of_bounds`].
fn fmt_point_length(expected: usize, failures: &[WrongLengthAt]) -> String {
    let show = show_index(failures.iter().map(|at| at.index));
    match failures {
        [at] => format!(
            "{} has length {}, expected {expected} for {expected}-D interpolation",
            point_at(at.index, show),
            at.found
        ),
        many => {
            // Unlike out-of-bounds, each failure here is a distinct point, so reaching
            // this arm always means more than one.
            let mut s = format!("points have the wrong length for {expected}-D interpolation:");
            for at in many {
                s.push_str(&format!(
                    "\n    {} has length {}",
                    point_at(at.index, show),
                    at.found
                ));
            }
            s
        }
    }
}
