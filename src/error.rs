//! Crate error types

use std::fmt;
use thiserror::Error;

use crate::strategy::Transform;

/// Error in interpolator data validation
#[derive(Error, Clone, PartialEq)]
#[non_exhaustive]
pub enum ValidateError {
    /// The strategy's [`allow_extrapolate`](crate::strategy::traits::Strategy1D::allow_extrapolate)
    /// returns `false`, so `Extrapolate::Enable` isn't supported by it.
    #[error("`Extrapolate::Enable` is not supported by this strategy")]
    ExtrapolateUnsupported,

    /// A grid axis has fewer than the 2 points every strategy needs. The `usize` is
    /// the offending dimension.
    #[error("at least 2 grid points are required per dimension: dim {0}")]
    InsufficientGridPoints(usize),

    /// A grid axis isn't strictly increasing. The `usize` is the offending
    /// dimension.
    #[error("supplied coordinates must be strictly increasing: dim {0}")]
    NotStrictlyIncreasing(usize),

    /// Raised by [`crate::strategy::utils::validate_uniform_grid`] and
    /// [`crate::strategy::utils::validate_uniform_grid_epsilon`], so any strategy
    /// requiring uniform spacing reports it the same way, not just `LinearUniform`.
    ///
    /// `index` is the first coordinate whose following interval, `grid[index + 1] -
    /// grid[index]`, differs from the grid's first interval, in either direction.
    #[error("grid[{dim}] is not uniformly spaced (spacing changes at index {index})")]
    NonUniform {
        /// The axis that isn't uniformly spaced.
        dim: usize,
        /// The first coordinate whose following interval differs from the grid's
        /// first interval.
        index: usize,
    },

    /// The supplied grid and values don't have compatible shapes. The `usize` is the
    /// offending dimension.
    #[error("supplied grid and values are not compatible shapes: dim {0}")]
    IncompatibleShapes(usize),

    /// Number of grid axes doesn't match the dimensionality of `values`. Only reachable
    /// for `InterpDataND`, whose axis count isn't fixed by the type. Distinct from
    /// [`ValidateError::IncompatibleShapes`], which compares extents within one axis.
    #[error("grid has {found} axes, expected {expected} to match the values")]
    GridAxisCount {
        /// Number of axes `values`'s own dimensionality requires.
        expected: usize,
        /// Number of grid axes actually supplied.
        found: usize,
    },

    /// Raised by [`crate::strategy::broadcast::Broadcastable::validate_len`], so any
    /// strategy built on [`crate::strategy::broadcast::Broadcastable`] reports a
    /// mismatched per-axis count
    /// the same way. `label` names the strategy (e.g. `"Step"`); `noun` names what's being
    /// counted (e.g. `"directions"`).
    #[error("{label} has {found} {noun} but interpolator is {ndim}-D (expected {ndim})")]
    PerAxisLen {
        /// Name of the strategy that raised this, e.g. `"Step"`.
        label: &'static str,
        /// What's being counted, e.g. `"directions"`.
        noun: &'static str,
        /// The interpolator's dimensionality.
        ndim: usize,
        /// How many entries were actually supplied.
        found: usize,
    },

    /// A grid coordinate lies outside `transform`'s valid domain (see
    /// [`Transform::in_domain`]), raised per-axis by
    /// [`crate::strategy::GridTransform`]. `index` is the offending coordinate's
    /// position within `grid[dim]`, mirroring [`ValidateError::NonUniform`].
    #[error("GridTransform: grid[{dim}][{index}] is outside {transform:?}'s domain")]
    GridTransformDomain {
        /// The transform whose domain was violated.
        transform: Transform,
        /// The axis the offending coordinate is on.
        dim: usize,
        /// The offending coordinate's position within `grid[dim]`.
        index: usize,
    },

    /// `transform` isn't monotonic across the whole of `grid[dim]`, raised by
    /// [`crate::strategy::GridTransform`]. Every coordinate can individually pass
    /// [`Transform::in_domain`] and still trigger this: `transform`'s domain can be
    /// disconnected (e.g. [`Transform::Reciprocal`]'s `x != 0` is two separate
    /// pieces), and a transform that's monotonic on each piece separately isn't
    /// necessarily monotonic across a raw grid that spans both. `index` is the first
    /// position where the transformed direction breaks, mirroring
    /// [`ValidateError::NonUniform`].
    #[error(
        "GridTransform: grid[{dim}] is not monotonic under {transform:?} \
         (direction changes at index {index})"
    )]
    GridTransformNotMonotonic {
        /// The transform under which `grid[dim]` isn't monotonic.
        transform: Transform,
        /// The axis that isn't monotonic under `transform`.
        dim: usize,
        /// The first position where the transformed direction breaks.
        index: usize,
    },

    /// A data value lies outside `transform`'s valid domain (see
    /// [`Transform::in_domain`]), raised by [`crate::strategy::ValuesTransform`].
    /// `index` is the offending element's position in `values`, e.g. `[6, 153, 2]`.
    #[error(
        "ValuesTransform: values{} is outside {transform:?}'s domain",
        fmt_values_index(index)
    )]
    ValuesTransformDomain {
        /// The transform whose domain was violated.
        transform: Transform,
        /// The offending element's position in `values`, e.g. `[6, 153, 2]`.
        index: Vec<usize>,
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
        /// The dimensionality every point must match.
        expected: usize,
        /// One entry per point with the wrong length.
        failures: Vec<WrongLengthAt>,
    },

    /// The output slice passed to a `*_into` batch call doesn't have the same length
    /// as the points slice.
    #[error("output slice has length {found}, expected {expected}")]
    OutputLength {
        /// Length the output slice needed to be: the number of points.
        expected: usize,
        /// Length the output slice actually was.
        found: usize,
    },

    /// One entry per query point coordinate that fell outside its axis's configured
    /// transform's domain (see [`Transform::in_domain`]), checked by
    /// [`crate::strategy::GridTransform`] before its own [`Transform::forward`] call:
    /// without this, `Extrapolate::Enable` pushing a query below e.g.
    /// [`Transform::Log`]'s lower bound would silently produce `NaN` instead of a
    /// clear error. A single point out of domain in two dimensions yields two
    /// entries.
    #[error("{}", fmt_outside_domain(.0))]
    GridTransformDomain(Vec<OutsideDomainAt>),

    /// Escape hatch for conditions this crate doesn't model, chiefly custom
    /// strategies' own fallible work.
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

/// A query point coordinate outside its axis's transform domain, in
/// [`InterpolateError::GridTransformDomain`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct OutsideDomainAt {
    /// Position of the point within the batch, or 0 for a single-point call.
    pub index: usize,
    /// Dimension whose transform domain the point coordinate violated.
    pub dim: usize,
    /// The transform whose domain was violated.
    pub transform: Transform,
}

/// Renders a values-array index as `values[i][j][k]`, e.g. `values[6][153][2]`.
fn fmt_values_index(index: &[usize]) -> String {
    let mut s = String::from("");
    for i in index {
        s.push_str(&format!("[{i}]"));
    }
    s
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
fn fmt_outside_domain(failures: &[OutsideDomainAt]) -> String {
    let show = show_index(failures.iter().map(|at| at.index));
    match failures {
        [at] => format!(
            "GridTransform: {} is outside {:?}'s domain in dim {}",
            point_at(at.index, show),
            at.transform,
            at.dim
        ),
        many => {
            let subject = if many.iter().any(|at| at.index != many[0].index) {
                "points"
            } else {
                "point"
            };
            let mut s = format!("GridTransform: {subject} outside transform's domain:");
            for at in many {
                s.push_str(&format!(
                    "\n    {} in dim {} ({:?}'s domain)",
                    point_at(at.index, show),
                    at.dim,
                    at.transform
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
