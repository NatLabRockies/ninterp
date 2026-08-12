//! Single-axis primitives for locating a query point within one grid dimension.
//! Strategies compose these per axis to get 1D/2D/3D/ND behavior; none of them
//! iterate over dimensions themselves.
//!
//! These are the same building blocks [`super::Linear`], [`super::LinearUniform`], and
//! [`super::Step`] are implemented with, and are public for reuse in custom strategies
//! (see [`crate::strategy::traits`]).

use super::*;

/// Returns the lower bracket index for `point` in `grid`: the largest `i` such that
/// `grid[i] <= point`, clamped to `[0, grid.len() - 2]` for out-of-range points.
///
/// This method contains code from RouteE Compass, another open-source NLR-developed tool
/// <https://www.nlr.gov/transportation/route-energy-prediction-model.html>
/// <https://github.com/NatLabRockies/routee-compass/>
pub fn locate_lower_index<T: PartialOrd>(grid: ArrayView1<T>, point: &T) -> usize {
    if point < grid.first().unwrap() {
        return 0;
    }
    if point >= grid.last().unwrap() {
        return grid.len() - 2;
    }

    let mut low = 0;
    let mut high = grid.len() - 1;

    while low < high {
        let mid = low + (high - low) / 2;

        if &grid[mid] >= point {
            high = mid;
        } else {
            low = mid + 1;
        }
    }

    if low > 0 && &grid[low] >= point {
        low - 1
    } else {
        low
    }
}

/// Per-axis locate for linear-family strategies: either an exact grid hit,
/// or an interior interpolation position.
pub enum AxisLocation<T> {
    /// `point` coincides exactly with `grid[_]` at this index.
    Exact(usize),
    /// `point` falls strictly between two grid coordinates.
    Between {
        /// Index of the lower bracketing grid coordinate.
        lower: usize,
        /// Fractional position of `point` between `grid[lower]` and `grid[lower + 1]`, in `[0, 1)`.
        frac: T,
    },
}

/// Locates `point` along `grid`, resolving to an exact grid hit or an interpolation
/// position. Combines [`locate_lower_index`] (search + extrapolation clamp) with
/// [`exact_index`] (exact-match short-circuit) into the single call linear-family
/// strategies need per axis.
pub fn locate_axis<T: Float>(grid: ArrayView1<T>, point: &T) -> AxisLocation<T> {
    let lower = locate_lower_index(grid, point);
    match exact_index(grid, lower, point) {
        Some(idx) => AxisLocation::Exact(idx),
        None => {
            let frac = (*point - grid[lower]) / (grid[lower + 1] - grid[lower]);
            AxisLocation::Between { lower, frac }
        }
    }
}

/// Returns the step index for `point` in `grid` using the given [`StepDirection`].
///
/// Handles all exact grid-point edge cases that arise from [`locate_lower_index`]'s
/// interval semantics (returning the lower bracket rather than the exact position).
pub fn locate_step_index<T: PartialOrd + Copy>(
    dir: StepDirection,
    grid: ArrayView1<T>,
    point: &T,
) -> usize {
    match dir {
        StepDirection::Lower => {
            let x_l = locate_lower_index(grid, point);
            // locate_lower_index returns i where grid[i] < point <= grid[i+1] for interior
            // matches, so an exact match at grid[i+1] gives i instead of i+1. Correct both:
            if point == grid.last().unwrap() {
                grid.len() - 1
            } else if *point == grid[x_l + 1] {
                x_l + 1
            } else {
                x_l
            }
        }
        StepDirection::Upper => {
            // locate_lower_index returns 0 when point == grid[0], giving x_l+1 = 1
            // which would skip values[0]. Handle the first-element case explicitly:
            if point == grid.first().unwrap() {
                0
            } else {
                locate_lower_index(grid, point) + 1
            }
        }
    }
}

/// Returns the exact grid index if `point` lies on `grid[lower]` or `grid[lower+1]`, else `None`.
///
/// Used to short-circuit interpolation when a query point coincides with a grid coordinate.
pub fn exact_index<T: PartialOrd>(grid: ArrayView1<T>, lower: usize, point: &T) -> Option<usize> {
    if grid[lower] == *point {
        Some(lower)
    } else if grid[lower + 1] == *point {
        Some(lower + 1)
    } else {
        None
    }
}

/// Computes the lower bracket index for a uniformly-spaced grid in O(1).
///
/// Equivalent to [`locate_lower_index`] but replaces binary search with direct arithmetic.
/// Only valid when the grid spacing is uniform; validate with [`validate_uniform_grid`] first.
pub fn locate_lower_index_uniform<T: Float>(grid0: T, step: T, n: usize, point: T) -> usize {
    let t = (point - grid0) / step;
    if t < T::zero() {
        0
    } else {
        t.floor().to_usize().unwrap_or(0).min(n - 2)
    }
}

/// Validates that `grid` is uniformly spaced within `tolerance`.
///
/// `tolerance` is an absolute bound, in the same units as `grid`, on how far an interval
/// may drift from the grid's first interval (either wider or narrower). Pass `None` for a
/// sensible default: a relative tolerance of 1024 × ε, scaled by the grid's own step
/// size, loose enough to absorb the rounding error that accumulates in grids built from
/// repeated arithmetic (e.g. `Array::linspace`) while still catching a grid that was
/// never meant to be uniform. Pass `Some(_)` when a strategy's precision needs differ
/// from that default.
///
/// Pair with [`locate_lower_index_uniform`] for the matching O(1) lookup.
pub fn validate_uniform_grid<T: Float>(
    grid: ArrayView1<T>,
    dim: usize,
    tolerance: Option<T>,
) -> Result<(), ValidateError> {
    let step = grid[1] - grid[0];
    let tolerance = tolerance.unwrap_or_else(|| {
        // 1024 * epsilon via 10 doublings, avoids numeric literal casting
        let mut tol = T::epsilon();
        for _ in 0..10 {
            tol = tol + tol;
        }
        step.abs() * tol
    });
    for i in 1..grid.len() - 1 {
        // Wider or narrower than the first interval both count as non-uniform.
        let spacing = grid[i + 1] - grid[i];
        if (spacing - step).abs() > tolerance {
            return Err(ValidateError::NonUniform { dim, index: i });
        }
    }
    Ok(())
}
