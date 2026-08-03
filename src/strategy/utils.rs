//! Shared index and grid utilities for interpolation strategies.

use super::*;

/// Find nearest index in `arr` left of `target`
///
/// This method contains code from RouteE Compass, another open-source NLR-developed tool
/// <https://www.nlr.gov/transportation/route-energy-prediction-model.html>
/// <https://github.com/NatLabRockies/routee-compass/>
pub fn find_nearest_index<T: PartialOrd>(arr: ArrayView1<T>, target: &T) -> usize {
    if target == arr.last().unwrap() {
        return arr.len() - 2;
    }

    let mut low = 0;
    let mut high = arr.len() - 1;

    while low < high {
        let mid = low + (high - low) / 2;

        if &arr[mid] >= target {
            high = mid;
        } else {
            low = mid + 1;
        }
    }

    if low > 0 && &arr[low] >= target {
        low - 1
    } else {
        low
    }
}

/// Returns the step index for `point` in `grid` using the given [`StepDirection`].
///
/// Handles all exact grid-point edge cases that arise from [`find_nearest_index`]'s
/// interval semantics (returning the lower bracket rather than the exact position).
pub(crate) fn step_index<T: PartialOrd + Copy>(
    dir: StepDirection,
    grid: ArrayView1<T>,
    point: &T,
) -> usize {
    match dir {
        StepDirection::Lower => {
            let x_l = find_nearest_index(grid, point);
            // find_nearest_index returns i where grid[i] < point <= grid[i+1] for interior
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
            // find_nearest_index returns 0 when point == grid[0], giving x_l+1 = 1
            // which would skip values[0]. Handle the first-element case explicitly:
            if point == grid.first().unwrap() {
                0
            } else {
                find_nearest_index(grid, point) + 1
            }
        }
    }
}

/// Returns the exact grid index if `point` lies on `grid[lower]` or `grid[lower+1]`, else `None`.
///
/// Used to short-circuit interpolation when a query point coincides with a grid coordinate.
pub(crate) fn exact_index<T: PartialOrd>(
    grid: ArrayView1<T>,
    lower: usize,
    point: &T,
) -> Option<usize> {
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
/// Equivalent to [`find_nearest_index`] but replaces binary search with direct arithmetic.
/// Only valid when the grid spacing is uniform — validate with [`check_uniform_grid`] first.
pub(crate) fn uniform_lower_index<T: Float>(grid0: T, step: T, n: usize, point: T) -> usize {
    let t = (point - grid0) / step;
    if t < T::zero() {
        0
    } else {
        t.floor().to_usize().unwrap_or(0).min(n - 2)
    }
}

/// Validates that `grid` is uniformly spaced within floating-point tolerance.
///
/// Uses a relative tolerance of 1024 × ε to accommodate accumulated floating-point rounding
/// error in grids constructed from repeated arithmetic.
pub(crate) fn check_uniform_grid<T: Float>(
    grid: ArrayView1<T>,
    dim: usize,
) -> Result<(), ValidateError> {
    let step = grid[1] - grid[0];
    // 1024 * epsilon via 10 doublings — avoids numeric literal casting
    let tolerance = {
        let mut tol = T::epsilon();
        for _ in 0..10 {
            tol = tol + tol;
        }
        step.abs() * tol
    };
    for i in 1..grid.len() - 1 {
        let gap = grid[i + 1] - grid[i];
        if (gap - step).abs() > tolerance {
            return Err(ValidateError::Other(format!(
                "LinearUniform: grid[{dim}] is not uniformly spaced (gap at index {i})"
            )));
        }
    }
    Ok(())
}
