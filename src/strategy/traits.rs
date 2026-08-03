//! Strategy trait definitions for all dimensionalities.

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

/// 1-D interpolation strategy.
pub trait Strategy1D<D>: Debug + DynClone
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    /// Initialize strategy struct, with access to interpolation data.
    fn init(&mut self, _data: &InterpData1D<D>) -> Result<(), ValidateError> {
        Ok(())
    }

    /// Execute interpolation (after handling [`Extrapolate`] setting).
    fn interpolate(
        &self,
        data: &InterpData1D<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError>;

    /// Does this type's [`Strategy1D::interpolate`] provision for extrapolation?
    fn allow_extrapolate(&self) -> bool;
}

clone_trait_object!(<D> Strategy1D<D>);

impl<D> Strategy1D<D> for Box<dyn Strategy1D<D>>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    /// Initialize strategy struct, with access to interpolation data.
    #[inline]
    fn init(&mut self, data: &InterpData1D<D>) -> Result<(), ValidateError> {
        (**self).init(data)
    }

    #[inline]
    fn interpolate(
        &self,
        data: &InterpData1D<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError> {
        (**self).interpolate(data, point)
    }

    #[inline]
    fn allow_extrapolate(&self) -> bool {
        (**self).allow_extrapolate()
    }
}

/// 2-D interpolation strategy.
pub trait Strategy2D<D>: Debug + DynClone
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    /// Initialize strategy struct, with access to interpolation data.
    fn init(&mut self, _data: &InterpData2D<D>) -> Result<(), ValidateError> {
        Ok(())
    }

    /// Execute interpolation (after handling [`Extrapolate`] setting).
    fn interpolate(
        &self,
        data: &InterpData2D<D>,
        point: &[D::Elem; 2],
    ) -> Result<D::Elem, InterpolateError>;

    /// Does this type's [`Strategy2D::interpolate`] provision for extrapolation?
    fn allow_extrapolate(&self) -> bool;
}

clone_trait_object!(<D> Strategy2D<D>);

impl<D> Strategy2D<D> for Box<dyn Strategy2D<D>>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    /// Initialize strategy struct, with access to interpolation data.
    #[inline]
    fn init(&mut self, data: &InterpData2D<D>) -> Result<(), ValidateError> {
        (**self).init(data)
    }

    #[inline]
    fn interpolate(
        &self,
        data: &InterpData2D<D>,
        point: &[D::Elem; 2],
    ) -> Result<D::Elem, InterpolateError> {
        (**self).interpolate(data, point)
    }

    #[inline]
    fn allow_extrapolate(&self) -> bool {
        (**self).allow_extrapolate()
    }
}

/// 3-D interpolation strategy.
pub trait Strategy3D<D>: Debug + DynClone
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    /// Initialize strategy struct, with access to interpolation data.
    fn init(&mut self, _data: &InterpData3D<D>) -> Result<(), ValidateError> {
        Ok(())
    }

    /// Execute interpolation (after handling [`Extrapolate`] setting).
    fn interpolate(
        &self,
        data: &InterpData3D<D>,
        point: &[D::Elem; 3],
    ) -> Result<D::Elem, InterpolateError>;

    /// Does this type's [`Strategy3D::interpolate`] provision for extrapolation?
    fn allow_extrapolate(&self) -> bool;
}

clone_trait_object!(<D> Strategy3D<D>);

impl<D> Strategy3D<D> for Box<dyn Strategy3D<D>>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    /// Initialize strategy struct, with access to interpolation data.
    #[inline]
    fn init(&mut self, data: &InterpData3D<D>) -> Result<(), ValidateError> {
        (**self).init(data)
    }

    #[inline]
    fn interpolate(
        &self,
        data: &InterpData3D<D>,
        point: &[D::Elem; 3],
    ) -> Result<D::Elem, InterpolateError> {
        (**self).interpolate(data, point)
    }

    #[inline]
    fn allow_extrapolate(&self) -> bool {
        (**self).allow_extrapolate()
    }
}

/// N-D interpolation strategy.
pub trait StrategyND<D>: Debug + DynClone
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    /// Initialize strategy struct, with access to interpolation data.
    fn init(&mut self, _data: &InterpDataND<D>) -> Result<(), ValidateError> {
        Ok(())
    }

    /// Execute interpolation (after handling [`Extrapolate`] setting).
    fn interpolate(
        &self,
        data: &InterpDataND<D>,
        point: &[D::Elem],
    ) -> Result<D::Elem, InterpolateError>;

    /// Does this type's [`StrategyND::interpolate`] provision for extrapolation?
    fn allow_extrapolate(&self) -> bool;
}

clone_trait_object!(<D> StrategyND<D>);

/// Solve for natural cubic spline coefficients and evaluate at `point` in one pass.
///
/// Implements the Thomas algorithm for the tridiagonal second-derivative system,
/// then evaluates the resulting cubic polynomial. Used by the 2-D, 3-D, and N-D
/// [`CubicSpline`](crate::strategy::CubicSpline) strategy implementations, which
/// cannot precompute coefficients across all slices.
///
/// `x` and `y` must each have length ≥ 2 (guaranteed by [`InterpData`](crate::data::InterpData) validation).
pub(crate) fn spline_eval_1d<T: Float>(x: ArrayView1<T>, y: ArrayView1<T>, point: T) -> T {
    let n = x.len() - 1;
    let two = T::one() + T::one();
    let six = two + two + two;

    let h: Vec<T> = (0..n).map(|i| x[i + 1] - x[i]).collect();
    let m = n - 1;
    let mut m_inner = vec![T::zero(); m];

    if m > 0 {
        let rhs: Vec<T> = (0..m)
            .map(|k| six * ((y[k + 2] - y[k + 1]) / h[k + 1] - (y[k + 1] - y[k]) / h[k]))
            .collect();

        let mut cp = vec![T::zero(); m];
        let mut dp = vec![T::zero(); m];

        let w0 = two * (h[0] + h[1]);
        cp[0] = if m > 1 { h[1] / w0 } else { T::zero() };
        dp[0] = rhs[0] / w0;

        for k in 1..m {
            let w = two * (h[k] + h[k + 1]) - h[k] * cp[k - 1];
            cp[k] = if k < m - 1 { h[k + 1] / w } else { T::zero() };
            dp[k] = (rhs[k] - h[k] * dp[k - 1]) / w;
        }

        m_inner[m - 1] = dp[m - 1];
        for k in (0..m - 1).rev() {
            m_inner[k] = dp[k] - cp[k] * m_inner[k + 1];
        }
    }

    let m_at = |i: usize| -> T {
        if i == 0 || i == n {
            T::zero()
        } else {
            m_inner[i - 1]
        }
    };

    let i = if point < *x.first().unwrap() {
        0
    } else if point > *x.last().unwrap() {
        n - 1
    } else {
        find_nearest_index(x, &point)
    };

    let dx = point - x[i];
    let b = (y[i + 1] - y[i]) / h[i] - h[i] * (two * m_at(i) + m_at(i + 1)) / six;
    let c = m_at(i) / two;
    let d = (m_at(i + 1) - m_at(i)) / (six * h[i]);
    y[i] + b * dx + c * dx * dx + d * dx * dx * dx
}

/// Recursively evaluates an N-D natural cubic spline via sequential 1-D slicing.
///
/// For each grid point along the first axis, recursively evaluates the remaining
/// (N-1)-D spline, producing a 1-D array of values. A final [`spline_eval_1d`] call
/// interpolates those values along the first axis at `point[0]`.
///
/// `grids[k].len() == values.len_of(Axis(k))` must hold at every recursion level,
/// which is guaranteed by [`InterpData`](crate::data::InterpData) validation.
pub(crate) fn spline_eval_nd_recursive<T: Float>(
    grids: &[ArrayView1<T>],
    values: ArrayViewD<T>,
    point: &[T],
) -> T {
    debug_assert_eq!(grids.len(), point.len());

    if grids.len() == 1 {
        let y = values
            .into_dimensionality::<Ix1>()
            .expect("internal: 1-D base case reached with non-1-D values array");
        return spline_eval_1d(grids[0], y, point[0]);
    }

    let n = grids[0].len();
    let g: Vec<T> = (0..n)
        .map(|i| spline_eval_nd_recursive(&grids[1..], values.index_axis(Axis(0), i), &point[1..]))
        .collect();

    spline_eval_1d(grids[0], ArrayView1::from(&g), point[0])
}

impl<D> StrategyND<D> for Box<dyn StrategyND<D>>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    #[inline]
    fn init(&mut self, data: &InterpDataND<D>) -> Result<(), ValidateError> {
        (**self).init(data)
    }

    #[inline]
    fn interpolate(
        &self,
        data: &InterpDataND<D>,
        point: &[D::Elem],
    ) -> Result<D::Elem, InterpolateError> {
        (**self).interpolate(data, point)
    }

    #[inline]
    fn allow_extrapolate(&self) -> bool {
        (**self).allow_extrapolate()
    }
}
