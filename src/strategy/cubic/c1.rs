//! [`CubicC1`]: C¹ local cubic Hermite spline (finite-difference derivative estimate).

use super::utils::evaluate_hermite_1d;
use super::*;

/// Local cubic Hermite spline interpolation (<https://en.wikipedia.org/wiki/Cubic_Hermite_spline>).
///
/// Constructs a C¹ piecewise cubic Hermite polynomial through all data points, with
/// per-knot derivatives estimated locally (see [`CubicC1DerivativeMode`]) rather than
/// [`CubicC2`]'s global tridiagonal solve. Cheaper to build, at the cost of C¹ (not C²)
/// continuity and no configurable boundary condition. See [`CubicC1CacheMode`] for the
/// caching tradeoff at 2-D and above.
///
/// Supports [`Extrapolate::Enable`](crate::interpolator::Extrapolate::Enable):
/// evaluation beyond the grid extends the boundary Hermite segment.
///
/// # Example
/// ```
/// use ndarray::prelude::*;
/// use ninterp::prelude::*;
///
/// // f(x) = 2x + 1 (linear: reproduced exactly by any Hermite spline)
/// let interp: Interp1D<f64, _> = Interp1D::new(
///     array![0., 1., 2., 3.],
///     array![1., 3., 5., 7.],
///     strategy::CubicC1::default(),
///     Extrapolate::Enable,
/// )
/// .unwrap();
/// assert_eq!(interp.interpolate(&[1.5]).unwrap(), 4.0);
/// assert_eq!(interp.interpolate(&[4.0]).unwrap(), 9.0); // extrapolation
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(serialize = "T: Serialize", deserialize = "T: Deserialize<'de>"))
)]
pub struct CubicC1<T> {
    /// How per-knot derivatives are estimated.
    pub derivative_mode: CubicC1DerivativeMode,
    /// Caching tradeoff at 2-D and above; ignored at 1-D, where the cache is already
    /// O(1) either way.
    pub cache_mode: CubicC1CacheMode,
    /// Precomputed derivative data under [`CubicC1CacheMode::Full`]; same shape as
    /// [`CubicC2::cache`]. Empty under [`CubicC1CacheMode::None`] above 1-D.
    #[cfg_attr(feature = "serde", serde(skip, default = "empty_cache"))]
    pub(crate) cache: ArrayD<T>,
}

/// Derivative-estimation method for [`CubicC1`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub enum CubicC1DerivativeMode {
    /// Unclipped (non-monotonicity-preserving) finite differences: central in the
    /// interior, one-sided at each boundary.
    FiniteDifference,
}

/// Caching strategy for [`CubicC1`] at 2-D and above.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub enum CubicC1CacheMode {
    /// Precompute the full corner-derivative tensor at `init()`, same mechanism as
    /// [`CubicC2`]: every query afterward is an O(1) cache lookup.
    Full,
    /// Skip precomputation: `init()` only validates, and every query derives its
    /// corner derivatives fresh from a local neighborhood. Cost depends only on grid
    /// dimensionality, never grid resolution; agrees with `Full` exactly.
    None,
}

impl<T> Default for CubicC1<T> {
    /// [`CubicC1DerivativeMode::FiniteDifference`] + [`CubicC1CacheMode::Full`].
    fn default() -> Self {
        Self {
            derivative_mode: CubicC1DerivativeMode::FiniteDifference,
            cache_mode: CubicC1CacheMode::Full,
            cache: empty_cache(),
        }
    }
}

impl<T> CubicC1<T> {
    /// Equivalent to [`default`](Self::default).
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets [`cache_mode`](Self::cache_mode).
    pub fn with_cache_mode(mut self, cache_mode: CubicC1CacheMode) -> Self {
        self.cache_mode = cache_mode;
        self
    }
}

// `Full` and `None` cache modes below call the same functions, just at different
// scales: `Full` on the whole grid, `None` on a small local window. This works because
// `fd_derivatives` is a width-<=3 local stencil (unlike `CubicC2`'s `compute_m`
// global solve), so windowing never changes which branch (central vs. one-sided) fires
// for a given knot: the window is only ever clipped at a true grid boundary, exactly
// where the one-sided branch should fire anyway. `Full`/`None` agreeing exactly on any
// query is a consequence of this, not a separate property to maintain by hand.

/// Finite-difference derivative at every knot of a 1-D lane: central in the interior,
/// one-sided at each end. `CubicC1`'s per-axis analog of `CubicC2`'s `compute_m`.
fn fd_derivatives<T: Float>(x: ArrayView1<T>, y: ArrayView1<T>) -> Vec<T> {
    let n = x.len();
    (0..n)
        .map(|i| {
            if i == 0 {
                (y[1] - y[0]) / (x[1] - x[0])
            } else if i == n - 1 {
                (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])
            } else {
                (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
            }
        })
        .collect()
}

/// Caches [`fd_derivatives`]'s output for [`Strategy1D::init`](crate::strategy::traits::Strategy1D::init).
pub(crate) fn compute_fd_cache<T: Float>(x: ArrayView1<T>, y: ArrayView1<T>) -> ArrayD<T> {
    let d = fd_derivatives(x, y);
    ArrayD::from_shape_vec(IxDyn(&[d.len()]), d)
        .expect("fd_derivatives's output length matches its own shape")
}

/// Evaluates [`Strategy1D`](crate::strategy::traits::Strategy1D)'s cached derivatives
/// (from [`compute_fd_cache`]) via [`evaluate_hermite_1d`] at `point`.
pub(crate) fn evaluate_hermite_1d_cached<T: Float>(
    x: ArrayView1<T>,
    y: ArrayView1<T>,
    deriv_cache: ArrayViewD<T>,
    point: T,
) -> Result<T, InterpolateError> {
    let m = deriv_cache.into_dimensionality::<Ix1>().map_err(|_| {
        InterpolateError::Other(
            "internal: non-1-D derivative cache, Strategy1D::cache invariant broken".into(),
        )
    })?;
    let i = locate_lower_index(x, &point);
    let h = x[i + 1] - x[i];
    let t = (point - x[i]) / h;
    Ok(evaluate_hermite_1d(y[i], m[i], y[i + 1], m[i + 1], h, t))
}

/// Same role as `CubicC2`'s function of the same name (differences one axis of a field
/// for the corner-derivative tensor), but no boundary condition to branch on, so there's
/// no homogenization pass either: every field (raw values or an already-differentiated
/// field) is just differenced the same way.
fn corner_cache_axis_pass<T: Float>(
    grid: ArrayView1<T>,
    field: ArrayViewD<T>,
    axis: usize,
) -> ArrayD<T> {
    let axis = Axis(axis);
    let mut out = ArrayD::<T>::zeros(IxDyn(field.shape()));
    for (y, mut out_lane) in field.lanes(axis).into_iter().zip(out.lanes_mut(axis)) {
        let d = fd_derivatives(grid, y);
        out_lane.assign(&ArrayView1::from(&d));
    }
    out
}

/// Finite-difference analog of `CubicC2`'s `compute_corner_cache`: builds the same
/// `2^N`-wide corner-derivative tensor shape, over whatever `grids`/`values` it's given.
/// Called on the whole grid for [`CubicC1CacheMode::Full`], or on a small local window
/// (see [`evaluate_spline_corner_local`]) for [`CubicC1CacheMode::None`].
pub(crate) fn compute_corner_cache_fd<T: Float>(
    grids: &[ArrayView1<T>],
    values: ArrayViewD<T>,
) -> ArrayD<T> {
    let n_axes = grids.len();
    let n_bits = 1usize << n_axes;
    let mask_axis = Axis(n_axes);

    let mut out_shape = values.shape().to_vec();
    out_shape.push(n_bits);
    let mut cache = ArrayD::<T>::zeros(IxDyn(&out_shape));
    cache.index_axis_mut(mask_axis, 0).assign(&values);

    for (axis, grid) in grids.iter().enumerate() {
        let bit = 1usize << axis;
        for mask in 0..bit {
            let field = cache.index_axis(mask_axis, mask).to_owned();
            let deriv = corner_cache_axis_pass(*grid, field.view(), axis);
            cache.index_axis_mut(mask_axis, mask | bit).assign(&deriv);
        }
    }
    cache
}

/// [`CubicC1CacheMode::None`]: extracts a local window (up to 4 points per axis) around
/// `point`'s bracket, builds a small corner-derivative tensor scoped to just that window
/// via [`compute_corner_cache_fd`], then evaluates it via [`evaluate_spline_corner_cached`]
/// unchanged. Cost is bounded by `grids.len()`, never by grid resolution.
pub(crate) fn evaluate_spline_corner_local<T: Float>(
    grids: &[ArrayView1<T>],
    values: ArrayViewD<T>,
    point: &[T],
) -> T {
    let n_axes = grids.len();
    let mut lowers = vec![0usize; n_axes];
    let mut uppers = vec![0usize; n_axes];
    for axis in 0..n_axes {
        let l = locate_lower_index(grids[axis], &point[axis]);
        lowers[axis] = l.saturating_sub(1);
        uppers[axis] = (l + 2).min(grids[axis].len() - 1);
    }

    let window_grids: Vec<ArrayView1<T>> = grids
        .iter()
        .enumerate()
        .map(|(axis, g)| g.slice_axis(Axis(0), Slice::from(lowers[axis]..=uppers[axis])))
        .collect();

    let window_values = values.slice_each_axis(|ax| {
        let axis = ax.axis.index();
        Slice::from(lowers[axis]..=uppers[axis])
    });

    let local_cache = compute_corner_cache_fd(&window_grids, window_values);
    evaluate_spline_corner_cached(&window_grids, local_cache.view(), point)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fd_derivatives_matches_hand_computed() {
        // Non-uniform grid, non-polynomial data.
        let x = [0.0, 0.4, 1.1, 1.8, 2.9, 4.0];
        let y = [0.5, 1.2, 0.3, 2.1, 1.0, 3.3];
        let xv = ArrayView1::from(&x);
        let yv = ArrayView1::from(&y);

        let got = fd_derivatives(xv, yv);
        let expected = [
            (y[1] - y[0]) / (x[1] - x[0]), // one-sided, lower boundary
            (y[2] - y[0]) / (x[2] - x[0]), // central
            (y[3] - y[1]) / (x[3] - x[1]), // central
            (y[4] - y[2]) / (x[4] - x[2]), // central
            (y[5] - y[3]) / (x[5] - x[3]), // central
            (y[5] - y[4]) / (x[5] - x[4]), // one-sided, upper boundary
        ];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert_approx_eq!(*g, *e, 1e-12);
        }
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_serde() {
        assert_eq!(
            serde_json::to_string(&CubicC1::<f64>::default()).unwrap(),
            r#"{"derivative_mode":"FiniteDifference","cache_mode":"Full"}"#
        );
        let none_mode = CubicC1::<f64>::new().with_cache_mode(CubicC1CacheMode::None);
        assert_eq!(
            serde_json::to_string(&none_mode).unwrap(),
            r#"{"derivative_mode":"FiniteDifference","cache_mode":"None"}"#
        );
        assert_eq!(
            serde_json::from_str::<CubicC1<f64>>(&serde_json::to_string(&none_mode).unwrap())
                .unwrap(),
            none_mode
        );
    }
}
