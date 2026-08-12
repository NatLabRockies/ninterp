//! Cubic interpolation algorithms shared across all dimensionalities, for [`CubicC2`].

use super::*;
use ndarray::Zip;

/// Cubic spline interpolation (<https://en.wikipedia.org/wiki/Spline_interpolation>).
///
/// Constructs a C² piecewise cubic polynomial through all data points.
/// The boundary condition is set by [`boundary_conditions`](CubicC2::boundary_conditions).
/// Coefficients are precomputed in [`Strategy1D::init`], called automatically
/// by [`Interp1D::new`](crate::interpolator::Interp1D::new) and
/// [`Interp1D::set_strategy`](crate::interpolator::Interp1D::set_strategy).
///
/// Supports [`Extrapolate::Enable`](crate::interpolator::Extrapolate::Enable):
/// evaluation beyond the grid extends the boundary cubic polynomials.
///
/// # Example
/// ```
/// use ndarray::prelude::*;
/// use ninterp::prelude::*;
///
/// // f(x) = 2x + 1 (linear: reproduced exactly by any spline)
/// let interp: Interp1D<f64, _> = Interp1D::new(
///     array![0., 1., 2., 3.],
///     array![1., 3., 5., 7.],
///     strategy::CubicC2::not_a_knot(),
///     Extrapolate::Enable,
/// )
/// .unwrap();
/// assert_eq!(interp.interpolate(&[1.5]).unwrap(), 4.0);
/// assert_eq!(interp.interpolate(&[4.0]).unwrap(), 9.0); // extrapolation
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct CubicC2<T> {
    /// Boundary conditions, one per dimension or a single entry broadcast to all.
    pub boundary_conditions: Vec<CubicC2BoundaryConditions<T>>,
    /// Precomputed derivative data, populated by `Strategy1D`/`2D`/`3D`/`ND::init`. Its
    /// shape depends on which of those populated it:
    ///
    /// - `Strategy1D`/`StrategyND`: cached second derivatives (`M[i] = S''(x_i)`) for
    ///   every 1-D pencil along the innermost (last) grid axis, one
    ///   [`compute_m`] result per combination of the other axes' indices. Same shape
    ///   as the value grid, with the same last-axis length as the innermost grid; for
    ///   1-D data (no outer axes) this degenerates to a single pencil, i.e. shape
    ///   `[n + 1]` for `n` intervals. Only the innermost axis's coefficients are
    ///   query-independent, so this is as far as caching goes; outer axes still solve
    ///   fresh per call.
    /// - `Strategy2D`/`Strategy3D`: the full corner-derivative tensor, shaped like the
    ///   value grid with one extra trailing axis of length `2^N` (`N` = 2 or 3): see
    ///   [`compute_corner_cache`]. Every query is then an O(1) lookup, no solving.
    ///
    /// Not included in the serialized form. After deserializing, call the
    /// interpolator's `init_strategy` method (e.g.
    /// [`Interp1D::init_strategy`](crate::interpolator::Interp1D::init_strategy)) to
    /// recompute this before use.
    #[cfg_attr(feature = "serde", serde(skip, default = "empty_cache"))]
    pub(crate) cache: ArrayD<T>,
}

/// Boundary conditions for [`CubicC2`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub enum CubicC2BoundaryConditions<T> {
    /// C³ continuity at the second and penultimate knots; no extra input required.
    /// Generally gives better accuracy than
    /// [`Natural`](CubicC2BoundaryConditions::Natural) for smooth functions.
    NotAKnot,
    /// Zero second derivative at both endpoints.
    Natural,
    /// Specified first derivative at both endpoints.
    Clamped {
        /// First derivative at the left (lower) endpoint.
        left: T,
        /// First derivative at the right (upper) endpoint.
        right: T,
    },
    /// First and second derivatives match at both endpoints. By convention
    /// `values[n]` (the last point along this axis) should equal `values[0]`, since
    /// that's what makes the axis periodic. This isn't enforced: `values[n]` is read and
    /// used like any other data point (both to build the periodic system and to
    /// evaluate the last interval), just never compared against `values[0]`.
    Periodic,
}

/// Placeholder for [`CubicC2::cache`] before [`Strategy1D::init`]/`2D`/`3D`/`ND`
/// populates it. No `T: Clone + Zero` bound needed (unlike `ArrayD::zeros`), so the
/// constructors below stay unconstrained.
fn empty_cache<T>() -> ArrayD<T> {
    ArrayD::from_shape_vec(IxDyn(&[0]), Vec::new()).expect("empty shape matches empty vec")
}

impl<T> CubicC2<T> {
    /// Returns the boundary condition for the given dimension.
    /// A single-entry vec is broadcast to all dimensions.
    pub(crate) fn bc_for_dim(&self, dim: usize) -> &CubicC2BoundaryConditions<T> {
        let bcs = &self.boundary_conditions;
        &bcs[if bcs.len() == 1 { 0 } else { dim }]
    }

    /// Create a cubic spline with a distinct boundary condition per grid dimension.
    /// A single-entry vec is broadcast to all dimensions instead (same rule
    /// `interpolate` uses internally to pick each axis's condition).
    ///
    /// Use [`not_a_knot`](Self::not_a_knot), [`natural`](Self::natural),
    /// [`clamped`](Self::clamped), or [`periodic`](Self::periodic) instead when every
    /// dimension shares the same condition.
    pub fn new(boundary_conditions: Vec<CubicC2BoundaryConditions<T>>) -> Self {
        Self {
            boundary_conditions,
            cache: empty_cache(),
        }
    }

    /// Create a cubic spline with not-a-knot boundary conditions.
    /// Requires at least 4 data points per dimension.
    pub fn not_a_knot() -> Self {
        Self {
            boundary_conditions: vec![CubicC2BoundaryConditions::NotAKnot],
            cache: empty_cache(),
        }
    }

    /// Create a cubic spline with natural (zero second derivative at endpoints) BCs.
    pub fn natural() -> Self {
        Self {
            boundary_conditions: vec![CubicC2BoundaryConditions::Natural],
            cache: empty_cache(),
        }
    }

    /// Create a cubic spline with specified first derivatives at both endpoints.
    pub fn clamped(left: T, right: T) -> Self {
        Self {
            boundary_conditions: vec![CubicC2BoundaryConditions::Clamped { left, right }],
            cache: empty_cache(),
        }
    }

    /// Create a cubic spline with periodic boundary conditions. By convention
    /// `values[n]` (the last point along each periodic axis) should equal
    /// `values[0]`, though this isn't enforced.
    pub fn periodic() -> Self {
        Self {
            boundary_conditions: vec![CubicC2BoundaryConditions::Periodic],
            cache: empty_cache(),
        }
    }
}

/// Thomas algorithm (tridiagonal matrix algorithm). Solves `A * x = rhs`.
/// `sub.len() == sup.len() == diag.len() - 1`.
pub(crate) fn thomas<T: Float>(sub: &[T], diag: &[T], sup: &[T], rhs: &[T]) -> Vec<T> {
    let n = diag.len();
    let mut cp = vec![T::zero(); n];
    let mut dp = vec![T::zero(); n];
    cp[0] = if n > 1 { sup[0] / diag[0] } else { T::zero() };
    dp[0] = rhs[0] / diag[0];
    for k in 1..n {
        let w = diag[k] - sub[k - 1] * cp[k - 1];
        cp[k] = if k < n - 1 { sup[k] / w } else { T::zero() };
        dp[k] = (rhs[k] - sub[k - 1] * dp[k - 1]) / w;
    }
    let mut x = vec![T::zero(); n];
    x[n - 1] = dp[n - 1];
    for k in (0..n - 1).rev() {
        x[k] = dp[k] - cp[k] * x[k + 1];
    }
    x
}

/// Sherman-Morrison cyclic tridiagonal solver.
/// Corner elements `corner` appear at `(0, n-1)` and `(n-1, 0)`.
/// `sub.len() == sup.len() == n - 1`.
pub(crate) fn cyclic_thomas<T: Float>(
    sub: &[T],
    diag: &[T],
    sup: &[T],
    rhs: &[T],
    corner: T,
) -> Vec<T> {
    let n = diag.len();
    if n == 1 {
        return vec![rhs[0] / (diag[0] + corner + corner)];
    }
    let gamma = -diag[0];
    let c_over_g = corner / gamma;
    let mut diag_mod = diag.to_vec();
    diag_mod[0] = diag_mod[0] - gamma;
    diag_mod[n - 1] = diag_mod[n - 1] - corner * corner / gamma;
    let y = thomas(sub, &diag_mod, sup, rhs);
    let mut u_vec = vec![T::zero(); n];
    u_vec[0] = gamma;
    u_vec[n - 1] = corner;
    let z = thomas(sub, &diag_mod, sup, &u_vec);
    let vt_y = y[0] + c_over_g * y[n - 1];
    let vt_z = z[0] + c_over_g * z[n - 1];
    let factor = vt_y / (T::one() + vt_z);
    y.into_iter()
        .zip(z.iter())
        .map(|(yi, zi)| yi - factor * *zi)
        .collect()
}

/// Computes the second-derivative vector `M[0..=n]` for the given cubic spline BC.
///
/// Used by [`Strategy1D::init`] (stored in `CubicC2::cache`) and
/// [`spline_eval_1d`] (on the fly).
pub(crate) fn compute_m<T: Float>(
    x: ArrayView1<T>,
    y: ArrayView1<T>,
    bc: &CubicC2BoundaryConditions<T>,
) -> Vec<T> {
    let n = x.len() - 1;
    let two = T::one() + T::one();
    let six = two + two + two;
    let h: Vec<T> = (0..n).map(|i| x[i + 1] - x[i]).collect();
    let slopes: Vec<T> = (0..n).map(|i| (y[i + 1] - y[i]) / h[i]).collect();
    let u: Vec<T> = (0..n.saturating_sub(1))
        .map(|k| six * (slopes[k + 1] - slopes[k]))
        .collect();

    match bc {
        CubicC2BoundaryConditions::NotAKnot => {
            // n >= 3 (grid_len >= 4) is validated by `validate_bc_min_points` before
            // this is ever reached through the normal `validate()` -> `init()` path.
            let mut sub: Vec<T> = h[1..n - 2].to_vec();
            sub.push(h[n - 2] * h[n - 2] - h[n - 1] * h[n - 1]);
            let mut sup = vec![h[1] * h[1] - h[0] * h[0]];
            sup.extend_from_slice(&h[2..n - 1]);
            let mut diag = vec![(h[0] + h[1]) * (h[0] + two * h[1])];
            for k in 1..n - 2 {
                diag.push(two * (h[k] + h[k + 1]));
            }
            diag.push((h[n - 2] + h[n - 1]) * (two * h[n - 2] + h[n - 1]));
            let mut rhs = vec![h[1] * u[0]];
            rhs.extend_from_slice(&u[1..n - 2]);
            rhs.push(h[n - 2] * u[n - 2]);
            let inner = thomas(&sub, &diag, &sup, &rhs);
            let m0 = ((h[0] + h[1]) * inner[0] - h[0] * inner[1]) / h[1];
            let mn = ((h[n - 2] + h[n - 1]) * inner[n - 2] - h[n - 1] * inner[n - 3]) / h[n - 2];
            let mut m = vec![m0];
            m.extend(inner);
            m.push(mn);
            m
        }
        CubicC2BoundaryConditions::Natural => {
            let mut sub = h[..n.saturating_sub(1)].to_vec();
            sub.push(T::zero());
            let mut diag = vec![T::one()];
            for k in 0..n.saturating_sub(1) {
                diag.push(two * (h[k] + h[k + 1]));
            }
            diag.push(T::one());
            let mut sup = vec![T::zero()];
            sup.extend_from_slice(&h[1..n]);
            let mut rhs = vec![T::zero()];
            rhs.extend_from_slice(&u);
            rhs.push(T::zero());
            thomas(&sub, &diag, &sup, &rhs)
        }
        CubicC2BoundaryConditions::Clamped { left, right } => {
            let (l, r) = (*left, *right);
            let mut diag = vec![two * h[0]];
            for k in 0..n.saturating_sub(1) {
                diag.push(two * (h[k] + h[k + 1]));
            }
            diag.push(two * h[n - 1]);
            let mut rhs = vec![six * (slopes[0] - l)];
            rhs.extend_from_slice(&u);
            rhs.push(six * (r - slopes[n - 1]));
            thomas(&h, &diag, &h, &rhs)
        }
        CubicC2BoundaryConditions::Periodic => {
            // `y[n]` is read below, via `slopes[n - 1]` in the `rhs[0]` line, and
            // `eval_spline_from_m` also reads `y[n]` directly when evaluating the last
            // interval. By convention `y[n]` should equal `y[0]`, but nothing here
            // compares them; `y[n]` is just used as ordinary data either way.
            if n < 2 {
                vec![T::zero(); n + 1]
            } else {
                let sub_sup = h[..n - 1].to_vec();
                let mut diag = vec![two * (h[n - 1] + h[0])];
                for k in 1..n {
                    diag.push(two * (h[k - 1] + h[k]));
                }
                let mut rhs = vec![six * (slopes[0] - slopes[n - 1])];
                rhs.extend_from_slice(&u);
                let corner = h[n - 1];
                let mut m_vals = cyclic_thomas(&sub_sup, &diag, &sub_sup, &rhs, corner);
                let m0 = m_vals[0];
                m_vals.push(m0);
                m_vals
            }
        }
    }
}

/// Evaluates the M-form cubic spline at `point` using precomputed second derivatives `m`.
pub(crate) fn eval_spline_from_m<T: Float>(
    x: ArrayView1<T>,
    y: ArrayView1<T>,
    m: ArrayView1<T>,
    point: T,
) -> T {
    let two = T::one() + T::one();
    let six = two + two + two;
    let i = locate_lower_index(x, &point);
    let h = x[i + 1] - x[i];
    let dx = point - x[i];
    let dx_r = h - dx;
    let six_h = six * h;
    let h2_over_six = h * h / six;
    m[i] * dx_r * dx_r * dx_r / six_h
        + m[i + 1] * dx * dx * dx / six_h
        + (y[i] - m[i] * h2_over_six) * dx_r / h
        + (y[i + 1] - m[i + 1] * h2_over_six) * dx / h
}

/// Computes and evaluates a 1-D cubic spline at `point` in one pass, respecting `bc`.
///
/// Used for the outer (query-dependent) axes in [`spline_eval_nd_cached`], which cannot
/// be precomputed since their inputs change with the query point.
pub(crate) fn spline_eval_1d<T: Float>(
    x: ArrayView1<T>,
    y: ArrayView1<T>,
    point: T,
    bc: &CubicC2BoundaryConditions<T>,
) -> T {
    let m = compute_m(x, y, bc);
    eval_spline_from_m(x, y, ArrayView1::from(&m), point)
}

/// Checks `grid_len` against boundary condition `bc`'s minimum point requirement (e.g.
/// [`CubicC2BoundaryConditions::NotAKnot`] needs at least 4), ahead of the real work in
/// [`compute_m`].
///
/// Pure, no mutation; used by each dimensionality's `Strategy*D::validate`.
pub(crate) fn validate_bc_min_points<T>(
    bc: &CubicC2BoundaryConditions<T>,
    grid_len: usize,
    dim: usize,
) -> Result<(), ValidateError> {
    if matches!(bc, CubicC2BoundaryConditions::NotAKnot) && grid_len < 4 {
        return Err(ValidateError::Other(format!(
            "CubicC2: dim {dim} has {grid_len} grid points; NotAKnot requires at least 4"
        )));
    }
    Ok(())
}

/// Checks `boundary_conditions.len()` against `ndim`: must be either 1 (broadcast to
/// every axis, [`CubicC2::bc_for_dim`]'s other case) or exactly `ndim` (one per axis).
/// `bc_for_dim` indexes unchecked assuming this already holds, so every
/// `Strategy1D`/`2D`/`3D`/`ND::validate` must call this before any `bc_for_dim` call --
/// mirrors [`Step::validate_len`](crate::strategy::Step::validate_len)'s check for the
/// same kind of per-dimension config vec.
pub(crate) fn validate_bc_count<T>(
    bcs: &[CubicC2BoundaryConditions<T>],
    ndim: usize,
) -> Result<(), ValidateError> {
    let found = bcs.len();
    if found == 1 || found == ndim {
        return Ok(());
    }
    let expected = if ndim == 1 {
        "1".to_string()
    } else {
        format!("1 or {ndim}")
    };
    Err(ValidateError::Other(format!(
        "CubicC2 has {found} boundary conditions but interpolator is {ndim}-D (expected {expected})"
    )))
}

/// Precomputes second-derivative coefficients for every 1-D pencil along the innermost
/// (last) axis of `values`, for [`spline_eval_nd_cached`] to look up in O(1) instead of
/// re-solving on every `interpolate` call. Only the innermost axis's system is
/// query-independent; outer axes still solve fresh via [`spline_eval_1d`] since their
/// inputs (the collapsed values from inner axes) depend on the query point.
///
/// Returns an array shaped like `values`, except the last axis is `inner_grid.len()`
/// long (matching [`compute_m`]'s output length). For 1-D `values` (no outer axes) this
/// degenerates to a single pencil, i.e. one [`compute_m`] call.
pub(crate) fn compute_m_inner_cache<T: Float>(
    inner_grid: ArrayView1<T>,
    values: ArrayViewD<T>,
    bc: &CubicC2BoundaryConditions<T>,
) -> ArrayD<T> {
    let last_axis = Axis(values.ndim() - 1);
    let mut out_shape = values.shape().to_vec();
    out_shape[values.ndim() - 1] = inner_grid.len();
    let mut cache = ArrayD::<T>::zeros(IxDyn(&out_shape));
    for (y, mut out_lane) in values
        .lanes(last_axis)
        .into_iter()
        .zip(cache.lanes_mut(last_axis))
    {
        let m = compute_m(inner_grid, y, bc);
        out_lane.assign(&ArrayView1::from(&m));
    }
    cache
}

/// Recursively evaluates an N-D cubic spline via sequential 1-D slicing, respecting `bcs`.
/// Looks up `m_cache` (from [`compute_m_inner_cache`]) at the innermost axis instead of
/// re-solving it; outer axes still solve fresh via [`spline_eval_1d`].
///
/// `bcs` length 1 broadcasts to all dimensions; length N applies per-dimension.
pub(crate) fn spline_eval_nd_cached<T: Float>(
    grids: &[ArrayView1<T>],
    values: ArrayViewD<T>,
    m_cache: ArrayViewD<T>,
    point: &[T],
    bcs: &[CubicC2BoundaryConditions<T>],
) -> Result<T, InterpolateError> {
    debug_assert_eq!(grids.len(), point.len());
    debug_assert!(!bcs.is_empty());

    let current_bc = &bcs[0];
    let next_bcs = if bcs.len() > 1 { &bcs[1..] } else { bcs };

    if grids.len() == 1 {
        let y = values.into_dimensionality::<Ix1>().map_err(|_| {
            InterpolateError::Other(
                "internal: non-1-D values at 1-D base case, grids.len() == values.ndim() invariant broken".into(),
            )
        })?;
        let m = m_cache.into_dimensionality::<Ix1>().map_err(|_| {
            InterpolateError::Other(
                "internal: non-1-D m_cache at 1-D base case, grids.len() == m_cache.ndim() invariant broken".into(),
            )
        })?;
        return Ok(eval_spline_from_m(grids[0], y, m, point[0]));
    }

    let n = grids[0].len();
    let g: Vec<T> = (0..n)
        .map(|i| {
            spline_eval_nd_cached(
                &grids[1..],
                values.index_axis(Axis(0), i),
                m_cache.index_axis(Axis(0), i),
                &point[1..],
                next_bcs,
            )
        })
        .collect::<Result<Vec<T>, _>>()?;

    Ok(spline_eval_1d(
        grids[0],
        ArrayView1::from(&g),
        point[0],
        current_bc,
    ))
}

/// Closed-form first derivative `S'(x_i)` at every knot, from an already-solved moment
/// vector `m` (no extra solve) -- the companion to [`compute_m`], used to build
/// [`compute_corner_cache`]'s derivative fields.
pub(crate) fn knot_derivatives_from_m<T: Float>(
    x: ArrayView1<T>,
    y: ArrayView1<T>,
    m: ArrayView1<T>,
) -> Vec<T> {
    let n = x.len() - 1;
    let two = T::one() + T::one();
    let six = two + two + two;
    let mut d: Vec<T> = (0..n)
        .map(|i| {
            let h = x[i + 1] - x[i];
            (y[i + 1] - y[i]) / h - h * (two * m[i] + m[i + 1]) / six
        })
        .collect();
    let h_last = x[n] - x[n - 1];
    d.push((y[n] - y[n - 1]) / h_last + h_last * (m[n - 1] + two * m[n]) / six);
    d
}

/// Splines every 1-D lane of `field` along `axis` and replaces it with its knot
/// derivatives (via [`compute_m`] + [`knot_derivatives_from_m`]), returning a new array
/// the same shape as `field`. Generalizes [`compute_m_inner_cache`]'s per-lane iteration
/// (hardcoded to the last axis) to an arbitrary one, for [`compute_corner_cache`].
fn corner_cache_axis_pass<T: Float>(
    grid: ArrayView1<T>,
    field: ArrayViewD<T>,
    axis: usize,
    bc: &CubicC2BoundaryConditions<T>,
) -> ArrayD<T> {
    let axis = Axis(axis);
    let mut out = ArrayD::<T>::zeros(IxDyn(field.shape()));
    for (y, mut out_lane) in field.lanes(axis).into_iter().zip(out.lanes_mut(axis)) {
        let m = compute_m(grid, y, bc);
        let d = knot_derivatives_from_m(grid, y, ArrayView1::from(&m));
        out_lane.assign(&ArrayView1::from(&d));
    }
    out
}

/// Precomputes the full corner-derivative tensor for [`CubicC2`]'s `Strategy2D`/
/// `Strategy3D` full-cache upgrade: for every grid point, all `2^N` partial-derivative
/// combinations (value, first partials, and mixed partials) needed to evaluate a Hermite
/// patch in O(1) via [`spline_eval_corner_cached`].
///
/// Returns an array shaped like `values`, with one extra trailing axis of length `2^N`
/// (`N = grids.len()`). Index `k` in that axis is a bitmask over the `N` grid axes: bit
/// `i` set means "this entry is `d/dx_i` of the value", so `k == 0` is the raw value and
/// `k == 2^N - 1` is the full mixed partial.
///
/// Built by processing axes `0..N` in order: for every mask not yet including axis `i`'s
/// bit, splines the corresponding field along axis `i` ([`corner_cache_axis_pass`]) to
/// populate the entry with that bit added. Boundary condition per axis is
/// [`CubicC2::bc_for_dim`] when splining the raw values (`mask == 0`, this axis's own
/// first-derivative pass). For every other `mask` (a cross-derivative pass over an
/// already-differentiated field, not the original data), `Clamped` falls back to
/// `Clamped { left: 0, right: 0 }`: a clamped axis fixes the same scalar derivative at
/// every point along that boundary, so differentiating that (constant-along-the-boundary)
/// field with respect to any other axis must itself be zero at that axis's own endpoints.
/// That's a zero-*first*-derivative condition on the field being splined, not `Natural`'s
/// zero-*second*-derivative one. Verified empirically: with non-separable data, only the
/// homogeneous-`Clamped` fallback makes this cached path agree (to float precision) with
/// `StrategyND`'s independent recursive solve; `Natural` does not.
pub(crate) fn compute_corner_cache<T: Float>(
    grids: &[ArrayView1<T>],
    values: ArrayViewD<T>,
    bcs: &[CubicC2BoundaryConditions<T>],
) -> ArrayD<T> {
    let n_axes = grids.len();
    let n_bits = 1usize << n_axes;
    let mask_axis = Axis(n_axes);

    let mut out_shape = values.shape().to_vec();
    out_shape.push(n_bits);
    let mut cache = ArrayD::<T>::zeros(IxDyn(&out_shape));
    cache.index_axis_mut(mask_axis, 0).assign(&values);

    for axis in 0..n_axes {
        let bit = 1usize << axis;
        let bc_axis = &bcs[if bcs.len() == 1 { 0 } else { axis }];
        let cross_bc = match bc_axis {
            CubicC2BoundaryConditions::Clamped { .. } => CubicC2BoundaryConditions::Clamped {
                left: T::zero(),
                right: T::zero(),
            },
            other => other.clone(),
        };
        for mask in 0..bit {
            let bc = if mask == 0 { bc_axis } else { &cross_bc };
            let field = cache.index_axis(mask_axis, mask).to_owned();
            let deriv = corner_cache_axis_pass(grids[axis], field.view(), axis, bc);
            cache.index_axis_mut(mask_axis, mask | bit).assign(&deriv);
        }
    }
    cache
}

/// Standard cubic Hermite basis blend of endpoint values `p0`/`p1` and their derivatives
/// `m0`/`m1` (actual derivatives, not yet scaled by the interval width `h`) at
/// fractional position `t` in `[0, 1]` between them.
fn hermite_eval_1d<T: Float>(p0: T, m0: T, p1: T, m1: T, h: T, t: T) -> T {
    let two = T::one() + T::one();
    let three = two + T::one();
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = two * t3 - three * t2 + T::one();
    let h10 = t3 - two * t2 + t;
    let h01 = -two * t3 + three * t2;
    let h11 = t3 - t2;
    h00 * p0 + h10 * h * m0 + h01 * p1 + h11 * h * m1
}

/// Evaluates a Hermite patch from the full corner-derivative tensor built by
/// [`compute_corner_cache`] in O(1), no solving, unlike [`spline_eval_nd_cached`]'s outer
/// axes. Used by [`Strategy2D`]/[`Strategy3D`]'s `CubicC2::interpolate`.
///
/// First localizes every axis down to just its query cell's two bounding grid indices,
/// into an owned `2 x 2 x ... x 2 x 2^N` buffer (`N = grids.len()`) that's independent of
/// grid size. Without this step the recursive blend would run over the full extent of
/// every not-yet-eliminated axis at each level, making the cost grow with grid size
/// instead of being the O(1) this cache exists to provide.
/// [`spline_eval_corner_cached_local`] then recurses on that small buffer like
/// [`spline_eval_nd_cached`] (peels the first axis, recurses on the rest),
/// Hermite-blending pairs of entries instead of solving a fresh 1-D spline per level. Bit
/// `i` of the cache's trailing mask axis is assigned to `grids[i]` (see
/// [`compute_corner_cache`]), and recursion always eliminates the lowest-numbered
/// remaining axis first, so at every level the bit being eliminated is bit 0 of the
/// *current* mask numbering: exactly the even/odd split of that axis. Each level's
/// blended output re-numbers the survivors contiguously (`mask / 2`), matching what the
/// next level expects.
pub(crate) fn spline_eval_corner_cached<T: Float>(
    grids: &[ArrayView1<T>],
    cache: ArrayViewD<T>,
    point: &[T],
) -> T {
    debug_assert_eq!(grids.len(), point.len());
    debug_assert_eq!(cache.ndim(), grids.len() + 1);

    let n_axes = grids.len();
    let n_bits = cache.len_of(Axis(n_axes));

    let mut lower = vec![0usize; n_axes];
    let mut hts = Vec::with_capacity(n_axes);
    for (axis, grid) in grids.iter().enumerate() {
        let l = locate_lower_index(*grid, &point[axis]);
        let h = grid[l + 1] - grid[l];
        let t = (point[axis] - grid[l]) / h;
        lower[axis] = l;
        hts.push((h, t));
    }

    let mut local_shape = vec![2usize; n_axes];
    local_shape.push(n_bits);
    let local = ArrayD::from_shape_fn(IxDyn(&local_shape), |idx| {
        let mut src = vec![0usize; n_axes + 1];
        for axis in 0..n_axes {
            src[axis] = lower[axis] + idx[axis];
        }
        src[n_axes] = idx[n_axes];
        cache[IxDyn(&src)]
    });

    spline_eval_corner_cached_local(local.view(), &hts)
}

/// Recursive Hermite-blend step of [`spline_eval_corner_cached`], operating on the small
/// grid-size-independent local buffer it builds. See that function's doc for the
/// recursion scheme.
fn spline_eval_corner_cached_local<T: Float>(cache: ArrayViewD<T>, hts: &[(T, T)]) -> T {
    if hts.is_empty() {
        return *cache
            .iter()
            .next()
            .expect("corner cache base case has exactly one entry");
    }

    let (h, t) = hts[0];
    let mask_axis = Axis(cache.ndim() - 1);
    let half = cache.len_of(mask_axis) / 2;

    let lower_slice = cache.index_axis(Axis(0), 0);
    let upper_slice = cache.index_axis(Axis(0), 1);
    let slice_mask_axis = Axis(lower_slice.ndim() - 1);

    let mut new_shape = lower_slice.shape()[..lower_slice.ndim() - 1].to_vec();
    new_shape.push(half);
    let mut new_field = ArrayD::<T>::zeros(IxDyn(&new_shape));
    let new_mask_axis = Axis(new_field.ndim() - 1);

    for sub in 0..half {
        let p0 = lower_slice.index_axis(slice_mask_axis, 2 * sub);
        let m0 = lower_slice.index_axis(slice_mask_axis, 2 * sub + 1);
        let p1 = upper_slice.index_axis(slice_mask_axis, 2 * sub);
        let m1 = upper_slice.index_axis(slice_mask_axis, 2 * sub + 1);
        let mut out_slot = new_field.index_axis_mut(new_mask_axis, sub);
        Zip::from(&mut out_slot)
            .and(&p0)
            .and(&m0)
            .and(&p1)
            .and(&m1)
            .for_each(|o, &p0v, &m0v, &p1v, &m1v| {
                *o = hermite_eval_1d(p0v, m0v, p1v, m1v, h, t);
            });
    }

    spline_eval_corner_cached_local(new_field.view(), &hts[1..])
}
