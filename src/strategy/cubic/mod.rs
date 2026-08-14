//! Cubic interpolation algorithms shared across all dimensionalities, for [`CubicC2`].

use super::*;

mod utils;
pub(crate) use utils::evaluate_spline_corner_cached;

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
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "T: Serialize + Zero",
        deserialize = "T: Deserialize<'de> + Zero"
    ))
)]
pub struct CubicC2<T> {
    /// Boundary conditions, one per dimension or a single entry broadcast to all.
    // Serializes under the key "CubicC2" rather than "boundary_conditions", making the
    // strategy type explicit in the output, consistent with how Linear, Nearest, Step,
    // etc. serialize to their type name.
    #[cfg_attr(feature = "serde", serde(rename = "CubicC2"))]
    pub boundary_conditions: Broadcastable<CubicC2BoundaryConditions<T>>,
    /// Precomputed derivative data, populated by `Strategy1D`/`2D`/`3D`/`ND::init`. Its
    /// shape depends on which of those populated it:
    ///
    /// - `Strategy1D`: cached second derivatives (`M[i] = S''(x_i)`), one [`compute_m`]
    ///   result for the single 1-D pencil, i.e. shape `[n + 1]` for `n` intervals.
    /// - `Strategy2D`/`Strategy3D`/`StrategyND`: the full corner-derivative tensor,
    ///   shaped like the value grid with one extra trailing axis of length `2^N` (`N` =
    ///   the grid's dimensionality): see [`compute_corner_cache`]. Every query is then
    ///   an O(1) lookup, no solving.
    ///
    /// Not included in the serialized form. After deserializing, call the
    /// interpolator's `init_strategy` method (e.g.
    /// [`Interp1D::init_strategy`](crate::interpolator::Interp1D::init_strategy)) to
    /// recompute this before use.
    #[cfg_attr(feature = "serde", serde(skip, default = "empty_cache"))]
    pub(crate) cache: ArrayD<T>,
}

/// Boundary conditions for [`CubicC2`].
///
/// [`Endpoints::lower`](CubicC2BoundaryConditions::Endpoints)/`upper` are independent, so
/// mixing types (e.g. [`NotAKnot`](CubicC2Endpoint::NotAKnot) on one side,
/// [`FirstDerivative`](CubicC2Endpoint::FirstDerivative) on the other) is allowed. The
/// common symmetric cases have shorthand constructors: [`not_a_knot`](Self::not_a_knot),
/// [`first_derivative`](Self::first_derivative), [`second_derivative`](Self::second_derivative);
/// [`CubicC2::natural`] additionally shorthands `second_derivative(0, 0)`.
///
/// Serializes as a bare string for [`NotAKnot`](CubicC2Endpoint::NotAKnot) (both
/// endpoints) and `Periodic`, matching their pre-`Endpoints` representation; a bare
/// `"Natural"` string is accepted and produced as a shorthand for symmetric zero
/// [`SecondDerivative`](CubicC2Endpoint::SecondDerivative) too. Anything else (an
/// explicit value, or asymmetric endpoints) uses the general `{"Endpoints": {"lower":
/// ..., "upper": ...}}` form.
#[derive(Debug, Clone, PartialEq)]
pub enum CubicC2BoundaryConditions<T> {
    /// Condition applied independently at each end of the axis.
    Endpoints {
        /// Condition at the lower endpoint.
        lower: CubicC2Endpoint<T>,
        /// Condition at the upper endpoint.
        upper: CubicC2Endpoint<T>,
    },
    /// First and second derivatives match at both endpoints. By convention
    /// `values[n]` (the last point along this axis) should equal `values[0]`, since
    /// that's what makes the axis periodic. This isn't enforced: `values[n]` is read and
    /// used like any other data point (both to build the periodic system and to
    /// evaluate the last interval), just never compared against `values[0]`.
    ///
    /// Deliberately not validated, even approximately: unlike grid coordinates (usually
    /// synthetic, so float rounding is the only source of nonuniformity), `values` is
    /// often real measured data, where the two ends of a period can legitimately differ
    /// by far more than rounding error for reasons that have nothing to do with a
    /// mistake (sensor noise, distinct samples at each end of the period, etc.). A
    /// tolerance tight enough to catch a genuine mismatch would also reject that valid
    /// data.
    ///
    /// If `values[n] != values[0]`, the spline within `[x[0], x[n]]` itself is still
    /// perfectly smooth (both derivatives matched at the endpoints, same as if the axis
    /// really were periodic) and passes through every supplied value exactly, `values[n]`
    /// included. The mismatch only shows up if the axis is then treated as periodic
    /// beyond its own bounds, e.g. via [`Extrapolate::Wrap`](crate::interpolator::Extrapolate::Wrap):
    /// crossing the seam is a jump in *value* of exactly `values[n] - values[0]`, with
    /// slope and curvature matching continuously on both sides of it.
    Periodic,
}

/// A single endpoint's condition, for [`CubicC2BoundaryConditions::Endpoints`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub enum CubicC2Endpoint<T> {
    /// C³ continuity at the second (from this end) knot; no extra input required.
    /// Generally gives better accuracy than natural (a zero
    /// [`SecondDerivative`](Self::SecondDerivative)) for smooth functions. Requires at
    /// least 3 grid points along this axis (4 if both endpoints are `NotAKnot`).
    NotAKnot,
    /// Specified first derivative at this endpoint ("clamped").
    FirstDerivative(T),
    /// Specified second derivative at this endpoint. Zero, the classic "natural"
    /// condition, has a shorthand: [`CubicC2::natural`].
    SecondDerivative(T),
}

impl<T> CubicC2BoundaryConditions<T> {
    /// Not-a-knot at both ends. Requires at least 4 data points per dimension.
    pub fn not_a_knot() -> Self {
        Self::Endpoints {
            lower: CubicC2Endpoint::NotAKnot,
            upper: CubicC2Endpoint::NotAKnot,
        }
    }

    /// Specified first derivative at both endpoints.
    pub fn first_derivative(lower: T, upper: T) -> Self {
        Self::Endpoints {
            lower: CubicC2Endpoint::FirstDerivative(lower),
            upper: CubicC2Endpoint::FirstDerivative(upper),
        }
    }

    /// Specified second derivative at both endpoints.
    pub fn second_derivative(lower: T, upper: T) -> Self {
        Self::Endpoints {
            lower: CubicC2Endpoint::SecondDerivative(lower),
            upper: CubicC2Endpoint::SecondDerivative(upper),
        }
    }
}

/// Placeholder for [`CubicC2::cache`] before [`Strategy1D::init`]/`2D`/`3D`/`ND`
/// populates it. No `T: Clone + Zero` bound needed (unlike `ArrayD::zeros`), so the
/// constructors below stay unconstrained.
fn empty_cache<T>() -> ArrayD<T> {
    ArrayD::from_shape_vec(IxDyn(&[0]), Vec::new()).expect("empty shape matches empty vec")
}

impl<T> From<CubicC2BoundaryConditions<T>> for CubicC2<T> {
    /// Broadcasts `bc` to all dimensions.
    ///
    /// Use [`CubicC2::not_a_knot`], [`natural`](CubicC2::natural),
    /// [`clamped`](CubicC2::clamped), or [`periodic`](CubicC2::periodic) instead when the
    /// desired condition is known at the call site; this is for a `CubicC2BoundaryConditions`
    /// value obtained generically (e.g. from runtime config), without matching on it first.
    fn from(bc: CubicC2BoundaryConditions<T>) -> Self {
        Self {
            boundary_conditions: Broadcastable::Broadcast(bc),
            cache: empty_cache(),
        }
    }
}

impl<T> CubicC2<T> {
    /// Create a cubic spline with a distinct boundary condition per grid dimension.
    ///
    /// Use [`not_a_knot`](Self::not_a_knot), [`natural`](Self::natural),
    /// [`clamped`](Self::clamped), or [`periodic`](Self::periodic) instead when every
    /// dimension shares the same condition.
    pub fn new(boundary_conditions: Vec<CubicC2BoundaryConditions<T>>) -> Self {
        Self {
            boundary_conditions: Broadcastable::Each(boundary_conditions),
            cache: empty_cache(),
        }
    }

    /// Create a cubic spline with not-a-knot boundary conditions.
    /// Requires at least 4 data points per dimension.
    pub fn not_a_knot() -> Self {
        Self {
            boundary_conditions: Broadcastable::Broadcast(CubicC2BoundaryConditions::not_a_knot()),
            cache: empty_cache(),
        }
    }

    /// Create a cubic spline with natural (zero second derivative at endpoints) BCs.
    pub fn natural() -> Self
    where
        T: Zero,
    {
        Self {
            boundary_conditions: Broadcastable::Broadcast(
                CubicC2BoundaryConditions::second_derivative(T::zero(), T::zero()),
            ),
            cache: empty_cache(),
        }
    }

    /// Create a cubic spline with specified first derivatives at both endpoints.
    pub fn clamped(lower: T, upper: T) -> Self {
        Self {
            boundary_conditions: Broadcastable::Broadcast(
                CubicC2BoundaryConditions::first_derivative(lower, upper),
            ),
            cache: empty_cache(),
        }
    }

    /// Create a cubic spline with periodic boundary conditions. By convention
    /// `values[n]` (the last point along each periodic axis) should equal
    /// `values[0]`, though this isn't enforced.
    pub fn periodic() -> Self {
        Self {
            boundary_conditions: Broadcastable::Broadcast(CubicC2BoundaryConditions::Periodic),
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
/// Used by [`compute_m_cache`] (called from `Strategy1D::init`, stored in `CubicC2::cache`)
/// and [`corner_cache_axis_pass`] (called from [`compute_corner_cache`]).
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
        CubicC2BoundaryConditions::Endpoints { lower, upper } => {
            // Each endpoint contributes either a direct boundary row (`Some`, referencing
            // only its own knot and its immediate interior neighbor) or, for `NotAKnot`,
            // `None`: that side's M is eliminated via the third-derivative-continuity
            // relation at knot 1 (or n-1), folded into the interior equation there instead
            // of appearing as its own unknown. `lower_row`/`upper_row` give
            // `(diag, off_diagonal, rhs)` for the direct case.
            let lower_row = match lower {
                CubicC2Endpoint::NotAKnot => None,
                CubicC2Endpoint::SecondDerivative(v) => Some((T::one(), T::zero(), *v)),
                CubicC2Endpoint::FirstDerivative(v) => {
                    Some((two * h[0], h[0], six * (slopes[0] - *v)))
                }
            };
            let upper_row = match upper {
                CubicC2Endpoint::NotAKnot => None,
                CubicC2Endpoint::SecondDerivative(v) => Some((T::one(), T::zero(), *v)),
                CubicC2Endpoint::FirstDerivative(v) => {
                    Some((two * h[n - 1], h[n - 1], six * (*v - slopes[n - 1])))
                }
            };

            // `validate_bc_min_points` guarantees `count >= 2` (grid_len >= 3 with one
            // `NotAKnot` side, >= 4 with both), so the `j == 0` and `j == count - 1`
            // branches below are always distinct rows.
            let start = if lower_row.is_some() { 0 } else { 1 };
            let end = if upper_row.is_some() { n } else { n - 1 };
            let count = end - start + 1;

            let mut sub = Vec::with_capacity(count - 1);
            let mut diag = Vec::with_capacity(count);
            let mut sup = Vec::with_capacity(count - 1);
            let mut rhs = Vec::with_capacity(count);
            for j in 0..count {
                let i = start + j;
                if j == 0 {
                    if let Some((diag_v, sup_v, rhs_v)) = lower_row {
                        diag.push(diag_v);
                        sup.push(sup_v);
                        rhs.push(rhs_v);
                    } else {
                        // Folds the eliminated M[0] into knot 1's interior equation.
                        diag.push((h[0] + h[1]) * (h[0] + two * h[1]));
                        sup.push(h[1] * h[1] - h[0] * h[0]);
                        rhs.push(h[1] * u[0]);
                    }
                } else if j == count - 1 {
                    if let Some((diag_v, sub_v, rhs_v)) = upper_row {
                        sub.push(sub_v);
                        diag.push(diag_v);
                        rhs.push(rhs_v);
                    } else {
                        // Mirrors the lower fold, at knot n-1 eliminating M[n].
                        sub.push(h[n - 2] * h[n - 2] - h[n - 1] * h[n - 1]);
                        diag.push((h[n - 2] + h[n - 1]) * (two * h[n - 2] + h[n - 1]));
                        rhs.push(h[n - 2] * u[n - 2]);
                    }
                } else {
                    sub.push(h[i - 1]);
                    diag.push(two * (h[i - 1] + h[i]));
                    sup.push(h[i]);
                    rhs.push(u[i - 1]);
                }
            }
            let inner = thomas(&sub, &diag, &sup, &rhs);

            let mut m = Vec::with_capacity(n + 1);
            if lower_row.is_none() {
                m.push(((h[0] + h[1]) * inner[0] - h[0] * inner[1]) / h[1]);
            }
            m.extend_from_slice(&inner);
            if upper_row.is_none() {
                let last = inner.len() - 1;
                m.push(
                    ((h[n - 2] + h[n - 1]) * inner[last] - h[n - 1] * inner[last - 1]) / h[n - 2],
                );
            }
            m
        }
        CubicC2BoundaryConditions::Periodic => {
            // `y[n]` is read below, via `slopes[n - 1]` in the `rhs[0]` line, and
            // `evaluate_spline_from_m` also reads `y[n]` directly when evaluating the last
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
pub(crate) fn evaluate_spline_from_m<T: Float>(
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

/// Checks `grid_len` against boundary condition `bc`'s minimum point requirement (e.g.
/// [`CubicC2Endpoint::NotAKnot`] needs at least 3 grid points on its own side, 4 if both
/// endpoints are `NotAKnot`), ahead of the real work in [`compute_m`].
///
/// Pure, no mutation; used by each dimensionality's `Strategy*D::validate`.
pub(crate) fn validate_bc_min_points<T>(
    bc: &CubicC2BoundaryConditions<T>,
    grid_len: usize,
    dim: usize,
) -> Result<(), ValidateError> {
    let CubicC2BoundaryConditions::Endpoints { lower, upper } = bc else {
        return Ok(());
    };
    let min = match (
        matches!(lower, CubicC2Endpoint::NotAKnot),
        matches!(upper, CubicC2Endpoint::NotAKnot),
    ) {
        (true, true) => 4,
        (true, false) | (false, true) => 3,
        (false, false) => return Ok(()),
    };
    if grid_len < min {
        return Err(ValidateError::Other(format!(
            "CubicC2: dim {dim} has {grid_len} grid points; NotAKnot requires at least {min}"
        )));
    }
    Ok(())
}

/// Computes and caches `M[0..=n]` (`CubicC2::cache`) for [`Strategy1D::init`], so
/// [`evaluate_spline_1d_cached`] can look them up in O(1) instead of re-solving on every
/// `interpolate` call.
pub(crate) fn compute_m_cache<T: Float>(
    x: ArrayView1<T>,
    y: ArrayView1<T>,
    bc: &CubicC2BoundaryConditions<T>,
) -> ArrayD<T> {
    let m = compute_m(x, y, bc);
    ArrayD::from_shape_vec(IxDyn(&[m.len()]), m)
        .expect("compute_m's output length matches its own shape")
}

/// Evaluates [`Strategy1D`]'s cached spline (`m_cache`, from [`compute_m_cache`]) at `point`.
pub(crate) fn evaluate_spline_1d_cached<T: Float>(
    x: ArrayView1<T>,
    y: ArrayView1<T>,
    m_cache: ArrayViewD<T>,
    point: T,
) -> Result<T, InterpolateError> {
    let m = m_cache.into_dimensionality::<Ix1>().map_err(|_| {
        InterpolateError::Other(
            "internal: non-1-D m_cache, Strategy1D::cache invariant broken".into(),
        )
    })?;
    Ok(evaluate_spline_from_m(x, y, m, point))
}

/// Closed-form first derivative `S'(x_i)` at every knot, from an already-solved moment
/// vector `m` (no extra solve). The companion to [`compute_m`], used to build
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
/// the same shape as `field`, for [`compute_corner_cache`].
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
/// patch in O(1) via [`evaluate_spline_corner_cached`].
///
/// Returns an array shaped like `values`, with one extra trailing axis of length `2^N`
/// (`N = grids.len()`). Index `k` in that axis is a bitmask over the `N` grid axes: bit
/// `i` set means "this entry is `d/dx_i` of the value", so `k == 0` is the raw value and
/// `k == 2^N - 1` is the full mixed partial.
///
/// Built by processing axes `0..N` in order: for every mask not yet including axis `i`'s
/// bit, splines the corresponding field along axis `i` ([`corner_cache_axis_pass`]) to
/// populate the entry with that bit added. Boundary condition per axis is
/// `bcs[axis]` when splining the raw values (`mask == 0`, this axis's own
/// first-derivative pass). For every other `mask` (a cross-derivative pass over an
/// already-differentiated field, not the original data), each endpoint's
/// [`FirstDerivative`](CubicC2Endpoint::FirstDerivative)/[`SecondDerivative`](CubicC2Endpoint::SecondDerivative)
/// falls back to the same-order condition with a zero value: a `FirstDerivative`/
/// `SecondDerivative` endpoint fixes the same scalar derivative at every point along that
/// boundary, so differentiating that (constant-along-the-boundary) field with respect to
/// any other axis must itself be zero at that endpoint, at the same order. `NotAKnot`
/// endpoints pass through unchanged (they carry no value to homogenize). Verified
/// empirically: with non-separable data, only these homogeneous fallbacks make this
/// cached path agree (to float precision) with `StrategyND`'s independent recursive
/// solve; leaving a nonzero endpoint value unchanged in a cross-derivative pass does not.
pub(crate) fn compute_corner_cache<T: Float>(
    grids: &[ArrayView1<T>],
    values: ArrayViewD<T>,
    bcs: &Broadcastable<CubicC2BoundaryConditions<T>>,
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
        let bc_axis = &bcs[axis];
        let cross_endpoint = |e: &CubicC2Endpoint<T>| match e {
            CubicC2Endpoint::FirstDerivative(_) => CubicC2Endpoint::FirstDerivative(T::zero()),
            CubicC2Endpoint::SecondDerivative(_) => CubicC2Endpoint::SecondDerivative(T::zero()),
            other @ CubicC2Endpoint::NotAKnot => other.clone(),
        };
        let cross_bc = match bc_axis {
            CubicC2BoundaryConditions::Endpoints { lower, upper } => {
                CubicC2BoundaryConditions::Endpoints {
                    lower: cross_endpoint(lower),
                    upper: cross_endpoint(upper),
                }
            }
            CubicC2BoundaryConditions::Periodic => CubicC2BoundaryConditions::Periodic,
        };
        for mask in 0..bit {
            let bc = if mask == 0 { bc_axis } else { &cross_bc };
            let field = cache.index_axis(mask_axis, mask).to_owned();
            let deriv = corner_cache_axis_pass(*grid, field.view(), axis, bc);
            cache.index_axis_mut(mask_axis, mask | bit).assign(&deriv);
        }
    }
    cache
}

#[cfg(feature = "serde")]
mod cubic_serde {
    use super::*;
    use serde::Deserializer;

    /// Bare-string wire form for the two `CubicC2BoundaryConditions::Endpoints` cases
    /// that carry no distinguishing per-endpoint data (symmetric `NotAKnot`, symmetric
    /// zero `SecondDerivative`, i.e. "natural"), plus `Periodic`. Anything else uses the
    /// general `Endpoints` shape instead.
    #[derive(Deserialize, Serialize)]
    enum BcName {
        NotAKnot,
        Natural,
        Periodic,
    }

    /// General `Endpoints` shape, owned form for [`Deserialize`].
    #[derive(Deserialize)]
    struct EndpointsOwned<T> {
        lower: CubicC2Endpoint<T>,
        upper: CubicC2Endpoint<T>,
    }

    /// General `Endpoints` shape, borrowed form for [`Serialize`]: avoids requiring `T:
    /// Clone` just to serialize.
    #[derive(Serialize)]
    struct EndpointsRef<'a, T> {
        lower: &'a CubicC2Endpoint<T>,
        upper: &'a CubicC2Endpoint<T>,
    }

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum BcWireDe<T> {
        Named(BcName),
        Endpoints {
            #[serde(rename = "Endpoints")]
            endpoints: EndpointsOwned<T>,
        },
    }

    #[derive(Serialize)]
    #[serde(untagged)]
    enum BcWireSer<'a, T> {
        Named(BcName),
        Endpoints {
            #[serde(rename = "Endpoints")]
            endpoints: EndpointsRef<'a, T>,
        },
    }

    impl<T: Serialize + Zero> Serialize for CubicC2BoundaryConditions<T> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            match self {
                Self::Periodic => BcWireSer::<T>::Named(BcName::Periodic).serialize(serializer),
                Self::Endpoints {
                    lower: CubicC2Endpoint::NotAKnot,
                    upper: CubicC2Endpoint::NotAKnot,
                } => BcWireSer::<T>::Named(BcName::NotAKnot).serialize(serializer),
                Self::Endpoints {
                    lower: CubicC2Endpoint::SecondDerivative(lower),
                    upper: CubicC2Endpoint::SecondDerivative(upper),
                } if lower.is_zero() && upper.is_zero() => {
                    BcWireSer::<T>::Named(BcName::Natural).serialize(serializer)
                }
                Self::Endpoints { lower, upper } => BcWireSer::Endpoints {
                    endpoints: EndpointsRef { lower, upper },
                }
                .serialize(serializer),
            }
        }
    }

    impl<'de, T: Deserialize<'de> + Zero> Deserialize<'de> for CubicC2BoundaryConditions<T> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            Ok(match BcWireDe::<T>::deserialize(deserializer)? {
                BcWireDe::Named(BcName::NotAKnot) => CubicC2BoundaryConditions::not_a_knot(),
                BcWireDe::Named(BcName::Natural) => Self::Endpoints {
                    lower: CubicC2Endpoint::SecondDerivative(T::zero()),
                    upper: CubicC2Endpoint::SecondDerivative(T::zero()),
                },
                BcWireDe::Named(BcName::Periodic) => Self::Periodic,
                BcWireDe::Endpoints { endpoints } => Self::Endpoints {
                    lower: endpoints.lower,
                    upper: endpoints.upper,
                },
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_bc_matches_named_constructors() {
        assert_eq!(
            CubicC2::from(CubicC2BoundaryConditions::<f64>::not_a_knot()),
            CubicC2::not_a_knot()
        );
        assert_eq!(
            CubicC2::from(CubicC2BoundaryConditions::<f64>::second_derivative(
                0.0, 0.0
            )),
            CubicC2::natural()
        );
        assert_eq!(
            CubicC2::from(CubicC2BoundaryConditions::first_derivative(1.0, 2.0)),
            CubicC2::clamped(1.0, 2.0)
        );
        assert_eq!(
            CubicC2::from(CubicC2BoundaryConditions::<f64>::Periodic),
            CubicC2::periodic()
        );
    }

    /// Builds the `(n+1) x (n+1)` moment system directly from `bc`'s mathematical
    /// definition, independent of `compute_m`'s reduced/eliminated form, to cross-check
    /// it on asymmetric endpoint combinations that this crate's other tests (all
    /// symmetric until now) wouldn't have caught a mismatched row for.
    fn dense_m_system(
        x: &[f64],
        y: &[f64],
        bc: &CubicC2BoundaryConditions<f64>,
    ) -> (Vec<Vec<f64>>, Vec<f64>) {
        let n = x.len() - 1;
        let h: Vec<f64> = (0..n).map(|i| x[i + 1] - x[i]).collect();
        let slopes: Vec<f64> = (0..n).map(|i| (y[i + 1] - y[i]) / h[i]).collect();
        let mut a = vec![vec![0.0; n + 1]; n + 1];
        let mut b = vec![0.0; n + 1];

        for i in 1..n {
            a[i][i - 1] = h[i - 1];
            a[i][i] = 2.0 * (h[i - 1] + h[i]);
            a[i][i + 1] = h[i];
            b[i] = 6.0 * (slopes[i] - slopes[i - 1]);
        }

        let CubicC2BoundaryConditions::Endpoints { lower, upper } = bc else {
            panic!("dense_m_system only supports Endpoints for this test");
        };
        match lower {
            CubicC2Endpoint::NotAKnot => {
                a[0][0] = h[1];
                a[0][1] = -(h[0] + h[1]);
                a[0][2] = h[0];
            }
            CubicC2Endpoint::SecondDerivative(v) => {
                a[0][0] = 1.0;
                b[0] = *v;
            }
            CubicC2Endpoint::FirstDerivative(v) => {
                a[0][0] = 2.0 * h[0];
                a[0][1] = h[0];
                b[0] = 6.0 * (slopes[0] - v);
            }
        }
        match upper {
            CubicC2Endpoint::NotAKnot => {
                a[n][n] = h[n - 2];
                a[n][n - 1] = -(h[n - 2] + h[n - 1]);
                a[n][n - 2] = h[n - 1];
            }
            CubicC2Endpoint::SecondDerivative(v) => {
                a[n][n] = 1.0;
                b[n] = *v;
            }
            CubicC2Endpoint::FirstDerivative(v) => {
                a[n][n] = 2.0 * h[n - 1];
                a[n][n - 1] = h[n - 1];
                b[n] = 6.0 * (v - slopes[n - 1]);
            }
        }
        (a, b)
    }

    /// Plain Gaussian elimination with partial pivoting, independent of `thomas`/
    /// `cyclic_thomas`, for cross-checking their output.
    fn dense_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
        let n = b.len();
        for col in 0..n {
            let pivot = (col..n)
                .max_by(|&r1, &r2| a[r1][col].abs().partial_cmp(&a[r2][col].abs()).unwrap())
                .unwrap();
            a.swap(col, pivot);
            b.swap(col, pivot);
            for row in (col + 1)..n {
                let factor = a[row][col] / a[col][col];
                let pivot_row = a[col].clone();
                for (c, pivot_val) in pivot_row.iter().enumerate().skip(col) {
                    a[row][c] -= factor * pivot_val;
                }
                b[row] -= factor * b[col];
            }
        }
        let mut x = vec![0.0; n];
        for row in (0..n).rev() {
            let sum: f64 = (row + 1..n).map(|c| a[row][c] * x[c]).sum();
            x[row] = (b[row] - sum) / a[row][row];
        }
        x
    }

    #[test]
    fn compute_m_matches_dense_solve_for_mixed_endpoints() {
        // Non-uniform grid, non-polynomial data: exercises the linear algebra for real,
        // rather than something any BC would reproduce exactly regardless of correctness.
        let x = [0.0, 0.4, 1.1, 1.8, 2.9, 4.0];
        let y = [0.5, 1.2, 0.3, 2.1, 1.0, 3.3];
        let xv = ArrayView1::from(&x);
        let yv = ArrayView1::from(&y);

        let endpoints = [
            CubicC2Endpoint::NotAKnot,
            CubicC2Endpoint::SecondDerivative(0.0),
            CubicC2Endpoint::SecondDerivative(2.5),
            CubicC2Endpoint::FirstDerivative(0.0),
            CubicC2Endpoint::FirstDerivative(-1.5),
        ];
        for lower in &endpoints {
            for upper in &endpoints {
                let bc = CubicC2BoundaryConditions::Endpoints {
                    lower: lower.clone(),
                    upper: upper.clone(),
                };
                let got = compute_m(xv, yv, &bc);
                let (a, b) = dense_m_system(&x, &y, &bc);
                let expected = dense_solve(a, b);
                for (g, e) in got.iter().zip(expected.iter()) {
                    assert_approx_eq!(*g, *e, 1e-9);
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_serde() {
        // The whole-axis cases collapse to a bare string, keyed by strategy name like
        // every other built-in strategy.
        assert_eq!(
            serde_json::to_string(&CubicC2::<f64>::not_a_knot()).unwrap(),
            r#"{"CubicC2":"NotAKnot"}"#
        );
        assert_eq!(
            serde_json::to_string(&CubicC2::<f64>::natural()).unwrap(),
            r#"{"CubicC2":"Natural"}"#
        );
        assert_eq!(
            serde_json::to_string(&CubicC2::<f64>::periodic()).unwrap(),
            r#"{"CubicC2":"Periodic"}"#
        );
        // Anything carrying an explicit value, or mixing endpoint types, uses the
        // general `Endpoints` form instead.
        assert_eq!(
            serde_json::to_string(&CubicC2::<f64>::clamped(1.0, 2.0)).unwrap(),
            r#"{"CubicC2":{"Endpoints":{"lower":{"FirstDerivative":1.0},"upper":{"FirstDerivative":2.0}}}}"#
        );
        let mixed = CubicC2::<f64>::from(CubicC2BoundaryConditions::Endpoints {
            lower: CubicC2Endpoint::NotAKnot,
            upper: CubicC2Endpoint::FirstDerivative(3.0),
        });
        assert_eq!(
            serde_json::to_string(&mixed).unwrap(),
            r#"{"CubicC2":{"Endpoints":{"lower":"NotAKnot","upper":{"FirstDerivative":3.0}}}}"#
        );
        assert_eq!(
            serde_json::from_str::<CubicC2<f64>>(&serde_json::to_string(&mixed).unwrap()).unwrap(),
            mixed
        );

        // Bare "NotAKnot"/"Natural" round-trip; the fully expanded `Endpoints` form is
        // also accepted on read for both, even though only the bare form is ever written.
        assert_eq!(
            serde_json::from_str::<CubicC2<f64>>(r#"{"CubicC2":"NotAKnot"}"#).unwrap(),
            CubicC2::<f64>::not_a_knot()
        );
        assert_eq!(
            serde_json::from_str::<CubicC2<f64>>(r#"{"CubicC2":"Natural"}"#).unwrap(),
            CubicC2::<f64>::natural()
        );
        assert_eq!(
            serde_json::from_str::<CubicC2<f64>>(
                r#"{"CubicC2":{"Endpoints":{"lower":"NotAKnot","upper":"NotAKnot"}}}"#
            )
            .unwrap(),
            CubicC2::<f64>::not_a_knot()
        );
        assert_eq!(
            serde_json::from_str::<CubicC2<f64>>(
                r#"{"CubicC2":{"Endpoints":{"lower":{"SecondDerivative":0.0},"upper":{"SecondDerivative":0.0}}}}"#
            )
            .unwrap(),
            CubicC2::<f64>::natural()
        );
    }
}
