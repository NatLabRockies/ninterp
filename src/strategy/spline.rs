//! Cubic spline algorithms shared across all dimensionalities.

use super::*;

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
/// Used by [`Strategy1D::init`] (stored in `CubicSpline::m_cache`) and
/// [`spline_eval_1d`] (on the fly).
pub(crate) fn compute_m<T: Float>(
    x: ArrayView1<T>,
    y: ArrayView1<T>,
    bc: &CubicBC<T>,
) -> Result<Vec<T>, ValidateError> {
    let n = x.len() - 1;
    let two = T::one() + T::one();
    let six = two + two + two;
    let h: Vec<T> = (0..n).map(|i| x[i + 1] - x[i]).collect();
    let slopes: Vec<T> = (0..n).map(|i| (y[i + 1] - y[i]) / h[i]).collect();
    let u: Vec<T> = (0..n.saturating_sub(1))
        .map(|k| six * (slopes[k + 1] - slopes[k]))
        .collect();

    Ok(match bc {
        CubicBC::NotAKnot => {
            if n < 3 {
                return Err(ValidateError::Other(
                    "CubicSpline with NotAKnot requires at least 4 data points".into(),
                ));
            }
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
            for k in 1..n - 2 {
                rhs.push(u[k]);
            }
            rhs.push(h[n - 2] * u[n - 2]);
            let inner = thomas(&sub, &diag, &sup, &rhs);
            let m0 = ((h[0] + h[1]) * inner[0] - h[0] * inner[1]) / h[1];
            let mn = ((h[n - 2] + h[n - 1]) * inner[n - 2] - h[n - 1] * inner[n - 3]) / h[n - 2];
            let mut m = vec![m0];
            m.extend(inner);
            m.push(mn);
            m
        }
        CubicBC::Natural => {
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
        CubicBC::Clamped { left, right } => {
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
        CubicBC::Periodic => {
            if y[0] != y[n] {
                return Err(ValidateError::Other(
                    "CubicSpline with Periodic BC requires values[0] == values[n]".into(),
                ));
            }
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
    })
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
    bc: &CubicBC<T>,
) -> Result<T, InterpolateError> {
    let m = compute_m(x, y, bc).map_err(|e| InterpolateError::Other(e.to_string()))?;
    Ok(eval_spline_from_m(x, y, ArrayView1::from(&m), point))
}

/// Checks `grid_len` against boundary condition `bc`'s minimum point requirement (e.g.
/// [`CubicBC::NotAKnot`] needs at least 4), ahead of the real work in [`compute_m`].
///
/// Pure, no mutation; used by each dimensionality's `Strategy*D::validate`.
pub(crate) fn validate_bc_min_points<T>(
    bc: &CubicBC<T>,
    grid_len: usize,
    dim: usize,
) -> Result<(), ValidateError> {
    if matches!(bc, CubicBC::NotAKnot) && grid_len < 4 {
        return Err(ValidateError::Other(format!(
            "CubicSpline: dim {dim} has {grid_len} grid points; NotAKnot requires at least 4"
        )));
    }
    Ok(())
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
    bc: &CubicBC<T>,
) -> Result<ArrayD<T>, ValidateError> {
    let last_axis = Axis(values.ndim() - 1);
    let mut out_shape = values.shape().to_vec();
    out_shape[values.ndim() - 1] = inner_grid.len();
    let mut cache = ArrayD::<T>::zeros(IxDyn(&out_shape));
    for (y, mut out_lane) in values
        .lanes(last_axis)
        .into_iter()
        .zip(cache.lanes_mut(last_axis))
    {
        let m = compute_m(inner_grid, y, bc)?;
        out_lane.assign(&ArrayView1::from(&m));
    }
    Ok(cache)
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
    bcs: &[CubicBC<T>],
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

    spline_eval_1d(grids[0], ArrayView1::from(&g), point[0], current_bc)
}
