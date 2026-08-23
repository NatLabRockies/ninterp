//! Corner-derivative-tensor evaluation, shared by [`c1`](super::c1) and
//! [`c2`](super::c2): BC-agnostic, works the same regardless of how the tensor's
//! entries were populated (a solve for `CubicC2`, finite differences for `CubicC1`).

use super::*;
use ndarray::Zip;

/// Standard cubic Hermite basis blend of endpoint values `p0`/`p1` and their derivatives
/// `m0`/`m1` (actual derivatives, not yet scaled by the interval width `h`) at
/// fractional position `t` in `[0, 1]` between them.
pub(crate) fn evaluate_hermite_1d<T: Float>(p0: T, m0: T, p1: T, m1: T, h: T, t: T) -> T {
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
/// [`compute_corner_cache`] in O(1), no solving. Used by [`Strategy2D`]/[`Strategy3D`]/
/// [`StrategyND`]'s `CubicC2::interpolate`.
///
/// First localizes every axis down to just its query cell's two bounding grid indices,
/// into an owned `2 x 2 x ... x 2 x 2^N` buffer (`N = grids.len()`) that's independent of
/// grid size. Without this step the recursive blend would run over the full extent of
/// every not-yet-eliminated axis at each level, making the cost grow with grid size
/// instead of being the O(1) this cache exists to provide.
/// [`evaluate_spline_corner_cached_local`] then recurses on that small buffer, peeling the
/// first axis and recursing on the rest, Hermite-blending pairs of entries instead of
/// solving a fresh 1-D spline per level. Bit
/// `i` of the cache's trailing mask axis is assigned to `grids[i]` (see
/// [`compute_corner_cache`]), and recursion always eliminates the lowest-numbered
/// remaining axis first, so at every level the bit being eliminated is bit 0 of the
/// *current* mask numbering: exactly the even/odd split of that axis. Each level's
/// blended output re-numbers the survivors contiguously (`mask / 2`), matching what the
/// next level expects.
pub(crate) fn evaluate_spline_corner_cached<T: Float>(
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

    evaluate_spline_corner_cached_local(local.view(), &hts)
}

/// Recursive Hermite-blend step of [`evaluate_spline_corner_cached`], operating on the small
/// grid-size-independent local buffer it builds. See that function's doc for the
/// recursion scheme.
fn evaluate_spline_corner_cached_local<T: Float>(cache: ArrayViewD<T>, hts: &[(T, T)]) -> T {
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
                *o = evaluate_hermite_1d(p0v, m0v, p1v, m1v, h, t);
            });
    }

    evaluate_spline_corner_cached_local(new_field.view(), &hts[1..])
}
