use super::*;
use strategy::*;

impl<D> StrategyND<D> for Linear
where
    D: Data + RawDataClone + Clone,
    D::Elem: Num + PartialOrd + Copy + Debug,
{
    fn interpolate(
        &self,
        data: &InterpDataND<D>,
        point: &[D::Elem],
    ) -> Result<D::Elem, InterpolateError> {
        // Dimensionality
        let mut n = data.values.ndim();

        // Point can share up to N values of a grid point, which reduces the problem dimensionality
        // i.e. the point shares one of three values of a 3-D grid point, then the interpolation becomes 2-D at that slice
        // or   if the point shares two of three values of a 3-D grid point, then the interpolation becomes 1-D
        let mut point = point.to_owned();
        let mut grid: Vec<_> = data.grid.iter().map(|arr| arr.view()).collect();
        let mut values_view = data.values.view();
        for dim in (0..n).rev() {
            // Binary search for an exact match: find_nearest_index returns the lower bracket,
            // so the point can only be exactly equal to grid[lower] or grid[lower+1].
            let lower = if &point[dim] < grid[dim].first().unwrap() {
                0
            } else if &point[dim] > grid[dim].last().unwrap() {
                grid[dim].len() - 2
            } else {
                find_nearest_index(grid[dim].view(), &point[dim])
            };
            let pos = exact_index(grid[dim].view(), lower, &point[dim]);
            if let Some(pos) = pos {
                point.remove(dim);
                grid.remove(dim);
                values_view.index_axis_inplace(Axis(dim), pos);
            }
        }
        if values_view.len() == 1 {
            // Supplied point is coincident with a grid point, so just return the value
            return Ok(values_view.first().copied().unwrap());
        }
        // Simplified dimensionality
        n = values_view.ndim();

        // Extract the lower and upper indices for each dimension,
        // as well as the fraction of how far the supplied point is between the surrounding grid points
        let mut lower_idxs = Vec::with_capacity(n);
        let mut interp_diffs = Vec::with_capacity(n);
        for dim in 0..n {
            // Extrapolation is checked previously in Interpolator::interpolate,
            // meaning by now, point is within grid bounds or extrapolation is enabled
            let lower_idx = if &point[dim] < grid[dim].first().unwrap() {
                0
            } else if &point[dim] > grid[dim].last().unwrap() {
                grid[dim].len() - 2
            } else {
                find_nearest_index(grid[dim].view(), &point[dim])
            };
            let interp_diff = (point[dim] - grid[dim][lower_idx])
                / (grid[dim][lower_idx + 1] - grid[dim][lower_idx]);
            lower_idxs.push(lower_idx);
            interp_diffs.push(interp_diff);
        }
        // Fill all 2^n corner values into a flat array indexed by bitmask.
        // Bit (n-1-d) of the mask = 1 selects the upper index in dimension d.
        // This layout supports an in-place butterfly reduction with no coordinate permutation tables.
        let size = 1usize << n;
        let mut vals = vec![D::Elem::zero(); size];
        let mut idx = vec![0usize; n];
        for mask in 0..size {
            for d in 0..n {
                idx[d] = lower_idxs[d] + ((mask >> (n - 1 - d)) & 1);
            }
            vals[mask] = values_view[idx.as_slice()];
        }

        // Butterfly reduction: one pass per dimension.
        // After pass d, vals[0..2^(n-d-1)] holds the result with dimensions 0..=d blended.
        for d in 0..n {
            let half = 1 << (n - 1 - d);
            for i in 0..half {
                vals[i] =
                    vals[i] * (D::Elem::one() - interp_diffs[d]) + vals[i + half] * interp_diffs[d];
            }
        }

        Ok(vals[0])
    }

    /// Returns `true`.
    fn allow_extrapolate(&self) -> bool {
        true
    }
}

impl<D> StrategyND<D> for Nearest
where
    D: Data + RawDataClone + Clone,
    D::Elem: Num + PartialOrd + Copy + Debug,
{
    fn interpolate(
        &self,
        data: &InterpDataND<D>,
        point: &[D::Elem],
    ) -> Result<D::Elem, InterpolateError> {
        let n = data.values.ndim();
        // Nearest-neighbor on a rectilinear grid factorizes: select the nearest index
        // independently per dimension, then do a single lookup. No corner extraction or
        // dimensionality reduction needed — the distance comparison handles exact matches correctly.
        let mut idx = vec![0usize; n];
        for dim in 0..n {
            let lower_idx = find_nearest_index(data.grid[dim].view(), &point[dim]);
            idx[dim] = if point[dim] - data.grid[dim][lower_idx]
                < data.grid[dim][lower_idx + 1] - point[dim]
            {
                lower_idx
            } else {
                lower_idx + 1
            };
        }
        Ok(data.values.view()[idx.as_slice()])
    }

    /// Returns `false`.
    fn allow_extrapolate(&self) -> bool {
        false
    }
}
