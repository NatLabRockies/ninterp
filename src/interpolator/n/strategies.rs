use super::*;
use strategy::*;

impl<D> StrategyND<D> for Linear
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    fn interpolate(
        &self,
        data: &InterpDataNDBase<D>,
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
            // Skip empty grid dimensions (e.g. the 0-D multilinear case uses an empty grid).
            // The original iter().position() returned None on empty grids without touching point[dim];
            // the binary search path would panic on first().unwrap(), so we guard it here.
            if grid[dim].is_empty() {
                continue;
            }
            let lower = locate_lower_index(grid[dim].view(), &point[dim]);
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
            let lower_idx = locate_lower_index(grid[dim].view(), &point[dim]);
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
        for (mask, val) in vals.iter_mut().enumerate() {
            for d in 0..n {
                idx[d] = lower_idxs[d] + ((mask >> (n - 1 - d)) & 1);
            }
            *val = values_view[idx.as_slice()];
        }

        // Butterfly reduction: one pass per dimension.
        // After pass d, vals[0..2^(n-d-1)] holds the result with dimensions 0..=d blended.
        for (d, diff) in interp_diffs.iter().enumerate() {
            let half = 1 << (n - 1 - d);
            for i in 0..half {
                vals[i] = vals[i] * (D::Elem::one() - *diff) + vals[i + half] * *diff;
            }
        }

        Ok(vals[0])
    }

    /// Returns `true`.
    fn allow_extrapolate(&self) -> bool {
        true
    }
}

impl<D> StrategyND<D> for LinearUniform
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    /// Ensures grid uniformity in all dimensions
    fn validate(&self, data: &InterpDataNDBase<D>) -> Result<(), ValidateError> {
        for (dim, grid) in data.grid.iter().enumerate() {
            validate_uniform_grid_epsilon(grid.view(), dim, None)?;
        }
        Ok(())
    }

    fn interpolate(
        &self,
        data: &InterpDataNDBase<D>,
        point: &[D::Elem],
    ) -> Result<D::Elem, InterpolateError> {
        let n = data.values.ndim();
        let mut lower_idxs = Vec::with_capacity(n);
        let mut interp_diffs = Vec::with_capacity(n);
        for (grid_dim, &point_dim) in data.grid.iter().zip(point.iter()) {
            let step = grid_dim[1] - grid_dim[0];
            let lower_idx =
                locate_lower_index_uniform(grid_dim[0], step, grid_dim.len(), point_dim);
            let diff = (point_dim - grid_dim[lower_idx]) / step;
            lower_idxs.push(lower_idx);
            interp_diffs.push(diff);
        }
        // Same bitmask/butterfly reduction as Linear ND
        let size = 1usize << n;
        let mut vals = vec![D::Elem::zero(); size];
        let mut idx = vec![0usize; n];
        for (mask, val) in vals.iter_mut().enumerate() {
            for d in 0..n {
                idx[d] = lower_idxs[d] + ((mask >> (n - 1 - d)) & 1);
            }
            *val = data.values.view()[idx.as_slice()];
        }
        for (d, diff) in interp_diffs.iter().enumerate() {
            let half = 1 << (n - 1 - d);
            for i in 0..half {
                vals[i] = vals[i] * (D::Elem::one() - *diff) + vals[i + half] * *diff;
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
    D::Elem: Sub<Output = D::Elem> + PartialOrd + Copy + Debug,
{
    fn interpolate(
        &self,
        data: &InterpDataNDBase<D>,
        point: &[D::Elem],
    ) -> Result<D::Elem, InterpolateError> {
        let n = data.values.ndim();
        // Nearest-neighbor on a rectilinear grid factorizes: select the nearest index
        // independently per dimension, then do a single lookup. No corner extraction or
        // dimensionality reduction needed; the distance comparison handles exact matches correctly.
        let mut idx = vec![0usize; n];
        for dim in 0..n {
            let lower_idx = locate_lower_index(data.grid[dim].view(), &point[dim]);
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

impl<D> StrategyND<D> for Step
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialOrd + Copy + Debug,
{
    /// Ensures the number of provided step directions matches the dimensionality of the interpolator
    fn validate(&self, data: &InterpDataNDBase<D>) -> Result<(), ValidateError> {
        self.directions
            .validate_len(data.values.ndim(), "Step", "directions")
    }

    fn interpolate(
        &self,
        data: &InterpDataNDBase<D>,
        point: &[D::Elem],
    ) -> Result<D::Elem, InterpolateError> {
        let n = data.values.ndim();
        let mut idx = vec![0usize; n];
        for dim in 0..n {
            idx[dim] = locate_step_index(self.directions[dim], data.grid[dim].view(), &point[dim]);
        }
        Ok(data.values.view()[idx.as_slice()])
    }

    /// Returns `false`.
    fn allow_extrapolate(&self) -> bool {
        false
    }
}

impl<D> StrategyND<D> for CubicC2<D::Elem>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    fn validate(&self, data: &InterpDataNDBase<D>) -> Result<(), ValidateError> {
        self.boundary_conditions
            .validate_len(data.ndim(), "CubicC2", "boundary conditions")?;
        for dim in 0..data.ndim() {
            validate_bc_min_points(&self.boundary_conditions[dim], data.grid[dim].len(), dim)?;
        }
        Ok(())
    }

    /// Precomputes the full corner-derivative tensor via `compute_corner_cache`.
    fn init(&mut self, data: &InterpDataNDBase<D>) -> Result<(), ValidateError> {
        if data.ndim() == 0 {
            return Ok(());
        }
        let data_view = data.view();
        self.cache =
            compute_corner_cache(&data_view.grid, data_view.values, &self.boundary_conditions);
        Ok(())
    }

    fn interpolate(
        &self,
        data: &InterpDataNDBase<D>,
        point: &[D::Elem],
    ) -> Result<D::Elem, InterpolateError> {
        if data.ndim() == 0 {
            return data.values.first().copied().ok_or_else(|| {
                InterpolateError::Other("internal: 0-D interpolation data has no value".into())
            });
        }
        let grids: Vec<ArrayView1<D::Elem>> = data.grid.iter().map(|g| g.view()).collect();
        Ok(evaluate_spline_corner_cached(
            &grids,
            self.cache.view(),
            point,
        ))
    }

    /// Returns `true`: the boundary cubic polynomials extend naturally.
    fn allow_extrapolate(&self) -> bool {
        true
    }
}

impl<D, S> StrategyND<D> for GridTransform<D::Elem, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
    S: for<'a> StrategyND<ViewRepr<&'a D::Elem>> + Clone + Debug,
{
    /// Checks the axis count and that every raw grid coordinate is in its axis's
    /// configured transform's domain.
    fn validate(&self, data: &InterpDataNDBase<D>) -> Result<(), ValidateError> {
        self.axes
            .validate_len(data.ndim(), "GridTransform", "axes")?;
        let transformed_grid: Vec<Array1<D::Elem>> = data
            .grid
            .iter()
            .enumerate()
            .map(|(dim, grid)| self.transform_axis(dim, grid.view()))
            .collect::<Result<_, _>>()?;
        let values = self.transformed_values_view(data.values.view());
        let view = InterpDataNDView {
            grid: transformed_grid.iter().map(|g| g.view()).collect(),
            values,
        };
        self.inner.validate(&view)
    }

    /// Transforms the grid into `grid_cache`, then initializes `inner` against a
    /// transient view zipping `grid_cache` with `data.values`.
    ///
    /// A raw grid is always strictly increasing, but a decreasing transform (e.g.
    /// `Reciprocal`) would otherwise leave that axis of `grid_cache` decreasing;
    /// `Transform::is_increasing` flags that case so the transformed axis (and the
    /// matching `values` axis) can be reversed back to ascending, matching every
    /// downstream strategy's ascending-grid assumption.
    fn init(&mut self, data: &InterpDataNDBase<D>) -> Result<(), ValidateError> {
        self.grid_cache = data
            .grid
            .iter()
            .enumerate()
            .map(|(dim, grid)| self.transform_axis(dim, grid.view()))
            .collect::<Result<_, _>>()?;
        let values = self.transformed_values_view(data.values.view());
        let view = InterpDataNDView {
            grid: self.grid_cache.iter().map(|g| g.view()).collect(),
            values,
        };
        self.inner.init(&view)
    }

    fn interpolate(
        &self,
        data: &InterpDataNDBase<D>,
        point: &[D::Elem],
    ) -> Result<D::Elem, InterpolateError> {
        self.check_point_domain(point)?;
        let transformed_point: Vec<D::Elem> = point
            .iter()
            .enumerate()
            .map(|(dim, &x)| self.axes[dim].forward(x))
            .collect();
        let values = self.transformed_values_view(data.values.view());
        let view = InterpDataNDView {
            grid: self.grid_cache.iter().map(|g| g.view()).collect(),
            values,
        };
        self.inner.interpolate(&view, &transformed_point)
    }

    /// Wraps in the transformed coordinate space (against `grid_cache`'s bounds),
    /// not `data.grid`'s raw bounds: a nonlinear transform doesn't commute with
    /// wrapping.
    fn interpolate_wrapped(
        &self,
        data: &InterpDataNDBase<D>,
        point: &[D::Elem],
    ) -> Result<D::Elem, InterpolateError>
    where
        D::Elem: Num + Euclid + Copy,
    {
        self.check_point_domain(point)?;
        let wrapped: Vec<D::Elem> = point
            .iter()
            .enumerate()
            .map(|(dim, &x)| self.wrap_axis(dim, x))
            .collect();
        let values = self.transformed_values_view(data.values.view());
        let view = InterpDataNDView {
            grid: self.grid_cache.iter().map(|g| g.view()).collect(),
            values,
        };
        self.inner.interpolate(&view, &wrapped)
    }

    fn allow_extrapolate(&self) -> bool {
        self.inner.allow_extrapolate()
    }
}

impl<D, S> StrategyND<D> for ValuesTransform<D::Elem, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
    S: for<'a> StrategyND<ViewRepr<&'a D::Elem>> + Clone + Debug,
{
    /// Checks that every data value is in the configured transform's domain.
    fn validate(&self, data: &InterpDataNDBase<D>) -> Result<(), ValidateError> {
        let transformed_values = self.transform_values(data.values.view())?;
        let view = InterpDataNDView {
            grid: data.grid.iter().map(|g| g.view()).collect(),
            values: transformed_values.view(),
        };
        self.inner.validate(&view)
    }

    /// Transforms `data.values` into `values_cache`, then initializes `inner`
    /// against a transient view zipping `data.grid` (untouched) with `values_cache`.
    fn init(&mut self, data: &InterpDataNDBase<D>) -> Result<(), ValidateError> {
        self.values_cache = self.transform_values(data.values.view())?;
        let view = InterpDataNDView {
            grid: data.grid.iter().map(|g| g.view()).collect(),
            values: self.values_cache.view(),
        };
        self.inner.init(&view)
    }

    fn interpolate(
        &self,
        data: &InterpDataNDBase<D>,
        point: &[D::Elem],
    ) -> Result<D::Elem, InterpolateError> {
        let view = InterpDataNDView {
            grid: data.grid.iter().map(|g| g.view()).collect(),
            values: self.values_cache.view(),
        };
        let result = self.inner.interpolate(&view, point)?;
        Ok(self.transform.inverse(result))
    }

    /// Hands `point` unmodified to `inner.interpolate_wrapped`, so a nested
    /// `GridTransform` handles the actual raw-space wrap; must not wrap here itself.
    fn interpolate_wrapped(
        &self,
        data: &InterpDataNDBase<D>,
        point: &[D::Elem],
    ) -> Result<D::Elem, InterpolateError>
    where
        D::Elem: Num + Euclid + Copy,
    {
        let view = InterpDataNDView {
            grid: data.grid.iter().map(|g| g.view()).collect(),
            values: self.values_cache.view(),
        };
        let result = self.inner.interpolate_wrapped(&view, point)?;
        Ok(self.transform.inverse(result))
    }

    fn allow_extrapolate(&self) -> bool {
        self.inner.allow_extrapolate()
    }
}
