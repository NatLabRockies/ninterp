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

impl<D> StrategyND<D> for CubicC1<D::Elem>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    /// Precomputes the full corner-derivative tensor under [`CubicC1CacheMode::Full`]
    /// (the default); under [`CubicC1CacheMode::None`], only validates.
    fn init(&mut self, data: &InterpDataNDBase<D>) -> Result<(), ValidateError> {
        if data.ndim() == 0 {
            return Ok(());
        }
        if self.cache_mode == CubicC1CacheMode::Full {
            let data_view = data.view();
            self.cache = compute_corner_cache_fd(&data_view.grid, data_view.values);
        }
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
        Ok(match self.cache_mode {
            CubicC1CacheMode::Full => {
                evaluate_spline_corner_cached(&grids, self.cache.view(), point)
            }
            CubicC1CacheMode::None => {
                evaluate_spline_corner_local(&grids, data.values.view(), point)
            }
        })
    }

    /// Returns `true`: the boundary Hermite patch extends naturally.
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
        self.transforms
            .validate_len(data.ndim(), "GridTransform", "transforms")?;
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
            .map(|(dim, &x)| self.transforms[dim].forward(x))
            .collect();
        let values = self.transformed_values_view(data.values.view());
        let view = InterpDataNDView {
            grid: self.grid_cache.iter().map(|g| g.view()).collect(),
            values,
        };
        self.inner.interpolate(&view, &transformed_point)
    }

    /// Forward-transforms into `grid_cache`'s coordinate space, then delegates the
    /// actual wrap to `inner.interpolate_wrapped` rather than wrapping here itself:
    /// wrapping doesn't commute with a nonlinear transform, so it must happen in the
    /// *final* (innermost) transformed space where the periodic strategy actually
    /// lives, mirroring how `ValuesTransform::interpolate_wrapped` defers to `inner`
    /// for the same reason (composing two `GridTransform`s and wrapping at the outer
    /// layer's space, then forward-transforming again, uses the wrong period). For a
    /// non-transform `inner`, `StrategyND::interpolate_wrapped`'s default wraps
    /// directly against `grid_cache`, reproducing this layer's own wrap exactly.
    fn interpolate_wrapped(
        &self,
        data: &InterpDataNDBase<D>,
        point: &[D::Elem],
    ) -> Result<D::Elem, InterpolateError>
    where
        D::Elem: Num + Euclid + Copy,
    {
        self.check_point_domain(point)?;
        let transformed_point: Vec<D::Elem> = point
            .iter()
            .enumerate()
            .map(|(dim, &x)| self.transforms[dim].forward(x))
            .collect();
        let values = self.transformed_values_view(data.values.view());
        let view = InterpDataNDView {
            grid: self.grid_cache.iter().map(|g| g.view()).collect(),
            values,
        };
        self.inner.interpolate_wrapped(&view, &transformed_point)
    }

    /// Skips the domain check `interpolate` does, and calls `inner.interpolate_fast`
    /// rather than `inner.interpolate`, so "fast" propagates through nested
    /// `GridTransform`/`ValuesTransform` layers instead of stopping at the outermost
    /// one.
    ///
    /// An out-of-domain point is the caller's problem here, same as any other
    /// unchecked `_fast` method: `forward`-ing it produces `NaN` (`Log`/`Sqrt`) or
    /// `+-inf` (`Reciprocal`, only at `x == 0`). `NaN` poisons the grid search's
    /// comparisons and panics; `+-inf` compares normally, so it's treated like an
    /// ordinary out-of-bounds extrapolation query and silently produces `NaN` output
    /// instead. Expected, not a bug: check [`Transform::in_domain`] yourself first if
    /// you need a guarantee either way.
    fn interpolate_fast(&self, data: &InterpDataNDBase<D>, point: &[D::Elem]) -> D::Elem {
        let transformed_point: Vec<D::Elem> = point
            .iter()
            .enumerate()
            .map(|(dim, &x)| self.transforms[dim].forward(x))
            .collect();
        let values = self.transformed_values_view(data.values.view());
        let view = InterpDataNDView {
            grid: self.grid_cache.iter().map(|g| g.view()).collect(),
            values,
        };
        self.inner.interpolate_fast(&view, &transformed_point)
    }

    /// Domain-checks every point in the batch before transforming, aggregating
    /// every violation across the *whole batch* into one
    /// [`InterpolateError::GridTransformDomain`] instead of erroring on the first
    /// one, mirroring how `Extrapolate::Error` aggregates out-of-bounds points.
    fn batch_interpolate_into(
        &self,
        data: &InterpDataNDBase<D>,
        points: &[&[D::Elem]],
        out: &mut [D::Elem],
    ) -> Result<(), InterpolateError> {
        if out.len() != points.len() {
            return Err(InterpolateError::OutputLength {
                expected: points.len(),
                found: out.len(),
            });
        }
        self.check_batch_domain(points.iter().copied())?;
        let transformed_points: Vec<Vec<D::Elem>> = points
            .iter()
            .map(|point| {
                point
                    .iter()
                    .enumerate()
                    .map(|(dim, &x)| self.transforms[dim].forward(x))
                    .collect()
            })
            .collect();
        let transformed_refs: Vec<&[D::Elem]> =
            transformed_points.iter().map(Vec::as_slice).collect();
        let values = self.transformed_values_view(data.values.view());
        let view = InterpDataNDView {
            grid: self.grid_cache.iter().map(|g| g.view()).collect(),
            values,
        };
        self.inner
            .batch_interpolate_into(&view, &transformed_refs, out)
    }

    /// Checks this layer's own domain per point (not bailing on the first violation),
    /// then recurses into `inner` with only the *outer-valid* points' forward
    /// transforms, remapping `inner`'s failure indices back to their true batch
    /// position: an outer-invalid point can't be forward-transformed meaningfully, but
    /// its presence must not hide `inner`'s own violations for the *other* points, so
    /// a `GridTransform` nested inside another one still gets every one of its
    /// violations aggregated, not just the outer layer's. Exposed here so generic
    /// callers (e.g. `Extrapolate::Wrap`'s batch dispatch) can pre-scan a batch
    /// without knowing the concrete strategy type.
    fn check_batch_domain(&self, points: &[&[D::Elem]]) -> Result<(), InterpolateError> {
        let mut failures = Vec::new();
        let mut valid_indices = Vec::new();
        let mut transformed_points: Vec<Vec<D::Elem>> = Vec::new();
        for (index, &point) in points.iter().enumerate() {
            let point_failures = self.point_domain_failures(index, point);
            if point_failures.is_empty() {
                valid_indices.push(index);
                transformed_points.push(
                    point
                        .iter()
                        .enumerate()
                        .map(|(dim, &x)| self.transforms[dim].forward(x))
                        .collect(),
                );
            } else {
                failures.extend(point_failures);
            }
        }
        let transformed_refs: Vec<&[D::Elem]> =
            transformed_points.iter().map(Vec::as_slice).collect();
        match self.inner.check_batch_domain(&transformed_refs) {
            Ok(()) => {}
            Err(InterpolateError::GridTransformDomain(inner_failures)) => {
                failures.extend(inner_failures.into_iter().map(|f| OutsideDomainAt {
                    index: valid_indices[f.index],
                    ..f
                }));
            }
            Err(other) => return Err(other),
        }
        failures.sort_by_key(|f| (f.index, f.dim));
        if failures.is_empty() {
            Ok(())
        } else {
            Err(InterpolateError::GridTransformDomain(failures))
        }
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

    /// Calls `inner.interpolate_fast` rather than `inner.interpolate`, so "fast"
    /// propagates through nested `GridTransform`/`ValuesTransform` layers instead of
    /// stopping at the outermost one.
    fn interpolate_fast(&self, data: &InterpDataNDBase<D>, point: &[D::Elem]) -> D::Elem {
        let view = InterpDataNDView {
            grid: data.grid.iter().map(|g| g.view()).collect(),
            values: self.values_cache.view(),
        };
        let result = self.inner.interpolate_fast(&view, point);
        self.transform.inverse(result)
    }

    /// Delegates the whole batch to `inner` via a transient view over `values_cache`,
    /// so a nested `GridTransform`'s batch-domain aggregation isn't lost to the
    /// point-by-point default; inverse-transforms every output afterward.
    fn batch_interpolate_into(
        &self,
        data: &InterpDataNDBase<D>,
        points: &[&[D::Elem]],
        out: &mut [D::Elem],
    ) -> Result<(), InterpolateError> {
        if out.len() != points.len() {
            return Err(InterpolateError::OutputLength {
                expected: points.len(),
                found: out.len(),
            });
        }
        let view = InterpDataNDView {
            grid: data.grid.iter().map(|g| g.view()).collect(),
            values: self.values_cache.view(),
        };
        self.inner.batch_interpolate_into(&view, points, out)?;
        for o in out.iter_mut() {
            *o = self.transform.inverse(*o);
        }
        Ok(())
    }

    /// Forwards to `inner`, so a nested `GridTransform`'s domain is still checked
    /// through a wrapping `ValuesTransform`.
    fn check_batch_domain(&self, points: &[&[D::Elem]]) -> Result<(), InterpolateError> {
        self.inner.check_batch_domain(points)
    }

    fn allow_extrapolate(&self) -> bool {
        self.inner.allow_extrapolate()
    }
}
