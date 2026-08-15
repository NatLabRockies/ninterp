use super::*;
use strategy::*;

impl<D> Strategy1D<D> for Linear
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    fn interpolate(
        &self,
        data: &InterpData1DBase<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError> {
        // Extrapolation is checked previously in Interpolator::interpolate,
        // meaning by now, point is within grid bounds or extrapolation is enabled
        match locate_axis(data.grid[0].view(), &point[0]) {
            AxisLocation::Exact(i) => Ok(data.values[i]),
            AxisLocation::Between { lower, frac } => {
                let upper = lower + 1;
                Ok(data.values[lower] * (D::Elem::one() - frac) + data.values[upper] * frac)
            }
        }
    }

    /// Returns `true`.
    fn allow_extrapolate(&self) -> bool {
        true
    }
}

impl<D> Strategy1D<D> for LinearUniform
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    /// Ensures the grid is uniformly spaced.
    fn validate(&self, data: &InterpData1DBase<D>) -> Result<(), ValidateError> {
        validate_uniform_grid_epsilon(data.grid[0].view(), 0, None)
    }

    fn interpolate(
        &self,
        data: &InterpData1DBase<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError> {
        let grid = data.grid[0].view();
        let step = grid[1] - grid[0];
        let x_l = locate_lower_index_uniform(grid[0], step, grid.len(), point[0]);
        let x_u = x_l + 1;
        let x_diff = (point[0] - grid[x_l]) / step;
        Ok(data.values[x_l] * (D::Elem::one() - x_diff) + data.values[x_u] * x_diff)
    }

    /// Returns `true`.
    fn allow_extrapolate(&self) -> bool {
        true
    }
}

impl<D> Strategy1D<D> for Nearest
where
    D: Data + RawDataClone + Clone,
    D::Elem: Sub<Output = D::Elem> + PartialOrd + Copy + Debug,
{
    fn interpolate(
        &self,
        data: &InterpData1DBase<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError> {
        let x_l = locate_lower_index(data.grid[0].view(), &point[0]);
        let x_u = x_l + 1;
        let i = if point[0] - data.grid[0][x_l] < data.grid[0][x_u] - point[0] {
            x_l
        } else {
            x_u
        };
        Ok(data.values[i])
    }

    /// Returns `false`.
    fn allow_extrapolate(&self) -> bool {
        false
    }
}

impl<D> Strategy1D<D> for Step
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialOrd + Copy + Debug,
{
    /// Ensures the number of provided step directions matches the interpolator dimensionality.
    fn validate(&self, _data: &InterpData1DBase<D>) -> Result<(), ValidateError> {
        self.directions.validate_len(1, "Step", "directions")
    }

    fn interpolate(
        &self,
        data: &InterpData1DBase<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError> {
        Ok(data.values[locate_step_index(self.directions[0], data.grid[0].view(), &point[0])])
    }

    fn allow_extrapolate(&self) -> bool {
        false
    }
}

impl<D> Strategy1D<D> for CubicC2<D::Elem>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    /// Checks the boundary-condition count (must be 1) and the grid size against the
    /// configured BC's minimum point requirement (e.g. [`CubicC2Endpoint::NotAKnot`]
    /// needs at least 3 or 4, depending on whether one or both endpoints use it) before
    /// [`Strategy1D::init`] attempts the real computation.
    fn validate(&self, data: &InterpData1DBase<D>) -> Result<(), ValidateError> {
        self.boundary_conditions
            .validate_len(1, "CubicC2", "boundary conditions")?;
        validate_bc_min_points(&self.boundary_conditions[0], data.grid[0].len(), 0)
    }

    /// Computes and caches `M[0..=n]` for the configured BC via `compute_m_cache`.
    fn init(&mut self, data: &InterpData1DBase<D>) -> Result<(), ValidateError> {
        self.cache = compute_m_cache(
            data.grid[0].view(),
            data.values.view(),
            &self.boundary_conditions[0],
        );
        Ok(())
    }

    fn interpolate(
        &self,
        data: &InterpData1DBase<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError> {
        evaluate_spline_1d_cached(
            data.grid[0].view(),
            data.values.view(),
            self.cache.view(),
            point[0],
        )
    }

    /// Returns `true`: the boundary cubic polynomials extend naturally.
    fn allow_extrapolate(&self) -> bool {
        true
    }
}

impl<D, S> Strategy1D<D> for GridTransform<D::Elem, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
    S: for<'a> Strategy1D<ViewRepr<&'a D::Elem>> + Clone + Debug,
{
    /// Checks the axis count (must be 1) and that every raw grid coordinate is in
    /// the configured transform's domain.
    fn validate(&self, data: &InterpData1DBase<D>) -> Result<(), ValidateError> {
        self.transforms
            .validate_len(1, "GridTransform", "transforms")?;
        let transformed_grid = self.transform_axis(0, data.grid[0].view())?;
        let values = self.transformed_values_view(data.values.view());
        let view = InterpData1DView {
            grid: [transformed_grid.view()],
            values,
        };
        self.inner.validate(&view)
    }

    /// Transforms the grid into `grid_cache`, then initializes `inner` against a
    /// transient view zipping `grid_cache` with `data.values`.
    ///
    /// A raw grid is always strictly increasing, but a decreasing transform (e.g.
    /// `Reciprocal`) would otherwise leave `grid_cache` decreasing; `Transform::
    /// is_increasing` flags that case so the transformed axis (and the matching
    /// `values` axis) can be reversed back to ascending, matching every downstream
    /// strategy's ascending-grid assumption.
    fn init(&mut self, data: &InterpData1DBase<D>) -> Result<(), ValidateError> {
        self.grid_cache = vec![self.transform_axis(0, data.grid[0].view())?];
        let values = self.transformed_values_view(data.values.view());
        let view = InterpData1DView {
            grid: [self.grid_cache[0].view()],
            values,
        };
        self.inner.init(&view)
    }

    fn interpolate(
        &self,
        data: &InterpData1DBase<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError> {
        self.check_point_domain(point)?;
        let transformed_point = [self.transforms[0].forward(point[0])];
        let values = self.transformed_values_view(data.values.view());
        let view = InterpData1DView {
            grid: [self.grid_cache[0].view()],
            values,
        };
        self.inner.interpolate(&view, &transformed_point)
    }

    /// Wraps in the transformed coordinate space (against `grid_cache`'s bounds),
    /// not `data.grid`'s raw bounds: a nonlinear transform doesn't commute with
    /// wrapping.
    fn interpolate_wrapped(
        &self,
        data: &InterpData1DBase<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError>
    where
        D::Elem: Num + Euclid + Copy,
    {
        self.check_point_domain(point)?;
        let wrapped = [self.wrap_axis(0, point[0])];
        let values = self.transformed_values_view(data.values.view());
        let view = InterpData1DView {
            grid: [self.grid_cache[0].view()],
            values,
        };
        self.inner.interpolate(&view, &wrapped)
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
    fn interpolate_fast(&self, data: &InterpData1DBase<D>, point: &[D::Elem; 1]) -> D::Elem {
        let transformed_point = [self.transforms[0].forward(point[0])];
        let values = self.transformed_values_view(data.values.view());
        let view = InterpData1DView {
            grid: [self.grid_cache[0].view()],
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
        data: &InterpData1DBase<D>,
        points: &[[D::Elem; 1]],
        out: &mut [D::Elem],
    ) -> Result<(), InterpolateError> {
        if out.len() != points.len() {
            return Err(InterpolateError::OutputLength {
                expected: points.len(),
                found: out.len(),
            });
        }
        self.check_batch_domain(points.iter().map(|p| p.as_slice()))?;
        let transformed_points: Vec<[D::Elem; 1]> = points
            .iter()
            .map(|point| [self.transforms[0].forward(point[0])])
            .collect();
        let values = self.transformed_values_view(data.values.view());
        let view = InterpData1DView {
            grid: [self.grid_cache[0].view()],
            values,
        };
        self.inner
            .batch_interpolate_into(&view, &transformed_points, out)
    }

    /// Delegates to the inherent [`GridTransform::check_batch_domain`], the same
    /// aggregating check `batch_interpolate_into` already runs; exposed here so
    /// generic callers (e.g. `Extrapolate::Wrap`'s batch dispatch) can pre-scan a
    /// batch without knowing the concrete strategy type.
    fn check_batch_domain(&self, points: &[[D::Elem; 1]]) -> Result<(), InterpolateError> {
        GridTransform::check_batch_domain(self, points.iter().map(|p| p.as_slice()))
    }

    fn allow_extrapolate(&self) -> bool {
        self.inner.allow_extrapolate()
    }
}

impl<D, S> Strategy1D<D> for ValuesTransform<D::Elem, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
    S: for<'a> Strategy1D<ViewRepr<&'a D::Elem>> + Clone + Debug,
{
    /// Checks that every data value is in the configured transform's domain.
    fn validate(&self, data: &InterpData1DBase<D>) -> Result<(), ValidateError> {
        let transformed_values = self.transform_values(data.values.view())?;
        let view = InterpData1DView {
            grid: [data.grid[0].view()],
            values: transformed_values.view(),
        };
        self.inner.validate(&view)
    }

    /// Transforms `data.values` into `values_cache`, then initializes `inner`
    /// against a transient view zipping `data.grid` (untouched) with `values_cache`.
    fn init(&mut self, data: &InterpData1DBase<D>) -> Result<(), ValidateError> {
        self.values_cache = self.transform_values(data.values.view())?.into_dyn();
        let view = InterpData1DView {
            grid: [data.grid[0].view()],
            values: self
                .values_cache
                .view()
                .into_dimensionality::<Ix1>()
                .expect("values_cache shape matches 1-D values"),
        };
        self.inner.init(&view)
    }

    fn interpolate(
        &self,
        data: &InterpData1DBase<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError> {
        let view = InterpData1DView {
            grid: [data.grid[0].view()],
            values: self
                .values_cache
                .view()
                .into_dimensionality::<Ix1>()
                .expect("values_cache shape matches 1-D values"),
        };
        let result = self.inner.interpolate(&view, point)?;
        Ok(self.transform.inverse(result))
    }

    /// Hands `point` unmodified to `inner.interpolate_wrapped`, so a nested
    /// `GridTransform` handles the actual raw-space wrap; must not wrap here itself.
    fn interpolate_wrapped(
        &self,
        data: &InterpData1DBase<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError>
    where
        D::Elem: Num + Euclid + Copy,
    {
        let view = InterpData1DView {
            grid: [data.grid[0].view()],
            values: self
                .values_cache
                .view()
                .into_dimensionality::<Ix1>()
                .expect("values_cache shape matches 1-D values"),
        };
        let result = self.inner.interpolate_wrapped(&view, point)?;
        Ok(self.transform.inverse(result))
    }

    /// Calls `inner.interpolate_fast` rather than `inner.interpolate`, so "fast"
    /// propagates through nested `GridTransform`/`ValuesTransform` layers instead of
    /// stopping at the outermost one.
    fn interpolate_fast(&self, data: &InterpData1DBase<D>, point: &[D::Elem; 1]) -> D::Elem {
        let view = InterpData1DView {
            grid: [data.grid[0].view()],
            values: self
                .values_cache
                .view()
                .into_dimensionality::<Ix1>()
                .expect("values_cache shape matches 1-D values"),
        };
        let result = self.inner.interpolate_fast(&view, point);
        self.transform.inverse(result)
    }

    /// Delegates the whole batch to `inner` via a transient view over `values_cache`,
    /// so a nested `GridTransform`'s batch-domain aggregation isn't lost to the
    /// point-by-point default; inverse-transforms every output afterward.
    fn batch_interpolate_into(
        &self,
        data: &InterpData1DBase<D>,
        points: &[[D::Elem; 1]],
        out: &mut [D::Elem],
    ) -> Result<(), InterpolateError> {
        if out.len() != points.len() {
            return Err(InterpolateError::OutputLength {
                expected: points.len(),
                found: out.len(),
            });
        }
        let view = InterpData1DView {
            grid: [data.grid[0].view()],
            values: self
                .values_cache
                .view()
                .into_dimensionality::<Ix1>()
                .expect("values_cache shape matches 1-D values"),
        };
        self.inner.batch_interpolate_into(&view, points, out)?;
        for o in out.iter_mut() {
            *o = self.transform.inverse(*o);
        }
        Ok(())
    }

    /// Forwards to `inner`, so a nested `GridTransform`'s domain is still checked
    /// through a wrapping `ValuesTransform`.
    fn check_batch_domain(&self, points: &[[D::Elem; 1]]) -> Result<(), InterpolateError> {
        self.inner.check_batch_domain(points)
    }

    fn allow_extrapolate(&self) -> bool {
        self.inner.allow_extrapolate()
    }
}
