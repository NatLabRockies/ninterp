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
        self.axes.validate_len(1, "GridTransform", "axes")?;
        let transform = self.axes[0];
        for &x in data.grid[0].iter() {
            if !transform.in_domain(x) {
                return Err(ValidateError::TransformDomain {
                    label: "GridTransform",
                    transform,
                });
            }
        }
        let mut transformed_grid = data.grid[0].mapv(|x| transform.forward(x));
        let values = if transform.is_increasing() {
            data.values.view()
        } else {
            transformed_grid.invert_axis(Axis(0));
            data.values.slice(s![..;-1])
        };
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
        let transform = self.axes[0];
        let mut transformed = data.grid[0].mapv(|x| transform.forward(x));
        let values = if transform.is_increasing() {
            data.values.view()
        } else {
            transformed.invert_axis(Axis(0));
            data.values.slice(s![..;-1])
        };
        self.grid_cache = vec![transformed];
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
        let transform = self.axes[0];
        if !transform.in_domain(point[0]) {
            return Err(InterpolateError::TransformDomain {
                label: "GridTransform",
                transform,
            });
        }
        let transformed_point = [transform.forward(point[0])];
        let values = if transform.is_increasing() {
            data.values.view()
        } else {
            data.values.slice(s![..;-1])
        };
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
        let transform = self.axes[0];
        if !transform.in_domain(point[0]) {
            return Err(InterpolateError::TransformDomain {
                label: "GridTransform",
                transform,
            });
        }
        let transformed = transform.forward(point[0]);
        let lo = *self.grid_cache[0].first().unwrap();
        let hi = *self.grid_cache[0].last().unwrap();
        let wrapped = [wrap(transformed, lo, hi)];
        let values = if transform.is_increasing() {
            data.values.view()
        } else {
            data.values.slice(s![..;-1])
        };
        let view = InterpData1DView {
            grid: [self.grid_cache[0].view()],
            values,
        };
        self.inner.interpolate(&view, &wrapped)
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
        for &v in data.values.iter() {
            if !self.transform.in_domain(v) {
                return Err(ValidateError::TransformDomain {
                    label: "ValuesTransform",
                    transform: self.transform,
                });
            }
        }
        let transformed_values = data.values.mapv(|v| self.transform.forward(v));
        let view = InterpData1DView {
            grid: [data.grid[0].view()],
            values: transformed_values.view(),
        };
        self.inner.validate(&view)
    }

    /// Transforms `data.values` into `values_cache`, then initializes `inner`
    /// against a transient view zipping `data.grid` (untouched) with `values_cache`.
    fn init(&mut self, data: &InterpData1DBase<D>) -> Result<(), ValidateError> {
        self.values_cache = data.values.mapv(|v| self.transform.forward(v)).into_dyn();
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

    fn allow_extrapolate(&self) -> bool {
        self.inner.allow_extrapolate()
    }
}
