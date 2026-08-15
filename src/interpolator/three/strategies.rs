use super::*;
use strategy::*;

impl<D> Strategy3D<D> for Linear
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    fn interpolate(
        &self,
        data: &InterpData3DBase<D>,
        point: &[D::Elem; 3],
    ) -> Result<D::Elem, InterpolateError> {
        // Extrapolation is checked previously in Interpolator::interpolate,
        // meaning by now, point is within grid bounds or extrapolation is enabled.
        //
        // Short-circuit if the point lies exactly on a grid coordinate in one or more
        // dimensions, reducing value lookups from 8 down to 4, 2, or 1.
        match (
            locate_axis(data.grid[0].view(), &point[0]),
            locate_axis(data.grid[1].view(), &point[1]),
            locate_axis(data.grid[2].view(), &point[2]),
        ) {
            (AxisLocation::Exact(i), AxisLocation::Exact(j), AxisLocation::Exact(k)) => {
                Ok(data.values[[i, j, k]])
            }
            (
                AxisLocation::Exact(i),
                AxisLocation::Exact(j),
                AxisLocation::Between {
                    lower: z_l,
                    frac: z_diff,
                },
            ) => {
                let z_u = z_l + 1;
                Ok(data.values[[i, j, z_l]] * (D::Elem::one() - z_diff)
                    + data.values[[i, j, z_u]] * z_diff)
            }
            (
                AxisLocation::Exact(i),
                AxisLocation::Between {
                    lower: y_l,
                    frac: y_diff,
                },
                AxisLocation::Exact(k),
            ) => {
                let y_u = y_l + 1;
                Ok(data.values[[i, y_l, k]] * (D::Elem::one() - y_diff)
                    + data.values[[i, y_u, k]] * y_diff)
            }
            (
                AxisLocation::Between {
                    lower: x_l,
                    frac: x_diff,
                },
                AxisLocation::Exact(j),
                AxisLocation::Exact(k),
            ) => {
                let x_u = x_l + 1;
                Ok(data.values[[x_l, j, k]] * (D::Elem::one() - x_diff)
                    + data.values[[x_u, j, k]] * x_diff)
            }
            (
                AxisLocation::Exact(i),
                AxisLocation::Between {
                    lower: y_l,
                    frac: y_diff,
                },
                AxisLocation::Between {
                    lower: z_l,
                    frac: z_diff,
                },
            ) => {
                let y_u = y_l + 1;
                let z_u = z_l + 1;
                let f0 = data.values[[i, y_l, z_l]] * (D::Elem::one() - y_diff)
                    + data.values[[i, y_u, z_l]] * y_diff;
                let f1 = data.values[[i, y_l, z_u]] * (D::Elem::one() - y_diff)
                    + data.values[[i, y_u, z_u]] * y_diff;
                Ok(f0 * (D::Elem::one() - z_diff) + f1 * z_diff)
            }
            (
                AxisLocation::Between {
                    lower: x_l,
                    frac: x_diff,
                },
                AxisLocation::Exact(j),
                AxisLocation::Between {
                    lower: z_l,
                    frac: z_diff,
                },
            ) => {
                let x_u = x_l + 1;
                let z_u = z_l + 1;
                let f0 = data.values[[x_l, j, z_l]] * (D::Elem::one() - x_diff)
                    + data.values[[x_u, j, z_l]] * x_diff;
                let f1 = data.values[[x_l, j, z_u]] * (D::Elem::one() - x_diff)
                    + data.values[[x_u, j, z_u]] * x_diff;
                Ok(f0 * (D::Elem::one() - z_diff) + f1 * z_diff)
            }
            (
                AxisLocation::Between {
                    lower: x_l,
                    frac: x_diff,
                },
                AxisLocation::Between {
                    lower: y_l,
                    frac: y_diff,
                },
                AxisLocation::Exact(k),
            ) => {
                let x_u = x_l + 1;
                let y_u = y_l + 1;
                let f0 = data.values[[x_l, y_l, k]] * (D::Elem::one() - x_diff)
                    + data.values[[x_u, y_l, k]] * x_diff;
                let f1 = data.values[[x_l, y_u, k]] * (D::Elem::one() - x_diff)
                    + data.values[[x_u, y_u, k]] * x_diff;
                Ok(f0 * (D::Elem::one() - y_diff) + f1 * y_diff)
            }
            (
                AxisLocation::Between {
                    lower: x_l,
                    frac: x_diff,
                },
                AxisLocation::Between {
                    lower: y_l,
                    frac: y_diff,
                },
                AxisLocation::Between {
                    lower: z_l,
                    frac: z_diff,
                },
            ) => {
                let x_u = x_l + 1;
                let y_u = y_l + 1;
                let z_u = z_l + 1;
                // interpolate in the x-direction
                let f00 = data.values[[x_l, y_l, z_l]] * (D::Elem::one() - x_diff)
                    + data.values[[x_u, y_l, z_l]] * x_diff;
                let f01 = data.values[[x_l, y_l, z_u]] * (D::Elem::one() - x_diff)
                    + data.values[[x_u, y_l, z_u]] * x_diff;
                let f10 = data.values[[x_l, y_u, z_l]] * (D::Elem::one() - x_diff)
                    + data.values[[x_u, y_u, z_l]] * x_diff;
                let f11 = data.values[[x_l, y_u, z_u]] * (D::Elem::one() - x_diff)
                    + data.values[[x_u, y_u, z_u]] * x_diff;
                // interpolate in the y-direction
                let f0 = f00 * (D::Elem::one() - y_diff) + f10 * y_diff;
                let f1 = f01 * (D::Elem::one() - y_diff) + f11 * y_diff;
                // interpolate in the z-direction
                Ok(f0 * (D::Elem::one() - z_diff) + f1 * z_diff)
            }
        }
    }

    /// Returns `true`.
    fn allow_extrapolate(&self) -> bool {
        true
    }
}

impl<D> Strategy3D<D> for LinearUniform
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    /// Ensures all grid dimensions are uniformly spaced.
    fn validate(&self, data: &InterpData3DBase<D>) -> Result<(), ValidateError> {
        validate_uniform_grid_epsilon(data.grid[0].view(), 0, None)?;
        validate_uniform_grid_epsilon(data.grid[1].view(), 1, None)?;
        validate_uniform_grid_epsilon(data.grid[2].view(), 2, None)
    }

    fn interpolate(
        &self,
        data: &InterpData3DBase<D>,
        point: &[D::Elem; 3],
    ) -> Result<D::Elem, InterpolateError> {
        let x_step = data.grid[0][1] - data.grid[0][0];
        let y_step = data.grid[1][1] - data.grid[1][0];
        let z_step = data.grid[2][1] - data.grid[2][0];
        let x_l = locate_lower_index_uniform(data.grid[0][0], x_step, data.grid[0].len(), point[0]);
        let y_l = locate_lower_index_uniform(data.grid[1][0], y_step, data.grid[1].len(), point[1]);
        let z_l = locate_lower_index_uniform(data.grid[2][0], z_step, data.grid[2].len(), point[2]);
        let x_u = x_l + 1;
        let y_u = y_l + 1;
        let z_u = z_l + 1;
        let x_diff = (point[0] - data.grid[0][x_l]) / x_step;
        let y_diff = (point[1] - data.grid[1][y_l]) / y_step;
        let z_diff = (point[2] - data.grid[2][z_l]) / z_step;
        let f00 = data.values[[x_l, y_l, z_l]] * (D::Elem::one() - x_diff)
            + data.values[[x_u, y_l, z_l]] * x_diff;
        let f01 = data.values[[x_l, y_l, z_u]] * (D::Elem::one() - x_diff)
            + data.values[[x_u, y_l, z_u]] * x_diff;
        let f10 = data.values[[x_l, y_u, z_l]] * (D::Elem::one() - x_diff)
            + data.values[[x_u, y_u, z_l]] * x_diff;
        let f11 = data.values[[x_l, y_u, z_u]] * (D::Elem::one() - x_diff)
            + data.values[[x_u, y_u, z_u]] * x_diff;
        let f0 = f00 * (D::Elem::one() - y_diff) + f10 * y_diff;
        let f1 = f01 * (D::Elem::one() - y_diff) + f11 * y_diff;
        Ok(f0 * (D::Elem::one() - z_diff) + f1 * z_diff)
    }

    /// Returns `true`.
    fn allow_extrapolate(&self) -> bool {
        true
    }
}

impl<D> Strategy3D<D> for Nearest
where
    D: Data + RawDataClone + Clone,
    D::Elem: Sub<Output = D::Elem> + PartialOrd + Copy + Debug,
{
    fn interpolate(
        &self,
        data: &InterpData3DBase<D>,
        point: &[D::Elem; 3],
    ) -> Result<D::Elem, InterpolateError> {
        // x
        let x_l = locate_lower_index(data.grid[0].view(), &point[0]);
        let x_u = x_l + 1;
        let i = if point[0] - data.grid[0][x_l] < data.grid[0][x_u] - point[0] {
            x_l
        } else {
            x_u
        };
        // y
        let y_l = locate_lower_index(data.grid[1].view(), &point[1]);
        let y_u = y_l + 1;
        let j = if point[1] - data.grid[1][y_l] < data.grid[1][y_u] - point[1] {
            y_l
        } else {
            y_u
        };
        // z
        let z_l = locate_lower_index(data.grid[2].view(), &point[2]);
        let z_u = z_l + 1;
        let k = if point[2] - data.grid[2][z_l] < data.grid[2][z_u] - point[2] {
            z_l
        } else {
            z_u
        };

        Ok(data.values[[i, j, k]])
    }

    /// Returns `false`.
    fn allow_extrapolate(&self) -> bool {
        false
    }
}

impl<D> Strategy3D<D> for Step
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialOrd + Copy + Debug,
{
    /// Ensures the number of provided step directions matches the interpolator dimensionality.
    fn validate(&self, _data: &InterpData3DBase<D>) -> Result<(), ValidateError> {
        self.directions.validate_len(3, "Step", "directions")
    }

    fn interpolate(
        &self,
        data: &InterpData3DBase<D>,
        point: &[D::Elem; 3],
    ) -> Result<D::Elem, InterpolateError> {
        let i = locate_step_index(self.directions[0], data.grid[0].view(), &point[0]);
        let j = locate_step_index(self.directions[1], data.grid[1].view(), &point[1]);
        let k = locate_step_index(self.directions[2], data.grid[2].view(), &point[2]);
        Ok(data.values[[i, j, k]])
    }

    /// Returns `false`.
    fn allow_extrapolate(&self) -> bool {
        false
    }
}

impl<D> Strategy3D<D> for CubicC2<D::Elem>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    fn validate(&self, data: &InterpData3DBase<D>) -> Result<(), ValidateError> {
        self.boundary_conditions
            .validate_len(3, "CubicC2", "boundary conditions")?;
        for (dim, grid) in data.grid.iter().enumerate() {
            validate_bc_min_points(&self.boundary_conditions[dim], grid.len(), dim)?;
        }
        Ok(())
    }

    /// Precomputes the full corner-derivative tensor via `compute_corner_cache`, so
    /// [`interpolate`](Self::interpolate) is an O(1) Hermite-patch lookup instead of
    /// re-solving the outer axes on every call.
    fn init(&mut self, data: &InterpData3DBase<D>) -> Result<(), ValidateError> {
        let data_view = data.view();
        self.cache = compute_corner_cache(
            &data_view.grid,
            data_view.values.into_dyn(),
            &self.boundary_conditions,
        );
        Ok(())
    }

    fn interpolate(
        &self,
        data: &InterpData3DBase<D>,
        point: &[D::Elem; 3],
    ) -> Result<D::Elem, InterpolateError> {
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

impl<D, S> Strategy3D<D> for GridTransform<D::Elem, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
    S: for<'a> Strategy3D<ViewRepr<&'a D::Elem>> + Clone + Debug,
{
    /// Checks the axis count (must be 1 or 3) and that every raw grid coordinate is
    /// in its axis's configured transform's domain.
    fn validate(&self, data: &InterpData3DBase<D>) -> Result<(), ValidateError> {
        self.transforms
            .validate_len(3, "GridTransform", "transforms")?;
        let transformed_grid: Vec<Array1<D::Elem>> = data
            .grid
            .iter()
            .enumerate()
            .map(|(dim, grid)| self.transform_axis(dim, grid.view()))
            .collect::<Result<_, _>>()?;
        let values = self.transformed_values_view(data.values.view());
        let view = InterpData3DView {
            grid: core::array::from_fn(|i| transformed_grid[i].view()),
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
    fn init(&mut self, data: &InterpData3DBase<D>) -> Result<(), ValidateError> {
        self.grid_cache = data
            .grid
            .iter()
            .enumerate()
            .map(|(dim, grid)| self.transform_axis(dim, grid.view()))
            .collect::<Result<_, _>>()?;
        let values = self.transformed_values_view(data.values.view());
        let view = InterpData3DView {
            grid: core::array::from_fn(|i| self.grid_cache[i].view()),
            values,
        };
        self.inner.init(&view)
    }

    fn interpolate(
        &self,
        data: &InterpData3DBase<D>,
        point: &[D::Elem; 3],
    ) -> Result<D::Elem, InterpolateError> {
        self.check_point_domain(point)?;
        let transformed_point: [D::Elem; 3] =
            core::array::from_fn(|i| self.transforms[i].forward(point[i]));
        let values = self.transformed_values_view(data.values.view());
        let view = InterpData3DView {
            grid: core::array::from_fn(|i| self.grid_cache[i].view()),
            values,
        };
        self.inner.interpolate(&view, &transformed_point)
    }

    /// Wraps in the transformed coordinate space (against `grid_cache`'s bounds),
    /// not `data.grid`'s raw bounds: a nonlinear transform doesn't commute with
    /// wrapping.
    fn interpolate_wrapped(
        &self,
        data: &InterpData3DBase<D>,
        point: &[D::Elem; 3],
    ) -> Result<D::Elem, InterpolateError>
    where
        D::Elem: Num + Euclid + Copy,
    {
        self.check_point_domain(point)?;
        let wrapped: [D::Elem; 3] = core::array::from_fn(|i| self.wrap_axis(i, point[i]));
        let values = self.transformed_values_view(data.values.view());
        let view = InterpData3DView {
            grid: core::array::from_fn(|i| self.grid_cache[i].view()),
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
    fn interpolate_fast(&self, data: &InterpData3DBase<D>, point: &[D::Elem; 3]) -> D::Elem {
        let transformed_point: [D::Elem; 3] =
            core::array::from_fn(|i| self.transforms[i].forward(point[i]));
        let values = self.transformed_values_view(data.values.view());
        let view = InterpData3DView {
            grid: core::array::from_fn(|i| self.grid_cache[i].view()),
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
        data: &InterpData3DBase<D>,
        points: &[[D::Elem; 3]],
        out: &mut [D::Elem],
    ) -> Result<(), InterpolateError> {
        if out.len() != points.len() {
            return Err(InterpolateError::OutputLength {
                expected: points.len(),
                found: out.len(),
            });
        }
        self.check_batch_domain(points.iter().map(|p| p.as_slice()))?;
        let transformed_points: Vec<[D::Elem; 3]> = points
            .iter()
            .map(|point| core::array::from_fn(|i| self.transforms[i].forward(point[i])))
            .collect();
        let values = self.transformed_values_view(data.values.view());
        let view = InterpData3DView {
            grid: core::array::from_fn(|i| self.grid_cache[i].view()),
            values,
        };
        self.inner
            .batch_interpolate_into(&view, &transformed_points, out)
    }

    /// Delegates to the inherent [`GridTransform::check_batch_domain`], the same
    /// aggregating check `batch_interpolate_into` already runs; exposed here so
    /// generic callers (e.g. `Extrapolate::Wrap`'s batch dispatch) can pre-scan a
    /// batch without knowing the concrete strategy type.
    fn check_batch_domain(&self, points: &[[D::Elem; 3]]) -> Result<(), InterpolateError> {
        GridTransform::check_batch_domain(self, points.iter().map(|p| p.as_slice()))
    }

    fn allow_extrapolate(&self) -> bool {
        self.inner.allow_extrapolate()
    }
}

impl<D, S> Strategy3D<D> for ValuesTransform<D::Elem, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
    S: for<'a> Strategy3D<ViewRepr<&'a D::Elem>> + Clone + Debug,
{
    /// Checks that every data value is in the configured transform's domain.
    fn validate(&self, data: &InterpData3DBase<D>) -> Result<(), ValidateError> {
        let transformed_values = self.transform_values(data.values.view())?;
        let view = InterpData3DView {
            grid: core::array::from_fn(|i| data.grid[i].view()),
            values: transformed_values.view(),
        };
        self.inner.validate(&view)
    }

    /// Transforms `data.values` into `values_cache`, then initializes `inner`
    /// against a transient view zipping `data.grid` (untouched) with `values_cache`.
    fn init(&mut self, data: &InterpData3DBase<D>) -> Result<(), ValidateError> {
        self.values_cache = self.transform_values(data.values.view())?.into_dyn();
        let view = InterpData3DView {
            grid: core::array::from_fn(|i| data.grid[i].view()),
            values: self
                .values_cache
                .view()
                .into_dimensionality::<Ix3>()
                .expect("values_cache shape matches 3-D values"),
        };
        self.inner.init(&view)
    }

    fn interpolate(
        &self,
        data: &InterpData3DBase<D>,
        point: &[D::Elem; 3],
    ) -> Result<D::Elem, InterpolateError> {
        let view = InterpData3DView {
            grid: core::array::from_fn(|i| data.grid[i].view()),
            values: self
                .values_cache
                .view()
                .into_dimensionality::<Ix3>()
                .expect("values_cache shape matches 3-D values"),
        };
        let result = self.inner.interpolate(&view, point)?;
        Ok(self.transform.inverse(result))
    }

    /// Hands `point` unmodified to `inner.interpolate_wrapped`, so a nested
    /// `GridTransform` handles the actual raw-space wrap; must not wrap here itself.
    fn interpolate_wrapped(
        &self,
        data: &InterpData3DBase<D>,
        point: &[D::Elem; 3],
    ) -> Result<D::Elem, InterpolateError>
    where
        D::Elem: Num + Euclid + Copy,
    {
        let view = InterpData3DView {
            grid: core::array::from_fn(|i| data.grid[i].view()),
            values: self
                .values_cache
                .view()
                .into_dimensionality::<Ix3>()
                .expect("values_cache shape matches 3-D values"),
        };
        let result = self.inner.interpolate_wrapped(&view, point)?;
        Ok(self.transform.inverse(result))
    }

    /// Calls `inner.interpolate_fast` rather than `inner.interpolate`, so "fast"
    /// propagates through nested `GridTransform`/`ValuesTransform` layers instead of
    /// stopping at the outermost one.
    fn interpolate_fast(&self, data: &InterpData3DBase<D>, point: &[D::Elem; 3]) -> D::Elem {
        let view = InterpData3DView {
            grid: core::array::from_fn(|i| data.grid[i].view()),
            values: self
                .values_cache
                .view()
                .into_dimensionality::<Ix3>()
                .expect("values_cache shape matches 3-D values"),
        };
        let result = self.inner.interpolate_fast(&view, point);
        self.transform.inverse(result)
    }

    /// Delegates the whole batch to `inner` via a transient view over `values_cache`,
    /// so a nested `GridTransform`'s batch-domain aggregation isn't lost to the
    /// point-by-point default; inverse-transforms every output afterward.
    fn batch_interpolate_into(
        &self,
        data: &InterpData3DBase<D>,
        points: &[[D::Elem; 3]],
        out: &mut [D::Elem],
    ) -> Result<(), InterpolateError> {
        if out.len() != points.len() {
            return Err(InterpolateError::OutputLength {
                expected: points.len(),
                found: out.len(),
            });
        }
        let view = InterpData3DView {
            grid: core::array::from_fn(|i| data.grid[i].view()),
            values: self
                .values_cache
                .view()
                .into_dimensionality::<Ix3>()
                .expect("values_cache shape matches 3-D values"),
        };
        self.inner.batch_interpolate_into(&view, points, out)?;
        for o in out.iter_mut() {
            *o = self.transform.inverse(*o);
        }
        Ok(())
    }

    /// Forwards to `inner`, so a nested `GridTransform`'s domain is still checked
    /// through a wrapping `ValuesTransform`.
    fn check_batch_domain(&self, points: &[[D::Elem; 3]]) -> Result<(), InterpolateError> {
        self.inner.check_batch_domain(points)
    }

    fn allow_extrapolate(&self) -> bool {
        self.inner.allow_extrapolate()
    }
}
