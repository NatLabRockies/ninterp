use super::*;
use strategy::*;

impl<D> Strategy2D<D> for Linear
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    fn interpolate(
        &self,
        data: &InterpData2DBase<D>,
        point: &[D::Elem; 2],
    ) -> Result<D::Elem, InterpolateError> {
        // Extrapolation is checked previously in Interpolator::interpolate,
        // meaning by now, point is within grid bounds or extrapolation is enabled.
        //
        // Short-circuit if the point lies exactly on a grid coordinate in one or both
        // dimensions, reducing value lookups from 4 to 2 or 1.
        match (
            locate_axis(data.grid[0].view(), &point[0]),
            locate_axis(data.grid[1].view(), &point[1]),
        ) {
            (AxisLocation::Exact(i), AxisLocation::Exact(j)) => Ok(data.values[[i, j]]),
            (
                AxisLocation::Exact(i),
                AxisLocation::Between {
                    lower: y_l,
                    frac: y_diff,
                },
            ) => {
                let y_u = y_l + 1;
                Ok(data.values[[i, y_l]] * (D::Elem::one() - y_diff)
                    + data.values[[i, y_u]] * y_diff)
            }
            (
                AxisLocation::Between {
                    lower: x_l,
                    frac: x_diff,
                },
                AxisLocation::Exact(j),
            ) => {
                let x_u = x_l + 1;
                Ok(data.values[[x_l, j]] * (D::Elem::one() - x_diff)
                    + data.values[[x_u, j]] * x_diff)
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
            ) => {
                let x_u = x_l + 1;
                let y_u = y_l + 1;
                // interpolate in the x-direction
                let f0 = data.values[[x_l, y_l]] * (D::Elem::one() - x_diff)
                    + data.values[[x_u, y_l]] * x_diff;
                let f1 = data.values[[x_l, y_u]] * (D::Elem::one() - x_diff)
                    + data.values[[x_u, y_u]] * x_diff;
                // interpolate in the y-direction
                Ok(f0 * (D::Elem::one() - y_diff) + f1 * y_diff)
            }
        }
    }

    /// Returns `true`.
    fn allow_extrapolate(&self) -> bool {
        true
    }
}

impl<D> Strategy2D<D> for LinearUniform
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    /// Ensures all grid dimensions are uniformly spaced.
    fn validate(&self, data: &InterpData2DBase<D>) -> Result<(), ValidateError> {
        validate_uniform_grid_epsilon(data.grid[0].view(), 0, None)?;
        validate_uniform_grid_epsilon(data.grid[1].view(), 1, None)
    }

    fn interpolate(
        &self,
        data: &InterpData2DBase<D>,
        point: &[D::Elem; 2],
    ) -> Result<D::Elem, InterpolateError> {
        let x_step = data.grid[0][1] - data.grid[0][0];
        let y_step = data.grid[1][1] - data.grid[1][0];
        let x_l = locate_lower_index_uniform(data.grid[0][0], x_step, data.grid[0].len(), point[0]);
        let y_l = locate_lower_index_uniform(data.grid[1][0], y_step, data.grid[1].len(), point[1]);
        let x_u = x_l + 1;
        let y_u = y_l + 1;
        let x_diff = (point[0] - data.grid[0][x_l]) / x_step;
        let y_diff = (point[1] - data.grid[1][y_l]) / y_step;
        let f0 =
            data.values[[x_l, y_l]] * (D::Elem::one() - x_diff) + data.values[[x_u, y_l]] * x_diff;
        let f1 =
            data.values[[x_l, y_u]] * (D::Elem::one() - x_diff) + data.values[[x_u, y_u]] * x_diff;
        Ok(f0 * (D::Elem::one() - y_diff) + f1 * y_diff)
    }

    /// Returns `true`.
    fn allow_extrapolate(&self) -> bool {
        true
    }
}

impl<D> Strategy2D<D> for Nearest
where
    D: Data + RawDataClone + Clone,
    D::Elem: Sub<Output = D::Elem> + PartialOrd + Copy + Debug,
{
    fn interpolate(
        &self,
        data: &InterpData2DBase<D>,
        point: &[D::Elem; 2],
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

        Ok(data.values[[i, j]])
    }

    /// Returns `false`.
    fn allow_extrapolate(&self) -> bool {
        false
    }
}

impl<D> Strategy2D<D> for Step
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialOrd + Copy + Debug,
{
    /// Ensures the number of provided step directions matches the interpolator dimensionality.
    fn validate(&self, _data: &InterpData2DBase<D>) -> Result<(), ValidateError> {
        self.directions.validate_len(2, "Step", "directions")
    }

    fn interpolate(
        &self,
        data: &InterpData2DBase<D>,
        point: &[D::Elem; 2],
    ) -> Result<D::Elem, InterpolateError> {
        let i = locate_step_index(self.directions[0], data.grid[0].view(), &point[0]);
        let j = locate_step_index(self.directions[1], data.grid[1].view(), &point[1]);
        Ok(data.values[[i, j]])
    }

    /// Returns `false`.
    fn allow_extrapolate(&self) -> bool {
        false
    }
}

impl<D> Strategy2D<D> for CubicC2<D::Elem>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    fn validate(&self, data: &InterpData2DBase<D>) -> Result<(), ValidateError> {
        self.boundary_conditions
            .validate_len(2, "CubicC2", "boundary conditions")?;
        for (dim, grid) in data.grid.iter().enumerate() {
            validate_bc_min_points(&self.boundary_conditions[dim], grid.len(), dim)?;
        }
        Ok(())
    }

    /// Precomputes the full corner-derivative tensor via `compute_corner_cache`, so
    /// [`interpolate`](Self::interpolate) is an O(1) Hermite-patch lookup instead of
    /// re-solving the outer axis on every call.
    fn init(&mut self, data: &InterpData2DBase<D>) -> Result<(), ValidateError> {
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
        data: &InterpData2DBase<D>,
        point: &[D::Elem; 2],
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

impl<D> Strategy2D<D> for CubicC1<D::Elem>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    /// Precomputes the full corner-derivative tensor under [`CubicC1CacheMode::Full`]
    /// (the default); under [`CubicC1CacheMode::None`], only validates.
    fn init(&mut self, data: &InterpData2DBase<D>) -> Result<(), ValidateError> {
        if self.cache_mode == CubicC1CacheMode::Full {
            let data_view = data.view();
            self.cache = compute_corner_cache_fd(&data_view.grid, data_view.values.into_dyn());
        }
        Ok(())
    }

    fn interpolate(
        &self,
        data: &InterpData2DBase<D>,
        point: &[D::Elem; 2],
    ) -> Result<D::Elem, InterpolateError> {
        let grids: Vec<ArrayView1<D::Elem>> = data.grid.iter().map(|g| g.view()).collect();
        Ok(match self.cache_mode {
            CubicC1CacheMode::Full => {
                evaluate_spline_corner_cached(&grids, self.cache.view(), point)
            }
            CubicC1CacheMode::None => {
                evaluate_spline_corner_local(&grids, data.values.view().into_dyn(), point)
            }
        })
    }

    /// Returns `true`: the boundary Hermite patch extends naturally.
    fn allow_extrapolate(&self) -> bool {
        true
    }
}

grid_transform_strategy_impl!(Strategy2D, InterpData2DBase, InterpData2DView, 2);
values_transform_strategy_impl!(Strategy2D, InterpData2DBase, InterpData2DView, Ix2, 2);
