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

impl<D> Strategy1D<D> for CubicC1<D::Elem>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    /// Caches the finite-difference derivative vector. `cache_mode` is ignored here:
    /// the cache is already O(1) regardless.
    fn init(&mut self, data: &InterpData1DBase<D>) -> Result<(), ValidateError> {
        self.cache = compute_fd_cache(data.grid[0].view(), data.values.view());
        Ok(())
    }

    fn interpolate(
        &self,
        data: &InterpData1DBase<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError> {
        evaluate_hermite_1d_cached(
            data.grid[0].view(),
            data.values.view(),
            self.cache.view(),
            point[0],
        )
    }

    /// Returns `true`: the boundary Hermite segment extends naturally.
    fn allow_extrapolate(&self) -> bool {
        true
    }
}

grid_transform_strategy_impl!(Strategy1D, InterpData1DBase, InterpData1DView, 1);
values_transform_strategy_impl!(Strategy1D, InterpData1DBase, InterpData1DView, Ix1, 1);
