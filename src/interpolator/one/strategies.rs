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
        validate_uniform_grid(data.grid[0].view(), 0, None)
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
        self.validate_len(1)
    }

    fn interpolate(
        &self,
        data: &InterpData1DBase<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError> {
        Ok(data.values[locate_step_index(self.dir(0), data.grid[0].view(), &point[0])])
    }

    fn allow_extrapolate(&self) -> bool {
        false
    }
}

impl<D> Strategy1D<D> for CubicSpline<D::Elem>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    /// Computes and caches `M[0..=n]` using [`compute_m`] for the configured BC.
    fn init(&mut self, data: &InterpData1DBase<D>) -> Result<(), ValidateError> {
        let new_m = compute_m(data.grid[0].view(), data.values.view(), self.bc_for_dim(0))?;
        self.m = new_m;
        Ok(())
    }

    fn interpolate(
        &self,
        data: &InterpData1DBase<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError> {
        let grid = data.grid[0].view();
        // locate_lower_index already clamps to the boundary interval for extrapolation.
        let i = locate_lower_index(grid, &point[0]);
        let two = D::Elem::one() + D::Elem::one();
        let six = two + two + two;
        let h = grid[i + 1] - grid[i];
        let dx = point[0] - grid[i]; // t - x[i]
        let dx_r = h - dx; // x[i+1] - t
        let six_h = six * h;
        let h2_over_six = h * h / six;
        Ok(self.m[i] * dx_r * dx_r * dx_r / six_h
            + self.m[i + 1] * dx * dx * dx / six_h
            + (data.values[i] - self.m[i] * h2_over_six) * dx_r / h
            + (data.values[i + 1] - self.m[i + 1] * h2_over_six) * dx / h)
    }

    /// Returns `true`: the boundary cubic polynomials extend naturally.
    fn allow_extrapolate(&self) -> bool {
        true
    }
}

impl<D> Strategy1D<D> for StepLower
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialOrd + Copy + Debug,
{
    fn interpolate(
        &self,
        data: &InterpData1DBase<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError> {
        Ok(data.values[locate_step_index(StepDirection::Lower, data.grid[0].view(), &point[0])])
    }

    /// Returns `false`.
    fn allow_extrapolate(&self) -> bool {
        false
    }
}

impl<D> Strategy1D<D> for StepUpper
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialOrd + Copy + Debug,
{
    fn interpolate(
        &self,
        data: &InterpData1DBase<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError> {
        Ok(data.values[locate_step_index(StepDirection::Upper, data.grid[0].view(), &point[0])])
    }

    /// Returns `false`.
    fn allow_extrapolate(&self) -> bool {
        false
    }
}
