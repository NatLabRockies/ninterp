use super::*;
use strategy::*;

impl<D> Strategy1D<D> for Linear
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    fn interpolate(
        &self,
        data: &InterpData1D<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError> {
        if let Some(i) = data.grid[0].iter().position(|&x_val| x_val == point[0]) {
            return Ok(data.values[i]);
        }
        // Extrapolation is checked previously in Interpolator::interpolate,
        // meaning by now, point is within grid bounds or extrapolation is enabled
        let x_l = if &point[0] < data.grid[0].first().unwrap() {
            0
        } else if &point[0] > data.grid[0].last().unwrap() {
            data.grid[0].len() - 2
        } else {
            find_nearest_index(data.grid[0].view(), &point[0])
        };
        let x_u = x_l + 1;
        let x_diff = (point[0] - data.grid[0][x_l]) / (data.grid[0][x_u] - data.grid[0][x_l]);
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
    D::Elem: Float + Debug,
{
    fn interpolate(
        &self,
        data: &InterpData1D<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError> {
        if let Some(i) = data.grid[0].iter().position(|&x_val| x_val == point[0]) {
            return Ok(data.values[i]);
        }
        let x_l = find_nearest_index(data.grid[0].view(), &point[0]);
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

impl<D> Strategy1D<D> for LeftNearest
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    fn interpolate(
        &self,
        data: &InterpData1D<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError> {
        if let Some(i) = data.grid[0].iter().position(|&x_val| x_val == point[0]) {
            return Ok(data.values[i]);
        }
        let x_l = find_nearest_index(data.grid[0].view(), &point[0]);
        Ok(data.values[x_l])
    }

    /// Returns `false`.
    fn allow_extrapolate(&self) -> bool {
        false
    }
}

impl<D> Strategy1D<D> for RightNearest
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    fn interpolate(
        &self,
        data: &InterpData1D<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError> {
        if let Some(i) = data.grid[0].iter().position(|&x_val| x_val == point[0]) {
            return Ok(data.values[i]);
        }
        let x_u = find_nearest_index(data.grid[0].view(), &point[0]) + 1;
        Ok(data.values[x_u])
    }

    /// Returns `false`.
    fn allow_extrapolate(&self) -> bool {
        false
    }
}
