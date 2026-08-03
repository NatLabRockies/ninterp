use super::*;
use strategy::*;

impl<D> Strategy2D<D> for Linear
where
    D: Data + RawDataClone + Clone,
    D::Elem: Num + PartialOrd + Copy + Debug,
{
    fn interpolate(
        &self,
        data: &InterpData2D<D>,
        point: &[D::Elem; 2],
    ) -> Result<D::Elem, InterpolateError> {
        // Extrapolation is checked previously in Interpolator::interpolate,
        // meaning by now, point is within grid bounds or extrapolation is enabled
        let lowers: [usize; 2] = std::array::from_fn(|dim| {
            if &point[dim] < data.grid[dim].first().unwrap() {
                0
            } else if &point[dim] > data.grid[dim].last().unwrap() {
                data.grid[dim].len() - 2
            } else {
                find_nearest_index(data.grid[dim].view(), &point[dim])
            }
        });
        // x
        let x_l = lowers[0];
        let x_u = x_l + 1;
        let x_diff = (point[0] - data.grid[0][x_l]) / (data.grid[0][x_u] - data.grid[0][x_l]);
        // y
        let y_l = lowers[1];
        let y_u = y_l + 1;
        let y_diff = (point[1] - data.grid[1][y_l]) / (data.grid[1][y_u] - data.grid[1][y_l]);
        // interpolate in the x-direction
        let f0 =
            data.values[[x_l, y_l]] * (D::Elem::one() - x_diff) + data.values[[x_u, y_l]] * x_diff;
        let f1 =
            data.values[[x_l, y_u]] * (D::Elem::one() - x_diff) + data.values[[x_u, y_u]] * x_diff;
        // interpolate in the y-direction
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
    D::Elem: Num + PartialOrd + Copy + Debug,
{
    fn interpolate(
        &self,
        data: &InterpData2D<D>,
        point: &[D::Elem; 2],
    ) -> Result<D::Elem, InterpolateError> {
        // x
        let x_l = find_nearest_index(data.grid[0].view(), &point[0]);
        let x_u = x_l + 1;
        let i = if point[0] - data.grid[0][x_l] < data.grid[0][x_u] - point[0] {
            x_l
        } else {
            x_u
        };
        // y
        let y_l = find_nearest_index(data.grid[1].view(), &point[1]);
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
    D::Elem: Num + PartialOrd + Copy + Debug,
{
    fn init(&mut self, _data: &InterpData2D<D>) -> Result<(), ValidateError> {
        if self.0.len() != 1 && self.0.len() != 2 {
            return Err(ValidateError::Other(format!(
                "Step strategy has {} directions but interpolator is 2-D (expected 1 or 2)",
                self.0.len()
            )));
        }
        Ok(())
    }

    fn interpolate(
        &self,
        data: &InterpData2D<D>,
        point: &[D::Elem; 2],
    ) -> Result<D::Elem, InterpolateError> {
        let i = step_index(self.dir(0), data.grid[0].view(), &point[0]);
        let j = step_index(self.dir(1), data.grid[1].view(), &point[1]);
        Ok(data.values[[i, j]])
    }

    fn allow_extrapolate(&self) -> bool {
        false
    }
}
