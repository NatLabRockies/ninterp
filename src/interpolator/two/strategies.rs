use super::*;
use strategy::*;

impl<D> Strategy2D<D> for Linear
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
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
        // y
        let y_l = lowers[1];
        let y_u = y_l + 1;

        // Short-circuit if the point lies exactly on a grid coordinate in one or both dimensions,
        // reducing value lookups from 4 to 2 or 1. find_nearest_index returns the lower bracket,
        // so exact matches appear at grid[lower] or grid[lower+1].
        let x_exact = exact_index(data.grid[0].view(), x_l, &point[0]);
        let y_exact = exact_index(data.grid[1].view(), y_l, &point[1]);
        match (x_exact, y_exact) {
            (Some(i), Some(j)) => return Ok(data.values[[i, j]]),
            (Some(i), None) => {
                let y_diff =
                    (point[1] - data.grid[1][y_l]) / (data.grid[1][y_u] - data.grid[1][y_l]);
                return Ok(data.values[[i, y_l]] * (D::Elem::one() - y_diff)
                    + data.values[[i, y_u]] * y_diff);
            }
            (None, Some(j)) => {
                let x_diff =
                    (point[0] - data.grid[0][x_l]) / (data.grid[0][x_u] - data.grid[0][x_l]);
                return Ok(data.values[[x_l, j]] * (D::Elem::one() - x_diff)
                    + data.values[[x_u, j]] * x_diff);
            }
            (None, None) => {}
        }

        let x_diff = (point[0] - data.grid[0][x_l]) / (data.grid[0][x_u] - data.grid[0][x_l]);
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

impl<D> Strategy2D<D> for LinearUniform
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    fn init(&mut self, data: &InterpData2D<D>) -> Result<(), ValidateError> {
        check_uniform_grid(data.grid[0].view(), 0)?;
        check_uniform_grid(data.grid[1].view(), 1)
    }

    fn interpolate(
        &self,
        data: &InterpData2D<D>,
        point: &[D::Elem; 2],
    ) -> Result<D::Elem, InterpolateError> {
        let x_step = data.grid[0][1] - data.grid[0][0];
        let y_step = data.grid[1][1] - data.grid[1][0];
        let x_l = uniform_lower_index(data.grid[0][0], x_step, data.grid[0].len(), point[0]);
        let y_l = uniform_lower_index(data.grid[1][0], y_step, data.grid[1].len(), point[1]);
        let x_u = x_l + 1;
        let y_u = y_l + 1;
        let x_diff = (point[0] - data.grid[0][x_l]) / x_step;
        let y_diff = (point[1] - data.grid[1][y_l]) / y_step;
        let f0 = data.values[[x_l, y_l]] * (D::Elem::one() - x_diff)
            + data.values[[x_u, y_l]] * x_diff;
        let f1 = data.values[[x_l, y_u]] * (D::Elem::one() - x_diff)
            + data.values[[x_u, y_u]] * x_diff;
        Ok(f0 * (D::Elem::one() - y_diff) + f1 * y_diff)
    }

    fn allow_extrapolate(&self) -> bool {
        true
    }
}

impl<D> Strategy2D<D> for Nearest
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
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
