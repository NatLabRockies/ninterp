use super::*;
use strategy::*;

impl<D> Strategy3D<D> for Linear
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    fn interpolate(
        &self,
        data: &InterpData3D<D>,
        point: &[D::Elem; 3],
    ) -> Result<D::Elem, InterpolateError> {
        // Extrapolation is checked previously in Interpolator::interpolate,
        // meaning by now, point is within grid bounds or extrapolation is enabled
        let lowers: [usize; 3] = std::array::from_fn(|dim| {
            if &point[dim] < data.grid[dim].first().unwrap() {
                0
            } else if &point[dim] > data.grid[dim].last().unwrap() {
                data.grid[dim].len() - 2
            } else {
                find_nearest_index(data.grid[dim].view(), &point[dim])
            }
        });
        let x_l = lowers[0];
        let x_u = x_l + 1;
        let y_l = lowers[1];
        let y_u = y_l + 1;
        let z_l = lowers[2];
        let z_u = z_l + 1;

        // Short-circuit if the point lies exactly on a grid coordinate in one or more dimensions,
        // reducing value lookups from 8 down to 4, 2, or 1.
        let x_exact = exact_index(data.grid[0].view(), x_l, &point[0]);
        let y_exact = exact_index(data.grid[1].view(), y_l, &point[1]);
        let z_exact = exact_index(data.grid[2].view(), z_l, &point[2]);
        match (x_exact, y_exact, z_exact) {
            (Some(i), Some(j), Some(k)) => return Ok(data.values[[i, j, k]]),
            (Some(i), Some(j), None) => {
                let z_diff =
                    (point[2] - data.grid[2][z_l]) / (data.grid[2][z_u] - data.grid[2][z_l]);
                return Ok(data.values[[i, j, z_l]] * (D::Elem::one() - z_diff)
                    + data.values[[i, j, z_u]] * z_diff);
            }
            (Some(i), None, Some(k)) => {
                let y_diff =
                    (point[1] - data.grid[1][y_l]) / (data.grid[1][y_u] - data.grid[1][y_l]);
                return Ok(data.values[[i, y_l, k]] * (D::Elem::one() - y_diff)
                    + data.values[[i, y_u, k]] * y_diff);
            }
            (None, Some(j), Some(k)) => {
                let x_diff =
                    (point[0] - data.grid[0][x_l]) / (data.grid[0][x_u] - data.grid[0][x_l]);
                return Ok(data.values[[x_l, j, k]] * (D::Elem::one() - x_diff)
                    + data.values[[x_u, j, k]] * x_diff);
            }
            (Some(i), None, None) => {
                let y_diff =
                    (point[1] - data.grid[1][y_l]) / (data.grid[1][y_u] - data.grid[1][y_l]);
                let z_diff =
                    (point[2] - data.grid[2][z_l]) / (data.grid[2][z_u] - data.grid[2][z_l]);
                let f0 = data.values[[i, y_l, z_l]] * (D::Elem::one() - y_diff)
                    + data.values[[i, y_u, z_l]] * y_diff;
                let f1 = data.values[[i, y_l, z_u]] * (D::Elem::one() - y_diff)
                    + data.values[[i, y_u, z_u]] * y_diff;
                return Ok(f0 * (D::Elem::one() - z_diff) + f1 * z_diff);
            }
            (None, Some(j), None) => {
                let x_diff =
                    (point[0] - data.grid[0][x_l]) / (data.grid[0][x_u] - data.grid[0][x_l]);
                let z_diff =
                    (point[2] - data.grid[2][z_l]) / (data.grid[2][z_u] - data.grid[2][z_l]);
                let f0 = data.values[[x_l, j, z_l]] * (D::Elem::one() - x_diff)
                    + data.values[[x_u, j, z_l]] * x_diff;
                let f1 = data.values[[x_l, j, z_u]] * (D::Elem::one() - x_diff)
                    + data.values[[x_u, j, z_u]] * x_diff;
                return Ok(f0 * (D::Elem::one() - z_diff) + f1 * z_diff);
            }
            (None, None, Some(k)) => {
                let x_diff =
                    (point[0] - data.grid[0][x_l]) / (data.grid[0][x_u] - data.grid[0][x_l]);
                let y_diff =
                    (point[1] - data.grid[1][y_l]) / (data.grid[1][y_u] - data.grid[1][y_l]);
                let f0 = data.values[[x_l, y_l, k]] * (D::Elem::one() - x_diff)
                    + data.values[[x_u, y_l, k]] * x_diff;
                let f1 = data.values[[x_l, y_u, k]] * (D::Elem::one() - x_diff)
                    + data.values[[x_u, y_u, k]] * x_diff;
                return Ok(f0 * (D::Elem::one() - y_diff) + f1 * y_diff);
            }
            (None, None, None) => {}
        }

        let x_diff = (point[0] - data.grid[0][x_l]) / (data.grid[0][x_u] - data.grid[0][x_l]);
        // y
        let y_diff = (point[1] - data.grid[1][y_l]) / (data.grid[1][y_u] - data.grid[1][y_l]);
        // z
        let z_diff = (point[2] - data.grid[2][z_l]) / (data.grid[2][z_u] - data.grid[2][z_l]);
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
    fn init(&mut self, data: &InterpData3D<D>) -> Result<(), ValidateError> {
        check_uniform_grid(data.grid[0].view(), 0)?;
        check_uniform_grid(data.grid[1].view(), 1)?;
        check_uniform_grid(data.grid[2].view(), 2)
    }

    fn interpolate(
        &self,
        data: &InterpData3D<D>,
        point: &[D::Elem; 3],
    ) -> Result<D::Elem, InterpolateError> {
        let x_step = data.grid[0][1] - data.grid[0][0];
        let y_step = data.grid[1][1] - data.grid[1][0];
        let z_step = data.grid[2][1] - data.grid[2][0];
        let x_l = uniform_lower_index(data.grid[0][0], x_step, data.grid[0].len(), point[0]);
        let y_l = uniform_lower_index(data.grid[1][0], y_step, data.grid[1].len(), point[1]);
        let z_l = uniform_lower_index(data.grid[2][0], z_step, data.grid[2].len(), point[2]);
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
    D::Elem: Float + Debug,
{
    fn interpolate(
        &self,
        data: &InterpData3D<D>,
        point: &[D::Elem; 3],
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
        // z
        let z_l = find_nearest_index(data.grid[2].view(), &point[2]);
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
    D::Elem: Float + Debug,
{
    /// Ensures the number of provided step directions matches the interpolator dimensionality.
    fn init(&mut self, _data: &InterpData3D<D>) -> Result<(), ValidateError> {
        if self.0.len() != 1 && self.0.len() != 3 {
            return Err(ValidateError::Other(format!(
                "Step strategy has {} directions but interpolator is 3-D (expected 1 or 3)",
                self.0.len()
            )));
        }
        Ok(())
    }

    fn interpolate(
        &self,
        data: &InterpData3D<D>,
        point: &[D::Elem; 3],
    ) -> Result<D::Elem, InterpolateError> {
        let i = step_index(self.dir(0), data.grid[0].view(), &point[0]);
        let j = step_index(self.dir(1), data.grid[1].view(), &point[1]);
        let k = step_index(self.dir(2), data.grid[2].view(), &point[2]);
        Ok(data.values[[i, j, k]])
    }

    /// Returns `false`.
    fn allow_extrapolate(&self) -> bool {
        false
    }
}

impl<D> Strategy3D<D> for CubicSpline<D::Elem>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    fn interpolate(
        &self,
        data: &InterpData3D<D>,
        point: &[D::Elem; 3],
    ) -> Result<D::Elem, InterpolateError> {
        let grids: Vec<ArrayView1<D::Elem>> = data.grid.iter().map(|g| g.view()).collect();
        spline_eval_nd_recursive(
            &grids,
            data.values.view().into_dyn(),
            point,
            &self.boundary_conditions,
        )
    }

    /// Returns `true`: the boundary cubic polynomials extend naturally.
    fn allow_extrapolate(&self) -> bool {
        true
    }
}
