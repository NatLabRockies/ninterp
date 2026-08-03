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
        let x_diff = (point[0] - data.grid[0][x_l]) / (data.grid[0][x_u] - data.grid[0][x_l]);
        // y
        let y_l = lowers[1];
        let y_u = y_l + 1;
        let y_diff = (point[1] - data.grid[1][y_l]) / (data.grid[1][y_u] - data.grid[1][y_l]);
        // z
        let z_l = lowers[2];
        let z_u = z_l + 1;
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
