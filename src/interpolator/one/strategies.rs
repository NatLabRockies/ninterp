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

impl<D> Strategy1D<D> for LinearUniform
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    /// Ensures the grid is uniformly spaced.
    fn init(&mut self, data: &InterpData1D<D>) -> Result<(), ValidateError> {
        check_uniform_grid(data.grid[0].view(), 0)
    }

    fn interpolate(
        &self,
        data: &InterpData1D<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError> {
        let grid = data.grid[0].view();
        let step = grid[1] - grid[0];
        let x_l = uniform_lower_index(grid[0], step, grid.len(), point[0]);
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
    D::Elem: Float + Debug,
{
    fn interpolate(
        &self,
        data: &InterpData1D<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError> {
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

impl<D> Strategy1D<D> for Step
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    /// Ensures the number of provided step directions matches the interpolator dimensionality.
    fn init(&mut self, _data: &InterpData1D<D>) -> Result<(), ValidateError> {
        if self.0.len() != 1 {
            return Err(ValidateError::Other(format!(
                "Step strategy has {} directions but interpolator is 1-D (expected 1)",
                self.0.len()
            )));
        }
        Ok(())
    }

    fn interpolate(
        &self,
        data: &InterpData1D<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError> {
        Ok(data.values[step_index(self.dir(0), data.grid[0].view(), &point[0])])
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
    /// Computes and caches the per-interval spline coefficients.
    ///
    /// Uses a natural cubic spline (zero second derivative at endpoints).
    /// Solves the resulting tridiagonal system via the Thomas algorithm.
    fn init(&mut self, data: &InterpData1D<D>) -> Result<(), ValidateError> {
        let x = data.grid[0].view();
        let y = data.values.view();
        let n = x.len() - 1; // number of intervals

        let two = D::Elem::one() + D::Elem::one();
        let six = two + two + two;

        // h[i] = x[i+1] - x[i]
        let h: Vec<D::Elem> = (0..n).map(|i| x[i + 1] - x[i]).collect();

        // Solve for interior second derivatives M[1..n-1] (natural BCs: M[0] = M[n] = 0).
        // System size m = n-1; for n=1 (two points) the system is empty and the spline
        // degenerates to linear interpolation.
        let m = n - 1;
        let mut m_inner = vec![D::Elem::zero(); m];

        if m > 0 {
            // RHS for equation k (unknown = M[k+1]):
            // 6 * ((y[k+2]-y[k+1])/h[k+1] - (y[k+1]-y[k])/h[k])
            let rhs: Vec<D::Elem> = (0..m)
                .map(|k| six * ((y[k + 2] - y[k + 1]) / h[k + 1] - (y[k + 1] - y[k]) / h[k]))
                .collect();

            // Thomas algorithm (tridiagonal solve).
            // Sub-diagonal: h[k], main: 2*(h[k]+h[k+1]), super: h[k+1].
            let mut cp = vec![D::Elem::zero(); m]; // modified super-diagonal
            let mut dp = vec![D::Elem::zero(); m]; // modified RHS

            let w0 = two * (h[0] + h[1]);
            cp[0] = if m > 1 { h[1] / w0 } else { D::Elem::zero() };
            dp[0] = rhs[0] / w0;

            for k in 1..m {
                let w = two * (h[k] + h[k + 1]) - h[k] * cp[k - 1];
                cp[k] = if k < m - 1 {
                    h[k + 1] / w
                } else {
                    D::Elem::zero()
                };
                dp[k] = (rhs[k] - h[k] * dp[k - 1]) / w;
            }

            m_inner[m - 1] = dp[m - 1];
            for k in (0..m - 1).rev() {
                m_inner[k] = dp[k] - cp[k] * m_inner[k + 1];
            }
        }

        // M[i]: M[0]=0, M[1..n-1]=m_inner[0..m-1], M[n]=0.
        // For interval i, M[i] = m_inner[i-1] (0 at boundaries).
        let m_at = |i: usize| -> D::Elem {
            if i == 0 || i == n {
                D::Elem::zero()
            } else {
                m_inner[i - 1]
            }
        };

        // S_i(x) = y[i] + b[i]*dx + c[i]*dx^2 + d[i]*dx^3  where dx = x - x[i]
        self.b = (0..n)
            .map(|i| (y[i + 1] - y[i]) / h[i] - h[i] * (two * m_at(i) + m_at(i + 1)) / six)
            .collect();
        self.c = (0..n).map(|i| m_at(i) / two).collect();
        self.d = (0..n)
            .map(|i| (m_at(i + 1) - m_at(i)) / (six * h[i]))
            .collect();

        Ok(())
    }

    fn interpolate(
        &self,
        data: &InterpData1D<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError> {
        let grid = data.grid[0].view();
        // For extrapolation, clamp to the boundary interval.
        let i = if &point[0] < grid.first().unwrap() {
            0
        } else if &point[0] > grid.last().unwrap() {
            grid.len() - 2
        } else {
            find_nearest_index(grid, &point[0])
        };
        let dx = point[0] - grid[i];
        Ok(data.values[i] + self.b[i] * dx + self.c[i] * dx * dx + self.d[i] * dx * dx * dx)
    }

    /// Returns `true`: the boundary cubic polynomials extend naturally.
    fn allow_extrapolate(&self) -> bool {
        true
    }
}
