use super::*;
use strategy::*;

/// Casts a single grid coordinate to `f64`, the shared blend/query precision every
/// ND strategy computes in (see [`StrategyND`]'s docs).
fn to_f64<T: NumCast>(x: T) -> f64 {
    num_traits::cast(x).expect("grid element must cast to f64")
}

/// Casts a whole grid axis to `f64`, for a strategy that needs to do real arithmetic
/// across the axis (`LinearUniform`'s uniformity check, `CubicC2`'s corner cache,
/// `GridTransform`'s forward transform) rather than a single per-probe comparison
/// (which `locate_lower_index_cast`/`exact_index_cast`/`locate_step_index_cast`
/// handle without this allocation).
fn grid_to_f64<T: NumCast + Copy>(grid: ArrayView1<T>) -> Array1<f64> {
    grid.iter().map(|&x| to_f64(x)).collect()
}

/// Casts a blended `f64` result back down to `Dv::Elem`, for the checked (`Result`
/// returning) interpolation path.
///
/// Known limitation: this goes through `NumCast`, whose float-to-integer conversion
/// truncates toward zero (`as`'s own semantics) rather than rounding to nearest. So
/// for an integer `Dv`, a blend that lands on e.g. `15.5` casts down to `15`, not a
/// rounded `16`, silently biasing every non-exact blend downward. Fixing this
/// properly needs a type-aware cast (round for integer `Dv`, exact passthrough for a
/// float `Dv`, since rounding *that* would wrongly destroy real fractional output),
/// deferred for now; revisit alongside a real `Tp` type if this prototype proceeds.
fn from_f64_checked<Tv: NumCast>(x: f64) -> Result<Tv, InterpolateError> {
    num_traits::cast(x)
        .ok_or_else(|| InterpolateError::Other("blended value doesn't fit in value type".into()))
}

impl<Dg, Dv> StrategyND<Dg, Dv> for Linear
where
    Dg: Data + RawDataClone + Clone,
    Dg::Elem: NumCast + PartialOrd + Copy + Debug,
    Dv: Data + RawDataClone + Clone,
    Dv::Elem: NumCast + Copy + Debug + PartialEq,
{
    fn interpolate(
        &self,
        data: &InterpDataNDBase<Dg, Dv>,
        point: &[f64],
    ) -> Result<Dv::Elem, InterpolateError> {
        // Dimensionality
        let mut n = data.values.ndim();

        // Point can share up to N values of a grid point, which reduces the problem dimensionality
        // i.e. the point shares one of three values of a 3-D grid point, then the interpolation becomes 2-D at that slice
        // or   if the point shares two of three values of a 3-D grid point, then the interpolation becomes 1-D
        let mut point = point.to_owned();
        let mut grid: Vec<_> = data.grid.iter().map(|arr| arr.view()).collect();
        let mut values_view = data.values.view();
        for dim in (0..n).rev() {
            // Skip empty grid dimensions (e.g. the 0-D multilinear case uses an empty grid).
            if grid[dim].is_empty() {
                continue;
            }
            let lower = locate_lower_index_cast(grid[dim], point[dim]);
            let pos = exact_index_cast(grid[dim], lower, point[dim]);
            if let Some(pos) = pos {
                point.remove(dim);
                grid.remove(dim);
                values_view.index_axis_inplace(Axis(dim), pos);
            }
        }
        if values_view.len() == 1 {
            // Supplied point is coincident with a grid point, so just return the value
            // directly: no blending, so no precision lost to the f64 round trip below.
            return Ok(values_view.first().copied().unwrap());
        }
        // Simplified dimensionality
        n = values_view.ndim();

        // Extract the lower and upper indices for each dimension,
        // as well as the fraction of how far the supplied point is between the surrounding grid points
        let mut lower_idxs = Vec::with_capacity(n);
        let mut interp_diffs = Vec::with_capacity(n);
        for dim in 0..n {
            // Extrapolation is checked previously in Interpolator::interpolate,
            // meaning by now, point is within grid bounds or extrapolation is enabled
            let lower_idx = locate_lower_index_cast(grid[dim], point[dim]);
            let g_lower = to_f64(grid[dim][lower_idx]);
            let g_upper = to_f64(grid[dim][lower_idx + 1]);
            let interp_diff = (point[dim] - g_lower) / (g_upper - g_lower);
            lower_idxs.push(lower_idx);
            interp_diffs.push(interp_diff);
        }
        // Fill all 2^n corner values into a flat array indexed by bitmask, blending in
        // f64 (the shared precision, independent of Dv) so an integer-ish Dv (e.g. an
        // image's u8 pixel values) doesn't lose precision to repeated integer blends.
        let size = 1usize << n;
        let mut vals = vec![0f64; size];
        let mut idx = vec![0usize; n];
        for (mask, val) in vals.iter_mut().enumerate() {
            for d in 0..n {
                idx[d] = lower_idxs[d] + ((mask >> (n - 1 - d)) & 1);
            }
            *val = to_f64(values_view[idx.as_slice()]);
        }

        // Butterfly reduction: one pass per dimension.
        for (d, diff) in interp_diffs.iter().enumerate() {
            let half = 1 << (n - 1 - d);
            for i in 0..half {
                vals[i] = vals[i] * (1.0 - *diff) + vals[i + half] * *diff;
            }
        }

        from_f64_checked(vals[0])
    }

    /// Returns `true`.
    fn allow_extrapolate(&self) -> bool {
        true
    }
}

impl<Dg, Dv> StrategyND<Dg, Dv> for LinearUniform
where
    Dg: Data + RawDataClone + Clone,
    Dg::Elem: NumCast + PartialOrd + Copy + Debug,
    Dv: Data + RawDataClone + Clone,
    Dv::Elem: NumCast + Copy + Debug + PartialEq,
{
    /// Ensures grid uniformity in all dimensions
    fn validate(&self, data: &InterpDataNDBase<Dg, Dv>) -> Result<(), ValidateError> {
        for (dim, grid) in data.grid.iter().enumerate() {
            let grid_f64 = grid_to_f64(grid.view());
            validate_uniform_grid_epsilon(grid_f64.view(), dim, None)?;
        }
        Ok(())
    }

    fn interpolate(
        &self,
        data: &InterpDataNDBase<Dg, Dv>,
        point: &[f64],
    ) -> Result<Dv::Elem, InterpolateError> {
        let n = data.values.ndim();
        let mut lower_idxs = Vec::with_capacity(n);
        let mut interp_diffs = Vec::with_capacity(n);
        for (grid_dim, &point_dim) in data.grid.iter().zip(point.iter()) {
            let g0 = to_f64(grid_dim[0]);
            let g1 = to_f64(grid_dim[1]);
            let step = g1 - g0;
            let lower_idx = locate_lower_index_uniform(g0, step, grid_dim.len(), point_dim);
            let g_lower = to_f64(grid_dim[lower_idx]);
            let diff = (point_dim - g_lower) / step;
            lower_idxs.push(lower_idx);
            interp_diffs.push(diff);
        }
        // Same bitmask/butterfly reduction as Linear ND
        let size = 1usize << n;
        let mut vals = vec![0f64; size];
        let mut idx = vec![0usize; n];
        for (mask, val) in vals.iter_mut().enumerate() {
            for d in 0..n {
                idx[d] = lower_idxs[d] + ((mask >> (n - 1 - d)) & 1);
            }
            *val = to_f64(data.values.view()[idx.as_slice()]);
        }
        for (d, diff) in interp_diffs.iter().enumerate() {
            let half = 1 << (n - 1 - d);
            for i in 0..half {
                vals[i] = vals[i] * (1.0 - *diff) + vals[i + half] * *diff;
            }
        }
        from_f64_checked(vals[0])
    }

    /// Returns `true`.
    fn allow_extrapolate(&self) -> bool {
        true
    }
}

impl<Dg, Dv> StrategyND<Dg, Dv> for Nearest
where
    Dg: Data + RawDataClone + Clone,
    Dg::Elem: NumCast + PartialOrd + Copy + Debug,
    Dv: Data + RawDataClone + Clone,
    Dv::Elem: Copy + Debug + PartialEq,
{
    fn interpolate(
        &self,
        data: &InterpDataNDBase<Dg, Dv>,
        point: &[f64],
    ) -> Result<Dv::Elem, InterpolateError> {
        let n = data.values.ndim();
        // Nearest-neighbor on a rectilinear grid factorizes: select the nearest index
        // independently per dimension, then do a single lookup: a direct, uncast
        // `Dv::Elem` read, so no precision is lost regardless of `Dv`.
        let mut idx = vec![0usize; n];
        for dim in 0..n {
            let lower_idx = locate_lower_index_cast(data.grid[dim].view(), point[dim]);
            let lo = to_f64(data.grid[dim][lower_idx]);
            let hi = to_f64(data.grid[dim][lower_idx + 1]);
            idx[dim] = if point[dim] - lo < hi - point[dim] {
                lower_idx
            } else {
                lower_idx + 1
            };
        }
        Ok(data.values.view()[idx.as_slice()])
    }

    /// Returns `false`.
    fn allow_extrapolate(&self) -> bool {
        false
    }
}

impl<Dg, Dv> StrategyND<Dg, Dv> for Step
where
    Dg: Data + RawDataClone + Clone,
    Dg::Elem: NumCast + PartialOrd + Copy + Debug,
    Dv: Data + RawDataClone + Clone,
    Dv::Elem: Copy + Debug + PartialEq,
{
    /// Ensures the number of provided step directions matches the dimensionality of the interpolator
    fn validate(&self, data: &InterpDataNDBase<Dg, Dv>) -> Result<(), ValidateError> {
        self.directions
            .validate_len(data.values.ndim(), "Step", "directions")
    }

    fn interpolate(
        &self,
        data: &InterpDataNDBase<Dg, Dv>,
        point: &[f64],
    ) -> Result<Dv::Elem, InterpolateError> {
        let n = data.values.ndim();
        let mut idx = vec![0usize; n];
        for dim in 0..n {
            idx[dim] =
                locate_step_index_cast(self.directions[dim], data.grid[dim].view(), point[dim]);
        }
        Ok(data.values.view()[idx.as_slice()])
    }

    /// Returns `false`.
    fn allow_extrapolate(&self) -> bool {
        false
    }
}

impl<Dg, Dv> StrategyND<Dg, Dv> for CubicC2<f64>
where
    Dg: Data + RawDataClone + Clone,
    Dg::Elem: NumCast + PartialOrd + Copy + Debug,
    Dv: Data + RawDataClone + Clone,
    Dv::Elem: NumCast + Copy + Debug + PartialEq,
{
    fn validate(&self, data: &InterpDataNDBase<Dg, Dv>) -> Result<(), ValidateError> {
        self.boundary_conditions
            .validate_len(data.ndim(), "CubicC2", "boundary conditions")?;
        for dim in 0..data.ndim() {
            validate_bc_min_points(&self.boundary_conditions[dim], data.grid[dim].len(), dim)?;
        }
        Ok(())
    }

    /// Precomputes the full corner-derivative tensor via `compute_corner_cache`,
    /// casting the grid and values to `f64` first (the tensor is a blend
    /// intermediate, like `Linear`'s corner blend, so it's `f64`-typed regardless of
    /// `Dg`/`Dv`). A one-time cost per `init`, not per query.
    fn init(&mut self, data: &InterpDataNDBase<Dg, Dv>) -> Result<(), ValidateError> {
        if data.ndim() == 0 {
            return Ok(());
        }
        let grids_f64: Vec<Array1<f64>> = data.grid.iter().map(|g| grid_to_f64(g.view())).collect();
        let grid_views: Vec<ArrayView1<f64>> = grids_f64.iter().map(|g| g.view()).collect();
        let values_f64: ArrayD<f64> = data.values.map(|&x| to_f64(x));
        self.cache =
            compute_corner_cache(&grid_views, values_f64.view(), &self.boundary_conditions);
        Ok(())
    }

    fn interpolate(
        &self,
        data: &InterpDataNDBase<Dg, Dv>,
        point: &[f64],
    ) -> Result<Dv::Elem, InterpolateError> {
        if data.ndim() == 0 {
            return data.values.first().copied().ok_or_else(|| {
                InterpolateError::Other("internal: 0-D interpolation data has no value".into())
            });
        }
        // Re-cast the grid to `f64` per query: `CubicC2`'s own struct (shared with
        // `Interp1D`/`2D`/`3D`) isn't touched by this prototype, so there's no spare
        // field here to cache it in the way `GridTransform::grid_cache` does.
        let grids_f64: Vec<Array1<f64>> = data.grid.iter().map(|g| grid_to_f64(g.view())).collect();
        let grid_views: Vec<ArrayView1<f64>> = grids_f64.iter().map(|g| g.view()).collect();
        let result = evaluate_spline_corner_cached(&grid_views, self.cache.view(), point);
        from_f64_checked(result)
    }

    /// Returns `true`: the boundary cubic polynomials extend naturally.
    fn allow_extrapolate(&self) -> bool {
        true
    }
}

impl<Dg, Dv, S> StrategyND<Dg, Dv> for GridTransform<f64, S>
where
    Dg: Data + RawDataClone + Clone,
    Dg::Elem: NumCast + PartialOrd + Copy + Debug,
    Dv: Data + RawDataClone + Clone,
    Dv::Elem: PartialEq + Debug,
    // Both sides are view-repr'd here: `data.values.view()` produces a view
    // regardless of what `Dv` itself is (owned or already a view), matching how the
    // pre-split code bounded this the same way on its single shared type param.
    S: for<'a> StrategyND<ViewRepr<&'a f64>, ViewRepr<&'a Dv::Elem>> + Clone + Debug,
{
    /// Checks the axis count and that every raw grid coordinate (cast to `f64`) is
    /// in its axis's configured transform's domain.
    fn validate(&self, data: &InterpDataNDBase<Dg, Dv>) -> Result<(), ValidateError> {
        self.transforms
            .validate_len(data.ndim(), "GridTransform", "transforms")?;
        let transformed_grid: Vec<Array1<f64>> = data
            .grid
            .iter()
            .enumerate()
            .map(|(dim, grid)| self.transform_axis(dim, grid_to_f64(grid.view()).view()))
            .collect::<Result<_, _>>()?;
        let values = self.transformed_values_view(data.values.view());
        let view = InterpDataNDBase {
            grid: transformed_grid.iter().map(|g| g.view()).collect(),
            values,
        };
        self.inner.validate(&view)
    }

    /// Transforms the (`f64`-cast) grid into `grid_cache`, then initializes `inner`
    /// against a transient view zipping `grid_cache` with `data.values` (untouched;
    /// `GridTransform` is grid-side only).
    fn init(&mut self, data: &InterpDataNDBase<Dg, Dv>) -> Result<(), ValidateError> {
        self.grid_cache = data
            .grid
            .iter()
            .enumerate()
            .map(|(dim, grid)| self.transform_axis(dim, grid_to_f64(grid.view()).view()))
            .collect::<Result<_, _>>()?;
        let values = self.transformed_values_view(data.values.view());
        let view = InterpDataNDBase {
            grid: self.grid_cache.iter().map(|g| g.view()).collect(),
            values,
        };
        self.inner.init(&view)
    }

    fn interpolate(
        &self,
        data: &InterpDataNDBase<Dg, Dv>,
        point: &[f64],
    ) -> Result<Dv::Elem, InterpolateError> {
        self.check_point_domain(point)?;
        let transformed_point: Vec<f64> = point
            .iter()
            .enumerate()
            .map(|(dim, &x)| self.transforms[dim].forward(x))
            .collect();
        let values = self.transformed_values_view(data.values.view());
        let view = InterpDataNDBase {
            grid: self.grid_cache.iter().map(|g| g.view()).collect(),
            values,
        };
        self.inner.interpolate(&view, &transformed_point)
    }

    fn interpolate_wrapped(
        &self,
        data: &InterpDataNDBase<Dg, Dv>,
        point: &[f64],
    ) -> Result<Dv::Elem, InterpolateError>
    where
        Dg::Elem: NumCast + Copy,
    {
        self.check_point_domain(point)?;
        let transformed_point: Vec<f64> = point
            .iter()
            .enumerate()
            .map(|(dim, &x)| self.transforms[dim].forward(x))
            .collect();
        let values = self.transformed_values_view(data.values.view());
        let view = InterpDataNDBase {
            grid: self.grid_cache.iter().map(|g| g.view()).collect(),
            values,
        };
        self.inner.interpolate_wrapped(&view, &transformed_point)
    }

    fn interpolate_fast(&self, data: &InterpDataNDBase<Dg, Dv>, point: &[f64]) -> Dv::Elem {
        let transformed_point: Vec<f64> = point
            .iter()
            .enumerate()
            .map(|(dim, &x)| self.transforms[dim].forward(x))
            .collect();
        let values = self.transformed_values_view(data.values.view());
        let view = InterpDataNDBase {
            grid: self.grid_cache.iter().map(|g| g.view()).collect(),
            values,
        };
        self.inner.interpolate_fast(&view, &transformed_point)
    }

    fn batch_interpolate_into(
        &self,
        data: &InterpDataNDBase<Dg, Dv>,
        points: &[&[f64]],
        out: &mut [Dv::Elem],
    ) -> Result<(), InterpolateError> {
        if out.len() != points.len() {
            return Err(InterpolateError::OutputLength {
                expected: points.len(),
                found: out.len(),
            });
        }
        self.check_batch_domain(points.iter().copied())?;
        let transformed_points: Vec<Vec<f64>> = points
            .iter()
            .map(|point| {
                point
                    .iter()
                    .enumerate()
                    .map(|(dim, &x)| self.transforms[dim].forward(x))
                    .collect()
            })
            .collect();
        let transformed_refs: Vec<&[f64]> = transformed_points.iter().map(Vec::as_slice).collect();
        let values = self.transformed_values_view(data.values.view());
        let view = InterpDataNDBase {
            grid: self.grid_cache.iter().map(|g| g.view()).collect(),
            values,
        };
        self.inner
            .batch_interpolate_into(&view, &transformed_refs, out)
    }

    fn check_batch_domain(&self, points: &[&[f64]]) -> Result<(), InterpolateError> {
        let mut failures = Vec::new();
        let mut valid_indices = Vec::new();
        let mut transformed_points: Vec<Vec<f64>> = Vec::new();
        for (index, &point) in points.iter().enumerate() {
            let point_failures = self.point_domain_failures(index, point);
            if point_failures.is_empty() {
                valid_indices.push(index);
                transformed_points.push(
                    point
                        .iter()
                        .enumerate()
                        .map(|(dim, &x)| self.transforms[dim].forward(x))
                        .collect(),
                );
            } else {
                failures.extend(point_failures);
            }
        }
        let transformed_refs: Vec<&[f64]> = transformed_points.iter().map(Vec::as_slice).collect();
        match self.inner.check_batch_domain(&transformed_refs) {
            Ok(()) => {}
            Err(InterpolateError::GridTransformDomain(inner_failures)) => {
                failures.extend(inner_failures.into_iter().map(|f| OutsideDomainAt {
                    index: valid_indices[f.index],
                    ..f
                }));
            }
            Err(other) => return Err(other),
        }
        failures.sort_by_key(|f| (f.index, f.dim));
        if failures.is_empty() {
            Ok(())
        } else {
            Err(InterpolateError::GridTransformDomain(failures))
        }
    }

    fn allow_extrapolate(&self) -> bool {
        self.inner.allow_extrapolate()
    }
}

impl<Dg, Dv, S> StrategyND<Dg, Dv> for ValuesTransform<f64, S>
where
    Dg: Data + RawDataClone + Clone,
    Dg::Elem: PartialEq + Debug,
    Dv: Data + RawDataClone + Clone,
    Dv::Elem: NumCast + Copy + Debug + PartialEq,
    // Both sides are view-repr'd here: `data.grid[i].view()` produces a view
    // regardless of what `Dg` itself is, matching how the pre-split code bounded
    // this the same way on its single shared type param.
    S: for<'a> StrategyND<ViewRepr<&'a Dg::Elem>, ViewRepr<&'a f64>> + Clone + Debug,
{
    /// Checks that every data value (cast to `f64`) is in the configured transform's
    /// domain.
    fn validate(&self, data: &InterpDataNDBase<Dg, Dv>) -> Result<(), ValidateError> {
        let values_f64: ArrayD<f64> = data.values.map(|&x| to_f64(x));
        let transformed_values = self.transform_values(values_f64.view())?;
        let view = InterpDataNDBase {
            grid: data.grid.iter().map(|g| g.view()).collect(),
            values: transformed_values.view(),
        };
        self.inner.validate(&view)
    }

    /// Transforms `data.values` (cast to `f64`) into `values_cache`, then
    /// initializes `inner` against a transient view zipping `data.grid` (untouched;
    /// `ValuesTransform` is value-side only) with `values_cache`.
    fn init(&mut self, data: &InterpDataNDBase<Dg, Dv>) -> Result<(), ValidateError> {
        let values_f64: ArrayD<f64> = data.values.map(|&x| to_f64(x));
        self.values_cache = self.transform_values(values_f64.view())?;
        let view = InterpDataNDBase {
            grid: data.grid.iter().map(|g| g.view()).collect(),
            values: self.values_cache.view(),
        };
        self.inner.init(&view)
    }

    fn interpolate(
        &self,
        data: &InterpDataNDBase<Dg, Dv>,
        point: &[f64],
    ) -> Result<Dv::Elem, InterpolateError> {
        let view = InterpDataNDBase {
            grid: data.grid.iter().map(|g| g.view()).collect(),
            values: self.values_cache.view(),
        };
        let result = self.inner.interpolate(&view, point)?;
        from_f64_checked(self.transform.inverse(result))
    }

    /// Hands `point` unmodified to `inner.interpolate_wrapped`, so a nested
    /// `GridTransform` handles the actual raw-space wrap; must not wrap here itself.
    fn interpolate_wrapped(
        &self,
        data: &InterpDataNDBase<Dg, Dv>,
        point: &[f64],
    ) -> Result<Dv::Elem, InterpolateError>
    where
        Dg::Elem: NumCast + Copy,
    {
        let view = InterpDataNDBase {
            grid: data.grid.iter().map(|g| g.view()).collect(),
            values: self.values_cache.view(),
        };
        let result = self.inner.interpolate_wrapped(&view, point)?;
        from_f64_checked(self.transform.inverse(result))
    }

    /// Calls `inner.interpolate_fast` rather than `inner.interpolate`, so "fast"
    /// propagates through nested `GridTransform`/`ValuesTransform` layers instead of
    /// stopping at the outermost one.
    ///
    /// Truncates rather than rounds for an integer `Dv`, same as `from_f64_checked`.
    fn interpolate_fast(&self, data: &InterpDataNDBase<Dg, Dv>, point: &[f64]) -> Dv::Elem {
        let view = InterpDataNDBase {
            grid: data.grid.iter().map(|g| g.view()).collect(),
            values: self.values_cache.view(),
        };
        let result = self.inner.interpolate_fast(&view, point);
        num_traits::cast(self.transform.inverse(result))
            .expect("inverse-transformed value doesn't fit in value type")
    }

    /// Delegates the whole batch to `inner` via a transient `f64` view over
    /// `values_cache`, then casts + inverse-transforms every output afterward.
    fn batch_interpolate_into(
        &self,
        data: &InterpDataNDBase<Dg, Dv>,
        points: &[&[f64]],
        out: &mut [Dv::Elem],
    ) -> Result<(), InterpolateError> {
        if out.len() != points.len() {
            return Err(InterpolateError::OutputLength {
                expected: points.len(),
                found: out.len(),
            });
        }
        let view = InterpDataNDBase {
            grid: data.grid.iter().map(|g| g.view()).collect(),
            values: self.values_cache.view(),
        };
        let mut scratch = vec![0f64; points.len()];
        self.inner
            .batch_interpolate_into(&view, points, &mut scratch)?;
        for (o, v) in out.iter_mut().zip(scratch) {
            *o = from_f64_checked(self.transform.inverse(v))?;
        }
        Ok(())
    }

    /// Forwards to `inner`, so a nested `GridTransform`'s domain is still checked
    /// through a wrapping `ValuesTransform`.
    fn check_batch_domain(&self, points: &[&[f64]]) -> Result<(), InterpolateError> {
        self.inner.check_batch_domain(points)
    }

    fn allow_extrapolate(&self) -> bool {
        self.inner.allow_extrapolate()
    }
}
