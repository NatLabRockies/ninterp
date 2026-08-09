//! Strategy trait definitions for all dimensionalities.

use super::*;

/// 1-D interpolation strategy.
pub trait Strategy1D<D>: Debug + DynClone
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    /// Validate strategy state against interpolation data. Pure check, no mutation.
    ///
    /// Default no-op. Override for invariant checks that don't require precomputed
    /// state (grid uniformity, direction-count matching, etc).
    fn validate(&self, _data: &InterpData1D<D>) -> Result<(), ValidateError> {
        Ok(())
    }

    /// Initialize/recompute cached derived state, with access to interpolation data.
    ///
    /// Default no-op. Override only when the strategy caches something derived from
    /// `data` (e.g. precomputed spline coefficients). Unlike [`Strategy1D::validate`],
    /// this may do real, non-trivial calculation.
    fn init(&mut self, _data: &InterpData1D<D>) -> Result<(), ValidateError> {
        Ok(())
    }

    /// Execute interpolation (after handling [`Extrapolate`](`crate::interpolator::Extrapolate`) setting).
    ///
    /// # Note for custom strategies
    /// Index `data.grid[i]` directly via `ArrayView` indexing; avoid `.as_slice()`, which
    /// panics on non-contiguous storage (possible with `Interp*Viewed`). See
    /// [`crate::strategy::utils`] for ready-made per-axis search helpers (bracket search,
    /// exact-match short-circuit, step-direction lookup, uniform-grid fast path) built from
    /// the same primitives the built-in strategies use.
    fn interpolate(
        &self,
        data: &InterpData1D<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError>;

    /// Interpolate without the `Result` wrapper, assuming `point` and `data` are
    /// already valid.
    ///
    /// Default just unwraps [`Strategy1D::interpolate`]. Override only if this
    /// strategy's checked path does real internal fallible work beyond producing
    /// the final `Ok(...)`; otherwise the default already compiles to the same thing.
    #[inline]
    fn interpolate_fast(&self, data: &InterpData1D<D>, point: &[D::Elem; 1]) -> D::Elem {
        self.interpolate(data, point)
            .expect("interpolate_fast: invalid point or data")
    }

    /// Interpolate at each of several points, sharing one grid across all of them.
    ///
    /// Default just loops [`Strategy1D::interpolate`]. Override only if locating a
    /// point in the grid can be amortized across the batch (e.g. sorting points once
    /// for a locate sweep instead of one binary search per point); no strategy
    /// shipped in this crate does that today.
    fn batch_interpolate(
        &self,
        data: &InterpData1D<D>,
        points: &[[D::Elem; 1]],
    ) -> Result<Vec<D::Elem>, InterpolateError> {
        points
            .iter()
            .map(|point| self.interpolate(data, point))
            .collect()
    }

    /// Batched [`Strategy1D::interpolate_fast`], assuming every point and `data` are
    /// already valid.
    ///
    /// Default just loops [`Strategy1D::interpolate_fast`]. Override under the same
    /// condition as [`Strategy1D::batch_interpolate`].
    fn batch_interpolate_fast(
        &self,
        data: &InterpData1D<D>,
        points: &[[D::Elem; 1]],
    ) -> Vec<D::Elem> {
        points
            .iter()
            .map(|point| self.interpolate_fast(data, point))
            .collect()
    }

    /// Does this type's [`Strategy1D::interpolate`] provision for extrapolation?
    fn allow_extrapolate(&self) -> bool;
}

clone_trait_object!(<D> Strategy1D<D>);

impl<D> Strategy1D<D> for Box<dyn Strategy1D<D>>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    /// Validate strategy state against interpolation data. Pure check, no mutation.
    #[inline]
    fn validate(&self, data: &InterpData1D<D>) -> Result<(), ValidateError> {
        (**self).validate(data)
    }

    /// Initialize strategy struct, with access to interpolation data.
    #[inline]
    fn init(&mut self, data: &InterpData1D<D>) -> Result<(), ValidateError> {
        (**self).init(data)
    }

    #[inline]
    fn interpolate(
        &self,
        data: &InterpData1D<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError> {
        (**self).interpolate(data, point)
    }

    #[inline]
    fn interpolate_fast(&self, data: &InterpData1D<D>, point: &[D::Elem; 1]) -> D::Elem {
        (**self).interpolate_fast(data, point)
    }

    #[inline]
    fn batch_interpolate(
        &self,
        data: &InterpData1D<D>,
        points: &[[D::Elem; 1]],
    ) -> Result<Vec<D::Elem>, InterpolateError> {
        (**self).batch_interpolate(data, points)
    }

    #[inline]
    fn batch_interpolate_fast(
        &self,
        data: &InterpData1D<D>,
        points: &[[D::Elem; 1]],
    ) -> Vec<D::Elem> {
        (**self).batch_interpolate_fast(data, points)
    }

    #[inline]
    fn allow_extrapolate(&self) -> bool {
        (**self).allow_extrapolate()
    }
}

/// 2-D interpolation strategy.
pub trait Strategy2D<D>: Debug + DynClone
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    /// Validate strategy state against interpolation data. Pure check, no mutation.
    ///
    /// Default no-op. Override for invariant checks that don't require precomputed
    /// state (grid uniformity, direction-count matching, etc).
    fn validate(&self, _data: &InterpData2D<D>) -> Result<(), ValidateError> {
        Ok(())
    }

    /// Initialize/recompute cached derived state, with access to interpolation data.
    ///
    /// Default no-op. Override only when the strategy caches something derived from
    /// `data` (e.g. precomputed spline coefficients). Unlike [`Strategy2D::validate`],
    /// this may do real, non-trivial calculation.
    fn init(&mut self, _data: &InterpData2D<D>) -> Result<(), ValidateError> {
        Ok(())
    }

    /// Execute interpolation (after handling [`Extrapolate`](`crate::interpolator::Extrapolate`) setting).
    ///
    /// # Note for custom strategies
    /// Index `data.grid[i]` directly via `ArrayView` indexing; avoid `.as_slice()`, which
    /// panics on non-contiguous storage (possible with `Interp*Viewed`). See
    /// [`crate::strategy::utils`] for ready-made per-axis search helpers (bracket search,
    /// exact-match short-circuit, step-direction lookup, uniform-grid fast path) built from
    /// the same primitives the built-in strategies use.
    fn interpolate(
        &self,
        data: &InterpData2D<D>,
        point: &[D::Elem; 2],
    ) -> Result<D::Elem, InterpolateError>;

    /// Interpolate without the `Result` wrapper, assuming `point` and `data` are
    /// already valid.
    ///
    /// Default just unwraps [`Strategy2D::interpolate`]. Override only if this
    /// strategy's checked path does real internal fallible work beyond producing
    /// the final `Ok(...)`; otherwise the default already compiles to the same thing.
    #[inline]
    fn interpolate_fast(&self, data: &InterpData2D<D>, point: &[D::Elem; 2]) -> D::Elem {
        self.interpolate(data, point)
            .expect("interpolate_fast: invalid point or data")
    }

    /// Interpolate at each of several points, sharing one grid across all of them.
    ///
    /// Default just loops [`Strategy2D::interpolate`]. Override only if locating a
    /// point in the grid can be amortized across the batch (e.g. sorting points once
    /// for a locate sweep instead of one binary search per point); no strategy
    /// shipped in this crate does that today.
    fn batch_interpolate(
        &self,
        data: &InterpData2D<D>,
        points: &[[D::Elem; 2]],
    ) -> Result<Vec<D::Elem>, InterpolateError> {
        points
            .iter()
            .map(|point| self.interpolate(data, point))
            .collect()
    }

    /// Batched [`Strategy2D::interpolate_fast`], assuming every point and `data` are
    /// already valid.
    ///
    /// Default just loops [`Strategy2D::interpolate_fast`]. Override under the same
    /// condition as [`Strategy2D::batch_interpolate`].
    fn batch_interpolate_fast(
        &self,
        data: &InterpData2D<D>,
        points: &[[D::Elem; 2]],
    ) -> Vec<D::Elem> {
        points
            .iter()
            .map(|point| self.interpolate_fast(data, point))
            .collect()
    }

    /// Does this type's [`Strategy2D::interpolate`] provision for extrapolation?
    fn allow_extrapolate(&self) -> bool;
}

clone_trait_object!(<D> Strategy2D<D>);

impl<D> Strategy2D<D> for Box<dyn Strategy2D<D>>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    /// Validate strategy state against interpolation data. Pure check, no mutation.
    #[inline]
    fn validate(&self, data: &InterpData2D<D>) -> Result<(), ValidateError> {
        (**self).validate(data)
    }

    /// Initialize strategy struct, with access to interpolation data.
    #[inline]
    fn init(&mut self, data: &InterpData2D<D>) -> Result<(), ValidateError> {
        (**self).init(data)
    }

    #[inline]
    fn interpolate(
        &self,
        data: &InterpData2D<D>,
        point: &[D::Elem; 2],
    ) -> Result<D::Elem, InterpolateError> {
        (**self).interpolate(data, point)
    }

    #[inline]
    fn interpolate_fast(&self, data: &InterpData2D<D>, point: &[D::Elem; 2]) -> D::Elem {
        (**self).interpolate_fast(data, point)
    }

    #[inline]
    fn batch_interpolate(
        &self,
        data: &InterpData2D<D>,
        points: &[[D::Elem; 2]],
    ) -> Result<Vec<D::Elem>, InterpolateError> {
        (**self).batch_interpolate(data, points)
    }

    #[inline]
    fn batch_interpolate_fast(
        &self,
        data: &InterpData2D<D>,
        points: &[[D::Elem; 2]],
    ) -> Vec<D::Elem> {
        (**self).batch_interpolate_fast(data, points)
    }

    #[inline]
    fn allow_extrapolate(&self) -> bool {
        (**self).allow_extrapolate()
    }
}

/// 3-D interpolation strategy.
pub trait Strategy3D<D>: Debug + DynClone
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    /// Validate strategy state against interpolation data. Pure check, no mutation.
    ///
    /// Default no-op. Override for invariant checks that don't require precomputed
    /// state (grid uniformity, direction-count matching, etc).
    fn validate(&self, _data: &InterpData3D<D>) -> Result<(), ValidateError> {
        Ok(())
    }

    /// Initialize/recompute cached derived state, with access to interpolation data.
    ///
    /// Default no-op. Override only when the strategy caches something derived from
    /// `data` (e.g. precomputed spline coefficients). Unlike [`Strategy3D::validate`],
    /// this may do real, non-trivial calculation.
    fn init(&mut self, _data: &InterpData3D<D>) -> Result<(), ValidateError> {
        Ok(())
    }

    /// Execute interpolation (after handling [`Extrapolate`](`crate::interpolator::Extrapolate`) setting).
    ///
    /// # Note for custom strategies
    /// Index `data.grid[i]` directly via `ArrayView` indexing; avoid `.as_slice()`, which
    /// panics on non-contiguous storage (possible with `Interp*Viewed`). See
    /// [`crate::strategy::utils`] for ready-made per-axis search helpers (bracket search,
    /// exact-match short-circuit, step-direction lookup, uniform-grid fast path) built from
    /// the same primitives the built-in strategies use.
    fn interpolate(
        &self,
        data: &InterpData3D<D>,
        point: &[D::Elem; 3],
    ) -> Result<D::Elem, InterpolateError>;

    /// Interpolate without the `Result` wrapper, assuming `point` and `data` are
    /// already valid.
    ///
    /// Default just unwraps [`Strategy3D::interpolate`]. Override only if this
    /// strategy's checked path does real internal fallible work beyond producing
    /// the final `Ok(...)`; otherwise the default already compiles to the same thing.
    #[inline]
    fn interpolate_fast(&self, data: &InterpData3D<D>, point: &[D::Elem; 3]) -> D::Elem {
        self.interpolate(data, point)
            .expect("interpolate_fast: invalid point or data")
    }

    /// Interpolate at each of several points, sharing one grid across all of them.
    ///
    /// Default just loops [`Strategy3D::interpolate`]. Override only if locating a
    /// point in the grid can be amortized across the batch (e.g. sorting points once
    /// for a locate sweep instead of one binary search per point); no strategy
    /// shipped in this crate does that today.
    fn batch_interpolate(
        &self,
        data: &InterpData3D<D>,
        points: &[[D::Elem; 3]],
    ) -> Result<Vec<D::Elem>, InterpolateError> {
        points
            .iter()
            .map(|point| self.interpolate(data, point))
            .collect()
    }

    /// Batched [`Strategy3D::interpolate_fast`], assuming every point and `data` are
    /// already valid.
    ///
    /// Default just loops [`Strategy3D::interpolate_fast`]. Override under the same
    /// condition as [`Strategy3D::batch_interpolate`].
    fn batch_interpolate_fast(
        &self,
        data: &InterpData3D<D>,
        points: &[[D::Elem; 3]],
    ) -> Vec<D::Elem> {
        points
            .iter()
            .map(|point| self.interpolate_fast(data, point))
            .collect()
    }

    /// Does this type's [`Strategy3D::interpolate`] provision for extrapolation?
    fn allow_extrapolate(&self) -> bool;
}

clone_trait_object!(<D> Strategy3D<D>);

impl<D> Strategy3D<D> for Box<dyn Strategy3D<D>>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    /// Validate strategy state against interpolation data. Pure check, no mutation.
    #[inline]
    fn validate(&self, data: &InterpData3D<D>) -> Result<(), ValidateError> {
        (**self).validate(data)
    }

    /// Initialize strategy struct, with access to interpolation data.
    #[inline]
    fn init(&mut self, data: &InterpData3D<D>) -> Result<(), ValidateError> {
        (**self).init(data)
    }

    #[inline]
    fn interpolate(
        &self,
        data: &InterpData3D<D>,
        point: &[D::Elem; 3],
    ) -> Result<D::Elem, InterpolateError> {
        (**self).interpolate(data, point)
    }

    #[inline]
    fn interpolate_fast(&self, data: &InterpData3D<D>, point: &[D::Elem; 3]) -> D::Elem {
        (**self).interpolate_fast(data, point)
    }

    #[inline]
    fn batch_interpolate(
        &self,
        data: &InterpData3D<D>,
        points: &[[D::Elem; 3]],
    ) -> Result<Vec<D::Elem>, InterpolateError> {
        (**self).batch_interpolate(data, points)
    }

    #[inline]
    fn batch_interpolate_fast(
        &self,
        data: &InterpData3D<D>,
        points: &[[D::Elem; 3]],
    ) -> Vec<D::Elem> {
        (**self).batch_interpolate_fast(data, points)
    }

    #[inline]
    fn allow_extrapolate(&self) -> bool {
        (**self).allow_extrapolate()
    }
}

/// N-D interpolation strategy.
pub trait StrategyND<D>: Debug + DynClone
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    /// Validate strategy state against interpolation data. Pure check, no mutation.
    ///
    /// Default no-op. Override for invariant checks that don't require precomputed
    /// state (grid uniformity, direction-count matching, etc).
    fn validate(&self, _data: &InterpDataND<D>) -> Result<(), ValidateError> {
        Ok(())
    }

    /// Initialize/recompute cached derived state, with access to interpolation data.
    ///
    /// Default no-op. Override only when the strategy caches something derived from
    /// `data` (e.g. precomputed spline coefficients). Unlike [`StrategyND::validate`],
    /// this may do real, non-trivial calculation.
    fn init(&mut self, _data: &InterpDataND<D>) -> Result<(), ValidateError> {
        Ok(())
    }

    /// Execute interpolation (after handling [`Extrapolate`](`crate::interpolator::Extrapolate`) setting).
    ///
    /// # Note for custom strategies
    /// Index `data.grid[i]` directly via `ArrayView` indexing; avoid `.as_slice()`, which
    /// panics on non-contiguous storage (possible with `Interp*Viewed`). See
    /// [`crate::strategy::utils`] for ready-made per-axis search helpers (bracket search,
    /// exact-match short-circuit, step-direction lookup, uniform-grid fast path) built from
    /// the same primitives the built-in strategies use.
    fn interpolate(
        &self,
        data: &InterpDataND<D>,
        point: &[D::Elem],
    ) -> Result<D::Elem, InterpolateError>;

    /// Interpolate without the `Result` wrapper, assuming `point` and `data` are
    /// already valid.
    ///
    /// Default just unwraps [`StrategyND::interpolate`]. Override only if this
    /// strategy's checked path does real internal fallible work beyond producing
    /// the final `Ok(...)`; otherwise the default already compiles to the same thing.
    #[inline]
    fn interpolate_fast(&self, data: &InterpDataND<D>, point: &[D::Elem]) -> D::Elem {
        self.interpolate(data, point)
            .expect("interpolate_fast: invalid point or data")
    }

    /// Interpolate at each of several points, sharing one grid across all of them.
    ///
    /// Default just loops [`StrategyND::interpolate`]. Override only if locating a
    /// point in the grid can be amortized across the batch (e.g. sorting points once
    /// for a locate sweep instead of one binary search per point); no strategy
    /// shipped in this crate does that today.
    fn batch_interpolate(
        &self,
        data: &InterpDataND<D>,
        points: &[&[D::Elem]],
    ) -> Result<Vec<D::Elem>, InterpolateError> {
        points
            .iter()
            .map(|point| self.interpolate(data, point))
            .collect()
    }

    /// Batched [`StrategyND::interpolate_fast`], assuming every point and `data` are
    /// already valid.
    ///
    /// Default just loops [`StrategyND::interpolate_fast`]. Override under the same
    /// condition as [`StrategyND::batch_interpolate`].
    fn batch_interpolate_fast(
        &self,
        data: &InterpDataND<D>,
        points: &[&[D::Elem]],
    ) -> Vec<D::Elem> {
        points
            .iter()
            .map(|point| self.interpolate_fast(data, point))
            .collect()
    }

    /// Does this type's [`StrategyND::interpolate`] provision for extrapolation?
    fn allow_extrapolate(&self) -> bool;
}

clone_trait_object!(<D> StrategyND<D>);

impl<D> StrategyND<D> for Box<dyn StrategyND<D>>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    /// Validate strategy state against interpolation data. Pure check, no mutation.
    #[inline]
    fn validate(&self, data: &InterpDataND<D>) -> Result<(), ValidateError> {
        (**self).validate(data)
    }

    #[inline]
    fn init(&mut self, data: &InterpDataND<D>) -> Result<(), ValidateError> {
        (**self).init(data)
    }

    #[inline]
    fn interpolate(
        &self,
        data: &InterpDataND<D>,
        point: &[D::Elem],
    ) -> Result<D::Elem, InterpolateError> {
        (**self).interpolate(data, point)
    }

    #[inline]
    fn allow_extrapolate(&self) -> bool {
        (**self).allow_extrapolate()
    }

    #[inline]
    fn interpolate_fast(&self, data: &InterpDataND<D>, point: &[D::Elem]) -> D::Elem {
        (**self).interpolate_fast(data, point)
    }

    #[inline]
    fn batch_interpolate(
        &self,
        data: &InterpDataND<D>,
        points: &[&[D::Elem]],
    ) -> Result<Vec<D::Elem>, InterpolateError> {
        (**self).batch_interpolate(data, points)
    }

    #[inline]
    fn batch_interpolate_fast(
        &self,
        data: &InterpDataND<D>,
        points: &[&[D::Elem]],
    ) -> Vec<D::Elem> {
        (**self).batch_interpolate_fast(data, points)
    }
}
