//! Strategy trait definitions for all dimensionalities.

use super::*;

/// 1-D interpolation strategy.
pub trait Strategy1D<D>: Debug + DynClone
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    /// Initialize strategy struct, with access to interpolation data.
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

    /// Does this type's [`Strategy1D::interpolate`] provision for extrapolation?
    fn allow_extrapolate(&self) -> bool;
}

clone_trait_object!(<D> Strategy1D<D>);

impl<D> Strategy1D<D> for Box<dyn Strategy1D<D>>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
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
    /// Initialize strategy struct, with access to interpolation data.
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

    /// Does this type's [`Strategy2D::interpolate`] provision for extrapolation?
    fn allow_extrapolate(&self) -> bool;
}

clone_trait_object!(<D> Strategy2D<D>);

impl<D> Strategy2D<D> for Box<dyn Strategy2D<D>>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
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
    /// Initialize strategy struct, with access to interpolation data.
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

    /// Does this type's [`Strategy3D::interpolate`] provision for extrapolation?
    fn allow_extrapolate(&self) -> bool;
}

clone_trait_object!(<D> Strategy3D<D>);

impl<D> Strategy3D<D> for Box<dyn Strategy3D<D>>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
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
    /// Initialize strategy struct, with access to interpolation data.
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

    /// Does this type's [`StrategyND::interpolate`] provision for extrapolation?
    fn allow_extrapolate(&self) -> bool;
}

clone_trait_object!(<D> StrategyND<D>);

impl<D> StrategyND<D> for Box<dyn StrategyND<D>>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
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
}
