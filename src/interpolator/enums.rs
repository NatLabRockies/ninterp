//! This module provides an [`InterpolatorEnum`] that allow mutable interpolator swapping.

// NOTE: `enum_dispatch` does essentially what this module does, but with less boilerplate.
// However, it does not currently support using a generic trait on a non-generic enum.
// https://gitlab.com/antonok/enum_dispatch/-/issues/67

use super::*;

use strategy::enums::*;

/// This is an alternative to using a `Box<dyn Interpolator<_>>` with a few key differences:
/// - Better runtime performance
/// - Compatible with serde
/// - **Incompatible** with custom strategies
///   - Must use a [`Strategy1DEnum`]/[`Strategy2DEnum`]/etc. internally
///
/// # Example:
/// ```
/// use ndarray::prelude::*;
/// use ninterp::prelude::*;
///
/// // 1-D linear
/// // type annotation for clarity
/// let mut interp: InterpolatorEnumOwned<_> = InterpolatorEnum::new_1d(
///     // x
///     array![0., 1., 2., 3., 4.],
///     // f(x)
///     array![0.2, 0.4, 0.6, 0.8, 1.0],
///     strategy::Linear, // strategy mod is exposed via `use ndarray::prelude::*;`
///     Extrapolate::Error,
/// )
/// .unwrap();
/// assert_eq!(interp.interpolate(&[3.00]).unwrap(), 0.8);
/// assert_eq!(interp.interpolate(&[3.75]).unwrap(), 0.95);
/// assert_eq!(interp.interpolate(&[4.00]).unwrap(), 1.0);
///
/// // 2-D nearest
/// interp = InterpolatorEnum::new_2d(
///     // x
///     array![0.05, 0.10, 0.15],
///     // y
///     array![0.10, 0.20, 0.30],
///     // f(x, y)
///     array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
///     strategy::Nearest,
///     Extrapolate::Error,
/// )
/// .unwrap();
/// let f_xy = match &interp {
///     InterpolatorEnum::Interp2D(interp) => &interp.data.values,
///     _ => unreachable!(),
/// };
///
/// assert_eq!(interp.interpolate(&[0.08, 0.21]).unwrap(), f_xy[[1, 1]]);
/// assert_eq!(interp.interpolate(&[0.11, 0.26]).unwrap(), f_xy[[1, 2]]);
/// assert_eq!(interp.interpolate(&[0.13, 0.12]).unwrap(), f_xy[[2, 0]]);
/// assert_eq!(interp.interpolate(&[0.14, 0.29]).unwrap(), f_xy[[2, 2]]);
///
/// // 0-D
/// interp = InterpolatorEnum::new_0d(0.5);
/// assert_eq!(interp.interpolate(&[]).unwrap(), 0.5);
/// ```
/// See also: `examples/dynamic_interpolator.rs`
#[allow(missing_docs)]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(feature = "serde", serde(untagged))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "D::Elem: Serialize",
        deserialize = "
            D: DataOwned,
            D::Elem: Deserialize<'de>,
        "
    ))
)]
pub enum InterpolatorEnum<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Num + PartialOrd + Copy + Debug,
{
    Interp0D(Interp0D<D::Elem>),
    Interp1D(Interp1D<D, Strategy1DEnum>),
    Interp2D(Interp2D<D, Strategy2DEnum>),
    Interp3D(Interp3D<D, Strategy3DEnum>),
    InterpND(InterpND<D, StrategyNDEnum>),
}
/// [`InterpolatorEnum`] that views data.
pub type InterpolatorEnumViewed<T> = InterpolatorEnum<ndarray::ViewRepr<T>>;
/// [`InterpolatorEnum`] that owns data.
pub type InterpolatorEnumOwned<T> = InterpolatorEnum<ndarray::OwnedRepr<T>>;

impl<D> PartialEq for InterpolatorEnum<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Num + PartialOrd + Copy + Debug,
    ArrayBase<D, Ix1>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Interp0D(l), Self::Interp0D(r)) => l == r,
            (Self::Interp1D(l), Self::Interp1D(r)) => l == r,
            (Self::Interp2D(l), Self::Interp2D(r)) => l == r,
            (Self::Interp3D(l), Self::Interp3D(r)) => l == r,
            (Self::InterpND(l), Self::InterpND(r)) => l == r,
            _ => false,
        }
    }
}

impl<D> InterpolatorEnum<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Num + PartialOrd + Copy + Debug,
{
    /// Create [`InterpolatorEnum::Interp0D`], internally calling [`Interp0D::new`].
    #[inline]
    pub fn new_0d(value: D::Elem) -> Self {
        Self::Interp0D(Interp0D::new(value))
    }

    /// Create [`InterpolatorEnum::Interp1D`], internally calling [`Interp1D::new`].
    #[inline]
    pub fn new_1d(
        x: ArrayBase<D, Ix1>,
        f_x: ArrayBase<D, Ix1>,
        strategy: impl Into<Strategy1DEnum>,
        extrapolate: Extrapolate<D::Elem>,
    ) -> Result<Self, ValidateError> {
        Ok(Self::Interp1D(Interp1D::new(
            x,
            f_x,
            strategy.into(),
            extrapolate,
        )?))
    }

    /// Create [`InterpolatorEnum::Interp2D`], internally calling [`Interp2D::new`].
    #[inline]
    pub fn new_2d(
        x: ArrayBase<D, Ix1>,
        y: ArrayBase<D, Ix1>,
        f_xy: ArrayBase<D, Ix2>,
        strategy: impl Into<Strategy2DEnum>,
        extrapolate: Extrapolate<D::Elem>,
    ) -> Result<Self, ValidateError> {
        Ok(Self::Interp2D(Interp2D::new(
            x,
            y,
            f_xy,
            strategy.into(),
            extrapolate,
        )?))
    }

    /// Create [`InterpolatorEnum::Interp3D`], internally calling [`Interp3D::new`].
    #[inline]
    pub fn new_3d(
        x: ArrayBase<D, Ix1>,
        y: ArrayBase<D, Ix1>,
        z: ArrayBase<D, Ix1>,
        f_xyz: ArrayBase<D, Ix3>,
        strategy: impl Into<Strategy3DEnum>,
        extrapolate: Extrapolate<D::Elem>,
    ) -> Result<Self, ValidateError> {
        Ok(Self::Interp3D(Interp3D::new(
            x,
            y,
            z,
            f_xyz,
            strategy.into(),
            extrapolate,
        )?))
    }

    /// Create [`InterpolatorEnum::InterpND`], internally calling [`InterpND::new`].
    #[inline]
    pub fn new_nd(
        grid: Vec<ArrayBase<D, Ix1>>,
        values: ArrayBase<D, IxDyn>,
        strategy: impl Into<StrategyNDEnum>,
        extrapolate: Extrapolate<D::Elem>,
    ) -> Result<Self, ValidateError> {
        Ok(Self::InterpND(InterpND::new(
            grid,
            values,
            strategy.into(),
            extrapolate,
        )?))
    }
}

impl<D> Interpolator<D::Elem> for InterpolatorEnum<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Num + Euclid + PartialOrd + Copy + Debug,
{
    #[inline]
    fn ndim(&self) -> usize {
        match self {
            InterpolatorEnum::Interp0D(_) => 0,
            InterpolatorEnum::Interp1D(_) => 1,
            InterpolatorEnum::Interp2D(_) => 2,
            InterpolatorEnum::Interp3D(_) => 3,
            InterpolatorEnum::InterpND(interp) => interp.ndim(),
        }
    }

    #[inline]
    fn validate(&mut self) -> Result<(), ValidateError> {
        match self {
            InterpolatorEnum::Interp0D(_) => Ok(()),
            InterpolatorEnum::Interp1D(interp) => interp.validate(),
            InterpolatorEnum::Interp2D(interp) => interp.validate(),
            InterpolatorEnum::Interp3D(interp) => interp.validate(),
            InterpolatorEnum::InterpND(interp) => interp.validate(),
        }
    }

    #[inline]
    fn interpolate(&self, point: &[D::Elem]) -> Result<D::Elem, InterpolateError> {
        match self {
            InterpolatorEnum::Interp0D(interp) => interp.interpolate(point),
            InterpolatorEnum::Interp1D(interp) => interp.interpolate(point),
            InterpolatorEnum::Interp2D(interp) => interp.interpolate(point),
            InterpolatorEnum::Interp3D(interp) => interp.interpolate(point),
            InterpolatorEnum::InterpND(interp) => interp.interpolate(point),
        }
    }

    #[inline]
    fn set_extrapolate(&mut self, extrapolate: Extrapolate<D::Elem>) -> Result<(), ValidateError> {
        match self {
            InterpolatorEnum::Interp0D(_) => Ok(()),
            InterpolatorEnum::Interp1D(interp) => interp.set_extrapolate(extrapolate),
            InterpolatorEnum::Interp2D(interp) => interp.set_extrapolate(extrapolate),
            InterpolatorEnum::Interp3D(interp) => interp.set_extrapolate(extrapolate),
            InterpolatorEnum::InterpND(interp) => interp.set_extrapolate(extrapolate),
        }
    }
}

impl<D> From<Interp0D<D::Elem>> for InterpolatorEnum<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Num + PartialOrd + Copy + Debug,
{
    #[inline]
    fn from(interpolator: Interp0D<D::Elem>) -> Self {
        InterpolatorEnum::Interp0D(interpolator)
    }
}

impl<D> From<Interp1D<D, Strategy1DEnum>> for InterpolatorEnum<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Num + PartialOrd + Copy + Debug,
{
    #[inline]
    fn from(interpolator: Interp1D<D, Strategy1DEnum>) -> Self {
        InterpolatorEnum::Interp1D(interpolator)
    }
}

impl<D> From<Interp2D<D, Strategy2DEnum>> for InterpolatorEnum<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Num + PartialOrd + Copy + Debug,
{
    #[inline]
    fn from(interpolator: Interp2D<D, Strategy2DEnum>) -> Self {
        InterpolatorEnum::Interp2D(interpolator)
    }
}

impl<D> From<Interp3D<D, Strategy3DEnum>> for InterpolatorEnum<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Num + PartialOrd + Copy + Debug,
{
    #[inline]
    fn from(interpolator: Interp3D<D, Strategy3DEnum>) -> Self {
        InterpolatorEnum::Interp3D(interpolator)
    }
}

impl<D> From<InterpND<D, StrategyNDEnum>> for InterpolatorEnum<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Num + PartialOrd + Copy + Debug,
{
    #[inline]
    fn from(interpolator: InterpND<D, StrategyNDEnum>) -> Self {
        InterpolatorEnum::InterpND(interpolator)
    }
}

mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_partialeq() {
        #[derive(PartialEq)]
        #[allow(unused)]
        struct MyStruct(InterpolatorEnumOwned<f64>);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_serde() {
        let interp0: Interp1D<_, strategy::enums::Strategy1DEnum> = Interp1D::new(
            array![0., 1., 2., 3., 4.],
            array![0.2, 0.4, 0.6, 0.8, 1.0],
            strategy::LeftNearest.into(),
            Extrapolate::Error,
        )
        .unwrap();
        let interp1 = InterpolatorEnum::from(interp0.clone());

        assert_eq!(
            serde_json::to_string(&interp0).unwrap(),
            serde_json::to_string(&interp1).unwrap()
        );
    }
}
