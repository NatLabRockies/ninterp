//! This module provides an [`InterpolatorEnum`] that allow mutable interpolator swapping.

// NOTE: `enum_dispatch` does essentially what this module does, but with less boilerplate.
// However, it does not currently support using a generic trait on a non-generic enum.
// https://gitlab.com/antonok/enum_dispatch/-/issues/67

use super::*;

use strategy::enums::*;

/// Generates all five `From` impls for InterpolatorEnum variants.
macro_rules! from_impls {
    () => {
        impl<D> From<Interp0D<D::Elem>> for InterpolatorEnum<D>
        where
            D: Data + RawDataClone + Clone,
            D::Elem: Float + Debug,
        {
            #[inline]
            fn from(interpolator: Interp0D<D::Elem>) -> Self {
                InterpolatorEnum::Interp0D(interpolator)
            }
        }

        impl<D> From<Interp1D<D, Strategy1DEnum>> for InterpolatorEnum<D>
        where
            D: Data + RawDataClone + Clone,
            D::Elem: Float + Debug,
        {
            #[inline]
            fn from(interpolator: Interp1D<D, Strategy1DEnum>) -> Self {
                InterpolatorEnum::Interp1D(interpolator)
            }
        }

        impl<D> From<Interp2D<D, Strategy2DEnum>> for InterpolatorEnum<D>
        where
            D: Data + RawDataClone + Clone,
            D::Elem: Float + Debug,
        {
            #[inline]
            fn from(interpolator: Interp2D<D, Strategy2DEnum>) -> Self {
                InterpolatorEnum::Interp2D(interpolator)
            }
        }

        impl<D> From<Interp3D<D, Strategy3DEnum>> for InterpolatorEnum<D>
        where
            D: Data + RawDataClone + Clone,
            D::Elem: Float + Debug,
        {
            #[inline]
            fn from(interpolator: Interp3D<D, Strategy3DEnum>) -> Self {
                InterpolatorEnum::Interp3D(interpolator)
            }
        }

        impl<D> From<InterpND<D, StrategyNDEnum>> for InterpolatorEnum<D>
        where
            D: Data + RawDataClone + Clone,
            D::Elem: Float + Debug,
        {
            #[inline]
            fn from(interpolator: InterpND<D, StrategyNDEnum>) -> Self {
                InterpolatorEnum::InterpND(interpolator)
            }
        }
    };
}

/// Generates simple forwarding methods where Interp0D returns Ok(()), others forward to variant.
macro_rules! enum_method_forward {
    (validate_strategy) => {
        #[doc = "Re-run the current variant's strategy `validate` against its data.\n\n`new_0d`/`new_1d`/etc. already call this internally, so this is only needed\nafter mutating a variant's `data`/`strategy` fields directly (via `match`)."]
        pub fn validate_strategy(&self) -> Result<(), ValidateError> {
            match self {
                InterpolatorEnum::Interp0D(_) => Ok(()),
                InterpolatorEnum::Interp1D(interp) => interp.validate_strategy(),
                InterpolatorEnum::Interp2D(interp) => interp.validate_strategy(),
                InterpolatorEnum::Interp3D(interp) => interp.validate_strategy(),
                InterpolatorEnum::InterpND(interp) => interp.validate_strategy(),
            }
        }
    };
    (check_extrapolate) => {
        #[doc = "Check applicability of the current variant's strategy, data, and extrapolate setting."]
        pub fn check_extrapolate(
            &self,
            extrapolate: &Extrapolate<D::Elem>,
        ) -> Result<(), ValidateError> {
            match self {
                InterpolatorEnum::Interp0D(_) => Ok(()),
                InterpolatorEnum::Interp1D(interp) => interp.check_extrapolate(extrapolate),
                InterpolatorEnum::Interp2D(interp) => interp.check_extrapolate(extrapolate),
                InterpolatorEnum::Interp3D(interp) => interp.check_extrapolate(extrapolate),
                InterpolatorEnum::InterpND(interp) => interp.check_extrapolate(extrapolate),
            }
        }
    };
    (mut init_strategy) => {
        #[doc = "Re-run the current variant's strategy `init` against its data.\n\n`new_0d`/`new_1d`/etc. already call this internally, so this is only needed\nafter bypassing them: mutating a variant's `data`/`strategy` fields directly\n(via `match`), or deserializing an interpolator with a stateful custom strategy\n(`Deserialize` does not call `init`)."]
        pub fn init_strategy(&mut self) -> Result<(), ValidateError> {
            match self {
                InterpolatorEnum::Interp0D(_) => Ok(()),
                InterpolatorEnum::Interp1D(interp) => interp.init_strategy(),
                InterpolatorEnum::Interp2D(interp) => interp.init_strategy(),
                InterpolatorEnum::Interp3D(interp) => interp.init_strategy(),
                InterpolatorEnum::InterpND(interp) => interp.init_strategy(),
            }
        }
    };
}

/// Generates enum-wrapping forwarding methods (view, into_owned).
macro_rules! enum_method_wrap {
    (view) => {
        #[doc = "Return an interpolator with viewed data."]
        pub fn view(&self) -> InterpolatorEnumViewed<&D::Elem> {
            match self {
                InterpolatorEnum::Interp0D(interp) => InterpolatorEnum::Interp0D(interp.clone()),
                InterpolatorEnum::Interp1D(interp) => InterpolatorEnum::Interp1D(interp.view()),
                InterpolatorEnum::Interp2D(interp) => InterpolatorEnum::Interp2D(interp.view()),
                InterpolatorEnum::Interp3D(interp) => InterpolatorEnum::Interp3D(interp.view()),
                InterpolatorEnum::InterpND(interp) => InterpolatorEnum::InterpND(interp.view()),
            }
        }
    };
    (into_owned) => {
        #[doc = "Turn the interpolator into an [`InterpolatorEnumOwned`], cloning the array elements if necessary."]
        pub fn into_owned(self) -> InterpolatorEnumOwned<D::Elem>
        where
            D::Elem: Clone,
        {
            match self {
                InterpolatorEnum::Interp0D(interp) => InterpolatorEnum::Interp0D(interp.clone()),
                InterpolatorEnum::Interp1D(interp) => InterpolatorEnum::Interp1D(interp.into_owned()),
                InterpolatorEnum::Interp2D(interp) => InterpolatorEnum::Interp2D(interp.into_owned()),
                InterpolatorEnum::Interp3D(interp) => InterpolatorEnum::Interp3D(interp.into_owned()),
                InterpolatorEnum::InterpND(interp) => InterpolatorEnum::InterpND(interp.into_owned()),
            }
        }
    };
}

/// Generates trait forwarding methods for Interpolator impl.
macro_rules! enum_trait_method {
    (ndim) => {
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
    };
    ($method:ident) => {
        fn $method(&self) -> Result<(), ValidateError> {
            match self {
                Self::Interp0D(_) => Ok(()),
                Self::Interp1D(i) => i.$method(),
                Self::Interp2D(i) => i.$method(),
                Self::Interp3D(i) => i.$method(),
                Self::InterpND(i) => i.$method(),
            }
        }
    };
    (mut $method:ident, $param:ident: $param_ty:ty) => {
        fn $method(&mut self, $param: $param_ty) -> Result<(), ValidateError> {
            match self {
                Self::Interp0D(_) => Ok(()),
                Self::Interp1D(i) => i.$method($param),
                Self::Interp2D(i) => i.$method($param),
                Self::Interp3D(i) => i.$method($param),
                Self::InterpND(i) => i.$method($param),
            }
        }
    };
}

/// Generates full PartialEq impl for enum variants. Takes enum type as argument for reusability
/// with InterpolatorMultiEnum and others.
macro_rules! enum_partialeq_impl {
    ($enum_type:ident) => {
        impl<D> PartialEq for $enum_type<D>
        where
            D: Data + RawDataClone + Clone,
            D::Elem: PartialEq + Debug,
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
    };
}

/// Generates full SerializeNested impl for enum variants. Takes enum type as argument for reusability
/// with InterpolatorMultiEnum and others.
#[cfg(feature = "serde")]
macro_rules! enum_serialize_nested_impl {
    ($enum_type:ident) => {
        impl<D> SerializeNested for $enum_type<D>
        where
            D: Data + RawDataClone + Clone,
            D::Elem: PartialEq + Debug + Serialize,
        {
            #[doc = "`#[serde(untagged)]`, so each variant serializes as its inner value."]
            fn serialize_nested<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                match self {
                    Self::Interp0D(interp) => Nested(interp).serialize(serializer),
                    Self::Interp1D(interp) => Nested(interp).serialize(serializer),
                    Self::Interp2D(interp) => Nested(interp).serialize(serializer),
                    Self::Interp3D(interp) => Nested(interp).serialize(serializer),
                    Self::InterpND(interp) => Nested(interp).serialize(serializer),
                }
            }
        }
    };
}

/// Generates trait impl methods with slice-to-array conversion and doc comments built in.
macro_rules! slice_to_array_forward {
    (interpolate) => {
        fn interpolate(&self, point: &[D::Elem]) -> Result<D::Elem, InterpolateError> {
            match self {
                InterpolatorEnum::Interp0D(interp) => interp.interpolate(point),
                InterpolatorEnum::Interp1D(interp) => interp.interpolate(
                    point
                        .try_into()
                        .map_err(|_| InterpolateError::PointLength(1))?,
                ),
                InterpolatorEnum::Interp2D(interp) => interp.interpolate(
                    point
                        .try_into()
                        .map_err(|_| InterpolateError::PointLength(2))?,
                ),
                InterpolatorEnum::Interp3D(interp) => interp.interpolate(
                    point
                        .try_into()
                        .map_err(|_| InterpolateError::PointLength(3))?,
                ),
                InterpolatorEnum::InterpND(interp) => interp.interpolate(point),
            }
        }
    };
    (interpolate_fast) => {
        fn interpolate_fast(&self, point: &[D::Elem]) -> D::Elem {
            match self {
                InterpolatorEnum::Interp0D(interp) => interp.0,
                InterpolatorEnum::Interp1D(interp) => interp.interpolate_fast(
                    point
                        .try_into()
                        .expect("interpolate_fast: point length mismatch"),
                ),
                InterpolatorEnum::Interp2D(interp) => interp.interpolate_fast(
                    point
                        .try_into()
                        .expect("interpolate_fast: point length mismatch"),
                ),
                InterpolatorEnum::Interp3D(interp) => interp.interpolate_fast(
                    point
                        .try_into()
                        .expect("interpolate_fast: point length mismatch"),
                ),
                InterpolatorEnum::InterpND(interp) => interp.interpolate_fast(point),
            }
        }
    };
    (batch_interpolate) => {
        fn batch_interpolate(
            &self,
            points: &[&[D::Elem]],
        ) -> Result<Vec<D::Elem>, InterpolateError> {
            match self {
                InterpolatorEnum::Interp0D(interp) => {
                    Interpolator::batch_interpolate(interp, points)
                }
                InterpolatorEnum::Interp1D(interp) => {
                    let points: Vec<[D::Elem; 1]> = points
                        .iter()
                        .map(|&point| {
                            point
                                .try_into()
                                .map_err(|_| InterpolateError::PointLength(1))
                        })
                        .collect::<Result<_, _>>()?;
                    interp.batch_interpolate(&points)
                }
                InterpolatorEnum::Interp2D(interp) => {
                    let points: Vec<[D::Elem; 2]> = points
                        .iter()
                        .map(|&point| {
                            point
                                .try_into()
                                .map_err(|_| InterpolateError::PointLength(2))
                        })
                        .collect::<Result<_, _>>()?;
                    interp.batch_interpolate(&points)
                }
                InterpolatorEnum::Interp3D(interp) => {
                    let points: Vec<[D::Elem; 3]> = points
                        .iter()
                        .map(|&point| {
                            point
                                .try_into()
                                .map_err(|_| InterpolateError::PointLength(3))
                        })
                        .collect::<Result<_, _>>()?;
                    interp.batch_interpolate(&points)
                }
                InterpolatorEnum::InterpND(interp) => interp.batch_interpolate(points),
            }
        }
    };
    (batch_interpolate_fast) => {
        fn batch_interpolate_fast(&self, points: &[&[D::Elem]]) -> Vec<D::Elem> {
            match self {
                InterpolatorEnum::Interp0D(interp) => vec![interp.0; points.len()],
                InterpolatorEnum::Interp1D(interp) => {
                    let points: Vec<[D::Elem; 1]> = points
                        .iter()
                        .map(|&point| {
                            point
                                .try_into()
                                .expect("batch_interpolate_fast: point length mismatch")
                        })
                        .collect();
                    interp.batch_interpolate_fast(&points)
                }
                InterpolatorEnum::Interp2D(interp) => {
                    let points: Vec<[D::Elem; 2]> = points
                        .iter()
                        .map(|&point| {
                            point
                                .try_into()
                                .expect("batch_interpolate_fast: point length mismatch")
                        })
                        .collect();
                    interp.batch_interpolate_fast(&points)
                }
                InterpolatorEnum::Interp3D(interp) => {
                    let points: Vec<[D::Elem; 3]> = points
                        .iter()
                        .map(|&point| {
                            point
                                .try_into()
                                .expect("batch_interpolate_fast: point length mismatch")
                        })
                        .collect();
                    interp.batch_interpolate_fast(&points)
                }
                InterpolatorEnum::InterpND(interp) => interp.batch_interpolate_fast(points),
            }
        }
    };
}

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
/// See also: `examples/swap_interpolator.rs`
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
    D::Elem: PartialEq + Debug,
{
    Interp0D(Interp0D<D::Elem>),
    Interp1D(Interp1D<D, Strategy1DEnum>),
    Interp2D(Interp2D<D, Strategy2DEnum>),
    Interp3D(Interp3D<D, Strategy3DEnum>),
    InterpND(InterpND<D, StrategyNDEnum>),
}
/// [`InterpolatorEnum`] that views data.
pub type InterpolatorEnumViewed<T> = InterpolatorEnum<ViewRepr<T>>;
/// [`InterpolatorEnum`] that owns data.
pub type InterpolatorEnumOwned<T> = InterpolatorEnum<OwnedRepr<T>>;

#[cfg(feature = "serde")]
enum_serialize_nested_impl!(InterpolatorEnum);

enum_partialeq_impl!(InterpolatorEnum);

impl<D> InterpolatorEnum<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
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

    enum_method_wrap!(view);
    enum_method_wrap!(into_owned);
    enum_method_forward!(check_extrapolate);
    enum_method_forward!(validate_strategy);
    enum_method_forward!(mut init_strategy);
}

impl<D> Interpolator<D::Elem> for InterpolatorEnum<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Euclid + Debug,
{
    enum_trait_method!(ndim);
    enum_trait_method!(validate);
    enum_trait_method!(mut set_extrapolate, extrapolate: Extrapolate<D::Elem>);
    slice_to_array_forward!(interpolate);
    slice_to_array_forward!(interpolate_fast);
    slice_to_array_forward!(batch_interpolate);
    slice_to_array_forward!(batch_interpolate_fast);
}

from_impls!();

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
    fn test_point_length() {
        // `InterpolatorEnum::interpolate` takes a real slice (not an inherent
        // fixed-size array), so a wrong-length point is still a runtime `Err` here,
        // for every variant with a fixed `N`.
        let interp_1d = InterpolatorEnum::new_1d(
            array![0., 1., 2., 3., 4.],
            array![0.2, 0.4, 0.6, 0.8, 1.0],
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap();
        assert!(matches!(
            interp_1d.interpolate(&[]).unwrap_err(),
            InterpolateError::PointLength(1)
        ));

        let interp_2d = InterpolatorEnum::new_2d(
            array![0.05, 0.10, 0.15],
            array![0.10, 0.20, 0.30],
            array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap();
        assert!(matches!(
            interp_2d.interpolate(&[]).unwrap_err(),
            InterpolateError::PointLength(2)
        ));

        let interp_3d = InterpolatorEnum::new_3d(
            array![0.05, 0.10],
            array![0.10, 0.20],
            array![0.20, 0.40],
            array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]],
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap();
        assert!(matches!(
            interp_3d.interpolate(&[]).unwrap_err(),
            InterpolateError::PointLength(3)
        ));
    }

    #[test]
    fn test_batch_interpolate() {
        let interp = InterpolatorEnum::new_1d(
            array![0., 1., 2., 3., 4.],
            array![0.2, 0.4, 0.6, 0.8, 1.0],
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap();
        let points: [&[f64]; 2] = [&[1.0], &[3.0]];
        assert_eq!(interp.batch_interpolate(&points).unwrap(), vec![0.4, 0.8]);
        assert_eq!(interp.batch_interpolate_fast(&points), vec![0.4, 0.8]);
    }

    #[test]
    fn test_batch_interpolate_0d() {
        let interp: InterpolatorEnumOwned<f64> = InterpolatorEnum::new_0d(0.5);
        let points: [&[f64]; 3] = [&[], &[], &[]];
        assert_eq!(
            interp.batch_interpolate(&points).unwrap(),
            vec![0.5, 0.5, 0.5]
        );
        assert_eq!(interp.batch_interpolate_fast(&points), vec![0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_batch_interpolate_nd() {
        let interp = InterpolatorEnum::new_nd(
            vec![array![0., 1.], array![0., 1.], array![0., 1.]],
            array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]].into_dyn(),
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap();
        let points: [&[f64]; 2] = [&[0.25, 0.65, 0.9], &[0.5, 0.5, 0.5]];
        let batched = interp.batch_interpolate(&points).unwrap();
        let looped: Vec<_> = points
            .iter()
            .map(|point| interp.interpolate(point).unwrap())
            .collect();
        assert_eq!(batched, looped);
    }

    #[test]
    fn test_batch_interpolate_point_length_mismatch() {
        let interp = InterpolatorEnum::new_2d(
            array![0.05, 0.10, 0.15],
            array![0.10, 0.20, 0.30],
            array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap();
        let points: [&[f64]; 1] = [&[0.075]];
        assert!(matches!(
            interp.batch_interpolate(&points).unwrap_err(),
            InterpolateError::PointLength(2)
        ));
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_serde() {
        let x = array![0.05, 0.10, 0.15];
        let y = array![0.10, 0.20, 0.30];
        let f_xy = array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]];
        let f_xy_dyn = f_xy.clone().into_dyn();

        let interp0: Interp2D<_, strategy::enums::Strategy2DEnum> = Interp2D::new(
            x.view(),
            y.view(),
            f_xy.view(),
            strategy::Nearest.into(),
            Extrapolate::Error,
        )
        .unwrap();
        let interp1 = InterpolatorEnum::from(interp0.clone());

        let interp2: InterpND<_, strategy::enums::StrategyNDEnum> = InterpND::new(
            vec![x.view(), y.view()],
            f_xy_dyn.view(),
            strategy::Nearest.into(),
            Extrapolate::Error,
        )
        .unwrap();
        let interp3 = InterpolatorEnum::from(interp2.view());

        assert_eq!(
            serde_json::to_string(&interp0).unwrap(),
            serde_json::to_string(&interp1).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&interp1).unwrap(),
            serde_json::to_string(&interp2).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&interp2).unwrap(),
            serde_json::to_string(&interp3).unwrap(),
        );
    }
}
