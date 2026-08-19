//! This module provides an [`InterpolatorEnum`] that allow mutable interpolator swapping.

// NOTE: `enum_dispatch` does essentially what this module does, but with less boilerplate.
// However, it does not currently support using a generic trait on a non-generic enum.
// https://gitlab.com/antonok/enum_dispatch/-/issues/67

use super::*;

use strategy::enums::*;

/// Generates simple forwarding methods where Interp0D returns Ok(()), others forward to variant.
macro_rules! enum_method_forward {
    (validate_strategy) => {
        #[doc = "Re-run the current variant's strategy `validate` against its data.\n\n`new_0d`/`new_1d`/etc. already call this internally, so this is only needed\nafter mutating a variant's `data`/`strategy` fields directly (via `match`)."]
        pub fn validate_strategy(&self) -> Result<(), ValidateError> {
            match self {
                InterpolatorEnumBase::Interp0D(_) => Ok(()),
                InterpolatorEnumBase::Interp1D(interp) => interp.validate_strategy(),
                InterpolatorEnumBase::Interp2D(interp) => interp.validate_strategy(),
                InterpolatorEnumBase::Interp3D(interp) => interp.validate_strategy(),
                InterpolatorEnumBase::InterpND(interp) => interp.validate_strategy(),
            }
        }
    };
    (validate_extrapolate) => {
        #[doc = "Check that `extrapolate` is applicable to the current variant's strategy.\n\nForwards to the variant's own `validate_extrapolate`; `Interp0D` accepts any\nsetting, since it never extrapolates."]
        pub fn validate_extrapolate(
            &self,
            extrapolate: &Extrapolate<D::Elem>,
        ) -> Result<(), ValidateError> {
            match self {
                InterpolatorEnumBase::Interp0D(_) => Ok(()),
                InterpolatorEnumBase::Interp1D(interp) => interp.validate_extrapolate(extrapolate),
                InterpolatorEnumBase::Interp2D(interp) => interp.validate_extrapolate(extrapolate),
                InterpolatorEnumBase::Interp3D(interp) => interp.validate_extrapolate(extrapolate),
                InterpolatorEnumBase::InterpND(interp) => interp.validate_extrapolate(extrapolate),
            }
        }
    };
    (mut init_strategy) => {
        #[doc = "Re-run the current variant's strategy `init` against its data.\n\n`new_0d`/`new_1d`/etc. already call this internally, so this is only needed\nafter bypassing them: mutating a variant's `data`/`strategy` fields directly\n(via `match`), or deserializing an interpolator with a stateful custom strategy\n(`Deserialize` does not call `init`)."]
        pub fn init_strategy(&mut self) -> Result<(), ValidateError> {
            match self {
                InterpolatorEnumBase::Interp0D(_) => Ok(()),
                InterpolatorEnumBase::Interp1D(interp) => interp.init_strategy(),
                InterpolatorEnumBase::Interp2D(interp) => interp.init_strategy(),
                InterpolatorEnumBase::Interp3D(interp) => interp.init_strategy(),
                InterpolatorEnumBase::InterpND(interp) => interp.init_strategy(),
            }
        }
    };
}

/// Generates enum-wrapping forwarding methods (view, into_owned).
macro_rules! enum_method_wrap {
    (view) => {
        #[doc = "Return an interpolator with viewed data."]
        pub fn view(&self) -> InterpolatorEnumView<&D::Elem> {
            match self {
                InterpolatorEnumBase::Interp0D(interp) => InterpolatorEnumBase::Interp0D(interp.clone()),
                InterpolatorEnumBase::Interp1D(interp) => InterpolatorEnumBase::Interp1D(interp.view()),
                InterpolatorEnumBase::Interp2D(interp) => InterpolatorEnumBase::Interp2D(interp.view()),
                InterpolatorEnumBase::Interp3D(interp) => InterpolatorEnumBase::Interp3D(interp.view()),
                InterpolatorEnumBase::InterpND(interp) => InterpolatorEnumBase::InterpND(interp.view()),
            }
        }
    };
    (into_owned) => {
        #[doc = "Turn the interpolator into an [`InterpolatorEnum`], cloning the array elements if necessary."]
        pub fn into_owned(self) -> InterpolatorEnum<D::Elem>
        where
            D::Elem: Clone,
        {
            match self {
                InterpolatorEnumBase::Interp0D(interp) => InterpolatorEnumBase::Interp0D(interp.clone()),
                InterpolatorEnumBase::Interp1D(interp) => InterpolatorEnumBase::Interp1D(interp.into_owned()),
                InterpolatorEnumBase::Interp2D(interp) => InterpolatorEnumBase::Interp2D(interp.into_owned()),
                InterpolatorEnumBase::Interp3D(interp) => InterpolatorEnumBase::Interp3D(interp.into_owned()),
                InterpolatorEnumBase::InterpND(interp) => InterpolatorEnumBase::InterpND(interp.into_owned()),
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
                InterpolatorEnumBase::Interp0D(_) => 0,
                InterpolatorEnumBase::Interp1D(_) => 1,
                InterpolatorEnumBase::Interp2D(_) => 2,
                InterpolatorEnumBase::Interp3D(_) => 3,
                InterpolatorEnumBase::InterpND(interp) => interp.ndim(),
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

/// Generates trait impl methods with slice-to-array conversion and doc comments built in.
macro_rules! slice_to_array_forward {
    (interpolate) => {
        fn interpolate(&self, point: &[D::Elem]) -> Result<D::Elem, InterpolateError> {
            match self {
                InterpolatorEnumBase::Interp0D(interp) => interp.interpolate(point),
                InterpolatorEnumBase::Interp1D(interp) => {
                    interp.interpolate(to_fixed_point(point)?)
                }
                InterpolatorEnumBase::Interp2D(interp) => {
                    interp.interpolate(to_fixed_point(point)?)
                }
                InterpolatorEnumBase::Interp3D(interp) => {
                    interp.interpolate(to_fixed_point(point)?)
                }
                InterpolatorEnumBase::InterpND(interp) => interp.interpolate(point),
            }
        }
    };
    (interpolate_fast) => {
        fn interpolate_fast(&self, point: &[D::Elem]) -> D::Elem {
            match self {
                InterpolatorEnumBase::Interp0D(interp) => interp.0,
                InterpolatorEnumBase::Interp1D(interp) => interp.interpolate_fast(
                    point
                        .try_into()
                        .expect("interpolate_fast: point length mismatch"),
                ),
                InterpolatorEnumBase::Interp2D(interp) => interp.interpolate_fast(
                    point
                        .try_into()
                        .expect("interpolate_fast: point length mismatch"),
                ),
                InterpolatorEnumBase::Interp3D(interp) => interp.interpolate_fast(
                    point
                        .try_into()
                        .expect("interpolate_fast: point length mismatch"),
                ),
                InterpolatorEnumBase::InterpND(interp) => interp.interpolate_fast(point),
            }
        }
    };
    (batch_interpolate) => {
        fn batch_interpolate(
            &self,
            points: &[&[D::Elem]],
        ) -> Result<Vec<D::Elem>, InterpolateError> {
            match self {
                InterpolatorEnumBase::Interp0D(interp) => {
                    Interpolator::batch_interpolate(interp, points)
                }
                InterpolatorEnumBase::Interp1D(interp) => {
                    interp.batch_interpolate(&to_fixed_points(points)?)
                }
                InterpolatorEnumBase::Interp2D(interp) => {
                    interp.batch_interpolate(&to_fixed_points(points)?)
                }
                InterpolatorEnumBase::Interp3D(interp) => {
                    interp.batch_interpolate(&to_fixed_points(points)?)
                }
                InterpolatorEnumBase::InterpND(interp) => interp.batch_interpolate(points),
            }
        }
    };
    (batch_interpolate_fast) => {
        fn batch_interpolate_fast(&self, points: &[&[D::Elem]]) -> Vec<D::Elem> {
            match self {
                InterpolatorEnumBase::Interp0D(interp) => vec![interp.0; points.len()],
                InterpolatorEnumBase::Interp1D(interp) => {
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
                InterpolatorEnumBase::Interp2D(interp) => {
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
                InterpolatorEnumBase::Interp3D(interp) => {
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
                InterpolatorEnumBase::InterpND(interp) => interp.batch_interpolate_fast(points),
            }
        }
    };
    (batch_interpolate_into) => {
        fn batch_interpolate_into(
            &self,
            points: &[&[D::Elem]],
            out: &mut [D::Elem],
        ) -> Result<(), InterpolateError> {
            match self {
                InterpolatorEnumBase::Interp0D(interp) => {
                    if out.len() != points.len() {
                        return Err(InterpolateError::OutputLength {
                            expected: points.len(),
                            found: out.len(),
                        });
                    }
                    for o in out.iter_mut() {
                        *o = interp.0;
                    }
                    Ok(())
                }
                InterpolatorEnumBase::Interp1D(interp) => {
                    interp.batch_interpolate_into(&to_fixed_points(points)?, out)
                }
                InterpolatorEnumBase::Interp2D(interp) => {
                    interp.batch_interpolate_into(&to_fixed_points(points)?, out)
                }
                InterpolatorEnumBase::Interp3D(interp) => {
                    interp.batch_interpolate_into(&to_fixed_points(points)?, out)
                }
                InterpolatorEnumBase::InterpND(interp) => {
                    interp.batch_interpolate_into(points, out)
                }
            }
        }
    };
    (batch_interpolate_fast_into) => {
        fn batch_interpolate_fast_into(&self, points: &[&[D::Elem]], out: &mut [D::Elem]) {
            match self {
                InterpolatorEnumBase::Interp0D(interp) => {
                    for o in out.iter_mut() {
                        *o = interp.0;
                    }
                }
                InterpolatorEnumBase::Interp1D(interp) => {
                    let points: Vec<[D::Elem; 1]> = points
                        .iter()
                        .map(|&point| {
                            point
                                .try_into()
                                .expect("batch_interpolate_fast_into: point length mismatch")
                        })
                        .collect();
                    interp.batch_interpolate_fast_into(&points, out)
                }
                InterpolatorEnumBase::Interp2D(interp) => {
                    let points: Vec<[D::Elem; 2]> = points
                        .iter()
                        .map(|&point| {
                            point
                                .try_into()
                                .expect("batch_interpolate_fast_into: point length mismatch")
                        })
                        .collect();
                    interp.batch_interpolate_fast_into(&points, out)
                }
                InterpolatorEnumBase::Interp3D(interp) => {
                    let points: Vec<[D::Elem; 3]> = points
                        .iter()
                        .map(|&point| {
                            point
                                .try_into()
                                .expect("batch_interpolate_fast_into: point length mismatch")
                        })
                        .collect();
                    interp.batch_interpolate_fast_into(&points, out)
                }
                InterpolatorEnumBase::InterpND(interp) => {
                    interp.batch_interpolate_fast_into(points, out)
                }
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
/// let mut interp: InterpolatorEnum<_> = InterpolatorEnumBase::new_1d(
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
/// interp = InterpolatorEnumBase::new_2d(
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
///     InterpolatorEnumBase::Interp2D(interp) => &interp.data.values,
///     _ => unreachable!(),
/// };
///
/// assert_eq!(interp.interpolate(&[0.08, 0.21]).unwrap(), f_xy[[1, 1]]);
/// assert_eq!(interp.interpolate(&[0.11, 0.26]).unwrap(), f_xy[[1, 2]]);
/// assert_eq!(interp.interpolate(&[0.13, 0.12]).unwrap(), f_xy[[2, 0]]);
/// assert_eq!(interp.interpolate(&[0.14, 0.29]).unwrap(), f_xy[[2, 2]]);
///
/// // 0-D
/// interp = InterpolatorEnumBase::new_0d(0.5);
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
        // `Zero` is `CubicC2`'s requirement (via `Strategy*Enum`'s own bound), not
        // something every variant here needs on its own.
        serialize = "D::Elem: Serialize + Zero",
        deserialize = "
            D: DataOwned,
            D::Elem: Deserialize<'de> + Zero,
        "
    ))
)]
pub enum InterpolatorEnumBase<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug + Clone,
{
    Interp0D(Interp0D<D::Elem>),
    Interp1D(Interp1DBase<D, Strategy1DEnum<D::Elem>>),
    Interp2D(Interp2DBase<D, Strategy2DEnum<D::Elem>>),
    Interp3D(Interp3DBase<D, Strategy3DEnum<D::Elem>>),
    InterpND(InterpNDBase<D, StrategyNDEnum<D::Elem>>),
}
/// Owned interpolator enum (see [`InterpolatorEnumBase`] for the generic form).
pub type InterpolatorEnum<T> = InterpolatorEnumBase<OwnedRepr<T>>;
/// Viewed interpolator enum (see [`InterpolatorEnumBase`] for the generic form).
pub type InterpolatorEnumView<T> = InterpolatorEnumBase<ViewRepr<T>>;

#[cfg(feature = "serde")]
impl<D> SerializeNested for InterpolatorEnumBase<D>
where
    D: Data + RawDataClone + Clone,
    // `Zero` is `CubicC2`'s requirement (via `Strategy*Enum`'s own bound), not
    // something every variant here needs on its own.
    D::Elem: PartialEq + Debug + Clone + Serialize + Zero,
{
    /// `#[serde(untagged)]`, so each variant serializes as its inner value.
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

impl<D> PartialEq for InterpolatorEnumBase<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug + Clone,
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

impl<D> From<Interp0D<D::Elem>> for InterpolatorEnumBase<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    #[inline]
    fn from(interpolator: Interp0D<D::Elem>) -> Self {
        InterpolatorEnumBase::Interp0D(interpolator)
    }
}

impl<D, S> From<Interp1DBase<D, S>> for InterpolatorEnumBase<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
    S: Into<Strategy1DEnum<D::Elem>> + Clone,
{
    #[inline]
    fn from(interpolator: Interp1DBase<D, S>) -> Self {
        InterpolatorEnumBase::Interp1D(Interp1DBase {
            data: interpolator.data,
            strategy: interpolator.strategy.into(),
            extrapolate: interpolator.extrapolate,
        })
    }
}

impl<D, S> From<Interp2DBase<D, S>> for InterpolatorEnumBase<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
    S: Into<Strategy2DEnum<D::Elem>> + Clone,
{
    #[inline]
    fn from(interpolator: Interp2DBase<D, S>) -> Self {
        InterpolatorEnumBase::Interp2D(Interp2DBase {
            data: interpolator.data,
            strategy: interpolator.strategy.into(),
            extrapolate: interpolator.extrapolate,
        })
    }
}

impl<D, S> From<Interp3DBase<D, S>> for InterpolatorEnumBase<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
    S: Into<Strategy3DEnum<D::Elem>> + Clone,
{
    #[inline]
    fn from(interpolator: Interp3DBase<D, S>) -> Self {
        InterpolatorEnumBase::Interp3D(Interp3DBase {
            data: interpolator.data,
            strategy: interpolator.strategy.into(),
            extrapolate: interpolator.extrapolate,
        })
    }
}

impl<D, S> From<InterpNDBase<D, S>> for InterpolatorEnumBase<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
    S: Into<StrategyNDEnum<D::Elem>> + Clone,
{
    #[inline]
    fn from(interpolator: InterpNDBase<D, S>) -> Self {
        InterpolatorEnumBase::InterpND(InterpNDBase {
            data: interpolator.data,
            strategy: interpolator.strategy.into(),
            extrapolate: interpolator.extrapolate,
        })
    }
}

impl<D> InterpolatorEnumBase<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    /// Create [`InterpolatorEnumBase::Interp0D`], internally calling [`Interp0D::new`].
    #[inline]
    pub fn new_0d(value: D::Elem) -> Self {
        Self::Interp0D(Interp0D::new(value))
    }

    /// Create [`InterpolatorEnumBase::Interp1D`], internally calling [`Interp1D::new`].
    #[inline]
    pub fn new_1d(
        x: ArrayBase<D, Ix1>,
        f_x: ArrayBase<D, Ix1>,
        strategy: impl Into<Strategy1DEnum<D::Elem>>,
        extrapolate: Extrapolate<D::Elem>,
    ) -> Result<Self, ValidateError> {
        Ok(Self::Interp1D(Interp1DBase::new(
            x,
            f_x,
            strategy.into(),
            extrapolate,
        )?))
    }

    /// Create [`InterpolatorEnumBase::Interp2D`], internally calling [`Interp2D::new`].
    #[inline]
    pub fn new_2d(
        x: ArrayBase<D, Ix1>,
        y: ArrayBase<D, Ix1>,
        f_xy: ArrayBase<D, Ix2>,
        strategy: impl Into<Strategy2DEnum<D::Elem>>,
        extrapolate: Extrapolate<D::Elem>,
    ) -> Result<Self, ValidateError> {
        Ok(Self::Interp2D(Interp2DBase::new(
            x,
            y,
            f_xy,
            strategy.into(),
            extrapolate,
        )?))
    }

    /// Create [`InterpolatorEnumBase::Interp3D`], internally calling [`Interp3D::new`].
    #[inline]
    pub fn new_3d(
        x: ArrayBase<D, Ix1>,
        y: ArrayBase<D, Ix1>,
        z: ArrayBase<D, Ix1>,
        f_xyz: ArrayBase<D, Ix3>,
        strategy: impl Into<Strategy3DEnum<D::Elem>>,
        extrapolate: Extrapolate<D::Elem>,
    ) -> Result<Self, ValidateError> {
        Ok(Self::Interp3D(Interp3DBase::new(
            x,
            y,
            z,
            f_xyz,
            strategy.into(),
            extrapolate,
        )?))
    }

    /// Create [`InterpolatorEnumBase::InterpND`], internally calling [`InterpND::new`].
    #[inline]
    pub fn new_nd(
        grid: Vec<ArrayBase<D, Ix1>>,
        values: ArrayBase<D, IxDyn>,
        strategy: impl Into<StrategyNDEnum<D::Elem>>,
        extrapolate: Extrapolate<D::Elem>,
    ) -> Result<Self, ValidateError> {
        Ok(Self::InterpND(InterpNDBase::new(
            grid,
            values,
            strategy.into(),
            extrapolate,
        )?))
    }

    enum_method_wrap!(view);
    enum_method_wrap!(into_owned);
    enum_method_forward!(validate_extrapolate);
    enum_method_forward!(validate_strategy);
    enum_method_forward!(mut init_strategy);
}

impl<D> Interpolator<D::Elem> for InterpolatorEnumBase<D>
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
    slice_to_array_forward!(batch_interpolate_into);
    slice_to_array_forward!(batch_interpolate_fast_into);
}

mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_partialeq() {
        #[derive(PartialEq)]
        #[allow(unused)]
        struct MyStruct(InterpolatorEnum<f64>);
    }

    #[test]
    fn test_point_length() {
        // `InterpolatorEnumBase::interpolate` takes a real slice (not an inherent
        // fixed-size array), so a wrong-length point is still a runtime `Err` here,
        // for every variant with a fixed `N`.
        let interp_1d = InterpolatorEnumBase::new_1d(
            array![0., 1., 2., 3., 4.],
            array![0.2, 0.4, 0.6, 0.8, 1.0],
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap();
        assert!(matches!(
            interp_1d.interpolate(&[]).unwrap_err(),
            InterpolateError::PointLength {
                expected: 1,
                ref failures
            } if failures.as_slice() == [WrongLengthAt { index: 0, found: 0 }]
        ));

        let interp_2d = InterpolatorEnumBase::new_2d(
            array![0.05, 0.10, 0.15],
            array![0.10, 0.20, 0.30],
            array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap();
        assert!(matches!(
            interp_2d.interpolate(&[]).unwrap_err(),
            InterpolateError::PointLength {
                expected: 2,
                ref failures
            } if failures.as_slice() == [WrongLengthAt { index: 0, found: 0 }]
        ));

        let interp_3d = InterpolatorEnumBase::new_3d(
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
            InterpolateError::PointLength {
                expected: 3,
                ref failures
            } if failures.as_slice() == [WrongLengthAt { index: 0, found: 0 }]
        ));
    }

    #[test]
    fn test_batch_interpolate() {
        let interp = InterpolatorEnumBase::new_1d(
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
        let interp: InterpolatorEnum<f64> = InterpolatorEnumBase::new_0d(0.5);
        let points: [&[f64]; 3] = [&[], &[], &[]];
        assert_eq!(
            interp.batch_interpolate(&points).unwrap(),
            vec![0.5, 0.5, 0.5]
        );
        assert_eq!(interp.batch_interpolate_fast(&points), vec![0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_batch_interpolate_nd() {
        let interp = InterpolatorEnumBase::new_nd(
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
        let interp = InterpolatorEnumBase::new_2d(
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
            InterpolateError::PointLength {
                expected: 2,
                ref failures
            } if failures.as_slice() == [WrongLengthAt { index: 0, found: 1 }]
        ));
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_serde() {
        let x = array![0.05, 0.10, 0.15];
        let y = array![0.10, 0.20, 0.30];
        let f_xy = array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]];
        let f_xy_dyn = f_xy.clone().into_dyn();

        let interp0: Interp2DBase<_, strategy::enums::Strategy2DEnum<f64>> = Interp2DBase::new(
            x.view(),
            y.view(),
            f_xy.view(),
            strategy::Nearest.into(),
            Extrapolate::Error,
        )
        .unwrap();
        let interp1 = InterpolatorEnumBase::from(interp0.clone());

        let interp2: InterpNDBase<_, strategy::enums::StrategyNDEnum<f64>> = InterpNDBase::new(
            vec![x.view(), y.view()],
            f_xy_dyn.view(),
            strategy::Nearest.into(),
            Extrapolate::Error,
        )
        .unwrap();
        let interp3 = InterpolatorEnumBase::from(interp2.view());

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
