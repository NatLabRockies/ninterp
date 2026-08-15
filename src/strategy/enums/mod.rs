//! This module provides enums that allow mutable strategy swapping.
//! The enum variants and `From` impls here are hand-maintained per strategy type.
//! Adding a new strategy requires explicit wiring in each of the 1-D/2-D/3-D/N-D enum modules.
//!
//! This is an alternative to using a `Box<dyn Strategy1D>`/etc. with a few key differences:
//! - Better runtime performance
//! - Compatible with serde
//! - **Incompatible** with custom strategies
//!
//! # Example:
//! ```
//! use ndarray::prelude::*;
//! use ninterp::prelude::*;
//!
//! let mut interp: Interp1D<_, strategy::enums::Strategy1DEnum<f64>> = Interp1D::new(
//!     // x
//!     array![0., 1., 2., 3., 4.],
//!     // f(x)
//!     array![0.2, 0.4, 0.6, 0.8, 1.0],
//!     strategy::Linear.into(),
//!     Extrapolate::Error,
//! )
//! .unwrap();
//! assert_eq!(interp.interpolate(&[3.00]).unwrap(), 0.8);
//! assert_eq!(interp.interpolate(&[3.75]).unwrap(), 0.95);
//! assert_eq!(interp.interpolate(&[4.00]).unwrap(), 1.0);
//!
//! interp.set_strategy(strategy::Nearest).unwrap();
//! assert_eq!(interp.interpolate(&[3.00]).unwrap(), 0.8);
//! assert_eq!(interp.interpolate(&[3.25]).unwrap(), 0.8);
//! assert_eq!(interp.interpolate(&[3.50]).unwrap(), 1.0);
//!
//! // Swap to LinearUniform: O(1) index lookup for this uniform grid
//! interp.set_strategy(strategy::LinearUniform).unwrap();
//! assert_eq!(interp.interpolate(&[3.00]).unwrap(), 0.8);
//! assert_eq!(interp.interpolate(&[3.75]).unwrap(), 0.95);
//! assert_eq!(interp.interpolate(&[4.00]).unwrap(), 1.0);
//!
//! // Piecewise-constant: fixed lower direction
//! interp.set_strategy(strategy::Step::lower()).unwrap();
//! assert_eq!(interp.interpolate(&[3.75]).unwrap(), 0.8);
//! assert_eq!(interp.interpolate(&[4.00]).unwrap(), 1.0);
//! ```
//! See also: `examples/swap_strategy.rs`

// NOTE: `enum_dispatch` does essentially what this module does, but with less boilerplate.
// However, it does not currently support using a generic trait on a non-generic enum.
// https://gitlab.com/antonok/enum_dispatch/-/issues/67

use super::*;

/// Generates enum + From impls + trait dispatch, shared by Strategy*DEnum types.
/// Takes enum name, trait name, data type name, point parameter type, and the list
/// of strategies (variant name + type pairs). This allows future customization if
/// strategies are added/removed per dimensionality.
macro_rules! strategy_enum_impl {
    (
        $EnumName:ident,
        $TraitName:ident,
        $DataType:ident,
        $PointType:ty,
        $PointsType:ty,
        [$(($Variant:ident, $StrategyType:ty)),* $(,)?]
    ) => {
        /// See [enums module](super) documentation.
        #[allow(missing_docs)]
        #[derive(Debug, Clone, PartialEq)]
        #[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
        #[cfg_attr(feature = "serde", serde(untagged))]
        // `T` is otherwise unused by every non-generic strategy (`Linear`, `Nearest`,
        // `Step`, `LinearUniform`); only `CubicC2<T>` actually depends on it, and its own
        // `Serialize`/`Deserialize` needs `Zero` (for the `Natural` bare-string
        // shorthand), so that's the bound the whole enum needs too.
        #[cfg_attr(
            feature = "serde",
            serde(bound(
                serialize = "T: Serialize + Zero",
                deserialize = "T: Deserialize<'de> + Zero"
            ))
        )]
        #[non_exhaustive]
        pub enum $EnumName<T> {
            $($Variant($StrategyType),)*
        }

        $(
            impl<T> From<$StrategyType> for $EnumName<T> {
                #[inline]
                fn from(strategy: $StrategyType) -> Self {
                    Self::$Variant(strategy)
                }
            }
        )*

        impl<D> $TraitName<D> for $EnumName<D::Elem>
        where
            D: Data + RawDataClone + Clone,
            D::Elem: Float + Debug,
        {
            #[inline]
            fn validate(&self, data: &$DataType<D>) -> Result<(), ValidateError> {
                match self {
                    $(Self::$Variant(strategy) => $TraitName::<D>::validate(strategy, data),)*
                }
            }

            #[inline]
            fn init(&mut self, data: &$DataType<D>) -> Result<(), ValidateError> {
                match self {
                    $(Self::$Variant(strategy) => $TraitName::<D>::init(strategy, data),)*
                }
            }

            #[inline]
            fn interpolate(
                &self,
                data: &$DataType<D>,
                point: $PointType,
            ) -> Result<D::Elem, InterpolateError> {
                match self {
                    $(Self::$Variant(strategy) => $TraitName::<D>::interpolate(strategy, data, point),)*
                }
            }

            #[inline]
            fn interpolate_wrapped(
                &self,
                data: &$DataType<D>,
                point: $PointType,
            ) -> Result<D::Elem, InterpolateError>
            where
                D::Elem: Num + Euclid + Copy,
            {
                match self {
                    $(Self::$Variant(strategy) => $TraitName::<D>::interpolate_wrapped(strategy, data, point),)*
                }
            }

            #[inline]
            fn interpolate_fast(&self, data: &$DataType<D>, point: $PointType) -> D::Elem {
                match self {
                    $(Self::$Variant(strategy) => $TraitName::<D>::interpolate_fast(strategy, data, point),)*
                }
            }

            #[inline]
            fn batch_interpolate_into(
                &self,
                data: &$DataType<D>,
                points: $PointsType,
                out: &mut [D::Elem],
            ) -> Result<(), InterpolateError> {
                match self {
                    $(Self::$Variant(strategy) => $TraitName::<D>::batch_interpolate_into(strategy, data, points, out),)*
                }
            }

            #[inline]
            fn batch_interpolate_fast_into(
                &self,
                data: &$DataType<D>,
                points: $PointsType,
                out: &mut [D::Elem],
            ) {
                match self {
                    $(Self::$Variant(strategy) => $TraitName::<D>::batch_interpolate_fast_into(strategy, data, points, out),)*
                }
            }

            #[inline]
            fn batch_interpolate(
                &self,
                data: &$DataType<D>,
                points: $PointsType,
            ) -> Result<Vec<D::Elem>, InterpolateError>
            where
                D::Elem: Num,
            {
                match self {
                    $(Self::$Variant(strategy) => $TraitName::<D>::batch_interpolate(strategy, data, points),)*
                }
            }

            #[inline]
            fn batch_interpolate_fast(&self, data: &$DataType<D>, points: $PointsType) -> Vec<D::Elem>
            where
                D::Elem: Num + Copy,
            {
                match self {
                    $(Self::$Variant(strategy) => $TraitName::<D>::batch_interpolate_fast(strategy, data, points),)*
                }
            }

            #[inline]
            fn allow_extrapolate(&self) -> bool {
                match self {
                    $(Self::$Variant(strategy) => $TraitName::<D>::allow_extrapolate(strategy),)*
                }
            }

            #[inline]
            fn check_batch_domain(&self, points: $PointsType) -> Result<(), InterpolateError> {
                match self {
                    $(Self::$Variant(strategy) => $TraitName::<D>::check_batch_domain(strategy, points),)*
                }
            }
        }

        /// See [enums module](super) documentation. `Box` here wraps the concrete
        /// enum (not `dyn Trait`), breaking the infinite size a wrapper strategy's
        /// `inner: Self` field would otherwise have if it named the enum directly
        /// (e.g. `GridTransform`'s `inner`); serde's blanket `Box<T>` impl covers it,
        /// so unlike `Box<dyn Trait>` this is fully serde-compatible.
        impl<D> $TraitName<D> for Box<$EnumName<D::Elem>>
        where
            D: Data + RawDataClone + Clone,
            D::Elem: Float + Debug,
        {
            #[inline]
            fn validate(&self, data: &$DataType<D>) -> Result<(), ValidateError> {
                (**self).validate(data)
            }

            #[inline]
            fn init(&mut self, data: &$DataType<D>) -> Result<(), ValidateError> {
                (**self).init(data)
            }

            #[inline]
            fn interpolate(&self, data: &$DataType<D>, point: $PointType) -> Result<D::Elem, InterpolateError> {
                (**self).interpolate(data, point)
            }

            #[inline]
            fn interpolate_wrapped(
                &self,
                data: &$DataType<D>,
                point: $PointType,
            ) -> Result<D::Elem, InterpolateError>
            where
                D::Elem: Num + Euclid + Copy,
            {
                (**self).interpolate_wrapped(data, point)
            }

            #[inline]
            fn interpolate_fast(&self, data: &$DataType<D>, point: $PointType) -> D::Elem {
                (**self).interpolate_fast(data, point)
            }

            #[inline]
            fn batch_interpolate_into(
                &self,
                data: &$DataType<D>,
                points: $PointsType,
                out: &mut [D::Elem],
            ) -> Result<(), InterpolateError> {
                (**self).batch_interpolate_into(data, points, out)
            }

            #[inline]
            fn batch_interpolate_fast_into(&self, data: &$DataType<D>, points: $PointsType, out: &mut [D::Elem]) {
                (**self).batch_interpolate_fast_into(data, points, out)
            }

            #[inline]
            fn batch_interpolate(
                &self,
                data: &$DataType<D>,
                points: $PointsType,
            ) -> Result<Vec<D::Elem>, InterpolateError>
            where
                D::Elem: Num,
            {
                (**self).batch_interpolate(data, points)
            }

            #[inline]
            fn batch_interpolate_fast(&self, data: &$DataType<D>, points: $PointsType) -> Vec<D::Elem>
            where
                D::Elem: Num + Copy,
            {
                (**self).batch_interpolate_fast(data, points)
            }

            #[inline]
            fn allow_extrapolate(&self) -> bool {
                $TraitName::<D>::allow_extrapolate(&**self)
            }

            #[inline]
            fn check_batch_domain(&self, points: $PointsType) -> Result<(), InterpolateError> {
                $TraitName::<D>::check_batch_domain(&**self, points)
            }
        }
    };
}
#[allow(unused_imports)]
pub(crate) use strategy_enum_impl;

mod n;
mod one;
mod three;
mod two;

pub use n::*;
pub use one::*;
pub use three::*;
pub use two::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use ndarray::prelude::*;

    #[test]
    fn test_1d() {
        let mut interp: Interp1D<_, strategy::enums::Strategy1DEnum<f64>> = Interp1D::new(
            array![0., 1., 2., 3., 4.],
            array![0.2, 0.4, 0.6, 0.8, 1.0],
            strategy::Linear.into(),
            Extrapolate::Error,
        )
        .unwrap();
        assert_eq!(interp.interpolate(&[3.00]).unwrap(), 0.8);
        assert_eq!(interp.interpolate(&[3.75]).unwrap(), 0.95);
        assert_eq!(interp.interpolate(&[4.00]).unwrap(), 1.0);

        interp.set_strategy(strategy::Nearest).unwrap();
        assert_eq!(interp.interpolate(&[3.00]).unwrap(), 0.8);
        assert_eq!(interp.interpolate(&[3.25]).unwrap(), 0.8);
        assert_eq!(interp.interpolate(&[3.50]).unwrap(), 1.0);
        assert_eq!(interp.interpolate(&[3.75]).unwrap(), 1.0);
        assert_eq!(interp.interpolate(&[4.00]).unwrap(), 1.0);

        interp.set_strategy(strategy::Step::lower()).unwrap();
        assert_eq!(interp.interpolate(&[3.00]).unwrap(), 0.8);
        assert_eq!(interp.interpolate(&[3.75]).unwrap(), 0.8);
        assert_eq!(interp.interpolate(&[4.00]).unwrap(), 1.0);
    }

    #[test]
    fn test_2d() {
        let mut interp: Interp2D<_, strategy::enums::Strategy2DEnum<f64>> = Interp2D::new(
            array![0.05, 0.10, 0.15],
            array![0.10, 0.20, 0.30],
            array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
            strategy::Linear.into(),
            Extrapolate::Error,
        )
        .unwrap();
        let x = &interp.data.grid[0];
        let y = &interp.data.grid[1];
        let f_xy = &interp.data.values;

        assert_eq!(interp.interpolate(&[x[2], y[1]]).unwrap(), f_xy[[2, 1]]);
        assert_eq!(interp.interpolate(&[0.075, 0.25]).unwrap(), 3.);

        interp.set_strategy(strategy::Nearest).unwrap();
        let f_xy = &interp.data.values; // need fresh reference after mutation

        assert_eq!(interp.interpolate(&[0.05, 0.12]).unwrap(), f_xy[[0, 0]]);
        assert_eq!(
            // float imprecision
            interp.interpolate(&[0.07, 0.15 + 0.0001]).unwrap(),
            f_xy[[0, 1]]
        );
        assert_eq!(interp.interpolate(&[0.08, 0.21]).unwrap(), f_xy[[1, 1]]);
        assert_eq!(interp.interpolate(&[0.11, 0.26]).unwrap(), f_xy[[1, 2]]);
        assert_eq!(interp.interpolate(&[0.13, 0.12]).unwrap(), f_xy[[2, 0]]);
        assert_eq!(interp.interpolate(&[0.14, 0.29]).unwrap(), f_xy[[2, 2]]);
    }

    #[test]
    fn test_3d() {
        let mut interp: Interp3D<_, strategy::enums::Strategy3DEnum<f64>> = Interp3D::new(
            array![0.05, 0.10, 0.15],
            array![0.10, 0.20, 0.30],
            array![0.20, 0.40, 0.60],
            array![
                [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
                [[9., 10., 11.], [12., 13., 14.], [15., 16., 17.]],
                [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.],],
            ],
            strategy::Linear.into(),
            Extrapolate::Error,
        )
        .unwrap();
        let x = &interp.data.grid[0];
        let y = &interp.data.grid[1];
        let z = &interp.data.grid[2];

        assert_approx_eq!(interp.interpolate(&[x[0], y[0], 0.3]).unwrap(), 0.5);
        assert_approx_eq!(interp.interpolate(&[x[0], 0.15, z[0]]).unwrap(), 1.5);
        assert_approx_eq!(interp.interpolate(&[x[0], 0.15, 0.3]).unwrap(), 2.);
        assert_approx_eq!(interp.interpolate(&[0.075, y[0], z[0]]).unwrap(), 4.5);
        assert_approx_eq!(interp.interpolate(&[0.075, y[0], 0.3]).unwrap(), 5.);
        assert_approx_eq!(interp.interpolate(&[0.075, 0.15, z[0]]).unwrap(), 6.);

        interp.set_strategy(strategy::Nearest).unwrap();
        assert_eq!(interp.interpolate(&[0.06, 0.11, 0.22]).unwrap(), 0.);
        assert_eq!(interp.interpolate(&[0.06, 0.11, 0.31]).unwrap(), 1.);
        assert_eq!(interp.interpolate(&[0.06, 0.19, 0.22]).unwrap(), 3.);
        assert_eq!(interp.interpolate(&[0.06, 0.19, 0.31]).unwrap(), 4.);
        assert_eq!(interp.interpolate(&[0.09, 0.11, 0.22]).unwrap(), 9.);
        assert_eq!(interp.interpolate(&[0.09, 0.11, 0.31]).unwrap(), 10.);
        assert_eq!(interp.interpolate(&[0.09, 0.19, 0.22]).unwrap(), 12.);
        assert_eq!(interp.interpolate(&[0.09, 0.19, 0.31]).unwrap(), 13.);
    }

    #[test]
    fn test_nd() {
        let mut interp: InterpND<_, strategy::enums::StrategyNDEnum<f64>> = InterpND::new(
            vec![
                array![0.05, 0.10, 0.15],
                array![0.10, 0.20, 0.30],
                array![0.20, 0.40, 0.60],
            ],
            array![
                [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
                [[9., 10., 11.], [12., 13., 14.], [15., 16., 17.]],
                [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.],],
            ]
            .into_dyn(),
            strategy::Linear.into(),
            Extrapolate::Error,
        )
        .unwrap();
        let x = &interp.data.grid[0];
        let y = &interp.data.grid[1];
        let z = &interp.data.grid[2];

        assert_approx_eq!(interp.interpolate(&[x[0], y[0], 0.3]).unwrap(), 0.5);
        assert_approx_eq!(interp.interpolate(&[x[0], 0.15, z[0]]).unwrap(), 1.5);
        assert_approx_eq!(interp.interpolate(&[x[0], 0.15, 0.3]).unwrap(), 2.);
        assert_approx_eq!(interp.interpolate(&[0.075, y[0], z[0]]).unwrap(), 4.5);
        assert_approx_eq!(interp.interpolate(&[0.075, y[0], 0.3]).unwrap(), 5.);
        assert_approx_eq!(interp.interpolate(&[0.075, 0.15, z[0]]).unwrap(), 6.);

        interp.set_strategy(strategy::Nearest).unwrap();
        assert_eq!(interp.interpolate(&[0.06, 0.11, 0.22]).unwrap(), 0.);
        assert_eq!(interp.interpolate(&[0.06, 0.11, 0.31]).unwrap(), 1.);
        assert_eq!(interp.interpolate(&[0.06, 0.19, 0.22]).unwrap(), 3.);
        assert_eq!(interp.interpolate(&[0.06, 0.19, 0.31]).unwrap(), 4.);
        assert_eq!(interp.interpolate(&[0.09, 0.11, 0.22]).unwrap(), 9.);
        assert_eq!(interp.interpolate(&[0.09, 0.11, 0.31]).unwrap(), 10.);
        assert_eq!(interp.interpolate(&[0.09, 0.19, 0.22]).unwrap(), 12.);
        assert_eq!(interp.interpolate(&[0.09, 0.19, 0.31]).unwrap(), 13.);
    }
}
