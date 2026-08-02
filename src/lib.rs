#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

/// The `prelude` module exposes a variety of types:
/// - All interpolator structs:
///   - [`Interp0D`](`interpolator::Interp0D`)
///   - [`Interp1D`](`interpolator::Interp1D`)
///   - [`Interp2D`](`interpolator::Interp2D`)
///   - [`Interp3D`](`interpolator::Interp3D`)
///   - [`InterpND`](`interpolator::InterpND`)
///   - A `serde`-compatible interpolator enum [`InterpolatorEnum`](`interpolator::enums::InterpolatorEnum`)
///   - `Owned` and `Viewed` type aliases for all of the above
/// - Their common trait: [`Interpolator`]
/// - The [`strategy`] mod, containing pre-defined interpolation strategies:
///   - [`strategy::Linear`]
///   - [`strategy::Nearest`]
///   - [`strategy::LeftNearest`]
///   - [`strategy::RightNearest`]
///   - `serde`-compatible strategy enums: [`strategy::enums::Strategy1DEnum`]/etc.
/// - The extrapolation setting enum: [`Extrapolate`]
pub mod prelude {
    pub use crate::strategy;

    pub use crate::interpolator::{Extrapolate, Interpolator};

    pub use crate::interpolator::Interp0D;
    pub use crate::interpolator::{Interp1D, Interp1DOwned, Interp1DViewed};
    pub use crate::interpolator::{Interp2D, Interp2DOwned, Interp2DViewed};
    pub use crate::interpolator::{Interp3D, Interp3DOwned, Interp3DViewed};
    pub use crate::interpolator::{InterpND, InterpNDOwned, InterpNDViewed};

    pub use crate::interpolator::enums::{
        InterpolatorEnum, InterpolatorEnumOwned, InterpolatorEnumViewed,
    };
}

pub mod error;
pub mod strategy;

pub mod interpolator;
pub use interpolator::data;
pub(crate) use interpolator::data::*;
pub(crate) use interpolator::*;

pub(crate) use error::*;
pub(crate) use strategy::traits::*;

pub(crate) use std::fmt::Debug;

pub use ndarray;
pub(crate) use ndarray::prelude::*;
pub(crate) use ndarray::{Data, Ix, OwnedRepr, RawDataClone, ViewRepr};

pub use num_traits;
pub(crate) use num_traits::{clamp, Euclid, Num, One};

pub(crate) use dyn_clone::*;

#[cfg(feature = "serde")]
#[path = "serde.rs"]
mod serde_mod;
#[cfg(feature = "serde")]
pub(crate) use serde_mod::*;

#[cfg(test)]
/// Alias for [`approx::assert_abs_diff_eq`] with `epsilon = 1e-6`
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr $(,)?) => {
        approx::assert_abs_diff_eq!($a, $b, epsilon = 1e-6)
    };
    ($a:expr, $b:expr, $eps:expr $(,)?) => {
        approx::assert_abs_diff_eq!($a, $b, epsilon = $eps)
    };
}
#[cfg(test)]
pub(crate) use assert_approx_eq;

/// Wrap value around data bounds.
/// Assumes `min` < `max`.
pub(crate) fn wrap<T: Num + Euclid + Copy>(input: T, min: T, max: T) -> T {
    min + (input - min).rem_euclid(&(max - min))
}

#[cfg(test)]
mod tests {
    use super::wrap;

    #[test]
    fn test_wrap() {
        assert_eq!(wrap(-3, -2, 5), 4);
        assert_eq!(wrap(3, -2, 5), 3);
        assert_eq!(wrap(6, -2, 5), -1);
        assert_eq!(wrap(5, 0, 10), 5);
        assert_eq!(wrap(11, 0, 10), 1);
        assert_eq!(wrap(-3, 0, 10), 7);
        assert_eq!(wrap(-11, 0, 10), 9);
        assert_eq!(wrap(-0.1, -2., -1.), -1.1);
        assert_eq!(wrap(-0., -2., -1.), -2.0);
        assert_eq!(wrap(0.1, -2., -1.), -1.9);
        assert_eq!(wrap(-0.5, -1., 1.), -0.5);
        assert_eq!(wrap(0., -1., 1.), 0.);
        assert_eq!(wrap(0.5, -1., 1.), 0.5);
        assert_eq!(wrap(0.8, -1., 1.), 0.8);
    }
}
