//! Pre-defined interpolation strategies and traits for custom strategies.
//!
//! "What can I pass as `strategy`?" Every strategy is re-exported here. Their
//! configuration types, which aren't themselves strategies, live in [`cubic`]/[`step`]
//! instead.

use super::*;

pub mod broadcast;
use broadcast::Broadcastable;

pub mod cubic;
pub use cubic::CubicC2;

pub mod step;
pub use step::Step;

pub mod transform;
pub(crate) use transform::{grid_transform_strategy_impl, values_transform_strategy_impl};
pub use transform::{GridTransform, Transform, ValuesTransform};

pub mod enums;
pub mod traits;
pub mod utils;

/// Placeholder for a strategy's coefficient/transform cache before
/// [`Strategy1D::init`](traits::Strategy1D::init)/`2D`/`3D`/`ND` populates it. Shared
/// by [`CubicC2::cache`] and [`GridTransform::grid_cache`]/[`ValuesTransform::
/// values_cache`]. No `T: Clone + Zero` bound needed (unlike `ArrayD::zeros`), so
/// callers stay unconstrained.
fn empty_cache<T>() -> ArrayD<T> {
    ArrayD::from_shape_vec(IxDyn(&[0]), Vec::new()).expect("empty shape matches empty vec")
}

/// Linear interpolation: <https://en.wikipedia.org/wiki/Linear_interpolation>
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize_unit_struct, Serialize_unit_struct)
)]
pub struct Linear;

/// Linear interpolation optimized for uniformly-spaced grids.
///
/// Uses an O(1) direct index calculation instead of the O(log N) binary search in [`Linear`].
/// The grid is validated to be uniformly spaced at construction time; a [`ValidateError`] is
/// returned if it is not.
///
/// # Examples
/// ```
/// use ndarray::prelude::*;
/// use ninterp::prelude::*;
///
/// let interp = Interp1D::new(
///     array![0., 1., 2., 3., 4.],
///     array![0.2, 0.4, 0.6, 0.8, 1.0],
///     strategy::LinearUniform,
///     Extrapolate::Error,
/// )
/// .unwrap();
/// assert_eq!(interp.interpolate(&[2.5]).unwrap(), 0.7);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize_unit_struct, Serialize_unit_struct)
)]
pub struct LinearUniform;

/// Nearest value interpolation: <https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation>
///
/// # Note
/// Float imprecision may affect the value returned near midpoints.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize_unit_struct, Serialize_unit_struct)
)]
pub struct Nearest;

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    #[cfg(feature = "serde")]
    fn test_serde() {
        assert_eq!(
            serde_json::to_string(&Linear).unwrap(),
            format!("\"{}\"", stringify!(Linear))
        );
        assert_eq!(
            serde_json::to_string(&LinearUniform).unwrap(),
            format!("\"{}\"", stringify!(LinearUniform))
        );
        assert_eq!(
            serde_json::to_string(&Nearest).unwrap(),
            format!("\"{}\"", stringify!(Nearest))
        );
    }
}
