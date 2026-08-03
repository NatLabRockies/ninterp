//! Pre-defined interpolation strategies and traits for custom strategies

use super::*;

pub mod enums;
pub mod traits;

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
/// ).unwrap();
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

/// Left-nearest (previous value) interpolation: <https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation>
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize_unit_struct, Serialize_unit_struct)
)]
pub struct LeftNearest;

/// Right-nearest (next value) interpolation: <https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation>
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize_unit_struct, Serialize_unit_struct)
)]
pub struct RightNearest;

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
            serde_json::to_string(&Nearest).unwrap(),
            format!("\"{}\"", stringify!(Nearest))
        );
        assert_eq!(
            serde_json::to_string(&LeftNearest).unwrap(),
            format!("\"{}\"", stringify!(LeftNearest))
        );
        assert_eq!(
            serde_json::to_string(&RightNearest).unwrap(),
            format!("\"{}\"", stringify!(RightNearest))
        );
    }
}
