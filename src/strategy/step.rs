//! Piecewise-constant (step) interpolation.

use super::*;

/// Direction used by [`Step`] interpolation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub enum StepDirection {
    /// Return the value at the nearest **lower** grid point (floor / previous value).
    Lower,
    /// Return the value at the nearest **upper** grid point (ceiling / next value).
    Upper,
}

/// Piecewise-constant (step) interpolation.
///
/// Returns the value at the nearest lower or upper grid point in each dimension.
///
/// Use [`Step`] when mixed per-dimension behavior is needed, or when direction is chosen at
/// runtime (for example from config). Use [`StepLower`] or [`StepUpper`] when a single direction
/// is known at compile time, especially in hot loops.
///
/// Construct [`Step`] from a single [`StepDirection`] to broadcast the same direction across
/// all dimensions, or supply one direction per dimension for mixed behavior.
///
/// # Examples
/// ```
/// use ndarray::prelude::*;
/// use ninterp::prelude::*;
///
/// // Returns the value at the nearest lower grid point
/// let interp = Interp1D::new(
///     array![0., 1., 2., 3., 4.],
///     array![0.2, 0.4, 0.6, 0.8, 1.0],
///     strategy::StepLower,
///     Extrapolate::Error,
/// )
/// .unwrap();
/// assert_eq!(interp.interpolate(&[3.75]).unwrap(), 0.8); // floor → value at 3.0
/// assert_eq!(interp.interpolate(&[4.00]).unwrap(), 1.0); // exact grid point
///
/// // Returns the value at the nearest upper grid point
/// let interp = Interp1D::new(
///     array![0., 1., 2., 3., 4.],
///     array![0.2, 0.4, 0.6, 0.8, 1.0],
///     strategy::StepUpper,
///     Extrapolate::Error,
/// )
/// .unwrap();
/// assert_eq!(interp.interpolate(&[3.25]).unwrap(), 1.0); // ceil → value at 4.0
/// assert_eq!(interp.interpolate(&[3.00]).unwrap(), 0.8); // exact grid point
///
/// // Behaves exactly like `StepUpper`, but allows the direction to be chosen at runtime
/// let interp = Interp1D::new(
///     array![0., 1., 2., 3., 4.],
///     array![0.2, 0.4, 0.6, 0.8, 1.0],
///     strategy::Step::from(strategy::step::StepDirection::Upper),
///     Extrapolate::Error,
/// )
/// .unwrap();
/// assert_eq!(interp.interpolate(&[3.25]).unwrap(), 1.0); // ceil → value at 4.0
/// assert_eq!(interp.interpolate(&[3.00]).unwrap(), 0.8); // exact grid point
///
/// // Per-dimension: floor in x, ceiling in y (2-D interpolator)
/// let interp = Interp2D::new(
///     array![0., 1., 2.],
///     array![0., 1., 2.],
///     array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
///     strategy::Step(strategy::per_axis::PerAxis::Axes(vec![
///         strategy::step::StepDirection::Lower,
///         strategy::step::StepDirection::Upper,
///     ])),
///     Extrapolate::Error,
/// )
/// .unwrap();
/// assert_eq!(interp.interpolate(&[0.7, 1.4]).unwrap(), 2.); // floor x→0, ceil y→2
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Step(pub PerAxis<StepDirection>);

impl From<StepDirection> for Step {
    /// Broadcasts `dir` to all dimensions.
    fn from(dir: StepDirection) -> Self {
        Step(PerAxis::Broadcast(dir))
    }
}

impl Step {
    /// Checks the stored direction count against an interpolator's dimensionality: one
    /// direction broadcasts to every dimension, otherwise there must be exactly one per
    /// dimension. Shared by the `Strategy1D`/`2D`/`3D`/`ND` impls so all four report it
    /// identically.
    pub(crate) fn validate_len(&self, ndim: usize) -> Result<(), ValidateError> {
        self.0.validate_len(ndim, "Step strategy", "directions")
    }

    /// Returns the direction for dimension `dim`.
    /// A broadcast direction applies to every dimension.
    pub(crate) fn dir(&self, dim: usize) -> StepDirection {
        *self.0.get(dim)
    }
}

/// Piecewise-constant interpolation that always selects the nearest **lower** grid point.
///
/// This is the zero-allocation, fixed-direction variant of [`Step`]. Prefer this when the
/// direction is known at compile time.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize_unit_struct))]
pub struct StepLower;

/// Piecewise-constant interpolation that always selects the nearest **upper** grid point.
///
/// This is the zero-allocation, fixed-direction variant of [`Step`]. Prefer this when the
/// direction is known at compile time.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize_unit_struct))]
pub struct StepUpper;

#[cfg(feature = "serde")]
mod step_serde {
    use super::*;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// Helper that serializes/deserializes `Step` as `{"Step": <directions>}`. This makes
    /// the strategy type explicit in the output, consistent with how `Linear`, `Nearest`,
    /// etc. serialize to their type name.
    #[derive(Serialize, Deserialize)]
    struct StepHelper {
        #[serde(rename = "Step")]
        directions: PerAxis<StepDirection>,
    }

    impl Serialize for Step {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            StepHelper {
                directions: self.0.clone(),
            }
            .serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for Step {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let helper = StepHelper::deserialize(deserializer)?;
            Ok(Step(helper.directions))
        }
    }
}

#[cfg(feature = "serde")]
mod step_marker_serde {
    use super::*;
    use serde::{Deserialize, Deserializer};

    #[derive(Deserialize)]
    enum StepLowerDe {
        #[serde(alias = "LeftNearest")]
        StepLower,
    }

    #[derive(Deserialize)]
    enum StepUpperDe {
        #[serde(alias = "RightNearest")]
        StepUpper,
    }

    impl<'de> Deserialize<'de> for StepLower {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let _ = StepLowerDe::deserialize(deserializer)?;
            Ok(StepLower)
        }
    }

    impl<'de> Deserialize<'de> for StepUpper {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let _ = StepUpperDe::deserialize(deserializer)?;
            Ok(StepUpper)
        }
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    #[cfg(feature = "serde")]
    fn test_serde() {
        assert_eq!(
            serde_json::to_string(&StepLower).unwrap(),
            format!("\"{}\"", stringify!(StepLower))
        );
        assert_eq!(
            serde_json::to_string(&StepUpper).unwrap(),
            format!("\"{}\"", stringify!(StepUpper))
        );
        assert_eq!(
            serde_json::to_string(&Step::from(StepDirection::Lower)).unwrap(),
            r#"{"Step":"Lower"}"#
        );
        assert_eq!(
            serde_json::to_string(&Step::from(StepDirection::Upper)).unwrap(),
            r#"{"Step":"Upper"}"#
        );
        assert_eq!(
            serde_json::to_string(&Step(PerAxis::Axes(vec![
                StepDirection::Lower,
                StepDirection::Upper
            ])))
            .unwrap(),
            r#"{"Step":["Lower","Upper"]}"#
        );

        let step_lower: StepLower = serde_json::from_str("\"StepLower\"").unwrap();
        assert_eq!(step_lower, StepLower);
        let step_upper: StepUpper = serde_json::from_str("\"StepUpper\"").unwrap();
        assert_eq!(step_upper, StepUpper);
        // Backward-compatibility aliases for pre-Step serialized names.
        assert_eq!(
            serde_json::from_str::<StepLower>("\"LeftNearest\"").unwrap(),
            StepLower
        );
        assert_eq!(
            serde_json::from_str::<StepUpper>("\"RightNearest\"").unwrap(),
            StepUpper
        );
        // Aliases are intentionally scoped to StepLower/StepUpper, not StepDirection.
        assert!(serde_json::from_str::<Step>(r#"{"Step":"LeftNearest"}"#).is_err());
    }
}
