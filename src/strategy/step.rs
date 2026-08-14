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
/// Use [`Step::lower`]/[`Step::upper`] to broadcast the same direction across all
/// dimensions, or [`Step::new`] to supply one direction per dimension for mixed behavior.
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
///     strategy::Step::lower(),
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
///     strategy::Step::upper(),
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
///     strategy::Step::new(vec![
///         strategy::step::StepDirection::Lower,
///         strategy::step::StepDirection::Upper,
///     ]),
///     Extrapolate::Error,
/// )
/// .unwrap();
/// assert_eq!(interp.interpolate(&[0.7, 1.4]).unwrap(), 2.); // floor x→0, ceil y→2
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct Step {
    /// Broadcast to every dimension, or one per dimension.
    // Serializes under the key "Step" rather than "directions", making the strategy type
    // explicit in the output, consistent with how Linear, Nearest, etc. serialize to
    // their type name.
    #[cfg_attr(feature = "serde", serde(rename = "Step"))]
    pub directions: Broadcastable<StepDirection>,
}

impl From<StepDirection> for Step {
    /// Broadcasts `dir` to all dimensions.
    fn from(dir: StepDirection) -> Self {
        Step {
            directions: Broadcastable::Broadcast(dir),
        }
    }
}

impl Step {
    /// Create a `Step` with a distinct direction per grid dimension.
    ///
    /// Use [`lower`](Self::lower)/[`upper`](Self::upper) instead when every dimension
    /// shares the same direction.
    pub fn new(directions: Vec<StepDirection>) -> Self {
        Step {
            directions: Broadcastable::Each(directions),
        }
    }

    /// Broadcasts [`StepDirection::Lower`] to every dimension.
    pub fn lower() -> Self {
        StepDirection::Lower.into()
    }

    /// Broadcasts [`StepDirection::Upper`] to every dimension.
    pub fn upper() -> Self {
        StepDirection::Upper.into()
    }
}

#[cfg(feature = "serde")]
mod step_serde {
    use super::*;
    use serde::{Deserialize, Deserializer};

    /// `Step`'s own shape (`{"Step": <directions>}`), for `Deserialize` only: `Serialize`
    /// is derived directly on [`Step`], but reading also needs the legacy fallback below,
    /// which a derived `Deserialize` on `Step` itself can't express.
    #[derive(Deserialize)]
    struct StepShape {
        #[serde(rename = "Step")]
        directions: Broadcastable<StepDirection>,
    }

    /// Pre-`Step` serialized names for a fixed-direction strategy (from the removed
    /// `LeftNearest`/`RightNearest` unit structs), accepted on deserialize only. Bare
    /// strings, not wrapped like [`StepShape`], since that's how they were originally
    /// serialized.
    #[derive(Deserialize)]
    enum LegacyDirection {
        LeftNearest,
        RightNearest,
    }

    /// Tries the current `{"Step": ..}` form first (the common case, so it succeeds without
    /// a wasted attempt), falling back to the legacy bare-string form; the two are
    /// structurally distinct (object vs. bare string), so there's no ambiguity between them
    /// regardless of order.
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StepWire {
        Current(StepShape),
        Legacy(LegacyDirection),
    }

    impl<'de> Deserialize<'de> for Step {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            Ok(match StepWire::deserialize(deserializer)? {
                StepWire::Current(shape) => Step {
                    directions: shape.directions,
                },
                StepWire::Legacy(LegacyDirection::LeftNearest) => Step::lower(),
                StepWire::Legacy(LegacyDirection::RightNearest) => Step::upper(),
            })
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
            serde_json::to_string(&Step::lower()).unwrap(),
            r#"{"Step":"Lower"}"#
        );
        assert_eq!(
            serde_json::to_string(&Step::upper()).unwrap(),
            r#"{"Step":"Upper"}"#
        );
        assert_eq!(
            serde_json::to_string(&Step::new(vec![StepDirection::Lower, StepDirection::Upper]))
                .unwrap(),
            r#"{"Step":["Lower","Upper"]}"#
        );

        // Backward-compatibility aliases for pre-Step serialized names
        // (the removed `LeftNearest`/`RightNearest` unit structs).
        assert_eq!(
            serde_json::from_str::<Step>("\"LeftNearest\"").unwrap(),
            Step::lower()
        );
        assert_eq!(
            serde_json::from_str::<Step>("\"RightNearest\"").unwrap(),
            Step::upper()
        );
        // The aliases are bare strings, not wrapped like the current `{"Step": ..}` form.
        assert!(serde_json::from_str::<Step>(r#"{"Step":"LeftNearest"}"#).is_err());
    }
}
