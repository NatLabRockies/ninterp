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
/// Construct from a single [`StepDirection`] to broadcast the same direction across all
/// dimensions, or supply one direction per dimension for mixed behavior.
///
/// # Examples
/// ```
/// use ninterp::prelude::*;
/// use ninterp::strategy::{Step, StepDirection};
///
/// // All dimensions floor (previous value)
/// let floor = Step::from(StepDirection::Lower);
///
/// // Per-dimension: floor in x, ceiling in y (for a 2-D interpolator)
/// let mixed = Step(vec![StepDirection::Lower, StepDirection::Upper]);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Step(pub Vec<StepDirection>);

impl From<StepDirection> for Step {
    /// Broadcasts `dir` to all dimensions.
    fn from(dir: StepDirection) -> Self {
        Step(vec![dir])
    }
}

impl Step {
    /// Returns the direction for dimension `dim`.
    /// A single stored direction broadcasts to all dimensions.
    pub(crate) fn dir(&self, dim: usize) -> StepDirection {
        if self.0.len() == 1 {
            self.0[0]
        } else {
            self.0[dim]
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
