//! Pre-defined interpolation strategies and traits for custom strategies

use super::*;

pub mod enums;
pub(crate) mod spline;
pub mod traits;
pub(crate) mod utils;

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
/// use ndarray::prelude::*;
/// use ninterp::prelude::*;
///
/// // Floor (previous value): returns the value at the nearest lower grid point
/// let interp = Interp1D::new(
///     array![0., 1., 2., 3., 4.],
///     array![0.2, 0.4, 0.6, 0.8, 1.0],
///     strategy::Step::from(strategy::StepDirection::Lower),
///     Extrapolate::Error,
/// )
/// .unwrap();
/// assert_eq!(interp.interpolate(&[3.75]).unwrap(), 0.8); // floor → value at 3.0
/// assert_eq!(interp.interpolate(&[4.00]).unwrap(), 1.0); // exact grid point
///
/// // Ceiling (next value): returns the value at the nearest upper grid point
/// let interp = Interp1D::new(
///     array![0., 1., 2., 3., 4.],
///     array![0.2, 0.4, 0.6, 0.8, 1.0],
///     strategy::Step::from(strategy::StepDirection::Upper),
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
///     strategy::Step(vec![
///         strategy::StepDirection::Lower,
///         strategy::StepDirection::Upper,
///     ]),
///     Extrapolate::Error,
/// )
/// .unwrap();
/// assert_eq!(interp.interpolate(&[0.7, 1.4]).unwrap(), 2.); // floor x→0, ceil y→2
/// ```
#[derive(Debug, Clone, PartialEq)]
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

#[cfg(feature = "serde")]
mod step_serde {
    use super::*;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// Helper that serializes/deserializes `Step` as `{"Step": [...directions...]}`.
    /// This makes the strategy type explicit in the output, consistent with how
    /// `Linear`, `Nearest`, etc. serialize to their type name.
    #[derive(Serialize, Deserialize)]
    struct StepHelper {
        #[serde(rename = "Step")]
        directions: Vec<StepDirection>,
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

/// Cubic spline interpolation (<https://en.wikipedia.org/wiki/Spline_interpolation>).
///
/// Constructs a C² piecewise cubic polynomial through all data points.
/// The boundary condition is set by [`boundary_conditions`](CubicSpline::boundary_conditions).
/// Coefficients are precomputed in [`Strategy1D::init`], called automatically
/// by [`Interp1D::new`] and [`Interp1D::set_strategy`].
///
/// Supports [`Extrapolate::Enable`]: evaluation beyond the grid extends the
/// boundary cubic polynomials.
///
/// # Example
/// ```
/// use ndarray::prelude::*;
/// use ninterp::prelude::*;
///
/// // f(x) = 2x + 1 (linear — reproduced exactly by any spline)
/// let interp: Interp1DOwned<f64, _> = Interp1D::new(
///     array![0., 1., 2., 3.],
///     array![1., 3., 5., 7.],
///     strategy::CubicSpline::not_a_knot(),
///     Extrapolate::Enable,
/// )
/// .unwrap();
/// assert_eq!(interp.interpolate(&[1.5]).unwrap(), 4.0);
/// assert_eq!(interp.interpolate(&[4.0]).unwrap(), 9.0); // extrapolation
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct CubicSpline<T> {
    /// Boundary conditions, one per dimension or a single entry broadcast to all.
    pub boundary_conditions: Vec<CubicBC<T>>,
    /// Second derivatives `M[i] = S''(x_i)` at each grid point, length `n + 1`
    /// for `n` intervals. Populated by [`Strategy1D::init`]; boundary values
    /// are determined by [`boundary_conditions`](CubicBC).
    ///
    /// Not included in the serialized form. Call [`Interpolator::validate`] after
    /// deserializing a 1-D interpolator to recompute these coefficients before use.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) m: Vec<T>,
}

/// Boundary conditions for [`CubicSpline`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub enum CubicBC<T> {
    /// C³ continuity at the second and penultimate knots; no extra input required.
    /// Generally gives better accuracy than [`Natural`](CubicBC::Natural)
    /// for smooth functions.
    NotAKnot,
    /// Zero second derivative at both endpoints.
    Natural,
    /// Specified first derivative at both endpoints.
    Clamped {
        /// First derivative at the left (lower) endpoint.
        left: T,
        /// First derivative at the right (upper) endpoint.
        right: T,
    },
    /// First and second derivatives match at both endpoints.
    /// Requires `values[0] == values[n]`.
    Periodic,
}

impl<T> CubicSpline<T> {
    /// Returns the boundary condition for the given dimension.
    /// A single-entry vec is broadcast to all dimensions.
    pub(crate) fn bc_for_dim(&self, dim: usize) -> &CubicBC<T> {
        let bcs = &self.boundary_conditions;
        &bcs[if bcs.len() == 1 { 0 } else { dim }]
    }

    /// Create a cubic spline with not-a-knot boundary conditions.
    /// Requires at least 4 data points per dimension.
    pub fn not_a_knot() -> Self {
        Self {
            boundary_conditions: vec![CubicBC::NotAKnot],
            m: Vec::new(),
        }
    }

    /// Create a cubic spline with natural (zero second derivative at endpoints) BCs.
    pub fn natural() -> Self {
        Self {
            boundary_conditions: vec![CubicBC::Natural],
            m: Vec::new(),
        }
    }

    /// Create a cubic spline with specified first derivatives at both endpoints.
    pub fn clamped(left: T, right: T) -> Self {
        Self {
            boundary_conditions: vec![CubicBC::Clamped { left, right }],
            m: Vec::new(),
        }
    }

    /// Create a cubic spline with periodic boundary conditions.
    /// Requires `values[0] == values[n]`.
    pub fn periodic() -> Self {
        Self {
            boundary_conditions: vec![CubicBC::Periodic],
            m: Vec::new(),
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
            serde_json::to_string(&LinearUniform).unwrap(),
            format!("\"{}\"", stringify!(LinearUniform))
        );
        assert_eq!(
            serde_json::to_string(&Nearest).unwrap(),
            format!("\"{}\"", stringify!(Nearest))
        );
        assert_eq!(
            serde_json::to_string(&Step::from(StepDirection::Lower)).unwrap(),
            r#"{"Step":["Lower"]}"#
        );
        assert_eq!(
            serde_json::to_string(&Step::from(StepDirection::Upper)).unwrap(),
            r#"{"Step":["Upper"]}"#
        );
        assert_eq!(
            serde_json::to_string(&Step(vec![StepDirection::Lower, StepDirection::Upper])).unwrap(),
            r#"{"Step":["Lower","Upper"]}"#
        );
    }
}
