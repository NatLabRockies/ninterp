//! Pre-defined interpolation strategies and traits for custom strategies

use super::*;

pub(crate) mod cubic;
pub mod enums;
pub mod traits;
pub mod utils;

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
    /// Checks the stored direction count against an interpolator's dimensionality: one
    /// direction broadcasts to every dimension, otherwise there must be exactly one per
    /// dimension. Shared by the `Strategy1D`/`2D`/`3D`/`ND` impls so all four report it
    /// identically.
    ///
    /// Stays a [`ValidateError::Other`] rather than earning a variant: it describes how
    /// this strategy was configured, not anything about the data.
    pub(crate) fn validate_len(&self, ndim: usize) -> Result<(), ValidateError> {
        let found = self.0.len();
        if found == 1 || found == ndim {
            return Ok(());
        }
        let expected = if ndim == 1 {
            "1".to_string()
        } else {
            format!("1 or {ndim}")
        };
        Err(ValidateError::Other(format!(
            "Step strategy has {found} directions but interpolator is {ndim}-D (expected {expected})"
        )))
    }

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

/// Cubic spline interpolation (<https://en.wikipedia.org/wiki/Spline_interpolation>).
///
/// Constructs a C² piecewise cubic polynomial through all data points.
/// The boundary condition is set by [`boundary_conditions`](CubicC2::boundary_conditions).
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
/// // f(x) = 2x + 1 (linear: reproduced exactly by any spline)
/// let interp: Interp1D<f64, _> = Interp1D::new(
///     array![0., 1., 2., 3.],
///     array![1., 3., 5., 7.],
///     strategy::CubicC2::not_a_knot(),
///     Extrapolate::Enable,
/// )
/// .unwrap();
/// assert_eq!(interp.interpolate(&[1.5]).unwrap(), 4.0);
/// assert_eq!(interp.interpolate(&[4.0]).unwrap(), 9.0); // extrapolation
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct CubicC2<T> {
    /// Boundary conditions, one per dimension or a single entry broadcast to all.
    pub boundary_conditions: Vec<CubicBoundaryConditions<T>>,
    /// Precomputed derivative data, populated by `Strategy1D`/`2D`/`3D`/`ND::init`. Its
    /// shape depends on which of those populated it:
    ///
    /// - `Strategy1D`/`StrategyND`: cached second derivatives (`M[i] = S''(x_i)`) for
    ///   every 1-D pencil along the innermost (last) grid axis, one
    ///   [`compute_m`](crate::strategy::cubic::compute_m) result per combination of the
    ///   other axes' indices. Same shape as the value grid, with the same last-axis
    ///   length as the innermost grid; for 1-D data (no outer axes) this degenerates to
    ///   a single pencil, i.e. shape `[n + 1]` for `n` intervals. Only the innermost
    ///   axis's coefficients are query-independent, so this is as far as caching goes;
    ///   outer axes still solve fresh per call.
    /// - `Strategy2D`/`Strategy3D`: the full corner-derivative tensor, shaped like the
    ///   value grid with one extra trailing axis of length `2^N` (`N` = 2 or 3): see
    ///   [`compute_corner_cache`](crate::strategy::cubic::compute_corner_cache). Every
    ///   query is then an O(1) lookup, no solving.
    ///
    /// Not included in the serialized form. Call [`Interpolator::validate`] after
    /// deserializing to recompute this before use.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) cache: ArrayD<T>,
}

/// Boundary conditions for [`CubicC2`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub enum CubicBoundaryConditions<T> {
    /// C³ continuity at the second and penultimate knots; no extra input required.
    /// Generally gives better accuracy than
    /// [`Natural`](CubicBoundaryConditions::Natural) for smooth functions.
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
    /// First and second derivatives match at both endpoints. By convention
    /// `values[n]` (the last point along this axis) should equal `values[0]`, since
    /// that's what makes the axis periodic, but it isn't read or enforced.
    Periodic,
}

/// Placeholder for [`CubicC2::cache`] before [`Strategy1D::init`]/`2D`/`3D`/`ND`
/// populates it. No `T: Clone + Zero` bound needed (unlike `ArrayD::zeros`), so the
/// constructors below stay unconstrained.
fn empty_cache<T>() -> ArrayD<T> {
    ArrayD::from_shape_vec(IxDyn(&[0]), Vec::new()).expect("empty shape matches empty vec")
}

impl<T> CubicC2<T> {
    /// Returns the boundary condition for the given dimension.
    /// A single-entry vec is broadcast to all dimensions.
    pub(crate) fn bc_for_dim(&self, dim: usize) -> &CubicBoundaryConditions<T> {
        let bcs = &self.boundary_conditions;
        &bcs[if bcs.len() == 1 { 0 } else { dim }]
    }

    /// Create a cubic spline with a distinct boundary condition per grid dimension.
    /// A single-entry vec is broadcast to all dimensions instead, matching
    /// [`bc_for_dim`](Self::bc_for_dim).
    ///
    /// Use [`not_a_knot`](Self::not_a_knot), [`natural`](Self::natural),
    /// [`clamped`](Self::clamped), or [`periodic`](Self::periodic) instead when every
    /// dimension shares the same condition.
    pub fn new(boundary_conditions: Vec<CubicBoundaryConditions<T>>) -> Self {
        Self {
            boundary_conditions,
            cache: empty_cache(),
        }
    }

    /// Create a cubic spline with not-a-knot boundary conditions.
    /// Requires at least 4 data points per dimension.
    pub fn not_a_knot() -> Self {
        Self {
            boundary_conditions: vec![CubicBoundaryConditions::NotAKnot],
            cache: empty_cache(),
        }
    }

    /// Create a cubic spline with natural (zero second derivative at endpoints) BCs.
    pub fn natural() -> Self {
        Self {
            boundary_conditions: vec![CubicBoundaryConditions::Natural],
            cache: empty_cache(),
        }
    }

    /// Create a cubic spline with specified first derivatives at both endpoints.
    pub fn clamped(left: T, right: T) -> Self {
        Self {
            boundary_conditions: vec![CubicBoundaryConditions::Clamped { left, right }],
            cache: empty_cache(),
        }
    }

    /// Create a cubic spline with periodic boundary conditions. By convention
    /// `values[n]` (the last point along each periodic axis) should equal
    /// `values[0]`, though this isn't enforced.
    pub fn periodic() -> Self {
        Self {
            boundary_conditions: vec![CubicBoundaryConditions::Periodic],
            cache: empty_cache(),
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
            serde_json::to_string(&StepLower).unwrap(),
            format!("\"{}\"", stringify!(StepLower))
        );
        assert_eq!(
            serde_json::to_string(&StepUpper).unwrap(),
            format!("\"{}\"", stringify!(StepUpper))
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
        assert!(serde_json::from_str::<Step>(r#"{"Step":["LeftNearest"]}"#).is_err());
    }
}
