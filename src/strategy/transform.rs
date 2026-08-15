//! Coordinate/value transform wrappers: see [`Transform`], [`GridTransform`],
//! [`ValuesTransform`].

use super::*;

/// A monotonic 1-D transform applied to grid coordinates ([`GridTransform`]) or data
/// values ([`ValuesTransform`]) before interpolating: standard for data spanning many
/// orders of magnitude, or following a known nonlinear relationship (power-law,
/// Arrhenius, diffusion-type).
///
/// | Variant | `forward(x)` | `inverse(x)` | Domain | Monotonicity |
/// |---|---|---|---|---|
/// | [`Identity`](Transform::Identity) | `x` | `x` | none | increasing |
/// | [`Log`](Transform::Log) | `ln(x)` | `exp(x)` | `x > 0` | increasing |
/// | [`Sqrt`](Transform::Sqrt) | `sqrt(x)` | `x^2` | `x >= 0` | increasing |
/// | [`Reciprocal`](Transform::Reciprocal) | `1/x` | `1/x` | `x != 0` | **decreasing** |
///
/// [`GridTransform`] relies on this: a raw grid is always strictly increasing, so
/// wrapping it in a *decreasing* transform (only [`Reciprocal`](Transform::Reciprocal)
/// today) would otherwise leave the transformed grid decreasing. `GridTransform`
/// detects that case and reverses the transformed grid (and the matching `values`
/// axis, kept in lockstep) back to ascending internally, so every downstream strategy
/// still sees an ascending grid, exactly like every other `Transform`. See
/// [`GridTransform`]'s docs for the one user-visible consequence of this: which
/// physical endpoint a `clamped`/`second_derivative` boundary condition applies to.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[non_exhaustive]
pub enum Transform {
    /// No transform: `forward`/`inverse` are both identity.
    Identity,
    /// Natural log / exponential. Requires `x > 0`.
    Log,
    /// Square root / square. Requires `x >= 0`.
    Sqrt,
    /// Reciprocal, its own inverse. Requires `x != 0`. Monotonically *decreasing*,
    /// unlike every other variant: see [`Transform`]'s own docs for what that means
    /// for [`GridTransform`].
    Reciprocal,
}

impl Transform {
    /// Applies this transform.
    ///
    /// Does not itself check [`Transform::in_domain`]; callers ([`GridTransform`],
    /// [`ValuesTransform`]) check first and return a
    /// [`ValidateError::GridTransformDomain`]/[`ValidateError::ValuesTransformDomain`]/
    /// [`InterpolateError::GridTransformDomain`] instead of calling this out of domain.
    pub fn forward<T: Float>(self, x: T) -> T {
        match self {
            Transform::Identity => x,
            Transform::Log => x.ln(),
            Transform::Sqrt => x.sqrt(),
            Transform::Reciprocal => x.recip(),
        }
    }

    /// Inverts [`Transform::forward`].
    pub fn inverse<T: Float>(self, x: T) -> T {
        match self {
            Transform::Identity => x,
            Transform::Log => x.exp(),
            Transform::Sqrt => x * x,
            Transform::Reciprocal => x.recip(),
        }
    }

    /// Whether `x` is valid input to [`Transform::forward`].
    pub fn in_domain<T: Float>(self, x: T) -> bool {
        match self {
            Transform::Identity => true,
            Transform::Log => x > T::zero(),
            Transform::Sqrt => x >= T::zero(),
            Transform::Reciprocal => x != T::zero(),
        }
    }

    /// Whether `forward` is increasing (vs. decreasing) on its domain.
    ///
    /// `GridTransform` uses this to keep its transformed grid ascending: a raw grid is
    /// always strictly increasing (enforced by [`InterpDataBase::validate`]
    /// (crate::interpolator::data::InterpDataBase::validate)), but `Reciprocal`
    /// (`1/x`) is decreasing on both branches of its domain, so `forward`-ing an
    /// increasing grid through it produces a decreasing one. Left uncorrected, every
    /// downstream ascending-grid assumption (binary search, `Extrapolate::Wrap`'s
    /// `min < max` precondition) would silently break instead of erroring.
    /// `GridTransform` reverses the transformed axis (and the matching `values` axis)
    /// whenever this is `false`.
    pub(crate) fn is_increasing(self) -> bool {
        match self {
            Transform::Identity | Transform::Log | Transform::Sqrt => true,
            Transform::Reciprocal => false,
        }
    }
}

/// Placeholder for [`GridTransform::grid_cache`]/[`ValuesTransform::values_cache`]
/// before `init` populates them. Mirrors [`crate::strategy::cubic`]'s own
/// `empty_cache`.
fn empty_cache<T>() -> ArrayD<T> {
    ArrayD::from_shape_vec(IxDyn(&[0]), Vec::new()).expect("empty shape matches empty vec")
}

/// Wraps a [`Strategy1D`]/[`Strategy2D`]/[`Strategy3D`]/[`StrategyND`] `inner`
/// strategy, interpolating in a transformed grid coordinate space (see [`Transform`])
/// instead of the raw one.
///
/// `inner`'s own working space is the transformed grid: e.g. wrapping
/// [`CubicC2`]'s `clamped`/`second_derivative` boundary
/// conditions supplies derivatives *with respect to the transformed coordinate*, not
/// the raw one. `not_a_knot`/`periodic` are unaffected: those are structural
/// conditions on the spline's own space, not physical-value targets.
///
/// Under a monotonically *decreasing* transform (only [`Transform::Reciprocal`]
/// today), the transformed grid comes out backwards, since the raw grid is always
/// increasing; `GridTransform` reverses it (and `values` in lockstep) back to
/// ascending internally, so `inner` always sees an ascending grid like normal. This
/// swaps which physical endpoint sits at index 0 vs. the last index: if `inner` is
/// [`CubicC2`] with `clamped`/`second_derivative` boundary conditions, its `lower`
/// endpoint ends up applying to what was the raw grid's *highest* coordinate, and
/// vice versa, under a decreasing transform.
///
/// # Example
/// ```
/// use ndarray::prelude::*;
/// use ninterp::prelude::*;
/// use ninterp::strategy::{GridTransform, Linear};
///
/// // f(x) = 2*ln(x) + 1: linear in log-x, so exact once the grid is log-transformed
/// let x = array![
///     1.,
///     std::f64::consts::E,
///     std::f64::consts::E.powi(2),
///     std::f64::consts::E.powi(3),
/// ];
/// let y = x.mapv(|v: f64| 2. * v.ln() + 1.);
/// let interp = Interp1D::new(x, y, GridTransform::log(Linear), Extrapolate::Error).unwrap();
/// let query = std::f64::consts::E.powf(1.5);
/// assert!((interp.interpolate(&[query]).unwrap() - 4.0).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "T: Serialize, S: Serialize",
        deserialize = "T: Deserialize<'de>, S: Deserialize<'de>"
    ))
)]
pub struct GridTransform<T, S> {
    /// Transform applied to each grid axis, one per dimension or a single entry
    /// broadcast to all.
    pub axes: Broadcastable<Transform>,
    /// Wrapped strategy, evaluated in the transformed coordinate space.
    pub inner: S,
    /// Transformed grid, one axis per dimension. Not included in the serialized
    /// form; call the interpolator's `init_strategy` after deserializing to
    /// recompute it.
    #[cfg_attr(feature = "serde", serde(skip, default))]
    pub(crate) grid_cache: Vec<Array1<T>>,
}

impl<T, S> GridTransform<T, S> {
    /// Distinct transform per grid dimension. `axes.len()` must equal the
    /// interpolator's dimensionality; checked by [`Broadcastable::validate_len`] in
    /// `validate`.
    pub fn new(axes: Vec<Transform>, inner: S) -> Self {
        Self {
            axes: Broadcastable::Each(axes),
            inner,
            grid_cache: Vec::new(),
        }
    }

    /// `transform` broadcast to every axis. Prefer [`log`](Self::log)/
    /// [`sqrt`](Self::sqrt)/[`reciprocal`](Self::reciprocal) when the desired
    /// transform is known at the call site; this is for a [`Transform`] obtained
    /// generically (e.g. from runtime config), without matching on it first.
    pub fn broadcast(transform: Transform, inner: S) -> Self {
        Self {
            axes: Broadcastable::Broadcast(transform),
            inner,
            grid_cache: Vec::new(),
        }
    }

    /// [`Transform::Log`] broadcast to every axis.
    pub fn log(inner: S) -> Self {
        Self::broadcast(Transform::Log, inner)
    }

    /// [`Transform::Sqrt`] broadcast to every axis.
    pub fn sqrt(inner: S) -> Self {
        Self::broadcast(Transform::Sqrt, inner)
    }

    /// [`Transform::Reciprocal`] broadcast to every axis.
    pub fn reciprocal(inner: S) -> Self {
        Self::broadcast(Transform::Reciprocal, inner)
    }

    /// A view of `values` with every decreasing-transform axis reversed, matching
    /// `grid_cache`'s ascending order. Shared by `validate`/`init`/`interpolate`/
    /// `interpolate_wrapped` across all four dimensionalities.
    pub(crate) fn transformed_values_view<'v, A, Dim: Dimension>(
        &self,
        mut values: ArrayView<'v, A, Dim>,
    ) -> ArrayView<'v, A, Dim> {
        values.slice_each_axis_inplace(|ax| {
            if self.axes[ax.axis.index()].is_increasing() {
                Slice::new(0, None, 1)
            } else {
                Slice::new(0, None, -1)
            }
        });
        values
    }
}

impl<T: Float, S> GridTransform<T, S> {
    /// Domain-checks and forward-transforms one grid axis (single pass, fused),
    /// reversing it to stay ascending if `dim`'s configured transform is
    /// decreasing. Shared by `validate` (result discarded) and `init` (result
    /// cached into `grid_cache`) across all four dimensionalities.
    pub(crate) fn transform_axis(
        &self,
        dim: usize,
        grid: ArrayView1<T>,
    ) -> Result<Array1<T>, ValidateError> {
        let transform = self.axes[dim];
        let mut transformed = Vec::with_capacity(grid.len());
        for (index, &x) in grid.iter().enumerate() {
            if !transform.in_domain(x) {
                return Err(ValidateError::GridTransformDomain {
                    transform,
                    dim,
                    index,
                });
            }
            transformed.push(transform.forward(x));
        }
        let mut transformed = Array1::from_vec(transformed);
        if !transform.is_increasing() {
            transformed.invert_axis(Axis(0));
        }
        Ok(transformed)
    }

    /// Domain-checks `point` against each axis's configured transform.
    pub(crate) fn check_point_domain(&self, point: &[T]) -> Result<(), InterpolateError> {
        for (dim, &x) in point.iter().enumerate() {
            let transform = self.axes[dim];
            if !transform.in_domain(x) {
                return Err(InterpolateError::GridTransformDomain { transform, dim });
            }
        }
        Ok(())
    }

    /// Forward-transforms `x` on axis `dim` and wraps it against `grid_cache[dim]`'s
    /// (ascending) bounds. Assumes [`check_point_domain`](Self::check_point_domain)
    /// already passed and `grid_cache` is populated.
    pub(crate) fn wrap_axis(&self, dim: usize, x: T) -> T
    where
        T: Num + Euclid,
    {
        let transformed = self.axes[dim].forward(x);
        let lo = *self.grid_cache[dim].first().unwrap();
        let hi = *self.grid_cache[dim].last().unwrap();
        wrap(transformed, lo, hi)
    }
}

/// Wraps a [`Strategy1D`]/[`Strategy2D`]/[`Strategy3D`]/[`StrategyND`] `inner`
/// strategy, interpolating in a transformed *value* space (see [`Transform`])
/// instead of the raw one: e.g. keeping interpolated output within a range a raw
/// polynomial spline doesn't guarantee (always-positive, via [`log`](Self::log)).
///
/// Composes with [`GridTransform`] by nesting, for full log-log interpolation:
/// `ValuesTransform::log(GridTransform::log(CubicC2::not_a_knot()))`.
///
/// # Example
/// ```
/// use ndarray::prelude::*;
/// use ninterp::prelude::*;
/// use ninterp::strategy::{Linear, ValuesTransform};
///
/// // f(x) = exp(2x): ln(f(x)) = 2x is linear, so exact under log-values Linear
/// let x = array![0., 1., 2., 3.];
/// let y = x.mapv(|v: f64| (2. * v).exp());
/// let interp = Interp1D::new(x, y, ValuesTransform::log(Linear), Extrapolate::Error).unwrap();
/// assert!((interp.interpolate(&[1.5]).unwrap() - 3f64.exp()).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "T: Serialize, S: Serialize",
        deserialize = "T: Deserialize<'de>, S: Deserialize<'de>"
    ))
)]
pub struct ValuesTransform<T, S> {
    /// Transform applied to `values`.
    pub transform: Transform,
    /// Wrapped strategy, evaluated against transformed values.
    pub inner: S,
    /// Transformed values. Not included in the serialized form; call the
    /// interpolator's `init_strategy` after deserializing to recompute it.
    #[cfg_attr(feature = "serde", serde(skip, default = "empty_cache"))]
    pub(crate) values_cache: ArrayD<T>,
}

impl<T, S> ValuesTransform<T, S> {
    /// Wraps `inner`, applying `transform` to data values.
    pub fn new(transform: Transform, inner: S) -> Self {
        Self {
            transform,
            inner,
            values_cache: empty_cache(),
        }
    }

    /// [`Transform::Log`].
    pub fn log(inner: S) -> Self {
        Self::new(Transform::Log, inner)
    }

    /// [`Transform::Sqrt`].
    pub fn sqrt(inner: S) -> Self {
        Self::new(Transform::Sqrt, inner)
    }

    /// [`Transform::Reciprocal`].
    pub fn reciprocal(inner: S) -> Self {
        Self::new(Transform::Reciprocal, inner)
    }
}

impl<T: Float, S> ValuesTransform<T, S> {
    /// Domain-checks and forward-transforms `values` (single pass, fused). Shared
    /// by `validate` (result discarded) and `init` (result cached into
    /// `values_cache`) across all four dimensionalities.
    pub(crate) fn transform_values<Dim: Dimension>(
        &self,
        values: ArrayView<T, Dim>,
    ) -> Result<Array<T, Dim>, ValidateError> {
        let mut transformed = Vec::with_capacity(values.len());
        for (pattern, &v) in indices_of(&values).into_iter().zip(values.iter()) {
            if !self.transform.in_domain(v) {
                let index: Dim = pattern.into_dimension();
                return Err(ValidateError::ValuesTransformDomain {
                    transform: self.transform,
                    index: index.slice().to_vec(),
                });
            }
            transformed.push(self.transform.forward(v));
        }
        Ok(Array::from_shape_vec(values.raw_dim(), transformed)
            .expect("transformed shape matches values shape"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_inverse_round_trip() {
        for transform in [
            Transform::Identity,
            Transform::Log,
            Transform::Sqrt,
            Transform::Reciprocal,
        ] {
            for x in [0.5_f64, 1., 2., 10., 123.456] {
                let round_tripped = transform.inverse(transform.forward(x));
                assert!(
                    (round_tripped - x).abs() < 1e-9,
                    "{transform:?}: forward/inverse round trip failed for x={x}"
                );
            }
        }
    }

    #[test]
    fn test_in_domain() {
        assert!(Transform::Identity.in_domain(-1.));
        assert!(Transform::Identity.in_domain(0.));

        assert!(!Transform::Log.in_domain(0.));
        assert!(!Transform::Log.in_domain(-1.));
        assert!(Transform::Log.in_domain(0.1));

        assert!(Transform::Sqrt.in_domain(0.));
        assert!(!Transform::Sqrt.in_domain(-0.1));

        assert!(!Transform::Reciprocal.in_domain(0.));
        assert!(Transform::Reciprocal.in_domain(-1.));
        assert!(Transform::Reciprocal.in_domain(1.));
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_serde() {
        assert_eq!(
            serde_json::to_string(&Transform::Log).unwrap(),
            "\"Log\"".to_string()
        );
        assert_eq!(
            serde_json::from_str::<Transform>("\"Sqrt\"").unwrap(),
            Transform::Sqrt
        );
    }
}
