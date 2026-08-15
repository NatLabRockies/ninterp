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
    ///
    /// `NaN` is always in-domain: `forward(NaN)` is a clean `NaN` for every variant
    /// (no panic), so there's no mathematical reason to exclude it here. Callers
    /// that need to reject NaN for a different reason (e.g. a query point reaching
    /// the inner strategy's grid search, which panics on NaN comparisons) check
    /// `x.is_nan()` themselves on top of this.
    pub fn in_domain<T: Float>(self, x: T) -> bool {
        x.is_nan()
            || match self {
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
    /// broadcast to all. Serializes as `grid_transform`.
    #[cfg_attr(feature = "serde", serde(rename = "grid_transform"))]
    pub transforms: Broadcastable<Transform>,
    /// Wrapped strategy, evaluated in the transformed coordinate space.
    pub inner: S,
    /// Transformed grid, one axis per dimension. Not included in the serialized
    /// form; call the interpolator's `init_strategy` after deserializing to
    /// recompute it.
    #[cfg_attr(feature = "serde", serde(skip, default))]
    pub(crate) grid_cache: Vec<Array1<T>>,
}

impl<T, S> GridTransform<T, S> {
    /// Distinct transform per grid dimension. `transforms.len()` must equal the
    /// interpolator's dimensionality; checked by [`Broadcastable::validate_len`] in
    /// `validate`.
    pub fn new(transforms: Vec<Transform>, inner: S) -> Self {
        Self {
            transforms: Broadcastable::Each(transforms),
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
            transforms: Broadcastable::Broadcast(transform),
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
            if self.transforms[ax.axis.index()].is_increasing() {
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
    ///
    /// Also checks that `forward` is monotonic across the *whole* axis, not just
    /// per-element domain membership: `transform`'s domain can be disconnected (e.g.
    /// [`Transform::Reciprocal`]'s `x != 0`), in which case every coordinate can
    /// individually be in-domain while the transformed sequence as a whole still
    /// isn't monotonic, and no single reversal can fix that up to ascending.
    pub(crate) fn transform_axis(
        &self,
        dim: usize,
        grid: ArrayView1<T>,
    ) -> Result<Array1<T>, ValidateError> {
        let transform = self.transforms[dim];
        let mut transformed: Vec<T> = Vec::with_capacity(grid.len());
        for (index, &x) in grid.iter().enumerate() {
            // Domain check.
            if !transform.in_domain(x) {
                return Err(ValidateError::GridTransformDomain {
                    transform,
                    dim,
                    index,
                });
            }
            // Forward-transform.
            let fx = transform.forward(x);
            // Monotonicity check, against the previous transformed coordinate.
            if let Some(&prev) = transformed.last() {
                let ordered = if transform.is_increasing() {
                    fx > prev
                } else {
                    fx < prev
                };
                if !ordered {
                    return Err(ValidateError::GridTransformNotMonotonic {
                        transform,
                        dim,
                        index,
                    });
                }
            }
            transformed.push(fx);
        }
        // Reversal, to stay ascending under a decreasing transform.
        let mut transformed = Array1::from_vec(transformed);
        if !transform.is_increasing() {
            transformed.invert_axis(Axis(0));
        }
        Ok(transformed)
    }

    /// Domain-checks every axis of `point` (batch position `index`), collecting
    /// every violation instead of stopping at the first. Shared by
    /// [`check_point_domain`](Self::check_point_domain) (`index` fixed at 0) and
    /// [`check_batch_domain`](Self::check_batch_domain) across all four
    /// dimensionalities.
    ///
    /// Rejects NaN explicitly here, overriding `transform.in_domain` (which treats
    /// NaN as in-domain). A NaN *query point* reaching the inner strategy's grid
    /// search breaks its comparisons and panics, so it can't be let through here the
    /// way `in_domain` alone would allow. A NaN *grid* coordinate can't reach this
    /// far (the raw grid's own strictly-increasing check rejects it first), and a
    /// NaN *data value* is a different, safe case entirely (see
    /// [`ValuesTransform::transform_values`]), so this override is deliberately
    /// scoped to query points only.
    pub(crate) fn point_domain_failures(&self, index: usize, point: &[T]) -> Vec<OutsideDomainAt> {
        point
            .iter()
            .enumerate()
            .filter_map(|(dim, &x)| {
                let transform = self.transforms[dim];
                (x.is_nan() || !transform.in_domain(x)).then_some(OutsideDomainAt {
                    index,
                    dim,
                    transform,
                })
            })
            .collect()
    }

    /// Domain-checks `point` against each axis's configured transform, aggregating
    /// every violating dimension (not just the first) into one
    /// [`InterpolateError::GridTransformDomain`], mirroring how `Extrapolate::Error`
    /// aggregates `OutOfBoundsAt` failures across every dimension of a point.
    pub(crate) fn check_point_domain(&self, point: &[T]) -> Result<(), InterpolateError> {
        let failures = self.point_domain_failures(0, point);
        if failures.is_empty() {
            Ok(())
        } else {
            Err(InterpolateError::GridTransformDomain(failures))
        }
    }

    /// Domain-checks every point in a batch, aggregating every violation across the
    /// *whole batch* (not just the first point) into one
    /// [`InterpolateError::GridTransformDomain`], mirroring how `Extrapolate::Error`
    /// aggregates `OutOfBoundsAt` failures across a whole batch instead of erroring
    /// on the first out-of-bounds point. Shared by `batch_interpolate_into` across
    /// all four dimensionalities.
    pub(crate) fn check_batch_domain<'p>(
        &self,
        points: impl Iterator<Item = &'p [T]>,
    ) -> Result<(), InterpolateError>
    where
        T: 'p,
    {
        let failures: Vec<OutsideDomainAt> = points
            .enumerate()
            .flat_map(|(index, point)| self.point_domain_failures(index, point))
            .collect();
        if failures.is_empty() {
            Ok(())
        } else {
            Err(InterpolateError::GridTransformDomain(failures))
        }
    }
}

/// Generates `impl $Strategy<D> for GridTransform<D::Elem, S>`, for 1D/2D/3D
/// (`$N` in `[D::Elem; $N]`); `StrategyND` stays hand-written in `n/strategies.rs`,
/// since its point type is a runtime-length slice rather than a fixed-size array and
/// so can't share this shape, mirroring why `StrategyND` itself isn't generated by
/// [`crate::strategy::traits::fixed_strategy_trait`].
macro_rules! grid_transform_strategy_impl {
    ($Strategy:ident, $InterpDataBase:ident, $InterpDataView:ident, $N:literal, $validate_doc:literal) => {
        impl<D, S> $Strategy<D> for GridTransform<D::Elem, S>
        where
            D: Data + RawDataClone + Clone,
            D::Elem: Float + Debug,
            S: for<'a> $Strategy<ViewRepr<&'a D::Elem>> + Clone + Debug,
        {
            #[doc = $validate_doc]
            fn validate(&self, data: &$InterpDataBase<D>) -> Result<(), ValidateError> {
                self.transforms
                    .validate_len($N, "GridTransform", "transforms")?;
                let transformed_grid: Vec<Array1<D::Elem>> = data
                    .grid
                    .iter()
                    .enumerate()
                    .map(|(dim, grid)| self.transform_axis(dim, grid.view()))
                    .collect::<Result<_, _>>()?;
                let values = self.transformed_values_view(data.values.view());
                let view = $InterpDataView {
                    grid: core::array::from_fn(|i| transformed_grid[i].view()),
                    values,
                };
                self.inner.validate(&view)
            }

            /// Transforms the grid into `grid_cache`, then initializes `inner` against
            /// a transient view zipping `grid_cache` with `data.values`.
            ///
            /// A raw grid is always strictly increasing, but a decreasing transform
            /// (e.g. `Reciprocal`) would otherwise leave that axis of `grid_cache`
            /// decreasing; `Transform::is_increasing` flags that case so the
            /// transformed axis (and the matching `values` axis) can be reversed back
            /// to ascending, matching every downstream strategy's ascending-grid
            /// assumption.
            fn init(&mut self, data: &$InterpDataBase<D>) -> Result<(), ValidateError> {
                self.grid_cache = data
                    .grid
                    .iter()
                    .enumerate()
                    .map(|(dim, grid)| self.transform_axis(dim, grid.view()))
                    .collect::<Result<_, _>>()?;
                let values = self.transformed_values_view(data.values.view());
                let view = $InterpDataView {
                    grid: core::array::from_fn(|i| self.grid_cache[i].view()),
                    values,
                };
                self.inner.init(&view)
            }

            fn interpolate(
                &self,
                data: &$InterpDataBase<D>,
                point: &[D::Elem; $N],
            ) -> Result<D::Elem, InterpolateError> {
                self.check_point_domain(point)?;
                let transformed_point: [D::Elem; $N] =
                    core::array::from_fn(|i| self.transforms[i].forward(point[i]));
                let values = self.transformed_values_view(data.values.view());
                let view = $InterpDataView {
                    grid: core::array::from_fn(|i| self.grid_cache[i].view()),
                    values,
                };
                self.inner.interpolate(&view, &transformed_point)
            }

            #[doc = concat!(
                " Forward-transforms into `grid_cache`'s coordinate space, then delegates\n",
                " the actual wrap to `inner.interpolate_wrapped` rather than wrapping here\n",
                " itself: wrapping doesn't commute with a nonlinear transform, so it must\n",
                " happen in the *final* (innermost) transformed space where the periodic\n",
                " strategy actually lives, mirroring how `ValuesTransform::interpolate_wrapped`\n",
                " defers to `inner` for the same reason (composing two `GridTransform`s and\n",
                " wrapping at the outer layer's space, then forward-transforming again, uses\n",
                " the wrong period). For a non-transform `inner`, `", stringify!($Strategy),
                "::interpolate_wrapped`'s default wraps directly against `grid_cache`,\n",
                " reproducing this layer's own wrap exactly.",
            )]
            fn interpolate_wrapped(
                &self,
                data: &$InterpDataBase<D>,
                point: &[D::Elem; $N],
            ) -> Result<D::Elem, InterpolateError>
            where
                D::Elem: Num + Euclid + Copy,
            {
                self.check_point_domain(point)?;
                let transformed_point: [D::Elem; $N] =
                    core::array::from_fn(|i| self.transforms[i].forward(point[i]));
                let values = self.transformed_values_view(data.values.view());
                let view = $InterpDataView {
                    grid: core::array::from_fn(|i| self.grid_cache[i].view()),
                    values,
                };
                self.inner.interpolate_wrapped(&view, &transformed_point)
            }

            /// Skips the domain check `interpolate` does, and calls
            /// `inner.interpolate_fast` rather than `inner.interpolate`, so "fast"
            /// propagates through nested `GridTransform`/`ValuesTransform` layers
            /// instead of stopping at the outermost one.
            ///
            /// An out-of-domain point is the caller's problem here, same as any other
            /// unchecked `_fast` method: `forward`-ing it produces `NaN` (`Log`/`Sqrt`)
            /// or `+-inf` (`Reciprocal`, only at `x == 0`). `NaN` poisons the grid
            /// search's comparisons and panics; `+-inf` compares normally, so it's
            /// treated like an ordinary out-of-bounds extrapolation query and silently
            /// produces `NaN` output instead. Expected, not a bug: check
            /// [`Transform::in_domain`] yourself first if you need a guarantee either
            /// way.
            fn interpolate_fast(
                &self,
                data: &$InterpDataBase<D>,
                point: &[D::Elem; $N],
            ) -> D::Elem {
                let transformed_point: [D::Elem; $N] =
                    core::array::from_fn(|i| self.transforms[i].forward(point[i]));
                let values = self.transformed_values_view(data.values.view());
                let view = $InterpDataView {
                    grid: core::array::from_fn(|i| self.grid_cache[i].view()),
                    values,
                };
                self.inner.interpolate_fast(&view, &transformed_point)
            }

            /// Domain-checks every point in the batch before transforming, aggregating
            /// every violation across the *whole batch* into one
            /// [`InterpolateError::GridTransformDomain`] instead of erroring on the
            /// first one, mirroring how `Extrapolate::Error` aggregates out-of-bounds
            /// points.
            fn batch_interpolate_into(
                &self,
                data: &$InterpDataBase<D>,
                points: &[[D::Elem; $N]],
                out: &mut [D::Elem],
            ) -> Result<(), InterpolateError> {
                if out.len() != points.len() {
                    return Err(InterpolateError::OutputLength {
                        expected: points.len(),
                        found: out.len(),
                    });
                }
                self.check_batch_domain(points.iter().map(|p| p.as_slice()))?;
                let transformed_points: Vec<[D::Elem; $N]> = points
                    .iter()
                    .map(|point| core::array::from_fn(|i| self.transforms[i].forward(point[i])))
                    .collect();
                let values = self.transformed_values_view(data.values.view());
                let view = $InterpDataView {
                    grid: core::array::from_fn(|i| self.grid_cache[i].view()),
                    values,
                };
                self.inner
                    .batch_interpolate_into(&view, &transformed_points, out)
            }

            /// Checks this layer's own domain per point (not bailing on the first
            /// violation), then recurses into `inner` with only the *outer-valid*
            /// points' forward transforms, remapping `inner`'s failure indices back to
            /// their true batch position: an outer-invalid point can't be
            /// forward-transformed meaningfully, but its presence must not hide
            /// `inner`'s own violations for the *other* points, so a `GridTransform`
            /// nested inside another one still gets every one of its violations
            /// aggregated, not just the outer layer's. Exposed here so generic callers
            /// (e.g. `Extrapolate::Wrap`'s batch dispatch) can pre-scan a batch without
            /// knowing the concrete strategy type.
            fn check_batch_domain(
                &self,
                points: &[[D::Elem; $N]],
            ) -> Result<(), InterpolateError> {
                let mut failures = Vec::new();
                let mut valid_indices = Vec::new();
                let mut transformed_points: Vec<[D::Elem; $N]> = Vec::new();
                for (index, point) in points.iter().enumerate() {
                    let point_failures = self.point_domain_failures(index, point.as_slice());
                    if point_failures.is_empty() {
                        valid_indices.push(index);
                        transformed_points.push(core::array::from_fn(|i| {
                            self.transforms[i].forward(point[i])
                        }));
                    } else {
                        failures.extend(point_failures);
                    }
                }
                match self.inner.check_batch_domain(&transformed_points) {
                    Ok(()) => {}
                    Err(InterpolateError::GridTransformDomain(inner_failures)) => {
                        failures.extend(inner_failures.into_iter().map(|f| OutsideDomainAt {
                            index: valid_indices[f.index],
                            ..f
                        }));
                    }
                    Err(other) => return Err(other),
                }
                failures.sort_by_key(|f| (f.index, f.dim));
                if failures.is_empty() {
                    Ok(())
                } else {
                    Err(InterpolateError::GridTransformDomain(failures))
                }
            }

            fn allow_extrapolate(&self) -> bool {
                self.inner.allow_extrapolate()
            }
        }
    };
}
pub(crate) use grid_transform_strategy_impl;

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
    /// Transform applied to `values`. Serializes as `values_transform`.
    #[cfg_attr(feature = "serde", serde(rename = "values_transform"))]
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

    /// `values_cache` reshaped to `Dim`. Shared by `interpolate`/`interpolate_wrapped`/
    /// `interpolate_fast`/`batch_interpolate_into` across 1D/2D/3D (ND keeps
    /// `values_cache` dynamically-shaped throughout, so it doesn't need this).
    ///
    /// # Panics
    /// If the cached shape doesn't match `Dim`. Doesn't happen in practice: `init`
    /// populates `values_cache` from a `Dim`-shaped `values` array before any of the
    /// above can run.
    pub(crate) fn cached_values_view<Dim: Dimension>(&self) -> ArrayView<'_, T, Dim> {
        self.values_cache
            .view()
            .into_dimensionality::<Dim>()
            .expect("values_cache shape matches values")
    }
}

impl<T: Float, S> ValuesTransform<T, S> {
    /// Domain-checks and forward-transforms `values` (single pass, fused). Shared
    /// by `validate` (result discarded) and `init` (result cached into
    /// `values_cache`) across all four dimensionalities.
    ///
    /// `NaN` values pass the domain check (`in_domain` treats `NaN` as always
    /// in-domain) and forward-transform to `NaN` untouched: real datasets commonly
    /// use `NaN` as a "no data" sentinel for unmeasured grid points, and that should
    /// propagate through, not fail construction. This is unlike a NaN *query point*
    /// under `GridTransform`, which is rejected (see
    /// [`GridTransform::point_domain_failures`]) since it would otherwise reach the
    /// inner strategy's grid search and panic there instead.
    pub(crate) fn transform_values<Dim: Dimension>(
        &self,
        values: ArrayView<T, Dim>,
    ) -> Result<Array<T, Dim>, ValidateError> {
        let mut transformed = Vec::with_capacity(values.len());
        for (pattern, &v) in values.indexed_iter() {
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

/// Generates `impl $Strategy<D> for ValuesTransform<D::Elem, S>`, for 1D/2D/3D
/// (`$N` in `[D::Elem; $N]`, `$Ix` the matching `ndarray` dimension type);
/// `StrategyND` stays hand-written in `n/strategies.rs`, for the same reason
/// [`grid_transform_strategy_impl`] doesn't cover it.
macro_rules! values_transform_strategy_impl {
    ($Strategy:ident, $InterpDataBase:ident, $InterpDataView:ident, $Ix:ident, $N:literal) => {
        impl<D, S> $Strategy<D> for ValuesTransform<D::Elem, S>
        where
            D: Data + RawDataClone + Clone,
            D::Elem: Float + Debug,
            S: for<'a> $Strategy<ViewRepr<&'a D::Elem>> + Clone + Debug,
        {
            /// Checks that every data value is in the configured transform's domain.
            fn validate(&self, data: &$InterpDataBase<D>) -> Result<(), ValidateError> {
                let transformed_values = self.transform_values(data.values.view())?;
                let view = $InterpDataView {
                    grid: core::array::from_fn(|i| data.grid[i].view()),
                    values: transformed_values.view(),
                };
                self.inner.validate(&view)
            }

            /// Transforms `data.values` into `values_cache`, then initializes `inner`
            /// against a transient view zipping `data.grid` (untouched) with
            /// `values_cache`.
            fn init(&mut self, data: &$InterpDataBase<D>) -> Result<(), ValidateError> {
                self.values_cache = self.transform_values(data.values.view())?.into_dyn();
                // Not `cached_values_view`: that borrows all of `self`, conflicting
                // with `self.inner.init`'s `&mut self.inner` below. Direct field
                // access keeps the two borrows disjoint.
                let view = $InterpDataView {
                    grid: core::array::from_fn(|i| data.grid[i].view()),
                    values: self
                        .values_cache
                        .view()
                        .into_dimensionality::<$Ix>()
                        .expect("values_cache shape matches values"),
                };
                self.inner.init(&view)
            }

            fn interpolate(
                &self,
                data: &$InterpDataBase<D>,
                point: &[D::Elem; $N],
            ) -> Result<D::Elem, InterpolateError> {
                let view = $InterpDataView {
                    grid: core::array::from_fn(|i| data.grid[i].view()),
                    values: self.cached_values_view(),
                };
                let result = self.inner.interpolate(&view, point)?;
                Ok(self.transform.inverse(result))
            }

            /// Hands `point` unmodified to `inner.interpolate_wrapped`, so a nested
            /// `GridTransform` handles the actual raw-space wrap; must not wrap here
            /// itself.
            fn interpolate_wrapped(
                &self,
                data: &$InterpDataBase<D>,
                point: &[D::Elem; $N],
            ) -> Result<D::Elem, InterpolateError>
            where
                D::Elem: Num + Euclid + Copy,
            {
                let view = $InterpDataView {
                    grid: core::array::from_fn(|i| data.grid[i].view()),
                    values: self.cached_values_view(),
                };
                let result = self.inner.interpolate_wrapped(&view, point)?;
                Ok(self.transform.inverse(result))
            }

            /// Calls `inner.interpolate_fast` rather than `inner.interpolate`, so
            /// "fast" propagates through nested `GridTransform`/`ValuesTransform`
            /// layers instead of stopping at the outermost one.
            fn interpolate_fast(
                &self,
                data: &$InterpDataBase<D>,
                point: &[D::Elem; $N],
            ) -> D::Elem {
                let view = $InterpDataView {
                    grid: core::array::from_fn(|i| data.grid[i].view()),
                    values: self.cached_values_view(),
                };
                let result = self.inner.interpolate_fast(&view, point);
                self.transform.inverse(result)
            }

            /// Delegates the whole batch to `inner` via a transient view over
            /// `values_cache`, so a nested `GridTransform`'s batch-domain aggregation
            /// isn't lost to the point-by-point default; inverse-transforms every
            /// output afterward.
            fn batch_interpolate_into(
                &self,
                data: &$InterpDataBase<D>,
                points: &[[D::Elem; $N]],
                out: &mut [D::Elem],
            ) -> Result<(), InterpolateError> {
                if out.len() != points.len() {
                    return Err(InterpolateError::OutputLength {
                        expected: points.len(),
                        found: out.len(),
                    });
                }
                let view = $InterpDataView {
                    grid: core::array::from_fn(|i| data.grid[i].view()),
                    values: self.cached_values_view(),
                };
                self.inner.batch_interpolate_into(&view, points, out)?;
                for o in out.iter_mut() {
                    *o = self.transform.inverse(*o);
                }
                Ok(())
            }

            /// Forwards to `inner`, so a nested `GridTransform`'s domain is still
            /// checked through a wrapping `ValuesTransform`.
            fn check_batch_domain(&self, points: &[[D::Elem; $N]]) -> Result<(), InterpolateError> {
                self.inner.check_batch_domain(points)
            }

            fn allow_extrapolate(&self) -> bool {
                self.inner.allow_extrapolate()
            }
        }
    };
}
pub(crate) use values_transform_strategy_impl;

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

        assert!(Transform::Identity.in_domain(f64::NAN));
        assert!(Transform::Log.in_domain(f64::NAN));
        assert!(Transform::Sqrt.in_domain(f64::NAN));
        assert!(Transform::Reciprocal.in_domain(f64::NAN));
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
