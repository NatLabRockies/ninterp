//! Module for all interpolation types.

use core::any::Any;

use super::*;

mod n;
mod one;
mod three;
mod two;
mod zero;

pub mod data;
pub mod enums;

pub use data::InterpDataBase;
pub use n::{InterpND, InterpNDBase, InterpNDView};
pub use one::{Interp1D, Interp1DBase, Interp1DView};
pub use three::{Interp3D, Interp3DBase, Interp3DView};
pub use two::{Interp2D, Interp2DBase, Interp2DView};
pub use zero::Interp0D;

/// An interpolator of data type `T`
///
/// This trait is dyn-compatible, meaning you can use:
/// `Box<dyn Interpolator<_>>`
/// and swap the contained interpolator at runtime.
pub trait Interpolator<T>: DynClone {
    /// Interpolator dimensionality.
    fn ndim(&self) -> usize;
    /// Validate interpolator data.
    fn validate(&self) -> Result<(), ValidateError>;
    /// Interpolate at supplied point.
    fn interpolate(&self, point: &[T]) -> Result<T, InterpolateError>;
    /// Set [`Extrapolate`] variant, checking validity.
    fn set_extrapolate(&mut self, extrapolate: Extrapolate<T>) -> Result<(), ValidateError>;
    /// Interpolate without bounds/extrapolation checks, for use in hot loops where the
    /// caller has already checked bounds and knows that extrapolation is not needed.
    ///
    /// Default just unwraps [`Interpolator::interpolate`], non-breaking for existing
    /// implementors. `Interp1D`/`2D`/`3D`/`ND` override this to actually skip the checks
    /// instead of just discarding the `Result`; through `Box<dyn Interpolator<T>>` this
    /// still pays vtable dispatch either way, so the win is smaller than calling a
    /// concrete `InterpXD<D, S>::interpolate_fast` directly.
    ///
    /// # Panics
    /// Panics if `point.len()` doesn't match [`Interpolator::ndim`], or if the strategy's
    /// checked `interpolate` would have returned `Err` for a reason other than length or
    /// the outer bounds/extrapolation check. No strategy shipped in this crate does that
    /// today; it would only apply to a strategy with real per-point fallible work inside
    /// its own `interpolate`, distinct from anything already catchable once via
    /// `validate`/`init` at construction time.
    fn interpolate_fast(&self, point: &[T]) -> T {
        self.interpolate(point)
            .expect("interpolate_fast: invalid point or data")
    }

    /// Interpolate at each of several points, writing results into `out` instead of
    /// allocating. `out.len()` must equal `points.len()`.
    ///
    /// `self.extrapolate` is one setting for the whole call, not resolved per
    /// point. Default just loops [`Interpolator::interpolate`] into `out`;
    /// `Interp1D`/`2D`/`3D`/`ND` override this to funnel every point into at most
    /// one call to the strategy instead.
    fn batch_interpolate_into(
        &self,
        points: &[&[T]],
        out: &mut [T],
    ) -> Result<(), InterpolateError> {
        if out.len() != points.len() {
            return Err(InterpolateError::OutputLength {
                expected: points.len(),
                found: out.len(),
            });
        }
        for (o, point) in out.iter_mut().zip(points) {
            *o = self.interpolate(point)?;
        }
        Ok(())
    }

    /// Unchecked [`Interpolator::batch_interpolate_into`], assuming every point is valid.
    ///
    /// Default loops [`Interpolator::interpolate_fast`] into `out`, panicking on
    /// length mismatch. `Interp1D`/`2D`/`3D`/`ND` override this the same way as
    /// [`Interpolator::batch_interpolate_into`].
    ///
    /// # Panics
    /// Panics if `out.len() != points.len()`.
    fn batch_interpolate_fast_into(&self, points: &[&[T]], out: &mut [T]) {
        assert_eq!(
            out.len(),
            points.len(),
            "batch_interpolate_fast_into: length mismatch"
        );
        for (o, point) in out.iter_mut().zip(points) {
            *o = self.interpolate_fast(point);
        }
    }

    /// Interpolate at each of several points, sharing one grid across all of them.
    ///
    /// `self.extrapolate` is one setting for the whole call, not resolved per
    /// point. Default allocates an output buffer and calls [`Interpolator::batch_interpolate_into`];
    /// `Interp1D`/`2D`/`3D`/`ND` override [`Self::batch_interpolate_into`] to funnel every point into
    /// at most one call to the strategy instead. Do not override this method.
    fn batch_interpolate(&self, points: &[&[T]]) -> Result<Vec<T>, InterpolateError>
    where
        T: Num,
    {
        let mut out = Vec::with_capacity(points.len());
        for _ in 0..points.len() {
            out.push(T::zero());
        }
        self.batch_interpolate_into(points, &mut out)?;
        Ok(out)
    }

    /// Batched [`Interpolator::interpolate_fast`], assuming every point is already
    /// valid.
    ///
    /// Default allocates an output buffer and calls [`Interpolator::batch_interpolate_fast_into`].
    /// Do not override this method. If you need to amortize work across the batch, override
    /// [`Self::batch_interpolate_into`] instead; the fast variant is rarely optimized.
    fn batch_interpolate_fast(&self, points: &[&[T]]) -> Vec<T>
    where
        T: Num + Copy,
    {
        let mut out = Vec::with_capacity(points.len());
        for _ in 0..points.len() {
            out.push(T::zero());
        }
        self.batch_interpolate_fast_into(points, &mut out);
        out
    }
}

clone_trait_object!(<T> Interpolator<T>);

impl<T> Interpolator<T> for Box<dyn Interpolator<T>> {
    fn ndim(&self) -> usize {
        (**self).ndim()
    }
    fn validate(&self) -> Result<(), ValidateError> {
        (**self).validate()
    }
    fn interpolate(&self, point: &[T]) -> Result<T, InterpolateError> {
        (**self).interpolate(point)
    }
    fn set_extrapolate(&mut self, extrapolate: Extrapolate<T>) -> Result<(), ValidateError> {
        (**self).set_extrapolate(extrapolate)
    }
    fn interpolate_fast(&self, point: &[T]) -> T {
        (**self).interpolate_fast(point)
    }
    fn batch_interpolate_into(
        &self,
        points: &[&[T]],
        out: &mut [T],
    ) -> Result<(), InterpolateError> {
        (**self).batch_interpolate_into(points, out)
    }
    fn batch_interpolate_fast_into(&self, points: &[&[T]], out: &mut [T]) {
        (**self).batch_interpolate_fast_into(points, out)
    }
    fn batch_interpolate(&self, points: &[&[T]]) -> Result<Vec<T>, InterpolateError>
    where
        T: Num,
    {
        (**self).batch_interpolate(points)
    }
    fn batch_interpolate_fast(&self, points: &[&[T]]) -> Vec<T>
    where
        T: Num + Copy,
    {
        (**self).batch_interpolate_fast(points)
    }
}

/// A `Send + Sync`, downcastable counterpart to [`Interpolator<T>`], for storing
/// heterogeneous interpolators behind `Box<dyn AnyInterpolator<T>>`.
///
/// Not in the [`prelude`](`crate::prelude`); reach for it explicitly
/// (`ninterp::interpolator::AnyInterpolator`).
///
/// Implemented for owned `Interp1D`/`2D`/`3D`/`ND` types only:
/// [`as_any`](AnyInterpolator::as_any) requires `Self: 'static`, which the borrowed
/// `Interp*View` types can't satisfy. A viewed interpolator can still be used
/// through [`Interpolator<T>`].
pub trait AnyInterpolator<T>: Interpolator<T> + Send + Sync {
    /// Downcast to the concrete interpolator type.
    fn as_any(&self) -> &dyn Any;
}

/// Extrapolation strategy
///
/// Controls what happens when supplied interpolation point
/// is outside the bounds of the coordinate grid.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[non_exhaustive]
pub enum Extrapolate<T> {
    /// Evaluate beyond the grid limits. Not applicable for all strategies.
    Enable,
    /// If point is beyond grid limits, return this value instead.
    Fill(T),
    /// Restrict interpolant point to the grid limits using [`num_traits::clamp`].
    Clamp,
    /// Wrap around to other end of (periodic) data.
    /// Does NOT check that first and last values are equal.
    Wrap,
    /// Return an error.
    #[default]
    Error,
}

// === Interpolation Logic ===

macro_rules! extrapolate_impl {
    ($InterpType:ident, $Strategy:ident) => {
        impl<D, S> $InterpType<D, S>
        where
            D: Data + RawDataClone + Clone,
            D::Elem: PartialEq + Debug,
            S: $Strategy<D> + Clone,
        {
            /// Check that `extrapolate` is applicable to the current strategy.
            ///
            /// Only [`Extrapolate::Enable`] can be rejected, and only by a strategy whose
            /// `allow_extrapolate` returns `false`. Takes the setting as an argument rather
            /// than reading `self.extrapolate`, so `set_extrapolate` can vet a candidate
            /// before storing it.
            pub fn validate_extrapolate(
                &self,
                extrapolate: &Extrapolate<D::Elem>,
            ) -> Result<(), ValidateError> {
                if matches!(extrapolate, Extrapolate::Enable) && !self.strategy.allow_extrapolate()
                {
                    return Err(ValidateError::ExtrapolateUnsupported);
                }
                Ok(())
            }
        }
    };
}
pub(crate) use extrapolate_impl;

/// Generates the inherent `interpolate` shared by `Interp1D`/`2D`/`3D`: loops over each
/// dimension, applying the [`Extrapolate`] setting per axis, then falls through to the
/// strategy once the point is known to be in-bounds (or already resolved via
/// `Fill`/`Clamp`/`Wrap`). `InterpND` has no fixed `N` to loop `0..N` over at compile
/// time, so it implements this by hand directly in its [`Interpolator`] impl instead.
macro_rules! interpolate_impl {
    () => {
        /// Interpolate at the supplied point.
        ///
        /// Unlike [`Interpolator::interpolate`], the point length is checked at compile
        /// time via `N`, so this cannot fail with [`InterpolateError::PointLength`].
        pub fn interpolate(&self, point: &[D::Elem; N]) -> Result<D::Elem, InterpolateError>
        where
            D::Elem: Num + PartialOrd + Euclid + Copy,
        {
            let mut errors = Vec::new();
            for dim in 0..N {
                if !(self.data.grid[dim].first().unwrap()..=self.data.grid[dim].last().unwrap())
                    .contains(&&point[dim])
                {
                    match &self.extrapolate {
                        Extrapolate::Enable => {}
                        Extrapolate::Fill(value) => return Ok(*value),
                        Extrapolate::Clamp => {
                            let clamped_point = std::array::from_fn(|i| {
                                *clamp(
                                    &point[i],
                                    self.data.grid[i].first().unwrap(),
                                    self.data.grid[i].last().unwrap(),
                                )
                            });
                            return self.strategy.interpolate(&self.data, &clamped_point);
                        }
                        Extrapolate::Wrap => {
                            let wrapped_point = std::array::from_fn(|i| {
                                wrap(
                                    point[i],
                                    *self.data.grid[i].first().unwrap(),
                                    *self.data.grid[i].last().unwrap(),
                                )
                            });
                            return self.strategy.interpolate(&self.data, &wrapped_point);
                        }
                        Extrapolate::Error => {
                            errors.push(OutOfBoundsAt { index: 0, dim });
                        }
                    };
                }
            }
            if !errors.is_empty() {
                return Err(InterpolateError::OutOfBounds(errors));
            }
            self.strategy.interpolate(&self.data, point)
        }
    };
}
pub(crate) use interpolate_impl;

/// Reinterpret a runtime-length point as a fixed-size array, for the `Interpolator<T>`
/// trait methods on `Interp1D`/`2D`/`3D`, whose inherent counterparts take `&[T; N]`.
pub(crate) fn to_fixed_point<T, const N: usize>(point: &[T]) -> Result<&[T; N], InterpolateError> {
    <&[T; N]>::try_from(point).map_err(|_| InterpolateError::PointLength {
        expected: N,
        failures: vec![WrongLengthAt {
            index: 0,
            found: point.len(),
        }],
    })
}

/// Batched [`to_fixed_point`]. Every offending point is reported, not just the first, so
/// one call surfaces the whole problem with a malformed batch. Mirrors how the
/// `Extrapolate::Error` path aggregates out-of-bounds points.
pub(crate) fn to_fixed_points<T: Copy, const N: usize>(
    points: &[&[T]],
) -> Result<Vec<[T; N]>, InterpolateError> {
    let mut converted = Vec::with_capacity(points.len());
    let mut failures = Vec::new();
    for (i, &point) in points.iter().enumerate() {
        match <&[T; N]>::try_from(point) {
            Ok(arr) => converted.push(*arr),
            Err(_) => failures.push(WrongLengthAt {
                index: i,
                found: point.len(),
            }),
        }
    }
    if failures.is_empty() {
        Ok(converted)
    } else {
        Err(InterpolateError::PointLength {
            expected: N,
            failures,
        })
    }
}

/// Is `point` out of `grid`'s bounds in any dimension?
///
/// Shared by `Interp1D`/`2D`/`3D`/`ND`'s `batch_interpolate`: both `grid` and `point`
/// are taken as slices, so one function covers `Interp1D`/`2D`/`3D`'s fixed-size
/// `[ArrayBase<D, Ix1>; N]` grid (via array-to-slice coercion) and `InterpND`'s
/// `Vec<ArrayBase<D, Ix1>>` alike.
pub(crate) fn out_of_bounds<D>(grid: &[ArrayBase<D, Ix1>], point: &[D::Elem]) -> bool
where
    D: Data,
    D::Elem: PartialOrd,
{
    grid.iter()
        .zip(point)
        .any(|(axis, coord)| !(axis.first().unwrap()..=axis.last().unwrap()).contains(&coord))
}

/// Generates the inherent `batch_interpolate_into` shared by `Interp1D`/`2D`/`3D`:
/// resolves `self.extrapolate` once for the whole batch (see the extrapolate-mode
/// partitioning table in issue #21), then funnels every point into at most one call
/// to the strategy, writing results into the provided output slice rather than
/// allocating a Vec.
macro_rules! batch_interpolate_into_impl {
    () => {
        /// Interpolate at each of several points, writing results into `out` instead of
        /// allocating.
        ///
        /// `self.extrapolate` is one setting for the whole call, not resolved per
        /// point: every point still funnels into at most one call to the strategy,
        /// rather than calling [`Self::interpolate`] once per point.
        ///
        /// # Errors
        /// Returns [`InterpolateError::OutputLength`] if `out.len() != points.len()`.
        /// Returns other interpolation errors from the strategy or extrapolation check.
        pub fn batch_interpolate_into(
            &self,
            points: &[[D::Elem; N]],
            out: &mut [D::Elem],
        ) -> Result<(), InterpolateError>
        where
            D::Elem: Num + PartialOrd + Euclid + Copy,
        {
            if out.len() != points.len() {
                return Err(InterpolateError::OutputLength {
                    expected: points.len(),
                    found: out.len(),
                });
            }
            match &self.extrapolate {
                Extrapolate::Enable => self
                    .strategy
                    .batch_interpolate_into(&self.data, points, out),
                Extrapolate::Clamp => {
                    // Clamping an in-bounds point is already identity, so every point
                    // can be clamped unconditionally.
                    let clamped: Vec<[D::Elem; N]> = points
                        .iter()
                        .map(|point| {
                            std::array::from_fn(|i| {
                                *clamp(
                                    &point[i],
                                    self.data.grid[i].first().unwrap(),
                                    self.data.grid[i].last().unwrap(),
                                )
                            })
                        })
                        .collect();
                    self.strategy
                        .batch_interpolate_into(&self.data, &clamped, out)
                }
                Extrapolate::Wrap => {
                    // Unlike `Clamp`, `wrap()` isn't identity exactly at the
                    // boundary, so only out-of-bounds points get wrapped.
                    let wrapped: Vec<[D::Elem; N]> = points
                        .iter()
                        .map(|point| {
                            if out_of_bounds(&self.data.grid, point) {
                                std::array::from_fn(|i| {
                                    wrap(
                                        point[i],
                                        *self.data.grid[i].first().unwrap(),
                                        *self.data.grid[i].last().unwrap(),
                                    )
                                })
                            } else {
                                *point
                            }
                        })
                        .collect();
                    self.strategy
                        .batch_interpolate_into(&self.data, &wrapped, out)
                }
                Extrapolate::Fill(value) => {
                    // Pre-fill output with the fill value, then scatter interpolated
                    // results from in-bounds points into their corresponding indices.
                    for o in out.iter_mut() {
                        *o = *value;
                    }
                    let (in_bounds_indices, in_bounds_points): (Vec<usize>, Vec<[D::Elem; N]>) =
                        points
                            .iter()
                            .enumerate()
                            .filter(|(_, point)| !out_of_bounds(&self.data.grid, *point))
                            .map(|(i, point)| (i, *point))
                            .unzip();
                    if !in_bounds_indices.is_empty() {
                        let mut scratch = vec![D::Elem::zero(); in_bounds_indices.len()];
                        self.strategy.batch_interpolate_into(
                            &self.data,
                            &in_bounds_points,
                            &mut scratch,
                        )?;
                        for (idx, value) in in_bounds_indices.into_iter().zip(scratch) {
                            out[idx] = value;
                        }
                    }
                    Ok(())
                }
                Extrapolate::Error => {
                    let mut errors = Vec::new();
                    let mut in_bounds_points = Vec::new();
                    for (i, point) in points.iter().enumerate() {
                        let mut point_errors = Vec::new();
                        for dim in 0..N {
                            if !(self.data.grid[dim].first().unwrap()
                                ..=self.data.grid[dim].last().unwrap())
                                .contains(&&point[dim])
                            {
                                point_errors.push(OutOfBoundsAt { index: i, dim });
                            }
                        }
                        if point_errors.is_empty() {
                            in_bounds_points.push(*point);
                        } else {
                            errors.extend(point_errors);
                        }
                    }
                    if !errors.is_empty() {
                        return Err(InterpolateError::OutOfBounds(errors));
                    }
                    self.strategy
                        .batch_interpolate_into(&self.data, &in_bounds_points, out)
                }
            }
        }
    };
}
pub(crate) use batch_interpolate_into_impl;

/// Generates the inherent `batch_interpolate` shared by `Interp1D`/`2D`/`3D`:
/// allocates an output buffer and delegates to [`batch_interpolate_into`].
macro_rules! batch_interpolate_impl {
    () => {
        /// Interpolate at each of several points, sharing one grid across all of
        /// them.
        ///
        /// Allocates an output buffer and calls [`Self::batch_interpolate_into`],
        /// which handles `self.extrapolate` resolution and funnels every point into
        /// at most one call to the strategy rather than calling [`Self::interpolate`]
        /// once per point.
        pub fn batch_interpolate(
            &self,
            points: &[[D::Elem; N]],
        ) -> Result<Vec<D::Elem>, InterpolateError>
        where
            D::Elem: Num + PartialOrd + Euclid + Copy,
        {
            let mut out = vec![D::Elem::zero(); points.len()];
            self.batch_interpolate_into(points, &mut out)?;
            Ok(out)
        }
    };
}
pub(crate) use batch_interpolate_impl;

// === Strategy Access & Updates ===

/// Generates the inherent `validate_strategy`/`init_strategy`, shared by
/// `Interp1D`/`2D`/`3D`/`ND`. Both only need `D::Elem: PartialEq + Debug` (the struct's
/// own bound), not the extra bounds of `new`/`interpolate`/etc. Kept on those methods
/// directly rather than the enclosing impl block, so all inherent methods can share one
/// block per type instead of being split across several.
macro_rules! strategy_accessors_impl {
    ($Strategy:ident) => {
        #[doc = concat!(
            " Re-run the strategy's [`", stringify!($Strategy), "::validate`] against the current data.\n",
            "\n",
            " `new`, `set_strategy`, and [`Interpolator::validate`] already call this\n",
            " internally, so this is only needed after mutating the public `data`/`strategy`\n",
            " fields directly.",
        )]
        pub fn validate_strategy(&self) -> Result<(), ValidateError> {
            self.strategy.validate(&self.data)
        }

        #[doc = concat!(
            " Re-run the strategy's [`", stringify!($Strategy), "::init`] against the current data.\n",
            "\n",
            " `new` and `set_strategy` already call this internally, so this is only needed\n",
            " after bypassing them: mutating the public `data`/`strategy` fields directly, or\n",
            " deserializing an interpolator whose strategy skips its cached state from\n",
            " serialization (e.g. via `#[serde(skip)]`, to avoid bloating the wire format with\n",
            " a large derived array). `Deserialize` does not call `init`; if the cached state\n",
            " is instead stored in ordinary serialized fields, it comes back as-is and this\n",
            " isn't needed.",
        )]
        pub fn init_strategy(&mut self) -> Result<(), ValidateError> {
            self.strategy.init(&self.data)
        }
    };
}
pub(crate) use strategy_accessors_impl;

/// Generates the `Box<dyn $Strategy<D>>`-backed inherent `set_strategy`, shared by
/// `Interp1D`/`2D`/`3D`/`ND`.
macro_rules! set_strategy_box_impl {
    ($InterpType:ident, $Strategy:ident) => {
        impl<D> $InterpType<D, Box<dyn $Strategy<D>>>
        where
            D: Data + RawDataClone + Clone,
            D::Elem: PartialEq + Debug,
        {
            #[doc = concat!(
                " Update strategy at runtime, calling [`", stringify!($Strategy), "::init`] on the new strategy\n",
                " against the current data.\n",
                "\n",
                " To swap in a strategy without re-running `init` (e.g. one whose state was\n",
                " already established elsewhere), assign the `strategy` field directly instead.",
            )]
            pub fn set_strategy(
                &mut self,
                strategy: Box<dyn $Strategy<D>>,
            ) -> Result<(), ValidateError> {
                self.strategy = strategy;
                self.validate_extrapolate(&self.extrapolate)?;
                self.validate_strategy()?;
                self.init_strategy()
            }
        }
    };
}
pub(crate) use set_strategy_box_impl;

/// Generates the `$StrategyEnum`-backed inherent `set_strategy`, shared by
/// `Interp1D`/`2D`/`3D`/`ND`.
macro_rules! set_strategy_enum_impl {
    ($InterpType:ident, $StrategyEnum:path) => {
        impl<D> $InterpType<D, $StrategyEnum>
        where
            D: Data + RawDataClone + Clone,
            D::Elem: Float + Debug,
        {
            #[doc = concat!(
                " Update strategy at runtime, calling [`", stringify!($StrategyEnum), "::init`] on the new strategy\n",
                " against the current data.\n",
                "\n",
                " To swap in a strategy without re-running `init` (e.g. one whose state was\n",
                " already established elsewhere), assign the `strategy` field directly instead.",
            )]
            pub fn set_strategy(
                &mut self,
                strategy: impl Into<$StrategyEnum>,
            ) -> Result<(), ValidateError> {
                self.strategy = strategy.into();
                self.validate_extrapolate(&self.extrapolate)?;
                self.validate_strategy()?;
                self.init_strategy()
            }
        }
    };
}
pub(crate) use set_strategy_enum_impl;

// === Trait Implementations ===

macro_rules! partialeq_impl {
    ($InterpType:ident, $Data:ident, $Strategy:ident) => {
        impl<D, S> PartialEq for $InterpType<D, S>
        where
            D: Data + RawDataClone + Clone,
            D::Elem: PartialEq + Debug,
            S: Clone + PartialEq,
            $Data<D>: PartialEq,
        {
            fn eq(&self, other: &Self) -> bool {
                self.data == other.data
                    && self.strategy == other.strategy
                    && self.extrapolate == other.extrapolate
            }
        }
    };
}
pub(crate) use partialeq_impl;

/// Generates the entire [`Interpolator<T>`] trait impl shared by `Interp1D`/`2D`/`3D`/`ND`.
/// Parameterized by interpolator type, strategy trait, and the ndim return value
/// (a literal for fixed-dimensionality types, `self.data.ndim()` for ND). Includes slice-to-array
/// conversion logic for `interpolate`/`interpolate_fast`/`batch_interpolate`/`batch_interpolate_fast`.
macro_rules! interpolator_trait_impl {
    ($InterpType:ident, $Strategy:ident, $NdimExpr:expr) => {
        impl<D, S> Interpolator<D::Elem> for $InterpType<D, S>
        where
            D: Data + RawDataClone + Clone,
            D::Elem: Num + PartialOrd + Euclid + Copy + Debug,
            S: $Strategy<D> + Clone,
        {
            #[inline]
            fn ndim(&self) -> usize {
                $NdimExpr
            }

            fn validate(&self) -> Result<(), ValidateError> {
                self.validate_extrapolate(&self.extrapolate)?;
                self.data.validate()?;
                self.validate_strategy()?;
                Ok(())
            }

            fn interpolate(&self, point: &[D::Elem]) -> Result<D::Elem, InterpolateError> {
                self.interpolate(to_fixed_point(point)?)
            }

            fn interpolate_fast(&self, point: &[D::Elem]) -> D::Elem {
                let point: &[D::Elem; N] = point
                    .try_into()
                    .expect("interpolate_fast: point length mismatch");
                self.interpolate_fast(point)
            }

            fn batch_interpolate_into(
                &self,
                points: &[&[D::Elem]],
                out: &mut [D::Elem],
            ) -> Result<(), InterpolateError> {
                let points: Vec<[D::Elem; N]> = to_fixed_points(points)?;
                self.batch_interpolate_into(&points, out)
            }

            fn batch_interpolate_fast_into(&self, points: &[&[D::Elem]], out: &mut [D::Elem]) {
                let points: Vec<[D::Elem; N]> = points
                    .iter()
                    .map(|&point| {
                        *<&[D::Elem; N]>::try_from(point)
                            .expect("batch_interpolate_fast_into: point length mismatch")
                    })
                    .collect();
                self.batch_interpolate_fast_into(&points, out)
            }

            fn batch_interpolate(
                &self,
                points: &[&[D::Elem]],
            ) -> Result<Vec<D::Elem>, InterpolateError> {
                let points: Vec<[D::Elem; N]> = to_fixed_points(points)?;
                self.batch_interpolate(&points)
            }

            fn batch_interpolate_fast(&self, points: &[&[D::Elem]]) -> Vec<D::Elem> {
                let points: Vec<[D::Elem; N]> = points
                    .iter()
                    .map(|&point| {
                        *<&[D::Elem; N]>::try_from(point)
                            .expect("batch_interpolate_fast: point length mismatch")
                    })
                    .collect();
                self.batch_interpolate_fast(&points)
            }

            fn set_extrapolate(
                &mut self,
                extrapolate: Extrapolate<D::Elem>,
            ) -> Result<(), ValidateError> {
                self.validate_extrapolate(&extrapolate)?;
                self.extrapolate = extrapolate;
                Ok(())
            }
        }
    };
}
pub(crate) use interpolator_trait_impl;

#[cfg(feature = "serde")]
macro_rules! serialize_nested_impl {
    ($InterpType:ident, $Data:ident, $Strategy:ident) => {
        impl<D, S> SerializeNested for $InterpType<D, S>
        where
            D: Data + RawDataClone + Clone,
            D::Elem: PartialEq + Debug + Serialize,
            S: Clone + Serialize,
            $Data<D>: SerializeNested + Serialize,
        {
            fn serialize_nested<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
            where
                Ser: Serializer,
            {
                let mut s = serializer.serialize_struct(stringify!($InterpType), 3)?;
                s.serialize_field("data", &Nested(&self.data))?;
                s.serialize_field("strategy", &self.strategy)?;
                s.serialize_field("extrapolate", &self.extrapolate)?;
                s.end()
            }
        }
    };
}

/// Generates the [`AnyInterpolator`] impl shared by `Interp1D`/`2D`/`3D`/`ND`'s owned
/// variants. Bounded on `Float + Euclid` (rather than the `Num + PartialOrd + Euclid +
/// Copy` the [`Interpolator`] impls use) because that's what [`AnyInterpolator`] itself
/// requires transitively; only owned types implement it, since `as_any` requires
/// `Self: 'static`.
macro_rules! any_interpolator_impl {
    ($InterpTypeOwned:ident, $Strategy:ident) => {
        impl<T, S> AnyInterpolator<T> for $InterpTypeOwned<T, S>
        where
            T: Float + Euclid + Debug + Send + Sync + 'static,
            S: $Strategy<OwnedRepr<T>> + Clone + Send + Sync + 'static,
        {
            fn as_any(&self) -> &dyn Any {
                self
            }
        }
    };
}
pub(crate) use any_interpolator_impl;

#[cfg(feature = "serde")]
pub(crate) use serialize_nested_impl;

// === Data Conversion ===

/// Generates `view()` and `into_owned()` inherent methods shared by `Interp1D`/`2D`/`3D`.
/// These methods have identical bodies except for return types, which are passed as parameters.
macro_rules! view_into_owned_impl {
    ($InterpType:ident, $Strategy:ident, $Viewed:ty, $Owned:ty) => {
        /// Return an interpolator with viewed data.
        pub fn view(&self) -> $Viewed
        where
            S: for<'a> $Strategy<ViewRepr<&'a D::Elem>>,
            D::Elem: Clone,
        {
            $InterpType {
                data: self.data.view(),
                strategy: self.strategy.clone(),
                extrapolate: self.extrapolate.clone(),
            }
        }

        /// Turn the interpolator into an owned variant, cloning the array elements if necessary.
        pub fn into_owned(self) -> $Owned
        where
            S: $Strategy<OwnedRepr<D::Elem>>,
            D::Elem: Clone,
        {
            $InterpType {
                data: self.data.into_owned(),
                strategy: self.strategy.clone(),
                extrapolate: self.extrapolate.clone(),
            }
        }
    };
}
pub(crate) use view_into_owned_impl;

// === High-level Groupers ===

/// Generates all trait impls for an interpolator: `PartialEq`, `SerializeNested`,
/// `extrapolate_impl`, `Interpolator`, `set_strategy` for `Box<dyn Strategy>`,
/// `set_strategy` for the strategy enum, and `AnyInterpolator`.
macro_rules! interp_trait_impls {
    ($InterpType:ident, $InterpTypeOwned:ident, $InterpData:ident, $Strategy:ident, $StrategyEnum:path, $N:expr) => {
        partialeq_impl!($InterpType, $InterpData, $Strategy);
        extrapolate_impl!($InterpType, $Strategy);
        interpolator_trait_impl!($InterpType, $Strategy, $N);
        set_strategy_box_impl!($InterpType, $Strategy);
        set_strategy_enum_impl!($InterpType, $StrategyEnum);
        any_interpolator_impl!($InterpTypeOwned, $Strategy);
        #[cfg(feature = "serde")]
        serialize_nested_impl!($InterpType, $InterpData, $Strategy);
    };
}
pub(crate) use interp_trait_impls;

/// Generates inherent methods for an interpolator: strategy accessors, fast paths,
/// data access (view/into_owned), and batch interpolation. Called inside the
/// `impl<D, S>` block, leaving `pub fn new()` to be hand-written.
macro_rules! interp_inherent_methods {
    ($InterpType:ident, $Strategy:ident, $Viewed:ty, $Owned:ty) => {
        interpolate_impl!();
        /// Interpolate without bounds/extrapolation checks, for use in hot loops where the
        /// caller has already checked bounds or knows that extrapolation handling is not needed.
        pub fn interpolate_fast(&self, point: &[D::Elem; N]) -> D::Elem {
            self.strategy.interpolate_fast(&self.data, point)
        }
        batch_interpolate_into_impl!();
        batch_interpolate_impl!();
        /// Unchecked batched [`Self::interpolate_fast`], assuming every point is valid
        /// and `out.len() == points.len()`.
        ///
        /// # Panics
        /// Panics if `out.len() != points.len()`.
        pub fn batch_interpolate_fast_into(&self, points: &[[D::Elem; N]], out: &mut [D::Elem]) {
            assert_eq!(
                out.len(),
                points.len(),
                "batch_interpolate_fast_into: length mismatch"
            );
            self.strategy
                .batch_interpolate_fast_into(&self.data, points, out)
        }
        /// Batched [`Self::interpolate_fast`], for use in hot loops where the caller
        /// has already checked bounds. Allocates an output buffer and calls
        /// [`Self::batch_interpolate_fast_into`].
        pub fn batch_interpolate_fast(&self, points: &[[D::Elem; N]]) -> Vec<D::Elem>
        where
            D::Elem: Num + Copy,
        {
            let mut out = Vec::with_capacity(points.len());
            for _ in 0..points.len() {
                out.push(D::Elem::zero());
            }
            self.batch_interpolate_fast_into(points, &mut out);
            out
        }
        strategy_accessors_impl!($Strategy);
        view_into_owned_impl!($InterpType, $Strategy, $Viewed, $Owned);
    };
}
pub(crate) use interp_inherent_methods;
