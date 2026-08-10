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

pub use n::{InterpND, InterpNDOwned, InterpNDViewed};
pub use one::{Interp1D, Interp1DOwned, Interp1DViewed};
pub use three::{Interp3D, Interp3DOwned, Interp3DViewed};
pub use two::{Interp2D, Interp2DOwned, Interp2DViewed};
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

    /// Interpolate at each of several points, sharing one grid across all of them.
    ///
    /// `self.extrapolate` is one setting for the whole call, not resolved per
    /// point. Default just loops [`Interpolator::interpolate`]; `Interp1D`/`2D`/
    /// `3D`/`ND` override this to funnel every point into at most one call to the
    /// strategy instead.
    fn batch_interpolate(&self, points: &[&[T]]) -> Result<Vec<T>, InterpolateError> {
        points.iter().map(|point| self.interpolate(point)).collect()
    }

    /// Batched [`Interpolator::interpolate_fast`], assuming every point is already
    /// valid.
    ///
    /// Default just loops [`Interpolator::interpolate_fast`]. `Interp1D`/`2D`/`3D`/
    /// `ND` override this the same way as [`Interpolator::batch_interpolate`].
    fn batch_interpolate_fast(&self, points: &[&[T]]) -> Vec<T> {
        points
            .iter()
            .map(|point| self.interpolate_fast(point))
            .collect()
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
    fn batch_interpolate(&self, points: &[&[T]]) -> Result<Vec<T>, InterpolateError> {
        (**self).batch_interpolate(points)
    }
    fn batch_interpolate_fast(&self, points: &[&[T]]) -> Vec<T> {
        (**self).batch_interpolate_fast(points)
    }
}

/// A `Send + Sync`, downcastable counterpart to [`Interpolator<T>`], for storing
/// heterogeneous interpolators behind `Box<dyn DynInterpolator<T>>`.
///
/// Not in the [`prelude`](`crate::prelude`); reach for it explicitly
/// (`ninterp::interpolator::DynInterpolator`).
///
/// Implemented for owned `Interp1D`/`2D`/`3D`/`ND` types only:
/// [`as_any`](DynInterpolator::as_any) requires `Self: 'static`, which the borrowed
/// `Interp*Viewed` types can't satisfy. A viewed interpolator can still be used
/// through [`Interpolator<T>`].
pub trait DynInterpolator<T>: Interpolator<T> + Send + Sync {
    /// Downcast to the concrete interpolator type.
    fn as_any(&self) -> &dyn Any;
}

/// Extrapolation strategy
///
/// Controls what happens when supplied interpolation point
/// is outside the bounds of the coordinate grid.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
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
            /// Check applicability of strategy, data, and extrapolate setting.
            pub fn check_extrapolate(
                &self,
                extrapolate: &Extrapolate<D::Elem>,
            ) -> Result<(), ValidateError> {
                // Check applicability of strategy and extrapolate setting
                if matches!(extrapolate, Extrapolate::Enable) && !self.strategy.allow_extrapolate()
                {
                    return Err(ValidateError::InvalidExtrapolate(format!(
                        "{:?}",
                        self.extrapolate
                    )));
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
                            errors.push(format!(
                                "\n    point[{dim}] = {:?} is out of bounds for grid[{dim}] = {:?}",
                                point[dim], self.data.grid[dim],
                            ));
                        }
                    };
                }
            }
            if !errors.is_empty() {
                return Err(InterpolateError::ExtrapolateError(errors.join("")));
            }
            self.strategy.interpolate(&self.data, point)
        }
    };
}
pub(crate) use interpolate_impl;

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

/// Generates the inherent `batch_interpolate` shared by `Interp1D`/`2D`/`3D`:
/// resolves `self.extrapolate` once for the whole batch (see the extrapolate-mode
/// partitioning table in issue #21), then funnels every point into at most one call
/// to the strategy, rather than calling the existing hand-written `interpolate` once
/// per point.
macro_rules! batch_interpolate_impl {
    () => {
        /// Interpolate at each of several points, sharing one grid across all of
        /// them.
        ///
        /// `self.extrapolate` is one setting for the whole call, not resolved per
        /// point: every point still funnels into at most one call to the strategy,
        /// rather than calling [`Self::interpolate`] once per point.
        pub fn batch_interpolate(
            &self,
            points: &[[D::Elem; N]],
        ) -> Result<Vec<D::Elem>, InterpolateError>
        where
            D::Elem: Num + PartialOrd + Euclid + Copy,
        {
            match &self.extrapolate {
                Extrapolate::Enable => self.strategy.batch_interpolate(&self.data, points),
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
                    self.strategy.batch_interpolate(&self.data, &clamped)
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
                    self.strategy.batch_interpolate(&self.data, &wrapped)
                }
                Extrapolate::Fill(value) => {
                    let mut results = vec![*value; points.len()];
                    let (in_bounds_indices, in_bounds_points): (Vec<usize>, Vec<[D::Elem; N]>) =
                        points
                            .iter()
                            .enumerate()
                            .filter(|(_, point)| !out_of_bounds(&self.data.grid, *point))
                            .map(|(i, point)| (i, *point))
                            .unzip();
                    let interpolated =
                        self.strategy.batch_interpolate(&self.data, &in_bounds_points)?;
                    for (i, value) in in_bounds_indices.into_iter().zip(interpolated) {
                        results[i] = value;
                    }
                    Ok(results)
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
                                point_errors.push(format!(
                                    "\n    point[{i}][{dim}] = {:?} is out of bounds for grid[{dim}] = {:?}",
                                    point[dim], self.data.grid[dim],
                                ));
                            }
                        }
                        if point_errors.is_empty() {
                            in_bounds_points.push(*point);
                        } else {
                            errors.extend(point_errors);
                        }
                    }
                    if !errors.is_empty() {
                        return Err(InterpolateError::ExtrapolateError(errors.join("")));
                    }
                    self.strategy.batch_interpolate(&self.data, &in_bounds_points)
                }
            }
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
                self.check_extrapolate(&self.extrapolate)?;
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
                self.check_extrapolate(&self.extrapolate)?;
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
/// (a literal for sized types, `self.data.ndim()` for ND). Includes slice-to-array
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
                self.check_extrapolate(&self.extrapolate)?;
                self.data.validate()?;
                self.validate_strategy()?;
                Ok(())
            }

            fn interpolate(&self, point: &[D::Elem]) -> Result<D::Elem, InterpolateError> {
                let point: &[D::Elem; N] = point
                    .try_into()
                    .map_err(|_| InterpolateError::PointLength(N))?;
                self.interpolate(point)
            }

            fn interpolate_fast(&self, point: &[D::Elem]) -> D::Elem {
                let point: &[D::Elem; N] = point
                    .try_into()
                    .expect("interpolate_fast: point length mismatch");
                self.interpolate_fast(point)
            }

            fn batch_interpolate(
                &self,
                points: &[&[D::Elem]],
            ) -> Result<Vec<D::Elem>, InterpolateError> {
                let points: Vec<[D::Elem; N]> = points
                    .iter()
                    .map(|&point| {
                        <&[D::Elem; N]>::try_from(point)
                            .map(|arr| *arr)
                            .map_err(|_| InterpolateError::PointLength(N))
                    })
                    .collect::<Result<_, _>>()?;
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
                self.check_extrapolate(&extrapolate)?;
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

/// Generates the [`DynInterpolator`] impl shared by `Interp1D`/`2D`/`3D`/`ND`'s owned
/// variants. Bounded on `Float + Euclid` (rather than the `Num + PartialOrd + Euclid +
/// Copy` the [`Interpolator`] impls use) because that's what [`DynInterpolator`] itself
/// requires transitively; only owned types implement it, since `as_any` requires
/// `Self: 'static`.
macro_rules! dyn_interpolator_impl {
    ($InterpTypeOwned:ident, $Strategy:ident) => {
        impl<T, S> DynInterpolator<T> for $InterpTypeOwned<T, S>
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
pub(crate) use dyn_interpolator_impl;

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
/// `set_strategy` for the strategy enum, and `DynInterpolator`.
macro_rules! interp_trait_impls {
    ($InterpType:ident, $InterpTypeOwned:ident, $InterpData:ident, $Strategy:ident, $StrategyEnum:path, $N:expr) => {
        partialeq_impl!($InterpType, $InterpData, $Strategy);
        extrapolate_impl!($InterpType, $Strategy);
        interpolator_trait_impl!($InterpType, $Strategy, $N);
        set_strategy_box_impl!($InterpType, $Strategy);
        set_strategy_enum_impl!($InterpType, $StrategyEnum);
        dyn_interpolator_impl!($InterpTypeOwned, $Strategy);
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
        batch_interpolate_impl!();
        /// Batched [`Self::interpolate_fast`], for use in hot loops where the caller
        /// has already checked bounds or knows that extrapolation handling is not
        /// needed.
        pub fn batch_interpolate_fast(&self, points: &[[D::Elem; N]]) -> Vec<D::Elem> {
            self.strategy.batch_interpolate_fast(&self.data, points)
        }
        strategy_accessors_impl!($Strategy);
        view_into_owned_impl!($InterpType, $Strategy, $Viewed, $Owned);
    };
}
pub(crate) use interp_inherent_methods;
