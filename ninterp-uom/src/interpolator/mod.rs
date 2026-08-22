//! Interpolator types, one module per dimensionality (mirroring `ninterp`'s own
//! `interpolator::{one, two, three}` layout).
//!
//! Covers **0-D through 3-D**. Three things core `ninterp` has are intentionally *not*
//! mirrored here, all for the same underlying reason: each needs one element type shared
//! across every grid axis (and, for the last one, across every dimensionality too), which
//! per-axis `uom` units don't provide:
//! - **`N`-D** (`ninterp::interpolator::InterpND`): a runtime-variable axis count can't
//!   carry a per-axis Rust type, and the only fallback (one shared unit for every axis)
//!   wasn't judged worth shipping.
//! - **The dyn-compatible `Interpolator<T>` trait and `AnyInterpolator<T>`**: both are
//!   built on `interpolate(&[T]) -> T`, one `T` for every axis *and* the output.
//! - **`InterpolatorEnumBase`** (`ninterp`'s runtime-swappable `Interp0D|1D|2D|3D|ND`
//!   enum): same shape problem, one level up.

use super::*;

pub mod one;
pub mod three;
pub mod two;
pub mod zero;

pub use one::{UomInterp1D, UomInterp1DBase, UomInterp1DView};
pub use three::{UomInterp3D, UomInterp3DBase, UomInterp3DView};
pub use two::{UomInterp2D, UomInterp2DBase, UomInterp2DView};
pub use zero::UomInterp0D;

/// Generates the inherent methods shared by every `UomInterp{1,2,3}DBase`, beyond `new`
/// and `interpolate` (which stay hand-written per dimensionality: grid construction and
/// the point signature genuinely differ in arity). Invoked from inside the same
/// `impl<D, $($Qi),+, Qv, S> ... where ...` block that hosts `interpolate`, so it inherits
/// that block's bounds.
///
/// `$Qi`/`$qi` are parallel axis type/value identifier lists (`Qx` / `x` for 1-D; `Qx, Qy`
/// / `x, y` for 2-D; etc). This falls out correctly for 1-D too, without a separate case:
/// `($($Qi),+)` with one identifier is just `Qx` (Rust parens around a single
/// type/pattern are grouping, not a 1-tuple), matching the existing bare-`Qx` point type
/// rather than an awkward `(Qx,)`.
macro_rules! uom_interp_common_methods {
    (
        $Base:ident,
        $Strategy:ident,
        $Viewed:ty,
        $Owned:ty,
        ($($Qi:ident),+ $(,)?),
        ($($qi:ident),+ $(,)?)
    ) => {
        /// Interpolate without bounds/extrapolation checks, for use in hot loops where
        /// the caller has already checked bounds or knows that extrapolation handling
        /// is not needed.
        pub fn interpolate_fast(&self, $($qi: $Qi),+) -> Qv {
            Qv::from_base(self.inner.interpolate_fast(&[$($qi.to_base()),+]))
        }

        /// Interpolate at each of several points, sharing one grid across all of them.
        #[allow(unused_parens)]
        pub fn batch_interpolate(
            &self,
            points: &[($($Qi),+)],
        ) -> Result<Vec<Qv>, InterpolateError> {
            let raw_points: Vec<_> = points
                .iter()
                .map(|&($($qi),+)| [$($qi.to_base()),+])
                .collect();
            let out = self.inner.batch_interpolate(&raw_points)?;
            // SAFETY: `Qv` is `#[repr(transparent)]` over `D::Elem` (same size/align,
            // no `Drop` of its own - see `one` module docs for the full argument), so a
            // `Vec<D::Elem>` is layout-identical to a `Vec<Qv>`; re-tagging the element
            // type in place is sound and avoids a second allocation/copy.
            Ok(unsafe { mem::transmute::<Vec<D::Elem>, Vec<Qv>>(out) })
        }

        /// Interpolate at each of several points, writing results into `out` instead of
        /// allocating.
        #[allow(unused_parens)]
        pub fn batch_interpolate_into(
            &self,
            points: &[($($Qi),+)],
            out: &mut [Qv],
        ) -> Result<(), InterpolateError> {
            let raw_points: Vec<_> = points
                .iter()
                .map(|&($($qi),+)| [$($qi.to_base()),+])
                .collect();
            // SAFETY: same reasoning as `batch_interpolate` above, applied to a
            // borrowed mutable slice instead of an owned `Vec`.
            let raw_out: &mut [D::Elem] = unsafe { mem::transmute::<&mut [Qv], _>(out) };
            self.inner.batch_interpolate_into(&raw_points, raw_out)
        }

        /// Unchecked batched [`Self::interpolate_fast`], assuming every point is valid.
        #[allow(unused_parens)]
        pub fn batch_interpolate_fast(&self, points: &[($($Qi),+)]) -> Vec<Qv> {
            let raw_points: Vec<_> = points
                .iter()
                .map(|&($($qi),+)| [$($qi.to_base()),+])
                .collect();
            let out = self.inner.batch_interpolate_fast(&raw_points);
            // SAFETY: see `batch_interpolate`.
            unsafe { mem::transmute::<Vec<D::Elem>, Vec<Qv>>(out) }
        }

        /// Unchecked [`Self::batch_interpolate_into`], assuming every point is valid.
        #[allow(unused_parens)]
        pub fn batch_interpolate_fast_into(&self, points: &[($($Qi),+)], out: &mut [Qv]) {
            let raw_points: Vec<_> = points
                .iter()
                .map(|&($($qi),+)| [$($qi.to_base()),+])
                .collect();
            // SAFETY: see `batch_interpolate_into`.
            let raw_out: &mut [D::Elem] = unsafe { mem::transmute::<&mut [Qv], _>(out) };
            self.inner.batch_interpolate_fast_into(&raw_points, raw_out)
        }

        /// Re-run the strategy's `validate` against the current data.
        ///
        /// `new`, `set_strategy`, and [`Self::validate`] already call this internally,
        /// so this is only needed after mutating `self.inner`'s `data`/`strategy`
        /// fields directly.
        pub fn validate_strategy(&self) -> Result<(), ValidateError> {
            self.inner.validate_strategy()
        }

        /// Re-run the strategy's `init` against the current data.
        ///
        /// `new` and `set_strategy` already call this internally, so this is only
        /// needed after bypassing them via `self.inner`.
        pub fn init_strategy(&mut self) -> Result<(), ValidateError> {
            self.inner.init_strategy()
        }

        /// Check that `extrapolate` is applicable to the current strategy. Takes the
        /// setting as an argument (rather than reading `self`) so a candidate can be
        /// vetted before storing it.
        ///
        /// Stays raw `D::Elem`, not uom-typed: core's own `Extrapolate<T>` doesn't yet
        /// distinguish grid-coordinate values from interpolated-output values, so
        /// there's no single `Qx`/`Qv` this could soundly be.
        pub fn validate_extrapolate(
            &self,
            extrapolate: &Extrapolate<D::Elem>,
        ) -> Result<(), ValidateError> {
            self.inner.validate_extrapolate(extrapolate)
        }

        /// Update `extrapolate` at runtime, checking it's applicable to the current
        /// strategy first.
        pub fn set_extrapolate(
            &mut self,
            extrapolate: Extrapolate<D::Elem>,
        ) -> Result<(), ValidateError> {
            self.inner.set_extrapolate(extrapolate)
        }

        /// Interpolator dimensionality.
        #[inline]
        pub fn ndim(&self) -> usize {
            self.inner.ndim()
        }

        /// Validate interpolator data.
        pub fn validate(&self) -> Result<(), ValidateError> {
            self.inner.validate()
        }

        /// Return an interpolator with viewed data.
        pub fn view(&self) -> $Viewed
        where
            S: for<'a> $Strategy<ViewRepr<&'a D::Elem>>,
            D::Elem: Clone,
        {
            $Base {
                inner: self.inner.view(),
                _units: PhantomData,
            }
        }

        /// Turn the interpolator into an owned variant, cloning the array elements if
        /// necessary.
        pub fn into_owned(self) -> $Owned
        where
            S: $Strategy<OwnedRepr<D::Elem>>,
            D::Elem: Clone,
        {
            $Base {
                inner: self.inner.into_owned(),
                _units: PhantomData,
            }
        }
    };
}
pub(crate) use uom_interp_common_methods;

/// Generates a hand-written `PartialEq` (not derived: see [`uom_interp_common_methods`]'s
/// neighbors below for why) forwarding straight to `self.inner`.
macro_rules! uom_interp_partial_eq {
    ($Base:ident, $Inner:ident, ($($Qi:ident),+ $(,)?)) => {
        impl<D, $($Qi,)* Qv, S> PartialEq for $Base<D, $($Qi,)* Qv, S>
        where
            D: Data + RawDataClone + Clone,
            D::Elem: PartialEq + Debug + Clone,
            $($Qi: BaseUnit<D::Elem>,)*
            Qv: BaseUnit<D::Elem>,
            S: Clone,
            $Inner<D, S>: PartialEq,
        {
            fn eq(&self, other: &Self) -> bool {
                self.inner == other.inner
            }
        }
    };
}
pub(crate) use uom_interp_partial_eq;

/// Generates the `Box<dyn $Strategy<D>>`-backed inherent `set_strategy`.
macro_rules! uom_interp_set_strategy_box {
    ($Base:ident, $Strategy:ident, ($($Qi:ident),+ $(,)?)) => {
        impl<D, $($Qi,)* Qv> $Base<D, $($Qi,)* Qv, Box<dyn $Strategy<D>>>
        where
            D: Data + RawDataClone + Clone,
            D::Elem: PartialEq + Debug + Clone,
            $($Qi: BaseUnit<D::Elem>,)*
            Qv: BaseUnit<D::Elem>,
        {
            /// Update strategy at runtime, calling `init` on the new strategy against
            /// the current data.
            ///
            /// To swap in a strategy without re-running `init`, mutate `self.inner`'s
            /// `strategy` field directly instead.
            pub fn set_strategy(
                &mut self,
                strategy: Box<dyn $Strategy<D>>,
            ) -> Result<(), ValidateError> {
                self.inner.set_strategy(strategy)
            }
        }
    };
}
pub(crate) use uom_interp_set_strategy_box;

/// Generates the `$StrategyEnum`-backed inherent `set_strategy`.
macro_rules! uom_interp_set_strategy_enum {
    ($Base:ident, $StrategyEnum:ident, ($($Qi:ident),+ $(,)?)) => {
        impl<D, $($Qi,)* Qv> $Base<D, $($Qi,)* Qv, $StrategyEnum<D::Elem>>
        where
            D: Data + RawDataClone + Clone,
            D::Elem: Float + Debug + Clone,
            $($Qi: BaseUnit<D::Elem>,)*
            Qv: BaseUnit<D::Elem>,
        {
            /// Update strategy at runtime, calling `init` on the new strategy against
            /// the current data.
            pub fn set_strategy(
                &mut self,
                strategy: impl Into<$StrategyEnum<D::Elem>>,
            ) -> Result<(), ValidateError> {
                self.inner.set_strategy(strategy)
            }
        }
    };
}
pub(crate) use uom_interp_set_strategy_enum;

/// Generates `Serialize`/`Deserialize`, gated on `feature = "serde"`, that delegate
/// straight to `self.inner`'s existing (core-provided) serde impl. The wire format is
/// identical to a bare `ninterp` interpolator - units are compile-time-only via
/// `Qx`/`Qy`/`Qz`/`Qv`, never serialized.
macro_rules! uom_interp_serde {
    ($Base:ident, $Inner:ident, ($($Qi:ident),+ $(,)?)) => {
        #[cfg(feature = "serde")]
        impl<D, $($Qi,)* Qv, S> Serialize for $Base<D, $($Qi,)* Qv, S>
        where
            D: Data + RawDataClone + Clone,
            D::Elem: PartialEq + Debug + Clone,
            $($Qi: BaseUnit<D::Elem>,)*
            Qv: BaseUnit<D::Elem>,
            S: Clone,
            $Inner<D, S>: Serialize,
        {
            fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
            where
                Ser: Serializer,
            {
                self.inner.serialize(serializer)
            }
        }

        #[cfg(feature = "serde")]
        impl<'de, D, $($Qi,)* Qv, S> Deserialize<'de> for $Base<D, $($Qi,)* Qv, S>
        where
            D: Data + RawDataClone + Clone,
            D::Elem: PartialEq + Debug + Clone,
            $($Qi: BaseUnit<D::Elem>,)*
            Qv: BaseUnit<D::Elem>,
            S: Clone,
            $Inner<D, S>: Deserialize<'de>,
        {
            fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
            where
                De: Deserializer<'de>,
            {
                Ok(Self {
                    inner: $Inner::deserialize(deserializer)?,
                    _units: PhantomData,
                })
            }
        }
    };
}
pub(crate) use uom_interp_serde;
