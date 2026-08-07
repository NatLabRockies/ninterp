//! Module for all interpolation types.

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

macro_rules! partialeq_impl {
    ($InterpType:ident, $Data:ident, $Strategy:ident) => {
        impl<D, S> PartialEq for $InterpType<D, S>
        where
            D: Data + RawDataClone + Clone,
            D::Elem: PartialEq + Debug,
            S: $Strategy<D> + Clone + PartialEq,
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
