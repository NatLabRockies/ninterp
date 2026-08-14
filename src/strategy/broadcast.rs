//! Shared per-axis configuration wrapper for strategies whose settings can vary by grid
//! dimension: see [`Step`] and [`CubicC2`].

use super::*;

/// Per-axis strategy configuration: either one value broadcast to every grid dimension, or
/// a distinct value for each.
///
/// [`Step`] and [`CubicC2`] both wrap this instead of a bare `Vec`, so "one setting for
/// the whole interpolator" and "one setting per axis" are distinguished by the type itself
/// instead of by a `Vec`'s length. A `len() == 1` vec used to mean "broadcast" even when a
/// genuinely 1-D interpolator wanted a single per-axis entry; those two cases are now
/// different variants.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(feature = "serde", serde(untagged))]
pub enum Broadcastable<T> {
    /// Same value applied to every dimension.
    Broadcast(T),
    /// One value per dimension. Length must equal the interpolator's dimensionality;
    /// checked by each strategy's `validate` via `validate_len`.
    Each(Vec<T>),
}

impl<T> From<T> for Broadcastable<T> {
    /// Broadcasts `value` to every dimension.
    fn from(value: T) -> Self {
        Broadcastable::Broadcast(value)
    }
}

impl<T> Broadcastable<T> {
    /// Checks the stored count against an interpolator's dimensionality: `Broadcast` is
    /// always valid, `Each` must have exactly `ndim` entries. `label` names the strategy
    /// (e.g. `"Step"`) and `noun` names what's being counted (e.g. `"directions"`).
    ///
    /// Custom strategies built on `Broadcastable` should call this from `validate`, before
    /// indexing via [`Index`](core::ops::Index), the same way [`Step`] and [`CubicC2`] do.
    pub fn validate_len(
        &self,
        ndim: usize,
        label: &'static str,
        noun: &'static str,
    ) -> Result<(), ValidateError> {
        let Broadcastable::Each(values) = self else {
            return Ok(());
        };
        let found = values.len();
        if found == ndim {
            return Ok(());
        }
        Err(ValidateError::PerAxisLen {
            label,
            noun,
            ndim,
            found,
        })
    }
}

impl<T> core::ops::Index<usize> for Broadcastable<T> {
    type Output = T;

    /// Returns the value for dimension `dim`. `Broadcast` returns the same value for every
    /// `dim`; `Each` indexes normally (panics if `dim` is out of range). Call
    /// [`validate_len`](Self::validate_len) first to confirm the count matches the
    /// interpolator's dimensionality.
    fn index(&self, dim: usize) -> &T {
        match self {
            Broadcastable::Broadcast(value) => value,
            Broadcastable::Each(values) => &values[dim],
        }
    }
}
