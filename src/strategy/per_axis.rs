//! Shared per-axis configuration wrapper for strategies whose settings can vary by grid
//! dimension: see [`Step`] and [`CubicC2`].

use super::*;

/// Per-axis strategy configuration: either one value broadcast to every grid dimension, or
/// a distinct value for each.
///
/// [`Step`] and [`CubicC2`] both wrap this instead of a bare `Vec`, so "one setting for
/// the whole interpolator" and "one setting per axis" are distinguished by the type itself
/// instead of by a `Vec`'s length (a `len() == 1` vec used to mean "broadcast" even when a
/// genuinely 1-D interpolator wanted a single per-axis entry -- those two cases are now
/// different variants).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(feature = "serde", serde(untagged))]
pub enum PerAxis<T> {
    /// Same value applied to every dimension.
    Broadcast(T),
    /// One value per dimension. Length must equal the interpolator's dimensionality;
    /// checked by each strategy's `validate` via `validate_len`.
    Axes(Vec<T>),
}

impl<T> From<T> for PerAxis<T> {
    /// Broadcasts `value` to every dimension.
    fn from(value: T) -> Self {
        PerAxis::Broadcast(value)
    }
}

impl<T> PerAxis<T> {
    /// Returns the value for dimension `dim`. `Broadcast` returns the same value for every
    /// `dim`; `Axes` indexes unchecked, assuming [`validate_len`](Self::validate_len) has
    /// already confirmed the count matches the interpolator's dimensionality.
    pub(crate) fn get(&self, dim: usize) -> &T {
        match self {
            PerAxis::Broadcast(value) => value,
            PerAxis::Axes(values) => &values[dim],
        }
    }

    /// Returns this configuration as a slice: length 1 for `Broadcast`, length N for
    /// `Axes`. Bridges to the numeric code in [`cubic`](super::cubic), which already treats
    /// a length-1 slice as "reuse this for every axis" and is left untouched by this
    /// wrapper.
    pub(crate) fn as_slice(&self) -> &[T] {
        match self {
            PerAxis::Broadcast(value) => core::slice::from_ref(value),
            PerAxis::Axes(values) => values,
        }
    }

    /// Checks the stored count against an interpolator's dimensionality: `Broadcast` is
    /// always valid, `Axes` must have exactly `ndim` entries. `label` names the strategy
    /// (e.g. `"Step strategy"`) and `noun` names what's being counted (e.g.
    /// `"directions"`).
    pub(crate) fn validate_len(
        &self,
        ndim: usize,
        label: &str,
        noun: &str,
    ) -> Result<(), ValidateError> {
        let PerAxis::Axes(values) = self else {
            return Ok(());
        };
        let found = values.len();
        if found == ndim {
            return Ok(());
        }
        let expected = if ndim == 1 {
            "1".to_string()
        } else {
            format!("1 or {ndim}")
        };
        Err(ValidateError::Other(format!(
            "{label} has {found} {noun} but interpolator is {ndim}-D (expected {expected})"
        )))
    }
}
