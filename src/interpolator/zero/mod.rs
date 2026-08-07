//! 0-dimensional interpolation

use super::*;

const N: usize = 0;

/// 0-D interpolator
// #[repr(transparent)]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Interp0D<T>(pub T);
impl<T> Interp0D<T>
where
    T: PartialEq + Debug,
{
    /// Construct a constant-value 'interpolator'.
    ///
    /// # Example:
    /// ```
    /// use ninterp::prelude::*;
    /// let const_value = 0.5;
    /// let interp = Interp0D::new(const_value);
    /// assert_eq!(
    ///     interp.interpolate(&[]).unwrap(), // an empty point is required for 0-D
    ///     const_value
    /// );
    /// ```
    pub fn new(value: T) -> Self {
        Self(value)
    }
}
impl<T> Interpolator<T> for Interp0D<T>
where
    T: Clone + Debug,
{
    /// Returns `0`.
    #[inline]
    fn ndim(&self) -> usize {
        N
    }

    /// Returns `Ok(())`.
    #[inline]
    fn validate(&self) -> Result<(), ValidateError> {
        Ok(())
    }

    /// Returns the contained value [`Interp0D::0`].
    ///
    /// Errors if `!point.is_empty()`.
    fn interpolate(&self, point: &[T]) -> Result<T, InterpolateError> {
        if !point.is_empty() {
            return Err(InterpolateError::PointLength(N));
        }
        Ok(self.0.clone())
    }

    /// Returns `Ok(())`.
    #[inline]
    fn set_extrapolate(&mut self, _extrapolate: Extrapolate<T>) -> Result<(), ValidateError> {
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    #[cfg(feature = "serde")]
    fn test_serde() {
        let interp = Interp0D::new(0.5);

        let ser = serde_json::to_string(&interp).unwrap();
        let de: Interp0D<_> = serde_json::from_str(&ser).unwrap();
        assert_eq!(interp, de);
    }
}
