//! Object-safe, downcastable interpolator trait for storing heterogeneous
//! interpolators behind `Box<dyn DynInterpolator<T>>`.

use std::any::Any;

use super::*;

/// A `Send + Sync`, downcastable counterpart to [`Interpolator<T>`], implemented for
/// owned `Interp1D`/`2D`/`3D`/`ND` types.
///
/// [`Interpolator<T>`] can't provide `as_any` itself: that requires `Self: 'static`,
/// which the borrowed `Interp*Viewed` types can't satisfy. `Send + Sync` is likewise
/// per-impl rather than on `Interpolator<T>`, since a custom strategy may hold
/// non-thread-safe state.
pub trait DynInterpolator<T>: Send + Sync {
    /// Interpolate at the supplied point.
    fn interpolate_slice(&self, point: &[T]) -> Result<T, InterpolateError>;

    /// Interpolate at each of several points, sharing one grid across all of them.
    ///
    /// Default loops [`DynInterpolator::interpolate_slice`]; the blanket impls
    /// override it to reuse the concrete type's inherent `batch_interpolate` instead
    /// of paying this trait's vtable dispatch once per point.
    fn batch_interpolate_slice(&self, points: &[&[T]]) -> Result<Vec<T>, InterpolateError> {
        points
            .iter()
            .map(|point| self.interpolate_slice(point))
            .collect()
    }

    /// Downcast to the concrete interpolator type.
    fn as_any(&self) -> &dyn Any;
}
