//! 2-dimensional interpolation

use super::*;

mod strategies;
#[cfg(test)]
mod tests;

const N: usize = 2;

/// [`InterpData`] for 2-D data.
pub type InterpData2D<D> = InterpData<D, N>;
/// [`InterpData2D`] that views data.
pub type InterpData2DViewed<T> = InterpData2D<ViewRepr<T>>;
/// [`InterpData2D`] that owns data.
pub type InterpData2DOwned<T> = InterpData2D<OwnedRepr<T>>;

impl<D> InterpData2D<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialOrd + Debug,
{
    /// Construct and validate a new [`InterpData2D`].
    pub fn new(
        x: ArrayBase<D, Ix1>,
        y: ArrayBase<D, Ix1>,
        f_xy: ArrayBase<D, Ix2>,
    ) -> Result<Self, ValidateError> {
        let data = Self {
            grid: [x, y],
            values: f_xy,
        };
        data.validate()?;
        Ok(data)
    }
}

/// 2-D interpolator
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "
            D::Elem: Serialize,
            S: Serialize,
        ",
        deserialize = "
            D: DataOwned,
            D::Elem: Deserialize<'de>,
            S: Deserialize<'de>,
        "
    ))
)]
pub struct Interp2D<D, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
    S: Clone,
{
    /// Interpolator data.
    pub data: InterpData2D<D>,
    /// Interpolation strategy.
    pub strategy: S,
    /// Extrapolation setting.
    pub extrapolate: Extrapolate<D::Elem>,
}
/// [`Interp2D`] that views data.
pub type Interp2DViewed<T, S> = Interp2D<ViewRepr<T>, S>;
/// [`Interp2D`] that owns data.
pub type Interp2DOwned<T, S> = Interp2D<OwnedRepr<T>, S>;

impl<D, S> Interp2D<D, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
    S: Strategy2D<D> + Clone,
{
    /// Construct and validate a 2-D interpolator.
    ///
    /// # Example:
    /// ```
    /// use ndarray::prelude::*;
    /// use ninterp::prelude::*;
    /// // f(x, y) = 0.2 * x + 0.4 * y
    /// // type annotation for clarity
    /// let interp: Interp2DOwned<f64, _> = Interp2D::new(
    ///     // x
    ///     array![0., 1., 2.], // x0, x1, x2
    ///     // y
    ///     array![0., 1., 2., 3.], // y0, y1, y2, y3
    ///     // f(x, y)
    ///     array![
    ///         [0.0, 0.4, 0.8, 1.2], // f(x0, y0), f(x0, y1), f(x0, y2), f(x0, y3)
    ///         [0.2, 0.6, 1.0, 1.4], // f(x1, y0), f(x1, y1), f(x1, y2), f(x1, y3)
    ///         [0.4, 0.8, 1.2, 1.6], // f(x2, y0), f(x2, y1), f(x2, y2), f(x2, y3)
    ///     ],
    ///     strategy::Linear,   // strategy mod is exposed via `use ndarray::prelude::*;`
    ///     Extrapolate::Clamp, // restrict point within grid bounds
    /// )
    /// .unwrap();
    /// assert_eq!(interp.interpolate(&[1.5, 1.5]).unwrap(), 0.9);
    /// assert_eq!(
    ///     interp.interpolate(&[-1., 3.5]).unwrap(),
    ///     interp.interpolate(&[0., 3.]).unwrap()
    /// ); // point is restricted to within grid bounds
    /// ```
    pub fn new(
        x: ArrayBase<D, Ix1>,
        y: ArrayBase<D, Ix1>,
        f_xy: ArrayBase<D, Ix2>,
        strategy: S,
        extrapolate: Extrapolate<D::Elem>,
    ) -> Result<Self, ValidateError>
    where
        D::Elem: PartialOrd,
    {
        let mut interpolator = Self {
            data: InterpData2D::new(x, y, f_xy)?,
            strategy,
            extrapolate,
        };
        interpolator.check_extrapolate(&interpolator.extrapolate)?;
        interpolator.validate_strategy()?;
        interpolator.init_strategy()?;
        Ok(interpolator)
    }

    interp_inherent_methods!(
        Interp2D,
        Strategy2D,
        Interp2DViewed<&D::Elem, S>,
        Interp2DOwned<D::Elem, S>
    );
}

interp_trait_impls!(
    Interp2D,
    Interp2DOwned,
    InterpData2D,
    Strategy2D,
    strategy::enums::Strategy2DEnum,
    N
);
