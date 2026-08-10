//! 1-dimensional interpolation

use super::*;

mod strategies;
#[cfg(test)]
mod tests;

const N: usize = 1;

/// [`InterpData`] for 1-D data.
pub type InterpData1D<D> = InterpData<D, N>;
/// [`InterpData1D`] that views data.
pub type InterpData1DViewed<T> = InterpData1D<ViewRepr<T>>;
/// [`InterpData1D`] that owns data.
pub type InterpData1DOwned<T> = InterpData1D<OwnedRepr<T>>;

impl<D> InterpData1D<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialOrd + Debug,
{
    /// Construct and validate a new [`InterpData1D`].
    pub fn new(x: ArrayBase<D, Ix1>, f_x: ArrayBase<D, Ix1>) -> Result<Self, ValidateError> {
        let data = Self {
            grid: [x],
            values: f_x,
        };
        data.validate()?;
        Ok(data)
    }
}

/// 1-D interpolator
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
pub struct Interp1D<D, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
    S: Clone,
{
    /// Interpolator data.
    pub data: InterpData1D<D>,
    /// Interpolation strategy.
    pub strategy: S,
    /// Extrapolation setting.
    pub extrapolate: Extrapolate<D::Elem>,
}
/// [`Interp1D`] that views data.
pub type Interp1DViewed<T, S> = Interp1D<ViewRepr<T>, S>;
/// [`Interp1D`] that owns data.
pub type Interp1DOwned<T, S> = Interp1D<OwnedRepr<T>, S>;

impl<D, S> Interp1D<D, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
    S: Strategy1D<D> + Clone,
{
    /// Instantiate one-dimensional interpolator.
    ///
    /// # Example:
    /// ```
    /// use ndarray::prelude::*;
    /// use ninterp::prelude::*;
    /// // f(x) = 0.4 * x
    /// // type annotation for clarity
    /// let interp: Interp1DOwned<f64, _> = Interp1D::new(
    ///     // x
    ///     array![0., 1., 2.], // x0, x1, x2
    ///     // f(x)
    ///     array![0.0, 0.4, 0.8], // f(x0), f(x1), f(x2)
    ///     strategy::Linear,      // strategy mod is exposed via `use ndarray::prelude::*;`
    ///     Extrapolate::Enable,
    /// )
    /// .unwrap();
    /// assert_eq!(interp.interpolate(&[1.4]).unwrap(), 0.56);
    /// assert_eq!(interp.interpolate(&[3.6]).unwrap(), 1.44);
    /// ```
    pub fn new(
        x: ArrayBase<D, Ix1>,
        f_x: ArrayBase<D, Ix1>,
        strategy: S,
        extrapolate: Extrapolate<D::Elem>,
    ) -> Result<Self, ValidateError>
    where
        D::Elem: PartialOrd,
    {
        let mut interpolator = Self {
            data: InterpData1D::new(x, f_x)?,
            strategy,
            extrapolate,
        };
        interpolator.check_extrapolate(&interpolator.extrapolate)?;
        interpolator.validate_strategy()?;
        interpolator.init_strategy()?;
        Ok(interpolator)
    }

    interp_inherent_methods!(
        Interp1D,
        Strategy1D,
        Interp1DViewed<&D::Elem, S>,
        Interp1DOwned<D::Elem, S>
    );
}

interp_trait_impls!(
    Interp1D,
    Interp1DOwned,
    InterpData1D,
    Strategy1D,
    strategy::enums::Strategy1DEnum,
    N
);
