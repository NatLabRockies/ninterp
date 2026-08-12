//! 1-dimensional interpolation

use super::*;

mod strategies;
#[cfg(test)]
mod tests;

const N: usize = 1;

/// Generic (base) form for 1-D data; parameterized by data representation.
pub type InterpData1DBase<D> = InterpDataBase<D, N>;
/// Owned data variant for 1-D data (see [`InterpData1DBase`] for the generic form).
pub type InterpData1D<T> = InterpData1DBase<OwnedRepr<T>>;
/// Viewed data variant for 1-D data (see [`InterpData1DBase`] for the generic form).
pub type InterpData1DView<T> = InterpData1DBase<ViewRepr<T>>;

impl<D> InterpData1DBase<D>
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
pub struct Interp1DBase<D, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
    S: Clone,
{
    /// Interpolator data.
    pub data: InterpData1DBase<D>,
    /// Interpolation strategy.
    pub strategy: S,
    /// Extrapolation setting.
    pub extrapolate: Extrapolate<D::Elem>,
}
/// Owned interpolator variant (see [`Interp1DBase`] for the generic form).
pub type Interp1D<T, S> = Interp1DBase<OwnedRepr<T>, S>;
/// Viewed interpolator variant (see [`Interp1DBase`] for the generic form).
pub type Interp1DView<T, S> = Interp1DBase<ViewRepr<T>, S>;

impl<D, S> Interp1DBase<D, S>
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
    /// let interp = Interp1D::new(
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
            data: InterpData1DBase::new(x, f_x)?,
            strategy,
            extrapolate,
        };
        interpolator.validate_extrapolate(&interpolator.extrapolate)?;
        interpolator.validate_strategy()?;
        interpolator.init_strategy()?;
        Ok(interpolator)
    }

    interp_inherent_methods!(
        Interp1DBase,
        Strategy1D,
        Interp1DView<&D::Elem, S>,
        Interp1D<D::Elem, S>
    );
}

interp_trait_impls!(
    Interp1DBase,
    Interp1D,
    InterpData1DBase,
    Strategy1D,
    strategy::enums::Strategy1DEnum,
    strategy::enums::Strategy1DEnum<D::Elem>,
    N
);
