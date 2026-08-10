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

extrapolate_impl!(Interp1D, Strategy1D);
partialeq_impl!(Interp1D, InterpData1D, Strategy1D);
#[cfg(feature = "serde")]
serialize_nested_impl!(Interp1D, InterpData1D, Strategy1D);

impl<D, S> Interp1D<D, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
    S: Strategy1D<D> + Clone,
{
    /// Re-run the strategy's [`Strategy1D::validate`] against the current data.
    ///
    /// `new`, `set_strategy`, and [`Interpolator::validate`] already call this
    /// internally, so this is only needed after mutating the public `data`/`strategy`
    /// fields directly.
    pub fn validate_strategy(&self) -> Result<(), ValidateError> {
        self.strategy.validate(&self.data)
    }

    /// Re-run the strategy's [`Strategy1D::init`] against the current data.
    ///
    /// `new` and `set_strategy` already call this internally, so this is only needed
    /// after bypassing them: mutating the public `data`/`strategy` fields directly, or
    /// deserializing an interpolator whose strategy skips its cached state from
    /// serialization (e.g. via `#[serde(skip)]`, to avoid bloating the wire format with
    /// a large derived array). `Deserialize` does not call `init`; if the cached state
    /// is instead stored in ordinary serialized fields, it comes back as-is and this
    /// isn't needed.
    pub fn init_strategy(&mut self) -> Result<(), ValidateError> {
        self.strategy.init(&self.data)
    }

    /// Interpolate without bounds/extrapolation checks, for use in hot loops where the
    /// caller has already checked bounds or knows that extrapolation handling is not needed.
    pub fn interpolate_fast(&self, point: &[D::Elem; 1]) -> D::Elem {
        self.strategy.interpolate_fast(&self.data, point)
    }

    batch_interpolate_fast_impl!();
}

impl<D, S> Interp1D<D, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialOrd + Debug,
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
    ) -> Result<Self, ValidateError> {
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

    /// Return an interpolator with viewed data.
    pub fn view(&self) -> Interp1DViewed<&D::Elem, S>
    where
        S: for<'a> Strategy1D<ViewRepr<&'a D::Elem>>,
        D::Elem: Clone,
    {
        Interp1DViewed {
            data: self.data.view(),
            strategy: self.strategy.clone(),
            extrapolate: self.extrapolate.clone(),
        }
    }

    /// Turn the interpolator into an [`Interp1DOwned`], cloning the array elements if necessary.
    pub fn into_owned(self) -> Interp1DOwned<D::Elem, S>
    where
        S: Strategy1D<OwnedRepr<D::Elem>>,
        D::Elem: Clone,
    {
        Interp1DOwned {
            data: self.data.into_owned(),
            strategy: self.strategy.clone(),
            extrapolate: self.extrapolate.clone(),
        }
    }
}

impl<D, S> Interp1D<D, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Num + PartialOrd + Euclid + Copy + Debug,
    S: Strategy1D<D> + Clone,
{
    interpolate_impl!();
    batch_interpolate_impl!();
}

impl<D, S> Interpolator<D::Elem> for Interp1D<D, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Num + PartialOrd + Euclid + Copy + Debug,
    S: Strategy1D<D> + Clone,
{
    /// Returns `1`.
    #[inline]
    fn ndim(&self) -> usize {
        N
    }

    fn validate(&self) -> Result<(), ValidateError> {
        self.check_extrapolate(&self.extrapolate)?;
        self.data.validate()?;
        self.validate_strategy()?;
        Ok(())
    }

    sized_interpolate_impl!();

    fn set_extrapolate(&mut self, extrapolate: Extrapolate<D::Elem>) -> Result<(), ValidateError> {
        self.check_extrapolate(&extrapolate)?;
        self.extrapolate = extrapolate;
        Ok(())
    }
}

set_strategy_box_impl!(Interp1D, Strategy1D);
set_strategy_enum_impl!(Interp1D, strategy::enums::Strategy1DEnum);
dyn_interpolator_impl!(Interp1DOwned, Strategy1D);
