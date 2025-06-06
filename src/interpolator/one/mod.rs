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
    S: Strategy1D<D> + Clone,
{
    /// Interpolator data.
    pub data: InterpData1D<D>,
    /// Interpolation strategy.
    pub strategy: S,
    /// Extrapolation setting.
    #[cfg_attr(feature = "serde", serde(default))]
    pub extrapolate: Extrapolate<D::Elem>,
}
/// [`Interp1D`] that views data.
pub type Interp1DViewed<T, S> = Interp1D<ViewRepr<T>, S>;
/// [`Interp1D`] that owns data.
pub type Interp1DOwned<T, S> = Interp1D<OwnedRepr<T>, S>;

extrapolate_impl!(Interp1D, Strategy1D);
partialeq_impl!(Interp1D, InterpData1D, Strategy1D);

impl<D, S> Interp1D<D, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialOrd + Debug,
    S: Strategy1D<D> + Clone,
{
    /// Instantiate one-dimensional interpolator.
    ///
    /// Applicable interpolation strategies:
    /// - [`strategy::Linear`]
    /// - [`strategy::Nearest`]
    /// - [`strategy::LeftNearest`]
    /// - [`strategy::RightNearest`]
    ///
    /// [`Extrapolate::Enable`] is valid for [`strategy::Linear`]
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
    ///     strategy::Linear, // strategy mod is exposed via `use ndarray::prelude::*;`
    ///     Extrapolate::Enable,
    /// )
    /// .unwrap();
    /// assert_eq!(interp.interpolate(&[1.4]).unwrap(), 0.56);
    /// assert_eq!(
    ///     interp.interpolate(&[3.6]).unwrap(),
    ///     1.44
    /// );
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
        interpolator.strategy.init(&interpolator.data)?;
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

impl<D, S> Interpolator<D::Elem> for Interp1D<D, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Num + Euclid + PartialOrd + Debug + Copy,
    S: Strategy1D<D> + Clone,
{
    /// Returns `1`.
    #[inline]
    fn ndim(&self) -> usize {
        N
    }

    fn validate(&mut self) -> Result<(), ValidateError> {
        self.check_extrapolate(&self.extrapolate)?;
        self.data.validate()?;
        self.strategy.init(&self.data)?;
        Ok(())
    }

    fn interpolate(&self, point: &[D::Elem]) -> Result<D::Elem, InterpolateError> {
        let point: &[D::Elem; N] = point
            .try_into()
            .map_err(|_| InterpolateError::PointLength(N))?;
        if !(self.data.grid[0].first().unwrap()..=self.data.grid[0].last().unwrap())
            .contains(&&point[0])
        {
            match &self.extrapolate {
                Extrapolate::Enable => {}
                Extrapolate::Fill(value) => return Ok(*value),
                Extrapolate::Clamp => {
                    let clamped_point = [*clamp(
                        &point[0],
                        self.data.grid[0].first().unwrap(),
                        self.data.grid[0].last().unwrap(),
                    )];
                    return self.strategy.interpolate(&self.data, &clamped_point);
                }
                Extrapolate::Wrap => {
                    let wrapped_point = [wrap(
                        point[0],
                        *self.data.grid[0].first().unwrap(),
                        *self.data.grid[0].last().unwrap(),
                    )];
                    return self.strategy.interpolate(&self.data, &wrapped_point);
                }
                Extrapolate::Error => {
                    return Err(InterpolateError::ExtrapolateError(format!(
                        "\n    point[0] = {:?} is out of bounds for grid[0] = {:?}",
                        point[0], self.data.grid[0]
                    )))
                }
            }
        };
        self.strategy.interpolate(&self.data, point)
    }

    fn set_extrapolate(&mut self, extrapolate: Extrapolate<D::Elem>) -> Result<(), ValidateError> {
        self.check_extrapolate(&extrapolate)?;
        self.extrapolate = extrapolate;
        Ok(())
    }
}

impl<D> Interp1D<D, Box<dyn Strategy1D<D>>>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    /// Update strategy dynamically.
    pub fn set_strategy(&mut self, strategy: Box<dyn Strategy1D<D>>) -> Result<(), ValidateError> {
        self.strategy = strategy;
        self.check_extrapolate(&self.extrapolate)
    }
}

impl<D> Interp1D<D, strategy::enums::Strategy1DEnum>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Num + PartialOrd + Copy + Debug,
{
    /// Update strategy dynamically.
    pub fn set_strategy(
        &mut self,
        strategy: impl Into<strategy::enums::Strategy1DEnum>,
    ) -> Result<(), ValidateError> {
        self.strategy = strategy.into();
        self.check_extrapolate(&self.extrapolate)
    }
}
