//! 3-dimensional interpolation

use super::*;

mod strategies;
#[cfg(test)]
mod tests;

const N: usize = 3;

/// [`InterpData`] for 3-D data.
pub type InterpData3D<D> = InterpData<D, N>;
/// [`InterpData3D`] that views data.
pub type InterpData3DViewed<T> = InterpData3D<ViewRepr<T>>;
/// [`InterpData3D`] that owns data.
pub type InterpData3DOwned<T> = InterpData3D<OwnedRepr<T>>;

impl<D> InterpData3D<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialOrd + Debug,
{
    /// Construct and validate a new [`InterpData3D`].
    pub fn new(
        x: ArrayBase<D, Ix1>,
        y: ArrayBase<D, Ix1>,
        z: ArrayBase<D, Ix1>,
        f_xyz: ArrayBase<D, Ix3>,
    ) -> Result<Self, ValidateError> {
        let data = Self {
            grid: [x, y, z],
            values: f_xyz,
        };
        data.validate()?;
        Ok(data)
    }
}

/// 3-D interpolator
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
pub struct Interp3D<D, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
    S: Strategy3D<D> + Clone,
{
    /// Interpolator data.
    pub data: InterpData3D<D>,
    /// Interpolation strategy.
    pub strategy: S,
    /// Extrapolation setting.
    #[cfg_attr(feature = "serde", serde(default))]
    pub extrapolate: Extrapolate<D::Elem>,
}
/// [`Interp3D`] that views data.
pub type Interp3DViewed<T, S> = Interp3D<ViewRepr<T>, S>;
/// [`Interp3D`] that owns data.
pub type Interp3DOwned<T, S> = Interp3D<OwnedRepr<T>, S>;

extrapolate_impl!(Interp3D, Strategy3D);
partialeq_impl!(Interp3D, InterpData3D, Strategy3D);

impl<D, S> Interp3D<D, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialOrd + Debug,
    S: Strategy3D<D> + Clone,
{
    /// Construct and validate a 3-D interpolator.
    ///
    /// Applicable interpolation strategies:
    /// - [`strategy::Linear`]
    /// - [`strategy::Nearest`]
    ///
    /// [`Extrapolate::Enable`] is valid for [`strategy::Linear`]
    ///
    /// # Example:
    /// ```
    /// use ndarray::prelude::*;
    /// use ninterp::prelude::*;
    /// // f(x, y, z) = 0.2 * x + 0.2 * y + 0.2 * z
    /// // type annotation for clarity
    /// let interp: Interp3DOwned<f64, _> = Interp3D::new(
    ///     // x
    ///     array![1., 2.], // x0, x1
    ///     // y
    ///     array![1., 2., 3.], // y0, y1, y2
    ///     // z
    ///     array![1., 2., 3., 4.], // z0, z1, z2, z3
    ///     // f(x, y, z)
    ///     array![
    ///         [
    ///             [0.6, 0.8, 1.0, 1.2], // f(x0, y0, z0), f(x0, y0, z1), f(x0, y0, z2), f(x0, y0, z3)
    ///             [0.8, 1.0, 1.2, 1.4], // f(x0, y1, z0), f(x0, y1, z1), f(x0, y1, z2), f(x0, y1, z3)
    ///             [1.0, 1.2, 1.4, 1.6], // f(x0, y2, z0), f(x0, y2, z1), f(x0, y2, z2), f(x0, y2, z3)
    ///         ],
    ///         [
    ///             [0.8, 1.0, 1.2, 1.4], // f(x1, y0, z0), f(x1, y0, z1), f(x1, y0, z2), f(x1, y0, z3)
    ///             [1.0, 1.2, 1.4, 1.6], // f(x1, y1, z0), f(x1, y1, z1), f(x1, y1, z2), f(x1, y1, z3)
    ///             [1.2, 1.4, 1.6, 1.8], // f(x1, y2, z0), f(x1, y2, z1), f(x1, y2, z2), f(x1, y2, z3)
    ///         ],
    ///     ],
    ///     strategy::Linear, // strategy mod is exposed via `use ndarray::prelude::*;`
    ///     Extrapolate::Error, // return an error when point is out of bounds
    /// )
    /// .unwrap();
    /// assert_eq!(interp.interpolate(&[1.5, 1.5, 1.5]).unwrap(), 0.9);
    /// // out of bounds point with `Extrapolate::Error` fails
    /// assert!(matches!(
    ///     interp.interpolate(&[5.5, 5.5, 5.5]).unwrap_err(),
    ///     ninterp::error::InterpolateError::ExtrapolateError(_)
    /// ));
    /// ```
    pub fn new(
        x: ArrayBase<D, Ix1>,
        y: ArrayBase<D, Ix1>,
        z: ArrayBase<D, Ix1>,
        f_xyz: ArrayBase<D, Ix3>,
        strategy: S,
        extrapolate: Extrapolate<D::Elem>,
    ) -> Result<Self, ValidateError> {
        let mut interpolator = Self {
            data: InterpData3D::new(x, y, z, f_xyz)?,
            strategy,
            extrapolate,
        };
        interpolator.check_extrapolate(&interpolator.extrapolate)?;
        interpolator.strategy.init(&interpolator.data)?;
        Ok(interpolator)
    }

    /// Return an interpolator with viewed data.
    pub fn view(&self) -> Interp3DViewed<&D::Elem, S>
    where
        S: for<'a> Strategy3D<ViewRepr<&'a D::Elem>>,
        D::Elem: Clone,
    {
        Interp3DViewed {
            data: self.data.view(),
            strategy: self.strategy.clone(),
            extrapolate: self.extrapolate.clone(),
        }
    }

    /// Turn the interpolator into an [`Interp3DOwned`], cloning the array elements if necessary.
    pub fn into_owned(self) -> Interp3DOwned<D::Elem, S>
    where
        S: Strategy3D<OwnedRepr<D::Elem>>,
        D::Elem: Clone,
    {
        Interp3DOwned {
            data: self.data.into_owned(),
            strategy: self.strategy.clone(),
            extrapolate: self.extrapolate.clone(),
        }
    }
}

impl<D, S> Interpolator<D::Elem> for Interp3D<D, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Num + Euclid + PartialOrd + Debug + Copy,
    S: Strategy3D<D> + Clone,
{
    /// Returns `3`.
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
        let mut errors = Vec::new();
        for dim in 0..N {
            if !(self.data.grid[dim].first().unwrap()..=self.data.grid[dim].last().unwrap())
                .contains(&&point[dim])
            {
                match &self.extrapolate {
                    Extrapolate::Enable => {}
                    Extrapolate::Fill(value) => return Ok(*value),
                    Extrapolate::Clamp => {
                        let clamped_point = std::array::from_fn(|i| {
                            *clamp(
                                &point[i],
                                self.data.grid[i].first().unwrap(),
                                self.data.grid[i].last().unwrap(),
                            )
                        });
                        return self.strategy.interpolate(&self.data, &clamped_point);
                    }
                    Extrapolate::Wrap => {
                        let wrapped_point = std::array::from_fn(|i| {
                            wrap(
                                point[i],
                                *self.data.grid[i].first().unwrap(),
                                *self.data.grid[i].last().unwrap(),
                            )
                        });
                        return self.strategy.interpolate(&self.data, &wrapped_point);
                    }
                    Extrapolate::Error => {
                        errors.push(format!(
                            "\n    point[{dim}] = {:?} is out of bounds for grid[{dim}] = {:?}",
                            point[dim], self.data.grid[dim],
                        ));
                    }
                };
            }
        }
        if !errors.is_empty() {
            return Err(InterpolateError::ExtrapolateError(errors.join("")));
        }
        self.strategy.interpolate(&self.data, point)
    }

    fn set_extrapolate(&mut self, extrapolate: Extrapolate<D::Elem>) -> Result<(), ValidateError> {
        self.check_extrapolate(&extrapolate)?;
        self.extrapolate = extrapolate;
        Ok(())
    }
}

impl<D> Interp3D<D, Box<dyn Strategy3D<D>>>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    /// Update strategy dynamically.
    pub fn set_strategy(&mut self, strategy: Box<dyn Strategy3D<D>>) -> Result<(), ValidateError> {
        self.strategy = strategy;
        self.check_extrapolate(&self.extrapolate)
    }
}

impl<D> Interp3D<D, strategy::enums::Strategy3DEnum>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Num + PartialOrd + Copy + Debug,
{
    /// Update strategy dynamically.
    pub fn set_strategy(
        &mut self,
        strategy: impl Into<strategy::enums::Strategy3DEnum>,
    ) -> Result<(), ValidateError> {
        self.strategy = strategy.into();
        self.check_extrapolate(&self.extrapolate)
    }
}
