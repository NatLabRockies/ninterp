//! N-dimensional interpolation

use super::*;

use ndarray::prelude::*;

mod strategies;
#[cfg(test)]
mod tests;

/// Interpolator data for N-dimensional interpolators, where N can vary at runtime.
///
/// See [`InterpData`] and its aliases for concrete-dimensionality interpolator data structs.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "D::Elem: Serialize",
        deserialize = "
            D: DataOwned,
            D::Elem: Deserialize<'de>,
        "
    ))
)]
pub struct InterpDataND<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    /// Coordinate grid: a vector of 1-dimensional [`ArrayBase<D, Ix1>`].
    #[cfg_attr(feature = "serde", serde(with = "serde_vec_array"))]
    pub grid: Vec<ArrayBase<D, Ix1>>,
    /// Function values at coordinates: a single dynamic-dimensional [`ArrayBase`].
    #[cfg_attr(feature = "serde", serde(serialize_with = "serde_ndim::serialize"))]
    #[cfg_attr(feature = "serde", serde(deserialize_with = "deserialize_dyn"))]
    pub values: ArrayBase<D, IxDyn>,
}
/// [`InterpDataND`] that views data.
pub type InterpDataNDViewed<T> = InterpDataND<ViewRepr<T>>;
/// [`InterpDataND`] that owns data.
pub type InterpDataNDOwned<T> = InterpDataND<OwnedRepr<T>>;

impl<D> PartialEq for InterpDataND<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
    ArrayBase<D, Ix1>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.grid == other.grid && self.values == other.values
    }
}

impl<D> InterpDataND<D>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    /// Construct and validate a new [`InterpDataND`].
    pub fn new(
        grid: Vec<ArrayBase<D, Ix1>>,
        values: ArrayBase<D, IxDyn>,
    ) -> Result<Self, ValidateError>
    where
        D::Elem: PartialOrd,
    {
        let data = Self { grid, values };
        data.validate()?;
        Ok(data)
    }

    /// Validate interpolator data.
    pub fn validate(&self) -> Result<(), ValidateError>
    where
        D::Elem: PartialOrd,
    {
        let n = self.ndim();
        if (self.grid.len() != n) && !(n == 0 && self.grid.iter().all(|g| g.is_empty())) {
            // Only possible for `InterpDataND`
            return Err(ValidateError::Other(format!(
                "grid length {} does not match dimensionality {}",
                self.grid.len(),
                n,
            )));
        }
        for i in 0..n {
            let i_grid_len = self.grid[i].len();
            // Check that each grid dimension has elements
            // Indexing `grid` directly is okay because empty dimensions are caught at compilation
            if i_grid_len == 0 {
                return Err(ValidateError::EmptyGrid(i));
            }
            // Check that grid points are monotonically increasing
            if !self.grid[i].windows(2).into_iter().all(|w| w[0] <= w[1]) {
                return Err(ValidateError::Monotonicity(i));
            }
            // Check that grid and values are compatible shapes
            if i_grid_len != self.values.shape()[i] {
                return Err(ValidateError::IncompatibleShapes(i));
            }
        }
        Ok(())
    }

    /// Get data dimensionality.
    pub fn ndim(&self) -> usize {
        if self.values.len() == 1 {
            0
        } else {
            self.values.ndim()
        }
    }

    /// View interpolator data.
    pub fn view(&self) -> InterpDataNDViewed<&D::Elem> {
        InterpDataNDViewed {
            grid: self.grid.iter().map(|g| g.view()).collect(),
            values: self.values.view(),
        }
    }

    /// Turn the data into an [`InterpDataNDOwned`], cloning the array elements if necessary.
    pub fn into_owned(self) -> InterpDataNDOwned<D::Elem>
    where
        D::Elem: Clone,
    {
        InterpDataNDOwned {
            grid: self.grid.into_iter().map(|g| g.into_owned()).collect(),
            values: self.values.into_owned(),
        }
    }
}

/// N-D interpolator
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
pub struct InterpND<D, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
    S: StrategyND<D> + Clone,
{
    /// Interpolator data.
    pub data: InterpDataND<D>,
    /// Interpolation strategy.
    pub strategy: S,
    /// Extrapolation setting.
    #[cfg_attr(feature = "serde", serde(default))]
    pub extrapolate: Extrapolate<D::Elem>,
}
/// [`InterpND`] that views data.
pub type InterpNDViewed<T, S> = InterpND<ViewRepr<T>, S>;
/// [`InterpND`] that owns data.
pub type InterpNDOwned<T, S> = InterpND<OwnedRepr<T>, S>;

extrapolate_impl!(InterpND, StrategyND);
partialeq_impl!(InterpND, InterpDataND, StrategyND);

impl<D, S> InterpND<D, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialOrd + Debug,
    S: StrategyND<D> + Clone,
{
    /// Construct and validate an N-D (any dimensionality) interpolator.
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
    /// let interp: InterpNDOwned<f64, _> = InterpND::new(
    ///     // grid
    ///     vec![
    ///         // x
    ///         array![1., 2.], // x0, x1
    ///         // y
    ///         array![1., 2., 3.], // y0, y1, y2
    ///         // z
    ///         array![1., 2., 3., 4.], // z0, z1, z2, z3
    ///     ],
    ///     // values
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
    ///     ].into_dyn(),
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
        grid: Vec<ArrayBase<D, Ix1>>,
        values: ArrayBase<D, IxDyn>,
        strategy: S,
        extrapolate: Extrapolate<D::Elem>,
    ) -> Result<Self, ValidateError> {
        let mut interpolator = Self {
            data: InterpDataND::new(grid, values)?,
            strategy,
            extrapolate,
        };
        interpolator.check_extrapolate(&interpolator.extrapolate)?;
        interpolator.strategy.init(&interpolator.data)?;
        Ok(interpolator)
    }

    /// Return an interpolator with viewed data.
    pub fn view(&self) -> InterpNDViewed<&D::Elem, S>
    where
        S: for<'a> StrategyND<ViewRepr<&'a D::Elem>>,
        D::Elem: Clone,
    {
        InterpNDViewed {
            data: self.data.view(),
            strategy: self.strategy.clone(),
            extrapolate: self.extrapolate.clone(),
        }
    }

    /// Turn the interpolator into an [`InterpNDOwned`], cloning the array elements if necessary.
    pub fn into_owned(self) -> InterpNDOwned<D::Elem, S>
    where
        S: StrategyND<OwnedRepr<D::Elem>>,
        D::Elem: Clone,
    {
        InterpNDOwned {
            data: self.data.into_owned(),
            strategy: self.strategy.clone(),
            extrapolate: self.extrapolate.clone(),
        }
    }
}

impl<D, S> Interpolator<D::Elem> for InterpND<D, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Num + Euclid + PartialOrd + Debug + Copy,
    S: StrategyND<D> + Clone,
{
    #[inline]
    fn ndim(&self) -> usize {
        self.data.ndim()
    }

    fn validate(&mut self) -> Result<(), ValidateError> {
        self.check_extrapolate(&self.extrapolate)?;
        self.data.validate()?;
        self.strategy.init(&self.data)?;
        Ok(())
    }

    fn interpolate(&self, point: &[D::Elem]) -> Result<D::Elem, InterpolateError> {
        let n = self.ndim();
        if point.len() != n {
            return Err(InterpolateError::PointLength(n));
        }
        let mut errors = Vec::new();
        for dim in 0..n {
            if !(self.data.grid[dim].first().unwrap()..=self.data.grid[dim].last().unwrap())
                .contains(&&point[dim])
            {
                match &self.extrapolate {
                    Extrapolate::Enable => {}
                    Extrapolate::Fill(value) => return Ok(*value),
                    Extrapolate::Clamp => {
                        let clamped_point: Vec<_> = point
                            .iter()
                            .enumerate()
                            .map(|(dim, pt)| {
                                *clamp(
                                    pt,
                                    self.data.grid[dim].first().unwrap(),
                                    self.data.grid[dim].last().unwrap(),
                                )
                            })
                            .collect();
                        return self.strategy.interpolate(&self.data, &clamped_point);
                    }
                    Extrapolate::Wrap => {
                        let wrapped_point: Vec<_> = point
                            .iter()
                            .enumerate()
                            .map(|(dim, pt)| {
                                wrap(
                                    *pt,
                                    *self.data.grid[dim].first().unwrap(),
                                    *self.data.grid[dim].last().unwrap(),
                                )
                            })
                            .collect();
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

impl<D> InterpND<D, Box<dyn StrategyND<D>>>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    /// Update strategy dynamically.
    pub fn set_strategy(&mut self, strategy: Box<dyn StrategyND<D>>) -> Result<(), ValidateError> {
        self.strategy = strategy;
        self.check_extrapolate(&self.extrapolate)
    }
}

impl<D> InterpND<D, strategy::enums::StrategyNDEnum>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Num + PartialOrd + Copy + Debug,
{
    /// Update strategy dynamically.
    pub fn set_strategy(
        &mut self,
        strategy: impl Into<strategy::enums::StrategyNDEnum>,
    ) -> Result<(), ValidateError> {
        self.strategy = strategy.into();
        self.check_extrapolate(&self.extrapolate)
    }
}
