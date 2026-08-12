//! 3-dimensional interpolation

use super::*;

mod strategies;
#[cfg(test)]
mod tests;

const N: usize = 3;

/// Generic (base) form for 3-D data; parameterized by data representation.
pub type InterpData3DBase<D> = InterpDataBase<D, N>;
/// Owned data variant for 3-D data (see [`InterpData3DBase`] for the generic form).
pub type InterpData3D<T> = InterpData3DBase<OwnedRepr<T>>;
/// Viewed data variant for 3-D data (see [`InterpData3DBase`] for the generic form).
pub type InterpData3DView<T> = InterpData3DBase<ViewRepr<T>>;

impl<D> InterpData3DBase<D>
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
pub struct Interp3DBase<D, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
    S: Clone,
{
    /// Interpolator data.
    pub data: InterpData3DBase<D>,
    /// Interpolation strategy.
    pub strategy: S,
    /// Extrapolation setting.
    pub extrapolate: Extrapolate<D::Elem>,
}
/// Owned interpolator variant (see [`Interp3DBase`] for the generic form).
pub type Interp3D<T, S> = Interp3DBase<OwnedRepr<T>, S>;
/// Viewed interpolator variant (see [`Interp3DBase`] for the generic form).
pub type Interp3DView<T, S> = Interp3DBase<ViewRepr<T>, S>;

impl<D, S> Interp3DBase<D, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
    S: Strategy3D<D> + Clone,
{
    /// Construct and validate a 3-D interpolator.
    ///
    /// # Example:
    /// ```
    /// use ndarray::prelude::*;
    /// use ninterp::prelude::*;
    /// // f(x, y, z) = 0.2 * x + 0.2 * y + 0.2 * z
    /// let interp = Interp3D::new(
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
    ///     strategy::Linear,   // strategy mod is exposed via `use ndarray::prelude::*;`
    ///     Extrapolate::Error, // return an error when point is out of bounds
    /// )
    /// .unwrap();
    /// assert_eq!(interp.interpolate(&[1.5, 1.5, 1.5]).unwrap(), 0.9);
    /// // out of bounds point with `Extrapolate::Error` fails
    /// assert!(matches!(
    ///     interp.interpolate(&[5.5, 5.5, 5.5]).unwrap_err(),
    ///     ninterp::error::InterpolateError::OutOfBounds(_)
    /// ));
    /// ```
    pub fn new(
        x: ArrayBase<D, Ix1>,
        y: ArrayBase<D, Ix1>,
        z: ArrayBase<D, Ix1>,
        f_xyz: ArrayBase<D, Ix3>,
        strategy: S,
        extrapolate: Extrapolate<D::Elem>,
    ) -> Result<Self, ValidateError>
    where
        D::Elem: PartialOrd,
    {
        let mut interpolator = Self {
            data: InterpData3DBase::new(x, y, z, f_xyz)?,
            strategy,
            extrapolate,
        };
        interpolator.validate_extrapolate(&interpolator.extrapolate)?;
        interpolator.validate_strategy()?;
        interpolator.init_strategy()?;
        Ok(interpolator)
    }

    interp_inherent_methods!(
        Interp3DBase,
        Strategy3D,
        Interp3DView<&D::Elem, S>,
        Interp3D<D::Elem, S>
    );
}

interp_trait_impls!(
    Interp3DBase,
    Interp3D,
    InterpData3DBase,
    Strategy3D,
    strategy::enums::Strategy3DEnum,
    strategy::enums::Strategy3DEnum<D::Elem>,
    N
);
