//! Interpolator data-containing types for fixed dimensionalities.

use super::*;

pub use n::{InterpDataND, InterpDataNDBase, InterpDataNDView};
pub use one::{InterpData1D, InterpData1DBase, InterpData1DView};
pub use three::{InterpData3D, InterpData3DBase, InterpData3DView};
pub use two::{InterpData2D, InterpData2DBase, InterpData2DView};

/// Interpolator data for interpolators of concrete dimensionality `const N: usize`.
///
/// See [`InterpDataND`] for the N-dimensional interpolator data struct.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "
            D::Elem: Serialize,
            Dim<[usize; N]>: Serialize,
            [ArrayBase<D, Ix1>; N]: Serialize,
        ",
        deserialize = "
            D: DataOwned,
            D::Elem: Deserialize<'de>,
            Dim<[usize; N]>: Deserialize<'de> + Dimension,
            [usize; N]: IntoDimension<Dim = Dim<[usize; N]>>,
            [ArrayBase<D, Ix1>; N]: Deserialize<'de>,
        "
    ))
)]
pub struct InterpDataBase<D, const N: usize>
where
    Dim<[Ix; N]>: Dimension,
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    /// Coordinate grid: an `N`-length array of 1-dimensional [`ArrayBase<D, Ix1>`].
    /// - 1-D: `[x]`
    /// - 2-D: `[x, y]`
    /// - 3-D: `[x, y, z]`
    #[cfg_attr(feature = "serde", serde(deserialize_with = "deserialize_grid_arr"))]
    pub grid: [ArrayBase<D, Ix1>; N],
    /// Function values at coordinates: a single `N`-dimensional [`ArrayBase`].
    #[cfg_attr(feature = "serde", serde(deserialize_with = "deserialize_fixed"))]
    pub values: ArrayBase<D, Dim<[Ix; N]>>,
}
/// [`InterpData`] that views data.
pub type InterpDataView<T, const N: usize> = InterpDataBase<ViewRepr<T>, N>;
/// [`InterpData`] that owns data.
pub type InterpDataOwned<T, const N: usize> = InterpDataBase<OwnedRepr<T>, N>;

#[cfg(feature = "serde")]
impl<D, const N: usize> SerializeNested for InterpDataBase<D, N>
where
    Dim<[Ix; N]>: Dimension,
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug + Serialize,
    ArrayBase<D, Dim<[Ix; N]>>: Serialize,
{
    fn serialize_nested<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = serializer.serialize_struct("InterpData", 2)?;
        s.serialize_field("grid", &GridArrWrapper(&self.grid))?;
        s.serialize_field("values", &ArrayWrapper(&self.values))?;
        s.end()
    }
}

impl<D, const N: usize> PartialEq for InterpDataBase<D, N>
where
    Dim<[Ix; N]>: Dimension,
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
    ArrayBase<D, Ix1>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.grid == other.grid && self.values == other.values
    }
}

impl<D, const N: usize> InterpDataBase<D, N>
where
    Dim<[Ix; N]>: Dimension,
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    /// Validate interpolator data.
    pub fn validate(&self) -> Result<(), ValidateError>
    where
        D::Elem: PartialOrd,
    {
        for i in 0..N {
            let i_grid_len = self.grid[i].len();
            // Every strategy needs at least 2 points per dimension to bracket a query point
            if i_grid_len < 2 {
                return Err(ValidateError::InsufficientGridPoints(i));
            }
            // Check that grid points are monotonically increasing
            if !self.grid[i].windows(2).into_iter().all(|w| w[0] <= w[1]) {
                return Err(ValidateError::NonMonotonic(i));
            }
            // Check that grid and values are compatible shapes
            if i_grid_len != self.values.shape()[i] {
                return Err(ValidateError::IncompatibleShapes(i));
            }
        }
        Ok(())
    }

    /// View interpolator data.
    pub fn view(&self) -> InterpDataView<&D::Elem, N> {
        InterpDataView {
            grid: std::array::from_fn(|i| self.grid[i].view()),
            values: self.values.view(),
        }
    }

    /// Turn the data into an [`InterpDataOwned`], cloning the array elements if necessary.
    pub fn into_owned(self) -> InterpDataOwned<D::Elem, N>
    where
        Dim<[Ix; N]>: Dimension,
        D::Elem: Clone,
    {
        InterpDataOwned {
            grid: self.grid.map(|arr| arr.into_owned()),
            values: self.values.into_owned(),
        }
    }
}
