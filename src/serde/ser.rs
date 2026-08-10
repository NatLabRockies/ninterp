//! Serialization.
//!
//! Arrays are written in the [`ndarray`] format unless wrapped in [`Nested`].

use super::*;

/// Serialization in the nested-array format.
///
/// Implemented for every ninterp type that contains an [`ArrayBase`]. Each implementation
/// re-wraps its children, so the format choice propagates all the way down the value.
///
/// Prefer [`Nested`] or [`serialize_nested`] over calling this directly. This trait is not
/// sealed: implement it for your own types that contain interpolators to extend the recursion.
pub trait SerializeNested {
    /// Serialize `self`, writing any contained [`ArrayBase`] as nested sequences.
    fn serialize_nested<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer;
}

/// Serialize a value using the nested-array format.
///
/// Use this when serializing a value directly. For a field of your own type, use
/// [`serialize_nested`] with serde's `serialize_with` attribute:
/// ```
/// use ninterp::prelude::*;
///
/// #[derive(serde::Serialize)]
/// struct Config {
///     #[serde(serialize_with = "serialize_nested")]
///     curve: Interp1D<f64, strategy::Linear>,
/// }
/// ```
///
/// Non-self-describing formats (bincode, postcard, ...) cannot read the nested format back,
/// so for those this is a no-op and the [`ndarray`] format is written instead.
///
/// # Example
/// ```
/// # use ndarray::array;
/// # use ninterp::prelude::*;
/// # use ninterp::data::InterpData1D;
/// let interp = Interp1D::new(
///     array![0., 1., 2.],
///     array![0.0, 0.4, 0.8],
///     strategy::Linear,
///     Extrapolate::Error,
/// )
/// .unwrap();
///
/// let json = serde_json::to_string(&Nested(&interp.data)).unwrap();
/// assert_eq!(json, r#"{"grid":[[0.0,1.0,2.0]],"values":[0.0,0.4,0.8]}"#);
///
/// // ...and reads back regardless of which format it was written in
/// let de: InterpData1D<f64> = serde_json::from_str(&json).unwrap();
/// assert_eq!(de, interp.data);
/// ```
pub struct Nested<'a, T: ?Sized>(pub &'a T);

impl<T> Serialize for Nested<'_, T>
where
    T: SerializeNested + Serialize + ?Sized,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            self.0.serialize_nested(serializer)
        } else {
            self.0.serialize(serializer)
        }
    }
}

/// Serialize a value using the nested-array format, for use with `serialize_with`.
///
/// Equivalent to wrapping the value in [`Nested`].
///
/// # Example
/// ```
/// # use ninterp::prelude::*;
/// #[derive(serde::Serialize)]
/// struct Config {
///     #[serde(serialize_with = "serialize_nested")]
///     curve: Interp1D<f64, strategy::Linear>,
/// }
/// ```
pub fn serialize_nested<T, S>(value: &T, serializer: S) -> Result<S::Ok, S::Error>
where
    T: SerializeNested + Serialize + ?Sized,
    S: Serializer,
{
    Nested(value).serialize(serializer)
}

impl<T> SerializeNested for [T]
where
    T: SerializeNested + Serialize,
{
    fn serialize_nested<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for item in self {
            seq.serialize_element(&Nested(item))?;
        }
        seq.end()
    }
}

impl<T> SerializeNested for Vec<T>
where
    T: SerializeNested + Serialize,
{
    fn serialize_nested<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.as_slice().serialize_nested(serializer)
    }
}

impl<T, const N: usize> SerializeNested for [T; N]
where
    T: SerializeNested + Serialize,
{
    fn serialize_nested<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.as_slice().serialize_nested(serializer)
    }
}

impl<T> SerializeNested for Option<T>
where
    T: SerializeNested + Serialize,
{
    fn serialize_nested<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Some(value) => serializer.serialize_some(&Nested(value)),
            None => serializer.serialize_none(),
        }
    }
}

/// Writes a single array in the nested format, at any dimensionality.
pub(crate) struct ArrayWrapper<'a, D, Dm>(pub &'a ArrayBase<D, Dm>)
where
    D: Data,
    Dm: Dimension;

impl<D, Dm> Serialize for ArrayWrapper<'_, D, Dm>
where
    D: Data,
    D::Elem: Serialize,
    Dm: Dimension,
    ArrayBase<D, Dm>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // `serde_ndim` requires at least 1 dimension, and a 0-D array has no nested form that
        // could be read back anyway, so fall back to the `ndarray` format for that case.
        if self.0.ndim() == 0 {
            self.0.serialize(serializer)
        } else {
            serde_ndim::serialize(self.0, serializer)
        }
    }
}

/// Writes a fixed-length grid in the nested format.
#[derive(Serialize)]
pub(crate) struct GridArrWrapper<'a, D, const N: usize>(
    #[serde(serialize_with = "serialize_grid_arr")] pub &'a [ArrayBase<D, Ix1>; N],
)
where
    D: Data,
    D::Elem: Serialize;

/// Writes a variable-length grid in the nested format.
#[derive(Serialize)]
pub(crate) struct GridVecWrapper<'a, D>(
    #[serde(serialize_with = "serialize_grid_vec")] pub &'a [ArrayBase<D, Ix1>],
)
where
    D: Data,
    D::Elem: Serialize;

/// Write a fixed-length coordinate grid as a sequence of nested arrays.
pub fn serialize_grid_arr<D, const N: usize, S>(
    grid: &[ArrayBase<D, Ix1>; N],
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    D: Data,
    D::Elem: Serialize,
    S: Serializer,
{
    let mut seq = serializer.serialize_seq(Some(N))?;
    for arr in grid {
        seq.serialize_element(&ArrayWrapper(arr))?;
    }
    seq.end()
}

/// Write a variable-length coordinate grid as a sequence of nested arrays.
pub fn serialize_grid_vec<D, S>(
    grid: &[ArrayBase<D, Ix1>],
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    D: Data,
    D::Elem: Serialize,
    S: Serializer,
{
    let mut seq = serializer.serialize_seq(Some(grid.len()))?;
    for arr in grid {
        seq.serialize_element(&ArrayWrapper(arr))?;
    }
    seq.end()
}
