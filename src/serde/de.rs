//! Deserialization.
//!
//! Both array formats are accepted, whichever was written. Self-describing formats dispatch on
//! the first token: a sequence is the nested format, a map is the [`ndarray`] format. Formats
//! that are not self-describing cannot support [`Deserializer::deserialize_any`] at all, so for
//! those the [`ndarray`] format is assumed, as it is the only one they can produce.

use super::*;

use core::marker::PhantomData;

use serde::de::{
    value::{MapAccessDeserializer, SeqAccessDeserializer},
    DeserializeSeed, Deserializer, Error, MapAccess, SeqAccess, Visitor,
};
use serde_ndim::de::MakeNDim;

struct ArrayFormatVisitor<A>(PhantomData<fn() -> A>);

impl<A> ArrayFormatVisitor<A> {
    const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<'de, A> Visitor<'de> for ArrayFormatVisitor<A>
where
    A: Deserialize<'de> + MakeNDim,
    A::Item: Deserialize<'de>,
{
    type Value = A;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("an ndarray object or a nested array sequence")
    }

    fn visit_seq<S>(self, seq: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        serde_ndim::deserialize(SeqAccessDeserializer::new(seq))
    }

    fn visit_map<M>(self, map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        A::deserialize(MapAccessDeserializer::new(map))
    }
}

fn deserialize_array_format<'de, A, D>(deserializer: D) -> Result<A, D::Error>
where
    A: Deserialize<'de> + MakeNDim,
    A::Item: Deserialize<'de>,
    D: Deserializer<'de>,
{
    // Accepting either format requires `deserialize_any`, which non-self-describing formats
    // (bincode, postcard, ...) cannot support. Those only ever produce the `ndarray` format,
    // so defer to its own impl rather than failing outright.
    if deserializer.is_human_readable() {
        deserializer.deserialize_any(ArrayFormatVisitor::<A>::new())
    } else {
        A::deserialize(deserializer)
    }
}

struct ArraySeed<D>(PhantomData<fn() -> D>);

impl<D> ArraySeed<D> {
    const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<'de, D> DeserializeSeed<'de> for ArraySeed<D>
where
    D: DataOwned,
    D::Elem: Deserialize<'de>,
    ArrayBase<D, Ix1>: Deserialize<'de> + MakeNDim<Item = D::Elem>,
{
    type Value = ArrayBase<D, Ix1>;

    fn deserialize<De>(self, deserializer: De) -> Result<Self::Value, De::Error>
    where
        De: Deserializer<'de>,
    {
        deserialize_array_format(deserializer)
    }
}

struct GridVisitor<D, const N: usize>(PhantomData<fn() -> D>);

impl<D, const N: usize> GridVisitor<D, N> {
    const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<'de, D, const N: usize> Visitor<'de> for GridVisitor<D, N>
where
    D: DataOwned,
    D::Elem: Deserialize<'de> + Debug,
    ArrayBase<D, Ix1>: Deserialize<'de> + MakeNDim<Item = D::Elem>,
{
    type Value = [ArrayBase<D, Ix1>; N];

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a sequence of arrays")
    }

    fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        let mut grid = Vec::with_capacity(N);
        while let Some(array) = seq.next_element_seed(ArraySeed::<D>::new())? {
            grid.push(array);
        }

        grid.try_into().map_err(|e: Vec<_>| {
            Error::custom(format_args!(
                "expected {N} array(s), found {}: {e:?}",
                e.len()
            ))
        })
    }
}

struct GridVecVisitor<D>(PhantomData<fn() -> D>);

impl<D> GridVecVisitor<D> {
    const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<'de, D> Visitor<'de> for GridVecVisitor<D>
where
    D: DataOwned,
    D::Elem: Deserialize<'de>,
    ArrayBase<D, Ix1>: Deserialize<'de> + MakeNDim<Item = D::Elem>,
{
    type Value = Vec<ArrayBase<D, Ix1>>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a sequence of arrays")
    }

    fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
    where
        S: SeqAccess<'de>,
    {
        let mut grid = Vec::new();
        while let Some(array) = seq.next_element_seed(ArraySeed::<D>::new())? {
            grid.push(array);
        }
        Ok(grid)
    }
}

/// Read a fixed-length coordinate grid, in either array format.
pub fn deserialize_grid_arr<'de, D, const N: usize, De>(
    deserializer: De,
) -> Result<[ArrayBase<D, Ix1>; N], De::Error>
where
    D: DataOwned,
    D::Elem: Deserialize<'de> + Debug,
    [ArrayBase<D, Ix1>; N]: Deserialize<'de>,
    De: Deserializer<'de>,
{
    // A fixed-size array is written as a tuple, which non-self-describing formats encode without
    // a length prefix, so reading it back as a seq would desynchronize the stream.
    if deserializer.is_human_readable() {
        deserializer.deserialize_seq(GridVisitor::<D, N>::new())
    } else {
        <[ArrayBase<D, Ix1>; N]>::deserialize(deserializer)
    }
}

/// Read a variable-length coordinate grid, in either array format.
pub fn deserialize_grid_vec<'de, D, De>(
    deserializer: De,
) -> Result<Vec<ArrayBase<D, Ix1>>, De::Error>
where
    D: DataOwned,
    D::Elem: Deserialize<'de>,
    De: Deserializer<'de>,
{
    if deserializer.is_human_readable() {
        deserializer.deserialize_seq(GridVecVisitor::<D>::new())
    } else {
        Vec::<ArrayBase<D, Ix1>>::deserialize(deserializer)
    }
}

/// Read a fixed-dimensionality values array, in either array format.
pub fn deserialize_fixed<'de, D, const N: usize, De>(
    deserializer: De,
) -> Result<ArrayBase<D, Dim<[Ix; N]>>, De::Error>
where
    D: DataOwned,
    D::Elem: Deserialize<'de>,
    Dim<[Ix; N]>: Dimension + Deserialize<'de>,
    ArrayBase<D, Dim<[Ix; N]>>: MakeNDim<Item = D::Elem>,
    De: Deserializer<'de>,
{
    deserialize_array_format(deserializer)
}

/// Read a dynamic-dimensionality values array, in either array format.
pub fn deserialize_dyn<'de, D, De>(deserializer: De) -> Result<ArrayBase<D, IxDyn>, De::Error>
where
    D: DataOwned,
    D::Elem: Deserialize<'de>,
    ArrayBase<D, IxDyn>: MakeNDim<Item = D::Elem>,
    De: Deserializer<'de>,
{
    deserialize_array_format(deserializer)
}
