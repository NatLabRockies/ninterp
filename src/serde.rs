use super::*;

pub(crate) use ndarray::{DataOwned, IntoDimension};
pub(crate) use serde::{Deserialize, Serialize};
pub(crate) use serde_unit_struct::{Deserialize_unit_struct, Serialize_unit_struct};

use core::marker::PhantomData;
use serde::de::{
    value::{MapAccessDeserializer, SeqAccessDeserializer},
    DeserializeSeed, Deserializer, Error, MapAccess, SeqAccess, Visitor,
};
use serde::ser::{SerializeSeq, Serializer};
use serde_ndim::de::MakeNDim;

#[derive(Serialize)]
struct ArrayWrapper<'a, D>(
    #[serde(serialize_with = "serde_ndim::serialize")] &'a ArrayBase<D, Ix1>,
)
where
    D: Data,
    D::Elem: Serialize;

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
        formatter.write_str("a legacy ndarray object or a nested array sequence")
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
    deserializer.deserialize_any(ArrayFormatVisitor::<A>::new())
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
            Error::custom(format_args!("expected {N} array(s), found {}: {e:?}", e.len()))
        })
    }
}

pub(crate) mod serde_arr_array {
    use super::*;

    pub fn serialize<D, const N: usize, Ser>(
        grid: &[ArrayBase<D, Ix1>; N],
        serializer: Ser,
    ) -> Result<Ser::Ok, Ser::Error>
    where
        D: Data,
        D::Elem: Serialize,
        Ser: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(N))?;
        for arr in grid {
            seq.serialize_element(&ArrayWrapper(arr))?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D, const N: usize, De>(
        deserializer: De,
    ) -> Result<[ArrayBase<D, Ix1>; N], De::Error>
    where
        D: DataOwned,
        D::Elem: Deserialize<'de> + Debug,
        De: Deserializer<'de>,
    {
        deserializer.deserialize_seq(GridVisitor::<D, N>::new())
    }
}

pub(crate) mod serde_vec_array {
    use super::*;

    pub fn serialize<D, Ser>(
        grid: &[ArrayBase<D, Ix1>],
        serializer: Ser,
    ) -> Result<Ser::Ok, Ser::Error>
    where
        D: Data,
        D::Elem: Serialize,
        Ser: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(grid.len()))?;
        for arr in grid {
            seq.serialize_element(&ArrayWrapper(arr))?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D, De>(deserializer: De) -> Result<Vec<ArrayBase<D, Ix1>>, De::Error>
    where
        D: DataOwned,
        D::Elem: Deserialize<'de>,
        De: Deserializer<'de>,
    {
        struct VecGridVisitor<D>(PhantomData<fn() -> D>);

        impl<D> VecGridVisitor<D> {
            const fn new() -> Self {
                Self(PhantomData)
            }
        }

        impl<'de, D> Visitor<'de> for VecGridVisitor<D>
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

        deserializer.deserialize_seq(VecGridVisitor::<D>::new())
    }
}

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

pub fn deserialize_dyn<'de, D, De>(deserializer: De) -> Result<ArrayBase<D, IxDyn>, De::Error>
where
    D: DataOwned,
    D::Elem: Deserialize<'de>,
    ArrayBase<D, IxDyn>: MakeNDim<Item = D::Elem>,
    De: Deserializer<'de>,
{
    deserialize_array_format(deserializer)
}
