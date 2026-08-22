use super::*;

/// A `uom` quantity backed by storage type `V`, convertible to/from its dimension's base
/// unit. Implemented for every `Quantity<D, U, V>`, i.e. every `uom` quantity of every
/// unit system: `Length`, `Power`, `Ratio`, in `f32` or `f64`, all alike.
pub trait BaseUnit<V>: Copy {
    fn to_base(self) -> V;
    fn from_base(value: V) -> Self;
}

impl<D, U, V> BaseUnit<V> for Quantity<D, U, V>
where
    D: Dimension + ?Sized,
    U: Units<V> + ?Sized,
    V: Num + Conversion<V> + Copy,
{
    fn to_base(self) -> V {
        self.value
    }

    fn from_base(value: V) -> Self {
        Quantity {
            dimension: PhantomData,
            units: PhantomData,
            value,
        }
    }
}
