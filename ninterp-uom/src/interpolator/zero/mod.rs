//! 0-D "interpolation": a constant value, still uom-typed.

use super::*;

#[cfg(all(test, feature = "f64"))]
mod tests;

/// 0-D 'interpolator': wraps a constant value of unit `Q`, backed by storage type `V`.
///
/// Has no axes, so the per-axis heterogeneous-unit machinery the other dimensionalities
/// need doesn't apply here; kept minimal (`new`/`interpolate`/serde only) rather than
/// pulled through the batch/`view`/`set_strategy` surface those share.
#[derive(Clone, Debug, PartialEq)]
pub struct UomInterp0D<Q, V> {
    /// The wrapped, unit-erased `ninterp` interpolator - see the `inner` field on
    /// [`UomInterp1DBase`] for the escape-hatch rationale
    /// and the base-unit caveat.
    pub inner: Interp0D<V>,
    _unit: PhantomData<fn() -> Q>,
}

impl<Q, V> UomInterp0D<Q, V>
where
    Q: BaseUnit<V>,
    V: PartialEq + Debug,
{
    /// Construct a constant-value 'interpolator'.
    pub fn new(value: Q) -> Self {
        Self {
            inner: Interp0D::new(value.to_base()),
            _unit: PhantomData,
        }
    }

    /// Returns the contained value. Infallible: 0-D has no point argument to get wrong
    /// (core's `InterpolateError::PointLength` case can't occur here).
    pub fn interpolate(&self) -> Q
    where
        V: Clone,
    {
        Q::from_base(self.inner.0.clone())
    }
}

#[cfg(feature = "serde")]
impl<Q, V> Serialize for UomInterp0D<Q, V>
where
    Q: BaseUnit<V>,
    V: PartialEq + Debug + Serialize,
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: Serializer,
    {
        self.inner.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, Q, V> Deserialize<'de> for UomInterp0D<Q, V>
where
    Q: BaseUnit<V>,
    V: PartialEq + Debug + Deserialize<'de>,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        Ok(Self {
            inner: Interp0D::deserialize(deserializer)?,
            _unit: PhantomData,
        })
    }
}
