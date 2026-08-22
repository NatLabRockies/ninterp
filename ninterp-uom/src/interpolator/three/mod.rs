//! Zero-copy 3-D interpolation over `uom` quantities: `Qx`/`Qy`/`Qz` are independent
//! per-axis units. See the [`one`] module docs for the transmute-soundness
//! argument every dimensionality here relies on.

use super::*;

#[cfg(all(test, feature = "f64"))]
mod tests;

/// 3-D interpolator over `uom` quantities: grid points of unit `Qx`/`Qy`/`Qz`, values of
/// unit `Qv`, all backed by storage representation `D` (`OwnedRepr<V>` or
/// `ViewRepr<&'a V>` - see the [`UomInterp3D`]/[`UomInterp3DView`] aliases below).
#[derive(Clone, Debug)]
pub struct UomInterp3DBase<D, Qx, Qy, Qz, Qv, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug + Clone,
    Qx: BaseUnit<D::Elem>,
    Qy: BaseUnit<D::Elem>,
    Qz: BaseUnit<D::Elem>,
    Qv: BaseUnit<D::Elem>,
    S: Clone,
{
    /// The wrapped, unit-erased `ninterp` interpolator - see the `inner` field on
    /// [`UomInterp1DBase`] for the escape-hatch rationale
    /// and the base-unit caveat.
    pub inner: Interp3DBase<D, S>,
    #[allow(clippy::type_complexity)]
    _units: PhantomData<fn() -> (Qx, Qy, Qz, Qv)>,
}

/// Owned variant (see [`UomInterp3DBase`] for the generic form).
pub type UomInterp3D<Qx, Qy, Qz, Qv, V, S> = UomInterp3DBase<OwnedRepr<V>, Qx, Qy, Qz, Qv, S>;
/// Viewed variant (see [`UomInterp3DBase`] for the generic form).
pub type UomInterp3DView<'a, Qx, Qy, Qz, Qv, V, S> =
    UomInterp3DBase<ViewRepr<&'a V>, Qx, Qy, Qz, Qv, S>;

impl<'a, Qx, Qy, Qz, Qv, V, S> UomInterp3DView<'a, Qx, Qy, Qz, Qv, V, S>
where
    Qx: BaseUnit<V>,
    Qy: BaseUnit<V>,
    Qz: BaseUnit<V>,
    Qv: BaseUnit<V>,
    V: Num + PartialOrd + Euclid + Copy + Debug + 'a,
    S: Strategy3D<ViewRepr<&'a V>> + Clone,
{
    /// Construct a viewed (borrowed, zero-copy) interpolator over `uom` quantity arrays.
    pub fn new(
        x: ArrayView1<'a, Qx>,
        y: ArrayView1<'a, Qy>,
        z: ArrayView1<'a, Qz>,
        f_xyz: ArrayView3<'a, Qv>,
        strategy: S,
        extrapolate: Extrapolate<V>,
    ) -> Result<Self, ValidateError> {
        // SAFETY: see `one` module docs - `Qx`/`Qy`/`Qz`/`Qv` are `uom` quantities
        // backed by `V`, `#[repr(transparent)]` over it, so reinterpreting each view's
        // element type as `V` is sound and each view's shape/strides are unaffected.
        let x: ArrayView1<'a, V> = unsafe { mem::transmute::<ArrayView1<'a, Qx>, _>(x) };
        let y: ArrayView1<'a, V> = unsafe { mem::transmute::<ArrayView1<'a, Qy>, _>(y) };
        let z: ArrayView1<'a, V> = unsafe { mem::transmute::<ArrayView1<'a, Qz>, _>(z) };
        let f_xyz: ArrayView3<'a, V> = unsafe { mem::transmute::<ArrayView3<'a, Qv>, _>(f_xyz) };
        Ok(Self {
            inner: Interp3DView::new(x, y, z, f_xyz, strategy, extrapolate)?,
            _units: PhantomData,
        })
    }
}

impl<Qx, Qy, Qz, Qv, V, S> UomInterp3D<Qx, Qy, Qz, Qv, V, S>
where
    Qx: BaseUnit<V>,
    Qy: BaseUnit<V>,
    Qz: BaseUnit<V>,
    Qv: BaseUnit<V>,
    V: Num + PartialOrd + Euclid + Copy + Debug,
    S: Strategy3D<OwnedRepr<V>> + Clone,
{
    /// Construct an owned interpolator over `uom` quantity arrays.
    pub fn new(
        x: Array1<Qx>,
        y: Array1<Qy>,
        z: Array1<Qz>,
        f_xyz: Array3<Qv>,
        strategy: S,
        extrapolate: Extrapolate<V>,
    ) -> Result<Self, ValidateError> {
        // SAFETY: same reasoning as the view constructor above, applied to owned
        // storage.
        let x: Array1<V> = unsafe { mem::transmute::<Array1<Qx>, _>(x) };
        let y: Array1<V> = unsafe { mem::transmute::<Array1<Qy>, _>(y) };
        let z: Array1<V> = unsafe { mem::transmute::<Array1<Qz>, _>(z) };
        let f_xyz: Array3<V> = unsafe { mem::transmute::<Array3<Qv>, _>(f_xyz) };
        Ok(Self {
            inner: Interp3D::new(x, y, z, f_xyz, strategy, extrapolate)?,
            _units: PhantomData,
        })
    }
}

impl<D, Qx, Qy, Qz, Qv, S> UomInterp3DBase<D, Qx, Qy, Qz, Qv, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Num + PartialOrd + Euclid + Copy + Debug,
    Qx: BaseUnit<D::Elem>,
    Qy: BaseUnit<D::Elem>,
    Qz: BaseUnit<D::Elem>,
    Qv: BaseUnit<D::Elem>,
    S: Strategy3D<D> + Clone,
{
    /// Interpolate at `(x, y, z)`, returning a value in `Qv`.
    pub fn interpolate(&self, x: Qx, y: Qy, z: Qz) -> Result<Qv, InterpolateError> {
        self.inner
            .interpolate(&[x.to_base(), y.to_base(), z.to_base()])
            .map(Qv::from_base)
    }

    uom_interp_common_methods!(
        UomInterp3DBase,
        Strategy3D,
        UomInterp3DView<'_, Qx, Qy, Qz, Qv, D::Elem, S>,
        UomInterp3D<Qx, Qy, Qz, Qv, D::Elem, S>,
        (Qx, Qy, Qz),
        (x, y, z)
    );
}

uom_interp_partial_eq!(UomInterp3DBase, Interp3DBase, (Qx, Qy, Qz));
uom_interp_set_strategy_box!(UomInterp3DBase, Strategy3D, (Qx, Qy, Qz));
uom_interp_set_strategy_enum!(UomInterp3DBase, Strategy3DEnum, (Qx, Qy, Qz));
uom_interp_serde!(UomInterp3DBase, Interp3DBase, (Qx, Qy, Qz));
