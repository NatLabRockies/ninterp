//! Zero-copy 2-D interpolation over `uom` quantities: `Qx`/`Qy` are independent per-axis
//! units, so e.g. an RPM x Torque grid producing a Power value is one type. See the
//! [`one`] module docs for the transmute-soundness argument every
//! dimensionality here relies on.

use super::*;

#[cfg(all(test, feature = "f64"))]
mod tests;

/// 2-D interpolator over `uom` quantities: grid points of unit `Qx`/`Qy`, values of unit
/// `Qv`, all backed by storage representation `D` (`OwnedRepr<V>` or `ViewRepr<&'a V>` -
/// see the [`UomInterp2D`]/[`UomInterp2DView`] aliases below).
#[derive(Clone, Debug)]
pub struct UomInterp2DBase<D, Qx, Qy, Qv, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug + Clone,
    Qx: BaseUnit<D::Elem>,
    Qy: BaseUnit<D::Elem>,
    Qv: BaseUnit<D::Elem>,
    S: Clone,
{
    /// The wrapped, unit-erased `ninterp` interpolator - see the `inner` field on
    /// [`UomInterp1DBase`] for the escape-hatch rationale
    /// and the base-unit caveat.
    pub inner: Interp2DBase<D, S>,
    #[allow(clippy::type_complexity)]
    _units: PhantomData<fn() -> (Qx, Qy, Qv)>,
}

/// Owned variant (see [`UomInterp2DBase`] for the generic form).
pub type UomInterp2D<Qx, Qy, Qv, V, S> = UomInterp2DBase<OwnedRepr<V>, Qx, Qy, Qv, S>;
/// Viewed variant (see [`UomInterp2DBase`] for the generic form).
pub type UomInterp2DView<'a, Qx, Qy, Qv, V, S> = UomInterp2DBase<ViewRepr<&'a V>, Qx, Qy, Qv, S>;

impl<'a, Qx, Qy, Qv, V, S> UomInterp2DView<'a, Qx, Qy, Qv, V, S>
where
    Qx: BaseUnit<V>,
    Qy: BaseUnit<V>,
    Qv: BaseUnit<V>,
    V: Num + PartialOrd + Euclid + Copy + Debug + 'a,
    S: Strategy2D<ViewRepr<&'a V>> + Clone,
{
    /// Construct a viewed (borrowed, zero-copy) interpolator over `uom` quantity arrays.
    pub fn new(
        x: ArrayView1<'a, Qx>,
        y: ArrayView1<'a, Qy>,
        f_xy: ArrayView2<'a, Qv>,
        strategy: S,
        extrapolate: Extrapolate<V>,
    ) -> Result<Self, ValidateError> {
        // SAFETY: see `one` module docs - `Qx`/`Qy`/`Qv` are `uom` quantities backed by
        // `V`, `#[repr(transparent)]` over it, so reinterpreting each view's element
        // type as `V` is sound and each view's shape/strides are unaffected.
        let x: ArrayView1<'a, V> = unsafe { mem::transmute::<ArrayView1<'a, Qx>, _>(x) };
        let y: ArrayView1<'a, V> = unsafe { mem::transmute::<ArrayView1<'a, Qy>, _>(y) };
        let f_xy: ArrayView2<'a, V> = unsafe { mem::transmute::<ArrayView2<'a, Qv>, _>(f_xy) };
        Ok(Self {
            inner: Interp2DView::new(x, y, f_xy, strategy, extrapolate)?,
            _units: PhantomData,
        })
    }
}

impl<Qx, Qy, Qv, V, S> UomInterp2D<Qx, Qy, Qv, V, S>
where
    Qx: BaseUnit<V>,
    Qy: BaseUnit<V>,
    Qv: BaseUnit<V>,
    V: Num + PartialOrd + Euclid + Copy + Debug,
    S: Strategy2D<OwnedRepr<V>> + Clone,
{
    /// Construct an owned interpolator over `uom` quantity arrays.
    pub fn new(
        x: Array1<Qx>,
        y: Array1<Qy>,
        f_xy: Array2<Qv>,
        strategy: S,
        extrapolate: Extrapolate<V>,
    ) -> Result<Self, ValidateError> {
        // SAFETY: same reasoning as the view constructor above, applied to owned
        // storage.
        let x: Array1<V> = unsafe { mem::transmute::<Array1<Qx>, _>(x) };
        let y: Array1<V> = unsafe { mem::transmute::<Array1<Qy>, _>(y) };
        let f_xy: Array2<V> = unsafe { mem::transmute::<Array2<Qv>, _>(f_xy) };
        Ok(Self {
            inner: Interp2D::new(x, y, f_xy, strategy, extrapolate)?,
            _units: PhantomData,
        })
    }
}

impl<D, Qx, Qy, Qv, S> UomInterp2DBase<D, Qx, Qy, Qv, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Num + PartialOrd + Euclid + Copy + Debug,
    Qx: BaseUnit<D::Elem>,
    Qy: BaseUnit<D::Elem>,
    Qv: BaseUnit<D::Elem>,
    S: Strategy2D<D> + Clone,
{
    /// Interpolate at `(x, y)`, returning a value in `Qv`.
    pub fn interpolate(&self, x: Qx, y: Qy) -> Result<Qv, InterpolateError> {
        self.inner
            .interpolate(&[x.to_base(), y.to_base()])
            .map(Qv::from_base)
    }

    uom_interp_common_methods!(
        UomInterp2DBase,
        Strategy2D,
        UomInterp2DView<'_, Qx, Qy, Qv, D::Elem, S>,
        UomInterp2D<Qx, Qy, Qv, D::Elem, S>,
        (Qx, Qy),
        (x, y)
    );
}

uom_interp_partial_eq!(UomInterp2DBase, Interp2DBase, (Qx, Qy));
uom_interp_set_strategy_box!(UomInterp2DBase, Strategy2D, (Qx, Qy));
uom_interp_set_strategy_enum!(UomInterp2DBase, Strategy2DEnum, (Qx, Qy));
uom_interp_serde!(UomInterp2DBase, Interp2DBase, (Qx, Qy));
