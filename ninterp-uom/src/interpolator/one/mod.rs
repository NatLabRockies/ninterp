//! Zero-copy 1-D interpolation over `uom` quantities: generic over any dimension/unit,
//! any storage type (`f32`, `f64`, ...), and both owned and borrowed data.
//!
//! `Quantity<D, U, V>` is `#[repr(transparent)]` over `V`; `dimension`/`units` are
//! `PhantomData` (zero-sized regardless of `D`/`U`), so `V` is the only field that
//! contributes to layout. `ArrayView1<A>`/`Array1<A>`'s own size doesn't depend on `A`
//! either (a view holds a thin pointer plus shape/strides; an owned array is
//! `Vec`-shaped), which is why `mem::transmute` type-checks even with `Qx`/`Qv`/`V` left
//! generic - unlike a transmute of a bare scalar type parameter, which fails to compile
//! (rustc can't prove two abstract types have equal size, only that *this particular*
//! wrapper struct's size is element-type-independent). That reasoning only goes through
//! when the concrete container (`OwnedRepr` or `ViewRepr`) is spelled out on both sides
//! of the transmute, not left as a fully abstract `D: Data` bound - so unlike the rest
//! of this module (struct definition, `interpolate`, all written once over generic `D`),
//! `new` itself needs two small `impl` blocks, one per concrete `D`.

use super::*;

/// 1-D interpolator over `uom` quantities: grid points of unit `Qx`, values of unit `Qv`,
/// both backed by storage representation `D` (`OwnedRepr<V>` or `ViewRepr<&'a V>` - see
/// the [`UomInterp1D`]/[`UomInterp1DView`] aliases below).
#[derive(Clone)]
pub struct UomInterp1DBase<D, Qx, Qv, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug + Clone,
    Qx: BaseUnit<D::Elem>,
    Qv: BaseUnit<D::Elem>,
    S: Clone,
{
    inner: Interp1DBase<D, S>,
    _units: PhantomData<fn() -> (Qx, Qv)>,
}

/// Owned variant (see [`UomInterp1DBase`] for the generic form).
pub type UomInterp1D<Qx, Qv, V, S> = UomInterp1DBase<OwnedRepr<V>, Qx, Qv, S>;
/// Viewed variant (see [`UomInterp1DBase`] for the generic form).
pub type UomInterp1DView<'a, Qx, Qv, V, S> = UomInterp1DBase<ViewRepr<&'a V>, Qx, Qv, S>;

impl<'a, Qx, Qv, V, S> UomInterp1DView<'a, Qx, Qv, V, S>
where
    Qx: BaseUnit<V>,
    Qv: BaseUnit<V>,
    V: Num + PartialOrd + Euclid + Copy + Debug + 'a,
    S: Strategy1D<ViewRepr<&'a V>> + Clone,
{
    /// Construct a viewed (borrowed, zero-copy) interpolator over `uom` quantity arrays.
    pub fn new(
        x: ArrayView1<'a, Qx>,
        f_x: ArrayView1<'a, Qv>,
        strategy: S,
        extrapolate: Extrapolate<V>,
    ) -> Result<Self, ValidateError> {
        // SAFETY: see module docs - `Qx`/`Qv` are `uom` quantities backed by `V`,
        // `#[repr(transparent)]` over it, so reinterpreting the view's element type as
        // `V` is sound and the view's shape/strides are unaffected.
        let x: ArrayView1<'a, V> = unsafe { mem::transmute::<ArrayView1<'a, Qx>, _>(x) };
        let f_x: ArrayView1<'a, V> = unsafe { mem::transmute::<ArrayView1<'a, Qv>, _>(f_x) };
        Ok(Self {
            inner: Interp1DView::new(x, f_x, strategy, extrapolate)?,
            _units: PhantomData,
        })
    }
}

impl<Qx, Qv, V, S> UomInterp1D<Qx, Qv, V, S>
where
    Qx: BaseUnit<V>,
    Qv: BaseUnit<V>,
    V: Num + PartialOrd + Euclid + Copy + Debug,
    S: Strategy1D<OwnedRepr<V>> + Clone,
{
    /// Construct an owned interpolator over `uom` quantity arrays.
    pub fn new(
        x: Array1<Qx>,
        f_x: Array1<Qv>,
        strategy: S,
        extrapolate: Extrapolate<V>,
    ) -> Result<Self, ValidateError> {
        // SAFETY: same reasoning as the view constructor above, applied to owned storage:
        // `Array1<A>` is `Vec`-shaped regardless of `A`, so the transmute only changes the
        // element type, not the container's own layout. Dropping is unaffected too -
        // `Quantity<D, U, V>` has no `Drop` of its own, so dropping it is exactly dropping
        // the wrapped `V`.
        let x: Array1<V> = unsafe { mem::transmute::<Array1<Qx>, _>(x) };
        let f_x: Array1<V> = unsafe { mem::transmute::<Array1<Qv>, _>(f_x) };
        Ok(Self {
            inner: Interp1D::new(x, f_x, strategy, extrapolate)?,
            _units: PhantomData,
        })
    }
}

impl<D, Qx, Qv, S> UomInterp1DBase<D, Qx, Qv, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Num + PartialOrd + Euclid + Copy + Debug,
    Qx: BaseUnit<D::Elem>,
    Qv: BaseUnit<D::Elem>,
    S: Strategy1D<D> + Clone,
{
    /// Interpolate at `point`, returning a value in `Qv`.
    pub fn interpolate(&self, point: Qx) -> Result<Qv, InterpolateError> {
        self.inner
            .interpolate(&[point.to_base()])
            .map(Qv::from_base)
    }
}
