//! N-dimensional interpolation

use super::*;

use ndarray::prelude::*;

mod strategies;
#[cfg(test)]
mod tests;

/// Interpolator data for N-dimensional interpolators, where N can vary at runtime.
///
/// Split into a grid storage type `Dg` and a value storage type `Dv` (ninterp's
/// Tg/Tv split, prototyped here first, see issue #57). `InterpDataND<T>` /
/// `InterpND<T, S>` cover the common same-type case (`Dg::Elem == Dv::Elem == T`);
/// reach for `InterpDataNDBase<OwnedRepr<Tg>, OwnedRepr<Tv>>` directly when they
/// should genuinely differ.
///
/// See [`InterpDataBase`] and its dimension-specific aliases
/// for concrete-dimensionality interpolator data structs.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "Dg::Elem: Serialize, Dv::Elem: Serialize",
        deserialize = "
            Dg: DataOwned,
            Dg::Elem: Deserialize<'de>,
            Dv: DataOwned,
            Dv::Elem: Deserialize<'de>,
        "
    ))
)]
pub struct InterpDataNDBase<Dg, Dv>
where
    Dg: Data + RawDataClone + Clone,
    Dg::Elem: PartialEq + Debug,
    Dv: Data + RawDataClone + Clone,
    Dv::Elem: PartialEq + Debug,
{
    /// Coordinate grid: a vector of 1-dimensional [`ArrayBase<Dg, Ix1>`].
    #[cfg_attr(feature = "serde", serde(deserialize_with = "deserialize_grid_vec"))]
    pub grid: Vec<ArrayBase<Dg, Ix1>>,
    /// Function values at coordinates: a single dynamic-dimensional [`ArrayBase`].
    #[cfg_attr(feature = "serde", serde(deserialize_with = "deserialize_dyn"))]
    pub values: ArrayBase<Dv, IxDyn>,
}
/// Owned data variant for N-D data, same-type convenience (see [`InterpDataNDBase`]
/// for the general grid/value-split form).
pub type InterpDataND<T> = InterpDataNDBase<OwnedRepr<T>, OwnedRepr<T>>;
/// Viewed data variant for N-D data, same-type convenience (see [`InterpDataNDBase`]
/// for the general grid/value-split form).
pub type InterpDataNDView<T> = InterpDataNDBase<ViewRepr<T>, ViewRepr<T>>;

#[cfg(feature = "serde")]
impl<Dg, Dv> SerializeNested for InterpDataNDBase<Dg, Dv>
where
    Dg: Data + RawDataClone + Clone,
    Dg::Elem: PartialEq + Debug + Serialize,
    Dv: Data + RawDataClone + Clone,
    Dv::Elem: PartialEq + Debug + Serialize,
{
    fn serialize_nested<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = serializer.serialize_struct("InterpDataNDBase", 2)?;
        s.serialize_field("grid", &GridVecWrapper(&self.grid))?;
        s.serialize_field("values", &ArrayWrapper(&self.values))?;
        s.end()
    }
}

impl<Dg, Dv> PartialEq for InterpDataNDBase<Dg, Dv>
where
    Dg: Data + RawDataClone + Clone,
    Dg::Elem: PartialEq + Debug,
    Dv: Data + RawDataClone + Clone,
    Dv::Elem: PartialEq + Debug,
    ArrayBase<Dg, Ix1>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.grid == other.grid && self.values == other.values
    }
}

impl<Dg, Dv> InterpDataNDBase<Dg, Dv>
where
    Dg: Data + RawDataClone + Clone,
    Dg::Elem: PartialEq + Debug,
    Dv: Data + RawDataClone + Clone,
    Dv::Elem: PartialEq + Debug,
{
    /// Construct and validate a new [`InterpDataND`].
    pub fn new(
        grid: Vec<ArrayBase<Dg, Ix1>>,
        values: ArrayBase<Dv, IxDyn>,
    ) -> Result<Self, ValidateError>
    where
        Dg::Elem: PartialOrd,
    {
        let data = Self { grid, values };
        data.validate()?;
        Ok(data)
    }

    /// Validate interpolator data.
    pub fn validate(&self) -> Result<(), ValidateError>
    where
        Dg::Elem: PartialOrd,
    {
        let n = self.ndim();
        if (self.grid.len() != n) && !(n == 0 && self.grid.iter().all(|g| g.is_empty())) {
            // Only possible for `InterpDataND`
            return Err(ValidateError::GridAxisCount {
                expected: n,
                found: self.grid.len(),
            });
        }
        for i in 0..n {
            let i_grid_len = self.grid[i].len();
            // Every strategy needs at least 2 points per dimension to bracket a query point
            if i_grid_len < 2 {
                return Err(ValidateError::InsufficientGridLength(i));
            }
            // Check that grid points are strictly increasing; a repeated coordinate would
            // give a zero-width interval, dividing by zero in any strategy that computes a
            // fractional position or slope across it (e.g. `Linear`'s `frac`, `CubicC2`'s
            // `compute_m`), silently producing NaN/Inf instead of a validation error.
            if !self.grid[i].windows(2).into_iter().all(|w| w[0] < w[1]) {
                return Err(ValidateError::NotStrictlyIncreasing(i));
            }
            // Check that grid and values are compatible shapes
            if i_grid_len != self.values.shape()[i] {
                return Err(ValidateError::IncompatibleShapes(i));
            }
        }
        Ok(())
    }

    /// Get data dimensionality.
    pub fn ndim(&self) -> usize {
        if self.values.len() == 1 {
            0
        } else {
            self.values.ndim()
        }
    }

    /// View interpolator data.
    pub fn view(&self) -> InterpDataNDBase<ViewRepr<&Dg::Elem>, ViewRepr<&Dv::Elem>> {
        InterpDataNDBase {
            grid: self.grid.iter().map(|g| g.view()).collect(),
            values: self.values.view(),
        }
    }

    /// Turn the data into an [`InterpDataND`], cloning the array elements if necessary.
    pub fn into_owned(self) -> InterpDataNDBase<OwnedRepr<Dg::Elem>, OwnedRepr<Dv::Elem>>
    where
        Dg::Elem: Clone,
        Dv::Elem: Clone,
    {
        InterpDataNDBase {
            grid: self.grid.into_iter().map(|g| g.into_owned()).collect(),
            values: self.values.into_owned(),
        }
    }
}

/// N-D interpolator
///
/// Split into a grid storage type `Dg` and a value storage type `Dv` (ninterp's
/// Tg/Tv split, prototyped here first, see issue #57). `InterpND<T, S>` covers the
/// common same-type case (`Dg::Elem == Dv::Elem == T`), and is the only shape that
/// implements [`Interpolator<T>`]: the general `Dg != Dv` case is reached through
/// the inherent methods below instead, since `Interpolator<T>`'s single type
/// parameter can't express a split grid/value type. Reach for
/// `InterpNDBase<OwnedRepr<Tg>, OwnedRepr<Tv>, S>` directly when they should
/// genuinely differ.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "
            Dg::Elem: Serialize,
            Dv::Elem: Serialize,
            S: Serialize,
        ",
        deserialize = "
            Dg: DataOwned,
            Dg::Elem: Deserialize<'de>,
            Dv: DataOwned,
            Dv::Elem: Deserialize<'de>,
            S: Deserialize<'de>,
        "
    ))
)]
pub struct InterpNDBase<Dg, Dv, S>
where
    Dg: Data + RawDataClone + Clone,
    Dg::Elem: PartialEq + Debug,
    Dv: Data + RawDataClone + Clone,
    Dv::Elem: PartialEq + Debug,
    S: Clone,
{
    /// Interpolator data.
    pub data: InterpDataNDBase<Dg, Dv>,
    /// Interpolation strategy.
    pub strategy: S,
    /// Extrapolation setting.
    pub extrapolate: Extrapolate<Dv::Elem>,
}
/// Owned interpolator variant, same-type convenience (see [`InterpNDBase`] for the
/// general grid/value-split form).
pub type InterpND<T, S> = InterpNDBase<OwnedRepr<T>, OwnedRepr<T>, S>;
/// Viewed interpolator variant, same-type convenience (see [`InterpNDBase`] for the
/// general grid/value-split form).
pub type InterpNDView<T, S> = InterpNDBase<ViewRepr<T>, ViewRepr<T>, S>;

// The shared macros below (`partialeq_impl!`, `serialize_nested_impl!`,
// `extrapolate_impl!`, `set_strategy_box_impl!`, `set_strategy_enum_impl!`,
// `any_interpolator_impl!`) are single-`D` and shared with `Interp1D`/`2D`/`3D`, so
// reusing them here would force a signature change onto types this prototype isn't
// touching. Left commented out (not deleted): when `Interp1D`/`2D`/`3D` eventually
// follow the same Tg/Tv split, the macros themselves get updated to take `Dg`/`Dv`,
// and these call sites are what make that swap-back a diff instead of a rewrite.
//
// partialeq_impl!(InterpNDBase, InterpDataNDBase, StrategyND);
// #[cfg(feature = "serde")]
// serialize_nested_impl!(InterpNDBase, InterpDataNDBase, StrategyND);
// extrapolate_impl!(InterpNDBase, StrategyND);
// set_strategy_box_impl!(InterpNDBase, StrategyND);
// set_strategy_enum_impl!(
//     InterpNDBase,
//     strategy::enums::StrategyNDEnum,
//     strategy::enums::StrategyNDEnum<D::Elem>
// );
// any_interpolator_impl!(InterpND, StrategyND);

impl<Dg, Dv, S> PartialEq for InterpNDBase<Dg, Dv, S>
where
    Dg: Data + RawDataClone + Clone,
    Dg::Elem: PartialEq + Debug,
    Dv: Data + RawDataClone + Clone,
    Dv::Elem: PartialEq + Debug,
    S: Clone + PartialEq,
    InterpDataNDBase<Dg, Dv>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
            && self.strategy == other.strategy
            && self.extrapolate == other.extrapolate
    }
}

#[cfg(feature = "serde")]
impl<Dg, Dv, S> SerializeNested for InterpNDBase<Dg, Dv, S>
where
    Dg: Data + RawDataClone + Clone,
    Dg::Elem: PartialEq + Debug + Serialize,
    Dv: Data + RawDataClone + Clone,
    Dv::Elem: PartialEq + Debug + Serialize,
    S: Clone + Serialize,
    InterpDataNDBase<Dg, Dv>: SerializeNested + Serialize,
{
    fn serialize_nested<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: Serializer,
    {
        let mut s = serializer.serialize_struct("InterpNDBase", 3)?;
        s.serialize_field("data", &Nested(&self.data))?;
        s.serialize_field("strategy", &self.strategy)?;
        s.serialize_field("extrapolate", &self.extrapolate)?;
        s.end()
    }
}

impl<Dg, Dv, S> InterpNDBase<Dg, Dv, S>
where
    Dg: Data + RawDataClone + Clone,
    Dg::Elem: PartialEq + Debug,
    Dv: Data + RawDataClone + Clone,
    Dv::Elem: PartialEq + Debug,
    S: StrategyND<Dg, Dv> + Clone,
{
    /// Check that `extrapolate` is applicable to the current strategy.
    ///
    /// Only [`Extrapolate::Enable`] can be rejected, and only by a strategy whose
    /// `allow_extrapolate` returns `false`. Takes the setting as an argument rather
    /// than reading `self.extrapolate`, so `set_extrapolate` can vet a candidate
    /// before storing it.
    pub fn validate_extrapolate(
        &self,
        extrapolate: &Extrapolate<Dv::Elem>,
    ) -> Result<(), ValidateError> {
        if matches!(extrapolate, Extrapolate::Enable) && !self.strategy.allow_extrapolate() {
            return Err(ValidateError::ExtrapolateUnsupported);
        }
        Ok(())
    }

    /// Re-run the strategy's [`StrategyND::validate`] against the current data.
    ///
    /// `new`, `set_strategy`, and [`Self::validate`] already call this internally,
    /// so this is only needed after mutating the public `data`/`strategy` fields
    /// directly.
    pub fn validate_strategy(&self) -> Result<(), ValidateError> {
        self.strategy.validate(&self.data)
    }

    /// Re-run the strategy's [`StrategyND::init`] against the current data.
    ///
    /// `new` and `set_strategy` already call this internally, so this is only needed
    /// after bypassing them: mutating the public `data`/`strategy` fields directly, or
    /// deserializing an interpolator whose strategy skips its cached state from
    /// serialization. `Deserialize` does not call `init`; if the cached state is
    /// instead stored in ordinary serialized fields, it comes back as-is and this
    /// isn't needed.
    pub fn init_strategy(&mut self) -> Result<(), ValidateError> {
        self.strategy.init(&self.data)
    }
}

impl<Dg, Dv> InterpNDBase<Dg, Dv, Box<dyn StrategyND<Dg, Dv>>>
where
    Dg: Data + RawDataClone + Clone,
    Dg::Elem: PartialEq + Debug,
    Dv: Data + RawDataClone + Clone,
    Dv::Elem: PartialEq + Debug,
{
    /// Update strategy at runtime, calling [`StrategyND::init`] on the new strategy
    /// against the current data.
    ///
    /// To swap in a strategy without re-running `init` (e.g. one whose state was
    /// already established elsewhere), assign the `strategy` field directly instead.
    pub fn set_strategy(
        &mut self,
        strategy: Box<dyn StrategyND<Dg, Dv>>,
    ) -> Result<(), ValidateError> {
        self.strategy = strategy;
        self.validate_extrapolate(&self.extrapolate)?;
        self.validate_strategy()?;
        self.init_strategy()
    }
}

impl<Dg, Dv> InterpNDBase<Dg, Dv, strategy::enums::StrategyNDEnum>
where
    Dg: Data + RawDataClone + Clone,
    Dg::Elem: PartialEq + Debug + NumCast + PartialOrd + Copy,
    Dv: Data + RawDataClone + Clone,
    Dv::Elem: PartialEq + Debug + NumCast + Copy,
{
    /// Update strategy at runtime, calling [`strategy::enums::StrategyNDEnum::init`] on
    /// the new strategy against the current data.
    ///
    /// To swap in a strategy without re-running `init` (e.g. one whose state was
    /// already established elsewhere), assign the `strategy` field directly instead.
    pub fn set_strategy(
        &mut self,
        strategy: impl Into<strategy::enums::StrategyNDEnum>,
    ) -> Result<(), ValidateError> {
        self.strategy = strategy.into();
        self.validate_extrapolate(&self.extrapolate)?;
        self.validate_strategy()?;
        self.init_strategy()
    }
}

impl<T, S> AnyInterpolator<T> for InterpND<T, S>
where
    T: Float + Euclid + Debug + Send + Sync + 'static,
    S: StrategyND<OwnedRepr<T>, OwnedRepr<T>> + Clone + Send + Sync + 'static,
{
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl<Dg, Dv, S> InterpNDBase<Dg, Dv, S>
where
    Dg: Data + RawDataClone + Clone,
    Dg::Elem: PartialOrd + Debug + NumCast + Copy,
    Dv: Data + RawDataClone + Clone,
    Dv::Elem: PartialEq + Debug + Copy,
    S: StrategyND<Dg, Dv> + Clone,
{
    /// Construct and validate an N-D (any dimensionality) interpolator.
    ///
    /// # Example:
    /// ```
    /// use ndarray::prelude::*;
    /// use ninterp::prelude::*;
    /// // f(x, y, z) = 0.2 * x + 0.2 * y + 0.2 * z
    /// let interp = InterpND::new(
    ///     // grid
    ///     vec![
    ///         // x
    ///         array![1., 2.], // x0, x1
    ///         // y
    ///         array![1., 2., 3.], // y0, y1, y2
    ///         // z
    ///         array![1., 2., 3., 4.], // z0, z1, z2, z3
    ///     ],
    ///     // values
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
    ///     ]
    ///     .into_dyn(),
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
        grid: Vec<ArrayBase<Dg, Ix1>>,
        values: ArrayBase<Dv, IxDyn>,
        strategy: S,
        extrapolate: Extrapolate<Dv::Elem>,
    ) -> Result<Self, ValidateError> {
        let mut interpolator = Self {
            data: InterpDataNDBase::new(grid, values)?,
            strategy,
            extrapolate,
        };
        interpolator.validate_extrapolate(&interpolator.extrapolate)?;
        interpolator.validate_strategy()?;
        interpolator.init_strategy()?;
        Ok(interpolator)
    }

    /// Get data dimensionality.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Validate interpolator data.
    pub fn validate(&self) -> Result<(), ValidateError> {
        self.validate_extrapolate(&self.extrapolate)?;
        self.data.validate()?;
        self.validate_strategy()?;
        Ok(())
    }

    /// Set [`Extrapolate`] variant, checking validity.
    pub fn set_extrapolate(
        &mut self,
        extrapolate: Extrapolate<Dv::Elem>,
    ) -> Result<(), ValidateError> {
        self.validate_extrapolate(&extrapolate)?;
        self.extrapolate = extrapolate;
        Ok(())
    }

    /// Casts `data.grid[dim]`'s bounds to `f64`, to compare against a `point`
    /// (always `f64`-typed; see [`StrategyND`]'s docs for why).
    fn grid_bounds_f64(&self, dim: usize) -> (f64, f64) {
        let lo: f64 = num_traits::cast(*self.data.grid[dim].first().unwrap())
            .expect("grid element must cast to f64");
        let hi: f64 = num_traits::cast(*self.data.grid[dim].last().unwrap())
            .expect("grid element must cast to f64");
        (lo, hi)
    }

    /// Interpolate at the supplied point.
    ///
    /// Named `interpolate_f64` (not `interpolate`) to avoid colliding with
    /// [`Interpolator::interpolate`]'s `&[T]`-typed method of the same name on
    /// `InterpND<T, S>` (`Dg = Dv = T`): an inherent method always wins over a
    /// trait method in dot-call resolution regardless of whether argument types
    /// actually match, so a same-named `&[f64]`-typed inherent method here would
    /// silently break `Interpolator<T>::interpolate` for any `T != f64`.
    pub fn interpolate_f64(&self, point: &[f64]) -> Result<Dv::Elem, InterpolateError> {
        let n = self.ndim();
        if point.len() != n {
            return Err(InterpolateError::PointLength {
                expected: n,
                failures: vec![WrongLengthAt {
                    index: 0,
                    found: point.len(),
                }],
            });
        }
        let mut errors = Vec::new();
        for dim in 0..n {
            let (lo, hi) = self.grid_bounds_f64(dim);
            if !(lo..=hi).contains(&point[dim]) {
                match &self.extrapolate {
                    Extrapolate::Enable => {}
                    Extrapolate::Fill(value) => return Ok(*value),
                    Extrapolate::Clamp => {
                        let clamped_point: Vec<f64> = point
                            .iter()
                            .enumerate()
                            .map(|(dim, &pt)| {
                                let (lo, hi) = self.grid_bounds_f64(dim);
                                clamp(pt, lo, hi)
                            })
                            .collect();
                        return self.strategy.interpolate(&self.data, &clamped_point);
                    }
                    Extrapolate::Wrap => {
                        return self.strategy.interpolate_wrapped(&self.data, point);
                    }
                    Extrapolate::Error => {
                        errors.push(OutOfBoundsAt { index: 0, dim });
                    }
                };
            }
        }
        if !errors.is_empty() {
            return Err(InterpolateError::OutOfBounds(errors));
        }
        self.strategy.interpolate(&self.data, point)
    }

    /// Interpolate without bounds/extrapolation checks, for use in hot loops where the
    /// caller has already checked bounds and knows that extrapolation is not needed.
    ///
    /// Named `interpolate_f64_fast`, not `interpolate_fast`; see
    /// [`Self::interpolate_f64`]'s doc for why.
    ///
    /// # Panics
    /// Panics if `point.len()` doesn't match [`Self::ndim`].
    pub fn interpolate_f64_fast(&self, point: &[f64]) -> Dv::Elem {
        assert_eq!(
            point.len(),
            self.ndim(),
            "interpolate_fast: point length mismatch"
        );
        self.strategy.interpolate_fast(&self.data, point)
    }

    /// Interpolate at each of several points, writing results into `out` instead of
    /// allocating.
    ///
    /// `self.extrapolate` is one setting for the whole call, not resolved per point:
    /// every point still funnels into at most one call to the strategy.
    ///
    /// Named `batch_interpolate_f64_into`, not `batch_interpolate_into`; see
    /// [`Self::interpolate_f64`]'s doc for why.
    pub fn batch_interpolate_f64_into(
        &self,
        points: &[&[f64]],
        out: &mut [Dv::Elem],
    ) -> Result<(), InterpolateError> {
        let n = self.ndim();
        let failures: Vec<WrongLengthAt> = points
            .iter()
            .enumerate()
            .filter(|(_, point)| point.len() != n)
            .map(|(i, point)| WrongLengthAt {
                index: i,
                found: point.len(),
            })
            .collect();
        if !failures.is_empty() {
            return Err(InterpolateError::PointLength {
                expected: n,
                failures,
            });
        }
        if out.len() != points.len() {
            return Err(InterpolateError::OutputLength {
                expected: points.len(),
                found: out.len(),
            });
        }
        match &self.extrapolate {
            Extrapolate::Enable => self
                .strategy
                .batch_interpolate_into(&self.data, points, out),
            Extrapolate::Clamp => {
                // Clamping an in-bounds point is already identity, so every point
                // can be clamped unconditionally.
                let clamped: Vec<Vec<f64>> = points
                    .iter()
                    .map(|&point| {
                        point
                            .iter()
                            .enumerate()
                            .map(|(dim, &pt)| {
                                let (lo, hi) = self.grid_bounds_f64(dim);
                                clamp(pt, lo, hi)
                            })
                            .collect()
                    })
                    .collect();
                let clamped: Vec<&[f64]> = clamped.iter().map(Vec::as_slice).collect();
                self.strategy
                    .batch_interpolate_into(&self.data, &clamped, out)
            }
            Extrapolate::Wrap => {
                self.strategy.check_batch_domain(points)?;
                for (o, &point) in out.iter_mut().zip(points) {
                    *o = if self.point_out_of_bounds(point) {
                        self.strategy.interpolate_wrapped(&self.data, point)?
                    } else {
                        self.strategy.interpolate(&self.data, point)?
                    };
                }
                Ok(())
            }
            Extrapolate::Fill(value) => {
                // Pre-fill output with the fill value, then scatter interpolated
                // results from in-bounds points into their corresponding indices.
                for o in out.iter_mut() {
                    *o = *value;
                }
                let mut in_bounds_indices = Vec::new();
                let mut in_bounds_points: Vec<&[f64]> = Vec::new();
                for (i, &point) in points.iter().enumerate() {
                    if !self.point_out_of_bounds(point) {
                        in_bounds_indices.push(i);
                        in_bounds_points.push(point);
                    }
                }
                if !in_bounds_indices.is_empty() {
                    let mut scratch: Vec<Dv::Elem> = in_bounds_indices
                        .iter()
                        .map(|_| out[in_bounds_indices[0]])
                        .collect();
                    self.strategy.batch_interpolate_into(
                        &self.data,
                        &in_bounds_points,
                        &mut scratch,
                    )?;
                    for (idx, value) in in_bounds_indices.into_iter().zip(scratch) {
                        out[idx] = value;
                    }
                }
                Ok(())
            }
            Extrapolate::Error => {
                let mut errors = Vec::new();
                let mut in_bounds_points: Vec<&[f64]> = Vec::new();
                for (i, &point) in points.iter().enumerate() {
                    let mut point_errors = Vec::new();
                    for (dim, &pt) in point.iter().enumerate() {
                        let (lo, hi) = self.grid_bounds_f64(dim);
                        if !(lo..=hi).contains(&pt) {
                            point_errors.push(OutOfBoundsAt { index: i, dim });
                        }
                    }
                    if point_errors.is_empty() {
                        in_bounds_points.push(point);
                    } else {
                        errors.extend(point_errors);
                    }
                }
                if !errors.is_empty() {
                    return Err(InterpolateError::OutOfBounds(errors));
                }
                self.strategy
                    .batch_interpolate_into(&self.data, &in_bounds_points, out)
            }
        }
    }

    /// Is `point` out of `self.data.grid`'s bounds in any dimension?
    fn point_out_of_bounds(&self, point: &[f64]) -> bool {
        (0..point.len()).any(|dim| {
            let (lo, hi) = self.grid_bounds_f64(dim);
            !(lo..=hi).contains(&point[dim])
        })
    }

    /// Interpolate at each of several points, sharing one grid across all of them.
    ///
    /// Named `batch_interpolate_f64`, not `batch_interpolate`; see
    /// [`Self::interpolate_f64`]'s doc for why.
    pub fn batch_interpolate_f64(
        &self,
        points: &[&[f64]],
    ) -> Result<Vec<Dv::Elem>, InterpolateError>
    where
        Dv::Elem: Num,
    {
        let mut out = vec![Dv::Elem::zero(); points.len()];
        self.batch_interpolate_f64_into(points, &mut out)?;
        Ok(out)
    }

    /// Unchecked batched [`Self::interpolate_f64_fast`], assuming every point is valid.
    ///
    /// Named `batch_interpolate_f64_fast_into`, not `batch_interpolate_fast_into`;
    /// see [`Self::interpolate_f64`]'s doc for why.
    ///
    /// # Panics
    /// Panics if `out.len() != points.len()`.
    pub fn batch_interpolate_f64_fast_into(&self, points: &[&[f64]], out: &mut [Dv::Elem]) {
        let n = self.ndim();
        for point in points {
            assert_eq!(
                point.len(),
                n,
                "batch_interpolate_f64_fast_into: point length mismatch"
            );
        }
        assert_eq!(
            out.len(),
            points.len(),
            "batch_interpolate_f64_fast_into: length mismatch"
        );
        self.strategy
            .batch_interpolate_fast_into(&self.data, points, out)
    }

    /// Batched [`Self::interpolate_f64_fast`], for use in hot loops where the caller
    /// has already checked bounds.
    ///
    /// Named `batch_interpolate_f64_fast`, not `batch_interpolate_fast`; see
    /// [`Self::interpolate_f64`]'s doc for why.
    pub fn batch_interpolate_f64_fast(&self, points: &[&[f64]]) -> Vec<Dv::Elem>
    where
        Dv::Elem: Num,
    {
        let mut out = vec![Dv::Elem::zero(); points.len()];
        self.batch_interpolate_f64_fast_into(points, &mut out);
        out
    }

    /// Return an interpolator with viewed data.
    pub fn view(&self) -> InterpNDBase<ViewRepr<&Dg::Elem>, ViewRepr<&Dv::Elem>, S>
    where
        S: for<'a> StrategyND<ViewRepr<&'a Dg::Elem>, ViewRepr<&'a Dv::Elem>>,
        Dg::Elem: Clone,
        Dv::Elem: Clone,
    {
        InterpNDBase {
            data: self.data.view(),
            strategy: self.strategy.clone(),
            extrapolate: self.extrapolate,
        }
    }

    /// Turn the interpolator into an owned variant, cloning the array elements if necessary.
    pub fn into_owned(self) -> InterpNDBase<OwnedRepr<Dg::Elem>, OwnedRepr<Dv::Elem>, S>
    where
        S: StrategyND<OwnedRepr<Dg::Elem>, OwnedRepr<Dv::Elem>>,
        Dg::Elem: Clone,
        Dv::Elem: Clone,
    {
        InterpNDBase {
            data: self.data.clone().into_owned(),
            strategy: self.strategy.clone(),
            extrapolate: self.extrapolate,
        }
    }
}

impl<D, S> Interpolator<D::Elem> for InterpNDBase<D, D, S>
where
    D: Data + RawDataClone + Clone,
    D::Elem: Num + PartialOrd + Euclid + NumCast + Copy + Debug,
    S: StrategyND<D, D> + Clone,
{
    #[inline]
    fn ndim(&self) -> usize {
        self.data.ndim()
    }

    fn validate(&self) -> Result<(), ValidateError> {
        InterpNDBase::validate(self)
    }

    fn interpolate(&self, point: &[D::Elem]) -> Result<D::Elem, InterpolateError> {
        let point: Vec<f64> = point
            .iter()
            .map(|&x| num_traits::cast(x).expect("point element must cast to f64"))
            .collect();
        InterpNDBase::interpolate_f64(self, &point)
    }

    fn set_extrapolate(&mut self, extrapolate: Extrapolate<D::Elem>) -> Result<(), ValidateError> {
        InterpNDBase::set_extrapolate(self, extrapolate)
    }

    fn interpolate_fast(&self, point: &[D::Elem]) -> D::Elem {
        let point: Vec<f64> = point
            .iter()
            .map(|&x| num_traits::cast(x).expect("point element must cast to f64"))
            .collect();
        InterpNDBase::interpolate_f64_fast(self, &point)
    }

    fn batch_interpolate_into(
        &self,
        points: &[&[D::Elem]],
        out: &mut [D::Elem],
    ) -> Result<(), InterpolateError> {
        let points: Vec<Vec<f64>> = points
            .iter()
            .map(|point| {
                point
                    .iter()
                    .map(|&x| num_traits::cast(x).expect("point element must cast to f64"))
                    .collect()
            })
            .collect();
        let points: Vec<&[f64]> = points.iter().map(Vec::as_slice).collect();
        InterpNDBase::batch_interpolate_f64_into(self, &points, out)
    }

    fn batch_interpolate_fast_into(&self, points: &[&[D::Elem]], out: &mut [D::Elem]) {
        let points: Vec<Vec<f64>> = points
            .iter()
            .map(|point| {
                point
                    .iter()
                    .map(|&x| num_traits::cast(x).expect("point element must cast to f64"))
                    .collect()
            })
            .collect();
        let points: Vec<&[f64]> = points.iter().map(Vec::as_slice).collect();
        InterpNDBase::batch_interpolate_f64_fast_into(self, &points, out)
    }

    fn batch_interpolate_fast(&self, points: &[&[D::Elem]]) -> Vec<D::Elem> {
        let n = self.ndim();
        for point in points {
            assert_eq!(
                point.len(),
                n,
                "batch_interpolate_fast: point length mismatch"
            );
        }
        let mut out = vec![D::Elem::zero(); points.len()];
        self.batch_interpolate_fast_into(points, &mut out);
        out
    }
}
