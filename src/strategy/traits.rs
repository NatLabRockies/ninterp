//! Strategy trait definitions for all dimensionalities.

use super::*;

/// Generates a fixed-dimensionality strategy trait (`Strategy1D`/`2D`/`3D`) plus its
/// `impl … for Box<dyn …>` forwarding impl. `StrategyND` takes points as `&[D::Elem]`
/// instead of `&[D::Elem; N]`, so it can't share this shape and stays hand-written below.
macro_rules! fixed_strategy_trait {
    ($Trait:ident, $InterpData:ident, $N:literal, $doc:literal) => {
        #[doc = $doc]
        pub trait $Trait<D>: Debug + DynClone
        where
            D: Data + RawDataClone + Clone,
            D::Elem: PartialEq + Debug,
        {
            /// Validate strategy state against interpolation data. Pure check, no mutation.
            ///
            /// Default no-op. Override for invariant checks that don't require precomputed
            /// state (grid uniformity, direction-count matching, etc).
            fn validate(&self, _data: &$InterpData<D>) -> Result<(), ValidateError> {
                Ok(())
            }

            #[doc = concat!(
                " Initialize/recompute cached derived state, with access to interpolation data.\n",
                "\n",
                " Default no-op. Override only when the strategy caches something derived from\n",
                " `data` (e.g. precomputed spline coefficients). Unlike [`", stringify!($Trait), "::validate`],\n",
                " this may do real, non-trivial calculation.",
            )]
            fn init(&mut self, _data: &$InterpData<D>) -> Result<(), ValidateError> {
                Ok(())
            }

            /// Execute interpolation (after handling [`Extrapolate`](`crate::interpolator::Extrapolate`) setting).
            ///
            /// # Note for custom strategies
            /// Index `data.grid[i]` directly via `ArrayView` indexing; avoid `.as_slice()`, which
            /// panics on non-contiguous storage (possible with `Interp*View`). See
            /// [`crate::strategy::utils`] for ready-made per-axis search helpers (bracket search,
            /// exact-match short-circuit, step-direction lookup, uniform-grid fast path) built from
            /// the same primitives the built-in strategies use.
            fn interpolate(
                &self,
                data: &$InterpData<D>,
                point: &[D::Elem; $N],
            ) -> Result<D::Elem, InterpolateError>;

            /// Resolves an out-of-bounds point under [`Extrapolate::Wrap`](`crate::interpolator::Extrapolate::Wrap`), then interpolates.
            ///
            /// Default wraps in `data.grid`'s own (raw) coordinate space. Override only if
            /// this strategy's working coordinate space differs from `data.grid`'s.
            fn interpolate_wrapped(
                &self,
                data: &$InterpData<D>,
                point: &[D::Elem; $N],
            ) -> Result<D::Elem, InterpolateError>
            where
                D::Elem: Num + Euclid + Copy,
            {
                let wrapped = core::array::from_fn(|i| {
                    wrap(
                        point[i],
                        *data.grid[i].first().unwrap(),
                        *data.grid[i].last().unwrap(),
                    )
                });
                self.interpolate(data, &wrapped)
            }

            #[doc = concat!(
                " Interpolate without the `Result` wrapper, assuming `point` and `data` are\n",
                " already valid.\n",
                "\n",
                " Default just unwraps [`", stringify!($Trait), "::interpolate`]. Override only if this\n",
                " strategy's checked path does real internal fallible work beyond producing\n",
                " the final `Ok(...)`; otherwise the default already compiles to the same thing.",
            )]
            #[inline]
            fn interpolate_fast(&self, data: &$InterpData<D>, point: &[D::Elem; $N]) -> D::Elem {
                self.interpolate(data, point)
                    .expect("interpolate_fast: invalid point or data")
            }

            #[doc = concat!(
                " Interpolate at each of several points, writing results into `out` instead of\n",
                " allocating. `out.len()` must equal `points.len()`.\n",
                "\n",
                " Default loops [`", stringify!($Trait), "::interpolate`] into `out`; no allocation. Override\n",
                " only if locating a point in the grid can be amortized across the batch (e.g. sorting\n",
                " points once for a locate sweep instead of one binary search per point); no strategy\n",
                " shipped in this crate does that today.",
            )]
            fn batch_interpolate_into(
                &self,
                data: &$InterpData<D>,
                points: &[[D::Elem; $N]],
                out: &mut [D::Elem],
            ) -> Result<(), InterpolateError> {
                if out.len() != points.len() {
                    return Err(InterpolateError::OutputLength {
                        expected: points.len(),
                        found: out.len(),
                    });
                }
                for (o, point) in out.iter_mut().zip(points) {
                    *o = self.interpolate(data, point)?;
                }
                Ok(())
            }

            #[doc = concat!(
                " Unchecked [`", stringify!($Trait), "::batch_interpolate_into`].\n",
                "\n",
                " # Panics\n",
                " Panics if `out.len() != points.len()`, or under the same conditions as\n",
                " [`", stringify!($Trait), "::interpolate_fast`].",
            )]
            fn batch_interpolate_fast_into(
                &self,
                data: &$InterpData<D>,
                points: &[[D::Elem; $N]],
                out: &mut [D::Elem],
            ) {
                assert_eq!(out.len(), points.len(), "batch_interpolate_fast_into: length mismatch");
                for (o, point) in out.iter_mut().zip(points) {
                    *o = self.interpolate_fast(data, point);
                }
            }

            #[doc = concat!(
                " Interpolate at each of several points, sharing one grid across all of them.\n",
                "\n",
                " Default allocates an output buffer and calls [`", stringify!($Trait), "::batch_interpolate_into`].\n",
                " Do not override this method; override [`", stringify!($Trait), "::batch_interpolate_into`] instead\n",
                " if you need to amortize locating a point across the batch (e.g. sorting points once\n",
                " for a locate sweep instead of one binary search per point).",
            )]
            fn batch_interpolate(
                &self,
                data: &$InterpData<D>,
                points: &[[D::Elem; $N]],
            ) -> Result<Vec<D::Elem>, InterpolateError>
            where
                D::Elem: Num,
            {
                let mut out = Vec::with_capacity(points.len());
                for _ in 0..points.len() {
                    out.push(D::Elem::zero());
                }
                self.batch_interpolate_into(data, points, &mut out)?;
                Ok(out)
            }

            #[doc = concat!(
                " Batched [`", stringify!($Trait), "::interpolate_fast`], assuming every point and `data` are\n",
                " already valid.\n",
                "\n",
                " Default allocates an output buffer and calls [`", stringify!($Trait), "::batch_interpolate_fast_into`].\n",
                " Do not override this method. Override [`", stringify!($Trait), "::batch_interpolate_into`] instead\n",
                " if you need to amortize per-point work; the fast variant inherits those optimizations with no\n",
                " additional override needed.",
            )]
            fn batch_interpolate_fast(
                &self,
                data: &$InterpData<D>,
                points: &[[D::Elem; $N]],
            ) -> Vec<D::Elem>
            where
                D::Elem: Num + Copy,
            {
                let mut out = Vec::with_capacity(points.len());
                for _ in 0..points.len() {
                    out.push(D::Elem::zero());
                }
                self.batch_interpolate_fast_into(data, points, &mut out);
                out
            }

            #[doc = concat!(
                " Domain-checks a batch of points against this strategy's own validity domain\n",
                " (distinct from `data.grid`'s bounds), aggregating every violation across the\n",
                " whole batch instead of just the first.\n",
                "\n",
                " Default no-op: most strategies accept any `D::Elem`. Override only if `self`\n",
                " (or a wrapped inner strategy) restricts the domain further, e.g.\n",
                " [`GridTransform`]'s configured [`Transform`]. Called as a pre-scan\n",
                " before doing any actual interpolation work, so it must stay cheap.",
            )]
            fn check_batch_domain(
                &self,
                _points: &[[D::Elem; $N]],
            ) -> Result<(), InterpolateError> {
                Ok(())
            }

            #[doc = concat!(
                " Does this type's [`", stringify!($Trait), "::interpolate`] provision for extrapolation?",
            )]
            fn allow_extrapolate(&self) -> bool;
        }

        clone_trait_object!(<D> $Trait<D>);

        impl<D> $Trait<D> for Box<dyn $Trait<D>>
        where
            D: Data + RawDataClone + Clone,
            D::Elem: PartialEq + Debug,
        {
            #[inline]
            fn validate(&self, data: &$InterpData<D>) -> Result<(), ValidateError> {
                (**self).validate(data)
            }

            #[inline]
            fn init(&mut self, data: &$InterpData<D>) -> Result<(), ValidateError> {
                (**self).init(data)
            }

            #[inline]
            fn interpolate(
                &self,
                data: &$InterpData<D>,
                point: &[D::Elem; $N],
            ) -> Result<D::Elem, InterpolateError> {
                (**self).interpolate(data, point)
            }

            #[inline]
            fn interpolate_wrapped(
                &self,
                data: &$InterpData<D>,
                point: &[D::Elem; $N],
            ) -> Result<D::Elem, InterpolateError>
            where
                D::Elem: Num + Euclid + Copy,
            {
                (**self).interpolate_wrapped(data, point)
            }

            #[inline]
            fn interpolate_fast(&self, data: &$InterpData<D>, point: &[D::Elem; $N]) -> D::Elem {
                (**self).interpolate_fast(data, point)
            }

            #[inline]
            fn batch_interpolate_into(
                &self,
                data: &$InterpData<D>,
                points: &[[D::Elem; $N]],
                out: &mut [D::Elem],
            ) -> Result<(), InterpolateError> {
                (**self).batch_interpolate_into(data, points, out)
            }

            #[inline]
            fn batch_interpolate_fast_into(
                &self,
                data: &$InterpData<D>,
                points: &[[D::Elem; $N]],
                out: &mut [D::Elem],
            ) {
                (**self).batch_interpolate_fast_into(data, points, out)
            }

            #[inline]
            fn batch_interpolate(
                &self,
                data: &$InterpData<D>,
                points: &[[D::Elem; $N]],
            ) -> Result<Vec<D::Elem>, InterpolateError>
            where
                D::Elem: Num,
            {
                (**self).batch_interpolate(data, points)
            }

            #[inline]
            fn batch_interpolate_fast(
                &self,
                data: &$InterpData<D>,
                points: &[[D::Elem; $N]],
            ) -> Vec<D::Elem>
            where
                D::Elem: Num + Copy,
            {
                (**self).batch_interpolate_fast(data, points)
            }

            #[inline]
            fn check_batch_domain(
                &self,
                points: &[[D::Elem; $N]],
            ) -> Result<(), InterpolateError> {
                (**self).check_batch_domain(points)
            }

            #[inline]
            fn allow_extrapolate(&self) -> bool {
                (**self).allow_extrapolate()
            }
        }
    };
}

fixed_strategy_trait!(
    Strategy1D,
    InterpData1DBase,
    1,
    "1-D interpolation strategy."
);
fixed_strategy_trait!(
    Strategy2D,
    InterpData2DBase,
    2,
    "2-D interpolation strategy."
);
fixed_strategy_trait!(
    Strategy3D,
    InterpData3DBase,
    3,
    "3-D interpolation strategy."
);

/// N-D interpolation strategy.
pub trait StrategyND<D>: Debug + DynClone
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    /// Validate strategy state against interpolation data. Pure check, no mutation.
    ///
    /// Default no-op. Override for invariant checks that don't require precomputed
    /// state (grid uniformity, direction-count matching, etc).
    fn validate(&self, _data: &InterpDataNDBase<D>) -> Result<(), ValidateError> {
        Ok(())
    }

    /// Initialize/recompute cached derived state, with access to interpolation data.
    ///
    /// Default no-op. Override only when the strategy caches something derived from
    /// `data` (e.g. precomputed spline coefficients). Unlike [`StrategyND::validate`],
    /// this may do real, non-trivial calculation.
    fn init(&mut self, _data: &InterpDataNDBase<D>) -> Result<(), ValidateError> {
        Ok(())
    }

    /// Execute interpolation (after handling [`Extrapolate`](`crate::interpolator::Extrapolate`) setting).
    ///
    /// # Note for custom strategies
    /// Index `data.grid[i]` directly via `ArrayView` indexing; avoid `.as_slice()`, which
    /// panics on non-contiguous storage (possible with `Interp*View`). See
    /// [`crate::strategy::utils`] for ready-made per-axis search helpers (bracket search,
    /// exact-match short-circuit, step-direction lookup, uniform-grid fast path) built from
    /// the same primitives the built-in strategies use.
    fn interpolate(
        &self,
        data: &InterpDataNDBase<D>,
        point: &[D::Elem],
    ) -> Result<D::Elem, InterpolateError>;

    /// Resolves an out-of-bounds point under [`Extrapolate::Wrap`](`crate::interpolator::Extrapolate::Wrap`), then interpolates.
    ///
    /// Default wraps in `data.grid`'s own (raw) coordinate space. Override only if
    /// this strategy's working coordinate space differs from `data.grid`'s.
    fn interpolate_wrapped(
        &self,
        data: &InterpDataNDBase<D>,
        point: &[D::Elem],
    ) -> Result<D::Elem, InterpolateError>
    where
        D::Elem: Num + Euclid + Copy,
    {
        let wrapped: Vec<D::Elem> = point
            .iter()
            .enumerate()
            .map(|(dim, &pt)| {
                wrap(
                    pt,
                    *data.grid[dim].first().unwrap(),
                    *data.grid[dim].last().unwrap(),
                )
            })
            .collect();
        self.interpolate(data, &wrapped)
    }

    /// Interpolate without the `Result` wrapper, assuming `point` and `data` are
    /// already valid.
    ///
    /// Default just unwraps [`StrategyND::interpolate`]. Override only if this
    /// strategy's checked path does real internal fallible work beyond producing
    /// the final `Ok(...)`; otherwise the default already compiles to the same thing.
    #[inline]
    fn interpolate_fast(&self, data: &InterpDataNDBase<D>, point: &[D::Elem]) -> D::Elem {
        self.interpolate(data, point)
            .expect("interpolate_fast: invalid point or data")
    }

    /// Interpolate at each of several points, writing results into `out` instead of
    /// allocating. `out.len()` must equal `points.len()`.
    ///
    /// Default loops [`StrategyND::interpolate`] into `out`; no allocation. Override
    /// only if locating a point in the grid can be amortized across the batch (e.g.
    /// sorting points once for a locate sweep instead of one binary search per point);
    /// no strategy shipped in this crate does that today.
    fn batch_interpolate_into(
        &self,
        data: &InterpDataNDBase<D>,
        points: &[&[D::Elem]],
        out: &mut [D::Elem],
    ) -> Result<(), InterpolateError> {
        if out.len() != points.len() {
            return Err(InterpolateError::OutputLength {
                expected: points.len(),
                found: out.len(),
            });
        }
        for (o, point) in out.iter_mut().zip(points) {
            *o = self.interpolate(data, point)?;
        }
        Ok(())
    }

    /// Unchecked [`StrategyND::batch_interpolate_into`].
    ///
    /// # Panics
    /// Panics if `out.len() != points.len()`, or under the same conditions as
    /// [`StrategyND::interpolate_fast`].
    fn batch_interpolate_fast_into(
        &self,
        data: &InterpDataNDBase<D>,
        points: &[&[D::Elem]],
        out: &mut [D::Elem],
    ) {
        assert_eq!(
            out.len(),
            points.len(),
            "batch_interpolate_fast_into: length mismatch"
        );
        for (o, point) in out.iter_mut().zip(points) {
            *o = self.interpolate_fast(data, point);
        }
    }

    /// Interpolate at each of several points, sharing one grid across all of them.
    ///
    /// Default allocates an output buffer and calls [`StrategyND::batch_interpolate_into`].
    /// Do not override this method; override [`StrategyND::batch_interpolate_into`] instead
    /// if you need to amortize locating a point across the batch (e.g. sorting points once
    /// for a locate sweep instead of one binary search per point).
    fn batch_interpolate(
        &self,
        data: &InterpDataNDBase<D>,
        points: &[&[D::Elem]],
    ) -> Result<Vec<D::Elem>, InterpolateError>
    where
        D::Elem: Num,
    {
        let mut out = Vec::with_capacity(points.len());
        for _ in 0..points.len() {
            out.push(D::Elem::zero());
        }
        self.batch_interpolate_into(data, points, &mut out)?;
        Ok(out)
    }

    /// Batched [`StrategyND::interpolate_fast`], assuming every point and `data` are
    /// already valid.
    ///
    /// Default allocates an output buffer and calls [`StrategyND::batch_interpolate_fast_into`].
    /// Do not override this method. Override [`StrategyND::batch_interpolate_into`] instead
    /// if you need to amortize per-point work; the fast variant inherits those optimizations
    /// with no additional override needed.
    fn batch_interpolate_fast(
        &self,
        data: &InterpDataNDBase<D>,
        points: &[&[D::Elem]],
    ) -> Vec<D::Elem>
    where
        D::Elem: Num + Copy,
    {
        let mut out = Vec::with_capacity(points.len());
        for _ in 0..points.len() {
            out.push(D::Elem::zero());
        }
        self.batch_interpolate_fast_into(data, points, &mut out);
        out
    }

    /// Domain-checks a batch of points against this strategy's own validity domain
    /// (distinct from `data.grid`'s bounds), aggregating every violation across the
    /// whole batch instead of just the first.
    ///
    /// Default no-op: most strategies accept any `D::Elem`. Override only if `self`
    /// (or a wrapped inner strategy) restricts the domain further, e.g.
    /// [`GridTransform`]'s configured [`Transform`]. Called as a pre-scan before
    /// doing any actual interpolation work, so it must stay cheap.
    fn check_batch_domain(&self, _points: &[&[D::Elem]]) -> Result<(), InterpolateError> {
        Ok(())
    }

    /// Does this type's [`StrategyND::interpolate`] provision for extrapolation?
    fn allow_extrapolate(&self) -> bool;
}

clone_trait_object!(<D> StrategyND<D>);

impl<D> StrategyND<D> for Box<dyn StrategyND<D>>
where
    D: Data + RawDataClone + Clone,
    D::Elem: PartialEq + Debug,
{
    #[inline]
    fn validate(&self, data: &InterpDataNDBase<D>) -> Result<(), ValidateError> {
        (**self).validate(data)
    }

    #[inline]
    fn init(&mut self, data: &InterpDataNDBase<D>) -> Result<(), ValidateError> {
        (**self).init(data)
    }

    #[inline]
    fn interpolate(
        &self,
        data: &InterpDataNDBase<D>,
        point: &[D::Elem],
    ) -> Result<D::Elem, InterpolateError> {
        (**self).interpolate(data, point)
    }

    #[inline]
    fn allow_extrapolate(&self) -> bool {
        (**self).allow_extrapolate()
    }

    #[inline]
    fn check_batch_domain(&self, points: &[&[D::Elem]]) -> Result<(), InterpolateError> {
        (**self).check_batch_domain(points)
    }

    #[inline]
    fn interpolate_wrapped(
        &self,
        data: &InterpDataNDBase<D>,
        point: &[D::Elem],
    ) -> Result<D::Elem, InterpolateError>
    where
        D::Elem: Num + Euclid + Copy,
    {
        (**self).interpolate_wrapped(data, point)
    }

    #[inline]
    fn interpolate_fast(&self, data: &InterpDataNDBase<D>, point: &[D::Elem]) -> D::Elem {
        (**self).interpolate_fast(data, point)
    }

    #[inline]
    fn batch_interpolate_into(
        &self,
        data: &InterpDataNDBase<D>,
        points: &[&[D::Elem]],
        out: &mut [D::Elem],
    ) -> Result<(), InterpolateError> {
        (**self).batch_interpolate_into(data, points, out)
    }

    #[inline]
    fn batch_interpolate_fast_into(
        &self,
        data: &InterpDataNDBase<D>,
        points: &[&[D::Elem]],
        out: &mut [D::Elem],
    ) {
        (**self).batch_interpolate_fast_into(data, points, out)
    }

    #[inline]
    fn batch_interpolate(
        &self,
        data: &InterpDataNDBase<D>,
        points: &[&[D::Elem]],
    ) -> Result<Vec<D::Elem>, InterpolateError>
    where
        D::Elem: Num,
    {
        (**self).batch_interpolate(data, points)
    }

    #[inline]
    fn batch_interpolate_fast(
        &self,
        data: &InterpDataNDBase<D>,
        points: &[&[D::Elem]],
    ) -> Vec<D::Elem>
    where
        D::Elem: Num + Copy,
    {
        (**self).batch_interpolate_fast(data, points)
    }
}
