//! Walks through implementing a custom strategy from scratch by reimplementing the
//! built-in `Step::lower()` strategy under a different name, `StepLower`. Real
//! `lower`-direction step interpolation: for each axis, snap the query point down to the
//! nearest grid coordinate at or below it (floor), then read the value stored at that
//! grid cell. See [`ninterp::strategy::Step`] for the built-in, configurable version this
//! is modeled on (it supports `Upper` as well as per-axis directions; `StepLower` only
//! ever floors).

use ninterp::prelude::*;

use ninterp::data::InterpData2DBase;
use ninterp::strategy::step::StepDirection;
use ninterp::strategy::traits::Strategy2D;
use ninterp::strategy::utils::locate_step_index;

// Note: ninterp also re-exposes the internally used `ndarray` crate
// `use ninterp::ndarray;`
use ndarray::prelude::*;
use ndarray::{Data, RawDataClone};

// Debug and Clone must be derived for custom strategies.
//
// No fields are needed here. The built-in `Step` strategy stores a `StepDirection` (or
// one per axis) so callers can choose `Lower` or `Upper` at runtime; `StepLower` bakes
// `Lower` in at the type level instead, so there's nothing to store on `self`. A strategy
// that did need configuration (e.g. our own direction choice, or precomputed state)
// would carry it as fields here.
#[derive(Debug, Clone)]
struct StepLower;

// Implement strategy for 2-D f32 interpolation.
impl<D> Strategy2D<D> for StepLower
where
    // Implement for any 2-D interpolator where the contained type is `f32`
    // e.g. `Array2<f32>`, `ArrayView2<f32>`, `CowArray<<'a, f32>, Ix2>`, etc.
    // For a more generic bound, consider introducing a bound for D::Elem
    // e.g. D::Elem: num_traits::Num + PartialOrd
    //
    // Note: when reading grid coordinates in `interpolate`, index `data.grid[i]` directly
    // via ArrayView indexing (e.g. `data.grid[i][idx]`), not `.as_slice()`, which panics
    // on non-contiguous storage. `Interp*View` can produce that from a strided slice.
    // `ninterp::strategy::utils` has ready-made per-axis search helpers (bracket search,
    // exact-match short-circuit, step-direction lookup, uniform-grid fast path) built from
    // the same primitives the built-in strategies use. Below, we lean on one of those
    // helpers (`locate_step_index`) directly, rather than reimplementing the search by
    // hand, exactly as the built-in `Step` strategy itself does.
    D: Data<Elem = f32> + RawDataClone + Clone,
{
    // We can optionally define a validation step for invariants that don't require
    // precomputed state (e.g. grid uniformity, direction-count matching). It's a pure
    // `&self` check, no mutation allowed, and runs before `init` below: on construction
    // (`new`), whenever the strategy is swapped (`set_strategy`), and whenever
    // `Interpolator::validate` is called on the interpolator directly.
    //
    // There is a default implementation that just returns `Ok(())`, so leave this out if
    // not needed. Real step interpolation has no such requirement (it works on any grid,
    // uniform or not), so this override exists purely to demonstrate the hook; a
    // production `StepLower` could delete it entirely.
    fn validate(&self, data: &InterpData2DBase<D>) -> Result<(), ninterp::error::ValidateError> {
        // Dummy invariant: reject non-uniformly-spaced grids. `strategy::utils` has a
        // ready-made helper for exactly this, used by the built-in `LinearUniform`
        // strategy. Swap in whatever check your own strategy actually needs.
        ninterp::strategy::utils::validate_uniform_grid_epsilon(data.grid[0].view(), 0, None)?;
        ninterp::strategy::utils::validate_uniform_grid_epsilon(data.grid[1].view(), 1, None)?;
        println!("validated!");
        Ok(())
    }

    // We can also optionally define an initialization step, useful for strategies that
    // need precalculation. It takes a mutable reference, so unlike `validate`, you can
    // edit any data contained in `StepLower` (e.g. cache something derived from `data`,
    // such as spline coefficients). It runs after `validate` above, at the same two
    // points: on construction (`new`) and whenever the strategy is swapped
    // (`set_strategy`). Unlike `validate`, it is not re-run by a standalone
    // `Interpolator::validate` call.
    //
    // There is a default implementation that just returns `Ok(())`, so leave this out if
    // not needed. `StepLower` has no state worth precomputing (each lookup is already an
    // O(log n) search per axis), so this override exists only to show where `init` sits
    // in the lifecycle relative to `validate` and `interpolate`.
    fn init(&mut self, _data: &InterpData2DBase<D>) -> Result<(), ninterp::error::ValidateError> {
        println!("initialized!");
        Ok(())
    }

    // Returns the interpolated value for the supplied point.
    fn interpolate(
        &self,
        data: &InterpData2DBase<D>,
        point: &[f32; 2],
    ) -> Result<f32, ninterp::error::InterpolateError> {
        // For each axis independently: find the index of the grid coordinate at or
        // below `point[axis]` ("Lower"), then read `data.values` at the resulting
        // `(i, j)` cell. No blending between neighbors, unlike `Linear`: the whole point
        // of step interpolation is that the result is piecewise-constant.
        //
        // `locate_step_index` is the exact per-axis primitive the built-in `Step`
        // strategy calls (see `src/interpolator/two/strategies.rs`); it also handles the
        // exact-grid-point edge cases (e.g. `point` landing precisely on the last grid
        // coordinate) that a naive floor search gets wrong. We just fix the direction to
        // `Lower` on every axis, instead of taking it as a per-dimension parameter.
        let i = locate_step_index(StepDirection::Lower, data.grid[0].view(), &point[0]);
        let j = locate_step_index(StepDirection::Lower, data.grid[1].view(), &point[1]);
        Ok(data.values[[i, j]])
    }

    // We can also override the batched entry point instead of relying on the default,
    // which just loops `interpolate` once per point. A real strategy might use this to
    // amortize per-call setup across the batch (e.g. sorting points once for a single
    // locate sweep instead of one binary search per point). A strategy with per-cell
    // coefficients (e.g. a cubic spline solving for derivatives on each grid cell) could
    // similarly cache those coefficients here, keyed by cell index, so repeated queries
    // into the same cell across a batch reuse the solve instead of redoing it per point.
    // `StepLower` has nothing to amortize or cache since every point's lookup is already
    // an independent, cheap search, so this override just demonstrates the shape and
    // prints how many points came through.
    //
    // There is a default implementation that loops `interpolate`, so leave this out if
    // there's nothing to amortize.
    fn batch_interpolate_into(
        &self,
        data: &InterpData2DBase<D>,
        points: &[[f32; 2]],
        out: &mut [f32],
    ) -> Result<(), ninterp::error::InterpolateError> {
        if out.len() != points.len() {
            return Err(ninterp::error::InterpolateError::OutputLength {
                expected: points.len(),
                found: out.len(),
            });
        }
        println!("batch interpolating {} points!", points.len());
        for (o, point) in out.iter_mut().zip(points) {
            let i = locate_step_index(StepDirection::Lower, data.grid[0].view(), &point[0]);
            let j = locate_step_index(StepDirection::Lower, data.grid[1].view(), &point[1]);
            *o = data.values[[i, j]];
        }
        Ok(())
    }

    // Disallow extrapolation.
    //
    // Returning `false` means a combination of `Extrapolate::Enable` and `StepLower` will
    // fail on validation.
    //
    // Only set this to `true` if the `interpolate` implementation provisions for
    // extrapolation. `locate_step_index` already clamps out-of-range points to the
    // nearest grid edge, so `StepLower` could support `Extrapolate::Enable` by returning
    // `true` here. We leave it `false` to mirror the built-in `Step`, which treats
    // extrapolation as something the caller must opt into explicitly, rather than
    // silently falling out of a floor search.
    //
    // All extrapolation settings besides `Extrapolate::Enable` are handled before the
    // strategy's `interpolate` call. `Extrapolate::Enable` itself carries no
    // configuration, so if your strategy needs multiple extrapolation behaviors, model
    // them as a field on `StepLower` (e.g. an enum) and branch on it inside `interpolate`.
    fn allow_extrapolate(&self) -> bool {
        false
    }
}

fn main() {
    // A 3x3 grid where each value encodes its own (row, col) as `row * 10 + col`, so the
    // expected result of every lookup below is easy to read off directly.
    //
    //        y=0   y=4   y=8
    // x=0  [  0.,   1.,   2. ]
    // x=2  [ 10.,  11.,  12. ]
    // x=4  [ 20.,  21.,  22. ]
    let interp: Interp2D<f32, StepLower> = Interp2D::new(
        array![0., 2., 4.],
        array![0., 4., 8.],
        array![[0., 1., 2.], [10., 11., 12.], [20., 21., 22.]],
        StepLower,
        Extrapolate::Error,
    )
    .unwrap(); // if we had a non-uniform grid, validation would fail here

    // (3., 5.) sits strictly inside the top-right cell: 2. <= 3. < 4. on the x-axis
    // and 4. <= 5. < 8. on the y-axis. Flooring each axis lands on grid coordinate
    // (2., 4.), i.e. row 1, col 1, so the result is 11.
    assert_eq!(interp.interpolate(&[3., 5.]).unwrap(), 11.);

    // An exact grid point floors to itself: (4., 8.) is row 2, col 2, so the result is 22.
    assert_eq!(interp.interpolate(&[4., 8.]).unwrap(), 22.);

    // `batch_interpolate` funnels every point into the `batch_interpolate_into` override
    // above in one call, instead of invoking `interpolate` once per point.
    let ys = interp.batch_interpolate(&[[3., 5.], [4., 8.]]).unwrap();
    assert_eq!(ys, vec![11., 22.]);
}
