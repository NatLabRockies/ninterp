use ninterp::prelude::*;

use ninterp::data::InterpData2D;
use ninterp::strategy::traits::Strategy2D;

// Note: ninterp also re-exposes the internally used `ndarray` crate
// `use ninterp::ndarray;`
use ndarray::prelude::*;
use ndarray::{Data, RawDataClone};

// Debug and Clone must be derived for custom strategies
#[derive(Debug, Clone)]
struct CustomStrategy;

// Implement strategy for 2-D f32 interpolation
impl<D> Strategy2D<D> for CustomStrategy
where
    // Implement for any 2-D interpolator where the contained type is `f32`
    // e.g. `Array2<f32>`, `ArrayView2<f32>`, `CowArray<<'a, f32>, Ix2>`, etc.
    // For a more generic bound, consider introducing a bound for D::Elem
    // e.g. D::Elem: num_traits::Num + PartialOrd
    //
    // Note: when reading grid coordinates in `interpolate`, index `data.grid[i]` directly
    // via ArrayView indexing (e.g. `data.grid[i][idx]`), not `.as_slice()`, which panics
    // on non-contiguous storage. `Interp*Viewed` can produce that from a strided slice.
    // `ninterp::strategy::utils` has ready-made per-axis search helpers (bracket search,
    // exact-match short-circuit, step-direction lookup, uniform-grid fast path) built from
    // the same primitives the built-in strategies use.
    D: Data<Elem = f32> + RawDataClone + Clone,
{
    // We can optionally define an initialization step, useful for strategies that need precalculation.
    // This is called on construction (`new`) and whenever the strategy is swapped (`set_strategy`),
    // for all interpolators. It takes a mutable reference, so you can edit any data contained in
    // `CustomStrategy`.
    //
    // There is a default implementation that just returns `Ok(())`, so leave this out if not needed.
    fn init(&mut self, _data: &InterpData2D<D>) -> Result<(), ninterp::error::ValidateError> {
        println!("initialized!");
        Ok(())
    }

    // Returns interpolated value for the supplied point.
    fn interpolate(
        &self,
        _data: &InterpData2D<D>,
        point: &[f32; 2],
    ) -> Result<f32, ninterp::error::InterpolateError> {
        // Dummy interpolation strategy, product of all point components
        //
        // Here we could access the `InterpData2D` (and/or data in `self`) instead,
        // but this is just an example.
        Ok(point.iter().fold(1., |acc, x| acc * x))
    }

    // Disallow extrapolation.
    //
    // Returning `false` will mean a combination of
    // `Extrapolate::Enable` and `CustomStrategy` will fail on validation.
    //
    // Only set this to `true` if the `interpolate` implementation provisions for extrapolation.
    //
    // All extrapolation settings besides `Extrapolate::Enable` are handled before the strategy `interpolate` call.
    // If you need different options for extrapolation beyond 'Enable', use an enum in your `CustomStrategy`.
    fn allow_extrapolate(&self) -> bool {
        false
    }
}

fn main() {
    // type annotation for clarity
    let interp: Interp2DOwned<f32, CustomStrategy> = Interp2D::new(
        array![0., 2.],
        array![0., 4., 8.],
        array![[0., 0., 0.], [0., 0., 0.]],
        CustomStrategy,
        Extrapolate::Error,
    )
    .unwrap();
    // 2 * 3 == 6
    assert_eq!(interp.interpolate(&[2., 3.]).unwrap(), 6.);
}
