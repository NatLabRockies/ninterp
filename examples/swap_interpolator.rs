use ndarray::prelude::*;

use ninterp::prelude::*;

fn main() {
    using_enum();
    using_boxdyn();
}

/// Use a provided interpolator enum to allow interpolator swapping.
/// - serde compatible
/// - Statically dispatched (faster runtime)
/// - **NOT** compatible with custom strategies
fn using_enum() {
    // Create `InterpolatorEnum`
    let mut interp = InterpolatorEnum::new_2d(
        array![0., 1.],
        array![0., 1., 2.],
        array![[2., 4., 6.], [4., 16., 32.]],
        strategy::Linear,
        Extrapolate::Enable,
    )
    .unwrap();
    assert_eq!(interp.interpolate(&[1.5, -0.5]).unwrap(), -3.5);

    // Change interpolator variant
    interp = Interp1D::new(
        array![0., 1., 2.],
        array![0., 4., 8.],
        strategy::Nearest.into(),
        Extrapolate::Error,
    )
    .unwrap()
    .into(); // `.into()` converts the `Interp1D` into an `InterpolatorEnum::Interp1D(...)`
    assert_eq!(interp.interpolate(&[1.75]).unwrap(), 8.);

    // Change interpolator variant again, using alternate syntax
    interp = InterpolatorEnum::new_3d(
        array![0., 1.],
        array![0., 1.],
        array![0., 1.],
        array![[[0., 1.], [0.1, 1.1]], [[0.2, 1.2], [0.3, 1.3]]],
        strategy::Nearest,
        Extrapolate::Error,
    )
    .unwrap(); // `.into()` converts the `Interp1D` into an `InterpolatorEnum::Interp1D(...)`
    assert_eq!(interp.interpolate(&[0.8, 0.7, 0.6]).unwrap(), 1.3);
}

/// Use a provided interpolator enum to allow interpolator swapping.
/// - **NOT** serde compatible
/// - Dynamically dispatched (slower runtime)
/// - Compatible with custom strategies
fn using_boxdyn() {
    // Create `Interpolator` trait object
    let mut boxed: Box<dyn Interpolator<_>> = Box::new(
        Interp2D::new(
            array![0., 1.],
            array![0., 1., 2.],
            array![[2., 4., 6.], [4., 16., 32.]],
            strategy::Linear,
            Extrapolate::Enable,
        )
        .unwrap(),
    );
    assert_eq!(boxed.interpolate(&[1.5, -0.5]).unwrap(), -3.5);

    // Change underlying interpolator
    boxed = Box::new(
        Interp1D::new(
            array![0., 1., 2.],
            array![0., 4., 8.],
            strategy::Nearest,
            Extrapolate::Error,
        )
        .unwrap(),
    );
    assert_eq!(boxed.interpolate(&[1.75]).unwrap(), 8.);
}
