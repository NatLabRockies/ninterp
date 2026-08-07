use ndarray::prelude::*;

use ninterp::prelude::*;

fn main() {
    using_enum();
    using_boxdyn();
}

/// Use a provided strategy enum to allow strategy swapping.
/// - serde compatible
/// - Statically dispatched (faster runtime)
/// - **NOT** compatible with custom strategies
fn using_enum() {
    // Create mutable interpolator
    let mut interp: Interp1D<_, strategy::enums::Strategy1DEnum> = Interp1D::new(
        array![0., 1., 2.],
        array![0., 3., 6.],
        // Provide the strategy as an enum
        strategy::Linear.into(),
        Extrapolate::Error,
    )
    .unwrap();
    assert_eq!(interp.interpolate(&[1.75]).unwrap(), 5.25);
    // Change strategy to `Nearest`
    interp.set_strategy(strategy::Nearest).unwrap();
    assert_eq!(interp.interpolate(&[1.75]).unwrap(), 6.);
}

/// Use a provided strategy enum to allow strategy swapping.
/// - **NOT** serde compatible
/// - Dynamically dispatched (slower runtime)
/// - Compatible with custom strategies
fn using_boxdyn() {
    // Create mutable interpolator
    let mut interp = Interp1D::new(
        array![0., 1., 2.],
        array![0., 3., 6.],
        // Provide the strategy as a trait object
        Box::new(strategy::Linear) as Box<dyn strategy::traits::Strategy1D<_>>,
        Extrapolate::Error,
    )
    .unwrap();
    assert_eq!(interp.interpolate(&[1.75]).unwrap(), 5.25);
    // Change strategy to `Nearest`
    interp.set_strategy(Box::new(strategy::Nearest)).unwrap();
    assert_eq!(interp.interpolate(&[1.75]).unwrap(), 6.);
}
