use ndarray::prelude::*;
use ninterp::prelude::*;

use uom::si::f64::Ratio;
use uom::si::ratio::ratio;

use uom::si::f64::Power;
use uom::si::power::kilowatt;

fn main() {
    let x = array![Ratio::new::<ratio>(0.), Ratio::new::<ratio>(1.)];
    let f_x = array![Power::new::<kilowatt>(0.25), Power::new::<kilowatt>(0.75)];
    // `uom::si::Quantity` is repr(transparent), meaning it has the same memory layout as its contained type.
    // This means we can get the contained type via transmuting.
    let interp: Interp1DViewed<&f64, _> = unsafe {
        Interp1D::new(
            std::mem::transmute::<ArrayView1<Ratio>, ArrayView1<f64>>(x.view()),
            std::mem::transmute::<ArrayView1<Power>, ArrayView1<f64>>(f_x.view()),
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap()
    };
    let output = interp.interpolate(&[0.5]).unwrap();
    // Note the result is not 0.5
    assert!(output != 0.5);
    // It is instead returned in the base units of f_x `Power`, i.e. 500 W
    assert_eq!(output, 500.);
}
