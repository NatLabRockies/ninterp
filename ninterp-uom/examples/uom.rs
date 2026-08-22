use ninterp_uom::ndarray::prelude::*;
use ninterp_uom::prelude::*;

use uom::si::power::kilowatt;
use uom::si::ratio::ratio;

fn main() {
    // f(x) = 0.25 kW + 0.5 kW * x, in `f64`, viewed (borrowed, zero-copy).
    // Requires the `f64` feature (on by default).
    #[cfg(feature = "f64")]
    {
        use uom::si::f64::{Power, Ratio};

        let x = array![Ratio::new::<ratio>(0.), Ratio::new::<ratio>(1.)];
        let f_x = array![Power::new::<kilowatt>(0.25), Power::new::<kilowatt>(0.75)];

        let interp: UomInterp1DView<Ratio, Power, _, _> =
            UomInterp1DView::new(x.view(), f_x.view(), strategy::Linear, Extrapolate::Error)
                .unwrap();

        // No `unsafe` at the call site, and the output comes back as a `Power`, not a
        // bare `f64`.
        let output = interp.interpolate(Ratio::new::<ratio>(0.5)).unwrap();
        assert_eq!(output, Power::new::<kilowatt>(0.5));
    }

    // The same wrapper, unmodified, over `f32` storage instead - proving `V` is generic,
    // not just `Qx`/`Qv`. Requires the `f32` feature (`cargo run --example uom --features f32`).
    #[cfg(feature = "f32")]
    {
        use uom::si::f32::{Power, Ratio};

        let x = array![Ratio::new::<ratio>(0.), Ratio::new::<ratio>(1.)];
        let f_x = array![Power::new::<kilowatt>(0.25), Power::new::<kilowatt>(0.75)];

        let interp: UomInterp1DView<Ratio, Power, _, _> =
            UomInterp1DView::new(x.view(), f_x.view(), strategy::Linear, Extrapolate::Error)
                .unwrap();

        let output = interp.interpolate(Ratio::new::<ratio>(0.5)).unwrap();
        assert_eq!(output, Power::new::<kilowatt>(0.5));
    }
}
