//! This module provides enums that allow mutable strategy swapping.
//!
//! This is an alternative to using a `Box<dyn Strategy1D>`/etc. with a few key differences:
//! - Better runtime performance
//! - Compatible with serde
//! - **Incompatible** with custom strategies
//!
//! # Example:
//! ```
//! use ndarray::prelude::*;
//! use ninterp::prelude::*;
//!
//! let mut interp: Interp1DOwned<_, strategy::enums::Strategy1DEnum> = Interp1D::new(
//!     // x
//!     array![0., 1., 2., 3., 4.],
//!     // f(x)
//!     array![0.2, 0.4, 0.6, 0.8, 1.0],
//!     strategy::Linear.into(),
//!     Extrapolate::Error,
//! )
//! .unwrap();
//! assert_eq!(interp.interpolate(&[3.00]).unwrap(), 0.8);
//! assert_eq!(interp.interpolate(&[3.75]).unwrap(), 0.95);
//! assert_eq!(interp.interpolate(&[4.00]).unwrap(), 1.0);
//!
//! interp.set_strategy(strategy::Nearest).unwrap();
//! assert_eq!(interp.interpolate(&[3.00]).unwrap(), 0.8);
//! assert_eq!(interp.interpolate(&[3.25]).unwrap(), 0.8);
//! assert_eq!(interp.interpolate(&[3.50]).unwrap(), 1.0);
//! ```
//! See also: `examples/dynamic_strategy.rs`

// NOTE: `enum_dispatch` does essentially what this module does, but with less boilerplate.
// However, it does not currently support using a generic trait on a non-generic enum.
// https://gitlab.com/antonok/enum_dispatch/-/issues/67

use super::*;

mod n;
mod one;
mod three;
mod two;

pub use n::*;
pub use one::*;
pub use three::*;
pub use two::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use ndarray::prelude::*;

    #[test]
    fn test_1d() {
        let mut interp: Interp1D<_, strategy::enums::Strategy1DEnum> = Interp1D::new(
            array![0., 1., 2., 3., 4.],
            array![0.2, 0.4, 0.6, 0.8, 1.0],
            strategy::Linear.into(),
            Extrapolate::Error,
        )
        .unwrap();
        assert_eq!(interp.interpolate(&[3.00]).unwrap(), 0.8);
        assert_eq!(interp.interpolate(&[3.75]).unwrap(), 0.95);
        assert_eq!(interp.interpolate(&[4.00]).unwrap(), 1.0);

        interp.set_strategy(strategy::Nearest).unwrap();
        assert_eq!(interp.interpolate(&[3.00]).unwrap(), 0.8);
        assert_eq!(interp.interpolate(&[3.25]).unwrap(), 0.8);
        assert_eq!(interp.interpolate(&[3.50]).unwrap(), 1.0);
        assert_eq!(interp.interpolate(&[3.75]).unwrap(), 1.0);
        assert_eq!(interp.interpolate(&[4.00]).unwrap(), 1.0);

        interp.set_strategy(strategy::LeftNearest).unwrap();
        assert_eq!(interp.interpolate(&[3.00]).unwrap(), 0.8);
        assert_eq!(interp.interpolate(&[3.75]).unwrap(), 0.8);
        assert_eq!(interp.interpolate(&[4.00]).unwrap(), 1.0);
    }

    #[test]
    fn test_2d() {
        let mut interp: Interp2D<_, strategy::enums::Strategy2DEnum> = Interp2D::new(
            array![0.05, 0.10, 0.15],
            array![0.10, 0.20, 0.30],
            array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
            strategy::Linear.into(),
            Extrapolate::Error,
        )
        .unwrap();
        let x = &interp.data.grid[0];
        let y = &interp.data.grid[1];
        let f_xy = &interp.data.values;

        assert_eq!(interp.interpolate(&[x[2], y[1]]).unwrap(), f_xy[[2, 1]]);
        assert_eq!(interp.interpolate(&[0.075, 0.25]).unwrap(), 3.);

        interp.set_strategy(strategy::Nearest).unwrap();
        let f_xy = &interp.data.values; // need fresh reference after mutation

        assert_eq!(interp.interpolate(&[0.05, 0.12]).unwrap(), f_xy[[0, 0]]);
        assert_eq!(
            // float imprecision
            interp.interpolate(&[0.07, 0.15 + 0.0001]).unwrap(),
            f_xy[[0, 1]]
        );
        assert_eq!(interp.interpolate(&[0.08, 0.21]).unwrap(), f_xy[[1, 1]]);
        assert_eq!(interp.interpolate(&[0.11, 0.26]).unwrap(), f_xy[[1, 2]]);
        assert_eq!(interp.interpolate(&[0.13, 0.12]).unwrap(), f_xy[[2, 0]]);
        assert_eq!(interp.interpolate(&[0.14, 0.29]).unwrap(), f_xy[[2, 2]]);
    }

    #[test]
    fn test_3d() {
        let mut interp: Interp3D<_, strategy::enums::Strategy3DEnum> = Interp3D::new(
            array![0.05, 0.10, 0.15],
            array![0.10, 0.20, 0.30],
            array![0.20, 0.40, 0.60],
            array![
                [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
                [[9., 10., 11.], [12., 13., 14.], [15., 16., 17.]],
                [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.],],
            ],
            strategy::Linear.into(),
            Extrapolate::Error,
        )
        .unwrap();
        let x = &interp.data.grid[0];
        let y = &interp.data.grid[1];
        let z = &interp.data.grid[2];

        assert_approx_eq!(interp.interpolate(&[x[0], y[0], 0.3]).unwrap(), 0.5);
        assert_approx_eq!(interp.interpolate(&[x[0], 0.15, z[0]]).unwrap(), 1.5);
        assert_approx_eq!(interp.interpolate(&[x[0], 0.15, 0.3]).unwrap(), 2.);
        assert_approx_eq!(interp.interpolate(&[0.075, y[0], z[0]]).unwrap(), 4.5);
        assert_approx_eq!(interp.interpolate(&[0.075, y[0], 0.3]).unwrap(), 5.);
        assert_approx_eq!(interp.interpolate(&[0.075, 0.15, z[0]]).unwrap(), 6.);

        interp.set_strategy(strategy::Nearest).unwrap();
        assert_eq!(interp.interpolate(&[0.06, 0.11, 0.22]).unwrap(), 0.);
        assert_eq!(interp.interpolate(&[0.06, 0.11, 0.31]).unwrap(), 1.);
        assert_eq!(interp.interpolate(&[0.06, 0.19, 0.22]).unwrap(), 3.);
        assert_eq!(interp.interpolate(&[0.06, 0.19, 0.31]).unwrap(), 4.);
        assert_eq!(interp.interpolate(&[0.09, 0.11, 0.22]).unwrap(), 9.);
        assert_eq!(interp.interpolate(&[0.09, 0.11, 0.31]).unwrap(), 10.);
        assert_eq!(interp.interpolate(&[0.09, 0.19, 0.22]).unwrap(), 12.);
        assert_eq!(interp.interpolate(&[0.09, 0.19, 0.31]).unwrap(), 13.);
    }

    #[test]
    fn test_nd() {
        let mut interp: InterpND<_, strategy::enums::StrategyNDEnum> = InterpND::new(
            vec![
                array![0.05, 0.10, 0.15],
                array![0.10, 0.20, 0.30],
                array![0.20, 0.40, 0.60],
            ],
            array![
                [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
                [[9., 10., 11.], [12., 13., 14.], [15., 16., 17.]],
                [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.],],
            ]
            .into_dyn(),
            strategy::Linear.into(),
            Extrapolate::Error,
        )
        .unwrap();
        let x = &interp.data.grid[0];
        let y = &interp.data.grid[1];
        let z = &interp.data.grid[2];

        assert_approx_eq!(interp.interpolate(&[x[0], y[0], 0.3]).unwrap(), 0.5);
        assert_approx_eq!(interp.interpolate(&[x[0], 0.15, z[0]]).unwrap(), 1.5);
        assert_approx_eq!(interp.interpolate(&[x[0], 0.15, 0.3]).unwrap(), 2.);
        assert_approx_eq!(interp.interpolate(&[0.075, y[0], z[0]]).unwrap(), 4.5);
        assert_approx_eq!(interp.interpolate(&[0.075, y[0], 0.3]).unwrap(), 5.);
        assert_approx_eq!(interp.interpolate(&[0.075, 0.15, z[0]]).unwrap(), 6.);

        interp.set_strategy(strategy::Nearest).unwrap();
        assert_eq!(interp.interpolate(&[0.06, 0.11, 0.22]).unwrap(), 0.);
        assert_eq!(interp.interpolate(&[0.06, 0.11, 0.31]).unwrap(), 1.);
        assert_eq!(interp.interpolate(&[0.06, 0.19, 0.22]).unwrap(), 3.);
        assert_eq!(interp.interpolate(&[0.06, 0.19, 0.31]).unwrap(), 4.);
        assert_eq!(interp.interpolate(&[0.09, 0.11, 0.22]).unwrap(), 9.);
        assert_eq!(interp.interpolate(&[0.09, 0.11, 0.31]).unwrap(), 10.);
        assert_eq!(interp.interpolate(&[0.09, 0.19, 0.22]).unwrap(), 12.);
        assert_eq!(interp.interpolate(&[0.09, 0.19, 0.31]).unwrap(), 13.);
    }
}
