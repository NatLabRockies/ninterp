//! Interpolator types, one module per dimensionality (mirroring `ninterp`'s own
//! `interpolator::{one, two, three}` layout).

use super::*;

pub mod one;
pub mod three;
pub mod two;

pub use one::{UomInterp1D, UomInterp1DBase, UomInterp1DView};
