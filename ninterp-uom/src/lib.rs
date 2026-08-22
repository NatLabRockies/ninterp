#![cfg_attr(not(feature = "std"), no_std)]

/// Re-exports [`ninterp::prelude`] alongside this crate's own types, so downstream
/// crates need only `use ninterp_uom::prelude::*;`.
pub mod prelude {
    pub use ninterp::prelude::*;

    pub use crate::base_unit::BaseUnit;
    pub use crate::interpolator::{UomInterp1D, UomInterp1DBase, UomInterp1DView};
}

mod base_unit;
pub mod interpolator;

pub use base_unit::BaseUnit;
pub use interpolator::{UomInterp1D, UomInterp1DBase, UomInterp1DView};

pub(crate) use ninterp::error::{InterpolateError, ValidateError};
pub(crate) use ninterp::prelude::*;
pub(crate) use ninterp::strategy::traits::Strategy1D;

pub(crate) use core::fmt::Debug;
pub(crate) use core::marker::PhantomData;
pub(crate) use core::mem;

pub use ninterp::ndarray;
pub(crate) use ninterp::ndarray::prelude::*;
pub(crate) use ninterp::ndarray::{Data, OwnedRepr, RawDataClone, ViewRepr};

pub use uom;
pub(crate) use uom::{
    Conversion,
    si::{Dimension, Quantity, Units},
};

pub use ninterp::num_traits;
pub(crate) use ninterp::num_traits::{Euclid, Num};
