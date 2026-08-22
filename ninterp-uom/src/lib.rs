#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

/// Re-exports [`ninterp::prelude`] alongside this crate's own types, so downstream
/// crates need only `use ninterp_uom::prelude::*;`.
pub mod prelude {
    pub use ninterp::prelude::*;

    pub use crate::base_unit::BaseUnit;
    pub use crate::interpolator::{
        UomInterp0D, UomInterp1D, UomInterp1DBase, UomInterp1DView, UomInterp2D, UomInterp2DBase,
        UomInterp2DView, UomInterp3D, UomInterp3DBase, UomInterp3DView,
    };
}

mod base_unit;
pub mod interpolator;

pub use base_unit::BaseUnit;
pub use interpolator::{
    UomInterp0D, UomInterp1D, UomInterp1DBase, UomInterp1DView, UomInterp2D, UomInterp2DBase,
    UomInterp2DView, UomInterp3D, UomInterp3DBase, UomInterp3DView,
};

pub(crate) use ninterp::error::{InterpolateError, ValidateError};
pub(crate) use ninterp::prelude::*;
pub(crate) use ninterp::strategy::enums::{Strategy1DEnum, Strategy2DEnum, Strategy3DEnum};
pub(crate) use ninterp::strategy::traits::{Strategy1D, Strategy2D, Strategy3D};

pub(crate) use alloc::boxed::Box;
// Only reached through test modules' `use super::*;` chains, which the unused-import
// lint doesn't track through a macro-namespace re-export.
#[allow(unused_imports)]
pub(crate) use alloc::vec;
pub(crate) use alloc::vec::Vec;

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
pub(crate) use ninterp::num_traits::{Euclid, Float, Num};

#[cfg(feature = "serde")]
pub(crate) use serde::{Deserialize, Deserializer, Serialize, Serializer};
