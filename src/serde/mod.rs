//! `serde` support.
//!
//! Interpolators write their arrays in the [`ndarray`] format by default:
//! ```json
//! {"v":1,"dim":[3],"data":[0.0,1.0,2.0]}
//! ```
//! Wrap a value in [`Nested`](crate::Nested) to write the more readable nested-array format:
//! ```json
//! [0.0,1.0,2.0]
//! ```
//! Reading accepts either format, so [`Nested`](crate::Nested) only affects what is written.
//!
//! Split by direction: [`ser`] writes, [`de`] reads. Note that neither is the `serde` crate —
//! refer to that as `::serde` from inside this module.

use super::*;

pub(crate) mod de;
pub(crate) mod ser;

pub(crate) use de::*;
pub(crate) use ser::*;

/// The public surface, re-exported at the crate root.
pub use ser::{serialize_nested, Nested, SerializeNested};

pub(crate) use ::serde::ser::{SerializeSeq, SerializeStruct, Serializer};
pub(crate) use ::serde::{Deserialize, Serialize};
pub(crate) use ndarray::{DataOwned, IntoDimension};
pub(crate) use serde_unit_struct::{Deserialize_unit_struct, Serialize_unit_struct};
