//! Cubic interpolation algorithms shared across all dimensionalities,
//! for [`CubicC1`] and [`CubicC2`].

use super::*;

mod c1;
mod c2;
mod utils;

pub use c1::{CubicC1, CubicC1CacheMode, CubicC1DerivativeMode};
pub use c2::{CubicC2, CubicC2BoundaryConditions, CubicC2Endpoint};

pub(crate) use c1::{
    compute_corner_cache_fd, compute_fd_cache, evaluate_hermite_1d_cached,
    evaluate_spline_corner_local,
};
pub(crate) use c2::{
    compute_corner_cache, compute_m_cache, evaluate_spline_1d_cached, validate_bc_min_points,
};
pub(crate) use utils::evaluate_spline_corner_cached;
