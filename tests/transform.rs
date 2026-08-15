//! End-to-end coverage for `GridTransform`/`ValuesTransform` (issue #56): oracle tests
//! against an equivalent interpolator built directly on an already-transformed grid,
//! domain-violation checks at all three checkpoints, `Extrapolate::Wrap` correctness
//! (the reason `interpolate_wrapped` exists at all for these wrappers), and the
//! composed `ValuesTransform(GridTransform(...))` case.

use ndarray::prelude::*;
use ninterp::error::{InterpolateError, ValidateError};
use ninterp::prelude::*;
use ninterp::strategy::*;

#[test]
fn grid_transform_log_cubic_matches_manual_log_grid() {
    // f(x) = x^2 on a log-spaced grid.
    let x = array![1., 10., 100., 1000.];
    let y = x.mapv(|v: f64| v * v);

    let via_wrapper = Interp1D::new(
        x.clone(),
        y.clone(),
        GridTransform::log(CubicC2::not_a_knot()),
        Extrapolate::Error,
    )
    .unwrap();

    let x_log = x.mapv(f64::ln);
    let manual = Interp1D::new(x_log, y, CubicC2::not_a_knot(), Extrapolate::Error).unwrap();

    for query in [1.5_f64, 15., 234., 999.] {
        let a = via_wrapper.interpolate(&[query]).unwrap();
        let b = manual.interpolate(&[query.ln()]).unwrap();
        assert!(
            (a - b).abs() < 1e-9,
            "query={query}: wrapper={a}, manual={b}"
        );
    }
}

#[test]
fn grid_transform_reciprocal_reproduces_linear_in_reciprocal_space() {
    // `Reciprocal` is decreasing, so forward-transforming an ascending raw grid
    // produces a descending one; `GridTransform` must reverse it (and `values` to
    // match) back to ascending. f(x) = 2/x + 1 is linear in 1/x, so `Linear` must
    // reproduce it exactly at every interval, not just the first, if the reversal
    // (grid *and* values, kept in lockstep) is correct.
    let x = array![1., 2., 4., 8.];
    let y = x.mapv(|v: f64| 2. / v + 1.);

    let via_wrapper =
        Interp1D::new(x, y, GridTransform::reciprocal(Linear), Extrapolate::Error).unwrap();

    for query in [1.5_f64, 3., 5., 7.] {
        let expected = 2. / query + 1.;
        let got = via_wrapper.interpolate(&[query]).unwrap();
        assert!(
            (got - expected).abs() < 1e-9,
            "query={query}: got={got}, expected={expected}"
        );
    }
}

#[test]
fn grid_transform_reciprocal_wrap_matches_manually_reversed_grid() {
    let x = array![1., 2., 4., 8.];
    let y = array![8., 4., 2., 1.]; // roughly periodic in 1/x space

    let via_wrapper = Interp1D::new(
        x.clone(),
        y.clone(),
        GridTransform::reciprocal(CubicC2::periodic()),
        Extrapolate::Wrap,
    )
    .unwrap();

    // `Reciprocal` is its own inverse, but forward-transforming ascending `x` gives
    // descending `1/x`; build the manual reference by reversing both arrays by hand,
    // exactly what `GridTransform` must do internally.
    let mut u = x.mapv(|v: f64| 1. / v);
    u.invert_axis(Axis(0));
    let mut y_rev = y;
    y_rev.invert_axis(Axis(0));
    let manual = Interp1D::new(u, y_rev, CubicC2::periodic(), Extrapolate::Wrap).unwrap();

    for query in [0.5_f64, 10., 16.] {
        let a = via_wrapper.interpolate(&[query]).unwrap();
        let b = manual.interpolate(&[1. / query]).unwrap();
        assert!(
            (a - b).abs() < 1e-9,
            "query={query}: wrapper={a}, manual={b}"
        );
    }
}

#[test]
fn grid_transform_mixed_increasing_and_decreasing_axes_2d() {
    // Axis 0 uses `Reciprocal` (decreasing), axis 1 uses `Log` (increasing): confirms
    // `slice_each_axis` reverses only the axes that actually need it.
    let x = array![1., 2., 4.];
    let y = array![1., std::f64::consts::E, std::f64::consts::E.powi(2)];
    let f_xy = Array2::from_shape_fn((3, 3), |(i, j)| 2. / x[i] + 3. * y[j].ln() + 1.);

    let interp = Interp2D::new(
        x,
        y,
        f_xy,
        GridTransform::new(vec![Transform::Reciprocal, Transform::Log], Linear),
        Extrapolate::Error,
    )
    .unwrap();

    for (qx, qy) in [(1.5_f64, 2.0_f64), (3.0, 5.0)] {
        let expected = 2. / qx + 3. * qy.ln() + 1.;
        let got = interp.interpolate(&[qx, qy]).unwrap();
        assert!(
            (got - expected).abs() < 1e-9,
            "query=({qx},{qy}): got={got}, expected={expected}"
        );
    }
}

#[test]
fn grid_transform_log_linear_uniform_checks_transformed_grid_uniformity() {
    // Raw grid is geometric (ratio 2), so ln(x) is uniformly spaced.
    let x = array![1., 2., 4., 8., 16.];
    let y = x.mapv(|v: f64| v);
    let interp = Interp1D::new(x, y, GridTransform::log(LinearUniform), Extrapolate::Error);
    assert!(
        interp.is_ok(),
        "log-uniform grid should validate: {interp:?}"
    );

    // Raw grid is linearly (not geometrically) uniform, so ln(x) is not.
    let x = array![1., 2., 3., 4.];
    let y = x.mapv(|v: f64| v);
    let interp = Interp1D::new(x, y, GridTransform::log(LinearUniform), Extrapolate::Error);
    assert!(matches!(
        interp.unwrap_err(),
        ValidateError::NonUniform { dim: 0, .. }
    ));
}

#[test]
fn grid_transform_per_axis_count_mismatch() {
    let interp = Interp2D::new(
        array![1., 2., 3.],
        array![1., 2., 3.],
        array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
        GridTransform::new(vec![Transform::Log, Transform::Log, Transform::Log], Linear),
        Extrapolate::Error,
    );
    assert!(matches!(
        interp.unwrap_err(),
        ValidateError::PerAxisLen {
            label: "GridTransform",
            ndim: 2,
            found: 3,
            ..
        }
    ));
}

#[test]
fn grid_transform_domain_violation_at_construction() {
    let x = array![-1., 0., 1., 2.];
    let y = array![1., 2., 3., 4.];
    let interp = Interp1D::new(x, y, GridTransform::log(Linear), Extrapolate::Error);
    assert!(matches!(
        interp.unwrap_err(),
        ValidateError::GridTransformDomain {
            transform: Transform::Log,
            dim: 0,
            index: 0,
        }
    ));
}

#[test]
fn values_transform_domain_violation_at_construction() {
    let x = array![1., 2., 3., 4.];
    let y = array![1., 0., -1., 2.];
    let interp = Interp1D::new(x, y, ValuesTransform::log(Linear), Extrapolate::Error);
    assert!(matches!(
        interp.unwrap_err(),
        ValidateError::ValuesTransformDomain {
            transform: Transform::Log,
            index,
        } if index == [1]
    ));
}

#[test]
fn grid_transform_reciprocal_rejects_sign_crossing_grid() {
    // `Reciprocal`'s domain (`x != 0`) is two disconnected pieces, and `1/x` is only
    // decreasing *within* each piece, not across the gap between them. A raw grid
    // that crosses zero (e.g. [-2, -1, 1, 2]) transforms to a non-monotonic sequence
    // ([-0.5, -1, 1, 0.5]) that no single reversal can restore to ascending order:
    // reversing it gives [0.5, 1, -1, -0.5], still not ascending. Every individual
    // coordinate passes the per-element `in_domain` check (none is exactly 0), so
    // only a monotonicity check across the whole axis catches this.
    let x = array![-2., -1., 1., 2.];
    let y = array![10., 20., 30., 40.];
    let interp = Interp1D::new(x, y, GridTransform::reciprocal(Linear), Extrapolate::Error);
    assert!(matches!(
        interp.unwrap_err(),
        ValidateError::GridTransformNotMonotonic {
            transform: Transform::Reciprocal,
            dim: 0,
            index: 2,
        }
    ));
}

#[test]
fn grid_transform_domain_violation_at_query_time_under_enable() {
    // Without the query-time check, `Extrapolate::Enable` pushing the query below
    // `Log`'s domain would silently `ln()` into `NaN` instead of a clear error.
    let x = array![1., 2., 3., 4.];
    let y = array![1., 2., 3., 4.];
    let interp = Interp1D::new(x, y, GridTransform::log(Linear), Extrapolate::Enable).unwrap();
    assert!(matches!(
        interp.interpolate(&[-1.]).unwrap_err(),
        InterpolateError::GridTransformDomain {
            transform: Transform::Log,
            dim: 0,
        }
    ));
}

#[test]
fn grid_transform_wrap_matches_manual_log_grid() {
    let x = array![1., 2., 4., 8., 16.];
    let y = array![1., 2., 4., 8., 1.]; // not exactly periodic; Periodic doesn't require it

    let via_wrapper = Interp1D::new(
        x.clone(),
        y.clone(),
        GridTransform::log(CubicC2::periodic()),
        Extrapolate::Wrap,
    )
    .unwrap();

    let x_log = x.mapv(f64::ln);
    let manual = Interp1D::new(x_log, y, CubicC2::periodic(), Extrapolate::Wrap).unwrap();

    // Negative/zero raw queries are covered separately (`Log`'s domain is `x > 0`,
    // checked before wrapping even applies): see
    // `grid_transform_wrap_out_of_domain_point_errors_not_nan`.
    for query in [20.0_f64, 32., 0.5] {
        let a = via_wrapper.interpolate(&[query]).unwrap();
        let b = manual.interpolate(&[query.ln()]).unwrap();
        assert!(
            (a - b).abs() < 1e-9,
            "query={query}: wrapper={a}, manual={b}"
        );
    }
}

#[test]
fn grid_transform_wrap_out_of_domain_point_errors_not_nan() {
    let x = array![1., 2., 4., 8., 16.];
    let y = array![1., 2., 4., 8., 1.];
    let interp = Interp1D::new(
        x,
        y,
        GridTransform::log(CubicC2::periodic()),
        Extrapolate::Wrap,
    )
    .unwrap();
    assert!(matches!(
        interp.interpolate(&[-5.]).unwrap_err(),
        InterpolateError::GridTransformDomain {
            transform: Transform::Log,
            dim: 0,
        }
    ));
}

#[test]
fn grid_transform_batch_wrap_matches_single_point() {
    // `batch_interpolate`/`batch_interpolate_into`'s `Extrapolate::Wrap` arm used to
    // wrap in the *raw* grid's coordinate space directly, bypassing
    // `interpolate_wrapped` entirely, unlike the single-point `interpolate` path
    // (which has always gone through `interpolate_wrapped`). Wrapping doesn't
    // commute with a nonlinear transform, so this silently gave batch calls a
    // different (wrong) answer from the single-point call for the exact same query.
    let x = array![1., 2., 4., 8., 16.];
    let y = array![1., 2., 4., 8., 1.];
    let interp = Interp1D::new(
        x,
        y,
        GridTransform::log(CubicC2::periodic()),
        Extrapolate::Wrap,
    )
    .unwrap();

    for query in [20.0_f64, 32., 0.5, 4.] {
        let single = interp.interpolate(&[query]).unwrap();
        let batch = interp.batch_interpolate(&[[query]]).unwrap()[0];
        assert!(
            (single - batch).abs() < 1e-9,
            "query={query}: single={single}, batch={batch}"
        );
    }
}

#[test]
fn grid_transform_nd_batch_wrap_matches_single_point() {
    // Same as `grid_transform_batch_wrap_matches_single_point`, but for `InterpND`,
    // which implements `batch_interpolate_into` by hand rather than through the
    // macro shared by `Interp1D`/`2D`/`3D`, and had the identical bug independently.
    let x = array![1., 2., 4., 8., 16.];
    let y = x.mapv(|v: f64| v.ln());

    let interp = InterpND::new(
        vec![x],
        y.into_dyn(),
        GridTransform::log(CubicC2::periodic()),
        Extrapolate::Wrap,
    )
    .unwrap();

    for query in [20.0_f64, 32., 0.5, 4.] {
        let single = interp.interpolate(&[query]).unwrap();
        let point = [query];
        let batch = interp.batch_interpolate(&[&point[..]]).unwrap()[0];
        assert!(
            (single - batch).abs() < 1e-9,
            "query={query}: single={single}, batch={batch}"
        );
    }
}

#[test]
fn grid_transform_clamp_and_fill() {
    let x = array![1., 2., 4., 8., 16.];
    let y = array![1., 4., 16., 64., 256.];

    let clamped = Interp1D::new(
        x.clone(),
        y.clone(),
        GridTransform::log(Linear),
        Extrapolate::Clamp,
    )
    .unwrap();
    assert_eq!(
        clamped.interpolate(&[100.]).unwrap(),
        clamped.interpolate(&[16.]).unwrap()
    );
    assert_eq!(
        clamped.interpolate(&[0.1]).unwrap(),
        clamped.interpolate(&[1.]).unwrap()
    );

    let filled = Interp1D::new(x, y, GridTransform::log(Linear), Extrapolate::Fill(-1.)).unwrap();
    assert_eq!(filled.interpolate(&[100.]).unwrap(), -1.);
}

#[test]
fn composed_values_log_grid_log_under_wrap_delegates_to_inner() {
    // ValuesTransform must hand `point` unmodified to `inner.interpolate_wrapped`, not
    // compute its own raw-space wrap: this is the case that would break if it did.
    let x = array![1., 2., 4., 8., 16.];
    let y = array![1., 2., 4., 8., 1.]; // strictly positive, for the outer log(values)

    let composed = Interp1D::new(
        x.clone(),
        y.clone(),
        ValuesTransform::log(GridTransform::log(CubicC2::periodic())),
        Extrapolate::Wrap,
    )
    .unwrap();

    let x_log = x.mapv(f64::ln);
    let y_log = y.mapv(f64::ln);
    let manual = Interp1D::new(x_log, y_log, CubicC2::periodic(), Extrapolate::Wrap).unwrap();

    for query in [20.0_f64, 32., 0.5] {
        let a = composed.interpolate(&[query]).unwrap();
        let b = manual.interpolate(&[query.ln()]).unwrap().exp();
        assert!(
            (a - b).abs() < 1e-6,
            "query={query}: composed={a}, manual={b}"
        );
    }
}

#[cfg(feature = "serde")]
mod serde_round_trip {
    use ninterp::strategy::enums::Strategy1DEnum;
    use ninterp::strategy::*;

    fn round_trip<T>(value: &T)
    where
        T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        let json = serde_json::to_string(value).unwrap();
        let de: T = serde_json::from_str(&json).unwrap();
        assert_eq!(&de, value, "failed to round-trip: {json}");
    }

    #[test]
    fn transform_round_trips() {
        round_trip(&Transform::Identity);
        round_trip(&Transform::Log);
        round_trip(&Transform::Sqrt);
        round_trip(&Transform::Reciprocal);
    }

    #[test]
    fn bare_grid_and_values_transform_round_trip() {
        let g: GridTransform<f64, Linear> = GridTransform::log(Linear);
        round_trip(&g);
        let v: ValuesTransform<f64, Linear> = ValuesTransform::log(Linear);
        round_trip(&v);
        let composed: ValuesTransform<f64, GridTransform<f64, Linear>> =
            ValuesTransform::log(GridTransform::sqrt(Linear));
        round_trip(&composed);
    }

    #[test]
    fn nested_enum_grid_transform_round_trips_and_deserializes_deep_chains() {
        let mut chain: Strategy1DEnum<f64> = Linear.into();
        for _ in 0..64 {
            chain = GridTransform::log(Box::new(chain)).into();
        }
        round_trip(&chain);
    }
}
