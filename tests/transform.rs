//! End-to-end coverage for `GridTransform`/`ValuesTransform` (issue #56): oracle tests
//! against an equivalent interpolator built directly on an already-transformed grid,
//! domain-violation checks at all three checkpoints, `Extrapolate::Wrap` correctness
//! (the reason `interpolate_wrapped` exists at all for these wrappers), and the
//! composed `ValuesTransform(GridTransform(...))` case.

use ndarray::prelude::*;
use ninterp::error::{InterpolateError, OutsideDomainAt, ValidateError};
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
fn grid_transform_mixed_increasing_and_decreasing_axes_3d() {
    // `Interp3D`'s `GridTransform` strategy impl is hand-written, not derived from the
    // 2-D/ND paths, so it needs its own axis-reversal coverage: axis 0 uses
    // `Reciprocal` (decreasing), axes 1 and 2 use `Log`/`Sqrt` (increasing). `Linear`
    // interpolation on a function that's additively separable and linear in each
    // axis's *transformed* coordinate is exact everywhere, not just at grid points, so
    // any mismatch means an axis didn't get reversed/transformed correctly.
    let x = array![1., 2., 4.];
    let y = array![1., std::f64::consts::E, std::f64::consts::E.powi(2)];
    let z: Array1<f64> = array![1., 4., 9.];
    let f_xyz = Array3::from_shape_fn((3, 3, 3), |(i, j, k)| {
        2. / x[i] + 3. * y[j].ln() + 4. * z[k].sqrt()
    });

    let interp = Interp3D::new(
        x,
        y,
        z,
        f_xyz,
        GridTransform::new(
            vec![Transform::Reciprocal, Transform::Log, Transform::Sqrt],
            Linear,
        ),
        Extrapolate::Error,
    )
    .unwrap();

    for (qx, qy, qz) in [(1.5_f64, 2.0_f64, 2.0_f64), (3.0, 5.0, 6.25)] {
        let expected = 2. / qx + 3. * qy.ln() + 4. * qz.sqrt();
        let got = interp.interpolate(&[qx, qy, qz]).unwrap();
        assert!(
            (got - expected).abs() < 1e-9,
            "query=({qx},{qy},{qz}): got={got}, expected={expected}"
        );
    }
}

#[test]
fn grid_transform_3d_wrap_matches_nd() {
    // Same hand-written-vs-shared-path concern as
    // `grid_transform_mixed_increasing_and_decreasing_axes_3d`, but for
    // `Extrapolate::Wrap` dispatch specifically: confirms `Interp3D`'s
    // `interpolate_wrapped` matches the well-covered `InterpND` path exactly, for a
    // query out of bounds on every axis so every axis actually wraps.
    let x = array![1., 2., 4., 8., 16.];
    let y = array![1., 2., 4., 8., 16.];
    let z = array![1., 2., 4., 8., 16.];
    let f_xyz = Array3::from_shape_fn((5, 5, 5), |(i, j, k)| x[i] + y[j] + z[k]);

    let interp_3d = Interp3D::new(
        x.clone(),
        y.clone(),
        z.clone(),
        f_xyz.clone(),
        GridTransform::log(CubicC2::periodic()),
        Extrapolate::Wrap,
    )
    .unwrap();
    let interp_nd = InterpND::new(
        vec![x, y, z],
        f_xyz.into_dyn(),
        GridTransform::log(CubicC2::periodic()),
        Extrapolate::Wrap,
    )
    .unwrap();

    for query in [[20.0_f64, 20., 20.], [0.5, 0.5, 0.5], [20., 0.5, 3.]] {
        let a = interp_3d.interpolate(&query).unwrap();
        let b = interp_nd.interpolate(&query).unwrap();
        assert!((a - b).abs() < 1e-9, "query={query:?}: 3d={a}, nd={b}");
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
fn grid_transform_nd_0d_placeholder_does_not_panic() {
    // `InterpND`'s 0-D case (a single value, no real axes) represents "no axes" with
    // one empty grid entry rather than an empty `Vec`, so a `GridTransform` with the
    // matching 0 transforms must not index into `transforms` while validating/
    // transforming that placeholder axis.
    let interp = InterpND::new(
        vec![array![]],
        array![42.].into_dyn(),
        GridTransform::new(vec![], Linear),
        Extrapolate::Error,
    )
    .unwrap();
    assert_eq!(interp.interpolate(&[]).unwrap(), 42.);
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
fn values_transform_allows_nan_as_missing_data_sentinel() {
    // Real N-D datasets commonly leave grid corners unmeasured (NaN) while a "blob"
    // of real data covers the interior: three NaN cells per corner here (an L-shaped
    // notch), leaving a rounded-corner diamond of real data in the middle.
    // ValuesTransform must let NaN propagate through construction and interpolation
    // instead of rejecting it as a domain violation: forward(NaN) is a clean NaN for
    // every transform, and NaN only poisons interpolated results near the missing
    // corners, not the whole dataset.
    let x = array![1., 2., 3., 4., 5.];
    let y = array![1., 2., 3., 4., 5.];
    let f_xy = array![
        [f64::NAN, f64::NAN, 4., f64::NAN, f64::NAN],
        [f64::NAN, 4., 5., 6., f64::NAN],
        [4., 5., 6., 7., 8.],
        [f64::NAN, 6., 7., 8., f64::NAN],
        [f64::NAN, f64::NAN, 8., f64::NAN, f64::NAN],
    ];

    let interp =
        Interp2D::new(x, y, f_xy, ValuesTransform::log(Linear), Extrapolate::Error).unwrap();

    // Exactly at a missing corner: NaN propagates through unchanged.
    assert!(interp.interpolate(&[1., 1.]).unwrap().is_nan());
    // Blending toward a missing corner: NaN poisons the local result.
    assert!(interp.interpolate(&[1.5, 1.5]).unwrap().is_nan());
    // Inside the real interior blob: interpolation is unaffected.
    let got = interp.interpolate(&[2.5, 2.5]).unwrap();
    assert!(got.is_finite(), "expected a finite value, got {got}");
}

#[test]
fn grid_transform_reciprocal_rejects_nan_query() {
    // `x != 0` alone accepts NaN (`NaN != 0.0` is `true` in IEEE 754), unlike `Log`'s
    // `x > 0`/`Sqrt`'s `x >= 0`, which are `false` for NaN and so already exclude it.
    // Without excluding NaN explicitly, a NaN query would pass this domain check,
    // `forward()` to NaN via `recip()`, and panic deep in grid search instead of
    // returning a clear domain error.
    let x = array![1., 2., 4., 8.];
    let y = array![1., 2., 4., 8.];
    let interp =
        Interp1D::new(x, y, GridTransform::reciprocal(Linear), Extrapolate::Enable).unwrap();
    let err = interp.interpolate(&[f64::NAN]).unwrap_err();
    let InterpolateError::GridTransformDomain(failures) = &err else {
        panic!("expected GridTransformDomain, got {err:?}");
    };
    assert_eq!(failures.len(), 1);
    assert!(matches!(
        failures[0],
        OutsideDomainAt {
            index: 0,
            dim: 0,
            transform: Transform::Reciprocal,
            ..
        }
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
fn grid_transform_reciprocal_rejects_equal_transformed_coordinates() {
    // Distinct subnormal-magnitude grid coordinates can both overflow to
    // `f64::INFINITY` under `Reciprocal`, producing two *equal* transformed
    // coordinates even though the raw grid is strictly increasing and every
    // coordinate individually passes the `x != 0` domain check. The
    // decreasing-transform branch of the monotonicity check must reject that (not
    // just a transformed increase), or a zero-width interval reaches the inner
    // strategy.
    let x = array![1e-320, 2e-320, 3., 4.];
    let y = array![10., 20., 30., 40.];
    let interp = Interp1D::new(x, y, GridTransform::reciprocal(Linear), Extrapolate::Error);
    assert!(matches!(
        interp.unwrap_err(),
        ValidateError::GridTransformNotMonotonic {
            transform: Transform::Reciprocal,
            dim: 0,
            index: 1,
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
    let err = interp.interpolate(&[-1.]).unwrap_err();
    let InterpolateError::GridTransformDomain(failures) = &err else {
        panic!("expected GridTransformDomain, got {err:?}");
    };
    assert_eq!(failures.len(), 1);
    assert!(matches!(
        failures[0],
        OutsideDomainAt {
            index: 0,
            dim: 0,
            transform: Transform::Log,
            ..
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
fn nested_grid_transform_wrap_matches_manual_composed_grid() {
    // Composing two `GridTransform`s (`GridTransform::sqrt(GridTransform::log(...))`:
    // `Sqrt` applied to the raw query first, `Log` applied to that result second) must
    // wrap in the *fully composed* `ln(sqrt(x))` space, not just the outer `sqrt(x)`
    // space: wrapping doesn't commute with either nonlinear transform, so wrapping only
    // at the outer layer (then forward-transforming the already-wrapped value through
    // the inner layer, as if via plain `interpolate`) uses the wrong period entirely.
    // Confirmed against a manual oracle built directly on the fully composed grid,
    // mirroring `grid_transform_wrap_matches_manual_log_grid`'s single-layer version.
    let x = array![1., 2., 4., 8., 16.];
    let y = array![1., 2., 4., 8., 1.]; // not exactly periodic; Periodic doesn't require it

    let via_wrapper = Interp1D::new(
        x.clone(),
        y.clone(),
        GridTransform::sqrt(GridTransform::log(CubicC2::periodic())),
        Extrapolate::Wrap,
    )
    .unwrap();

    let x_composed = x.mapv(|v: f64| v.sqrt().ln());
    let manual = Interp1D::new(x_composed, y, CubicC2::periodic(), Extrapolate::Wrap).unwrap();

    for query in [20.0_f64, 32., 0.5] {
        let a = via_wrapper.interpolate(&[query]).unwrap();
        let b = manual.interpolate(&[query.sqrt().ln()]).unwrap();
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
    let err = interp.interpolate(&[-5.]).unwrap_err();
    let InterpolateError::GridTransformDomain(failures) = &err else {
        panic!("expected GridTransformDomain, got {err:?}");
    };
    assert_eq!(failures.len(), 1);
    assert!(matches!(
        failures[0],
        OutsideDomainAt {
            index: 0,
            dim: 0,
            transform: Transform::Log,
            ..
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
fn grid_transform_batch_domain_violations_all_reported() {
    // `GridTransform::batch_interpolate_into` pre-scans the whole batch and
    // aggregates every domain violation, rather than the trait default's
    // short-circuit on the first `interpolate` call: query 1 and 3 both violate
    // `Log`'s domain (`x > 0`), and both must be reported, not just the first.
    let x = array![1., 2., 3., 4.];
    let y = array![1., 2., 3., 4.];
    let interp = Interp1D::new(x, y, GridTransform::log(Linear), Extrapolate::Enable).unwrap();

    let points = [[-1.], [2.5], [-3.], [0.]];
    let mut out = [0.; 4];
    let err = interp
        .batch_interpolate_into(&points, &mut out)
        .unwrap_err();
    let InterpolateError::GridTransformDomain(failures) = &err else {
        panic!("expected GridTransformDomain, got {err:?}");
    };
    assert_eq!(failures.len(), 3);
    for (failure, expected_index) in failures.iter().zip([0, 2, 3]) {
        assert!(matches!(
            failure,
            OutsideDomainAt {
                index,
                dim: 0,
                transform: Transform::Log,
                ..
            } if *index == expected_index
        ));
    }
}

#[test]
fn grid_transform_wrap_batch_domain_violations_all_reported() {
    // The `Extrapolate::Wrap` batch dispatch can't reach `GridTransform`'s own
    // aggregating `batch_interpolate_into` (which point wraps vs. interpolates
    // normally is decided per point), so it pre-scans with `check_batch_domain`
    // instead: query 0 and 2 both violate `Log`'s domain (`x > 0`), and both must be
    // reported, not just the first.
    let x = array![1., 2., 3., 4.];
    let y = array![1., 2., 3., 4.];
    let interp = Interp1D::new(x, y, GridTransform::log(Linear), Extrapolate::Wrap).unwrap();

    let points = [[-1.], [2.5], [-3.], [10.]];
    let mut out = [0.; 4];
    let err = interp
        .batch_interpolate_into(&points, &mut out)
        .unwrap_err();
    let InterpolateError::GridTransformDomain(failures) = &err else {
        panic!("expected GridTransformDomain, got {err:?}");
    };
    assert_eq!(failures.len(), 2);
    for (failure, expected_index) in failures.iter().zip([0, 2]) {
        assert!(matches!(
            failure,
            OutsideDomainAt {
                index,
                dim: 0,
                transform: Transform::Log,
                ..
            } if *index == expected_index
        ));
    }
}

#[test]
fn grid_transform_nested_wrap_batch_domain_violations_all_reported() {
    // `GridTransform::check_batch_domain`'s trait-level pre-scan must recurse into
    // `inner` with the forward-transformed batch (mirroring `batch_interpolate_into`),
    // not just check its own layer. Composing two transforms
    // (`GridTransform::sqrt(GridTransform::log(Linear))`: `Sqrt` applied to the raw
    // query first, then `Log` applied to that result) means a query of exactly `0.`
    // passes the outer `Sqrt` domain (`x >= 0`) but violates the inner `Log` domain
    // once sqrt-transformed (`sqrt(0.) == 0.`, not `> 0.`). Without recursing into
    // `inner`, that violation is invisible to the pre-scan and only surfaces (aborting
    // the whole batch, mislabeled as index 0) once the per-point loop happens to reach
    // it, instead of being aggregated with the other violation at its true index.
    let x = array![1., 2., 3., 4.];
    let y = array![1., 2., 3., 4.];
    let interp = Interp1D::new(
        x,
        y,
        GridTransform::sqrt(GridTransform::log(Linear)),
        Extrapolate::Wrap,
    )
    .unwrap();

    let points = [[0.], [2.], [0.], [3.]];
    let mut out = [0.; 4];
    let err = interp
        .batch_interpolate_into(&points, &mut out)
        .unwrap_err();
    let InterpolateError::GridTransformDomain(failures) = &err else {
        panic!("expected GridTransformDomain, got {err:?}");
    };
    assert_eq!(failures.len(), 2);
    for (failure, expected_index) in failures.iter().zip([0, 2]) {
        assert!(matches!(
            failure,
            OutsideDomainAt {
                index,
                dim: 0,
                transform: Transform::Log,
                ..
            } if *index == expected_index
        ));
    }
}

#[test]
fn grid_transform_nested_wrap_batch_reports_outer_and_inner_violations_together() {
    // Bailing out of the outer `Sqrt` layer's own domain check (via `?`) as soon as it
    // finds *any* violation, before ever recursing into `inner`, would hide `Log`'s own
    // violations for the *other* points in the same batch entirely (not just mislabel
    // them): a batch with an outer (`Sqrt`) violation at one point and an inner
    // (`Log`, once sqrt-transformed) violation at a different point must report both.
    let x = array![1., 2., 3., 4.];
    let y = array![1., 2., 3., 4.];
    let interp = Interp1D::new(
        x,
        y,
        GridTransform::sqrt(GridTransform::log(Linear)),
        Extrapolate::Wrap,
    )
    .unwrap();

    // index 0: -1. violates outer Sqrt (x >= 0).
    // index 1: 0. passes outer Sqrt but violates inner Log once sqrt-transformed
    //          (sqrt(0.) == 0., not > 0.).
    // index 2: 4. valid all the way through.
    // index 3: -5. violates outer Sqrt again.
    let points = [[-1.], [0.], [4.], [-5.]];
    let mut out = [0.; 4];
    let err = interp
        .batch_interpolate_into(&points, &mut out)
        .unwrap_err();
    let InterpolateError::GridTransformDomain(failures) = &err else {
        panic!("expected GridTransformDomain, got {err:?}");
    };
    assert_eq!(failures.len(), 3);
    for (failure, (expected_index, expected_transform)) in failures.iter().zip([
        (0, Transform::Sqrt),
        (1, Transform::Log),
        (3, Transform::Sqrt),
    ]) {
        assert!(matches!(
            failure,
            OutsideDomainAt { index, dim: 0, transform, .. }
            if *index == expected_index && *transform == expected_transform
        ));
    }
}

#[test]
fn grid_transform_enum_wrap_batch_domain_violations_all_reported() {
    // `Strategy1DEnum`'s generated `check_batch_domain` forwarding (both the
    // match-based enum impl and its `Box` impl) must reach a `GridTransform`
    // variant's own aggregating check instead of silently falling back to the
    // trait's no-op default: query 0 and 2 both violate `Log`'s domain, and both
    // must be reported, not just the first.
    use ninterp::strategy::enums::Strategy1DEnum;

    let x = array![1., 2., 3., 4.];
    let y = array![1., 2., 3., 4.];
    let strategy: Strategy1DEnum<f64> =
        GridTransform::log(Box::new(Strategy1DEnum::from(Linear))).into();
    let interp: Interp1D<_, Strategy1DEnum<f64>> =
        Interp1D::new(x, y, strategy, Extrapolate::Wrap).unwrap();

    let points = [[-1.], [2.5], [-3.], [10.]];
    let mut out = [0.; 4];
    let err = interp
        .batch_interpolate_into(&points, &mut out)
        .unwrap_err();
    let InterpolateError::GridTransformDomain(failures) = &err else {
        panic!("expected GridTransformDomain, got {err:?}");
    };
    assert_eq!(failures.len(), 2);
    for (failure, expected_index) in failures.iter().zip([0, 2]) {
        assert!(matches!(
            failure,
            OutsideDomainAt {
                index,
                dim: 0,
                transform: Transform::Log,
                ..
            } if *index == expected_index
        ));
    }
}

#[test]
fn grid_transform_nd_wrap_batch_domain_violations_all_reported() {
    // Same as `grid_transform_wrap_batch_domain_violations_all_reported`, but for
    // `InterpND`, which implements `batch_interpolate_into`'s `Extrapolate::Wrap` arm
    // by hand rather than through the macro shared by `Interp1D`/`2D`/`3D`.
    let x = array![1., 2., 3., 4.];
    let y = x.mapv(|v: f64| v.ln());
    let interp = InterpND::new(
        vec![x],
        y.into_dyn(),
        GridTransform::log(Linear),
        Extrapolate::Wrap,
    )
    .unwrap();

    let p0 = [-1.];
    let p1 = [2.5];
    let p2 = [-3.];
    let p3 = [10.];
    let points: [&[f64]; 4] = [&p0, &p1, &p2, &p3];
    let mut out = [0.; 4];
    let err = interp
        .batch_interpolate_into(&points, &mut out)
        .unwrap_err();
    let InterpolateError::GridTransformDomain(failures) = &err else {
        panic!("expected GridTransformDomain, got {err:?}");
    };
    assert_eq!(failures.len(), 2);
    for (failure, expected_index) in failures.iter().zip([0, 2]) {
        assert!(matches!(
            failure,
            OutsideDomainAt {
                index,
                dim: 0,
                transform: Transform::Log,
                ..
            } if *index == expected_index
        ));
    }
}

#[test]
fn grid_transform_batch_domain_violations_multiple_dims_per_point() {
    // A single point can violate the domain on more than one axis; both must be
    // reported, mirroring how `Extrapolate::Error` aggregates `OutOfBoundsAt` per
    // dimension for a single out-of-bounds point.
    let x = array![1., 2., 3.];
    let y = array![1., 2., 3.];
    let f_xy = Array2::from_shape_fn((3, 3), |(i, j)| x[i] + y[j]);
    let interp =
        Interp2D::new(x, y, f_xy, GridTransform::log(Linear), Extrapolate::Enable).unwrap();

    let points = [[1.5, 1.5], [-1., -1.]];
    let mut out = [0.; 2];
    let err = interp
        .batch_interpolate_into(&points, &mut out)
        .unwrap_err();
    let InterpolateError::GridTransformDomain(failures) = &err else {
        panic!("expected GridTransformDomain, got {err:?}");
    };
    assert_eq!(failures.len(), 2);
    for (failure, expected_dim) in failures.iter().zip([0, 1]) {
        assert!(matches!(
            failure,
            OutsideDomainAt {
                index: 1,
                dim,
                transform: Transform::Log,
                ..
            } if *dim == expected_dim
        ));
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
