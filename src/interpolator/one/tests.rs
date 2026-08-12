use super::*;

#[test]
fn test_cubic_spline() {
    // Linear data: any spline reproduces it exactly (all second derivatives = 0)
    let interp = Interp1D::new(
        array![0., 1., 2., 3.],
        array![1., 3., 5., 7.], // f(x) = 2x + 1
        strategy::CubicC2::not_a_knot(),
        Extrapolate::Enable,
    )
    .unwrap();
    // Knot values
    let x = interp.data.grid[0].clone();
    for (i, xi) in x.iter().enumerate() {
        assert_approx_eq!(interp.interpolate(&[*xi]).unwrap(), interp.data.values[i]);
    }
    // Midpoints
    assert_approx_eq!(interp.interpolate(&[0.5]).unwrap(), 2.0);
    assert_approx_eq!(interp.interpolate(&[1.5]).unwrap(), 4.0);
    assert_approx_eq!(interp.interpolate(&[2.5]).unwrap(), 6.0);
    // Extrapolation via boundary polynomials
    assert_approx_eq!(interp.interpolate(&[-1.0]).unwrap(), -1.0);
    assert_approx_eq!(interp.interpolate(&[4.0]).unwrap(), 9.0);
}

#[test]
fn test_cubic_c2_knot_exactness() {
    // Values at all knots must be reproduced exactly regardless of data shape
    let interp = Interp1D::new(
        array![0., 1., 2., 3., 4.],
        array![0., 1., 4., 9., 16.], // f(x) = x^2
        strategy::CubicC2::not_a_knot(),
        Extrapolate::Error,
    )
    .unwrap();
    let x = interp.data.grid[0].clone();
    for (i, xi) in x.iter().enumerate() {
        assert_approx_eq!(interp.interpolate(&[*xi]).unwrap(), interp.data.values[i]);
    }
}

#[test]
fn test_cubic_c2_two_points() {
    // Degenerate case: 2 points → degenerates to linear interpolation.
    // Uses Natural BC since NotAKnot requires ≥ 4 points.
    let interp = Interp1D::new(
        array![0., 1.],
        array![0., 2.],
        strategy::CubicC2::natural(),
        Extrapolate::Enable,
    )
    .unwrap();
    assert_approx_eq!(interp.interpolate(&[0.5]).unwrap(), 1.0);
    assert_approx_eq!(interp.interpolate(&[2.0]).unwrap(), 4.0); // extrapolation
}

#[test]
fn test_cubic_c2_natural() {
    // Uses example from https://tools.timodenk.com/cubic-spline-interpolation
    let interp = Interp1D::new(
        array![-1.5, -0.2, 1., 5., 10., 15., 20.],
        array![-1.2, 0., 0.5, 1., 1.2, 2., 1.],
        strategy::CubicC2::natural(),
        Extrapolate::Error,
    )
    .unwrap();
    let inputs = &[[-1.4], [6.], [7.], [14.], [19.]];
    let expected = &[
        -1.095065419952828,
        1.015165170952125,
        1.0218810233458848,
        1.932091193114578,
        1.301403522754169,
    ];
    let outputs = interp.batch_interpolate(inputs).unwrap();
    for (out, exp) in outputs.iter().zip(expected.iter()) {
        assert_approx_eq!(out, exp);
    }
}

#[test]
fn test_cubic_c2_not_a_knot() {
    let interp = Interp1D::new(
        array![-1.5, -0.2, 1., 5., 10., 15., 20.],
        array![-1.2, 0., 0.5, 1., 1.2, 2., 1.],
        strategy::CubicC2::not_a_knot(),
        Extrapolate::Error,
    )
    .unwrap();
    let inputs = &[[-1.4], [6.], [7.], [14.], [19.]];
    let expected = &[-1.07209658, 1.01588337, 1.02825421, 1.87938333, 1.50092501];
    let outputs = interp.batch_interpolate(inputs).unwrap();
    for (out, exp) in outputs.iter().zip(expected.iter()) {
        assert_approx_eq!(out, exp);
    }
}

#[test]
fn test_cubic_c2_clamped() {
    let interp = Interp1D::new(
        array![-1.5, -0.2, 1., 5., 10., 15., 20.],
        array![-1.2, 0., 0.5, 1., 1.2, 2., 1.],
        strategy::CubicC2::clamped(-1., 1.),
        Extrapolate::Error,
    )
    .unwrap();
    let inputs = &[[-1.4], [6.], [7.], [14.], [19.]];
    let expected = &[-1.27364033, 1.03070187, 1.01669998, 2.16516227, 0.41055472];
    let outputs = interp.batch_interpolate(inputs).unwrap();
    for (out, exp) in outputs.iter().zip(expected.iter()) {
        assert_approx_eq!(out, exp);
    }
}

#[test]
fn test_cubic_c2_periodic() {
    let interp = Interp1D::new(
        array![-4., -1.5, -0.2, 1., 5., 10., 15., 20.],
        array![1., -1.2, 0., 0.5, 1., 1.2, 2., 1.],
        strategy::CubicC2::periodic(),
        Extrapolate::Error,
    )
    .unwrap();
    let inputs = &[[-1.4], [6.], [7.], [14.], [19.]];
    let expected = &[-1.15382906, 1.04798649, 1.07838449, 1.77694949, 1.87460604];
    let outputs = interp.batch_interpolate(inputs).unwrap();
    for (out, exp) in outputs.iter().zip(expected.iter()) {
        assert_approx_eq!(out, exp);
    }
}

#[test]
fn test_cubic_c2_not_a_knot_cubic_exact() {
    // f(x) = x^3: a genuine cubic (nonzero third derivative), unlike the quadratic data
    // in `knot_exactness` above. `NotAKnot`'s defining property is exact reproduction of
    // any degree-<=3 polynomial, at interior points and via boundary extrapolation, not
    // just at grid points -- quadratics satisfy that trivially since their third
    // derivative is already zero everywhere, so this is a stronger test.
    let interp = Interp1D::new(
        array![0., 1., 2., 3.],
        array![0., 1., 8., 27.], // f(x) = x^3
        strategy::CubicC2::not_a_knot(),
        Extrapolate::Enable,
    )
    .unwrap();
    assert_approx_eq!(interp.interpolate(&[0.5]).unwrap(), 0.125);
    assert_approx_eq!(interp.interpolate(&[1.5]).unwrap(), 3.375);
    assert_approx_eq!(interp.interpolate(&[2.5]).unwrap(), 15.625);
    // Extrapolation via the boundary cubic polynomial is exact too, since the "boundary
    // polynomial" already equals the true global cubic everywhere.
    assert_approx_eq!(interp.interpolate(&[-1.0]).unwrap(), -1.0);
    assert_approx_eq!(interp.interpolate(&[4.0]).unwrap(), 64.0);
}

#[test]
fn test_cubic_c2_notaknot_enough_points() {
    // NotAKnot requires ≥ 4 points to define a cubic spline
    // not enough points
    let result = Interp1D::new(
        array![0., 1., 2.],
        array![0., 1., 4.],
        strategy::CubicC2::not_a_knot(),
        Extrapolate::Error,
    );
    assert!(
        result.is_err(),
        "NotAKnot on a 3-point axis should fail validation, got Ok"
    );
    // enough points
    let result = Interp1D::new(
        array![0., 1., 2., 3.],
        array![0., 1., 4., 9.],
        strategy::CubicC2::not_a_knot(),
        Extrapolate::Error,
    );
    assert!(
        result.is_ok(),
        "NotAKnot on a 4-point axis should succeed, got Err: {:?}",
        result.unwrap_err()
    );
}

#[test]
fn test_cubic_c2_clamped_cubic_exact() {
    // Same f(x) = x^3, but with `Clamped` given the true endpoint derivatives
    // (f'(x) = 3x^2, so f'(0) = 0, f'(3) = 27). Exact reproduction here confirms
    // `Clamped` correctly uses the supplied derivative, not just that the solve runs.
    let interp = Interp1D::new(
        array![0., 1., 2., 3.],
        array![0., 1., 8., 27.],
        strategy::CubicC2::clamped(0., 27.),
        Extrapolate::Enable,
    )
    .unwrap();
    assert_approx_eq!(interp.interpolate(&[0.5]).unwrap(), 0.125);
    assert_approx_eq!(interp.interpolate(&[1.5]).unwrap(), 3.375);
    assert_approx_eq!(interp.interpolate(&[2.5]).unwrap(), 15.625);
    assert_approx_eq!(interp.interpolate(&[-1.0]).unwrap(), -1.0);
    assert_approx_eq!(interp.interpolate(&[4.0]).unwrap(), 64.0);
}

#[test]
fn test_cubic_c2_clamped_uses_given_derivative() {
    // Differential check for the previous test: `clamped_cubic_exact` alone can't tell
    // "Clamped correctly used the supplied derivative" apart from "Clamped silently
    // behaved like NotAKnot" -- for a genuine cubic like f(x) = x^3, NotAKnot *also*
    // reproduces it exactly with no derivative info at all, so a bug that dropped
    // `left`/`right` entirely would still pass that test. Supplying deliberately wrong
    // derivatives here and confirming the result moves far from the true value proves
    // they're actually being read.
    let interp = Interp1D::new(
        array![0., 1., 2., 3.],
        array![0., 1., 8., 27.], // f(x) = x^3; true derivatives are 0 and 27
        strategy::CubicC2::clamped(999., 999.),
        Extrapolate::Enable,
    )
    .unwrap();
    let wrong = interp.interpolate(&[0.5]).unwrap();
    assert!(
        (wrong - 0.125_f64).abs() > 1.0,
        "Clamped(999, 999) gave {wrong}, suspiciously close to the true f(0.5) = 0.125 \
         as if the supplied derivatives were ignored"
    );
}

#[test]
fn test_invalid_args() {
    let interp = Interp1D::new(
        array![0., 1., 2., 3., 4.],
        array![0.2, 0.4, 0.6, 0.8, 1.0],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    // Wrong-length points on a concretely-typed `Interp1D` are caught at compile time
    // via the inherent `interpolate(&[D::Elem; N])`; the trait's checked path (used by
    // generic/`dyn` callers passing a real slice) still catches it at runtime.
    assert!(matches!(
        Interpolator::interpolate(&interp, &[]).unwrap_err(),
        InterpolateError::PointLength { .. }
    ));
    assert_eq!(interp.interpolate(&[1.0]).unwrap(), 0.4);
}

#[test]
fn test_invalid_args_dyn() {
    let interp: Box<dyn Interpolator<f64>> = Box::new(
        Interp1D::new(
            array![0., 1., 2., 3., 4.],
            array![0.2, 0.4, 0.6, 0.8, 1.0],
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap(),
    );
    // Through `Box<dyn Interpolator<T>>`, only the trait's slice-taking method is ever
    // reachable, so a wrong-length point still fails at runtime, not compile time.
    assert!(matches!(
        interp.interpolate(&[]).unwrap_err(),
        InterpolateError::PointLength { .. }
    ));
    assert_eq!(interp.interpolate(&[1.0]).unwrap(), 0.4);
}

#[test]
fn test_dyn_interpolator() {
    let interp = Interp1D::new(
        array![0., 1., 2., 3., 4.],
        array![0.2, 0.4, 0.6, 0.8, 1.0],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    let points: [&[f64]; 2] = [&[1.0], &[2.5]];

    let boxed: Box<dyn AnyInterpolator<f64>> = Box::new(interp.clone());
    assert_eq!(boxed.interpolate(&[1.0]).unwrap(), 0.4);
    assert_eq!(
        boxed.batch_interpolate(&points).unwrap(),
        interp.batch_interpolate(&[[1.0], [2.5]]).unwrap(),
    );
    assert!(matches!(
        boxed.interpolate(&[]).unwrap_err(),
        InterpolateError::PointLength { expected: 1, .. }
    ));
    assert_eq!(
        boxed.as_any().downcast_ref::<Interp1D<f64, _>>(),
        Some(&interp)
    );
}

#[test]
fn test_insufficient_grid_points() {
    // A single grid point can't bracket anything, regardless of `Extrapolate` setting.
    // Previously this passed construction and panicked on the first `interpolate` call.
    assert!(matches!(
        Interp1D::new(
            array![5.0],
            array![10.0],
            strategy::Linear,
            Extrapolate::Error
        )
        .unwrap_err(),
        ValidateError::InsufficientGridPoints(0)
    ));
}

#[test]
fn test_linear() {
    let interp = Interp1D::new(
        array![0., 1., 2., 3., 4.],
        array![0.2, 0.4, 0.6, 0.8, 1.0],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    // Check that interpolating at grid points just retrieves the value
    let x = &interp.data.grid[0];
    for (i, x_i) in x.iter().enumerate() {
        assert_eq!(interp.interpolate(&[*x_i]).unwrap(), interp.data.values[i]);
    }
    assert_eq!(interp.interpolate(&[3.00]).unwrap(), 0.8);
    assert_eq!(interp.interpolate(&[3.75]).unwrap(), 0.95);
    assert_eq!(interp.interpolate(&[4.00]).unwrap(), 1.0);
}

#[test]
fn test_left_nearest() {
    let interp = Interp1D::new(
        array![0., 1., 2., 3., 4.],
        array![0.2, 0.4, 0.6, 0.8, 1.0],
        strategy::Step::from(strategy::step::StepDirection::Lower),
        Extrapolate::Error,
    )
    .unwrap();
    // Check that interpolating at grid points just retrieves the value
    let x = &interp.data.grid[0];
    let f_x = &interp.data.values;
    for (i, x_i) in x.iter().enumerate() {
        assert_eq!(interp.interpolate(&[*x_i]).unwrap(), f_x[i]);
    }
    assert_eq!(interp.interpolate(&[3.00]).unwrap(), 0.8);
    assert_eq!(interp.interpolate(&[3.75]).unwrap(), 0.8);
    assert_eq!(interp.interpolate(&[4.00]).unwrap(), 1.0);
}

#[test]
fn test_right_nearest() {
    let interp = Interp1D::new(
        array![0., 1., 2., 3., 4.],
        array![0.2, 0.4, 0.6, 0.8, 1.0],
        strategy::Step::from(strategy::step::StepDirection::Upper),
        Extrapolate::Error,
    )
    .unwrap();
    // Check that interpolating at grid points just retrieves the value
    let x = &interp.data.grid[0];
    let f_x = &interp.data.values;
    for (i, x_i) in x.iter().enumerate() {
        assert_eq!(interp.interpolate(&[*x_i]).unwrap(), f_x[i]);
    }
    assert_eq!(interp.interpolate(&[3.00]).unwrap(), 0.8);
    assert_eq!(interp.interpolate(&[3.25]).unwrap(), 1.0);
    assert_eq!(interp.interpolate(&[4.00]).unwrap(), 1.0);
}

#[test]
fn test_step_markers() {
    let lower = Interp1D::new(
        array![0., 1., 2., 3., 4.],
        array![0.2, 0.4, 0.6, 0.8, 1.0],
        strategy::StepLower,
        Extrapolate::Error,
    )
    .unwrap();
    assert_eq!(lower.interpolate(&[3.75]).unwrap(), 0.8);
    assert_eq!(lower.interpolate(&[4.00]).unwrap(), 1.0);

    let upper = Interp1D::new(
        array![0., 1., 2., 3., 4.],
        array![0.2, 0.4, 0.6, 0.8, 1.0],
        strategy::StepUpper,
        Extrapolate::Error,
    )
    .unwrap();
    assert_eq!(upper.interpolate(&[3.25]).unwrap(), 1.0);
    assert_eq!(upper.interpolate(&[3.00]).unwrap(), 0.8);
}

#[test]
fn test_nearest() {
    let interp = Interp1D::new(
        array![0., 1., 2., 3., 4.],
        array![0.2, 0.4, 0.6, 0.8, 1.0],
        strategy::Nearest,
        Extrapolate::Error,
    )
    .unwrap();
    // Check that interpolating at grid points just retrieves the value
    let x = &interp.data.grid[0];
    let f_x = &interp.data.values;
    for (i, x_i) in x.iter().enumerate() {
        assert_eq!(interp.interpolate(&[*x_i]).unwrap(), f_x[i]);
    }
    assert_eq!(interp.interpolate(&[3.00]).unwrap(), 0.8);
    assert_eq!(interp.interpolate(&[3.25]).unwrap(), 0.8);
    assert_eq!(interp.interpolate(&[3.50]).unwrap(), 1.0);
    assert_eq!(interp.interpolate(&[3.75]).unwrap(), 1.0);
    assert_eq!(interp.interpolate(&[4.00]).unwrap(), 1.0);
}

#[test]
fn test_integer_nearest_and_wrap_step() {
    let nearest = Interp1D::new(
        array![0, 10, 20],
        array![100, 200, 300],
        strategy::Nearest,
        Extrapolate::Error,
    )
    .unwrap();
    assert_eq!(nearest.interpolate(&[14]).unwrap(), 200);
    // Midpoint ties resolve to the upper bracket in Nearest.
    assert_eq!(nearest.interpolate(&[15]).unwrap(), 300);

    let step_wrap = Interp1D::new(
        array![0, 10, 20],
        array![100, 200, 300],
        strategy::Step::from(strategy::step::StepDirection::Lower),
        Extrapolate::Wrap,
    )
    .unwrap();
    // -1 wraps to 19 -> lower step bucket [10, 20].
    assert_eq!(step_wrap.interpolate(&[-1]).unwrap(), 200);
    // 21 wraps to 1 -> lower step bucket [0, 10].
    assert_eq!(step_wrap.interpolate(&[21]).unwrap(), 100);
}

#[test]
fn test_linear_uniform() {
    let grid = array![0., 1., 2., 3., 4.];
    let values = array![0.2, 0.4, 0.6, 0.8, 1.0];

    let uniform = Interp1D::new(
        grid.clone(),
        values.clone(),
        strategy::LinearUniform,
        Extrapolate::Error,
    )
    .unwrap();
    let linear = Interp1D::new(grid, values, strategy::Linear, Extrapolate::Error).unwrap();

    // Results must match Linear exactly at grid points and between them
    let x = &uniform.data.grid[0];
    let f_x = &uniform.data.values;
    for (i, x_i) in x.iter().enumerate() {
        assert_eq!(uniform.interpolate(&[*x_i]).unwrap(), f_x[i]);
    }
    for point in [0.5, 1.25, 2.75, 3.99] {
        assert_eq!(
            uniform.interpolate(&[point]).unwrap(),
            linear.interpolate(&[point]).unwrap()
        );
    }
}

#[test]
fn test_linear_uniform_non_uniform_grid_error() {
    assert!(Interp1D::new(
        array![0., 1., 2., 3., 4.5], // not uniform
        array![0.2, 0.4, 0.6, 0.8, 1.0],
        strategy::LinearUniform,
        Extrapolate::Error,
    )
    .is_err());
}

#[test]
fn test_linear_uniform_extrapolate() {
    let interp = Interp1D::new(
        array![0., 1., 2., 3., 4.],
        array![0.2, 0.4, 0.6, 0.8, 1.0],
        strategy::LinearUniform,
        Extrapolate::Enable,
    )
    .unwrap();
    assert_eq!(interp.interpolate(&[-1.0]).unwrap(), 0.0);
    assert_eq!(interp.interpolate(&[5.0]).unwrap(), 1.2);
}

#[test]
fn test_step_invalid_direction_count() {
    // 2 directions for a 1-D interpolator → ValidateError
    assert!(Interp1D::new(
        array![0., 1., 2., 3., 4.],
        array![0.2, 0.4, 0.6, 0.8, 1.0],
        strategy::Step(vec![
            strategy::step::StepDirection::Lower,
            strategy::step::StepDirection::Upper,
        ]),
        Extrapolate::Error,
    )
    .is_err());
}

#[test]
fn test_extrapolate_inputs() {
    // Incorrect extrapolation selection
    assert!(matches!(
        Interp1D::new(
            array![0., 1., 2., 3., 4.],
            array![0.2, 0.4, 0.6, 0.8, 1.0],
            strategy::Nearest,
            Extrapolate::Enable,
        )
        .unwrap_err(),
        ValidateError::ExtrapolateUnsupported
    ));

    // Extrapolate::Error
    let interp = Interp1D::new(
        array![0., 1., 2., 3., 4.],
        array![0.2, 0.4, 0.6, 0.8, 1.0],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    // Fail to extrapolate below lowest grid value
    assert!(matches!(
        interp.interpolate(&[-1.]).unwrap_err(),
        InterpolateError::OutOfBounds(_)
    ));
    // Fail to extrapolate above highest grid value
    assert!(matches!(
        interp.interpolate(&[5.]).unwrap_err(),
        InterpolateError::OutOfBounds(_)
    ));
}

#[test]
fn test_extrapolate_fill() {
    let interp = Interp1D::new(
        array![0., 1., 2., 3., 4.],
        array![0.2, 0.4, 0.6, 0.8, 1.0],
        strategy::Linear,
        Extrapolate::Fill(f64::NAN),
    )
    .unwrap();
    assert_eq!(interp.interpolate(&[1.5]).unwrap(), 0.5);
    assert_eq!(interp.interpolate(&[2.]).unwrap(), 0.6);
    assert!(interp.interpolate(&[-1.]).unwrap().is_nan());
    assert!(interp.interpolate(&[5.]).unwrap().is_nan());
}

#[test]
fn test_extrapolate_clamp() {
    let interp = Interp1D::new(
        array![0., 1., 2., 3., 4.],
        array![0.2, 0.4, 0.6, 0.8, 1.0],
        strategy::Linear,
        Extrapolate::Clamp,
    )
    .unwrap();
    assert_eq!(interp.interpolate(&[-1.]).unwrap(), 0.2);
    assert_eq!(interp.interpolate(&[5.]).unwrap(), 1.0);
}

#[test]
fn test_extrapolate() {
    let interp = Interp1D::new(
        array![0., 1., 2., 3., 4.],
        array![0.2, 0.4, 0.6, 0.8, 1.0],
        strategy::Linear,
        Extrapolate::Enable,
    )
    .unwrap();
    assert_eq!(interp.interpolate(&[-1.]).unwrap(), 0.0);
    assert_approx_eq!(interp.interpolate(&[-0.75]).unwrap(), 0.05);
    assert_eq!(interp.interpolate(&[5.]).unwrap(), 1.2);
}

#[test]
fn test_batch_interpolate_matches_interpolate() {
    let interp = Interp1D::new(
        array![0., 1., 2., 3., 4.],
        array![0.2, 0.4, 0.6, 0.8, 1.0],
        strategy::Linear,
        Extrapolate::Enable,
    )
    .unwrap();
    let points = [[-1.], [0.5], [1.5], [3.75], [5.]];
    let batched = interp.batch_interpolate(&points).unwrap();
    let looped: Vec<_> = points
        .iter()
        .map(|point| interp.interpolate(point).unwrap())
        .collect();
    assert_eq!(batched, looped);

    let batched_fast = interp.batch_interpolate_fast(&points);
    let looped_fast: Vec<_> = points
        .iter()
        .map(|point| interp.interpolate_fast(point))
        .collect();
    assert_eq!(batched_fast, looped_fast);
}

#[test]
fn test_batch_interpolate_clamp() {
    let interp = Interp1D::new(
        array![0., 1., 2., 3., 4.],
        array![0.2, 0.4, 0.6, 0.8, 1.0],
        strategy::Linear,
        Extrapolate::Clamp,
    )
    .unwrap();
    assert_eq!(
        interp.batch_interpolate(&[[-1.], [2.], [5.]]).unwrap(),
        vec![0.2, 0.6, 1.0]
    );
}

#[test]
fn test_batch_interpolate_fill() {
    let interp = Interp1D::new(
        array![0., 1., 2., 3., 4.],
        array![0.2, 0.4, 0.6, 0.8, 1.0],
        strategy::Linear,
        Extrapolate::Fill(f64::NAN),
    )
    .unwrap();
    let results = interp.batch_interpolate(&[[-1.], [2.], [5.]]).unwrap();
    assert!(results[0].is_nan());
    assert_eq!(results[1], 0.6);
    assert!(results[2].is_nan());
}

#[test]
fn test_batch_interpolate_wrap_boundary() {
    // `wrap()` isn't identity exactly at the boundary (`wrap(max, min, max) == min`),
    // so an in-bounds point sitting exactly on the upper edge must not get wrapped,
    // while a genuinely out-of-bounds point still does.
    let interp = Interp1D::new(
        array![0., 1., 2., 3., 4.],
        array![0.2, 0.4, 0.6, 0.8, 1.0],
        strategy::Linear,
        Extrapolate::Wrap,
    )
    .unwrap();
    let results = interp.batch_interpolate(&[[4.], [5.]]).unwrap();
    assert_eq!(results[0], interp.interpolate(&[4.]).unwrap());
    assert_eq!(results[1], interp.interpolate(&[5.]).unwrap());
    assert_eq!(results[0], 1.0); // untouched boundary point
    assert_eq!(results[1], 0.4); // wrap(5, 0, 4) == 1 -> f(1) == 0.4
}

#[test]
fn test_batch_interpolate_error_aggregates_all_points() {
    let interp = Interp1D::new(
        array![0., 1., 2., 3., 4.],
        array![0.2, 0.4, 0.6, 0.8, 1.0],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    // Two bad points among good ones: the batch error must mention both, not just
    // the first one it finds.
    let err = interp
        .batch_interpolate(&[[1.], [-1.], [2.], [5.]])
        .unwrap_err();
    let InterpolateError::OutOfBounds(failures) = err else {
        panic!("expected InterpolateError::OutOfBounds");
    };
    let offending: Vec<usize> = failures.iter().map(|at| at.index).collect();
    assert!(offending.contains(&1));
    assert!(offending.contains(&3));
    assert!(!offending.contains(&0));
    assert!(!offending.contains(&2));
}

#[test]
fn test_batch_interpolate_dyn() {
    let interp: Box<dyn Interpolator<f64>> = Box::new(
        Interp1D::new(
            array![0., 1., 2., 3., 4.],
            array![0.2, 0.4, 0.6, 0.8, 1.0],
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap(),
    );
    let points: [&[f64]; 2] = [&[1.0], &[3.0]];
    assert_eq!(interp.batch_interpolate(&points).unwrap(), vec![0.4, 0.8]);
    assert_eq!(interp.batch_interpolate_fast(&points), vec![0.4, 0.8]);
}

#[test]
fn test_partialeq() {
    #[derive(PartialEq)]
    #[allow(unused)]
    struct MyStruct(InterpData1D<f64>);

    #[derive(PartialEq)]
    #[allow(unused)]
    struct MyStruct2(Interp1D<f64, strategy::Linear>);
}

#[test]
#[cfg(feature = "serde")]
fn test_serde() {
    let interp = Interp1D::new(
        array![0., 1., 2., 3., 4.],
        array![0.2, 0.4, 0.6, 0.8, 1.0],
        strategy::Step::from(strategy::step::StepDirection::Lower),
        Extrapolate::Error,
    )
    .unwrap();

    let ser = serde_json::to_string(&interp).unwrap();
    let de: Interp1D<f64, strategy::Step> = serde_json::from_str(&ser).unwrap();
    assert_eq!(interp, de);

    // `ndarray` format by default
    let data_ser = serde_json::to_string(&interp.data).unwrap();
    assert_eq!(
        data_ser,
        "{\"grid\":[{\"v\":1,\"dim\":[5],\"data\":[0.0,1.0,2.0,3.0,4.0]}],\"values\":{\"v\":1,\"dim\":[5],\"data\":[0.2,0.4,0.6,0.8,1.0]}}"
    );
    // nested-array format on request
    let data_ser_nested = serde_json::to_string(&crate::prelude::Nested(&interp.data)).unwrap();
    assert_eq!(
        data_ser_nested,
        "{\"grid\":[[0.0,1.0,2.0,3.0,4.0]],\"values\":[0.2,0.4,0.6,0.8,1.0]}"
    );
    // ...and the whole interpolator nests too
    let interp_ser_nested = serde_json::to_string(&crate::prelude::Nested(&interp)).unwrap();
    let de_nested: Interp1D<f64, strategy::Step> =
        serde_json::from_str(&interp_ser_nested).unwrap();
    assert_eq!(interp, de_nested);

    // simple format (new serialization output)
    let ser0 = "{\"grid\":[[0.0,1.0,2.0,3.0,4.0]],\"values\":[0.2,0.4,0.6,0.8,1.0]}";
    let de0: InterpData1D<_> = serde_json::from_str(ser0).unwrap();
    assert_eq!(interp.data, de0);
    // mixed format (simple grid)
    let ser1 = "{\"grid\":[[0.0,1.0,2.0,3.0,4.0]],\"values\":{\"v\":1,\"dim\":[5],\"data\":[0.2,0.4,0.6,0.8,1.0]}}";
    let de1: InterpData1D<_> = serde_json::from_str(ser1).unwrap();
    assert_eq!(interp.data, de1);
    // mixed format (simple values)
    let ser2 = "{\"grid\":[{\"v\":1,\"dim\":[5],\"data\":[0.0,1.0,2.0,3.0,4.0]}],\"values\":[0.2,0.4,0.6,0.8,1.0]}";
    let de2: InterpData1D<_> = serde_json::from_str(ser2).unwrap();
    assert_eq!(interp.data, de2);
    // complex format (legacy serialization output)
    let ser3 = "{\"grid\":[{\"v\":1,\"dim\":[5],\"data\":[0.0,1.0,2.0,3.0,4.0]}],\"values\":{\"v\":1,\"dim\":[5],\"data\":[0.2,0.4,0.6,0.8,1.0]}}";
    let de3: InterpData1D<_> = serde_json::from_str(ser3).unwrap();
    assert_eq!(interp.data, de3);
}
