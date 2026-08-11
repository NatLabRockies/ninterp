use super::*;

#[test]
fn test_cubic_spline() {
    // f(x, y, z) = x + 2y + 3z: linear, reproduced exactly by any spline
    let interp = Interp3D::new(
        array![0., 1., 2.],
        array![0., 1., 2.],
        array![0., 1., 2.],
        array![
            [[0., 3., 6.], [2., 5., 8.], [4., 7., 10.]],
            [[1., 4., 7.], [3., 6., 9.], [5., 8., 11.]],
            [[2., 5., 8.], [4., 7., 10.], [6., 9., 12.]],
        ],
        strategy::CubicC2::natural(),
        Extrapolate::Enable,
    )
    .unwrap();
    // Knots
    assert_approx_eq!(interp.interpolate(&[1., 1., 1.]).unwrap(), 6.);
    // Midpoints
    assert_approx_eq!(interp.interpolate(&[0.5, 0.5, 0.5]).unwrap(), 3.);
    assert_approx_eq!(interp.interpolate(&[1.0, 0.5, 1.0]).unwrap(), 5.);
    // Extrapolation
    assert_approx_eq!(interp.interpolate(&[3., 1., 1.]).unwrap(), 8.);
}

#[test]
fn test_cubic_spline_knot_exactness() {
    let interp = Interp3D::new(
        array![0., 1., 2.],
        array![0., 1., 2.],
        array![0., 1., 2.],
        array![
            [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
            [[9., 10., 11.], [12., 13., 14.], [15., 16., 17.]],
            [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.]],
        ],
        strategy::CubicC2::natural(),
        Extrapolate::Error,
    )
    .unwrap();
    let x = interp.data.grid[0].clone();
    let y = interp.data.grid[1].clone();
    let z = interp.data.grid[2].clone();
    for (i, xi) in x.iter().enumerate() {
        for (j, yj) in y.iter().enumerate() {
            for (k, zk) in z.iter().enumerate() {
                assert_approx_eq!(
                    interp.interpolate(&[*xi, *yj, *zk]).unwrap(),
                    interp.data.values[[i, j, k]]
                );
            }
        }
    }
}

#[test]
fn test_cubic_c2_interior_accuracy() {
    // f(x, y, z) = x^2*y + y^2*z + z^2*x: quadratic along every axis (well within
    // cubic-spline capacity) with a genuine three-way mixed-partial term, so a
    // `NotAKnot` spline reproduces it exactly everywhere, not just at grid points.
    fn f(x: f64, y: f64, z: f64) -> f64 {
        x * x * y + y * y * z + z * z * x
    }
    let grid = [0., 1., 2., 3.];
    let values = Array3::from_shape_fn((4, 4, 4), |(i, j, k)| f(grid[i], grid[j], grid[k]));
    let interp = Interp3D::new(
        array![0., 1., 2., 3.],
        array![0., 1., 2., 3.],
        array![0., 1., 2., 3.],
        values,
        strategy::CubicC2::not_a_knot(),
        Extrapolate::Error,
    )
    .unwrap();
    for &(x, y, z) in &[(0.5, 0.5, 0.5), (1.5, 2.5, 0.25), (2.25, 0.75, 1.5)] {
        assert_approx_eq!(interp.interpolate(&[x, y, z]).unwrap(), f(x, y, z));
    }
}

#[test]
fn test_cubic_c2_cached_vs_uncached() {
    // `Strategy3D`'s corner-cache path must agree with `StrategyND`'s unchanged
    // recursive-collapse path on the same grid/values/BC.
    fn f(x: f64, y: f64, z: f64) -> f64 {
        x * x * y + y * y * z + z * z * x
    }
    let grid = [0., 1., 2., 3.];
    let values = Array3::from_shape_fn((4, 4, 4), |(i, j, k)| f(grid[i], grid[j], grid[k]));
    let interp3d = Interp3D::new(
        array![0., 1., 2., 3.],
        array![0., 1., 2., 3.],
        array![0., 1., 2., 3.],
        values.clone(),
        strategy::CubicC2::not_a_knot(),
        Extrapolate::Error,
    )
    .unwrap();
    let interp_nd = InterpND::new(
        vec![
            array![0., 1., 2., 3.],
            array![0., 1., 2., 3.],
            array![0., 1., 2., 3.],
        ],
        values.into_dyn(),
        strategy::CubicC2::not_a_knot(),
        Extrapolate::Error,
    )
    .unwrap();
    for &(x, y, z) in &[(0.5, 0.5, 0.5), (1.5, 2.5, 0.25), (2.25, 0.75, 1.5)] {
        assert_approx_eq!(
            interp3d.interpolate(&[x, y, z]).unwrap(),
            interp_nd.interpolate(&[x, y, z]).unwrap()
        );
    }
}

#[test]
fn test_cubic_c2_clamped_short_axis() {
    // A `Clamped` axis with only 2 points must still validate under the corner-cache
    // upgrade -- regression test for the `Natural` (not `NotAKnot`) fallback choice.
    let mut strategy = strategy::CubicC2::natural();
    strategy.boundary_conditions = vec![
        strategy::CubicBoundaryConditions::Clamped {
            left: 1.,
            right: 1.,
        },
        strategy::CubicBoundaryConditions::Natural,
        strategy::CubicBoundaryConditions::Natural,
    ];
    let interp = Interp3D::new(
        array![0., 1.], // only 2 points on the Clamped axis
        array![0., 1., 2.],
        array![0., 1., 2.],
        array![
            [[0., 1., 2.], [1., 2., 3.], [2., 3., 4.]],
            [[1., 2., 3.], [2., 3., 4.], [3., 4., 5.]],
        ], // f(x, y, z) = x + y + z
        strategy,
        Extrapolate::Error,
    )
    .unwrap();
    assert_approx_eq!(interp.interpolate(&[0.5, 1.5, 0.5]).unwrap(), 2.5);
}

#[test]
fn test_cubic_c2_not_a_knot_cubic_exact() {
    // f(x, y, z) = x^3+y^3+z^3 + x^2*y+y^2*z+z^2*x: every axis slice is a genuine
    // cubic (unlike `interior_accuracy`'s quadratic data above), and every one of the
    // 8 corner-derivative slots (value, 3 first partials, 3 mixed second partials, the
    // triple mixed partial) is exercised with a non-constant value except the triple
    // mixed partial itself (which really is 0 for this f, since no term has xyz jointly).
    fn f(x: f64, y: f64, z: f64) -> f64 {
        x.powi(3) + y.powi(3) + z.powi(3) + x.powi(2) * y + y.powi(2) * z + z.powi(2) * x
    }
    let grid = [0., 1., 2., 3.];
    let values = Array3::from_shape_fn((4, 4, 4), |(i, j, k)| f(grid[i], grid[j], grid[k]));
    let interp = Interp3D::new(
        array![0., 1., 2., 3.],
        array![0., 1., 2., 3.],
        array![0., 1., 2., 3.],
        values,
        strategy::CubicC2::not_a_knot(),
        Extrapolate::Error,
    )
    .unwrap();
    for &(x, y, z) in &[(0.5, 0.5, 0.5), (1.5, 2.5, 0.25), (2.25, 0.75, 1.5)] {
        assert_approx_eq!(interp.interpolate(&[x, y, z]).unwrap(), f(x, y, z));
    }
}

#[test]
fn test_cubic_c2_clamped_cubic_exact() {
    // f(x, y, z) = x^3 + y^3 + z^3 (separable, no cross terms) -- `Clamped`'s endpoint
    // derivative is one scalar per axis shared by every pencil along it, so it's only
    // exact when the true boundary partial doesn't vary with the other axes; a
    // cross-term function can't satisfy that in general (see the 2D version of this
    // test for the full reasoning). Cross-term correctness is already covered by
    // `not_a_knot_cubic_exact` above.
    fn f(x: f64, y: f64, z: f64) -> f64 {
        x.powi(3) + y.powi(3) + z.powi(3)
    }
    let grid = [0., 1., 2., 3.];
    let values = Array3::from_shape_fn((4, 4, 4), |(i, j, k)| f(grid[i], grid[j], grid[k]));
    let mut strategy = strategy::CubicC2::natural();
    strategy.boundary_conditions = vec![
        strategy::CubicBoundaryConditions::Clamped {
            left: 0.,
            right: 27.,
        }; // f'(0) = 0, f'(3) = 27 on every axis
        3
    ];
    let interp = Interp3D::new(
        array![0., 1., 2., 3.],
        array![0., 1., 2., 3.],
        array![0., 1., 2., 3.],
        values,
        strategy,
        Extrapolate::Error,
    )
    .unwrap();
    for &(x, y, z) in &[(0.5, 0.5, 0.5), (1.5, 2.5, 0.25), (2.25, 0.75, 1.5)] {
        assert_approx_eq!(interp.interpolate(&[x, y, z]).unwrap(), f(x, y, z));
    }
}

#[test]
fn test_invalid_args() {
    let interp = Interp3D::new(
        array![0.05, 0.10, 0.15],
        array![0.10, 0.20, 0.30],
        array![0.20, 0.40, 0.60],
        array![
            [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
            [[9., 10., 11.], [12., 13., 14.], [15., 16., 17.]],
            [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.],],
        ],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    // Wrong-length points on a concretely-typed `Interp3D` are caught at compile time
    // via the inherent `interpolate(&[D::Elem; N])`; the trait's checked path (used by
    // generic/`dyn` callers passing a real slice) still catches it at runtime.
    assert!(matches!(
        Interpolator::interpolate(&interp, &[]).unwrap_err(),
        InterpolateError::PointLength { .. }
    ));
}

#[test]
fn test_dyn_interpolator() {
    let interp = Interp3D::new(
        array![0.05, 0.10, 0.15],
        array![0.10, 0.20, 0.30],
        array![0.20, 0.40, 0.60],
        array![
            [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
            [[9., 10., 11.], [12., 13., 14.], [15., 16., 17.]],
            [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.],],
        ],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    let points: [&[f64]; 2] = [&[0.05, 0.10, 0.20], &[0.15, 0.30, 0.60]];

    let boxed: Box<dyn AnyInterpolator<f64>> = Box::new(interp.clone());
    assert_eq!(boxed.interpolate(&[0.05, 0.10, 0.20]).unwrap(), 0.);
    assert_eq!(
        boxed.batch_interpolate(&points).unwrap(),
        interp
            .batch_interpolate(&[[0.05, 0.10, 0.20], [0.15, 0.30, 0.60]])
            .unwrap(),
    );
    assert!(matches!(
        boxed.interpolate(&[]).unwrap_err(),
        InterpolateError::PointLength { expected: 3, .. }
    ));
    assert_eq!(
        boxed.as_any().downcast_ref::<Interp3D<f64, _>>(),
        Some(&interp)
    );
}

#[test]
fn test_linear() {
    let interp = Interp3D::new(
        array![0.05, 0.10, 0.15],
        array![0.10, 0.20, 0.30],
        array![0.20, 0.40, 0.60],
        array![
            [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
            [[9., 10., 11.], [12., 13., 14.], [15., 16., 17.]],
            [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.],],
        ],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    // Check that interpolating at grid points just retrieves the value
    let x = &interp.data.grid[0];
    let y = &interp.data.grid[1];
    let z = &interp.data.grid[2];
    for (i, x_i) in x.iter().enumerate() {
        for (j, y_j) in y.iter().enumerate() {
            for (k, z_k) in z.iter().enumerate() {
                assert_eq!(
                    interp.interpolate(&[*x_i, *y_j, *z_k]).unwrap(),
                    interp.data.values[[i, j, k]]
                );
            }
        }
    }
    assert_approx_eq!(interp.interpolate(&[x[0], y[0], 0.3]).unwrap(), 0.5);
    assert_approx_eq!(interp.interpolate(&[x[0], 0.15, z[0]]).unwrap(), 1.5);
    assert_approx_eq!(interp.interpolate(&[x[0], 0.15, 0.3]).unwrap(), 2.);
    assert_approx_eq!(interp.interpolate(&[0.075, y[0], z[0]]).unwrap(), 4.5);
    assert_approx_eq!(interp.interpolate(&[0.075, y[0], 0.3]).unwrap(), 5.);
    assert_approx_eq!(interp.interpolate(&[0.075, 0.15, z[0]]).unwrap(), 6.);
}

#[test]
fn test_linear_extrapolation() {
    let interp = Interp3D::new(
        array![0.05, 0.10, 0.15],
        array![0.10, 0.20, 0.30],
        array![0.20, 0.40, 0.60],
        array![
            [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
            [[9., 10., 11.], [12., 13., 14.], [15., 16., 17.]],
            [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.],],
        ],
        strategy::Linear,
        Extrapolate::Enable,
    )
    .unwrap();
    // below x, below y, below z
    assert_approx_eq!(interp.interpolate(&[0.01, 0.06, 0.17]).unwrap(), -8.55);
    assert_approx_eq!(interp.interpolate(&[0.02, 0.08, 0.19]).unwrap(), -6.05);
    // below x, below y, above z
    assert_approx_eq!(interp.interpolate(&[0.01, 0.06, 0.63]).unwrap(), -6.25);
    assert_approx_eq!(interp.interpolate(&[0.02, 0.08, 0.65]).unwrap(), -3.75);
    // below x, above y, below z
    assert_approx_eq!(interp.interpolate(&[0.01, 0.33, 0.17]).unwrap(), -0.45);
    assert_approx_eq!(interp.interpolate(&[0.02, 0.36, 0.19]).unwrap(), 2.35);
    // below x, above y, above z
    assert_approx_eq!(interp.interpolate(&[0.01, 0.33, 0.63]).unwrap(), 1.85);
    assert_approx_eq!(interp.interpolate(&[0.02, 0.36, 0.65]).unwrap(), 4.65);
    // above x, below y, below z
    assert_approx_eq!(interp.interpolate(&[0.17, 0.06, 0.17]).unwrap(), 20.25);
    assert_approx_eq!(interp.interpolate(&[0.19, 0.08, 0.19]).unwrap(), 24.55);
    // above x, below y, above z
    assert_approx_eq!(interp.interpolate(&[0.17, 0.06, 0.63]).unwrap(), 22.55);
    assert_approx_eq!(interp.interpolate(&[0.19, 0.08, 0.65]).unwrap(), 26.85);
    // above x, above y, below z
    assert_approx_eq!(interp.interpolate(&[0.17, 0.33, 0.17]).unwrap(), 28.35);
    assert_approx_eq!(interp.interpolate(&[0.19, 0.36, 0.19]).unwrap(), 32.95);
    // above x, above y, above z
    assert_approx_eq!(interp.interpolate(&[0.17, 0.33, 0.63]).unwrap(), 30.65);
    assert_approx_eq!(interp.interpolate(&[0.19, 0.36, 0.65]).unwrap(), 35.25);
}

#[test]
fn test_linear_offset() {
    let interp = Interp3D::new(
        array![0., 1.],
        array![0., 1.],
        array![0., 1.],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    assert_approx_eq!(interp.interpolate(&[0.25, 0.65, 0.9]).unwrap(), 3.2);
}

#[test]
fn test_nearest() {
    let interp = Interp3D::new(
        array![0., 1.],
        array![0., 1.],
        array![0., 1.],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]],
        strategy::Nearest,
        Extrapolate::Error,
    )
    .unwrap();
    // Check that interpolating at grid points just retrieves the value
    let x = &interp.data.grid[0];
    let y = &interp.data.grid[1];
    let z = &interp.data.grid[2];
    for (i, x_i) in x.iter().enumerate() {
        for (j, y_j) in y.iter().enumerate() {
            for (k, z_k) in z.iter().enumerate() {
                assert_eq!(
                    interp.interpolate(&[*x_i, *y_j, *z_k]).unwrap(),
                    interp.data.values[[i, j, k]]
                );
            }
        }
    }
    assert_eq!(interp.interpolate(&[0., 0., 0.]).unwrap(), 0.);
    assert_eq!(interp.interpolate(&[0.25, 0.25, 0.25]).unwrap(), 0.);
    assert_eq!(interp.interpolate(&[0.25, 0.75, 0.25]).unwrap(), 2.);
    assert_eq!(interp.interpolate(&[0., 1., 0.]).unwrap(), 2.);
    assert_eq!(interp.interpolate(&[0.75, 0.25, 0.75]).unwrap(), 5.);
    assert_eq!(interp.interpolate(&[0.75, 0.75, 0.75]).unwrap(), 7.);
    assert_eq!(interp.interpolate(&[1., 1., 1.]).unwrap(), 7.);
}

#[test]
fn test_step() {
    let interp = Interp3D::new(
        array![0., 1.],
        array![0., 1.],
        array![0., 1.],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]],
        strategy::Step::from(strategy::StepDirection::Lower),
        Extrapolate::Error,
    )
    .unwrap();
    // At grid points: exact value
    let x = &interp.data.grid[0];
    let y = &interp.data.grid[1];
    let z = &interp.data.grid[2];
    for (i, xi) in x.iter().enumerate() {
        for (j, yj) in y.iter().enumerate() {
            for (k, zk) in z.iter().enumerate() {
                assert_eq!(
                    interp.interpolate(&[*xi, *yj, *zk]).unwrap(),
                    interp.data.values[[i, j, k]]
                );
            }
        }
    }
    // Between points: floor each dimension to index 0
    assert_eq!(interp.interpolate(&[0.3, 0.7, 0.6]).unwrap(), 0.); // floor→[0,0,0]
    assert_eq!(interp.interpolate(&[0.9, 0.4, 0.1]).unwrap(), 0.); // floor→[0,0,0]

    let interp_lower = Interp3D::new(
        array![0., 1.],
        array![0., 1.],
        array![0., 1.],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]],
        strategy::StepLower,
        Extrapolate::Error,
    )
    .unwrap();
    assert_eq!(interp_lower.interpolate(&[0.3, 0.7, 0.6]).unwrap(), 0.);

    // Uniform Upper (ceiling)
    let interp_upper = Interp3D::new(
        array![0., 1.],
        array![0., 1.],
        array![0., 1.],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]],
        strategy::Step::from(strategy::StepDirection::Upper),
        Extrapolate::Error,
    )
    .unwrap();
    assert_eq!(interp_upper.interpolate(&[0.3, 0.7, 0.6]).unwrap(), 7.); // ceil→[1,1,1]

    let interp_marker_upper = Interp3D::new(
        array![0., 1.],
        array![0., 1.],
        array![0., 1.],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]],
        strategy::StepUpper,
        Extrapolate::Error,
    )
    .unwrap();
    assert_eq!(
        interp_marker_upper.interpolate(&[0.3, 0.7, 0.6]).unwrap(),
        7.
    );

    // Per-dimension: Lower in x, Upper in y, Lower in z
    let interp_mixed = Interp3D::new(
        array![0., 1.],
        array![0., 1.],
        array![0., 1.],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]],
        strategy::Step(vec![
            strategy::StepDirection::Lower,
            strategy::StepDirection::Upper,
            strategy::StepDirection::Lower,
        ]),
        Extrapolate::Error,
    )
    .unwrap();
    assert_eq!(interp_mixed.interpolate(&[0.6, 0.4, 0.8]).unwrap(), 2.); // floor x→0, ceil y→1, floor z→0 → [0,1,0]
}

#[test]
fn test_extrapolate_inputs() {
    // Extrapolate::Extrapolate
    assert!(matches!(
        Interp3D::new(
            array![0.1, 1.1],
            array![0.2, 1.2],
            array![0.3, 1.3],
            array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],],
            strategy::Nearest,
            Extrapolate::Enable,
        )
        .unwrap_err(),
        ValidateError::ExtrapolateUnsupported
    ));
    // Extrapolate::Error
    let interp = Interp3D::new(
        array![0.1, 1.1],
        array![0.2, 1.2],
        array![0.3, 1.3],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    assert!(matches!(
        interp.interpolate(&[-1., -1., -1.]).unwrap_err(),
        InterpolateError::OutOfBounds(_)
    ));
    assert!(matches!(
        interp.interpolate(&[2., 2., 2.]).unwrap_err(),
        InterpolateError::OutOfBounds(_)
    ));
}

#[test]
fn test_extrapolate_fill() {
    let interp = Interp3D::new(
        array![0.1, 1.1],
        array![0.2, 1.2],
        array![0.3, 1.3],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],],
        strategy::Linear,
        Extrapolate::Fill(f64::NAN),
    )
    .unwrap();
    assert_approx_eq!(interp.interpolate(&[0.4, 0.4, 0.4]).unwrap(), 1.7);
    assert_approx_eq!(interp.interpolate(&[0.8, 0.8, 0.8]).unwrap(), 4.5);
    assert!(interp.interpolate(&[0., 0., 0.]).unwrap().is_nan());
    assert!(interp.interpolate(&[0., 0., 2.]).unwrap().is_nan());
    assert!(interp.interpolate(&[0., 2., 0.]).unwrap().is_nan());
    assert!(interp.interpolate(&[0., 2., 2.]).unwrap().is_nan());
    assert!(interp.interpolate(&[2., 0., 0.]).unwrap().is_nan());
    assert!(interp.interpolate(&[2., 0., 2.]).unwrap().is_nan());
    assert!(interp.interpolate(&[2., 2., 0.]).unwrap().is_nan());
    assert!(interp.interpolate(&[2., 2., 2.]).unwrap().is_nan());
}

#[test]
fn test_extrapolate_clamp() {
    let interp = Interp3D::new(
        array![0.1, 1.1],
        array![0.2, 1.2],
        array![0.3, 1.3],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],],
        strategy::Linear,
        Extrapolate::Clamp,
    )
    .unwrap();
    assert_eq!(interp.interpolate(&[-1., -1., -1.]).unwrap(), 0.);
    assert_eq!(interp.interpolate(&[2., 2., 2.]).unwrap(), 7.);
}

#[test]
fn test_batch_interpolate_matches_interpolate() {
    let interp = Interp3D::new(
        array![0.05, 0.10, 0.15],
        array![0.10, 0.20, 0.30],
        array![0.20, 0.40, 0.60],
        array![
            [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
            [[9., 10., 11.], [12., 13., 14.], [15., 16., 17.]],
            [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.],],
        ],
        strategy::Linear,
        Extrapolate::Enable,
    )
    .unwrap();
    let points = [[0.075, 0.25, 0.3], [0.05, 0.10, 0.20], [0.2, 0.4, 0.8]];
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
    let interp = Interp3D::new(
        array![0.1, 1.1],
        array![0.2, 1.2],
        array![0.3, 1.3],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],],
        strategy::Linear,
        Extrapolate::Clamp,
    )
    .unwrap();
    assert_eq!(
        interp
            .batch_interpolate(&[[-1., -1., -1.], [2., 2., 2.]])
            .unwrap(),
        vec![0., 7.]
    );
}

#[test]
fn test_batch_interpolate_error_aggregates_all_points() {
    let interp = Interp3D::new(
        array![0.1, 1.1],
        array![0.2, 1.2],
        array![0.3, 1.3],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    let err = interp
        .batch_interpolate(&[[0.4, 0.4, 0.4], [-1., -1., -1.], [2., 2., 2.]])
        .unwrap_err();
    let InterpolateError::OutOfBounds(failures) = err else {
        panic!("expected InterpolateError::OutOfBounds");
    };
    let offending: Vec<usize> = failures.iter().map(|at| at.index).collect();
    assert!(offending.contains(&1));
    assert!(offending.contains(&2));
    assert!(!offending.contains(&0));
}

#[test]
fn test_partialeq() {
    #[derive(PartialEq)]
    #[allow(unused)]
    struct MyStruct(InterpData3D<f64>);

    #[derive(PartialEq)]
    #[allow(unused)]
    struct MyStruct2(Interp3D<f64, strategy::Linear>);
}

#[test]
#[cfg(feature = "serde")]
fn test_serde() {
    let interp = Interp3D::new(
        array![0., 1.],
        array![0., 1., 2.],
        array![0., 1., 2., 3.],
        array![
            [
                [0.6, 0.8, 1.0, 1.2],
                [0.8, 1.0, 1.2, 1.4],
                [1.0, 1.2, 1.4, 1.6],
            ],
            [
                [0.8, 1.0, 1.2, 1.4],
                [1.0, 1.2, 1.4, 1.6],
                [1.2, 1.4, 1.6, 1.8],
            ],
        ],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();

    let ser = serde_json::to_string(&interp).unwrap();
    let de: Interp3D<_, _> = serde_json::from_str(&ser).unwrap();
    assert_eq!(interp, de);

    // simple format (new serialization output)
    let ser0 = "{\"grid\":[[0.0,1.0],[0.0,1.0,2.0],[0.0,1.0,2.0,3.0]],\"values\":[[[0.6,0.8,1.0,1.2],[0.8,1.0,1.2,1.4],[1.0,1.2,1.4,1.6]],[[0.8,1.0,1.2,1.4],[1.0,1.2,1.4,1.6],[1.2,1.4,1.6,1.8]]]}";
    let de0: InterpData3D<_> = serde_json::from_str(ser0).unwrap();
    assert_eq!(interp.data, de0);
    // mixed format (simple grid)
    let ser1 = "{\"grid\":[[0.0,1.0],[0.0,1.0,2.0],[0.0,1.0,2.0,3.0]],\"values\":{\"v\":1,\"dim\":[2,3,4],\"data\":[0.6,0.8,1.0,1.2,0.8,1.0,1.2,1.4,1.0,1.2,1.4,1.6,0.8,1.0,1.2,1.4,1.0,1.2,1.4,1.6,1.2,1.4,1.6,1.8]}}";
    let de1: InterpData3D<_> = serde_json::from_str(ser1).unwrap();
    assert_eq!(interp.data, de1);
    // mixed format (simple values)
    let ser2 = "{\"grid\":[{\"v\":1,\"dim\":[2],\"data\":[0.0,1.0]},{\"v\":1,\"dim\":[3],\"data\":[0.0,1.0,2.0]},{\"v\":1,\"dim\":[4],\"data\":[0.0,1.0,2.0,3.0]}],\"values\":[[[0.6,0.8,1.0,1.2],[0.8,1.0,1.2,1.4],[1.0,1.2,1.4,1.6]],[[0.8,1.0,1.2,1.4],[1.0,1.2,1.4,1.6],[1.2,1.4,1.6,1.8]]]}";
    let de2: InterpData3D<_> = serde_json::from_str(ser2).unwrap();
    assert_eq!(interp.data, de2);
    // complex format (legacy serialization output)
    let ser3 = "{\"grid\":[{\"v\":1,\"dim\":[2],\"data\":[0.0,1.0]},{\"v\":1,\"dim\":[3],\"data\":[0.0,1.0,2.0]},{\"v\":1,\"dim\":[4],\"data\":[0.0,1.0,2.0,3.0]}],\"values\":{\"v\":1,\"dim\":[2,3,4],\"data\":[0.6,0.8,1.0,1.2,0.8,1.0,1.2,1.4,1.0,1.2,1.4,1.6,0.8,1.0,1.2,1.4,1.0,1.2,1.4,1.6,1.2,1.4,1.6,1.8]}}";
    let de3: InterpData3D<_> = serde_json::from_str(ser3).unwrap();
    assert_eq!(interp.data, de3);
}
