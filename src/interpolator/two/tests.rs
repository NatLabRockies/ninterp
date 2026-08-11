use super::*;

#[test]
fn test_cubic_spline() {
    // f(x, y) = 2x + y: linear in both dims, reproduced exactly by any spline
    let interp = Interp2D::new(
        array![0., 1., 2.],
        array![0., 1., 2.],
        array![[0., 1., 2.], [2., 3., 4.], [4., 5., 6.]],
        strategy::CubicC2::natural(),
        Extrapolate::Enable,
    )
    .unwrap();
    // Knots
    assert_approx_eq!(interp.interpolate(&[1., 1.]).unwrap(), 3.);
    assert_approx_eq!(interp.interpolate(&[2., 0.]).unwrap(), 4.);
    // Midpoints
    assert_approx_eq!(interp.interpolate(&[0.5, 0.5]).unwrap(), 1.5);
    assert_approx_eq!(interp.interpolate(&[1.5, 1.0]).unwrap(), 4.);
    // Extrapolation
    assert_approx_eq!(interp.interpolate(&[3., 1.]).unwrap(), 7.);
    assert_approx_eq!(interp.interpolate(&[1., 3.]).unwrap(), 5.);
}

#[test]
fn test_cubic_spline_knot_exactness() {
    let interp = Interp2D::new(
        array![0., 1., 2., 3.],
        array![0., 1., 2., 3.],
        array![
            [0., 1., 4., 9.],
            [1., 2., 5., 10.],
            [4., 5., 8., 13.],
            [9., 10., 13., 18.],
        ], // f(x, y) = x^2 + y
        strategy::CubicC2::not_a_knot(),
        Extrapolate::Error,
    )
    .unwrap();
    let x = interp.data.grid[0].clone();
    let y = interp.data.grid[1].clone();
    for (i, xi) in x.iter().enumerate() {
        for (j, yj) in y.iter().enumerate() {
            assert_approx_eq!(
                interp.interpolate(&[*xi, *yj]).unwrap(),
                interp.data.values[[i, j]]
            );
        }
    }
}

#[test]
fn test_cubic_c2_interior_accuracy() {
    // f(x, y) = x^2*y + x*y^2: quadratic in each axis (well within cubic-spline
    // capacity), so a `NotAKnot` spline reproduces it exactly everywhere, not just at
    // grid points -- unlike knot-exactness, this exercises the corner cache's
    // mixed-partial term (d^2f/dxdy = 2x + 2y, non-constant) at interior points.
    fn f(x: f64, y: f64) -> f64 {
        x * x * y + x * y * y
    }
    let interp = Interp2D::new(
        array![0., 1., 2., 3.],
        array![0., 1., 2., 3.],
        array![
            [0., 0., 0., 0.],
            [0., 2., 6., 12.],
            [0., 6., 16., 30.],
            [0., 12., 30., 54.],
        ],
        strategy::CubicC2::not_a_knot(),
        Extrapolate::Error,
    )
    .unwrap();
    for &(x, y) in &[(0.5, 0.5), (1.5, 2.5), (2.5, 1.5), (0.25, 2.75)] {
        assert_approx_eq!(interp.interpolate(&[x, y]).unwrap(), f(x, y));
    }
}

#[test]
fn test_cubic_c2_cached_vs_uncached() {
    // `Strategy2D`'s corner-cache path (`compute_corner_cache` +
    // `spline_eval_corner_cached`) must agree with `StrategyND`'s unchanged
    // recursive-collapse path (`spline_eval_nd_cached`) on the same grid/values/BC.
    let grid = array![0., 1., 2., 3.];
    let values = array![
        [0., 1., 4., 9.],
        [1., 2., 5., 10.],
        [4., 5., 8., 13.],
        [9., 10., 13., 18.],
    ]; // f(x, y) = x^2 + y
    let interp2d = Interp2D::new(
        grid.clone(),
        grid.clone(),
        values.clone(),
        strategy::CubicC2::not_a_knot(),
        Extrapolate::Error,
    )
    .unwrap();
    let interp_nd = InterpND::new(
        vec![grid.clone(), grid],
        values.into_dyn(),
        strategy::CubicC2::not_a_knot(),
        Extrapolate::Error,
    )
    .unwrap();
    for &(x, y) in &[(0.5, 0.5), (1.5, 2.5), (2.5, 1.5), (0.25, 2.75)] {
        assert_approx_eq!(
            interp2d.interpolate(&[x, y]).unwrap(),
            interp_nd.interpolate(&[x, y]).unwrap()
        );
    }
}

#[test]
fn test_cubic_c2_clamped_short_axis() {
    // A `Clamped` axis with only 2 points must still validate under the corner-cache
    // upgrade: the second-pass BC substituted for `Clamped` (`Natural`, not `NotAKnot`)
    // has no minimum-point requirement -- regression test for that fallback choice.
    let mut strategy = strategy::CubicC2::natural();
    strategy.boundary_conditions = vec![
        strategy::CubicBoundaryConditions::Clamped {
            left: 1.,
            right: 1.,
        },
        strategy::CubicBoundaryConditions::Natural,
    ];
    let interp = Interp2D::new(
        array![0., 1.], // only 2 points on the Clamped axis
        array![0., 1., 2., 3.],
        array![[0., 1., 2., 3.], [1., 2., 3., 4.]], // f(x, y) = x + y
        strategy,
        Extrapolate::Error,
    )
    .unwrap();
    assert_approx_eq!(interp.interpolate(&[0.5, 1.5]).unwrap(), 2.0);
}

#[test]
fn test_invalid_args() {
    let interp = Interp2D::new(
        array![0.05, 0.10, 0.15],
        array![0.10, 0.20, 0.30],
        array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    // Wrong-length points on a concretely-typed `Interp2D` are caught at compile time
    // via the inherent `interpolate(&[D::Elem; N])`; the trait's checked path (used by
    // generic/`dyn` callers passing a real slice) still catches it at runtime.
    assert!(matches!(
        Interpolator::interpolate(&interp, &[]).unwrap_err(),
        InterpolateError::PointLength { .. }
    ));
    assert_eq!(interp.interpolate(&[0.075, 0.25]).unwrap(), 3.);
}

#[test]
fn test_linear() {
    let interp = Interp2D::new(
        array![0.05, 0.10, 0.15],
        array![0.10, 0.20, 0.30],
        array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    // Check that interpolating at grid points just retrieves the value
    let x = &interp.data.grid[0];
    let y = &interp.data.grid[1];
    let f_xy = &interp.data.values;
    for (i, x_i) in x.iter().enumerate() {
        for (j, y_j) in y.iter().enumerate() {
            assert_eq!(interp.interpolate(&[*x_i, *y_j]).unwrap(), f_xy[[i, j]]);
        }
    }
    assert_eq!(interp.interpolate(&[x[2], y[1]]).unwrap(), f_xy[[2, 1]]);
    assert_eq!(interp.interpolate(&[0.075, 0.25]).unwrap(), 3.);
}

#[test]
fn test_linear_offset() {
    let interp = Interp2D::new(
        array![0., 1.],
        array![0., 1.],
        array![[0., 1.], [2., 3.]],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    assert_approx_eq!(interp.interpolate(&[0.25, 0.65]).unwrap(), 1.15);
}

#[test]
fn test_linear_extrapolation() {
    let interp = Interp2D::new(
        array![0.05, 0.10, 0.15],
        array![0.10, 0.20, 0.30],
        array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
        strategy::Linear,
        Extrapolate::Enable,
    )
    .unwrap();
    // RHS are coplanar neighboring data planes according to:
    // https://www.ambrbit.com/TrigoCalc/Plan3D/PointsCoplanar.htm
    // below x, below y
    assert_approx_eq!(interp.interpolate(&[0.0, 0.0]).unwrap(), -4.);
    assert_approx_eq!(interp.interpolate(&[0.03, 0.04]).unwrap(), -1.8);
    // below x, above y
    assert_approx_eq!(interp.interpolate(&[0.0, 0.32]).unwrap(), -0.8);
    assert_approx_eq!(interp.interpolate(&[0.03, 0.36]).unwrap(), 1.4);
    // above x, below y
    assert_approx_eq!(interp.interpolate(&[0.17, 0.0]).unwrap(), 6.2);
    assert_approx_eq!(interp.interpolate(&[0.19, 0.04]).unwrap(), 7.8);
    // above x, above y
    assert_approx_eq!(interp.interpolate(&[0.17, 0.32]).unwrap(), 9.4);
    assert_approx_eq!(interp.interpolate(&[0.19, 0.36]).unwrap(), 11.);
}

#[test]
fn test_nearest() {
    let interp = Interp2D::new(
        array![0.05, 0.10, 0.15],
        array![0.10, 0.20, 0.30],
        array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
        strategy::Nearest,
        Extrapolate::Error,
    )
    .unwrap();
    // Check that interpolating at grid points just retrieves the value
    let x = &interp.data.grid[0];
    let y = &interp.data.grid[1];
    let f_xy = &interp.data.values;
    for (i, x_i) in x.iter().enumerate() {
        for (j, y_j) in y.iter().enumerate() {
            assert_eq!(interp.interpolate(&[*x_i, *y_j]).unwrap(), f_xy[[i, j]]);
        }
    }
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
fn test_step() {
    let grid_x = array![0., 1., 2.];
    let grid_y = array![0., 1., 2.];
    let values = array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]];

    // Uniform Lower (floor) in all dimensions
    let interp = Interp2DView::new(
        grid_x.view(),
        grid_y.view(),
        values.view(),
        strategy::Step::from(strategy::StepDirection::Lower),
        Extrapolate::Error,
    )
    .unwrap();
    let x = &interp.data.grid[0];
    let y = &interp.data.grid[1];
    let f = &interp.data.values;
    for (i, xi) in x.iter().enumerate() {
        for (j, yj) in y.iter().enumerate() {
            assert_eq!(interp.interpolate(&[*xi, *yj]).unwrap(), f[[i, j]]);
        }
    }
    assert_eq!(interp.interpolate(&[0.7, 1.4]).unwrap(), f[[0, 1]]); // floor x→0, floor y→1
    assert_eq!(interp.interpolate(&[1.9, 0.1]).unwrap(), f[[1, 0]]); // floor x→1, floor y→0

    let interp_lower = Interp2DView::new(
        grid_x.view(),
        grid_y.view(),
        values.view(),
        strategy::StepLower,
        Extrapolate::Error,
    )
    .unwrap();
    assert_eq!(interp_lower.interpolate(&[0.7, 1.4]).unwrap(), f[[0, 1]]);

    let interp_upper = Interp2DView::new(
        grid_x.view(),
        grid_y.view(),
        values.view(),
        strategy::StepUpper,
        Extrapolate::Error,
    )
    .unwrap();
    assert_eq!(interp_upper.interpolate(&[0.7, 1.4]).unwrap(), f[[1, 2]]);

    // Per-dimension: Lower in x, Upper in y
    let interp_mixed = Interp2DView::new(
        grid_x.view(),
        grid_y.view(),
        values.view(),
        strategy::Step(vec![
            strategy::StepDirection::Lower,
            strategy::StepDirection::Upper,
        ]),
        Extrapolate::Error,
    )
    .unwrap();
    assert_eq!(interp_mixed.interpolate(&[0.7, 1.4]).unwrap(), f[[0, 2]]); // floor x→0, ceil y→2
    assert_eq!(interp_mixed.interpolate(&[1.3, 0.8]).unwrap(), f[[1, 1]]); // floor x→1, ceil y→1

    // Invalid: 3 directions for 2-D
    assert!(Interp2DView::new(
        grid_x.view(),
        grid_y.view(),
        values.view(),
        strategy::Step(vec![
            strategy::StepDirection::Lower,
            strategy::StepDirection::Lower,
            strategy::StepDirection::Lower,
        ]),
        Extrapolate::Error,
    )
    .is_err());
}

#[test]
fn test_extrapolate_inputs() {
    // Extrapolate::Extrapolate
    assert!(matches!(
        Interp2D::new(
            array![0.1, 1.1],
            array![0.2, 1.2],
            array![[0., 1.], [2., 3.]],
            strategy::Nearest,
            Extrapolate::Enable,
        )
        .unwrap_err(),
        ValidateError::ExtrapolateUnsupported
    ));
    // Extrapolate::Error
    let interp = Interp2D::new(
        array![0.1, 1.1],
        array![0.2, 1.2],
        array![[0., 1.], [2., 3.]],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    assert!(matches!(
        interp.interpolate(&[-1., -1.]).unwrap_err(),
        InterpolateError::OutOfBounds(_)
    ));
    assert!(matches!(
        interp.interpolate(&[2., 2.]).unwrap_err(),
        InterpolateError::OutOfBounds(_)
    ));
}

#[test]
fn test_extrapolate_fill() {
    let interp = Interp2D::new(
        array![0.1, 1.1],
        array![0.2, 1.2],
        array![[0., 1.], [2., 3.]],
        strategy::Linear,
        Extrapolate::Fill(f64::NAN),
    )
    .unwrap();
    assert_eq!(interp.interpolate(&[0.5, 0.5]).unwrap(), 1.1);
    assert_eq!(interp.interpolate(&[0.1, 1.2]).unwrap(), 1.);
    assert!(interp.interpolate(&[0., 0.]).unwrap().is_nan());
    assert!(interp.interpolate(&[0., 2.]).unwrap().is_nan());
    assert!(interp.interpolate(&[2., 0.]).unwrap().is_nan());
    assert!(interp.interpolate(&[2., 2.]).unwrap().is_nan());
}

#[test]
fn test_dyn_strategy() {
    let mut interp = Interp2D::new(
        array![0., 1.],
        array![0., 1.],
        array![[0., 1.], [2., 3.]],
        Box::new(strategy::Linear) as Box<dyn Strategy2D<_>>,
        Extrapolate::Error,
    )
    .unwrap();
    assert_eq!(interp.interpolate(&[0.2, 0.]).unwrap(), 0.4);
    interp.set_strategy(Box::new(strategy::Nearest)).unwrap();
    assert_eq!(interp.interpolate(&[0.2, 0.]).unwrap(), 0.);
}

#[test]
fn test_dyn_strategy_batch_interpolate() {
    // Strategy2D<D>'s batch_interpolate/batch_interpolate_fast for
    // Box<dyn Strategy2D<D>> forward to the wrapped concrete strategy; confirm that
    // forward actually produces correct output, not just that it compiles.
    let interp = Interp2D::new(
        array![0., 1.],
        array![0., 1.],
        array![[0., 1.], [2., 3.]],
        Box::new(strategy::Linear) as Box<dyn Strategy2D<_>>,
        Extrapolate::Error,
    )
    .unwrap();
    let points = [[0.2, 0.], [1., 1.]];
    assert_eq!(interp.batch_interpolate(&points).unwrap(), vec![0.4, 3.]);
    assert_eq!(interp.batch_interpolate_fast(&points), vec![0.4, 3.]);
}

#[test]
fn test_set_strategy_runs_init() {
    // `Step`'s `validate` checks its direction count against dimensionality,
    // so swapping in a `Step` with the wrong count via `set_strategy` must
    // surface that error rather than silently leaving the strategy unvalidated.
    let mut interp: Interp2D<_, strategy::enums::Strategy2DEnum> = Interp2D::new(
        array![0., 1.],
        array![0., 1.],
        array![[0., 1.], [2., 3.]],
        strategy::Linear.into(),
        Extrapolate::Error,
    )
    .unwrap();
    let bad_step = strategy::Step(vec![strategy::StepDirection::Lower; 3]);
    assert!(matches!(
        interp.set_strategy(bad_step).unwrap_err(),
        ValidateError::Other(_)
    ));
}

#[test]
fn test_set_strategy_runs_validate() {
    // Same as `test_set_strategy_runs_init`, but for `LinearUniform`'s uniform-grid
    // check: `Linear` doesn't care about grid spacing, so a non-uniform grid builds
    // fine, but swapping to `LinearUniform` via `set_strategy` must catch it via
    // `Strategy2D::validate` rather than silently accepting a strategy that will
    // produce wrong results at query time.
    let mut interp: Interp2D<_, strategy::enums::Strategy2DEnum> = Interp2D::new(
        array![0., 1., 5.], // non-uniform
        array![0., 1., 2.],
        array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
        strategy::Linear.into(),
        Extrapolate::Error,
    )
    .unwrap();
    assert!(matches!(
        interp.set_strategy(strategy::LinearUniform).unwrap_err(),
        // grid[0] = [0., 1., 5.]: the interval after index 1 is 4, not 1
        ValidateError::NonUniform { dim: 0, index: 1 }
    ));
}

#[test]
fn test_validate_strategy() {
    // `LinearUniform`'s uniform-grid check is a pure invariant check (`Strategy2D::validate`),
    // not a precomputation (`Strategy2D::init`), so `Interpolator::validate` (which calls
    // `validate_strategy` internally) catches directly mutating `data` to break that
    // invariant, while `init_strategy` (a no-op for `LinearUniform`, since it caches
    // nothing) does not.
    let mut interp = Interp2D::new(
        array![0., 1., 2.],
        array![0., 1., 2.],
        array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
        strategy::LinearUniform,
        Extrapolate::Error,
    )
    .unwrap();
    interp.data.grid[0] = array![0., 1., 5.]; // still monotonic, no longer uniform
    assert!(interp.validate().is_err());
    assert!(interp.validate_strategy().is_err());
    assert!(interp.init_strategy().is_ok());
}

#[test]
fn test_extrapolate_clamp() {
    let interp = Interp2D::new(
        array![0.1, 1.1],
        array![0.2, 1.2],
        array![[0., 1.], [2., 3.]],
        strategy::Linear,
        Extrapolate::Clamp,
    )
    .unwrap();
    assert_eq!(interp.interpolate(&[-1., -1.]).unwrap(), 0.);
    assert_eq!(interp.interpolate(&[2., 2.]).unwrap(), 3.);
}

#[test]
fn test_batch_interpolate_matches_interpolate() {
    let interp = Interp2D::new(
        array![0.05, 0.10, 0.15],
        array![0.10, 0.20, 0.30],
        array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
        strategy::Linear,
        Extrapolate::Enable,
    )
    .unwrap();
    let points = [[0.075, 0.25], [0.05, 0.10], [0.2, 0.4]];
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
    let interp = Interp2D::new(
        array![0.1, 1.1],
        array![0.2, 1.2],
        array![[0., 1.], [2., 3.]],
        strategy::Linear,
        Extrapolate::Clamp,
    )
    .unwrap();
    assert_eq!(
        interp.batch_interpolate(&[[-1., -1.], [2., 2.]]).unwrap(),
        vec![0., 3.]
    );
}

#[test]
fn test_batch_interpolate_error_aggregates_all_points() {
    let interp = Interp2D::new(
        array![0.1, 1.1],
        array![0.2, 1.2],
        array![[0., 1.], [2., 3.]],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    let err = interp
        .batch_interpolate(&[[0.5, 0.5], [-1., -1.], [2., 2.]])
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
fn test_batch_interpolate_dyn() {
    let interp: Box<dyn Interpolator<f64>> = Box::new(
        Interp2D::new(
            array![0.05, 0.10, 0.15],
            array![0.10, 0.20, 0.30],
            array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap(),
    );
    let points: [&[f64]; 2] = [&[0.075, 0.25], &[0.05, 0.10]];
    assert_eq!(interp.batch_interpolate(&points).unwrap(), vec![3., 0.]);
    assert_eq!(interp.batch_interpolate_fast(&points), vec![3., 0.]);
}

#[test]
fn test_dyn_interpolator() {
    let interp = Interp2D::new(
        array![0.05, 0.10, 0.15],
        array![0.10, 0.20, 0.30],
        array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    let points: [&[f64]; 2] = [&[0.075, 0.25], &[0.05, 0.10]];

    let boxed: Box<dyn AnyInterpolator<f64>> = Box::new(interp.clone());
    assert_eq!(boxed.interpolate(&[0.075, 0.25]).unwrap(), 3.);
    assert_eq!(
        boxed.batch_interpolate(&points).unwrap(),
        interp
            .batch_interpolate(&[[0.075, 0.25], [0.05, 0.10]])
            .unwrap(),
    );
    assert!(matches!(
        boxed.interpolate(&[]).unwrap_err(),
        InterpolateError::PointLength { expected: 2, .. }
    ));
    assert_eq!(
        boxed.as_any().downcast_ref::<Interp2D<f64, _>>(),
        Some(&interp)
    );
}

#[test]
fn test_batch_interpolate_into_matches_interpolate() {
    let interp = Interp2D::new(
        array![0.05, 0.10, 0.15],
        array![0.10, 0.20, 0.30],
        array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    let points = [[0.075, 0.25], [0.05, 0.10]];
    let batched = interp.batch_interpolate(&points).unwrap();
    let mut out = vec![0.0; points.len()];
    interp.batch_interpolate_into(&points, &mut out).unwrap();
    assert_eq!(out, batched);
}

#[test]
fn test_batch_interpolate_into_output_length_error() {
    let interp = Interp2D::new(
        array![0.05, 0.10, 0.15],
        array![0.10, 0.20, 0.30],
        array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    let mut out = vec![0.0; 1];
    assert!(matches!(
        interp.batch_interpolate_into(&[[0.075, 0.25], [0.05, 0.10]], &mut out),
        Err(InterpolateError::OutputLength {
            expected: 2,
            found: 1
        })
    ));
}

#[test]
fn test_batch_interpolate_fast_into_matches_interpolate() {
    let interp = Interp2D::new(
        array![0.05, 0.10, 0.15],
        array![0.10, 0.20, 0.30],
        array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    let batched_fast = interp.batch_interpolate_fast(&[[0.075, 0.25], [0.05, 0.10]]);
    let mut out = vec![0.0; 2];
    interp.batch_interpolate_fast_into(&[[0.075, 0.25], [0.05, 0.10]], &mut out);
    assert_eq!(out, batched_fast);
}

#[test]
#[should_panic(expected = "batch_interpolate_fast_into: length mismatch")]
fn test_batch_interpolate_fast_into_length_mismatch() {
    let interp = Interp2D::new(
        array![0.05, 0.10, 0.15],
        array![0.10, 0.20, 0.30],
        array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    let mut out = vec![0.0; 1];
    interp.batch_interpolate_fast_into(&[[0.075, 0.25], [0.05, 0.10]], &mut out);
}

#[test]
fn test_batch_interpolate_into_clamp() {
    let interp = Interp2D::new(
        array![0.05, 0.10, 0.15],
        array![0.10, 0.20, 0.30],
        array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
        strategy::Linear,
        Extrapolate::Clamp,
    )
    .unwrap();
    let batched = interp.batch_interpolate(&[[-1., -1.], [2., 2.]]).unwrap();
    let mut out = vec![0.0; 2];
    interp
        .batch_interpolate_into(&[[-1., -1.], [2., 2.]], &mut out)
        .unwrap();
    assert_eq!(out, batched);
}

#[test]
fn test_batch_interpolate_into_fill() {
    let interp = Interp2D::new(
        array![0.05, 0.10, 0.15],
        array![0.10, 0.20, 0.30],
        array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
        strategy::Linear,
        Extrapolate::Fill(99.0),
    )
    .unwrap();
    let mut out = vec![0.0; 3];
    interp
        .batch_interpolate_into(&[[0.075, 0.25], [-1., -1.], [0.05, 0.10]], &mut out)
        .unwrap();
    assert_eq!(out[0], 3.0); // in-bounds
    assert_eq!(out[1], 99.0); // out-of-bounds, filled
    assert_eq!(out[2], 0.0); // in-bounds
}

#[test]
fn test_batch_interpolate_into_error_aggregates_all_points() {
    let interp = Interp2D::new(
        array![0.05, 0.10, 0.15],
        array![0.10, 0.20, 0.30],
        array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    let mut out = vec![0.0; 3];
    let err = interp
        .batch_interpolate_into(&[[0.5, 0.5], [-1., -1.], [2., 2.]], &mut out)
        .unwrap_err();
    match err {
        InterpolateError::OutOfBounds(failures) => {
            // Should report all out-of-bounds points
            let offending: Vec<usize> = failures.iter().map(|at| at.index).collect();
            assert!(offending.contains(&1));
            assert!(offending.contains(&2));
        }
        _ => panic!("expected InterpolateError::OutOfBounds"),
    }
}

#[test]
fn test_batch_interpolate_into_dyn() {
    let interp = Interp2D::new(
        array![0.05, 0.10, 0.15],
        array![0.10, 0.20, 0.30],
        array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    let boxed: Box<dyn Interpolator<_>> = Box::new(interp.clone());
    let points: [&[f64]; 2] = [&[0.075, 0.25], &[0.05, 0.10]];
    let batched = boxed.batch_interpolate(&points).unwrap();
    let mut out = vec![0.0; points.len()];
    boxed.batch_interpolate_into(&points, &mut out).unwrap();
    assert_eq!(out, batched);

    let mut out_fast = vec![0.0; points.len()];
    boxed.batch_interpolate_fast_into(&points, &mut out_fast);
    let batched_fast = boxed.batch_interpolate_fast(&points);
    assert_eq!(out_fast, batched_fast);
}

#[test]
fn test_partialeq() {
    #[derive(PartialEq)]
    #[allow(unused)]
    struct MyStruct(InterpData2D<f64>);

    #[derive(PartialEq)]
    #[allow(unused)]
    struct MyStruct2(Interp2D<f64, strategy::Linear>);
}

#[test]
#[cfg(feature = "serde")]
fn test_serde() {
    let interp = Interp2D::new(
        array![0.05, 0.10, 0.15],
        array![0.10, 0.20, 0.30],
        array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
        strategy::Linear,
        Extrapolate::Enable,
    )
    .unwrap();

    let ser = serde_json::to_string(&interp).unwrap();
    let de: Interp2D<f64, strategy::Linear> = serde_json::from_str(&ser).unwrap();
    assert_eq!(interp, de);

    // simple format (new serialization output)
    let ser0 = "{\"grid\":[[0.05,0.1,0.15],[0.1,0.2,0.3]],\"values\":[[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]]}";
    let de0: InterpData2D<_> = serde_json::from_str(ser0).unwrap();
    assert_eq!(interp.data, de0);
    // mixed format (simple grid)
    let ser1 = "{\"grid\":[[0.05,0.1,0.15],[0.1,0.2,0.3]],\"values\":{\"v\":1,\"dim\":[3,3],\"data\":[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]}}";
    let de1: InterpData2D<_> = serde_json::from_str(ser1).unwrap();
    assert_eq!(interp.data, de1);
    // mixed format (simple values)
    let ser2 = "{\"grid\":[{\"v\":1,\"dim\":[3],\"data\":[0.05,0.1,0.15]},{\"v\":1,\"dim\":[3],\"data\":[0.1,0.2,0.3]}],\"values\":[[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]]}";
    let de2: InterpData2D<_> = serde_json::from_str(ser2).unwrap();
    assert_eq!(interp.data, de2);
    // complex format (legacy serialization output)
    let ser3 = "{\"grid\":[{\"v\":1,\"dim\":[3],\"data\":[0.05,0.1,0.15]},{\"v\":1,\"dim\":[3],\"data\":[0.1,0.2,0.3]}],\"values\":{\"v\":1,\"dim\":[3,3],\"data\":[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]}}";
    let de3: InterpData2D<_> = serde_json::from_str(ser3).unwrap();
    assert_eq!(interp.data, de3);
}
