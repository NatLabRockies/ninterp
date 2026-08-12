use super::*;

#[test]
fn test_cubic_spline() {
    // f(x, y) = 2x + y: linear, reproduced exactly by any spline
    let interp = InterpND::new(
        vec![array![0., 1., 2.], array![0., 1., 2.]],
        array![[0., 1., 2.], [2., 3., 4.], [4., 5., 6.]].into_dyn(),
        strategy::CubicC2::natural(),
        Extrapolate::Enable,
    )
    .unwrap();
    assert_approx_eq!(interp.interpolate(&[0.5, 0.5]).unwrap(), 1.5);
    assert_approx_eq!(interp.interpolate(&[1., 1.]).unwrap(), 3.);
    assert_approx_eq!(interp.interpolate(&[3., 1.]).unwrap(), 7.); // extrapolation
}

#[test]
fn test_cubic_spline_knot_exactness() {
    let interp = InterpND::new(
        vec![array![0., 1., 2., 3.], array![0., 1., 2., 3.]],
        array![
            [0., 1., 4., 9.],
            [1., 2., 5., 10.],
            [4., 5., 8., 13.],
            [9., 10., 13., 18.],
        ]
        .into_dyn(),
        strategy::CubicC2::natural(),
        Extrapolate::Error,
    )
    .unwrap();
    for i in 0..4usize {
        for j in 0..4usize {
            let xi = interp.data.grid[0][i];
            let yj = interp.data.grid[1][j];
            assert_approx_eq!(
                interp.interpolate(&[xi, yj]).unwrap(),
                interp.data.values[[i, j]]
            );
        }
    }
}

#[test]
fn test_linear_0d() {
    let interp = InterpND::new(
        vec![array![]],
        array![0.5].into_dyn(),
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    assert_eq!(interp.interpolate(&[]).unwrap(), 0.5);
}

#[test]
fn test_cubic_spline_0d() {
    let interp = InterpND::new(
        vec![array![]],
        array![0.5].into_dyn(),
        strategy::CubicC2::natural(),
        Extrapolate::Error,
    )
    .unwrap();
    assert_eq!(interp.interpolate(&[]).unwrap(), 0.5);
}

#[test]
fn test_cubic_c2_periodic_outer_axis() {
    // Regression coverage: `Periodic` on the outer axis (axis 0, re-solved on every
    // `interpolate()` call via `spline_eval_1d`) used to run an exact `y[0] != y[n]`
    // equality check against *derived* intermediate values from the inner-axis spline
    // evaluation, not raw grid data -- values that can differ from their
    // mathematically-equal counterpart by a few ULPs, causing spurious failures. That
    // check is gone now; this just confirms Periodic on an outer axis still
    // interpolates successfully.
    let interp = InterpND::new(
        vec![array![0., 1., 2., 3.], array![0., 1., 2.]],
        array![
            [0., 1., 2.],
            [1., 3., 2.],
            [2., 1., 3.],
            [0., 1., 2.], // matches the x=0 row, by Periodic's convention
        ]
        .into_dyn(),
        strategy::CubicC2::new(vec![
            strategy::CubicBoundaryConditions::Periodic,
            strategy::CubicBoundaryConditions::Natural,
        ]),
        Extrapolate::Error,
    )
    .unwrap();
    // No exact-value oracle needed here -- this is regression coverage for the removed
    // check, not a numerical-accuracy test.
    for &(x, y) in &[(0.5, 0.5), (2.5, 1.5), (3.0, 1.0), (0.0, 1.0)] {
        assert!(interp.interpolate(&[x, y]).unwrap().is_finite());
    }
}

#[test]
fn test_cubic_c2_not_a_knot_cubic_exact() {
    // f(x, y) = x^3 + x^2*y + x*y^2 + y^3: same genuine-cubic cross-term function as the
    // `Interp2D` version. `test_cubic_c2_cached_vs_uncached` (in two/tests.rs) only
    // proves `InterpND` agrees with `Interp2D` on a zero-mixed-partial function
    // (f(x,y)=x^2+y); it doesn't independently confirm `InterpND`'s own outer-axis
    // recursive-collapse path (`spline_eval_1d`, not `Interp2D`'s corner-derivative
    // cache) reproduces a genuine bicubic -- one with a nonzero, non-constant mixed
    // partial d^2f/dxdy = 2x + 2y -- exactly rather than approximately.
    fn f(x: f64, y: f64) -> f64 {
        x.powi(3) + x.powi(2) * y + x * y.powi(2) + y.powi(3)
    }
    let grid = [0., 1., 2., 3.];
    let interp = InterpND::new(
        vec![array![0., 1., 2., 3.], array![0., 1., 2., 3.]],
        Array2::from_shape_fn((4, 4), |(i, j)| f(grid[i], grid[j])).into_dyn(),
        strategy::CubicC2::not_a_knot(),
        Extrapolate::Error,
    )
    .unwrap();
    for &(x, y) in &[(0.5, 0.5), (1.5, 2.5), (2.5, 1.5), (0.25, 2.75)] {
        assert_approx_eq!(interp.interpolate(&[x, y]).unwrap(), f(x, y));
    }
}

#[test]
fn test_cubic_c2_clamped_uses_given_derivative() {
    // Differential check, same reasoning as the 1D/2D versions: exact reproduction alone
    // can't tell "Clamped used the supplied derivative" apart from "Clamped silently
    // behaved like NotAKnot", since NotAKnot reproduces this same separable-cubic data
    // exactly with no derivative info at all. Unlike the 1D/2D versions, this exercises
    // `InterpND`'s own `Clamped` handling on an outer axis (plain `compute_m` via
    // `spline_eval_1d`, not the 2D/3D corner cache's Clamped-to-Natural substitution for
    // the mixed-partial second pass) -- a code path `Clamped` had never been run through
    // at all before this test, let alone verified to actually use its supplied value.
    fn f(x: f64, y: f64) -> f64 {
        x.powi(3) + y.powi(3)
    }
    let grid = [0., 1., 2., 3.];
    let interp = InterpND::new(
        vec![array![0., 1., 2., 3.], array![0., 1., 2., 3.]],
        Array2::from_shape_fn((4, 4), |(i, j)| f(grid[i], grid[j])).into_dyn(),
        strategy::CubicC2::clamped(999., 999.), // true derivatives are 0 and 27
        Extrapolate::Error,
    )
    .unwrap();
    let wrong = interp.interpolate(&[0.5, 0.5]).unwrap();
    assert!(
        (wrong - f(0.5, 0.5)).abs() > 1.0,
        "Clamped(999, 999) gave {wrong}, suspiciously close to the true f(0.5, 0.5) = {} \
         as if the supplied derivatives were ignored",
        f(0.5, 0.5)
    );
}

#[test]
fn test_cubic_c2_periodic_scipy_oracle() {
    // Same additive-corner-derivative reasoning and ground truth as
    // `Interp2D::test_cubic_c2_2d_periodic` (product of two independent periodic 1D
    // splines, scipy `CubicSpline(bc_type='periodic')`), but built via `InterpND`
    // directly with the 5-point axis first (outer, re-solved per query via
    // `spline_eval_1d`) and the 3-point axis second (inner, cached at construction) --
    // the same axis roles `test_cubic_c2_periodic_outer_axis` above uses, but with a
    // real numerical oracle instead of just an is-finite check. This is `InterpND`'s
    // own periodic outer-axis solve, independent of `Interp2D`'s corner-cache path.
    let interp = InterpND::new(
        vec![array![0., 1., 2., 3., 4.], array![0., 1., 2.]],
        array![
            [2., 1., 2.],
            [4., 2., 4.],
            [2., 1., 2.],
            [6., 3., 6.],
            [2., 1., 2.],
        ]
        .into_dyn(),
        strategy::CubicC2::periodic(),
        Extrapolate::Error,
    )
    .unwrap();

    // (query x, query y, scipy-derived S_A(x) * S_B(y))
    let cases = [
        (0.5, 0.5, 2.109375),
        (2.5, 1.5, 3.140625),
        (3.5, 0.5, 3.140625),
        (0.25, 1.75, 1.9373779296875),
    ];

    for (x, y, expected) in cases {
        assert_approx_eq!(interp.interpolate(&[x, y]).unwrap(), expected);
    }
}

#[test]
fn test_cubic_c2_notaknot_enough_points() {
    // NotAKnot requires >= 4 points per axis; axis 0 has enough on its own, so this
    // pins down that `StrategyND::validate` checks every axis, not just the first.
    let result = InterpND::new(
        vec![array![0., 1., 2., 3.], array![0., 1., 2.]],
        Array2::from_shape_fn((4, 3), |(i, j)| (i + j) as f64).into_dyn(),
        strategy::CubicC2::not_a_knot(),
        Extrapolate::Error,
    );
    assert!(
        result.is_err(),
        "NotAKnot with a 3-point axis should fail validation, got Ok"
    );
    let result = InterpND::new(
        vec![array![0., 1., 2., 3.], array![0., 1., 2., 3.]],
        Array2::from_shape_fn((4, 4), |(i, j)| (i + j) as f64).into_dyn(),
        strategy::CubicC2::not_a_knot(),
        Extrapolate::Error,
    );
    assert!(
        result.is_ok(),
        "NotAKnot with all axes >= 4 points should succeed, got Err: {:?}",
        result.unwrap_err()
    );
}

#[test]
fn test_linear() {
    let interp = InterpND::new(
        vec![
            array![0.05, 0.10, 0.15],
            array![0.10, 0.20, 0.30],
            array![0.20, 0.40, 0.60],
        ],
        array![
            [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
            [[9., 10., 11.], [12., 13., 14.], [15., 16., 17.]],
            [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.]],
        ]
        .into_dyn(),
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    // Check that interpolating at grid points just retrieves the value
    let x = &interp.data.grid[0];
    let y = &interp.data.grid[1];
    let z = &interp.data.grid[2];
    for i in 0..x.len() {
        for j in 0..y.len() {
            for k in 0..z.len() {
                assert_eq!(
                    interp.interpolate(&[x[i], y[j], z[k]]).unwrap(),
                    interp.data.values[[i, j, k]]
                );
            }
        }
    }
    assert_approx_eq!(interp.interpolate(&[x[0], y[0], 0.3]).unwrap(), 0.5);
    assert_approx_eq!(interp.interpolate(&[x[0], 0.15, z[0]]).unwrap(), 1.5);
    assert_approx_eq!(interp.interpolate(&[x[0], 0.15, 0.3]).unwrap(), 2.0);
    assert_approx_eq!(interp.interpolate(&[0.075, y[0], z[0]]).unwrap(), 4.5);
    assert_approx_eq!(interp.interpolate(&[0.075, y[0], 0.3]).unwrap(), 5.);
    assert_approx_eq!(interp.interpolate(&[0.075, 0.15, z[0]]).unwrap(), 6.);
}

#[test]
fn test_dyn_interpolator() {
    let interp = InterpND::new(
        vec![
            array![0.05, 0.10, 0.15],
            array![0.10, 0.20, 0.30],
            array![0.20, 0.40, 0.60],
        ],
        array![
            [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
            [[9., 10., 11.], [12., 13., 14.], [15., 16., 17.]],
            [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.]],
        ]
        .into_dyn(),
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    let points: [&[f64]; 2] = [&[0.05, 0.10, 0.20], &[0.15, 0.30, 0.60]];

    let boxed: Box<dyn AnyInterpolator<f64>> = Box::new(interp.clone());
    assert_eq!(
        boxed.interpolate(&[0.05, 0.10, 0.20]).unwrap(),
        interp.interpolate(&[0.05, 0.10, 0.20]).unwrap(),
    );
    assert_eq!(
        boxed.batch_interpolate(&points).unwrap(),
        interp.batch_interpolate(&points).unwrap(),
    );
    assert!(matches!(
        boxed.interpolate(&[]).unwrap_err(),
        InterpolateError::PointLength { expected: 3, .. }
    ));
    assert_eq!(
        boxed.as_any().downcast_ref::<InterpND<f64, _>>(),
        Some(&interp)
    );
}

#[test]
fn test_linear_offset() {
    let interp = InterpND::new(
        vec![array![0., 1.], array![0., 1.], array![0., 1.]],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    assert_approx_eq!(interp.interpolate(&[0.25, 0.65, 0.9]).unwrap(), 3.2)
}

#[test]
fn test_linear_extrapolation_2d() {
    let interp_2d = crate::interpolator::Interp2D::new(
        array![0.05, 0.10, 0.15],
        array![0.10, 0.20, 0.30],
        array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
        strategy::Linear,
        Extrapolate::Enable,
    )
    .unwrap();
    let interp_nd = InterpND::new(
        vec![array![0.05, 0.10, 0.15], array![0.10, 0.20, 0.30]],
        array![[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]].into_dyn(),
        strategy::Linear,
        Extrapolate::Enable,
    )
    .unwrap();
    // below x, below y
    assert_eq!(
        interp_2d.interpolate(&[0.0, 0.0]).unwrap(),
        interp_nd.interpolate(&[0.0, 0.0]).unwrap()
    );
    assert_eq!(
        interp_2d.interpolate(&[0.03, 0.04]).unwrap(),
        interp_nd.interpolate(&[0.03, 0.04]).unwrap(),
    );
    // below x, above y
    assert_eq!(
        interp_2d.interpolate(&[0.0, 0.32]).unwrap(),
        interp_nd.interpolate(&[0.0, 0.32]).unwrap(),
    );
    assert_eq!(
        interp_2d.interpolate(&[0.03, 0.36]).unwrap(),
        interp_nd.interpolate(&[0.03, 0.36]).unwrap()
    );
    // above x, below y
    assert_eq!(
        interp_2d.interpolate(&[0.17, 0.0]).unwrap(),
        interp_nd.interpolate(&[0.17, 0.0]).unwrap(),
    );
    assert_eq!(
        interp_2d.interpolate(&[0.19, 0.04]).unwrap(),
        interp_nd.interpolate(&[0.19, 0.04]).unwrap(),
    );
    // above x, above y
    assert_eq!(
        interp_2d.interpolate(&[0.17, 0.32]).unwrap(),
        interp_nd.interpolate(&[0.17, 0.32]).unwrap()
    );
    assert_eq!(
        interp_2d.interpolate(&[0.19, 0.36]).unwrap(),
        interp_nd.interpolate(&[0.19, 0.36]).unwrap()
    );
}

#[test]
fn test_linear_extrapolate_3d() {
    let interp_3d = crate::interpolator::Interp3D::new(
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
    let interp_nd = InterpND::new(
        vec![
            array![0.05, 0.10, 0.15],
            array![0.10, 0.20, 0.30],
            array![0.20, 0.40, 0.60],
        ],
        array![
            [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]],
            [[9., 10., 11.], [12., 13., 14.], [15., 16., 17.]],
            [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.]],
        ]
        .into_dyn(),
        strategy::Linear,
        Extrapolate::Enable,
    )
    .unwrap();
    // below x, below y, below z
    assert_eq!(
        interp_3d.interpolate(&[0.01, 0.06, 0.17]).unwrap(),
        interp_nd.interpolate(&[0.01, 0.06, 0.17]).unwrap()
    );
    assert_eq!(
        interp_3d.interpolate(&[0.02, 0.08, 0.19]).unwrap(),
        interp_nd.interpolate(&[0.02, 0.08, 0.19]).unwrap()
    );
    // below x, below y, above z
    assert_eq!(
        interp_3d.interpolate(&[0.01, 0.06, 0.63]).unwrap(),
        interp_nd.interpolate(&[0.01, 0.06, 0.63]).unwrap()
    );
    assert_eq!(
        interp_3d.interpolate(&[0.02, 0.08, 0.65]).unwrap(),
        interp_nd.interpolate(&[0.02, 0.08, 0.65]).unwrap()
    );
    // below x, above y, below z
    assert_eq!(
        interp_3d.interpolate(&[0.01, 0.33, 0.17]).unwrap(),
        interp_nd.interpolate(&[0.01, 0.33, 0.17]).unwrap()
    );
    assert_eq!(
        interp_3d.interpolate(&[0.02, 0.36, 0.19]).unwrap(),
        interp_nd.interpolate(&[0.02, 0.36, 0.19]).unwrap()
    );
    // below x, above y, above z
    assert_eq!(
        interp_3d.interpolate(&[0.01, 0.33, 0.63]).unwrap(),
        interp_nd.interpolate(&[0.01, 0.33, 0.63]).unwrap()
    );
    assert_eq!(
        interp_3d.interpolate(&[0.02, 0.36, 0.65]).unwrap(),
        interp_nd.interpolate(&[0.02, 0.36, 0.65]).unwrap()
    );
    // above x, below y, below z
    assert_eq!(
        interp_3d.interpolate(&[0.17, 0.06, 0.17]).unwrap(),
        interp_nd.interpolate(&[0.17, 0.06, 0.17]).unwrap()
    );
    assert_eq!(
        interp_3d.interpolate(&[0.19, 0.08, 0.19]).unwrap(),
        interp_nd.interpolate(&[0.19, 0.08, 0.19]).unwrap()
    );
    // above x, below y, above z
    assert_eq!(
        interp_3d.interpolate(&[0.17, 0.06, 0.63]).unwrap(),
        interp_nd.interpolate(&[0.17, 0.06, 0.63]).unwrap()
    );
    assert_eq!(
        interp_3d.interpolate(&[0.19, 0.08, 0.65]).unwrap(),
        interp_nd.interpolate(&[0.19, 0.08, 0.65]).unwrap()
    );
    // above x, above y, below z
    assert_eq!(
        interp_3d.interpolate(&[0.17, 0.33, 0.17]).unwrap(),
        interp_nd.interpolate(&[0.17, 0.33, 0.17]).unwrap()
    );
    assert_eq!(
        interp_3d.interpolate(&[0.19, 0.36, 0.19]).unwrap(),
        interp_nd.interpolate(&[0.19, 0.36, 0.19]).unwrap()
    );
    // above x, above y, above z
    assert_eq!(
        interp_3d.interpolate(&[0.17, 0.33, 0.63]).unwrap(),
        interp_nd.interpolate(&[0.17, 0.33, 0.63]).unwrap()
    );
    assert_eq!(
        interp_3d.interpolate(&[0.19, 0.36, 0.65]).unwrap(),
        interp_nd.interpolate(&[0.19, 0.36, 0.65]).unwrap()
    );
}

#[test]
fn test_nearest() {
    let interp = InterpND::new(
        vec![array![0., 1.], array![0., 1.], array![0., 1.]],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
        strategy::Nearest,
        Extrapolate::Error,
    )
    .unwrap();
    // Check that interpolating at grid points just retrieves the value
    let x = &interp.data.grid[0];
    let y = &interp.data.grid[1];
    let z = &interp.data.grid[2];
    for i in 0..x.len() {
        for j in 0..y.len() {
            for k in 0..z.len() {
                assert_eq!(
                    interp.interpolate(&[x[i], y[j], z[k]]).unwrap(),
                    interp.data.values[[i, j, k]]
                );
            }
        }
    }
    assert_eq!(interp.interpolate(&[0.25, 0.25, 0.25]).unwrap(), 0.);
    assert_eq!(interp.interpolate(&[0.25, 0.75, 0.25]).unwrap(), 2.);
    assert_eq!(interp.interpolate(&[0.75, 0.25, 0.75]).unwrap(), 5.);
    assert_eq!(interp.interpolate(&[0.75, 0.75, 0.75]).unwrap(), 7.);
}

#[test]
fn test_integer_nearest_with_clamp() {
    let interp = InterpND::new(
        vec![array![0, 10], array![0, 10]],
        array![[0, 1], [2, 3]].into_dyn(),
        strategy::Nearest,
        Extrapolate::Clamp,
    )
    .unwrap();

    // In-bounds nearest still works on integer coordinates.
    assert_eq!(interp.interpolate(&[8, 3]).unwrap(), 2);
    // Out-of-bounds point is clamped to [0, 10], selecting the top-left row/right column.
    assert_eq!(interp.interpolate(&[-3, 12]).unwrap(), 1);
}

#[test]
fn test_step() {
    // Uniform Lower (floor), same grid as test_nearest
    let interp = InterpND::new(
        vec![array![0., 1.], array![0., 1.], array![0., 1.]],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
        strategy::Step::from(strategy::StepDirection::Lower),
        Extrapolate::Error,
    )
    .unwrap();
    // At grid points: exact value
    let x = &interp.data.grid[0];
    let y = &interp.data.grid[1];
    let z = &interp.data.grid[2];
    for i in 0..x.len() {
        for j in 0..y.len() {
            for k in 0..z.len() {
                assert_eq!(
                    interp.interpolate(&[x[i], y[j], z[k]]).unwrap(),
                    interp.data.values[[i, j, k]]
                );
            }
        }
    }
    // Between points: floor each dimension to index 0
    assert_eq!(interp.interpolate(&[0.3, 0.7, 0.6]).unwrap(), 0.); // floor→[0,0,0]
    assert_eq!(interp.interpolate(&[0.9, 0.9, 0.9]).unwrap(), 0.); // floor→[0,0,0]

    let interp_lower = InterpND::new(
        vec![array![0., 1.], array![0., 1.], array![0., 1.]],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
        strategy::StepLower,
        Extrapolate::Error,
    )
    .unwrap();
    assert_eq!(interp_lower.interpolate(&[0.3, 0.7, 0.6]).unwrap(), 0.);

    // Uniform Upper (ceiling)
    let interp_upper = InterpND::new(
        vec![array![0., 1.], array![0., 1.], array![0., 1.]],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
        strategy::Step::from(strategy::StepDirection::Upper),
        Extrapolate::Error,
    )
    .unwrap();
    assert_eq!(interp_upper.interpolate(&[0.3, 0.7, 0.6]).unwrap(), 7.); // ceil→[1,1,1]

    let interp_marker_upper = InterpND::new(
        vec![array![0., 1.], array![0., 1.], array![0., 1.]],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
        strategy::StepUpper,
        Extrapolate::Error,
    )
    .unwrap();
    assert_eq!(
        interp_marker_upper.interpolate(&[0.3, 0.7, 0.6]).unwrap(),
        7.
    );

    // Per-dimension: Lower in x, Upper in y, Lower in z
    let interp_mixed = InterpND::new(
        vec![array![0., 1.], array![0., 1.], array![0., 1.]],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
        strategy::Step(vec![
            strategy::StepDirection::Lower,
            strategy::StepDirection::Upper,
            strategy::StepDirection::Lower,
        ]),
        Extrapolate::Error,
    )
    .unwrap();
    assert_eq!(interp_mixed.interpolate(&[0.6, 0.4, 0.8]).unwrap(), 2.); // floor x→0, ceil y→1, floor z→0 → [0,1,0]

    // Invalid: direction count mismatch
    assert!(InterpND::new(
        vec![array![0., 1.], array![0., 1.], array![0., 1.]],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
        strategy::Step(vec![
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
        InterpND::new(
            vec![array![0., 1.], array![0., 1.], array![0., 1.]],
            array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
            strategy::Nearest,
            Extrapolate::Enable,
        )
        .unwrap_err(),
        ValidateError::ExtrapolateUnsupported
    ));
    // Extrapolate::Error
    let interp = InterpND::new(
        vec![array![0., 1.], array![0., 1.], array![0., 1.]],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
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
    let interp = InterpND::new(
        vec![array![0.1, 1.1], array![0.2, 1.2], array![0.3, 1.3]],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
        strategy::Linear,
        Extrapolate::Fill(f64::NAN),
    )
    .unwrap();
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
    let interp = InterpND::new(
        vec![array![0.1, 1.1], array![0.2, 1.2], array![0.3, 1.3]],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
        strategy::Linear,
        Extrapolate::Clamp,
    )
    .unwrap();
    assert_eq!(
        interp.interpolate(&[-1., -1., -1.]).unwrap(),
        interp.data.values[[0, 0, 0]]
    );
    assert_eq!(
        interp.interpolate(&[-1., 2., -1.]).unwrap(),
        interp.data.values[[0, 1, 0]]
    );
    assert_eq!(
        interp.interpolate(&[2., -1., 2.]).unwrap(),
        interp.data.values[[1, 0, 1]]
    );
    assert_eq!(
        interp.interpolate(&[2., 2., 2.]).unwrap(),
        interp.data.values[[1, 1, 1]]
    );
}

#[test]
fn test_extrapolate_wrap() {
    let interp = InterpND::new(
        vec![array![0., 1.], array![0., 1.], array![0., 1.]],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
        strategy::Linear,
        Extrapolate::Wrap,
    )
    .unwrap();
    assert_eq!(
        interp.interpolate(&[-0.25, -0.2, -0.4]).unwrap(),
        interp.interpolate(&[0.75, 0.8, 0.6]).unwrap(),
    );
    assert_eq!(
        interp.interpolate(&[-0.25, 2.1, -0.4]).unwrap(),
        interp.interpolate(&[0.75, 0.1, 0.6]).unwrap(),
    );
    assert_eq!(
        interp.interpolate(&[-0.25, 2.1, 2.3]).unwrap(),
        interp.interpolate(&[0.75, 0.1, 0.3]).unwrap(),
    );
    assert_eq!(
        interp.interpolate(&[2.5, 2.1, 2.3]).unwrap(),
        interp.interpolate(&[0.5, 0.1, 0.3]).unwrap(),
    );
}

#[test]
fn test_interpolate_fast() {
    let interp = InterpND::new(
        vec![array![0., 1.], array![0., 1.], array![0., 1.]],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    assert_eq!(
        interp.interpolate_fast(&[0.25, 0.65, 0.9]),
        interp.interpolate(&[0.25, 0.65, 0.9]).unwrap()
    );
}

#[test]
#[should_panic(expected = "interpolate_fast: point length mismatch")]
fn test_interpolate_fast_point_too_short() {
    let interp = InterpND::new(
        vec![array![0., 1.], array![0., 1.], array![0., 1.]],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    interp.interpolate_fast(&[0.5, 0.5]);
}

#[test]
#[should_panic(expected = "interpolate_fast: point length mismatch")]
fn test_interpolate_fast_point_too_long() {
    let interp = InterpND::new(
        vec![array![0., 1.], array![0., 1.], array![0., 1.]],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    interp.interpolate_fast(&[0.5, 0.5, 0.5, 0.5]);
}

#[test]
fn test_batch_interpolate_matches_interpolate() {
    let interp = InterpND::new(
        vec![array![0., 1.], array![0., 1.], array![0., 1.]],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
        strategy::Linear,
        Extrapolate::Enable,
    )
    .unwrap();
    let points: [&[f64]; 3] = [&[0.25, 0.65, 0.9], &[0., 0., 0.], &[2., 2., 2.]];
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
    let interp = InterpND::new(
        vec![array![0.1, 1.1], array![0.2, 1.2], array![0.3, 1.3]],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
        strategy::Linear,
        Extrapolate::Clamp,
    )
    .unwrap();
    let points: [&[f64]; 2] = [&[-1., -1., -1.], &[2., 2., 2.]];
    assert_eq!(
        interp.batch_interpolate(&points).unwrap(),
        vec![interp.data.values[[0, 0, 0]], interp.data.values[[1, 1, 1]]]
    );
}

#[test]
fn test_batch_interpolate_error_aggregates_all_points() {
    let interp = InterpND::new(
        vec![array![0.1, 1.1], array![0.2, 1.2], array![0.3, 1.3]],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    let points: [&[f64]; 3] = [&[0.4, 0.4, 0.4], &[-1., -1., -1.], &[2., 2., 2.]];
    let err = interp.batch_interpolate(&points).unwrap_err();
    let InterpolateError::OutOfBounds(failures) = err else {
        panic!("expected InterpolateError::OutOfBounds");
    };
    let offending: Vec<usize> = failures.iter().map(|at| at.index).collect();
    assert!(offending.contains(&1));
    assert!(offending.contains(&2));
    assert!(!offending.contains(&0));
}

#[test]
fn test_batch_interpolate_point_length_mismatch() {
    let interp = InterpND::new(
        vec![array![0., 1.], array![0., 1.], array![0., 1.]],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    let points: [&[f64]; 2] = [&[0.25, 0.65, 0.9], &[0.5, 0.5]];
    assert!(matches!(
        interp.batch_interpolate(&points).unwrap_err(),
        InterpolateError::PointLength { expected: 3, .. }
    ));
}

#[test]
#[should_panic(expected = "batch_interpolate_fast: point length mismatch")]
fn test_batch_interpolate_fast_point_length_mismatch() {
    let interp = InterpND::new(
        vec![array![0., 1.], array![0., 1.], array![0., 1.]],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap();
    let points: [&[f64]; 2] = [&[0.25, 0.65, 0.9], &[0.5, 0.5]];
    interp.batch_interpolate_fast(&points);
}

#[test]
fn test_mismatched_grid() {
    assert!(matches!(
        InterpND::new(
            // 3-D grid
            vec![array![0., 1.], array![0., 1.], array![0., 1.]],
            // 2-D values
            array![[0., 1.], [2., 3.]].into_dyn(),
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap_err(),
        ValidateError::GridAxisCount {
            expected: 2,
            found: 3
        }
    ));
    assert!(InterpND::new(
        vec![array![]],
        array![0.].into_dyn(),
        strategy::Linear,
        Extrapolate::Error,
    )
    .is_ok(),);
    assert!(matches!(
        InterpND::new(
            // non-empty grid
            vec![array![1.]],
            // 0-D values
            array![0.].into_dyn(),
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap_err(),
        // A single value reads as 0-D, so a non-empty grid has one axis too many
        ValidateError::GridAxisCount {
            expected: 0,
            found: 1
        }
    ));
}

#[test]
fn test_partialeq() {
    #[derive(PartialEq)]
    #[allow(unused)]
    struct MyStruct(InterpDataND<f64>);

    #[derive(PartialEq)]
    #[allow(unused)]
    struct MyStruct2(InterpND<f64, strategy::Linear>);
}

#[test]
#[cfg(feature = "serde")]
fn test_serde() {
    let interp = InterpND::new(
        vec![array![0.1, 1.1], array![0.2, 1.2], array![0.3, 1.3]],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]],].into_dyn(),
        strategy::Nearest,
        Extrapolate::Error,
    )
    .unwrap();

    let ser = serde_json::to_string(&interp).unwrap();
    let de: InterpND<f64, strategy::Nearest> = serde_json::from_str(&ser).unwrap();
    assert_eq!(interp, de);

    // `ndarray` format by default
    let data_ser = serde_json::to_string(&interp.data).unwrap();
    assert_eq!(
        data_ser,
        "{\"grid\":[{\"v\":1,\"dim\":[2],\"data\":[0.1,1.1]},{\"v\":1,\"dim\":[2],\"data\":[0.2,1.2]},{\"v\":1,\"dim\":[2],\"data\":[0.3,1.3]}],\"values\":{\"v\":1,\"dim\":[2,2,2],\"data\":[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0]}}"
    );
    // nested-array format on request
    let data_ser_nested = serde_json::to_string(&crate::prelude::Nested(&interp.data)).unwrap();
    assert_eq!(
        data_ser_nested,
        "{\"grid\":[[0.1,1.1],[0.2,1.2],[0.3,1.3]],\"values\":[[[0.0,1.0],[2.0,3.0]],[[4.0,5.0],[6.0,7.0]]]}"
    );
    // ...and the whole interpolator nests too
    let interp_ser_nested = serde_json::to_string(&crate::prelude::Nested(&interp)).unwrap();
    let de_nested: InterpND<f64, strategy::Nearest> =
        serde_json::from_str(&interp_ser_nested).unwrap();
    assert_eq!(interp, de_nested);

    // simple format (new serialization output)
    let ser0 = "{\"grid\":[[0.1,1.1],[0.2,1.2],[0.3,1.3]],\"values\":[[[0.0,1.0],[2.0,3.0]],[[4.0,5.0],[6.0,7.0]]]}";
    let de0: InterpDataND<_> = serde_json::from_str(ser0).unwrap();
    assert_eq!(interp.data, de0);
    // mixed format (simple grid)
    let ser1 = "{\"grid\":[[0.1,1.1],[0.2,1.2],[0.3,1.3]],\"values\":{\"v\":1,\"dim\":[2,2,2],\"data\":[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0]}}";
    let de1: InterpDataND<_> = serde_json::from_str(ser1).unwrap();
    assert_eq!(interp.data, de1);
    // mixed format (simple values)
    let ser2 = "{\"grid\":[{\"v\":1,\"dim\":[2],\"data\":[0.1,1.1]},{\"v\":1,\"dim\":[2],\"data\":[0.2,1.2]},{\"v\":1,\"dim\":[2],\"data\":[0.3,1.3]}],\"values\":[[[0.0,1.0],[2.0,3.0]],[[4.0,5.0],[6.0,7.0]]]}";
    let de2: InterpDataND<_> = serde_json::from_str(ser2).unwrap();
    assert_eq!(interp.data, de2);
    // complex format (legacy serialization output)
    let ser3 = "{\"grid\":[{\"v\":1,\"dim\":[2],\"data\":[0.1,1.1]},{\"v\":1,\"dim\":[2],\"data\":[0.2,1.2]},{\"v\":1,\"dim\":[2],\"data\":[0.3,1.3]}],\"values\":{\"v\":1,\"dim\":[2,2,2],\"data\":[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0]}}";
    let de3: InterpDataND<_> = serde_json::from_str(ser3).unwrap();
    assert_eq!(interp.data, de3);
}

#[test]
fn test_dyn_interpolator_heterogeneous_storage() {
    // The motivating use case for `AnyInterpolator`: interpolators of differing
    // dimensionality and strategy, stored behind one `Vec<Box<dyn AnyInterpolator<T>>>`.
    let interp1d: Box<dyn AnyInterpolator<f64>> = Box::new(
        Interp1D::new(
            array![0., 1., 2.],
            array![0.0, 0.4, 0.8],
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap(),
    );
    let interp2d: Box<dyn AnyInterpolator<f64>> = Box::new(
        Interp2D::new(
            array![0., 1.],
            array![0., 1.],
            array![[0., 1.], [2., 3.]],
            strategy::Nearest,
            Extrapolate::Error,
        )
        .unwrap(),
    );
    let interpnd: Box<dyn AnyInterpolator<f64>> = Box::new(
        InterpND::new(
            vec![array![0., 1.]],
            array![0.0, 0.4].into_dyn(),
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap(),
    );

    let interps: Vec<Box<dyn AnyInterpolator<f64>>> = vec![interp1d, interp2d, interpnd];
    let points: [&[f64]; 3] = [&[1.5], &[0.6, 0.6], &[0.5]];
    let results: Vec<f64> = interps
        .iter()
        .zip(&points)
        .map(|(interp, point)| interp.interpolate(point).unwrap())
        .collect();
    assert_approx_eq!(results[0], 0.6);
    assert_approx_eq!(results[1], 3.);
    assert_approx_eq!(results[2], 0.2);

    // Downcast back to the concrete type for the first entry.
    assert!(interps[0]
        .as_any()
        .downcast_ref::<Interp1D<f64, strategy::Linear>>()
        .is_some());
    assert!(interps[0]
        .as_any()
        .downcast_ref::<Interp2D<f64, strategy::Nearest>>()
        .is_none());
}
