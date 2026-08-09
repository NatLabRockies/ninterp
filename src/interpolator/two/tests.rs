use super::*;

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
        InterpolateError::PointLength(_)
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
    let interp = Interp2D::new(
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

    let interp_lower = Interp2D::new(
        grid_x.view(),
        grid_y.view(),
        values.view(),
        strategy::StepLower,
        Extrapolate::Error,
    )
    .unwrap();
    assert_eq!(interp_lower.interpolate(&[0.7, 1.4]).unwrap(), f[[0, 1]]);

    let interp_upper = Interp2D::new(
        grid_x.view(),
        grid_y.view(),
        values.view(),
        strategy::StepUpper,
        Extrapolate::Error,
    )
    .unwrap();
    assert_eq!(interp_upper.interpolate(&[0.7, 1.4]).unwrap(), f[[1, 2]]);

    // Per-dimension: Lower in x, Upper in y
    let interp_mixed = Interp2D::new(
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
    assert!(Interp2D::new(
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
        ValidateError::InvalidExtrapolate(_)
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
        InterpolateError::ExtrapolateError(_)
    ));
    assert!(matches!(
        interp.interpolate(&[2., 2.]).unwrap_err(),
        InterpolateError::ExtrapolateError(_)
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
        ValidateError::Other(_)
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
fn test_partialeq() {
    #[derive(PartialEq)]
    #[allow(unused)]
    struct MyStruct(InterpData2DOwned<f64>);

    #[derive(PartialEq)]
    #[allow(unused)]
    struct MyStruct2(Interp2DOwned<f64, strategy::Linear>);
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
    let de: Interp2DOwned<f64, strategy::Linear> = serde_json::from_str(&ser).unwrap();
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
