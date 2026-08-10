use super::*;

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
        InterpolateError::PointLength(_)
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
        InterpolateError::PointLength(_)
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

    let boxed: Box<dyn DynInterpolator<f64>> = Box::new(interp.clone());
    assert_eq!(boxed.interpolate_slice(&[1.0]).unwrap(), 0.4);
    assert_eq!(
        boxed.batch_interpolate_slice(&points).unwrap(),
        interp.batch_interpolate(&[[1.0], [2.5]]).unwrap(),
    );
    assert!(matches!(
        boxed.interpolate_slice(&[]).unwrap_err(),
        InterpolateError::PointLength(1)
    ));
    assert_eq!(
        boxed.as_any().downcast_ref::<Interp1DOwned<f64, _>>(),
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
        strategy::Step::from(strategy::StepDirection::Lower),
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
        strategy::Step::from(strategy::StepDirection::Upper),
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
        strategy::Step::from(strategy::StepDirection::Lower),
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
            strategy::StepDirection::Lower,
            strategy::StepDirection::Upper,
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
        ValidateError::InvalidExtrapolate(_)
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
        InterpolateError::ExtrapolateError(_)
    ));
    // Fail to extrapolate above highest grid value
    assert!(matches!(
        interp.interpolate(&[5.]).unwrap_err(),
        InterpolateError::ExtrapolateError(_)
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
    let InterpolateError::ExtrapolateError(message) = err else {
        panic!("expected ExtrapolateError");
    };
    assert!(message.contains("point[1]"));
    assert!(message.contains("point[3]"));
    assert!(!message.contains("point[0]"));
    assert!(!message.contains("point[2]"));
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
    struct MyStruct(InterpData1DOwned<f64>);

    #[derive(PartialEq)]
    #[allow(unused)]
    struct MyStruct2(Interp1DOwned<f64, strategy::Linear>);
}

#[test]
#[cfg(feature = "serde")]
fn test_serde() {
    let interp = Interp1D::new(
        array![0., 1., 2., 3., 4.],
        array![0.2, 0.4, 0.6, 0.8, 1.0],
        strategy::Step::from(strategy::StepDirection::Lower),
        Extrapolate::Error,
    )
    .unwrap();

    let ser = serde_json::to_string(&interp).unwrap();
    let de: Interp1DOwned<f64, strategy::Step> = serde_json::from_str(&ser).unwrap();
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
    let de_nested: Interp1DOwned<f64, strategy::Step> =
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
