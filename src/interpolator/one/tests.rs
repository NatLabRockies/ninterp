use super::*;

#[test]
fn test_cubic_spline() {
    // Linear data: any spline reproduces it exactly (all second derivatives = 0)
    let interp = Interp1D::new(
        array![0., 1., 2., 3.],
        array![1., 3., 5., 7.], // f(x) = 2x + 1
        strategy::CubicSpline::new(),
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
fn test_cubic_spline_knot_exactness() {
    // Values at all knots must be reproduced exactly regardless of data shape
    let interp = Interp1D::new(
        array![0., 1., 2., 3., 4.],
        array![0., 1., 4., 9., 16.], // f(x) = x^2
        strategy::CubicSpline::new(),
        Extrapolate::Error,
    )
    .unwrap();
    let x = interp.data.grid[0].clone();
    for (i, xi) in x.iter().enumerate() {
        assert_approx_eq!(interp.interpolate(&[*xi]).unwrap(), interp.data.values[i]);
    }
}

#[test]
fn test_cubic_spline_two_points() {
    // Degenerate case: 2 points → degenerates to linear interpolation
    let interp = Interp1D::new(
        array![0., 1.],
        array![0., 2.],
        strategy::CubicSpline::new(),
        Extrapolate::Enable,
    )
    .unwrap();
    assert_approx_eq!(interp.interpolate(&[0.5]).unwrap(), 1.0);
    assert_approx_eq!(interp.interpolate(&[2.0]).unwrap(), 4.0); // extrapolation
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
    assert!(matches!(
        interp.interpolate(&[]).unwrap_err(),
        InterpolateError::PointLength(_)
    ));
    assert_eq!(interp.interpolate(&[1.0]).unwrap(), 0.4);
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
        ValidateError::ExtrapolateSelection(_)
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

    let data_ser = serde_json::to_string(&interp.data).unwrap();
    #[cfg(feature = "serde_ndim")]
    assert_eq!(
        data_ser,
        "{\"grid\":[[0.0,1.0,2.0,3.0,4.0]],\"values\":[0.2,0.4,0.6,0.8,1.0]}"
    );
    #[cfg(not(feature = "serde_ndim"))]
    assert_eq!(
        data_ser,
        "{\"grid\":[{\"v\":1,\"dim\":[5],\"data\":[0.0,1.0,2.0,3.0,4.0]}],\"values\":{\"v\":1,\"dim\":[5],\"data\":[0.2,0.4,0.6,0.8,1.0]}}"
    );

    // simple format (new serialization output)
    let ser0 = "{\"grid\":[[0.0,1.0,2.0,3.0,4.0]],\"values\":[0.2,0.4,0.6,0.8,1.0]}";
    let de0: InterpData1D<_> = serde_json::from_str(&ser0).unwrap();
    assert_eq!(interp.data, de0);
    // mixed format (simple grid)
    let ser1 = "{\"grid\":[[0.0,1.0,2.0,3.0,4.0]],\"values\":{\"v\":1,\"dim\":[5],\"data\":[0.2,0.4,0.6,0.8,1.0]}}";
    let de1: InterpData1D<_> = serde_json::from_str(&ser1).unwrap();
    assert_eq!(interp.data, de1);
    // mixed format (simple values)
    let ser2 = "{\"grid\":[{\"v\":1,\"dim\":[5],\"data\":[0.0,1.0,2.0,3.0,4.0]}],\"values\":[0.2,0.4,0.6,0.8,1.0]}";
    let de2: InterpData1D<_> = serde_json::from_str(&ser2).unwrap();
    assert_eq!(interp.data, de2);
    // complex format (legacy serialization output)
    let ser3 = "{\"grid\":[{\"v\":1,\"dim\":[5],\"data\":[0.0,1.0,2.0,3.0,4.0]}],\"values\":{\"v\":1,\"dim\":[5],\"data\":[0.2,0.4,0.6,0.8,1.0]}}";
    let de3: InterpData1D<_> = serde_json::from_str(&ser3).unwrap();
    assert_eq!(interp.data, de3);
}
