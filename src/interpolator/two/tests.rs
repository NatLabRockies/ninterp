use super::*;

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
        ValidateError::ExtrapolateSelection(_)
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
    let de0: InterpData2D<_> = serde_json::from_str(&ser0).unwrap();
    assert_eq!(interp.data, de0);
    // mixed format (simple grid)
    let ser1 = "{\"grid\":[[0.05,0.1,0.15],[0.1,0.2,0.3]],\"values\":{\"v\":1,\"dim\":[3,3],\"data\":[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]}}";
    let de1: InterpData2D<_> = serde_json::from_str(&ser1).unwrap();
    assert_eq!(interp.data, de1);
    // mixed format (simple values)
    let ser2 = "{\"grid\":[{\"v\":1,\"dim\":[3],\"data\":[0.05,0.1,0.15]},{\"v\":1,\"dim\":[3],\"data\":[0.1,0.2,0.3]}],\"values\":[[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]]}";
    let de2: InterpData2D<_> = serde_json::from_str(&ser2).unwrap();
    assert_eq!(interp.data, de2);
    // complex format (legacy serialization output)
    let ser3 = "{\"grid\":[{\"v\":1,\"dim\":[3],\"data\":[0.05,0.1,0.15]},{\"v\":1,\"dim\":[3],\"data\":[0.1,0.2,0.3]}],\"values\":{\"v\":1,\"dim\":[3,3],\"data\":[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]}}";
    let de3: InterpData2D<_> = serde_json::from_str(&ser3).unwrap();
    assert_eq!(interp.data, de3);
}
