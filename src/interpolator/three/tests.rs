use super::*;

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
        ValidateError::ExtrapolateSelection(_)
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
        InterpolateError::ExtrapolateError(_)
    ));
    assert!(matches!(
        interp.interpolate(&[2., 2., 2.]).unwrap_err(),
        InterpolateError::ExtrapolateError(_)
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
fn test_partialeq() {
    #[derive(PartialEq)]
    #[allow(unused)]
    struct MyStruct(InterpData3DOwned<f64>);

    #[derive(PartialEq)]
    #[allow(unused)]
    struct MyStruct2(Interp3DOwned<f64, strategy::Linear>);
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
    let de0: InterpData3D<_> = serde_json::from_str(&ser0).unwrap();
    assert_eq!(interp.data, de0);
    // mixed format (simple grid)
    let ser1 = "{\"grid\":[[0.0,1.0],[0.0,1.0,2.0],[0.0,1.0,2.0,3.0]],\"values\":{\"v\":1,\"dim\":[2,3,4],\"data\":[0.6,0.8,1.0,1.2,0.8,1.0,1.2,1.4,1.0,1.2,1.4,1.6,0.8,1.0,1.2,1.4,1.0,1.2,1.4,1.6,1.2,1.4,1.6,1.8]}}";
    let de1: InterpData3D<_> = serde_json::from_str(&ser1).unwrap();
    assert_eq!(interp.data, de1);
    // mixed format (simple values)
    let ser2 = "{\"grid\":[{\"v\":1,\"dim\":[2],\"data\":[0.0,1.0]},{\"v\":1,\"dim\":[3],\"data\":[0.0,1.0,2.0]},{\"v\":1,\"dim\":[4],\"data\":[0.0,1.0,2.0,3.0]}],\"values\":[[[0.6,0.8,1.0,1.2],[0.8,1.0,1.2,1.4],[1.0,1.2,1.4,1.6]],[[0.8,1.0,1.2,1.4],[1.0,1.2,1.4,1.6],[1.2,1.4,1.6,1.8]]]}";
    let de2: InterpData3D<_> = serde_json::from_str(&ser2).unwrap();
    assert_eq!(interp.data, de2);
    // complex format (legacy serialization output)
    let ser3 = "{\"grid\":[{\"v\":1,\"dim\":[2],\"data\":[0.0,1.0]},{\"v\":1,\"dim\":[3],\"data\":[0.0,1.0,2.0]},{\"v\":1,\"dim\":[4],\"data\":[0.0,1.0,2.0,3.0]}],\"values\":{\"v\":1,\"dim\":[2,3,4],\"data\":[0.6,0.8,1.0,1.2,0.8,1.0,1.2,1.4,1.0,1.2,1.4,1.6,0.8,1.0,1.2,1.4,1.0,1.2,1.4,1.6,1.2,1.4,1.6,1.8]}}";
    let de3: InterpData3D<_> = serde_json::from_str(&ser3).unwrap();
    assert_eq!(interp.data, de3);
}
