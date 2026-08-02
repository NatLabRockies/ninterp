use super::*;

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
        ValidateError::ExtrapolateSelection(_)
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
        InterpolateError::ExtrapolateError(_)
    ));
    assert!(matches!(
        interp.interpolate(&[2., 2., 2.]).unwrap_err(),
        InterpolateError::ExtrapolateError(_)
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
        ValidateError::Other(_)
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
        ValidateError::Other(_)
    ));
}

#[test]
fn test_partialeq() {
    #[derive(PartialEq)]
    #[allow(unused)]
    struct MyStruct(InterpDataNDOwned<f64>);

    #[derive(PartialEq)]
    #[allow(unused)]
    struct MyStruct2(InterpNDOwned<f64, strategy::Linear>);
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
    let de: InterpNDOwned<f64, strategy::Nearest> = serde_json::from_str(&ser).unwrap();
    assert_eq!(interp, de);

    // simple format (new serialization output)
    let ser0 = "{\"grid\":[[0.1,1.1],[0.2,1.2],[0.3,1.3]],\"values\":[[[0.0,1.0],[2.0,3.0]],[[4.0,5.0],[6.0,7.0]]]}";
    let de0: InterpDataND<_> = serde_json::from_str(&ser0).unwrap();
    assert_eq!(interp.data, de0);
    // mixed format (simple grid)
    let ser1 = "{\"grid\":[[0.1,1.1],[0.2,1.2],[0.3,1.3]],\"values\":{\"v\":1,\"dim\":[2,2,2],\"data\":[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0]}}";
    let de1: InterpDataND<_> = serde_json::from_str(&ser1).unwrap();
    assert_eq!(interp.data, de1);
    // mixed format (simple values)
    let ser2 = "{\"grid\":[{\"v\":1,\"dim\":[2],\"data\":[0.1,1.1]},{\"v\":1,\"dim\":[2],\"data\":[0.2,1.2]},{\"v\":1,\"dim\":[2],\"data\":[0.3,1.3]}],\"values\":[[[0.0,1.0],[2.0,3.0]],[[4.0,5.0],[6.0,7.0]]]}";
    let de2: InterpDataND<_> = serde_json::from_str(&ser2).unwrap();
    assert_eq!(interp.data, de2);
    // complex format (legacy serialization output)
    let ser3 = "{\"grid\":[{\"v\":1,\"dim\":[2],\"data\":[0.1,1.1]},{\"v\":1,\"dim\":[2],\"data\":[0.2,1.2]},{\"v\":1,\"dim\":[2],\"data\":[0.3,1.3]}],\"values\":{\"v\":1,\"dim\":[2,2,2],\"data\":[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0]}}";
    let de3: InterpDataND<_> = serde_json::from_str(&ser3).unwrap();
    assert_eq!(interp.data, de3);
}
