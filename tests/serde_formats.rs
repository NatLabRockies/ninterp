//! Round-trip coverage for both array formats, across every serializable type.
//!
//! The `Nested` impls are written by hand, so they can drift out of sync with the derived
//! `Deserialize` (a renamed field, a field added to one but not the other). Round-tripping every
//! type through both formats is what catches that.
#![cfg(feature = "serde")]

use ndarray::prelude::*;
use ninterp::data::{InterpData3DOwned, InterpDataND, InterpDataNDOwned};
use ninterp::prelude::*;
use ninterp::Nested;

/// Assert that a value survives a round trip in both the `ndarray` and nested-array formats,
/// and that the two formats really are different on the wire.
fn both_formats<T>(value: &T)
where
    T: serde::Serialize + serde::de::DeserializeOwned + ninterp::SerializeNested + PartialEq,
{
    let plain = serde_json::to_string(value).unwrap();
    let nested = serde_json::to_string(&Nested(value)).unwrap();

    let from_plain: T = serde_json::from_str(&plain).unwrap();
    assert!(
        from_plain == *value,
        "`ndarray` format failed to round-trip"
    );

    let from_nested: T = serde_json::from_str(&nested).unwrap();
    assert!(
        from_nested == *value,
        "nested format failed to round-trip: {nested}"
    );

    // Each format must also read the other's output, since the reader is format-agnostic.
    let cross: T = serde_json::from_str(&plain).unwrap();
    assert!(cross == *value);
}

fn interp_1d() -> Interp1DOwned<f64, strategy::Linear> {
    Interp1D::new(
        array![0., 1., 2., 3.],
        array![0.2, 0.4, 0.6, 0.8],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap()
}

fn interp_2d() -> Interp2DOwned<f64, strategy::Linear> {
    Interp2D::new(
        array![0., 1.],
        array![0., 1.],
        array![[0., 1.], [2., 3.]],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap()
}

fn interp_3d() -> Interp3DOwned<f64, strategy::Linear> {
    Interp3D::new(
        array![0., 1.],
        array![0., 1.],
        array![0., 1.],
        array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap()
}

fn interp_nd() -> InterpNDOwned<f64, strategy::Linear> {
    InterpND::new(
        vec![array![0., 1.], array![0., 1.]],
        array![[0., 1.], [2., 3.]].into_dyn(),
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap()
}

#[test]
fn round_trip_interpolators() {
    both_formats(&Interp0D::new(0.5));
    both_formats(&interp_1d());
    both_formats(&interp_2d());
    both_formats(&interp_3d());
    both_formats(&interp_nd());
}

#[test]
fn round_trip_data() {
    both_formats(&interp_1d().data);
    both_formats(&interp_2d().data);
    both_formats(&interp_3d().data);
    both_formats(&interp_nd().data);
}

/// Untagged enums fail confusingly when a variant's field names shift, so cover every variant.
#[test]
fn round_trip_interpolator_enum() {
    let variants: Vec<InterpolatorEnumOwned<f64>> = vec![
        InterpolatorEnum::new_0d(0.5),
        InterpolatorEnum::new_1d(
            array![0., 1., 2., 3.],
            array![0.2, 0.4, 0.6, 0.8],
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap(),
        InterpolatorEnum::new_2d(
            array![0., 1.],
            array![0., 1.],
            array![[0., 1.], [2., 3.]],
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap(),
        InterpolatorEnum::new_3d(
            array![0., 1.],
            array![0., 1.],
            array![0., 1.],
            array![[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]],
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap(),
        // 4-D: `InterpolatorEnum` is `#[serde(untagged)]`, so an N-D interpolator whose
        // dimensionality also fits a concrete variant deserializes as that concrete variant.
        // Use a dimensionality no concrete variant can claim.
        InterpolatorEnum::new_nd(
            vec![array![0., 1.]; 4],
            Array4::from_shape_fn((2, 2, 2, 2), |(i, j, k, l)| (i + j + k + l) as f64).into_dyn(),
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap(),
    ];
    for variant in &variants {
        both_formats(variant);
    }
}

/// `Nested` must reach through containers, not just bare values.
#[test]
fn round_trip_through_containers() {
    let curves = vec![interp_1d(), interp_1d()];
    let nested = serde_json::to_string(&Nested(&curves)).unwrap();
    assert!(nested.starts_with("[{\"data\":{\"grid\":[[0.0,1.0,2.0,3.0]]"));

    let de: Vec<Interp1DOwned<f64, strategy::Linear>> = serde_json::from_str(&nested).unwrap();
    assert_eq!(de, curves);

    let some = Some(interp_1d());
    let de: Option<Interp1DOwned<f64, strategy::Linear>> =
        serde_json::from_str(&serde_json::to_string(&Nested(&some)).unwrap()).unwrap();
    assert_eq!(de, some);
}

/// `serialize_with` on a field of someone else's struct.
#[test]
fn serialize_with_on_a_field() {
    #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
    struct Config {
        name: String,
        #[serde(serialize_with = "ninterp::serialize_nested")]
        curve: Interp1DOwned<f64, strategy::Linear>,
    }

    let config = Config {
        name: "efficiency".into(),
        curve: interp_1d(),
    };
    let ser = serde_json::to_string(&config).unwrap();
    assert!(
        ser.contains("\"grid\":[[0.0,1.0,2.0,3.0]]"),
        "field was not nested: {ser}"
    );
    assert_eq!(serde_json::from_str::<Config>(&ser).unwrap(), config);
}

/// Non-self-describing formats cannot support `deserialize_any`, so the reader must fall back to
/// the `ndarray` format for them rather than failing outright.
#[test]
fn binary_formats_round_trip() {
    let data = interp_3d().data;
    let bytes = bincode::serialize(&data).unwrap();
    assert_eq!(
        bincode::deserialize::<InterpData3DOwned<f64>>(&bytes).unwrap(),
        data
    );

    let nd = interp_nd().data;
    let bytes = bincode::serialize(&nd).unwrap();
    assert_eq!(
        bincode::deserialize::<InterpDataNDOwned<f64>>(&bytes).unwrap(),
        nd
    );

    let interp = interp_1d();
    let bytes = bincode::serialize(&interp).unwrap();
    assert_eq!(
        bincode::deserialize::<Interp1DOwned<f64, strategy::Linear>>(&bytes).unwrap(),
        interp
    );

    // `Nested` degrades to the `ndarray` format here, so it must still round-trip.
    let bytes = bincode::serialize(&Nested(&interp)).unwrap();
    assert_eq!(
        bincode::deserialize::<Interp1DOwned<f64, strategy::Linear>>(&bytes).unwrap(),
        interp
    );
}

/// `InterpDataND` permits zero dimensions, which has no nested representation, so the values
/// array must fall back to the `ndarray` format for the value to still round-trip.
#[test]
fn zero_dimensional_nd_data() {
    let data = InterpDataND::new(vec![], Array0::from_elem((), 0.5).into_dyn()).unwrap();

    let plain = serde_json::to_string(&data).unwrap();
    assert_eq!(
        serde_json::from_str::<InterpDataNDOwned<f64>>(&plain).unwrap(),
        data
    );

    let nested = serde_json::to_string(&Nested(&data)).unwrap();
    assert_eq!(
        serde_json::from_str::<InterpDataNDOwned<f64>>(&nested).unwrap(),
        data,
        "0-D values failed to round-trip in nested format: {nested}"
    );
}
