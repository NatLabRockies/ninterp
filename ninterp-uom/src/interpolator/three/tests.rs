use super::*;
use uom::si::f64::{Length, Power, Ratio, Time};
use uom::si::length::meter;
use uom::si::power::kilowatt;
use uom::si::ratio::ratio;
use uom::si::time::second;

fn build() -> UomInterp3D<Ratio, Time, Length, Power, f64, strategy::Linear> {
    UomInterp3D::new(
        array![Ratio::new::<ratio>(0.), Ratio::new::<ratio>(1.)],
        array![Time::new::<second>(0.), Time::new::<second>(1.)],
        array![Length::new::<meter>(0.), Length::new::<meter>(1.)],
        array![
            [
                [Power::new::<kilowatt>(0.), Power::new::<kilowatt>(1.)],
                [Power::new::<kilowatt>(1.), Power::new::<kilowatt>(2.)],
            ],
            [
                [Power::new::<kilowatt>(1.), Power::new::<kilowatt>(2.)],
                [Power::new::<kilowatt>(2.), Power::new::<kilowatt>(3.)],
            ],
        ],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap()
}

#[test]
fn interpolate_matches_linear() {
    let interp = build();
    let (x, y, z) = (
        Ratio::new::<ratio>(0.5),
        Time::new::<second>(0.5),
        Length::new::<meter>(0.5),
    );
    assert_eq!(
        interp.interpolate(x, y, z).unwrap(),
        Power::new::<kilowatt>(1.5)
    );
    assert_eq!(
        interp.interpolate_fast(x, y, z),
        Power::new::<kilowatt>(1.5)
    );
}

#[test]
fn view_and_into_owned_round_trip() {
    let owned = build();
    let (x, y, z) = (
        Ratio::new::<ratio>(0.25),
        Time::new::<second>(0.75),
        Length::new::<meter>(0.5),
    );
    let viewed = owned.view();
    assert_eq!(viewed.interpolate(x, y, z), owned.interpolate(x, y, z));
    let back = viewed.into_owned();
    assert_eq!(back.interpolate(x, y, z), owned.interpolate(x, y, z));
}

#[test]
fn batch_interpolate_matches_loop() {
    let interp = build();
    let points = [
        (
            Ratio::new::<ratio>(0.25),
            Time::new::<second>(0.25),
            Length::new::<meter>(0.25),
        ),
        (
            Ratio::new::<ratio>(0.5),
            Time::new::<second>(0.75),
            Length::new::<meter>(1.),
        ),
    ];
    let batched = interp.batch_interpolate(&points).unwrap();
    let looped: Vec<Power> = points
        .iter()
        .map(|&(x, y, z)| interp.interpolate(x, y, z).unwrap())
        .collect();
    assert_eq!(batched, looped);
    assert_eq!(interp.batch_interpolate_fast(&points), looped);

    let mut into = vec![Power::new::<kilowatt>(0.); points.len()];
    interp.batch_interpolate_into(&points, &mut into).unwrap();
    assert_eq!(into, looped);
    interp.batch_interpolate_fast_into(&points, &mut into);
    assert_eq!(into, looped);
}

#[test]
fn set_strategy_enum_changes_result() {
    let mut interp: UomInterp3D<Ratio, Time, Length, Power, f64, Strategy3DEnum<f64>> =
        UomInterp3D::new(
            array![Ratio::new::<ratio>(0.), Ratio::new::<ratio>(1.)],
            array![Time::new::<second>(0.), Time::new::<second>(1.)],
            array![Length::new::<meter>(0.), Length::new::<meter>(1.)],
            array![
                [
                    [Power::new::<kilowatt>(0.), Power::new::<kilowatt>(1.)],
                    [Power::new::<kilowatt>(1.), Power::new::<kilowatt>(2.)],
                ],
                [
                    [Power::new::<kilowatt>(1.), Power::new::<kilowatt>(2.)],
                    [Power::new::<kilowatt>(2.), Power::new::<kilowatt>(3.)],
                ],
            ],
            strategy::Linear.into(),
            Extrapolate::Error,
        )
        .unwrap();
    let (x, y, z) = (
        Ratio::new::<ratio>(0.25),
        Time::new::<second>(0.),
        Length::new::<meter>(0.),
    );
    assert_eq!(
        interp.interpolate(x, y, z).unwrap(),
        Power::new::<kilowatt>(0.25)
    );
    interp.set_strategy(strategy::Nearest).unwrap();
    assert_eq!(
        interp.interpolate(x, y, z).unwrap(),
        Power::new::<kilowatt>(0.)
    );
}

#[test]
fn extrapolate_and_validate() {
    let mut interp = build();
    assert_eq!(interp.ndim(), 3);
    assert!(interp.validate().is_ok());
    let oob = (
        Ratio::new::<ratio>(5.),
        Time::new::<second>(0.),
        Length::new::<meter>(0.),
    );
    assert!(interp.interpolate(oob.0, oob.1, oob.2).is_err());
    interp.set_extrapolate(Extrapolate::Clamp).unwrap();
    // Clamped to (x=1, y=0, z=0): f(1, 0, 0) = 1 kW.
    assert_eq!(
        interp.interpolate(oob.0, oob.1, oob.2).unwrap(),
        Power::new::<kilowatt>(1.)
    );
}

#[test]
fn partial_eq() {
    assert_eq!(build(), build());
}

#[test]
#[cfg(feature = "serde")]
fn serde_round_trip() {
    let interp = build();
    let json = serde_json::to_string(&interp).unwrap();
    let de: UomInterp3D<Ratio, Time, Length, Power, f64, strategy::Linear> =
        serde_json::from_str(&json).unwrap();
    assert_eq!(interp, de);
}
