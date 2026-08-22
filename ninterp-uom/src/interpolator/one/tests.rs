use super::*;
use uom::si::f64::{Power, Ratio};
use uom::si::power::kilowatt;
use uom::si::ratio::ratio;

fn build() -> UomInterp1D<Ratio, Power, f64, strategy::Linear> {
    UomInterp1D::new(
        array![
            Ratio::new::<ratio>(0.),
            Ratio::new::<ratio>(1.),
            Ratio::new::<ratio>(2.),
        ],
        array![
            Power::new::<kilowatt>(0.),
            Power::new::<kilowatt>(1.),
            Power::new::<kilowatt>(2.),
        ],
        strategy::Linear,
        Extrapolate::Error,
    )
    .unwrap()
}

#[test]
fn interpolate_matches_linear() {
    let interp = build();
    assert_eq!(
        interp.interpolate(Ratio::new::<ratio>(0.5)).unwrap(),
        Power::new::<kilowatt>(0.5)
    );
    assert_eq!(
        interp.interpolate_fast(Ratio::new::<ratio>(0.5)),
        Power::new::<kilowatt>(0.5)
    );
}

#[test]
fn view_and_into_owned_round_trip() {
    let owned = build();
    let point = Ratio::new::<ratio>(1.5);
    let viewed = owned.view();
    assert_eq!(viewed.interpolate(point), owned.interpolate(point));
    let back = viewed.into_owned();
    assert_eq!(back.interpolate(point), owned.interpolate(point));
}

#[test]
fn batch_interpolate_matches_loop() {
    let interp = build();
    let points = [
        Ratio::new::<ratio>(0.25),
        Ratio::new::<ratio>(1.5),
        Ratio::new::<ratio>(1.75),
    ];
    let batched = interp.batch_interpolate(&points).unwrap();
    let looped: Vec<Power> = points
        .iter()
        .map(|&p| interp.interpolate(p).unwrap())
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
fn set_strategy_box_changes_result() {
    let mut interp: UomInterp1D<Ratio, Power, f64, Box<dyn Strategy1D<OwnedRepr<f64>>>> =
        UomInterp1D::new(
            array![Ratio::new::<ratio>(0.), Ratio::new::<ratio>(1.)],
            array![Power::new::<kilowatt>(0.), Power::new::<kilowatt>(1.)],
            Box::new(strategy::Linear) as Box<dyn Strategy1D<OwnedRepr<f64>>>,
            Extrapolate::Error,
        )
        .unwrap();
    let point = Ratio::new::<ratio>(0.25);
    assert_eq!(
        interp.interpolate(point).unwrap(),
        Power::new::<kilowatt>(0.25)
    );
    interp.set_strategy(Box::new(strategy::Nearest)).unwrap();
    assert_eq!(
        interp.interpolate(point).unwrap(),
        Power::new::<kilowatt>(0.)
    );
}

#[test]
fn set_strategy_enum_changes_result() {
    let mut interp: UomInterp1D<Ratio, Power, f64, Strategy1DEnum<f64>> = UomInterp1D::new(
        array![Ratio::new::<ratio>(0.), Ratio::new::<ratio>(1.)],
        array![Power::new::<kilowatt>(0.), Power::new::<kilowatt>(1.)],
        strategy::Linear.into(),
        Extrapolate::Error,
    )
    .unwrap();
    let point = Ratio::new::<ratio>(0.25);
    assert_eq!(
        interp.interpolate(point).unwrap(),
        Power::new::<kilowatt>(0.25)
    );
    interp.set_strategy(strategy::Nearest).unwrap();
    assert_eq!(
        interp.interpolate(point).unwrap(),
        Power::new::<kilowatt>(0.)
    );
}

#[test]
fn extrapolate_and_validate() {
    let mut interp = build();
    assert_eq!(interp.ndim(), 1);
    assert!(interp.validate().is_ok());
    assert!(interp.validate_extrapolate(&Extrapolate::Clamp).is_ok());
    assert!(interp.interpolate(Ratio::new::<ratio>(5.)).is_err());
    interp.set_extrapolate(Extrapolate::Clamp).unwrap();
    assert_eq!(
        interp.interpolate(Ratio::new::<ratio>(5.)).unwrap(),
        Power::new::<kilowatt>(2.)
    );
    interp.validate_strategy().unwrap();
    interp.init_strategy().unwrap();
}

#[test]
fn inner_escape_hatch_is_reachable() {
    let mut interp = build();
    // Mutate the raw, unit-erased interpolator directly rather than going through
    // `set_strategy`, then re-run `init_strategy` (mirrors core's own documented usage
    // of its public `strategy` field).
    interp.inner.strategy = strategy::Linear;
    interp.init_strategy().unwrap();
    // Base-unit caveat: the grid was built from dimensionless `Ratio`s, so the stored
    // raw value equals what was typed in here (not always true in general).
    assert_eq!(interp.inner.data.grid[0][0], 0.);
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
    let de: UomInterp1D<Ratio, Power, f64, strategy::Linear> = serde_json::from_str(&json).unwrap();
    assert_eq!(interp, de);
}
