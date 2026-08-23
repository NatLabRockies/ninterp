//! Serde round-trip coverage for every built-in strategy, both bare and wrapped in each
//! dimensionality's `Strategy*Enum`. The enums are `#[serde(untagged)]`, so a new strategy
//! (or a new `CubicC2` boundary condition) can silently fail to round-trip through the wrong
//! variant if its JSON shape collides with another's; round-tripping every variant here is
//! what would catch that.
#![cfg(feature = "serde")]

use ninterp::strategy::enums::*;
use ninterp::strategy::step::StepDirection;
use ninterp::strategy::*;

fn round_trip<T>(value: &T)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let json = serde_json::to_string(value).unwrap();
    let de: T = serde_json::from_str(&json).unwrap();
    assert_eq!(&de, value, "failed to round-trip: {json}");
}

fn cubic_c2_variants() -> Vec<CubicC2<f64>> {
    vec![
        CubicC2::not_a_knot(),
        CubicC2::natural(),
        CubicC2::clamped(0.5, -0.5),
        CubicC2::periodic(),
    ]
}

fn cubic_c1_variants() -> Vec<CubicC1<f64>> {
    vec![
        CubicC1::default(),
        CubicC1::new().with_cache_mode(CubicC1CacheMode::None),
    ]
}

#[test]
fn bare_strategies_round_trip() {
    round_trip(&Nearest);
    round_trip(&Linear);
    round_trip(&LinearUniform);
    round_trip(&Step::lower());
    round_trip(&Step::upper());
    round_trip(&Step::new(vec![StepDirection::Lower, StepDirection::Upper]));
    for bc in cubic_c2_variants() {
        round_trip(&bc);
    }
    for c1 in cubic_c1_variants() {
        round_trip(&c1);
    }
    round_trip(&GridTransform::<f64, _>::log(Linear));
    round_trip(&ValuesTransform::<f64, _>::log(Linear));
}

/// Generates a round-trip test for `$Enum` over every strategy variant, `CubicC2` and
/// `GridTransform`/`ValuesTransform` (nested one level, since each enum's own variant
/// boxes its `inner`) included.
macro_rules! enum_round_trip_test {
    ($test_name:ident, $Enum:ident) => {
        #[test]
        fn $test_name() {
            round_trip(&$Enum::<f64>::from(Nearest));
            round_trip(&$Enum::<f64>::from(Linear));
            round_trip(&$Enum::<f64>::from(LinearUniform));
            round_trip(&$Enum::<f64>::from(Step::lower()));
            round_trip(&$Enum::<f64>::from(Step::upper()));
            round_trip(&$Enum::<f64>::from(Step::new(vec![
                StepDirection::Lower,
                StepDirection::Upper,
            ])));
            for bc in cubic_c2_variants() {
                round_trip(&$Enum::from(bc));
            }
            for c1 in cubic_c1_variants() {
                round_trip(&$Enum::from(c1));
            }
            let inner: Box<$Enum<f64>> = Box::new($Enum::<f64>::from(Linear));
            round_trip(&$Enum::<f64>::from(GridTransform::log(inner.clone())));
            round_trip(&$Enum::<f64>::from(ValuesTransform::log(inner)));
        }
    };
}

enum_round_trip_test!(strategy_1d_enum_round_trips_every_variant, Strategy1DEnum);
enum_round_trip_test!(strategy_2d_enum_round_trips_every_variant, Strategy2DEnum);
enum_round_trip_test!(strategy_3d_enum_round_trips_every_variant, Strategy3DEnum);
enum_round_trip_test!(strategy_nd_enum_round_trips_every_variant, StrategyNDEnum);
