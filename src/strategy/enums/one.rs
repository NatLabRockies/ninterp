use super::*;

strategy_enum_impl!(
    Strategy1DEnum,
    Strategy1D,
    InterpData1DBase,
    &[D::Elem; 1],
    [
        (Nearest, strategy::Nearest),
        (Step, strategy::Step),
        (StepLower, strategy::StepLower),
        (StepUpper, strategy::StepUpper),
        (Linear, strategy::Linear),
        (LinearUniform, strategy::LinearUniform),
        (CubicC2, strategy::CubicC2<T>),
    ]
);

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    #[cfg(feature = "serde")]
    fn test_serde() {
        assert_eq!(
            serde_json::to_string(&Strategy1DEnum::<f64>::from(Linear)).unwrap(),
            serde_json::to_string(&Linear).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&Strategy1DEnum::<f64>::from(Nearest)).unwrap(),
            serde_json::to_string(&Nearest).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&Strategy1DEnum::<f64>::from(StepLower)).unwrap(),
            serde_json::to_string(&StepLower).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&Strategy1DEnum::<f64>::from(StepUpper)).unwrap(),
            serde_json::to_string(&StepUpper).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&Strategy1DEnum::from(CubicC2::<f64>::not_a_knot())).unwrap(),
            serde_json::to_string(&CubicC2::<f64>::not_a_knot()).unwrap(),
        );

        // Legacy aliases deserialize through StepLower/StepUpper only.
        assert!(matches!(
            serde_json::from_str::<Strategy1DEnum<f64>>("\"LeftNearest\"").unwrap(),
            Strategy1DEnum::StepLower(_)
        ));
        assert!(matches!(
            serde_json::from_str::<Strategy1DEnum<f64>>("\"RightNearest\"").unwrap(),
            Strategy1DEnum::StepUpper(_)
        ));
    }
}
