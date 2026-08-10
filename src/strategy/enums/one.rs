use super::*;

strategy_enum_impl!(
    Strategy1DEnum,
    Strategy1D,
    InterpData1DBase,
    &[D::Elem; 1],
    [
        (Linear, strategy::Linear),
        (LinearUniform, strategy::LinearUniform),
        (Nearest, strategy::Nearest),
        (Step, strategy::Step),
        (StepLower, strategy::StepLower),
        (StepUpper, strategy::StepUpper),
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
            serde_json::to_string(&Strategy1DEnum::from(Linear)).unwrap(),
            serde_json::to_string(&Linear).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&Strategy1DEnum::from(Nearest)).unwrap(),
            serde_json::to_string(&Nearest).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&Strategy1DEnum::from(StepLower)).unwrap(),
            serde_json::to_string(&StepLower).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&Strategy1DEnum::from(StepUpper)).unwrap(),
            serde_json::to_string(&StepUpper).unwrap(),
        );

        // Legacy aliases deserialize through StepLower/StepUpper only.
        assert!(matches!(
            serde_json::from_str::<Strategy1DEnum>("\"LeftNearest\"").unwrap(),
            Strategy1DEnum::StepLower(_)
        ));
        assert!(matches!(
            serde_json::from_str::<Strategy1DEnum>("\"RightNearest\"").unwrap(),
            Strategy1DEnum::StepUpper(_)
        ));
    }
}
