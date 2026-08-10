use super::*;

strategy_enum_impl!(
    StrategyNDEnum,
    StrategyND,
    InterpDataND,
    &[D::Elem],
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
            serde_json::to_string(&StrategyNDEnum::from(Linear)).unwrap(),
            serde_json::to_string(&Linear).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&StrategyNDEnum::from(Nearest)).unwrap(),
            serde_json::to_string(&Nearest).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&StrategyNDEnum::from(StepLower)).unwrap(),
            serde_json::to_string(&StepLower).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&StrategyNDEnum::from(StepUpper)).unwrap(),
            serde_json::to_string(&StepUpper).unwrap(),
        );
    }
}
