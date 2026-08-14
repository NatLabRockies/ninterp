use super::*;

strategy_enum_impl!(
    StrategyNDEnum,
    StrategyND,
    InterpDataNDBase,
    &[D::Elem],
    &[&[D::Elem]],
    [
        (Nearest, strategy::Nearest),
        (Step, strategy::Step),
        (Linear, strategy::Linear),
        (LinearUniform, strategy::LinearUniform),
        (CubicC2, strategy::CubicC2<T>),
        (GridTransform, strategy::GridTransform<T, Box<StrategyNDEnum<T>>>),
        (ValuesTransform, strategy::ValuesTransform<T, Box<StrategyNDEnum<T>>>),
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
            serde_json::to_string(&StrategyNDEnum::<f64>::from(Linear)).unwrap(),
            serde_json::to_string(&Linear).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&StrategyNDEnum::<f64>::from(Nearest)).unwrap(),
            serde_json::to_string(&Nearest).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&StrategyNDEnum::<f64>::from(Step::from(
                strategy::step::StepDirection::Lower
            )))
            .unwrap(),
            serde_json::to_string(&Step::from(strategy::step::StepDirection::Lower)).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&StrategyNDEnum::from(CubicC2::<f64>::not_a_knot())).unwrap(),
            serde_json::to_string(&CubicC2::<f64>::not_a_knot()).unwrap(),
        );
    }
}
