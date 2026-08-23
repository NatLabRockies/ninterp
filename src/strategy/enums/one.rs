use super::*;

strategy_enum_impl!(
    Strategy1DEnum,
    Strategy1D,
    InterpData1DBase,
    &[D::Elem; 1],
    &[[D::Elem; 1]],
    [
        (Nearest, strategy::Nearest),
        (Step, strategy::Step),
        (Linear, strategy::Linear),
        (LinearUniform, strategy::LinearUniform),
        (CubicC1, strategy::CubicC1<T>),
        (CubicC2, strategy::CubicC2<T>),
        (GridTransform, strategy::GridTransform<T, Box<Strategy1DEnum<T>>>),
        (ValuesTransform, strategy::ValuesTransform<T, Box<Strategy1DEnum<T>>>),
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
            serde_json::to_string(&Strategy1DEnum::<f64>::from(Step::from(
                strategy::step::StepDirection::Lower
            )))
            .unwrap(),
            serde_json::to_string(&Step::from(strategy::step::StepDirection::Lower)).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&Strategy1DEnum::from(CubicC2::<f64>::not_a_knot())).unwrap(),
            serde_json::to_string(&CubicC2::<f64>::not_a_knot()).unwrap(),
        );

        // Legacy aliases (from the removed `LeftNearest`/`RightNearest` unit structs)
        // deserialize through `Step`'s broadcast form.
        assert_eq!(
            serde_json::from_str::<Strategy1DEnum<f64>>("\"LeftNearest\"").unwrap(),
            Strategy1DEnum::Step(Step::lower())
        );
        assert_eq!(
            serde_json::from_str::<Strategy1DEnum<f64>>("\"RightNearest\"").unwrap(),
            Strategy1DEnum::Step(Step::upper())
        );
    }
}
