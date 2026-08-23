use super::*;

strategy_enum_impl!(
    Strategy3DEnum,
    Strategy3D,
    InterpData3DBase,
    &[D::Elem; 3],
    &[[D::Elem; 3]],
    [
        (Nearest, strategy::Nearest),
        (Step, strategy::Step),
        (Linear, strategy::Linear),
        (LinearUniform, strategy::LinearUniform),
        (CubicC1, strategy::CubicC1<T>),
        (CubicC2, strategy::CubicC2<T>),
        (GridTransform, strategy::GridTransform<T, Box<Strategy3DEnum<T>>>),
        (ValuesTransform, strategy::ValuesTransform<T, Box<Strategy3DEnum<T>>>),
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
            serde_json::to_string(&Strategy3DEnum::<f64>::from(Linear)).unwrap(),
            serde_json::to_string(&Linear).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&Strategy3DEnum::<f64>::from(Nearest)).unwrap(),
            serde_json::to_string(&Nearest).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&Strategy3DEnum::<f64>::from(Step::from(
                strategy::step::StepDirection::Lower
            )))
            .unwrap(),
            serde_json::to_string(&Step::from(strategy::step::StepDirection::Lower)).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&Strategy3DEnum::from(CubicC2::<f64>::not_a_knot())).unwrap(),
            serde_json::to_string(&CubicC2::<f64>::not_a_knot()).unwrap(),
        );
    }
}
