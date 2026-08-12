use super::*;

strategy_enum_impl!(
    Strategy3DEnum,
    Strategy3D,
    InterpData3DBase,
    &[D::Elem; 3],
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
            serde_json::to_string(&Strategy3DEnum::<f64>::from(Linear)).unwrap(),
            serde_json::to_string(&Linear).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&Strategy3DEnum::<f64>::from(Nearest)).unwrap(),
            serde_json::to_string(&Nearest).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&Strategy3DEnum::<f64>::from(StepLower)).unwrap(),
            serde_json::to_string(&StepLower).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&Strategy3DEnum::<f64>::from(StepUpper)).unwrap(),
            serde_json::to_string(&StepUpper).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&Strategy3DEnum::from(CubicC2::<f64>::not_a_knot())).unwrap(),
            serde_json::to_string(&CubicC2::<f64>::not_a_knot()).unwrap(),
        );
    }
}
