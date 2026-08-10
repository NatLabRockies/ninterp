use super::*;

strategy_enum_impl!(
    Strategy3DEnum,
    Strategy3D,
    InterpData3D,
    &[D::Elem; 3],
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
            serde_json::to_string(&Strategy3DEnum::from(Linear)).unwrap(),
            serde_json::to_string(&Linear).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&Strategy3DEnum::from(Nearest)).unwrap(),
            serde_json::to_string(&Nearest).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&Strategy3DEnum::from(StepLower)).unwrap(),
            serde_json::to_string(&StepLower).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&Strategy3DEnum::from(StepUpper)).unwrap(),
            serde_json::to_string(&StepUpper).unwrap(),
        );
    }
}
