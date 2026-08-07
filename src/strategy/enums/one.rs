use super::*;

/// See [enums module](super) documentation.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(feature = "serde", serde(untagged))]
pub enum Strategy1DEnum {
    Linear(strategy::Linear),
    LinearUniform(strategy::LinearUniform),
    Nearest(strategy::Nearest),
    Step(strategy::Step),
    StepLower(strategy::StepLower),
    StepUpper(strategy::StepUpper),
}

impl From<Linear> for Strategy1DEnum {
    #[inline]
    fn from(strategy: Linear) -> Self {
        Self::Linear(strategy)
    }
}

impl From<LinearUniform> for Strategy1DEnum {
    #[inline]
    fn from(strategy: LinearUniform) -> Self {
        Self::LinearUniform(strategy)
    }
}

impl From<Nearest> for Strategy1DEnum {
    #[inline]
    fn from(strategy: Nearest) -> Self {
        Self::Nearest(strategy)
    }
}

impl From<Step> for Strategy1DEnum {
    #[inline]
    fn from(strategy: Step) -> Self {
        Self::Step(strategy)
    }
}

impl From<StepLower> for Strategy1DEnum {
    #[inline]
    fn from(strategy: StepLower) -> Self {
        Self::StepLower(strategy)
    }
}

impl From<StepUpper> for Strategy1DEnum {
    #[inline]
    fn from(strategy: StepUpper) -> Self {
        Self::StepUpper(strategy)
    }
}

impl<D> Strategy1D<D> for Strategy1DEnum
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    #[inline]
    fn validate(&self, data: &InterpData1D<D>) -> Result<(), ValidateError> {
        match self {
            Strategy1DEnum::Linear(strategy) => Strategy1D::<D>::validate(strategy, data),
            Strategy1DEnum::LinearUniform(strategy) => Strategy1D::<D>::validate(strategy, data),
            Strategy1DEnum::Nearest(strategy) => Strategy1D::<D>::validate(strategy, data),
            Strategy1DEnum::Step(strategy) => Strategy1D::<D>::validate(strategy, data),
            Strategy1DEnum::StepLower(strategy) => Strategy1D::<D>::validate(strategy, data),
            Strategy1DEnum::StepUpper(strategy) => Strategy1D::<D>::validate(strategy, data),
        }
    }

    #[inline]
    fn init(&mut self, data: &InterpData1D<D>) -> Result<(), ValidateError> {
        match self {
            Strategy1DEnum::Linear(strategy) => Strategy1D::<D>::init(strategy, data),
            Strategy1DEnum::LinearUniform(strategy) => Strategy1D::<D>::init(strategy, data),
            Strategy1DEnum::Nearest(strategy) => Strategy1D::<D>::init(strategy, data),
            Strategy1DEnum::Step(strategy) => Strategy1D::<D>::init(strategy, data),
            Strategy1DEnum::StepLower(strategy) => Strategy1D::<D>::init(strategy, data),
            Strategy1DEnum::StepUpper(strategy) => Strategy1D::<D>::init(strategy, data),
        }
    }

    #[inline]
    fn interpolate(
        &self,
        data: &InterpData1D<D>,
        point: &[D::Elem; 1],
    ) -> Result<D::Elem, InterpolateError> {
        match self {
            Strategy1DEnum::Linear(strategy) => Strategy1D::<D>::interpolate(strategy, data, point),
            Strategy1DEnum::LinearUniform(strategy) => {
                Strategy1D::<D>::interpolate(strategy, data, point)
            }
            Strategy1DEnum::Nearest(strategy) => {
                Strategy1D::<D>::interpolate(strategy, data, point)
            }
            Strategy1DEnum::Step(strategy) => Strategy1D::<D>::interpolate(strategy, data, point),
            Strategy1DEnum::StepLower(strategy) => {
                Strategy1D::<D>::interpolate(strategy, data, point)
            }
            Strategy1DEnum::StepUpper(strategy) => {
                Strategy1D::<D>::interpolate(strategy, data, point)
            }
        }
    }

    #[inline]
    fn allow_extrapolate(&self) -> bool {
        match self {
            Strategy1DEnum::Linear(strategy) => Strategy1D::<D>::allow_extrapolate(strategy),
            Strategy1DEnum::LinearUniform(strategy) => Strategy1D::<D>::allow_extrapolate(strategy),
            Strategy1DEnum::Nearest(strategy) => Strategy1D::<D>::allow_extrapolate(strategy),
            Strategy1DEnum::Step(strategy) => Strategy1D::<D>::allow_extrapolate(strategy),
            Strategy1DEnum::StepLower(strategy) => Strategy1D::<D>::allow_extrapolate(strategy),
            Strategy1DEnum::StepUpper(strategy) => Strategy1D::<D>::allow_extrapolate(strategy),
        }
    }
}

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
