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

impl<D> Strategy1D<D> for Strategy1DEnum
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    #[inline]
    fn init(&mut self, data: &InterpData1D<D>) -> Result<(), ValidateError> {
        match self {
            Strategy1DEnum::Linear(strategy) => Strategy1D::<D>::init(strategy, data),
            Strategy1DEnum::LinearUniform(strategy) => Strategy1D::<D>::init(strategy, data),
            Strategy1DEnum::Nearest(strategy) => Strategy1D::<D>::init(strategy, data),
            Strategy1DEnum::Step(strategy) => Strategy1D::<D>::init(strategy, data),
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
        }
    }

    #[inline]
    fn allow_extrapolate(&self) -> bool {
        match self {
            Strategy1DEnum::Linear(strategy) => Strategy1D::<D>::allow_extrapolate(strategy),
            Strategy1DEnum::LinearUniform(strategy) => Strategy1D::<D>::allow_extrapolate(strategy),
            Strategy1DEnum::Nearest(strategy) => Strategy1D::<D>::allow_extrapolate(strategy),
            Strategy1DEnum::Step(strategy) => Strategy1D::<D>::allow_extrapolate(strategy),
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
    }
}
