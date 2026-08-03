use super::*;

/// See [enums module](super) documentation.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(feature = "serde", serde(untagged))]
pub enum StrategyNDEnum {
    Linear(strategy::Linear),
    LinearUniform(strategy::LinearUniform),
    Nearest(strategy::Nearest),
}

impl From<Linear> for StrategyNDEnum {
    #[inline]
    fn from(strategy: Linear) -> Self {
        StrategyNDEnum::Linear(strategy)
    }
}

impl From<LinearUniform> for StrategyNDEnum {
    #[inline]
    fn from(strategy: LinearUniform) -> Self {
        StrategyNDEnum::LinearUniform(strategy)
    }
}

impl From<Nearest> for StrategyNDEnum {
    #[inline]
    fn from(strategy: Nearest) -> Self {
        StrategyNDEnum::Nearest(strategy)
    }
}

impl<D> StrategyND<D> for StrategyNDEnum
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    #[inline]
    fn init(&mut self, data: &InterpDataND<D>) -> Result<(), ValidateError> {
        match self {
            StrategyNDEnum::Linear(strategy) => StrategyND::<D>::init(strategy, data),
            StrategyNDEnum::LinearUniform(strategy) => StrategyND::<D>::init(strategy, data),
            StrategyNDEnum::Nearest(strategy) => StrategyND::<D>::init(strategy, data),
        }
    }

    #[inline]
    fn interpolate(
        &self,
        data: &InterpDataND<D>,
        point: &[D::Elem],
    ) -> Result<D::Elem, InterpolateError> {
        match self {
            StrategyNDEnum::Linear(strategy) => StrategyND::<D>::interpolate(strategy, data, point),
            StrategyNDEnum::LinearUniform(strategy) => StrategyND::<D>::interpolate(strategy, data, point),
            StrategyNDEnum::Nearest(strategy) => {
                StrategyND::<D>::interpolate(strategy, data, point)
            }
        }
    }

    #[inline]
    fn allow_extrapolate(&self) -> bool {
        match self {
            StrategyNDEnum::Linear(strategy) => StrategyND::<D>::allow_extrapolate(strategy),
            StrategyNDEnum::LinearUniform(strategy) => StrategyND::<D>::allow_extrapolate(strategy),
            StrategyNDEnum::Nearest(strategy) => StrategyND::<D>::allow_extrapolate(strategy),
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
            serde_json::to_string(&StrategyNDEnum::from(Linear)).unwrap(),
            serde_json::to_string(&Linear).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&StrategyNDEnum::from(Nearest)).unwrap(),
            serde_json::to_string(&Nearest).unwrap(),
        );
    }
}
