use super::*;

/// See [enums module](super) documentation.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[cfg_attr(feature = "serde", serde(untagged))]
pub enum Strategy3DEnum {
    Linear(strategy::Linear),
    LinearUniform(strategy::LinearUniform),
    Nearest(strategy::Nearest),
    Step(strategy::Step),
}

impl From<Linear> for Strategy3DEnum {
    #[inline]
    fn from(strategy: Linear) -> Self {
        Self::Linear(strategy)
    }
}

impl From<LinearUniform> for Strategy3DEnum {
    #[inline]
    fn from(strategy: LinearUniform) -> Self {
        Self::LinearUniform(strategy)
    }
}

impl From<Nearest> for Strategy3DEnum {
    #[inline]
    fn from(strategy: Nearest) -> Self {
        Self::Nearest(strategy)
    }
}

impl From<Step> for Strategy3DEnum {
    #[inline]
    fn from(strategy: Step) -> Self {
        Self::Step(strategy)
    }
}

impl<D> Strategy3D<D> for Strategy3DEnum
where
    D: Data + RawDataClone + Clone,
    D::Elem: Float + Debug,
{
    #[inline]
    fn init(&mut self, data: &InterpData3D<D>) -> Result<(), ValidateError> {
        match self {
            Strategy3DEnum::Linear(strategy) => Strategy3D::<D>::init(strategy, data),
            Strategy3DEnum::LinearUniform(strategy) => Strategy3D::<D>::init(strategy, data),
            Strategy3DEnum::Nearest(strategy) => Strategy3D::<D>::init(strategy, data),
            Strategy3DEnum::Step(strategy) => Strategy3D::<D>::init(strategy, data),
        }
    }

    #[inline]
    fn interpolate(
        &self,
        data: &InterpData3D<D>,
        point: &[D::Elem; 3],
    ) -> Result<D::Elem, InterpolateError> {
        match self {
            Strategy3DEnum::Linear(strategy) => Strategy3D::<D>::interpolate(strategy, data, point),
            Strategy3DEnum::LinearUniform(strategy) => {
                Strategy3D::<D>::interpolate(strategy, data, point)
            }
            Strategy3DEnum::Nearest(strategy) => {
                Strategy3D::<D>::interpolate(strategy, data, point)
            }
            Strategy3DEnum::Step(strategy) => Strategy3D::<D>::interpolate(strategy, data, point),
        }
    }

    #[inline]
    fn allow_extrapolate(&self) -> bool {
        match self {
            Strategy3DEnum::Linear(strategy) => Strategy3D::<D>::allow_extrapolate(strategy),
            Strategy3DEnum::LinearUniform(strategy) => Strategy3D::<D>::allow_extrapolate(strategy),
            Strategy3DEnum::Nearest(strategy) => Strategy3D::<D>::allow_extrapolate(strategy),
            Strategy3DEnum::Step(strategy) => Strategy3D::<D>::allow_extrapolate(strategy),
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
            serde_json::to_string(&Strategy3DEnum::from(Linear)).unwrap(),
            serde_json::to_string(&Linear).unwrap(),
        );
        assert_eq!(
            serde_json::to_string(&Strategy3DEnum::from(Nearest)).unwrap(),
            serde_json::to_string(&Nearest).unwrap(),
        );
    }
}
