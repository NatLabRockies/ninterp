use super::*;
use uom::si::f64::Power;
use uom::si::power::kilowatt;

#[test]
fn interpolate_returns_constant() {
    let interp = UomInterp0D::new(Power::new::<kilowatt>(0.5));
    assert_eq!(interp.interpolate(), Power::new::<kilowatt>(0.5));
}

#[test]
fn partial_eq() {
    assert_eq!(
        UomInterp0D::new(Power::new::<kilowatt>(0.5)),
        UomInterp0D::new(Power::new::<kilowatt>(0.5))
    );
}

#[test]
fn inner_escape_hatch_is_reachable() {
    let interp = UomInterp0D::new(Power::new::<kilowatt>(0.5));
    // Base-unit caveat: `inner.0` is watts (`uom`'s SI base unit for power), not
    // kilowatts.
    assert_eq!(interp.inner.0, 500.);
}

#[test]
#[cfg(feature = "serde")]
fn serde_round_trip() {
    let interp = UomInterp0D::new(Power::new::<kilowatt>(0.5));
    let json = serde_json::to_string(&interp).unwrap();
    let de: UomInterp0D<Power, f64> = serde_json::from_str(&json).unwrap();
    assert_eq!(interp, de);
}
