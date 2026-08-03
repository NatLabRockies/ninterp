# ninterp

[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-ninterp-F74C00?style=for-the-badge&logo=docs.rs" height=25>](https://docs.rs/ninterp/latest/ninterp)
[<img alt="crates.io" src="https://img.shields.io/crates/v/ninterp?style=for-the-badge&color=FFC932&logo=rust" height=25>](https://crates.io/crates/ninterp)
[<img alt="github.com" src="https://img.shields.io/badge/github-NatLabRockies/ninterp-0076BD?style=for-the-badge&logo=github" height=25>](https://github.com/NatLabRockies/ninterp/)

The `ninterp` crate provides [multivariate interpolation](https://en.wikipedia.org/wiki/Multivariate_interpolation#Regular_grid) over rectilinear grids of any dimensionality.

There are hard-coded interpolators for lower dimensionalities (up to N = 3) for better runtime performance.
All interpolators work with both owned and borrowed arrays (array views) of various types.

A variety of interpolation strategies are implemented and exposed in the [`prelude`](https://docs.rs/ninterp/latest/ninterp/prelude/index.html) module.
Custom interpolation strategies can be defined in downstream crates.

```text
cargo add ninterp
```

#### Cargo Features
- `serde`: support for [`serde`](https://crates.io/crates/serde) 1.x using ndarray's built-in legacy wire format
  ```text
  cargo add ninterp --features serde
  ```
- `serde_ndim`: enable `serde` feature and switch output to the column-major nested array format from [`serde-ndim`](https://crates.io/crates/serde-ndim)
  ```text
  cargo add ninterp --features serde_ndim
  ```

## Examples
See examples in `new` method documentation:
- [`Interp0D::new`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.Interp0D.html#method.new)
- [`Interp1D::new`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.Interp1D.html#method.new)
- [`Interp2D::new`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.Interp2D.html#method.new)
- [`Interp3D::new`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.Interp3D.html#method.new)
- [`InterpND::new`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.InterpND.html#method.new)

Also see the [`examples`](https://github.com/NatLabRockies/ninterp/tree/main/examples) directory for advanced examples:
- Swapping strategies at runtime: **[`dynamic_strategy.rs`](examples/dynamic_strategy.rs)**
  - Using strategy enums (`strategy::enums::Strategy1DEnum`/etc.)
    - Compatible with `serde`
    - Incompatible with custom strategies
  - Using `Box<dyn Strategy1D>`/etc. (dynamic dispatch)
    - Incompatible with `serde`
    - Compatible with custom strategies
    - Runtime cost

- Swapping interpolators at runtime: **[`dynamic_interpolator.rs`](examples/dynamic_interpolator.rs)**
  - Using `InterpolatorEnum`
    - Compatible with `serde`
    - Incompatible with custom strategies
  - Using `Box<dyn Interpolator>` (dynamic dispatch)
    - Incompatible with `serde`
    - Compatible with custom strategies
    - Runtime cost

- Defining custom strategies: **[`custom_strategy.rs`](examples/custom_strategy.rs)**

- Using transmutable (transparent) types, such as [`uom::si::Quantity`](https://docs.rs/uom/0.36.0/uom/si/struct.Quantity.html):
  **[`uom.rs`](examples/uom.rs)**

## Overview
A [`prelude`](https://docs.rs/ninterp/latest/ninterp/prelude/index.html) module has been defined: 
```rust
use ninterp::prelude::*;
```

This exposes all strategies and a variety of interpolators:
- [`Interp1D`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.Interp1D.html)
- [`Interp2D`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.Interp2D.html)
- [`Interp3D`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.Interp3D.html)
- [`InterpND`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.InterpND.html)

There is also a constant-value 'interpolator':
[`Interp0D`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.Interp0D.html).
This is useful when working with an `InterpolatorEnum` or `Box<dyn Interpolator>`

Instantiation is done by calling an interpolator's `new` method.
For dimensionalities N ≥ 1, this executes a validation step, preventing runtime panics.
After editing interpolator data,
call the InterpData's `validate` method
or [`Interpolator::validate`](https://docs.rs/ninterp/latest/ninterp/interpolator/trait.Interpolator.html#tymethod.validate)
to rerun these checks.

To change the extrapolation setting, call `set_extrapolate`.

To change the interpolation strategy,
supply a `Strategy1DEnum`/etc. or `Box<dyn Strategy1D>`/etc. upon instantiation,
and call `set_strategy`.

### Strategies
An interpolation strategy (e.g.
[`Linear`](https://docs.rs/ninterp/latest/ninterp/strategy/struct.Linear.html),
[`Nearest`](https://docs.rs/ninterp/latest/ninterp/strategy/struct.Nearest.html),
[`LeftNearest`](https://docs.rs/ninterp/latest/ninterp/strategy/struct.LeftNearest.html),
[`RightNearest`](https://docs.rs/ninterp/latest/ninterp/strategy/struct.RightNearest.html))
must be specified.
Not all interpolation strategies are implemented for every dimensionality.
`Linear` and `Nearest` are implemented for all dimensionalities.

Custom strategies can be defined. See
[`examples/custom_strategy.rs`](examples/custom_strategy.rs)
for an example.

### Extrapolation
An [`Extrapolate`](https://docs.rs/ninterp/latest/ninterp/interpolator/enum.Extrapolate.html)
setting must be provided in the `new` method.
This controls what happens when a point is beyond the range of supplied coordinates.
The following settings are applicable for all interpolators:
- `Extrapolate::Fill(T)`
- `Extrapolate::Clamp`
- `Extrapolate::Wrap`
- `Extrapolate::Error`

`Extrapolate::Enable` is valid for `Linear` for all dimensionalities.

If you are unsure which variant to choose, `Extrapolate::Error` is likely what you want.

### Interpolation
Interpolation is executed by calling [`Interpolator::interpolate`](https://docs.rs/ninterp/latest/ninterp/interpolator/trait.Interpolator.html#tymethod.interpolate).

The length of the interpolant point slice must be equal to the interpolator dimensionality.
The interpolator dimensionality can be retrieved by calling [`Interpolator::ndim`](https://docs.rs/ninterp/latest/ninterp/interpolator/trait.Interpolator.html#tymethod.ndim).

## Using Owned and Borrowed (Viewed) Data
All interpolators work with both owned and borrowed data.
This is accomplished by the generic `D`, which has a bound on the
[`ndarray::Data`](https://docs.rs/ndarray/latest/ndarray/trait.Data.html)
trait.

Type aliases are provided in the
[`prelude`](https://docs.rs/ninterp/latest/ninterp/prelude/index.html)
for convenience, e.g. for 1-D:
- [`Interp1DOwned`](https://docs.rs/ninterp/latest/ninterp/interpolator/type.Interp1DOwned.html)
  - Data is *owned* by the interpolator object
  - Useful for struct fields
  ```rust
  use ndarray::prelude::*;
  use ninterp::prelude::*;
  let interp: Interp1DOwned<f64, _> = Interp1D::new(
      array![0.0, 1.0, 2.0, 3.0],
      array![0.0, 1.0, 4.0, 9.0],
      strategy::Linear,
      Extrapolate::Error,
  )
  .unwrap();
  ```
- [`Interp1DViewed`](https://docs.rs/ninterp/latest/ninterp/interpolator/type.Interp1DViewed.html)
  - Data is *borrowed* by the interpolator object
  - Use when interpolator data should be owned by another object
  ```rust
  use ndarray::prelude::*;
  use ninterp::prelude::*;
  let x = array![0.0, 1.0, 2.0, 3.0];
  let f_x = array![0.0, 1.0, 4.0, 9.0];
  let interp: Interp1DViewed<&f64, _> = Interp1D::new(
      x.view(),
      f_x.view(),
      strategy::Linear,
      Extrapolate::Error,
  )
  .unwrap();
  ```

Typically, the compiler can determine concrete types using the arguments provided to `new` methods.
Examples throughout this crate have type annotions for clarity purposes; they are often unnecessary.
