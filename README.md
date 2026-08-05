# ninterp

[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-ninterp-F74C00?style=for-the-badge&logo=docs.rs" height=25>](https://docs.rs/ninterp/latest/ninterp)
[<img alt="crates.io" src="https://img.shields.io/crates/v/ninterp?style=for-the-badge&color=FFC932&logo=rust" height=25>](https://crates.io/crates/ninterp)
[<img alt="github.com" src="https://img.shields.io/badge/github-NatLabRockies/ninterp-0076BD?style=for-the-badge&logo=github" height=25>](https://github.com/NatLabRockies/ninterp/)

The `ninterp` crate provides [multivariate interpolation](https://en.wikipedia.org/wiki/Multivariate_interpolation#Regular_grid) over rectilinear grids of any dimensionality.

It is built on [`ndarray`](https://crates.io/crates/ndarray) and uses ndarray arrays/views throughout its API.

Hard-coded interpolators are provided for N = 1, 2, and 3, based on the observed runtime tradeoff versus a general N-D implementation.
For higher dimensionalities (N >= 4), use `InterpND`.
All interpolators work with both owned and borrowed arrays (array views) of various types.

A variety of interpolation strategies are implemented and exposed in the [`prelude`](https://docs.rs/ninterp/latest/ninterp/prelude/index.html) module.
Custom interpolation strategies can be defined in downstream crates.

## Quick Start

```text
cargo add ninterp
```

Bring common API types into scope:

```rust
use ninterp::prelude::*;
```

Minimal end-to-end interpolation example:

```rust
use ndarray::prelude::*;
use ninterp::prelude::*;

let interp = Interp1D::new(
    array![0.0, 1.0, 2.0, 3.0], // x
    array![0.0, 1.0, 4.0, 9.0], // f(x)
    strategy::Linear,
    Extrapolate::Error,
)
.unwrap();

let y = interp.interpolate(&[1.5]).unwrap();
assert_eq!(y, 2.5);
```

Minimal N-D interpolation example:

```rust
use ndarray::prelude::*;
use ninterp::prelude::*;

let interp_nd = InterpND::new(
    // grid
    vec![
        array![0.0, 1.0], // x0, x1
        array![0.0, 1.0], // y0, y1
    ],
    // values
    array![
        [0.0, 1.0], // f(x0, y0), f(x0, y1)
        [1.0, 2.0], // f(x1, y0), f(x1, y1)
    ].into_dyn(),
    strategy::Linear,
    Extrapolate::Error,
)
.unwrap();

let z = interp_nd.interpolate(&[0.25, 0.75]).unwrap();
assert_eq!(z, 1.0);
```

Instantiation is done by calling an interpolator's `new` method.
For dimensionalities N >= 1, this executes a validation step that prevents runtime panics.

## Cargo Features
- `serde`: support for [`serde`](https://crates.io/crates/serde) 1.x
  ```text
  cargo add ninterp --features serde
  ```

  By default, arrays are written in `ndarray`'s built-in format, which is performant to parse and works with every serialization format (text and binary):
  ```json
  {"grid":[{"v":1,"dim":[2],"data":[0.0,1.0]},{"v":1,"dim":[3],"data":[0.0,1.0,2.0]}],"values":{"v":1,"dim":[2,3],"data":[0.0,1.0,2.0,3.0,4.0,5.0]}}
  ```

  You can also serialize interpolators using the nested-array format from
  [`serde-ndim`](https://crates.io/crates/serde-ndim), which is far easier to read and hand-edit. This works for any `is_human_readable` serde format (binary formats will still necessarily serialize `ndarray`'s format).

  - On fields, using the `ninterp::serialize_nested` helper function:

    ```rust,ignore
    #[derive(serde::Serialize)]
    struct MyConfig {
        #[serde(serialize_with = "ninterp::serialize_nested")]
        surface: Interp2DOwned<f64, strategy::Linear>,
    }
    ```

  - Using the `ninterp::Nested` wrapper:

    ```rust,ignore
    use ninterp::Nested;

    let json = serde_json::to_string(&Nested(&interp.data)).unwrap();
    // {"grid":[[0.0,1.0],[0.0,1.0,2.0]],"values":[[0.0,1.0,2.0],[3.0,4.0,5.0]]}
    ```

  Deserialization accepts **either** format, so this is purely a choice about what you write:

  - Prefer the default when deserialization is on a hot path: nested arrays cost roughly 20% more to read,
  since `ndarray`'s format carries the shape up front and can allocate exactly once,
  while `serde-ndim` must parse the shape from the nested array every read.

  - Prefer `Nested` / `serialize_with = "ninterp::serialize_nested"` for config files and anything a human will look at.

## Choosing an Interpolator
The [`prelude`](https://docs.rs/ninterp/latest/ninterp/prelude/index.html) exposes these interpolators:
- [`Interp0D`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.Interp0D.html): constant-value interpolator
- [`Interp1D`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.Interp1D.html): hard-coded 1-D interpolator
- [`Interp2D`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.Interp2D.html): hard-coded 2-D interpolator
- [`Interp3D`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.Interp3D.html): hard-coded 3-D interpolator
- [`InterpND`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.InterpND.html): general N-D interpolator

Use `Interp0D` when working with heterogeneous collections such as an `InterpolatorEnum` or `Box<dyn Interpolator>`.

### Flexibility Model
| Approach | Runtime swapping | `serde` | Custom strategies | Runtime cost |
| --- | --- | --- | --- | --- |
| `Interp*<_, ConcreteStrategy>` | No | Yes | N/A | Lowest |
| `Interp*<_, strategy::enums::Strategy*Enum>` | Strategy only | Yes | No | Low |
| `Interp*<_, Box<dyn Strategy*>>` | Strategy only | No | Yes | Medium |
| `InterpolatorEnum` | Interpolator + strategy | Yes | No | Low |
| `Box<dyn Interpolator<_>>` | Interpolator + strategy | No | Yes | Highest |

## Core Concepts
### Validation Lifecycle
After editing interpolator data, call the InterpData `validate` method or
[`Interpolator::validate`](https://docs.rs/ninterp/latest/ninterp/interpolator/trait.Interpolator.html#tymethod.validate)
to rerun validation checks.

### Data Shape Contract
Grid and values shapes must match by axis order.

Examples:
- 1-D: `x.len() == f_x.len()`
- 2-D: `x.len() == f_xy.shape()[0]` and `y.len() == f_xy.shape()[1]`
- 3-D: `x.len() == f_xyz.shape()[0]`, `y.len() == f_xyz.shape()[1]`, `z.len() == f_xyz.shape()[2]`
- N-D: for every dimension `n`, `grid[n].len() == values.shape()[n]`

Grid coordinates in each dimension must be monotonically increasing.

### Strategies
An interpolation strategy (for example
[`Linear`](https://docs.rs/ninterp/latest/ninterp/strategy/struct.Linear.html),
[`LinearUniform`](https://docs.rs/ninterp/latest/ninterp/strategy/struct.LinearUniform.html),
[`Nearest`](https://docs.rs/ninterp/latest/ninterp/strategy/struct.Nearest.html),
[`Step`](https://docs.rs/ninterp/latest/ninterp/strategy/struct.Step.html)) must be specified.

To change the interpolation strategy, supply a `Strategy1DEnum`/etc. or `Box<dyn Strategy1D>`/etc. at instantiation and call `set_strategy`.
Custom strategies can be defined. See [`examples/custom_strategy.rs`](https://github.com/NatLabRockies/ninterp/blob/main/examples/custom_strategy.rs).

### Extrapolation
An [`Extrapolate`](https://docs.rs/ninterp/latest/ninterp/interpolator/enum.Extrapolate.html)
setting must be provided in `new`.
This controls behavior when points are beyond the supplied coordinate range.

Available for all interpolation strategies:
- `Extrapolate::Fill(T)`
- `Extrapolate::Clamp`
- `Extrapolate::Wrap`
- `Extrapolate::Error`

`Extrapolate::Enable` is valid for `Linear` and `LinearUniform` for all dimensionalities.
If you are unsure which variant to choose, `Extrapolate::Error` is a good default.

To change extrapolation behavior after construction, call `set_extrapolate`.

### Interpolation Calls
Interpolation is executed by calling [`Interpolator::interpolate`](https://docs.rs/ninterp/latest/ninterp/interpolator/trait.Interpolator.html#tymethod.interpolate).

The query point must contain one coordinate per dimension.
For example:
- 1-D interpolator: `&[x]`
- 2-D interpolator: `&[x, y]`
- 3-D interpolator: `&[x, y, z]`
- N-D interpolator: `&[x0, x1, ..., x_{N-1}]`

If the number of coordinates does not match dimensionality, interpolation returns an error.
Retrieve dimensionality using [`Interpolator::ndim`](https://docs.rs/ninterp/latest/ninterp/interpolator/trait.Interpolator.html#tymethod.ndim).

### Common Errors
Validation-time (`new` / `validate`):
- Empty grid coordinates (`ValidateError::EmptyGrid`)
- Non-monotonic coordinates (`ValidateError::Monotonicity`)
- Grid/value shape mismatch (`ValidateError::IncompatibleShapes`)
- Inapplicable extrapolation setting (`ValidateError::ExtrapolateSelection`)

Interpolation-time (`interpolate`):
- Query point has wrong dimensionality (`InterpolateError::PointLength`)
- Query point is out of bounds while using `Extrapolate::Error` (`InterpolateError::ExtrapolateError`)

## Using Owned and Borrowed (Viewed) Data
All interpolators support both owned and borrowed data via the generic `D` bound on
[`ndarray::Data`](https://docs.rs/ndarray/latest/ndarray/trait.Data.html).

The crate also re-exports [`ndarray`](https://docs.rs/ninterp/latest/ninterp/ndarray/index.html)
and [`num_traits`](https://docs.rs/ninterp/latest/ninterp/num_traits/index.html),
so either of these import styles are valid:

```rust
use ninterp::ndarray::prelude::*;
// or
use ndarray::prelude::*;
```

Type aliases in the [`prelude`](https://docs.rs/ninterp/latest/ninterp/prelude/index.html)
make ownership intent explicit, for example in 1-D:
- [`Interp1DOwned`](https://docs.rs/ninterp/latest/ninterp/interpolator/type.Interp1DOwned.html)
  - Data is owned by the interpolator object
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
  - Data is borrowed by the interpolator object
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

Typically, the compiler can infer concrete types from arguments passed to `new`.
Some examples use explicit annotations for clarity.

## Examples
See examples in `new` method documentation:
- [`Interp0D::new`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.Interp0D.html#method.new)
- [`Interp1D::new`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.Interp1D.html#method.new)
- [`Interp2D::new`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.Interp2D.html#method.new)
- [`Interp3D::new`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.Interp3D.html#method.new)
- [`InterpND::new`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.InterpND.html#method.new)

Also see the [`examples`](https://github.com/NatLabRockies/ninterp/tree/main/examples) directory for advanced examples:
- Swapping strategies at runtime: [`dynamic_strategy.rs`](https://github.com/NatLabRockies/ninterp/blob/main/examples/dynamic_strategy.rs)
  - Strategy enums (`strategy::enums::Strategy1DEnum`/etc.): `serde`-compatible, custom strategies not supported
  - `Box<dyn Strategy1D>`/etc. (dynamic dispatch): custom strategies supported, not `serde`-compatible, runtime cost
- Swapping interpolators at runtime: [`dynamic_interpolator.rs`](https://github.com/NatLabRockies/ninterp/blob/main/examples/dynamic_interpolator.rs)
  - `InterpolatorEnum`: `serde`-compatible, custom strategies not supported
  - `Box<dyn Interpolator>` (dynamic dispatch): custom strategies supported, not `serde`-compatible, runtime cost
- Defining custom strategies: [`custom_strategy.rs`](https://github.com/NatLabRockies/ninterp/blob/main/examples/custom_strategy.rs)
- Using transmutable (transparent) types such as [`uom::si::Quantity`](https://docs.rs/uom/0.36.0/uom/si/struct.Quantity.html): [`uom.rs`](https://github.com/NatLabRockies/ninterp/blob/main/examples/uom.rs)
