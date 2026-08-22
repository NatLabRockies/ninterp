# ninterp

[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-ninterp-F74C00?style=for-the-badge&logo=docs.rs" height=25>](https://docs.rs/ninterp/latest/ninterp)
[<img alt="crates.io" src="https://img.shields.io/crates/v/ninterp?style=for-the-badge&color=FFC932&logo=rust" height=25>](https://crates.io/crates/ninterp)
[<img alt="github.com" src="https://img.shields.io/badge/github-NatLabRockies/ninterp-0076BD?style=for-the-badge&logo=github" height=25>](https://github.com/NatLabRockies/ninterp/)

The `ninterp` crate provides [multivariate interpolation](https://en.wikipedia.org/wiki/Multivariate_interpolation#Regular_grid) over rectilinear grids of any dimensionality.

It is built on [`ndarray`](https://crates.io/crates/ndarray) and uses ndarray arrays/views throughout its API.
`ndarray` and [`num_traits`](https://crates.io/crates/num_traits) are re-exposed as `ninterp::ndarray` and
`ninterp::num_traits` for convenience.

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

Minimal end-to-end interpolation example, using the hard-coded 2-D interpolator:

```rust
use ndarray::prelude::*;
use ninterp::prelude::*;

let interp = Interp2D::new(
    array![0.0, 1.0], // x
    array![0.0, 1.0], // y
    array![
        [0.0, 1.0], // f(x0, y0), f(x0, y1)
        [1.0, 2.0], // f(x1, y0), f(x1, y1)
    ],
    strategy::Linear,
    Extrapolate::Error,
)
.unwrap();

let z = interp.interpolate(&[0.25, 0.75]).unwrap();
assert_eq!(z, 1.0);
```

The same example, generalized to `InterpND` (grid axes become a `Vec`, values become
dynamically-dimensioned via `.into_dyn()`):

```rust
use ndarray::prelude::*;
use ninterp::prelude::*;

let interp_nd = InterpND::new(
    vec![
        array![0.0, 1.0], // x
        array![0.0, 1.0], // y
    ],
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

## Choosing an Interpolator
The [`prelude`](https://docs.rs/ninterp/latest/ninterp/prelude/index.html) exposes these interpolators:
- [`Interp0D`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.Interp0D.html): constant-value interpolator
- [`Interp1D`](https://docs.rs/ninterp/latest/ninterp/interpolator/type.Interp1D.html): hard-coded 1-D interpolator
- [`Interp2D`](https://docs.rs/ninterp/latest/ninterp/interpolator/type.Interp2D.html): hard-coded 2-D interpolator
- [`Interp3D`](https://docs.rs/ninterp/latest/ninterp/interpolator/type.Interp3D.html): hard-coded 3-D interpolator
- [`InterpND`](https://docs.rs/ninterp/latest/ninterp/interpolator/type.InterpND.html): general N-D interpolator

Use `Interp0D` when working with heterogeneous collections such as an `InterpolatorEnum` or `Box<dyn Interpolator>`.

### Compiled vs. Runtime Flexibility

This crate is designed to maximize both performance and runtime flexibility.
There are multiple ways to specify interpolators and strategies;
You can pin a concrete dimensionality and strategy at compile-time for best performance,
or swap either via provided enums or dynamic dispatch.
Each approach has trade-offs:

| Approach | Runtime cost | Runtime swapping | Custom strategies | `serde` |
| --- | --- | --- | --- | --- |
| `Interp*<_, ConcreteStrategy>` | Lowest | No | Yes | Yes |
| `Interp*<_, strategy::enums::Strategy*Enum>` | Low | Strategy only | No | Yes |
| `InterpolatorEnum` | Low | Interpolator + strategy | No | Yes |
| `Interp*<_, Box<dyn Strategy*>>` | Medium | Strategy only | Yes | No |
| `Box<dyn Interpolator<_>>` | Highest | Interpolator + strategy | Yes | No |

For heterogeneous storage that also needs to recover the concrete interpolator type later,
[`AnyInterpolator`](https://docs.rs/ninterp/latest/ninterp/interpolator/trait.AnyInterpolator.html)
(`ninterp::interpolator::AnyInterpolator`, not in the `prelude`) adds `Send + Sync` and an
`as_any(&self) -> &dyn Any` method on top of `Interpolator<T>`, so
`boxed.as_any().downcast_ref::<Interp2D<f64, MyStrategy>>()` recovers the concrete type.
It's implemented for owned `Interp1D`/`2D`/`3D`/`ND` only, since downcasting requires
`Self: 'static`.

## Core Concepts
### Data Shape Contract
Grid and values shapes must match by axis order:
for every dimension `n`, `grid[n].len() == values.shape()[n]`.

Grid coordinates in each dimension must be strictly increasing (no repeated adjacent
coordinates), with at least 2 points per dimension (`ValidateError::InsufficientGridPoints`
otherwise).

### Strategies
An interpolation strategy must be specified. Provided strategies:

| Strategy | Description |
| --- | --- |
| [`Nearest`](https://docs.rs/ninterp/latest/ninterp/strategy/struct.Nearest.html) | Nearest-neighbor interpolation |
| [`Step`](https://docs.rs/ninterp/latest/ninterp/strategy/step/struct.Step.html) | Step interpolation to the previous/next grid value, broadcast or per-dimension |
| [`Linear`](https://docs.rs/ninterp/latest/ninterp/strategy/struct.Linear.html) | Linear interpolation |
| [`LinearUniform`](https://docs.rs/ninterp/latest/ninterp/strategy/struct.LinearUniform.html) | Linear interpolation for uniformly spaced grids |
| [`CubicC2`](https://docs.rs/ninterp/latest/ninterp/strategy/cubic/struct.CubicC2.html) | C²-continuous cubic spline with boundary conditions ([`CubicC2BoundaryConditions`](https://docs.rs/ninterp/latest/ninterp/strategy/cubic/enum.CubicC2BoundaryConditions.html): not-a-knot, first derivative ("clamped"), second derivative (zero is "natural"), or periodic, with per-endpoint, per-axis configuration flexibility) |

More strategies will be added in the future, see [Issue #24](https://github.com/NatLabRockies/ninterp/issues/24).

To change the interpolation strategy, supply a `Strategy*DEnum` or `Box<dyn Strategy*D>` at instantiation and call `set_strategy`.
Custom strategies can be defined, see [`examples/custom_strategy.rs`](https://github.com/NatLabRockies/ninterp/blob/main/examples/custom_strategy.rs).

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
Call `interpolate` with one coordinate per dimension:
- 1-D interpolator: `interp.interpolate(&[x])`
- 2-D interpolator: `interp.interpolate(&[x, y])`
- 3-D interpolator: `interp.interpolate(&[x, y, z])`
- N-D interpolator: `interp.interpolate(&[x0, x1, ..., x_{N-1}])`

For `Interp1D`/`Interp2D`/`Interp3D`, a wrong-length point is a compile error. For
`InterpND`, `InterpolatorEnum`, and `Box<dyn Interpolator<T>>`, it's instead a runtime
`InterpolateError::PointLength`. Retrieve dimensionality using
[`Interpolator::ndim`](https://docs.rs/ninterp/latest/ninterp/interpolator/trait.Interpolator.html#tymethod.ndim)
if necessary.

For hot loops where the point is already known to be in-bounds and extrapolation isn't
needed, [`interpolate_fast`](https://docs.rs/ninterp/latest/ninterp/interpolator/trait.Interpolator.html#method.interpolate_fast)
skips the bounds check and extrapolation handling that `interpolate` does, returning
`D::Elem` directly instead of a `Result` (panicking, instead of erroring, on an invalid
point). It takes the same point argument as `interpolate` above.

### Batch Interpolation
To interpolate several points against one interpolator,
[`batch_interpolate`](https://docs.rs/ninterp/latest/ninterp/interpolator/trait.Interpolator.html#method.batch_interpolate)
(and its `interpolate_fast`-style counterpart `batch_interpolate_fast`) resolve
`self.extrapolate` once for the whole batch and funnel every point into at most one call
to the strategy, instead of calling `interpolate` per point. For a `Box<dyn
Interpolator<T>>` or `Box<dyn Strategy*D>`, this also collapses one virtual dispatch
per point into a single dispatch for the whole batch:

```rust
use ndarray::prelude::*;
use ninterp::prelude::*;

let interp = Interp1D::new(
    array![0.0, 1.0, 2.0, 3.0],
    array![0.0, 1.0, 4.0, 9.0],
    strategy::Linear,
    Extrapolate::Error,
)
.unwrap();

let ys = interp.batch_interpolate(&[[0.5], [1.5], [2.5]]).unwrap();
assert_eq!(ys, vec![0.5, 2.5, 6.5]);
```

`batch_interpolate_into`/`batch_interpolate_fast_into` write into a caller-supplied output
slice instead of allocating a `Vec`. `batch_interpolate_into` returns
`InterpolateError::OutputLength` if the slice length doesn't match the point count;
`batch_interpolate_fast_into` instead panics on that mismatch, matching `interpolate_fast`'s
panic-instead-of-`Result` convention.

### Validation Lifecycle
After editing interpolator data, call
[`Interpolator::validate`](https://docs.rs/ninterp/latest/ninterp/interpolator/trait.Interpolator.html#tymethod.validate)
to rerun data, extrapolate, and strategy validation checks together, or `interp.data.validate()`
directly for just the narrower check that grid/value shapes match and grid coordinates are
strictly increasing.

`validate` checks the data (shapes, strictly increasing grid coordinates), the extrapolate
setting, and runs the strategy's own `validate`, a pure check for invariants that don't
need precomputed state (for example `LinearUniform`'s uniform-grid requirement, or
`Step`'s per-dimension direction count). It does not re-run the strategy's `init`, the
mutating counterpart that only strategies caching derived state (such as precomputed
spline coefficients) need to override.

`new` and `set_strategy` call both `validate` and `init` on the strategy. If you mutate
the public `data`/`strategy` fields directly instead, call `validate_strategy`/
`init_strategy` to re-run just those steps.

Deserialization does not call `validate` or `init`. Call `validate` afterward regardless:
deserialized data isn't guaranteed to satisfy grid/strategy invariants (hand-edited JSON,
a foreign writer, etc.), and the check is cheap. Whether you also need `init_strategy`
depends on the strategy: if its cached state is stored in its own serialized fields,
deserialize restores it as-is and `init` doesn't need to rerun; if that state is instead
skipped from serialization (e.g. via `#[serde(skip)]`, to avoid bloating the wire format
with a large derived array), call `init_strategy` to rebuild it.

### Errors
Validation-time (`new` / `validate`):

| Error | Meaning |
| --- | --- |
| `ValidateError::InsufficientGridPoints` | Fewer than 2 grid coordinates in a dimension |
| `ValidateError::NotStrictlyIncreasing` | Grid coordinates in a dimension aren't strictly increasing (a repeat or a decrease) |
| `ValidateError::NonUniform` | Non-uniformly-spaced grid, required by `LinearUniform` and any strategy calling `validate_uniform_grid`/`validate_uniform_grid_epsilon` |
| `ValidateError::IncompatibleShapes` | Grid/value shape mismatch |
| `ValidateError::GridAxisCount` | `InterpDataND` grid axis count doesn't match the dimensionality of `values` |
| `ValidateError::ExtrapolateUnsupported` | `Extrapolate::Enable` on a strategy that can't extrapolate |

Interpolation-time (`interpolate` / batch interpolation):

| Error | Meaning |
| --- | --- |
| `InterpolateError::PointLength` | Query point has wrong dimensionality; carries a `WrongLengthAt` per offending point |
| `InterpolateError::OutOfBounds` | Query point is out of bounds while using `Extrapolate::Error`; carries an `OutOfBoundsAt` per offending coordinate |
| `InterpolateError::OutputLength` | `batch_interpolate_into` output slice length doesn't match the point count |

## Using Owned and Borrowed (Viewed) Data
All interpolators support both owned and borrowed data via the generic `D` bound on
[`ndarray::Data`](https://docs.rs/ndarray/latest/ndarray/trait.Data.html).

Type aliases in the [`prelude`](https://docs.rs/ninterp/latest/ninterp/prelude/index.html)
follow Rust idioms (like `String` vs `&str` and `Vec` vs `&[T]`), making ownership intent explicit.
For example, in 1-D:

- [`Interp1D`](https://docs.rs/ninterp/latest/ninterp/interpolator/type.Interp1D.html) (owned data)
  - Default type for owned arrays
  - Examples: struct fields, general use
  ```rust
  use ndarray::prelude::*;
  use ninterp::prelude::*;
  let interp = Interp1D::new(
      array![0.0, 1.0, 2.0, 3.0],
      array![0.0, 1.0, 4.0, 9.0],
      strategy::Linear,
      Extrapolate::Error,
  )
  .unwrap();
  ```

- [`Interp1DView`](https://docs.rs/ninterp/latest/ninterp/interpolator/type.Interp1DView.html) (borrowed data)
  - For viewed arrays (data borrowed from elsewhere)
  - Examples: data lives in a larger struct, data is shared without copying
  ```rust
  use ndarray::prelude::*;
  use ninterp::prelude::*;
  let x = array![0.0, 1.0, 2.0, 3.0];
  let f_x = array![0.0, 1.0, 4.0, 9.0];
  let interp = Interp1DView::new(
      x.view(),
      f_x.view(),
      strategy::Linear,
      Extrapolate::Error,
  )
  .unwrap();
  ```

The same pattern applies to 2-D, 3-D, and N-D interpolators (`Interp2D`/`Interp2DView`, etc.).

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
  [`serde-ndim`](https://crates.io/crates/serde-ndim), which is far easier to read and hand-edit. This works for any `is_human_readable` serde format (serializing to binary formats necessarily uses the standard `ndarray` style).

  - On fields, using the `serialize_nested` helper function from [`ninterp::prelude`](https://docs.rs/ninterp/latest/ninterp/prelude/index.html):

    ```rust,ignore
    use ninterp::prelude::*;

    #[derive(serde::Serialize)]
    struct MyConfig {
        #[serde(serialize_with = "serialize_nested")]
        surface: Interp2D<f64, strategy::Linear>,
    }
    ```

  - Directly, using the `Nested` wrapper:

    ```rust,ignore
    use ninterp::prelude::*;

    let json = serde_json::to_string(&Nested(&interp.data)).unwrap();
    // {"grid":[[0.0,1.0],[0.0,1.0,2.0]],"values":[[0.0,1.0,2.0],[3.0,4.0,5.0]]}
    ```

  Deserialization accepts **either** format, so this is purely a choice about what you write:

  - Prefer the default when deserialization is on a hot path: nested arrays cost roughly 20% more to read,
    since `ndarray`'s format carries the shape up front and can allocate exactly once,
    while `serde-ndim` must parse the shape from the nested array every read.

  - Prefer `Nested` / `serialize_with = "serialize_nested"` for config files and anything a human will look at,
    so long as the array read cost is worth it.

## Examples
See examples in `new` method documentation:
- [`Interp0D::new`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.Interp0D.html#method.new)
- [`Interp1D::new`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.Interp1DBase.html#method.new)
- [`Interp2D::new`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.Interp2DBase.html#method.new)
- [`Interp3D::new`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.Interp3DBase.html#method.new)
- [`InterpND::new`](https://docs.rs/ninterp/latest/ninterp/interpolator/struct.InterpNDBase.html#method.new)

Also see the [`examples`](https://github.com/NatLabRockies/ninterp/tree/main/examples) directory for advanced examples:
- Swapping strategies at runtime: [`swap_strategy.rs`](https://github.com/NatLabRockies/ninterp/blob/main/examples/swap_strategy.rs)
  - Strategy enums (`strategy::enums::Strategy1DEnum`/etc.): `serde`-compatible, custom strategies not supported
  - `Box<dyn Strategy1D>`/etc. (dynamic dispatch): custom strategies supported, not `serde`-compatible, runtime cost
- Swapping interpolators at runtime: [`swap_interpolator.rs`](https://github.com/NatLabRockies/ninterp/blob/main/examples/swap_interpolator.rs)
  - `InterpolatorEnum`: `serde`-compatible, custom strategies not supported
  - `Box<dyn Interpolator>` (dynamic dispatch): custom strategies supported, not `serde`-compatible, runtime cost
- Defining custom strategies: [`custom_strategy.rs`](https://github.com/NatLabRockies/ninterp/blob/main/examples/custom_strategy.rs)
- Using transmutable (transparent) types such as [`uom::si::Quantity`](https://docs.rs/uom/0.36.0/uom/si/struct.Quantity.html): [`uom.rs`](https://github.com/NatLabRockies/ninterp/blob/main/examples/uom.rs)
