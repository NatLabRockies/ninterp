# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project is pre-1.0 and follows the common pre-1.0 convention: breaking changes
bump the minor version (`0.x` -> `0.(x+1)`), other changes bump the patch version.

## [Unreleased]

### Added
- `strategy::CubicC1`: a C¹ local cubic Hermite spline strategy (finite-difference
  derivative estimate, no global solve). Cheaper to build than `CubicC2`, matching the
  local/uncached recipe LHAPDF-style consumers (e.g. neopdf) use by default at 3D+, but
  not aiming for bit-for-bit LHAPDF parity. `derivative_mode` carries the
  derivative-estimate method (`FiniteDifference` for now, `#[non_exhaustive]` for future
  monotonicity-preserving variants); `cache_mode` chooses between precomputing the full
  corner-derivative tensor at `init()` (`Full`, the default, same mechanism as
  `CubicC2`) or deriving it fresh from a bounded local neighborhood on every query
  (`None`) at 2-D and above. Closes #55.

## [0.11.1] - 2026-08-22

### Added
- `std` feature, on by default: opting out via `default-features = false` makes the
  crate `no_std`-compatible (`alloc` still required). The `serde` feature doesn't yet
  build without `std`, blocked on an upstream fix
  ([RReverser/serde-ndim#8](https://github.com/RReverser/serde-ndim/pull/8)).

## [0.11.0] - 2026-08-18

### Changed
- **Breaking:** `InterpolatorEnumBase`'s `From<Interp1DBase<D, S>>`/`From<Interp2DBase<D,
  S>>`/`From<Interp3DBase<D, S>>`/`From<InterpNDBase<D, S>>` impls are now generic over
  any strategy `S: Into<Strategy*DEnum<D::Elem>>`, not just the enum strategy type
  itself, so e.g. `Interp1D::new(x, f_x, strategy::Nearest, extrapolate)?.into()`
  converts directly into an `InterpolatorEnum` without first converting the strategy to
  `Strategy1DEnum`. Loosening the bound can change type inference at call sites that
  chained `.into()` without a turbofish; add an explicit type/turbofish if that happens.

## [0.10.0] - 2026-08-17

### Added
- `strategy::LinearUniform`: an O(1)-index alternative to `Linear` for uniformly-spaced
  grids.
- `strategy::Step`: a parameterized step (piecewise-constant) strategy, replacing
  `LeftNearest`/`RightNearest`, with per-axis direction control.
- `strategy::CubicC2`: a C² piecewise cubic spline strategy with configurable boundary
  conditions (not-a-knot, clamped, natural, periodic).
- `strategy::GridTransform`/`ValuesTransform`: interpolate in a transformed
  coordinate/value space (log, sqrt, reciprocal) instead of the raw one, e.g. for
  log-log interpolation or bounding output to always-positive values. Closes #56.
- `interpolate_fast`/`batch_interpolate`/`batch_interpolate_fast` (plus allocation-free
  `_into` variants): panic-instead-of-`Result` and batched point interpolation paths for
  hot loops.
- `interpolator::AnyInterpolator<T>`: a downcastable counterpart to `Interpolator<T>`
  for storing heterogeneous interpolators behind one trait object.
- `Nested`/`serialize_nested` (serde): opt into nested-array serialization at a specific
  call site instead of crate-wide.
- `strategy::Broadcastable<T>`: shared per-axis config helper (`Broadcast` vs `Each`)
  used by `Step`, `CubicC2`, and `GridTransform`, and reusable by custom strategies.

### Changed
- **Breaking:** owned data types drop their `Owned` suffix (e.g. `Interp1DOwned` ->
  `Interp1D`, `InterpolatorEnumOwned` -> `InterpolatorEnum`); the forms generic over the
  `ndarray` representation gain a `Base` suffix instead (`Interp1D` -> `Interp1DBase`);
  `Viewed` becomes `View`.
- **Breaking:** `LeftNearest`/`RightNearest` removed; use `Step::lower()`/`Step::upper()`.
- **Breaking:** `check_extrapolate` renamed `validate_extrapolate`; `find_nearest_index`
  renamed `locate_lower_index` and moved (with the other index-search helpers) into a
  new `strategy::utils` module.
- **Breaking:** `Strategy1DEnum`/`2DEnum`/`3DEnum`/`NDEnum` are now generic over the
  element type (`Strategy1DEnum<f64>`).
- **Breaking:** `InterpolateError`, `ValidateError`, `Extrapolate<T>`, and the strategy
  enums are now `#[non_exhaustive]`; an exhaustive downstream `match` needs a `_` arm.
- **Breaking:** `Strategy1D`/`2D`/`3D`/`ND::init` is split into a pure `validate` and a
  mutating `init`.
- **Breaking:** grid coordinates must now be strictly increasing (a repeated adjacent
  coordinate previously passed validation and silently divided by zero).
- **Breaking:** interpolation failures (`InterpolateError::PointLength`/`OutOfBounds`)
  carry structured positions instead of prose, and aggregate every failure across a
  batch instead of stopping at the first one.
- **Breaking:** error renames/removals: `ValidateError::Monotonicity` ->
  `NotStrictlyIncreasing`, `ExtrapolateSelection` -> `ExtrapolateUnsupported`,
  `EmptyGrid` removed (folds into `InsufficientGridLength`);
  `InterpolateError::ExtrapolateError` -> `OutOfBounds`.
- **Breaking:** the `serde_ndim` Cargo feature is removed (it silently affected every
  `ninterp` consumer in a binary, not just the opting-in crate); use `Nested` at the
  call site instead. Reading still accepts either wire format.
- **Breaking:** `extrapolate` is now a required field on deserialize.
- **Breaking:** `Interp1D`/`2D`/`3D` gain an inherent `interpolate(&self, point:
  &[D::Elem; N])`, so a wrong-length point is a compile error instead of a runtime one.
  A call site passing a runtime-length slice needs `let point: &[D::Elem; N] =
  slice.try_into()?;`.
- Significant ND performance work: dropped `itertools`, cut per-call allocations from
  O(N·2^N) to O(1)-O(N). 54-64% faster on 1D and up to 63% on 2D multilinear; smaller
  gains (roughly 1-12%) on 3D and on paths that already hit an exact grid point. See
  PR #13.

### Fixed
- `InterpND` panicked on a 0-D grid and on any grid dimension with exactly 1 point;
  both now raise `ValidateError::InsufficientGridLength`/handle the 0-D case cleanly.
- Serde: non-self-describing formats (bincode, postcard, ...) couldn't round-trip an
  interpolator they had just written.

## [0.9.1] - 2026-08-03

### Changed
- **Breaking:** human-readable array output via `serde-ndim` is now opt-in through the
  `serde_ndim` feature, rather than always enabled alongside `serde`.

### Notes
- Repository moved from the `NREL` to `NatLabRockies` GitHub organization (URLs
  redirect automatically).

## [0.9.0] - 2026-08-02

### Added
- `serde-ndim` integration under the existing `serde` feature.
- Compatibility deserializers that accept the new simple array representation, the
  legacy `serde-ndim` representation, or a mix of the two (simple grid + legacy values,
  or vice versa).

### Changed
- **Breaking (serialization output only):** default serialized output for interpolator
  grid/values now uses a simpler, sequence-style representation instead of the prior
  `serde-ndim` format. If you snapshot-test or schema-validate the exact serialized
  payload shape, update your fixtures.

### Notes
- Deserialization remains backward compatible with the prior payload structure, so
  existing persisted data is still readable without migration.

## [0.8.2] - 2026-02-19

### Fixed
- Misleading error message for coordinate validation ([#12], [@meredithdoan]).

## [0.8.1] - 2025-11-25

### Changed
- `ndarray` dependency loosened to `^0.16` for downstream compatibility ([#11],
  [@robfitzgerald]).

## [0.8.0] - 2025-11-15

### Changed
- **Breaking (version bump only):** raised the maximum supported `ndarray` version to
  include the 0.17.x line. Bumped as a new "major" (`0.x`) version specifically so a
  downstream `Cargo.lock` rebuild wouldn't silently pull in the new `ndarray` major
  version without an explicit opt-in.

## [0.7.3] - 2025-05-29

### Added
- `into_owned()` methods.

### Fixed
- Bug in `InterpDataOwned`.

## [0.7.2] - 2025-05-29

### Added
- `view()` method for interpolators.

## [0.7.1] - 2025-05-19

### Changed
- Error types now have a hand-written `Debug` impl that delegates to the
  `thiserror`-derived `Display`, instead of `#[derive(Debug)]`. Unwrapped errors read
  as a message instead of a raw struct dump.

### Notes
- Documentation improvements.

## [0.7.0] - 2025-05-02

### Changed
- `#[serde(untagged)]` applied to all enum types: they now (de)serialize identically to
  their contained variant, so switching a downstream project from a concrete
  interpolator type to `InterpolatorEnum` doesn't change the serialized shape.
- Strategies now serialize to their stringified name instead of `null`.

## [0.6.4] - 2025-04-22

### Changed
- Serde deserialize bounds changed from `DeserializeOwned` to `Deserialize<'de>`.
- Minor syntax cleanup; removed some unnecessary allocations.

## [0.6.3] - 2025-03-24

### Changed
- `set_extrapolate` moved onto the `Interpolator` trait.

## [0.6.2] - 2025-03-21

### Fixed
- `PartialEq` impls: `#[derive(PartialEq)]` doesn't work for types with a `D: Data`
  bound, since `ndarray::Data` itself doesn't implement `PartialEq` even though
  `ArrayBase<D, _>` does. Switched to manual impls.

### Changed
- Owned and viewed type aliases now exposed in `prelude`.

## [0.6.1] - 2025-03-20

### Added
- Strategy and interpolator enums (`Strategy1DEnum`/etc., `InterpolatorEnum`), enabling
  `serde` support for runtime-swappable interpolators and strategies.

## [0.6.0] - 2025-03-19

### Changed
- **Breaking:** namespace reorganized. Strategies are now accessed as `strategy::Linear`
  etc. after `use prelude::*`, instead of being re-exported flat at the top level. This
  makes room for more complex strategy organization (e.g. cubic strategies) without
  polluting the downstream namespace.

### Added
- Strategy `init` step, letting a strategy mutate/precompute its own internal state
  ahead of interpolation calls, enabling more complex strategies.
- `Extrapolate::Wrap`: wrap around to the other end of periodic data.

## [0.5.2] - 2025-03-12

### Added
- `Clone` now derived for all public types ([#4], [@kylecarow]).

## [0.5.1] - 2025-03-09

### Changed
- Extrapolation handling moved into the macro-generated impls; a separate `extrapolate`
  call is no longer necessary, and it's no longer incorrectly applicable to `Interp0D`.

## [0.5.0] - 2025-03-08

### Changed
- **Breaking:** whole-crate rewrite onto generics, operating directly on `ndarray` data
  (owned and viewed) instead of a fixed internal representation ([#3], "NDArray &
  generics rewrite").

## [0.4.0] - 2025-03-07

### Changed
- **Breaking:** full rewrite ([#2]). Introduced custom strategies via
  `Strategy1D`/`Strategy2D`/`Strategy3D` traits (renamed from `Interp1DStrategy`/etc.),
  added `set_strategy`, reorganized modules into per-dimensionality folders, and
  removed the old `Interpolator` enum in favor of the concrete-type design used since.

## [0.3.0] - 2025-03-03

### Changed
- **Breaking:** error types renamed for clarity: `ValidationError` -> `ValidateError`,
  `InterpolationError` -> `InterpolateError`,
  `InterpolationError::ExtrapolationError` -> `InterpolateError::ExtrapolateError`,
  `Error::NoSuchField` now carries a `&'static str`.
- **Breaking:** `Extrapolate::FillValue` renamed to `Extrapolate::Fill`.

## [0.2.7] - 2025-03-03

### Added
- `Extrapolate::FillValue(f64)`.

### Fixed
- `Interp3D` clamp bug.

## [0.2.6] - 2025-03-01

### Added
- `Extrapolate::Enable` support for `Linear` across all dimensionalities.
- `Nearest` strategy available for all dimensionalities.

## [0.2.5] - 2025-02-25

### Changed
- Minor internal cleanup (removed an unnecessary `return`) and documentation cleanup.

## [0.2.4] - 2025-02-21

### Added
- `prelude` module to simplify downstream imports.

## [0.2.3] - 2025-02-21

### Changed
- `Clone` now derived on relevant types, per a downstream request from the FASTSim
  team.

### Notes
- Minor internal/CI polish.

## [0.2.2] - 2025-02-21

### Changed
- `new_*d` constructor methods now return `crate::Error`.

## [0.2.1] - 2025-01-24

### Changed
- Extrapolation error messages improved to include every out-of-bounds grid dimension,
  not just the first.

## [0.2.0] - 2025-01-23

### Changed
- **Breaking:** instantiation moved to dimensionality-specific `new_1d`/`new_2d`/
  `new_3d`/`new_nd` methods.

## [0.1.0] - 2024-11-27

Initial release.

[#2]: https://github.com/NatLabRockies/ninterp/pull/2
[#3]: https://github.com/NatLabRockies/ninterp/pull/3
[#4]: https://github.com/NatLabRockies/ninterp/pull/4
[#11]: https://github.com/NatLabRockies/ninterp/pull/11
[#12]: https://github.com/NatLabRockies/ninterp/pull/12
[@kylecarow]: https://github.com/kylecarow
[@robfitzgerald]: https://github.com/robfitzgerald
[@meredithdoan]: https://github.com/meredithdoan

[Unreleased]: https://github.com/NatLabRockies/ninterp/compare/v0.10.0...main
[0.10.0]: https://github.com/NatLabRockies/ninterp/compare/v0.9.1...v0.10.0
[0.9.1]: https://github.com/NatLabRockies/ninterp/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/NatLabRockies/ninterp/compare/v0.8.2...v0.9.0
[0.8.2]: https://github.com/NatLabRockies/ninterp/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/NatLabRockies/ninterp/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/NatLabRockies/ninterp/compare/v0.7.3...v0.8.0
[0.7.3]: https://github.com/NatLabRockies/ninterp/compare/v0.7.2...v0.7.3
[0.7.2]: https://github.com/NatLabRockies/ninterp/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/NatLabRockies/ninterp/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/NatLabRockies/ninterp/compare/v0.6.4...v0.7.0
[0.6.4]: https://github.com/NatLabRockies/ninterp/compare/v0.6.3...v0.6.4
[0.6.3]: https://github.com/NatLabRockies/ninterp/compare/v0.6.2...v0.6.3
[0.6.2]: https://github.com/NatLabRockies/ninterp/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/NatLabRockies/ninterp/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/NatLabRockies/ninterp/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/NatLabRockies/ninterp/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/NatLabRockies/ninterp/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/NatLabRockies/ninterp/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/NatLabRockies/ninterp/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/NatLabRockies/ninterp/compare/v0.2.7...v0.3.0
[0.2.7]: https://github.com/NatLabRockies/ninterp/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/NatLabRockies/ninterp/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/NatLabRockies/ninterp/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/NatLabRockies/ninterp/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/NatLabRockies/ninterp/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/NatLabRockies/ninterp/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/NatLabRockies/ninterp/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/NatLabRockies/ninterp/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/NatLabRockies/ninterp/releases/tag/v0.1.0