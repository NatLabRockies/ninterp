# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project is pre-1.0 and follows the common pre-1.0 convention: breaking changes
bump the minor version (`0.x` -> `0.(x+1)`), other changes bump the patch version.

## [Unreleased]

Everything below is merged to `main` but not yet tagged/released.

### Added
- `strategy::LinearUniform`: an O(1)-index alternative to `Linear` for uniformly-spaced
  grids (1D/2D/3D/ND). Validates grid uniformity at construction/`init` time (1024 * ε
  relative tolerance) instead of silently falling back to a search.
- `strategy::Step`: a parameterized step (piecewise-constant) strategy, replacing
  `LeftNearest`/`RightNearest` with a single strategy that works across all
  dimensionalities. A single `StepDirection` broadcasts to every axis, or a `Vec` gives
  per-axis control.
- `strategy::StepLower` / `strategy::StepUpper`: zero-sized marker strategies for the
  common fixed-direction case, avoiding `Step`'s direction-vector allocation and
  per-call direction matching. `Step` remains the choice for mixed per-dimension or
  runtime-selected direction.
- `strategy::LinearUniform`, `Step`, `StepLower`, and `StepUpper` are all available for
  every dimensionality and included in the corresponding `Strategy*Enum` types.
- `Nested` wrapper / `serialize_nested` helper / `SerializeNested` trait (`prelude`, behind
  the `serde` feature): opt into the nested-array format at a specific serialization call
  site, e.g. `serde_json::to_string(&Nested(&interp))` or
  `#[serde(serialize_with = "serialize_nested")]` on a field. Falls back to the `ndarray`
  format on non-`is_human_readable` (binary) serializers, since there's nothing to nest
  there and those formats can't read it back anyway.
- `strategy::utils::exact_index`, `locate_step_index`, `locate_lower_index_uniform`,
  `check_uniform_grid`, and `AxisLocation`/`locate_axis` are now `pub` (previously
  `pub(crate)`). They're the same per-axis primitives `Linear`/`LinearUniform`/`Step`/
  `StepLower`/`StepUpper` are built from, now reusable from custom strategies instead of
  needing to be reimplemented. `check_uniform_grid`'s error message no longer hardcodes
  `"LinearUniform:"`, since other strategies can call it directly now too.
- `interpolate_fast` on `Strategy1D`/`2D`/`3D`/`ND` and `Interpolator<T>` (both
  default-provided, non-breaking): a `Result`-free interpolation path for hot loops
  where the caller has already checked the point is in-bounds and knows extrapolation
  isn't needed. `Interp0D`/`1D`/`2D`/`3D`/`ND`, `InterpolatorEnum`, and
  `Box<dyn Interpolator<T>>` all override it with the real skip-checks path;
  `Interp1D`/`2D`/`3D` additionally get an inherent `interpolate_fast(&self, &[D::Elem; N])`
  taking the point as a fixed-size array, so the point length is guaranteed by the type
  system instead of checked at runtime.
- `InterpolatorEnum` gains `check_extrapolate`/`validate_strategy`/`init_strategy`
  forwarding to the current variant, matching what `Interp1D`/`2D`/`3D`/`ND` already
  expose as public inherent methods.

### Changed
- **Breaking:** `Strategy1D`/`Strategy2D`/`Strategy3D`/`StrategyND::init` is split into
  a pure `validate(&self, data)` and a mutating `init(&mut self, data)`, both
  default-no-op (non-breaking for existing custom strategy implementations, which keep
  compiling unchanged). `validate` is for invariant checks that don't need precomputed
  state (`LinearUniform`'s grid-uniformity check and `Step`'s direction-count check
  both moved from `init` to `validate`); `init` stays reserved for real precomputation.
  `new` and `set_strategy` call both; `Interpolator::validate` now also calls
  `validate_strategy` (see below), so invariant violations like a non-uniform
  `LinearUniform` grid are caught there too, not just via `init_strategy`.
- Each `Interp1D`/`Interp2D`/`Interp3D`/`InterpND` gains a public `validate_strategy()`,
  mirroring the existing `init_strategy()`: re-runs the strategy's `validate` against
  the current data, for use after mutating the public `data`/`strategy` fields
  directly.
- **Breaking:** `find_nearest_index` is renamed to `locate_lower_index` and, along with
  the other grid/index search helpers (`step_index` -> `locate_step_index`,
  `uniform_lower_index` -> `locate_lower_index_uniform`, `exact_index`,
  `check_uniform_grid`), moves from `strategy::traits` to a new `strategy::utils`
  module — `traits` now holds only the `Strategy1D`/`2D`/`3D`/`ND` trait definitions.
  No deprecation shim, matching the other breaking renames in this release.
  `locate_lower_index` also now clamps out-of-range points to `[0, len - 2]` itself,
  rather than relying on each `Linear` call site to inline the same clamp before
  calling it.
- **Breaking:** `LeftNearest` and `RightNearest` are removed. Migrate to
  `Step::from(StepDirection::Lower)` / `Step::from(StepDirection::Upper)`, or the
  leaner `StepLower` / `StepUpper` markers.
- **Breaking:** `Linear` and `LinearUniform` now require `D::Elem: Float` (previously
  `Num + PartialOrd`). Other strategies (`Nearest`, `Step`, etc.) keep looser numeric
  bounds after an initial, overly broad `Float` restriction across the whole strategy
  surface was narrowed back down to just the two strategies that actually need it.
- **Breaking:** `ValidateError` variants renamed for consistency, and no longer read as
  full sentences: `ExtrapolateSelection` -> `InvalidExtrapolate`, `Monotonicity` ->
  `NonMonotonic`. `EmptyGrid` is removed outright; a grid dimension with 0 or 1 points
  is now rejected by the same `InsufficientGridPoints`, since a single point can't
  bracket a query either.
- **Breaking:** the `serde_ndim` Cargo feature is removed. It switched the nested-array
  write format on for every array field crate-wide, and because Cargo features are
  additive and unify across the dependency graph, enabling it anywhere in a binary
  silently flipped the wire format for every other `ninterp` consumer in that binary too.
  Migrate to wrapping values in `Nested` (or `serialize_with = "serialize_nested"` on a
  field) at the specific call site that wants it. Reading already accepted either format
  and continues to, so data written by prior versions still loads fine.
- Significant ND performance work: `Linear`/`Nearest` no longer build coordinate
  permutation tables via `itertools::multi_cartesian_product` (removing the `itertools`
  dependency); corner values are now gathered into a flat buffer and reduced with an
  in-place bitmask/butterfly pass, cutting allocations from O(N * 2^N) to O(1)-O(N).
  1D strategies no longer open with an O(M) linear scan for exact grid-point matches.
  2D/3D `Linear` now short-circuits per-dimension when a query point lands exactly on a
  grid coordinate. See PR #13 for benchmark numbers (roughly 50-65% faster on 1D/2D
  hardcoded and multilinear paths).
- Serde: `StepLower`/`StepUpper` accept the legacy `"LeftNearest"`/`"RightNearest"`
  names on deserialization for backward compatibility; `Step`'s own wire format
  (`{"Step": [...]}`) is unchanged and does not accept those aliases.
- Loosened `Step`/`Nearest` strategy trait bounds back down after the `LinearUniform`
  work had temporarily tightened them further than necessary.
- Loosened `InterpolatorEnum`'s `PartialEq`/`SerializeNested` impls off `D::Elem: Float`
  down to `PartialEq + Debug` (`+ Serialize`), matching what those impls actually need:
  they only compare/serialize fields and never touch the strategy trait. Construction and
  interpolation still require `Float`, unchanged.
- **Breaking:** `extrapolate` is now a required field on deserialize instead of silently
  defaulting to `Extrapolate::Error` when omitted. `data` and `strategy` were already
  required; this was the only field with that leniency, and nothing pinned the behavior
  down, so a missing field now fails loudly (`missing field "extrapolate"`) instead of
  risking a silently different interpolator than intended.
- Various documentation and README improvements; CI workflow polish.

### Fixed
- `InterpND` panicked on `n == 0` (the 0-D-via-`InterpND` case, e.g. after
  dimensionality reduction collapses every axis). The ND `Linear` strategy's exact-match
  scan was rewritten from `iter().position()` (which returned `None` on an empty grid
  dimension without touching it) to `find_nearest_index`, whose binary search calls
  `.first().unwrap()` and panics on an empty dimension. Fixed by skipping empty grid
  dimensions in that loop.
- A grid dimension with exactly 1 point passed construction and then panicked on the
  first `interpolate` call (integer underflow in the lower-bracket search, or an
  out-of-bounds index right after it), for every strategy except when
  `Extrapolate::Enable` was selected, since the "at least 2 points" check only ran for
  that one setting. It's now checked unconditionally at construction, so this is a
  `ValidateError::InsufficientGridPoints` instead of a panic.
- Serde: non-self-describing formats (bincode, postcard, ...) could never actually read
  back an interpolator they had just written, in either feature configuration.
  `deserialize_any` was called unconditionally, which those formats don't support at all
  (`Bincode does not support the serde::Deserializer::deserialize_any method`); separately,
  a fixed-size grid (`[ArrayBase<D, Ix1>; N]`) serializes as a tuple, which those formats
  encode without a length prefix, so reading it back as a seq desynchronized the byte
  stream (`unknown array version: 0`). Both are now gated on `is_human_readable()`, falling
  back to `ndarray`'s own (de)serialization for non-human-readable formats.

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

[Unreleased]: https://github.com/NatLabRockies/ninterp/compare/v0.9.1...main
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