# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project is pre-1.0 and follows the common pre-1.0 convention: breaking changes
bump the minor version (`0.x` -> `0.(x+1)`), other changes bump the patch version.

## [Unreleased]

Everything below is merged to `main` but not yet tagged/released.

### Added
- `strategy::LinearUniform`: an O(1)-index alternative to `Linear` for uniformly-spaced
  grids (1D/2D/3D/ND), validated at construction/`init` time.
- `strategy::broadcast::Broadcastable<T>`: `Broadcast(T)` applies one value to every
  grid dimension, `Each(Vec<T>)` gives one value per dimension. Shared by `Step` and
  `CubicC2` for their per-axis configuration; indexes via `Index<usize>` (panics like a
  slice if `dim` is out of range) and validates via the public `validate_len`, which
  reports a mismatched count as `ValidateError::PerAxisLen { label, noun, ndim, found }`.
  Both are usable by custom strategies built on `Broadcastable`.
- `strategy::Step`: a parameterized step (piecewise-constant) strategy across all
  dimensionalities, replacing `LeftNearest`/`RightNearest`. Its `directions` field is a
  `Broadcastable<StepDirection>`: `Broadcast` applies one direction to every axis, `Each`
  gives per-axis control. `Step::lower()`/`Step::upper()` construct the common broadcast
  cases; `Step::new(Vec<StepDirection>)` constructs the per-axis case.
- `LinearUniform` and `Step` are available for every dimensionality and included in the
  `Strategy*Enum` types.
- `strategy::CubicC2`: a C² piecewise cubic spline strategy (1D/2D/3D/ND) with
  configurable boundary conditions given via
  `boundary_conditions: Broadcastable<CubicC2BoundaryConditions<T>>` (`Broadcast` for one
  condition on every axis, `Each` for one per axis), a single tridiagonal solve per axis.
  `CubicC2BoundaryConditions` is either `Periodic`, or `Endpoints { lower, upper }` where
  `lower`/`upper` are independent `CubicC2Endpoint`s (`NotAKnot`, `FirstDerivative(T)`
  "clamped", or `SecondDerivative(T)`, zero being the classic "natural" case); the two
  ends of an axis can mix types, e.g. `NotAKnot` on one side and a specific derivative on
  the other. A single-`NotAKnot` axis needs 3 grid points; 4 if both ends are `NotAKnot`.
  `not_a_knot()`/`natural()`/`clamped(lower, upper)`/`periodic()` construct the common
  symmetric cases; `CubicC2BoundaryConditions::{not_a_knot, first_derivative,
  second_derivative}` are the per-condition constructors mixed endpoints are built from.
  `Strategy1D` caches its second-derivative coefficients; `Strategy2D`/`3D`/`ND` cache
  the full corner-derivative tensor for O(1) queries.
  `From<CubicC2BoundaryConditions<T>>` broadcasts a condition obtained generically (e.g.
  from runtime config) without matching on it first. Included in the `Strategy*Enum`
  types. With the `serde` feature, serializes keyed by `"CubicC2"` like the other
  built-in strategies; `NotAKnot`/`Natural`/`Periodic` (symmetric, no explicit value)
  serialize as bare strings, anything else (an explicit value, or mixed endpoint types)
  as the general `{"Endpoints": {"lower": ..., "upper": ...}}` form.
- `Nested` wrapper / `serialize_nested` helper / `SerializeNested` trait (`prelude`,
  `serde` feature): opt into the nested-array serialization format at a specific call
  site, e.g. `serde_json::to_string(&Nested(&interp))` or
  `#[serde(serialize_with = "serialize_nested")]` on a field. Falls back to the `ndarray`
  format for non-human-readable (binary) serializers.
- `strategy::utils` per-axis primitives (`exact_index`, `locate_step_index`,
  `locate_lower_index_uniform`, `validate_uniform_grid`/`validate_uniform_grid_epsilon`,
  `AxisLocation`/`locate_axis`) are now `pub`, for reuse by custom strategies.
  `validate_uniform_grid_epsilon` is `Float`-bound and takes a `tolerance: Option<T>`
  (`None` keeps the default of 1024 × ε); `validate_uniform_grid` is its generic core,
  taking an explicit `tolerance: T` under just `Copy + PartialOrd + Sub`, for grids over
  non-`Float` types. Both raise the dedicated `ValidateError::NonUniform { dim, index }`
  on failure.
- `interpolate_fast` on `Strategy1D`/`2D`/`3D`/`ND` and `Interpolator<T>`: a `Result`-free
  interpolation path (panics instead of erroring) for hot loops where the point is
  already known to be in-bounds. `Interp1D`/`2D`/`3D` also get an inherent
  `interpolate_fast(&self, &[D::Elem; N])` taking the point as a fixed-size array.
- `InterpolatorEnum` gains `validate_extrapolate`/`validate_strategy`/`init_strategy`,
  matching what `Interp1D`/`2D`/`3D`/`ND` already expose.
- `batch_interpolate`/`batch_interpolate_fast` on `Interpolator<T>`,
  `Strategy1D`/`2D`/`3D`/`ND`, and the concrete interpolator types: interpolate several
  points in one call, resolving `self.extrapolate` once and funneling every point into
  at most one strategy call instead of one `interpolate` call per point. For a
  `Box<dyn Interpolator<T>>`/`Box<dyn Strategy*D>`, this also collapses what would be one
  virtual dispatch per point into a single dispatch. Under `Extrapolate::Error`, every
  out-of-range point is reported in one `InterpolateError::OutOfBounds`, not just the
  first.
- `batch_interpolate_into`/`batch_interpolate_fast_into`: allocation-free counterparts
  that write into a caller-supplied output slice instead of returning a `Vec`.
  `batch_interpolate`/`batch_interpolate_fast` now defer to these internally. New
  `InterpolateError::OutputLength` variant for a slice/point-count length mismatch.
- `interpolator::AnyInterpolator<T>: Interpolator<T> + Send + Sync` (not in the
  `prelude`): a downcastable counterpart to `Interpolator<T>` for storing heterogeneous
  interpolators behind `Box<dyn AnyInterpolator<T>>` and recovering the concrete type via
  `as_any`. Implemented for owned `Interp1D`/`2D`/`3D`/`ND` only, since `as_any` requires
  `Self: 'static`; viewed interpolators can still be used through `Interpolator<T>`.
- `Strategy1D`/`2D`/`3D`/`ND::interpolate_wrapped`: resolves an `Extrapolate::Wrap`
  query point, then interpolates. Default reproduces existing behavior (wraps in the
  grid's own coordinate space) for every built-in strategy; a strategy whose working
  coordinate space differs from the grid's can override it. Groundwork for #56.
- `strategy::Transform`, `strategy::GridTransform<T, S>`, `strategy::ValuesTransform<T,
  S>`: composable wrappers for interpolating in a transformed coordinate/value space
  (`Log`, `Sqrt`, `Reciprocal`, or `Identity`) instead of the raw one: standard for data
  spanning many orders of magnitude or following a known nonlinear relationship, and for
  bounding interpolated output (e.g. always-positive via `ValuesTransform::log`).
  `GridTransform`'s `axes: Broadcastable<Transform>` gives one transform per grid
  dimension or one broadcast to all; `ValuesTransform::transform` applies to the whole
  values array. Both wrap any `Strategy1D`/`2D`/`3D`/`ND` `inner` and compose by nesting,
  e.g. `ValuesTransform::log(GridTransform::log(CubicC2::not_a_knot()))` for full log-log
  interpolation. `log`/`sqrt`/`reciprocal`/`broadcast`/`new` constructors mirror
  `CubicC2`'s. Included in the `Strategy*Enum` types, with `inner` boxed to break the
  self-referential size (`Box` around the concrete enum, not `dyn Trait`, so still
  serde-compatible); `Strategy1D`/`2D`/`3D`/`ND` also gain a `Box<Strategy*DEnum<T>>`
  forwarding impl for this. New `ValidateError::GridTransformDomain`/
  `ValidateError::GridTransformNotMonotonic`/`ValidateError::ValuesTransformDomain`
  variants for a grid coordinate outside a transform's domain (e.g. `Log` requires `x >
  0`), a raw grid that's non-monotonic once transformed (possible even when every
  coordinate individually passes the domain check, since a transform's domain can be
  disconnected, e.g. `Reciprocal`'s `x != 0`), or a data value outside a transform's
  domain. New `InterpolateError::GridTransformDomain(Vec<OutsideDomainAt>)` for a query
  point outside a transform's domain under `Extrapolate::Enable`, catching what would
  otherwise silently `ln()` into `NaN`; one entry per offending point coordinate,
  aggregated across a whole `batch_interpolate`/`batch_interpolate_into` call rather than
  erroring on the first one, mirroring `InterpolateError::OutOfBounds`. Closes #56.
- `strategy_enum_impl!`'s generated `Strategy*DEnum` impl now forwards every
  `Strategy1D`/`2D`/`3D`/`ND` method (previously only `validate`/`init`/`interpolate`/
  `allow_extrapolate`), so a strategy overriding `interpolate_wrapped` or the batch
  methods dispatches correctly when nested inside the enum instead of silently falling
  back to the trait default.

### Changed
- **Breaking:** owned data is now the default in type names, following Rust idioms
  (`String` vs `&str`, `Vec<T>` vs `&[T]`). Every form generic over the `ndarray` data
  representation gains a `Base` suffix, the unsuffixed name becomes the owned alias, and
  `Viewed` becomes `View`:
  - `Interp{1D,2D,3D,ND}` -> `Interp{1D,2D,3D,ND}Base`, `InterpData` -> `InterpDataBase`,
    `InterpDataND` -> `InterpDataNDBase`, `InterpolatorEnum` -> `InterpolatorEnumBase`
    (generic forms). `Interp0D<T>` is unaffected; it was never generic over the
    representation.
  - `Interp{1D,2D,3D,ND}Owned`/`InterpData{1D,2D,3D,ND}Owned`/`InterpolatorEnumOwned` drop
    the `Owned` suffix entirely (they're now the plain, unsuffixed names).
  - `Interp{1D,2D,3D,ND}Viewed`/`InterpData{1D,2D,3D,ND}Viewed`/`InterpolatorEnumViewed`
    become `...View`.
  - `InterpDataBase` is now also re-exported from `ninterp::interpolator` (previously
    only `ninterp::data`).
  - **Migrating generic-over-`D` code:** custom strategy impls and anything else written
    against the representation parameter must move to the `Base` names. Keeping the old
    unsuffixed name is a compile error, though an indirect one: `InterpData1D<D>` now
    means `InterpDataBase<OwnedRepr<D>, 1>`, so the failure surfaces as an unsatisfied
    `D::Elem: PartialEq + Debug` bound rather than a clean "not found".
- **Breaking:** `check_extrapolate` renamed to `validate_extrapolate`, matching the
  `validate`/`validate_strategy` naming used elsewhere.
- **Breaking:** `InterpolateError`, `ValidateError`, `Extrapolate<T>`, and
  `Strategy1DEnum`/`2DEnum`/`3DEnum`/`NDEnum` are now `#[non_exhaustive]`. An existing
  exhaustive downstream `match` over any of these needs a `_` arm; construction is
  unaffected.
- **Breaking:** `Strategy1DEnum`/`2DEnum`/`3DEnum`/`NDEnum` are now generic over the
  element type (`Strategy1DEnum<T>` etc.), needed to hold `CubicC2<T>` as a variant.
  Bare usages (e.g. `Interp1D<_, Strategy1DEnum>`) need a type argument:
  `Interp1D<_, Strategy1DEnum<f64>>`.
- **Breaking:** `Strategy1D`/`2D`/`3D`/`ND::init` is split into a pure
  `validate(&self, data)` and a mutating `init(&mut self, data)`, both default no-op
  (existing custom strategies keep compiling unchanged). `LinearUniform`'s uniform-grid
  check and `Step`'s direction-count check moved from `init` to `validate`. `new` and
  `set_strategy` call both; `Interpolator::validate` now also calls `validate_strategy`.
- **Breaking:** `find_nearest_index` renamed to `locate_lower_index` (now also clamps
  out-of-range points to `[0, len - 2]` itself); it and the other index-search helpers
  (`step_index` -> `locate_step_index`, `uniform_lower_index` ->
  `locate_lower_index_uniform`, `exact_index`, `validate_uniform_grid`) move from
  `strategy::traits` to a new `strategy::utils` module.
- **Breaking:** `LeftNearest`/`RightNearest` removed. Migrate to `Step::lower()`/
  `Step::upper()`.
- **Breaking:** `Linear`/`LinearUniform` now require `D::Elem: Float` (previously
  `Num + PartialOrd`); other strategies keep looser numeric bounds.
- **Breaking:** interpolation-time failures carry structured positions instead of prose,
  and aggregate across a batch instead of stopping at the first failure:
  - `InterpolateError::PointLength(usize)` ->
    `PointLength { expected: usize, failures: Vec<WrongLengthAt> }`.
  - `InterpolateError::OutOfBounds(String)` -> `OutOfBounds(Vec<OutOfBoundsAt>)`; the
    offending coordinate/bounds are no longer echoed into the message (they're
    `D::Elem`, which the error isn't generic over); index the `points`/`grid` you
    already have instead.
  - Both entry types are `#[non_exhaustive]` structs rather than tuples; read by field
    (`at.index`, `at.dim`, `at.found`).
  - Both variants render a lone failure as a sentence and several as an indented list,
    omitting point indices for single-point calls.
- **Breaking:** `ValidateError::ExtrapolateSelection` becomes the payload-free
  `ExtrapolateUnsupported`.
- New `ValidateError::GridAxisCount { expected: usize, found: usize }`, for an
  `InterpDataND` whose grid axis count doesn't match its values' dimensionality
  (previously fell through to `ValidateError::Other(String)`).
- **Breaking:** `ValidateError::Monotonicity` -> `NotStrictlyIncreasing`;
  `InterpolateError::ExtrapolateError` -> `OutOfBounds` (message rewritten to
  ``point out of bounds with `Extrapolate::Error` set``). `ValidateError::EmptyGrid` is
  removed; a grid dimension with 0 or 1 points is now `InsufficientGridPoints`.
- **Breaking:** grid coordinates must now be strictly increasing; a repeated adjacent
  coordinate previously passed validation and gave a zero-width interval, dividing by
  zero (silent NaN/Inf) in any strategy that computes a fractional position or slope
  across it, instead of the `NotStrictlyIncreasing` error it now raises.
- **Breaking:** the `serde_ndim` Cargo feature is removed. It switched the nested-array
  write format on for every array field crate-wide, and because Cargo features are
  additive, enabling it anywhere in a binary silently flipped the wire format for every
  other `ninterp` consumer in that binary too. Migrate to wrapping values in `Nested` (or
  `serialize_with = "serialize_nested"`) at the specific call site that wants it. Reading
  still accepts either format.
- **Breaking:** `extrapolate` is now a required field on deserialize instead of silently
  defaulting to `Extrapolate::Error` when omitted, matching `data`/`strategy`.
- **Breaking:** `Interp1D`/`2D`/`3D` gain an inherent
  `interpolate(&self, point: &[D::Elem; N])`, so a wrong-length point is a compile error
  instead of a runtime `InterpolateError::PointLength`. Because inherent methods shadow
  trait methods, a call site holding a genuine runtime-length slice (not an array
  literal) for a concretely-typed `Interp1D`/`2D`/`3D` stops compiling; migrate with
  `let point: &[D::Elem; N] = slice.try_into()?;`. `Interpolator::interpolate(&self,
  point: &[T])` (used by `InterpND`, `InterpolatorEnum`, `Box<dyn Interpolator<T>>`) is
  unaffected.
- `Interp1D`/`2D`/`3D`/`InterpND` gain a public `validate_strategy()`, mirroring
  `init_strategy()`, for use after mutating the public `data`/`strategy` fields directly.
- Significant ND performance work: `Linear`/`Nearest` no longer build coordinate
  permutation tables via `itertools::multi_cartesian_product` (dropping the `itertools`
  dependency); corner values are gathered into a flat buffer and reduced via a
  bitmask/butterfly pass, cutting allocations from O(N·2^N) to O(1)-O(N). 1D strategies
  no longer linear-scan for exact grid-point matches; 2D/3D `Linear` short-circuits
  per-dimension on an exact match. Roughly 50-65% faster on 1D/2D hardcoded and
  multilinear paths (see PR #13).
- Serde: `Step` accepts the legacy bare `"LeftNearest"`/`"RightNearest"` strings (from the
  removed unit structs of the same name) on deserialization, in addition to its own wire
  format (`{"Step": "Lower"}` for `Broadcast`, `{"Step": ["Lower", "Upper"]}` for `Axes`).
- Various documentation and README improvements; CI workflow polish.

### Fixed
- `InterpND` panicked on `n == 0` (the 0-D-via-`InterpND` case, e.g. after
  dimensionality reduction collapses every axis): the ND `Linear` strategy's exact-match
  scan now skips empty grid dimensions instead of calling into a binary search that
  panics on one.
- A grid dimension with exactly 1 point passed construction and then panicked on the
  first `interpolate` call, for every strategy except `Extrapolate::Enable` (the only
  setting that ran the "at least 2 points" check). Now checked unconditionally at
  construction, raising `ValidateError::InsufficientGridPoints` instead.
- Serde: non-self-describing formats (bincode, postcard, ...) could never actually read
  back an interpolator they had just written: `deserialize_any` was called
  unconditionally, which those formats don't support at all, and a fixed-size grid
  serializes as a tuple, which those formats encode without a length prefix, so reading
  it back as a seq desynchronized the byte stream. Both paths are now gated on
  `is_human_readable()`, falling back to `ndarray`'s own (de)serialization for
  non-human-readable formats.

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