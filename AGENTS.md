# Agent guidelines

Guidance for coding agents (and people) working in this repository. The
[README](README.md) says what the library does; [DESIGN.md](DESIGN.md) says why
it is built this way. Read DESIGN.md's "Where to start" table before changing
evaluation code.

## The one rule

Every change must keep the complete `precision`, `recall` and `scores` arrays
**bit-identical** to pycocotools, not just the rounded AP. Speed and memory come
second. In particular:

- Do not reassociate floating-point arithmetic, reorder divisions, or enable
  fast-math, SIMD or FMA that changes results
  ([DESIGN.md, rule 1](DESIGN.md#rule-1-do-not-reassociate-floating-point-arithmetic)).
- Keep every sort stable (`sort_by`, never `sort_unstable` on keyed data;
  [rule 2](DESIGN.md#rule-2-preserve-stable-ordering)).
- Take IoU/recall threshold grids from the caller
  ([rule 3](DESIGN.md#rule-3-the-caller-constructs-the-grids)).
- Parsing must give the same `f64` as serde_json with `float_roundtrip`, and the
  same errors or fallbacks for invalid input.

When an exact claim rests on reasoning (for example "this skip cannot change a
match"), write the argument in a comment and add a test that compares against
the previous implementation or pycocotools.

## Layout

| Path | Contents |
| --- | --- |
| `rust/ufcoco-core` | Evaluation engine and RLE code; no Python dependency |
| `rust/ufcoco-py` | PyO3 bindings, JSON loading, compact file inputs (`compact.rs`) |
| `python/ultrafast_pycocotools` | `COCO`, `COCOeval`, `mask` Python API |
| `tests/` | Python parity, API, determinism and integration tests |
| `bench/` | Benchmark and profiling scripts; `bench/results/` holds condensed evidence |
| `docs/` | Reports (one per change or measurement) and [docs/benchmarks](docs/benchmarks/README.md) index |

## Build and test

```bash
python -m venv .venv && . .venv/bin/activate
pip install maturin
maturin develop --release              # add --features alloc-stats for allocation counters
pip install -e ".[test,lvis-test]"
python bench/fetch_lvis_fixture.py     # otherwise the LVIS tests skip
python -m pytest -q
cargo test -p ufcoco-core --release
PYO3_PYTHON=$(which python) cargo test -p ufcoco-py --release --test pose_numbers
```

`cargo` commands on `ufcoco-py` need a Python interpreter (`PYO3_PYTHON` or an
active virtual environment). If both `VIRTUAL_ENV` and `CONDA_PREFIX` are set,
maturin refuses to run; unset one.

Tests needing full COCO val2017 skip unless `bench/data/` contains
`instances_val2017.json`, `person_keypoints_val2017.json` and the output of
`bench/make_dets.py`. `tests/test_rfdetr_integration.py` and
`tests/test_ultralytics_integration.py` skip without those frameworks; CI runs
them in [upstream-integrations.yml](.github/workflows/upstream-integrations.yml).

## Performance work

- Profile before optimizing: `bench/profile_tasks.py` (per-phase wall time,
  CPU, RSS and, with `alloc-stats`, Rust allocations) and
  `bench/profile_engine.py` (engine timers).
- Compare builds made the same way (`maturin build --release`, same flags) in
  fresh processes; report medians, CPU time next to wall time, and results at
  one and two threads ([DESIGN.md, measurement methods](DESIGN.md#measurement-methods)).
- Check exactness on real data, not only unit tests: complete arrays against a
  saved pycocotools oracle for bbox, segm and keypoints.
- State what got worse or did not move. Record commits, host and input hashes
  in a `docs/*.md` report and link it from `docs/benchmarks/README.md`.

## Adding or upgrading an external dependency

Treat every new crate or package as code that runs on users' machines and in
CI. This repository was hardened after the
[2026-08-20 crates.io supply-chain attack](https://blog.rust-lang.org/2026/08/20/supply-chain-attack-on-arrayref/)
(#14). Before adding or upgrading a Rust crate (including a new feature of an
existing one) or a Python runtime dependency:

1. **Need.** Prefer the standard library or an existing dependency. Show with a
   measurement that the dependency is worth it; a speedup claim needs numbers
   from this repository's benchmarks.
2. **Identity.** Confirm the exact crate name (watch for typosquats such as
   `serde-json` vs `serde_json`), that the crates.io page links the expected
   repository, and who the owners are. Be wary of recent ownership changes, a
   first release, or a sudden version jump.
3. **Advisories.** Search the [RustSec advisory database](https://rustsec.org/)
   for the crate and run the repository gate after changing `Cargo.lock`:
   `cargo deny --all-features --workspace --locked check advisories`
   (the same check as [cargo-deny.yml](.github/workflows/cargo-deny.yml)).
   It must pass with no new ignore entries.
4. **License.** Compatible with BSD-2-Clause distribution in wheels: MIT,
   Apache-2.0, BSD, ISC, Zlib or Unlicense. `deny.toml` does not check licenses,
   so check the crate and every new transitive crate by hand
   (`cargo tree -e normal --prefix none -f '{p} {l}'`).
5. **Transitive footprint.** Review everything the change adds to `Cargo.lock`
   (`git diff Cargo.lock`, `cargo tree -i <crate>`). Disable default features you
   do not use. Prefer crates with no build script and no proc macros; a
   `build.rs` or proc macro runs arbitrary code at build time, so read it.
6. **Code.** Skim the source you will actually call, especially `unsafe` blocks,
   and any network, file-system or environment access. Numeric crates used in
   exact paths need their own bit-for-bit tests against the current behavior.
7. **Pinning.** Registry versions only (no git or path dependencies), a committed
   `Cargo.lock`, and builds with `--locked` (CI sets
   `MATURIN_PEP517_ARGS: --locked`). Python test and CI dependencies are pinned
   to exact versions; the only runtime Python dependency is NumPy.
8. **Record it.** Put the checks above and their results in the pull request
   description, and note the dependency's purpose next to it in `Cargo.toml`.

Upgrades follow the same steps for the versions in between: read the
changelog and the diff of anything that runs at build time.

## Git and pull requests

- Commits need a DCO sign-off (`git commit -s`); a DCO check runs on every
  pull request.
- Branch from `main`; one topic per pull request. Documentation, comments and
  examples are written in English.
- The required `CI` status includes the RF-DETR and Ultralytics integration
  jobs, which run pinned upstream commits against the pull request build.
  `upstream-canary.yml` checks the latest upstream revisions weekly.
