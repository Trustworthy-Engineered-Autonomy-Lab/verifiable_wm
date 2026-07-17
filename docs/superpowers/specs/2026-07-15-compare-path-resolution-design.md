# Compare Path Resolution Design

## Scope

This change fixes only the runtime path-selection behavior in `compare.py`.
It does not modify verification configuration, regenerate safety results, or
change trajectory-containment calculations.

## Problem

`compare.py` currently defines editable `DEFAULT_*` paths at module scope, but
`apply_args()` replaces them with paths derived from `--env` whenever explicit
path arguments are absent. Consequently, running the script without path
arguments ignores the documented defaults and can select nonexistent files.

The paths are also relative to the process working directory. Invoking the
script by absolute path from another directory therefore looks for `results/`
and `datasets/` beneath that unrelated directory.

## Runtime Behavior

The module-level `DEFAULT_SAFETY_PATH`, `DEFAULT_REAL_TRAJ_PATH`,
`DEFAULT_DWM_TRAJ_PATH`, and `DEFAULT_OUT_DIR` are the single source of default
paths. They are anchored to the repository directory obtained from
`Path(__file__).resolve().parent`.

`apply_args()` selects each path independently:

- use the corresponding command-line path when provided;
- otherwise use the corresponding module-level default.

`--env` continues to select environment-specific plotting and containment
dimensions. It no longer constructs input or output paths. This separation
allows an explicit `--env` to be combined with an explicit experiment dataset
without an undocumented directory convention.

Relative paths supplied explicitly on the command line retain standard CLI
behavior and remain relative to the caller's working directory. Only built-in
defaults are anchored to the repository.

## Command-Line Help

The help strings for `--safety`, `--real`, `--dwm`, and `--outdir` report the
actual `DEFAULT_*` values. They no longer advertise the obsolete
`datasets/<env>/data/dataset_v1` convention.

## Error Handling

Existing input-path validation remains responsible for reporting missing
files. This change does not add fallback searches or silently select alternate
experiments. A missing resolved path continues to raise `FileNotFoundError`
with the path and logical label.

## Tests and Verification

Automated tests will verify that:

1. absent path arguments resolve to the four module-level defaults;
2. each explicit command-line path overrides only its corresponding default;
3. built-in defaults remain stable when the process working directory changes;
4. `--env` still selects the expected plotting/checking dimensions without
   changing the default experiment paths.

Runtime verification will run `compare.py --print-keys` from both the
repository root and an external working directory. The command must load the
configured real and DWM NPZ files without attempting to create plots. Full
comparison plotting is intentionally deferred because the separate safety
bounds-history issue is outside this phase.

## Files

The implementation is limited to:

- `compare.py`;
- a focused path-resolution test file under `tests/`.

No verification configuration or generated result file is changed.
