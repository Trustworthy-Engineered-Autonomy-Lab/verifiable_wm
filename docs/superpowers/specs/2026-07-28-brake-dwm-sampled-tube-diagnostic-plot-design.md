# Brake DWM Sampled-Tube Diagnostic Plot Design

## Goal

Create one focused diagnostic figure that explains why the Brake DWM sampled
tube has low containment, using the current seed-728 sampling artifacts and
the same real/DWM evaluation trajectories consumed by
`signed_tube_margin.py`.

## Inputs

- Sampled tube:
  `results/brake_system/sampled_tube/recomputed_20260728/sampled_reachable_tube_old_seed_728.json`
- Real trajectories:
  `safety_results/brake_system/real_trajectories.npz`, key `test_traj`
- DWM trajectories:
  `safety_results/brake_system/dwm_trajectories_saliency.npz`, key
  `test_traj`
- Output directory:
  `results/brake_system/signed_tube_margin/recomputed_20260728/b_dwm_sampled/diagnostics`

The diagnostic must reuse the grid lookup, tube loading, interval handling,
and signed-margin semantics from `signed_tube_margin.py`.

## Trajectory Selection

Evaluate every real test trajectory against the raw sampled tube and select
the trajectory with the smallest trajectory-level signed margin. Use its
initial state to select the matching grid cell. Pair it with the DWM
trajectory at the same test index.

## Figure Layout

Write one four-panel PNG:

1. **Phase plane:** overlay real and DWM trajectories with the selected
   cell's reachable rectangle at every time step. Highlight the initial state
   and the first step outside the tube.
2. **Distance vs. time:** plot real and DWM distance with the corresponding
   tube interval as a shaded band.
3. **Velocity vs. time:** plot real and DWM velocity with the corresponding
   tube interval as a shaded band.
4. **Error and margin:** plot the real signed margin, DWM signed margin, and
   real-minus-DWM state difference by step. Mark zero margin and the first
   violating step.

The title and annotations must include the trajectory index, grid-cell index,
first violating step, worst real margin, and maximum real/DWM state
difference.

## Numerical Companion

Write a JSON file beside the PNG containing:

- selected trajectory and cell indices;
- first violating step;
- per-step real and DWM states;
- per-step tube bounds;
- per-step signed margins;
- per-step and maximum real/DWM differences.

This keeps exact values available when visual overlap makes the trajectories
look identical.

## Safety and Scope

- Do not modify sampling, controller, dynamics, or margin calculations.
- Do not overwrite the existing signed-margin summary, CSVs, or comparison
  plots.
- Only add the diagnostic PNG and JSON under the new `diagnostics` directory.

## Verification

- The output PNG and JSON exist and are nonempty.
- The JSON references seed 728 and the expected source files.
- Recomputing the selected trajectory margin through
  `signed_tube_margin.py` matches the JSON values.
- The paired real and DWM trajectories use the same test index and horizon.
- Existing signed-margin artifacts remain unchanged.
