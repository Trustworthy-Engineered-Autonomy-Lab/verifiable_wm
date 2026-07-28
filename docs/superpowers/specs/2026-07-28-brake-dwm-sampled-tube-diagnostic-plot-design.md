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

## Random Complete-Trajectory Figure

Create a second PNG that shows six complete paired trajectories rather than
only the worst case.

- Use `numpy.random.default_rng(728)` and sample six distinct trajectory
  indices without replacement from the real test trajectories that match a
  valid sampled-tube cell.
- Use a single 2-by-3 figure. Each subplot represents one sampled trajectory
  index and its own initial-state cell; tubes from different cells must not be
  mixed in the same subplot.
- In every subplot, overlay the complete 11-state Real trajectory, the
  complete same-index DWM trajectory, and all 11 per-step reachable-tube
  rectangles for that cell.
- Color the tube rectangles by time step using one shared color scale.
- Mark the initial state and the first Real state outside the raw tube, when
  one exists.
- Include the trajectory index, cell index, first violating step, and
  trajectory-level signed margin in each subplot title.
- Write the figure as
  `brake_dwm_sampled_tube_random6_complete_trajectories.png` in the existing
  `diagnostics` directory.
- Write
  `brake_dwm_sampled_tube_random6_complete_trajectories.json` beside it with
  seed 728, the six selected indices, their cell indices, margins, first
  violating steps, complete Real/DWM states, and complete tube bounds.

## Safety and Scope

- Do not modify sampling, controller, dynamics, or margin calculations.
- Do not overwrite the existing signed-margin summary, CSVs, or comparison
  plots.
- Only add diagnostic PNG and JSON files under the `diagnostics` directory.

## Verification

- The output PNG and JSON exist and are nonempty.
- The JSON references seed 728 and the expected source files.
- Recomputing the selected trajectory margin through
  `signed_tube_margin.py` matches the JSON values.
- The paired real and DWM trajectories use the same test index and horizon.
- The random-six indices exactly match
  `numpy.random.default_rng(728).choice(valid_indices, 6, replace=False)`.
- Each random-six subplot uses the cell selected from its own Real initial
  state and contains all 11 Real states, DWM states, and tube steps.
- Existing signed-margin artifacts remain unchanged.
