#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified trajectory-vs-reachable-tube comparison.

Only outputs two figures:
1) real_vs_reachable_tube_examples.png
2) dwm_vs_reachable_tube_examples.png

Designed for:
- mountain_car : state [pos, vel], default plot dims 0 1
- pendulum     : state [theta, omega], default plot dims 0 1
- cartpole     : state [x, x_dot, theta, theta_dot], default plot/check dims 0 1

Meaning of fully contained:
A trajectory is fully contained if every checked state is inside the corresponding
reachable tube bound for the chosen verification horizon and chosen check dimensions.

--check-dims controls which state dimensions are used for containment checking.
For CartPole, the default is position/angle: --plot-dims 0 2 --check-dims 0 2

The containment math lives in tube_geometry.py, per-trajectory comparison and
summary stats in tube_report.py, and the matplotlib rendering in tube_plot.py.
This file is just the CLI: parse args, resolve file paths, load npz/json, call
those three modules, print/save the results.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np

from tube_geometry import load_safety_result, normalize_check_dims
from tube_report import (
    ENV_DEFAULTS,
    build_summary_row,
    compare_set,
    print_summary,
)
from tube_plot import plot_examples

# Pendulum's configured mainline is saliency. MountainCar still uses the old
# shared decoder, while CartPole has multiple active variants and therefore
# requires an explicit --variant when auto-resolving its DWM trajectory.
DEFAULT_VARIANT = {
    "mountain_car": "old",
    "pendulum": "saliency",
}


def npz_keys(path: Path) -> List[str]:
    with np.load(path, allow_pickle=False) as z:
        return list(z.files)


def load_npz_array(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        if key not in z.files:
            raise KeyError(f"Key '{key}' not found in {path}. Available keys: {list(z.files)}")
        return np.asarray(z[key])


def default_key(split: str, kind: str) -> str:
    """
    Both data scripts (make_decoder_dataset.py / sampling.py) always write
    "{split}_traj" / "{split}_states" (see readme's npz key tables), so the
    key name is fixed rather than guessed. Use --real-key/--dwm-key/--state-key
    to override for a one-off file that doesn't follow the convention.
    """
    if kind == "traj":
        return f"{split}_traj"
    if kind == "states":
        return f"{split}_states"
    raise ValueError(kind)


def ensure_traj_shape(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"{name} should have shape (N, T+1, dim), got {arr.shape}")
    return arr.astype(float, copy=False)


def ensure_state_shape(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{name} should have shape (N, dim), got {arr.shape}")
    return arr.astype(float, copy=False)


def validate_trajectory_initial_states(
    states: np.ndarray,
    real_traj: np.ndarray,
    dwm_traj: np.ndarray,
) -> np.ndarray:
    states = ensure_state_shape(states, "trajectory states")
    real_init = np.asarray(real_traj[:, 0, :], dtype=float)
    dwm_init = np.asarray(dwm_traj[:, 0, :], dtype=float)

    if states.shape != real_init.shape:
        raise ValueError(
            "trajectory states must contain the full trajectory state and match "
            f"real trajectory t=0; got states={states.shape}, real={real_init.shape}"
        )
    if dwm_init.shape != real_init.shape:
        raise ValueError(
            f"DWM trajectory t=0 shape {dwm_init.shape} != real trajectory t=0 shape {real_init.shape}"
        )
    if not np.allclose(states, real_init, rtol=0.0, atol=1e-7):
        raise ValueError("trajectory states do not match real trajectory t=0")
    if not np.allclose(states, dwm_init, rtol=0.0, atol=1e-7):
        raise ValueError("trajectory states do not match DWM trajectory t=0")
    return states


def validate_states_in_grid(states: np.ndarray, grid) -> None:
    outside = [
        index
        for index, state in enumerate(np.asarray(states, dtype=float))
        if grid.point_to_linear_index(state) is None
    ]
    if outside:
        preview = outside[:10]
        raise ValueError(
            f"{len(outside)} trajectory initial states are outside the StarV grid; "
            f"first indices: {preview}"
        )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """argv=None reads sys.argv (normal CLI use); pass an explicit list to
    call this programmatically, e.g. from a notebook: parse_args(["--env",
    "cartpole", "--variant", "saliency"])."""
    p = argparse.ArgumentParser(description="Generate two trajectory-vs-reachable-tube figures for multiple environments.")

    p.add_argument("--env", choices=["cartpole", "mountain_car", "pendulum"], default=None)
    p.add_argument("--safety", type=Path, default=None)
    p.add_argument("--real", type=Path, default=None)
    p.add_argument("--dwm", type=Path, default=None)
    p.add_argument("--states", type=Path, default=None)

    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--real-key", default=None)
    p.add_argument("--dwm-key", default=None)
    p.add_argument("--state-key", default=None)

    p.add_argument("--dataset-name", default="dataset_v1",
                   help="Dataset version folder under datasets/<env>/data/, used only when auto-resolving paths via --env.")
    p.add_argument("--variant", default=None,
                   help="Decoder variant used to auto-resolve --dwm as dwm_trajectories_<variant>.npz. "
                        "Defaults to 'old' for mountain_car/pendulum; required for cartpole "
                        "(intensity/saliency/hybrid) since there is no single mainline choice.")
    p.add_argument("--outdir", type=Path, default=None)

    p.add_argument("--real-delta", type=float, default=0.0)
    p.add_argument("--dwm-delta", type=float, default=0.0)
    p.add_argument("--delta", type=float, default=None)

    p.add_argument("--init-source", choices=["auto", "states", "traj"], default="auto")
    p.add_argument("--only-in-grid", action="store_true")
    p.add_argument("--plot-dims", type=int, nargs=2, default=None,
                   help="Two state dimensions to plot. Defaults by env: mountain_car/pendulum=0 1, cartpole=0 2.")
    p.add_argument("--check-dims", type=int, nargs="+", default=None,
                   help="State dimensions used for containment checking. Default by env; for cartpole default is 0 2. Example: --check-dims 0 2")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Maximum transition steps to compare. Example: --max-steps 20 checks states t=0..20.")
    p.add_argument("--print-keys", action="store_true")

    return p.parse_args(argv)


def resolve_paths_and_defaults(args: argparse.Namespace) -> None:
    """
    Fixed layout (see readme "数据文件" / "StarV 验证"):
      results/<env>/safety_result.json
      datasets/<env>/data/<dataset_name>/{starv_states,real_trajectories,dwm_trajectories_<variant>}.npz
    --safety/--real/--dwm/--states always win when passed explicitly.
    """
    root = Path.cwd()

    if args.env is not None:
        data_dir = root / "datasets" / args.env / "data" / args.dataset_name

        if args.safety is None:
            args.safety = root / "results" / args.env / "safety_result.json"
        if args.real is None:
            args.real = data_dir / "real_trajectories.npz"
        if args.states is None:
            default_states = data_dir / "starv_states.npz"
            if default_states.exists():
                args.states = default_states
        if args.dwm is None:
            variant = args.variant or DEFAULT_VARIANT.get(args.env)
            if variant is None:
                raise SystemExit(
                    f"--variant is required to auto-resolve --dwm for env={args.env} "
                    f"(no single mainline decoder variant to assume); pass --variant "
                    f"or --dwm explicitly."
                )
            args.dwm = data_dir / f"dwm_trajectories_{variant}.npz"
        if args.outdir is None:
            args.outdir = root / "results" / args.env / "compare_tube"

    if args.outdir is None:
        args.outdir = Path("compare_outputs")

    if args.safety is None or args.real is None or args.dwm is None:
        raise SystemExit("Please provide --safety --real --dwm, or use --env <cartpole|mountain_car|pendulum>.")


def default_plot_dims(env_name: Optional[str], grid) -> tuple:
    if env_name in ENV_DEFAULTS:
        dims = ENV_DEFAULTS[env_name]["plot_dims"]
        return int(dims[0]), int(dims[1])

    if grid.ndim >= 2:
        return 0, 1

    raise ValueError(f"Cannot plot a grid with ndim={grid.ndim}; need at least 2 dimensions.")


def default_check_dims(env_name: Optional[str], grid) -> Optional[tuple]:
    """
    Default containment-check dimensions.

    If env has a check_dims default, use it.
    Otherwise return None, meaning all grid dimensions.
    """
    if env_name in ENV_DEFAULTS and "check_dims" in ENV_DEFAULTS[env_name]:
        dims = ENV_DEFAULTS[env_name]["check_dims"]
        return tuple(int(d) for d in dims)
    return None


def run(args: argparse.Namespace) -> None:
    """The actual work, taking an already-parsed/built Namespace. Call this
    directly (e.g. from a notebook) with args built via parse_args([...]) to
    avoid going through sys.argv."""
    if args.delta is not None:
        args.real_delta = args.delta
        args.dwm_delta = args.delta

    resolve_paths_and_defaults(args)

    if args.print_keys:
        print("real keys :", npz_keys(args.real))
        print("dwm keys  :", npz_keys(args.dwm))
        if args.states is not None:
            print("state keys:", npz_keys(args.states))
        return

    args.outdir.mkdir(parents=True, exist_ok=True)

    safety, grid, cells = load_safety_result(args.safety)

    plot_dims = tuple(args.plot_dims) if args.plot_dims is not None else default_plot_dims(args.env, grid)
    plot_dims = (int(plot_dims[0]), int(plot_dims[1]))

    if max(plot_dims) >= grid.ndim:
        raise ValueError(f"--plot-dims {plot_dims} is invalid for grid dims {grid.names}")

    requested_check_dims = args.check_dims
    if requested_check_dims is None:
        requested_check_dims = default_check_dims(args.env, grid)
    active_check_dims = normalize_check_dims(requested_check_dims, grid)

    real_key = args.real_key or default_key(args.split, "traj")
    dwm_key = args.dwm_key or default_key(args.split, "traj")

    real_traj = ensure_traj_shape(load_npz_array(args.real, real_key), f"real[{real_key}]")
    dwm_traj = ensure_traj_shape(load_npz_array(args.dwm, dwm_key), f"dwm[{dwm_key}]")

    states_arr = None
    if args.states is not None:
        state_key = args.state_key or default_key(args.split, "states")
        states_arr = ensure_state_shape(load_npz_array(args.states, state_key), f"states[{state_key}]")

    if real_traj.shape[2] < grid.ndim:
        raise ValueError(f"real trajectory dim {real_traj.shape[2]} < grid dim {grid.ndim}")
    if dwm_traj.shape[2] < grid.ndim:
        raise ValueError(f"dwm trajectory dim {dwm_traj.shape[2]} < grid dim {grid.ndim}")

    if args.init_source == "states" and states_arr is None:
        raise ValueError("--init-source states requested, but starv_states.npz was not provided")

    if args.init_source == "traj" or states_arr is None:
        shared_init = validate_trajectory_initial_states(real_traj[:, 0, :], real_traj, dwm_traj)
        init_source = "real_traj[:,0,:]"
    else:
        shared_init = validate_trajectory_initial_states(states_arr, real_traj, dwm_traj)
        init_source = "starv_states.npz"

    real_init = shared_init
    dwm_init = shared_init
    real_init_source = init_source
    dwm_init_source = init_source
    validate_states_in_grid(shared_init, grid)

    print("========== Loaded ==========")
    print(f"env        : {args.env}")
    print(f"safety     : {args.safety}")
    print(f"real       : {args.real} | key={real_key} | shape={real_traj.shape}")
    print(f"dwm        : {args.dwm} | key={dwm_key} | shape={dwm_traj.shape}")
    print(f"states     : {args.states}")
    print(f"grid names : {grid.names}")
    print(f"grid nums  : {grid.nums.tolist()} | cells={len(cells)}")
    print(f"plot dims  : {plot_dims} -> {grid.names[plot_dims[0]]}, {grid.names[plot_dims[1]]}")
    print(f"check dims : {active_check_dims} -> {[grid.names[d] for d in active_check_dims]}")
    print(f"max_steps  : {args.max_steps}")
    print(f"real init  : {real_init_source}")
    print(f"dwm init   : {dwm_init_source}")

    real_rows = compare_set(
        traj=real_traj,
        init_states=real_init,
        kind="real",
        split=args.split,
        grid=grid,
        cells=cells,
        delta=args.real_delta,
        only_in_grid=args.only_in_grid,
        max_steps=args.max_steps,
        check_dims=active_check_dims,
    )

    dwm_rows = compare_set(
        traj=dwm_traj,
        init_states=dwm_init,
        kind="dwm",
        split=args.split,
        grid=grid,
        cells=cells,
        delta=args.dwm_delta,
        only_in_grid=args.only_in_grid,
        max_steps=args.max_steps,
        check_dims=active_check_dims,
    )

    real_summary = build_summary_row(
        real_rows, "real", args.split, args.real_delta, args.only_in_grid, args.max_steps, active_check_dims
    )
    dwm_summary = build_summary_row(
        dwm_rows, "dwm", args.split, args.dwm_delta, args.only_in_grid, args.max_steps, active_check_dims
    )

    print_summary(real_summary)
    print_summary(dwm_summary)

    real_rows_by_idx = {int(r["traj_index"]): r for r in real_rows}
    dwm_rows_by_idx = {int(r["traj_index"]): r for r in dwm_rows}

    plot_examples(
        out_path=args.outdir / "real_vs_reachable_tube_examples.png",
        kind="real",
        trajs=real_traj,
        init_states=real_init,
        rows_by_idx=real_rows_by_idx,
        grid=grid,
        cells=cells,
        safety=safety,
        summary_row=real_summary,
        plot_dims=plot_dims,
        max_steps=args.max_steps,
    )

    plot_examples(
        out_path=args.outdir / "dwm_vs_reachable_tube_examples.png",
        kind="dwm",
        trajs=dwm_traj,
        init_states=dwm_init,
        rows_by_idx=dwm_rows_by_idx,
        grid=grid,
        cells=cells,
        safety=safety,
        summary_row=dwm_summary,
        plot_dims=plot_dims,
        max_steps=args.max_steps,
    )

    print(f"[Saved] {args.outdir / 'real_vs_reachable_tube_examples.png'}")
    print(f"[Saved] {args.outdir / 'dwm_vs_reachable_tube_examples.png'}")


def main(argv: Optional[List[str]] = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
