"""Decide which dynamics vintage a trajectory npz was generated with.

config/sampling/*.json only pins {"name": "Pendulum"} with no args, so dt and
friends come from dynamic.py's defaults. When a default changes, every
previously generated npz silently describes a different system: fd3f5d2
(2026-07-23) moved Pendulum dt from 0.05 to 0.02, which is why
pendulum/dataset_v1 (dt=0.05) and dataset_v1_clamp_100 (dt=0.02) coexist in
this repo. Mixing the two across a comparison produced a spurious delta of
7.97.

Files written after the dynamics_* provenance fields were added report their
parameters directly. Older files carry no such record, so dt is recovered
from the trajectory itself: every environment here integrates position with
the already-updated velocity, i.e.

    x' = x + dt * v'      =>      dt = (x' - x) / v'

Samples near a wrap-around, a clamp, or v' ~ 0 are dropped before taking the
median, so the estimate reflects the integrator rather than the boundary
handling.

    python -m tools.check_dynamics_vintage datasets/pendulum/data/*/*.npz
    python -m tools.check_dynamics_vintage --expect-dt 0.02 a.npz b.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Each environment integrates position differently, so dt has to be inverted
# from the matching form (see dynamic.py):
#   Pendulum  semi-implicit, next velocity:  theta' = theta + omega' * dt
#   CartPole  explicit Euler, current one:   x'     = x     + x_dot  * dt
#   Brake     explicit, closing distance:    dist'  = dist  - vel    * dt
# MountainCar is excluded on purpose: it integrates position with velocity and
# no dt factor at all, so there is no dt to recover.
ENV_INTEGRATORS = {
    # position, velocity, use_next_velocity, sign, wrap guard, velocity cap
    "pendulum": (0, 1, True, +1.0, 1.0, 8.0),
    "cartpole": (0, 1, False, +1.0, 1.0e9, 1.0e9),
    "brake_system": (0, 1, False, -1.0, 1.0e9, 1.0e9),
}
TRAJ_KEYS = ("test_traj", "val_traj", "train_traj")


def recorded_provenance(data):
    return {
        key[len("dynamics_") :]: data[key].item()
        for key in data.files
        if key.startswith("dynamics_")
    }


def estimate_dt(traj, spec):
    """Median of (x' - x) / (sign * v) over steps safely inside the domain."""
    position, velocity, use_next, sign, jump_guard, velocity_cap = spec
    pos, vel = traj[..., position], traj[..., velocity]
    delta = pos[:, 1:] - pos[:, :-1]
    driver = sign * (vel[:, 1:] if use_next else vel[:, :-1])
    # Clamped steps (Brake floors distance and velocity at 0, MountainCar-style
    # walls, Pendulum's pi wrap) do not obey the integrator, so drop them.
    usable = (
        (np.abs(delta) < jump_guard)
        & (np.abs(driver) > 1e-3)
        & (np.abs(driver) < velocity_cap * 0.99)
    )
    if sign < 0:
        # Brake clamps distance and velocity at 0; once either floors out the
        # step no longer reflects dt.
        usable &= (pos[:, 1:] > 1e-9) & (pos[:, :-1] > 1e-9)
    if usable.sum() < 20:
        return None, int(usable.sum())
    return float(np.median(delta[usable] / driver[usable])), int(usable.sum())


def guess_env(path):
    parts = {part.lower() for part in Path(path).parts}
    for env in ENV_INTEGRATORS:
        if env in parts:
            return env
    return None


def inspect(path, env_override=None):
    report = {"path": str(path), "recorded": {}, "dt": None, "samples": 0}
    with np.load(path, allow_pickle=False) as data:
        report["recorded"] = recorded_provenance(data)
        traj_key = next((key for key in TRAJ_KEYS if key in data.files), None)
        if traj_key is None:
            report["note"] = "no trajectory array"
            return report
        traj = data[traj_key]

    env = env_override or guess_env(path)
    if env not in ENV_INTEGRATORS:
        report["note"] = f"no integrator model for env={env!r}"
        return report
    if traj.ndim != 3 or traj.shape[1] < 2:
        report["note"] = f"unusable trajectory shape {traj.shape}"
        return report

    spec = ENV_INTEGRATORS[env]
    if traj.shape[2] <= max(spec[0], spec[1]):
        report["note"] = f"trajectory has only {traj.shape[2]} dims"
        return report

    report["env"] = env
    report["dt"], report["samples"] = estimate_dt(traj, spec)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--env",
        default=None,
        choices=sorted(ENV_INTEGRATORS),
        help="Override environment detection (default: infer from path).",
    )
    parser.add_argument(
        "--expect-dt",
        type=float,
        default=None,
        help="Exit non-zero unless every file matches this dt.",
    )
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()

    observed = {}
    for path in args.paths:
        report = inspect(path, args.env)
        recorded_dt = report["recorded"].get("dt")
        if report["dt"] is None:
            detail = report.get("note", f"too few usable steps ({report['samples']})")
            print(f"{path}\n    dt=?        {detail}")
            continue

        source = "recorded" if recorded_dt is not None else "recovered"
        mismatch = ""
        if recorded_dt is not None and abs(recorded_dt - report["dt"]) > args.tol:
            mismatch = f"  !! recorded dt={recorded_dt:g} disagrees with data"
        print(
            f"{path}\n    dt={report['dt']:.6f}  ({source}, n={report['samples']})"
            f"{mismatch}"
        )
        observed.setdefault(round(report["dt"], 6), []).append(str(path))

    status = 0
    if len(observed) > 1:
        print("\nMIXED VINTAGES — these files describe different systems:")
        for dt_value, paths in sorted(observed.items()):
            print(f"  dt={dt_value:g}")
            for path in paths:
                print(f"      {path}")
        status = 1
    if args.expect_dt is not None:
        off = [
            dt_value
            for dt_value in observed
            if abs(dt_value - args.expect_dt) > args.tol
        ]
        if off:
            print(f"\nEXPECTED dt={args.expect_dt:g}, found {sorted(off)}")
            status = 1
    if status == 0 and observed:
        print(f"\nOK — all files share dt={next(iter(observed)):g}")
    return status


if __name__ == "__main__":
    sys.exit(main())
