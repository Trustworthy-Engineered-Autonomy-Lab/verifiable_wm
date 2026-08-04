"""Shared client-side plumbing for driving the CARLA AEBS environment.

The braking benchmark is the only one whose frames come from a simulator
instead of a gym renderer, and every tool that talks to it needs the same
defensive handling: the sensor queues have to be drained before a reset or a
stale frame comes back, a reset can raise queue.Empty under load and must be
retried, and after enough failures the server has to be restarted. Keeping
one copy here means the ground-truth sweep, the grid capture and the real
trajectory collection all drive the simulator identically.

Importing this module registers the CARLA AEBS gym environments as a side
effect (via env.py), so `gym.make(ENV_ID)` works afterwards.
"""

from __future__ import annotations

import queue
import sys
import time
from pathlib import Path

import gym
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import env as _env  # noqa: F401,E402  (registers the CARLA AEBS envs)

ENV_ID = "AdvancedEmergencyBrakingSystemWithRendering-v0"


def safe_reset(env):
    """Reset and return the observation for either gym API generation."""
    out = env.reset()
    if isinstance(out, tuple):
        out = out[0]
    return out


def safe_render_rgb(env) -> np.ndarray:
    """Return the current camera frame as uint8 RGB.

    Prefers the frame the env already latched, because calling render() again
    can block on a sensor queue that has nothing new in it yet.
    """
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "image"):
        img = env.unwrapped.image
        if img is not None:
            return np.array(img, dtype=np.uint8)
    try:
        img = env.render(mode="rgb_array")
    except TypeError:
        img = env.render()
    return np.array(img, dtype=np.uint8)


def drain_sensor_queues(env, max_drain: int = 50) -> None:
    """Discard buffered sensor frames so the next reset sees a fresh one."""
    unwrapped = getattr(env, "unwrapped", env)
    if hasattr(unwrapped, "_queues"):
        for q in unwrapped._queues:
            drained = 0
            while drained < max_drain:
                try:
                    q.get_nowait()
                    drained += 1
                except Exception:
                    break
    for attr in ["_collision_queue", "collision_queue"]:
        if hasattr(unwrapped, attr):
            q = getattr(unwrapped, attr)
            drained = 0
            while drained < max_drain:
                try:
                    q.get_nowait()
                    drained += 1
                except Exception:
                    break


def make_env(env_id: str = ENV_ID) -> gym.Env:
    """Create the CARLA env and let the server settle before first use."""
    env = gym.make(env_id)
    _ = safe_reset(env)
    time.sleep(2.0)
    return env


def set_env_state(env, dist_m: float, vel_val: float,
                  max_tries: int = 6, sleep_sec: float = 2.0,
                  env_id: str = ENV_ID):
    """Teleport the ego vehicle to (distance, velocity).

    Returns (observation, env). The env is returned because an unrecoverable
    reset restarts the simulator and hands back a new handle.
    """
    for attempt in range(1, max_tries + 1):
        try:
            drain_sensor_queues(env)

            if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "reset_to_state"):
                out = env.unwrapped.reset_to_state(dist_m, vel_val)
            elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "reset"):
                out = env.unwrapped.reset(state=[dist_m, vel_val])
            else:
                raise AttributeError(
                    "Cannot find reset_to_state or reset(state=...) on env."
                )

            if isinstance(out, tuple):
                out = out[0]
            return np.array(out, dtype=np.float32), env

        except queue.Empty:
            wait = sleep_sec * attempt
            print(f"[Warn] queue.Empty on reset (d={dist_m:.4f}, v={vel_val:.4f}), "
                  f"try {attempt}/{max_tries}, sleeping {wait:.0f}s")
            time.sleep(wait)

        except Exception:
            raise

    print(f"[Warn] All {max_tries} retries failed. Restarting CARLA env...")
    try:
        env.close()
    except Exception:
        pass
    time.sleep(5.0)
    env = make_env(env_id)

    drain_sensor_queues(env)
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "reset_to_state"):
        out = env.unwrapped.reset_to_state(dist_m, vel_val)
    else:
        out = env.unwrapped.reset(state=[dist_m, vel_val])
    if isinstance(out, tuple):
        out = out[0]
    return np.array(out, dtype=np.float32), env
