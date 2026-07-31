import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYGLET_HEADLESS", "True")

import numpy as np
import torch

from model import *
from dynamic import *
from utils import (
    load_config,
    resolve_device,
    load_state_splits,
    starv_states_path,
    render_images,
    to_numpy,
    dynamics_provenance,
)
from sampled_tube import (
    build_sampled_safety_result,
    grid_cell_bounds,
    load_or_create_cellwise_states,
    sample_cellwise_latents,
    sampled_cells_from_trajectories,
    validate_cellwise_rollout_artifact,
    validate_cellwise_setup,
    write_json_atomic,
    write_npz_atomic,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIGS = tuple(
    PROJECT_ROOT / "config" / "sampling" / f"{environment}.json"
    for environment in ("cartpole", "mountain_car", "pendulum")
)


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_controller(config, device):
    controller_config = config["controller"]
    controller_name = controller_config["name"]
    controller_cls = globals()[controller_name]
    controller_args = controller_config.get("args", {})

    controller = controller_cls(**controller_args).to(device).eval()
    controller.load_state_dict(load_state_dict(controller_config["weights"], device))

    print(f"[Load] {controller_name}={controller_config['weights']}")
    return controller


def decoder_variant(config):
    return config["decoder"].get("variant", "old")


def resolve_decoder_weights(config):
    weights = config["decoder"]["weights"]
    if isinstance(weights, dict):
        weights = weights[decoder_variant(config)]
    return str(weights)


def load_decoder(config, device):
    decoder_config = config["decoder"]
    decoder_name = decoder_config["name"]
    decoder_cls = globals()[decoder_name]
    decoder_args = decoder_config.get("args", {})

    weights = resolve_decoder_weights(config)

    decoder = decoder_cls(**decoder_args).to(device).eval()
    decoder.load_state_dict(load_state_dict(weights, device))

    print(f"[Load] {decoder_name}[{decoder_variant(config)}]={weights}")
    return decoder


def build_initial_state_splits(config, device):
    starv_config_path = config.get("starv_config")
    if not starv_config_path:
        raise KeyError("sampling config must define 'starv_config'")
    starv_config = load_config(starv_config_path)
    path = starv_states_path(starv_config)
    saved = load_state_splits(path, device)
    print(f"[Load] starv states={path}")
    return {
        split: saved[f"{split}_states"]
        for split in ("train", "val", "test")
    }


def save_dwm_trajectories(config, trajectory_splits, dynamic=None):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for split_name, split_data in trajectory_splits.items():
        arrays[f"{split_name}_traj"] = to_numpy(split_data["traj"])
        arrays[f"{split_name}_actions"] = to_numpy(split_data["actions"])

    # 溯源信息直接存进 npz（同 saliency_occlusion.npz 的做法）：消融会反复覆盖
    # 同名文件，只有文件内部记录才说得清当前这份是哪个 checkpoint 跑出来的
    arrays["variant"] = np.array(decoder_variant(config))
    arrays["decoder_weights"] = np.array(resolve_decoder_weights(config))
    arrays["rollout_steps"] = np.array(int(config["rollout_steps"]))
    arrays["starv_config"] = np.array(str(config["starv_config"]))
    arrays["controller_weights"] = np.array(
        str(config["controller"]["weights"])
    )
    if dynamic is not None:
        arrays.update(dynamics_provenance(dynamic))

    output_path = output_dir / f"dwm_trajectories_{decoder_variant(config)}.npz"
    np.savez_compressed(output_path, **arrays)
    print(f"[Saved] {output_path} (decoder={arrays['decoder_weights']})")


@torch.no_grad()
def rollout_dwm_trajectory(
    states0,
    steps,
    decoder,
    controller,
    dynamic,
    device,
    decoder_state_indices=None,
    latents=None,
):
    num_samples, state_dim = states0.shape
    trajectories = torch.empty(num_samples, steps + 1, state_dim, device=device)
    action_steps = []

    states = states0.clone()
    trajectories[:, 0, :] = states

    for step in range(steps):
        decoder_states = states
        if decoder_state_indices is not None:
            decoder_states = states[:, decoder_state_indices]

        if latents is None:
            images = decoder(decoder_states)
        else:
            images = decoder(decoder_states, latents[:, step, :])
        actions = controller(images)
        states = dynamic.step(states, actions)

        action_steps.append(actions)
        trajectories[:, step + 1, :] = states

    actions = torch.stack(action_steps, dim=1)
    return trajectories, actions


@torch.no_grad()
def rollout_cellwise_batches(
    *,
    states,
    steps,
    decoder,
    controller,
    dynamic,
    device,
    cell_batch_size,
    decoder_state_indices=None,
    latents=None,
    progress=None,
):
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 3:
        raise ValueError(
            "cellwise states must have shape (cells, samples, state_dim)"
        )
    if int(cell_batch_size) <= 0:
        raise ValueError("cell_batch_size must be positive")

    latent_array = None
    if latents is not None:
        latent_array = np.asarray(latents, dtype=np.float32)
        expected_prefix = (states.shape[0], states.shape[1], int(steps))
        if latent_array.ndim != 4 or latent_array.shape[:3] != expected_prefix:
            raise ValueError(
                f"cellwise latent shape mismatch: actual={latent_array.shape}, "
                f"expected prefix={expected_prefix}"
            )

    trajectory_chunks = []
    action_chunks = []
    samples_per_cell = states.shape[1]
    for cell_start in range(0, len(states), int(cell_batch_size)):
        cell_stop = min(cell_start + int(cell_batch_size), len(states))
        batch_cells = cell_stop - cell_start
        batch_states = torch.from_numpy(
            states[cell_start:cell_stop].reshape(-1, states.shape[2])
        ).to(device)
        batch_latents = None
        if latent_array is not None:
            batch_latents = torch.from_numpy(
                latent_array[cell_start:cell_stop].reshape(
                    -1,
                    int(steps),
                    latent_array.shape[3],
                )
            ).to(device)

        trajectories, actions = rollout_dwm_trajectory(
            batch_states,
            int(steps),
            decoder,
            controller,
            dynamic,
            device,
            decoder_state_indices,
            batch_latents,
        )
        trajectory_chunks.append(
            to_numpy(trajectories).reshape(
                batch_cells,
                samples_per_cell,
                int(steps) + 1,
                states.shape[2],
            )
        )
        action_chunks.append(
            to_numpy(actions).reshape(
                batch_cells,
                samples_per_cell,
                int(steps),
                actions.shape[2],
            )
        )
        if progress is not None:
            progress(cell_stop, len(states))

    return (
        np.concatenate(trajectory_chunks, axis=0),
        np.concatenate(action_chunks, axis=0),
    )


def generate_dataset(config):
    device = resolve_device(config.get("device", "auto"))
    controller = load_controller(config, device)
    decoder = load_decoder(config, device)

    steps = int(config["rollout_steps"])
    decoder_state_indices = config.get(
        "decoder_state_indices",
        config["decoder"].get("state_indices"),
    )

    dynamic = globals()[config["dynamic"]["name"]](**config["dynamic"].get("args", {}))
    print(f"[Dynamic] {config['dynamic']['name']}")

    initial_splits = build_initial_state_splits(config, device)

    # Sampling only produces variant-aware DWM trajectories. The (s, a, s')
    # single-step transition pairs existed for learned dynamics, but this
    # pipeline's dynamics are analytic (dynamic.py) and nothing downstream
    # consumed them, so those helpers were removed.
    trajectory_splits = {}
    for split_name, states0 in initial_splits.items():
        traj, dwm_actions = rollout_dwm_trajectory(
            states0,
            steps,
            decoder,
            controller,
            dynamic,
            device,
            decoder_state_indices,
        )
        trajectory_splits[split_name] = {
            "traj": traj,
            "actions": dwm_actions,
        }

        print(
            f"[DWM Trajectory] {split_name}: "
            f"traj={tuple(traj.shape)}, "
            f"actions={tuple(dwm_actions.shape)}"
        )

    output_dir = Path(config["output_dir"])
    save_dwm_trajectories(config, trajectory_splits, dynamic)
    return output_dir


def _project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def generate_cellwise_tube(wrapper, wrapper_path):
    base_config_path = _project_path(wrapper["base_sampling_config"])
    grid_config_path = _project_path(wrapper["grid_config"])
    base_config = load_config(base_config_path)
    model_starv_path = _project_path(base_config["starv_config"])
    model_starv_config = load_config(model_starv_path)
    grid_config = load_config(grid_config_path)
    metadata = validate_cellwise_setup(
        wrapper,
        base_config,
        model_starv_config,
        grid_config,
    )

    device = resolve_device(base_config.get("device", "auto"))
    controller = load_controller(base_config, device)
    decoder = load_decoder(base_config, device)
    dynamic_config = base_config["dynamic"]
    dynamic = globals()[dynamic_config["name"]](
        **dynamic_config.get("args", {})
    )
    print(f"[Dynamic] {dynamic_config['name']}")

    steps = int(base_config["rollout_steps"])
    samples_per_cell = int(wrapper["samples_per_cell"])
    seed = int(wrapper["seed"])
    cell_batch_size = int(wrapper["cell_batch_size"])
    cells = grid_cell_bounds(grid_config["grid"])
    print(
        f"[Cellwise Setup] model={metadata['model_id']} "
        f"({metadata['model_type']}), cells={len(cells)}, "
        f"samples_per_cell={samples_per_cell}, "
        f"trajectories={len(cells) * samples_per_cell}, "
        f"horizon={steps}, seed={seed}"
    )
    states_path = _project_path(wrapper["states_file"])
    states = load_or_create_cellwise_states(
        states_path,
        cells,
        samples_per_cell=samples_per_cell,
        seed=seed,
        grid=grid_config["grid"],
        horizon=steps,
        source_grid_config=str(wrapper["grid_config"]),
    )
    print(
        f"[Cellwise States] cells={len(cells)}, "
        f"samples_per_cell={samples_per_cell}, path={states_path}"
    )

    latents = None
    if metadata["model_type"] == "G_MLP":
        latent_dim = int(metadata["latent_dim"])
        latents = sample_cellwise_latents(
            num_cells=len(cells),
            samples_per_cell=samples_per_cell,
            num_steps=steps,
            latent_dim=latent_dim,
            z_range=float(metadata["z_range"]),
            seed=seed,
        )

    decoder_state_indices = base_config.get(
        "decoder_state_indices",
        base_config["decoder"].get("state_indices"),
    )
    trajectories, actions = rollout_cellwise_batches(
        states=states,
        steps=steps,
        decoder=decoder,
        controller=controller,
        dynamic=dynamic,
        device=device,
        cell_batch_size=cell_batch_size,
        decoder_state_indices=decoder_state_indices,
        latents=latents,
        progress=lambda completed, total: print(
            f"[Cellwise Rollout] {completed}/{total} cells "
            f"({100.0 * completed / total:.1f}%)"
        ),
    )

    latent_dim = None
    z_range = None
    if latents is not None:
        latent_dim = int(metadata["latent_dim"])
        z_range = float(metadata["z_range"])
    validate_cellwise_rollout_artifact(
        trajectories,
        actions,
        num_cells=len(cells),
        samples_per_cell=samples_per_cell,
        horizon=steps,
        state_dim=int(metadata["state_dim"]),
        initial_states=states,
        latents=latents,
        latent_dim=latent_dim,
        z_range=z_range,
    )
    sampled_cells = sampled_cells_from_trajectories(
        trajectories,
        cells,
        model_starv_config["verifier"],
    )

    trajectory_path = _project_path(wrapper["trajectory_file"])
    arrays = {
        "trajectories": trajectories,
        "actions": actions,
        "cell_indices": np.arange(len(cells), dtype=np.int64),
        "seed": np.array(seed),
        "samples_per_cell": np.array(samples_per_cell),
        "grid_fingerprint": np.array(metadata["grid_fingerprint"]),
        "rollout_steps": np.array(steps),
        "model_id": np.array(metadata["model_id"]),
        "model_type": np.array(metadata["model_type"]),
        "state_dim": np.array(int(metadata["state_dim"])),
        "decoder_state_indices": np.asarray(
            metadata["decoder_state_indices"],
            dtype=np.int64,
        ),
        "decoder_weights": np.array(metadata["decoder_weights"]),
        "controller_name": np.array(metadata["controller_name"]),
        "controller_weights": np.array(metadata["controller_weights"]),
        "controller_activation": np.array(
            metadata["controller_activation"]
        ),
        "dynamic_name": np.array(metadata["dynamic_name"]),
        "dynamic_args_json": np.array(
            json.dumps(
                metadata["dynamic_args"],
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "formal_semantics_match": np.array(
            bool(metadata["formal_semantics_match"])
        ),
        "source_wrapper_config": np.array(str(wrapper_path)),
        "source_sampling_config": np.array(str(wrapper["base_sampling_config"])),
        "source_grid_config": np.array(str(wrapper["grid_config"])),
        "source_starv_config": np.array(str(base_config["starv_config"])),
        "source_states_file": np.array(str(wrapper["states_file"])),
    }
    if latents is not None:
        arrays["latents"] = latents
        arrays["z_range"] = np.array(float(metadata["z_range"]))
        arrays["latent_dim"] = np.array(int(metadata["latent_dim"]))
        arrays["generator_output_activation"] = np.array(
            metadata["generator_output_activation"]
        )
    else:
        arrays["decoder_output_activation"] = np.array(
            metadata["decoder_output_activation"]
        )
    write_npz_atomic(trajectory_path, **arrays)
    print(
        f"[Cellwise Trajectories] shape={trajectories.shape}, "
        f"path={trajectory_path}"
    )

    result_metadata = {
        **metadata,
        "samples_per_cell": samples_per_cell,
        "seed": seed,
        "source_sampling_config": str(wrapper["base_sampling_config"]),
        "source_grid_config": str(wrapper["grid_config"]),
        "source_starv_config": str(base_config["starv_config"]),
        "source_states": str(wrapper["states_file"]),
        "source_trajectories": str(wrapper["trajectory_file"]),
    }
    payload = build_sampled_safety_result(
        model_starv_config,
        sampled_cells,
        result_metadata,
        grid_config=grid_config,
    )
    tube_path = _project_path(wrapper["tube_file"])
    write_json_atomic(tube_path, payload)
    print(f"[Cellwise Tube] cells={len(sampled_cells)}, path={tube_path}")
    return tube_path


def run_sampling_config(config, config_path, decoder_variant=None):
    mode = config.get("sampling_mode", "state_splits")
    if mode == "state_splits":
        if decoder_variant is not None:
            config["decoder"]["variant"] = decoder_variant
        return generate_dataset(config)
    if mode == "cellwise_tube":
        if decoder_variant is not None:
            raise ValueError(
                "--decoder-variant is not supported for cellwise wrapper "
                "configs; select the matching wrapper config instead"
            )
        return generate_cellwise_tube(config, Path(config_path))
    raise ValueError(f"Unsupported sampling_mode: {mode!r}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "configs",
        nargs="+",
        type=Path,
        help="Config JSON files.",
    )
    parser.add_argument(
        "--decoder-variant",
        default=None,
        help="Which decoder weights to roll out with (old / intensity / saliency). "
        "Overrides the config's decoder.variant.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for config_path in args.configs:
        print(f"[Config] {config_path}")
        config = load_config(config_path)
        output_path = run_sampling_config(
            config,
            config_path,
            decoder_variant=args.decoder_variant,
        )
        if config.get("sampling_mode", "state_splits") == "cellwise_tube":
            print(f"[Done] sampled tube saved to {output_path}")
        else:
            print(f"[Done] dwm trajectories saved to {output_path}")


if __name__ == "__main__":
    main()
