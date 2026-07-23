import argparse
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


# 当前主流程不调用下面两个 transition helper。它们只为未来可能恢复的
# learned-dynamics `(state, action, next_state)` 数据生成保留，请勿当作
# 当前 sampling 的输出路径；当前输出只有 variant-aware DWM trajectory。
@torch.no_grad()
def rollout_transition(states0, steps, controller, dynamic, device, render_batch_size):
    states = states0.clone()

    all_states = []
    all_actions = []
    all_next_states = []

    for step in range(steps):
        images = render_images(dynamic, states, device, render_batch_size)
        actions = controller(images)
        next_states = dynamic.step(states, actions)

        all_states.append(states)
        all_actions.append(actions)
        all_next_states.append(next_states)

        states = next_states

    states = torch.cat(all_states, dim=0)
    actions = torch.cat(all_actions, dim=0)
    next_states = torch.cat(all_next_states, dim=0)

    return states, actions, next_states


def save_dataset(config, dataset):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for split_name, split_data in dataset.items():
        arrays[f"{split_name}_states"] = to_numpy(split_data["states"])
        arrays[f"{split_name}_actions"] = to_numpy(split_data["actions"])
        arrays[f"{split_name}_next_states"] = to_numpy(split_data["next_states"])

    np.savez_compressed(output_dir / "transition_dataset.npz", **arrays)
    return output_dir


def save_dwm_trajectories(config, trajectory_splits):
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

        images = decoder(decoder_states)
        actions = controller(images)
        states = dynamic.step(states, actions)

        action_steps.append(actions)
        trajectories[:, step + 1, :] = states

    actions = torch.stack(action_steps, dim=1)
    return trajectories, actions


def generate_dataset(config):
    device = resolve_device(config.get("device", "auto"))
    controller = load_controller(config, device)
    decoder = load_decoder(config, device)

    steps = int(config["rollout_steps"])
    render_batch_size = int(config.get("render_batch_size", 64))
    decoder_state_indices = config.get(
        "decoder_state_indices",
        config["decoder"].get("state_indices"),
    )

    dynamic = globals()[config["dynamic"]["name"]](**config["dynamic"].get("args", {}))
    print(f"[Dynamic] {config['dynamic']['name']}")

    initial_splits = build_initial_state_splits(config, device)

    # transition_dataset.npz 生成已停用：(s,a,s') 单步转移对是给 learned dynamics 用的，
    # 当前 pipeline 的 dynamics 是解析已知的（dynamic.py），全仓库没有下游消费者，
    # 而它的真实渲染是 sampling 里最慢的部分。以后要做 learned dynamics 实验时，
    # 取消本函数里相关注释即可恢复（rollout_transition / save_dataset 都还保留着）。
    # dataset = {}
    trajectory_splits = {}
    for split_name, states0 in initial_splits.items():
        # states, actions, next_states = rollout_transition(
        #     states0,
        #     steps,
        #     controller,
        #     dynamic,
        #     device,
        #     render_batch_size,
        # )
        # dataset[split_name] = {
        #     "states": states,
        #     "actions": actions,
        #     "next_states": next_states,
        # }
        # print(
        #     f"[Rollout] {split_name}: "
        #     f"states={tuple(states.shape)}, "
        #     f"actions={tuple(actions.shape)}, "
        #     f"next_states={tuple(next_states.shape)}"
        # )

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

    # output_dir = save_dataset(config, dataset)
    output_dir = Path(config["output_dir"])
    save_dwm_trajectories(config, trajectory_splits)
    return output_dir


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
        if args.decoder_variant is not None:
            config["decoder"]["variant"] = args.decoder_variant
        output_dir = generate_dataset(config)
        print(f"[Done] dwm trajectories saved to {output_dir}")


if __name__ == "__main__":
    main()
