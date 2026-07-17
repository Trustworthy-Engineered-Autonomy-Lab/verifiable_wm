from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from visualize_datasets import (
    EnvironmentSpec,
    choose_indices,
    plot_2d_trajectories,
    summarize_npz,
    visualize_environment,
)


def _write_fixture_dataset(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True)
    rng = np.random.default_rng(11)

    np.savez_compressed(
        dataset_dir / "decoder_states.npz",
        train_states=rng.normal(size=(12, 2)).astype(np.float32),
        train_images=rng.random(size=(12, 1, 8, 8)).astype(np.float32),
    )

    real_arrays = {}
    dwm_arrays = {"variant": np.array("saliency")}
    for split in ("train", "val", "test"):
        real_traj = rng.normal(size=(12, 5, 2)).astype(np.float32)
        real_actions = rng.normal(size=(12, 4, 1)).astype(np.float32)
        real_arrays[f"{split}_traj"] = real_traj
        real_arrays[f"{split}_actions"] = real_actions
        dwm_arrays[f"{split}_traj"] = real_traj + 0.1
        dwm_arrays[f"{split}_actions"] = real_actions - 0.1
    np.savez_compressed(dataset_dir / "real_trajectories.npz", **real_arrays)
    np.savez_compressed(dataset_dir / "dwm_trajectories_saliency.npz", **dwm_arrays)


def test_choose_indices_is_deterministic_unique_and_in_range():
    first = choose_indices(20, 5, seed=7)
    second = choose_indices(20, 5, seed=7)

    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 5
    assert np.all((0 <= first) & (first < 20))


def test_summarize_npz_reports_numeric_array_metadata(tmp_path):
    path = tmp_path / "arrays.npz"
    np.savez_compressed(path, values=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

    lines = summarize_npz(path)

    assert lines == ["values: shape=(2, 2), dtype=float32, min=1, max=4"]


def test_visualize_environment_creates_nonempty_figures(tmp_path):
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "visualizations"
    _write_fixture_dataset(dataset_dir)
    spec = EnvironmentSpec(
        name="fixture",
        dataset_dir=dataset_dir,
        variant="saliency",
        decoder_state_names=("position", "angle"),
        trajectory_state_names=("position", "angle"),
    )

    outputs = visualize_environment(spec, output_dir)

    assert [path.name for path in outputs] == [
        "fixture_dataset_samples.png",
        "fixture_trajectory_comparison.png",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)


def test_plot_2d_trajectories_marks_every_stored_point(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "visualizations"
    _write_fixture_dataset(dataset_dir)
    spec = EnvironmentSpec(
        name="fixture",
        dataset_dir=dataset_dir,
        variant="saliency",
        decoder_state_names=("position", "angle"),
        trajectory_state_names=("position", "angle"),
    )
    plotted_lines = []
    original_plot = Axes.plot

    def record_plot(axis, x, y, *args, **kwargs):
        plotted_lines.append((len(x), kwargs.get("marker")))
        return original_plot(axis, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", record_plot)

    output = plot_2d_trajectories(spec, output_dir, dimensions=(0, 1))

    assert output.name == "fixture_2d_trajectories.png"
    assert output.is_file() and output.stat().st_size > 0
    assert plotted_lines
    assert all(point_count == 5 and marker == "." for point_count, marker in plotted_lines)
