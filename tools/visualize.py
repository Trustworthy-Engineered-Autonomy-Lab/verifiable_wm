"""Render a StarV safety_result.json as a red/green 2D safety map.

Reads the `grid` and `cells` written by verify.py and colours each cell by
its `result` flag (green = verified safe, red = not verified). Grids whose
extra dimensions are singletons are squeezed down to the two that vary.

    python -m tools.visualize results/pendulum/safety_result.json \
        --title "Pendulum" --save Figures/pendulum_safety.png
"""

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from typing import Dict, List, Tuple
from pathlib import Path

import argparse
import sys
import json
import os

def _plot_2d_safety_map(x_dim: Dict, y_dim: Dict, safety_matrix: np.ndarray, title = "") -> Tuple[Figure, Axes]:
    """Visualize safety map as a grid heatmap"""

    # Create custom colormap: red for unsafe (0), green for safe (1)
    colors = ['red', 'green']
    cmap = ListedColormap(colors)

    # Create the plot
    fig, ax = plt.subplots()

    # Plot the grid heatmap
    im = ax.imshow(safety_matrix[..., ], cmap=cmap, aspect='auto', origin='lower',
                   extent=[x_dim['start'], x_dim['stop'], y_dim['start'], y_dim['stop']])

    # Set aspect so that each cell looks like a square
    ax.set_aspect(x_dim['step']/y_dim['step'])

    # Add grid lines
    ax.grid(True)

    # Customize the plot
    ax.set_xlabel(f"{x_dim['name']}", fontsize=12)
    ax.set_ylabel(f"{y_dim['name']}", fontsize=12)
    ax.set_title(title , fontsize=14)

    # Create custom legend
    legend_elements = [
        Patch(facecolor='green', label=f'Success ({safe_count})'),
        Patch(facecolor='red', label=f'Failure ({unsafe_count})')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    fig.tight_layout()
    return fig, ax

def varying_dims(dims: List[Dict]) -> List[Dict]:
    """The grid dimensions that actually span a range.

    CartPole verifies over a four-dimensional grid whose velocity and angular
    velocity are pinned to single points, so the map it produces is really
    two-dimensional.
    """
    return [dim for dim in dims if int(dim["num"]) != 1]


def safety_matrix_from_cells(dims: List[Dict], cells: List[Dict]) -> np.ndarray:
    """0/1 safety map over the varying grid dimensions.

    `dims` is the full grid definition; singleton dimensions are dropped from
    the result. Cells are placed by the midpoint of the initial box they
    recorded rather than by their position in the list, so a reordered or
    partially regenerated result still lands on the right square, and a
    missing cell is reported instead of silently shifting the map. A cell that
    errored during verification carries no `result` and counts as not verified.
    """
    axes = [index for index, dim in enumerate(dims) if int(dim["num"]) != 1]
    edges = [
        np.linspace(float(dims[a]["start"]), float(dims[a]["stop"]), int(dims[a]["num"]) + 1)
        for a in axes
    ]
    shape = tuple(int(dims[a]["num"]) for a in axes)
    safety_matrix = np.full(shape, -1, dtype=np.int8)

    for cell in cells:
        initial = np.asarray(cell["bounds"][0], dtype=float)
        index = tuple(
            int(np.clip(
                np.searchsorted(edges[k], initial[a].mean()) - 1, 0, shape[k] - 1
            ))
            for k, a in enumerate(axes)
        )
        safety_matrix[index] = 1 if cell.get("result") else 0

    missing = int((safety_matrix < 0).sum())
    if missing:
        raise ValueError(f"{missing} grid cells are missing from the safety result")
    return safety_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--title", type=str, help="title for the plot", default="")
    parser.add_argument("--save", type=str, help="filename of the result to save", default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("result_file", type=str, help="result file path")
    args = parser.parse_args()

    result_path = Path(args.result_file)
    if not result_path.exists():
        print(f"Result file {args.result_file} does not exist")
        sys.exit(1)

    if result_path.suffix != ".json":
        print(f"Unsupported result file format {result_path.suffix}")
        sys.exit(1)

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON parse error at line {e.lineno}, column {e.colno}: {e.msg}")
        sys.exit(1)

    try:
        grid = result['grid']
        cells = result['cells']

        dims = grid['dims']
        effective_dims = varying_dims(dims)
        for dim in dims:
            if int(dim['num']) == 1:
                print(f"Ignored dimension {dim['name']} whose num is 1")

        if len(effective_dims) != 2:
            print(
                f"Need exactly 2 varying dimensions to plot, found "
                f"{len(effective_dims)}: {[d['name'] for d in effective_dims]}"
            )
            sys.exit(1)

        print(
            f"Grid size: {effective_dims[0]['num']} × {effective_dims[1]['num']}"
            f" = {effective_dims[0]['num'] * effective_dims[1]['num']} total cells"
        )
        
        # Create the safety matrix
        safety_matrix = safety_matrix_from_cells(dims, cells).T
    except KeyError as e:
        print(f"Could not find field {e.args[0]} in {args.result_file}")
        sys.exit(1)

    # Count safe and unsafe cells
    safe_count = int(np.sum(safety_matrix))
    unsafe_count = int(np.size(safety_matrix) - safe_count)
    print(f"Summary: {safe_count} safe cells, {unsafe_count} unsafe cells")

    save_matrix = False
    save_plot = False

    if args.save is not None:
        save_path = Path(args.save)
        if save_path.name == "":
            print(f"Invalid path {args.save} to save any result")
            sys.exit(1)
        
        save_folder = save_path.parent
        os.makedirs(save_folder,exist_ok=True)

        suffix = save_path.suffix.lower().lstrip('.')
        if suffix in ['png','jpeg','jpg']:
            save_plot = True
        elif suffix == 'npy':
            save_matrix = True


    if args.show or save_plot:
        fig, ax = _plot_2d_safety_map(*effective_dims, safety_matrix, title=args.title)
        if args.show:
            fig.show()
        if save_plot:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Safety map saved to: {args.save}")
        
    if save_matrix:
        np.save(save_path, safety_matrix)
        print(f"Safety matrix saved to {args.save}")
        