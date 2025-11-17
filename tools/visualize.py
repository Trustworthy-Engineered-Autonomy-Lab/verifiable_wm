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

def _get_safety_matrix(dims, cells: List[Dict]) -> np.ndarray:
    
    shape = [dim['num'] for dim in dims]
    safety_matrix = np.array([cell['result'] for cell in cells], dtype=int).reshape(shape)

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
        effective_dims = []
        for dim in dims:
            if dim['num'] != 1:
                print(f"Ignored dimension {dim['name']} whose num is 1")
                effective_dims.append(dim)

        print(f"Grid size: {effective_dims[0]['num']} × {effective_dims[0]['num']} = {effective_dims[0]['num'] * effective_dims[1]['num']} total cells")
        
        # Create the safety matrix
        safety_matrix = _get_safety_matrix(effective_dims, cells).T
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
        