import numpy as np
import pickle
from typing import List, Dict, Tuple, Sequence
import time
import os
import json
import sys
import pathlib
import importlib

from colorama import Fore, Style

import traceback
import shutil
import argparse

from mpi4py import MPI

from verifiers import Verifer
    
def generate_grid_cells(grid: Dict, comm = MPI.COMM_WORLD) -> List[Dict]:
    """Generate grid cells for verification"""

    if comm.rank== 0:
        starts = []
        ends = []
        for dim in grid['dims']:
            vals, step = np.linspace(dim['start'], dim['stop'], dim['num'] + 1, retstep=True)
            print(f"{dim['name']} range: [{dim['start']}, {dim['stop']}], step: {step}")
            dim['step'] = step
            starts.append(vals[0:-1])
            ends.append(vals[1:])
        
        # Generate grid cells
        starts_mesh = np.array(np.meshgrid(*starts, indexing='ij'))
        ends_mesh = np.array(np.meshgrid(*ends, indexing='ij'))

        cell_starts = starts_mesh.reshape(starts_mesh.shape[0], -1).T
        cell_ends = ends_mesh.reshape(ends_mesh.shape[0], -1).T

        cells = np.stack([cell_starts, cell_ends], axis=-1)
        print(f"Generated {len(cells)} cells for verification")

        splited_cells = np.array_split(cells, comm.size)
    else:
        splited_cells = None
    
    local_cells = comm.scatter(splited_cells)
    print(f"Rank {comm.rank} process gets {len(local_cells)} cells")

    return [
        {
            dim['name'] : list(cell[i])  for i, dim in enumerate(grid['dims'])
        }

        for cell in local_cells
    ]

def run_full_verification(verifier: Verifer, cells: List[Dict], 
                          num_steps: int = 20):
    """Run complete multi-cell verification"""

    for idx, cell in enumerate(cells):

        task_str = "Verified:"
        for k,v in cell.items():
            task_str += f" {k}∈[{v[0]},{v[1]}]"
        
        try:
            result = verifier.verify_single_cell(cell, num_steps)
        except KeyboardInterrupt as e:
            break
        except Exception as e:
            result = False
            cell['error_msg'] = str(e)
            status_str = "Error"
            color = Fore.RED
            traceback.print_exc(file=sys.stderr)
        else:
            status_str = "Safe" if result else "Unsafe"
            color = Fore.GREEN if result else Fore.YELLOW
        
        cell['result'] = result
        ndashs = shutil.get_terminal_size().columns - len(task_str) - len(status_str) - 2
        
        print(task_str, '-' * ndashs, color + status_str + Style.RESET_ALL)

def load_input(file_path: str):
    input_file_path = pathlib.Path(file_path)
    if not input_file_path.exists():
        print('Input file does not exist')
        return None

    if input_file_path.suffix != '.json':
        print(f'Unsupport input file type {input_file_path.suffix()}')
        return None

    print(f'Loading the input file {input_file_path}')
    
    with open(input_file_path, 'r') as config_file:
        config = json.load(config_file)

    if 'verifier' not in config:
        print('Verifier is not specified')
        return None
    
    if 'grid' not in config:
        print('Grid is not specified')
        return None
    
    if 'output_prefix' not in config:
        config['output_prefix'] = 'result'

    if 'num_steps' not in config:
        print(f"The number of steps is not specified, default to 20")
        config['num_steps'] = 20

    return config


# ============ Main Execution ============
if __name__ == "__main__":
    # Use line buffering
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    comm = MPI.COMM_WORLD

    # Load the input file
    if comm.rank == 0:
        parser = argparse.ArgumentParser()
        parser.add_argument("input_file", type=str, help="Input file")
        args = parser.parse_args()

        config = load_input(args.input_file)
    else:
        config = None
    
    config = comm.bcast(config)

    if config is None:
        sys.exit(1)

    # Create the verifier
    verifier_cfg = config['verifier']

    try:
        module = importlib.import_module("verifiers")
        verifier_cls = getattr(module, verifier_cfg['name'])
        verifier = verifier_cls(*(verifier_cfg.get('args') or []), **(verifier_cfg.get('kwargs') or {}))
    except Exception as e:
        if comm.rank == 0:
            print(f"Failed to create verifier {verifier_cfg['name']}: {e}")
        sys.exit(1)

    comm.barrier()

    if comm.rank== 0:
        print(f"Created the verifier {verifier_cfg['name']}")
        # Create output directory
        output_prefix = pathlib.Path(config['output_prefix'])
        output_dir = output_prefix.parent
        if not output_dir.exists():
            os.makedirs(output_prefix.parent)
            print(f"Created output directory {output_dir}")

    comm.barrier()

    # Generate grid cells
    local_cells = generate_grid_cells(config['grid'], comm)

    # Run full verification
    start_time = time.time()

    run_full_verification(
        verifier,
        local_cells,
        num_steps=config['num_steps']
    )

    end_time = time.time()
    local_time = end_time - start_time
    total_time = comm.reduce(local_time, MPI.SUM)

    all_local_cells = comm.gather(local_cells)

    # Save verification results
    if comm.rank == 0:
        global_time = total_time / comm.size
        print(f"Verification completed in {global_time:.2f} seconds")

        cells = [cell for local_cells in all_local_cells for cell in local_cells]

        output_path = output_prefix.with_suffix('.json')

        with open(output_path, "w") as f:
            json.dump({**config, "cells": cells}, f)

        print(f"Verification results saved to {output_path}")










