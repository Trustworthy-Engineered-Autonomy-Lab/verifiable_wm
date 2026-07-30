#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train-and-build Predictor configuration.

Edit this file, then run:

    python train_predictor.py
    python build_predictor.py

No command-line arguments are required.
"""

from pathlib import Path


# =============================================================================
# 1. Experiment
# =============================================================================

# Supported values: "pendulum", "mountain_car", "cartpole", "brake_system".
# This package is preset for the existing Brake dataset.
#ENVIRONMENT = "brake_system"
#ENVIRONMENT = "mountain_car"
ENVIRONMENT = "cartpole"
#ENVIRONMENT = "pendulum"

ENVIRONMENT_HORIZONS = {
    "pendulum": 20,
    "mountain_car": 20,
    "cartpole": 20,
    "brake_system": 10,
}
HORIZON = ENVIRONMENT_HORIZONS[ENVIRONMENT]


# =============================================================================
# 2. Input paths
# =============================================================================

# The real NPZ is both:
#   1) the initial-state source for Predictor trajectories; and
#   2) the format/provenance template copied into predictor_trajectories.npz.
REAL_TRAJECTORIES = Path(
    f"/home/tealab_shared/safety_results/{ENVIRONMENT}/real_trajectories.npz"
)

# JSON containing grid.dims.  Existing cells are used when their initial
# bounds are valid; otherwise cells are reconstructed from grid.dims.
GRID_RESULTS = {
    "pendulum": Path(
        "/home/tealab_shared/safety_results/pendulum/"
        "safety_result_big_cell_a16_lambda05.json"
    ),
    "mountain_car": Path(
        "/home/tealab_shared/safety_results/mountain_car/"
        "safety_result_big_cell_best.json"
    ),
    "cartpole": Path(
        "/home/tealab_shared/safety_results/cartpole/"
        "safety_result_big_cell_a8_lamda01.json"
    ),
    "brake_system": Path(
        "/home/tealab_shared/safety_results/brake_system/"
        "safety_result.json"
    ),
}
GRID_RESULT = GRID_RESULTS[ENVIRONMENT]

# The training program creates this checkpoint.  The build program then reads
# the same file.
CHECKPOINT = Path(
    f"/home/UFAD/xinyangwang/projects/verifiable_wm/trajectory_predictor/models/{ENVIRONMENT}/predictor_transformer.pth"
)


# =============================================================================
# 3. Output paths
# =============================================================================

OUTPUT_DIRECTORY = Path(
    f"/home/UFAD/xinyangwang/projects/verifiable_wm/trajectory_predictor/models/{ENVIRONMENT}"
)

TRAJECTORY_OUTPUT = OUTPUT_DIRECTORY / "predictor_trajectories.npz"
TUBE_OUTPUT = OUTPUT_DIRECTORY / "predictor_tube.json"

# When a legacy NPZ has no train_traj, build_predictor.py writes the held-out
# calibration view here.  The original REAL_TRAJECTORIES file is never
# modified.  Pass this file to --calibration-real.
CONFORMAL_REAL_OUTPUT = (
    OUTPUT_DIRECTORY / "conformal_real_trajectories.npz"
)


# =============================================================================
# 4. Training parameters
# =============================================================================

# train_traj is used when present.  If it is absent and split_val is enabled,
# a deterministic part of val_traj becomes the training source and the rest
# remains disjoint for calibration.  test_traj is always untouched.
TRAIN_FIT_RATIO = 0.9
TRAIN_EPOCHS = 300
TRAIN_BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
TERMINAL_LOSS_WEIGHT = 0.2
GRADIENT_CLIP = 1.0
EARLY_STOPPING_PATIENCE = 30
EARLY_STOPPING_MIN_DELTA = 1e-8
NUM_WORKERS = 0

# If train_traj exists, it is always used and this policy has no effect.
# For the existing Brake archive, split_val deterministically uses part of
# val_traj for training and reserves the rest for calibration.
MISSING_TRAIN_POLICY = "split_val"
DERIVED_TRAIN_RATIO = 0.8

# Transformer architecture.  These values are stored in the checkpoint.
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.1

# Reproducible train/selection split and parameter initialization.
TRAIN_SEED = 2025


# =============================================================================
# 5. Build parameters
# =============================================================================

# Every cell is sampled at exactly:
#   1) lower corner, 2) center, 3) upper corner.
SAMPLES_PER_CELL = 3

# Inference batch size.  Lower this if GPU/CPU memory is insufficient.
BATCH_SIZE = 1024

# "auto" selects CUDA when available, otherwise CPU.
DEVICE = "auto"

# Reproducibility seed used during construction/inference.
BUILD_SEED = 0


# =============================================================================
# 6. Compatibility policy
# =============================================================================

# Applied when the source has an independent train_traj.  For split_val,
# build_predictor.py derives the expected count from the checkpoint indices.
EXPECTED_CALIBRATION_COUNT = 400

# Refuse a derived split that leaves too little independent calibration data.
MIN_DERIVED_CALIBRATION_COUNT = 20

# Pendulum must use a checkpoint trained with continuous unwrapped theta.
# Keeping this True prevents silently using an incompatible legacy checkpoint.
REQUIRE_UNWRAPPED_PENDULUM_CHECKPOINT = True
