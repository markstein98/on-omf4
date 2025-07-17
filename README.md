# O(N) Theory Simulation on 2D Lattice

This repository contains code to simulate an O(N) theory on a 2-dimensional lattice using the Numerical Stochastic Perturbation Theory (NSPT) framework. The simulation is implemented in Julia and is designed to run in a SLURM cluster environment.

## Overview

The code performs Monte Carlo simulations of O(N) quantum field theory on a 2D lattice using Hybrid Monte Carlo (HMC) algorithms. The simulation supports:
- Variable lattice sizes (npoint × npoint)
- Configurable O(N) theory with N = ncomp + 1 components
- Perturbative expansion up to specified orders
- Checkpoint/resume functionality
- Site-per-site energy analysis

## Prerequisites

- **SLURM Environment**: This program is intended to be executed inside a SLURM environment
- **Julia**: The simulation requires Julia runtime
- **GPU Support**: The default configuration requests A100 80GB GPU resources, but it can be changed accordingly to the resources available in the SLURM environment, inside the `call_sbatch.sh` and the `julia/resume_sbatch.sh` files

## Installation and Setup

### 1. Configuration File Setup

**REQUIRED**: Before running the simulation, you must configure the environment:

1. **Rename the configuration file**:
   ```bash
   mv configuration_example.conf configuration.conf
   ```

2. **Edit `configuration.conf`** and modify the following lines:

   - **Line 2**: Set your email address for SLURM notifications
     ```bash
     NOTIFICATION_EMAIL="your.email@institution.edu"
     ```

   - **Line 3**: Set the full path to the root folder of this repository, i.e., the path to the folder in which this file is contained
     ```bash
     JULIA_FOLDER_PATH="/full/path/to/omf4/repository"
     ```

   - **Line 4**: Set the path where energy site-per-site `.mat` files will be saved
     ```bash
     SITE_ENERGY_PATH="/full/path/to/energy/output/directory"
     ```

### 2. Directory Structure

Ensure the following directories exist:
```
├── logs/                    # For SLURM job logs (create if it does not exist)
├── checkpoint/              # For simulation checkpoints (create if it does not exist)
├── julia/                   # Julia source files
├── call_sbatch.sh          # Main submission script
├── job_sbatch.bash         # SLURM job script (should exist)
├── configuration.conf      # Your configured settings
└── README.md               # This file
```

### 3. Julia Dependencies

The following Julia packages should be installed before executing any simulaiton:
- CUDA
- Combinatorics
- Distributions
- MAT
- Serialization

Install them using the following commands:
- Launch the Julia environment
- Press `]` to enter the Pkg environment
- Execute ```install CUDA Combinatorics Distributions MAT Serialization```

## Running the Simulation

### Simulation Parameters

The main simulation parameters are configured in `call_sbatch.sh` (lines 3-9):

- **Line 3**: `npoint=30` - Number of lattice sites per side (creates 30×30 lattice)
- **Line 4**: `nmeas=1000` - Number of energy measurements
- **Line 5**: `NHMC=12` - Number of HMC integration steps
- **Line 6**: `dt=0.05` - Stochastic time-step size
- **Line 7**: `ncomp=14` - Number of independent components (O(N) with N=ncomp+1=15)
- **Line 8**: `ptord=10` - Maximum perturbative order
- **Line 9**: `every=7` - Measurement interval (skip 7 measurements between recordings)

### SLURM Resource Configuration

The SLURM job parameters are configured in `call_sbatch.sh` (lines 15-25), you can change these parameters accordingly to your needs and your SLURM environment availability:

- **GPU**: Requests 1 A100 80GB GPU (`--gres=gpu:a100_80g:1`)
- **Memory**: 40GB RAM allocation
- **Time**: 1 day runtime limit
- **Partition**: Uses `gpu` partition with `gpu` QoS

### Execution Options

The script provides three execution modes (lines 28-30):

1. **Start new simulation**:
   ```bash
   ./call_sbatch.sh
   ```
   Uncomment line 28 in `call_sbatch.sh`

2. **Start with site-per-site energy saving**:
   ```bash
   ./call_sbatch.sh
   ```
   Uncomment line 29 in `call_sbatch.sh` (default)

3. **Resume from checkpoint**:
   ```bash
   ./call_sbatch.sh
   ```
   Uncomment line 30 in `call_sbatch.sh`

