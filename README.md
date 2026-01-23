# O(N) Theory Simulation on 2D Lattice

This repository contains code to simulate an O(N) theory on a 2-dimensional lattice using the Numerical Stochastic Perturbation Theory (NSPT) framework. The simulation is implemented in Julia and is designed to run in a SLURM cluster environment.

## Overview

The code performs Monte Carlo simulations of O(N) quantum field theory on a 2D lattice using Hybrid Monte Carlo (HMC) algorithms. The simulation supports:
- Variable lattice sizes (npoint × npoint)
- Configurable O(N) sigma model with N = ncomp + 1 components
- Perturbative expansion up to specified orders
- Checkpoint/resume functionality
- Site-per-site energy analysis

## Prerequisites

- **SLURM Environment**: This program is intended to be executed inside a SLURM environment
- **Julia**: The simulation requires Julia runtime
- **GPU Support**: The default configuration requests A100 80GB GPU resources, but it can be changed accordingly to the resources available in the SLURM environment, inside the `call_sbatch.sh` and the `julia/resume_sbatch.sh` files

## Installation and Setup

### 1. SLURM Configuration File Setup

**REQUIRED**: Before running the simulation, you must edit the configuration file for the slurm environment:

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


### 2. Directory Structure

Ensure the following directories exist:
```
├── checkpoint/                # For simulation checkpoints (create if it does not exist)
├── logs/                      # For SLURM job logs (create if it does not exist)
├── julia/                     # Julia source files
├── simulation_configurations/ # For simulations' TOML configuration files
├── call_sbatch.sh             # Main submission script
├── job_sbatch.bash            # SLURM job script (should exist)
├── configuration.conf         # SLURM environment variables
└── README.md                  # This file
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

The main simulation parameters are configured in `simulation_configurations/configuration.conf` (the file can have any name).

#### Required Parameters

**Simulation Parameters**
- `n_side_sites`: Defines the spatial dimensions of the simulation lattice. The complete lattice will be a 2D square grid with a total of `n_side_sites`×`n_side_sites` sites.
- `n_measurements`: Total number of measurements to be taken during the simulation run.
- `n_HMC_steps`: Average number of Hybrid Monte Carlo steps used for each time evolution iteration.
- `dt`: The time step size (in computer time units) for numerical integration of the Langevin equation during evolution.
- `n_indep_comps`: Number of independent field components. The simulated theory will be O(N) where N = n_indep_comps + 1.
- `max_perturbative_order`: Maximum order in the perturbative expansion. The simulation includes effects up to g^`max_perturbative_order` in the coupling constant g, neglecting terms of order g^`(max_perturbative_order+1)` and higher.

**Output Files**
- `energy_filename`: Full path where mean energy measurements file will be saved (`.txt` format).
- `checkpoint_filename`: Full path for the checkpoint file (`.jld` format) to save and restore simulation state.

#### Optional Parameters
##### Defaults will be used if the corresponding parameter is not present (or is commented out) in the configuration file

- `measure_every`: Controls measurement frequency by saving one measurement for every `measure_every`, in order to decrease autocorrelation. (Default: 1)
- `n_copies`: Number of independent lattice configurations to simulate in parallel, allowing for better statistics. (Default: 1)
- `energy_site_by_site_matlab`: If specified, saves the spatial distribution of energy across all lattice sites in MATLAB format for detailed analysis. (Default: disabled)
- `max_saving_time`: Maximum time in seconds allocated for saving the simulation state before the program exits. (Default: 600 seconds or 10 minutes)
- `cuda_seed`: Seed for GPU random number generator for reproducibility. (Default: `0` -> random)
- `cpu_rng_seed`: Seed for CPU random number generator for reproducibility. (Default: `0` -> random)
- `intBits`: Bit precision for integer variables on GPU. Allowed values: 32 for Int32, or 64 for Int64. (Default: 32)
- `floatBits`: Bit precision for floating-point variables on GPU. Allowed values: 32 for Float32, or 64 for Float64. (Default: 32). Note: Mismatched `intBits` and `floatBits` values will cause performance issues and will trigger a warning from the program (it will execute nonetheless).

### SLURM Resource Configuration

The simulation is launched by the script `call_sbatch.sh` (make sure it is executable, with the command `chmod +x call_sbatch.sh`).

The SLURM job parameters are configured in `call_sbatch.sh` (lines 10-20), you can change these parameters accordingly to your needs and your SLURM environment availability:

- **GPU**: Requests 1 A100 80GB GPU (`--gres=gpu:a100_80g:1`)
- **Memory**: 80GB RAM allocation
- **Time**: 1 day runtime limit
- **Partition**: Uses `gpu` partition with `gpu` QoS

### Execution Options

The path to the configuraton file (to start the execution) or the checkpoint file (to resume a previously started execution) can be configured in the variables at lines 3-6.

The script provides two execution modes (lines 23-24):

1. **Start new simulation**:
   ```bash
   ./call_sbatch.sh
   ```
   Uncomment line 23 in `call_sbatch.sh`

2. **Resume from checkpoint**:
   ```bash
   ./call_sbatch.sh
   ```
   Uncomment line 24 in `call_sbatch.sh`

### Manual Execution

The simulation can also be launched manually (for testing) via the commands:
```bash
julia julia/ON_OMF_cluster.jl start path/to/configuration.toml
```
to start a new simulation, or
```bash
julia julia/ON_OMF_cluster.jl load path/to/checkpoint.jld
```
to resume an already started simulation.

# Warning
This version is not backwards-compatible with the previous versions, where the simulation configuration parameters were passed as arguments to the julia program.

## Checkpoint Adaptation
### Usage

Use the proper function in the julia/adapt_checkpoint.jl file to generate a new checkpoint file that is compatible with this version, by running julia from the project directory.

```julia
julia> include("julia/adapt_checkpoint.jl")
```

- If you need to adapt a checkpoint from a simulation **without** different copies:  

   ```julia
   julia> adapt_old_checkpoint("checkpoint/old_checkpoint.jld")
   ```

- If you need to adapt a checkpoint from a simulation **with** different copies:  

   ```julia
   julia> adapt_old_checkpoint_copies("checkpoint/old_checkpoint.jld")
   ```

### Optional arguments

For both functions, there is an optional argument, that is the path to the matlab file in which to save the energy site-by-site (if provided in the previous execution):
```julia
julia> adapt_old_checkpoint("checkpoint/old_checkpoint.jld", "/path/to/file.mat")
```

And also two optional keyword arguments:
```julia
julia> adapt_old_checkpoint("checkpoint/old_checkpoint.jld"; max_saving_time::Int, config_fname::String="")
```
- `max_saving_time` is the time allocated for saving, as explained in [configuration file's optional parameters](#optional-parameters). By default it is set to 600 s, or 10 min.
- `config_fname` is the path to the [configuration file](#simulation-parameters). It is just used to fill the correct field in the struct and will be printed in the log output, it will not be used by the simulation. By default it is set to an empty string.