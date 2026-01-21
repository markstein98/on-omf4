using Serialization
using Random
using CUDA
using MAT
using Dates

function current_time()
    return string(now()) * ": "
end

function open_energy_file(fname::String, mode::String="w", header::Bool=false)
    file = open(fname, mode)
    if header # prints header if opening new file
        println(file, "# Each line is an energy measure,")
        println(file, "# each column corresponds to the perturbative order: from 0 to the max\n")
    end
    return file
end

function write_line(file::IOStream, array::Vector{T}) where {T}
    for j in 1:size(array)[1]
        @inbounds print(file, array[j], " ")
    end
    println(file)
end

function get_copy_energy_filename(energy_filename, n_copy)
    if n_copy < 2
        return energy_filename
    end
    return energy_filename[1:end-4] * "_copy" * string(n_copy) * ".txt"
end

function build_energy_fname(n_comps, Npoint, dt, max_ptord, NHMC, n_meas, measure_every)
    folder = "../energies/"
    fname = folder * "O" * string(n_comps+1) * "_Npoint" * string(Npoint)
    fname *= "_dt" * string(dt) * "_ord" * string(max_ptord) * "_NHMC" * string(NHMC)
    fname *= "_nMeas" * string(n_meas) * "_every" * string(measure_every) * ".txt"
    return fname
end

function get_checkpoint_filename(energy_fname::String)
    fname = "../checkpoint" * energy_fname[findlast('/', energy_fname):end] # changes directory
    fname = fname[1:findlast('.', fname)] * "jld" # changes the extension
    return fname
end

function get_energy_filename(checkpt_fname::String)
    fname = "../energies" * checkpt_fname[findlast('/', checkpt_fname):end] # changes directory
    fname = fname[1:findlast('.', fname)] * "txt" # changes the extension
    return fname
end

function get_lat_checkpoint(lat_fname::String) # TODO: fix '/' bug if '/' is not present in lat_fname
    fname = "../checkpoint" * lat_fname[findlast('/', lat_fname):end] # changes directory
    fname = fname[1:findlast('.', fname)] * "jld" # changes the extension
    return fname
end

function get_seconds(time::String)
    seconds = 0
    sep = findfirst('-', time)
    if !(sep === nothing)
        seconds += parse(Int, time[1:sep-1]) * 24 * 60 * 60 # days
        time = time[sep+1:end]
    end
    i = 0
    for t in reverse(split(time, ':'))
        seconds += parse(Int, t) * 60 ^ i # sec, min, hours
        i += 1
    end
    return seconds
end

function get_job_id()
    return get(ENV, "SLURM_JOB_ID", "")
end

function get_job_name()
    return get(ENV, "SLURM_JOB_NAME", "")
end

function get_remaining_time(jobid::String)
    if jobid == ""
        return 1000 # Only for testing on non-SLURM environments
    end
    t = read(`squeue -h -j $jobid -o "%L"`, String)
    return get_seconds(t)
end

function get_remaining_time()
    return get_remaining_time(get_job_id())
end

function save_state(fname::String, data)
    serialize(fname, data)
    println(current_time(), "State saved to file \"" * fname * "\"")
    return
end

function save_state(fname::String, data, lat_checkpt_fname::String, lat)
    serialize(fname, data)
    println(current_time(), "State saved to file \"" * fname * "\"")
    serialize(lat_checkpt_fname, lat)
    println(current_time(), "Lattice saved to file \"" * lat_checkpt_fname * "\"")
    return
end

function execute_self(fname::String)
    jname = get_job_name()
    bashfile = "./resume_sbatch.sh"
    command = `$bashfile $jname $fname`
    exec_time = current_time()
    println(exec_time, "Re-executing itself with command: ", command)
    println(" "^length(exec_time), read(command, String))
    return
end

function execute_self(fname::String, lat_fname::String)
    jname = get_job_name()
    bashfile = "./resume_sbatch.sh"
    command = `$bashfile $jname $fname $lat_fname`
    exec_time = current_time()
    println(exec_time, "Re-executing itself with command: ", command)
    println(" "^length(exec_time), read(command, String))
    return
end

function save_matlab_energy(lat_fname::String, lat)
    matwrite(lat_fname, Dict("energies"=>lat))
    println(current_time(), "Energy history written to file ", lat_fname)
    return
end

function remove_files(fnames...)
    println(current_time(), "Execution successful.")
    for fname in fnames
        if isfile(fname)
            rm(fname)
            println(current_time(), "Removed file:", fname)
        end
    end
end

mutable struct OMF_args_copies{F <: AbstractFloat, I <: Integer, I2 <: Integer}
    const Npoint::I
    const n_meas::I
    const NHMC::I
    const dt::F
    const n_comps::I
    const max_ptord::I
    const measure_every::I
    const n_copies::I
    const cuda_rng::CUDA.RNG
    const nhmc_rng::Random.TaskLocalRNG
    const en_fname::String
    const config_fname::String
    const checkpt_fname::String
    const max_saving_time::I2
    iter_start::I
    x::CuArray{F, 5}
    const lat_fname::Union{Nothing, String}
    ener_meas::Union{Nothing, CuArray{F, 5}}
    # OMF_args_copies{F <: AbstractFloat, I <: Integer}(
    #     Npoint::I, n_meas::I, NHMC::I, dt::F, n_comps::I, max_ptord::I, measure_every::I, n_copies::I,
    #     cuda_rng::CUDA.RNG, nhmc_rng::Random.TaskLocalRNG,
    #     en_fname::String, config_fname::String, iter_start::I, x::CuArray{F, 5},
    #     lat_fname::Union{Nothing, String} = nothing, ener_meas::Union{Nothing, CuArray{F, 5}} = nothing
    # ) = new(
    #     Npoint, n_meas, NHMC, dt, n_comps, max_ptord, measure_every, n_copies, cuda_rng, nhmc_rng,
    #     en_fname, config_fname, iter_start, x, lat_fname, ener_meas
    # )
end

mutable struct OMF_args{F <: AbstractFloat, I <: Integer, I2 <: Integer}
    const Npoint::I
    const n_meas::I
    const NHMC::I
    const dt::F
    const n_comps::I
    const max_ptord::I
    const measure_every::I
    const cuda_rng::CUDA.RNG
    const nhmc_rng::Random.TaskLocalRNG
    iter_start::I2
    x::CuArray{F, 4}
end

# Constructor to automatically match integer type to float type
# function OMF_args_copies{F}(
#     Npoint, n_meas, NHMC, dt::F, n_comps, max_ptord, measure_every, n_copies,
#     cuda_rng, nhmc_rng, en_fname, config_fname, lat_fname,
#     iter_start, x, ener_meas
# ) where {F <: AbstractFloat}
#     I = F == Float32 ? Int32 : Int64
    
#     return OMF_args_copies{F, I}(
#         I(Npoint), 
#         I(n_meas), 
#         I(NHMC), 
#         dt, 
#         I(n_comps), 
#         I(max_ptord), 
#         I(measure_every), 
#         I(n_copies),
#         cuda_rng, 
#         nhmc_rng, 
#         en_fname, 
#         config_fname, 
#         lat_fname,
#         I(iter_start), 
#         x, 
#         ener_meas
#     )
# end
