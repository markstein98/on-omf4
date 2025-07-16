using Serialization
using Random
using CUDA
using MAT

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

function build_energy_fname(n_comps, Npoint, dt, max_ptord, NHMC, n_meas, measure_every)
    folder = "../energies/"
    fname = folder * "O" * string(n_comps+1) * "_Npoint" * string(Npoint)
    fname *= "_dt" * string(dt) * "_ord" * string(max_ptord) * "_NHMC" * string(NHMC)
    fname *= "_nMeas" * string(n_meas) * "_every" * string(measure_every) * ".txt"
    return fname
end

function get_checkpoint_filename(energy_fname::AbstractString)
    fname = "../checkpoint" * energy_fname[findlast('/', energy_fname):end] # changes directory
    fname = fname[1:findlast('.', fname)] * "jld" # changes the extension
    return fname
end

function get_energy_filename(checkpt_fname::AbstractString)
    fname = "../energies" * checkpt_fname[findlast('/', checkpt_fname):end] # changes directory
    fname = fname[1:findlast('.', fname)] * "txt" # changes the extension
    return fname
end

function get_lat_checkpoint(lat_fname::AbstractString)
    fname = "../checkpoint" * lat_fname[findlast('/', lat_fname):end] # changes directory
    fname = fname[1:findlast('.', fname)] * "jld" # changes the extension
    return fname
end

function get_seconds(time::AbstractString)
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

function get_remaining_time(jobid::AbstractString)
    # return 1000 # REMOVE!!! Only for testing on non-SLURM environments
    t = read(`squeue -h -j $jobid -o "%L"`, String)
    return get_seconds(t)
end

function get_remaining_time()
    return get_remaining_time(get_job_id())
end

function save_state(fname::AbstractString, data)
    serialize(fname, data)
    println("State saved to file \"" * fname * "\"")
    return
end

function save_state(fname::AbstractString, data, lat_checkpt_fname::AbstractString, lat)
    serialize(fname, data)
    println("State saved to file \"" * fname * "\"")
    serialize(lat_checkpt_fname, lat)
    println("Lattice saved to file \"" * lat_checkpt_fname * "\"")
    return
end

function execute_self(fname::AbstractString)
    jname = get_job_name()
    bashfile = "./resume_sbatch.sh"
    command = `$bashfile $jname $fname`
    println("Re-executing itself with command: ", command)
    println(read(command, String))
    return
end

function execute_self(fname::AbstractString, lat_fname::AbstractString)
    jname = get_job_name()
    bashfile = "./resume_sbatch.sh"
    command = `$bashfile $jname $fname $lat_fname`
    println("Re-executing itself with command: ", command)
    println(read(command, String))
    return
end

function save_lat(lat_fname::AbstractString, lat)
    matwrite(lat_fname, Dict("energies"=>lat))
    println("Energy history written to file ", lat_fname)
    return
end

function remove_files(fnames...)
    println("Execution successful. Removing checkpoint files:")
    for fname in fnames
        if isfile(fname)
            rm(fname)
            println("Removed ", fname)
        end
    end
end

mutable struct OMF_args
    const Npoint::Int
    const n_meas::Int
    const NHMC::Int
    const dt::Real
    const n_comps::Int
    const max_ptord::Int
    const measure_every::Int
    const cuda_rng::CUDA.RNG
    const nhmc_rng::Random.TaskLocalRNG
    iter_start::Int
    x::CuArray
end
