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
    if jobid == "" # Only for testing on non-SLURM environments
        fname = "remaining_time.txt"
        if isfile(fname)
            f = open(fname)
            t = tryparse(Int, readline(f))
            close(f)
            t !== nothing && return t
        end
        return 100000
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

function execute_self(fname::String)
    jname = get_job_name()
    bashfile = "./resume_sbatch.sh"
    command = `$bashfile $jname $fname`
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

mutable struct OMF_args{F <: AbstractFloat, I <: Integer, I2 <: Integer}
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
end

function get_infos_string(sim_infos::OMF_args; n_decimals::Int=2, header::String="", sep::String="\n", prepend::String="", append::String="")
    head_spaces = " " ^ length(header)
    msg = header * "Simulation of O(" * string(sim_infos.n_comps + 1)
    msg *= ") sigma model up to perturbative order " * string(sim_infos.max_ptord) * " (included), on "
    msg *= string(sim_infos.n_copies) * " copies of a "
    msg *= string(sim_infos.Npoint) * "x" * string(sim_infos.Npoint) * " lattice." * sep
    msg *= head_spaces * "Computer-time integration steps of " * string(sim_infos.dt) * ", taking " * string(sim_infos.NHMC)
    msg *= " of them on average and saving a measurement every " * string(sim_infos.measure_every) * "." * sep
    msg *= head_spaces * "Starting from measurement #" * string(sim_infos.iter_start) * " out of " * string(sim_infos.n_meas)
    if n_decimals <= 0
        percentage = round(Int, sim_infos.iter_start / sim_infos.n_meas * 100)
    else
        percentage = round(sim_infos.iter_start / sim_infos.n_meas * 100 * 10^n_decimals) / 10^n_decimals
    end
    msg *= " (" * string(percentage) * "% of total)."
    return prepend * msg * append
end
