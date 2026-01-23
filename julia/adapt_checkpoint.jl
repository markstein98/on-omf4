using CUDA
using Serialization

include("read_write_funcs.jl")

mutable struct old_OMF_args_copies{F <: AbstractFloat, I <: Integer, I2 <: Integer}
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
    iter_start::I2
    x::CuArray{F, 5}
end

mutable struct old_OMF_args{F <: AbstractFloat, I <: Integer, I2 <: Integer}
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

function get_energy_filename(checkpt_fname::String)
    fname = "../energies" * checkpt_fname[findlast('/', checkpt_fname):end] # changes directory
    fname = fname[1:findlast('.', fname)] * "txt" # changes the extension
    return fname
end

function get_lat_checkpoint(lat_fname::String)
    fname = "../checkpoint" * lat_fname[findlast('/', lat_fname):end] # changes directory
    fname = fname[1:findlast('.', fname)] * "jld" # changes the extension
    return fname
end

function adapt_old_checkpoint_copies(old_checkpoint_fname::String, lat_fname::String=""; config_fname::String="", max_saving_time::Int=600)
    new_checkpoint_fname = old_checkpoint_fname[1:end-4] * "_new.jld"
    lat_chkpt = get_lat_checkpoint(lat_fname)
    lat_present = false
    if lat_fname != "" && isfile(lat_chkpt)
        println("Energy site-by-site checkpoint found.")
        lat_present = true
    elseif lat_fname != ""
        println("Energy site-by-site checkpoint NOT found: ", lat_chkpt)
    end
    old_args::old_OMF_args_copies = deserialize(old_checkpoint_fname)
    new_args = OMF_args(
        old_args.Npoint,
        old_args.n_meas,
        old_args.NHMC,
        old_args.dt,
        old_args.n_comps,
        old_args.max_ptord,
        old_args.measure_every,
        old_args.n_copies,
        old_args.cuda_rng,
        old_args.nhmc_rng,
        get_energy_filename(old_checkpoint_fname),
        config_fname,
        new_checkpoint_fname,
        max_saving_time,
        old_args.iter_start,
        old_args.x,
        lat_present ? nothing : lat_fname,
        lat_present ? nothing : deserialize(lat_chkpt)
    )
    serialize(new_checkpoint_fname, new_args)
    println("New checkpoint saved to file: ", new_checkpoint_fname)
    return
end

function adapt_old_checkpoint(old_checkpoint_fname::String, lat_fname::String=""; config_fname::String="", max_saving_time::Int=600)
    new_checkpoint_fname = old_checkpoint_fname[1:end-4] * "_new.jld"
    lat_chkpt = get_lat_checkpoint(lat_fname)
    lat_present = false
    if lat_fname != "" && isfile(lat_chkpt)
        println("Energy site-by-site checkpoint found.")
        old_ener_meas = deserialize(lat_chkpt)
        lat_present = true
    elseif lat_fname != ""
        println("Energy site-by-site checkpoint NOT found: ", lat_chkpt)
    end
    old_args::old_OMF_args = deserialize(old_checkpoint_fname)
    new_args = OMF_args(
        old_args.Npoint,
        old_args.n_meas,
        old_args.NHMC,
        old_args.dt,
        old_args.n_comps,
        old_args.max_ptord,
        old_args.measure_every,
        one(old_args.n_comps),
        old_args.cuda_rng,
        old_args.nhmc_rng,
        get_energy_filename(old_checkpoint_fname),
        config_fname,
        new_checkpoint_fname,
        max_saving_time,
        old_args.iter_start,
        reshape(old_args.x, size(old_args.x)..., 1),
        lat_present ? nothing : lat_fname,
        lat_present ? nothing : reshape(old_ener_meas, (size(old_ener_meas)[1:3]..., 1, size(old_ener_meas)[4:end]...))
    )
    serialize(new_checkpoint_fname, new_args)
    println("New checkpoint saved to file: ", new_checkpoint_fname)
    return
end
