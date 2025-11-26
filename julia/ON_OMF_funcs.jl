using CUDA
using Random
using Combinatorics
using Distributions
using Serialization
include("kernel_compilation.jl")
include("read_write_funcs.jl")
include("ON_OMF_kernels.jl")

function compute_gradient!(gx2::CompiledKernel, grad_ker::CompiledKernel)
    # calcola -gradiente (in parallelo) su ogni sito del reticolo e lo mette in gradient
    CUDA.@sync run_kernel(gx2)
    CUDA.@sync run_kernel(grad_ker)
    return nothing
end

function omf_evolution!(x::CuArray{F, 4}, Pi::CuArray{F, 4}, dt::F, gradient::CuArray{F, 4},
    gx2::CompiledKernel, grad_ker::CompiledKernel) where {F <: AbstractFloat}
    # esegue l'evoluzione di uno step secondo l'algoritmo OMF

    # costanti per l'algoritmo OMF
    ρ = F(+0.2750081212332419e+00)
    θ = F(-0.1347950099106792e+00)
    Θ = F(-0.8442961950707149e-01)
    λ = F(+0.3549000571574260e+00)

    CUDA.@sync x .+= (dt * ρ) .* Pi
    compute_gradient!(gx2, grad_ker)
    CUDA.@sync Pi .+= (dt * Θ) .* gradient
    CUDA.@sync x .+= (dt * θ) .* Pi
    compute_gradient!(gx2, grad_ker)
    CUDA.@sync Pi .+= (dt * λ) .* gradient
    CUDA.@sync x .+= (dt * 0.5 * (1 - 2*(θ+ρ))) .* Pi
    compute_gradient!(gx2, grad_ker)
    CUDA.@sync Pi .+= (dt * (1 - 2*(λ+Θ))) .* gradient
    CUDA.@sync x .+= (dt * 0.5 * (1 - 2*(θ+ρ))) .* Pi
    compute_gradient!(gx2, grad_ker)
    CUDA.@sync Pi .+= (dt * λ) .* gradient
    CUDA.@sync x .+= (dt * θ) .* Pi
    compute_gradient!(gx2, grad_ker)
    CUDA.@sync Pi .+= (dt * Θ) .* gradient
    CUDA.@sync x .+= (dt * ρ) .* Pi
    return nothing
end

function compute_energy!(
    energy::CuArray{F, 1},
    x::CuArray{F, 4},
    ener::CuArray{F, 3},
    zero_modo::CuArray{F, 2},
    Npoint::I,
    zero_mode::CompiledKernel,
    gx2::CompiledKernel,
    compute_ener::CompiledKernel
    ) where {F <: AbstractFloat, I <: Integer}
    # computes the mean of the energy of all sites at time t
    
    CUDA.@sync @inbounds @views zero_modo .= CUDA.sum(CUDA.sum(x, dims=3), dims=2)[:,1,1,:] ./ (Npoint^2)
    CUDA.fill!(ener, zero(F))

    # sottraggo lo zero modo da x (x = x - zero_modo) e calcolo l'energia media
    CUDA.@sync run_kernel(zero_mode)
    CUDA.@sync run_kernel(gx2)
    CUDA.@sync run_kernel(compute_ener)
    CUDA.@sync @inbounds @views energy .= CUDA.sum(CUDA.sum(ener, dims=3), dims=2)[:,1,1] ./ (Npoint^2)
    return nothing
end

@inline function reset_moment!(Pi::CuArray{F, 4}, rng::CUDA.RNG) where {F <: AbstractFloat}
    CUDA.fill!(Pi, zero(F))
    @inbounds @views CUDA.randn!(rng, Pi[1,:,:,:])
    return nothing
end

function main_omf(args::OMF_args{F, I, I2}, en_fname::AbstractString, lat_fname::AbstractString="") where {F <: AbstractFloat, I <: Integer, I2 <: Integer}
    
    # Determine if we should save lattice data
    save_lattice = lat_fname != ""
    
    # Extract (almost) all the arguments
    Npoint = args.Npoint
    n_meas = args.n_meas
    NHMC = args.NHMC
    dt = args.dt
    n_comps = args.n_comps
    max_ptord = args.max_ptord
    measure_every = args.measure_every
    x = args.x
    cuda_rng = args.cuda_rng
    nhmc_rng = args.nhmc_rng
    
    # Setup checkpoint filenames
    checkpt_fname = get_checkpoint_filename(en_fname)
    if save_lattice
        lat_checkpt = get_lat_checkpoint(lat_fname)
    end

    n_ords = max_ptord + one(max_ptord)
    Npoint2 = Npoint^2

    # Initialize additional variables
    x_back = CuArray{F}(undef, n_ords, Npoint, Npoint, n_comps)
    CUDA.copyto!(x_back, x)
    args.x = x_back
    Pi = CUDA.zeros(F, n_ords, Npoint, Npoint, n_comps)
    gradient = CUDA.zeros(F, n_ords, Npoint, Npoint, n_comps)
    x2 = CUDA.zeros(F, n_ords, Npoint, Npoint)

    energia = CUDA.zeros(F, n_ords)
    ener = CUDA.zeros(F, n_ords, Npoint, Npoint)
    zero_modo = CUDA.zeros(F, n_ords, n_comps)

    # Handle energy measurement array and file opening
    if args.iter_start == 1
        mean_energy_file = open_energy_file(en_fname, "w", true) # with header
        if save_lattice
            ener_meas = CUDA.zeros(F, n_ords, Npoint, Npoint, n_meas)
        end
    else
        mean_energy_file = open_energy_file(en_fname, "a", false) # without header, appending
        if save_lattice
            ener_meas = deserialize(lat_checkpt)
        end
    end

    # randomizzazione di NHMC
    binom_distr = Binomial(NHMC*10,1/10)

    # creazione delle funzioni a parametri fissati
    f_gx2, f_grad, f_zeromode, f_ener = create_kernels(F, Npoint, max_ptord, n_comps)

    # compilazione dei kernel e creazione di struct con i kernel compilati
    gx2 = compile_kernel(f_gx2, (x, x2), Npoint2)
    grad_ker = compile_kernel(f_grad, (x, gradient, x2), Npoint2)
    zero_mode = compile_kernel(f_zeromode, (x, zero_modo), Npoint2)
    compute_ener = compile_kernel(f_ener, (x, ener, x2), Npoint2)

    println(now(), ": Initialization and Kernel compilation successfully complete.")

    jobid = get_job_id()

    # Main computation loop
    for t = args.iter_start:n_meas
        for _ in 1:measure_every
            reset_moment!(Pi, cuda_rng)
            for __ = 1:rand(nhmc_rng, binom_distr)
                omf_evolution!(x, Pi, dt, gradient, gx2, grad_ker)
            end
        end
        
        compute_energy!(energia, x, ener, zero_modo, Npoint, zero_mode, gx2, compute_ener)
        
        # Store energy measurements if saving lattice
        if save_lattice
            @inbounds @views ener_meas[:,:,:,t] .= ener
        end
        
        write_line(mean_energy_file, Array(energia))
        x_back .= x
        args.iter_start = t + 1
        
        # Check remaining time and save state if needed
        if get_remaining_time(jobid) < 120
            # 2 min remaining, save state and exit
            if save_lattice
                save_state(checkpt_fname, args, lat_checkpt, ener_meas)
                execute_self(checkpt_fname, lat_fname)
            else
                save_state(checkpt_fname, args)
                execute_self(checkpt_fname)
            end
            return
        end
    end
    
    # Final cleanup and file operations
    close(mean_energy_file)
    println("Energy written to file ", en_fname)
    
    if save_lattice
        save_lat(lat_fname, Array(ener_meas))
        remove_files(checkpt_fname, lat_checkpt)
    else
        remove_files(checkpt_fname)
    end
    
    return
end

function launch_main_omf(
    Npoint::I1,
    n_meas::I1,
    NHMC::I1,
    dt::F,
    n_comps::I1,
    max_ptord::I1,
    measure_every::I1,
    lat_fname::AbstractString="";
    cuda_seed::I2=0,
    nhmc_seed::I3=0
    ) where {F <: AbstractFloat, I1 <: Integer, I2 <: Integer, I3 <: Integer}
    en_fname = build_energy_fname(n_comps, Npoint, dt, max_ptord, NHMC, n_meas, measure_every)
    checkpt_fname = get_checkpoint_filename(en_fname)
    println("Received simulation infos. Trying to see if there is already some data...")
    if isfile(checkpt_fname) && isfile(en_fname) && (lat_fname == "" || isfile(get_lat_checkpoint(lat_fname)))
        # Resume execution
        launch_main_omf(checkpt_fname, lat_fname)
        return
    end
    # Starts from scratch
    println("Some file with previous data was not found. Starting anew...")
    cuda_rng = cuda_seed == 0 ? CUDA.RNG() : CUDA.RNG(cuda_seed)
    nhmc_rng = Random.default_rng()
    nhmc_seed != 0 && Random.seed!(nhmc_rng, nhmc_seed)
    args = OMF_args(Npoint, n_meas, NHMC, dt, n_comps, max_ptord, measure_every, cuda_rng, nhmc_rng,
                    1, CUDA.zeros(F, max_ptord+1, Npoint, Npoint, n_comps))
    if lat_fname == ""
        main_omf(args, en_fname)
    else
        main_omf(args, en_fname, lat_fname)
    end
    return
end

function launch_main_omf(checkpt_fname::AbstractString, lat_fname::AbstractString="")
    # Resumes execution
    if !isfile(checkpt_fname)
        println("Error: file $checkpt_fname not found. Exiting")
        return
    end
    en_fname = get_energy_filename(checkpt_fname)
    if !isfile(en_fname)
        println("Error: file $en_fname not found. Exiting")
        return
    end
    if lat_fname != ""
        lat_checkpt = get_lat_checkpoint(lat_fname)
        if !isfile(lat_checkpt)
            println("Error: file $lat_checkpt not found. Exiting")
            return
        end
    end
    println("Found necessary files. Resuming execution...")
    args = deserialize(checkpt_fname)
    if lat_fname == ""
        main_omf(args, en_fname)
    else
        main_omf(args, en_fname, lat_fname)
    end
    return
end
