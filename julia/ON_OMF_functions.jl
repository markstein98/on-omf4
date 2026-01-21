using CUDA
using Random
using Combinatorics
using Distributions
using Serialization
include("kernel_compilation.jl")
include("read_write_funcs.jl")
include("ON_OMF_kernels.jl")
include("load_config.jl")

function compute_gradient!(gx2::Array{CompiledKernel}, grad_ker::Array{CompiledKernel})
    # calcola -gradiente (in parallelo) su ogni sito del reticolo e lo mette in gradient
    CUDA.@sync begin  
        for i in eachindex(gx2)  
            @inbounds run_kernel(gx2[i])
        end
    end
    CUDA.@sync begin
        for i in eachindex(grad_ker)
            @inbounds run_kernel(grad_ker[i])
        end 
    end
    return nothing
end

function omf_evolution!(x::CuArray{F, 5}, Pi::CuArray{F, 5}, dt::F, gradient::CuArray{F, 5},
    gx2::Array{CompiledKernel}, grad_ker::Array{CompiledKernel}) where {F <: AbstractFloat}
    # esegue l'evoluzione di uno step secondo l'algoritmo OMF

    # costanti per l'algoritmo OMF
    ρ = F(+0.2750081212332419e+00)
    θ = F(-0.1347950099106792e+00)
    Θ = F(-0.8442961950707149e-01)
    λ = F(+0.3549000571574260e+00)

    # queste che seguono sono element-wise operations, quindi in teoria non devo cambiare

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
    energy::CuArray{F, 2},
    x::CuArray{F, 5},
    ener::CuArray{F, 4},
    zero_modo::CuArray{F, 3},
    Npoint::I,
    zero_mode::Array{CompiledKernel},
    gx2::Array{CompiledKernel},
    compute_ener::Array{CompiledKernel}
    ) where {F <: AbstractFloat, I <: Integer}
    # computes the mean of the energy of all sites at time t
   
    CUDA.@sync @inbounds @views zero_modo .= CUDA.sum(CUDA.sum(x, dims=3), dims=2)[:,1,1,:,:] ./ (Npoint^2)
    CUDA.fill!(ener, zero(F))

    # sottraggo lo zero modo da x (x = x - zero_modo) e calcolo l'energia media
    CUDA.@sync begin 
        for i in eachindex(zero_mode)
            @inbounds run_kernel(zero_mode[i])
        end 
    end 

    CUDA.@sync begin
        for i in eachindex(gx2)
            @inbounds run_kernel(gx2[i])
        end 
    end 

    CUDA.@sync begin
        for i in eachindex(compute_ener)
            @inbounds run_kernel(compute_ener[i])
        end
    end 
    
    CUDA.@sync @inbounds @views energy.= CUDA.sum(CUDA.sum(ener, dims=3), dims=2)[:,1,1,:] ./ (Npoint^2)
    return nothing
end

@inline function reset_moment!(Pi::CuArray{F, 5}, rng::CUDA.RNG) where {F <: AbstractFloat}
    CUDA.fill!(Pi, zero(F))
    @inbounds @views CUDA.randn!(rng, Pi[1,:,:,:,:])
    return nothing
end

function main_omf(args::OMF_args_copies{F, I, I2}) where {F <: AbstractFloat, I <: Integer, I2 <: Integer}
    
    println(current_time(), "Variables initialization started.")

    # Extract (almost) all the arguments
    Npoint        = args.Npoint
    n_meas        = args.n_meas
    NHMC          = args.NHMC
    dt            = args.dt
    n_comps       = args.n_comps
    n_copies      = args.n_copies
    max_ptord     = args.max_ptord
    measure_every = args.measure_every
    x             = args.x
    cuda_rng      = args.cuda_rng
    nhmc_rng      = args.nhmc_rng
    en_fname      = args.en_fname
    checkpt_fname = args.checkpt_fname

    # Determine if we should save lattice data
    save_lattice = args.lat_fname !== nothing
    if save_lattice
        lat_fname::String = args.lat_fname
        ener_meas::CuArray{F, 5} = args.ener_meas
    end
    
    n_ords = max_ptord + one(max_ptord)
    Npoint2 = Npoint ^ 2

    # Initialize additional variables
    x_back = CuArray{F}(undef, n_ords, Npoint, Npoint, n_comps, n_copies)
    CUDA.copyto!(x_back, x) # copy x into x_back; 
    args.x = x_back
    Pi = CUDA.zeros(F, n_ords, Npoint, Npoint, n_comps, n_copies)
    gradient = CUDA.zeros(F, n_ords, Npoint, Npoint, n_comps, n_copies)
    x2 = CUDA.zeros(F, n_ords, Npoint, Npoint, n_copies)

    energia = CUDA.zeros(F, n_ords, n_copies)
    ener = CUDA.zeros(F, n_ords, Npoint, Npoint, n_copies)
    zero_modo = CUDA.zeros(F, n_ords, n_comps, n_copies)

    gx2 = Array{CompiledKernel}(undef, n_copies); 
    grad_ker = Array{CompiledKernel}(undef,n_copies); 
    zero_mode = Array{CompiledKernel}(undef,n_copies); 
    compute_ener = Array{CompiledKernel}(undef,n_copies); 

    en_fnames = [en_fname]
    for i in 2:n_copies
        push!(en_fnames, get_copy_energy_filename(en_fname, i))
    end

    # Handle energy measurement array and file opening
    if args.iter_start == 1
        mean_energy_files = [open_energy_file(en_fname, "w", true)] # with header
        for i in 2:n_copies
            push!(mean_energy_files, open_energy_file(en_fnames[i], "w", true))
        end
    else
        mean_energy_files = [open_energy_file(en_fname, "a", false)] # without header, appending
        for i in 2:n_copies
            push!(mean_energy_files, open_energy_file(en_fnames[i], "a", false))
        end
    end

    # randomization of NHMC
    binom_distr = Binomial(NHMC*10,1/10)

    # creation of fixed-parameters functions
    f_gx2, f_grad, f_zeromode, f_ener = create_kernels(F, Npoint, max_ptord, n_comps)

    # kernel compilation and compiled-kernel structs creation
    for i in eachindex(gx2)
        @inbounds @views gx2[i] = compile_kernel(f_gx2, (x[:,:,:,:,i], x2[:,:,:,i]), Npoint2)
        @inbounds @views grad_ker[i] = compile_kernel(f_grad, (x[:,:,:,:,i], gradient[:,:,:,:,i], x2[:,:,:,i]), Npoint2)
        @inbounds @views zero_mode[i] = compile_kernel(f_zeromode, (x[:,:,:,:,i], zero_modo[:,:,i]), Npoint2)
        @inbounds @views compute_ener[i] = compile_kernel(f_ener, (x[:,:,:,:,i], ener[:,:,:,i], x2[:,:,:,i]), Npoint2)
    end
    
    println(current_time(), "Variables initialization and Kernel compilation successfully completed.")

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
            @inbounds @views ener_meas[:,:,:,:,t] .= ener
        end
        
        for i in 1:n_copies
            @inbounds @views write_line(mean_energy_files[i], Array(energia[:, i]))
        end
        x_back .= x
        args.iter_start = t + 1
        
        # Check remaining time and save state if needed
        if get_remaining_time(jobid) < args.max_saving_time
            # time's almost up, save state and exit
            println(current_time(), "Saving status.")
            save_state(checkpt_fname, args)         
            println(current_time(), "Saving status completed.")
            execute_self(checkpt_fname)
            return
        end
    end
    
    # Final cleanup and file operations
    for i in 1:n_copies
        close(mean_energy_files[i])
        print(current_time(), "Mean energy ")
        n_copies > 1 && print("of copy $i ")
        println("written to file: ", en_fnames[i])
    end

    # Saving energy site-by-site on matlab file
    save_lattice && save_matlab_energy(lat_fname, Array(ener_meas))
    println(current_time(), "Execution successfully terminated.")
    return
end

function launch_main_omf(config_fname::String)
    # Starts from scratch
    # Checking if configuration file exists and has all the mandatory parameters
    if !isfile(args[2])
        error("Configuration file not found: ", args[1])
    end
    check_required_keys(args[2], true)
    curr_time = current_time()
    println(curr_time, "Configuration file found.")
    curr_time = " "^(length(curr_time)-length("[INFO]: ")) * "[INFO]: "
    println(curr_time, "Configuration file: ", config_fname)
    # Load config file
    conf = parse_config_file(config_fname)
    # checking writeability of checkpoint and energy filenames
    if !is_file_writeable(conf.checkpt_fname)
        error("Checkpoint file ", conf.checkpt_fname, " is not writeable.")
    end
    if !is_file_writeable(conf.en_fname)
        error("Energy file ", conf.en_fname, " is not writeable.")
    end
    # Setting specific rng seeds, if provided
    cuda_rng = conf.cuda_seed == 0 ? CUDA.RNG() : CUDA.RNG(conf.cuda_seed)
    nhmc_rng = Random.default_rng()
    conf.cpu_rng_seed != 0 && Random.seed!(nhmc_rng, conf.cpu_rng_seed)
    # Arguments initialization
    floatType = typeof(conf.dt)
    n_ords = conf.max_ptord + one(conf.max_ptord)
    args = OMF_args_copies(
        conf.Npoint, conf.n_meas, conf.NHMC, conf.dt, conf.n_comps, conf.max_ptord, conf.measure_every, conf.n_copies,
        cuda_rng, nhmc_rng, conf.en_fname, config_fname, conf.checkpt_fname, conf.max_saving_time, one(conf.Npoint),
        CUDA.zeros(floatType, n_ords, conf.Npoint, conf.Npoint, conf.n_comps, conf.n_copies),
        conf.lat_file,
        conf.lat_file == nothing ? nothing : CUDA.zeros(F, n_ords, conf.Npoint, conf.Npoint, conf.n_copies, conf.n_meas)
    )
    println(curr_time, "Checkpoint file: ", args.checkpt_fname)
    println(curr_time, "Energy file: ", args.en_fname)
    println(curr_time, "Starting new simulation...")
    main_omf(args)
    return
end

function resume_main_omf(checkpt_fname::String)
    # Resumes execution
    if !isfile(checkpt_fname)
        error("Checkpoint file $checkpt_fname not found.")
        return
    end
    # Load config file
    args = deserialize(checkpt_fname)
    curr_time = current_time()
    println(curr_time, "Checkpoint file found.")
    curr_time = " "^(length(curr_time)-length("[INFO]: ")) * "[INFO]: "
    println(curr_time, "Configuration file: ", args.config_fname, " (could be outdated).")
    println(curr_time, "Checkpoint file: ", checkpt_fname)
    println(curr_time, "Energy file: ", args.en_fname)
    println(curr_time, "Resuming execution...")
    main_omf(args)
    return
end