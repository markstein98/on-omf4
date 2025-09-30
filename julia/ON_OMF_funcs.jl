using CUDA
using Random
using Combinatorics
using Distributions
using Serialization
include("kernel_compilation.jl")
include("read_write_funcs.jl")

# di seguito, j è l'indice per l'ordine perturbativo
# i e k sono prima e seconda componente delle coordinate sul reticolo
# n ∈ {1, ..., N-1} è il numero della componente, dove N è quella di O(N)

@inline function set_unity!(v::CuDeviceArray{F, 3, 1}, i::I, k::I) where {F <: AbstractFloat, I <: Integer}
    @inbounds v[:, i, k] .= 0
    @inbounds v[1, i, k] = 1
end

@inline function set_unity_vec!(v::CuDeviceArray{F, 4, 1}, i::I, k::I, n::I) where {F <: AbstractFloat, I <: Integer}
    @inbounds v[:, i, k, n] .= 0
    @inbounds v[1, i, k, n] = 1
end

function create_kernels(::Type{F}, Npoint::I, max_ptord::I, n_comps::I) where {F <: AbstractFloat, I <: Integer}

    n_ptords = max_ptord + one(I) # total number of perturbative orders
    unity = CUDA.zeros(F, n_ptords)
    @inbounds CUDA.@allowscalar unity[1] = one(F)
    coupling = CUDA.zeros(F, n_ptords, Npoint, Npoint)
    @inbounds CUDA.@allowscalar coupling[2,:,:] .= one(F)
    
    # Buffers used in computations
    vec_buffer = CuArray{F}(undef, n_ptords, Npoint, Npoint, n_comps)
    scalar_buffs = CuArray{F}(undef, n_ords, Npoint, Npoint, 4)
    @inbounds sc_buffA = @view scalar_buffs[:,:,:,1]
    @inbounds sc_buffB = @view scalar_buffs[:,:,:,2]
    @inbounds sc_buffC = @view scalar_buffs[:,:,:,3]
    @inbounds sc_buffD = @view scalar_buffs[:,:,:,4]
    scalar_buffs2 = CuArray{F}(undef, n_ords, Npoint, Npoint, 4)
    fraction = CuArray{F}(undef, n_ords, Npoint, Npoint)

    # Funcitons returning the coordinates of the nearest neighbour of site (i, k) in the given direction
    @inline function neighbour_up(i::I, k::I)
        return (mod(i-I(2), Npoint) + one(I), k)
    end

    @inline function neighbour_down(i::I, k::I)
        return (mod(i, Npoint) + one(I), k)
    end

    @inline function neighbour_left(i::I, k::I)
        return (i, mod(k-I(2), Npoint) + one(I))
    end

    @inline function neighbour_right(i::I, k::I)
        return (i, mod(k, Npoint) + one(I))
    end

    function get_indexes()
        # converts from one-dimensional index ind to two-dimensional indexes i, k
        ind = (blockIdx().x - one(T)) * blockDim().x + threadIdx().x
        i::I = rem(ind-one(I), Npoint) + one(I)
        k::I = div(ind-one(I), Npoint) + one(I)
        return (i, k)
    end

    @inline function mult_doppia!(
        a::CuDeviceArray{F, 4, 1}, b::CuDeviceArray{F, 4, 1}, result::CuDeviceArray{F, 4, 1},
        i::I, k::I, i2::I, k2::I, n::I, n2::I
        )
        # questa moltiplicazione prende gli indici a coppie 
        # così si possono moltiplicare cose provenienti da siti diversi
        # i risultati sono memorizzati usando gli indici del primo
        @inbounds result[:, i, k, n] .= zero(F)
        for j::I = 1:n_ptords
            for l::I = 1:j
                @inbounds result[j, i, k, n] += a[l, i, k, n] * b[j-l+one(I), i2, k2, n2]
            end
        end
        return
    end

    @inline function mult!(
        a::CuDeviceArray{F, 4, 1}, b::CuDeviceArray{F, 4, 1}, result::CuDeviceArray{F, 4, 1}, i::I, k::I, n::I
        )
        # mette a*b in result
        mult_doppia!(a, b, result, i, k, i, k, n, n)
        return
    end

    @inline function mult_sc!(
        a::CuDeviceArray{F, 3, 1}, b::CuDeviceArray{F, 3, 1}, result::CuDeviceArray{F, 3, 1}, i::I, k::I
        )
        # questa moltiplicazione è pensata per moltiplicare dei moduli quadri
        # sono ancora oggetti perturbativi con la struttura di reticolo
        # ma sono scalari quindi non c'è l'indice di componente
        # reset_sc!(result, i, k)
        @inbounds result[:, i, k] .= zero(F)
        for j::I = 1:n_ptords
            for l::I = 1:j
                @inbounds result[j, i, k] += a[l, i, k] * b[j-l+one(I), i, k]
            end
        end
        return
    end

    @inline function mult_vec_sc_all_comps!(
        a::CuDeviceArray{F, 4, 1}, b::CuDeviceArray{F, 3, 1}, result::CuDeviceArray{F, 4, 1}, i::I, k::I
        )
        # questa moltiplicazione prende il primo oggetto vettoriale, il secondo scalare
        @inbounds result[:, i, k, :] .= 0
        for j::I = 1:n_ptords
            for l::I = 1:j
                @inbounds @views result[j, i, k, :] .+= a[l, i, k, :] .* b[j-l+one(I), i, k]
            end
        end
        return
    end

    @inline function power!(
        x::CuDeviceArray{F, 3, 1},
        esp::I, i::I, k::I,
        result_even::CuDeviceArray{F, 3, 1},
        result_odd::CuDeviceArray{F, 3, 1}
        )
        # se n è dispari ci sarà il risultato corretto in result_odd
        # se n è pari il risultato sarà in result_even
        # se metto ad esempio 6 calcolerò comunque fino a 7
        @inbounds @views result_odd[:, i, k] .= x[:, i, k]
        @inbounds result_even[:, i, k] .= zero(F)
        for _ = 2:2:esp
            mult_sc!(x, result_odd, result_even, i, k)
            mult_sc!(x, result_even, result_odd, i, k)
        end
        return
    end

    @inline function unmenx!(
        x::CuDeviceArray{F, 3, 1},
        i::I, k::I, i2::I, k2::I,
        result::CuDeviceArray{F, 3, 1}
        )
        # mette in result la sottrazione unity - x
        @inbounds @views result[:, i, k] .= unity .- x[:, i2, k2]
        return
    end

    function g_quadro_kernel!(x::CuDeviceArray{F, 4, 1}, gx2::CuDeviceArray{F, 3, 1})
        # calcola in parallelo gx^2 su tutto il reticolo e lo mette in gx2
        i, k = get_indexes()
        if (i <= Npoint && k <= Npoint)
            @inbounds gx2[:, i, k] .= zero(F)
            for n::I = 1:n_comps
                mult!(x, x, vec_buffer, i, k, n)
                @inbounds @views gx2[I(2):n_ptords, i, k] .+= vec_buffer[one(I):max_ptord, i, k, n]
            end
        end
        return nothing
    end

    @inline function frazione!(
        x::CuDeviceArray{F, 3, 1},
        i::I,
        k::I,
        risultato::CuDeviceArray{F, 3, 1},
        potenza_pari::CuDeviceArray{F, 3, 1},
        potenza_dispari::CuDeviceArray{F, 3, 1}
        )
        #qui calcolo lo sviluppo di Taylor di 1/(1-gx)
        set_unity!(risultato, i, k)
        # @inbounds risultato[:, i, k] .= unity # more efficient than previous line for greater n_ptords, to be tested
        # TODO: test this
        for esp::I = 1:2:n_ptords
            power!(x, esp, i, k, potenza_pari, potenza_dispari)
            for j::I = 1:n_ptords
                @inbounds risultato[j,i,k] += potenza_pari[j,i,k] + potenza_dispari[j,i,k]
            end
        end
    end

    @inline function radice!(
        x::CuDeviceArray{F, 3, 1},
        i::I,
        k::I,
        n::I,
        risultato::CuDeviceArray{F, 4, 1},
        potenza_pari::CuDeviceArray{F, 3, 1},
        potenza_dispari::CuDeviceArray{F, 3, 1},
        xmenun::CuDeviceArray{F, 3, 1}
        )
        # svilupo di taylor di √x attorno a x=(1,0,...,0) e lo metto in risultato[:,:,:,n] (n∈{1,...,4} rappresenta 1:dx, 2:sx, 3:su, 4:giù)
        set_unity_vec!(risultato, i, k, n)
        # TODO: same as set_unity in function frazione!
        @inbounds @views xmenun[:, i, k] .= x[:, i, k] .- unity
        for esp::I = 1:2:n_ptords
            power!(xmenun, esp, i, k, potenza_pari, potenza_dispari)
            for j::I = 1:n_ptords
                @inbounds risultato[j,i,k,n] += binomial(1/2,esp-one(I))*potenza_pari[j,i,k] + binomial(1/2,esp)*potenza_dispari[j,i,k]
            end
        end
    end

    function compute_ener_kernel!(x::CuDeviceArray{F, 4, 1}, ener::CuDeviceArray{F, 3, 1}, x2::CuDeviceArray{F, 3, 1})

        i, k = get_indexes()

        if (i <= Npoint && k <= Npoint)
            down = neighbour_down(i, k)
            right = neighbour_right(i, k)
            for n::I = 1:n_comps
                #per l'energia io sommo i contributi di link destro e link sotto
                #poi divido per 2, così conto tutti i link e faccio la media
                #alla fine dovrò comunque dividere per Npoint^2
                #sulle n_comps sommo perchè tanto è un prodotto scalare
                #poi c'è un g davanti quindi guardo un j indietro
                mult_doppia!(x, x, vec_buffer, i, k, right..., n, n)
                @inbounds @views ener[I(2):n_ptords,i,k] .+= F(0.5) .* vec_buffer[I(1):max_ptord,i,k,n]
                mult_doppia!(x, x, vec_buffer, i, k, down..., n, n)
                @inbounds @views ener[I(2):n_ptords,i,k] .+= F(0.5) .* vec_buffer[I(1):max_ptord,i,k,n]
            end

            #poi sommo le radici fuori dal ciclo su n
            #anche qui salvo i risultati sulle diverse n_comps di rad
            unmenx!(x2, i, k, i, k, sc_buffA)
            radice!(sc_buffA, i, k, I(1), scalar_buffs2, sc_buffB, sc_buffC, sc_buffD)

            unmenx!(x2, i, k, right..., sc_buffA)
            radice!(sc_buffA, i, k, I(2), scalar_buffs2, sc_buffB, sc_buffC, sc_buffD)

            unmenx!(x2, i, k, down..., sc_buffA)
            radice!(sc_buffA, i, k, I(3), scalar_buffs2, sc_buffB, sc_buffC, sc_buffD)

            #così troverò i 2 risultati importanti in
            #scalar_buffs[:,i,k,1] e scalar_buffs[:,i,k,3]
            mult_doppia!(scalar_buffs2, scalar_buffs2, scalar_buffs, i, k, i, k, I(1), I(2))
            mult_doppia!(scalar_buffs2, scalar_buffs2, scalar_buffs, i, k, i, k, I(3), I(1))

            @inbounds @views ener[:,i,k] .+= F(0.5) .* (scalar_buffs[:,i,k,I(1)] .+ scalar_buffs[:,i,k,I(3)])
        end
        return nothing
    end

# TODO: merge the following two functions in a single kernel (also with the instruction at line 357), first scalar, then vector
    
    function gradient_kernel!(x::CuDeviceArray{F, 4, 1}, gradient::CuDeviceArray{F, 4, 1}, gx2::CuDeviceArray{F, 3, 1})
        
        i, k = get_indexes()

        if (i <= Npoint && k <= Npoint)
            up = neighbour_up(i, k)
            down = neighbour_down(i, k)
            right = neighbour_right(i, k)
            left = neighbour_left(i, k)

            # fraction conterrà il valore della frazione 1/(1-gx^2)
            frazione!(gx2, i, k, fraction, sc_buffA, sc_buffB)

            #ora nel calcolo della radice metto anche una componente ennesima
            #questa avrà sempre valori da 1 a 4 e sono i 4 primi vicini
            #il primo passaggio fa 1-gx^2_{i+μ}
            #il secondo moltiplica questo per la frazione
            #il terzo fa la radice
            unmenx!(gx2, i, k, right..., sc_buffA)
            mult_sc!(sc_buffA, fraction, sc_buffD, i, k)
            radice!(sc_buffD, i, k, I(1), scalar_buffs2, sc_buffA, sc_buffB, sc_buffC)

            unmenx!(gx2, i, k, left..., sc_buffA)
            mult_sc!(sc_buffA, fraction, sc_buffD, i, k)
            radice!(sc_buffD, i, k, I(2), scalar_buffs2, sc_buffA, sc_buffB, sc_buffC)

            unmenx!(gx2, i, k, up..., sc_buffA)
            mult_sc!(sc_buffA, fraction, sc_buffD, i, k)
            radice!(sc_buffD, i, k, I(3), scalar_buffs2, sc_buffA, sc_buffB, sc_buffC)

            unmenx!(gx2, i, k, down..., sc_buffA)
            mult_sc!(sc_buffA, fraction, sc_buffD, i, k)
            radice!(sc_buffD, i, k, I(4), scalar_buffs2, sc_buffA, sc_buffB, sc_buffC)

            #in buff_scA metto g*fraz e poi ci sottraggo le radici
            mult_sc!(coupling, fraction, sc_buffA, i, k)
            for j::I = 1:n_ptords
                @inbounds @views sc_buffA[j,i,k] .+= .- scalar_buffs2[j,i,k,1] .- scalar_buffs2[j,i,k,2] .- scalar_buffs2[j,i,k,3] .- scalar_buffs2[j,i,k,4]
            end
            
            # in gradient metto il risultato della moltiplicazione di sc_buffA (passaggio precedente) per x
            mult_vec_sc_all_comps!(x, sc_buffA, gradient, i, k)
            # il risultato si ottiene sommando a gradient tutti i primi vicini del sito
            for n::I = 1:n_comps
                for j::I = 1:n_ptords
                    @inbounds gradient[j,i,k,n] += x[j,right...,n] + x[j,left...,n] + x[j,up...,n] + x[j,down...,n]
                end
            end
        end
        return nothing
    end

    function zero_mode_kernel!(x::CuDeviceArray{F, 4, 1}, zero_modo::CuDeviceArray{F, 2, 1})
        i, k = get_indexes()
        if (i <= Npoint && k <= Npoint)
            @inbounds @views x[:,i,k,:] .-= zero_modo
        end
        return nothing
    end

    return (g_quadro_kernel!, gradient_kernel!, zero_mode_kernel!, compute_ener_kernel!)
end

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

function main_omf(args::OMF_args, en_fname::AbstractString, lat_fname::AbstractString="")
    
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
    floatType = typeof(dt)
    
    # Setup checkpoint filenames
    checkpt_fname = get_checkpoint_filename(en_fname)
    if save_lattice
        lat_checkpt = get_lat_checkpoint(lat_fname)
    end

    n_ords = max_ptord + one(max_ptord)
    Npoint2 = Npoint^2

    # Initialize additional variables
    x_back = CuArray{floatType}(undef, n_ords, Npoint, Npoint, n_comps)
    CUDA.copyto!(x_back, x)
    args.x = x_back
    Pi = CUDA.zeros(floatType, n_ords, Npoint, Npoint, n_comps)
    gradient = CUDA.zeros(floatType, n_ords, Npoint, Npoint, n_comps)
    x2 = CUDA.zeros(floatType, n_ords, Npoint, Npoint)

    energia = CUDA.zeros(floatType, n_ords)
    ener = CUDA.zeros(floatType, n_ords, Npoint, Npoint)
    zero_modo = CUDA.zeros(floatType, n_ords, n_comps)

    # Handle energy measurement array and file opening
    if args.iter_start == 1
        mean_energy_file = open_energy_file(en_fname, "w", true) # with header
        if save_lattice
            ener_meas = CUDA.zeros(floatType, n_ords, Npoint, Npoint, n_meas)
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
    f_gx2, f_grad, f_zeromode, f_ener = create_kernels(floatType, Npoint, max_ptord, n_comps)

    # compilazione dei kernel e creazione di struct con i kernel compilati
    gx2 = compile_kernel(f_gx2, (x, x2), Npoint2)
    grad_ker = compile_kernel(f_grad, (x, gradient, x2), Npoint2)
    zero_mode = compile_kernel(f_zeromode, (x, zero_modo), Npoint2)
    compute_ener = compile_kernel(f_ener, (x, ener, x2), Npoint2)

    println("Initialization and Kernel compilation successfully complete.")

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
