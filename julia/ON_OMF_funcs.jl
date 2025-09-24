using CUDA
using Random
using Combinatorics
using Distributions
using Serialization
include("kernel_compilation.jl")
include("read_write_funcs.jl")

# di seguito, j è l'indice per l'ordine perturbativo
# i e k sono prima e seconda dim del reticolo
# n ∈ {1, ..., N-1} è il numero della componente

@inline function reset!(v::CuDeviceArray{F, 4, 1}, i::I, k::I, n::I) where {F <: AbstractFloat, I <: Integer}
    @inbounds v[:, i, k, n] .= 0
end

@inline function set_unity!(v::CuDeviceArray{F, 3, 1}, i::I, k::I) where {F <: AbstractFloat, I <: Integer}
    @inbounds v[:, i, k] .= 0
    @inbounds v[1, i, k] = 1
end

@inline function set_unity_vec!(v::CuDeviceArray{F, 4, 1}, i::I, k::I, n::I) where {F <: AbstractFloat, I <: Integer}
    @inbounds v[:, i, k, n] .= 0
    @inbounds v[1, i, k, n] = 1
end

@inline function reset_sc!(v::CuDeviceArray{F, 3, 1}, i::I, k::I) where {F <: AbstractFloat, I <: Integer}
    @inbounds v[:, i, k] .= 0
end

@inline function copia!(v::CuDeviceArray{F, 3, 1}, w::CuDeviceArray{F, 3, 1}, i::I, k::I) where {F <: AbstractFloat, I <: Integer}
    # mette w in v
    @inbounds @views v[:, i, k] .= w[:, i, k]
end

@inline function mult!(
    a::CuDeviceArray{F, 4, 1}, b::CuDeviceArray{F, 4, 1}, result::CuDeviceArray{F, 4, 1}, ord::I, i::I, k::I, n::I
    ) where {F <: AbstractFloat, I <: Integer}
    # mette a*b in result
    reset!(result, i, k, n)
    for j = one(I):(ord+one(I))
        for l = one(I):j
            @inbounds result[j, i, k, n] += a[l, i, k, n] * b[j-l+one(I), i, k, n]
        end
    end
end

@inline function mult_doppia!(
    a::CuDeviceArray{F, 4, 1}, b::CuDeviceArray{F, 4, 1}, result::CuDeviceArray{F, 4, 1},
    ord::I, i::I, k::I, i2::I, k2::I, n::I, n2::I
    ) where {F <: AbstractFloat, I <: Integer}
    # questa moltiplicazione prende gli indici a coppie 
    # così si possono moltiplicare cose provenienti da siti diversi
    # i risultati sono memorizzati usando gli indici del primo
    reset!(result, i, k, n)
    for j = one(I):(ord+one(I))
        for l = one(I):j
            @inbounds result[j, i, k, n] += a[l, i, k, n] * b[j-l+one(I), i2, k2, n2]
        end
    end 
end

@inline function mult_sc!(
    a::CuDeviceArray{F, 3, 1}, b::CuDeviceArray{F, 3, 1}, result::CuDeviceArray{F, 3, 1},
    ord::I, i::I, k::I
    ) where {F <: AbstractFloat, I <: Integer}
    # questa moltiplicazione è pensata per moltiplicare dei moduli quadri
    # sono ancora oggetti perturbativi con la struttura di reticolo
    # ma sono scalari quindi non c'è l'indice di componente
    # reset_sc!(result, i, k)
    @inbounds result[:, i, k] .= 0
    for j = one(I):(ord+one(I))
        for l = one(I):j
            @inbounds result[j, i, k] += a[l, i, k] * b[j-l+one(I), i, k]
        end
    end
end

@inline function mult_vec_sc_all_comps!(
    a::CuDeviceArray{F, 4, 1}, b::CuDeviceArray{F, 3, 1}, result::CuDeviceArray{F, 4, 1},
    ord::I, i::I, k::I
    ) where {F <: AbstractFloat, I <: Integer}
    # questa moltiplicazione prende il primo oggetto vettoriale
    # il secondo scalare
    @inbounds result[:, i, k, :] .= 0
    for j = one(I):(ord+one(I))
        for l = one(I):j
            @inbounds @views result[j, i, k, :] .+= a[l, i, k, :] .* b[j-l+one(I), i, k]
        end
    end
end

@inline function power!(
    x::CuDeviceArray{F, 3, 1},
    esp::I, ord::I, i::I, k::I,
    result_even::CuDeviceArray{F, 3, 1},
    result_odd::CuDeviceArray{F, 3, 1}
    ) where {F <: AbstractFloat, I <: Integer}
    # se n è dispari ci sarà il risultato corretto in result_odd
    # se n è pari il risultato sarà in result_even
    # se metto ad esempio 6 calcolerò comunque fino a 7
    copia!(result_odd, x, i, k)
    reset_sc!(result_even, i, k)
    for l = I(2):I(2):esp
        mult_sc!(x, result_odd, result_even, ord, i, k)
        mult_sc!(x, result_even, result_odd, ord, i, k)
    end
end

@inline function unmenx!(
    x::CuDeviceArray{F, 3, 1},
    i::I, k::I, i2::I, k2::I,
    result::CuDeviceArray{F, 3, 1},
    unita::CuDeviceArray{F, 1, 1}
    ) where {F <: AbstractFloat, I <: Integer}
    # mette in result la sottrazione unita - x
    @inbounds @views result[:, i, k] .= unita .- x[:, i2, k2]
end

function g_quadro!(
    x::CuDeviceArray{F, 4, 1},
    x2::CuDeviceArray{F, 3, 1},
    Npoint::I,
    n_comps::I,
    max_ptord::I,
    x2_vec::CuDeviceArray{F, 4, 1}
    ) where {F <: AbstractFloat, I <: Integer}
    # calcola in parallelo gx^2 su tutto il reticolo e lo mette in x2
    i, k = get_indexes(Npoint)
    if (i <= Npoint && k <= Npoint)
        reset_sc!(x2, i, k)
        for n = one(I):n_comps
            mult!(x, x, x2_vec, max_ptord, i, k, n)
            @inbounds @views x2[I(2):max_ptord+one(I), i, k] .+= x2_vec[one(I):max_ptord, i, k, n]
        end
    end
    return nothing
end

@inline function frazione!(
    x::CuDeviceArray{F, 3, 1},
    i::I,
    k::I,
    max_ptord::I,
    risultato::CuDeviceArray{F, 3, 1},
    potenza_pari::CuDeviceArray{F, 3, 1},
    potenza_dispari::CuDeviceArray{F, 3, 1}
    ) where {F <: AbstractFloat, I <: Integer}
#qui calcolo lo sviluppo di Taylor di 1/(1-gx)
    set_unity!(risultato, i, k)
    for esp = one(I):I(2):(max_ptord+one(I))
        power!(x, esp, max_ptord, i, k, potenza_pari, potenza_dispari)
        for j = one(I):(max_ptord+one(I))
            @inbounds risultato[j,i,k] += potenza_pari[j,i,k] + potenza_dispari[j,i,k]
        end
    end
end

@inline function radice!(
    x::CuDeviceArray{F, 3, 1},
    i::I,
    k::I,
    n::I,
    max_ptord::I,
    unita::CuDeviceArray{F, 1, 1},
    risultato::CuDeviceArray{F, 4, 1},
    potenza_pari::CuDeviceArray{F, 3, 1},
    potenza_dispari::CuDeviceArray{F, 3, 1},
    xmenun::CuDeviceArray{F, 3, 1}
    ) where {F <: AbstractFloat, I <: Integer}
    # svilupo di taylor di √x attorno a x=(1,0,...,0) e lo metto in risultato[:,:,:,n] (n∈{1,...,4} rappresenta 1:dx, 2:sx, 3:su, 4:giù)
    set_unity_vec!(risultato,i,k,n)
    @inbounds @views xmenun[:, i, k] .= x[:, i, k] .- unita
    for esp = one(I):I(2):(max_ptord+one(I))
        power!(xmenun, esp, max_ptord, i, k, potenza_pari, potenza_dispari)
        for j in one(I):max_ptord+one(I)
            @inbounds risultato[j,i,k,n] += binomial(1/2,esp-one(I))*potenza_pari[j,i,k] + binomial(1/2,esp)*potenza_dispari[j,i,k]
        end
    end
end

function compute_ener!(
    x::CuDeviceArray{F, 4, 1},
    ener::CuDeviceArray{F, 3, 1},
    gi_dx::CuDeviceArray{I, 1, 1},
    max_ptord::I,
    Npoint::I,
    n_comps::I,
    x2::CuDeviceArray{F, 3, 1},
    rad::CuDeviceArray{F, 4, 1},
    arg_rad::CuDeviceArray{F, 3, 1},
    buff_scA::CuDeviceArray{F, 3, 1},
    buff_scB::CuDeviceArray{F, 3, 1},
    buff_scC::CuDeviceArray{F, 3, 1},
    buff_vec::CuDeviceArray{F, 4, 1},
    buffs_sc::CuDeviceArray{F, 4, 1},
    unita::CuDeviceArray{F, 1, 1}
    ) where {F <: AbstractFloat, I <: Integer}

    i, k = get_indexes(Npoint)

    if (i <= Npoint && k <= Npoint)
        for n = I(1):n_comps
            #per l'energia io sommo i contributi di link destro e link sotto
            #poi divido per 2, così conto tutti i link e faccio la media
            #alla fine dovrò comunque dividere per Npoint^2
            #sulle n_comps sommo perchè tanto è un prodotto scalare
            #poi c'è un g davanti quindi guardo un j indietro
            mult_doppia!(x, x, buff_vec, max_ptord, i, k, i, gi_dx[k], n, n)
            @inbounds @views ener[I(2):max_ptord+I(1),i,k] .+= F(0.5) .* buff_vec[I(1):max_ptord,i,k,n]
            mult_doppia!(x, x, buff_vec, max_ptord, i, k, gi_dx[i], k, n, n)
            @inbounds @views ener[I(2):max_ptord+I(1),i,k] .+= F(0.5) .* buff_vec[I(1):max_ptord,i,k,n]
        end

        #poi sommo le radici fuori dal ciclo su n
        #anche qui salvo i risultati sulle diverse n_comps di rad
        unmenx!(x2, i, k, i, k, arg_rad, unita)
        radice!(arg_rad, i, k, I(1), max_ptord, unita, rad, buff_scA, buff_scB, buff_scC)

        unmenx!(x2, i, k, i, gi_dx[k], arg_rad, unita)
        radice!(arg_rad, i, k, I(2), max_ptord, unita, rad, buff_scA, buff_scB, buff_scC)

        unmenx!(x2, i, k, gi_dx[i], k, arg_rad, unita)
        radice!(arg_rad, i, k, I(3), max_ptord, unita, rad, buff_scA, buff_scB, buff_scC)

        #così troverò i 2 risultati importanti in
        #buffs_sc[:,i,k,1] e buffs_sc[:,i,k,3]
        mult_doppia!(rad, rad, buffs_sc, max_ptord, i, k, i, k, I(1), I(2))
        mult_doppia!(rad, rad, buffs_sc, max_ptord, i, k, i, k, I(3), I(1))

        @inbounds @views ener[:,i,k] .+= F(0.5) .* (buffs_sc[:,i,k,I(1)] .+ buffs_sc[:,i,k,I(3)])
    end
    return nothing
end

@inline function reset_moment!(Pi::CuArray{F, 4}, rng::CUDA.RNG) where {F <: AbstractFloat}
    CUDA.fill!(Pi, 0)
    @inbounds @views CUDA.randn!(rng, Pi[1,:,:,:])
    return nothing
end

function compute_vec_grad!(
    x::CuDeviceArray{F, 4, 1},
    gradient::CuDeviceArray{F, 4, 1},
    su_sx::CuDeviceArray{I, 1, 1},
    gi_dx::CuDeviceArray{I, 1, 1},
    buff_sc::CuDeviceArray{F, 3, 1},
    buff_vec::CuDeviceArray{F, 4, 1},
    max_ptord::I,
    n_comps::I,
    Npoint::I
    ) where {F <: AbstractFloat, I <: Integer}
    i, k = get_indexes(Npoint)
    if (i <= Npoint && k <= Npoint)
        # in buff_vec metto il risultato della moltiplicazione di buff_sc (passaggio precedente) per x
        mult_vec_sc_all_comps!(x, buff_sc, buff_vec, max_ptord, i, k)
        # il risultato si ottiene sommando a buff_vec tutti i primi vicini del sito
        for n in 1:n_comps
            for j in 1:max_ptord+1
                @inbounds gradient[j,i,k,n] = buff_vec[j,i,k,n] + x[j,i,gi_dx[k],n] + x[j,i,su_sx[k],n] + x[j,su_sx[i],k,n] + x[j,gi_dx[i],k,n]
            end
        end
    end
    return nothing
end

function gradient_kernel!(
    su_sx::CuDeviceArray{I, 1, 1},
    gi_dx::CuDeviceArray{I, 1, 1},
    ordine::I,
    Npoint::I,
    x2::CuDeviceArray{F, 3, 1},
    fraz::CuDeviceArray{F, 3, 1},
    rad::CuDeviceArray{F, 4, 1},
    arg_rad::CuDeviceArray{F, 3, 1},
    buff_scA::CuDeviceArray{F, 3, 1},
    buff_scB::CuDeviceArray{F, 3, 1},
    buff_scC::CuDeviceArray{F, 3, 1},
    unita::CuDeviceArray{F, 1, 1},
    coupling::CuDeviceArray{F, 3, 1}
    ) where {F <: AbstractFloat, I <: Integer}
    
    i, k = get_indexes(Npoint)

    if (i <= Npoint && k <= Npoint)
        frazione!(x2, i, k, ordine, fraz, buff_scA, buff_scB)

        #ora nel calcolo della radice metto anche una componente ennesima
        #questa avrà sempre valori da 1 a 4 e sono i 4 primi vicini
        #il primo passaggio fa 1-gx^2_{i+μ}
        #il secondo moltiplica questo per la frazione
        #il terzo fa la radice
        unmenx!(x2, i, k, i, gi_dx[k], buff_scA, unita)
        mult_sc!(buff_scA, fraz, arg_rad, ordine, i, k)
        radice!(arg_rad, i, k, I(1), ordine, unita, rad, buff_scA, buff_scB, buff_scC)

        unmenx!(x2, i, k, i, su_sx[k], buff_scA, unita)
        mult_sc!(buff_scA, fraz, arg_rad, ordine, i, k)
        radice!(arg_rad, i, k, I(2), ordine, unita, rad, buff_scA, buff_scB, buff_scC)

        unmenx!(x2, i, k, su_sx[i], k, buff_scA, unita)
        mult_sc!(buff_scA, fraz, arg_rad, ordine, i, k)
        radice!(arg_rad, i, k, I(3), ordine, unita, rad, buff_scA, buff_scB, buff_scC)

        unmenx!(x2, i, k, gi_dx[i], k, buff_scA, unita)
        mult_sc!(buff_scA, fraz, arg_rad, ordine, i, k)
        radice!(arg_rad, i, k, I(4), ordine, unita, rad, buff_scA, buff_scB, buff_scC)

        #in buff_scA metto g*fraz e poi ci sottraggo le radici
        mult_sc!(coupling, fraz, buff_scA, ordine, i, k)
    end
    return
end

function compute_gradient!(
    gx2::CompiledKernel,
    grad_ker::CompiledKernel,
    vec_grad::CompiledKernel,
    rad::CuArray{F, 4},
    buff_scA::CuArray{F, 3}
    ) where {F <: AbstractFloat}
    # calcola -gradiente (in parallelo) su ogni sito del reticolo e lo mette in gradient

    CUDA.@sync begin
        run_kernel(gx2)
    end

    CUDA.@sync begin
        run_kernel(grad_ker)
    end

    # e poi ci sottraggo le radici
    CUDA.@sync @inbounds @views buff_scA .+= .- rad[:,:,:,1] .- rad[:,:,:,2] .- rad[:,:,:,3] .- rad[:,:,:,4]

    CUDA.@sync begin
        run_kernel(vec_grad)
    end
    return nothing
end

function omf_evolution!(
    x::CuArray{F, 4},
    Pi::CuArray{F, 4},
    dt::F,
    rad::CuArray{F, 4},
    buff_scA::CuArray{F, 3},
    gradient::CuArray{F, 4},
    gx2::CompiledKernel,
    grad_ker::CompiledKernel,
    vec_grad::CompiledKernel
    ) where {F <: AbstractFloat}
    # esegue l'evoluzione di uno step secondo l'algoritmo OMF

    # costanti per l'algoritmo OMF
    ρ = F(+0.2750081212332419e+00)
    θ = F(-0.1347950099106792e+00)
    Θ = F(-0.8442961950707149e-01)
    λ = F(+0.3549000571574260e+00)

    CUDA.@sync x .+= (dt * ρ) .* Pi
    compute_gradient!(gx2, grad_ker, vec_grad, rad, buff_scA)
    CUDA.@sync Pi .+= (dt * Θ) .* gradient
    CUDA.@sync x .+= (dt * θ) .* Pi
    compute_gradient!(gx2, grad_ker, vec_grad, rad, buff_scA)
    CUDA.@sync Pi .+= (dt * λ) .* gradient
    CUDA.@sync x .+= (dt * 0.5 * (1 - 2*(θ+ρ))) .* Pi
    compute_gradient!(gx2, grad_ker, vec_grad, rad, buff_scA)
    CUDA.@sync Pi .+= (dt * (1 - 2*(λ+Θ))) .* gradient
    CUDA.@sync x .+= (dt * 0.5 * (1 - 2*(θ+ρ))) .* Pi
    compute_gradient!(gx2, grad_ker, vec_grad, rad, buff_scA)
    CUDA.@sync Pi .+= (dt * λ) .* gradient
    CUDA.@sync x .+= (dt * θ) .* Pi
    compute_gradient!(gx2, grad_ker, vec_grad, rad, buff_scA)
    CUDA.@sync Pi .+= (dt * Θ) .* gradient
    CUDA.@sync x .+= (dt * ρ) .* Pi
    return nothing
end

function subtract_zero_mode!(
    x::CuDeviceArray{F, 4, 1},
    zero_modo::CuDeviceArray{F, 2, 1},
    Npoint::I
    ) where {F <: AbstractFloat, I <: Integer}
    i, k = get_indexes(Npoint)
    if (i <= Npoint && k <= Npoint)
        @inbounds @views x[:,i,k,:] .-= zero_modo
    end
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
    comp_ener::CompiledKernel
    ) where {F <: AbstractFloat, I <: Integer}
    # computes the mean of the energy of all sites at time t
    
    CUDA.@sync @inbounds @views zero_modo .= CUDA.sum(CUDA.sum(x, dims=3), dims=2)[:,1,1,:] ./ (Npoint^2)
    CUDA.fill!(ener, 0)

    # sottraggo lo zero modo da x (x = x - zero_modo)
    CUDA.@sync begin
        run_kernel(zero_mode)
    end

    CUDA.@sync begin
        run_kernel(gx2)
    end
    CUDA.@sync begin
        run_kernel(comp_ener)
    end
    CUDA.@sync @inbounds @views energy .= CUDA.sum(CUDA.sum(ener, dims=3), dims=2)[:,1,1] ./ (Npoint^2)
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
    
    # Setup checkpoint filenames
    checkpt_fname = get_checkpoint_filename(en_fname)
    if save_lattice
        lat_checkpt = get_lat_checkpoint(lat_fname)
    end

    n_ords = max_ptord + one(max_ptord)
    Npoint2 = Npoint^2

    # Initialize additional variables
    x_back = CuArray{Float32}(undef, n_ords, Npoint, Npoint, n_comps)
    CUDA.copyto!(x_back, x)
    args.x = x_back
    Pi = CUDA.zeros(Float32, n_ords, Npoint, Npoint, n_comps)
    gradient = CUDA.zeros(Float32, n_ords, Npoint, Npoint, n_comps)

    buff_vec = CUDA.zeros(Float32, n_ords, Npoint, Npoint, n_comps)
    fraz = CUDA.zeros(Float32, n_ords, Npoint, Npoint)
    arg_rad = CUDA.zeros(Float32, n_ords, Npoint, Npoint)
    rad = CUDA.zeros(Float32, n_ords, Npoint, Npoint, 4)
    buffs_sc = CUDA.zeros(Float32, n_ords, Npoint, Npoint, 4)
    @inbounds buff_scA = @view buffs_sc[:,:,:,1]
    @inbounds buff_scB = @view buffs_sc[:,:,:,2]
    @inbounds buff_scC = @view buffs_sc[:,:,:,3]
    x2 = CUDA.zeros(Float32, n_ords, Npoint, Npoint)

    unita = CUDA.zeros(Float32, n_ords)
    @inbounds CUDA.@allowscalar unita[1] = 1.0f0
    coupling = CUDA.zeros(Float32, n_ords, Npoint, Npoint)
    @inbounds coupling[2,:,:] .= 1.0f0
    
    energia = CUDA.zeros(Float32, n_ords)
    ener = CUDA.zeros(Float32, n_ords, Npoint, Npoint)
    zero_modo = CUDA.zeros(Float32, n_ords, n_comps)

    # Handle energy measurement array and file opening
    if args.iter_start == 1
        file = open_energy_file(en_fname, "w", true) # with header
        if save_lattice
            ener_meas = CUDA.zeros(Float32, n_ords, Npoint, Npoint, n_meas)
        end
    else
        file = open_energy_file(en_fname, "a", false) # without header, appending
        if save_lattice
            ener_meas = deserialize(lat_checkpt)
        end
    end

    # randomizzazione di NHMC
    binom_distr = Binomial(NHMC*10,1/10)

    # Array con i primi vicini
    indici = CuArray{Int32}(1:Npoint)
    su_sx = CuArray{Int32}(undef, Npoint)
    gi_dx = CuArray{Int32}(undef, Npoint)
    circshift!(su_sx, indici, 1)
    circshift!(gi_dx, indici, -1)

    # definizione degli argomenti dei kernel
    gx2_args = (x, x2, Npoint, n_comps, max_ptord, buff_vec)
    grad_ker_args = (su_sx, gi_dx, max_ptord, Npoint, x2, fraz, rad, arg_rad,
                     buff_scA, buff_scB, buff_scC, unita, coupling)
    vec_grad_args = (x, gradient, su_sx, gi_dx, buff_scA, buff_vec, max_ptord, n_comps, Npoint)
    zero_mode_args = (x, zero_modo, Npoint)
    comp_ener_args = (x, ener, gi_dx, max_ptord, Npoint, n_comps, x2, rad, arg_rad,
                      buff_scA, buff_scB, buff_scC, buff_vec, buffs_sc, unita)
    
    # compilazione dei kernel e creazione di struct con i kernel compilati
    gx2 = compile_kernel(g_quadro!, gx2_args, Npoint2)
    grad_ker = compile_kernel(gradient_kernel!, grad_ker_args, Npoint2)
    vec_grad = compile_kernel(compute_vec_grad!, vec_grad_args, Npoint2)
    zero_mode = compile_kernel(subtract_zero_mode!, zero_mode_args, Npoint2)
    comp_ener = compile_kernel(compute_ener!, comp_ener_args, Npoint2)

    println("Kernel compilation successfully complete.")

    jobid = get_job_id()

    # Main computation loop
    for t = args.iter_start:n_meas
        for _ in 1:measure_every
            reset_moment!(Pi, cuda_rng)
            for __ = 1:rand(nhmc_rng, binom_distr)
                omf_evolution!(x, Pi, dt, rad, buff_scA, gradient, gx2, grad_ker, vec_grad)
            end
        end
        
        compute_energy!(energia, x, ener, zero_modo, Npoint, zero_mode, gx2, comp_ener)
        
        # Store energy measurements if saving lattice
        if save_lattice
            @inbounds @views ener_meas[:,:,:,t] .= ener
        end
        
        write_line(file, Array(energia))
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
    close(file)
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
                    1, CUDA.zeros(Float32, max_ptord+1, Npoint, Npoint, n_comps))
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
