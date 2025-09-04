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

@inline function reset!(v, i, k, n)
    @inbounds v[:, i, k, n] .= 0
end

@inline function set_unity!(v, i, k)
    @inbounds v[:, i, k] .= 0
    @inbounds v[1, i, k] = 1
end

@inline function set_unity_vec!(v, i, k, n)
    @inbounds v[:, i, k, n] .= 0
    @inbounds v[1, i, k, n] = 1
end

@inline function reset_sc!(v, i, k)
    @inbounds v[:, i, k] .= 0
end

@inline function copia!(v, w, i, k)
    # mette w in v
    @inbounds @views v[:, i, k] .= w[:, i, k]
end

@inline function mult!(a, b, result, ord, i, k, n)
    # mette a*b in result
    reset!(result, i, k, n)
    for j = 1:(ord+1)
        for l = 1:j
            @inbounds result[j, i, k, n] += a[l, i, k, n] * b[j-l+1, i, k, n]
        end
    end
end

@inline function mult_doppia!(a, b, result, ord, i, k, i2, k2, n, n2)
    # questa moltiplicazione prende gli indici a coppie 
    # così si possono moltiplicare cose provenienti da siti diversi
    # i risultati sono memorizzati usando gli indici del primo
    reset!(result, i, k, n)
    for j = 1:(ord+1)
        for l = 1:j
            @inbounds result[j, i, k, n] += a[l, i, k, n] * b[j-l+1, i2, k2, n2]
        end
    end 
end

@inline function mult_sc!(a, b, result, ord, i, k)
    # questa moltiplicazione è pensata per moltiplicare dei moduli quadri
    # sono ancora oggetti perturbativi con la struttura di reticolo
    # ma sono scalari quindi non c'è l'indice di componente
    reset_sc!(result, i, k)
    for j = 1:(ord+1)
        for l = 1:j
            @inbounds result[j, i, k] += a[l, i, k] * b[j-l+1, i, k]
        end
    end
end

@inline function mult_vec_sc_all_comps!(a, b, result, ord, i, k)
    # questa moltiplicazione prende il primo oggetto vettoriale
    # il secondo scalare
    @inbounds result[:, i, k, :] .= 0
    for j = 1:(ord+1)
        for l = 1:j
            @inbounds @views result[j, i, k, :] .+= a[l, i, k, :] .* b[j-l+1, i, k]
        end
    end
end

@inline function power!(x, esp, ord, i, k, result_even, result_odd)
    # se n è dispari ci sarà il risultato corretto in result_odd
    # se n è pari il risultato sarà in result_even
    # se metto ad esempio 6 calcolerò comunque fino a 7
    copia!(result_odd, x, i, k)
    reset_sc!(result_even, i, k)
    for l = 2:2:esp
        mult_sc!(x, result_odd, result_even, ord, i, k)
        mult_sc!(x, result_even, result_odd, ord, i, k)
    end
end

@inline function unmenx!(x, i, k, i2, k2, result, unita)
    # mette in result la sottrazione unita - x
    @inbounds @views result[:, i, k] .= unita .- x[:, i2, k2]
end

function g_quadro!(x, x2, Npoint, n_comps, max_ptord, x2_vec)
    # calcola in parallelo gx^2 su tutto il reticolo e lo mette in x2
    i, k = get_indexes((blockIdx().x - 1) * blockDim().x + threadIdx().x, Npoint)
    if (i <= Npoint && k <= Npoint)
        reset_sc!(x2, i, k)
        for n = 1:n_comps
            mult!(x, x, x2_vec, max_ptord, i, k, n)
            @inbounds @views x2[2:max_ptord+1, i, k] .+= x2_vec[1:max_ptord, i, k, n]
        end
    end
    return nothing
end

@inline function frazione!(x, i, k, max_ptord, risultato, potenza_pari, potenza_dispari)
#qui calcolo lo sviluppo di Taylor di 1/(1-gx)
    set_unity!(risultato, i, k)
    for esp = 1:2:(max_ptord+1)
        power!(x, esp, max_ptord, i, k, potenza_pari, potenza_dispari)
        for j = 1:(max_ptord+1)
            @inbounds risultato[j,i,k] += potenza_pari[j,i,k] + potenza_dispari[j,i,k]
        end
    end
end

@inline function radice!(x, i, k, n, max_ptord, unita, risultato, potenza_pari, potenza_dispari, xmenun)
    # svilupo di taylor di √x attorno a x=(1,0,...,0) e lo metto in risultato[:,:,:,n] (n∈{1,...,4} rappresenta 1:dx, 2:sx, 3:su, 4:giù)
    set_unity_vec!(risultato,i,k,n)
    @inbounds @views xmenun[:, i, k] .= x[:, i, k] .- unita
    for esp = 1:2:(max_ptord+1)
        power!(xmenun, esp, max_ptord, i, k, potenza_pari, potenza_dispari)
        for j in 1:max_ptord+1
            @inbounds risultato[j,i,k,n] += binomial(1/2,esp-1)*potenza_pari[j,i,k] + binomial(1/2,esp)*potenza_dispari[j,i,k]
        end
    end
end

function compute_ener!(x, ener, gi_dx, max_ptord, Npoint, n_comps,
    x2, rad, arg_rad, zeri_scA, zeri_scB, zeri_scC, zeri_vec, buffs_sc, unita)

    i, k = get_indexes((blockIdx().x - 1) * blockDim().x + threadIdx().x, Npoint)

    if (i <= Npoint && k <= Npoint)
        for n = 1:n_comps
            #per l'energia io sommo i contributi di link destro e link sotto
            #poi divido per 2, così conto tutti i link e faccio la media
            #alla fine dovrò comunque dividere per Npoint^2
            #sulle n_comps sommo perchè tanto è un prodotto scalare
            #poi c'è un g davanti quindi guardo un j indietro
            mult_doppia!(x, x, zeri_vec, max_ptord, i, k, i, gi_dx[k], n, n)
            @inbounds @views ener[2:max_ptord+1,i,k] .+= 0.5 .* zeri_vec[1:max_ptord,i,k,n]
            mult_doppia!(x, x, zeri_vec, max_ptord, i, k, gi_dx[i], k, n, n)
            @inbounds @views ener[2:max_ptord+1,i,k] .+= 0.5 .* zeri_vec[1:max_ptord,i,k,n]
        end

        #poi sommo le radici fuori dal ciclo su n
        #anche qui salvo i risultati sulle diverse n_comps di rad
        unmenx!(x2, i, k, i, k, arg_rad, unita)
        radice!(arg_rad, i, k, 1, max_ptord, unita, rad, zeri_scA, zeri_scB, zeri_scC)

        unmenx!(x2, i, k, i, gi_dx[k], arg_rad, unita)
        radice!(arg_rad, i, k, 2, max_ptord, unita, rad, zeri_scA, zeri_scB, zeri_scC)

        unmenx!(x2, i, k, gi_dx[i], k, arg_rad, unita)
        radice!(arg_rad, i, k, 3, max_ptord, unita, rad, zeri_scA, zeri_scB, zeri_scC)

        #così troverò i 2 risultati importanti in
        #buffs_sc[:,i,k,1] e buffs_sc[:,i,k,3]
        mult_doppia!(rad, rad, buffs_sc, max_ptord, i, k, i, k, 1, 2)
        mult_doppia!(rad, rad, buffs_sc, max_ptord, i, k, i, k, 3, 1)

        @inbounds @views ener[:,i,k] .+= 0.5 .* (buffs_sc[:,i,k,1] .+ buffs_sc[:,i,k,3])
    end
    return nothing
end

@inline function reset_moment!(Pi, rng)
    CUDA.fill!(Pi,0.0f0)
    @inbounds @views CUDA.randn!(rng, Pi[1,:,:,:])
end

function compute_vec_grad!(x, gradient, su_sx, gi_dx, buff_sc, buff_vec, max_ptord, n_comps, Npoint)
    i, k = get_indexes((blockIdx().x - 1) * blockDim().x + threadIdx().x, Npoint)
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

function gradient_kernel!(su_sx, gi_dx, ordine, Npoint,
    x2, fraz, rad, arg_rad, buff_scA, buff_scB, buff_scC, unita, coupling)
    
    i, k = get_indexes((blockIdx().x - 1) * blockDim().x + threadIdx().x, Npoint)

    if (i <= Npoint && k <= Npoint)
        frazione!(x2, i, k, ordine, fraz, buff_scA, buff_scB)

        #ora nel calcolo della radice metto anche una componente ennesima
        #questa avrà sempre valori da 1 a 4 e sono i 4 primi vicini
        #il primo passaggio fa 1-gx^2_{i+μ}
        #il secondo moltiplica questo per la frazione
        #il terzo fa la radice
        unmenx!(x2, i, k, i, gi_dx[k], buff_scA, unita)
        mult_sc!(buff_scA, fraz, arg_rad, ordine, i, k)
        radice!(arg_rad, i, k, 1, ordine, unita, rad, buff_scA, buff_scB, buff_scC)

        unmenx!(x2, i, k, i, su_sx[k], buff_scA, unita)
        mult_sc!(buff_scA, fraz, arg_rad, ordine, i, k)
        radice!(arg_rad, i, k, 2, ordine, unita, rad, buff_scA, buff_scB, buff_scC)

        unmenx!(x2, i, k, su_sx[i], k, buff_scA, unita)
        mult_sc!(buff_scA, fraz, arg_rad, ordine, i, k)
        radice!(arg_rad, i, k, 3, ordine, unita, rad, buff_scA, buff_scB, buff_scC)

        unmenx!(x2, i, k, gi_dx[i], k, buff_scA, unita)
        mult_sc!(buff_scA, fraz, arg_rad, ordine, i, k)
        radice!(arg_rad, i, k, 4, ordine, unita, rad, buff_scA, buff_scB, buff_scC)

        #in zeri_scA metto g*fraz e poi ci sottraggo le radici
        mult_sc!(coupling, fraz, buff_scA, ordine, i, k)
    end
    return
end

function compute_gradient!(gx2, grad_ker, vec_grad, rad, buff_scA)
    # calcola -gradiente (in parallelo) su ogni sito del reticolo e lo mette in gradient

    CUDA.@sync begin
        gx2.kernel(gx2.args...; threads=gx2.threads, blocks=gx2.blocks)
    end

    CUDA.@sync begin
        grad_ker.kernel(grad_ker.args...; threads=grad_ker.threads, blocks=grad_ker.blocks)
    end

    # e poi ci sottraggo le radici
    CUDA.@sync @inbounds @views buff_scA .+= .- rad[:,:,:,1] .- rad[:,:,:,2] .- rad[:,:,:,3] .- rad[:,:,:,4]

    CUDA.@sync begin
        vec_grad.kernel(vec_grad.args...; threads=vec_grad.threads, blocks=vec_grad.blocks)
    end
    return nothing
end

function omf_evolution!(x, Pi, dt, rad, buff_scA, gradient, gx2, grad_ker, vec_grad)
    # esegue l'evoluzione di uno step secondo l'algoritmo OMF

    # costanti per l'algoritmo OMF
    ρ = +0.2750081212332419e+00
    θ = -0.1347950099106792e+00
    Θ = -0.8442961950707149e-01
    λ = +0.3549000571574260e+00

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

@inline function subtract_zero_mode!(x, zero_modo, Npoint)
    i, k = get_indexes((blockIdx().x - 1) * blockDim().x + threadIdx().x, Npoint)
    if (i <= Npoint && k <= Npoint)
        @inbounds @views x[:,i,k,:] .-= zero_modo
    end
    return nothing
end

function compute_energy!(energy, x, ener, zero_modo, Npoint, zero_mode, gx2, comp_ener)
    # computes the mean of the energy of all sites at time t
    
    CUDA.@sync @inbounds @views zero_modo .= CUDA.sum(CUDA.sum(x, dims=3), dims=2)[:,1,1,:] ./ (Npoint^2)
    CUDA.fill!(ener, 0.0f0)

    # sottraggo lo zero modo da x (x = x - zero_modo)
    CUDA.@sync begin
        zero_mode.kernel(zero_mode.args...; threads=zero_mode.threads, blocks=zero_mode.blocks)
    end

    CUDA.@sync begin
        gx2.kernel(gx2.args...; threads=gx2.threads, blocks=gx2.blocks)
    end
    CUDA.@sync begin
        comp_ener.kernel(comp_ener.args...; threads=comp_ener.threads, blocks=comp_ener.blocks)
    end
    CUDA.@sync @inbounds @views energy .= CUDA.sum(CUDA.sum(ener, dims=3), dims=2)[:,1,1] ./ (Npoint^2)
    return nothing
end

function main_omf(args::OMF_args, en_fname::AbstractString)
    
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
    checkpt_fname = get_checkpoint_filename(en_fname)

    n_ords = max_ptord+1
    Npoint2 = Npoint^2

    x_back = CuArray{Float32}(undef, n_ords, Npoint, Npoint, n_comps)
    CUDA.copyto!(x_back, x)
    args.x = x_back
    Pi = CUDA.zeros(Float32, n_ords, Npoint, Npoint, n_comps)
    gradient = CUDA.zeros(Float32, n_ords, Npoint, Npoint, n_comps)

    buff_vec = CUDA.zeros(Float32, n_ords, Npoint, Npoint, n_comps)
    fraz = CUDA.zeros(Float32, n_ords, Npoint, Npoint)
    arg_rad = CUDA.zeros(Float32, n_ords, Npoint, Npoint, 4)
    rad = CUDA.zeros(Float32, n_ords, Npoint, Npoint, 4)
    buffs_sc = CUDA.zeros(Float32, n_ords, Npoint, Npoint, 4)
    buff_scA = selectdim(buffs_sc, 4, 1)
    buff_scB = selectdim(buffs_sc, 4, 2)
    buff_scC = selectdim(buffs_sc, 4, 3)
    x2 = CUDA.zeros(Float32, n_ords, Npoint, Npoint)

    unita = CUDA.zeros(Float32, n_ords)
    @inbounds CUDA.@allowscalar unita[1] = 1.0f0
    coupling = CUDA.zeros(Float32, n_ords, Npoint, Npoint)
    @inbounds coupling[2,:,:] .= 1.0f0
    
    energia = CUDA.zeros(Float32, n_ords)
    ener = CUDA.zeros(Float32, n_ords, Npoint, Npoint)
    zero_modo = CUDA.zeros(Float32, n_ords, n_comps)

    if args.iter_start == 1
        file = open_energy_file(en_fname, "w", true) # with header
    else
        file = open_energy_file(en_fname, "a", false) # without header
    end

    # randomizzazione di NHMC
    binom_distr = Binomial(NHMC*10,1/10)

    indici = CuArray{Int32}(1:Npoint)
    su__sx = CuArray{Int32}(undef, Npoint)
    giu_dx = CuArray{Int32}(undef, Npoint)
    circshift!(su__sx, indici, 1)
    circshift!(giu_dx, indici, -1)

    # definizione degli argomenti dei kernel
    gx2_args = (x, x2, Npoint, n_comps, max_ptord, buff_vec)
    grad_ker_args = (su__sx, giu_dx, max_ptord, Npoint, x2, fraz, rad, arg_rad,
                     buff_scA, buff_scB, buff_scC, unita, coupling)
    vec_grad_args = (x, gradient, su__sx, giu_dx, buff_scA, buff_vec, max_ptord, n_comps, Npoint)
    zero_mode_args = (x, zero_modo, Npoint)
    comp_ener_args = (x, ener, giu_dx, max_ptord, Npoint, n_comps, x2, rad, selectdim(arg_rad, 4, 1),
                      buff_scA, buff_scB, buff_scC, buff_vec, buffs_sc, unita)
    # compilazione dei kernel
    gx2_c = compile_kernel(g_quadro!, gx2_args, Npoint2)
    grad_ker_c = compile_kernel(gradient_kernel!, grad_ker_args, Npoint2)
    vec_grad_c = compile_kernel(compute_vec_grad!, vec_grad_args, Npoint2)
    zero_mode_c = compile_kernel(subtract_zero_mode!, zero_mode_args, Npoint2)
    comp_ener_c = compile_kernel(compute_ener!, comp_ener_args, Npoint2)
    # creazione di struct con i kernel compilati
    gx2 = CompiledKernel(gx2_args, gx2_c...)
    grad_ker = CompiledKernel(grad_ker_args, grad_ker_c...)
    vec_grad = CompiledKernel(vec_grad_args, vec_grad_c...)
    zero_mode = CompiledKernel(zero_mode_args, zero_mode_c...)
    comp_ener = CompiledKernel(comp_ener_args, comp_ener_c...)

    jobid = get_job_id()

    for t = args.iter_start:n_meas
        for _ in 1:measure_every
            reset_moment!(Pi, cuda_rng)
            for __ = 1:rand(nhmc_rng, binom_distr)
                omf_evolution!(x, Pi, dt, rad, buff_scA, gradient, gx2, grad_ker, vec_grad)
            end
        end
        compute_energy!(energia, x, ener, zero_modo, Npoint, zero_mode, gx2, comp_ener)
        write_line(file, Array(energia))
        x_back .= x
        args.iter_start = t + 1
        if get_remaining_time(jobid) < 120
            # 2 min remaining, save state and exit
            save_state(checkpt_fname, args)
            execute_self(checkpt_fname)
            return
        end
    end
    close(file)
    println("Energy written to file ", en_fname)
    if isfile(checkpt_fname)
        rm(checkpt_fname)  # cleanup
        println("Execution successful. Removed checkpoint file.")
    end
    return
end

function main_omf(args::OMF_args, en_fname::AbstractString, lat_fname::AbstractString)
    
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
    checkpt_fname = get_checkpoint_filename(en_fname)
    lat_checkpt = get_lat_checkpoint(lat_fname)

    n_ords = max_ptord+1
    Npoint2 = Npoint^2

    x_back = CuArray{Float32}(undef, n_ords, Npoint, Npoint, n_comps)
    CUDA.copyto!(x_back, x)
    args.x = x_back
    Pi = CUDA.zeros(Float32, n_ords, Npoint, Npoint, n_comps)
    gradient = CUDA.zeros(Float32, n_ords, Npoint, Npoint, n_comps)

    buff_vec = CUDA.zeros(Float32, n_ords, Npoint, Npoint, n_comps)
    fraz = CUDA.zeros(Float32, n_ords, Npoint, Npoint)
    arg_rad = CUDA.zeros(Float32, n_ords, Npoint, Npoint, 4)
    rad = CUDA.zeros(Float32, n_ords, Npoint, Npoint, 4)
    buffs_sc = CUDA.zeros(Float32, n_ords, Npoint, Npoint, 4)
    buff_scA = selectdim(buffs_sc, 4, 1)
    buff_scB = selectdim(buffs_sc, 4, 2)
    buff_scC = selectdim(buffs_sc, 4, 3)
    x2 = CUDA.zeros(Float32, n_ords, Npoint, Npoint)

    unita = CUDA.zeros(Float32, n_ords)
    @inbounds CUDA.@allowscalar unita[1] = 1.0f0
    coupling = CUDA.zeros(Float32, n_ords, Npoint, Npoint)
    @inbounds coupling[2,:,:] .= 1.0f0
    
    energia = CUDA.zeros(Float32, n_ords)
    ener = CUDA.zeros(Float32, n_ords, Npoint, Npoint)
    zero_modo = CUDA.zeros(Float32, n_ords, n_comps)

    if args.iter_start == 1
        ener_meas = CUDA.zeros(Float32, n_ords, Npoint, Npoint, n_meas)
        file = open_energy_file(en_fname, "w", true) # with header
    else
        ener_meas = deserialize(lat_checkpt)
        file = open_energy_file(en_fname, "a", false) # without header
    end

    # randomizzazione di NHMC
    binom_distr = Binomial(NHMC*10,1/10)

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
    comp_ener_args = (x, ener, gi_dx, max_ptord, Npoint, n_comps, x2, rad, selectdim(arg_rad, 4, 1),
                      buff_scA, buff_scB, buff_scC, buff_vec, buffs_sc, unita)
    # compilazione dei kernel
    gx2_c = compile_kernel(g_quadro!, gx2_args, Npoint2)
    grad_ker_c = compile_kernel(gradient_kernel!, grad_ker_args, Npoint2)
    vec_grad_c = compile_kernel(compute_vec_grad!, vec_grad_args, Npoint2)
    zero_mode_c = compile_kernel(subtract_zero_mode!, zero_mode_args, Npoint2)
    comp_ener_c = compile_kernel(compute_ener!, comp_ener_args, Npoint2)
    # creazione di struct con i kernel compilati
    gx2 = CompiledKernel(gx2_args, gx2_c...)
    grad_ker = CompiledKernel(grad_ker_args, grad_ker_c...)
    vec_grad = CompiledKernel(vec_grad_args, vec_grad_c...)
    zero_mode = CompiledKernel(zero_mode_args, zero_mode_c...)
    comp_ener = CompiledKernel(comp_ener_args, comp_ener_c...)

    jobid = get_job_id()

    for t = args.iter_start+1:n_meas
        for _ in 1:measure_every
            reset_moment!(Pi, cuda_rng)
            for __ = 1:rand(nhmc_rng, binom_distr)
                omf_evolution!(x, Pi, dt, rad, buff_scA, gradient, gx2, grad_ker, vec_grad)
            end
        end
        compute_energy!(energia, x, ener, zero_modo, Npoint, zero_mode, gx2, comp_ener)
        @inbounds @views ener_meas[:,:,:,t] .= ener
        write_line(file, Array(energia))
        x_back .= x
        args.iter_start = t
        if get_remaining_time(jobid) < 120
            # 2 min remaining, save state and exit
            save_state(checkpt_fname, args, lat_checkpt, ener_meas)
            execute_self(checkpt_fname, lat_fname)
            return
        end
    end
    save_lat(lat_fname, Array(ener_meas))
    close(file)
    println("Energy written to file ", en_fname)
    remove_files(checkpt_fname, lat_checkpt)
    return
end

function launch_main_omf(Npoint::Int, n_meas::Int, NHMC::Int, dt::Real, n_comps::Int,
                        max_ptord::Int, measure_every::Int, lat_fname::AbstractString=""; cuda_seed::Integer=0, nhmc_seed::Integer=0)
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
        return
    end
    main_omf(args, en_fname, lat_fname)
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
        return
    end
    main_omf(args, en_fname, lat_fname)
    return
end
