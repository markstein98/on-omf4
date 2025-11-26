using CUDA
using Combinatorics

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
    scalar_buffs = CuArray{F}(undef, n_ptords, Npoint, Npoint, 4)
    @inbounds sc_buffA = @view scalar_buffs[:,:,:,1]
    @inbounds sc_buffB = @view scalar_buffs[:,:,:,2]
    @inbounds sc_buffC = @view scalar_buffs[:,:,:,3]
    @inbounds sc_buffD = @view scalar_buffs[:,:,:,4]
    scalar_buffs2 = CuArray{F}(undef, n_ptords, Npoint, Npoint, 4)
    fraction = CuArray{F}(undef, n_ptords, Npoint, Npoint)

    # Functions returning the coordinates of the nearest neighbour of site (i, k) in the given direction
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

    @inline function get_indexes()
        # converts from one-dimensional index ind to two-dimensional indexes i, k
        ind = (blockIdx().x - one(I)) * blockDim().x + threadIdx().x
        i::I = rem(ind-one(I), Npoint) + one(I)
        k::I = div(ind-one(I), Npoint) + one(I)
        return (i, k)
    end

    @inline function multiplication!(
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
                multiplication!(x, x, vec_buffer, i, k, i, k, n, n)
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
                multiplication!(x, x, vec_buffer, i, k, right..., n, n)
                @inbounds @views ener[I(2):n_ptords,i,k] .+= F(0.5) .* vec_buffer[I(1):max_ptord,i,k,n]
                multiplication!(x, x, vec_buffer, i, k, down..., n, n)
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
            multiplication!(scalar_buffs2, scalar_buffs2, scalar_buffs, i, k, i, k, I(1), I(2))
            multiplication!(scalar_buffs2, scalar_buffs2, scalar_buffs, i, k, i, k, I(3), I(1))

            @inbounds @views ener[:,i,k] .+= F(0.5) .* (scalar_buffs[:,i,k,I(1)] .+ scalar_buffs[:,i,k,I(3)])
        end
        return nothing
    end

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
                @inbounds @views sc_buffA[j,i,k] += - scalar_buffs2[j,i,k,I(1)] - scalar_buffs2[j,i,k,I(2)] - scalar_buffs2[j,i,k,I(3)] - scalar_buffs2[j,i,k,I(4)]
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
