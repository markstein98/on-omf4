struct CompiledKernel{T <: Integer}
    args::Tuple
    kernel::CUDA.HostKernel
    threads::T
    blocks::T
end

function compile_kernel(fun::Function, fun_args::Tuple, len_arr::T) where {T <: Integer}
    kernel = @cuda launch=false fun(fun_args...)
    config = launch_configuration(kernel.fun)
    threads::T = min(len_arr, config.threads)
    blocks::T = cld(len_arr, threads)
    return CompiledKernel{T}(fun_args, kernel, threads, blocks)
end

@inline function run_kernel(kernel::CompiledKernel)
    kernel.kernel(kernel.args...; threads=kernel.threads, blocks=kernel.blocks)
end

@inline function get_indexes(ind::T, Npoint::T) where {T <: Integer}
    # converts from one-dimensional index ind to two-dimensional indexes i, k
    i::T = rem(ind-one(T), Npoint) + one(T)
    k::T = div(ind-one(T), Npoint) + one(T)
    return i, k
end

@inline function get_indexes(Npoint::T) where {T <: Integer}
    # converts from one-dimensional index, obtained by calling the CUDA API,
    # to two-dimensional indexes i, k
    return get_indexes((blockIdx().x - one(T)) * blockDim().x + threadIdx().x, Npoint)
end
