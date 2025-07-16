struct CompiledKernel
    args::Tuple
    kernel::CUDA.HostKernel
    threads::Int
    blocks::Int
end

function compile_kernel(fun::Function, fun_args::Tuple, len_arr::Int)
    kernel = @cuda launch=false fun(fun_args...)
    config = launch_configuration(kernel.fun)
    threads = min(len_arr, config.threads)
    blocks = cld(len_arr, threads)
    return kernel, threads, blocks
end

@inline function get_indexes(ind::Int, Npoint::Int)
    # converts from one-dimensional index ind to two-dimensional indexes i, k
    i = rem(ind-1, Npoint) + 1
    k = div(ind-1, Npoint) + 1
    return i, k
end
