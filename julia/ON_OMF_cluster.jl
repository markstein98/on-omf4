using Dates
include("ON_OMF_funcs.jl")

function print_usage()
    println("Usage: $PROGRAM_FILE checkpoint_fname")
    println("   or: $PROGRAM_FILE checkpoint_fname lat_file")
    println("   or: $PROGRAM_FILE Npoint n_meas NHMC dt n_indep_comps max_ptord")
    println("   or: $PROGRAM_FILE Npoint n_meas NHMC dt n_indep_comps max_ptord measure_every")
    println("   or: $PROGRAM_FILE Npoint n_meas NHMC dt n_indep_comps max_ptord measure_every lat_file")
end

function parse_args(args)
    possible_n_args = (1, 2, 6, 7, 8)
    n_args = length(args)
    if !(n_args in possible_n_args)
        print_usage()
        exit(1)
    end
    if n_args == 1 || n_args == 2
        return args
    end
    Npoint = parse(Int, args[1])
    n_meas = parse(Int, args[2])
    NHMC = parse(Int, args[3])
    dt = parse(Float32, args[4])
    n_comps = parse(Int, args[5])
    max_ptord = parse(Int, args[6])
    measure_every = n_args > 6 ? parse(Int, args[7]) : 1
    if n_args < 8
        return Npoint, n_meas, NHMC, dt, n_comps, max_ptord, measure_every
    end
    return Npoint, n_meas, NHMC, dt, n_comps, max_ptord, measure_every, args[8]
end

println(now(), ": Library compilation successful.")
@time launch_main_omf(parse_args(ARGS)...)
