using Dates
include("ON_OMF_funcs_copies.jl")

function print_usage()
    println("Usage: $PROGRAM_FILE checkpoint_fname")
    println("   or: $PROGRAM_FILE checkpoint_fname lat_file")
    println("   or: $PROGRAM_FILE Npoint n_meas NHMC dt n_indep_comps max_ptord")
    println("   or: $PROGRAM_FILE Npoint n_meas NHMC dt n_indep_comps max_ptord measure_every")
    println("   or: $PROGRAM_FILE Npoint n_meas NHMC dt n_indep_comps max_ptord measure_every lat_file")
end

function parse_args(args; floatType=Float32, intType=Int32)
    possible_n_args = (1, 2, 6, 8, 9)
    n_args = length(args)
    if !(n_args in possible_n_args)
        print_usage()
        exit(1)
    end
    if n_args == 1 || n_args == 2
        return args
    end
    Npoint = parse(intType, args[1])
    n_meas = parse(intType, args[2])
    NHMC = parse(intType, args[3])
    dt = parse(floatType, args[4])
    n_comps = parse(intType, args[5])
    n_copies = parse(intType, args[6])
    max_ptord = parse(intType, args[7])
    n_copies = parse(intType, args[8])
    measure_every = n_args > 7 ? parse(intType, args[9]) : 1
    if n_args < 9
        return Npoint, n_meas, NHMC, dt, n_comps, n_copies, max_ptord, measure_every
    end
    return Npoint, n_meas, NHMC, dt, n_comps, n_copies, max_ptord, measure_every, args[9]
end

println(now(), ": Library compilation successful.")
@time launch_main_omf(parse_args(ARGS)...)
