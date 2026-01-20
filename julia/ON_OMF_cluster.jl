using Dates

include("ON_OMF_funcs_copies.jl")
include("load_config.jl")

function print_usage()
    println("Usage: $PROGRAM_FILE config.toml")
    spaces = ' ' ^ length("Usage: ")
    println(spaces, "to start from scratch")
    println(spaces, "Required keys in config file:")
    println(spaces, required_config_keys)
    println(spaces, "See simulation_configurations/sample_configuration.toml for further details.")
    println("   or: $PROGRAM_FILE config.toml checkpoint.jld")
    println(spaces, "to resume from a previous checkpoint")
end

function parse_args(args)
    if length(args) > 0
        if !isfile(args[1])
            error("Configuration file not found: ", args[1])
        end
        if length(args) == 1
            check_required_keys(args[1], true)
            return args
        end
        if length(args) == 2
            check_required_keys(args[1], false)
            if !isfile(args[2])
                error("Checkpoint file not found: ", args[2])
            end
            return args
        end
    end
    print_usage()
    exit(1)
end

# ---- Program entry point ----

println(now(), ": Library compilation successful. Starting execution.")
launch_main_omf(parse_args(ARGS)...)
println(now(), ": Execution terminated.")
