using Dates

include("ON_OMF_funcs_copies.jl")
include("load_config.jl")

function print_usage()
    println("Usage: $PROGRAM_FILE start config.toml")
    spaces = ' ' ^ length("Usage: ")
    println(spaces, "to start from scratch")
    println(spaces, "Required keys in config file:")
    println(spaces, required_config_keys)
    println(spaces, "See simulation_configurations/sample_configuration.toml for further details.")
    println("   or: $PROGRAM_FILE load checkpoint.jld")
    println(spaces, "to resume from a previous checkpoint")
end

function parse_args(args)
    if length(args) == 2
        if lowercase(args[1]) in ["start", "s"]
            if !isfile(args[2])
                error("Configuration file not found: ", args[1])
            end
            check_required_keys(args[2], true)
            return 0
        elseif lowercase(args[1]) in ["load", "l", "resume", "r"]
            if !isfile(args[2])
                error("Checkpoint file not found: ", args[2])
            end
            return 1
        end
    end
    print_usage()
    return -1
end

# ---- Program entry point ----

println(now(), ": Library compilation successful. Starting execution.")
status = parse_args(ARGS)
if status == 0
    launch_main_omf(args[2])
elseif status == 1
    resume_main_omf(args[2])
end
println(now(), ": Execution terminated.")
