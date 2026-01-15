using Dates
using TOML

include("ON_OMF_funcs_copies.jl")

# Required fields
required_keys = ["n_side_sites", "n_measurements", "n_HMC_steps", "dt", "n_indep_comps", "max_perturbative_order"]


function print_usage()
    println("Usage: $PROGRAM_FILE config.toml")
    println("Required keys in config file:")
    println(required_keys)
    println("See simulation_configurations/sample_configuration.toml for further details.")
end

"""
    parse_config(fname; floatType=Float32, intType=Int32)

Reads and validates a TOML configuration file.
"""
function parse_config(fname; floatType=Float32, intType=Int32)
    if !isfile(fname)
        error("Configuration file not found: $fname")
    end

    cfg = TOML.parsefile(fname)

    for key in required_keys
        haskey(cfg, key) || error("Missing required config key: $key")
    end

    Npoint        = intType(cfg["n_side_sites"])
    n_meas        = intType(cfg["n_measurements"])
    NHMC          = intType(cfg["n_HMC_steps"])
    dt            = floatType(cfg["dt"])
    n_comps       = intType(cfg["n_indep_comps"])
    max_ptord     = intType(cfg["max_perturbative_order"])

    # Optional fields
    measure_every = intType(get(cfg, "measure_every", 1))
    n_copies      = intType(get(cfg, "n_copies", 1))
    lat_file      = get(cfg, "energy_site_by_site_matlab", "")

    return (Npoint, n_meas, NHMC, dt, n_comps, max_ptord, measure_every, n_copies, lat_file)
end

# ---- Program entry point ----

if length(ARGS) != 1
    print_usage()
    exit(1)
end

println(now(), ": Library compilation successful.")
@time launch_main_omf(parse_config(ARGS[1])...)
