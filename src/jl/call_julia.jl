#=
call_julia.jl:
- Julia version: 
- Author: hyeonah
- Date: 2021-12-23
=#
using TSPLIB
using CVRPLIB
# using CVRPSEP
using CPLEX, JuMP
import Base.@kwdef
include("./cvrp_cutting.jl")


@kwdef mutable struct CutOptions
    use_exact_rounded_capacity_cuts :: Bool = false
    use_learned_rounded_capacity_cuts :: Bool = false
    use_rounded_capacity_cuts :: Bool = true
end


function experiment(name::String)

    include("../jl/cvrp_cutting.jl")

    k = parse(Int, split(name, 'k')[end]) # minimum number of vehicles required
    cvrp, vrp_file, sol_file = readCVRP(name)

    cplex = optimizer_with_attributes(
        CPLEX.Optimizer,
        "CPX_PARAM_SCRIND" => 0, # output level
        # "CPX_PARAM_LPMETHOD" => 2, # 1:Primal Simplex, 2: Dual Simplex, 3: Network Simplex, 4: Barrier
        # "CPXPARAM_Barrier_ConvergeTol" => 1e-12,
        # "CPXPARAM_Simplex_Tolerances_Optimality" => 1e-9
        # "CPXPARAM_Advance" => 1,
    )
    my_optimizer = cplex

    cut_options = CutOptions(
        use_rounded_capacity_cuts               = false,
        use_learned_rounded_capacity_cuts       = false,
        use_exact_rounded_capacity_cuts         = true,
    )

    @time lowerbound, list_e, list_x, list_s, list_rhs, list_z = generate_data(cvrp, k, my_optimizer, cut_options; max_n_cuts=k)
#     @show name, opt, lowerbound

    return cvrp, lowerbound, list_e, list_x, list_s, list_rhs, list_z
end


# function get_cvrp_data(name::String, dimention::Int, edge_weight_type::String, capacity::Int, depot::Int, dummy::Int, customer::Vector, coords::Matrix, demands::Vector, k::Int, opt::Float64)
#     include("../jl/cvrp_cutting.jl")
#
# #     dimension = Int.(problem.dimension)
# #     depot = problem.depots[1]
# #     dummy = dimension + 1
# #     customers = setdiff(1:(dimension+1), [depot, dummy])
#
#     weights = Int.(TSPLIB.calc_weights(edge_weight_type, coords))
#
#     cvrp = CVRP(
#                 name,
#                 dimension,
#                 edge_weight_type,
#                 weights,
#                 capacity,
#                 coords[1:end-1, :],
#                 demands,
#                 depot,
#                 dummy,
#                 customers
#     )
#
#
#     cplex = optimizer_with_attributes(
#         CPLEX.Optimizer,
#         "CPX_PARAM_SCRIND" => 0, # output level
#         # "CPX_PARAM_LPMETHOD" => 2, # 1:Primal Simplex, 2: Dual Simplex, 3: Network Simplex, 4: Barrier
#         # "CPXPARAM_Barrier_ConvergeTol" => 1e-12,
#         # "CPXPARAM_Simplex_Tolerances_Optimality" => 1e-9
#         # "CPXPARAM_Advance" => 1,
#     )
#     my_optimizer = cplex
#
#     cut_options = CutOptions(
#         use_rounded_capacity_cuts               = false,
#         use_learned_rounded_capacity_cuts       = false,
#         use_exact_rounded_capacity_cuts         = true,
#     )
#
#     lowerbound, list_e, list_x, list_s, list_rhs, list_z = generate_data(cvrp, k, my_optimizer, cut_options; max_n_cuts=k)
# #     @time lowerbound, list_e, list_x, list_s, list_rhs, list_z = generate_data(cvrp, k, my_optimizer, cut_options; max_n_cuts=k)
# #     @show opt, lowerbound
#
#     return cvrp, lowerbound, list_e, list_x, list_s, list_rhs, list_z
# end


function get_cvrp_data(problem::Any, coords::Matrix, demands::Vector, k::Int, opt::Float64)
    include("../jl/cvrp_cutting.jl")

    dimension = Int.(problem.dimension)
    depot = problem.depots[1]
    dummy = dimension + 1
    customers = setdiff(1:(dimension+1), [depot, dummy])

    weights = Int.(TSPLIB.calc_weights(problem.edge_weight_type, coords))

    cvrp = CVRP(
                problem.name,
                dimension,
                problem.edge_weight_type,
                weights,
                problem.capacity,
                coords[1:end-1, :],
                demands,
                depot,
                dummy,
                customers
    )


    cplex = optimizer_with_attributes(
        CPLEX.Optimizer,
        "CPX_PARAM_SCRIND" => 0, # output level
        # "CPX_PARAM_LPMETHOD" => 2, # 1:Primal Simplex, 2: Dual Simplex, 3: Network Simplex, 4: Barrier
        # "CPXPARAM_Barrier_ConvergeTol" => 1e-12,
        # "CPXPARAM_Simplex_Tolerances_Optimality" => 1e-9
        # "CPXPARAM_Advance" => 1,
    )
    my_optimizer = cplex

    cut_options = CutOptions(
        use_rounded_capacity_cuts               = false,
        use_learned_rounded_capacity_cuts       = false,
        use_exact_rounded_capacity_cuts         = true,
    )

#     lowerbound, list_e, list_x, list_s, list_rhs, list_z = generate_data(cvrp, k, my_optimizer, cut_options; max_n_cuts=k)
    @time lowerbound, list_e, list_x, list_s, list_rhs, list_z = generate_data(cvrp, k, my_optimizer, cut_options; max_n_cuts=k)
    @show opt, lowerbound

    return cvrp, lowerbound, list_e, list_x, list_s, list_rhs, list_z
end