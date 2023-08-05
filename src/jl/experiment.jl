
using(Pickle)
include("cvrp_cutting.jl")

# name = "P-n16-k8"; opt = 450;
# name = "P-n19-k2"; opt = 212;
# name = "E-n13-k4"; opt = 247;
# name = "E-n22-k4"; opt = 375;
# name = "E-n23-k3"; opt = 569;
# name = "P-n101-k4"; opt = 681

# name = "E-n30-k3"; opt = 534
# name = "E-n33-k4"; opt = 835
# name = "E-n51-k5"; opt = 521
# name = "E-n76-k7"; opt = 682
# name = "E-n76-k8"; opt = 735
# name = "E-n76-k10"; opt = 830
# name = "E-n76-k14"; opt = 1021
# name = "E-n101-k8"; opt = 815
# name = "E-n101-k14"; opt = 1067

# name = "M-n101-k10"; opt = 820
# name = "F-n45-k4"; opt = 724
# name = "F-n72-k4"; opt = 237
# name = "F-n135-k7"; opt = 1162

# name = "M-n200-k17"; opt = 1275

# name = "X-n101-k25"; opt = 27591
# name = "X-n106-k14"; opt = 26362

instances = [
("P-n16-k8", 450)  # for build
# ("P-n16-k8", 450)
# ("P-n19-k2", 212)
# ("P-n101-k4", 681)
# ("E-n13-k4", 247)
# ("E-n22-k4", 375)
# ("E-n23-k3", 569)
# ("E-n51-k5", 521)
# ("E-n76-k7", 682)
# ("E-n76-k14", 1021)
# ("E-n101-k8", 815)
# ("E-n101-k14", 1067)
# ("F-n135-k7", 1162)
# ("M-n101-k10", 820)
# ("M-n121-k7", 1034)
# ("M-n151-k12", 1015)
# ("M-n200-k16", 1274)
# ("M-n200-k17", 1275)
# ("X-n101-k25", 27591)
# ("X-n106-k14", 26362)
# ("X-n110-k13", 14971)
# ("X-n115-k10", 12747)
# ("X-n120-k6", 13332)
# ("X-n125-k30", 55539)
# ("X-n129-k18", 28940)
# ("X-n134-k13", 10916)
# ("X-n139-k10", 13590)
# ("X-n143-k7", 15700)
# ("X-n148-k46", 43448)
# ("X-n153-k22", 21220)
# ("X-n157-k13", 16876)
# ("X-n162-k11", 14138)
# ("X-n167-k10", 20557)
# ("X-n172-k51", 45607)
# ("X-n176-k26", 47812)
# ("X-n181-k23", 25569)
# ("X-n186-k15", 24145)
# ("X-n190-k8", 16980)
# ("X-n195-k51", 44225)
# ("X-n200-k36", 58578)
# ("X-n204-k19", 19565)
# ("X-n209-k16", 30656)
# ("X-n214-k11", 10856)
# ("X-n219-k73", 117595)
# ("X-n223-k34", 40437)
# ("X-n228-k23", 25742)
# ("X-n233-k16", 19230)
# ("X-n237-k14", 27042)
# ("X-n242-k48", 82751)
# ("X-n247-k50", 37274)
# ("X-n251-k28", 38684)
# ("X-n256-k16", 18839)
# ("X-n261-k13", 26558)
# ("X-n266-k58", 75478)
# ("X-n270-k35", 35291)
# ("X-n275-k28", 21245)
# ("X-n280-k17", 33503)
# ("X-n284-k15", 20215)
# ("X-n289-k60", 95151)
# ("X-n294-k50", 47161)
# ("X-n298-k31", 34231)
# ("X-n303-k21", 21736)
# ("X-n308-k13", 25859)
# ("X-n313-k71", 94043)
# ("X-n317-k53", 78355)
# ("X-n322-k28", 29834)
# ("X-n327-k20", 27532)
# ("X-n331-k15", 31102)
# ("X-n336-k84", 139111)
# ("X-n344-k43", 42050)
# ("X-n351-k40", 25896)
# ("X-n359-k29", 51505)
# ("X-n367-k17", 22814)
# ("X-n376-k94", 147713)
# ("X-n384-k52", 65928)
# ("X-n393-k38", 38260)
# ("X-n401-k29", 66154)
# ("X-n411-k19", 19712)
# ("X-n420-k130", 107798)
# ("X-n429-k61", 65449)
# ("X-n439-k37", 36391)
# ("X-n449-k29", 55233)
# ("X-n459-k26", 24139)
# ("X-n469-k138", 221824)
# ("X-n480-k70", 89449)
# ("X-n491-k59", 66483)
# ("X-n502-k39", 69226)
# ("X-n513-k21", 24201)
# ("X-n524-k153", 154593)
# ("X-n536-k96", 94846)
# ("X-n548-k50", 86700)
# ("X-n561-k42", 42717)
# ("X-n573-k30", 50673)
# ("X-n586-k159", 190316)
# ("X-n599-k92", 108451)
# ("X-n613-k62", 59535)
# ("X-n627-k43", 62164)
# ("X-n641-k35", 63682)
# ("X-n655-k131", 106780)
# ("X-n670-k130", 146332)
# ("X-n685-k75", 68205)
# ("X-n701-k44", 81923)
# ("X-n716-k35", 43373)
# ("X-n733-k159", 136187)
# ("X-n749-k98", 77269)
# ("X-n766-k71", 114417)
# ("X-n783-k48", 72386)
# ("X-n801-k40", 73305)
# ("X-n819-k171", 158121)
# ("X-n837-k142", 193737)
# ("X-n856-k95", 88965)
# ("X-n876-k59", 99299)
# ("X-n895-k37", 53860)
("X-n916-k207", 329179)
("X-n936-k151", 132715)
("X-n957-k87", 85465)
("X-n979-k58", 118976)
("X-n1001-k43", 72355)
]

cut_options = CutOptions(
    use_exact_rounded_capacity_cuts         = false,
    use_learned_rounded_capacity_cuts       = true,
    use_rounded_capacity_cuts               = false
)

println(cut_options)

for ins in instances
    name = ins[1]
    opt = ins[2]

#     println(name)

    k = parse(Int, split(name, 'k')[end]) # minimum number of vehicles required
    cvrp, vrp_file, sol_file = readCVRPLIB(name, add_dummy=true)

    cplex = optimizer_with_attributes(
        CPLEX.Optimizer,
        "CPX_PARAM_SCRIND" => 0, # output level
        # "CPX_PARAM_LPMETHOD" => 2, # 1:Primal Simplex, 2: Dual Simplex, 3: Network Simplex, 4: Barrier
        # "CPXPARAM_Barrier_ConvergeTol" => 1e-12,
        # "CPXPARAM_Simplex_Tolerances_Optimality" => 1e-9
        # "CPXPARAM_Advance" => 1,
    )
    my_optimizer = cplex

    size = parse(Int, split(split(name, '-')[2], 'n')[end]) - 1

    if size <= 300
        max_iter = 200
    elseif size <= 500
        max_iter = 100
    else
        max_iter = 50
    end

#     if cut_options.use_rounded_capacity_cuts
#         max_iter *= 2
#     end

    start = time()
    @time lowerbound, iter_time, list_time, list_info, iter_cut, const_num, z_list, violations = solve_root_node_relaxation(cvrp, k, my_optimizer, cut_options; max_n_cuts=k, max_iter = max_iter)
    root_time = time() - start
    @show name, opt, lowerbound

    if cut_options.use_learned_rounded_capacity_cuts
        method =  "_learned_imp_128_5"
    elseif cut_options.use_exact_rounded_capacity_cuts
        method = "_exact"
    elseif cut_options.use_rounded_capacity_cuts
        method = "_heuristic_rci_sparse"
    end

    file = string("../../data/results/exe_time_", name, method, ".pkl")
    store(file, [lowerbound, root_time, iter_time, list_time, list_info, iter_cut, const_num, z_list, violations])

end