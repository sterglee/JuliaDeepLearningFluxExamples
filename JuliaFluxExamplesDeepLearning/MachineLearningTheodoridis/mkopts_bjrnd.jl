function mkopts_bjrnd(T)
    opts = (
        algorithm     = "bj",
        use_kd_tree   = false,
        sis           = 0,
        initial_K     = T,
        do_greedy     = false,
        do_split      = false,
        do_merge      = false,
        do_sort       = false
        )
    return opts
end

