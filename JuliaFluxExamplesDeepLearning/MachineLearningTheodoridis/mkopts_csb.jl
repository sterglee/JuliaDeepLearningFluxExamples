function mkopts_csb(T)
    opts = (
        algorithm     = "csb",
        use_kd_tree   = false,
        initial_K     = T,
        do_greedy     = false,
        do_split      = false,
        do_merge      = false,
        do_sort       = false
        )
    return opts
end
