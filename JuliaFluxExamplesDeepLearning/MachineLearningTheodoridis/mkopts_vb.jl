function mkopts_vb(K)
    opts = (
        algorithm     = "non_dp",
        use_kd_tree   = false,
        initial_K     = K,
        do_greedy     = false,
        do_split      = false,
        do_merge      = false
    )
    return opts
end
