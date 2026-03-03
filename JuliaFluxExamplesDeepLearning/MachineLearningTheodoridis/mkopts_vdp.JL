function mkopts_vdp()
    opts = (
        algorithm     = "vdp",
        use_kd_tree   = false,
        initial_K     = 1,
        do_greedy     = true,
        do_split      = false,
        do_merge      = false
    )
    return opts
end
