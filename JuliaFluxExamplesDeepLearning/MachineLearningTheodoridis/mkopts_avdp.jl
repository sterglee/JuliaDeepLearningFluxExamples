function mkopts_avdp()
    opts = (
        algorithm                      = "vdp",
        use_kd_tree                    = true,
        initial_K                      = 1,
        do_greedy                      = true,
        do_split                       = false,
        do_merge                       = false,
        do_sort                        = true,
        initial_depth                  = 4,
        max_target_ratio               = 0.1,
        recursive_expanding_depth      = 2,
        recursive_expanding_threshold  = 0.1,
        recursive_expanding_frequency  = 3
        )
    return opts
end

