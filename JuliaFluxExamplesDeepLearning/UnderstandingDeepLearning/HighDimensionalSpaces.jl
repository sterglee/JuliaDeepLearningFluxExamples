using Statistics
using LinearAlgebra
using Plots

# ---------------------------------------------------------
# 1. Distance Concentration
# ---------------------------------------------------------
# Calculate distances between pairs of random points in different dimensions
function study_distances(dims)
    n_pairs = 1000
    results = []
    for d in dims
        # Generate two sets of random points in d-dimensions
        pts1 = randn(Float32, d, n_pairs)
        pts2 = randn(Float32, d, n_pairs)

        # Calculate Euclidean distances
        dists = [norm(pts1[:, i] - pts2[:, i]) for i in 1:n_pairs]

            # Normalize by expected distance (sqrt(2d) for standard normal)
            # to show how the distribution "tightens" relative to the mean
            push!(results, dists ./ sqrt(2*d))
        end
        return results
    end

    dimensions = [2, 10, 100, 1000]
    dist_data = study_distances(dimensions)

    p1 = histogram(dist_data[1], alpha=0.5, label="2D", title="Distance Concentration", normalize=:pdf)
    for i in 2:4
        histogram!(p1, dist_data[i], alpha=0.5, label="$(dimensions[i])D", normalize=:pdf)
    end
    xlabel!("Normalized Distance")

    # ---------------------------------------------------------
    # 2. Orthogonality in High Dimensions
    # ---------------------------------------------------------
    # Calculate angles between random vectors
    function study_angles(dims)
        n_pairs = 1000
        results = []
        for d in dims
            angles = []
            for i in 1:n_pairs
                v1 = randn(d); v2 = randn(d)
                # Angle in degrees: acos(dot product / product of norms)
                cos_theta = dot(v1, v2) / (norm(v1) * norm(v2))
                push!(angles, rad2deg(acos(clamp(cos_theta, -1, 1))))
            end
            push!(results, angles)
        end
        return results
    end

    angle_data = study_angles(dimensions)
    p2 = histogram(angle_data[1], alpha=0.5, label="2D", title="Angular Distribution")
    for i in 2:4
        histogram!(p2, angle_data[i], alpha=0.5, label="$(dimensions[i])D")
    end
    xlabel!("Angle (Degrees)")

    # ---------------------------------------------------------
    # 3. Volume in the "Outer Shell"
    # ---------------------------------------------------------
    # Proportion of volume in the outer 1% of the radius: 1 - (0.99)^d
    get_shell_volume(d) = 1.0 - (0.99)^d

    dims_range = 1:500
    shell_vols = [get_shell_volume(d) for d in dims_range]

        p3 = plot(dims_range, shell_vols, lw=2, color=:black,
                  title="Volume in Outer 1% Shell", legend=false)
        ylabel!("Proportion of Volume")
        xlabel!("Dimension")



        display(plot(p1, p2, p3, layout=(1, 3), size=(1200, 400)))



