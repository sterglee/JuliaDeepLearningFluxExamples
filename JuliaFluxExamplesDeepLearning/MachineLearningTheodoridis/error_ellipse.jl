using LinearAlgebra
using Plots
using Distributions

# -------------------------------
# 1. Error ellipse / ellipsoid
# -------------------------------
function error_ellipse(C::AbstractMatrix; mu=nothing, conf=0.95, scale=1.0, style=:blue, clip=Inf)
  # Validate covariance
  r, c = size(C)
  if r != c || !(r in (2, 3))
    error("C must be 2x2 or 3x3 covariance matrix")
  end
  if any(real(eigvals(C)) .<= 0)
    error("Covariance matrix must be positive definite")
  end

  # Mean
  μ = isnothing(mu) ? zeros(r) : mu

  # Chi-square quantile for confidence (k is the scaling factor)
  # For r=2, 95% confidence corresponds to k ≈ 2.447
  k = sqrt(quantile(Chisq(r), conf))

  # Eigen decomposition
  # eigv contains eigenvalues, eigvec contains orthogonal eigenvectors
  eigv, eigvec = eigen(C)

  # Transformation matrix: Rotate by eigenvectors and scale by sqrt of eigenvalues
  # T = eigenvectors * diag(sqrt(eigenvalues))
  T = eigvec * diagonal(sqrt.(max.(0, eigv)))

  if r == 2
    # 2D ellipse

    θ = range(0, 2π, length=100)
    # Unit circle
    unit_circle = [cos(t) for t in θ, j in 1:1] # shape (100, 1)
      unit_circle = hcat([cos(t) for t in θ], [sin(t) for t in θ])' # 2x100 matrix

        # Transform unit circle to error ellipse
        # points = μ + k * T * unit_circle
        xy = μ .+ (k * scale .* (T * unit_circle))

        X = xy[1, :]
        Y = xy[2, :]

        # Optional clipping
        r_dist = sqrt.((X .- μ[1]).^2 .+ (Y .- μ[2]).^2)
        X[r_dist .> clip] .= NaN
        Y[r_dist .> clip] .= NaN

        plt = plot(X, Y, color=style, label="Conf: $conf", lw=2, aspect_ratio=:equal,
                   xlabel="X", ylabel="Y", title="2D Error Ellipse", legend=true)
        return plt

        elseif r == 3
        # 3D ellipsoid

        u = range(0, 2π, length=50)
        v = range(0, π, length=50)

        # Unit sphere coordinates
        xs = [cos(ui)*sin(vi) for vi in v, ui in u]
          ys = [sin(ui)*sin(vi) for vi in v, ui in u]
            zs = [cos(vi) for vi in v, ui in u]

              # Flatten for matrix multiplication: (3 x 2500)
              unit_sphere = vcat(vec(xs)', vec(ys)', vec(zs)')

              # Transform unit sphere
              # points = μ + k * T * unit_sphere
              xyz = μ .+ (k * scale .* (T * unit_sphere))

              # Reshape back to grid for surface plotting
              X = reshape(xyz[1, :], length(v), length(u))
              Y = reshape(xyz[2, :], length(v), length(u))
              Z = reshape(xyz[3, :], length(v), length(u))

              plt = surface(X, Y, Z, alpha=0.5, color=style, aspect_ratio=:equal,
                            xlabel="X", ylabel="Y", zlabel="Z", title="3D Error Ellipsoid")
              return plt
            end
          end

          # -------------------------------
          # 2. Main script
          # -------------------------------
          function main()
            # Ensure plots are displayed in a window
            gr()

            println("=== 2D Error Ellipse Example ===")
            # Defining a tilted covariance matrix
            C2 = [2.0 0.8; 0.8 1.0]
            μ2 = [1.0, 2.0]
            plt2 = error_ellipse(C2, mu=μ2, conf=0.95, style=:red)
            # Add the center point
            scatter!(plt2, [μ2[1]], [μ2[2]], color=:black, label="Mean")
            display(plt2)

            println("=== 3D Error Ellipsoid Example ===")
            C3 = [2.0 0.5 0.3;
                  0.5 1.0 0.2;
                  0.3 0.2 1.5]
            μ3 = [0.0, 0.0, 0.0]
            plt3 = error_ellipse(C3, mu=μ3, conf=0.9, style=:blue)
            display(plt3)
          end

          # Helper to avoid "diagonal not defined" if using older Julia versions
          diagonal(v) = Matrix(Diagonal(v))

          # -------------------------------
          # 3. Run
          # -------------------------------
          main()

