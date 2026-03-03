using LinearAlgebra

"""
sb1_kernel_function(X1, X2, kernel_str, length_scale)

Computes the kernel matrix K where K[i, j] is the kernel evaluation between
the i-th row of X1 and the j-th row of X2.
"""
function sb1_kernel_function(X1::AbstractMatrix, X2::AbstractMatrix, kernel_str::String, length_scale::Real)
  N1, d = size(X1)
  N2, _ = size(X2)

  # Pre-parse polynomial orders (e.g., 'poly3' -> p=3)
  p = 0
  kernel_type = lowercase(kernel_str)

  if startswith(kernel_type, "poly")
    p = parse(Int, kernel_type[5:end])
    kernel_type = "poly"
    elseif startswith(kernel_type, "hpoly")
    p = parse(Int, kernel_type[6:end])
    kernel_type = "hpoly"
  end

  eta = 1 / length_scale^2

  # Handle kernel types
  if kernel_type == "gauss"
    return exp.(-eta .* dist_sqrd(X1, X2))

    elseif kernel_type == "tps" # Thin-plate spline
    r2 = eta .* dist_sqrd(X1, X2)
    # Handle log(0) by adding (r2 == 0) logic
    return 0.5 .* r2 .* log.(r2 .+ (r2 .== 0))

    elseif kernel_type == "cauchy"
    r2 = eta .* dist_sqrd(X1, X2)
    return 1 ./ (1 .+ r2)

    elseif kernel_type == "cubic"
    r2 = eta .* dist_sqrd(X1, X2)
    return r2 .* sqrt.(r2)

    elseif kernel_type == "r" # Linear distance
    return sqrt.(eta .* dist_sqrd(X1, X2))

    elseif kernel_type == "bubble"
    return Float64.( (eta .* dist_sqrd(X1, X2)) .< 1 )

    elseif kernel_type == "laplace"
    return exp.(-sqrt.(eta .* dist_sqrd(X1, X2)))

    elseif kernel_type == "poly"
    return (X1 * (eta .* X2') .+ 1) .^ p

    elseif kernel_type == "hpoly"
    return (eta .* (X1 * X2')) .^ p

    elseif kernel_type == "spline"
    K = ones(N1, N2)
    X1_scaled = X1 ./ length_scale
    X2_scaled = X2 ./ length_scale
    for i in 1:d
      x1_col = X1_scaled[:, i]
      x2_col = X2_scaled[:, i]

      # Use broadcasting for outer-product-like operations
      XX = x1_col .* x2_col'
        minXX = min.(x1_col, x2_col')

      K .= K .* (1 .+ XX .+ XX .* minXX .- (x1_col .+ x2_col') ./ 2 .* (minXX.^2) .+ (minXX.^3) ./ 3)
    end
    return K

  else
    error("Unrecognised kernel function type: $kernel_str")
    end
  end

  """
  Helper: Squared Euclidean distance between rows of X and rows of Y.
  Equivalent to MATLAB's distSqrd helper but optimized for Julia.
    """
    function dist_sqrd(X, Y)
      # ||x - y||² = ||x||² + ||y||² - 2xᵀy
      sum_x2 = sum(abs2, X, dims=2)
      sum_y2 = sum(abs2, Y, dims=2)
      D2 = sum_x2 .+ sum_y2' .- 2 .* (X * Y')
      # Clean up numerical noise (ensure no negative distances)
      return max.(0.0, D2)
    end


