"""
log_sum_exp(x, dim)

Computes log(sum(exp(x), dims=dim)) using the max-offset trick for stability.
  Works with -Inf but assumes no +Inf.
  """
  function log_sum_exp(x::AbstractArray, dim::Int)
    # Find the maximum value along the specified dimension
    x_max = maximum(x, dims=dim)

    # Handle the case where all elements are -Inf (avoid -Inf - (-Inf) = NaN)
    x_max_cleaned = copy(x_max)
    x_max_cleaned[x_max .== -Inf] .= 0.0

    # x - x_max (Broadcasting handles the 'repmat' logic automatically)
    x_offset = x .- x_max_cleaned

    # y = x_max + log(sum(exp(x_offset)))
    y = x_max .+ log.(sum(exp.(x_offset), dims=dim))

    # Optional: If you want to drop the singleton dimension like MATLAB's default max behavior
    # return dropdims(y, dims=dim)
    return y
  end

