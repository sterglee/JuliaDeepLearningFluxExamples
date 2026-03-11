using Zygote  # For automatic differentiation
using Plots

# 1. Define the objective function
# Equivalent to the lambda or function defined in the Python notebook
f(x) = x^4 - 3x^3 + 2

# 2. Gradient Descent Algorithm
function gradient_descent(f, start_x; lr=0.01, precision=0.00001, max_iters=10000)
  x = start_x
  history = [x]

  for i in 1:max_iters
    # Zygote.gradient returns a tuple of gradients for each input
    grad = gradient(f, x)[1]

    prev_x = x
    x = x - lr * grad  # The update rule

    push!(history, x)

    # Check for convergence
    if abs(x - prev_x) < precision
      println("Converged in $i iterations.")
      break
    end
  end
  return x, history
end

# 3. Execution
start_pos = 6.0
min_x, x_history = gradient_descent(f, start_pos, lr=0.01)

println("The local minimum occurs at: ", round(min_x, digits=4))

# 4. Visualization
x_range = 0:0.1:6
plot(x_range, f.(x_range), label="f(x)", title="Gradient Descent Progress", lw=2)
scatter!(x_history, f.(x_history), label="Steps", color=:red, markersize=3, alpha=0.6)
