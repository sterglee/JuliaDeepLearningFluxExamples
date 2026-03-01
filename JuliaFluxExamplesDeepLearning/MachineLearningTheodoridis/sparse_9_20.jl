using LinearAlgebra

# 1. Define inputs
A = [0.5 2.0 1.5; 1.5 2.3 3.5]
x = [2.5, 0.0, 0.0]
y = A * x

# --- L2 Norm Minimization ---
# θ_L2 = A' * inv(A * A') * y is equivalent to pinv(A) * y
theta2 = pinv(A) * y
error_L2 = norm(y - A * theta2)

println("L2 norm minimization solution:")
display(theta2)
println("Error L2: ", error_L2)

# --- Exhaustive Search for 1-Sparse Solutions ---
println("\nChecking potential 1-sparse solutions:")

for i in 1:3
    subA = A[:, i]
    # Solving the 1D least squares: val = (A_i' * A_i) \ (A_i' * y)
    val = (subA' * subA) \ (subA' * y)

    xx = zeros(3)
    xx[i] = val
    err = norm(y - A * xx)

    println("Checking solution [column $i]: Error = $err")
    if err < 1e-10
        println("Found 1-sparse solution at column $i:")
        display(xx)
    end
end

# --- Comparison of Norms ---
# Using the 1-sparse solution from column 1 (xx1)
theta0 = [2.5, 0.0, 0.0]
println("\nFinal Comparison:")
println("L2 norm of L2 solution: ", norm(theta2))
println("L2 norm of L0 solution: ", norm(theta0))
\

