using LinearAlgebra
using Statistics
using Printf
using Random

# --- Utility Functions ---

"""
Standard sigmoid function.
    """
    function sigmoid(x)
        return 1.0 ./ (1.0 .+ exp.(-x))
    end

    """
    Diagnostic logger to mimic the SparseBayes behavior.
    """
    function sb1_diagnostic(level::Int, fmt::String, args...)
        # Set threshold; level 4 is for detailed PosteriorMode iterations
        if level <= 1
         println(level);
         println(args...)
        end
    end

    # --- The SparseBayes Posterior Mode Finder ---

    """
    sb1_posterior_mode(PHI, t, w, alpha, its)

    Find the mode of the posterior distribution for the Bernoulli case (Classification).
        This implements the IRLS (Iteratively Reweighted Least Squares) / Newton-Raphson
        algorithm with line-search (backtracking).
        """
        function sb1_posterior_mode(PHI::Matrix{Float64}, t::Vector, w_init::Vector{Float64}, alpha::Vector{Float64}, its::Int)

            # Constants
            GRAD_STOP   = 1e-6
            LAMBDA_MIN  = 2.0^(-8)

            N, d = size(PHI)
            M = length(w_init)
            w = copy(w_init)
            A = Diagonal(alpha)

            # Initial setup
            phi_w = PHI * w
            y = sigmoid(phi_w)
            t_bool = convert(Vector{Bool}, t)

            # Compute initial value of log posterior (as an error)
            # Using small epsilon to prevent log(0)
            eps_val = 1e-12
            data_term = -(sum(log.(y[t_bool] .+ eps_val)) + sum(log.(1.0 .- y[.!t_bool] .+ eps_val))) / N
            regulariser = (alpha' * (w.^2)) / (2 * N)
            err_new = data_term + regulariser

            errs = zeros(its)
            U = zeros(M, M) # Initialize for scope
            data_term_final = data_term # Initialize for scope

            for i in 1:its
                y_var = y .* (1.0 .- y)
                # Weight PHI by variance (equivalent to PHI' * V * PHI)
                phi_v = PHI .* y_var

                e = t .- y

                # Compute gradient vector and Hessian matrix
                g = PHI' * e .- alpha .* w
                Hessian = (phi_v' * PHI + A)

                # Condition check on first iteration
                if i == 1
                    cond_hess = 1.0 / cond(Hessian)
                    if cond_hess < eps()
                        error("Sorry! Cannot recover from ill-conditioned Hessian ($cond_hess). Try reducing kernel length scale.")
                    end
                end

                errs[i] = err_new
                sb1_diagnostic(4, "PosteriorMode Cycle: %2d\t error: %.6f\n", i, errs[i])

                # Check for convergence
                if i >= 2 && norm(g) / M < GRAD_STOP
                    sb1_diagnostic(4, "(PosteriorMode) converged after %d iterations\n", i)
                    break
                end

                # Newton step
                try
                    U = cholesky(Hermitian(Hessian)).U
                catch
                    @warn "Hessian not positive definite; aborting Newton step."
                    break
                end

                delta_w = U \ (U' \ g)
                lambda = 1.0

                while lambda > LAMBDA_MIN
                    w_new = w .+ lambda .* delta_w
                    phi_w_new = PHI * w_new
                    y_new = sigmoid(phi_w_new)

                    # Compute new error
                    if any(y_new[t_bool] .== 0) || any(y_new[.!t_bool] .== 1)
                        err_new = Inf
                    else
                        data_term_new = -(sum(log.(y_new[t_bool] .+ eps_val)) + sum(log.(1.0 .- y_new[.!t_bool] .+ eps_val))) / N
                        regulariser_new = (alpha' * (w_new.^2)) / (2 * N)
                        err_new = data_term_new + regulariser_new
                    end

                    if err_new > errs[i]
                        lambda /= 2.0
                        sb1_diagnostic(4, "(PosteriorMode) error increase! Backing off ... (%.3f)\n", lambda)
                    else
                        # Error decreased: update state
                        w = w_new
                        y = y_new
                        data_term_final = (-(sum(log.(y[t_bool] .+ eps_val)) + sum(log.(1.0 .- y[.!t_bool] .+ eps_val))) / N)
                        lambda = 0.0 # Exit search
                    end
                end

                if lambda > 0
                    sb1_diagnostic(4, "(PosteriorMode) stopping due to back-off limit.\n")
                    break
                end
            end

            # Ui is the inverse Cholesky factor (used in RVM for marginal likelihood)
            Ui = inv(U)
            log_mode = -N * data_term_final

            return w, Ui, log_mode
        end

        # --- Main Demo ---

        function main()
            Random.seed!(42)
            println("--- Running SB1_PosteriorMode Demo ---")

            # 1. Generate synthetic classification data
            N, D = 100, 5
            X = randn(N, D)
            true_w = [2.0, -1.5, 0.0, 0.5, -3.0]

            # Generate labels (t) using sigmoid
            logits = X * true_w
            probs = sigmoid(logits)
            t = [p > rand() ? 1.0 : 0.0 for p in probs]

                # 2. Parameters for the finder
                w_init = zeros(D)
                alpha = fill(1.0, D) # Hyperparameters (regularization)
                max_its = 25

                # 3. Call the function
                w_mode, Ui, lmode = sb1_posterior_mode(X, t, w_init, alpha, max_its)

                # 4. Results
                println("\nResults:")
                println("Weights at mode: ", round.(w_mode, digits=4))
                println("True weights:    ", true_w)
                @printf("Log Likelihood at mode: %.4f\n", lmode)
                println("UI (Inverse Cholesky of Hessian) size: ", size(Ui))
            end

            # Execute
            main()

