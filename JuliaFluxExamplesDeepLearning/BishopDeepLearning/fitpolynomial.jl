using Random
using LinearAlgebra
using GLMakie
using Statistics

# --- 1. Data Generation ---
Random.seed!(42)
f(x) = sin(2π * x)
N_train = 10
x_train = range(0, 1, length=N_train)
t_train = f.(x_train) .+ 0.1 * randn(N_train)

N_test = 100
x_test = range(0, 1, length=N_test)
t_test = f.(x_test)

# --- 2. Helper Functions ---
function design_matrix(x, M)
    return Float64[xi^m for xi in x, m in 0:M]
    end

    function fit_polynomial(X, t)
        return X \ t
    end

    function rms_error(X, t, w)
        return sqrt(mean((X * w .- t).^2))
    end

    # --- 3. Visualization Setup ---
    fig = Figure(size = (1000, 500))

    # Axis 1: The Curves
    ax1 = Axis(fig[1, 1], title = "Polynomial Fitting",
               xlabel = "x", ylabel = "t",
               limits = (nothing, (-1.5, 1.5)))

    # Axis 2: The Errors (Log scale is better for visualizing overfitting)
    ax2 = Axis(fig[1, 2], title = "RMS Error (Log Scale)",
               xlabel = "Degree (M)", ylabel = "ERMS",
               yscale = log10, xticks = 0:9)

    # --- 4. Plotting the Fits ---
    x_plot = range(0, 1, length=200)
    Ms_to_plot = [0, 1, 3, 9]
    colors = [:orange, :green, :blue, :red]

    # Reference lines
    lines!(ax1, x_plot, f.(x_plot), color = :black, linestyle = :dash, label = "True Sine")
    scatter!(ax1, x_train, t_train, color = :black, markersize = 12, label = "Data")

    for (i, M) in enumerate(Ms_to_plot)
        X_tr = design_matrix(x_train, M)
        w = fit_polynomial(X_tr, t_train)

        X_pl = design_matrix(x_plot, M)
        y_pl = X_pl * w
        lines!(ax1, x_plot, y_pl, color = colors[i], linewidth = 2, label = "M=$M")
    end
    axislegend(ax1, position = :rt)

    # --- 5. Calculating and Plotting Errors ---
    train_errors = Float64[]
    test_errors = Float64[]
    degrees = 0:9

    for M in degrees
        X_tr = design_matrix(x_train, M)
        X_te = design_matrix(x_test, M)
        w = fit_polynomial(X_tr, t_train)
        push!(train_errors, rms_error(X_tr, t_train, w))
        push!(test_errors, rms_error(X_te, t_test, w))
    end

    # Using scatterlines! for the combined Marker + Line effect
    scatterlines!(ax2, degrees, train_errors, color = :blue, marker = :circle, label = "Training")
    scatterlines!(ax2, degrees, test_errors,  color = :red,  marker = :rect,   label = "Test")
    axislegend(ax2, position = :lt)

    # --- 6. Finalize ---
    display(fig)
    save("poly_fit_glmakie.png", fig)
    println("Image saved as poly_fit_glmakie.png")

