using Flux
using Random
using Statistics
using LinearAlgebra

# --- 1. Model & Masking Setup ---
# We define a function to create a model and a corresponding "mask"
# to simulate sparsity (lottery tickets).
function create_model_with_masks(n_in, n_hidden, n_out)
    model = Chain(
        Dense(n_in, n_hidden, relu),
        Dense(n_hidden, n_hidden, relu),
        Dense(n_hidden, n_out)
        )
    # Create initial masks (all ones) for each weight matrix
    masks = [ones(Float32, size(p)) for p in Flux.params(model) if length(size(p)) > 1]
        return model, masks
    end

    # Helper to apply masks to model weights
    function apply_masks!(model, masks)
        m_idx = 1
        for p in Flux.params(model)
            if length(size(p)) > 1 # Only mask weights, not biases
                p .*= masks[m_idx]
                m_idx += 1
            end
        end
    end

    # --- 2. Pruning Logic ---
    # Identifying the "Winning Ticket" by keeping only the largest weights
    function prune_model!(masks, model, percent)
        m_idx = 1
        for p in Flux.params(model)
            if length(size(p)) > 1
                # Find the threshold for the bottom X percent of weights
                threshold = quantile(abs.(vec(p)), percent)
                # Update mask: 0 if weight is small, 1 if large
                masks[m_idx] .*= (abs.(p) .> threshold)
                m_idx += 1
            end
        end
    end

    # --- 3. Modern Training Function ---
    function train_lottery!(model, masks, X, Y, initial_weights; epochs=100)
        opt_state = Flux.setup(Adam(0.01f0), model)

        for ep in 1:epochs
            loss_val, grads = Flux.withgradient(model) do m
                y_hat = m(X)
                Flux.logitcrossentropy(y_hat, Y)
            end

            # Apply updates
            Flux.update!(opt_state, model, grads[1])

            # CRITICAL: Re-apply masks after every update to keep the network sparse
            apply_masks!(model, masks)
        end
    end

    # --- 4. The Lottery Ticket Workflow ---
    # Replicating the notebook's logic
    function run_lth_experiment()
        # Data Setup (Synthetic)
        X = randn(Float32, 10, 100)
        Y = Flux.onehotbatch(rand(1:3, 100), 1:3)

        # Step 1: Initialize and save original "Winning Ticket" weights
        model, masks = create_model_with_masks(10, 32, 3)
        initial_weights = deepcopy(Flux.params(model))

        println("Phase 1: Training initial dense network...")
        train_lottery!(model, masks, X, Y, initial_weights)

        # Step 2: Prune 50% of the weights
        println("Phase 2: Pruning 50% of weights to find the ticket...")
        prune_model!(masks, model, 0.5)

        # Step 3: Reset weights to original values (The "Ticket" condition)
        # We keep the mask but go back to the initialization
        for (p, p_init) in zip(Flux.params(model), initial_weights)
            p .= p_init
        end
        apply_masks!(model, masks) # Zero out the pruned weights

        println("Phase 3: Training the sparse winning ticket...")
        train_lottery!(model, masks, X, Y, initial_weights)

        return model
    end

    # Run
    run_lth_experiment()

