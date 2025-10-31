
using DrWatson

using InvertibleNetworks
using Random
using Distributions
using Statistics
using ProgressMeter
using PyPlot
using Seaborn
using LinearAlgebra
using Flux
using HDF5

font_prop = sef_plot_configs()[1]
args = read_config("test_rosenbrock_madcow.json")
args = parse_input_args(args)

if args["epoch"] == -1
    args["epoch"] = args["max_epoch"]
end

save_path = plotsdir(args["sim_name"], savename(args))

# Noise distribution
π_t = Normal(0.0f0, args["sigma"])


for j = 1:args["madcow"]

    args["madcow_idx"] = j

    # Define network
    G = NetworkGlow(2, args["n_hidden"], args["depth"], 1, freeze_conv = true)

    # Loading the experiment—only network weights and training loss
    loaded_keys = load_experiment(args, ["G", "fval", "fval_eval"])
    G = loaded_keys["G"]
    fval = loaded_keys["fval"]
    fval_eval = loaded_keys["fval_eval"]

    # Testing data
    test_size = 10000
    RB_dist = RosenbrockDistribution(0.0f0, 5.0f-1)
    X_test = rand(RB_dist, test_size)

    Zx = args["r"] .* randn(Float32, 1, 1, 2, test_size)

    # Precited samples
    X_ = G.inverse(Zx)

    # Training loss
    fig = figure("training logs", figsize = (7, 4))
    if args["epoch"] == args["max_epoch"]
        plot(
            range(0, args["epoch"], length = length(fval)),
            fval,
            color = "#4a4a4a",
            label = "training loss",
        )
        plot(
            range(0, args["epoch"], length = length(fval_eval)),
            fval_eval,
            color = "#a1a1a1",
            label = "validation loss",
        )
    else
        plot(
            range(0, args["epoch"], length = length(fval[1:findfirst(fval .== 0.0f0)-1])),
            fval[1:findfirst(fval .== 0.0f0)-1],
            color = "#4a4a4a",
            label = "training loss",
        )
        plot(
            range(
                0,
                args["epoch"],
                length = length(fval_eval[1:findfirst(fval_eval .== 0.0f0)-1]),
            ),
            fval_eval[1:findfirst(fval_eval .== 0.0f0)-1],
            color = "#a1a1a1",
            label = "validation loss",
        )
    end
    legend()
    title("Training objective at MC iteration " * string(j - 1))
    ylabel(L"KL divergence + $const.$")
    xlabel("Epochs")
    xlim([0.0, args["epoch"]])
    ylim([2.2, 3.88])
    wsave(joinpath(save_path, "training-obj" * string(j) * ".png"), fig)
    close(fig)

    # True samples from Rosenbrock distribuition.
    fig_real, ax_real = subplots(1, 1, figsize = (5, 5))
    ax_real.scatter(
        X_test[1, 1, 1, :],
        X_test[1, 1, 2, :],
        s = 0.5,
        color = "#000000",
        alpha = 0.3,
    )
    ax_real.set_xlim([-3.5, 3.5])
    ax_real.set_ylim([-2.5, 8])
    ax_real.grid(false)
    ax_real.set_title("True samples")
    wsave(joinpath(save_path, "true-samples.png"), fig_real)
    close(fig_real)

    fig, ax = subplots(1, 1, figsize = (5, 5))
    ax.scatter(X_[1, 1, 1, :], X_[1, 1, 2, :], s = 0.5, color = "#000000", alpha = 0.3)

    ax.set_xlim([-3.5, 3.5])
    ax.set_ylim([-2.5, 8])
    ax.grid(false)
    ax.set_title("Iteration " * string(args["madcow_idx"] - 1))
    wsave(
        joinpath(
            save_path,
            "nf-samples_itr-" * string(j - 1) * "_r-" * string(args["r"]) * ".png",
        ),
        fig,
    )
    close(fig)
end
