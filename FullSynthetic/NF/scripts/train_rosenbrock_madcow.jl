

using DrWatson


using InvertibleNetworks
using Random
using ProgressMeter
using Flux

# Random seed
Random.seed!(19)

args = read_config("train_rosenbrock_madcow.json")
args = parse_input_args(args)

device = cpu

# Define network.
G = NetworkGlow(2, args["n_hidden"], args["depth"], 1, freeze_conv = true)
G = G |> device

# Training data number.
ntrain = 5120

# Validation data number.
nval = 512

p = Progress(Int(floor(ntrain / args["batchsize"])) * (args["max_epoch"] * args["madcow"]))

for madcow_idx = 1:args["madcow"]

    if madcow_idx == 1
        X_train = rand(RosenbrockDistribution(0.0f0, 5.0f-1), ntrain) |> device
        X_val = rand(RosenbrockDistribution(0.0f0, 5.0f-1), nval) |> device
    else
        Zx = args["r"] .* randn(Float32, 1, 1, 2, ntrain) |> device
        X_train = G.inverse(Zx)

        Zx = args["r"] .* randn(Float32, 1, 1, 2, nval) |> device
        X_val = G.inverse(Zx)
        # Define network.
        global G = NetworkGlow(2, args["n_hidden"], args["depth"], 1, freeze_conv = true)
        global G = G |> device
    end

    # Training Batch extractor.
    train_loader = Flux.DataLoader(X_train, batchsize = args["batchsize"], shuffle = true)
    num_batches = length(train_loader)

    # Optimizer.
    opt = Flux.Optimiser(
        Flux.ExpDecay(args["lr"], 0.9f0, num_batches * args["lr_step"], 1.0f-6),
        Flux.ADAM(args["lr"]),
    )

    # Training log keeper.
    fval = zeros(Float32, num_batches * args["max_epoch"])
    fval_eval = zeros(Float32, args["max_epoch"])

    for epoch = 1:args["max_epoch"]

        fval_eval[epoch] = maximum_likelihood(G, X_val; grad = false)

        for (itr, X) in enumerate(train_loader)
            Base.flush(Base.stdout)

            fval[(epoch-1)*num_batches+itr] = maximum_likelihood(G, X)[1]

            ProgressMeter.next!(
                p;
                showvalues = [
                    (:Madcow_itr, madcow_idx),
                    (:Epoch, epoch),
                    (:Itreration, itr),
                    (:NLL, fval[(epoch-1)*num_batches+itr]),
                    (:NLL_eval, fval_eval[epoch]),
                ],
            )

            # Update params
            for p in get_params(G)
                Flux.update!(opt, p.data, p.grad)
            end
            clear_grad!(G)
        end

        if epoch % 10 == 0 || epoch == args["max_epoch"]

            save_dict = Dict{String,Any}()
            for (key, val) in args
                save_dict[key] = val
            end

            save_dict = merge(
                save_dict,
                Dict(
                    "epoch" => epoch,
                    "madcow_idx" => madcow_idx,
                    "fval" => fval,
                    "fval_eval" => fval_eval,
                    "opt" => opt,
                    "G" => G,
                ),
            )
            @tagsave(
                datadir(args["sim_name"], savename(save_dict, "jld2"; digits = 6)),
                save_dict;
                safe = true
            )
        end

    end

end
