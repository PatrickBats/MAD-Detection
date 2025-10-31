# Madcow.jl


Here, we heavily rely on [InvertibleNetworks.jl](https://github.com/slimgroup/InvertibleNetworks.jl), a recently-developed, memory-efficient framework for training invertible networks in Julia.

## Installation

Before starting installing the required packages in Julia, make sure you have `matplotlib` and `seaborn` installed in your Python environment since we depend on `PyPlot.jl` and `Seaborn.jl` for creating figures.

```bash
pip install matplotlib seaborn
```

Next, run the following commands in the command line to install the necessary libraries and setup the Julia project:

```bash
export PYTHON=$(which python3)
julia -e 'using Pkg; Pkg.add("DrWatson"); Pkg.add("PyCall"); Pkg.build("PyCall")'
julia --project -e 'using Pkg; Pkg.instantiate()'
```

After the last line, the necessary dependencies will be installed.

## Example

Run normalizing flow training on a 2D toy dataset for 10 madcow iterations:

```bash
julia --project scripts/train_rosenbrock_madcow.jl --madcow 10
```

To plot the results, run:

```bash
julia --project scripts/test_rosenbrock_madcow.jl --madcow 10
```

