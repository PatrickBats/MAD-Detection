export maximum_likelihood


function maximum_likelihood(Net, X::AbstractArray{Float32,4}; grad::Bool = true)

    Zx, logdet = Net.forward(X)
    z_size = size(Zx)

    f = sum(logpdf(0.0f0, 1.0f0, Zx)) + logdet * z_size[4]

    if grad
        ΔZx = -gradlogpdf(0.0f0, 1.0f0, Zx) / z_size[4]
        ΔX = Net.backward(ΔZx, Zx)[1]

        return -f / z_size[4], ΔX
    else
        return -f / z_size[4]
    end
end
