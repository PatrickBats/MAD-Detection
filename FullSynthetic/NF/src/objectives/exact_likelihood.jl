export exact_likelihood


"""
Compute p(X) given X and Net
"""
function exact_likelihood(Net::NetworkGlow, X::AbstractArray{Float32,4})

    Zx, logdet = Net.forward(X)

    loglike = sum(logpdf(0.0f0, 1.0f0, Zx), dims = [1, 2, 3])[1, 1, 1, :]
    loglike = loglike .+ logdet

    return loglike

end
