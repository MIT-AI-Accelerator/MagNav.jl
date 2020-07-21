# remove mean, slope from vector
function detrend(y)
    N = length(y)
    X = [ones(N) 1:N]
    β_hat = (X'*X) \ (X'*y)
    y_new = y - X*β_hat
    return (y_new)
end # function detrend

# lon_deriv,lat_deriv [nT/rad]
map_grad(interp_mapS,x,y) = ForwardDiff.gradient(z -> rad2deg(interp_mapS(
                            z[1],z[2])),SVector(x,y))
