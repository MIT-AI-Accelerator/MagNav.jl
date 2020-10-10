"""
    detrend(y)

Detrend (remove mean and slope from) evenly spaced signal.

**Arguments:**
- `y`: vector of observed values

**Returns:**
- `y_new`: detrended vector of observed values
"""
function detrend(y)
    N = length(y)
    X = [ones(N) 1:N]
    β_hat = (X'*X) \ (X'*y)
    y_new = y - X*β_hat
    return (y_new)
end # function detrend

"""
    map_grad(interp_mapS, x, y)

Get local map gradient.

**Arguments:**
- `interp_mapS`: scalar map grid interpolation
- `x`: x direction (longitude) map point [m]
- `y`: y direction (latitude)  map point [m]

**Returns:**
- `map_grad`: local map gradient: lon_deriv, lat_deriv [nT/m]
"""
map_grad(interp_mapS, x, y) = ForwardDiff.gradient(z -> interp_mapS(z[1],z[2]),
                                                   SVector(x,y))
