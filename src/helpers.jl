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
