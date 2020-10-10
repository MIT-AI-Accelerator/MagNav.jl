"""
    gen_interp_map(map_map::Array{Float64}, 
                   map_xx::Array{Float64}, 
                   map_yy::Array{Float64})

Generate map grid interpolation, equivalent of griddedInterpolant in MATLAB.

**Arguments:**
- `map_map`: 2D gridded magnetic anomaly map
- `map_xx`: x direction map indices
- `map_yy`: y direction map indices

**Returns:**
- `interp_map`: map grid interpolation
"""
function gen_interp_map(map_map::Array{Float64}, 
                        map_xx::Array{Float64}, 
                        map_yy::Array{Float64})
#   BSpline(Linear())
#   BSpline(Quadratic(Line(OnCell())))
#   BSpline(Cubic(Line(OnCell())))

    Interpolations.scale(
    interpolate(map_map', BSpline(Cubic(Line(OnCell()))) ),
    LinRange(minimum(map_xx),maximum(map_xx),length(map_xx)),
    LinRange(minimum(map_yy),maximum(map_yy),length(map_yy)))

end # function gen_interp_map
