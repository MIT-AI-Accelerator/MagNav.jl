"""
    delta_lat(delta_pos, lat)

Convert north-south position error to latitude error.

**Arguments:**
- `delta_pos`: north-south position error [m]
- `lat`: nominal latitude [rad]

**Returns:**
- `delta_lat`: latitude error [rad]
"""
function delta_lat(delta_pos, lat)
    r_earth   = 6378137         # WGS 84 radius of earth [m]
    e_earth   = 0.0818191908426 # first eccentricity of earth [-]
    delta_lat = delta_pos * sqrt(1-(e_earth*sin(lat))^2) / r_earth
    return (delta_lat)
end # function delta_lat

"""
    delta_lon(delta_pos, lat)

Convert east-west position error to longitude error.

**Arguments:**
- `delta_pos`: east-west position error [m]
- `lat`: nominal latitude [rad]

**Returns:**
- `delta_lon`: longitude error [rad]
"""
function delta_lon(delta_pos, lat)
    r_earth   = 6378137         # WGS 84 radius of earth [m]
    e_earth   = 0.0818191908426 # first eccentricity of earth [-]
    delta_lon = delta_pos * sqrt(1-(e_earth*sin(lat))^2) / r_earth / cos(lat)
    return (delta_lon)
end # function delta_lon

"""
    delta_north(delta_lat, lat)

Convert latitude error to north-south position error.

**Arguments:**
- `delta_lat`: latitude error [rad]
- `lat`: nominal latitude [rad]

**Returns:**
- `delta_pos`: north-south position error [m]
"""
function delta_north(delta_lat, lat)
    r_earth   = 6378137         # WGS 84 radius of earth [m]
    e_earth   = 0.0818191908426 # first eccentricity of earth [-]
    delta_pos = delta_lat / sqrt(1-(e_earth*sin(lat))^2) * r_earth
    return (delta_pos)
end # function delta_north

"""
    delta_east(delta_lon, lat)

Convert longitude error to east-west position error.

**Arguments:**
- `delta_lon`: longitude error [rad]
- `lat`: nominal latitude [rad]

**Returns:**
- `delta_pos`: east-west position error [m]
"""
function delta_east(delta_lon, lat)
    r_earth   = 6378137         # WGS 84 radius of earth [m]
    e_earth   = 0.0818191908426 # first eccentricity of earth [-]
    delta_pos = delta_lon / sqrt(1-(e_earth*sin(lat))^2) * r_earth * cos(lat)
    return (delta_pos)
end # function delta_east
