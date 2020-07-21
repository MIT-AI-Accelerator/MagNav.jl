function delta_lat(delta_pos,lat)
#   convert north-south position error to latitude error
#   delta_pos   [m]
#   lat         [rad]
#   delta_lat   [rad]

    r_earth     = 6378137           # WGS 84 radius of earth [m]
    e_earth     = 0.0818191908426   # first eccentricity of earth [-]

    delta_lat = delta_pos * sqrt(1-(e_earth*sin(lat))^2) / r_earth

    return (delta_lat)
end # function delta_lat

function delta_lon(delta_pos,lat)
#   convert east-west position error to longitude error
#   delta_pos   [m]
#   lat         [rad]
#   delta_lon   [rad]

    r_earth     = 6378137           # WGS 84 radius of earth [m]
    e_earth     = 0.0818191908426   # first eccentricity of earth [-]

    delta_lon = delta_pos * sqrt(1-(e_earth*sin(lat))^2) / r_earth / cos(lat)

    return (delta_lon)
end # function delta_lon

function delta_north(delta_lat,lat)
#   convert latitude error to north-south position error
#   delta_lat   [rad]
#   lat         [rad]
#   delta_pos   [m]

    r_earth     = 6378137           # WGS 84 radius of earth [m]
    e_earth     = 0.0818191908426   # first eccentricity of earth [-]

    delta_pos = delta_lat / sqrt(1-(e_earth*sin(lat))^2) * r_earth

    return (delta_pos)
end # function delta_north

function delta_east(delta_lon,lat)
#   convert longitude error to east-west position error
#   delta_lon   [rad]
#   lat         [rad]
#   delta_pos   [m]

    r_earth     = 6378137           # WGS 84 radius of earth [m]
    e_earth     = 0.0818191908426   # first eccentricity of earth [-]

    delta_pos = delta_lon / sqrt(1-(e_earth*sin(lat))^2) * r_earth * cos(lat)

    return (delta_pos)
end # function delta_east
