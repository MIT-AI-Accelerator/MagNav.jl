using MagNav, Test, LinearAlgebra, Statistics

# load map data HDF5 files & create map_cache
map_file  = joinpath(@__DIR__,"test_data/test_data_map.mat")
map_305   = get_map(map_file);
map_915   = upward_fft(map_305,3*map_305.alt);
xx_lim    = MagNav.get_lim(map_305.xx,0.2);
yy_lim    = MagNav.get_lim(map_305.yy,0.2);
namad     = map_fill(map_trim(get_map(MagNav.namad);
                              xx_lim=xx_lim,yy_lim=yy_lim));
map_cache = Map_Cache(maps=[map_305,map_915], # 2 maps, at 305 & 915 m
                      fallback=namad)            # trimmed NAMAD for speed

# test lat & lon point
lat = mean(map_305.yy)
lon = mean(map_305.xx)

# test alt points at 315, 610, & 965 m
alt_315 =   map_305.alt + 10
alt_610 = 2*map_305.alt
alt_965 =   map_915.alt + 50

# determine alt buckets for test points
alt_bucket_300_305 = max(floor(alt_315/100)*100, map_305.alt)
alt_bucket_600_305 = max(floor(alt_610/100)*100, map_305.alt)
alt_bucket_900_915 = max(floor(alt_965/100)*100, map_915.alt)

# upward continue to alt buckets & interpolate
mapS_300_305 = upward_fft(map_fill(map_trim(map_305)), alt_bucket_300_305)
mapS_600_305 = upward_fft(map_fill(map_trim(map_305)), alt_bucket_600_305)
mapS_900_915 = upward_fft(map_fill(map_trim(map_915)), alt_bucket_900_915)
itp_mapS_300_305 = map_interpolate(mapS_300_305)
itp_mapS_600_305 = map_interpolate(mapS_600_305)
itp_mapS_900_915 = map_interpolate(mapS_900_915)

# test alt, lat, & lon point outside of maps for NAMAD fallback testing
alt_out = map_305.alt - 5
lat_out = MagNav.get_lim(map_305.yy,0.1)[1]
lon_out = MagNav.get_lim(map_305.xx,0.1)[1]

# NAMAD maps for NAMAD fallback testing
alt_bucket_namad_1 = max(floor(alt_out/100)*100, namad.alt)
alt_bucket_namad_2 = max(floor(alt_315/100)*100, namad.alt)
mapS_namad_1       = upward_fft(map_fill(map_trim(namad)), alt_bucket_namad_1)
mapS_namad_2       = upward_fft(map_fill(map_trim(namad)), alt_bucket_namad_2)
itp_mapS_namad_1   = map_interpolate(mapS_namad_1)
itp_mapS_namad_2   = map_interpolate(mapS_namad_2)

# test that same alt bucket maps are used as within map_cache
@testset "Map_Cache tests" begin
    @test map_cache(lat,lon,alt_315) ≈ itp_mapS_300_305(lon,lat) # using map at 305 m (> 300 m)
    @test map_cache(lat,lon,alt_610) ≈ itp_mapS_600_305(lon,lat) # using map at 600 m (> 305 m)
    @test map_cache(lat,lon,alt_965) ≈ itp_mapS_900_915(lon,lat) # using map at 915 m (> 900 m)
    @test map_cache(lat,lon,alt_out) ≈ itp_mapS_namad_1(lon,lat) # NAMAD fallback, alt < all maps alts
    @test map_cache(lat_out,lon_out,alt_315) ≈ itp_mapS_namad_2(lon_out,lat_out) # NAMAD fallback, failed map_check()
end
