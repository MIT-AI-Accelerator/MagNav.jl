using MagNav, Test, MAT

xyz_file = joinpath(@__DIR__,"test_data/Flt1003_sample.xyz")
xyz_h5   = joinpath(@__DIR__,"Flt1003_sample.h5")

data = xyz2h5(xyz_file,xyz_h5,:Flt1003;return_data=true)

@testset "xyz2h5 tests" begin
    @test xyz2h5(xyz_file,xyz_h5,:Flt1003) isa Nothing
    rm(xyz_h5)
    @test xyz2h5(xyz_file,xyz_h5,:Flt1003;
                 lines=[(1003.02,50713.0,50713.2)],lines_type=:include) isa Nothing
    rm(xyz_h5)
    @test xyz2h5(xyz_file,xyz_h5,:Flt1003;
                 lines=[(1003.02,50713.0,50713.2)],lines_type=:exclude) isa Nothing
    rm(xyz_h5)
    @test_throws ErrorException xyz2h5(xyz_file,xyz_h5,:Flt1003;
                 lines=[(1003.02,50713.0,50713.2)],lines_type=:test)
    @test xyz2h5(xyz_file,xyz_h5,:Flt1003;return_data=true) isa Matrix
    @test xyz2h5(xyz_file,xyz_h5,:Flt1001_160Hz;return_data=true) isa Matrix
    @test xyz2h5(data,xyz_h5,:Flt1003) isa Nothing
end

xyz = get_XYZ20(xyz_h5)

@testset "h5 field tests" begin
    @test MagNav.delete_field(xyz_h5,:lat) isa Nothing
    @test MagNav.write_field(xyz_h5,:lat,xyz.traj.lat) isa Nothing
    @test MagNav.overwrite_field(xyz_h5,:lat,xyz.traj.lat) isa Nothing
    @test MagNav.read_field(xyz_h5,:lat) isa Vector
    @test MagNav.rename_field(xyz_h5,:lat,:lat) isa Nothing
    @test MagNav.clear_fields(xyz_h5) isa Nothing
end

rm(xyz_h5)

comp_params_0 = NNCompParams()
comp_params_1 = NNCompParams(comp_params_0,terms=[:p],reorient_vec=true)
comp_params_2 = NNCompParams(comp_params_1,model=MagNav.get_nn_m(1))
comp_params_3 = NNCompParams(comp_params_2,model=MagNav.get_nn_m(2))
comp_params_4 = NNCompParams(comp_params_3,TL_coef=zeros(18))

@testset "xyz field tests" begin
    @test MagNav.print_fields(xyz) isa Nothing
    @test MagNav.compare_fields(xyz,xyz;silent=true ) isa Int
    @test MagNav.compare_fields(xyz,xyz;silent=false) isa Nothing
    @test MagNav.compare_fields(comp_params_0,comp_params_1;silent=true) == 2
    @test MagNav.compare_fields(comp_params_1,comp_params_2;silent=true) == 1
    @test MagNav.compare_fields(comp_params_2,comp_params_3;silent=true) == 2
    @test MagNav.compare_fields(comp_params_3,comp_params_4;silent=true) == 1
    @test MagNav.field_check(xyz,MagNav.Traj) == [:traj]
    @test MagNav.field_check(xyz,:traj) isa Nothing
    @test MagNav.field_check(xyz,:traj,MagNav.Traj) isa Nothing
end

@testset "field_extrema tests" begin
    @test MagNav.field_extrema(xyz,:line,1003.01) == (49820.0,49820.2)
    @test_throws ErrorException MagNav.field_extrema(xyz,:flight,-1)
end

flights = [:fields20,:fields21,:fields160,
           :Flt1001,:Flt1002,:Flt1003,:Flt1004_1005,:Flt1004,:Flt1005,
           :Flt1006,:Flt1007,:Flt1008,:Flt1009,
           :Flt1001_160Hz,:Flt1002_160Hz,:Flt2001_2017,
           :Flt2001,:Flt2002,:Flt2004,:Flt2005,:Flt2006,
           :Flt2007,:Flt2008,:Flt2015,:Flt2016,:Flt2017]

@testset "xyz_fields tests" begin
    for flight in flights
        @test MagNav.xyz_fields(flight) isa Vector{Symbol}
    end
    @test_throws ErrorException MagNav.xyz_fields(:test)
end
