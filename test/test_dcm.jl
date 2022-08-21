using MagNav, Test, MAT

test_file = "test_data/test_data_dcm.mat"
dcm_data  = matopen(test_file,"r") do file
    read(file,"dcm_data")
end

roll  = vec(dcm_data["roll"])
pitch = vec(dcm_data["pitch"])
yaw   = vec(dcm_data["yaw"])
Cnb_1 = euler2dcm(roll[1],pitch[1],yaw[1],:body2nav)
Cnb   = euler2dcm(roll,pitch,yaw,:body2nav)
tilt_err       = dcm_data["tilt_err"]
Cnb_estimate_1 = correct_Cnb(Cnb_1,tilt_err[:,1])
Cnb_estimate   = correct_Cnb(Cnb  ,tilt_err)

rpy      = deg2rad.((45,60,15))
Cnb_temp = euler2dcm(rpy[1],rpy[2],rpy[3],:body2nav)
Cbn_temp = euler2dcm(rpy[1],rpy[2],rpy[3],:nav2body)
(roll_1,pitch_1,yaw_1) = dcm2euler(Cnb_temp,:body2nav)
(roll_2,pitch_2,yaw_2) = dcm2euler(Cbn_temp,:nav2body)

@testset "correct_Cnb tests" begin
    @test Cnb_1          ≈ dcm_data["Cnb"][:,:,1]
    @test Cnb            ≈ dcm_data["Cnb"]
    @test Cnb_estimate_1 ≈ dcm_data["Cnb_estimate"][:,:,1]
    @test Cnb_estimate   ≈ dcm_data["Cnb_estimate"]
    @test_nowarn correct_Cnb(Cnb_1,zeros(3,1))
end

@testset "euler2dcm & dcm2euler tests" begin
    @test roll_1  ≈ rpy[1]
    @test pitch_1 ≈ rpy[2]
    @test yaw_1   ≈ rpy[3]
    @test roll_2  ≈ rpy[1]
    @test pitch_2 ≈ rpy[2]
    @test yaw_2   ≈ rpy[3]
    @test_throws ErrorException euler2dcm(roll[1],pitch[1],yaw[1],:test)
    @test_throws ErrorException dcm2euler(Cnb_1,:test)
end
