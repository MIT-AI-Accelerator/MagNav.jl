"""
    euler2dcm(roll, pitch, yaw, order::Symbol=:body2nav)

Converts a (Euler) roll-pitch-yaw (`X`-`Y`-`Z`) right-handed body to navigation 
frame rotation (or the opposite rotation), to a DCM (direction cosine matrix). 
Yaw is synonymous with azimuth and heading here. 
If frame 1 is rotated to frame 2, then the returned DCM, when pre-multiplied, 
rotates a vector in frame 1 into frame 2. There are 2 use cases:

1) With `order = :body2nav`, the body frame is rotated in the standard 
-roll, -pitch, -yaw sequence to the navigation frame. For example, if v1 is 
a 3x1 vector in the body frame [nose, right wing, down], then that vector 
rotated into the navigation frame [north, east, down] would be v2 = dcm * v1.

2) With `order = :nav2body`, the navigation frame is rotated in the standard 
yaw, pitch, roll sequence to the body frame. For example, if v1 is a 3x1 
vector in the navigation frame [north, east, down], then that vector rotated 
into the body frame [nose, right wing, down] would be v2 = dcm * v1.

Reference: Titterton & Weston, Strapdown Inertial Navigation Technology, 2004, 
Section 3.6 (pg. 36-41 & 537).

**Arguments:**
- `roll`:  `N` roll  angles [rad], right-handed rotation about x-axis
- `pitch`: `N` pitch angles [rad], right-handed rotation about y-axis
- `yaw`:   `N` yaw   angles [rad], right-handed rotation about z-axis
- `order`: (optional) rotation order {`:body2nav`,`:nav2body`}

**Returns:**
- `dcm`: `3` x `3` x `N` direction cosine matrices [-]
"""
function euler2dcm(roll, pitch, yaw, order::Symbol=:body2nav)

    size(roll ,2) > 1 && (roll  = roll')
    size(pitch,2) > 1 && (pitch = pitch')
    size(yaw  ,2) > 1 && (yaw   = yaw')

    cr = cos.(roll)
    sr = sin.(roll)
    cp = cos.(pitch)
    sp = sin.(pitch)
    cy = cos.(yaw)
    sy = sin.(yaw)

    dcm = zeros(3,3,length(roll))

    if order == :body2nav # Cnb, shown in Titterton & Weston (pg. 41)
        dcm[1,1,:] .=  cp.*cy
        dcm[1,2,:] .= -cr.*sy + sr.*sp.*cy
        dcm[1,3,:] .=  sr.*sy + cr.*sp.*cy
        dcm[2,1,:] .=  cp.*sy
        dcm[2,2,:] .=  cr.*cy + sr.*sp.*sy
        dcm[2,3,:] .= -sr.*cy + cr.*sp.*sy
        dcm[3,1,:] .= -sp
        dcm[3,2,:] .=  sr.*cp
        dcm[3,3,:] .=  cr.*cp
    elseif order == :nav2body # Cbn, used by John Raquet in RpyToDcm()
        dcm[1,1,:] .=  cp.*cy
        dcm[1,2,:] .=  cp.*sy
        dcm[1,3,:] .= -sp
        dcm[2,1,:] .= -cr.*sy + sr.*sp.*cy
        dcm[2,2,:] .=  cr.*cy + sr.*sp.*sy
        dcm[2,3,:] .=  sr.*cp
        dcm[3,1,:] .=  sr.*sy + cr.*sp.*cy
        dcm[3,2,:] .= -sr.*cy + cr.*sp.*sy
        dcm[3,3,:] .=  cr.*cp
    else
        error("DCM rotation $order order not defined")
    end

    if length(roll) == 1
        return (dcm[:,:,1])
    else
        return (dcm)
    end

end # function euler2dcm

"""
    dcm2euler(dcm, order::Symbol=:body2nav)

Converts a DCM (direction cosine matrix) to yaw, pitch, and roll Euler angles. 
Yaw is synonymous with azimuth and heading here. There are 2 use cases:

1) With `order = :body2nav`, the provided DCM is assumed to rotate from the 
body frame in the standard -roll, -pitch, -yaw sequence to the navigation 
frame. For example, if v1 is a 3x1 vector in the body frame [nose, right wing, 
down], then that vector rotated into the navigation frame [north, east, down] 
would be v2 = dcm * v1.

2) With `order = :nav2body`, the provided DCM is assumed to rotate from the 
navigation frame in the standard yaw, pitch, roll sequence to the body frame. 
For example, if v1 is a 3x1 vector in the navigation frame [north, east, down], 
then that vector rotated into the body frame [nose, right wing, down] would be 
v2 = dcm * v1.

Reference: Titterton & Weston, Strapdown Inertial Navigation Technology, 2004, 
Section 3.6 (pg. 36-41 & 537).

**Arguments:**
- `dcm`:   `3` x `3` x `N` direction cosine matrices [-]
- `order`: (optional) rotation order {`:body2nav`,`:nav2body`}

**Returns:**
- `roll`:  `N` roll  angles [rad], right-handed rotation about x-axis
- `pitch`: `N` pitch angles [rad], right-handed rotation about y-axis
- `yaw`:   `N` yaw   angles [rad], right-handed rotation about z-axis
"""
function dcm2euler(dcm, order::Symbol=:body2nav)

    if order == :body2nav # Cnb, shown in Titterton & Weston (pg. 41)
        roll  =  atan.(dcm[3,2,:],dcm[3,3,:])
        pitch = -asin.(dcm[3,1,:])
        yaw   =  atan.(dcm[2,1,:],dcm[1,1,:])
    elseif order == :nav2body # Cbn, used by John Raquet in DcmToRpy()
        roll  =  atan.(dcm[2,3,:],dcm[3,3,:])
        pitch = -asin.(dcm[1,3,:])
        yaw   =  atan.(dcm[1,2,:],dcm[1,1,:])
    else
        error("DCM rotation $order order not defined")
    end

    # for debugging INS Cnb
    # println("dcm[3,1] ",round.(extrema(dcm[3,1,:]),digits=3))
    # println("dcm[1,3] ",round.(extrema(dcm[1,3,:]),digits=3))
    # println("pitch ",round.(rad2deg.(extrema(pitch)),digits=1)," deg")

    if length(roll) == 1
        return (roll[1], pitch[1], yaw[1])
    else
        return (roll, pitch, yaw)
    end

end # function dcm2euler

"""
    correct_Cnb(Cnb, tilt_err)

Internal helper function to correct a (Euler) roll-pitch-yaw (`X`-`Y`-`Z`)
right-handed body to navigation frame rotation DCM (direction cosine matrix)
with [`X`,`Y`,`Z`] tilt angle errors. The resulting DCM is `in error`, such
as INS data.

Reference: Titterton & Weston, Strapdown Inertial Navigation Technology, 2004,  
eq. 10.10 (pg. 284) and eq. 12.6 (pg. 342).

**Arguments:**
- `Cnb`:      `3` x `3` x `N` direction cosine matrices (BODY TO NAVIGATION) [-]
- `tilt_err`: `3` x `N` [`X`,`Y`,`Z`] tilt angle errors [rad]

**Returns:**
- `Cnb_estimate`: `3` x `3` x `N` "in error" direction cosine matrices [-]
"""
function correct_Cnb(Cnb, tilt_err)

    N = size(tilt_err,2)
    Cnb_estimate = zeros(3,3,N)
    for i = 1:N
        m = norm(tilt_err[:,i])
        if m != 0
            s = [             0 -tilt_err[3,i]  tilt_err[2,i]
                  tilt_err[3,i]              0 -tilt_err[1,i]
                 -tilt_err[2,i]  tilt_err[1,i]              0]
            B = I - sin(m)/m*s - (1-cos(m))/m^2*s^2 # ≈ I - s - 0.5*s^2
            Cnb_estimate[:,:,i] = B*Cnb[:,:,i]
        else
            Cnb_estimate[:,:,i] =   Cnb[:,:,i]
        end
    end

    return (Cnb_estimate)
end # function correct_Cnb
