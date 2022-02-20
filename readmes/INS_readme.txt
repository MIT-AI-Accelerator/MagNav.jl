Notes regarding recorded data

The data are generated from measurements taken by the AIRGrav system alone. The main sensor unit is a three axis gimballed inertial navigation system, run in a local-level, wander-angle mechanization. As such, the horizontal accelerations in the files are in a platform coordinate frame, with the X-axis nominally horizontal and slewed counter-clockwise from North by the wander angle. The Y-axis is also nominally horizontal and 90 degrees counter-clockwise from the X-axis, and the Z-axis nominally vertical (up). The system has the normal perturbations of its nominal axes due to sensor drifts, gravity disturbances, etc, as with any IMU or INS. These can be estimated using a Kalman filter with GPS observations as measurements.

The location of the aircraft cabin antenna with respect to the gravimeter inertial centre, using a coordinate frame of forward-left-up, is as follows:
[1.2243  -0.1796  1.1133]  
ie antenna is forward 1.2243 m, right 17.96 cm, and up 1.133 m from the gravimeter. The vector for the aircraft tail antenna is as follows:
[-7.0824  -0.2914  3.2384]

A brief description of the recorded quantities is as follows:

week: GPS week number

seconds in week: GPS seconds into the week

x accel: x platform sensed acceleration corrected for temperature, misalignment and scale factor (m/sec^2); note that the digitizers used for the accelerometers are of the sigma-delta type and require low-pass filtering to increase the resolution.

y accel: y platform sensed acceleration corrected for temperature, misalignment and scale factor (m/sec^2)

z accel: z platform sensed acceleration corrected for temperature, misalignment and scale factor (m/sec^2)

north vel: system computed north velocity (m/sec)

west vel: system computed west velocity (m/sec)

vert vel: system computed vertical (up) velocity (m/sec)

latitude: system computed latitude (radians)

longitude: system computed longitude (radians)

height: system computed height (m); note that this is a combined GPS/inertial height computed in real time, it has a lag and should not be relied on as a source of true height

wander: system computed wander angle, x ccw from North (radians)

pitch: system computed aircraft pitch (radians)

roll: system computed aircraft roll (radians)

azimuth: system computed aircraft azimuth (radians); note that the three aircraft attitude angles follow the normal convention of azimuth clockwise from North, then pitch up, then roll to starboard.
