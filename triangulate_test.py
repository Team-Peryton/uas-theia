from theia import position_estimation, spec

l = spec.LocationInfo(lat=51.292598, lon=-0.4849133, alt=73.77,heading=342.0, pitch=-0.19271160662174225, roll=-0.9371986985206604)
print(position_estimation.triangulate((1314, 1002), l))