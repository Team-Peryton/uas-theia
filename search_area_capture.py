from theia.main import ImageRecognition
from theia.position_estimation import search_perimeter
import time

ir = ImageRecognition("/media/pi/USB DISK/")

try:
    while True:   
        location = ir.get_location_info() 
        if search_perimeter(location):
                location = ir.get_location_info()
                loc = f"{location.alt},{location.lon},{location.lat}"
                heading = location.heading
                attitude = f"{location.roll},{location.rollspeed}, {location.pitch}, {location.pitchspeed}, {location.yaw}, {location.yawspeed}"
                t = time.time()
                metadata = ir.picam.capture_file(f"/media/pi/USB DISK/{t},{loc},{heading},{attitude}.jpg")
        else:
            pass

        if not ir.vehicle.armed:
            ir.cala


except KeyboardInterrupt:
    print("keyboard interupt")
    ir.calaculate_target_position()
    ir.get_found_targets()

except Exception as e:
    print(e)
    ir.calaculate_target_position()
    ir.get_found_targets()

ir.picam.close()