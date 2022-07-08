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
            heading = 0
            attitude = f"{0},{location.rollspeed}, {0}, {location.pitchspeed}, {location.yaw}, {location.yawspeed}"
            t = time.time()
            metadata = ir.picam.capture_file(f"/media/pi/USB DISK/{t},{loc},{heading},{attitude}.jpg")
        else:
            pass

        if not ir.vehicle.armed: #clustering?
            ir.image_recognition_from_files()
            ir.calaculate_target_position()
            ir.get_found_targets()
            break

except KeyboardInterrupt:
    print("keyboard interupt")
    ir.calaculate_target_position()
    ir.get_found_targets()

except Exception as e:
    print(e)
    ir.calaculate_target_position()
    ir.get_found_targets()

ir.picam.close()