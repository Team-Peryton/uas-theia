from picamera2.picamera2 import *
import time
from dronekit import connect
import logging



picam2 = Picamera2(verbose_console=0)

logger = logging.getLogger('picamera2')
logging.basicConfig(level=logging.CRITICAL)

vehicle = connect('/dev/ttyACM0', baud=56700, wait_ready=True)
print("connecte to autopilot")

print(vehicle)
print(picam2)

config = picam2.still_configuration(main={"size": (1920,1080)}) #these 3 lines should turn off preview
picam2.configure(config)
print(picam2.sensor_resolution)
picam2.start_preview(Preview.NULL)

picam2.start()
time.sleep(2)

try:
    while(vehicle.armed):
        location = vehicle.location.global_frame
        location = f"{location.alt},{location.lon},{location.lat}"
        
        heading = vehicle.heading
        
        roll = vehicle._roll
        rollspeed = vehicle._rollspeed

        pitch = vehicle._pitch
        pitchspeed = vehicle._pitchspeed
        
        yaw = vehicle._yaw
        yawspeed = vehicle._yawspeed

        attitude = f"{roll},{rollspeed}, {pitch}, {pitchspeed}, {yaw}, {yawspeed}"
        
        t = time.time()
        metadata = picam2.capture_file(f"/media/pi/USB DISK/{t},{location},{heading},{attitude}.jpg")
        
except KeyboardInterrupt:
    print("keyboard interupt")
except Exception:
    print("exception")

picam2.close()p