from picamera2.picamera2 import *
import time
from dronekit import connect
import logging



picam2 = Picamera2(verbose_console=0)

logger = logging.getLogger('picamera2')
logging.basicConfig(level=logging.CRITICAL)

vehicle = connect('/dev/ttyACM0', baud=56700, wait_ready=True) #(not sure how we're connecting to the pi)
print("connecte to autopilot")

print(vehicle)
print(picam2)

config = picam2.still_configuration(main={"size": (1920,1080)}) #these 3 lines should turn off preview
picam2.configure(config)
picam2.start_preview(Preview.NULL)

fps = 5 # how many fps to record

picam2.start()
time.sleep(2)

try:
    while(True): #change condition to something like "while armed" using dronekit. or use vehicle mission attribute for working with waypoints? 
        #time.sleep(1/fps) #check units
        location = vehicle.location.global_frame #check if this actually yields coorect gps coordinates. cant really test without pixhawk
        metadata = picam2.capture_file(f"/media/pi/USB DISK/{location.alt},{location.lon},{location.lat},{time.time()}.jpg")
except KeyboardInterrupt:
    print("keyboard interupt")
except Exception:
    print("exception")

picam2.close()