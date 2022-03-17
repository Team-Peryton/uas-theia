from picamera2.picamera2 import *
import time
from dronekit import connect

picam2 = Picamera2()
vehicle = connect('network address', wait_ready=True) #(not sure how we're connecting to the pi)

config = picam2.preview_configuration() #these 3 lines should turn off preview
picam2.configure(config)
picam2.start_preview(Preview.NULL)

fps = 5 # how many fps to record

picam2.start()
time.sleep(2)

while(True): #change condition to something like "while armed" using dronekit. or use vehicle mission attribute for working with waypoints? 
    time.sleep(1/fps) #check units
    location = str(vehicle.location.global_frame) #check if this actually yields coorect gps coordinates. cant really test without pixhawk
    metadata = picam2.capture_file(location+".jpg")

picam2.close()