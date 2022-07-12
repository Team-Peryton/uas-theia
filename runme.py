from theia.main import ImageRecognition

ir = ImageRecognition("/media/pi/USB DISK")


while True:
    try:
        ir.image_recognition(ir.get_image(), ir.get_location_info())

    except KeyboardInterrupt:
        print("keyboard interupt")
        ir.calaculate_target_position()
        ir.get_found_targets()

    except Exception as e:
        print(e)
        ir.calaculate_target_position()
        ir.get_found_targets()

ir.calculate_target_position()
ir.get_found_targets()
ir.picam.close()
