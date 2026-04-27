
from vision_drivers import run_vision
from vision_drivers import run_vision_with_sensor
from vision_drivers import run_recording
from vision_drivers import run_camera_test

def main():

    MODE = "vision"  # Change this to "vision", "sensor", "record", or "camera_test"

    if MODE == "vision":
        run_vision()

    elif MODE == "sensor":
        run_vision_with_sensor()

    elif MODE == "record":
        run_recording()

    elif MODE == "camera_test":
        run_camera_test()


if __name__ == "__main__":
    main()