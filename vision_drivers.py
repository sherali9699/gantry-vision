import cv2
import time
import numpy as np
import serial

from vision_pipeline import VisionPipeline
from vision_camera_config import VisionConfig



def run_vision():

    vision = VisionPipeline()

    #cap = cv2.VideoCapture(VisionConfig.CAMERA_ID)
    #cap = cv2.VideoCapture("./videos/vision-recording-homography.avi")
    cap = cv2.VideoCapture(2)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VisionConfig.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VisionConfig.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, VisionConfig.CAMERA_FPS)

    prev_time = 0
    count = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, top_down_view = vision.process_vision(frame)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(processed_frame,
                    f"Proc FPS: {fps:.1f}",
                    (20,700),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),2)

        cv2.imshow("Annotated",
                   cv2.resize(processed_frame,(0,0),fx=0.6,fy=0.6))

        cv2.imshow("TopDown",
                   cv2.resize(top_down_view,(0,0),fx=0.6,fy=0.6))

        # Capture key press ONCE
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'): 
            cv2.imwrite(f'./calib-images/calib_{count}.jpg', frame)
            print(f"Saved: calib_{count}.jpg") # Debug print to confirm
            count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_vision_with_sensor():

    try:
        ser = serial.Serial('/dev/ttyACM0',115200,timeout=0.01)
    except:
        ser = None

    vision = VisionPipeline()

    cap = cv2.VideoCapture(VisionConfig.CAMERA_ID)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VisionConfig.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VisionConfig.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, VisionConfig.CAMERA_FPS)

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, vis_angle = vision.process_vision(frame)

        if ser and ser.in_waiting > 0:
            line = ser.readline().decode(errors="ignore").rstrip()

        cv2.imshow("Telemetry", processed_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    if ser:
        ser.close()

    cap.release()
    cv2.destroyAllWindows()


def run_recording():

    vision = VisionPipeline()

    cap = cv2.VideoCapture(VisionConfig.CAMERA_ID)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VisionConfig.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VisionConfig.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, VisionConfig.CAMERA_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(
        "vision-recording-chessboard.avi",
        fourcc,
        VisionConfig.CAMERA_FPS,
        (VisionConfig.CAMERA_WIDTH, VisionConfig.CAMERA_HEIGHT)
    )

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, _ = vision.process_vision(frame)

        out.write(frame)

        cv2.imshow("Vision", processed_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def run_camera_test():

    cap = cv2.VideoCapture(VisionConfig.CAMERA_ID)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VisionConfig.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VisionConfig.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, VisionConfig.CAMERA_FPS)

    prev_time = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        curr_time = time.time()
        fps = 1/(curr_time-prev_time)
        prev_time = curr_time

        cv2.putText(frame,f"FPS: {fps:.1f}",
                    (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),2)

        cv2.imshow("Camera Test",frame)

        if cv2.waitKey(1)==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()