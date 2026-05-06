from ultralytics import YOLO
import cv2

# 1. Load your trained model
model_path = "best.pt"
print(f"📦 Loading model: {model_path}")
model = YOLO(model_path)

# 2. Define the path to your test image
# Make sure this matches the name of the image you put in the folder!
image_path = "image03.png" 

print(f"🔍 Running inference on: {image_path}")

# 3. Run YOLO inference
# conf=0.4 sets the confidence threshold
# save=True automatically saves the output image with the bounding box drawn on it
results = model.predict(source=image_path, conf=0.4, save=True, show=False)

# 4. Optional: Display the image right on your screen using OpenCV
# (results[0].plot() generates the image array with boxes and labels applied)
annotated_frame = results[0].plot()
cv2.imshow("Gantry Vision - Static Test", annotated_frame)

print("✅ Inference complete! Press any key on the image window to close it.")

# Wait for you to press a key, then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()



















# import cv2
# from ultralytics import YOLO

# # Load your downloaded custom model
# model_path = "best.pt"
# print(f"📦 Loading model: {model_path}")
# model = YOLO(model_path)

# # Initialize the camera stream
# # '0' is usually the built-in webcam. Change to 1 or 2 if you plug in your AR0234 via USB.
# cap = cv2.VideoCapture(0)

# # Force the camera resolution to match your training size
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# if not cap.isOpened():
#     print("❌ Error: Could not open camera.")
#     exit()

# print("✅ Camera active. Point it at a microwave! Press 'q' to quit.")

# # Live Inference Loop
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("❌ Error: Failed to grab frame.")
#         break

#     # Run YOLO inference. 
#     # conf=0.4 gives us that slight boost to recall we talked about earlier.
#     results = model.predict(source=frame, conf=0.4, show=False, verbose=False)

#     # Draw the bounding boxes onto the frame
#     annotated_frame = results[0].plot()

#     # Display the live feed
#     cv2.imshow("Gantry Vision - Live Test", annotated_frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Clean up
# cap.release()
# cv2.destroyAllWindows()