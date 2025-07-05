from ultralytics import YOLO
import cv2

# Test YOLO model loading
print("Loading YOLO model...")
model = YOLO("yolov8m.pt")
print("YOLO model loaded successfully!")

# Test OpenCV
print("\nTesting OpenCV...")
print(f"OpenCV version: {cv2.__version__}")
print("OpenCV is working!")

# Test video capture
print("\nTesting video capture...")
cap = cv2.VideoCapture(0)  # Try to open default camera
if cap.isOpened():
    print("Camera access successful!")
    cap.release()
else:
    print("Could not access camera") 