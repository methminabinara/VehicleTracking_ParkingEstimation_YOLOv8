import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Ensure you have the correct model file path

# Video input
video_path = r'E:\Intern_FOE\video.mp4'
cap = cv2.VideoCapture(video_path)

# Line coordinates for gate
start_point = (50, 300)  # Start point of the line (x1, y1)
end_point = (300, 800)   # End point of the line (x2, y2)
offset = 10  # Tolerance for counting vehicles
parked_vehicles = 0
enter_count = 0
exit_count = 0
parking_zones = []  # Define the coordinates for parking zones (if needed)

# Define the font for text on video
font = cv2.FONT_HERSHEY_SIMPLEX

# Store vehicle centroids
trackers = []

def get_center(x, y, w, h):
    return (x + w // 2, y + h // 2)

def calculate_slope_intercept(x1, y1, x2, y2):
    # Calculate the slope and intercept of the line
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept

def is_crossing_line(center, slope, intercept):
    # Check if a vehicle's center point crosses the indicator line
    x, y = center
    y_on_line = slope * x + intercept
    return abs(y - y_on_line) < offset  # Tolerance for proximity to the line

# Calculate slope and intercept for the indicator line
slope, intercept = calculate_slope_intercept(*start_point, *end_point)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLOv8 detection
    results = model(frame, conf=0.5)
    
    # Filter detected objects for vehicles (based on YOLOv8 class IDs)
    for result in results:
        for detection in result.boxes:
            x, y, w, h = map(int, detection.xywh[0].tolist())
            class_id = int(detection.cls)
            
            # Check if detected object is a vehicle (based on common vehicle classes in COCO dataset)
            if class_id in [2, 3, 5, 7]:  # Cars, motorcycles, buses, trucks
                center = get_center(x, y, w, h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, center, 4, (0, 0, 255), -1)
                
                # Check if the vehicle crosses the line (enter or exit)
                if is_crossing_line(center, slope, intercept):
                    if center[0] > (start_point[0] + end_point[0]) // 2:  # Going right (enter)
                        enter_count += 1
                    else:  # Going left (exit)
                        exit_count += 1
                
                # Parked vehicle detection (simple example, could be improved by tracking movement over time)
                for zone in parking_zones:
                    if cv2.pointPolygonTest(zone, center, False) >= 0:
                        parked_vehicles += 1
                        break
    
    # Draw the counting line
    cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
    
    # Display counts
    cv2.putText(frame, f'Entered: {enter_count}', (10, 50), font, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Exited: {exit_count}', (10, 100), font, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Parked: {parked_vehicles}', (10, 150), font, 1, (255, 255, 0), 2)

    # Show video
    cv2.imshow('Vehicle Detection', frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
