import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
from datetime import timedelta

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Define the video path
video_path = r'E:\Intern_FOE\video.mp4'

# Open video capture
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

# Get video properties for timestamp calculation
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize tracker and counts
tracker = Tracker()
entry_count, exit_count, parked_count = 0, 0, 0

# Class IDs and names for vehicles in COCO dataset
vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
    9: "Bicycle"
}

# Define the angled lines for counting vehicles (entry/exit lines)
line_start_red, line_end_red = (100, 300), (500, 300)  # Red line (Exit)
line_start_blue, line_end_blue = (1, 350), (450, 350)  # Blue line (Entry)

# Initialize tracking variables
exit_tracking, entry_tracking = {}, {}
exit_vehicles, entry_vehicles = [], []
frame_count = 0

# Lists to store entry and exit times with vehicle types
entry_data = []  # Will store (time, vehicle_type) tuples
exit_data = []   # Will store (time, vehicle_type) tuples

# Store vehicle types for each ID
vehicle_types = {}  # {vehicle_id: vehicle_type}

# Function to convert frame number to video time
def frame_to_time(frame_num):
    seconds = frame_num / fps
    return str(timedelta(seconds=seconds))

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = frame_to_time(frame_count)
    frame = cv2.resize(frame, (1020, 500))

    # Perform YOLOv8 inference on the frame
    results = model.predict(frame)
    detections = results[0].boxes.data.cpu().numpy()

    # Store detected vehicle boxes and their types
    vehicle_boxes = []
    current_detections = {}  # {bbox: class_id}

    # Filter results to keep only vehicle classes
    for det in detections:
        x1, y1, x2, y2, conf, class_id = det.tolist()
        class_id = int(class_id)
        if class_id in vehicle_classes:
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            vehicle_boxes.append(bbox)
            current_detections[tuple(bbox)] = class_id

    # Update tracker and get tracked boxes with IDs
    tracked_boxes = tracker.update(vehicle_boxes)

    # Draw entry/exit lines
    cv2.line(frame, line_start_red, line_end_red, (0, 0, 255), 3)  # Red for exit
    cv2.line(frame, line_start_blue, line_end_blue, (255, 0, 0), 3)  # Blue for entry

    # Track each vehicle and count based on movement direction
    for bbox in tracked_boxes:
        x3, y3, x4, y4, vehicle_id = bbox
        
        # Try to match the tracked box with one of our detections to get the class
        best_match = None
        best_iou = 0
        for det_bbox, class_id in current_detections.items():
            # Calculate IoU between tracked box and detected box
            x1, y1, x2, y2 = det_bbox
            # Calculate intersection
            inter_x1 = max(x1, x3)
            inter_y1 = max(y1, y3)
            inter_x2 = min(x2, x4)
            inter_y2 = min(y2, y4)
            if inter_x2 <inter_x1 or inter_y2 < inter_y1:
                continue
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            # Calculate union
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (x4 - x3) * (y4 - y3)
            union_area = area1 + area2 - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_match = class_id
        
        # Update vehicle type if we have a good match
        if best_match is not None and best_iou > 0.5:
            vehicle_types[vehicle_id] = vehicle_classes[best_match]
        
        # If we still don't have a type for this vehicle, use a default
        if vehicle_id not in vehicle_types:
            vehicle_types[vehicle_id] = "Unknown"
        
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2
        
        # Display vehicle type
        veh_type = vehicle_types[vehicle_id]
        cv2.putText(frame, f"{veh_type} #{vehicle_id}", (cx - 40, y3 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Define offset for movement tolerance
        offset = 7

        # Condition for vehicles moving upwards (entry)
        if line_start_blue[1] < (cy + offset) and line_start_blue[1] > (cy - offset):
            entry_tracking[vehicle_id] = cy
        if vehicle_id in entry_tracking:
            if line_start_red[1] < (cy + offset) and line_start_red[1] > (cy - offset):
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(vehicle_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                if vehicle_id not in entry_vehicles:
                    entry_vehicles.append(vehicle_id)
                    veh_type = vehicle_types[vehicle_id]
                    entry_data.append((current_time, veh_type))
                del entry_tracking[vehicle_id]

        # Condition for vehicles moving downwards (exit)
        if line_start_red[1] < (cy + offset) and line_start_red[1] > (cy - offset):
            exit_tracking[vehicle_id] = cy
        if vehicle_id in exit_tracking:
            if line_start_blue[1] < (cy + offset) and line_start_blue[1] > (cy - offset):
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(vehicle_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                if vehicle_id not in exit_vehicles:
                    exit_vehicles.append(vehicle_id)
                    veh_type = vehicle_types[vehicle_id]
                    exit_data.append((current_time, veh_type))
                del exit_tracking[vehicle_id]

    # Display entry, exit, and parked counts on the frame
    exits = len(set(exit_vehicles))
    entries = len(set(entry_vehicles))

    # Make text visible by adding a solid background behind text
    cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 0), -1)  # Black background for text
    cv2.putText(frame, f'Entries: {entries}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, f'Exits: {exits}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Display the frame with tracking annotations
    cv2.imshow("YOLOv8 Vehicle Detection with Tracking", frame)

    # Press 'q' to stop the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Save the counts, times, and vehicle types to a text file
with open('vehicle_types_and_times.txt', 'w') as f:
    f.write(f"Total vehicles entering: {entries}\n")
    f.write(f"Total vehicles exiting: {exits}\n\n")
    
    f.write("--- ENTRY DATA ---\n")
    for i, (time_stamp, veh_type) in enumerate(entry_data, 1):
        f.write(f"{time_stamp} - {veh_type}\n")
    
    f.write("\n--- EXIT DATA ---\n")
    for i, (time_stamp, veh_type) in enumerate(exit_data, 1):
        f.write(f"{time_stamp} - {veh_type}\n")
    
    # Add vehicle type statistics
    entry_type_counts = {}
    exit_type_counts = {}
    for _, veh_type in entry_data:
        entry_type_counts[veh_type] = entry_type_counts.get(veh_type, 0) + 1
    for _, veh_type in exit_data:
        exit_type_counts[veh_type] = exit_type_counts.get(veh_type, 0) + 1
    
    f.write("\n--- ENTRY STATISTICS BY VEHICLE TYPE ---\n")
    for veh_type, count in entry_type_counts.items():
        f.write(f"{veh_type}: {count}\n")
    
    f.write("\n--- EXIT STATISTICS BY VEHICLE TYPE ---\n")
    for veh_type, count in exit_type_counts.items():
        f.write(f"{veh_type}: {count}\n")

print(f"Total vehicles entering: {entries}")
print(f"Total vehicles exiting: {exits}")
print(f"Vehicle types, entry and exit times saved to 'vehicle_types_and_times.txt'")