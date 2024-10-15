import cv2
import numpy as np
from ultralytics import YOLO

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

# Vehicle class IDs in the COCO dataset (ignore humans and other objects)
vehicle_classes = [2, 3, 5, 7, 9]  # Car, Motorcycle, Bus, Truck, Bicycle

# Define the angled line for counting vehicles (entrance/exit line)
line_start = (50, 300)  # Start point of the line (x1, y1)
line_end = (300, 800)   # End point of the line (x2, y2)
enter_count = 0
exit_count = 0
parked_count = 0

# Dictionary to track vehicle positions to prevent double-counting and to track movement
vehicle_tracker = {}
stationary_threshold = 30  # Number of frames for a vehicle to be considered "parked"
movement_tolerance = 5  # Tolerance for detecting movement (in pixels)

# Function to check if a vehicle crosses the angled line from left to right (enter) or right to left (exit)
def crossed_the_line(prev_pos, curr_pos, line_start, line_end):
    def is_on_opposite_sides(p1, p2):
        cross_product1 = np.cross(np.subtract(line_end, line_start), np.subtract(p1, line_start))
        cross_product2 = np.cross(np.subtract(line_end, line_start), np.subtract(p2, line_start))
        return np.sign(cross_product1) != np.sign(cross_product2)

    return is_on_opposite_sides(prev_pos, curr_pos)

# Function to detect if the vehicle is stationary (parked)
def is_parked(vehicle_history, threshold, tolerance):
    if len(vehicle_history) < threshold:
        return False
    first_pos = vehicle_history[-threshold]
    return all(abs(p[0] - first_pos[0]) <= tolerance and abs(p[1] - first_pos[1]) <= tolerance for p in vehicle_history[-threshold:])

# Function to smooth out box positions to reduce shaking
def smooth_position(positions, alpha=0.2):
    if len(positions) < 2:
        return positions[-1]
    prev_pos = positions[-2]
    curr_pos = positions[-1]
    smoothed_pos = (int(alpha * curr_pos[0] + (1 - alpha) * prev_pos[0]),
                    int(alpha * curr_pos[1] + (1 - alpha) * prev_pos[1]))
    return smoothed_pos

# Process video frame by frame
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLOv8 inference on the frame
    results = model(frame)

    # Create a blank frame for annotations
    annotated_frame = frame.copy()

    # Draw the counting angled line on the frame
    cv2.line(annotated_frame, line_start, line_end, (0, 255, 255), 2)

    # Loop through the detected objects in the results
    for result in results[0].boxes:
        class_id = int(result.cls)  # Get the class ID

        # Only keep vehicle detections
        if class_id in vehicle_classes:
            # Get vehicle box coordinates
            box = result.xyxy[0].cpu().numpy()  # Coordinates of the box (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box)  # Convert to int for plotting
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2  # Get the center point of the vehicle

            # Vehicle ID for tracking (simple ID based on center coordinates and frame count)
            vehicle_id = f"{center_x}-{center_y}-{frame_count % 5}"

            # Draw the bounding box and center point
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(annotated_frame, (center_x, center_y), 4, (0, 0, 255), -1)  # Center point

            # Put the label (class name) with increased font size
            label = model.names[class_id]
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Track the vehicle's position and detect crossing the line or parking
            if vehicle_id in vehicle_tracker:
                vehicle_history = vehicle_tracker[vehicle_id]['history']
                prev_pos = vehicle_history[-1]
                curr_pos = smooth_position([(center_x, center_y)], alpha=0.5)
                vehicle_history.append(curr_pos)

                # Check if the vehicle crosses the line (left-to-right is entering, right-to-left is exiting)
                if crossed_the_line(prev_pos, curr_pos, line_start, line_end):
                    if curr_pos[0] > prev_pos[0]:  # Moving left to right (entering)
                        enter_count += 1
                    elif curr_pos[0] < prev_pos[0]:  # Moving right to left (exiting)
                        exit_count += 1

                # Check if the vehicle is parked (not moving for a set number of frames)
                if not vehicle_tracker[vehicle_id]['parked'] and is_parked(vehicle_history, stationary_threshold, movement_tolerance):
                    parked_count += 1
                    vehicle_tracker[vehicle_id]['parked'] = True

            else:
                # Initialize tracking if vehicle is new
                vehicle_tracker[vehicle_id] = {'history': [(center_x, center_y)], 'parked': False}

    # Display entry, exit, and parked counts on the frame
    cv2.putText(annotated_frame, f'Entries: {enter_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f'Exits: {exit_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(annotated_frame, f'Parked: {parked_count}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the annotated frame with vehicles
    cv2.imshow('YOLOv8 Vehicle Detection with Entry/Exit Counting and Parking', annotated_frame)

    # Press 'q' to stop the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Release video capture when done
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print(f"Total vehicles entered: {enter_count}")
print(f"Total vehicles exited: {exit_count}")
print(f"Total vehicles parked: {parked_count}")
