import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # You can use yolov8s.pt or yolov8m.pt for better accuracy

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
line_end = (300, 800)    # End point of the line (x2, y2)
enter_count = 0
exit_count = 0

# Dictionary to track vehicle positions to prevent double-counting
vehicle_tracker = {}

# Function to check if a vehicle crosses the angled line
def crossed_the_line(prev_pos, curr_pos, line_start, line_end):
    def is_on_opposite_sides(p1, p2):
        # Check if two points (previous and current) are on opposite sides of the line
        cross_product1 = np.cross(np.subtract(line_end, line_start), np.subtract(p1, line_start))
        cross_product2 = np.cross(np.subtract(line_end, line_start), np.subtract(p2, line_start))
        return np.sign(cross_product1) != np.sign(cross_product2)

    return is_on_opposite_sides(prev_pos, curr_pos)

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
            box = result.xyxy[0]  # Coordinates of the box (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box)  # Convert to int for plotting
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2  # Get the center point of the vehicle

            # Vehicle ID for tracking (simple ID based on center coordinates and frame count)
            vehicle_id = f"{center_x}-{center_y}-{frame_count % 5}"

            # Draw the bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put the label (class name) with increased font size
            label = model.names[class_id]
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Track the vehicle's position and detect crossing the line
            if vehicle_id in vehicle_tracker:
                prev_pos = vehicle_tracker[vehicle_id]
                curr_pos = (center_x, center_y)

                # Debug: print positions to ensure they are changing
                print(f"Prev pos: {prev_pos}, Curr pos: {curr_pos}")

                # Check if the vehicle crosses the line
                if crossed_the_line(prev_pos, curr_pos, line_start, line_end):
                    # Determine if entering or leaving
                    if curr_pos[1] > prev_pos[1]:  # Moving downward (entering)
                        enter_count += 1
                        print(f"Vehicle entered (Total enters: {enter_count})")
                    else:  # Moving upward (leaving)
                        exit_count += 1
                        print(f"Vehicle exited (Total exits: {exit_count})")

                # Update the tracked position
                vehicle_tracker[vehicle_id] = curr_pos
            else:
                # Initialize tracking if vehicle is new
                vehicle_tracker[vehicle_id] = (center_x, center_y)

    # Display the annotated frame with only vehicles
    cv2.imshow('YOLOv8 Vehicle Detection with Entry/Exit Counting', annotated_frame)

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
