import cv2
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

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLOv8 inference on the frame
    results = model(frame)

    # Create a blank frame for annotations
    annotated_frame = frame.copy()

    # Loop through the detected objects in the results
    for result in results[0].boxes:
        class_id = int(result.cls)  # Get the class ID

        # Only keep vehicle detections
        if class_id in vehicle_classes:
            # Plot only the vehicles onto the frame
            box = result.xyxy[0]  # Coordinates of the box (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box)  # Convert to int for plotting

            # Draw the bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put the label (class name)
            label = model.names[class_id]
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the annotated frame with only vehicles
    cv2.imshow('YOLOv8 Vehicle Detection (Vehicles Only)', annotated_frame)

    # Press 'q' to stop the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture when done
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print("Vehicle detection complete.")
