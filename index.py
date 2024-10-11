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

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLOv8 inference on the frame
    results = model(frame)

    # Extract boxes and class names from the results
    annotated_frame = results[0].plot()  # Plot results on the frame

    # Display the annotated frame
    cv2.imshow('YOLOv8 Vehicle Detection', annotated_frame)

    # Press 'q' to stop the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture when done
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print("Detection complete.")
