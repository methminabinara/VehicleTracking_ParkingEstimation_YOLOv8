import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Ask user for threshold levels at runtime
print("***ENTER VEHICLE COUNTS FOR EACH TRAFFIC LEVELS***")
moderate_threshold = int(input("Enter the vehicle count for moderate traffic level (e.g., 10): "))
critical_threshold = int(input("Enter the vehicle count for critical traffic level (e.g., 20): "))

# Initialize video capture
cap = cv2.VideoCapture(r'E:\Intern_FOE\video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

# Vehicle detection settings
min_width_react = 80
min_hieght_react = 80
count_line_position = 570

# Background subtractor algorithm
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# Initialize variables for counting
detect = []
offset = 6  # Allowable error between pixels
counter = 0

# Variables for tracking traffic flow over time
time_intervals = 10  # Track traffic every 10 seconds
traffic_flow = [0]
time_stamps = [0]
start_time = time.time()

# Warning settings
warning_duration = 5  # Duration to display the warning (in seconds)
warning_displayed = False  # Flag to track if warning is active
warning_type = None  # Tracks whether the warning is moderate or critical
warning_start_time = 0  # Store when the warning was triggered

# Set up Matplotlib plot for traffic flow comparison
fig, ax = plt.subplots()
ax.set_title("Traffic Flow Over Time")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Number of Vehicles")

# Define the plot update function
def update_plot(i):
    ax.clear()
    ax.plot(time_stamps, traffic_flow, label="Traffic Flow")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Number of Vehicles")
    ax.legend()

ani = FuncAnimation(fig, update_plot, interval=1000)

# Main video processing loop
while True:
    ret, frame1 = cap.read()
    if not ret:
        break

    # Convert to grayscale and apply blur
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    # Background subtraction and morphology operations
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    counterSahpe, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the detection line
    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)

    for (i, c) in enumerate(counterSahpe):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_react) and (h >= min_hieght_react)
        if not validate_counter:
            continue

        # Draw bounding box around the vehicle
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get the center of the vehicle
        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        # Count vehicle when it crosses the line
        for (cx, cy) in detect:
            if count_line_position - offset < cy < count_line_position + offset:
                counter += 1
                detect.remove((cx, cy))

    # Update traffic flow at time intervals
    current_time = time.time() - start_time
    if int(current_time) % time_intervals == 0 and int(current_time) not in time_stamps:
        time_stamps.append(int(current_time))
        traffic_flow.append(counter)

        # Check if traffic exceeds threshold and display appropriate warning message
        if counter >= critical_threshold and not warning_displayed:
            warning_type = "critical"
            warning_displayed = True
            warning_start_time = time.time()  # Record the start time of the warning
            counter = 0  # Reset counter for the next interval
        elif counter >= moderate_threshold and not warning_displayed:
            warning_type = "moderate"
            warning_displayed = True
            warning_start_time = time.time()  # Record the start time of the warning
            counter = 0  # Reset counter for the next interval


    # Display the warning message based on traffic thresholds
    if warning_displayed:
        if warning_type == "critical":
            warning_msg = "Critical Warning: Severe traffic expected at the next junction!"
            color = (0, 0, 255)  # Red color for critical warning
        elif warning_type == "moderate":
            warning_msg = "Warning: Traffic may increase at the next junction!"
            color = (0, 255, 255)  # Yellow color for moderate warning

        cv2.putText(frame1, warning_msg, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Check if the warning has been displayed for more than the set duration
        if time.time() - warning_start_time > warning_duration:
            warning_displayed = False  # Reset warning flag after the duration
            warning_type = None  # Reset warning type

    # Display vehicle counts on the video frame
    cv2.putText(frame1, "VEHICLE COUNTER : " + str(counter), (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
 
    # Display the video frame with warning if triggered
    cv2.imshow("video original", frame1)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

# Show the traffic flow plot
plt.show()