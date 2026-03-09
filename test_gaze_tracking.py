import cv2
import numpy as np
import time
import random
import gaze_tracking

# --- DIRECT IMPORTS FOR COMPUTER VISION ---
import mediapipe as mp
from mediapipe.python.solutions import face_mesh as eye_tracking

# Initialize FaceMesh for gaze tracking
eyes = eye_tracking.FaceMesh(max_num_faces=1,
                            refine_landmarks=True,
                            min_detection_confidence=0.8)
cap = cv2.VideoCapture(0)  # Connect to webcam
    
def gaze_calibration_setup(cap, frame_width, frame_height,
                           screen_width, screen_height, duration_per_calibration_point=3.0):
    """
    Collect gaze samples to calibrate the gaze tracking functionality
    returns:
        (average_x, average_y) or (None, None) if no samples were collected
    """
    gaze_collected_points = []
    calibration_start_time = time.time()
    while time.time() - calibration_start_time < duration_per_calibration_point:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)  # Mirror the camera for intuitive control
        
        # Process image with MediaPipe
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        eye_landmarks = eyes.process(rgb_image)
        if eye_landmarks.multi_face_landmarks:
            for landmarks in eye_landmarks.multi_face_landmarks:
                gaze_estimate_x, gaze_estimate_y = gaze_tracking.gaze_to_screen_point(landmarks,
                                                                                      frame_width,
                                                                                      frame_height,
                                                                                      screen_width,
                                                                                      screen_height)
                if not (gaze_estimate_x == 0.0 or gaze_estimate_y == 0.0):
                    # only add points that are not clipped by gaze_to_screen_points
                    gaze_collected_points.append((gaze_estimate_x, gaze_estimate_y))                    

        # pause 1 millisecond between each point estimation for computation
        cv2.waitKey(1)
    if gaze_collected_points:
        average_x = np.mean([point[0] for point in gaze_collected_points])
        average_y = np.mean([point[1] for point in gaze_collected_points])
        return average_x, average_y
    else:
        return None, None

# calibrate and smooth gaze vector
smooth_gaze_vector = gaze_tracking.GazeSmoothTracking(alpha=0.1)
calibrate_gaze = gaze_tracking.GazeCalibration()

# get screen dimensions of main monitor
screen_width = 3840
screen_height = 2400

# first calibrate the gaze tracking
is_calibrated = False

# --- MAIN EXECUTION LOOP ---
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)  # Mirror the camera for intuitive control
        
        # Process image with MediaPipe
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        eye_landmarks = eyes.process(rgb_image)

        height, width = frame.shape[:2]

        gaze_status = ""
        if eye_landmarks.multi_face_landmarks:
            for landmarks in eye_landmarks.multi_face_landmarks:
            
                if is_calibrated:
                    gaze_estimate_x, gaze_estimate_y = gaze_tracking.gaze_to_screen_point(landmarks, width, height, screen_width, screen_height)
                    
                    # compute calibrated position
                    calibrate_x, calibrated_y = calibrate_gaze.transform(gaze_estimate_x, gaze_estimate_y)

                    # Smooth using a moving average
                    screen_x, screen_y = smooth_gaze_vector.update(calibrate_x, calibrated_y)

                    # visualize on the screen
                    visualize_x = int(screen_x * width / screen_width)
                    visualize_y = int(screen_y * height / screen_height)
                    cv2.circle(frame, (visualize_x, visualize_y), 5, (0, 255, 0), -1)
                else:
                    # upper right, upper left, center, bottom left, and bottom right of the screen
                    calibration_points = [(0.1, 0.1), (0.9, 0.1), (0.5, 0.5), (0.1, 0.9), (0.9, 0.9)]
                    # create blank screen
                    display = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
                    cv2.namedWindow('Gaze Calibration', cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty('Gaze Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.imshow('Gaze Calibration', display)
                    
                    # show user the calibration instructions
                    cv2.putText(display, f"Please look at the designated points, there will be {len(calibration_points)} calibration positions.",
                                        (screen_width // 2 - 1000, screen_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 2)
                    cv2.imshow('Gaze Calibration', display)
                    
                    # Display instructions for 8 seconds and then clear the screen
                    cv2.waitKey(6000)
                    display = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
                    cv2.imshow('Gaze Calibration', display)

                    # define time of displaying each calibration dot
                    duration_per_calibration_point = 5.0

                    for i, (point_x, point_y) in enumerate(calibration_points):
                            # convert to pixel coordinates
                            true_x = int(point_x * screen_width)
                            true_y = int(point_y * screen_height)

                            # draw screen to overlay calibration points
                            display = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
                            cv2.circle(display, (true_x, true_y), 25, (0, 255, 0), -1)
                            cv2.putText(display, f"Calibrating Gaze: look at the dot ({i+1}/{len(calibration_points)})",
                                        (screen_width // 2 - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                            cv2.imshow('Gaze Calibration', display)
                            # cv2.imshow("WPI RBE 526: Precise Sorter", frame)

                            # wait 1 second between calibration points to allow
                            # the user's eye to track to the new point
                            cv2.waitKey(1000)

                            # collect gaze coordinates for the shown calibration point on the screen
                            average_x, average_y = gaze_calibration_setup(cap, width, height,
                                                    screen_width, screen_height, duration_per_calibration_point)
                            if average_x:
                                calibrate_gaze.add_sample(average_x, average_y, true_x, true_y)
                   
                    # fit the linear regression model for calibrating on the sampled points
                    calibrate_gaze.fit()
                    
                    cv2.destroyWindow('Gaze Calibration')
                    is_calibrated = True

        cv2.namedWindow("Test Gaze Tracking", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Test Gaze Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Test Gaze Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    # Safe shutdown of camera
    cap.release()
    cv2.destroyAllWindows()