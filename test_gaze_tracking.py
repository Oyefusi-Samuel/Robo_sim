import cv2
import time
import random
import gaze_tracking

# --- DIRECT IMPORTS FOR COMPUTER VISION ---
from mediapipe.python.solutions import face_mesh as eye_tracking

# Initialize FaceMesh for gaze tracking
eyes = eye_tracking.FaceMesh(max_num_faces=1,
                            refine_landmarks=True,
                            min_detection_confidence=0.8)
cap = cv2.VideoCapture(0)  # Connect to webcam

# smooth gaze vector with a moving average
smooth_gaze_vector = gaze_tracking.GazeSmoothTracking(alpha=0.15)

# get screen dimensions of main monitor
screen_width = 3840
screen_height = 2400

# --- MAIN EXECUTION LOOP ---
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Process image with MediaPipe
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        eye_landmarks = eyes.process(rgb_image)

        height, width = frame.shape[:2]

        gaze_status = ""
        if eye_landmarks.multi_face_landmarks:
            for landmarks in eye_landmarks.multi_face_landmarks:
        
                gaze_estimate_x, gaze_estimate_y, gaze_world = gaze_tracking.gaze_to_screen_point(landmarks.landmark,
                                                                                    width,
                                                                                    height,
                                                                                    screen_width,
                                                                                    screen_height)
                # Smooth using a moving average
                screen_x, screen_y = smooth_gaze_vector.update(gaze_estimate_x, gaze_estimate_y)

                # ensure the boundary of the screen is the limit of the x and y pixel coordinates
                if screen_x < 0:
                    screen_x = 0
                if screen_y < 0:
                    screen_y = 0
                if screen_x > width:
                    screen_x = width
                if screen_y > height:
                    screen_y = height
                # print screen coordinate of gaze
                cv2.circle(frame, (screen_x, screen_y), 5, (0, 255, 0), -1)
                
                # print gaze vector on the screen if available
                if gaze_world is not None:
                    average_gaze = f"Gaze: [{gaze_world[0]:+.3f}, {gaze_world[1]:+.3f}, {gaze_world[2]:+.3f}]"
                    right_eye = landmarks.landmark[468]
                    left_eye = landmarks.landmark[473]
                    eyes_center_x = int((right_eye.x+left_eye.x)/2*width)
                    eyes_center_y = int((right_eye.y+left_eye.y)/2*height)
                    end_vector_x = int(eyes_center_x+gaze_world[0]*300)
                    end_vector_y = int(eyes_center_y+gaze_world[1]*300)
                    cv2.arrowedLine(frame, (eyes_center_x,eyes_center_y), (end_vector_x,end_vector_y), (0, 255, 0), 2, tipLength=0.3)
                
                frame = cv2.flip(frame, 1)  # Mirror the camera 
                cv2.putText(frame, average_gaze, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                

        cv2.namedWindow("Test Gaze Tracking", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Test Gaze Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Test Gaze Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    # Safe shutdown of camera
    cap.release()
    cv2.destroyAllWindows()