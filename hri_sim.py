import pybullet as p
import pybullet_data
import cv2
import numpy as np
import time

# --- NEW DIRECT IMPORTS ---
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw

# --- 1. Simulation Setup (Same as before) ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")
p.loadURDF("table/table.urdf", [0.5, 0, 0], [0, 0, 0, 1])
robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

# --- 2. MediaPipe Hand Tracking Setup (UPDATED) ---
# Use the new mp_hands variable we imported directly
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

def control_gripper(dist):
    gripper_pos = 0.0 if dist < 0.06 else 0.04
    p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, gripper_pos)
    p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, gripper_pos)

print("Sim active. Mirror your hand in the camera to control the arm.")

# --- 3. Main Loop ---
try:
    while cap.isOpened() and p.isConnected():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                idx = hand_lms.landmark[8]
                thumb = hand_lms.landmark[4]

                # Mapping camera to robot workspace
                target_x = 0.4 + (idx.y * 0.4) 
                target_y = (idx.x - 0.5) * 0.8
                target_z = 0.7 

                joint_poses = p.calculateInverseKinematics(robot_id, 11, [target_x, target_y, target_z])
                
                for i in range(7):
                    p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joint_poses[i])

                pinch_dist = np.linalg.norm(np.array([idx.x - thumb.x, idx.y - thumb.y]))
                control_gripper(pinch_dist)

                # Use the new mp_draw variable
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        p.stepSimulation()
        cv2.imshow("HRI Interface", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    p.disconnect()