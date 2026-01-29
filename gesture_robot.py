import pybullet as p
import pybullet_data
import cv2
import mediapipe as mp
import numpy as np

# --- 1. Simulation Setup ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# Load a robot with a gripper (Franka Emika Panda)
robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
# Create a small object to pick up
target_obj = p.loadURDF("sphere_1cm.urdf", [0.5, 0, 0.05], [0,0,0,1], globalScaling=5)

# --- 2. MediaPipe Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

def move_gripper(open_gripper):
    # Joint indices for Panda gripper are 9 and 10
    target_pos = 0.04 if open_gripper else 0.0
    p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, target_pos)
    p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, target_pos)

print("Starting simulation... Keep your hand in view of the webcam.")

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) # Mirror for intuitive control
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get Index finger tip (ID 8) and Thumb tip (ID 4)
            idx_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            # Map coordinates (Adjust multipliers to fit your reach)
            # Hand X (0 to 1) -> Robot X (0.3 to 0.7)
            # Hand Y (0 to 1) -> Robot Y (-0.5 to 0.5)
            world_x = 0.3 + (idx_tip.y * 0.4) 
            world_y = (idx_tip.x - 0.5) * 1.0
            world_z = 0.2 # Keep at a fixed height for now

            # Calculate Inverse Kinematics
            joint_poses = p.calculateInverseKinematics(robot_id, 11, [world_x, world_y, world_z])
            
            # Apply to robot
            for i in range(len(joint_poses)):
                p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joint_poses[i])

            # Check Gesture for Grabbing (Distance between thumb and index)
            dist = np.linalg.norm(np.array([idx_tip.x - thumb_tip.x, idx_tip.y - thumb_tip.y]))
            move_gripper(dist > 0.05) # If fingers are close, close gripper

    p.stepSimulation()
    cv2.imshow("Hand Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
p.disconnect()