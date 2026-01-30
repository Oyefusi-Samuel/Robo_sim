import pybullet as p
import pybullet_data
import cv2
import numpy as np
import time

# --- DIRECT IMPORTS (Proven to bypass the solutions error) ---
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw

# --- 1. Simulation & Environment Setup ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0.5, 0, 0.5])

p.loadURDF("plane.urdf")
p.loadURDF("table/table.urdf", [0.5, 0, 0], [0, 0, 0, 1])

# Create Visual Zones (Bins) - Using transparent colors
blue_zone_viz = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.12, 0.12, 0.005], rgbaColor=[0, 0, 1, 0.4])
red_zone_viz = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.12, 0.12, 0.005], rgbaColor=[1, 0, 0, 0.4])

# Define exact bin positions
blue_bin_pos = [0.5, 0.35, 0.63]
red_bin_pos = [0.5, -0.35, 0.63]

p.createMultiBody(baseVisualShapeIndex=blue_zone_viz, basePosition=blue_bin_pos)
p.createMultiBody(baseVisualShapeIndex=red_zone_viz, basePosition=red_bin_pos)

# Load Robot
robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

# Load and Color the Cubes
cube_red = p.loadURDF("cube_small.urdf", [0.45, -0.1, 0.65])
p.changeVisualShape(cube_red, -1, rgbaColor=[1, 0, 0, 1])

cube_blue = p.loadURDF("cube_small.urdf", [0.45, 0.1, 0.65])
p.changeVisualShape(cube_blue, -1, rgbaColor=[0, 0, 1, 1])

# --- 2. Global Variables for HRI Logic ---
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
cap = cv2.VideoCapture(0)
cid = -1 # Constraint ID for picking up objects
score = 0
red_sorted = False
blue_sorted = False

def handle_grasping(gripper_pos, ee_pos, cubes):
    """Snaps the cube to the gripper when pinched."""
    global cid
    if gripper_pos < 0.01 and cid == -1:
        for cube in cubes:
            cube_pos, _ = p.getBasePositionAndOrientation(cube)
            dist = np.linalg.norm(np.array(ee_pos) - np.array(cube_pos))
            if dist < 0.06:
                # Create a fixed constraint between robot link 11 and the cube
                cid = p.createConstraint(robot_id, 11, cube, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])
                break
    elif gripper_pos > 0.03 and cid != -1:
        p.removeConstraint(cid)
        cid = -1

def check_scoring():
    """Checks if cubes are inside their respective color zones."""
    global score, red_sorted, blue_sorted
    
    r_pos, _ = p.getBasePositionAndOrientation(cube_red)
    b_pos, _ = p.getBasePositionAndOrientation(cube_blue)
    
    # Check Red Cube in Red Bin
    if not red_sorted and np.linalg.norm(np.array(r_pos[:2]) - np.array(red_bin_pos[:2])) < 0.1:
        score += 1
        red_sorted = True
        
    # Check Blue Cube in Blue Bin
    if not blue_sorted and np.linalg.norm(np.array(b_pos[:2]) - np.array(blue_bin_pos[:2])) < 0.1:
        score += 1
        blue_sorted = True

# --- 3. Main Loop ---
try:
    print("HRI Sorter System Running. Control the arm with your hand.")
    while cap.isOpened() and p.isConnected():
        success, frame = cap.read()
        if not success: continue
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        status_msg = "REACH FOR A CUBE"
        msg_color = (255, 255, 255)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                idx = hand_lms.landmark[8]    # Index finger tip
                thumb = hand_lms.landmark[4]  # Thumb tip
                
                # --- COORDINATE MAPPING ---
                # Hand horizontal (x) -> Robot Y (left/right)
                # Hand vertical (y) -> Robot X (forward/back)
                # Pinch distance -> Gripper state
                
                target_x = 0.35 + (idx.y * 0.4) 
                target_y = (idx.x - 0.5) * 1.0 
                target_z = 0.68 + (0.32 * (1.0 - idx.y))

                # Inverse Kinematics for Link 11 (Hand center)
                joint_poses = p.calculateInverseKinematics(robot_id, 11, [target_x, target_y, target_z])
                for i in range(7):
                    p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joint_poses[i])

                # Gripper Position calculation
                pinch_dist = np.linalg.norm(np.array([idx.x - thumb.x, idx.y - thumb.y]))
                g_pos = 0.0 if pinch_dist < 0.05 else 0.04
                p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, g_pos)
                p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, g_pos)

                # Logic for Grasping and Scoring
                handle_grasping(g_pos, [target_x, target_y, target_z], [cube_red, cube_blue])
                check_scoring()

                # HRI Visual Guidance Logic
                if cid != -1:
                    status_msg = "MOVING TO TARGET BIN..."
                    msg_color = (0, 255, 0)
                elif pinch_dist < 0.05:
                    status_msg = "PINCH DETECTED"
                    msg_color = (255, 255, 0)

                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        # UI OVERLAY (Score and Status)
        cv2.rectangle(frame, (0,0), (w, 65), (50, 50, 50), -1)
        cv2.putText(frame, f"STATUS: {status_msg}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, msg_color, 2)
        cv2.putText(frame, f"SCORE: {score}/2", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        p.stepSimulation()
        cv2.imshow("RBE 526: HRI Sorting System", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release()
    cv2.destroyAllWindows()
    p.disconnect()