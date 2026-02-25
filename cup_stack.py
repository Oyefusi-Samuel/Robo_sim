import pybullet as p
import pybullet_data
import cv2
import numpy as np
import time
import mediapipe as mp

# --- 1. Simulation Setup ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=[0.4, 0, 0.4])

p.loadURDF("plane.urdf")
# Load a heavy, static table
table_id = p.loadURDF("table/table.urdf", [0.4, 0, 0], useFixedBase=True, globalScaling=0.5)

# Load the Panda Robot
robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.31], useFixedBase=True, globalScaling=0.8)

# --- 2. Load "Cups" (Small Blocks) ---
# We use cubes as simplified cups for easier stacking physics
cup_ids = []
colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]] # Red, Green, Blue
for i in range(3):
    # Spread cups out on the table
    pos = [0.4, -0.2 + (i * 0.2), 0.4]
    c_id = p.loadURDF("cube_small.urdf", pos)
    p.changeVisualShape(c_id, -1, rgbaColor=colors[i])
    cup_ids.append(c_id)

# --- 3. Gesture Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# --- 4. Stacking Logic Variables ---
current_cup_idx = 0
state = "APPROACHING"
cid = -1
stack_base_pos = [0.5, 0.0, 0.35] # Where the tower will be built
cup_height = 0.06 # Offset for each new layer

try:
    while cap.isOpened() and p.isConnected():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Gesture Detection
        thumb_up = False
        index_up = False
        if results.multi_hand_landmarks:
            for lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)
                # Thumb Up Logic
                if lms.landmark[4].y < lms.landmark[8].y - 0.1: thumb_up = True
                # Index Up (Pointing) Logic
                if lms.landmark[8].y < lms.landmark[6].y - 0.05: index_up = True

        if current_cup_idx < len(cup_ids):
            target_cup = cup_ids[current_cup_idx]
            c_pos, _ = p.getBasePositionAndOrientation(target_cup)
            ee_pos = p.getLinkState(robot_id, 11)[0]

            if state == "APPROACHING":
                target = [c_pos[0], c_pos[1], 0.6]
                status = "GESTURE: Thumb Up to PICK cup"
                if np.linalg.norm(np.array(ee_pos[:2]) - np.array(c_pos[:2])) < 0.03:
                    if thumb_up: state = "PICKING"

            elif state == "PICKING":
                target = [c_pos[0], c_pos[1], 0.34]
                status = "ACTION: Grabbing..."
                if ee_pos[2] < 0.36:
                    cid = p.createConstraint(robot_id, 11, target_cup, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])
                    state = "LIFTING"

            elif state == "LIFTING":
                target = [ee_pos[0], ee_pos[1], 0.7]
                if ee_pos[2] > 0.65: state = "MOVING"

            elif state == "MOVING":
                # Move above the stack base
                target = [stack_base_pos[0], stack_base_pos[1], 0.7]
                status = "GESTURE: Index Up to STACK cup"
                if np.linalg.norm(np.array(ee_pos[:2]) - np.array(stack_base_pos[:2])) < 0.03:
                    if index_up: state = "STACKING"

            elif state == "STACKING":
                # Calculate height based on how many cups are already there
                current_stack_z = stack_base_pos[2] + (current_cup_idx * cup_height)
                target = [stack_base_pos[0], stack_base_pos[1], current_stack_z + 0.05]
                status = f"ACTION: Placing level {current_cup_idx + 1}"
                
                if abs(ee_pos[2] - (current_stack_z + 0.05)) < 0.01:
                    p.removeConstraint(cid)
                    time.sleep(0.5) # Let physics settle
                    current_cup_idx += 1
                    state = "APPROACHING"

            # Inverse Kinematics
            joint_poses = p.calculateInverseKinematics(robot_id, 11, target)
            for i in range(7):
                p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joint_poses[i], maxVelocity=1.0)

        p.stepSimulation()
        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Hand Gesture Cup Stacking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    cap.release(); cv2.destroyAllWindows(); p.disconnect()