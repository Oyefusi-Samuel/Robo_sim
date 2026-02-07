import pybullet as p
import pybullet_data
import cv2
import numpy as np
import time
import random

# --- DIRECT IMPORTS ---
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw

# --- 1. Simulation Setup ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0.5, 0, 0.6])

p.loadURDF("plane.urdf")
table_id = p.loadURDF("table/table.urdf", [0.5, 0, 0], [0, 0, 0, 1])
robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.62], useFixedBase=True)

# Trays (Increased height slightly to act as a container)
blue_tray_pos = [0.5, 0.35, 0.64]
red_tray_pos = [0.5, -0.35, 0.64]
tray_viz_b = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.12, 0.12, 0.02], rgbaColor=[0, 0, 1, 0.6])
tray_viz_r = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.12, 0.12, 0.02], rgbaColor=[1, 0, 0, 0.6])
p.createMultiBody(baseVisualShapeIndex=tray_viz_b, basePosition=blue_tray_pos)
p.createMultiBody(baseVisualShapeIndex=tray_viz_r, basePosition=red_tray_pos)

# --- 2. Scatter Mixed Cubes ---
cubes = []
for i in range(8):
    # Narrower scattering range to keep them away from the tray edges
    pos = [random.uniform(0.45, 0.6), random.uniform(-0.15, 0.15), 0.7]
    c_id = p.loadURDF("cube_small.urdf", pos)
    color = [1, 0, 0, 1] if random.random() > 0.5 else [0, 0, 1, 1]
    p.changeVisualShape(c_id, -1, rgbaColor=color)
    cubes.append({'id': c_id, 'color': 'red' if color[0]==1 else 'blue', 'sorted': False})

# --- 3. Control Variables ---
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
cap = cv2.VideoCapture(0)
current_idx = 0
state = "APPROACHING" # APPROACHING -> DESCENDING -> PICKING -> LIFTING -> MOVING -> DROPPING
cid = -1
SAFE_HEIGHT = 0.9  # Height to move across the table without hitting cubes
APPROACH_HEIGHT = 0.75 # Height just above the cube

def is_thumbs_up(hand_lms):
    thumb_tip = hand_lms.landmark[4].y
    index_tip = hand_lms.landmark[8].y
    return thumb_tip < index_tip - 0.1

# --- 4. Main Loop ---
try:
    while cap.isOpened() and p.isConnected():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        approved = False
        if res.multi_hand_landmarks:
            for lms in res.multi_hand_landmarks:
                if is_thumbs_up(lms): approved = True
                mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

        if current_idx < len(cubes):
            cube = cubes[current_idx]
            c_pos, _ = p.getBasePositionAndOrientation(cube['id'])
            ee_pos = p.getLinkState(robot_id, 11)[0]
            
            # --- PRECISE STATE MACHINE ---
            if state == "APPROACHING":
                target = [c_pos[0], c_pos[1], SAFE_HEIGHT]
                status = "STATUS: Approaching Cube. Give THUMBS UP!"
                if np.linalg.norm(np.array(ee_pos[:2]) - np.array(c_pos[:2])) < 0.02:
                    if approved: state = "DESCENDING"
            
            elif state == "DESCENDING":
                target = [c_pos[0], c_pos[1], 0.685] # Precise grab height
                status = "STATUS: Descending carefully..."
                if abs(ee_pos[2] - 0.685) < 0.01:
                    cid = p.createConstraint(robot_id, 11, cube['id'], -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])
                    state = "LIFTING"

            elif state == "LIFTING":
                target = [ee_pos[0], ee_pos[1], SAFE_HEIGHT]
                status = "STATUS: Lifting cube safely."
                if ee_pos[2] > SAFE_HEIGHT - 0.05: state = "MOVING"

            elif state == "MOVING":
                tray_pos = blue_tray_pos if cube['color'] == 'blue' else red_tray_pos
                target = [tray_pos[0], tray_pos[1], SAFE_HEIGHT]
                status = f"STATUS: Transporting to {cube['color']} tray"
                if np.linalg.norm(np.array(ee_pos[:2]) - np.array(tray_pos[:2])) < 0.02:
                    state = "DROPPING"

            elif state == "DROPPING":
                tray_pos = blue_tray_pos if cube['color'] == 'blue' else red_tray_pos
                target = [tray_pos[0], tray_pos[1], 0.72] # Lower into tray before release
                status = "STATUS: Soft release..."
                if ee_pos[2] < 0.73:
                    p.removeConstraint(cid)
                    time.sleep(0.2) # Wait for settle
                    cube['sorted'] = True
                    current_idx += 1
                    state = "APPROACHING"

            # Inverse Kinematics with limited max velocity for smoothness
            joint_poses = p.calculateInverseKinematics(robot_id, 11, target)
            for i in range(7):
                p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joint_poses[i], force=200, maxVelocity=1.5)
        else:
            status = "STATUS: Sorting Complete!"

        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("WPI RBE 526: Precise Sorter", frame)
        p.stepSimulation()
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release()
    cv2.destroyAllWindows()
    p.disconnect()