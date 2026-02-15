import pybullet as p
import pybullet_data
import cv2
import numpy as np
import time
import random
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw

# --- 1. Simulation Setup ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=90, cameraPitch=-35, cameraTargetPosition=[0.3, 0, 0.3])

p.loadURDF("plane.urdf")

# --- 2. Environment Setup ---
# Robot: Small scale (0.75) for better joint stability
robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.05], useFixedBase=True, globalScaling=0.75)

# Table: Moved to the SIDE (Y = -0.4)
table_id = p.loadURDF("table/table.urdf", [0, -0.4, 0], [0, 0, 0, 1], globalScaling=0.45)
p.changeDynamics(table_id, -1, mass=0) 

# Shelf: Moved to the FRONT (X = 0.45, Y = 0)
shelf_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.12, 0.15, 0.015], rgbaColor=[0.4, 0.2, 0.1, 1])
shelf_c = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.12, 0.15, 0.015])

LEVEL_Z = [0.05, 0.25, 0.45] 
for z in LEVEL_Z:
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shelf_c, baseVisualShapeIndex=shelf_v, basePosition=[0.45, 0, z])

# Targets updated for front shelf
targets = {'red': [0.45, 0, 0.5], 'blue': [0.45, 0, 0.3], 'yellow': [0.45, 0, 0.1]}

# --- 3. Cubes ---
cubes = []
colors_rgb = {'red': [1,0,0,1], 'blue': [0,0,1,1], 'yellow': [1,1,0,1]}
for _ in range(6):
    pos = [random.uniform(-0.05, 0.05), random.uniform(-0.45, -0.35), 0.35]
    c_id = p.loadURDF("cube_small.urdf", pos)
    name = random.choice(['red', 'blue', 'yellow'])
    p.changeVisualShape(c_id, -1, rgbaColor=colors_rgb[name])
    cubes.append({'id': c_id, 'color': name})

# --- 4. Control Loop ---
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
cap = cv2.VideoCapture(0)
current_idx, state, cid = 0, "APPROACHING", -1

try:
    while cap.isOpened() and p.isConnected():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        approved = False
        if res.multi_hand_landmarks:
            for lms in res.multi_hand_landmarks:
                if lms.landmark[4].y < lms.landmark[8].y - 0.1: approved = True
                mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

        if current_idx < len(cubes):
            cube = cubes[current_idx]
            c_pos, _ = p.getBasePositionAndOrientation(cube['id'])
            ee_pos = p.getLinkState(robot_id, 11)[0]
            
            if state == "APPROACHING":
                target = [c_pos[0], c_pos[1], 0.6] 
                status = f"SUPERVISE: Confirm {cube['color']}?"
                if np.linalg.norm(np.array(ee_pos[:2]) - np.array(c_pos[:2])) < 0.05:
                    if approved: state = "PICKING"

            elif state == "PICKING":
                target = [c_pos[0], c_pos[1], 0.32] 
                if ee_pos[2] < 0.34:
                    cid = p.createConstraint(robot_id, 11, cube['id'], -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])
                    state = "LIFTING"

            elif state == "LIFTING":
                # LIFT HIGH to clear the table completely
                target = [ee_pos[0], ee_pos[1], 0.7] 
                if ee_pos[2] > 0.65: state = "RETRACTING"

            elif state == "RETRACTING":
                # PULL ARM IN toward center to avoid getting stuck during rotation
                status = "ACTION: Retracting arm..."
                target = [0.1, -0.1, 0.7] 
                if np.linalg.norm(np.array(ee_pos[:2]) - np.array([0.1, -0.1])) < 0.05:
                    state = "MOVING"

            elif state == "MOVING":
                status = f"ACTION: Moving to shelf..."
                goal = targets[cube['color']]
                target = [goal[0], goal[1], goal[2]]
                if abs(ee_pos[0] - goal[0]) < 0.05: state = "DROPPING"

            elif state == "DROPPING":
                goal = targets[cube['color']]
                target = [0.48, goal[1], goal[2]] 
                if ee_pos[0] > 0.46:
                    p.removeConstraint(cid)
                    time.sleep(0.3)
                    current_idx += 1
                    state = "APPROACHING"

            joint_poses = p.calculateInverseKinematics(robot_id, 11, target, maxNumIterations=150)
            for i in range(7):
                p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joint_poses[i], maxVelocity=0.8)
        
        p.stepSimulation()
        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Front Shelf + Waypoint Fix", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    cap.release(); cv2.destroyAllWindows(); p.disconnect()