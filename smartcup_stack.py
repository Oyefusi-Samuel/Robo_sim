import pybullet as p
import pybullet_data
import cv2
import numpy as np
import time
import random
import mediapipe as mp

# --- 1. SIMULATION SETUP ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=90, cameraPitch=-35, cameraTargetPosition=[0.3, 0, 0.3])

p.loadURDF("plane.urdf")
robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.05], useFixedBase=True, globalScaling=0.75)
table_id = p.loadURDF("table/table.urdf", [0, -0.4, 0], [0, 0, 0, 1], globalScaling=0.45)
p.changeDynamics(table_id, -1, mass=0)

# --- 2. ORDERED STACKING DATA ---
REQUIRED_ORDER = ['red', 'green', 'blue']
stack_base_pos = [0.45, 0, 0.05]
cup_height = 0.06 

cups = []
colors_rgb = {'red': [1,0,0,1], 'green': [0,1,0,1], 'blue': [0,0,1,1]}

# Spawn cubes
for color_name in ['blue', 'red', 'green', 'red', 'blue']:
    pos = [random.uniform(-0.05, 0.05), random.uniform(-0.45, -0.35), 0.35]
    c_id = p.loadURDF("cube_small.urdf", pos)
    p.changeVisualShape(c_id, -1, rgbaColor=colors_rgb[color_name])
    cups.append({'id': c_id, 'color': color_name, 'sorted': False})

# --- 3. HRI SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
cap = cv2.VideoCapture(0)

current_order_idx = 0
state = "SEARCHING"
cid = -1
target_cup_data = None
status = "Initializing..."

# --- FIX: Initialize target to a safe home position ---
target = [0.2, 0, 0.6] 

try:
    while cap.isOpened() and p.isConnected():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        thumb_up = False
        if res.multi_hand_landmarks:
            for lms in res.multi_hand_landmarks:
                # Thumb higher than index tip
                if lms.landmark[4].y < lms.landmark[8].y - 0.1: thumb_up = True
                mp.solutions.drawing_utils.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

        if current_order_idx < len(REQUIRED_ORDER):
            goal_color = REQUIRED_ORDER[current_order_idx]
            ee_pos = p.getLinkState(robot_id, 11)[0]
            
            if state == "SEARCHING":
                status = f"SEARCHING: Need {goal_color} cup..."
                target = [0.2, 0, 0.6] # Stay at home while searching
                for cup in cups:
                    if cup['color'] == goal_color and not cup['sorted']:
                        target_cup_data = cup
                        state = "APPROACHING"
                        break

            elif state == "APPROACHING":
                c_pos, _ = p.getBasePositionAndOrientation(target_cup_data['id'])
                target = [c_pos[0], c_pos[1], 0.6]
                status = f"CONFIRM: Found {goal_color}. Give THUMBS UP!"
                if np.linalg.norm(np.array(ee_pos[:2]) - np.array(c_pos[:2])) < 0.04:
                    if thumb_up: state = "PICKING"

            elif state == "PICKING":
                c_pos, _ = p.getBasePositionAndOrientation(target_cup_data['id'])
                target = [c_pos[0], c_pos[1], 0.325]
                status = "ACTION: Picking..."
                if ee_pos[2] < 0.34:
                    cid = p.createConstraint(robot_id, 11, target_cup_data['id'], -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])
                    state = "RETRACTING"

            elif state == "RETRACTING":
                target = [0.1, -0.1, 0.7]
                status = "ACTION: Clearing table..."
                if np.linalg.norm(np.array(ee_pos[:2]) - np.array([0.1, -0.1])) < 0.05:
                    state = "MOVING"

            elif state == "MOVING":
                z_stack = stack_base_pos[2] + (current_order_idx * cup_height) + 0.12
                target = [stack_base_pos[0], stack_base_pos[1], z_stack]
                status = "ACTION: Moving to stack..."
                if np.linalg.norm(np.array(ee_pos[:2]) - np.array(stack_base_pos[:2])) < 0.04:
                    state = "DROPPING"

            elif state == "DROPPING":
                status = "ACTION: Releasing..."
                p.removeConstraint(cid)
                target_cup_data['sorted'] = True
                current_order_idx += 1
                time.sleep(1.0) # Wait for physics to settle
                state = "SEARCHING"

            # IK with a defined target
            joint_poses = p.calculateInverseKinematics(robot_id, 11, target, maxNumIterations=150)
            for i in range(7):
                p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joint_poses[i], maxVelocity=1.0)
        else:
            status = "SUCCESS: Ordered Stack Complete!"
            target = [0.2, 0, 0.6] # Return home

        p.stepSimulation()
        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Ordered Cup Stacker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    cap.release(); cv2.destroyAllWindows(); p.disconnect()