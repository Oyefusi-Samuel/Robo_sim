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
p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=40, cameraPitch=-25, cameraTargetPosition=[0.5, 0, 0.8])

p.loadURDF("plane.urdf")

# --- 2. Build the Shelf (Visual and Collision) ---
# We'll create a 3-tier shelf using scaled boxes
shelf_color = [0.4, 0.2, 0.1, 1] # Dark wood
shelf_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.4, 0.02], rgbaColor=shelf_color)
shelf_c = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.4, 0.02])

# Ground Shelf (Bottom)
p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shelf_c, baseVisualShapeIndex=shelf_v, basePosition=[0.6, 0, 0.1])
# Middle Shelf
p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shelf_c, baseVisualShapeIndex=shelf_v, basePosition=[0.6, 0, 0.5])
# Top Shelf
p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shelf_c, baseVisualShapeIndex=shelf_v, basePosition=[0.6, 0, 0.9])

# Load Robot (Placed on a pedestal to reach the top shelf)
p.loadURDF("cube.urdf", [0, 0, 0.2], globalScaling=0.4) # Pedestal
robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.4], useFixedBase=True)

# Define Drop-off Bins on different shelves
# Red cubes go to the Top Shelf (0.9m), Blue cubes go to Middle Shelf (0.5m)
red_target = [0.6, -0.2, 1.0] # Slightly above top shelf
blue_target = [0.6, 0.2, 0.6] # Slightly above middle shelf

# --- 3. Mixed Cubes on the Bottom Shelf ---
cubes = []
for i in range(6):
    pos = [random.uniform(0.5, 0.7), random.uniform(-0.2, 0.2), 0.2]
    c_id = p.loadURDF("cube_small.urdf", pos)
    color = [1, 0, 0, 1] if random.random() > 0.5 else [0, 0, 1, 1]
    p.changeVisualShape(c_id, -1, rgbaColor=color)
    cubes.append({'id': c_id, 'color': 'red' if color[0]==1 else 'blue', 'sorted': False})

# --- 4. HRI & Control Variables ---
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
cap = cv2.VideoCapture(0)
current_idx = 0
state = "APPROACHING" 
cid = -1
PICK_HEIGHT = 0.25 # Just above the bottom shelf

def is_thumbs_up(hand_lms):
    thumb_tip = hand_lms.landmark[4].y
    index_tip = hand_lms.landmark[8].y
    return thumb_tip < index_tip - 0.1

# --- 5. Main Loop ---
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
            
            if state == "APPROACHING":
                target = [c_pos[0], c_pos[1], 0.4] # Hover above bottom shelf
                status = "STATUS: Reached Cube. Approve pick?"
                if approved: state = "PICKING"
            
            elif state == "PICKING":
                target = [c_pos[0], c_pos[1], 0.22] # Lower to grab
                status = "STATUS: Picking..."
                if abs(ee_pos[2] - 0.22) < 0.02:
                    cid = p.createConstraint(robot_id, 11, cube['id'], -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])
                    state = "LIFTING"

            elif state == "LIFTING":
                target = [0.3, ee_pos[1], 0.6] # Pull BACK and UP to avoid hitting middle shelf
                status = "STATUS: Retracting from shelf..."
                if ee_pos[0] < 0.35: state = "MOVING"

            elif state == "MOVING":
                goal = red_target if cube['color'] == 'red' else blue_target
                target = [0.4, goal[1], goal[2]] # Align with shelf height
                status = f"STATUS: Moving to {cube['color']} shelf"
                if abs(ee_pos[1] - goal[1]) < 0.05: state = "DROPPING"

            elif state == "DROPPING":
                goal = red_target if cube['color'] == 'red' else blue_target
                target = [0.6, goal[1], goal[2]] # Move forward INTO the shelf
                status = "STATUS: Placing on shelf..."
                if ee_pos[0] > 0.58:
                    p.removeConstraint(cid)
                    time.sleep(0.3)
                    cube['sorted'] = True
                    current_idx += 1
                    state = "APPROACHING"

            # Smooth Motion Control
            joint_poses = p.calculateInverseKinematics(robot_id, 11, target)
            for i in range(7):
                p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joint_poses[i], maxVelocity=1.2)
        else:
            status = "STATUS: Shelf Stocking Complete!"

        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("WPI RBE 526: Shelf Supervision", frame)
        p.stepSimulation()
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release()
    cv2.destroyAllWindows()
    p.disconnect()