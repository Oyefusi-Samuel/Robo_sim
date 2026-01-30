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
p.resetDebugVisualizerCamera(1.5, 45, -30, [0.5, 0, 0.5])

p.loadURDF("plane.urdf")
p.loadURDF("table/table.urdf", [0.5, 0, 0])

# --- 2. Defining the Tray Areas ---
# Red tray on the left, Blue tray on the right
red_tray_pos = [0.6, -0.3, 0.63]
blue_tray_pos = [0.6, 0.3, 0.63]

def create_tray(pos, color):
    viz = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.01], rgbaColor=color)
    p.createMultiBody(baseVisualShapeIndex=viz, basePosition=pos)

create_tray(red_tray_pos, [1, 0, 0, 0.5])  # Semi-transparent red
create_tray(blue_tray_pos, [0, 0, 1, 0.5])  # Semi-transparent blue

# Robot & Cubes
robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
cube_red = p.loadURDF("cube_small.urdf", [0.5, -0.1, 0.65])
p.changeVisualShape(cube_red, -1, rgbaColor=[1, 0, 0, 1])
cube_blue = p.loadURDF("cube_small.urdf", [0.5, 0.1, 0.65])
p.changeVisualShape(cube_blue, -1, rgbaColor=[0, 0, 1, 1])

# --- 3. HRI & Gesture Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

def get_gesture(results):
    if not results.multi_hand_landmarks: return "NONE"
    lms = results.multi_hand_landmarks[0].landmark
    pinch = np.linalg.norm(np.array([lms[4].x - lms[8].x, lms[4].y - lms[8].y])) < 0.05
    if pinch: return "PINCH"
    # Open Palm check
    if lms[8].y < lms[6].y and lms[12].y < lms[10].y: return "REJECT"
    return "NONE"

def move_to(pos, gripper_open=True):
    joint_poses = p.calculateInverseKinematics(robot_id, 11, pos)
    for i in range(7):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joint_poses[i])
    g_val = 0.04 if gripper_open else 0.0
    p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, g_val)
    p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, g_val)

# --- 4. Logic Loop ---
cubes = [cube_red, cube_blue]
trays = [red_tray_pos, blue_tray_pos]
current_idx = 0
state = "HOVER"
cid = -1

try:
    while cap.isOpened():
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        gesture = get_gesture(res)

        target_cube = cubes[current_idx]
        target_tray = trays[current_idx]
        c_pos, _ = p.getBasePositionAndOrientation(target_cube)

        if state == "HOVER":
            move_to([c_pos[0], c_pos[1], c_pos[2] + 0.12], gripper_open=True)
            if gesture == "PINCH": state = "PICK"
            if gesture == "REJECT": 
                current_idx = (current_idx + 1) % 2
                time.sleep(0.5)

        elif state == "PICK":
            move_to([c_pos[0], c_pos[1], c_pos[2]], gripper_open=False)
            time.sleep(0.5)
            cid = p.createConstraint(robot_id, 11, target_cube, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])
            state = "PLACE"

        elif state == "PLACE":
            # Lift then move to tray
            move_to([target_tray[0], target_tray[1], target_tray[2] + 0.1], gripper_open=False)
            curr_ee = p.getLinkState(robot_id, 11)[0]
            if np.linalg.norm(np.array(curr_ee[:2]) - np.array(target_tray[:2])) < 0.05:
                p.removeConstraint(cid)
                state = "SUCCESS"

        elif state == "SUCCESS":
            move_to([0.4, 0, 1.0], gripper_open=True) # Home position
            time.sleep(1)
            current_idx = (current_idx + 1) % 2
            state = "HOVER"

        cv2.putText(frame, f"GOAL: {'RED' if current_idx==0 else 'BLUE'} | GESTURE: {gesture}", (10, 50), 1, 1.5, (0,255,0), 2)
        cv2.imshow("HRI Sorting", frame)
        p.stepSimulation()
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    cap.release()
    cv2.destroyAllWindows()
    p.disconnect()