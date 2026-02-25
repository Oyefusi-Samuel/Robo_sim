import pybullet as p
import pybullet_data
import cv2
import numpy as np
import time
import mediapipe as mp

# --- 1. SIMULATION SETUP ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=90, cameraPitch=-35, cameraTargetPosition=[0.3, 0, 0.3])

p.loadURDF("plane.urdf")
# Robot scaled to 0.75 to prevent elbow-table collisions
robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.05], useFixedBase=True, globalScaling=0.75)

# Peg Locations [X, Y, Z_Base]
PEGS = {
    'A': [0.1, -0.4, 0.35], # Side Table
    'B': [0.3, 0.0, 0.05],  # Buffer (Floor/Low)
    'C': [0.45, 0.0, 0.1]   # Front Shelf
}

# Fix Table mass to 0 to prevent scattering
table_id = p.loadURDF("table/table.urdf", [0, -0.4, 0], [0, 0, 0, 1], globalScaling=0.45)
p.changeDynamics(table_id, -1, mass=0)

# --- 2. PUZZLE SETUP (TOWER OF HANOI) ---
disk_ids = []
colors = [[1,0,0,1], [0,1,0,1], [0,0,1,1]] # S, M, L
scales = [0.5, 0.7, 0.9]
for i in range(3):
    # Stack on Peg A
    pos = [PEGS['A'][0], PEGS['A'][1], PEGS['A'][2] + (2-i)*0.06]
    d_id = p.loadURDF("cube_small.urdf", pos, globalScaling=scales[i])
    p.changeVisualShape(d_id, -1, rgbaColor=colors[i])
    disk_ids.append(d_id)

moves = [('A','C'), ('A','B'), ('C','B'), ('A','C'), ('B','A'), ('B','C'), ('A','C')]

# --- 3. HRI & STATE VARIABLES ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
cap = cv2.VideoCapture(0)

state = "APPROACHING"
move_idx = 0
cid = -1
target = [0.2, 0, 0.6] # Initial Target
peg_counts = {'A': 3, 'B': 0, 'C': 0}
status = "Starting..."

# --- 4. MAIN LOOP ---
try:
    while cap.isOpened() and p.isConnected():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        approved = False
        hand_detected = False
        if res.multi_hand_landmarks:
            hand_detected = True
            for lms in res.multi_hand_landmarks:
                # Thumb Up Check
                if lms.landmark[4].y < lms.landmark[8].y - 0.1: approved = True
                mp.solutions.drawing_utils.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

        if move_idx < len(moves):
            src, dst = moves[move_idx]
            ee_pos = p.getLinkState(robot_id, 11)[0]
            
            if state == "APPROACHING":
                target = [PEGS[src][0], PEGS[src][1], 0.65]
                status = f"MOVE {move_idx+1}: {src} to {dst}. [THUMBS UP]"
                if np.linalg.norm(np.array(ee_pos[:2]) - np.array(target[:2])) < 0.05:
                    if approved: state = "PICKING"

            elif state == "PICKING":
                z_pick = PEGS[src][2] + (peg_counts[src]-1) * 0.06 + 0.02
                target = [PEGS[src][0], PEGS[src][1], z_pick]
                status = "ACTION: Grabbing disk..."
                if ee_pos[2] < z_pick + 0.02:
                    # Find closest disk
                    dists = [np.linalg.norm(np.array(p.getBasePositionAndOrientation(d)[0]) - np.array(ee_pos)) for d in disk_ids]
                    closest_disk = disk_ids[np.argmin(dists)]
                    cid = p.createConstraint(robot_id, 11, closest_disk, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])
                    peg_counts[src] -= 1
                    state = "LIFTING"

            elif state == "LIFTING":
                target = [ee_pos[0], ee_pos[1], 0.7]
                if ee_pos[2] > 0.65: state = "RETRACTING"

            elif state == "RETRACTING":
                target = [0.1, -0.1, 0.7] # Waypoint to avoid table edge
                if np.linalg.norm(np.array(ee_pos[:2]) - np.array([0.1, -0.1])) < 0.05:
                    state = "MOVING"

            elif state == "MOVING":
                target = [PEGS[dst][0], PEGS[dst][1], 0.7]
                if np.linalg.norm(np.array(ee_pos[:2]) - np.array(target[:2])) < 0.05:
                    state = "DROPPING"

            elif state == "DROPPING":
                z_drop = PEGS[dst][2] + (peg_counts[dst]) * 0.06 + 0.12
                target = [PEGS[dst][0], PEGS[dst][1], z_drop]
                if ee_pos[2] < z_drop + 0.02:
                    p.removeConstraint(cid)
                    peg_counts[dst] += 1
                    time.sleep(0.5)
                    move_idx += 1
                    state = "APPROACHING"

            # Inverse Kinematics solve
            joint_poses = p.calculateInverseKinematics(robot_id, 11, target, maxNumIterations=150)
            for i in range(7):
                p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joint_poses[i], maxVelocity=1.0)
        else:
            status = "PUZZLE SOLVED!"
            target = [0.2, 0, 0.6]

        # UI & Simulation Step
        p.stepSimulation()
        cv2.circle(frame, (30, 30), 10, (0, 255, 0) if hand_detected else (0, 0, 255), -1)
        cv2.putText(frame, status, (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("WPI RBE 526: Tower of Hanoi Final", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    cap.release(); cv2.destroyAllWindows(); p.disconnect()