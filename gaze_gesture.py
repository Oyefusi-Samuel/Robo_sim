import pybullet as p
import pybullet_data
import cv2
import numpy as np
import time
import random
import mediapipe as mp

# --- 1. MEDIA PIPE SETUP (Dual Model) ---
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Eye landmarks
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# --- 2. SIMULATION SETUP ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0.5, 0, 0.6])

p.loadURDF("plane.urdf")
table_id = p.loadURDF("table/table.urdf", [0.5, 0, 0], [0, 0, 0, 1])
robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.62], useFixedBase=True)

blue_tray_pos = [0.5, 0.35, 0.64]
red_tray_pos = [0.5, -0.35, 0.64]

cubes = []
for i in range(6):
    pos = [random.uniform(0.45, 0.6), random.uniform(-0.15, 0.15), 0.7]
    c_id = p.loadURDF("cube_small.urdf", pos)
    color = [1, 0, 0, 1] if random.random() > 0.5 else [0, 0, 1, 1]
    p.changeVisualShape(c_id, -1, rgbaColor=color)
    cubes.append({'id': c_id, 'color': 'red' if color[0]==1 else 'blue'})

# --- 3. HRI CONTROLLERS ---
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
cap = cv2.VideoCapture(0)

current_idx, state, cid = 0, "APPROACHING", -1
SAFE_HEIGHT = 0.9

try:
    while cap.isOpened() and p.isConnected():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process both Face and Hands
        face_results = face_mesh.process(rgb_frame)
        hand_results = hands.process(rgb_frame)

        # HRI Logic Flags
        is_looking = False
        is_gesturing = False

        # 1. Check Gaze
        if face_results.multi_face_landmarks:
            is_looking = True # Simple presence-based gaze for this demo
            mesh_points = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) for p in face_results.multi_face_landmarks[0].landmark])
            for idx in LEFT_IRIS + RIGHT_IRIS:
                cv2.circle(frame, tuple(mesh_points[idx]), 1, (0, 255, 255), -1)

        # 2. Check Gesture (Thumbs Up)
        if hand_results.multi_hand_landmarks:
            for lms in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)
                if lms.landmark[4].y < lms.landmark[8].y - 0.1: # Thumb higher than Index
                    is_gesturing = True

        # --- 4. STATE MACHINE ---
        if current_idx < len(cubes):
            cube = cubes[current_idx]
            c_pos, _ = p.getBasePositionAndOrientation(cube['id'])
            ee_pos = p.getLinkState(robot_id, 11)[0]
            
            if state == "APPROACHING":
                target = [c_pos[0], c_pos[1], SAFE_HEIGHT]
                status = "HRI: Look AND Thumbs Up to Sort"
                if np.linalg.norm(np.array(ee_pos[:2]) - np.array(c_pos[:2])) < 0.03:
                    if is_looking and is_gesturing: 
                        state = "DESCENDING"

            elif state == "DESCENDING":
                target = [c_pos[0], c_pos[1], 0.685]
                status = "HRI: Dual-Auth Success! Picking..."
                if abs(ee_pos[2] - 0.685) < 0.01:
                    cid = p.createConstraint(robot_id, 11, cube['id'], -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])
                    state = "LIFTING"

            elif state == "LIFTING":
                target = [ee_pos[0], ee_pos[1], SAFE_HEIGHT]
                if ee_pos[2] > SAFE_HEIGHT - 0.05: state = "MOVING"

            elif state == "MOVING":
                tray_pos = blue_tray_pos if cube['color'] == 'blue' else red_tray_pos
                target = [tray_pos[0], tray_pos[1], SAFE_HEIGHT]
                if np.linalg.norm(np.array(ee_pos[:2]) - np.array(tray_pos[:2])) < 0.03:
                    state = "DROPPING"

            elif state == "DROPPING":
                tray_pos = blue_tray_pos if cube['color'] == 'blue' else red_tray_pos
                target = [tray_pos[0], tray_pos[1], 0.72]
                if ee_pos[2] < 0.73:
                    p.removeConstraint(cid)
                    current_idx += 1
                    state = "APPROACHING"

            joint_poses = p.calculateInverseKinematics(robot_id, 11, target)
            for i in range(7):
                p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joint_poses[i], force=200, maxVelocity=1.5)
        
        # --- UI FEEDBACK ---
        # Visual indicators for dual-auth
        cv2.circle(frame, (30, 100), 10, (0, 255, 0) if is_looking else (0, 0, 255), -1)
        cv2.putText(frame, "GAZE", (50, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.circle(frame, (30, 130), 10, (0, 255, 0) if is_gesturing else (0, 0, 255), -1)
        cv2.putText(frame, "GESTURE", (50, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Multi-Modal HRI Sorter", frame)
        p.stepSimulation()
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    cap.release(); cv2.destroyAllWindows(); p.disconnect()