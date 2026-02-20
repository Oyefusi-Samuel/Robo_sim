import pybullet as p
import pybullet_data
import cv2
import numpy as np
import time
import random
import mediapipe as mp

# --- 1. MEDIA PIPE SETUP (GAZE FOCUS) ---
mp_face_mesh = mp.solutions.face_mesh
# Landmark indices for eyes and irises
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# --- 2. SIMULATION SETUP (Same as your Precise Sorter) ---
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
for i in range(8):
    pos = [random.uniform(0.45, 0.6), random.uniform(-0.15, 0.15), 0.7]
    c_id = p.loadURDF("cube_small.urdf", pos)
    color = [1, 0, 0, 1] if random.random() > 0.5 else [0, 0, 1, 1]
    p.changeVisualShape(c_id, -1, rgbaColor=color)
    cubes.append({'id': c_id, 'color': 'red' if color[0]==1 else 'blue'})

# --- 3. CONTROL VARIABLES ---
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0)
current_idx = 0
state = "APPROACHING"
cid = -1
SAFE_HEIGHT = 0.9

# Gaze confirmation variables
gaze_counter = 0
CONFIRM_THRESHOLD = 15 # Frames of looking at screen to trigger pick

try:
    while cap.isOpened() and p.isConnected():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        gaze_confirmed = False
        if results.multi_face_landmarks:
            # Simple Iris Tracking Logic
            mesh_points = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            
            # Draw irises for visual feedback
            (l_cx, l_cy), l_rad = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_rad = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            cv2.circle(frame, (int(l_cx), int(l_cy)), int(l_rad), (0, 255, 255), 1)
            cv2.circle(frame, (int(r_cx), int(r_cy)), int(r_rad), (0, 255, 255), 1)

            # Logic: If irises are detected, assume focus. 
            # (In a pro version, you'd check iris center vs eye corner distance)
            gaze_counter += 1
            if gaze_counter >= CONFIRM_THRESHOLD:
                gaze_confirmed = True
        else:
            gaze_counter = 0 # Reset if you look away

        if current_idx < len(cubes):
            cube = cubes[current_idx]
            c_pos, _ = p.getBasePositionAndOrientation(cube['id'])
            ee_pos = p.getLinkState(robot_id, 11)[0]
            
            if state == "APPROACHING":
                target = [c_pos[0], c_pos[1], SAFE_HEIGHT]
                status = f"LOOK AT SCREEN TO SORT {cube['color'].upper()}"
                # If hovering, check gaze focus
                if np.linalg.norm(np.array(ee_pos[:2]) - np.array(c_pos[:2])) < 0.02:
                    if gaze_confirmed: 
                        state = "DESCENDING"
                        gaze_counter = 0

            elif state == "DESCENDING":
                target = [c_pos[0], c_pos[1], 0.685]
                status = "STATUS: Gaze confirmed. Descending..."
                if abs(ee_pos[2] - 0.685) < 0.01:
                    cid = p.createConstraint(robot_id, 11, cube['id'], -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])
                    state = "LIFTING"

            elif state == "LIFTING":
                target = [ee_pos[0], ee_pos[1], SAFE_HEIGHT]
                if ee_pos[2] > SAFE_HEIGHT - 0.05: state = "MOVING"

            elif state == "MOVING":
                tray_pos = blue_tray_pos if cube['color'] == 'blue' else red_tray_pos
                target = [tray_pos[0], tray_pos[1], SAFE_HEIGHT]
                if np.linalg.norm(np.array(ee_pos[:2]) - np.array(tray_pos[:2])) < 0.02:
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
        
        p.stepSimulation()
        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Visual Progress Bar for Gaze
        cv2.rectangle(frame, (20, 70), (20 + (gaze_counter * 10), 80), (0, 255, 255), -1)
        
        cv2.imshow("Gaze-Controlled Precise Sorter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    cap.release()
    cv2.destroyAllWindows()
    p.disconnect()