import pybullet as p
import pybullet_data
import cv2
import numpy as np
import time
import random

# --- DIRECT IMPORTS FOR COMPUTER VISION ---
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw

# --- 1. SIMULATION & PHYSICS SETUP ---
p.connect(p.GUI)  # Open the PyBullet graphical interface
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Access built-in URDF assets
p.setGravity(0, 0, -9.81)  # Set earth-like gravity

# Configure the virtual camera: Distance, Yaw (rotate), Pitch (tilt), and Focus Point
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0.5, 0, 0.6])

# Load world objects
p.loadURDF("plane.urdf")  # The ground floor
table_id = p.loadURDF("table/table.urdf", [0.5, 0, 0], [0, 0, 0, 1])  # The work surface

# Load the Franka Panda Robot. 
# Position [0, 0, 0.62] ensures it sits perfectly on the table surface.
robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0.62], useFixedBase=True)

# --- 2. WORKSPACE ELEMENTS (TRAYS & CUBES) ---
# Define goal positions for the blue and red trays
blue_tray_pos = [0.5, 0.35, 0.64]
red_tray_pos = [0.5, -0.35, 0.64]

# Create visual markers for the trays (transparent boxes)
tray_viz_b = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.12, 0.12, 0.02], rgbaColor=[0, 0, 1, 0.6])
tray_viz_r = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.12, 0.12, 0.02], rgbaColor=[1, 0, 0, 0.6])
p.createMultiBody(baseVisualShapeIndex=tray_viz_b, basePosition=blue_tray_pos)
p.createMultiBody(baseVisualShapeIndex=tray_viz_r, basePosition=red_tray_pos)

# Scatter 8 cubes with randomized positions and colors
cubes = []
for i in range(8):
    # Random position within a safe area on the table center
    pos = [random.uniform(0.45, 0.6), random.uniform(-0.15, 0.15), 0.7]
    c_id = p.loadURDF("cube_small.urdf", pos)
    
    # 50/50 chance for a cube to be red [1,0,0] or blue [0,0,1]
    color = [1, 0, 0, 1] if random.random() > 0.5 else [0, 0, 1, 1]
    p.changeVisualShape(c_id, -1, rgbaColor=color)
    
    # Store cube data in a list for the state machine to track
    cubes.append({'id': c_id, 'color': 'red' if color[0]==1 else 'blue', 'sorted': False})

# --- 3. HRI CONTROL VARIABLES ---
# Initialize MediaPipe Hands for gesture recognition
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
cap = cv2.VideoCapture(0)  # Connect to webcam

current_idx = 0  # Which cube is the robot currently targeting?
state = "APPROACHING"  # Start the State Machine in the initial phase
cid = -1  # Placeholder for the constraint (the "magnetic" grip)

# Vertical waypoints for safe movement
SAFE_HEIGHT = 0.9      # High enough to fly over other cubes
APPROACH_HEIGHT = 0.75 # Just above the cube before the grab

def is_thumbs_up(hand_lms):
    """
    Checks if the thumb tip (landmark 4) is significantly higher 
    on the screen than the index tip (landmark 8).
    Note: In screen coordinates, a smaller Y value is higher up.
    """
    thumb_tip = hand_lms.landmark[4].y
    index_tip = hand_lms.landmark[8].y
    return thumb_tip < index_tip - 0.1

# --- 4. MAIN EXECUTION LOOP ---
try:
    while cap.isOpened() and p.isConnected():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)  # Mirror the camera for intuitive control
        
        # Process image with MediaPipe
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb_image)
        
        approved = False  # Default to no approval
        if res.multi_hand_landmarks:
            for lms in res.multi_hand_landmarks:
                if is_thumbs_up(lms): 
                    approved = True  # User has given the green light
                mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

        # Only run sorting logic if there are cubes left to sort
        if current_idx < len(cubes):
            cube = cubes[current_idx]
            c_pos, _ = p.getBasePositionAndOrientation(cube['id'])
            ee_pos = p.getLinkState(robot_id, 11)[0] # Link 11 is the Panda Gripper center
            
            # --- SUPERVISORY STATE MACHINE LOGIC ---
            
            # PHASE 1: Move horizontally to hover above the next cube
            if state == "APPROACHING":
                target = [c_pos[0], c_pos[1], SAFE_HEIGHT]
                status = "STATUS: Approaching Cube. Give THUMBS UP!"
                # Transition: If close to target horizontally AND user approves
                if np.linalg.norm(np.array(ee_pos[:2]) - np.array(c_pos[:2])) < 0.02:
                    if approved: state = "DESCENDING"
            
            # PHASE 2: Lower the arm carefully to the cube's top
            elif state == "DESCENDING":
                target = [c_pos[0], c_pos[1], 0.685] # Target grab height
                status = "STATUS: Descending carefully..."
                # Transition: Once vertical height is reached, "grab" and lift
                if abs(ee_pos[2] - 0.685) < 0.01:
                    # Create a fixed constraint (snaps the cube to the gripper)
                    cid = p.createConstraint(robot_id, 11, cube['id'], -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])
                    state = "LIFTING"

            # PHASE 3: Lift the cube straight up to avoid knocking neighbors
            elif state == "LIFTING":
                target = [ee_pos[0], ee_pos[1], SAFE_HEIGHT]
                status = "STATUS: Lifting cube safely."
                # Transition: Once at safe height, move to tray
                if ee_pos[2] > SAFE_HEIGHT - 0.05: state = "MOVING"

            # PHASE 4: Transport cube horizontally to the correct color tray
            elif state == "MOVING":
                tray_pos = blue_tray_pos if cube['color'] == 'blue' else red_tray_pos
                target = [tray_pos[0], tray_pos[1], SAFE_HEIGHT]
                status = f"STATUS: Transporting to {cube['color']} tray"
                # Transition: Once above the tray, begin landing
                if np.linalg.norm(np.array(ee_pos[:2]) - np.array(tray_pos[:2])) < 0.02:
                    state = "DROPPING"

            # PHASE 5: Lower cube into tray and release
            elif state == "DROPPING":
                tray_pos = blue_tray_pos if cube['color'] == 'blue' else red_tray_pos
                target = [tray_pos[0], tray_pos[1], 0.72] # Hover just above tray floor
                status = "STATUS: Soft release..."
                if ee_pos[2] < 0.73:
                    p.removeConstraint(cid) # Release the "magnetic" grip
                    time.sleep(0.2) # Short pause for physics to settle
                    cube['sorted'] = True
                    current_idx += 1 # Move to the next cube in the list
                    state = "APPROACHING"

            # --- ROBOT ACTUATION ---
            # Calculate Inverse Kinematics (IK) to find the joint angles needed for 'target'
            joint_poses = p.calculateInverseKinematics(robot_id, 11, target)
            
            # Apply the calculated positions to the first 7 joints (arm)
            # maxVelocity=1.5 keeps the motion smooth and professional
            for i in range(7):
                p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joint_poses[i], force=200, maxVelocity=1.5)
        else:
            status = "STATUS: Sorting Complete!"

        # Update the OpenCV window with the current status message
        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("WPI RBE 526: Precise Sorter", frame)
        
        p.stepSimulation() # Advance the physics engine by one step
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    # Safe shutdown of camera and simulation
    cap.release()
    cv2.destroyAllWindows()
    p.disconnect()