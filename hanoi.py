import pybullet as p
import pybullet_data
import cv2
import numpy as np
import time
import math
import sys
import mediapipe as mp

# --- 1. SETUP ---
print("Connecting to Physics Server...")
cid_gui = p.connect(p.GUI)
if cid_gui < 0:
    print("Failed to open GUI.")
    sys.exit()

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)  # Manual stepping for stability

p.resetDebugVisualizerCamera(1.4, 60, -30, [0.5, -0.3, 0.20])

# --- 2. WORLD SETUP ---
def create_world():
    board_pos = [0.5, -0.3, 0.02]

    # Ground plane
    p.loadURDF("plane.urdf")

    # Board
    b_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.15, 0.02])
    b_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.15, 0.02], rgbaColor=[0.1, 0.5, 0.1, 1])
    p.createMultiBody(0, b_col, b_vis, board_pos)

    # Pegs — tall enough to hold 3 thick disks (height 0.50 total)
    peg_positions = [
        [board_pos[0] - 0.25, board_pos[1], 0.29],
        [board_pos[0],        board_pos[1], 0.29],
        [board_pos[0] + 0.25, board_pos[1], 0.29],
    ]
    peg_ids = []
    for pos in peg_positions:
        peg_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.014, height=0.50)
        peg_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.014, length=0.50,
                                      rgbaColor=[0.95, 0.85, 0.5, 1])  # golden pegs
        pid = p.createMultiBody(0, peg_col, peg_vis, pos)
        peg_ids.append(pid)
    return board_pos, peg_positions, peg_ids

board_pos, peg_coords, peg_ids = create_world()

# --- 3. DISK CREATION ---
# KEY FIX: Disable collision between disks and pegs so disks slide freely on pegs
# Disks rest on each other / board via gravity — pegs only guide them visually.
DISK_RADII   = [0.130, 0.090, 0.052]   # large / medium / small — wide spread
DISK_HEIGHT  = 0.052                    # thick so each layer is clearly visible
DISK_GAP     = 0.006                    # air gap between stacked disks
# High-contrast saturated colors: orange / cyan / yellow
DISK_COLORS  = [[1.0,  0.35, 0.0,  1],
                [0.0,  0.85, 0.95, 1],
                [0.95, 0.95, 0.0,  1]]

# Board top surface z
BOARD_TOP_Z = board_pos[2] + 0.02  # = 0.04

def disk_rest_z(peg_idx, stack_pos):
    """Return the z center for a disk at stack_pos (0=bottom), with a visible gap."""
    return BOARD_TOP_Z + DISK_HEIGHT / 2 + stack_pos * (DISK_HEIGHT + DISK_GAP)

disks = []
for i in range(3):
    d_col = p.createCollisionShape(p.GEOM_CYLINDER,
                                   radius=DISK_RADII[i], height=DISK_HEIGHT)
    d_vis = p.createVisualShape(p.GEOM_CYLINDER,
                                radius=DISK_RADII[i], length=DISK_HEIGHT,
                                rgbaColor=DISK_COLORS[i])
    spawn_z = disk_rest_z(0, 2 - i)  # large disk at bottom (stack pos 0), small at top
    disk_pos = [peg_coords[0][0], peg_coords[0][1], spawn_z]
    d_id = p.createMultiBody(0.15, d_col, d_vis, disk_pos)
    p.changeDynamics(d_id, -1, linearDamping=0.9, angularDamping=0.9,
                     lateralFriction=1.0, restitution=0.0)
    disks.append(d_id)

# Disable collision between every disk and every peg
for d_id in disks:
    for pid in peg_ids:
        p.setCollisionFilterPair(d_id, pid, -1, -1, enableCollision=0)

# Also disable disk–disk collision to avoid stacking instability
for i in range(len(disks)):
    for j in range(i + 1, len(disks)):
        p.setCollisionFilterPair(disks[i], disks[j], -1, -1, enableCollision=0)

# --- 4. SETTLE SIMULATION ---
print("Settling disks...")
for _ in range(300):
    p.stepSimulation()

# Now freeze disks in place by teleporting them to exact rest positions
for i, d_id in enumerate(disks):
    exact_z = disk_rest_z(0, 2 - i)
    p.resetBasePositionAndOrientation(
        d_id,
        [peg_coords[0][0], peg_coords[0][1], exact_z],
        [0, 0, 0, 1]
    )
    p.resetBaseVelocity(d_id, [0, 0, 0], [0, 0, 0])

print("Disks settled and locked.")

# --- 5. ROBOT LOAD ---
print("Loading Panda URDF...")
try:
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0],
                          useFixedBase=True, globalScaling=0.8)
except Exception as e:
    print(f"URDF Error: {e}")
    sys.exit()

# Home position — drive joints to a neutral pose
HOME_JOINTS = [0, -0.5, 0, -2.0, 0, 1.5, 0.8]
for i, angle in enumerate(HOME_JOINTS):
    p.resetJointState(robot_id, i, angle)

# --- 6. HELPER: MOVE ROBOT SMOOTHLY ---
def move_robot_to(target_xyz, steps=80):
    """Drive robot EE toward target_xyz over `steps` sim steps."""
    for _ in range(steps):
        joint_poses = p.calculateInverseKinematics(robot_id, 11, target_xyz)
        for i in range(7):
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL,
                                    joint_poses[i], force=240)
        p.stepSimulation()
        time.sleep(1. / 240.)

# --- 7. HRI CAMERA ---
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)

def detect_thumbs_up(frame):
    """Return True if a thumbs-up gesture is detected."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark
        # Thumb tip (4) above index MCP (5) in image coords (y flipped)
        if lm[4].y < lm[5].y:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            return True
    return False

def wait_for_gesture(prompt="Show thumbs-up to continue"):
    """Block until thumbs-up detected or 'q' pressed."""
    print(f"  [{prompt}]")
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            approved = detect_thumbs_up(frame)
            cv2.putText(frame, prompt, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("HRI Hanoi", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False
            if approved:
                return True
        p.stepSimulation()
        time.sleep(1. / 240.)

# --- 8. DISK PICK & PLACE ---
# Track which disks are on which peg: peg_stacks[peg] = [disk_ids, bottom to top]
peg_stacks = {0: [disks[0], disks[1], disks[2]], 1: [], 2: []}
# disks[0]=large(red), disks[1]=medium(green), disks[2]=small(blue)

LIFT_Z   = 0.55
HOVER_Z  = 0.40
held_disk = None
constraint_id = -1

def pick_disk(src_peg):
    global held_disk, constraint_id
    disk = peg_stacks[src_peg][-1]  # top disk
    stack_pos = len(peg_stacks[src_peg]) - 1
    disk_z = disk_rest_z(src_peg, stack_pos)
    px, py = peg_coords[src_peg][0], peg_coords[src_peg][1]

    # Approach above
    move_robot_to([px, py, HOVER_Z])
    # Descend to disk
    move_robot_to([px, py, disk_z + 0.05], steps=60)
    # Attach
    constraint_id = p.createConstraint(
        robot_id, 11, disk, -1,
        p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.05], [0, 0, 0])
    held_disk = disk
    peg_stacks[src_peg].pop()
    # Lift
    move_robot_to([px, py, LIFT_Z], steps=60)

def place_disk(dst_peg):
    global held_disk, constraint_id
    stack_pos = len(peg_stacks[dst_peg])  # next position
    disk_z = disk_rest_z(dst_peg, stack_pos)
    px, py = peg_coords[dst_peg][0], peg_coords[dst_peg][1]

    # Move over destination
    move_robot_to([px, py, LIFT_Z])
    # Descend
    move_robot_to([px, py, disk_z + 0.05], steps=60)
    # Release
    p.removeConstraint(constraint_id)
    constraint_id = -1
    # Teleport disk to exact rest position
    p.resetBasePositionAndOrientation(
        held_disk,
        [px, py, disk_z],
        [0, 0, 0, 1]
    )
    p.resetBaseVelocity(held_disk, [0, 0, 0], [0, 0, 0])
    peg_stacks[dst_peg].append(held_disk)
    held_disk = None
    move_robot_to([px, py, HOVER_Z], steps=40)

# --- 9. HANOI MOVES ---
# 7-move solution for 3 disks: (src, dst)
moves = [(0, 2), (0, 1), (2, 1), (0, 2), (1, 0), (1, 2), (0, 2)]

print("\n=== Tower of Hanoi Simulation ===")
print("Show THUMBS UP to approve each move.\nPress 'q' to quit.\n")

try:
    for move_idx, (src, dst) in enumerate(moves):
        top_disk = peg_stacks[src][-1] if peg_stacks[src] else None
        print(f"Move {move_idx+1}/{len(moves)}: Peg {src+1} → Peg {dst+1}")

        if not wait_for_gesture(f"Move {move_idx+1}: Peg {src+1}→{dst+1} | Thumbs Up!"):
            print("Quit signal received.")
            break

        pick_disk(src)
        place_disk(dst)
        print(f"  ✓ Done. Stacks: { {k: len(v) for k,v in peg_stacks.items()} }")

        # Run a few settle steps
        for _ in range(120):
            p.stepSimulation()
            time.sleep(1. / 240.)

    print("\n🎉 Tower of Hanoi Complete!")
    # Keep window open
    while p.isConnected():
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "Puzzle Complete! Press Q to exit",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
            cv2.imshow("HRI Hanoi", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        p.stepSimulation()
        time.sleep(1. / 240.)

except KeyboardInterrupt:
    print("Simulation stopped by user.")
finally:
    p.disconnect()
    cap.release()
    cv2.destroyAllWindows()