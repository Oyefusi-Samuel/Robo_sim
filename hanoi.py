import pybullet as p
import pybullet_data
<<<<<<< HEAD
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
=======
import math
import time

## WORLD PARAMS:
NUM_DISK = 3

## Helper Functions for World Gen
# Create Pegs for Puzzle
def create_peg(position):
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.02, height=0.2)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.02, length=0.2,
                              rgbaColor=[0.6, 0.3, 0.1, 1])
    peg = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=position
    )
    return peg

# Create board for puzzle
def create_board(position):
    halfExtents = [0.75, 0.3, 0.02]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=halfExtents,
                              rgbaColor=[0.6, 0.3, 0.1, 1])
    board = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=position
    )
    return board

# Generate full board (pegs + board)
def gen_board(position):
    board_pos = position
    board = create_board(board_pos)
    peg_positions = [[board_pos[0]-0.3, board_pos[1], board_pos[2]+0.1],
                     [board_pos[0], board_pos[1], board_pos[2]+0.1],
                     [board_pos[0]+0.3, board_pos[1], board_pos[2]+0.1]]
    pegs = [create_peg(pos) for pos in peg_positions]
    return board, pegs, peg_positions

# Generate a disk for puzzle (needs to have hole geometry to fit into puzzle)
def create_ring(radius_outer, radius_inner, height, position, color):
    num_segments = 16
    thickness = (radius_outer - radius_inner) / 2
    mid_radius = (radius_outer + radius_inner) / 2

    col_shapes = []
    col_positions = []
    col_orientations = []

    for i in range(num_segments):
        angle = 2 * math.pi * i / num_segments

        # segment center
        x = mid_radius * math.cos(angle)
        y = mid_radius * math.sin(angle)

        # rotate each box to follow the circle
        orn = p.getQuaternionFromEuler([0, 0, angle])

        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[thickness, thickness * 0.6, height/2]
        )

        col_shapes.append(col)
        col_positions.append([x, y, 0])
        col_orientations.append(orn)

    # visual (optional — cylinder is fine for looks)
    vis = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius_outer + 0.009,
        length=height,
        rgbaColor=color
    )

    disk = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=-1,  # IMPORTANT for compound
        baseVisualShapeIndex=vis,
        basePosition=position,
        linkMasses=[0.1]*num_segments,
        linkCollisionShapeIndices=col_shapes,
        linkVisualShapeIndices=[-1]*num_segments,
        linkPositions=col_positions,
        linkOrientations=col_orientations,
        linkInertialFramePositions=[[0,0,0]]*num_segments,
        linkInertialFrameOrientations=[[0,0,0,1]]*num_segments,
        linkParentIndices=[0]*num_segments,
        linkJointTypes=[p.JOINT_FIXED]*num_segments,
        linkJointAxis=[[0,0,1]]*num_segments
    )

    return disk

## Physics Environment init (plane + gravity)
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)
plane_id = p.loadURDF("plane.urdf")

## Load Board (Box + Pegs)
board_pos = [0, -0.5, 0.35]
board, pegs, peg_pos = gen_board(board_pos)

## Load disks
disks = []
for i in range(NUM_DISK):
    radius = 0.12 - i * 0.02
    height = 0.03
    z = height/2 + i * (height + 0.002)
    #disk = create_disk(radius, height, [peg_pos[0][0], peg_pos[0][1], peg_pos[0][1] + 1 + z] ,[1-i*0.2, 0.2, 0.8, 1])
    disk = create_ring(radius, 0.04, height, [peg_pos[0][0], peg_pos[0][1], peg_pos[0][1] + 1 + z] ,[1-i*0.2, 0.2, 0.8, 1])
    disks.append(disk)

## Load Robot into World
# Note:
# End effector --> 11
# Fingers -------> 9 and 10
# Arm -----------> 0 to 6
boxId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
t = 0
while p.isConnected:
    p.stepSimulation()
    time.sleep(1./240.)
    print(t)
    t += 1
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()
>>>>>>> f0338215151d7e29aa00b2145aed537ae1081295
