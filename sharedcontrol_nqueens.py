import pybullet as p
import pybullet_data
import cv2
import numpy as np
import math
import time
import sys

# ─────────────────────────────────────────────
#  WORLD HELPERS
# ─────────────────────────────────────────────

def create_table(position):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.75, 0.4, 0.02])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.75, 0.4, 0.02],
                              rgbaColor=[0.55, 0.27, 0.07, 1])
    return p.createMultiBody(0, col, vis, position)

def create_chess_board(board_pos):
    sqr, h = 0.06, 0.01
    mid = sqr * 4
    for r in range(8):
        for c in range(8):
            color = [0.9, 0.9, 0.9, 1] if (r+c)%2==0 else [0.1, 0.1, 0.1, 1]
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[sqr/2.1, sqr/2.1, h/2], rgbaColor=color)
            col_s = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sqr/2.1, sqr/2.1, h/2])
            x = r*sqr + board_pos[0] - (mid - sqr/2)
            y = c*sqr + board_pos[1] - (mid - sqr/2)
            p.createMultiBody(0, col_s, vis, [x, y, board_pos[2]+h/2])

def create_piece(position, color):
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.02, height=0.06)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.02, length=0.06, rgbaColor=color)
    body_id = p.createMultiBody(0.1, col, vis, position)
    # FIX: Add damping to prevent pieces from sliding/falling off
    p.changeDynamics(body_id, -1, linearDamping=0.5, angularDamping=0.5, frictionAnchor=True)
    return body_id

def board_to_world(row, col, board_pos, sqr=0.06, n=8):
    mid = sqr * (n/2)
    x = row*sqr + board_pos[0] - (mid - sqr/2)
    y = col*sqr + board_pos[1] - (mid - sqr/2)
    return [x, y, board_pos[2] + 0.05]

# ─────────────────────────────────────────────
#  ROBOT LOGIC
# ─────────────────────────────────────────────

class Gripper:
    def __init__(self, robot, ee):
        self.robot = robot; self.ee = ee
        self.cid = None

    def grasp(self, obj_id):
        if self.cid: return
        self.cid = p.createConstraint(self.robot, self.ee, obj_id, -1,
                                      p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0.03])

    def release(self):
        if self.cid:
            p.removeConstraint(self.cid)
            self.cid = None

def execute_move(robot, gripper, src, dst, piece_id):
    """Autonomous execution of the human's choice"""
    LIFT = 0.35
    # 1. Approach Source
    ik_move(robot, [src[0], src[1], LIFT]); step(40)
    ik_move(robot, [src[0], src[1], src[2]+0.01]); step(40)
    # 2. Pick
    gripper.grasp(piece_id); step(20)
    # 3. Move to Destination
    ik_move(robot, [src[0], src[1], LIFT]); step(40)
    ik_move(robot, [dst[0], dst[1], LIFT]); step(50)
    ik_move(robot, [dst[0], dst[1], dst[2]+0.01]); step(40)
    # 4. Release
    gripper.release(); step(20)
    ik_move(robot, [dst[0], dst[1], LIFT]); step(40)

def ik_move(robot, pos):
    orn = p.getQuaternionFromEuler([math.pi, 0, 0])
    joints = p.calculateInverseKinematics(robot, 11, pos, orn)
    for i in range(7):
        p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, joints[i], force=500)

def step(n=30):
    for _ in range(n):
        p.stepSimulation()
        time.sleep(1/240)

# ─────────────────────────────────────────────
#  STATE & HUD
# ─────────────────────────────────────────────

def get_safe_columns(current_row, placed):
    safe = []
    for col in range(8):
        conflict = False
        for r, c in placed.items():
            if c == col or abs(r - current_row) == abs(c - col):
                conflict = True; break
        if not conflict: safe.append(col)
    return safe

def draw_hud(frame, current_row, safe_cols, cursor_idx, last_action):
    cv2.rectangle(frame, (0,0), (frame.shape[1], 100), (30,30,30), -1)
    cv2.putText(frame, f"ROW: {current_row} | CHOOSE COLUMN", (20, 40), 2, 0.7, (255,255,255), 1)
    
    for i, c in enumerate(safe_cols):
        color = (0, 255, 0) if i == cursor_idx else (150, 150, 150)
        cv2.putText(frame, f"C{c}", (20 + i*60, 80), 2, 0.6, color, 2)
    
    cv2.putText(frame, f"Last: {last_action}", (frame.shape[1]-400, 40), 2, 0.5, (0, 255, 255), 1)
    return frame

# ─────────────────────────────────────────────
#  MAIN EXECUTION
# ─────────────────────────────────────────────

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")
p.resetDebugVisualizerCamera(1.5, 50, -35, [0, -0.4, 0.1])

table_pos = [0, -0.5, 0.15]
create_table(table_pos)
board_pos = [0, -0.5, 0.17]
create_chess_board(board_pos)

# Spawning Queens safely off-board
queen_ids = {}
queen_pos = {}
for i in range(8):
    pos = [-0.4, -0.2 - i*0.07, 0.25] # Staggered Y to avoid robot base
    qid = create_piece(pos, [0.2, 0.1, 0.8, 1])
    queen_ids[i] = qid
    queen_pos[i] = pos

robot = p.loadURDF("franka_panda/panda.urdf", [0.4, -0.5, 0.17], useFixedBase=True)
gripper = Gripper(robot, 11)

# State
placed = {}
current_row = 0
cursor_idx = 0
last_action = "System Ready"
history = []

cap = cv2.VideoCapture(0)

try:
    while p.isConnected() and current_row < 8:
        safe_cols = get_safe_columns(current_row, placed)
        if safe_cols: cursor_idx %= len(safe_cols)

        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = draw_hud(frame, current_row, safe_cols, cursor_idx, last_action)
            cv2.imshow("N-Queens Shared Control", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('a'): cursor_idx -= 1
        elif key == ord('d'): cursor_idx += 1
        elif key in (13, 32) and safe_cols: # ENTER / SPACE
            chosen_col = safe_cols[cursor_idx]
            dst = board_to_world(current_row, chosen_col, board_pos)
            src = p.getBasePositionAndOrientation(queen_ids[current_row])[0]
            
            last_action = f"Robot placing Queen {current_row} at Col {chosen_col}"
            execute_move(robot, gripper, src, dst, queen_ids[current_row])
            
            history.append((current_row, chosen_col, src))
            placed[current_row] = chosen_col
            current_row += 1
            cursor_idx = 0

        p.stepSimulation()
        time.sleep(1/240)

except Exception as e:
    print(f"Error: {e}")
finally:
    p.disconnect()
    cap.release()
    cv2.destroyAllWindows()