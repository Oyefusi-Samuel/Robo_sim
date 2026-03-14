"""
supervisory_control_nqueens.py
==============================
N-Queens HRI — SUPERVISORY CONTROL MODE
The robot has a pre-planned 8-queens solution.
Before each move the operator sees the proposed placement on the CV overlay.
  • THUMBS UP  → approve  (robot executes move)
  • OPEN PALM  → reject / skip this move (it is deferred to the end)
  • Q key      → quit

Architecture
------------
  Human role : SUPERVISOR  – approves or vetoes robot intentions
  Robot role : EXECUTOR    – autonomously plans & carries out approved moves
"""

import pybullet as p
import pybullet_data
import cv2
import numpy as np
import mediapipe as mp
import math
import time
import sys

# ─────────────────────────────────────────────
#  WORLD HELPERS  (unchanged from your base)
# ─────────────────────────────────────────────

def create_table(position):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.75, 0.3, 0.02])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.75, 0.3, 0.02],
                              rgbaColor=[0.55, 0.27, 0.07, 1])
    return p.createMultiBody(0, col, vis, position)

def create_chess_board(board_pos):
    sqr = 0.06
    h   = 0.005
    mid = sqr * 4
    for r in range(8):
        for c in range(8):
            color = [1,1,1,1] if (r+c)%2==0 else [0.1,0.1,0.1,1]
            vis = p.createVisualShape(p.GEOM_BOX,
                                      halfExtents=[sqr/2, sqr/2, h/2],
                                      rgbaColor=color)
            col = p.createCollisionShape(p.GEOM_BOX,
                                         halfExtents=[sqr/2, sqr/2, h/2])
            x = r*sqr + board_pos[0] - (mid - sqr/2)
            y = c*sqr + board_pos[1] - (mid - sqr/2)
            p.createMultiBody(0, col, vis, [x, y, board_pos[2]+h/2])

def create_piece(position, color=[0.25, 0, 1, 1]):
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.022, height=0.06)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.022, length=0.06,
                              rgbaColor=color)
    return p.createMultiBody(0.1, col, vis, position)

def board_to_world(row, col, board_pos, sqr=0.06, n=8):
    mid = sqr * (n/2)
    x = row*sqr + board_pos[0] - (mid - sqr/2)
    y = col*sqr + board_pos[1] - (mid - sqr/2)
    return [x, y, board_pos[2] + 0.03]

# ─────────────────────────────────────────────
#  ROBOT HELPERS
# ─────────────────────────────────────────────

def ik_move(robot, pos):
    orn = p.getQuaternionFromEuler([math.pi, 0, 0])
    joints = p.calculateInverseKinematics(robot, 11, pos, orn)
    for i in range(7):
        p.setJointMotorControl2(robot, i, p.POSITION_CONTROL,
                                joints[i], force=700)

def step(n=50):
    for _ in range(n):
        p.stepSimulation()
        time.sleep(1/240)

def pick_and_place(robot, src, dst, gripper, piece_id):
    """Full pick-place sequence. piece_id is the exact body to move."""
    LIFT = 0.28

    # Move above src
    ik_move(robot, [src[0], src[1], src[2]+LIFT]);  step(60)
    # Descend to piece
    ik_move(robot, [src[0], src[1], src[2]+0.04]);  step(60)
    # Grasp only this piece
    gripper.grasp([piece_id]);                       step(30)
    if gripper.held is None:
        print(f"  [WARN] Grasp missed — EE may not have reached piece.")
    # Lift
    ik_move(robot, [src[0], src[1], src[2]+LIFT]);  step(60)

    # Swing over destination
    ik_move(robot, [dst[0], dst[1], dst[2]+LIFT]);  step(60)
    # Descend to place
    ik_move(robot, [dst[0], dst[1], dst[2]+0.04]);  step(60)
    # Release
    gripper.release();                               step(30)

    # Snap piece to exact grid position — eliminates physics drift
    p.resetBasePositionAndOrientation(piece_id, dst, [0, 0, 0, 1])
    p.resetBaseVelocity(piece_id, [0, 0, 0], [0, 0, 0])

    # Retreat
    ik_move(robot, [dst[0], dst[1], dst[2]+LIFT]);  step(60)

class Gripper:
    def __init__(self, robot, ee):
        self.robot = robot
        self.ee    = ee
        self.cid   = None
        self.held  = None
        self.last_released = None

    def grasp(self, objects, thresh=0.06):
        if self.cid: return
        ee_pos = p.getLinkState(self.robot, self.ee)[0]
        closest, best_d = None, thresh
        for obj in objects:
            pos, _ = p.getBasePositionAndOrientation(obj)
            d = math.dist(ee_pos, pos)
            if d < best_d:
                best_d, closest = d, obj
        if closest:
            self.cid  = p.createConstraint(self.robot, self.ee,
                                           closest, -1, p.JOINT_FIXED,
                                           [0,0,0],[0,0,0.04],[0,0,0])
            self.held = closest

    def release(self):
        if self.cid:
            p.removeConstraint(self.cid)
            self.cid  = None
            self.last_released = self.held
            self.held = None

# ─────────────────────────────────────────────
#  GESTURE DETECTION
# ─────────────────────────────────────────────

mp_hands = mp.solutions.hands
hand_det = mp_hands.Hands(min_detection_confidence=0.75, max_num_hands=1)
draw_u   = mp.solutions.drawing_utils

def detect_gesture(frame):
    """
    Returns  'approve'  (thumbs-up)
             'reject'   (open palm – all 4 fingertips above their PIPs)
             None
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hand_det.process(rgb)
    if not res.multi_hand_landmarks:
        return None

    lm = res.multi_hand_landmarks[0].landmark
    draw_u.draw_landmarks(frame, res.multi_hand_landmarks[0],
                          mp_hands.HAND_CONNECTIONS)

    # Thumbs-up: thumb tip (4) clearly above index MCP (5)
    if lm[4].y < lm[5].y - 0.04:
        return 'approve'

    # Open palm: all 4 fingertips above their PIP joints
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    if all(lm[t].y < lm[p].y for t, p in zip(tips, pips)):
        return 'reject'

    return None

# ─────────────────────────────────────────────
#  OVERLAY DRAWING
# ─────────────────────────────────────────────

COLORS_BGR = {
    'approve': (0, 220, 0),
    'reject':  (0, 0, 220),
    'wait':    (200, 200, 0),
}

def draw_overlay(frame, move_idx, total, src_rc, dst_rc, gesture, status):
    h, w = frame.shape[:2]

    # Dark semi-transparent banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Move info
    cv2.putText(frame,
                f"Move {move_idx+1}/{total}  |  Queen row {src_rc[0]} col {src_rc[1]}"
                f"  ->  row {dst_rc[0]} col {dst_rc[1]}",
                (12, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)

    # Gesture instruction
    cv2.putText(frame,
                "THUMBS UP = Approve    OPEN PALM = Skip    Q = Quit",
                (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Status pill
    color = COLORS_BGR.get(gesture if gesture else 'wait', COLORS_BGR['wait'])
    cv2.putText(frame, f"  {status}  ",
                (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    return frame

# ─────────────────────────────────────────────
#  PHYSICS INIT
# ─────────────────────────────────────────────

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)
p.loadURDF("plane.urdf")
p.resetDebugVisualizerCamera(1.8, 45, -35, [0, -0.5, 0.2])

table_pos = [0, -0.5, 0.15]
create_table(table_pos)
board_pos = [table_pos[0], table_pos[1], table_pos[2] + 0.02]
create_chess_board(board_pos)

# ─────────────────────────────────────────────
#  N-QUEENS SOLUTION  (pre-computed, row→col)
# ─────────────────────────────────────────────
SOLUTION = {0: 4, 1: 2, 2: 0, 3: 6, 4: 1, 5: 7, 6: 5, 7: 3}

# Spawn queens on left staging column (col = 0 for each row)
queen_ids = {}   # row → body_id
for row in range(8):
    pos = board_to_world(row, 0, board_pos)
    pos[2] += 0.01
    qid = create_piece(pos, color=[0.3, 0.0, 1.0, 1])
    queen_ids[row] = qid

# ─────────────────────────────────────────────
#  ROBOT & GRIPPER
# ─────────────────────────────────────────────

robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
p.setJointMotorControl2(robot, 9, p.POSITION_CONTROL, 0.04)
p.setJointMotorControl2(robot, 10, p.POSITION_CONTROL, 0.04)
gripper = Gripper(robot, 11)

# Settle
for _ in range(100):
    p.stepSimulation()
    time.sleep(1/240)

# ─────────────────────────────────────────────
#  SUPERVISORY CONTROL LOOP
# ─────────────────────────────────────────────

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[WARN] No webcam found — press SPACE to auto-approve each move.")

# Build move list: (queen_row, src_col=0, dst_col=SOLUTION[row])
moves = [(row, 0, SOLUTION[row]) for row in range(8)]
deferred = []   # rejected moves come back at the end
move_queue = moves[:]
done_rows  = set()

print("\n=== SUPERVISORY CONTROL — N-Queens ===")
print("THUMBS UP → approve   |   OPEN PALM → defer   |   Q → quit\n")

try:
    while move_queue:
        move_idx_global = 8 - len(move_queue)
        queen_row, src_col, dst_col = move_queue.pop(0)

        src_world = board_to_world(queen_row, src_col, board_pos)
        dst_world = board_to_world(queen_row, dst_col, board_pos)

        src_rc = (queen_row, src_col)
        dst_rc = (queen_row, dst_col)

        # ── Wait for gesture ──────────────────────────────────────────
        decision = None
        print(f"  Proposing: row {queen_row} col {src_col} → col {dst_col}")
        print(f"  [Show THUMBS UP to approve / OPEN PALM to defer]")

        while decision is None:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                gesture = detect_gesture(frame)

                if gesture == 'approve':
                    decision = 'approve'
                    status_txt = "APPROVED — Executing..."
                elif gesture == 'reject':
                    decision = 'reject'
                    status_txt = "DEFERRED — Skipping..."
                else:
                    status_txt = "Awaiting gesture..."

                frame = draw_overlay(frame, move_idx_global, 8,
                                     src_rc, dst_rc, gesture, status_txt)
                cv2.imshow("Supervisory Control — N-Queens", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                raise KeyboardInterrupt
            if key == ord(' '):       # fallback: spacebar = approve
                decision = 'approve'

            p.stepSimulation()
            time.sleep(1/240)

        # ── Execute or defer ─────────────────────────────────────────
        if decision == 'approve':
            print(f"  ✓ Executing move row {queen_row}: col {src_col} → {dst_col}")
            pick_and_place(robot, src_world, dst_world, gripper, piece_id=queen_ids[queen_row])
            done_rows.add(queen_row)
            print(f"  ✓ Done. {8 - len(move_queue)} / 8 queens placed.")
        else:
            print(f"  ⏭ Deferred move for row {queen_row}")
            deferred.append((queen_row, src_col, dst_col))

        # When primary queue empty, retry deferred once
        if not move_queue and deferred:
            print("\n--- Retrying deferred moves ---")
            move_queue = deferred[:]
            deferred   = []

    print("\n🎉 All 8 queens placed — puzzle solved!")

    # Keep alive
    while p.isConnected():
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "Puzzle Solved! Press Q to exit",
                        (10, 50), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,220,0), 2)
            cv2.imshow("Supervisory Control — N-Queens", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        p.stepSimulation()
        time.sleep(1/240)

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    p.disconnect()
    cap.release()
    cv2.destroyAllWindows()