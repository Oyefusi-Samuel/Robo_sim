"""
shared_control_nqueens.py
=========================
N-Queens HRI — SHARED CONTROL MODE
Human and robot divide cognitive labour:

  Robot's job  : constraint solver — computes which columns are SAFE
                 for the current row given already-placed queens,
                 highlights them, and proposes the best one.

  Human's job  : column selector — uses keyboard to choose among the
                 robot-validated options and confirms the move.

Keyboard map
────────────
  LEFT  / A    ← cycle valid columns backwards
  RIGHT / D    → cycle valid columns forwards
  ENTER / SPACE  confirm and execute highlighted column
  U              undo last placed queen (re-opens that row)
  R              reset entire board
  Q              quit

What makes it "shared":
  Neither party alone solves the puzzle.
  Human picks the column, robot enforces legality + executes manipulation.
"""

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
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.75, 0.3, 0.02])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.75, 0.3, 0.02],
                              rgbaColor=[0.55, 0.27, 0.07, 1])
    return p.createMultiBody(0, col, vis, position)

def create_chess_board(board_pos):
    sqr, h = 0.06, 0.005
    mid = sqr * 4
    for r in range(8):
        for c in range(8):
            color = [1,1,1,1] if (r+c)%2==0 else [0.1,0.1,0.1,1]
            vis = p.createVisualShape(p.GEOM_BOX,
                                      halfExtents=[sqr/2, sqr/2, h/2],
                                      rgbaColor=color)
            col_s = p.createCollisionShape(p.GEOM_BOX,
                                           halfExtents=[sqr/2, sqr/2, h/2])
            x = r*sqr + board_pos[0] - (mid - sqr/2)
            y = c*sqr + board_pos[1] - (mid - sqr/2)
            p.createMultiBody(0, col_s, vis, [x, y, board_pos[2]+h/2])

def create_piece(position, color):
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
#  CONSTRAINT SOLVER  (robot's contribution)
# ─────────────────────────────────────────────

def get_safe_columns(current_row, placed):
    """
    placed: dict {row: col} for already-confirmed queens.
    Returns sorted list of column indices safe for current_row.
    """
    safe = []
    for col in range(8):
        conflict = False
        for r, c in placed.items():
            if c == col:                      # same column
                conflict = True; break
            if abs(r - current_row) == abs(c - col):   # diagonal
                conflict = True; break
        if not conflict:
            safe.append(col)
    return safe

# ─────────────────────────────────────────────
#  ROBOT HELPERS
# ─────────────────────────────────────────────

def ik_move(robot, pos):
    orn    = p.getQuaternionFromEuler([math.pi, 0, 0])
    joints = p.calculateInverseKinematics(robot, 11, pos, orn)
    for i in range(7):
        p.setJointMotorControl2(robot, i, p.POSITION_CONTROL,
                                joints[i], force=700)

def step(n=50):
    for _ in range(n):
        p.stepSimulation()
        time.sleep(1/240)

class Gripper:
    def __init__(self, robot, ee):
        self.robot = robot; self.ee = ee
        self.cid = None;    self.held = None

    def grasp(self, objects, thresh=0.07):
        if self.cid: return
        ee_pos = p.getLinkState(self.robot, self.ee)[0]
        closest, best = None, thresh
        for obj in objects:
            pos, _ = p.getBasePositionAndOrientation(obj)
            d = math.dist(ee_pos, pos)
            if d < best: best, closest = d, obj
        if closest:
            self.cid  = p.createConstraint(self.robot, self.ee, closest, -1,
                                           p.JOINT_FIXED, [0,0,0],[0,0,0.04],[0,0,0])
            self.held = closest

    def release(self):
        if self.cid:
            p.removeConstraint(self.cid)
            self.cid = None; self.held = None

def execute_move(robot, gripper, src_world, dst_world, piece_id):
    """Pick-place with a guaranteed teleport snap at the end."""
    LIFT = 0.28
    ik_move(robot, [src_world[0], src_world[1], src_world[2]+LIFT]); step(60)
    ik_move(robot, [src_world[0], src_world[1], src_world[2]+0.04]); step(60)
    gripper.grasp([piece_id]); step(30)
    ik_move(robot, [src_world[0], src_world[1], src_world[2]+LIFT]); step(60)
    ik_move(robot, [dst_world[0], dst_world[1], dst_world[2]+LIFT]); step(60)
    ik_move(robot, [dst_world[0], dst_world[1], dst_world[2]+0.04]); step(60)
    gripper.release(); step(30)
    # Snap
    p.resetBasePositionAndOrientation(piece_id, dst_world, [0,0,0,1])
    p.resetBaseVelocity(piece_id, [0,0,0], [0,0,0])
    ik_move(robot, [dst_world[0], dst_world[1], dst_world[2]+LIFT]); step(60)

# ─────────────────────────────────────────────
#  CV OVERLAY  (shared control HUD)
# ─────────────────────────────────────────────

def draw_hud(frame, current_row, placed, safe_cols,
             cursor_idx, last_action, undo_available):
    h, w = frame.shape[:2]

    # Banner background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Title
    cv2.putText(frame,
                f"SHARED CONTROL  |  Placing queen for ROW {current_row}  "
                f"({len(placed)}/8 placed)",
                (10, 22), cv2.FONT_HERSHEY_DUPLEX, 0.62, (255,255,255), 1)

    # Safe columns bar
    bar_x = 10
    for i, col in enumerate(safe_cols):
        is_cursor = (i == cursor_idx)
        box_color = (0, 200, 80)  if is_cursor else (80, 80, 200)
        txt_color = (0,   0,  0)  if is_cursor else (230,230,230)
        x0, y0 = bar_x + i*52, 32
        cv2.rectangle(frame, (x0, y0), (x0+46, y0+28), box_color, -1)
        cv2.putText(frame, f"C{col}", (x0+8, y0+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, txt_color, 1)

    # Arrow indicator above cursor
    cx = bar_x + cursor_idx*52 + 23
    cv2.arrowedLine(frame, (cx, 30), (cx, 35), (0,220,80), 2, tipLength=0.5)

    # Control legend
    cv2.putText(frame,
                "< A/LEFT  cycle    D/RIGHT cycle >    ENTER/SPACE confirm    U undo    R reset    Q quit",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1)

    # Last action
    cv2.putText(frame, f"Last: {last_action}",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100,220,255), 1)

    # Already placed queens summary
    summary = "  ".join([f"R{r}→C{c}" for r, c in sorted(placed.items())])
    if summary:
        cv2.putText(frame, f"Placed: {summary}",
                    (10, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,100), 1)

    # No safe moves warning
    if not safe_cols:
        cv2.rectangle(frame, (0,0), (w,h), (0,0,180), 4)
        cv2.putText(frame, "NO SAFE COLUMNS — press U to undo",
                    (w//2 - 200, h//2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,50,255), 2)
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

# Spawn all 8 queens in a staging column off the left edge
STAGING_COL = -1    # one column to the left of the board (world space offset)
STAGING_SQR = 0.06

def staging_pos(row):
    mid = STAGING_SQR * 4
    x   = row * STAGING_SQR + board_pos[0] - (mid - STAGING_SQR/2)
    y   = board_pos[1] - (mid - STAGING_SQR/2) - STAGING_SQR * 1.5
    return [x, y, board_pos[2] + 0.03]

queen_ids   = {}   # row → pybullet body id
queen_pos   = {}   # row → current world pos (updated after each place)
for row in range(8):
    pos = staging_pos(row)
    qid = create_piece(pos, color=[0.25, 0.0, 1.0, 1.0])
    queen_ids[row]  = qid
    queen_pos[row]  = pos

# ─────────────────────────────────────────────
#  ROBOT & GRIPPER
# ─────────────────────────────────────────────

robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
p.setJointMotorControl2(robot, 9, p.POSITION_CONTROL, 0.04)
p.setJointMotorControl2(robot, 10, p.POSITION_CONTROL, 0.04)
gripper = Gripper(robot, 11)

for _ in range(120): p.stepSimulation(); time.sleep(1/240)

# ─────────────────────────────────────────────
#  SHARED CONTROL STATE
# ─────────────────────────────────────────────

placed        = {}          # {row: col} confirmed placements
current_row   = 0           # which row we're placing next
cursor_idx    = 0           # index into safe_cols list
last_action   = "Initialised — select column for Row 0"
history       = []          # stack of (row, col, src_world, dst_world) for undo

cap = cv2.VideoCapture(0)
cv2.namedWindow("Shared Control — N-Queens", cv2.WINDOW_NORMAL)

print("\n=== SHARED CONTROL — N-Queens ===")
print("LEFT/RIGHT: cycle columns  |  ENTER: confirm  |  U: undo  |  R: reset  |  Q: quit\n")

def reset_board():
    global placed, current_row, cursor_idx, last_action, history
    for row in range(8):
        pos = staging_pos(row)
        p.resetBasePositionAndOrientation(queen_ids[row], pos, [0,0,0,1])
        p.resetBaseVelocity(queen_ids[row], [0,0,0], [0,0,0])
        queen_pos[row] = pos
    placed       = {}
    current_row  = 0
    cursor_idx   = 0
    history      = []
    last_action  = "Board reset."
    print("  Board reset.")

try:
    while p.isConnected():

        # ── Robot constraint solver fires every frame ───────────────
        safe_cols = get_safe_columns(current_row, placed)

        # Clamp cursor to valid range
        if safe_cols:
            cursor_idx = cursor_idx % len(safe_cols)

        # ── Render HUD ───────────────────────────────────────────────
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = draw_hud(frame, current_row, placed, safe_cols,
                             cursor_idx, last_action,
                             undo_available=bool(history))
            cv2.imshow("Shared Control — N-Queens", frame)

        # ── Keyboard input ───────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key in (ord('a'), 81):    # LEFT / A  — cycle left
            if safe_cols:
                cursor_idx = (cursor_idx - 1) % len(safe_cols)
                last_action = f"Row {current_row}: cursor → col {safe_cols[cursor_idx]}"

        elif key in (ord('d'), 83):    # RIGHT / D — cycle right
            if safe_cols:
                cursor_idx = (cursor_idx + 1) % len(safe_cols)
                last_action = f"Row {current_row}: cursor → col {safe_cols[cursor_idx]}"

        elif key in (13, 32):          # ENTER or SPACE — confirm
            if not safe_cols:
                last_action = "No safe column — undo first!"
            elif current_row >= 8:
                last_action = "All queens placed!"
            else:
                chosen_col  = safe_cols[cursor_idx]
                src_world   = queen_pos[current_row][:]
                dst_world   = board_to_world(current_row, chosen_col, board_pos)

                print(f"  Human chose: row {current_row} → col {chosen_col}")
                last_action = f"Executing: row {current_row} → col {chosen_col} ..."
                # Quick HUD refresh before blocking move
                ret2, frame2 = cap.read()
                if ret2:
                    frame2 = cv2.flip(frame2, 1)
                    frame2 = draw_hud(frame2, current_row, placed, safe_cols,
                                      cursor_idx, last_action, bool(history))
                    cv2.imshow("Shared Control — N-Queens", frame2)
                    cv2.waitKey(1)

                execute_move(robot, gripper, src_world, dst_world,
                             queen_ids[current_row])
                queen_pos[current_row] = dst_world

                history.append((current_row, chosen_col, src_world, dst_world))
                placed[current_row] = chosen_col
                current_row += 1
                cursor_idx   = 0

                if current_row == 8:
                    last_action = "🎉 All 8 queens placed — N-Queens SOLVED!"
                    print("\n🎉 Puzzle solved!")
                else:
                    safe_next   = get_safe_columns(current_row, placed)
                    last_action = (f"Row {current_row-1} done. "
                                   f"Row {current_row}: {len(safe_next)} safe columns")

        elif key == ord('u'):          # UNDO last move
            if history:
                undo_row, undo_col, undo_src, undo_dst = history.pop()
                # Move piece back to staging
                p.resetBasePositionAndOrientation(queen_ids[undo_row],
                                                  undo_src, [0,0,0,1])
                p.resetBaseVelocity(queen_ids[undo_row], [0,0,0], [0,0,0])
                queen_pos[undo_row] = undo_src
                del placed[undo_row]
                current_row  = undo_row
                cursor_idx   = 0
                last_action  = f"Undid row {undo_row} (was col {undo_col})"
                print(f"  Undo: row {undo_row} back to staging")
            else:
                last_action = "Nothing to undo."

        elif key == ord('r'):          # RESET
            reset_board()

        p.stepSimulation()
        time.sleep(1/240)

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    p.disconnect()
    cap.release()
    cv2.destroyAllWindows()