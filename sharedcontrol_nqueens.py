"""
shared_control_nqueens_v2.py
============================
N-Queens SHARED CONTROL — Improved
  • Human : keyboard selects target column (LEFT/RIGHT + ENTER)
  • Robot  : computes legal moves, autonomously picks & places
  • Gripper: fingers visibly OPEN before pick, CLOSE around queen, OPEN on release
  • Queens : compound shape (wide base + tall body) — easy to grip visually
  • Ghost  : semi-transparent yellow preview queen tracks selection in real-time
"""

import pybullet as p
import pybullet_data
import time
import math
import sys

# ═══════════════════════════════════════════════════════
#  LAYOUT  — single source of truth for all positions
# ═══════════════════════════════════════════════════════
SQR     = 0.055   # chess square side (m)
BOARD_N = 8

# ── Table ───────────────────────────────────────────────
# Narrower in Y so arm can comfortably span it
TABLE_CX     = 0.02
TABLE_CY     = -0.48
TABLE_HALF_Z = 0.09
TABLE_HALF_X = 0.38
TABLE_HALF_Y = 0.38
TABLE_TOP_Z  = TABLE_HALF_Z * 2   # = 0.18

# ── Chess board on table top ─────────────────────────────
# Board centred at (0.02, -0.48), shifted back slightly
# y: -0.62 → -0.235   x: -0.175 → +0.210
BOARD_SQ_H  = 0.008
BOARD_Z     = TABLE_TOP_Z + BOARD_SQ_H / 2   # 0.184
BOARD_X0    = -0.175
BOARD_Y0    = -0.620   # board back edge; front edge = -0.620 + 7*0.055 = -0.235

# ── Queen geometry ───────────────────────────────────────
Q_RADIUS = 0.020
Q_HEIGHT = 0.068
Q_BASE_R = 0.028
Q_BASE_H = 0.012
# queen centre z = TABLE_TOP_Z + Q_BASE_H + Q_HEIGHT/2 = 0.226

# ── Staging: BETWEEN robot and board, on table surface ───
# Robot front at y=-0.10, board front at y=-0.235
# → staging at y=-0.17 (1 row in front of board)
STAGE_X0 = -0.175    # aligns with board column 0, spaced by SQR
STAGE_Y  = -0.165    # in front of board; robot reaches here easily

# ── Robot: ON TABLE SURFACE at front edge ────────────────
# Table front edge = TABLE_CY + TABLE_HALF_Y = -0.48 + 0.38 = -0.10
# Robot base ON table: z = TABLE_TOP_Z = 0.18
# Arm now reaches: staging at y=-0.165 (0.065m), far board at y=-0.62 (0.52m)
# Panda reach @ scale 0.85 ≈ 0.76m  →  0.52m is comfortably within reach
ROBOT_POS = [0.02, -0.10, TABLE_TOP_Z]
ROBOT_SC  = 0.85

# ── Motion heights ───────────────────────────────────────
LIFT_Z     = 0.52     # transit clearance above table (TABLE_TOP_Z=0.18 + 0.34 headroom)
PICK_Z_OFF = 0.095    # link-11 to fingertip at scale 0.85

# ── Gripper ──────────────────────────────────────────────
FINGER_OPEN  = 0.038
FINGER_CLOSE = 0.005


# ═══════════════════════════════════════════════════════
#  WORLD BUILDERS
# ═══════════════════════════════════════════════════════

def create_world():
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    # ── Table slab ──────────────────────────────────────────────────────────
    # Centre at (TABLE_CX, TABLE_CY, TABLE_HALF_Z) so top = TABLE_TOP_Z = 0.18
    t_ext = [TABLE_HALF_X, TABLE_HALF_Y, TABLE_HALF_Z]
    p.createMultiBody(
        0,
        p.createCollisionShape(p.GEOM_BOX, halfExtents=t_ext),
        p.createVisualShape(p.GEOM_BOX, halfExtents=t_ext,
                            rgbaColor=[0.45, 0.25, 0.07, 1]),
        [TABLE_CX, TABLE_CY, TABLE_HALF_Z]   # centre z = halfExt → base sits on floor
    )

    # ── Chess board squares ON TOP of table ─────────────────────────────────
    # BOARD_Z = TABLE_TOP_Z + BOARD_SQ_H/2 so squares sit above table surface
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            color = [0.95, 0.95, 0.95, 1] if (r+c) % 2 == 0 else [0.08, 0.08, 0.08, 1]
            half  = [SQR/2.05, SQR/2.05, BOARD_SQ_H/2]
            x = BOARD_X0 + r * SQR
            y = BOARD_Y0 + c * SQR
            p.createMultiBody(
                0,
                p.createCollisionShape(p.GEOM_BOX, halfExtents=half),
                p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color),
                [x, y, BOARD_Z]   # BOARD_Z already = TABLE_TOP_Z + h/2
            )

    # ── Row index dots along front edge of board (toward robot) ─────────────
    for i in range(BOARD_N):
        x = BOARD_X0 + i * SQR
        p.createMultiBody(
            0, -1,
            p.createVisualShape(p.GEOM_SPHERE, radius=0.005,
                                rgbaColor=[0.9, 0.6, 0.1, 1]),
            [x, BOARD_Y0 + BOARD_N * SQR + SQR * 0.3, TABLE_TOP_Z + 0.01]
        )


def sq_world(row, col):
    """Board (row, col) → world [x, y, z] at queen base centre."""
    return [
        BOARD_X0 + row * SQR,
        BOARD_Y0 + col * SQR,
        TABLE_TOP_Z + Q_BASE_H + Q_HEIGHT / 2  # queen centre on table surface
    ]


def stage_world(row):
    """8 queens in a row along +x behind the board (more negative y)."""
    return [STAGE_X0 + row * SQR, STAGE_Y, TABLE_TOP_Z + Q_BASE_H + Q_HEIGHT / 2]


# ═══════════════════════════════════════════════════════
#  QUEEN  — compound: wide base disc + tall cylinder body
# ═══════════════════════════════════════════════════════

def create_queen(pos, color):
    """
    Compound body: base disc at link -1, body cylinder as a visual offset.
    We use a single multibody with two visual shapes stacked.
    """
    # Base disc collision + visual
    base_col = p.createCollisionShape(p.GEOM_CYLINDER,
                                      radius=Q_BASE_R, height=Q_BASE_H)
    base_vis = p.createVisualShape(p.GEOM_CYLINDER,
                                   radius=Q_BASE_R, length=Q_BASE_H,
                                   rgbaColor=color,
                                   visualFramePosition=[0, 0, -Q_HEIGHT/2])

    # Body cylinder — taller, narrower
    body_col = p.createCollisionShape(p.GEOM_CYLINDER,
                                      radius=Q_RADIUS, height=Q_HEIGHT)
    body_vis = p.createVisualShape(p.GEOM_CYLINDER,
                                   radius=Q_RADIUS, length=Q_HEIGHT,
                                   rgbaColor=[min(color[0]+0.15,1),
                                              min(color[1]+0.15,1),
                                              min(color[2]+0.15,1), 1])

    # Crown sphere on top
    crown_vis = p.createVisualShape(p.GEOM_SPHERE,
                                    radius=Q_RADIUS * 1.3,
                                    rgbaColor=color,
                                    visualFramePosition=[0, 0, Q_HEIGHT/2])

    qid = p.createMultiBody(
        baseMass=0.08,
        baseCollisionShapeIndex=body_col,
        baseVisualShapeIndex=body_vis,
        basePosition=pos,
        linkMasses=[0.02],
        linkCollisionShapeIndices=[base_col],
        linkVisualShapeIndices=[base_vis],
        linkPositions=[[0, 0, -Q_HEIGHT/2]],
        linkOrientations=[[0, 0, 0, 1]],
        linkInertialFramePositions=[[0, 0, 0]],
        linkInertialFrameOrientations=[[0, 0, 0, 1]],
        linkParentIndices=[0],
        linkJointTypes=[p.JOINT_FIXED],
        linkJointAxis=[[0, 0, 1]]
    )
    # Crown as extra visual on base body
    p.createMultiBody(0, -1,
                      crown_vis,
                      [pos[0], pos[1], pos[2] + Q_HEIGHT/2])
    p.changeDynamics(qid, -1, linearDamping=0.95, angularDamping=0.95,
                     lateralFriction=1.2, restitution=0.0)
    return qid


def create_ghost():
    """Semi-transparent yellow preview piece."""
    body_vis  = p.createVisualShape(p.GEOM_CYLINDER,
                                    radius=Q_RADIUS, length=Q_HEIGHT,
                                    rgbaColor=[1, 0.9, 0, 0.45])
    base_vis  = p.createVisualShape(p.GEOM_CYLINDER,
                                    radius=Q_BASE_R, length=Q_BASE_H,
                                    rgbaColor=[1, 0.9, 0, 0.35],
                                    visualFramePosition=[0, 0, -Q_HEIGHT/2])
    crown_vis = p.createVisualShape(p.GEOM_SPHERE,
                                    radius=Q_RADIUS * 1.3,
                                    rgbaColor=[1, 1, 0, 0.4],
                                    visualFramePosition=[0, 0, Q_HEIGHT/2])
    ghost = p.createMultiBody(0, -1, body_vis, [0, 0, -2])
    # base & crown are separate static visuals that we'll teleport manually
    ghost_base  = p.createMultiBody(0, -1, base_vis,  [0, 0, -2])
    ghost_crown = p.createMultiBody(0, -1, crown_vis, [0, 0, -2])
    return ghost, ghost_base, ghost_crown


def move_ghost(ghost_ids, pos):
    g, gb, gc = ghost_ids
    p.resetBasePositionAndOrientation(g,  pos, [0,0,0,1])
    p.resetBasePositionAndOrientation(gb,
        [pos[0], pos[1], pos[2] - Q_HEIGHT/2], [0,0,0,1])
    p.resetBasePositionAndOrientation(gc,
        [pos[0], pos[1], pos[2] + Q_HEIGHT/2], [0,0,0,1])


def hide_ghost(ghost_ids):
    for g in ghost_ids:
        p.resetBasePositionAndOrientation(g, [0, 0, -2], [0,0,0,1])


# ═══════════════════════════════════════════════════════
#  ROBOT CONTROL
# ═══════════════════════════════════════════════════════

def set_fingers(robot, val, steps=20):
    """Animate finger joints 9 & 10 over `steps` sim steps."""
    cur9  = p.getJointState(robot, 9)[0]
    cur10 = p.getJointState(robot, 10)[0]
    for i in range(steps):
        t   = (i + 1) / steps
        v9  = cur9  + (val - cur9)  * t
        v10 = cur10 + (val - cur10) * t
        p.setJointMotorControl2(robot, 9,  p.POSITION_CONTROL, v9,  force=100)
        p.setJointMotorControl2(robot, 10, p.POSITION_CONTROL, v10, force=100)
        p.stepSimulation()
        time.sleep(1/240)


def ik_step(robot, target_pos, steps=70, force=600):
    """Drive EE to target_pos over `steps` sim steps (smooth interpolation)."""
    orn = p.getQuaternionFromEuler([math.pi, 0, math.pi/2])  # gripper down, fingers along board rows
    ee_start = p.getLinkState(robot, 11)[0]

    for i in range(steps):
        t   = (i + 1) / steps
        # Linear interpolation for smoother motion
        interp = [
            ee_start[j] + (target_pos[j] - ee_start[j]) * t
            for j in range(3)
        ]
        joints = p.calculateInverseKinematics(robot, 11, interp, orn,
                                              maxNumIterations=80,
                                              residualThreshold=1e-5)
        for j in range(7):
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL,
                                    joints[j], force=force)
        p.stepSimulation()
        time.sleep(1/240)


def pick_queen(robot, queen_id, src_pos):
    """
    Full pick sequence with visible gripper open → descend → close → lift.
    src_pos = world position of queen centre.
    """
    grip_z = src_pos[2] + PICK_Z_OFF     # EE hovers just above queen centre

    print(f"    → Opening gripper...")
    set_fingers(robot, FINGER_OPEN, steps=25)

    # Step 1 — move to a neutral high waypoint above staging zone centre
    # High neutral point above staging zone before descending
    staging_mid_x = STAGE_X0 + (BOARD_N / 2) * SQR
    print(f"    → Moving to high waypoint above staging...")
    ik_step(robot, [staging_mid_x, STAGE_Y, LIFT_Z], steps=55)

    # Step 2 — move laterally above the exact queen (still at LIFT_Z)
    print(f"    → Approaching above queen...")
    ik_step(robot, [src_pos[0], src_pos[1], LIFT_Z], steps=70)

    # Step 3 — descend straight down to grip height
    print(f"    → Descending to grip height...")
    ik_step(robot, [src_pos[0], src_pos[1], grip_z], steps=80)

    print(f"    → Closing gripper on queen...")
    set_fingers(robot, FINGER_CLOSE, steps=30)

    # Get exact EE position AFTER IK settled
    ee_pos = p.getLinkState(robot, 11)[0]

    # Snap queen to directly below EE (fingertip level)
    # EE z-axis points DOWN (orn=[π,0,π/2]), so fingertips = ee_pos[2] - PICK_Z_OFF
    grip_world = [ee_pos[0], ee_pos[1], ee_pos[2] - PICK_Z_OFF]
    p.resetBasePositionAndOrientation(queen_id, grip_world, [0, 0, 0, 1])
    p.resetBaseVelocity(queen_id, [0, 0, 0], [0, 0, 0])

    # Attach constraint — parentFramePos along EE local +z (= world -z at this orn)
    # so the queen hangs PICK_Z_OFF below link 11
    cid = p.createConstraint(
        robot, 11, queen_id, -1,
        p.JOINT_FIXED, [0, 0, 0],
        [0, 0, PICK_Z_OFF],   # in EE local frame: +z = downward = fingertip direction
        [0, 0, 0]             # at queen centre
    )
    # Increase constraint max force so it doesn't slip
    p.changeConstraint(cid, maxForce=500)

    print(f"    → Lifting with queen...")
    ik_step(robot, [src_pos[0], src_pos[1], LIFT_Z], steps=80)
    return cid


def place_queen(robot, queen_id, cid, dst_pos):
    """
    Full place sequence: transit → descend → open → snap → lift.
    dst_pos = world position of target queen centre.
    """
    place_z = dst_pos[2] + PICK_Z_OFF

    print(f"    → Transiting to target column...")
    ik_step(robot, [dst_pos[0], dst_pos[1], LIFT_Z], steps=90)

    print(f"    → Descending to place height...")
    ik_step(robot, [dst_pos[0], dst_pos[1], place_z], steps=80)

    print(f"    → Opening gripper to release...")
    set_fingers(robot, FINGER_OPEN, steps=25)

    p.removeConstraint(cid)

    # Snap queen to exact destination — eliminates any constraint drift
    p.resetBasePositionAndOrientation(queen_id, dst_pos, [0, 0, 0, 1])
    p.resetBaseVelocity(queen_id, [0, 0, 0], [0, 0, 0])

    # Run a few settle steps so queen appears to land naturally
    for _ in range(30):
        p.stepSimulation()
        time.sleep(1/240)

    print(f"    → Retracting arm...")
    ik_step(robot, [dst_pos[0], dst_pos[1], LIFT_Z], steps=60)

    # Partially close fingers back to neutral (looks natural)
    set_fingers(robot, FINGER_OPEN * 0.5, steps=15)


# ═══════════════════════════════════════════════════════
#  N-QUEENS SOLVER (constraint checker)
# ═══════════════════════════════════════════════════════

def get_safe_cols(row, placed):
    safe = []
    for col in range(BOARD_N):
        ok = True
        for r, c in placed.items():
            if c == col or abs(r - row) == abs(c - col):
                ok = False
                break
        if ok:
            safe.append(col)
    return safe


# ═══════════════════════════════════════════════════════
#  INIT
# ═══════════════════════════════════════════════════════

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setRealTimeSimulation(0)
p.resetDebugVisualizerCamera(
    cameraDistance=1.45,
    cameraYaw=35,
    cameraPitch=-28,
    cameraTargetPosition=[0.02, -0.38, 0.26]
)
# Hide default GUI panels for cleaner view
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

create_world()

# Robot — positioned behind and to the right, arm reaches across the board
robot = p.loadURDF("franka_panda/panda.urdf",
                   ROBOT_POS, useFixedBase=True, globalScaling=ROBOT_SC)

# Set a good initial arm pose so it starts out of the way
HOME = [0.0,  0.15, 0.0, -1.90, 0.0, 2.05, 0.80]   # arm up & back, EE high and clear
for i, ang in enumerate(HOME):
    p.resetJointState(robot, i, ang)
set_fingers(robot, FINGER_OPEN * 0.5)   # neutral half-open

# Queens — spawn in a neat staging column beside the board
QUEEN_COLOR = [0.15, 0.25, 0.90, 1]    # vivid blue
queens = {}
queen_pos_cache = {}
for row in range(BOARD_N):
    sp = stage_world(row)
    qid = create_queen(sp, QUEEN_COLOR)
    queens[row] = qid
    queen_pos_cache[row] = sp

# Settle staging queens
for _ in range(150):
    p.stepSimulation()
    time.sleep(1/240)
# Snap all to exact positions after settling
for row in range(BOARD_N):
    p.resetBasePositionAndOrientation(queens[row], queen_pos_cache[row], [0,0,0,1])
    p.resetBaseVelocity(queens[row], [0,0,0], [0,0,0])

# Ghost preview
ghost_ids = create_ghost()

# State
placed      = {}
current_row = 0
cursor_idx  = 0

# ═══════════════════════════════════════════════════════
#  HUD — printed once per state change, not every frame
# ═══════════════════════════════════════════════════════

def print_hud(row, safe_cols, cursor):
    col = safe_cols[cursor] if safe_cols else -1
    bar = ""
    for i, c in enumerate(safe_cols):
        bar += f"[►C{c}◄]" if i == cursor else f" C{c} "
    sys.stdout.write(
        f"\r  ROW {row}  |  {bar}  |  Selected → Col {col}   "
        f"  [{len(placed)}/8 placed]   "
    )
    sys.stdout.flush()


print("\n" + "═"*60)
print("  N-QUEENS  —  SHARED CONTROL  v2")
print("═"*60)
print("  ←/→  (A/D)   Cycle legal columns")
print("  ENTER/SPACE  Confirm — robot picks & places")
print("  Q             Quit")
print("═"*60 + "\n")

# ═══════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════
try:
    while p.isConnected() and current_row < BOARD_N:

        safe_cols = get_safe_cols(current_row, placed)

        if not safe_cols:
            print("\n\n  [DEADLOCK] No safe columns — this shouldn't happen with")
            print("  a valid N-Queens algorithm. Restart and try again.")
            break

        cursor_idx %= len(safe_cols)
        selected_col = safe_cols[cursor_idx]

        # Update ghost preview position
        preview_pos = sq_world(current_row, selected_col)
        move_ghost(ghost_ids, preview_pos)

        # Terminal HUD
        print_hud(current_row, safe_cols, cursor_idx)

        # ── Keyboard ─────────────────────────────────────────────────
        keys = p.getKeyboardEvents()

        # Cycle right: RIGHT arrow or D
        if (p.B3G_RIGHT_ARROW in keys and
                keys[p.B3G_RIGHT_ARROW] & p.KEY_WAS_TRIGGERED) or \
           (ord('d') in keys and keys[ord('d')] & p.KEY_WAS_TRIGGERED):
            cursor_idx += 1

        # Cycle left: LEFT arrow or A
        if (p.B3G_LEFT_ARROW in keys and
                keys[p.B3G_LEFT_ARROW] & p.KEY_WAS_TRIGGERED) or \
           (ord('a') in keys and keys[ord('a')] & p.KEY_WAS_TRIGGERED):
            cursor_idx -= 1

        # Quit
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            print("\n\n  Quit by user.")
            break

        # Confirm: ENTER or SPACE
        confirmed = (
            (p.B3G_RETURN in keys and keys[p.B3G_RETURN] & p.KEY_WAS_TRIGGERED) or
            (ord(' ')     in keys and keys[ord(' ')]     & p.KEY_WAS_TRIGGERED)
        )

        if confirmed:
            hide_ghost(ghost_ids)   # hide preview during execution

            src = queen_pos_cache[current_row]
            dst = sq_world(current_row, selected_col)

            print(f"\n\n  ► EXECUTING: Row {current_row} → Col {selected_col}")
            print(f"    src={[round(v,3) for v in src]}  dst={[round(v,3) for v in dst]}")

            # ── Robot pick & place ────────────────────────────────────
            cid = pick_queen(robot, queens[current_row], src)
            place_queen(robot, queens[current_row], cid, dst)

            queen_pos_cache[current_row] = dst
            placed[current_row] = selected_col
            current_row += 1
            cursor_idx  = 0

            summary = "  ".join([f"R{r}→C{c}" for r, c in sorted(placed.items())])
            print(f"    ✓ Placed. Board: {summary}")
            print(f"    {len(placed)}/8 queens on board.\n")

        p.stepSimulation()
        time.sleep(1/240)

    # ── Puzzle complete ───────────────────────────────────────────────
    if current_row == BOARD_N:
        hide_ghost(ghost_ids)
        print("\n\n" + "═"*60)
        print("    ALL 8 QUEENS PLACED — PUZZLE SOLVED!")
        solution = "  ".join([f"R{r}:C{c}" for r, c in sorted(placed.items())])
        print(f"  Solution: {solution}")
        print("═"*60)
        # Colour queens gold to celebrate
        for row, qid in queens.items():
            p.changeVisualShape(qid, -1, rgbaColor=[1, 0.8, 0, 1])

        print("  Simulation running — press Ctrl-C to exit.\n")
        while p.isConnected():
            p.stepSimulation()
            time.sleep(1/240)

except KeyboardInterrupt:
    print("\n\n  Stopped by user.")
except Exception as e:
    import traceback
    print(f"\n\n  [ERROR] {e}")
    traceback.print_exc()
finally:
    p.disconnect()
    print("  Disconnected.")