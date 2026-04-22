"""
shared_control_nqueens_v2.py  — FINAL CLEAN VERSION
N-Queens SHARED CONTROL
  Human  : arrow keys select column, ENTER confirms
  Robot  : computes legal moves, picks & places autonomously
  Board  : chess-style GREEN/RED/ORANGE cell highlights
  HUD    : clean 5-line corner panel (no floating 3D clutter)
"""

import pybullet as p
import pybullet_data
import time, math, sys
import enum

# ═══════════════════════════════════════════════════════
#  LAYOUT
# ═══════════════════════════════════════════════════════
SQR      = 0.055
BOARD_N  = 8

TABLE_CX     =  0.02
TABLE_CY     = -0.48
TABLE_HALF_X =  0.38
TABLE_HALF_Y =  0.38
TABLE_HALF_Z =  0.09
TABLE_TOP_Z  =  TABLE_HALF_Z * 2          # 0.18

BOARD_SQ_H   =  0.008
BOARD_Z      =  TABLE_TOP_Z + BOARD_SQ_H / 2   # 0.184
BOARD_X0     = -0.175
BOARD_Y0     = -0.620

Q_RADIUS = 0.020
Q_HEIGHT = 0.055
Q_BASE_R = 0.026
Q_BASE_H = 0.010

STAGE_X0 = -0.175
STAGE_Y  = -0.165

ROBOT_POS = [0.02, -0.10, TABLE_TOP_Z]
ROBOT_SC  =  0.85

LIFT_Z      = 0.52
PICK_Z_OFF  = 0.095

FINGER_OPEN  = 0.038
FINGER_CLOSE = 0.005

class MODE(enum.Enum):
    L1 = 1 # Only Yellow Ghost
    L2 = 2 # Grid with just Green and Red
    L3 = 3 # Grid with Green, Yellow, and Red
    L4 = 4 # Blue ghost suggestion, unreliable
    L5 = 5 # Blue ghots suggestion, fully reliable


# ═══════════════════════════════════════════════════════
#  WORLD
# ═══════════════════════════════════════════════════════

def create_world():
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    t = [TABLE_HALF_X, TABLE_HALF_Y, TABLE_HALF_Z]
    p.createMultiBody(0,
        p.createCollisionShape(p.GEOM_BOX, halfExtents=t),
        p.createVisualShape(p.GEOM_BOX, halfExtents=t, rgbaColor=[0.45,0.25,0.07,1]),
        [TABLE_CX, TABLE_CY, TABLE_HALF_Z])
    h = BOARD_SQ_H / 2
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            col = [0.95,0.95,0.95,1] if (r+c)%2==0 else [0.08,0.08,0.08,1]
            half = [SQR/2.05, SQR/2.05, h]
            x = BOARD_X0 + r*SQR
            y = BOARD_Y0 + c*SQR
            p.createMultiBody(0,
                p.createCollisionShape(p.GEOM_BOX, halfExtents=half),
                p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=col),
                [x, y, BOARD_Z])

def sq_world(row, col):
    return [BOARD_X0 + row*SQR,
            BOARD_Y0 + col*SQR,
            TABLE_TOP_Z + Q_BASE_H + Q_HEIGHT/2]

def stage_world(row):
    return [STAGE_X0 + row*SQR, STAGE_Y, TABLE_TOP_Z + Q_BASE_H + Q_HEIGHT/2]

# ═══════════════════════════════════════════════════════
#  QUEEN  — simple two-body stack (no compound links)
# ═══════════════════════════════════════════════════════

def create_queen(pos, color):
    # Body cylinder
    qid = p.createMultiBody(0.08,
        p.createCollisionShape(p.GEOM_CYLINDER, radius=Q_RADIUS, height=Q_HEIGHT),
        p.createVisualShape(p.GEOM_CYLINDER, radius=Q_RADIUS, length=Q_HEIGHT,
                            rgbaColor=color),
        pos)
    p.changeDynamics(qid, -1, linearDamping=0.95, angularDamping=0.95,
                     lateralFriction=1.2, restitution=0.0)
    # Base disc — static visual only
    p.createMultiBody(0, -1,
        p.createVisualShape(p.GEOM_CYLINDER, radius=Q_BASE_R, length=Q_BASE_H,
                            rgbaColor=[min(color[0]+0.1,1), min(color[1]+0.1,1),
                                       min(color[2]+0.1,1), 1]),
        [pos[0], pos[1], TABLE_TOP_Z + Q_BASE_H/2])
    # Crown sphere
    p.createMultiBody(0, -1,
        p.createVisualShape(p.GEOM_SPHERE, radius=Q_RADIUS*1.25,
                            rgbaColor=color),
        [pos[0], pos[1], pos[2] + Q_HEIGHT/2])
    return qid

def create_ghost(color=[1,0.9,0,0.45]):
    gid = p.createMultiBody(0, -1,
        p.createVisualShape(p.GEOM_CYLINDER, radius=Q_RADIUS, length=Q_HEIGHT,
                            rgbaColor=color),
        [0, 0, -5])
    return gid

def move_ghost(gid, pos):
    p.resetBasePositionAndOrientation(gid, pos, [0,0,0,1])

def hide_ghost(gid):
    p.resetBasePositionAndOrientation(gid, [0,0,-5], [0,0,0,1])

# ═══════════════════════════════════════════════════════
#  CHESS-STYLE BOARD HIGHLIGHTS
# ═══════════════════════════════════════════════════════

_hl = {}   # (row,col) → body id

def init_highlights():
    half = [SQR*0.46, SQR*0.46, 0.0008]
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                      rgbaColor=[0,0,0,0])
            bid = p.createMultiBody(0, -1, vis, [0,0,-5])
            _hl[(r,c)] = bid

def _sq_z():
    return TABLE_TOP_Z + BOARD_SQ_H + 0.001

def _show(r, c, rgba):
    bid = _hl.get((r,c))
    if bid is None: return
    p.resetBasePositionAndOrientation(
        bid, [BOARD_X0+r*SQR, BOARD_Y0+c*SQR, _sq_z()], [0,0,0,1])
    p.changeVisualShape(bid, -1, rgbaColor=rgba)

def _hide(r, c):
    bid = _hl.get((r,c))
    if bid is None: return
    p.resetBasePositionAndOrientation(bid, [0,0,-5], [0,0,0,1])
    p.changeVisualShape(bid, -1, rgbaColor=[0,0,0,0])

def update_highlights_2d(placed, failure_threshold=0.0, mode=MODE.L5):
    if (mode != MODE.L2) and (mode != MODE.L3):
        suggestions = get_suggestions(placed, failure_thresh=failure_threshold)
    else:
        suggestions = None

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            #if (r,c) in placed:
            #    continue
            #if (r, c) in suggestions:
            #    _show(r, c, [0.2, 0.6, 1.0, 0.75])   # blue = solution path

            if suggestions:
                best_r, best_c = suggestions[0]
                move_ghost(ghost2, sq_world(best_r, best_c))
            else:
                hide_ghost(ghost2)

            if is_safe(r, c, placed) and r not in placed:
                if mode != MODE.L2:
                    if _deadlock(r, c, placed):
                        _show(r, c, [1.0, 0.55, 0.0, 0.72])   # orange = risky
                    else:
                        _show(r, c, [0.05, 0.90, 0.15, 0.68]) # green  = safe
                else:
                    _show(r, c, [0.05, 0.90, 0.15, 0.68]) # green  = safe
            else:
                _show(r, c, [0.92, 0.08, 0.08, 0.62])     # red    = illegal

def update_highlights(cur_row, safe_cols, placed):
    """Color all cells of cur_row: green/orange/red."""
    for col in range(BOARD_N):
        if col in safe_cols:
            if _deadlock(cur_row, col, placed):
                _show(cur_row, col, [1.0, 0.55, 0.0, 0.72])   # orange = risky
            else:
                _show(cur_row, col, [0.05, 0.90, 0.15, 0.68]) # green  = safe
        else:
            _show(cur_row, col, [0.92, 0.08, 0.08, 0.62])     # red    = illegal

def clear_row(row):
    for c in range(BOARD_N): _hide(row, c)

def clear_all_hl():
    for r in range(BOARD_N):
        for c in range(BOARD_N): _hide(r,c)

# Placed dot pool
_dots = {}

def add_dot(row, col):
    if row in _dots:
        try: p.removeBody(_dots[row])
        except: pass
    half = [SQR*0.18, SQR*0.18, 0.002]
    vis  = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.1,1.0,0.3,0.9])
    _dots[row] = p.createMultiBody(0,-1,vis,
        [BOARD_X0+row*SQR, BOARD_Y0+col*SQR, _sq_z()+0.002])

def clear_dots():
    for bid in _dots.values():
        try: p.removeBody(bid)
        except: pass
    _dots.clear()

def flash_cell(row, col):
    bid = _hl.get((row,col))
    if bid is None: return
    for _ in range(3):
        p.changeVisualShape(bid,-1,rgbaColor=[1,1,1,0.95])
        for _ in range(6): p.stepSimulation(); time.sleep(1/240)
        p.changeVisualShape(bid,-1,rgbaColor=[0.05,0.90,0.15,0.68])
        for _ in range(6): p.stepSimulation(); time.sleep(1/240)

# ═══════════════════════════════════════════════════════
#  HUD — fixed 5-line panel anchored above board
# ═══════════════════════════════════════════════════════

_hud = {}

def _txt(key, text, pos, rgb, sz=1.0):
    if key in _hud:
        try: p.removeUserDebugItem(_hud[key])
        except: pass
    _hud[key] = p.addUserDebugText(text, pos, textColorRGB=rgb,
                                    textSize=sz, lifeTime=0)

def draw_hud(cur_row, n_placed, status):
    # Anchored above top-left of board
    ax = BOARD_X0 - 0.01
    ay = BOARD_Y0 + (BOARD_N-1)*SQR + 0.05
    az = TABLE_TOP_Z + 0.31
    lh = 0.058
    _txt("h0","N-QUEENS  |  SHARED CONTROL",
         [ax,ay,az+lh*4],[1.0,0.85,0.0],1.2)
    _txt("h1",f"Row {cur_row}/8    Placed: {n_placed}/8",
         [ax,ay,az+lh*3],[0.9,0.9,0.9],1.0)
    _txt("h2","GREEN=safe   ORANGE=risky   RED=illegal",
         [ax,ay,az+lh*2],[0.55,0.95,0.55],0.85)
    _txt("h3","ARROW KEYS : move ghost    ENTER : place",
         [ax,ay,az+lh*1],[0.65,0.82,1.0],0.85)
    _txt("h4","U : undo    R : reset    Q : quit",
         [ax,ay,az+lh*0],[0.65,0.82,1.0],0.85)
    sc = [1.0,0.35,0.35] if any(w in status for w in ["RISKY","DEAD","block"]) \
         else [0.35,1.0,0.45]
    _txt("h5", status, [ax,ay,az-lh*1.1], sc, 0.90)

def clear_hud():
    for v in _hud.values():
        try: p.removeUserDebugItem(v)
        except: pass
    _hud.clear()

# ═══════════════════════════════════════════════════════
#  N-QUEENS LOGIC
# ═══════════════════════════════════════════════════════

def is_safe(row, col, placed):
    return all(
        c != col and abs(r - row) != abs(c - col)
        for r, c in placed.items()
    )

def get_safe(row, placed):
    safe = []
    for col in range(BOARD_N):
        if all(c!=col and abs(r-row)!=abs(c-col) for r,c in placed.items()):
            safe.append(col)
    return safe

def is_selectable(r, c, placed):
    if r in placed:
        return False
    if c in placed.values():
        return False
    return True

def _deadlock(row, col, placed):
    trial = dict(placed); trial[row] = col
    return any(not get_safe(fr, trial) for fr in range(row+1, BOARD_N))

# ═══════════════════════════════════════════════════════
#  ROBOT CONTROL
# ═══════════════════════════════════════════════════════

def set_fingers(robot, val, steps=20):
    c9  = p.getJointState(robot,9)[0]
    c10 = p.getJointState(robot,10)[0]
    for i in range(steps):
        t = (i+1)/steps
        p.setJointMotorControl2(robot,9, p.POSITION_CONTROL,c9 +(val-c9 )*t,force=80)
        p.setJointMotorControl2(robot,10,p.POSITION_CONTROL,c10+(val-c10)*t,force=80)
        p.stepSimulation(); time.sleep(1/240)

def ik_go(robot, tgt, steps=70):
    orn   = p.getQuaternionFromEuler([math.pi, 0, math.pi/2])
    start = list(p.getLinkState(robot,11)[0])
    for i in range(steps):
        t = (i+1)/steps
        interp = [start[j]+(tgt[j]-start[j])*t for j in range(3)]
        jt = p.calculateInverseKinematics(robot,11,interp,orn,
                                          maxNumIterations=100,
                                          residualThreshold=1e-5)
        for j in range(7):
            p.setJointMotorControl2(robot,j,p.POSITION_CONTROL,jt[j],force=600)
        p.stepSimulation(); time.sleep(1/240)

def pick(robot, qid, src):
    gz = src[2] + PICK_Z_OFF
    set_fingers(robot, FINGER_OPEN, 20)
    ik_go(robot, [src[0], src[1], LIFT_Z], 70)
    ik_go(robot, [src[0], src[1], gz],     80)
    set_fingers(robot, FINGER_CLOSE, 25)
    ee = p.getLinkState(robot,11)[0]
    snap = [ee[0], ee[1], ee[2]-PICK_Z_OFF]
    p.resetBasePositionAndOrientation(qid, snap, [0,0,0,1])
    p.resetBaseVelocity(qid,[0,0,0],[0,0,0])
    cid = p.createConstraint(robot,11,qid,-1,p.JOINT_FIXED,
                             [0,0,0],[0,0,PICK_Z_OFF],[0,0,0])
    p.changeConstraint(cid, maxForce=500)
    ik_go(robot, [src[0], src[1], LIFT_Z], 70)
    return cid

def place(robot, qid, cid, dst):
    pz = dst[2] + PICK_Z_OFF
    ik_go(robot, [dst[0], dst[1], LIFT_Z], 80)
    ik_go(robot, [dst[0], dst[1], pz],     80)
    set_fingers(robot, FINGER_OPEN, 20)
    p.removeConstraint(cid)
    p.resetBasePositionAndOrientation(qid, dst, [0,0,0,1])
    p.resetBaseVelocity(qid,[0,0,0],[0,0,0])
    for _ in range(25): p.stepSimulation(); time.sleep(1/240)
    ik_go(robot, [dst[0], dst[1], LIFT_Z], 55)
    set_fingers(robot, FINGER_OPEN*0.5, 10)

# ═══════════════════════════════════════════════════════
#  INIT
# ═══════════════════════════════════════════════════════

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setRealTimeSimulation(0)
p.resetDebugVisualizerCamera(1.11, 37.8, -48.4, [-0.12, -0.19, -0.08])  #(1.45, 35, -28, [0.02,-0.38,0.26])
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

create_world()
init_highlights()   # must be after p.connect

robot = p.loadURDF("franka_panda/panda.urdf",
                   ROBOT_POS, useFixedBase=True, globalScaling=ROBOT_SC)
HOME = [0.0, 0.15, 0.0, -1.90, 0.0, 2.05, 0.80]
for i,a in enumerate(HOME): p.resetJointState(robot,i,a)
set_fingers(robot, FINGER_OPEN*0.5)

QUEEN_COLOR = [0.15, 0.25, 0.90, 1]
queens        = {}
q_pos         = {}

def spawn_queens():
    global queens, q_pos
    for qid in queens.values():
        try: p.removeBody(qid)
        except: pass
    queens = {}; q_pos = {}
    for row in range(BOARD_N):
        sp = stage_world(row)
        qid = create_queen(sp, QUEEN_COLOR)
        queens[row] = qid; q_pos[row] = sp
    for _ in range(100): p.stepSimulation(); time.sleep(1/240)
    for row in range(BOARD_N):
        p.resetBasePositionAndOrientation(queens[row], q_pos[row], [0,0,0,1])
        p.resetBaseVelocity(queens[row],[0,0,0],[0,0,0])

spawn_queens()
ghost = create_ghost()
ghost2 = create_ghost(color=[0.2, 0.6, 1.0, 0.75])
redraw_flag = True

# Debug Camera Print
def print_camera():
    cam = p.getDebugVisualizerCamera()
    print("\n--- CAMERA POSE ---")
    print("Distance:", cam[10])
    print("Yaw:", cam[8])
    print("Pitch:", cam[9])
    print("Target:", cam[11])

# ── Game state ───────────────────────────────────────────────────────────────
placed      = {}
#cur_row     = 0
#cursor      = 0
cursor_row = 0
cursor_col = 0
status      = "Ready — use arrows to select, ENTER to place"
sim_mode:MODE = MODE.L4 #NOTE: Here is where to change the mode, short note on the modes at Enum class, L5 for full function

def do_reset():
    global placed, status, cursor_col, cursor_row #cur_row, cursor, status
    clear_all_hl(); clear_dots(); hide_ghost(ghost)
    for i,a in enumerate(HOME): p.resetJointState(robot,i,a)
    set_fingers(robot, FINGER_OPEN*0.5)
    spawn_queens()
    placed={}; cursor_row = 0; cursor_col = 0;#cur_row=0; cursor=0
    status="Board reset — ready"
    print("\n  ↺ RESET")

def do_undo():
    global placed, status, cursor_col, cursor_row #cur_row, cursor, status
    if not placed:
        status="Nothing to undo"; return
    row = max(placed.keys()); col = placed.pop(row)
    if row in _dots:
        try: p.removeBody(_dots.pop(row))
        except: pass
    sp = stage_world(row)
    p.resetBasePositionAndOrientation(queens[row],sp,[0,0,0,1])
    p.resetBaseVelocity(queens[row],[0,0,0],[0,0,0])
    q_pos[row]=sp; cursor_row = 0; cursor_col = 0;#cur_row=row; cursor=0
    status=f"Undid row {row} (was col {col})"
    print(f"\n  Undo row {row}")

def leads_to_solution(row, col, placed):
    trial = dict(placed)
    trial[row] = col

    def backtrack(r, state):
        if r == BOARD_N:
            return True
        if r in state:
            return backtrack(r + 1, state)

        for c in range(BOARD_N):
            if is_safe(r, c, state):
                state[r] = c
                if backtrack(r + 1, state):
                    return True
                del state[r]
        return False

    return backtrack(0, trial)

import random

def get_suggestions(placed, failure_thresh=0.0):
    roll = random.random()
    suggestions = []
    for r in range(BOARD_N):
        if r in placed:
            continue
        for c in range(BOARD_N):
            if is_safe(r, c, placed) and leads_to_solution(r, c, placed):
                if roll < failure_thresh:
                    dr = random.randint(-1, 1)
                    dc = random.randint(-1, 1)
                    suggestions.append(((r + dr) % BOARD_N, (c + dc) % BOARD_N))
                else:
                    suggestions.append((r, c))
    
    return suggestions
# ─────────────────────────────────────────────────────────────────────────────
print("\n"+"═"*55)
print("  N-QUEENS  —  SHARED CONTROL")
print("  A/← D/→  cycle   ENTER/SPC  place")
print("  U  undo   R  reset   Q  quit")
print("  GREEN=safe  ORANGE=risky  RED=illegal")
print("═"*55+"\n")

# ═══════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════
try:
    while p.isConnected():
        # ── Solved ───────────────────────────────────────────────────
        if len(placed) == BOARD_N:#if cur_row == BOARD_N:
            clear_all_hl(); hide_ghost(ghost); clear_hud()
            for qid in queens.values():
                p.changeVisualShape(qid,-1,rgbaColor=[1,0.82,0,1])
            _txt("win","  PUZZLE SOLVED!  Press R to reset  ",
                 [BOARD_X0, BOARD_Y0+3.5*SQR, TABLE_TOP_Z+0.32],
                 [0.15,1.0,0.15], 1.5)
            sol = " ".join([f"R{r}:C{c}" for r,c in sorted(placed.items())])
            print(f"\n  Solved! {sol}")
            while p.isConnected():
                keys = p.getKeyboardEvents()
                if ord('r') in keys and keys[ord('r')]&p.KEY_WAS_TRIGGERED:
                    clear_hud(); do_reset(); break
                if ord('q') in keys and keys[ord('q')]&p.KEY_WAS_TRIGGERED:
                    raise KeyboardInterrupt
                p.stepSimulation(); time.sleep(1/240)
            continue

        #safe = get_safe(cursor_row, placed)
        safe = is_safe(cursor_row, cursor_col, placed)

        # ── Auto-undo on deadlock ─────────────────────────────────────
        if not safe:
            pass
            #print(f"\n Deadlock - auto-undo")#print(f"\n  Deadlock row {cur_row} — auto-undo")
            #do_undo(); continue

        #cursor %= len(safe)
        #sel     = safe[cursor]
        #warn    = _deadlock(cur_row, sel, placed)

        # Board highlights + ghost
        if redraw_flag and (sim_mode != MODE.L1):
            if (sim_mode == MODE.L4) or (sim_mode == MODE.L3) or (sim_mode == MODE.L2): update_highlights_2d(placed, failure_threshold=0.5, mode=sim_mode)#update_highlights_(cur_row, safe, placed) # NOTE: Modify Reliability here
            elif sim_mode == MODE.L5: update_highlights_2d(placed, mode=sim_mode)
            redraw_flag = False
        move_ghost(ghost, sq_world(cursor_row, cursor_col)) # cur_row, sel

        warn = _deadlock(cursor_row, cursor_col, placed)
        # HUD
        status = (f"R{cursor_row} C{cursor_col} RISKY (dead end)" if warn
          else f"Row {cursor_row} → Col {cursor_col} | ENTER to place")
        draw_hud(cursor_row, len(placed), status)

        # Terminal
        sys.stdout.write(
            f"\r  ROW {cursor_row}  sel=C{cursor_col}  "
            f"{'[RISKY]' if warn else '[OK]'}  {len(placed)}/8   ")
        sys.stdout.flush()

        # ── Keys ─────────────────────────────────────────────────────
        keys = p.getKeyboardEvents()

        if (p.B3G_UP_ARROW in keys and 
                keys[p.B3G_UP_ARROW] & p.KEY_WAS_TRIGGERED) or \
            (ord('w') in keys and keys[ord('w')]&p.KEY_WAS_TRIGGERED):
            cursor_row = (cursor_row - 1) % BOARD_N

        if (p.B3G_DOWN_ARROW in keys and 
                keys[p.B3G_DOWN_ARROW] & p.KEY_WAS_TRIGGERED) or \
            (ord('s') in keys and keys[ord('s')]&p.KEY_WAS_TRIGGERED):
            cursor_row = (cursor_row + 1) % BOARD_N

        if (p.B3G_RIGHT_ARROW in keys and
                keys[p.B3G_RIGHT_ARROW]&p.KEY_WAS_TRIGGERED) or \
           (ord('d') in keys and keys[ord('d')]&p.KEY_WAS_TRIGGERED):
            cursor_col = (cursor_col + 1) % BOARD_N
            #cursor += 1

        if (p.B3G_LEFT_ARROW in keys and
                keys[p.B3G_LEFT_ARROW]&p.KEY_WAS_TRIGGERED) or \
           (ord('a') in keys and keys[ord('a')]&p.KEY_WAS_TRIGGERED):
            cursor_col = (cursor_col - 1) % BOARD_N
            #cursor -= 1

        if ord('q') in keys and keys[ord('q')]&p.KEY_WAS_TRIGGERED:
            print("\n  Quit."); break

        if ord('r') in keys and keys[ord('r')]&p.KEY_WAS_TRIGGERED:
            do_reset()

        if ord('u') in keys and keys[ord('u')]&p.KEY_WAS_TRIGGERED:
            clear_row(cursor_row); do_undo()

        confirm = (
            (p.B3G_RETURN in keys and keys[p.B3G_RETURN]&p.KEY_WAS_TRIGGERED) or
            (ord(' ')     in keys and keys[ord(' ')]    &p.KEY_WAS_TRIGGERED)
        )

        if confirm:
            r, c = cursor_row, cursor_col
            if r in placed and placed[r] == c:
                status = "Square already occupied"
                #status = "Row already occupied"
            #elif not is_safe(r, c, placed):
            #    status = "Illegal move"
            else:
                hide_ghost(ghost)
                clear_all_hl()
                flash_cell(r, c)
                redraw_flag = True

                src_pos = q_pos[r]
                dst_pos = sq_world(r, c)
                print(f"\n\n  ► Row {r} → Col {c}")

                cid = pick(robot, queens[r], src_pos)
                place(robot, queens[r], cid, dst_pos)

                q_pos[r] = dst_pos
                placed[r] = c#dst_pos
                add_dot(r, c)
                status = f"{len(placed)}/8 queens placed"
                print(f"    ✓  {len(placed)}/8")

        p.stepSimulation()
        time.sleep(1/240)

except KeyboardInterrupt:
    print("\n  Stopped.")
except Exception as e:
    import traceback
    print(f"\n  [ERROR] {e}")
    traceback.print_exc()
finally:
    p.disconnect()
    print("  Disconnected.")