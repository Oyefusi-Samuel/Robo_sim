"""
shared_control_nqueens_gaze.py  ─ GAZE + KEYBOARD SHARED CONTROL  (FINAL)
==========================================================================
Phase 1 ─ Calibration
  9 board cells highlighted CYAN.  Look at each for 3 s.
  Robust affine fit: median-per-cell + IQR outlier rejection.

Phase 2 ─ Gameplay
  GAZE         → cursor (row, col)
  ENTER/SPACE  → place queen at cursor
  DWELL 2 s    → auto-place
  ARROW KEYS   → cursor override (suppresses gaze for 1.5 s)
  U            → undo   R → reset   Q → quit

Facial expression (Smiling / Frowning / Neutral) is detected on the
MAIN thread from the latest webcam frame and shown as a PyBullet HUD
overlay — no threading issues, no double-detection.
==========================================================================
"""

import os, sys, time, math, threading
import pybullet as p
import pybullet_data
import cv2
import numpy as np
import gaze_tracking
from mediapipe.python.solutions import face_mesh as mp_face_mesh

# ═══════════════════════════════════════════════════════════════════════════════
#  NON-BLOCKING CAMERA READER THREAD
#  cap.read() on Windows can block 100-500 ms when webcam is slow/autofocusing.
#  This thread reads continuously; callers just grab the latest frame instantly.
# ═══════════════════════════════════════════════════════════════════════════════
class CamReader:
    def __init__(self, cap):
        self.cap   = cap
        self._lock = threading.Lock()
        self._frame= None
        self._stop = threading.Event()
        self._t    = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.005)

    def read(self):
        """Return (True, frame) or (False, None) — never blocks."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self):
        self._stop.set()


# ── Expression tracking (absolute path — fixes Windows MediaPipe bug) ─────────
try:
    import mediapipe as _mp
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision

    _MODEL_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'face_landmarker_v2_with_blendshapes.task'
    )
    _expr_detector = _mp_vision.FaceLandmarker.create_from_options(
        _mp_vision.FaceLandmarkerOptions(
            base_options=_mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
            output_face_blendshapes=True,
            num_faces=1
        )
    )
    EXPR_AVAILABLE = True
    print("  [EXPR] FaceLandmarker loaded OK")
except Exception as _e:
    EXPR_AVAILABLE = False
    print(f"  [EXPR] Expression tracking unavailable: {_e}")


def _detect_expression(bgr_frame):
    """Run blendshape detection on a BGR frame. Returns label string."""
    if not EXPR_AVAILABLE:
        return ''
    try:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mi  = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
        det = _expr_detector.detect(mi)
        if not det.face_blendshapes:
            return 'Neutral'
        cats  = det.face_blendshapes[0]
        smile = sum(c.score for c in cats if 'Smile' in c.category_name)
        frown = sum(c.score for c in cats if 'Frown' in c.category_name)
        return ('Smiling'  if smile > 1.0  else
                'Frowning' if frown > 0.005 else 'Neutral')
    except Exception:
        return ''


# ── Expression background thread (avoids freezing main loop) ─────────────────
_expr_result  = ['']       # latest label — main loop reads this
_expr_queue   = [None]     # main loop deposits frame here
_expr_tlock   = threading.Lock()

def _expr_bg_worker():
    while True:
        frm = None
        with _expr_tlock:
            if _expr_queue[0] is not None:
                frm = _expr_queue[0]
                _expr_queue[0] = None
        if frm is not None:
            label = _detect_expression(frm)
            if label:
                with _expr_tlock:
                    _expr_result[0] = label
        time.sleep(0.04)

threading.Thread(target=_expr_bg_worker, daemon=True).start()

# ═══════════════════════════════════════════════════════════════════════════════
#  LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
SQR     = 0.055
BOARD_N = 8

TABLE_CX, TABLE_CY          =  0.02, -0.48
TABLE_HALF_X, TABLE_HALF_Y  =  0.38,  0.38
TABLE_HALF_Z                =  0.09
TABLE_TOP_Z                 =  TABLE_HALF_Z * 2   # 0.18

BOARD_SQ_H = 0.008
BOARD_Z    = TABLE_TOP_Z + BOARD_SQ_H / 2
BOARD_X0   = -0.175
BOARD_Y0   = -0.620

Q_RADIUS, Q_HEIGHT = 0.020, 0.055
Q_BASE_R,  Q_BASE_H = 0.026, 0.010

STAGE_X0, STAGE_Y = -0.175, -0.165
ROBOT_POS         = [0.02, -0.10, TABLE_TOP_Z]
ROBOT_SC          =  0.55

LIFT_Z       = 0.52
PICK_Z_OFF   = 0.095
FINGER_OPEN  = 0.038
FINGER_CLOSE = 0.005

# ═══════════════════════════════════════════════════════════════════════════════
#  GAZE / CALIBRATION CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
DWELL_THRESH     = 2.0   # seconds held on same cell → auto-place
KB_OVERRIDE_SECS = 1.5   # keyboard takes priority over gaze for this long

CAL_CELLS = [
    (0, 0), (0, 3), (0, 7),
    (3, 0), (3, 3), (3, 7),
    (7, 0), (7, 3), (7, 7),
]
CAL_DWELL   = 3.0   # seconds per target
CAL_DISCARD = 0.6   # skip first N seconds (settling)
CAL_CLAMP   = 10000 # reject truly insane outliers only
EXPR_INTERVAL = 0.25  # run expression detection every N seconds

# ═══════════════════════════════════════════════════════════════════════════════
#  WORLD
# ═══════════════════════════════════════════════════════════════════════════════
def create_world():
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    p.createMultiBody(0,
        p.createCollisionShape(p.GEOM_BOX,
            halfExtents=[TABLE_HALF_X, TABLE_HALF_Y, TABLE_HALF_Z]),
        p.createVisualShape(p.GEOM_BOX,
            halfExtents=[TABLE_HALF_X, TABLE_HALF_Y, TABLE_HALF_Z],
            rgbaColor=[0.45, 0.25, 0.07, 1]),
        [TABLE_CX, TABLE_CY, TABLE_HALF_Z])
    h = BOARD_SQ_H / 2
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            col  = [0.95, 0.95, 0.95, 1] if (r+c) % 2 == 0 else [0.08, 0.08, 0.08, 1]
            half = [SQR/2.05, SQR/2.05, h]
            p.createMultiBody(0,
                p.createCollisionShape(p.GEOM_BOX, halfExtents=half),
                p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=col),
                [BOARD_X0+r*SQR, BOARD_Y0+c*SQR, BOARD_Z])


def sq_world(row, col):
    return [BOARD_X0+row*SQR, BOARD_Y0+col*SQR, TABLE_TOP_Z+Q_BASE_H+Q_HEIGHT/2]


def stage_world(row):
    return [STAGE_X0+row*SQR, STAGE_Y, TABLE_TOP_Z+Q_BASE_H+Q_HEIGHT/2]


# ═══════════════════════════════════════════════════════════════════════════════
#  QUEENS / GHOSTS
# ═══════════════════════════════════════════════════════════════════════════════
def create_queen(pos, color):
    qid = p.createMultiBody(0.08,
        p.createCollisionShape(p.GEOM_CYLINDER, radius=Q_RADIUS, height=Q_HEIGHT),
        p.createVisualShape(p.GEOM_CYLINDER, radius=Q_RADIUS, length=Q_HEIGHT,
                            rgbaColor=color), pos)
    p.changeDynamics(qid, -1, linearDamping=0.95, angularDamping=0.95,
                     lateralFriction=1.2, restitution=0.0)
    p.createMultiBody(0, -1,
        p.createVisualShape(p.GEOM_CYLINDER, radius=Q_BASE_R, length=Q_BASE_H,
            rgbaColor=[min(color[0]+.1,1), min(color[1]+.1,1), min(color[2]+.1,1), 1]),
        [pos[0], pos[1], TABLE_TOP_Z+Q_BASE_H/2])
    p.createMultiBody(0, -1,
        p.createVisualShape(p.GEOM_SPHERE, radius=Q_RADIUS*1.25, rgbaColor=color),
        [pos[0], pos[1], pos[2]+Q_HEIGHT/2])
    return qid


def create_ghost(color=(1, .9, 0, .45)):
    return p.createMultiBody(0, -1,
        p.createVisualShape(p.GEOM_CYLINDER, radius=Q_RADIUS, length=Q_HEIGHT,
                            rgbaColor=list(color)), [0, 0, -5])


def move_ghost(gid, pos): p.resetBasePositionAndOrientation(gid, pos, [0, 0, 0, 1])
def hide_ghost(gid):      p.resetBasePositionAndOrientation(gid, [0, 0, -5], [0, 0, 0, 1])


# ═══════════════════════════════════════════════════════════════════════════════
#  HIGHLIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
_hl = {}


def init_highlights():
    half = [SQR*.46, SQR*.46, 0.0008]
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0,0,0,0])
            _hl[(r,c)] = p.createMultiBody(0, -1, vis, [0, 0, -5])


def _sq_z(): return TABLE_TOP_Z+BOARD_SQ_H+0.001


def _show(r, c, rgba):
    bid = _hl.get((r,c))
    if bid is None: return
    p.resetBasePositionAndOrientation(bid, [BOARD_X0+r*SQR, BOARD_Y0+c*SQR, _sq_z()], [0,0,0,1])
    p.changeVisualShape(bid, -1, rgbaColor=rgba)


def _hide(r, c):
    bid = _hl.get((r,c))
    if bid is None: return
    p.resetBasePositionAndOrientation(bid, [0, 0, -5], [0, 0, 0, 1])
    p.changeVisualShape(bid, -1, rgbaColor=[0, 0, 0, 0])


def update_highlights(placed):
    sugg = get_suggestions(placed)
    if sugg: move_ghost(ghost2, sq_world(*sugg[0]))
    else:    hide_ghost(ghost2)
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            if is_safe(r, c, placed) and r not in placed:
                _show(r, c, [1., .55, 0., .72] if _deadlock(r, c, placed)
                            else [.05, .90, .15, .68])
            else:
                _show(r, c, [.92, .08, .08, .62])


def clear_all_hl():
    for r in range(BOARD_N):
        for c in range(BOARD_N): _hide(r, c)


_dots = {}


def add_dot(row, col):
    if row in _dots:
        try: p.removeBody(_dots[row])
        except: pass
    half = [SQR*.18, SQR*.18, 0.002]
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[.1, 1., .3, .9])
    _dots[row] = p.createMultiBody(0, -1, vis,
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
        p.changeVisualShape(bid, -1, rgbaColor=[1, 1, 1, .95])
        for _ in range(6): p.stepSimulation(); time.sleep(1/240)
        p.changeVisualShape(bid, -1, rgbaColor=[.05, .90, .15, .68])
        for _ in range(6): p.stepSimulation(); time.sleep(1/240)


# ═══════════════════════════════════════════════════════════════════════════════
#  HUD
# ═══════════════════════════════════════════════════════════════════════════════
_hud = {}


def _txt(key, text, pos, rgb, sz=1.0):
    if key in _hud:
        try: p.removeUserDebugItem(_hud[key])
        except: pass
    _hud[key] = p.addUserDebugText(text, pos, textColorRGB=rgb, textSize=sz, lifeTime=0)


def draw_hud(cur_row, n_placed, status, dwell_pct=0., gaze_on=True, expression=''):
    ax = BOARD_X0-0.01
    ay = BOARD_Y0+(BOARD_N-1)*SQR+0.05
    az = TABLE_TOP_Z+0.31
    lh = 0.058
    tag = "GAZE+KB" if gaze_on else "KB ONLY"
    _txt("h0", f"N-QUEENS  |  SHARED CONTROL  [{tag}]",
         [ax, ay, az+lh*4], [1., .85, 0.], 1.2)
    _txt("h1", f"Row {cur_row}/8    Placed: {n_placed}/8",
         [ax, ay, az+lh*3], [.9, .9, .9], 1.0)
    _txt("h2", "GREEN=safe   ORANGE=risky   RED=illegal",
         [ax, ay, az+lh*2], [.55, .95, .55], .85)
    _txt("h3", "GAZE=cursor   ENTER=place   DWELL 2s=auto",
         [ax, ay, az+lh*1], [.65, .82, 1.], .85)
    _txt("h4", "U=undo   R=reset   Q=quit   Arrows=override",
         [ax, ay, az+lh*0], [.65, .82, 1.], .85)
    sc = [1., .35, .35] if any(w in status for w in ["RISKY","DEAD","block"]) \
         else [.35, 1., .45]
    _txt("h5", status, [ax, ay, az-lh*1.1], sc, .90)
    if dwell_pct > 0.:
        bar = "▓"*int(dwell_pct*10) + "░"*(10-int(dwell_pct*10))
        _txt("h6", f"Dwell [{bar}]", [ax, ay, az-lh*2.1], [.85, .7, 1.], .80)
    elif "h6" in _hud:
        try: p.removeUserDebugItem(_hud.pop("h6"))
        except: pass
    if expression:
        ecol = ([.2, 1., .2]  if expression == 'Smiling'  else
                [1., .35, .35] if expression == 'Frowning' else
                [.85, .85, .85])
        _txt("h7", f"Face: {expression}", [ax, ay, az-lh*3.2], ecol, .95)
    elif "h7" in _hud:
        try: p.removeUserDebugItem(_hud.pop("h7"))
        except: pass


def clear_hud():
    for v in _hud.values():
        try: p.removeUserDebugItem(v)
        except: pass
    _hud.clear()


# ═══════════════════════════════════════════════════════════════════════════════
#  N-QUEENS LOGIC
# ═══════════════════════════════════════════════════════════════════════════════
def is_safe(row, col, placed):
    return all(c != col and abs(r-row) != abs(c-col) for r, c in placed.items())


def get_safe(row, placed):
    return [c for c in range(BOARD_N)
            if all(pc != c and abs(pr-row) != abs(pc-c) for pr, pc in placed.items())]


def _deadlock(row, col, placed):
    trial = dict(placed); trial[row] = col
    return any(not get_safe(fr, trial) for fr in range(row+1, BOARD_N))


def leads_to_solution(row, col, placed):
    trial = dict(placed); trial[row] = col
    def bt(r, state):
        if r == BOARD_N: return True
        if r in state:   return bt(r+1, state)
        for c in range(BOARD_N):
            if is_safe(r, c, state):
                state[r] = c
                if bt(r+1, state): return True
                del state[r]
        return False
    return bt(0, trial)


def get_suggestions(placed):
    out = []
    for r in range(BOARD_N):
        if r in placed: continue
        for c in range(BOARD_N):
            if is_safe(r, c, placed) and leads_to_solution(r, c, placed):
                out.append((r, c))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  ROBOT CONTROL
# ═══════════════════════════════════════════════════════════════════════════════
def set_fingers(robot, val, steps=20):
    c9 = p.getJointState(robot, 9)[0]; c10 = p.getJointState(robot, 10)[0]
    for i in range(steps):
        t = (i+1)/steps
        p.setJointMotorControl2(robot, 9,  p.POSITION_CONTROL, c9 +(val-c9 )*t, force=80)
        p.setJointMotorControl2(robot, 10, p.POSITION_CONTROL, c10+(val-c10)*t, force=80)
        p.stepSimulation(); time.sleep(1/240)


def ik_go(robot, tgt, steps=70):
    orn   = p.getQuaternionFromEuler([math.pi, 0, math.pi/2])
    start = list(p.getLinkState(robot, 11)[0])
    for i in range(steps):
        t      = (i+1)/steps
        interp = [start[j]+(tgt[j]-start[j])*t for j in range(3)]
        jt     = p.calculateInverseKinematics(robot, 11, interp, orn,
                                              maxNumIterations=100, residualThreshold=1e-5)
        for j in range(7):
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, jt[j], force=600)
        p.stepSimulation(); time.sleep(1/240)


def pick(robot, qid, src):
    gz = src[2]+PICK_Z_OFF
    set_fingers(robot, FINGER_OPEN, 20)
    ik_go(robot, [src[0], src[1], LIFT_Z], 70)
    ik_go(robot, [src[0], src[1], gz], 80)
    set_fingers(robot, FINGER_CLOSE, 25)
    ee   = p.getLinkState(robot, 11)[0]
    snap = [ee[0], ee[1], ee[2]-PICK_Z_OFF]
    p.resetBasePositionAndOrientation(qid, snap, [0, 0, 0, 1])
    p.resetBaseVelocity(qid, [0, 0, 0], [0, 0, 0])
    cid = p.createConstraint(robot, 11, qid, -1, p.JOINT_FIXED,
                             [0, 0, 0], [0, 0, PICK_Z_OFF], [0, 0, 0])
    p.changeConstraint(cid, maxForce=500)
    ik_go(robot, [src[0], src[1], LIFT_Z], 70)
    return cid


def place(robot, qid, cid, dst):
    pz = dst[2]+PICK_Z_OFF
    ik_go(robot, [dst[0], dst[1], LIFT_Z], 80)
    ik_go(robot, [dst[0], dst[1], pz], 80)
    set_fingers(robot, FINGER_OPEN, 20)
    p.removeConstraint(cid)
    p.resetBasePositionAndOrientation(qid, dst, [0, 0, 0, 1])
    p.resetBaseVelocity(qid, [0, 0, 0], [0, 0, 0])
    for _ in range(25): p.stepSimulation(); time.sleep(1/240)
    ik_go(robot, [dst[0], dst[1], LIFT_Z], 55)
    set_fingers(robot, FINGER_OPEN*.5, 10)


# ═══════════════════════════════════════════════════════════════════════════════
#  PURE-PYTHON LEAST SQUARES  (no numpy/LAPACK — avoids MKL crash on Windows)
# ═══════════════════════════════════════════════════════════════════════════════
def _py_lstsq3(rows, y):
    A = [[0.]*3 for _ in range(3)]; b = [0.]*3
    for row, yi in zip(rows, y):
        for j in range(3):
            b[j] += row[j]*yi
            for k in range(3): A[j][k] += row[j]*row[k]
    M = [A[i][:]+[b[i]] for i in range(3)]
    for col in range(3):
        pr = max(range(col, 3), key=lambda r: abs(M[r][col]))
        M[col], M[pr] = M[pr], M[col]
        if abs(M[col][col]) < 1e-12: return [0., 0., 0.]
        for row in range(3):
            if row == col: continue
            f = M[row][col]/M[col][col]
            M[row] = [M[row][j]-f*M[col][j] for j in range(4)]
    return [M[i][3]/M[i][i] for i in range(3)]


def gaze_to_cell(gx, gy, A_row, A_col):
    v = [gx, gy, 1.]
    row = int(max(0, min(BOARD_N-1, round(sum(A_row[i]*v[i] for i in range(3))))))
    col = int(max(0, min(BOARD_N-1, round(sum(A_col[i]*v[i] for i in range(3))))))
    return row, col


# ═══════════════════════════════════════════════════════════════════════════════
#  GAZE CALIBRATION  (main thread — robust)
# ═══════════════════════════════════════════════════════════════════════════════
def run_calibration(cap, face_mesh, smooth):
    """
    Collect 3 s of gaze per cell:
      - discard first CAL_DISCARD seconds
      - clamp |gaze| > CAL_CLAMP (unstable samples)
      - USE MEDIAN per cell (robust to saccades)
      - IQR outlier rejection across cells before fitting
    Returns (A_row, A_col) or None.
    """
    gaze_data = []

    for idx, (trow, tcol) in enumerate(CAL_CELLS):
        clear_all_hl()
        _show(trow, tcol, [0., 1., 1., .98])

        ax = BOARD_X0-0.01; ay = BOARD_Y0+(BOARD_N-1)*SQR+0.05
        az = TABLE_TOP_Z+0.38; lh = 0.060
        _txt("ct0", f"CALIBRATION  {idx+1} / {len(CAL_CELLS)}",
             [ax, ay, az+lh*0], [1., .85, 0.], 1.1)
        _txt("ct1", f"Look at CYAN cell  R{trow}  C{tcol}",
             [ax, ay, az-lh*1], [0., 1., 1.], .95)
        _txt("ct2", "Hold gaze steady — move EYES not head",
             [ax, ay, az-lh*2], [.75, .75, .75], .85)

        t_start = time.time(); samples = []

        while time.time()-t_start < CAL_DWELL:
            elapsed = time.time()-t_start
            p.stepSimulation(); time.sleep(1/240)
            ret, frame = cam.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = face_mesh.process(rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                gx, gy, _ = gaze_tracking.gaze_to_screen_point(lm, w, h, w, h)
                sx, sy    = smooth.update(gx, gy)
                if elapsed > CAL_DISCARD and abs(sx) < CAL_CLAMP and abs(sy) < CAL_CLAMP:
                    samples.append((sx, sy))
                cv2.circle(frame, (int(np.clip(sx, 0, w-1)),
                                   int(np.clip(sy, 0, h-1))), 8, (0, 255, 200), -1)
            pct = elapsed/CAL_DWELL
            cv2.rectangle(frame, (0, h-8), (int(w*pct), h), (0, 220, 150), -1)
            cv2.putText(frame,
                f"CAL {idx+1}/{len(CAL_CELLS)}  R{trow}C{tcol}  "
                f"clean={len(samples)}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 180), 2)
            cv2.imshow("Gaze Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                _cleanup_cal_hud(); return None

        if len(samples) >= 5:
            med_gx = float(np.median([s[0] for s in samples]))
            med_gy = float(np.median([s[1] for s in samples]))
            gaze_data.append((med_gx, med_gy, trow, tcol))
            print(f"  CAL {idx+1}: R{trow}C{tcol}  "
                  f"median=({med_gx:.1f},{med_gy:.1f})  n={len(samples)}")
        else:
            print(f"  CAL {idx+1}: R{trow}C{tcol}  "
                  f"only {len(samples)} clean samples — skipped")

        _show(trow, tcol, [.05, .90, .15, .95])
        t_f = time.time()
        while time.time()-t_f < 0.3:
            p.stepSimulation(); cv2.waitKey(1); time.sleep(1/240)

    # ── Fit ───────────────────────────────────────────────────────────────────
    print(f"  [CAL] {len(gaze_data)} usable points."); sys.stdout.flush()
    if len(gaze_data) < 4:
        print("  [CAL] Too few — keyboard-only fallback."); return None

    # IQR outlier rejection
    gxs = [d[0] for d in gaze_data]; gys = [d[1] for d in gaze_data]
    def _iqr_bounds(vals):
        sv = sorted(vals); n = len(sv)
        q1 = sv[n//4]; q3 = sv[(3*n)//4]; iqr = q3-q1
        return q1-1.5*iqr, q3+1.5*iqr
    lox, hix = _iqr_bounds(gxs); loy, hiy = _iqr_bounds(gys)
    clean = [d for d in gaze_data if lox <= d[0] <= hix and loy <= d[1] <= hiy]
    print(f"  [CAL] After IQR: {len(clean)}/{len(gaze_data)} points remain")
    if len(clean) < 4:
        clean = gaze_data
        print("  [CAL] IQR too aggressive — using all points")

    G = [[d[0], d[1], 1.] for d in clean]
    R = [float(d[2]) for d in clean]
    C = [float(d[3]) for d in clean]
    A_row = _py_lstsq3(G, R); A_col = _py_lstsq3(G, C)
    print(f"  [CAL] A_row = {[round(x,4) for x in A_row]}")
    print(f"  [CAL] A_col = {[round(x,4) for x in A_col]}")
    sys.stdout.flush()
    return A_row, A_col


def _cleanup_cal_hud():
    for k in ("ct0", "ct1", "ct2"):
        try: p.removeUserDebugItem(_hud.pop(k))
        except: pass


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED STATE  (gaze thread → main loop)
# ═══════════════════════════════════════════════════════════════════════════════
_gaze_lock    = threading.Lock()
_stop_gaze    = threading.Event()
_frame_lock   = threading.Lock()
_latest_frame = [None]        # annotated webcam frame for display + expression
_kb_last_press = [0.0]        # timestamp of last keyboard cursor press

gaze_state = {
    'row'      : 0,
    'col'      : 0,
    'active'   : False,
    'confirm'  : False,
    'dwell_pct': 0.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  GAZE WORKER THREAD  (lean — gaze only, no expression detection)
# ═══════════════════════════════════════════════════════════════════════════════
def gaze_worker(cap, cal_data):
    A_row, A_col = cal_data
    dwell_row = dwell_col = -1
    dwell_start = None

    fm      = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                     min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)
    smoother = gaze_tracking.GazeSmoothTracking(alpha=0.20)

    while not _stop_gaze.is_set():
        ret, frame = cam.read()
        if not ret: time.sleep(0.005); continue
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = fm.process(rgb)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            gx, gy, gaze_world = gaze_tracking.gaze_to_screen_point(lm, w, h, w, h)
            sx, sy = smoother.update(gx, gy)
            row, col = gaze_to_cell(sx, sy, A_row, A_col)

            confirm = False; dwell_pct = 0.
            if row == dwell_row and col == dwell_col:
                if dwell_start is None: dwell_start = time.time()
                elapsed   = time.time()-dwell_start
                dwell_pct = min(elapsed/DWELL_THRESH, 1.)
                if elapsed >= DWELL_THRESH:
                    confirm = True; dwell_start = None; dwell_pct = 0.
            else:
                dwell_row = row; dwell_col = col
                dwell_start = time.time(); dwell_pct = 0.

            with _gaze_lock:
                gaze_state['row']       = row
                gaze_state['col']       = col
                gaze_state['active']    = True
                gaze_state['dwell_pct'] = dwell_pct
                if confirm: gaze_state['confirm'] = True

            cv2.circle(frame, (int(np.clip(sx, 0, w-1)),
                                int(np.clip(sy, 0, h-1))), 8, (0, 255, 100), -1)
            cv2.putText(frame, f"R{row} C{col}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 100), 2)
            if gaze_world is not None:
                re = lm[468]; le = lm[473]
                ex = w-int((re.x+le.x)/2*w)
                ey = int((re.y+le.y)/2*h)
                cv2.arrowedLine(frame, (ex, ey),
                    (ex+int(-gaze_world[0]*200), ey+int(gaze_world[1]*200)),
                    (0, 255, 0), 2, tipLength=0.3)
            if dwell_pct > 0.:
                cv2.rectangle(frame, (0, h-8), (int(w*dwell_pct), h), (180, 80, 255), -1)
        else:
            with _gaze_lock: gaze_state['active'] = False

        with _frame_lock: _latest_frame[0] = frame.copy()

    fm.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  PYBULLET INIT
# ═══════════════════════════════════════════════════════════════════════════════
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setRealTimeSimulation(0)
p.resetDebugVisualizerCamera(1.11, 37.8, -48.4, [-0.12, -0.19, -0.08])
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

create_world()
init_highlights()

robot = p.loadURDF("franka_panda/panda.urdf", ROBOT_POS,
                   useFixedBase=True, globalScaling=ROBOT_SC)
HOME = [0., 0.15, 0., -1.90, 0., 2.05, .80]
for i, a in enumerate(HOME): p.resetJointState(robot, i, a)
set_fingers(robot, FINGER_OPEN*.5)

QUEEN_COLOR = [0.15, 0.25, 0.90, 1]
queens = {}; q_pos = {}


def spawn_queens():
    global queens, q_pos
    for qid in queens.values():
        try: p.removeBody(qid)
        except: pass
    queens = {}; q_pos = {}
    for row in range(BOARD_N):
        sp = stage_world(row)
        queens[row] = create_queen(sp, QUEEN_COLOR); q_pos[row] = sp
    for _ in range(100): p.stepSimulation(); time.sleep(1/240)
    for row in range(BOARD_N):
        p.resetBasePositionAndOrientation(queens[row], q_pos[row], [0, 0, 0, 1])
        p.resetBaseVelocity(queens[row], [0, 0, 0], [0, 0, 0])


spawn_queens()
ghost  = create_ghost()
ghost2 = create_ghost(color=(.2, .6, 1., .75))

# ═══════════════════════════════════════════════════════════════════════════════
#  WEBCAM INIT
# ═══════════════════════════════════════════════════════════════════════════════
_raw_cap = cv2.VideoCapture(0)
CAL_FALLBACK = False
A_row = A_col = None
cam = None

if not _raw_cap.isOpened():
    print("[WARN] No webcam — keyboard-only mode.")
    CAL_FALLBACK = True
else:
    cam = CamReader(_raw_cap)   # non-blocking frame grabber
    time.sleep(0.5)             # let first frames buffer

# ═══════════════════════════════════════════════════════════════════════════════
#  CALIBRATION PHASE
# ═══════════════════════════════════════════════════════════════════════════════
if not CAL_FALLBACK:
    face_mesh_cal = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    smooth_cal = gaze_tracking.GazeSmoothTracking(alpha=0.12)

    print("\n"+"═"*58)
    print("  GAZE CALIBRATION")
    print("  9 cells highlighted CYAN — look at each for 3 s.")
    print("  Move your EYES only, keep your head still.")
    print("═"*58); sys.stdout.flush()

    _txt("intro", "CALIBRATION — look at each CYAN cell (3 s each)",
         [BOARD_X0-.01, BOARD_Y0+(BOARD_N-1)*SQR+0.05, TABLE_TOP_Z+0.44],
         [0., 1., 1.], 1.0)
    for _ in range(300): p.stepSimulation(); time.sleep(1/240)  # display intro

    try:
        cal_result = run_calibration(cam, face_mesh_cal, smooth_cal)
    except Exception as _e:
        import traceback; traceback.print_exc(); cal_result = None

    try: face_mesh_cal.close()
    except: pass

    print("  [POST-CAL] Draining render queue …"); sys.stdout.flush()
    for _ in range(120): p.stepSimulation(); time.sleep(1/240)
    try: _cleanup_cal_hud()
    except: pass
    for _ in range(30): p.stepSimulation(); time.sleep(1/240)
    try: clear_all_hl()
    except: pass
    for _ in range(30): p.stepSimulation(); time.sleep(1/240)
    try: p.removeUserDebugItem(_hud.pop("intro"))
    except: pass

    if cal_result is None:
        print("  Calibration failed — keyboard-only mode.")
        CAL_FALLBACK = True
    else:
        A_row, A_col = cal_result
        print("  Calibration complete!  Starting game …")
        sys.stdout.flush()

# ═══════════════════════════════════════════════════════════════════════════════
#  START GAZE THREAD
# ═══════════════════════════════════════════════════════════════════════════════
if not CAL_FALLBACK:
    gaze_thread = threading.Thread(target=gaze_worker,
                                   args=(cam, (A_row, A_col)), daemon=True)
    gaze_thread.start()
    print("  Gaze thread started."); sys.stdout.flush()

# ═══════════════════════════════════════════════════════════════════════════════
#  GAME STATE
# ═══════════════════════════════════════════════════════════════════════════════
placed = {}; cursor_row = 0; cursor_col = 0
status = "Ready — look at a cell then ENTER or dwell to place"
cur_expression = ''
_expr_last_t   = 0.


def do_reset():
    global placed, status, cursor_row, cursor_col
    clear_all_hl(); clear_dots(); hide_ghost(ghost)
    for i, a in enumerate(HOME): p.resetJointState(robot, i, a)
    set_fingers(robot, FINGER_OPEN*.5)
    spawn_queens()
    placed = {}; cursor_row = 0; cursor_col = 0
    status = "Board reset — ready"
    print("\n  ↺ RESET")


def do_undo():
    global placed, status, cursor_row, cursor_col
    if not placed: status = "Nothing to undo"; return
    row = max(placed.keys()); col = placed.pop(row)
    if row in _dots:
        try: p.removeBody(_dots.pop(row))
        except: pass
    sp = stage_world(row)
    p.resetBasePositionAndOrientation(queens[row], sp, [0, 0, 0, 1])
    p.resetBaseVelocity(queens[row], [0, 0, 0], [0, 0, 0])
    q_pos[row] = sp; cursor_row = 0; cursor_col = 0
    status = f"Undid row {row} (was col {col})"
    print(f"\n  Undo row {row}")


print("\n"+"═"*58)
print("  N-QUEENS  —  GAZE + KEYBOARD SHARED CONTROL")
print("  GAZE=cursor   DWELL 2s=auto   ENTER/SPACE=place")
print("  ARROWS=override   U=undo   R=reset   Q=quit")
print("═"*58+"\n")

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════
try:
    while p.isConnected():

        # ── Solved ───────────────────────────────────────────────────
        if len(placed) == BOARD_N:
            clear_all_hl(); hide_ghost(ghost); clear_hud()
            for qid in queens.values():
                p.changeVisualShape(qid, -1, rgbaColor=[1, .82, 0, 1])
            _txt("win", "  PUZZLE SOLVED!  Press R to reset  ",
                 [BOARD_X0, BOARD_Y0+3.5*SQR, TABLE_TOP_Z+0.32], [.15, 1., .15], 1.5)
            sol = " ".join([f"R{r}:C{c}" for r, c in sorted(placed.items())])
            print(f"\n  Solved!  {sol}")
            while p.isConnected():
                keys = p.getKeyboardEvents()
                if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                    clear_hud(); do_reset(); break
                if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                    raise KeyboardInterrupt
                p.stepSimulation(); time.sleep(1/240)
            continue

        # ── Read gaze ─────────────────────────────────────────────────
        gaze_confirm = False; dwell_pct = 0.
        if not CAL_FALLBACK:
            kb_idle = (time.time()-_kb_last_press[0]) > KB_OVERRIDE_SECS
            with _gaze_lock:
                if kb_idle:
                    cursor_row = gaze_state['row']
                    cursor_col = gaze_state['col']
                dwell_pct    = gaze_state['dwell_pct']
                gaze_confirm = gaze_state['confirm']
                if gaze_confirm: gaze_state['confirm'] = False

        # ── Expression: post frame to bg thread, read latest result ────
        now = time.time()
        if EXPR_AVAILABLE and not CAL_FALLBACK and now-_expr_last_t > EXPR_INTERVAL:
            _expr_last_t = now
            with _frame_lock: frm = _latest_frame[0]
            if frm is not None:
                with _expr_tlock: _expr_queue[0] = frm   # non-blocking
            with _expr_tlock: cur_expression = _expr_result[0]

        # ── Highlights + ghost ────────────────────────────────────────
        update_highlights(placed)
        move_ghost(ghost, sq_world(cursor_row, cursor_col))

        warn        = _deadlock(cursor_row, cursor_col, placed)
        legal       = is_safe(cursor_row, cursor_col, placed) and cursor_row not in placed
        placed_here = cursor_row in placed and placed[cursor_row] == cursor_col

        if placed_here:
            status = f"R{cursor_row} C{cursor_col} [already placed]"
        elif warn and legal:
            status = f"R{cursor_row} C{cursor_col} RISKY — may dead-end"
        elif legal:
            status = f"R{cursor_row} C{cursor_col} [OK]  ENTER / dwell to place"
        else:
            status = f"R{cursor_row} C{cursor_col} illegal — look at green cell"

        draw_hud(cursor_row, len(placed), status, dwell_pct,
                 not CAL_FALLBACK, cur_expression)

        sys.stdout.write(
            f"\r  {'GAZE' if not CAL_FALLBACK else 'KB'}"
            f"  R{cursor_row}C{cursor_col}"
            f"  {'[RISKY]' if warn else '[OK]'}"
            f"  {len(placed)}/8  dwell={dwell_pct:.2f}"
            f"  [{cur_expression}]  ")
        sys.stdout.flush()

        # ── Keyboard ──────────────────────────────────────────────────
        keys = p.getKeyboardEvents()

        def _kb(): _kb_last_press[0] = time.time()

        if ((p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_WAS_TRIGGERED) or
                (ord('w') in keys and keys[ord('w')] & p.KEY_WAS_TRIGGERED)):
            cursor_row = (cursor_row-1) % BOARD_N; _kb()

        if ((p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_WAS_TRIGGERED) or
                (ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED)):
            cursor_row = (cursor_row+1) % BOARD_N; _kb()

        if ((p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_WAS_TRIGGERED) or
                (ord('d') in keys and keys[ord('d')] & p.KEY_WAS_TRIGGERED)):
            cursor_col = (cursor_col+1) % BOARD_N; _kb()

        if ((p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_WAS_TRIGGERED) or
                (ord('a') in keys and keys[ord('a')] & p.KEY_WAS_TRIGGERED)):
            cursor_col = (cursor_col-1) % BOARD_N; _kb()

        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            print("\n  Quit."); break
        if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
            do_reset()
        if ord('u') in keys and keys[ord('u')] & p.KEY_WAS_TRIGGERED:
            clear_all_hl(); do_undo()

        confirm = gaze_confirm or (
            (p.B3G_RETURN in keys and keys[p.B3G_RETURN] & p.KEY_WAS_TRIGGERED) or
            (ord(' ')     in keys and keys[ord(' ')]     & p.KEY_WAS_TRIGGERED))

        # ── Place queen ───────────────────────────────────────────────
        if confirm:
            r, c = cursor_row, cursor_col
            if placed_here:
                status = "Already placed here"
            elif not is_safe(r, c, placed) or r in placed:
                status = "Illegal — look at a green/orange cell"
            else:
                hide_ghost(ghost); clear_all_hl(); flash_cell(r, c)
                src_pos = q_pos[r]; dst_pos = sq_world(r, c)
                print(f"\n\n  ► Row {r} → Col {c}  [face: {cur_expression}]")
                cid = pick(robot, queens[r], src_pos)
                place(robot, queens[r], cid, dst_pos)
                q_pos[r] = dst_pos; placed[r] = c
                add_dot(r, c)
                status = f"{len(placed)}/8 queens placed"
                print(f"    ✓  {len(placed)}/8")

        # ── Show webcam feed (MUST be main thread on Windows) ─────────
        if not CAL_FALLBACK:
            with _frame_lock: frm = _latest_frame[0]
            if frm is not None:
                cv2.imshow("Gaze Feed", frm); cv2.waitKey(1)

        p.stepSimulation(); time.sleep(1/240)

except KeyboardInterrupt:
    print("\n  Stopped.")
except Exception as exc:
    import traceback
    print(f"\n  [ERROR] {exc}"); traceback.print_exc()
finally:
    _stop_gaze.set()
    time.sleep(0.3)
    try:
        if cam is not None: cam.stop()
    except: pass
    try:
        if _raw_cap is not None and _raw_cap.isOpened(): _raw_cap.release()
    except: pass
    cv2.destroyAllWindows()
    try: p.disconnect()
    except: pass 
    print("  Disconnected.")