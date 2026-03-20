"""
supervisory_control_nqueens.py  — AR Edition
N-Queens HRI SUPERVISORY CONTROL — all original logic preserved,
AR layers added on top.
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
#  WORLD HELPERS  (unchanged)
# ─────────────────────────────────────────────

def create_table(position):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.75, 0.3, 0.02])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.75, 0.3, 0.02],
                              rgbaColor=[0.55, 0.27, 0.07, 1])
    return p.createMultiBody(0, col, vis, position)

def create_chess_board(board_pos):
    sqr = 0.06; h = 0.005; mid = sqr * 4
    for r in range(8):
        for c in range(8):
            color = [1,1,1,1] if (r+c)%2==0 else [0.1,0.1,0.1,1]
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[sqr/2,sqr/2,h/2], rgbaColor=color)
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sqr/2,sqr/2,h/2])
            x = r*sqr + board_pos[0] - (mid - sqr/2)
            y = c*sqr + board_pos[1] - (mid - sqr/2)
            p.createMultiBody(0, col, vis, [x, y, board_pos[2]+h/2])

def create_piece(position, color=[0.25, 0, 1, 1]):
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.022, height=0.06)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.022, length=0.06, rgbaColor=color)
    return p.createMultiBody(0.1, col, vis, position)

def board_to_world(row, col, board_pos, sqr=0.06, n=8):
    mid = sqr*(n/2)
    x = row*sqr + board_pos[0] - (mid - sqr/2)
    y = col*sqr + board_pos[1] - (mid - sqr/2)
    return [x, y, board_pos[2] + 0.03]

# ─────────────────────────────────────────────
#  ROBOT HELPERS  (unchanged)
# ─────────────────────────────────────────────

def ik_move(robot, pos):
    orn = p.getQuaternionFromEuler([math.pi, 0, 0])
    joints = p.calculateInverseKinematics(robot, 11, pos, orn)
    for i in range(7):
        p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, joints[i], force=700)

def step(n=50):
    for _ in range(n):
        p.stepSimulation()
        time.sleep(1/240)

def pick_and_place(robot, src, dst, gripper, piece_id):
    LIFT = 0.28
    ik_move(robot, [src[0], src[1], src[2]+LIFT]);  step(60)
    ik_move(robot, [src[0], src[1], src[2]+0.04]);  step(60)
    gripper.grasp([piece_id]);                       step(30)
    if gripper.held is None:
        print("  [WARN] Grasp missed.")
    ik_move(robot, [src[0], src[1], src[2]+LIFT]);  step(60)
    ik_move(robot, [dst[0], dst[1], dst[2]+LIFT]);  step(60)
    ik_move(robot, [dst[0], dst[1], dst[2]+0.04]);  step(60)
    gripper.release();                               step(30)
    p.resetBasePositionAndOrientation(piece_id, dst, [0,0,0,1])
    p.resetBaseVelocity(piece_id, [0,0,0], [0,0,0])
    ik_move(robot, [dst[0], dst[1], dst[2]+LIFT]);  step(60)

class Gripper:
    def __init__(self, robot, ee):
        self.robot=robot; self.ee=ee; self.cid=None; self.held=None; self.last_released=None
    def grasp(self, objects, thresh=0.06):
        if self.cid: return
        ee_pos = p.getLinkState(self.robot, self.ee)[0]
        closest, best_d = None, thresh
        for obj in objects:
            pos, _ = p.getBasePositionAndOrientation(obj)
            d = math.dist(ee_pos, pos)
            if d < best_d: best_d, closest = d, obj
        if closest:
            self.cid = p.createConstraint(self.robot,self.ee,closest,-1,
                                          p.JOINT_FIXED,[0,0,0],[0,0,0.04],[0,0,0])
            self.held = closest
    def release(self):
        if self.cid:
            p.removeConstraint(self.cid)
            self.cid=None; self.last_released=self.held; self.held=None

# ─────────────────────────────────────────────
#  GESTURE DETECTION  (returns landmarks too)
# ─────────────────────────────────────────────

mp_hands = mp.solutions.hands
hand_det = mp_hands.Hands(min_detection_confidence=0.75, max_num_hands=1)
draw_u   = mp.solutions.drawing_utils

def detect_gesture(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hand_det.process(rgb)
    if not res.multi_hand_landmarks:
        return None, None
    lm = res.multi_hand_landmarks[0].landmark
    draw_u.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
    if lm[4].y < lm[5].y - 0.04:
        return 'approve', res.multi_hand_landmarks[0]
    tips=[8,12,16,20]; pips=[6,10,14,18]
    if all(lm[t].y < lm[p].y for t,p in zip(tips,pips)):
        return 'reject', res.multi_hand_landmarks[0]
    return None, res.multi_hand_landmarks[0]

# ═══════════════════════════════════════════════════════
#  AR — PYBULLET BOARD HIGHLIGHTS
# ═══════════════════════════════════════════════════════

_hl_ids  = {}
_dot_ids = {}
_hud_ids = {}

def _hl_pos(row, col, board_pos, sqr=0.06, n=8):
    mid = sqr*(n/2)
    return [row*sqr+board_pos[0]-(mid-sqr/2),
            col*sqr+board_pos[1]-(mid-sqr/2),
            board_pos[2]+0.007]

def init_board_ar(board_pos):
    half = [0.024, 0.024, 0.001]
    for key in ['src','dst']:
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0,0,0,0])
        _hl_ids[key] = p.createMultiBody(0,-1,vis,[0,0,-5])

def _show_hl(key, pos, rgba):
    bid = _hl_ids.get(key)
    if bid is None: return
    p.resetBasePositionAndOrientation(bid, pos, [0,0,0,1])
    p.changeVisualShape(bid,-1,rgbaColor=rgba)

def _hide_hl(key):
    bid = _hl_ids.get(key)
    if bid is None: return
    p.resetBasePositionAndOrientation(bid,[0,0,-5],[0,0,0,1])

def set_move_highlights(queen_row, src_col, dst_col, board_pos, t):
    a_src = 0.50 + 0.35*math.sin(t*4)
    a_dst = 0.55 + 0.32*math.sin(t*4+math.pi)
    _show_hl('src', _hl_pos(queen_row,src_col,board_pos), [0.1,0.4,1.0,a_src])
    _show_hl('dst', _hl_pos(queen_row,dst_col,board_pos), [1.0,0.78,0.0,a_dst])

def hide_move_highlights():
    _hide_hl('src'); _hide_hl('dst')

def add_placed_ar(queen_row, dst_col, board_pos):
    if queen_row in _dot_ids:
        try: p.removeBody(_dot_ids[queen_row])
        except: pass
    half=[0.015,0.015,0.002]
    vis=p.createVisualShape(p.GEOM_BOX,halfExtents=half,rgbaColor=[0.1,1.0,0.3,0.85])
    pos=_hl_pos(queen_row,dst_col,board_pos); pos[2]+=0.003
    _dot_ids[queen_row]=p.createMultiBody(0,-1,vis,pos)

def _txt(key, txt, pos, rgb, sz=1.0):
    if key in _hud_ids:
        try: p.removeUserDebugItem(_hud_ids[key])
        except: pass
    _hud_ids[key]=p.addUserDebugText(txt,pos,textColorRGB=rgb,textSize=sz,lifeTime=0)

def draw_pybullet_hud(move_idx, total, queen_row, src_col, dst_col, n_placed, status, board_pos):
    bx=board_pos[0]+0.32; by=board_pos[1]+0.18; bz=board_pos[2]+0.28; lh=0.055
    _txt("t0","N-QUEENS  |  SUPERVISORY CONTROL",[bx,by,bz+lh*5],[1.0,0.85,0.0],1.2)
    _txt("t1",f"Move {move_idx+1}/{total}   Row {queen_row}: col {src_col} -> col {dst_col}",
         [bx,by,bz+lh*3.8],[0.9,0.9,0.9],1.0)
    bar="█"*n_placed+"░"*(8-n_placed)
    _txt("t2",f"Progress: {bar}  {n_placed}/8",[bx,by,bz+lh*2.8],[0.3,1.0,0.4],0.95)
    sc=[0.2,1.0,0.2] if "Approv" in status else [1.0,0.35,0.35] if "Defer" in status else [1.0,0.85,0.1]
    _txt("t3",status,[bx,by,bz+lh*1.8],sc,1.0)
    _txt("t4","Thumbs Up = Approve",[bx,by,bz+lh*0.8],[0.5,1.0,0.5],0.85)
    _txt("t5","Open Palm = Defer",  [bx,by,bz+lh*0.2],[1.0,0.5,0.5],0.85)
    _txt("t6","SPACE = Force approve",[bx,by,bz-lh*0.4],[0.7,0.8,1.0],0.80)

def clear_pybullet_hud():
    for v in _hud_ids.values():
        try: p.removeUserDebugItem(v)
        except: pass
    _hud_ids.clear()

# ═══════════════════════════════════════════════════════
#  AR — CV CAMERA OVERLAY
# ═══════════════════════════════════════════════════════

MINI_ORIG=(20,130); MINI_SQ=26; MINI_N=8

def draw_mini_board(frame, placed_map, cur_row, src_col, dst_col):
    ox,oy=MINI_ORIG; sz=MINI_SQ*MINI_N
    cv2.rectangle(frame,(ox-3,oy-3),(ox+sz+3,oy+sz+3),(30,30,30),-1)
    for r in range(MINI_N):
        for c in range(MINI_N):
            x0=ox+c*MINI_SQ; y0=oy+r*MINI_SQ; x1=x0+MINI_SQ; y1=y0+MINI_SQ
            base=(210,210,210) if (r+c)%2==0 else (45,45,45)
            if r==cur_row and c==src_col:   base=(180,80,20)
            elif r==cur_row and c==dst_col: base=(20,160,255)
            elif r in placed_map and placed_map[r]==c: base=(30,150,55)
            cv2.rectangle(frame,(x0,y0),(x1,y1),base,-1)
            if r in placed_map and placed_map[r]==c:
                cx,cy=x0+MINI_SQ//2,y0+MINI_SQ//2
                cv2.circle(frame,(cx,cy),MINI_SQ//3-1,(255,255,255),-1)
                cv2.circle(frame,(cx,cy),MINI_SQ//3-1,(0,0,0),1)
    # Arrow src→dst on current row
    sp=(ox+src_col*MINI_SQ+MINI_SQ//2, oy+cur_row*MINI_SQ+MINI_SQ//2)
    dp=(ox+dst_col*MINI_SQ+MINI_SQ//2, oy+cur_row*MINI_SQ+MINI_SQ//2)
    cv2.arrowedLine(frame,sp,dp,(0,210,255),2,tipLength=0.4)
    cv2.rectangle(frame,(ox-3,oy-3),(ox+sz+3,oy+sz+3),(160,160,160),1)
    cv2.putText(frame,"BOARD",(ox,oy-8),cv2.FONT_HERSHEY_SIMPLEX,0.36,(180,180,180),1)

def draw_gesture_ring(frame, gesture, landmarks, ft):
    if landmarks is None: return
    h,w=frame.shape[:2]
    wrist=landmarks.landmark[0]
    cx,cy=int(wrist.x*w),int(wrist.y*h)
    pulse=int(36+13*math.sin(ft*6))
    color=(0,210,55) if gesture=='approve' else (0,55,210) if gesture=='reject' else (0,160,200)
    label="APPROVE" if gesture=='approve' else "DEFER" if gesture=='reject' else ""
    cv2.circle(frame,(cx,cy),pulse,color,2)
    cv2.circle(frame,(cx,cy),pulse+9,color,1)
    if label:
        cv2.putText(frame,label,(cx-32,cy-pulse-12),cv2.FONT_HERSHEY_DUPLEX,0.65,color,2)

def draw_progress_bar(frame, n_placed, total=8):
    h,w=frame.shape[:2]
    bx,by=20,h-22; bw,bh=w-40,12
    cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(40,40,40),-1)
    fill=int(bw*n_placed/total)
    if fill>0: cv2.rectangle(frame,(bx,by),(bx+fill,by+bh),(0,205,80),-1)
    cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(140,140,140),1)
    cv2.putText(frame,f"{n_placed}/8 queens placed",
                (bx+bw//2-58,by-5),cv2.FONT_HERSHEY_SIMPLEX,0.43,(190,190,190),1)

def draw_status_pill(frame, gesture, status_txt, ft):
    h,w=frame.shape[:2]
    pw,ph=430,38; px,py=w//2-pw//2,h-70
    if gesture=='approve':   bg=(0,90,0);   border=(0,210,55);   tc=(160,255,160)
    elif gesture=='reject':  bg=(70,0,0);   border=(0,55,210);   tc=(160,160,255)
    else:
        v=int((0.55+0.35*math.sin(ft*3))*130)
        bg=(18,18,18); border=(v,v,0);  tc=(210,210,90)
    ov=frame.copy()
    cv2.rectangle(ov,(px,py),(px+pw,py+ph),bg,-1)
    cv2.addWeighted(ov,0.75,frame,0.25,0,frame)
    cv2.rectangle(frame,(px,py),(px+pw,py+ph),border,2)
    cv2.putText(frame,f"  {status_txt}  ",(px+10,py+26),cv2.FONT_HERSHEY_DUPLEX,0.60,tc,1)

def draw_top_banner(frame, move_idx, total, queen_row, src_col, dst_col, gesture):
    h,w=frame.shape[:2]
    bc=(0,200,55) if gesture=='approve' else (0,55,200) if gesture=='reject' else (35,35,35)
    ov=frame.copy()
    cv2.rectangle(ov,(0,0),(w,80),(12,12,12),-1)
    cv2.addWeighted(ov,0.72,frame,0.28,0,frame)
    cv2.rectangle(frame,(0,0),(6,80),bc,-1)
    cv2.line(frame,(0,80),(w,80),bc,1)
    cv2.putText(frame,f"MOVE {move_idx+1}/{total}",
                (18,26),cv2.FONT_HERSHEY_DUPLEX,0.78,(255,255,90),1)
    cv2.putText(frame,f"Row {queen_row}  :  col {src_col}  ->  col {dst_col}",
                (18,52),cv2.FONT_HERSHEY_DUPLEX,0.65,(215,215,215),1)
    cv2.putText(frame,"Thumbs Up=Approve",(w-210,26),cv2.FONT_HERSHEY_SIMPLEX,0.48,(80,210,80),1)
    cv2.putText(frame,"Open Palm=Defer",  (w-210,50),cv2.FONT_HERSHEY_SIMPLEX,0.48,(80,100,210),1)
    cv2.putText(frame,"SPACE=Force",      (w-210,72),cv2.FONT_HERSHEY_SIMPLEX,0.40,(120,120,120),1)

def draw_ar_overlay(frame, move_idx, total, queen_row, src_col, dst_col,
                    gesture, landmarks, status_txt, placed_map, ft):
    draw_top_banner(frame, move_idx, total, queen_row, src_col, dst_col, gesture)
    draw_mini_board(frame, placed_map, queen_row, src_col, dst_col)
    draw_gesture_ring(frame, gesture, landmarks, ft)
    draw_progress_bar(frame, len(placed_map))
    draw_status_pill(frame, gesture, status_txt, ft)
    return frame

# ─────────────────────────────────────────────
#  PHYSICS INIT  (unchanged)
# ─────────────────────────────────────────────

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(0)
p.loadURDF("plane.urdf")
p.resetDebugVisualizerCamera(1.8, 45, -35, [0, -0.5, 0.2])
# GUI panels kept visible for supervisory mode

table_pos=[0,-0.5,0.15]
create_table(table_pos)
board_pos=[table_pos[0],table_pos[1],table_pos[2]+0.02]
create_chess_board(board_pos)
init_board_ar(board_pos)

# ─────────────────────────────────────────────
#  N-QUEENS SOLUTION  (unchanged)
# ─────────────────────────────────────────────
SOLUTION={0:4,1:2,2:0,3:6,4:1,5:7,6:5,7:3}

queen_ids={}
for row in range(8):
    pos=board_to_world(row,0,board_pos); pos[2]+=0.01
    queen_ids[row]=create_piece(pos,color=[0.3,0.0,1.0,1])

# ─────────────────────────────────────────────
#  ROBOT & GRIPPER  (unchanged)
# ─────────────────────────────────────────────
robot=p.loadURDF("franka_panda/panda.urdf",useFixedBase=True)
p.setJointMotorControl2(robot,9,p.POSITION_CONTROL,0.04)
p.setJointMotorControl2(robot,10,p.POSITION_CONTROL,0.04)
gripper=Gripper(robot,11)
# ─────────────────────────────────────────────
#  SUPERVISORY CONTROL LOOP  (unchanged logic)
# ─────────────────────────────────────────────
cap=cv2.VideoCapture(0)
if not cap.isOpened():
    print("[WARN] No webcam — press SPACE to auto-approve.")

moves=[(row,0,SOLUTION[row]) for row in range(8)]
deferred=[]; move_queue=moves[:]; done_rows=set(); placed_map={}

print("\n=== SUPERVISORY CONTROL — N-Queens (AR Edition) ===")
print("THUMBS UP -> approve  |  OPEN PALM -> defer  |  Q -> quit\n")

try:
    # Settle physics inside try so errors are caught
    for _ in range(60):
        p.stepSimulation()

    while move_queue:
        move_idx_global=8-len(move_queue)
        queen_row,src_col,dst_col=move_queue.pop(0)
        src_world=board_to_world(queen_row,src_col,board_pos)
        dst_world=board_to_world(queen_row,dst_col,board_pos)
        print(f"  Proposing: row {queen_row} col {src_col} -> col {dst_col}")
        print(f"  [Show THUMBS UP to approve / OPEN PALM to defer]")

        draw_pybullet_hud(move_idx_global,8,queen_row,src_col,dst_col,
                          len(placed_map),"Awaiting gesture...",board_pos)

        decision=None; status_txt="Awaiting gesture..."; ft=0.0

        while decision is None:
            ft += 1/60
            set_move_highlights(queen_row,src_col,dst_col,board_pos,ft)
            ret,frame=cap.read()
            if ret:
                frame=cv2.flip(frame,1)
                gesture,landmarks=detect_gesture(frame)
                if gesture=='approve':   decision='approve'; status_txt="APPROVED — Executing..."
                elif gesture=='reject': decision='reject';  status_txt="DEFERRED — Skipping..."
                else:                                          status_txt="Awaiting gesture..."
                frame=draw_ar_overlay(frame,move_idx_global,8,
                                      queen_row,src_col,dst_col,
                                      gesture,landmarks,status_txt,placed_map,ft)
                cv2.imshow("Supervisory Control — N-Queens",frame)
                draw_pybullet_hud(move_idx_global,8,queen_row,src_col,dst_col,
                                  len(placed_map),status_txt,board_pos)
            key=cv2.waitKey(1)&0xFF
            if key==ord('q'): raise KeyboardInterrupt
            if key==ord(' '): decision='approve'
            p.stepSimulation(); time.sleep(1/240)

        if decision=='approve':
            print(f"  Executing row {queen_row}: col {src_col} -> {dst_col}")
            hide_move_highlights()
            _show_hl('dst',_hl_pos(queen_row,dst_col,board_pos),[1.0,1.0,0.2,0.95])
            for _ in range(30): p.stepSimulation(); time.sleep(1/240)
            pick_and_place(robot,src_world,dst_world,gripper,piece_id=queen_ids[queen_row])
            done_rows.add(queen_row); placed_map[queen_row]=dst_col
            add_placed_ar(queen_row,dst_col,board_pos)
            hide_move_highlights()
            print(f"  Done. {len(placed_map)}/8 queens placed.")
        else:
            print(f"  Deferred row {queen_row}")
            hide_move_highlights()
            deferred.append((queen_row,src_col,dst_col))

        if not move_queue and deferred:
            print("\n--- Retrying deferred moves ---")
            move_queue=deferred[:]; deferred=[]

    print("\nAll 8 queens placed — puzzle solved!")
    clear_pybullet_hud()
    p.addUserDebugText("  PUZZLE SOLVED!  ",
        [board_pos[0],board_pos[1]+0.15,board_pos[2]+0.32],
        textColorRGB=[0.1,1.0,0.2],textSize=1.8,lifeTime=0)
    for qid in queen_ids.values():
        p.changeVisualShape(qid,-1,rgbaColor=[1,0.82,0,1])

    while p.isConnected():
        ret,frame=cap.read()
        if ret:
            frame=cv2.flip(frame,1)
            h,w=frame.shape[:2]
            ov=frame.copy()
            cv2.rectangle(ov,(0,h//2-42),(w,h//2+42),(0,55,0),-1)
            cv2.addWeighted(ov,0.7,frame,0.3,0,frame)
            cv2.putText(frame,"  PUZZLE SOLVED!  Press Q to exit  ",
                        (w//2-230,h//2+12),cv2.FONT_HERSHEY_DUPLEX,0.95,(0,235,80),2)
            draw_progress_bar(frame,8)
            cv2.imshow("Supervisory Control — N-Queens",frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break
        p.stepSimulation(); time.sleep(1/240)

except KeyboardInterrupt:
    print("\nStopped.")
except Exception as e:
    import traceback
    print(f"\n[ERROR] {e}"); traceback.print_exc()
finally:
    p.disconnect(); cap.release(); cv2.destroyAllWindows()