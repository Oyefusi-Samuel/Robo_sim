import pybullet as p
import pybullet_data
import math
import time

## WORLD PARAMS:

## Helper Functions for World Gen

# Create Table for puzzle
def create_table(position):
    halfExtents = [0.75, 0.3, 0.02]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=halfExtents,
                              rgbaColor=[0.6, 0.3, 0.1, 1])
    table = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=position
    )
    return table

# Create Chess board
def create_chess_board(board_pos):

    board_size = 8
    sqr_size = 0.06
    mid_point = sqr_size * (board_size / 2)
    height = 0.005

    for row in range(board_size):
        for col in range(board_size):
        
            if (row+col) % 2 == 0:
                color = [1,1,1,1]
            else:
                color = [0,0,0,1]

            visual = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[sqr_size/2, sqr_size/2, height/2],
                rgbaColor=color
            )

            collision = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[sqr_size/2, sqr_size/2, height/2]
            )

            x = (row * sqr_size) + board_pos[0] - (mid_point - sqr_size/2)
            y = (col * sqr_size) + board_pos[1] - (mid_point - sqr_size/2)
            z = (height/2) + board_pos[2]

            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=[x, y, z]
            )

# Create a chess peice for the N-queens problem
import pybullet as p

def create_piece(position, height, radius, color):

    collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=radius,
        height=height
    )

    visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=color
    )

    piece = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position
    )

    return piece

# convert board coordinates to world, given board position
def board_to_world(row, col, board_pos, sqr_size=0.06, board_size=8):

    mid_point = sqr_size * (board_size / 2)

    x = (row * sqr_size) + board_pos[0] - (mid_point - sqr_size/2)
    y = (col * sqr_size) + board_pos[1] - (mid_point - sqr_size/2)
    z = board_pos[2] + 0.03   # piece height offset

    return [x, y, z]

## Dynamics helpers

# Class for Gripping
class Gripper:

    def __init__(self, robot_id, ee_link):
        self.robot = robot_id
        self.ee = ee_link
        self.constraint = None
        self.held_object = None

    def grasp(self, object_ids, threshold=0.05):

        if self.constraint is not None:
            return

        ee_state = p.getLinkState(self.robot, self.ee)
        ee_pos = ee_state[0]

        for obj in object_ids:

            obj_pos, _ = p.getBasePositionAndOrientation(obj)

            dx = ee_pos[0] - obj_pos[0]
            dy = ee_pos[1] - obj_pos[1]
            dz = ee_pos[2] - obj_pos[2]

            dist = math.sqrt(dx*dx + dy*dy + dz*dz)

            if dist < threshold:

                self.constraint = p.createConstraint(
                    self.robot,
                    self.ee,
                    obj,
                    -1,
                    p.JOINT_FIXED,
                    [0,0,0],
                    [0,0,0],
                    [0,0,0]
                )

                self.held_object = obj
                break


    def release(self):

        if self.constraint is not None:
            p.removeConstraint(self.constraint)
            self.constraint = None
            self.held_object = None


def step_sim(n):

    for _ in range(n):
        p.stepSimulation()
        time.sleep(1/240)

#
def inv_kinematics(robot, target_pos):

    constraint = p.getQuaternionFromEuler([math.pi, 0, 0])

    jointPoses = p.calculateInverseKinematics(
        robot,
        11,
        target_pos,
        targetOrientation = constraint
    )
    for i in range(7):
        p.setJointMotorControl2(
            robot,
            i,
            p.POSITION_CONTROL,
            jointPoses[i],
            force=700
        )

def pick_piece(robot, piece_pos, gripper:Gripper, peiceIds):

    above = [piece_pos[0], piece_pos[1], piece_pos[2] + 0.25]
    at = [piece_pos[0], piece_pos[1], piece_pos[2] + 0.04]

    inv_kinematics(robot, above)
    step_sim(50)

    inv_kinematics(robot, at)
    step_sim(50)

    gripper.grasp(object_ids=peiceIds)
    step_sim(50)

    inv_kinematics(robot, above)
    step_sim(50)

def place_piece(robot, target_pos, gripper:Gripper):

    above = [target_pos[0], target_pos[1], target_pos[2] + 0.25]
    at = [target_pos[0], target_pos[1], target_pos[2] + 0.04]

    inv_kinematics(robot, above)
    step_sim(50)

    inv_kinematics(robot, at)
    step_sim(50)

    gripper.release()
    step_sim(50)

    inv_kinematics(robot, above)
    step_sim(50)

## Physics Environment init (plane + gravity)
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)
plane_id = p.loadURDF("plane.urdf")

## Load Table
tabel_pos = [0, -0.5, 0.15]
table = create_table(tabel_pos)

# Load Board
board_pos = [tabel_pos[0], tabel_pos[1], tabel_pos[2]+0.02]
chess_board = create_chess_board(board_pos)


# Load Queens
#a_queen = create_piece([0, -0.5, 0.4], 0.08, 0.025, [0.25,0,1,1])
queens = []
for row in range(4):
    pos = board_to_world(row+2, 0, board_pos)
    queens.append(create_piece(pos, 0.08, 0.025, [0.25, 0, 1, 1]))
for row in range(4):
    pos = board_to_world(row+2, 7, board_pos)
    queens.append(create_piece(pos, 0.08, 0.025, [0.25, 0, 1, 1]))

## Load Robot into World
# Note:
# End effector --> 11
# Fingers -------> 9 and 10
# Arm -----------> 0 to 6
boxId = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
p.setJointMotorControl2(boxId, 9, p.POSITION_CONTROL, 0.04)
p.setJointMotorControl2(boxId, 10, p.POSITION_CONTROL, 0.04)
gripper = Gripper(boxId, 11)
t = 0
#while p.isConnected:
#    p.stepSimulation()
#    time.sleep(1./240.)
#    print(t)
#    t += 1
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
start = [board_to_world(2,0,board_pos),
         board_to_world(3,0,board_pos),
         board_to_world(4,0,board_pos),
         board_to_world(5,0,board_pos),
         board_to_world(5,7,board_pos),
         board_to_world(4,7,board_pos),
         board_to_world(3,7,board_pos),
         board_to_world(2,7,board_pos)
]
targets = [board_to_world(0,4,board_pos),
           board_to_world(1,2,board_pos),
           board_to_world(2,0,board_pos),
           board_to_world(3,6,board_pos),
           board_to_world(4,1,board_pos),
           board_to_world(5,7,board_pos),
           board_to_world(6,5,board_pos),
           board_to_world(7,3,board_pos)
]
for i in range(len(start)):
    pick_piece(boxId, start[i], gripper=gripper, peiceIds=queens)
    place_piece(boxId, targets[i], gripper=gripper)
p.disconnect()