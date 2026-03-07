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

## Physics Environment init (plane + gravity)
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)
plane_id = p.loadURDF("plane.urdf")

## Load Table
tabel_pos = [0, -0.5, 0.35]
table = create_table(tabel_pos)

# Load Board
chess_board = create_chess_board([0,-0.5,0.37])

# Load Queens
a_queen = create_piece([0, -0.5, 0.4], 0.08, 0.025, [0.25,0,1,1])


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