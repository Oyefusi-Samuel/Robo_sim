import pybullet as p
import pybullet_data
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