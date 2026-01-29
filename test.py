import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

print("Sim is running. Close the window to exit.")
while p.isConnected():
    p.stepSimulation()
    time.sleep(1./240.)