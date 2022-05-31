import logging
import os
import time

import numpy as np
import pybullet as pb
import pybullet_data
import tacto  # import TACTO
from robot import Robot
from utils import Camera, get_forces

# physicsClient = pb.connect(pb.DIRECT)
physicsClient = pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
pb.setGravity(0, 0, -9.81)  # Major Tom to planet Earth

# Initialize digits
digits = tacto.Sensor(width=240, height=320, visualize_gui=True)

# Camera position
pb.resetDebugVisualizerCamera(
    cameraDistance=0.6,
    cameraYaw=15,
    cameraPitch=-20,
    # cameraTargetPosition=[-1.20, 0.69, -0.77],
    cameraTargetPosition=[0.5, 0, 0.08],
)

planeId = pb.loadURDF("plane.urdf")  # Create plane TODO: is this the place where to put the table?

robotURDF = "setup/robots/sawyer_wsg50.urdf"
# robotURDF = "robots/wsg50.urdf"
robotID = pb.loadURDF(robotURDF, useFixedBase=True)
rob = Robot(robotID)

cam = Camera()
color, depth = cam.get_image()

rob.go(rob.pos, wait=True)

sensorLinks = rob.get_id_by_name(
    ["joint_finger_tip_left", "joint_finger_tip_right"]
)  # [21, 24]
digits.add_camera(robotID, sensorLinks)

# Add object to pybullet and tacto simulator
urdfObj = "setup/objects/cube_small.urdf"
globalScaling = 0.6
objStartPos = [0.50, 0, 0.05]
objStartOrientation = pb.getQuaternionFromEuler([0, 0, np.pi / 2])

objID = digits.loadURDF(
    urdfObj, objStartPos, objStartOrientation, globalScaling=globalScaling
)

sensorID = rob.get_id_by_name(["joint_finger_tip_right", "joint_finger_tip_left"])


# Now I want to test the rendering capabilities
gripForce = 20

color, depth = digits.render()
digits.updateGUI(color, depth)



print()

