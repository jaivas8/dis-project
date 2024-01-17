import cv2 as cv
import time
import pandas as pd
import URBasic
import random
import math3d as m3d
import math
import time
ROBOT_IP = '169.254.76.5' 
ACCELERATION = 0.5  # Robot acceleration value
VELOCITY = 0.5  # Robot speed value
ROBOT_START_POSITION = (math.radians(0),
                    math.radians(-90),
                    math.radians(90),
                    math.radians(-90),
                    math.radians(-90),
                    math.radians(0))
print(ROBOT_START_POSITION)





print("initialising robot")
robotModel = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP,robotModel=robotModel)
robot.reset_error()

print("robot initialised")
time.sleep(1)
MOVEMENT_RANGE = 0.05
def random_position():
    return random.uniform(-MOVEMENT_RANGE,MOVEMENT_RANGE), random.uniform(-MOVEMENT_RANGE,MOVEMENT_RANGE)
# get current position
pos = robot.get_actual_tcp_pose()

x,y = random_position()
print(x)

pos[0] = pos[0]+x
print(pos)
robot.move(pos)    
#IK = robot.get_inverse_kin(pos,qnear=robot.get_actual_joint_positions())

#robot.servoj(IK, t =2, lookahead_time=0.04, gain=100)

