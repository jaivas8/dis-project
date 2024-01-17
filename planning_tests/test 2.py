import cv2 as cv
import time
import pandas as pd
import URBasic
import random
# import math3d as m3d
import math

ROBOT_IP = '169.254.76.5' 
ACCELERATION = 0.5  # Robot acceleration value
VELOCITY = 0.5  # Robot speed value
'''ROBOT_START_POSITION = (math.radians(0),
                    math.radians(-90),
                    math.radians(90),
                    math.radians(-90),
                    math.radians(-90),
                    math.radians(13))
'''
print("initialising robot")
robotModel = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP,robotModel=robotModel)
robot.init_realtime_control()
robot.reset_error()
print("robot initialised")
time.sleep(1)
FILE_PATH = "images/" # File path for images
def take_picture(cam, counter):
    """
    Function that takes a counter and outputs a image to a file

    input:
    - counter for naming the file
    - cam to take the picture 
    """
   
    result, image = cam.read()
    if result:

        cv.imwrite(f"{FILE_PATH}/robot_{counter}.png", image)
# get current position
# print(robot.get_actual_tcp_pose())
start_pos = robot.get_actual_tcp_pose()
print(robot.get_actual_tcp_pose())

# move circle
radius = 0.05
cam = cv.VideoCapture(0)
start = time.time()
for theta in range(0, 360,1):
    i = theta/20
    rad_theta = (2*theta*math.pi)/360
    x = radius * (math.cos(rad_theta))
    y = radius * (math.sin(rad_theta))
    pos = robot.get_actual_tcp_pose()
    pos[0] = start_pos[0] + x
    pos[1] = start_pos[1] + y

    
    print(pos[0],pos[1])
    robot.set_realtime_pose(pos) 
    #take_picture(cam,i)
print("Time taken:", time.time()-start)
print(pos)
print(robot.get_actual_tcp_pose())
print("moved done")
cam.release()