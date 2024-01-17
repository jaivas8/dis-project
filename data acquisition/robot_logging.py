import cv2 as cv
import time
import pandas as pd
import URBasic
import random
import math3d as m3d
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
random.seed(10)
MOVEMENT_RANGE = 0.04999
FILE_PATH = "images/" # File path for images
ROBOT_IP = '169.254.76.5'
ACCELERATION = 0.9  # Robot acceleration value
VELOCITY = 0.8  # Robot speed value
LOOPS = 300 # The amount of movement
START_X = 0
START_Y = 0
MAX= 0.05
# The Joint position the robot starts at

ROBOT_START_POSITION = (math.radians(-218),
                    math.radians(-63),
                    math.radians(-93),
                    math.radians(-20),
                    math.radians(88),
                    math.radians(0))

def take_picture(cam):
    """
    Function that takes a counter and outputs a image to a file

    input:
    - counter for naming the file
    - cam to take the picture 
    """
   
    result, image = cam.read()
    if result:
        return image
    return None



def random_position():
    """
    Returns a random poisition plus/minus Movement range from the start point
    """
    return START_X + random.uniform(-MOVEMENT_RANGE,MOVEMENT_RANGE), START_Y+ random.uniform(-MOVEMENT_RANGE,MOVEMENT_RANGE)


def bezier_curve(t, p0, p1, p2, p3):
    """
        Curve used for the movement of the robot. Allows for the movement to be
         curved while also allowing for control of how big the curve is and making sure it doesnt exceed the movement range box
    """
    return (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3


def generate_smooth_curve(point1, point2, control_points):
    """
        The smooth path that the robot will follow
        Point1 and 2 are the starting and end points
        The control points are what cause the curveture but also keep the robot bounded between the points
    """
    t = np.linspace(0, 1, 10)
    x = bezier_curve(t, point1[0], control_points[0][0], control_points[1][0], point2[0])
    y = bezier_curve(t, point1[1], control_points[0][1], control_points[1][1], point2[1])
    return x, y


def move_to_position(robot,x,y):
    """
    Moves the robot to the position required
    
    input:
    - robot object
    - x and y position
    """
    pos = robot.get_actual_tcp_pose()
    if not check_max(x,y):
        raise ValueError("Number not in range")
    pos[0] = x
    pos[1] = y
    pos[2] = START_Z
    robot.set_realtime_pose(pos) 

def check_max(x,y):
    """
    Checks that the coordinates do to not exceed the bounding box

    input:
    - x and y: position of movement
    """
    if x > (START_X +MAX) or x < (START_X-MAX):
        print("Failed check x")
        return False
    if y >(START_Y + MAX) or y<(START_Y - MAX):
        print("Failed check y")
        return False
    return True


def log_movement(robot):
    """
    logs the movement but also starts the movement chain

    input:
    - robot object
    
    """
    avg_time = []
    avg_mov_time = []


    start_time = time.time()
    timestamps = [start_time]

 

    cam = cv.VideoCapture(0)
    pos = robot.get_actual_tcp_pose()
    robot_xs = [pos[0]-START_X]
    robot_ys = [pos[1]-START_Y]
    photos = [take_picture(cam)]
    


    
    starting_point = START_X,START_Y
    counter =0
    while counter < LOOPS:
        # Randomly picks which corner the control points will be located for the specific curve
        if random.randint(0,1):

            control_point1 = (START_X+MAX, START_Y+MAX)
            control_point2 = (START_X-MAX, START_Y-MAX)
        else:
            control_point1 = (START_X-MAX, START_Y+MAX)
            control_point2 = (START_X+MAX, START_Y-MAX)
        
        end_point = random_position()

        # Generate the smooth curve
        xs, ys =  generate_smooth_curve(starting_point, end_point, [control_point1, control_point2])

        """# Following code is used to demonstrate the path that the robot will follow

        plt.plot(xs, ys, label='Smooth Curve')
        plt.scatter([starting_point[0], end_point[0]], [starting_point[1], end_point[1]], color='red', label='End Points')
        plt.scatter([control_point1[0], control_point2[0]], [control_point1[1], control_point2[1]], color='green', label='Control Points')
        plt.legend()
        plt.show()"""

        print(f"Start: {starting_point} End: {end_point}")
        print(f"Counter: {counter}")
        # Loops through the curve path xs, ys
        for x,y in zip(xs,ys):
            #print("coordinates:",x,y)
            # Moves to the position
            start_move = time.time()
            move_to_position(robot,x,y)
            avg_mov_time.append(time.time()-start_move)
            robot_position = robot.get_actual_tcp_pose()

            # Save the frame as a photo
            start_pic = time.time()
            image = take_picture(cam)
            
            #Add data to the DataFrame
            timestamps.append(time.time()-start_time)
            robot_xs.append(robot_position[0]-START_X)
            robot_ys.append(robot_position[1]-START_Y)
            
            photos.append(image)
            # Increment the photo counter
            avg_time.append(time.time()-start_pic)
        starting_point = end_point
        counter +=1
    
    

    
    xs, ys =  generate_smooth_curve(starting_point, (START_X,START_Y), [control_point1, control_point2])
    for x,y in zip(xs,ys):
        move_to_position(robot,x,y)
        time.sleep(0.1)
    photo_num=[]
    photo_counter = 0
    for image in tqdm(photos):
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        cv.imwrite(f"{FILE_PATH}/robot_{photo_counter}.png", image)
        photo_num.append(photo_counter)
        photo_counter+=1
        
    data= pd.DataFrame({"Timestamp":timestamps,"robot_x":robot_xs,"robot_y":robot_ys,"photo_num":photo_num})
    data.to_csv("robot_movement_data.csv", index=False)
    print(f"Avg_time for log: {sum(avg_time)/len(avg_time)}")
    print(f"Avg_time for movement: {sum(avg_mov_time)/len(avg_mov_time)}")
    # Release the webcam and close the data logging process
    cam.release()



def run():
    # initialise robot with URBasic
    print("initialising robot")
    robotModel = URBasic.robotModel.RobotModel()
    robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP,robotModel=robotModel)
    robot.reset_error()
    print("robot initialised")
    time.sleep(1)

    # get current position
    start = robot.get_actual_tcp_pose()
    print(robot.get_actual_tcp_pose())
    """
    # Used for robot testing when not in HereEast
    start = [-0.3278, -0.0936,  0.4363,    0.1991, -3.1020, -0.0460]
    robot = 0

    """
    global START_X 
    START_X = start[0]
    global START_Y 
    START_Y= start[1]
    global START_Z
    START_Z = start[2]
    print(f"Starting coordinates: ({START_X},{START_Y})")
    # Move Robot to the midpoint of the lookplane
    #robot.movej(q=ROBOT_START_POSITION, a= ACCELERATION, v= VELOCITY )


    robot.init_realtime_control()  # starts the realtime control loop on the Universal-Robot Controller

    time.sleep(1) # just a short wait to make sure everything is initialised
    log_movement(robot)
    print("done")
run()
