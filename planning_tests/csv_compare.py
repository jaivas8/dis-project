import pandas as pd

csv1 = pd.read_csv("robot_movement_data.csv")
csv2 = pd.read_csv("robot_movement_data_1.csv")
print(sum(csv1['robot_x']-csv2['robot_x'])/len(csv1['robot_x']))
print(sum(csv1['robot_y']-csv2['robot_y'])/len(csv1['robot_y']))