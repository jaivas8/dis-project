import pandas as pd
import numpy as np
from PIL import Image
import os
import cv2

def process_time(df):


    # Calculate the time deltas as the difference between entries
    """df['Timestamp'] = df['Timestamp'].diff()
    df.loc[0,'Timestamp']=0"""
    df = df.round(5)
    return df

def process_image(image_path,image_name):
    image = Image.open(f'{image_path}/{image_name}')
    # Define the area to crop (left, upper, right, lower)
    crop_area = (291, 0, 1371, 1080)
    new_size = (256, 256)
    # Crop the image
    cropped_image = image.crop(crop_area)
    resized_image = cropped_image.resize(new_size)
    resized_image.save(f'network/data/processed_images/{image_name}')



def extrapolate_time():
    def get_picture_from_time(df,time):
        num = int(df[df['Timestamp']==time]['photo_num'])
        return cv2.imread(f'network/data/processed_images/robot_{num}.png')
    # Load your dataset
    file_path = 'network/data/robot_movement_data.csv'  
    data = pd.read_csv(file_path)

    # Define the new time interval (1/30 seconds)
    time_interval = 1 / 30

    # Create a new timestamp series starting from the minimum timestamp in the original data
    # and ending at the maximum timestamp, with a step of 0.0333 seconds
    new_timestamps = np.arange(data['Timestamp'].min(), data['Timestamp'].max(), time_interval)

    # Interpolating the values of robot_x and robot_y at these new timestamps
    interpolated_x = np.interp(new_timestamps, data['Timestamp'], data['robot_x'])
    interpolated_y = np.interp(new_timestamps, data['Timestamp'], data['robot_y'])
    closest_times = [(max(data[data['Timestamp']<=timestamp]['Timestamp']), min(data[data['Timestamp']>timestamp]['Timestamp'])) for timestamp in new_timestamps]
    
    for i in range(len(closest_times)):
        min_time, max_time = closest_times[i]
        new_time = new_timestamps[i]
        weight1 = (max_time - new_time) / (max_time - min_time)
        weight2 = 1-weight1
        image1 = get_picture_from_time(data, min_time)
        image2 = get_picture_from_time(data, max_time)
        blended_image = cv2.addWeighted(image1, weight1, image2, weight2, 0)
        gray_image = cv2.cvtColor(blended_image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'network/data/blended_image/robot_{i}.png',gray_image)
    # Creating a new DataFrame with the interpolated values
    interpolated_data = pd.DataFrame({
        'Timestamp': new_timestamps,
        'robot_x': interpolated_x,
        'robot_y': interpolated_y,
        'photo_num': [i for i in range(len(closest_times))]
    })
    interpolated_data.to_csv('network/data/updated_robot_data.csv', index=False)