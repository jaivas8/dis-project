import pandas as pd
import numpy as np
from PIL import Image
import os
import cv2
from tqdm import tqdm
def process_time(df):
    """
    Process the time column in the DataFrame by calculating the time deltas.

    Args:
        df (pd.DataFrame): The DataFrame containing the robot data.

    Returns:
        pd.DataFrame: The DataFrame with the time deltas calculated.
    """


    # Calculate the time deltas as the difference between entries
    df['Timestamp'] = df['Timestamp'].diff()
    df.loc[0,'Timestamp']=0
    df = df.round(5)
    return df

def process_image(image_path,image_name):
    """
    Process the image by cropping and resizing it.
    
    Args:
        image_path (str): The path to the image.
        image_name (str): The name of the image.
    """
    image = Image.open(f'{image_path}/{image_name}')
    # Define the area to crop (left, upper, right, lower)
    crop_area = (291, 0, 1371, 1080)
    new_size = (256, 256)
    # Crop the image
    cropped_image = image.crop(crop_area)
    resized_image = cropped_image.resize(new_size)
    resized_image.save(f'network/data/processed_images/{image_name}')

# Function to apply mask and perform segmentation
def segment_image(image, x1, y1, x2, y2):
    """
    Segment the image using a mask and thresholding.
    
    Args:
        image (np.ndarray): The input image.
        x1 (int): The x-coordinate of the top-left corner of the ROI.
        y1 (int): The y-coordinate of the top-left corner of the ROI.
        x2 (int): The x-coordinate of the bottom-right corner of the ROI.
        y2 (int): The y-coordinate of the bottom-right corner of the ROI.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define and apply ROI mask
    mask = np.zeros_like(gray)
    cv2.rectangle(mask, (x1, y1), (x2, y2), (255), thickness=-1)
    roi_gray = cv2.bitwise_and(gray, mask)

    # Perform segmentation
    _, segmented = cv2.threshold(roi_gray, 85, 255, cv2.THRESH_BINARY)
    segmented = cv2.medianBlur(segmented, 3)

    # Invert the mask for the unsegmented section to be white
    mask_inv = cv2.bitwise_not(mask)  # Invert the mask: ROI becomes black, rest becomes white
    segmented_inv_mask = cv2.bitwise_or(segmented, mask_inv)  # Apply inverted mask to the segmented image
    flipped = cv2.bitwise_not(segmented_inv_mask)
    return flipped



def blended_segmented_images(image_sub_sub_folder, x1, y1, x2, y2):
    """
    Segment the images in the specified folder and save the segmented images in a new folder.
    
    Args:
        image_sub_sub_folder (str): The path to the folder containing the images.
        x1 (int): The x-coordinate of the top-left corner of the ROI.
        y1 (int): The y-coordinate of the top-left corner of the ROI.
        x2 (int): The x-coordinate of the bottom-right corner of the ROI.
        y2 (int): The y-coordinate of the bottom-right corner of the ROI.
    """
    for ribbons in os.listdir(image_sub_sub_folder):
        for folder in tqdm(os.listdir(f'{image_sub_sub_folder}/{ribbons}')):
            image_folder_path = os.path.join(image_sub_sub_folder, ribbons, folder)
            image_paths = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if f.endswith('.png')]
            
            for image_path in image_paths:
                image = cv2.imread(image_path)
                segmented = segment_image(image, x1, y1, x2, y2)
                
                output_folder_path = f'network/data/segmented_images/{ribbons}/{folder}'
                if not os.path.exists(output_folder_path):
                    os.makedirs(output_folder_path)
                
                # Extract the filename from the image_path and construct the output path
                image_filename = os.path.basename(image_path)
                output_path = os.path.join(output_folder_path, image_filename)
                
                cv2.imwrite(output_path, segmented)

def extrapolate_time(file_path,ribbons,index):
    """
    Extrapolate the time for the images and interpolate the robot_x and robot_y values.
    
    Args:
        file_path (str): The path to the file containing the robot data.
        ribbons (str): The type of ribbons.
        index (int): The index of the file.
        
    """

    def get_picture_from_time(df,time):
        """
        Get the picture from the time
        
        Args:
            df (pd.DataFrame): The DataFrame containing the robot data.
            time (float): The time to get the picture from.
            
        Returns:
            np.ndarray: The image at the time.
        """
        num = int(df[df['Timestamp']==time]['photo_num'])
        return cv2.imread(f'network/data/processed_images/{ribbons}/robot_{num}.png')

    data = pd.read_csv(file_path)
    time_interval = 1 / 30

    # Create a new timestamp series starting from the minimum timestamp in the original data
    # and ending at the maximum timestamp, with a step of 0.0333 seconds
    new_timestamps = np.arange(data['Timestamp'].min(), data['Timestamp'].max(), time_interval)

    # Interpolating the values of robot_x and robot_y at these new timestamps
    interpolated_x = np.interp(new_timestamps, data['Timestamp'], data['robot_x'])
    interpolated_y = np.interp(new_timestamps, data['Timestamp'], data['robot_y'])
    closest_times = [(max(data[data['Timestamp']<=timestamp]['Timestamp']), min(data[data['Timestamp']>timestamp]['Timestamp'])) for timestamp in new_timestamps]
    if not os.path.exists(f'network/data/blended_image/{ribbons}/{index}'):
        os.makedirs(f'network/data/blended_image/{ribbons}/{index}')
    for i in range(len(closest_times)):
        min_time, max_time = closest_times[i]
        new_time = new_timestamps[i]
        weight1 = (max_time - new_time) / (max_time - min_time)
        weight2 = 1-weight1
        image1 = get_picture_from_time(data, min_time)
        image2 = get_picture_from_time(data, max_time)
        blended_image = cv2.addWeighted(image1, weight1, image2, weight2, 0)
        gray_image = cv2.cvtColor(blended_image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'network/data/blended_image/{ribbons}/{index}/robot_{i}.png',gray_image)
    # Creating a new DataFrame with the interpolated values
    interpolated_data = pd.DataFrame({
        'Timestamp': new_timestamps,
        'robot_x': interpolated_x,
        'robot_y': interpolated_y,
        'photo_num': [i for i in range(len(closest_times))]
    })
    interpolated_data.to_csv(f'network/data/updated_robot_data_{index}.csv', index=False)


def perform_proccesing():
    """
    entry point for the preprocessing
    """
    for index in tqdm(range(0,20)):

        file_path = f'data_acquisition/robot_movement_data_{index*1000}.csv'
        ribbons = "two_ribbons" if index < 10 else "one_ribbon"
        extrapolate_time(file_path,ribbons, index)


def calculate_average_x_of_white_pixels(image):
    """
    Calculate the average x-coordinate of white pixels in the image.

    Args:
        image (np.ndarray): The input image.

    Returns:
        float: The average x-coordinate of white pixels.
    """
    white_pixels = np.where(np.all(image == [255, 255, 255], axis=-1))
    average_x = np.mean(white_pixels[1]) if white_pixels[1].size > 0 else 0
    return average_x

def shift_image(image, x_shift):
    """
    Shift an image by a specified amount in the x-direction.

    Args:
        image (np.ndarray): The input image.
        x_shift (int): The amount to shift the image in the x-direction.

    Returns:
        np.ndarray: The shifted image.
    """
    rows, cols = image.shape[:2]
    translation_matrix = np.float32([[1, 0, x_shift], [0, 1, 0]])
    shifted_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    return shifted_image

def convert_to_black_and_white(image):
    """
    Convert an image to black and white.

    Args:
        image (np.ndarray): The input image.

    Returns:
        np.ndarray: The black and white image.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, black_and_white_image) = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return black_and_white_image


def shift(folder_path='network/data/segmented_images/two_ribbons/'):
    """
    Shift the segmented images in the specified folder to align the ribbons.

    Args:
        folder_path (str): The path to the folder containing the segmented images.
    """
    
    directories = sorted(os.listdir(folder_path))
    first_image_avg_x = None

    # Calculate average x for the first image in the first directory
    first_dir = os.path.join(folder_path, directories[0])
    first_image_path = os.path.join(first_dir, sorted(os.listdir(first_dir))[0])
    first_image = cv2.imread(first_image_path)
    first_image_avg_x = calculate_average_x_of_white_pixels(first_image)

    # Process each directory starting from the second one
    for dir in directories[1:]: 
        print(f"Processing directory: {dir}")
        image_sub_folder = os.path.join(folder_path, dir)
        
        # Calculate shift needed based on the first image of the current directory
        first_image_path_current_dir = os.path.join(image_sub_folder, sorted(os.listdir(image_sub_folder))[0])
        first_image_current_dir = cv2.imread(first_image_path_current_dir)
        current_dir_first_image_avg_x = calculate_average_x_of_white_pixels(first_image_current_dir)
        x_shift = first_image_avg_x - current_dir_first_image_avg_x
        
        # Apply this shift to all images in the current directory
        for filename in sorted(os.listdir(image_sub_folder)):
            image_path = os.path.join(image_sub_folder, filename)
            image = cv2.imread(image_path)
            shifted_image = shift_image(image, x_shift)
            
            # Convert to black and white
            bw_image = convert_to_black_and_white(shifted_image)
            
            # Override the original image with the shifted black and white image
            cv2.imwrite(image_path, bw_image)

min_len = 1052
def time_fix():
    for index in tqdm(range(0,20)):

        file_path = f'network/data/updated_robot_data_{index}.csv'
        data = pd.read_csv(file_path)
        data = process_time(data)
        print(data)
        data.to_csv(f'network/data/updated_robot_data_{index}.csv', index=False)

def min_len():
    """
    Find the minimum length of the data
    
    Returns:
        pd.DataFrame: The DataFrame with the minimum length
        int: The minimum length
    """
    min_len =100000
    min_df =None
    for index in tqdm(range(0,20)):
        df = pd.read_csv(f'network/data/updated_robot_data_{index}.csv')
        if len(df) < min_len:
            min_len = len(df)
            min_df = df
    return min_df, min_len
def avg_time_end():
    """
    Calculate the average time taken by the robot to reach the end of the path.
    
    Returns:
        float: The average time taken by the robot to reach the end of the path.
    """
    time = 0
    for index in tqdm(range(0,20)):
        df = pd.read_csv(f'network/data/updated_robot_data_{index}.csv')
        time += df['Timestamp'].iloc[-1]
    return time/20
if __name__ == '__main__':
    blended_segmented_images('network/data/blended_image',  92, 0, 180, 256)
    shift(folder_path = 'network/data/segmented_images/two_ribbons/')
    shift(folder_path = 'network/data/segmented_images/one_ribbon/')