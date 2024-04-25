from tqdm import tqdm
import torch
import cv2
import numpy as np
import pandas as pd
from mpc import mpc_control_loop, create_curve_tensor
from utils import load_model
import time
from PIL import Image
import imageio
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def run_tests(model, target_path, scaler_mean,scaler_std,N=5,epochs=100, initial_points=10,model_name='model'):
    default = target_path
    all_accuracy = []
    for _ in tqdm(range(100)):
        random = np.random.randint(1,300)
        target_path = default + f'robot_{random}.png'
        target = cv2.imread(target_path,cv2.IMREAD_GRAYSCALE)
        target = torch.from_numpy(target).float()/255
        target= target.to(device)
        best_accuracy = 0
        best_coordinates = None

        for _ in range(initial_points):
            coordinates, accuracy_hist = mpc_control_loop(model,target,scaler_mean,scaler_std,N=N,epochs=1)
            accuracy = accuracy_hist[-1]['f1-score']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_coordinates = coordinates
        
        coordinates, accuracy_hist = mpc_control_loop(model,target,scaler_mean,scaler_std,N=N,epochs=epochs,initial_coords=best_coordinates)
        for acc in accuracy_hist:
            acc['target'] = random
        all_accuracy.extend(accuracy_hist)
    pd.DataFrame(all_accuracy).to_csv(f'results/transfer_learning/mpc_tests/accuracy_{model_name}.csv',index=False)

def run_efficiency_tests(target_path,model,scaler_mean,scaler_std,N=5,epochs=100, initial_points=10):
    default = target_path
    all_accuracy = []
    for _ in tqdm(range(100)):
        random = np.random.randint(1,300)
        target_path = default + f'robot_{random}.png'
        target = cv2.imread(target_path,cv2.IMREAD_GRAYSCALE)
        target = torch.from_numpy(target).float()/255
        target= target.to(device)
        best_accuracy = 0
        best_coordinates = None
        
        for repeat in range(2):
            global_time = time.perf_counter()
            for _ in range(initial_points):
                coordinates, accuracy_hist = mpc_control_loop(model,target,scaler_mean,scaler_std,N=N,epochs=1)
                accuracy = accuracy_hist[-1]['f1-score']
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_coordinates = coordinates
            global_time = time.perf_counter() - global_time

            mpc_time = time.perf_counter()  
            coordinates, accuracy_hist = mpc_control_loop(model,target,scaler_mean,scaler_std,N=N,epochs=epochs,initial_coords=best_coordinates)
            mpc_time = time.perf_counter() - mpc_time
            for acc in accuracy_hist:
                acc['repeat'] = repeat
                acc['target'] = random
                acc['global_search_time'] = global_time
                acc['mpc_time'] = mpc_time
            all_accuracy.extend(accuracy_hist)
    pd.DataFrame(all_accuracy).to_csv('results/Timings/accuracy.csv',index=False)

def run_efficiency_tests_n(target_path, model, scaler_mean, scaler_std, N_values=[3, 5, 10], epochs=100, initial_points=10):
    default = target_path
    all_accuracy = []
    for N in N_values:  # Loop over different values of N
        for _ in tqdm(range(100), desc=f"Testing for N={N}"):
            random = np.random.randint(1, 300)
            specific_target_path = default + f'robot_{random}.png'
            target = cv2.imread(specific_target_path, cv2.IMREAD_GRAYSCALE)
            target = torch.from_numpy(target).float() / 255
            target = target.to(device)
            best_accuracy = 0
            best_coordinates = None

            for repeat in range(2):  # With and without early stopping
                global_time = time.perf_counter()
                for _ in range(initial_points):
                    coordinates, accuracy_hist = mpc_control_loop(model, target, scaler_mean, scaler_std, N=N, epochs=1)
                    accuracy = accuracy_hist[-1]['f1-score']
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_coordinates = coordinates
                global_time = time.perf_counter() - global_time
                
                mpc_time = time.perf_counter()
                coordinates, accuracy_hist = mpc_control_loop(model, target, scaler_mean, scaler_std, N=N, epochs=epochs, initial_coords=best_coordinates)
                mpc_time = time.perf_counter() - mpc_time
                
                for acc in accuracy_hist:
                    acc['repeat'] = repeat
                    acc['target'] = random
                    acc['N'] = N
                    acc['global_search_time'] = global_time
                    acc['mpc_time'] = mpc_time
                all_accuracy.extend(accuracy_hist)

    pd.DataFrame(all_accuracy).to_csv('results/Timings/efficiency_vs_N.csv', index=False)

def run(target_path: str, model: torch.nn.Module, scaler_mean: list, scaler_std: list, N: int = 5, epochs: int = 100, initial_points: int = 10,model_name='baseline')-> None:
    """
    Run the MPC control loop.
    
    Args:
        target_path (str): The path to the target image.
        model (torch.nn.Module): The model to use.
        scaler_mean (List): The mean of the training data.
        scaler_std (List): The standard deviation of the training data.
        N (int): The number of random coordinates to use.
        epochs (int): The number of epochs to run.
        initial_points (int): The number of random points to use in global search.
    """
    target = cv2.imread(target_path,cv2.IMREAD_GRAYSCALE)
    target = torch.from_numpy(target).float()/255
    target= target.to(device)
    best_accuracy = 0
    best_coordinates = None
    for i in range(initial_points):
        coordinates, accuracy_hist = mpc_control_loop(model,target,scaler_mean,scaler_std,N=N,epochs=1)
        accuracy = accuracy_hist[-1]['f1-score']
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_coordinates = coordinates
            
    coordinates, accuracy_hist = mpc_control_loop(model,target,scaler_mean,scaler_std,N=N,epochs=epochs,initial_coords=best_coordinates)
    pil_imgs =[]

    sequence = create_curve_tensor(coordinates,scaler_mean,scaler_std,device)
    print(len(sequence))
    with torch.no_grad():  # Ensure model is in inference mode
        model.eval()
        decoded_images, mu, sigma, hidden_states = model(sequence)
        decoded_images =(decoded_images > 0.5).float()
    # Ensure unnormalize function adjusts image to 0-255 range and converts to uint8 if necessary
    images = (decoded_images.detach().cpu().squeeze().numpy() * 255).astype(np.uint8)
    for img in images:
        pil_img =Image.fromarray(img)
        pil_imgs.append(pil_img)
    
    imageio.mimsave(f'results/transfer_learning/{model_name}.gif', pil_imgs, fps=5)

if __name__ == "__main__":
    scaler_mean=[1.75166666e+01, -2.18374525e-04, -2.76295627e-03]
    scaler_std=[10.12287013, 0.01665567, 0.01509156]
    target_path = '/Users/jaiva/Documents/dis-project/network/data/segmented_images/two_ribbons/0/robot_96.png'
    folder_path = '/Users/jaiva/Documents/dis-project/network/data/segmented_images/two_ribbons/9/'
    model_path = '/Users/jaiva/Documents/dis-project/results/transfer_learning/'
    model_name = 'final.pth'
    #model = load_model(model_name, device)
    #run(target_path=target_path,model=model,scaler_mean=scaler_mean,scaler_std=scaler_std,N=5,epochs=100,model_name=model_name.split('.')[0])
    for model_name in os.listdir(model_path):
        if not model_name.endswith('.pth'):
            continue
        temp_model_path = os.path.join(model_path, model_name)
        model = load_model(temp_model_path, device)

        run_tests(target_path=folder_path,model=model,scaler_mean=scaler_mean,scaler_std=scaler_std,N=5,epochs=100,model_name=model_name.split('.')[0])
    #run(target_path=target_path,model=model,scaler_mean=scaler_mean,scaler_std=scaler_std,N=5,epochs=100,model_name=model_name.split('.')[0])

    #run_efficiency_tests(target_path=folder_path,model=model,scaler_mean=scaler_mean,scaler_std=scaler_std,N=5,epochs=100)
    #run_efficiency_tests_n(target_path=folder_path, model=model, scaler_mean=scaler_mean, scaler_std=scaler_std, N_values=[1,2,3,4, 5,6], epochs=100)