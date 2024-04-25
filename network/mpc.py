from utils import load_model, generate_smooth_curve, calculate_precision_recall_f1
import numpy as np
import torch
import cv2
import imageio
from PIL import Image
import warnings
from torch.autograd import gradcheck
warnings.filterwarnings("ignore", category=UserWarning)
torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

def bce_with_logits_loss(output: torch.Tensor, target: torch.Tensor, weight_for_positives: float = 1, weight_for_negatives: float = 1) -> torch.Tensor:
    """
    Calculate Binary Cross-Entropy (BCE) with logits loss between predictions and targets.

    Args:
        output (torch.Tensor): Predicted logits.
        target (torch.Tensor): Ground truth labels.
        weight_for_positives (float): Weight for positive class.
        weight_for_negatives (float): Weight for negative class.

    Returns:
        torch.Tensor: BCE with logits loss.
    """

    weights = torch.where(target == 1, torch.full_like(target, weight_for_positives), torch.full_like(target, weight_for_negatives))
    
    
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target, weight=weights)
    return loss.mean()

def iou_loss(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) loss between predictions and targets.
    
    Args:
        preds (torch.Tensor): Predicted segmentations.
        targets (torch.Tensor): Ground truth segmentations.
        smooth (float): A small value to avoid division by zero.
    
    Returns:
        torch.Tensor: IoU loss.
    """
    # Calculate intersection and union areas
    intersection = torch.sum(preds * targets)
    total = torch.sum(preds + targets)
    union = total - intersection
    
    # Compute IoU score
    IoU = (intersection + smooth) / (union + smooth)
    
    # Return IoU loss
    return 1 - IoU



def torch_scale(tensor: torch.Tensor, scaler_mean: list, scaler_std: list, device: torch.device) -> torch.Tensor:
    """
    scale the tensor using the mean and std of the training data

    Args:
        tensor (torch.Tensor): The tensor to scale.
        scaler_mean (List): The mean of the training data.
        scaler_std (List): The standard deviation of the training data.
        device (torch.device): The device to use.
    
    Returns:
        torch.Tensor: The scaled tensor.
    """
    

    scaler_mean_tensor = torch.tensor(scaler_mean, dtype=torch.float32, device=device)
    scaler_std_tensor = torch.tensor(scaler_std, dtype=torch.float32, device=device)
    scaled_tensor = (tensor - scaler_mean_tensor) / scaler_std_tensor
    return scaled_tensor

def create_curve_tensor(points: torch.Tensor, scaler_mean: list, scaler_std: list, device: torch.device) -> torch.Tensor:
    """
    Create a tensor of points that represent a curve.

    Args:
        points (torch.Tensor): The points to create a curve from.
        scaler_mean (List): The mean of the training data.
        scaler_std (List): The standard deviation of the training data.
        device (torch.device): The device to use.
    
    Returns:
        torch.Tensor: The tensor of points that represent a curve.
    """

    curve_control_points = [(0.5, 0.5), (-0.5, -0.5)]
    sequence = []
    points = points.to(device)
    
    starting = torch.tensor([(0,0)], device=device, dtype=torch.float32)
    points = torch.cat((starting, points), dim=0)
    
    for i in range(len(points)-1):
        x, y = generate_smooth_curve(points[i], points[i+1], curve_control_points, device)
        time = torch.linspace(0, 1, steps=len(x), device=device) * (1/30) + (i/30)
        
        combined_tensor = torch.stack((time, x, y), dim=-1)
        scaled_tensor = torch_scale(combined_tensor, scaler_mean, scaler_std, device)
        sequence.append(scaled_tensor)
    
    curve_tensor = torch.cat(sequence, dim=0)
    return curve_tensor


def mpc_control_loop(model: torch.nn.Module, target: torch.Tensor, scaler_mean: list, scaler_std: list, N: int = 10, epochs: int = 100, initial_coords: torch.Tensor = None, early_stopping: bool = True, patience: int = 3) -> tuple:
    """
    mpc control loop to optimize the curve to match the target image.

    Args:
        model (torch.nn.Module): The model to use.
        target (torch.Tensor): The target image.
        scaler_mean (List): The mean of the training data.
        scaler_std (List): The standard deviation of the training data.
        N (int): The number of random coordinates to use.
        epochs (int): The number of epochs to run.
        initial_coords (torch.Tensor): The initial coordinates to use.
        early_stopping (bool): Whether to use early stopping.
        patience (int): The patience to use.
    
    Returns:
        tuple: The coordinates and the accuracy history.
    """

    if initial_coords is not None:
        random_coordinates = initial_coords
    else:
        random_coordinates = torch.rand((N, 2)) * 0.1 - 0.05

    
    coordinates = torch.nn.Parameter(random_coordinates,requires_grad=True)
    optimizer = torch.optim.Adam([coordinates], lr=0.0001)
    lower_bound = -0.05
    upper_bound = 0.05
    no_improvement = 0
    prev_accuracy = None
    accuracy_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Generate points based on current coordinates
        generated_points = create_curve_tensor(coordinates,scaler_mean,scaler_std,device)
        
        model.train()
        decoded_image, mu, sigma, hidden_states = model(generated_points)
        
        decoded_image = decoded_image[-1].squeeze()
        loss = iou_loss(decoded_image, target)
        
        # Perform backpropagation
        loss.backward()
        optimizer.step()
        with torch.no_grad():  
            coordinates.data = coordinates.data.clamp(min=lower_bound, max=upper_bound)
        accuracy = calculate_precision_recall_f1((decoded_image > 0.5).float(), target)
        result = {
            'epoch': epoch,
            'loss': loss.item(),
            'precision': accuracy[0],
            'recall': accuracy[1],
            'f1-score': accuracy[2],
        }
        
        accuracy_history.append(result)
        
        if not early_stopping:
            print(f'Epoch {epoch+1}, Loss: {loss.item()} accuracy: {accuracy}')
            continue
        if prev_accuracy is None:
            prev_accuracy = accuracy[2]
            continue
        if np.isclose(accuracy[2], prev_accuracy, atol=1e-3) and no_improvement < patience:
            no_improvement += 1
        elif np.isclose(accuracy[2], prev_accuracy, atol=1e-3) and no_improvement >= patience:
            break
        elif not np.isclose(accuracy[2], prev_accuracy, atol=1e-3):
            no_improvement = 0

        
        
        prev_accuracy = accuracy[2]
        print(f'Epoch {epoch+1}, Loss: {loss.item()} accuracy: {accuracy}')
    
    return coordinates.detach(),accuracy_history



def run(target_path: str, model: torch.nn.Module, scaler_mean: list, scaler_std: list, N: int = 5, epochs: int = 100, initial_points: int = 10)-> None:
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
    image_seq = []
    
    sequence = create_curve_tensor(coordinates,scaler_mean,scaler_std,device)
    print(len(sequence))
    with torch.no_grad():  
        model.eval()
        decoded_images, mu, sigma, hidden_states = model(sequence)
        decoded_images =(decoded_images > 0.5).float()
    
    # Ensure unnormalize function adjusts image to 0-255 range and converts to uint8 if necessary
    images = (decoded_images.detach().cpu().squeeze().numpy() * 255).astype(np.uint8)
    for img in images:
        pil_img =Image.fromarray(img)
        pil_imgs.append(pil_img)
    
    imageio.mimsave(f'results/MPC/robot.gif', pil_imgs, fps=5)


if __name__ == "__main__":
    scaler_mean=[1.75166666e+01, -2.18374525e-04, -2.76295627e-03]
    scaler_std=[10.12287013, 0.01665567, 0.01509156]
    model_path = '/Users/jaiva/Documents/dis-project/final.pth'
    model = load_model(model_path, device)
    target_path = '/Users/jaiva/Documents/dis-project/network/data/segmented_images/one_ribbon/15/robot_180.png'
    folder_path = '/Users/jaiva/Documents/dis-project/network/data/segmented_images/one_ribbon/10/'
    run(target_path=target_path,model=model,scaler_mean=scaler_mean,scaler_std=scaler_std,N=5,epochs=100)
    #run_tests(target_path=folder_path,model=model,scaler_mean=scaler_mean,scaler_std=scaler_std,N=5,epochs=100)
    #run_efficiency_tests(target_path=folder_path,model=model,scaler_mean=scaler_mean,scaler_std=scaler_std,N=5,epochs=100)
    #run_efficiency_tests_n(target_path=folder_path, model=model, scaler_mean=scaler_mean, scaler_std=scaler_std, N_values=[1,2,3,4, 5,6], epochs=100)