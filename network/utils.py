import cv2
from LstmDecoder import LstmDecoder
import torch
import numpy as np
import imageio
from torchvision.transforms.functional import to_pil_image
from torch.nn import functional as F
from PIL import Image
def load_model(model_path, device):
    model = LstmDecoder(device=device, input_dim=3, l1_out=[8,16,32,32], lstm_hidden_dim=[32,32,64], l2_out=[64,128,128], decoder_chan=8).to(device)
    model_state_dict = torch.load(model_path, map_location=device)  # Ensure model is loaded to the correct device
    model.load_state_dict(model_state_dict)
    model.eval()  # Set the model to evaluation mode
    return model

def unnormalize(tensor):
    mean = 0.5
    std = 0.5
    return tensor.clone().mul_(std).add_(mean)

def get_ellipse(image_path=None,image=None):
    if image_path:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    elif image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the ribbon
    ribbon_contour = max(contours, key=cv2.contourArea)

    # Alternatively, calculate an average radius based on the distance from the centroid to various points on the contour
    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(ribbon_contour)
    center = ellipse[0] # (x, y)
    size = ellipse[1]    # (MA, ma)
    angle = ellipse[2]   # angle

    # Create a flat array with all elements
    ellipse_array = np.array([center[0], center[1], size[0], size[1], angle], dtype=float)
    return ellipse_array

def predict_next_position(optim, lstm_model, threshold=0.5):
    # Prepare input for the LSTM model

    input_data = np.array(optim)
    
    # Predict the next position
    next_image = lstm_model.predict(input_data)
    next_image = (next_image > threshold).float()
    next_image= unnormalize(next_image.cpu())
    return next_image  

def bezier_curve(t, p0, p1, p2, p3):
    """
    Curve used for the movement of the robot. Allows for the movement to be
    curved while also allowing for control of how big the curve is and making sure it doesn't exceed the movement range box.
    """
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3

def generate_smooth_curve(point1, point2, control_points,device):
    """
    The smooth path that the robot will follow.
    Point1 and 2 are the starting and end points.
    The control points are what cause the curvature but also keep the robot bounded between the points.
    """
    t = torch.linspace(0, 1, 10,device=device)
    x = bezier_curve(t, point1[0], control_points[0][0], control_points[1][0], point2[0])
    y = bezier_curve(t, point1[1], control_points[0][1], control_points[1][1], point2[1])
    return x, y



def create_gif_from_test(model, device, test_loader, output_gif_path, num_images=10, threshold=0.5):
    model.eval()
    images_to_save = []
    
    with torch.no_grad():
        for i, (sequences, actual_images) in enumerate(test_loader):
            if i >= num_images:  
                break
            sequences = sequences.to(device)
            produced_images, _, _, _ = model(sequences)
            
            # Apply thresholding to the produced_images
            produced_images_thresholded = (produced_images > threshold).float()
            
            # Process and add both actual and produced (thresholded) images to the list for the GIF
            for actual, produced in zip(actual_images, produced_images_thresholded):

                # Convert to PIL images
                actual_img_pil = to_pil_image(unnormalize(actual.cpu()))
                produced_img_pil = to_pil_image(unnormalize(produced.cpu()))  

                combined_img = Image.new('RGB', (actual_img_pil.width * 2, actual_img_pil.height))
                combined_img.paste(actual_img_pil, (0, 0))
                combined_img.paste(produced_img_pil, (actual_img_pil.width, 0))
                
                images_to_save.append(combined_img)
        print(f"Saving GIF to: {output_gif_path} of size {len(images_to_save)}")

    imageio.mimsave(output_gif_path, [np.array(img) for img in images_to_save], fps=5)

def calculate_precision_recall_f1(outputs: torch.Tensor, labels: torch.Tensor) -> tuple:
    """
    Calculate precision, recall, and F1 score between predictions and labels.
    
    Args:
        outputs (torch.Tensor): Predicted probabilities.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        tuple: Precision, recall, and F1 score.
    """
    preds = outputs > 0.5  
    labels = labels > 0.5  

    TP = (preds & labels).float().sum()
    FP = (preds & ~labels).float().sum()
    FN = (~preds & labels).float().sum()

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)  

    return precision.item(), recall.item(), f1.item()

def weighted_bce_loss(output, target, pos_weight=1.0):
    """
    Custom BCE loss that applies different weights to the positive class.

    Args:
    - output (tensor): Tensor of predicted values.
    - target (tensor): Tensor of actual values.
    - pos_weight (float): Weight for the positive class.
    """
    weight = torch.where(target == 1, pos_weight, 1)
    transformed_target = (target + 1) / 2  # Adjust target if necessary.
    bce_loss = F.binary_cross_entropy(output, transformed_target, weight=weight, reduction='none')
    return bce_loss.mean()