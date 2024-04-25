import os

import pandas as pd
import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from utils import create_gif_from_test, calculate_precision_recall_f1, weighted_bce_loss
from LstmDecoder import LstmDecoder
from preprocessing import *


def preprocess_image(image_path: str)->torch.Tensor:
    """
    Preprocess an image by loading, resizing, and normalizing it.

    Args:
    - image_path (str): Path to the image file.
    
    Returns:
    - torch.Tensor: Preprocessed image.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = Image.open(image_path)
    return transform(image)


def save_model(model:nn.Module, model_name:str)->None:
    """
    Save the model to a file.

    Args:
    - model (nn.Module): Model to save.
    - model_name (str): Name of the file to save the model to.
    """
    torch.save(model.state_dict(), model_name)



def get_data_loader(df:pd.DataFrame, image_path:str, batch_size:int=32)->DataLoader:
    """
    Create a DataLoader from a DataFrame and a path to a folder of images.

    Args:
    - df (pd.DataFrame): DataFrame containing sequence data.
    - image_path (str): Path to the folder containing images.
    - batch_size (int): Batch size for the DataLoader.

    Returns:
    - DataLoader: DataLoader containing the sequence data and images.
    """
    image_paths = df['photo_num'].values
    sequence_data = df.drop('photo_num', axis=1)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(sequence_data.values)

    images = [preprocess_image(f'{image_path}/robot_{path}.png') for path in image_paths]
    dataset = TensorDataset(torch.tensor(scaled_data, dtype=torch.float32), torch.stack(images))
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def get_train_test_loaders(image_path:str, ribbon_type:str,data_path:str,max_len:int, batch_size:int=32, train_test_split:float=0.8)->tuple:
    """
    Create DataLoaders for training and testing data.

    Args:
    - image_path (str): Path to the sub-folder containing images.
    - ribbon_type (str): Type of ribbon to use.
    - data_path (str): Path to the folder containing the data.
    - max_len (int): Maximum length of the sequence data.
    - batch_size (int): Batch size for the DataLoader.
    - train_test_split (float): Fraction of data to use for training.

    Returns:
    - tuple: Tuple containing the training and testing DataLoaders.
    """

    ribbon_path = os.path.join(image_path, ribbon_type)

    all_folders = os.listdir(ribbon_path)
    
    split_index = int(len(all_folders) * train_test_split)
    train_folders = all_folders[:split_index]
    test_folders = all_folders[split_index:]

    train_loaders = []
    for batch_folder in train_folders:
        csv_file_path = os.path.join(data_path, f'updated_robot_data_{batch_folder}.csv')
        if not os.path.exists(csv_file_path):
            raise(f"CSV file not found: {csv_file_path}")
            
        
        df = pd.read_csv(csv_file_path)
        df = df.head(max_len)
        train_loader = get_data_loader(df, os.path.join(ribbon_path, batch_folder), batch_size)

        train_loaders.append(train_loader)

    test_loaders = []
    for batch_folder in test_folders:
        csv_file_path = os.path.join(data_path, f'updated_robot_data_{batch_folder}.csv')
        if not os.path.exists(csv_file_path):
            raise(f"CSV file not found: {csv_file_path}")
        
        df = pd.read_csv(csv_file_path)
        df = df.head(max_len)
        test_loader = get_data_loader(df, os.path.join(ribbon_path, batch_folder), batch_size)
        test_loaders.append(test_loader)
    return train_loaders, test_loaders






def train(model:nn.Module,device:torch.device,train_loader:DataLoader,optimizer:optim.Optimizer,pos_weight:float=7.5)->None:
    """
    Train the model on a DataLoader.

    Args:
    - model (nn.Module): Model to train.
    - device (torch.device): Device to use for training.
    - train_loader (DataLoader): DataLoader containing training data.
    - optimizer (optim.Optimizer): Optimizer to use for training.
    - pos_weight (float): Positive weight for the loss function.

    Returns:
    - None
    """
    
    model.train()
    total_loss = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    for sequences, images in train_loader:
        sequences, images = sequences.to(device), images.to(device)
        optimizer.zero_grad()
        output_images, _, _, _ = model(sequences)

        loss = weighted_bce_loss(output_images, images, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calculate precision, recall, and F1 score
        precision, recall, f1 = calculate_precision_recall_f1(output_images, images)
        total_precision += precision
        total_recall += recall
        total_f1 += f1

    avg_loss = total_loss / len(train_loader)
    avg_precision = total_precision / len(train_loader)
    avg_recall = total_recall / len(train_loader)
    avg_f1 = total_f1 / len(train_loader)

    #print(f"Loss: {avg_loss}")


def evaluate(model:nn.Module, device:torch.device, test_loader:DataLoader, pos_weight:float=7.5)->tuple:
    """
    Evaluate the model on a DataLoader.

    Args:
    - model (nn.Module): Model to evaluate.
    - device (torch.device): Device to use for evaluation.
    - test_loader (DataLoader): DataLoader containing test data.
    - pos_weight (float): Positive weight for the loss function.
    
    Returns:
    - tuple: Tuple containing the average loss, precision, recall, and F1 score.
    """
    model.eval()
    total_loss = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for sequences, images in test_loader:
            sequences, images = sequences.to(device), images.to(device)
            output_images, _, _, _ = model(sequences)
            loss = weighted_bce_loss(output_images, images, pos_weight=pos_weight)
            total_loss += loss.item()
            precision, recall, f1 = calculate_precision_recall_f1(output_images, images)
            total_precision += precision
            total_recall += recall
            total_f1 += f1

    avg_loss = total_loss / len(test_loader)
    avg_precision = total_precision / len(test_loader)
    avg_recall = total_recall / len(test_loader)
    avg_f1 = total_f1 / len(test_loader)
    print(f"Test Loss: {avg_loss}, Test Precision: {avg_precision}, Test Recall: {avg_recall}, Test F1 Score: {avg_f1}")
    return avg_loss, avg_precision, avg_recall, avg_f1

def load_and_train(model:nn.Module, device:torch.device, image_path:str, data_path:str, ribbon_type:str="two_ribbons", model_name:str='final.pth', pos_weight:float=7.5,train_test_split:float=0.8, num_epochs:int=200, batch_size:int=32,max_len:int=1052, lr:float=0.001)->list:
    """
    Load and train a model on a dataset.

    Args:
    - model (nn.Module): Model to train.
    - device (torch.device): Device to use for training.
    - image_path (str): Path to the sub-folder containing images.
    - data_path (str): Path to the folder containing the data.
    - ribbon_type (str): Type of ribbon to use.
    - model_name (str): Name of the file to save the model to.
    - pos_weight (float): Positive weight for the loss function.
    - train_test_split (float): Fraction of data to use for training.
    - num_epochs (int): Number of epochs to train the model.
    - batch_size (int): Batch size for the DataLoader.
    - max_len (int): Maximum length of the sequence data.
    - lr (float): Learning rate for the optimizer.

    Returns:
    - list: List of dictionaries containing the results of training and evaluation.
    """


    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    start_weight = pos_weight
    train_loaders, test_loaders = get_train_test_loaders(image_path, ribbon_type, data_path, max_len, batch_size, train_test_split)
    results = []
    for epoch in tqdm(range(num_epochs)):
        
        for i, train_loader in enumerate(train_loaders):
            train(model, device, train_loader, optimizer, pos_weight=pos_weight)
        # Evaluate the model on the Validation set
        print('Test on a individual training set')
        train_loss, train_precision, train_recall, train_f1 = evaluate(model, device, train_loaders[0], pos_weight=pos_weight)
        print('Test on validation set')
        test_loss, test_precision, test_recall, test_f1 = evaluate(model, device, test_loaders[0], pos_weight=pos_weight)
        result = {
            'lr': lr,
            'epoch': epoch,
            'train_loss': train_loss,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'test_loss': test_loss,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        }
        results.append(result)
    
    
    # Evaluate the model on the test set
    evaluate(model, device, test_loaders[-1], pos_weight=pos_weight)
    output_gif_path = os.path.join("results", f"{epoch}_{ribbon_type}_{pos_weight}_test.gif")
    create_gif_from_test(model, device, test_loaders[-1], output_gif_path)
    output_gif_path = os.path.join("results", f"{epoch}_{ribbon_type}_{pos_weight}_train.gif")
    create_gif_from_test(model, device, train_loaders[-1], output_gif_path)
    save_model(model, model_name)
    print("Training and evaluation complete. Model saved.")
    return results


def main():
    """Main function to train the model and save the results to a CSV file."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LstmDecoder(device=device, input_dim=3, l1_out=[8,16,32,32], lstm_hidden_dim=[32,32,64], l2_out=[64,128,128], decoder_chan=8).to(device)
    if True:
        model_loader = torch.load('final.pth')
        model.load_state_dict(model_loader)
    print("Model loaded")
    print("Model created")
    image_path = 'network/data/segmented_images'
    data_path = 'network/data'
    all_results = []

    results = load_and_train(model, device, image_path, data_path, ribbon_type="one_ribbon", model_name='test.pth', pos_weight= 7.5,train_test_split=0.8, num_epochs=400, batch_size=32, max_len=50, lr=0.001)
    """ results = load_and_train(model, device, image_path, data_path, ribbon_type="one_ribbon", model_name='test.pth', pos_weight= 10,train_test_split=0.8, num_epochs=500, batch_size=32, max_len=100)
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/BCE_weights.csv', index=False)
    
    for lr in [0.0001,0.001,0.01]:
        for i in range(repititions):
            model = LstmDecoder(device=device, input_dim=3, l1_out=[8,16,32,32], lstm_hidden_dim=[32,32,64], l2_out=[64,128,128], decoder_chan=8).to(device)
            results = load_and_train(model, device, image_path, data_path, ribbon_type="one_ribbon", model_name='test.pth', pos_weight= 7.5,train_test_split=0.8, num_epochs=400, batch_size=32, max_len=50, lr=lr)
            for result in results:
                result['rep'] = i
            all_results.append(results)
    results_df = pd.concat([pd.DataFrame(result) for result in all_results])
    results_df.to_csv('results/BCE_weights.csv', index=False)"""
    
def transfer_learning():
    """Transfer learning code which was adapted to the specific use case of the project."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LstmDecoder(device=device, input_dim=3, l1_out=[8,16,32,32], lstm_hidden_dim=[32,32,64], l2_out=[64,128,128], decoder_chan=8).to(device)
    model_loader = torch.load('final.pth')
    model.load_state_dict(model_loader)
    print("Model loaded")
    print("Model created")
    
    #Freeze the first few layers
    for param in model.l1.parameters():
        param.requires_grad = False
    for param in model.lstm.parameters():
        param.requires_grad = False
    for param in model.l2.parameters():
        param.requires_grad = False
    for param in model.l3.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = True
    image_path = 'network/data/segmented_images'
    data_path = 'network/data'
    results = load_and_train(model, device, image_path, data_path, ribbon_type="two_ribbons", model_name='unfrozen_decoder.pth', pos_weight= 7.5,train_test_split=0.8, num_epochs=10, batch_size=32, max_len=200,lr=0.001)
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/transfer_learning_lowered_lr.csv', index=False)


    
    
if __name__ == "__main__":
    main()
