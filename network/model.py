import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as transforms
from LstmDecoder import LstmDecoder
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import imageio
from preprocessing import *
def preprocess_image(image_path):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        image = Image.open(image_path)
        return transform(image)
def unnormalize(tensor):
    """Unnormalize an image tensor."""
    mean = 0.5
    std = 0.5
    tensor = tensor.clone()  # Clone the tensor to avoid changing the original
    tensor.mul_(std).add_(mean)  # Reverse the normalization
    return tensor
def get_data_sets(df,image_path):

    image_paths = df['photo_num'].values  # Column with image references
    sequence_data = df.drop('photo_num', axis=1)  # All other columns

    # Step 2: Normalize or Standardize Sequence Data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(sequence_data.values)

    # Step 3: Preprocess Images

    images = [preprocess_image(f'{image_path}/robot_{path}.png') for path in image_paths]

    split_point = int(len(sequence_data) * 0.8)  # 80% for training, 20% for testing

    # Split sequences
    X_train = scaled_data[:split_point]
    X_test = scaled_data[split_point:]

    # Split images (assuming 'images' is a list of image tensors)
    y_train = images[:split_point]
    y_test = images[split_point:]

    # Step 6: Convert to Tensors and DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.stack(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.stack(y_test))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader,test_loader

def train(model,criterion, optimizer,train_loader,num_epochs=100):
    
    model.train()  # Set the model to training mode
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        for sequences, images in train_loader:
            sequences, images = sequences.to(device), images.to(device)

            # Forward pass
            optimizer.zero_grad()
            output_images, mu, sigma, _ = model(sequences)
            
            # Compute loss (here, using just the reconstruction loss as an example)
            loss = criterion(output_images, images)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}")

def evaluate(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        total_loss = 0
        for sequences, images in test_loader:
            sequences, images = sequences.to(device), images.to(device)
            output_images, _, _, _ = model(sequences)
            loss = criterion(output_images, images)
            total_loss += loss.item()
        print(f"Test Loss: {total_loss/len(test_loader)}")

def save_model(model):
    torch.save(model.state_dict(), 'new_model.pth')

def test_passthrough(model,df, image_path):

    length_seq = 50
    length = int(0.8*len(df))
    df = df.iloc[length:length+length_seq]
    image_paths = df['photo_num'].values  # Column with image references
    sequence_data = df.drop('photo_num', axis=1)  # All other columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(sequence_data.values)
    images = [preprocess_image(f'{image_path}/robot_{path}.png') for path in image_paths]
    
    dataset = TensorDataset(torch.tensor(scaled_data, dtype=torch.float32), torch.stack(images))
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    frames = []  # List to store frames for the GIF

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for sequences, actual_images in data_loader:
            sequences = sequences.to(device)  # Move sequences to the correct device
            produced_images, _, _, _ = model(sequences)

            for i in range(len(sequences)):
                actual_img = unnormalize(actual_images[i])
                actual_img = to_pil_image(actual_img)
                produced_img = unnormalize(produced_images[i])
                produced_img = to_pil_image(produced_img)

                # Create a new image to place the actual and produced images side by side
                new_img = Image.new('RGB', (actual_img.width * 2, actual_img.height))
                new_img.paste(actual_img, (0, 0))
                new_img.paste(produced_img, (actual_img.width, 0))

                # Convert the PIL image to a numpy array and add to frames
                frames.append(np.array(new_img))

    # Create a GIF from the frames
    imageio.mimsave('output.gif', frames, format='GIF', fps=5)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LstmDecoder(device=device,input_dim=3,l1_out=[8,16,32,32],lstm_hidden_dim = [32,32,64], l2_out= [64,128,128], decoder_chan=8) 
if False:
    model_loader = torch.load('new_model.pth')
    model.load_state_dict(model_loader)
   
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.to(device)
df = pd.read_csv('network/data/updated_robot_data.csv')
df = process_time(df)
df = df[1:100]
print("Setup complete")

train_loader, test_loader = get_data_sets(df,'network/data/blended_image')
print("Data loaded")
train(model=model,criterion=criterion, optimizer=optimizer, train_loader=train_loader, num_epochs=1000)
save_model(model)
evaluate(model, test_loader, criterion)
test_passthrough(model,df,'network/data/blended_image')
