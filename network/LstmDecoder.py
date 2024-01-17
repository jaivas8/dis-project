
"""LSTM + CNN Network Model"""

import torch.nn as nn
import torch

class LstmDecoder(nn.Module):
    # input_dim = 3, l1_out = [8,16,32,32], lstm_hidden_dim = [32,32,64], l2/l3 = [64,128,128] decoder_chan=8
    def __init__(self, device, input_dim, l1_out, lstm_hidden_dim, l2_out, decoder_chan=1):
        super().__init__()
        self.device = device
        
        # Linear to increase dimensions of input
        self.l1 = []
        layer = nn.Linear(input_dim,l1_out[0])
        self.l1.append(layer)
        for i in range(1,len(l1_out)):
            layer = nn.Linear(l1_out[i-1],l1_out[i])
            self.l1.append(layer)
        self.l1.append(nn.ReLU())
        self.l1 = nn.Sequential(*self.l1)

        # LSTM Encoder (Acts as the encoder in the VAE)
        self.lstm = []
        layer = nn.LSTM(l1_out[-1], lstm_hidden_dim[0], batch_first=True).to(self.device)
        self.lstm.append(layer)
        for i in range(1,len(lstm_hidden_dim)):          
            layer = nn.LSTM(lstm_hidden_dim[i-1],lstm_hidden_dim[i], batch_first=True).to(self.device)
            self.lstm.append(layer)
        self.lstm = nn.ModuleList(self.lstm)
        
        # Second and third linear layers are used to calculate mu and sd deviation

        self.l2 = [nn.ReLU()]
        self.l3 = [nn.ReLU()]
        layer1 = nn.Linear(lstm_hidden_dim[-1], l2_out[0])
        layer2 = nn.Linear(lstm_hidden_dim[-1], l2_out[0])
        self.l2.append(layer1)
        self.l3.append(layer2)
        for i in range(1,len(l2_out)):
            layer1 = nn.Linear(l2_out[i-1],l2_out[i])
            layer2 = nn.Linear(l2_out[i-1],l2_out[i])
            self.l2.append(layer1)
            self.l3.append(layer2)
            
        self.l2 = nn.Sequential(*self.l2)
        self.l3 = nn.Sequential(*self.l3)

        # Sets episilon for the encoder
        self.N = torch.distributions.Normal(0, 1)
        self.sampling_loss = 0
        # Image Decoder
        s = int((l2_out[-1]/decoder_chan)**(1/2))
        self.decoder = nn.Sequential(
            nn.Unflatten(1, torch.Size([decoder_chan,s,s])),
            nn.ConvTranspose2d(decoder_chan,8,kernel_size=1,stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8,16,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,64,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,128,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32,16,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,8,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8,4,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4,1,kernel_size=5, padding=2),
            nn.Sigmoid()
        )
        
        
    def forward(self, x):
        x = self.l1(x)

        # Process through LSTM layers
        hidden_states = []
        for layer in self.lstm:
            x, hidden = layer(x)
            hidden_states.append(hidden)

        # Calculate mu and sigma
        mu = self.l2(x)  # Assuming last output of LSTM is used
        sigma = self.l3(x)  # Assuming last output of LSTM is used
        sigma = torch.exp(sigma)  # Ensure sigma is positive

        # Sampling in the latent space
        epsilon = self.N.sample(mu.shape).to(self.device)
        z = mu + sigma * epsilon

        # Decode the sample to generate the image
        decoded_image = self.decoder(z)

        return decoded_image, mu, sigma, hidden_states