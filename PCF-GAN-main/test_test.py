#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:22:50 2024

@author: shanyinuo
"""

import torch
import torch.nn as nn
from src.utils import init_weights


from src.networks.generators import LSTMGenerator
from src.networks.discriminators import LSTMDiscriminator

device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_dim = 10  
output_dim = 10 
hidden_dim = 64 
n_layers = 2 
generator = LSTMGenerator(input_dim, output_dim, hidden_dim, n_layers)
discriminator = LSTMDiscriminator(output_dim, hidden_dim, n_layers, out_dim=1)


gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)


num_epochs = 100
batch_size = 32


for epoch in range(num_epochs):

    generated_data = generator(batch_size=batch_size, n_lags=20, device='cpu')
    

    real_data = torch.randn(batch_size, 20, 10)  
    gen_labels = torch.zeros(batch_size, 1)  
    real_labels = torch.ones(batch_size, 1)  
    
    gen_output = discriminator(generated_data)
    disc_loss_fake = nn.BCEWithLogitsLoss()(gen_output.squeeze(), gen_labels.squeeze())
    
    real_output = discriminator(real_data)
    disc_loss_real = nn.BCEWithLogitsLoss()(real_output.squeeze(), real_labels.squeeze())
    
    disc_loss = disc_loss_fake + disc_loss_real
    

    disc_optimizer.zero_grad()
    disc_loss.backward(retain_graph=True)
    disc_optimizer.step()
    

    generated_output = discriminator(generated_data)
    gen_loss = nn.BCEWithLogitsLoss()(generated_output.squeeze(), real_labels.squeeze())  # 生成器希望判别器把生成的数据判定为真
    

    gen_optimizer.zero_grad()
    gen_loss.backward()
    gen_optimizer.step()
    

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Generator Loss: {gen_loss.item():.4f}, Discriminator Loss: {disc_loss.item():.4f}')

# generated_data = generator(batch_size, n_lags, device).detach().cpu().numpy()





