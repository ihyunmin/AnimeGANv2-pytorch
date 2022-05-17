import numpy as np
import math
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision

writer = SummaryWriter()

device = torch.device('cuda')

# Real Data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=torch.float)
x = x.view(-1,1)
y = torch.sin(x)

# Simple MLP Model
model = torch.nn.Sequential(
    torch.nn.Linear(1,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,1)
).to(device)

# Optimizer & Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5,0.999))
criterion = torch.nn.MSELoss()
criterion2 = torch.nn.L1Loss()

# train
def train_model(iter):
    for epoch in range(iter):
        optimizer.zero_grad()

        pred = model(x)
        loss = criterion(y, pred)
        loss2  = criterion2(y, pred)
        writer.add_scalars('Loss/train', {'L2Loss':loss,
                                         'L1Loss':loss2}, epoch)        
        loss.backward()
        optimizer.step()
        print('{} Epoch Loss : [{}]'.format(epoch, round(loss.item(), 8)))
    
train_model(5000)

img_grid = torchvision.utils.make_grid(torch.rand(2,3,64,64))
writer.add_image('Two new images', img_grid)

writer.add_graph(model, x)

writer.close()