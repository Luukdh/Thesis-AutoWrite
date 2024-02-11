import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.utils.data import Dataset, DataLoader, random_split

from models.unet_das_1 import UNet
from random_transformation import RandomTransformation

# Start the timer.
start_time = time.time()
duration = 60*14

alpha = {}
dirty_alpha = {}
with open('./data/input/alphabet.pkl', 'rb') as f:
    alpha = pickle.load(f)
with open('./data/input/dirty_alphabet.pkl', 'rb') as f:
    dirty_alpha = pickle.load(f)


class MyDataset(Dataset):
    '''
    Contains the dataset.
    '''
    def __init__(self, alpha, dirty_alpha, transform=None):
        '''
        Initialize the dataset.
        '''
        self.data = []
        self.labels = []
        self.n = 3
        for key, value in alpha.items():
            if key in dirty_alpha:
                for i, stroke in enumerate(dirty_alpha[key]):
                    self.labels.append(value[0])
                    self.data.append(stroke)
            else:
                continue

    def applyTransform(self, data):
        '''
        Apply a random transformation to the data.
        '''
        rand = RandomTransformation()
        return rand(data)
    
    def getData(self, index):
        data = self.data[index // self.n]
        transformed_data = self.applyTransform(self.applyTransform(self.applyTransform(data)))
        return torch.Tensor(transformed_data.copy())
    
    def getLabels(self, index):
        return torch.Tensor(self.labels[index // self.n])

    def __getitem__(self, index):
        return self.getData(index), self.getLabels(index)

    def __len__(self):
        return len(self.data)*self.n
    

if __name__ == "__main__":
    '''
    Start the training run.
    '''
    name = sys.argv[1]
    num_epochs = int(sys.argv[2])
    min_loss = float(sys.argv[3])

    dataset = MyDataset(alpha, dirty_alpha)
    # dataloader = DataLoader(dataset=dataset, batch_size=362, shuffle=True, num_workers=1)
    dataset_train, dataset_valid = random_split(dataset, [len(dataset)-72, 72])
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=len(dataset)-72, shuffle=True, num_workers=1)
    dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=72, shuffle=True, num_workers=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(device)

    # num_epochs = 10
    learning_rate = 0.0001

    model = UNet(3)
    if os.path.isfile(f"./data/model_data/model_{name}.ckpt"):
        model.load_state_dict(torch.load(f"./data/model_data/model_{name}.ckpt"))
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if os.path.isfile(f"./data/optim_data/optim_{name}.ckpt"):
        optimizer.load_state_dict(torch.load(f"./data/optim_data/optim_{name}.ckpt"))

    n_total_steps = len(dataloader_train)
    lowest_loss = 1000
    loss_train = []
    loss_valid = []
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(dataloader_train):
            features = features.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(features)
            loss = criterion(outputs.requires_grad_(), labels)
            loss_train.append(loss.item())

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 1 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

        with torch.no_grad():
            for i, (features, labels) in enumerate(dataloader_valid):
                features = features.to(device)
                labels = labels.to(device)

                # forward pass
                outputs = model(features)
                loss_v = criterion(outputs.requires_grad_(), labels)
                loss_valid.append(loss_v.item())
        
        if loss.item() < lowest_loss:
            lowest_loss = loss.item()
            torch.save(model.state_dict(), f'./data/model_data/model_{name}.ckpt')
            torch.save(optimizer.state_dict(), f'./data/optim_data/optim_{name}.ckpt')
        
        loss_file = open(f"./data/loss_data/loss_{name}.csv", "a")
        loss_file.write(f"{loss.item()}, {loss_v.item()}\n")
        loss_file.close()
        
        if time.time() - start_time >= duration:
            print("Timeout!")
            break
        if loss_v.item() < min_loss:
            print("Found Loss!")
            break
    
    with open(f"./data/loss_data/loss_train_{name}.pkl", "wb") as f:
        pickle.dump(loss_train, f)
    with open(f"./data/loss_data/loss_valid_{name}.pkl", "wb") as f:
        pickle.dump(loss_valid, f)

