import torch
from torch import nn
import pickle as pkl
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

HIDDEN_SIZES = [336, 336]

NUM_OF_CLASSES = 36
INPUT_RESOLUTION = 56


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_RESOLUTION * INPUT_RESOLUTION, HIDDEN_SIZES[0])
        self.fc2 = nn.Linear(HIDDEN_SIZES[0], NUM_OF_CLASSES)
        #self.fc3 = nn.Linear(HIDDEN_SIZES[1], NUM_OF_CLASSES)

    def forward(self, x):
        #x = x.view(-1, INPUT_RESOLUTION * INPUT_RESOLUTION)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        return self.fc2(x)






'''def compress_training_set(x_train):
    new_x_train_array = []
    for x in x_train:
        max = len(x)
        new_x_array = []
        for i in range(0, max):
            if i % 56 != 0:
                new_x_array.append(x[i])
            else:
                i += 56
        new_x_train_array.append(np.array(new_x_array))

    return np.array(new_x_train_array)'''
