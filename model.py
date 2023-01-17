import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(784,512)
        self.layer_2 = nn.Linear(512,10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(-1)

    def forward(self,img):
        img = self.layer_1(img)
        img = self.relu(img)
        img = self.layer_2(img)
        return self.softmax(img)

