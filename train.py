import torch
from torchvision import datasets,transforms
import os
from model import Model 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange
transform_img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0), std=(1)),
    lambda img: img.reshape(784)
    ])

train_data= datasets.MNIST(root =  os.getcwd(),download = True,train = True,
        transform = transform_img)
train_data= DataLoader(train_data, batch_size=4096, shuffle=False,drop_last=True)
test_data = datasets.MNIST(root =  os.getcwd(),download = True,train = False,
        transform = transform_img)
test_data = DataLoader(test_data, batch_size=4096, shuffle=False,drop_last=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model().to(device)
opt = optim.Adam(params = model.parameters(),lr = .003)
loss_fn = nn.CrossEntropyLoss()
epochs = 30
def save_model(model):
    torch.save(model.state_dict(),f'{os.getcwd()}/model.pt')

def validate(test_data,model,device):
    it = iter(test_data)
    acc = []
    for _ in range(len(test_data)):
        input,target = next(it)
        input,target = input.to(device),target.to(device)
        pred = model(input)
        pred = torch.argmax(pred,-1)
        acc.append(((pred == target).nonzero()).size(0)/target.size(0))
    return sum(acc)/len(acc)
def train(train_data = train_data,test_data = test_data,model = model,loss_fn = loss_fn,opt = opt,epochs = epochs,device = device):
    for epoch in (t:= trange(epochs)):
        it = iter(train_data)
        for _ in range(len(train_data)):
            input,target = next(it)
            input,target = input.to(device),target.to(device)
            opt.zero_grad()
            output = model(input)
            loss = loss_fn(output,target)
            loss.backward()
            opt.step()
        t.set_description('accuracy: %.3f'% (validate(test_data,model,device)))

if __name__ == '__main__':
    train()
    save_model(model)
