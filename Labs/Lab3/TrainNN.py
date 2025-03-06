

import torch 
import torchvision
import torchvision.transforms as transforms;
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class CNN(torch.nn.Module):

    def __init__(self, input, unitsize, output):
        super(CNN, self).__init__() 
        self.layer1 = torch.nn.Linear(input, unitsize)
        self.layer2 = torch.nn.Linear(unitsize, unitsize)
        self.layer3 = torch.nn.Linear(unitsize, unitsize)
        self.outputlayer = torch.nn.Linear(unitsize, output)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
    
    def forward(self, X):
        
        firstlayer = self.relu(self.layer1(X))
        firstlayer = self.dropout(firstlayer)

        secondlayer = self.relu(self.layer2(firstlayer))
        secondlayer = self.dropout(secondlayer)

        thirdlayer = self.relu(self.layer3(secondlayer))
        thirdlayer = self.dropout(thirdlayer)

        outputlayer = self.outputlayer(thirdlayer)
        return outputlayer



classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
learning_rate = 0.0001
total_loss = 0
epocs = 10

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.FashionMNIST(root="./data", download=True, train=True, transform=transform)


test_dataset = torchvision.datasets.FashionMNIST(root="./data", download=True, train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

obj = CNN(28*28, 2048, 10)
criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(obj.parameters(), lr=learning_rate)


for i in range(epocs):
    total_loss = 0
    for batch, (inputs, labels) in enumerate(train_loader):
        input = inputs.view(inputs.size(0), -1) 
        print(input)
        output = obj.forward(input)
        loss = criterion(output, labels) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(total_loss)

    # print(output)
obj.eval()