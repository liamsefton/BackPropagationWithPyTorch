import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import MSELoss
import torch.optim as optim

class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(11, 15),
            nn.ReLU(),
            nn.Linear(15, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

model = ExampleNet()
num_epochs = 50
optimizer = optim.SGD(model.parameters(), lr=.0001) #learning rates higher than this tend to converge to local minima after first epoch
#Training
for outer_loop in range(num_epochs):
    f = open("winequality-red.csv")
    f.readline()
    sum_differences = 0
    for line in f:
        line = line.split(";")
        line = list(map(float, line))
        target = [line[-1]] #target must be in list in order to be converted to tensor
        target = torch.FloatTensor(target) 
        line = line[:-1]
        output = model(torch.FloatTensor(line))
        optimizer.zero_grad()
        loss_func = nn.MSELoss()
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        sum_differences += float(target.item() - output) ** 2
        #print(float(output - target.item()) ** 2)
        #print(float(output))
        #loss_function = nn.MSELoss()
        #loss = loss_function(output, target)
        #model.zero_grad() #gradients are accumulated so they must be zeroed before each back prop
        #loss.backward()
        #learning_rate = 0.5
        #for f in model.parameters():
        #    f.data.sub_(f.grad.data * learning_rate)
        #print(float(target.item() - output) ** 2)
    print(float(sum_differences))

#Testing
f = open("winequality-red.csv")
f.readline()
sum_differences = 0
for line in f:
    line = line.split(";")
    line = list(map(float, line))
    target = [line[-1]]
    target = torch.FloatTensor(target)
    line = line[:-1]
    output = model(torch.FloatTensor(line))
    sum_differences += float(target.item() - output) ** 2

print(sum_differences)