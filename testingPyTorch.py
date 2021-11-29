import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import MSELoss
import torch.optim as optim
import matplotlib.pyplot as plt

class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        #This creates the layers
        self.linear_relu_stack = nn.Sequential( 
            nn.Linear(11, 8),
            nn.ReLU(),
            nn.Linear(8, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

    def forward(self, x):
        #forward function must be defined, so we give it the layers we created above
        return self.linear_relu_stack(x)

#this function is used to initialize the weights to a set value to reduce randomness
def init_weights(layer):
    if type(layer) == nn.Linear:
        layer.weight.data.fill_(0.0)

model = ExampleNet()
model.apply(init_weights)
num_epochs = 1000
optimizer = optim.SGD(model.parameters(), lr=.00005) #learning rates higher than this tend to converge to local minima after first epoch
testing_differences = -1
curr_epoch = 0

#variables for pyplots
testing_error_y_vals = []
training_error_y_vals = []
x_vals_training = []
x_vals_testing = []

#Training
for outer_loop in range(num_epochs):
    if outer_loop % 10 == 0:
        print(outer_loop)
    f = open("training.csv")
    f.readline()
    sum_differences = 0
    num_samples = 0
    for line in f:
        line = line.split(";")
        line = list(map(float, line))
        target = [line[-1]] #target must be in list in order to be converted to tensor
        target = torch.FloatTensor(target) 
        line = line[:-1]
        #line = [float(i)/max(line) for i in line] #normalize the data
        output = model(torch.FloatTensor(line))
        optimizer.zero_grad()
        loss_func = nn.MSELoss()
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        sum_differences += abs(float(target.item() - output))
        num_samples += 1
    training_error_y_vals.append(float(sum_differences)/num_samples)
    x_vals_training.append(curr_epoch)

    #Overfitting prevention here
    divergence_threshold = .001 #if (new average error) - (prev average error) > divergence_threshold
    convergence_threshold = .00005 #if abs(new average error - prev average error) < convergence_threshold 
    f = open("testing.csv")
    f.readline()
    sum_differences = 0
    num_samples = 0
    for line in f:
        line = line.split(";")
        line = list(map(float, line))
        target = [line[-1]]
        target = torch.FloatTensor(target)
        line = line[:-1]
        #line = [float(i)/max(line) for i in line] 
        output = model(torch.FloatTensor(line))
        sum_differences += abs(float(target.item() - output))
        num_samples += 1
    exit_code = -1

    #checking for convergence and divergence
    if testing_differences == -1: #if it is first iteration
        testing_differences = float(sum_differences)/num_samples
    elif float(sum_differences)/num_samples <= testing_differences: 
        if testing_differences - float(sum_differences)/num_samples < convergence_threshold:
            exit_code = 0
            break
        else:
            testing_differences = float(sum_differences)/num_samples
    else:
        if float(sum_differences)/num_samples - testing_differences > divergence_threshold:
            exit_code = 1
            break

    curr_epoch += 1

    testing_error_y_vals.append(float(sum_differences)/num_samples)
    x_vals_testing.append(curr_epoch)

if exit_code == -1:
    print("Training stopped after " + str(curr_epoch) + " epochs from reaching max epochs.")
elif exit_code == 0:
    print("Training stopped after " + str(curr_epoch) + " epochs from achieving convergence.")
elif exit_code == 1:
    print("Training stopped after " + str(curr_epoch) + " epochs from overfitting prevention.")

f.close()

#Testing
f = open("testing.csv")
f.readline()
sum_differences = 0
num_samples = 0
for line in f:
    line = line.split(";")
    line = list(map(float, line))
    target = [line[-1]]
    target = torch.FloatTensor(target)
    line = line[:-1]
    #line = [float(i)/max(line) for i in line] #this does normalization
    output = model(torch.FloatTensor(line))
    sum_differences += abs(float(target.item() - output))
    num_samples += 1
print("Testing result: ", end="")
print(float(sum_differences)/num_samples)

#Look at this graph
plt.plot(x_vals_testing, testing_error_y_vals)
plt.savefig("testing_differences_by_epoch.png")
plt.close()
plt.plot(x_vals_training, training_error_y_vals)
plt.savefig("training_differences_by_epoch.png")