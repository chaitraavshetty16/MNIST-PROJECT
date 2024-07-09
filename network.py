# imports
import snntorch as snn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from random import randint

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

batch_size = 128
data_path = './data/mnist'


def data_loader(drop_last):
    # Define a transform
    transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    # Create DataLoaders
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    return train_loader,test_loader

# Network Architecture


num_inputs = 28*28
num_hidden = 10
num_outputs = 10
# Temporal Dynamics
num_steps = 25
beta = 0.99


# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


def init():
    net = Net().to(device)
    return net


def plot_loss(loss_hist,test_loss_hist,name):
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    plt.plot(loss_hist)
    plt.plot(test_loss_hist)
    plt.title("Loss Curves")
    plt.legend(["Train Loss", "Test Loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    name_of_file = str(name) + '.png'
    plt.savefig(name_of_file)

def train_printer(test_loss_hist,loss_hist,test_data,test_targets,data,targets,counter):
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print("\n")


def train(models):
    num_epochs = 1
    loss_hist = []
    test_loss_hist = []
    counter = 0
    test_local_hist = []
    loss_local_hist = []
    model_to_return = []
    loss_to_return = []
    train_loader,test_loader = data_loader(True)
    for model in models:
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))
        for epoch in range(num_epochs):
            iter_counter = 0
            train_batch = iter(train_loader)
        # Minibatch training loop
            for data, targets in train_batch:
                data = data.to(device)
                targets = targets.to(device)
                # forward pass
                model.train()
                spk_rec, mem_rec = model(data.view(batch_size, -1))
                # initialize the loss & sum over time
                loss_val = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    loss_val += loss(mem_rec[step], targets)
                loss_local_hist.append(loss_val.item())
                with torch.no_grad():
                    model.eval()
                    test_data, test_targets = next(iter(test_loader))
                    test_data = test_data.to(device)
                    test_targets = test_targets.to(device)
                    # Test set forward pass
                    test_spk, test_mem = model(test_data.view(batch_size, -1))
                    # Test set loss
                    test_loss = torch.zeros((1), dtype=dtype, device=device)
                    for step in range(num_steps):
                        test_loss += loss(test_mem[step], test_targets)
                    test_local_hist.append(test_loss.item())
                    if counter % 50 == 0:
                        train_printer(test_local_hist,loss_local_hist,test_data,test_targets,data,targets,counter)
                    counter += 1
                    iter_counter += 1
        loss_hist.append(loss_val.item())
        test_loss_hist.append(test_loss.item())
        model_to_return.append(model)
        loss_to_return.append(loss_val)
    value = randint(0,50)
    plot_loss(loss_local_hist,test_local_hist,value)
    return loss_to_return,model_to_return
