import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from models import MLP
import fire

def train(maxTimeSteps=1000, numEpoch=100, batchSizes=128, learningRate=0.001):
    # device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("Current Device: {}".format(device))

    beta = torch.zeros(size=(maxTimeSteps+1,))
    beta[1:] = torch.linspace(1e-4, 0.02, maxTimeSteps)
    alpha = 1 - beta
    alphaBar = torch.cumprod(alpha, dim=0)
    alphaBarSqrt = torch.sqrt(alphaBar).to(device)
    oneMinusAlphaBarSqrt = torch.sqrt(1 - alphaBar).to(device)
    imageDim = 784
    epsilonGenerator = torch.distributions.MultivariateNormal(torch.zeros(size=(imageDim,)), torch.eye(imageDim))

    imageTransform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Lambda(lambda x: torch.reshape(x, (-1,))),
        transforms.Lambda(lambda x: x * 2 / 255 - 1)
    ])
    dataset = MNIST(root="./dataset/", train=True, download=True, transform=imageTransform)
    dataLoader = DataLoader(dataset, batch_size=batchSizes, shuffle=True)

    sizes = [imageDim, 1000, 1000, imageDim]
    model = MLP(sizes, nn.Tanh()).to(device)
    criterier = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    for epoch in range(numEpoch):
        model.train()
        totalLosses = 0.0
        for images, _ in dataLoader:
            x_0 = images.to(device)
            currentBatchSize = images.size(0)
            t = torch.randint(1, maxTimeSteps+1, size=(1,))
            epsilon = epsilonGenerator.sample((currentBatchSize,)).to(device)
            x_t = alphaBarSqrt[t] * x_0 + oneMinusAlphaBarSqrt[t] * epsilon
            
            epsilon_theta = model(x_t)

            optimizer.zero_grad()
            loss = criterier(epsilon, epsilon_theta)
            loss.backward()
            optimizer.step()

            totalLosses += loss.item()

        print("Epoch: {}. Losses: {}".format(epoch, totalLosses))
    fileName = './models/MLP-{}-{}-{}-{}-{}.pt'.format(imageDim, 1000, 1000, imageDim, torch.randint(0, 10000, size=(1,)).item())
    object = {
        "model": model,
        "betas": beta,
        "maxTimeSteps": maxTimeSteps
    }
    torch.save(object, fileName)
    print('Model has been save to {}'.format(fileName))

def sample(modelFileName, n=10):
    object = torch.load(modelFileName)
    model = object['model'].cpu()
    beta = object['betas']
    maxTimeSteps = object['maxTimeSteps']

    alpha = 1 - beta
    alphaBar = torch.cumprod(alpha, dim=0)
    alphaSqrts = torch.sqrt(alpha)
    oneMinusAlphaBarSqrt = torch.sqrt(1 - alphaBar)
    sigma2 = torch.zeros(size=(maxTimeSteps+1,))
    sigma2[1:] = (1 - alphaBar[:-1]) / (1 - alphaBar[1:]) * beta[:-1]
    sigma = torch.sqrt(sigma2)
    
    imageDim = 784
    epsilonGenerator = torch.distributions.MultivariateNormal(torch.zeros(size=(imageDim,)), torch.eye(imageDim))
    
    id = torch.randint(0, 10000, size=(1,)).item()
    for i in range(10):
        x = epsilonGenerator.sample()
        for t in reversed(range(2, maxTimeSteps+1)):
            z = epsilonGenerator.sample()
            x = (x - beta[t] / oneMinusAlphaBarSqrt[t] * model(x)) / alphaSqrts[t] + sigma[t] * z
        mu_theta = (x - beta[1] / oneMinusAlphaBarSqrt[1] * model(x)) / alphaSqrts[1]
        x = torch.normal(mu_theta, sigma[1] * torch.ones_like(mu_theta))
        x = (x + 1) * 255 / 2
        x = torch.round(x)
        x = x.reshape(28, 28).detach().numpy()
        x[x > 255] = 255
        x[x < 0] = 0
        plt.imshow(x)
        plt.savefig('./figure/{}-{}.png'.format(id, i), dpi=400)

def classify(numEpoch=100, batchSizes=128, learningRate=0.001):
    # device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print("Current Device: {}".format(device))

    imageDim = 784

    imageTransform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Lambda(lambda x: torch.reshape(x, (-1,))),
        transforms.Lambda(lambda x: x * 2 / 255 - 1)
    ])
    trainDataset = MNIST(root="./dataset/", train=True, download=True, transform=imageTransform)
    dataLoader = DataLoader(trainDataset, batch_size=batchSizes, shuffle=True)
    testDataset = MNIST(root="./dataset/", train=False, download=True, transform=imageTransform)
    testDataloader = DataLoader(testDataset, batch_size=batchSizes, shuffle=False)


    sizes = [imageDim, 1000, 1000, 10]
    model = MLP(sizes, nn.Tanh()).to(device)
    criterier = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    for epoch in range(numEpoch):
        model.train()
        totalLosses = 0.0
        for images, targets in dataLoader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)

            optimizer.zero_grad()
            loss = criterier(outputs, targets)
            loss.backward()
            optimizer.step()

            totalLosses += loss.item()
        print("Epoch: {}, Losses: {}".format(epoch, totalLosses))

        model.eval()
        totalLosses = 0.0
        correct = 0
        for images, targets in testDataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)

            loss = criterier(outputs, targets)
            correct += torch.sum(torch.argmax(outputs, dim=1) == targets).item()

            totalLosses += loss.item()

        print("[Test] Losses: {}, Accurancy: {:.2%}".format(totalLosses, correct / len(testDataset)))






if __name__ == "__main__": 
    fire.Fire()