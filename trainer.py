import argparse
from common import *
from CNN import ConvNet

import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score

from torchsummary import summary

import torch
from torch import nn, optim, utils
import torchvision
import torchvision.utils as vutils

import random


class Trainer:
    def __init__(self, model, n_classes, device):
        self.n_classes = n_classes
        self.device = device
        self.model = model
        self.loss = []
        self.precision = []
        self.recall = []
        self.f1 = []
    
    def fit(self, trainLoader, testLoader, epochs):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

        print("TRAINING")
        for epoch in range(epochs):
            i = 0
            print('EPOCH:', epoch)

            for features, labels in iter(trainLoader):    

                input = features.to(self.device)
                l_input = labels.to(self.device)
                self.model.zero_grad()

                output = self.model(input).view(-1, self.n_classes)
                error = criterion(output, l_input)
                error.backward()
                optimizer.step()

                i += 1
                if i == 0 or i % 20 == 9:
                    print('[%d/%d][%d/%d]\tLoss: %.4f'% (epoch, epochs, i, len(trainLoader), error.item()))
                    self.get_scores(testLoader)
                self.loss.append(error.item())

                
            print('Epoch: %d |\tPrecision: %.4f\tRecall: %.4f\tF1: %.4f'% (epoch, self.precision[-1], self.recall[-1], self.f1[-1]))
            if error.item() < 0.001:
                    break

            torch.save(self.model.state_dict(), 'model.m')

    def predict(self, X):
        X = self.model(X).view(-1, self.n_classes)
        return torch.argmax(X, dim=1).cpu().numpy()

    def get_scores(self, testLoader):
        X, Y = next(iter(testLoader))

        X = self.predict(X.to(self.device))
        Y = Y.numpy()
        self.precision.append(precision_score(Y, X, average='weighted'))
        self.recall.append(recall_score(Y, X, average='weighted'))
        self.f1.append(f1_score(Y, X, average='weighted'))



def get_data(train=True, batch_size=128, shuffle=True, n_workers=1):
    return utils.data.DataLoader(
        torchvision.datasets.MNIST(root='Data', download=True, train=train, transform=torchvision.transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=n_workers)


def train(epochs=5, batch_size=50, gpu=0, verbose = 1):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # ЗЧИТАТИ ДАНІ
    data_train = get_data(True)
    data_test = get_data(False)

    # ІНІЦІАЛІЗУВАТИ МОДЕЛЬ
    model = ConvNet(gpu, 10).to(device)
    if verbose == 1:
            print(summary(model, (1, IMAGE_SIZE, IMAGE_SIZE)))

    trainer = Trainer(model, 10, device)
    trainer.fit(data_train, data_test, epochs)
    
    plt.figure()
    plt.plot(trainer.loss, label='model')
    plt.title('Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.figure()
    plt.plot(trainer.precision, label='precision')
    plt.plot(trainer.recall, label='recall')
    plt.plot(trainer.f1, label='f1')
    plt.title('Accuracy metrics')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Insert training parameteres')
    parser.add_argument('epochs', metavar='epochs', type=int,
                        help='number of training epochs;')
    parser.add_argument('batch_size', metavar='batch_size', type=int,
                        help='number of samples in a single training batch;')
    parser.add_argument('gpu_number', metavar='gpu_number', type=int,
                        help='number of availible GPUs to perform training;')
    parser.add_argument('verbose', metavar='verbose', type=int,
                    help='verbose [0 / 1];')

    args = parser.parse_args()
    
    manualSeed = 42
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    train(args.epochs, args.batch_size, args.gpu_number, args.verbose)
