"""
William Randall
Homework 3
UID 805167986
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image

from google.colab import drive
drive.mount('/content/drive')

######################################################################
# OneLayerNetwork
######################################################################

class OneLayerNetwork(torch.nn.Module):
    def __init__(self):
        super(OneLayerNetwork, self).__init__()

        ### ========== TODO : START ========== ###
        ### part d: implement OneLayerNetwork with torch.nn.Linear
        self.oneLayerNetwork = torch.nn.Linear(784,3)
        ### ========== TODO : END ========== ###

    def forward(self, x):
        # x.shape = (n_batch, n_features)

        ### ========== TODO : START ========== ###
        ### part d: implement the foward function
        outputs = self.oneLayerNetwork(x)
        ### ========== TODO : END ========== ###
        return outputs

######################################################################
# TwoLayerNetwork
######################################################################

class TwoLayerNetwork(torch.nn.Module):
    def __init__(self):
        super(TwoLayerNetwork, self).__init__()
        ### ========== TODO : START ========== ###
        ### part g: implement TwoLayerNetwork with torch.nn.Linear
        self.twoLayerNetwork_1 = torch.nn.Linear(784, 400)
        self.twoLayerNetwork_2 = torch.nn.Linear(400, 3)
        ### ========== TODO : END ========== ###

    def forward(self, x):
        # x.shape = (n_batch, n_features)

        ### ========== TODO : START ========== ###
        ### part g: implement the foward function
        sig = torch.nn.Sigmoid()
        layer_1 = self.twoLayerNetwork_1(x)
        layer_1 = sig(layer_1)
        outputs = self.twoLayerNetwork_2(layer_1)
        ### ========== TODO : END ========== ###
        return outputs

# load data from csv
# X.shape = (n_examples, n_features), y.shape = (n_examples, )
def load_data(filename):
    data = np.loadtxt(filename)
    y = data[:, 0].astype(int)
    X = data[:, 1:].astype(np.float32) / 255
    return X, y

# plot one example
# x.shape = (features, )
def plot_img(x):
    x = x.reshape(28, 28)
    img = Image.fromarray(x*255)
    plt.figure()
    plt.imshow(img)
    return

def evaluate_loss(model, criterion, dataloader):
    model.eval()
    total_loss = 0.0
    for batch_X, batch_y in dataloader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate_acc(model, dataloader):
    model.eval()
    total_acc = 0.0
    for batch_X, batch_y in dataloader:
        outputs = model(batch_X)
        predictions = torch.argmax(outputs, dim=1)
        total_acc += (predictions==batch_y).sum()
        
    return total_acc / len(dataloader.dataset)

def train(model, criterion, optimizer, train_loader, valid_loader):
    train_loss_list = []
    valid_loss_list = []
    train_acc_list = []
    valid_acc_list = []
    for epoch in range(1, 31):
        model.train()
        for batch_X, batch_y in train_loader:
            ### ========== TODO : START ========== ###
            ### part f: implement the training process
            #forward pass
            y_pred = model.forward(batch_X)
            #initializing gradients to zeroes
            model.zero_grad()
            #computing loss
            loss = criterion(y_pred, batch_y)
            #loss backward
            loss.backward()
            #updating model parameters
            optimizer.step()
            ### ========== TODO : END ========== ###
            
        train_loss = evaluate_loss(model, criterion, train_loader)
        valid_loss = evaluate_loss(model, criterion, valid_loader)
        train_acc = evaluate_acc(model, train_loader)
        valid_acc = evaluate_acc(model, valid_loader)
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

        print(f"| epoch {epoch:2d} | train loss {train_loss:.6f} | train acc {train_acc:.6f} | valid loss {valid_loss:.6f} | valid acc {valid_acc:.6f} |")

    return train_loss_list, valid_loss_list, train_acc_list, valid_acc_list

#######################################################################
# main
######################################################################

def main():

    # fix random seed
    np.random.seed(0)
    torch.manual_seed(0)

    # load data with correct file path

    ### ========== TODO : START ========== ###
    data_directory_path =  "/content/drive/MyDrive/Colab Notebooks/CS146/hw3"
    ### ========== TODO : END ========== ###

    # X.shape = (n_examples, n_features)
    # y.shape = (n_examples, )
    X_train, y_train = load_data(os.path.join(data_directory_path, "hw3_train.csv"))
    X_valid, y_valid = load_data(os.path.join(data_directory_path, "hw3_valid.csv"))
    X_test, y_test = load_data(os.path.join(data_directory_path, "hw3_test.csv"))

    ### ========== TODO : START ========== ###
    ### part a: print out three training images with different labels
    plot_img(X_train[20])
    plot_img(X_train[70])
    plot_img(X_train[299])
    ### ========== TODO : END ========== ###

    print("Data preparation...")

    ### ========== TODO : START ========== ###
    ### part b: convert numpy arrays to tensors
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)

    X_valid = torch.from_numpy(X_valid)
    y_valid = torch.from_numpy(y_valid)

    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)
    ### ========== TODO : END ========== ###

    ### ========== TODO : START ========== ###
    ### part c: prepare dataloaders for training, validation, and testing
    ###         we expect to get a batch of pairs (x_n, y_n) from the dataloader
    train_loader = DataLoader(TensorDataset(X_train,y_train), batch_size=10)
    valid_loader = DataLoader(TensorDataset(X_valid,y_valid), batch_size=10)
    test_loader = DataLoader(TensorDataset(X_test,y_test), batch_size=10)
    ### ========== TODO : END ========== ###

    ### ========== TODO : START ========== ###
    ### part e: prepare OneLayerNetwork, criterion, and optimizer
    model_one = OneLayerNetwork()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_one.parameters(), lr=0.0005)
    ### ========== TODO : END ========== ###

    print("Start training OneLayerNetwork...")
    results_one = train(model_one, criterion, optimizer, train_loader, valid_loader)
    print("Done!")

    ### ========== TODO : START ========== ###
    ### part h: prepare TwoLayerNetwork, criterion, and optimizer
    model_two = TwoLayerNetwork()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_two.parameters(), lr=0.0005)
    ### ========== TODO : END ========== ###

    print("Start training TwoLayerNetwork...")
    results_two = train(model_two, criterion, optimizer, train_loader, valid_loader)
    print("Done!")

    one_train_loss, one_valid_loss, one_train_acc, one_valid_acc = results_one
    two_train_loss, two_valid_loss, two_train_acc, two_valid_acc = results_two

    ### ========== TODO : START ========== ###
    ### part i: generate a plot to comare one_train_loss, one_valid_loss, two_train_loss, two_valid_loss
    epochs = np.arange(1, 31)
    plt.figure(figsize=[10,10])
    plt.plot(epochs, one_train_loss, color='red', label='1 layer train loss', marker='o')
    plt.plot(epochs, one_valid_loss, color='green', label='1 layer valid loss', marker='o')
    plt.plot(epochs, two_train_loss, color='blue', label='2 layer train loss', marker='o')
    plt.plot(epochs, two_valid_loss, color='lightblue', label='2 layer valid loss', marker='o')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('comare one layer and two layer train and valid loss')
    plt.legend()
    plt.show()
    ### ========== TODO : END ========== ###

    ### ========== TODO : START ========== ###
    ### part j: generate a plot to comare one_train_acc, one_valid_acc, two_train_acc, two_valid_acc
    epochs = np.arange(1, 31)
    plt.figure(figsize=[10,10])
    plt.plot(epochs, one_train_acc, color='red', label='1 layer train accuracy', marker='o')
    plt.plot(epochs, one_valid_acc, color='green', label='1 layer valid accuracy', marker='o')
    plt.plot(epochs, two_train_acc, color='blue', label='2 layer train accuracy', marker='o')
    plt.plot(epochs, two_valid_acc, color='lightblue', label='2 layer valid accuracy', marker='o')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('comare one layer and two layer train and valid accuracy')
    plt.legend()
    plt.show()
    ### ========== TODO : END ========== ##

    ### ========== TODO : START ========== ###
    ### part k: calculate the test accuracy
    print("test accuracy of one layer network: ", evaluate_acc(model_one, test_loader))
    print("test accuracy of two layer network: ", evaluate_acc(model_two, test_loader))
    ### ========== TODO : END ========== ###

    ### ========== TODO : START ========== ###
    ### part l: replace the SGD optimizer with the Adam optimizer and do the experiments again

    train_loader = DataLoader(TensorDataset(X_train,y_train), batch_size=10)
    valid_loader = DataLoader(TensorDataset(X_valid,y_valid), batch_size=10)
    test_loader = DataLoader(TensorDataset(X_test,y_test), batch_size=10)

    model_one = OneLayerNetwork()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_one.parameters(), lr=0.0005)

    print("Start training OneLayerNetwork...")
    results_one = train(model_one, criterion, optimizer, train_loader, valid_loader)
    print("Done!")

    model_two = TwoLayerNetwork()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_two.parameters(), lr=0.0005)

    print("Start training TwoLayerNetwork...")
    results_two = train(model_two, criterion, optimizer, train_loader, valid_loader)
    print("Done!")

    one_train_loss, one_valid_loss, one_train_acc, one_valid_acc = results_one
    two_train_loss, two_valid_loss, two_train_acc, two_valid_acc = results_two

    epochs = np.arange(1, 31)
    plt.figure(figsize=[10,10])
    plt.plot(epochs, one_train_loss, color='red', label='1 layer train loss', marker='o')
    plt.plot(epochs, one_valid_loss, color='green', label='1 layer valid loss', marker='o')
    plt.plot(epochs, two_train_loss, color='blue', label='2 layer train loss', marker='o')
    plt.plot(epochs, two_valid_loss, color='lightblue', label='2 layer valid loss', marker='o')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('comare one layer and two layer train and valid loss')
    plt.legend()
    plt.show()

    epochs = np.arange(1, 31)
    plt.figure(figsize=[10,10])
    plt.plot(epochs, one_train_acc, color='red', label='1 layer train accuracy', marker='o')
    plt.plot(epochs, one_valid_acc, color='green', label='1 layer valid accuracy', marker='o')
    plt.plot(epochs, two_train_acc, color='blue', label='2 layer train accuracy', marker='o')
    plt.plot(epochs, two_valid_acc, color='lightblue', label='2 layer valid accuracy', marker='o')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('comare one layer and two layer train and valid accuracy')
    plt.legend()
    plt.show()

    print("test accuracy of one layer network: ", evaluate_acc(model_one, test_loader))
    print("test accuracy of two layer network: ", evaluate_acc(model_two, test_loader))

    ### ========== TODO : END ========== ###



if __name__ == "__main__":
    main()