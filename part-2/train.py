
###########
# Imports #
###########
# Core
import os
import argparse

# 3rd-party
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Project
from workspace_utils import keep_awake
from utils import save_model, base_model, get_valid_device



################
# Data Loading #
################
def load_datasets(data_dir):
    """
        Load datasets with appropiate transforms
    """
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    ])
    
    train_dir, valid_dir, test_dir = (os.path.abspath(os.path.join(str(data_dir), subdir))  + '/' for subdir in ['train', 'valid', 'test'])

    # Load the datasets with transforms
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    return train_dataset, valid_dataset, test_dataset


def data_loaders(data_dir, batch_size=32, shuffle=True):
    """
        Makes data loaders from data directory
    
        Expects 3 sub-directories, ./train, ./valid, ./test
    """
    train_dataset, valid_dataset, test_dataset = load_datasets(data_dir)
    
    # Using the image datasets, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return train_loader, valid_loader, test_loader



###############################
# Model Building and Training #
###############################
def last_module(model):
    """
        Finds last element of OrderedDict

        some models call it 'classifier' some 'fc'
        this function makes calls make generic
    """
    key = str(next(reversed(model._modules)))
    modules_d = model._modules
    return modules_d, key


def hidden_node_count(module):
    """
        Returns the numbers of inputs and hidden layer inputs
    """
    if type(module) == torch.nn.modules.container.Sequential:
        # strip Sequential 
        inputs = list(module)[0].in_features
        outputs = list(module)[0].out_features
    else:
        inputs = module.in_features
        outputs = module.out_features

    return inputs, outputs


def build_model(arch, learning_rate, num_labels, hidden_units=None):
    """
        Loads base model and modifies classifiers
    """
    model = base_model(arch)
    module_d, key = last_module(model)

    input_nodes, hidden_nodes = hidden_node_count(model._modules[key])

    if hidden_units is not None:
        if hidden_nodes < num_labels:
            print(f'Too few hidden units, need at least {num_labels} outputs.\nSetting hidden_units to {hidden_nodes}')
        elif hidden_nodes > input_nodes:
            print(f'Too many hidden units, need at most {input_nodes} units.\nSetting hidden_units to {hidden_nodes}')
        else:
            hidden_nodes = hidden_units
        
    classifier = nn.Sequential(
        nn.Linear(input_nodes, hidden_nodes),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(hidden_nodes, hidden_nodes),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(hidden_nodes, num_labels),
        nn.LogSoftmax(dim=1)
    )

    model._modules[key] = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model._modules[key].parameters(), lr=learning_rate)
    
    return model, criterion, optimizer



#######################
# Testing the Network #
#######################
def accuracy(logps, labels):
    ps = torch.exp(logps)
    top_ps, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    acc = torch.mean(equals.type(torch.FloatTensor)).item()
    return acc

def train_network(model, criterion, optimizer, train_loader, test_loader, valid_loader, epochs, device, print_every=50):
    """
        Trains network according to specifications
    """
    step = 0
    
    model.to(device)

    for epoch in keep_awake(range(epochs)):
        train_loss = 0
        train_accuracy = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            step += 1

            optimizer.zero_grad()

            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            train_accuracy += accuracy(logps, labels)

            if step % print_every == 0:
                model.eval()

                # testing
                test_loss = 0
                test_accuracy = 0
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)

                    with torch.no_grad():
                        logps = model(images)
                        loss = criterion(logps, labels)

                    test_loss += loss.item()
                    test_accuracy += accuracy(logps, labels)

                # validation
                valid_loss = 0
                valid_accuracy = 0
                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)

                    with torch.no_grad():
                        logps = model(images)
                        loss = criterion(logps, labels)

                    valid_loss += loss.item()
                    valid_accuracy += accuracy(logps, labels)

                # report
                print(
                    f"Epoch {epoch+1}/{epochs} "
                    f"Train loss: {train_loss/print_every:.3f} "
                    f"Train accuracy: {train_accuracy/print_every:.3f} "
                    f"Test loss: {test_loss/len(test_loader):.3f} "
                    f"Test accuracy: {test_accuracy/len(test_loader):.3f} "
                    f"Valid loss: {valid_loss/len(valid_loader):.3f} "
                    f"Valid accuracy: {valid_accuracy/len(valid_loader):.3f}"
                )

                model.train()

    return model, test_accuracy/len(test_loader)


########
# Main #
########
def main(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    """
        Trains image classification model
    """
    device = get_valid_device(gpu)
    
    train_loader, valid_loader, test_loader = data_loaders(data_dir)

    idx_to_class = {str(v):str(k) for k,v in train_loader.dataset.class_to_idx.items()}
    num_labels = len(idx_to_class)
    
    model, criterion, optimizer = build_model(arch, learning_rate, num_labels, hidden_units)
    model.idx_to_class = idx_to_class
    
    print('Starting training...\n')
    model, accuracy = train_network(model, criterion, optimizer, train_loader, test_loader, valid_loader, epochs, device, print_every=50)
    print(f'Done training. Accuracy: {accuracy*100:2.2f} %\n')
    
    if save_dir:
        print('Saving model...')
        filename = f'{arch}-accuracy-{accuracy*100:2.2f}.pth'
        filepath = os.path.abspath(os.path.join(save_dir, filename))
        save_model(filepath, model, epochs, optimizer, idx_to_class)


if __name__ == '__main__':
    """
        Basic usage: python train.py data_directory
        Prints out training loss, validation loss, and validation accuracy as the network trains
        Options:
            Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
            Choose architecture: python train.py data_dir --arch "vgg13"
            Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
            Use GPU for training: python train.py data_dir --gpu
    """

    parser = argparse.ArgumentParser(
        description=
        """
        Image classifier training module
        
        Basic usage: python train.py data_directory
        Prints out training loss, validation loss, and validation accuracy as the network trains
        Options:
            Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
            Choose architecture: python train.py data_dir --arch "vgg13"
            Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
            Use GPU for training: python train.py data_dir --gpu
        """
    )

    parser.add_argument(action='store',
                        type=str,
                        dest='data_dir',
                        help='Data directory')

    parser.add_argument('--save_dir', action='store',
                        default=None,
                        type=str,
                        dest='save_dir',
                        help='Checkpoint save directory')
    
    parser.add_argument('--arch', action='store',
                        type=str,
                        default='vgg11',
                        dest='arch',
                        help='Choose architecture')

    parser.add_argument('--gpu', action='store_true',
                        default=False,
                        dest='gpu',
                        help='Use GPU for training')
    
    parser.add_argument('--learning_rate', action='store',
                        type=float,
                        default=0.00007,
                        dest='learning_rate',
                        help='Set learning rate (alpha)')
    
    parser.add_argument('--hidden_units', action='store',
                        type=int,
                        default=None,
                        dest='hidden_units',
                        help='Set number of hidden units')
    
    parser.add_argument('--epochs', action='store',
                        type=int,
                        default=3,
                        dest='epochs',
                        help='Set number of epochs')
    
    options = parser.parse_args()
    
    main(options.data_dir, options.save_dir, options.arch, options.learning_rate, options.hidden_units, options.epochs, options.gpu)
