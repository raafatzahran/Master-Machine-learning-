import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json

def load_cat_to_name (json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

# Implement a function to create a classifier for the pretrained model
def createClassifierForPreTrainedModel (input_size, hidden_sizes, output_size, drop_p = 0.5):
    classifier = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('Dropout1', nn.Dropout(p=drop_p)),
                      ('output', nn.Linear(hidden_sizes[0], output_size)),
                      ('softmax', nn.LogSoftmax(dim=1))]))
    return classifier

# Implement a function for the validation pass
def validation(model, validloader, criterion, device='cpu'):
    test_loss = 0
    accuracy = 0
    
    model.to(device)
    
    for images, labels in validloader:

        images, labels = images.to(device), labels.to(device)
        
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

# Implement a function for the training pass
def do_deep_learning(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device='cpu'):
    epochs = epochs
    print_every = print_every
    steps = 0
    running_loss = 0

    # change to cuda
    model.to(device)

    for e in range(epochs):
        model.train()
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    valid_loss, valid_accuracy = validation(model, validloader, criterion, device)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Test Accuracy: {:.3f}".format(valid_accuracy/len(validloader)))

                running_loss = 0

                # Make sure training is back on
                model.train()

# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    #model.optimizer = checkpoint('optimizer')
    return model

def load_key_from_checkpoint(filepath, key):
    checkpoint = torch.load(filepath)
    return checkpoint[key]

