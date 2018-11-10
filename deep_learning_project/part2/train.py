import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import utility as util
import helper

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', action='store',
                    dest='data_dir',
                    help='Store data_dir')

parser.add_argument('--save_dir', action='store',
                    dest='save_dir',
                    help='Store save_dir')

parser.add_argument('--arch', action='store',
                    dest='arch',
                    help='Store arch')

parser.add_argument('-epochs', type=int, 
                    dest='epochs',
                    help='Store epochs')

parser.add_argument('--learning_rate', type=float, 
                    dest='learning_rate',
                    help='Store learning_rate')

parser.add_argument('--hidden_units', action='append', type=int,
                    dest='hidden_units',
                    default=[],
                    help='Add hidden_units to a list')

parser.add_argument('--device', action='store',
                    dest='device',
                    help='Store device')

parser.add_argument('--version', action='version',
                    version='%(prog)s 1.0')

results = parser.parse_args()
print('data_dir     = {!r}'.format(results.data_dir))
print('save_dir   = {!r}'.format(results.save_dir))
print('arch        = {!r}'.format(results.arch))
print('epochs        = {!r}'.format(results.epochs))
print('learning_rate       = {!r}'.format(results.learning_rate))
print('hidden_units = {!r}'.format(results.hidden_units))
print('device   = {!r}'.format(results.device))

pretrained_models = {'vgg16' : 'vgg16', 'densenet121' : 'densenet121'}
data_dir =  results.data_dir if results.data_dir != None else 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
save_dir =  results.save_dir if results.save_dir != None else 'save'
arch =  pretrained_models[results.arch] if results.arch in pretrained_models.keys() else 'vgg16'
epochs =  results.epochs if results.epochs != None else 3
learning_rate =  results.learning_rate if results.learning_rate != None else 0.001
hidden_units =  results.hidden_units if results.hidden_units else [400]
device =  util.devices['gpu'] if results.device == 'gpu' or  torch.cuda.is_available() else 'cpu'
print ('----------------------')
print ('data_dir used: ', data_dir)
print ('train_dir used: ', train_dir)
print ('valid_dir used: ', valid_dir)
print ('test_dir used: ', test_dir)
print ('save_dir used: ', save_dir)
print ('arch used: ', arch)
print ('epochs used: ', epochs)
print ('learning_rate used: ', learning_rate)
print ('hidden_units used: ', hidden_units)
print ('device used: ', device)
#-----------------------------------------------------------
image_datasets = util.getImageDatasets(data_dir, util.data_transforms)
train_data = util.getTrainData(train_dir, util.train_transforms)
valid_data = util.getValidData(valid_dir, util.valid_transforms)
test_data = util.getTestData(test_dir, util.test_transforms)

dataloaders = util.getDataLoaders(image_datasets)
trainloader = util.getTrainLoader(train_data)
validloader = util.getValidLoader(valid_data)
testloader = util.getTestLoader(test_data)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#model = models.vgg16(pretrained=True)
model = util.model_switcher(util.pretrained_models, arch)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# Hyperparameters for our network
input_size = util.get_model_input_size(model, arch)
output_size = 102
hidden_sizes = hidden_units
drop_p = 0.2

classifier = helper.createClassifierForPreTrainedModel(input_size, hidden_sizes, output_size, drop_p)
model.classifier = classifier
print(model)

# Train a model with a pre-trained network
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
print_every = 40
helper.do_deep_learning(model, trainloader, validloader, epochs, 40, criterion, optimizer, device)

model.class_to_idx = train_data.class_to_idx
# Save the checkpoint 
checkpoint = {'input_size': input_size,
              'output_size': output_size,
              'hidden_sizes': hidden_sizes,
              'state_dict': model.state_dict(),
              'classifier': model.classifier,
              'optimizer.state_dict' : optimizer.state_dict,
              'optimizer' : optimizer,
              'cat_to_name' : cat_to_name,
              'arch' : arch,
              'epochs' : epochs,
              'training_rate' : learning_rate,
              'drop_p' : drop_p,
             'class_to_idx' : model.class_to_idx}
torch.save(checkpoint, 'checkpoint.pth')