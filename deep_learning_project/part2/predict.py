import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from matplotlib.pyplot import imshow
import utility as util
import helper

parser = argparse.ArgumentParser()

parser.add_argument('--img_path', action='store',
                    dest='img_path',
                    help='Store img_path')

parser.add_argument('--checkpoint_path', action='store',
                    dest='checkpoint_path',
                    help='Store checkpoint_path')

parser.add_argument('--cat_to_name', action='store',
                    dest='cat_to_name',
                    help='Store cat_to_name')

parser.add_argument('--device', action='store',
                    dest='device',
                    help='Store device')

parser.add_argument('--top_k', type=int, 
                    dest='top_k',
                    help='Store top_k')

results = parser.parse_args()
print('img_path     = {!r}'.format(results.img_path))
print('checkpoint_path   = {!r}'.format(results.checkpoint_path))
print('cat_to_name   = {!r}'.format(results.cat_to_name))
print('device   = {!r}'.format(results.device))
print('top_k        = {!r}'.format(results.top_k))

data_dir =  'flowers'
test_dir = data_dir + '/test'
img_path =  results.img_path if results.img_path != None else (test_dir + '/' + '10/image_07090.jpg')
checkpoint_path =  results.checkpoint_path if results.checkpoint_path != None else 'checkpoint.pth'
cat_to_name =  results.cat_to_name if results.cat_to_name != None else 'cat_to_name.json'
device =  util.devices['gpu'] if results.device == 'gpu' or  torch.cuda.is_available() else 'cpu'
top_k =  results.top_k if results.top_k != None else 5

print ('-------------------------')
print ('img_path used: ', img_path)
print ('checkpoint_path used: ', checkpoint_path)
print ('device used: ', device)
print ('top_k Used: ' ,top_k)
print ('cat_to_name Used: ' ,cat_to_name)
#-----------------------------------------------------------
model = helper.load_checkpoint('checkpoint.pth')
model
#print(model)

img_label = img_path.split(test_dir + '/')[1].split('/')[0]
cat_to_name = helper.load_cat_to_name(cat_to_name)
title = cat_to_name[img_label]
print ('-------------------------')
print('flower category = ',title)

probs, labs, flowers = util.predict(img_path, model, cat_to_name) 
print('probs = ',probs)
print('labs = ',labs)
print('flowers = ',flowers)