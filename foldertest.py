# Copyright <2019> Edward Verenich <verenie@clarkson.edu>
# MIT license <https://opensource.org/licenses/MIT>
# This script iteratively tests models within a 
# specified models folder. 

from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import csv
import metrx as mx

DATA_DIR = 'data/ood_tel'
MODELS_FOLDER = 'models'
LOG_DIR = 'logs'
STATUS_FILE = os.path.join(LOG_DIR, 'resultsOOD.csv')
BATCH_SIZE = 6
WORKERS = 6
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
# check for cuda and set gpu or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
# data tranforms
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
    ])
}

# use the ImageFolder class from torchvision that accepts datasets structured by datax.py
# create datasets with appropriate transformation
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in data_transforms}
# create dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS) for x in data_transforms}

dataset_sizes = {x: len(image_datasets[x]) for x in data_transforms}
class_names = image_datasets['test'].classes
number_classes = len(class_names)


def test_model(model_name):
    # load the model
    model_path = os.path.join(MODELS_FOLDER,model_name)
    model = torch.load(model_path)
    model.eval()
    model.to(device)
    cm = torch.zeros(number_classes,number_classes)
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            # forward pass
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            # record performance in the confusion matrix
            for t, p in zip(predictions.view(-1), labels.view(-1)):
                cm[t.long(), p.long()] +=1
    # record confusion matrix csv data
    with open(STATUS_FILE, 'a') as gt:
        # set print out precision
        R = 4
        writer = csv.writer(gt)
        writer.writerow([model_name,
            round(mx.accuracy(cm).item(),R),
            round(mx.sensitivity(cm, 0).item(),R),
            round(mx.precision(cm, 0).item(),R),
            round(mx.number_false_negative(cm, 0).item()),
            round(mx.number_false_positive(cm, 0).item())])
    
    
    print("----------------------------------------")
    print("Model name: ", model_name)
    print("Confusion Matrix: \n", cm)
    print("MX target sensitivity: ", mx.sensitivity(cm,0))
    print("MX class accuracy: ", mx.per_class_accuracy(cm))
    print("----------------------------------------")

if __name__ == '__main__':
    X = os.listdir(MODELS_FOLDER)
    print("Number of models: ", len(X))
    # write column names to csv file
    with open(STATUS_FILE, 'a') as gt:
        writer = csv.writer(gt)
        writer.writerow(['model','accuracy','sensitivity','precision',
                'false neg','false pos'])
    for name in X:
        test_model(name)