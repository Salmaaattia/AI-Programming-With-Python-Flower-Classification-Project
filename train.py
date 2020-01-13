import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torchvision.models as models
from collections import OrderedDict
from torch import optim
from PIL import Image
import pandas as pd
import argparse
import json

parser = argparse.ArgumentParser (description = "Parser for train.py script")

parser.add_argument ('data_dir', help = 'Provide data directory. required argument', type = str)
parser.add_argument ('--save_dir', help = 'Provide saving directory. Optional argument', type = str)
parser.add_argument ('--arch', help = 'Vgg13 can be used if this argument specified, otherwise Alexnet will be used', type = str)
parser.add_argument ('--lrn', help = 'Learning rate,Optional, default value is 0.001', type = float)
parser.add_argument ('--hidden_layers', help = 'Hidden layers in Classifier.optional, Default value is 2048', type = int)
parser.add_argument ('--epochs', help = 'Number of epochs', type = int)
parser.add_argument ('--GPU', help = "choose GPU", type = str)


args = parser.parse_args ()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#use cuda or cpu
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

#loading data and defining transforms
if data_dir:
    train_transform = transforms.Compose ([transforms.RandomRotation (30),
                                                transforms.RandomResizedCrop (224),
                                                transforms.RandomHorizontalFlip (),
                                                transforms.ToTensor (),
                                                transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])

    valid_transform = transforms.Compose ([transforms.Resize (255),
                                                transforms.CenterCrop (224),
                                                transforms.ToTensor (),
                                                transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])

    test_transform = transforms.Compose ([transforms.Resize (255),
                                                transforms.CenterCrop (224),
                                                transforms.ToTensor (),
                                                transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])
    # TODO: Load the datasets with ImageFolder 
    train_image_dataset = datasets.ImageFolder(train_dir,transform=train_transform)
    test_image_dataset= datasets.ImageFolder(test_dir,transform=test_transform)
    valid_image_dataset = datasets.ImageFolder(valid_dir,transform=valid_transform)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_image_dataset, batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_image_dataset, batch_size=64, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_image_dataset, batch_size=64, shuffle=True)


#mapping 
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def def_network(modeltype='',hiddenlayers=2048,learningrate=0.001):
    #load pretrained model
    if(modeltype=='vgg16'):
        pretrained_model= models.vgg16(pretrained=True)
        features_in=25088
    elif(modeltype=='vgg13'):
        pretrained_model=models.vgg13(pretrained=True)
        features_in=25088
    else:
        pretrained_model=models.alexnet(pretrained=True)
        features_in=9216
    #freeze parameters in the feature section
    for param in pretrained_model.parameters(): 
        param.requires_grad = False
        
    #define new classifier
    classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (features_in, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p =0.3)),
                            ('fc2', nn.Linear (4096, hiddenlayers)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (hiddenlayers, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
    
    #replace the classifier
    pretrained_model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(pretrained_model.classifier.parameters(), lr=learningrate)
    
    return pretrained_model,optimizer,criterion

# Defining validation Function. will be used during training
def validate(model,validloader,criterion):
    model.to (device)

    validloss = 0
    accuracy = 0
    for inputs, labels in validloader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        validloss += criterion(output, labels).item()
        p = torch.exp(output)
        equality = (labels.data == p.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return validloss, accuracy


if(args.arch=='vgg16'):
     modeltype='vgg16'
elif(args.arch=='vgg16'):
    modeltype='vgg16'
else:
    modeltype=''

if args.lrn: 
   lr=args.lrn
else:
    lr=0.001

if args.hidden_layers:
    hidden_layers=args.hidden_layers
else:
    hidden_layers=2048

model,optimizer,criterion= def_network(modeltype=modeltype, hiddenlayers=hidden_layers,learningrate=lr)





model.to (device) 
if args.epochs:
    epochs = args.epochs
else:
    epochs = 7

print_every = 40
steps = 0

#runing through epochs
for e in range (epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate (train_dataloader):
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad ()

        outputs = model.forward (inputs) 
        loss = criterion (outputs, labels) 
        loss.backward ()
        optimizer.step () 
        running_loss += loss.item () 

        if steps % print_every == 0:
            model.eval () #switching to evaluation mode so that dropout is turned off

            with torch.no_grad():
                valid_loss, accuracy = validate(model, valid_dataloader, criterion)

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Loss: {:.4f}.. ".format(running_loss/print_every),
                  "Validation Loss: {:.4f}.. ".format(valid_loss/len(valid_dataloader)),
                  "Valid Accuracy: {:.4f}%".format(accuracy/len(valid_dataloader)*100))

            running_loss = 0
            model.train()

model.to ('cpu') 
# Save the checkpoint
model.class_to_idx = train_image_dataset.class_to_idx 

checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict (),
              'arch': modeltype,
              'class_to_idx':  model.class_to_idx
             }

if args.save_dir:
    torch.save (checkpoint, args.save_dir + '/checkpoint.pth')
else:
    torch.save (checkpoint, 'checkpoint.pth')