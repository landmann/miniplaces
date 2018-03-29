### Section 1 - First, let's import everything we will be needing.

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import shutil
from fine_tuning_config_file import *

# custom datasets
from train_set import MiniPlacesDataset
from runningAvg import RunningAvg
from tester import compute_output
from accuracy import accuracy
from metrics import compute_metrics
## If you want to keep a track of your network on tensorboard, set USE_TENSORBOARD TO 1 in config file.

if USE_TENSORBOARD:
    from pycrayon import CrayonClient
    cc = CrayonClient(hostname=TENSORBOARD_SERVER)
    try:
        cc.remove_experiment(EXP_NAME)
    except:
        pass
    foo = cc.create_experiment(EXP_NAME)


## If you want to use the GPU, set GPU_MODE TO 1 in config file

use_gpu = GPU_MODE
print('Are you using your GPU? {}'.format("Yes!" if use_gpu else "Nope :("))
if use_gpu:
    torch.cuda.set_device(CUDA_DEVICE)


### SECTION 2 - data loading and shuffling/augmentation/normalization : all handled by torch automatically.

# This is a little hard to understand initially, so I'll explain in detail here!

# For training, the data gets transformed by undergoing augmentation and normalization. 
# The RandomSizedCrop basically takes a crop of an image at various scales between 0.01 to 0.8 times the size of the image and resizes it to given number
# Horizontal flip is a common technique in computer vision to augment the size of your data set. Firstly, it increases the number of times the network gets
# to see the same thing, and secondly it adds rotational invariance to your networks learning.


# Just normalization for validation, no augmentation. 

# You might be curious where these numbers came from? For the most part, they were used in popular architectures like the AlexNet paper. 
# It is important to normalize your dataset by calculating the mean and standard deviation of your dataset images and making your data unit normed. However,
# it takes a lot of computation to do so, and some papers have shown that it doesn't matter too much if they are slightly off. So, people just use imagenet
# dataset's mean and standard deviation to normalize their dataset approximately. These numbers are imagenet mean and standard deviation!

# If you want to read more, transforms is a function from torchvision, and you can go read more here - http://pytorch.org/docs/master/torchvision/transforms.html
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        #transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}



# Enter the absolute path of the dataset folder below. Keep in mind that this code expects data to be in same format as Imagenet. I encourage you to
# use your own dataset. In that case you need to organize your data such that your dataset folder has EXACTLY two folders. Name these 'train' and 'val'
# Yes, this is case sensitive. The 'train' folder contains training set and 'val' fodler contains validation set on which accuracy is measured. 

# The structure within 'train' and 'val' folders will be the same. They both contain one folder per class. All the images of that class are inside the 
# folder named by class name.

# So basically, if your dataset has 3 classes and you're trying to classify between pictures of 1) dogs 2) cats and 3) humans,
# say you name your dataset folder 'data_directory'. Then inside 'data_directory' will be 'train' and 'test'. Further, Inside 'train' will be 
# 3 folders - 'dogs', 'cats', 'humans'. All training images for dogs will be inside this 'dogs'. Similarly, within 'val' as well there will be the same
# 3 folders. 

## So, the structure looks like this : 
# data_dar
#      |- train 
#            |- dogs
#                 |- dog_image_1
#                 |- dog_image_2
#                        .....

#            |- cats
#                 |- cat_image_1
#                 |- cat_image_1
#                        .....
#            |- humans
#      |- val
#            |- dogs
#            |- cats
#            |- humans

data_dir = os.path.expanduser(DATA_PATH)

dsets = {}
for mode in ['train', 'val']: 
    dsets[mode] = MiniPlacesDataset(
        photos_path=os.path.join(data_dir, 'images/'),
        labels_path=os.path.join(data_dir, mode + '.txt'),
        transform = data_transforms[mode]
    )


dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=25)
                for x in ['train', 'val']}

dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
# dset_classes = dsets['train'].classes

### SECTION 3 : Writing the functions that do training and validation phase. 

# These functions basically do forward propogation, back propogation, loss calculation, update weights of model, and save best model!


## The below function will train the model. Here's a short basic outline - 

# For the number of specified epoch's, the function goes through a train and a validation phase. Hence the nested for loop. 

# In both train and validation phase, the loaded data is forward propogated through the model (architecture defined ahead). 
# In PyTorch, the data loader is basically an iterator. so basically there's a get_element function which gets called everytime 
# the program iterates over data loader. So, basically, get_item on dset_loader below gives data, which contains 2 tensors - input and target. 
# target is the class number. Class numbers are assigned by going through the train/val folder and reading folder names in alphabetical order.
# So in our case cats would be first, dogs second and humans third class.

# Forward prop is as simple as calling model() function and passing in the input. 

# Variables are basically wrappers on top of PyTorch tensors and all that they do is keep a track of every process that tensor goes through.
# The benefit of this is, that you don't need to write the equations for backpropogation, because the history of computations has been tracked
# and pytorch can automatically differentiate it! Thus, 2 things are SUPER important. ALWAYS check for these 2 things. 
# 1) NEVER overwrite a pytorch variable, as all previous history will be lost and autograd won't work.
# 2) Variables can only undergo operations that are differentiable.

def train_model(model, criterion, optimizer, lr_scheduler, checkpoint_file, num_epochs=100):
    since = time.time()
    print("##"*10)
    best_model = model

    # Loss history is saved below. Saved every epoch. 
    loss_history = {'train':[0], 'val':[0]}
    start_epoch  = 0
    best_top1    = 0
    best_top5    = 0

    if checkpoint_file:
        print()
        if os.path.isfile(checkpoint_file):
            try:
                checkpoint = torch.load(checkpoint_file)
                start_epoch = checkpoint['epoch']
                best_top1 = checkpoint['best_top1']
                best_top5 = checkpoint['best_top5']
                loss_history = checkpoint['loss_history']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(checkpoint_file, checkpoint['epoch']))
            except:
                print("Found the file, but couldn't load it.")
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_file))

    # Freeze all layers except last one and add a new layer at the end, initialized from 0 (like previous few).
    # Resnet 18 has 10 children, so freeze 8 of the 10 

    #for i, child in enumerate(model.children()):
    #    if i < 8: 
    #        print("Freezing layer {}".format(i))
    #        for param in child.parameters():
    #            param.requires_grad = False 
       
    #num_ftrs = model.fc.in_features
    # model.fc stands for last fully connected layer.
    #model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    #model.fc.weight.data.normal_(0, 0.01)
#   model.fc.bias.data.fill_(0)

    # params for gradient noise 
    t = start_epoch
    gamma = .55

    for epoch in range(start_epoch, num_epochs):
        sigma = BASE_LR/((1+t)**gamma)

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                mode='train'
                optimizer = lr_scheduler(optimizer, epoch)
                model.train()  # Set model to training mode
            else:
                model.eval()
                mode='val'

            losses = RunningAvg()
            epoch_acc_1 = RunningAvg()
            epoch_acc_5 = RunningAvg()

            counter=0
            # Iterate over data.
            for data in dset_loaders[phase]:
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.float().cuda())
                    labels = Variable(labels.long().cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # Set gradient to zero to delete history of computations in previous epoch. Track operations so that differentiation can be done automatically.
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds5 = torch.topk(outputs.data, 5)
                _, preds1 = torch.max(outputs.data, 1)
                
                loss = criterion(outputs, labels)
                # Just so that you can keep track that something's happening and don't feel like the program isn't running.

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    for p in model_ft.parameters(): 
                        #if p.requires_grad:
                        # add noise
                        p.grad = p.grad + np.random.normal(0, sigma**2)
                    optimizer.step()
                # try:
                losses.update(loss.data[0], inputs.size(0))
                acc_top_1, acc_top_5 = accuracy(outputs.data, labels.data)
                epoch_acc_1.update(acc_top_1[0], inputs.size(0))
                epoch_acc_5.update(acc_top_5[0], inputs.size(0))
                # except:
                # print(counter, epoch_acc_1.avg, losses.avg, epoch_acc_5.avg)
                # print('unexpected error, could not calculate loss or do a sum.')

                if counter % 100==0:
                    print("It: {}, Loss: {:.4f}, Top 1: {:.4f}, Top 5: {:.4f}".format(counter, losses.avg, epoch_acc_1.avg, epoch_acc_5.avg))
                    #print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                counter+=1
            # At the end of every epoch, tally up losses and accuracies
            time_elapsed = time.time() - since

            print_stats(epoch_num=epoch, train=mode, batch_time=time_elapsed, loss=losses, top1=epoch_acc_1, top5=epoch_acc_5)  

            loss_history[mode].append(losses.avg)
            is_best = epoch_acc_5.avg > best_top5
            best_top5 = max(epoch_acc_5.avg, best_top5)
            save_checkpoint({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'best_top1': epoch_acc_1.avg,
                'best_top5': best_top5,
                'loss_history': loss_history,
                'optimizer': optimizer.state_dict(),
                }, is_best)
            print('checkpoint saved!')

            # deep copy the model
            if phase == 'val':
                if USE_TENSORBOARD:
                    foo.add_scalar_value('epoch_loss',losses.avg,step=epoch)
                    foo.add_scalar_value('epoch_acc_1',epoch_acc_1,step=epoch)
                if epoch_acc_1.avg > best_top1:
                    best_top1= epoch_acc_1.avg
                    best_model = copy.deepcopy(model)
                    print('new best accuracy = ',best_top1)

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_top1))
    print('returning and looping back')
    return best_model

# This function changes the learning rate over the training model.
def exp_lr_scheduler(optimizer, epoch, init_lr=BASE_LR, lr_decay_epoch=EPOCH_DECAY):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    init_lr = 0.005
    lr = init_lr * (DECAY_WEIGHT**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


## Helper Functions

def save_checkpoint(state, is_best, filename='../../checkpoint_new_layers.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '../../model_best_new_layers.pth.tar')
    

def print_stats(epoch_num=None, it_num=None, train=True, batch_time=None, loss=None, top1=None, top5=None): 
    progress_string = "Epoch %d" % epoch_num if epoch_num else ''
    if it_num is not None: 
        progress_string += ", Iteration %d" % it_num
    else: 
        progress_string += " finished"
    progress_string += ", Training set = %s\n" % (train)
    print(progress_string + 
          #"\tBatch time: {batch_time.val:.3f}, Batch time average: {batch_time.val:.3f}\n"
          "\tLoss: {loss.avg:.4f}\n Accuracies: \n"
          "\tTop 1: {top1.avg:.3f}%\n"
          "\tTop 5: {top5.avg:.3f}%\n".format(batch_time=batch_time, loss=loss, top1=top1, top5=top5))

def save(filename='trained_alexnet'):
    """Saves model using file numbers to make sure previous models are not overwritten"""
    filenum = 0
    while (os.path.exists(os.path.abspath('{}_v{}.pt'.format(filename, filenum)))):
        filenum += 1
    torch.save(model.state_dict(), '{}_v{}.pt'.format(filename, filenum))

### SECTION 4 : DEFINING MODEL ARCHITECTURE.

# We use Resnet18 here. If you have more computational power, feel free to swap it with Resnet50, Resnet100 or Resnet152.
# Since we are doing fine-tuning, or transfer learning we will use the pretrained net weights. In the last line, the number of classes has been specified.
# Set the number of classes in the config file by setting the right value for NUM_CLASSES.



model_ft = models.resnet18(pretrained=False)
#model_ft= models.inceptionv4(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)


criterion = nn.CrossEntropyLoss()

if use_gpu:
    criterion.cuda()
    model_ft.cuda()

optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=BASE_LR)
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=BASE_LR, momentum=.9)

if len(sys.argv) < 1:
    print("Type 'tr' to train, 'test' to test, and 'metrics' to extract the error metrics'.  For 'test' and 'metrics' make sure to add another argument specifying the path of the model.")

if sys.argv[1] == 'tr': 
    #checkpoint_file = '../../saved_models/ResNet-18/checkpoint.pth.tar'
    checkpoint_file = os.path.expanduser('~/checkpoint_new_layers.pth.tar')
    # Run the functions and save the best model in the function model_ft.
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, checkpoint_file, num_epochs=100)

    # Save model
    model_ft.save_state_dict('fine_tuned_best_model.pt')

if sys.argv[1] in ('test', 'metrics'): 
    print("from file")
    print(torch.cuda.current_device())
    model_path = sys.argv[2]
    output_file_name = sys.argv[3] if len(sys.argv) > 3 else 'output.txt'

    test_options = {
       'photos_path': os.path.expanduser(TEST_DATA_PATH),
       'transform': data_transforms['val']
    }
    if sys.argv[1] == 'test':
        compute_output(model_path, output_file_name, model_ft, use_gpu, test_options)
    elif sys.argv[1] == 'metrics':
        test_options['photos_path'] = os.path.expanduser('~/data/images/')
        compute_output(model_path, 'train_'+output_file_name, model_ft, use_gpu, test_options, compute_metrics) 
        compute_output(model_path, 'val_'+output_file_name, model_ft, use_gpu, test_options, compute_metrics)

    

