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
import torchsample
# custom datasets
from train_set import MiniPlacesDataset
from runningAvg import RunningAvg
from tester import compute_output
from accuracy import accuracy
from metrics import compute_metrics

use_gpu = GPU_MODE
print('Are you using your GPU? {}'.format("Yes!" if use_gpu else "Nope :("))
if use_gpu:
    torch.cuda.set_device(CUDA_DEVICE)


### SECTION 2 - data loading and transformation

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        torchsample.transforms.RandomRotate(30),
        torchsample.transforms.RandomGamma(0.5, 1.5),
        torchsample.transforms.RandomSaturation(-0.8, 0.8),
        torchsample.transforms.RandomBrightness(-0.3, 0.3),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


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

### SECTION 3 : Writing the functions that do training and validation phase. 

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

    # params for gradient noise have been commented out, as they were not
    # used the final model. 
    # gamma = .55

    for epoch in range(start_epoch, num_epochs):
        t = epoch - 30 
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

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds5 = torch.topk(outputs.data, 5)
                _, preds1 = torch.max(outputs.data, 1)
                
                loss = criterion(outputs, labels)
                # Just so that you can keep track that something's happening and don't feel like the program isn't running.

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # add noise to the gradient
                    #for p in model_ft.parameters(): 
                    #    # add noise
                    #    p.grad = p.grad + np.random.normal(0, sigma**2)
                    optimizer.step()
                # try:
                losses.update(loss.data[0], inputs.size(0))
                acc_top_1, acc_top_5 = accuracy(outputs.data, labels.data)
                epoch_acc_1.update(acc_top_1[0], inputs.size(0))
                epoch_acc_5.update(acc_top_5[0], inputs.size(0))

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
            # save checkpoint at the end of every epoch
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
    lr = init_lr * (DECAY_WEIGHT**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


## Helper Functions

def save_checkpoint(state, is_best, filename='../../checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '../../model_best.pth.tar')
    

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

### SECTION 4 : Define model architecture

model_ft = models.resnet34(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)

criterion = nn.CrossEntropyLoss()

if use_gpu:
    criterion.cuda()
    model_ft.cuda()

optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=BASE_LR)

if len(sys.argv) < 1:
    print("Type 'tr' to train, 'test' to test, and 'metrics' to extract the error metrics'.  For 'test' and 'metrics' make sure to add another argument specifying the path of the model.")

if sys.argv[1] == 'tr': 
    checkpoint_file = '../../checkpoint.pth.tar'
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

