import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision 
from torchvision import models

from test_set import MiniPlacesTestLoader
from fine_tuning_config_file import *  

import sys
import os

def run_test(model, use_gpu, output_path, test_options):
    print("use_gpu", use_gpu)
    print("from test", torch.cuda.current_device())
    testdata = MiniPlacesTestLoader(**test_options)
    testloader = torch.utils.data.DataLoader(testdata, batch_size=1)
    model.eval()
    # for each example in the data loader, run it through, get an answer, and format it 
    output_labels = []
    for img, filename in testloader: 
        inputs = Variable(img.float().cuda()) if use_gpu else Variable(img)
        # run through network and get output

        output = model(inputs)
        # get the top 5 labels: should be a 1x5 tensor
        _, prediction = output.topk(5, dim=1, largest=True, sorted=True)
        # print(" ".join(str(prediction)))
        prediction = [str(elt) for elt in prediction.data[0]]
        # create the correct output
        prediction = " ".join(prediction) 
        print('here', filename)
        output_labels.append(filename[0] + " " + prediction)

    # output to file
    output_labels = "\n".join(output_labels)
    with open(output_path, 'w') as outfile: 
        outfile.write(output_labels)

def compute_output(checkpoint_file, output_file_name, model, use_gpu, test_options, comp_funct=run_test):# change the model to load the right thing
    parent_path = "/".join(checkpoint_file.split("/")[:-1])
    save_path = os.path.join(parent_path, output_file_name)

    checkpoint = torch.load(checkpoint_file)
    start_epoch = checkpoint['epoch']
    best_top1 = checkpoint['best_top1']
    best_top5 = checkpoint['best_top5']
    loss_history = checkpoint['loss_history']
    model.load_state_dict(checkpoint['state_dict'])
    print("Loaded checkpoint.")
    print("Epoch: {}, Top 1: {:.4f}, Top 5: {:.4f}".format(start_epoch, best_top1, best_top5))
    print("Saving results to %s" % save_path)

    comp_funct(model, use_gpu, output_path=save_path, test_options=test_options)
