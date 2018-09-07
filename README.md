# miniplaces-2017
Code for miniplaces challenge 2017. 

Original implementation uses ResNet-34 and achieves op-1 accuracy of 50.2% and top-5 accuracy of 78.8%.

### For a detailed explanation of the code within, please refer to the [project report](https://github.com/landmann/miniplaces/blob/master/Report.pdf)

## Brief tour of the code base: 

- The iPython notebooks `Transforms.ipynb` and `Visualization.ipynb` are used to generate visuals for the report 
- The folder `Resnet Code` contains the code used to train and evaluate the model 
    - `accuracy.py` simply calculates the accuracy attained given a target and output vector
    - `fine_tuning_config_file.py` contains constants (like batch size and learning rate) 
    - `metrics.py` contains code to generate the output file for testing
    - `train.py` is the meat of the code base, containing code to define and train the model 
    - `runningAvg.py` contains a class to keep track of running averages (for evaluation)
    - `test_set.py` contains the test set data loader
    - `train_set.py` contains the train/validation data loader 
    - `tester.py` contains code that given a saved model, produces the output.
    
## Datasets
More information, along with the dataset, can be found at the [MiniPlaces Challenge](https://github.com/CSAILVision/miniplaces) repo.

## To run the code

To train the code and generate a checkpoint each time an epoch is finished, type:

```python
python train.py tr 
```

To produce an `output.txt` file containing the properly formatted output file for submission, type:

```python
python train.py test '<path_to_checkpoint>'
```

This code assumes that the data folder is located outside of this directory at `/data`. Checkpoint files will be saved to `../../checkpoint.pth.tar`. 
## References

Parts of our code were taken from the following tutorials. Proper citation was given in the write-up. 

1. Madan, Spandan. "Pytorch Tutorial for Fine Tuning/Transfer Learning a Resnet for Image Classification." <https://github.com/Spandan-Madan/Pytorch_fine_tuning_Tutorial>
2. PyTorch. "ImageNet Training in Pytorch". <https://github.com/pytorch/examples/blob/master/imagenet/main.py>
3. ncullen93. "High-Level Training, Data Augmentation, and Utilities for Pytorch." <https://github.com/ncullen93/torchsample>
