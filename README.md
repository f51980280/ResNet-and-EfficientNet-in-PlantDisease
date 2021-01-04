# ResNet and EfficientNet in PlantDisease
Model Scaling by compound scaling method. 

In this project, we try to implement a compound scaling method proposed by「EfficientNet : Rethinking Model Scaling for Convolutional Neural Networks」. It scales a model's depth, width, resolution in a systematic method for enhancing a model’s accuracy. Using this method and analyzing its accuracy and efficiency in ResNet and EfficientNet. We also add MobileNet for evaluation of its performance. After that, we find its compound scaling method still has numerous dimension candidates. Thus, we tried to apply hyper-parameter search with its scaling dimension. Using the PlantDisease Dataset for the model scaling experiment.  

## We had implement the task base on following backbone modules:
1. https://pytorch.org/docs/stable/torchvision/index.html 
2. https://github.com/lukemelas/EfficientNet-PyTorch. 
3. https://github.com/optuna/optuna .  # for Optuna HyperparameterSearch. 

## Hardware
The following specs were used to create the original solution.

Ubuntu 16.04  
Intel® Xeon® Gold 6136 Processor @ 3.0GHHz.  
1x NVIDIA NVIDIA TESLA V100 32GB  


## Requirements
Using Anaconda is strongly recommended.  
Python >= 3.6(Conda).   
PyTorch 1.3.  
torchvision that matches the PyTorch installation. You can install them together at pytorch.org to make sure of this.  
OpenCV, needed by demo and visualization. 

## Several files must be changed by manually
```
mobilenet_scaling.py: 
  {your evn path}/lib/python3.7/site-packages/torchvision/models/mobilenet.py -> change to mobilenet.py(mobilenet_scaling.py)

resnet_scaling.py: 
  {your evn path}/lib/python3.7/site-packages/torchvision/models/resnet.py -> change to resnet.py(resnet_scaling.py)
  
model.py:
 {your evn path}lib/python3.7/site-packages/efficientnet_pytorch/model.py ->  change to model.py(model.py)
```


