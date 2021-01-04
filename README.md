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
https://github.com/lukemelas/EfficientNet-PyTorch. 
https://github.com/optuna/optuna .  # for Optuna HyperparameterSearch. 
torchvision that matches the PyTorch installation. You can install them together at pytorch.org to make sure of this.  
OpenCV, needed by demo and visualization. 

## Several files must be changed by manually
```
mobilenet_scaling.py: 
  {your env path}/lib/python3.7/site-packages/torchvision/models/mobilenet.py -> change to mobilenet.py(mobilenet_scaling.py)

resnet_scaling.py: 
  {your env path}/lib/python3.7/site-packages/torchvision/models/resnet.py -> change to resnet.py(resnet_scaling.py)
  
model.py:  # for EfficientNet scaling
 {your evn path}lib/python3.7/site-packages/efficientnet_pytorch/model.py ->  change to model.py(model.py)
```

## Use HyperSearch_with_model_scaling.py
If you follow the below rules, you will train the model which you want to use.   

```
python HyperSearch_with_model_scaling.py -h  # for more help
usage: HyperSearch_with_model_scaling.py [-h]
                                         [--train_data_path TRAIN_DATA_PATH]
                                         [--valid_data_path VALID_DATA_PATH]
                                         [--model MODEL] [--maxrange MAXRANGE]
                                         [--batch_size BATCH_SIZE]
                                         [--sample_num SAMPLE_NUM]

optional arguments:
  -h, --help            show this help message and exit
  --train_data_path TRAIN_DATA_PATH, -tp TRAIN_DATA_PATH, Note for trainning data set path. 
  --valid_data_path VALID_DATA_PATH, -vp VALID_DATA_PATH, Note for validation data set path. 
  --model MODEL, -m MODEL, Choose mode -> 0:EfficinetNet, 1:MobileNetV2, 2:ResNext. 
  --maxrange MAXRANGE, -r MAXRANGE, Choose compute resource -> default:2, increase number, if has more resource. 
  --batch_size BATCH_SIZE, -b BATCH_SIZE, Choose batch size, default as 32. 
  --sample_num SAMPLE_NUM, -s SAMPLE_NUM, Hyperparameter sample number, default as 10. 
```
Fianlly, you can get the best scaling dimension for your dataset and model.  
![image](https://github.com/f51980280/ResNet-and-EfficientNet-in-PlantDisease/blob/main/result/mobileNet.png)
