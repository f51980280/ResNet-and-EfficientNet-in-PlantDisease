# ResNet and EfficientNet in PlantDisease
Model Scaling by compound scaling method. 

In this project, we try to implement a compound scaling method proposed by「EfficientNet : Rethinking Model Scaling for Convolutional Neural Networks」. It scales a model's depth, width, resolution in a systematic method for enhancing a model’s accuracy. Using this method and analyzing its accuracy and efficiency in ResNet and EfficientNet. We also add MobileNet for evaluation of its performance. After that, we find its compound scaling method still has numerous dimension candidates. Thus, we tried to apply hyper-parameter search with its scaling dimension.

## We had implement the task base on following backbone modules:
1. Pytorch. 
2. Optuna.  # for HyperparameterSearch
