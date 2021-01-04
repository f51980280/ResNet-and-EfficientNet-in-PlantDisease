import os
import argparse 
import warnings
import random
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
import ray.tune as tune
import torchvision.transforms as trans
import numpy as np
from os import listdir
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from PIL import Image
from ray.tune import run
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.optuna import OptunaSearch
from optuna.samplers import TPESampler
from efficientnet_pytorch import EfficientNet


print("==> Check devices..")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Current device: ",device)
warnings.filterwarnings("ignore")

#Also can print your current GPU id, and the number of GPUs you can use.
print("Our selected device: ", torch.cuda.current_device())
print(torch.cuda.device_count(), " GPUs is available")


class Config():
        def __init__(self):


            self.FolderNames2English_names = {}
            self.num_class = 0
            self.model = 0
            
            #. Trainging setting
            self.folder_names2code = {}
            self.early_stop = 5
            self.max_epoch = 100
            self.best_epoch = 0
            self.train_batchsize = 32
            self.eva_val_batchsize = self.train_batchsize
            self.class_num = 15
            self.each_class_item_num = {}
            self.temperature = 1
            self.alpha = 0.9
            self.lr = 0.00001
            self.criterion = nn.CrossEntropyLoss() #定義損失函數
            
            self.image_size = 224  # resolution 

            #. Basic NyResNext config
            self.net = 'MyResNeXt'  # 0: resnet18 1: MyResNeXt
            self.pretrain = False
            self.building_block = models.resnet.Bottleneck  # ResNet building block (Bottleneck or BasicBlock)
            self.layers = [3, 4, 6, 3]  # depth(number of layers)
            self.width_per_group = 64
            self.groups = 1  # if using Bottleneck for building block, it can use groups and width which only set by ResNet

            #. Basic EffientNet config
            self.depth_mult = 1
            self.width_mult = 1

            #. Basic MobileNet config
            self.layers = [0,0, 0, 0, 0, 0]  # depth, defalut as [0, 0 , 0, 0, 0, 0]
            self.width_mult = 1

            #. DataSet direction setting
            self.train_dataset_path = ''
            self.validation_dataset_path = ''

            #. Weight Sampler
            self.class_folder_num = {}
            self.wts = []
        
class MobileNet_scale(models.mobilenet.MobileNetV2):
    def __init__(self, training=True, config = Config()):
        super(MobileNet_scale, self).__init__(
                                              num_classes = config.class_num,
                                              width_mult = config.width_mult,
                                              layers = config.layers)

class MyResNeXt(models.resnet.ResNet):
    def __init__(self, training=True, config = Config()):
        super(MyResNeXt, self).__init__(block=config.building_block,
                                        layers=config.layers, 
                                        groups=config.groups, 
                                        width_per_group=config.width_per_group)

def WRSampler(dataset, config):
    config.class_folder_name = listdir(config.train_dataset_path)
    for cf in config.class_folder_name:
        config.class_folder_num[cf] = len(listdir(config.train_dataset_path + '/' + cf))
    for i in range(config.class_num):
        config.wts.append(1/config.class_folder_num[config.FolderNames2English_names[str(i)]])

    class_name_list = dataset.classes
    num_per_classes = {}
    for img in dataset.imgs:
        if  img[1] not in num_per_classes:
            num_per_classes[int(img[1])] = 1
            config.each_class_item_num[int(img[1])] = 1
        else:
            num_per_classes[int(img[1])] += 1
            config.each_class_item_num[int(img[1])] += 1
            
    each_data_wts = []
    for class_name in class_name_list:
        class_name = list(config.FolderNames2English_names.keys())[list(config.FolderNames2English_names.values()).index(class_name)]
        class_item_num = num_per_classes[int(class_name)]
        for i in range(class_item_num):
            each_data_wts.append(config.wts[int(class_name)])
    
    sampler = torch.utils.data.sampler.WeightedRandomSampler(each_data_wts, len(each_data_wts), replacement=True)
    
    return sampler

def random_dim_by_CompoundMethod(max_range):
    dim=[]
    min_range= 1
    list_dim=[]
    record = 0
   
    for i in np.linspace(min_range, max_range, 10):
        for j in np.linspace(min_range, max_range, 10):
            for k in np.linspace(min_range, max_range, 10):
                if (i*j*j*k*k < max_range+1 and i*j*j*k*k > min_range):
                    dim = []
                    dim.append(round(i,2))
                    dim.append(round(j,2))
                    dim.append(round(k,2))
                    #if record <30:
                    #print(dim)
                    record+=1
                    list_dim.append(dim)

    return list_dim

def train(model, criterion, optimizer, max_epoch, train_loader, validation_loader, config):
    t_loss = []
    v_loss = []
    training_accuracy = []
    validation_accuracy = []
    total = 0
    min_val_loss = 0.0
    min_val_error = 0.0
    early_stop_timer = 0 

    for epoch in range(max_epoch):  # loop over the dataset multiple times
        train_loss = 0.0
        validation_loss = 0.0
        correct_train = 0
        correct_validation = 0
        train_num = 0
        val_num = 0
        train_img_num = 0
        validation_img_num = 0


        ########################
        # train the model      #
        ########################

        model.train()
        for i, (inputs, labels) in enumerate(train_loader, 0):

            #change the type into cuda tensor 
            inputs = inputs.to(device) 
            labels = labels.to(device) 

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            # select the class with highest probability
            #print(outputs)
            _, pred = outputs.max(1)
            # if the model predicts the same results as the true
            # label, then the correct counter will plus 1
            correct_train += pred.eq(labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss += loss.item()
            train_num += 1
            train_img_num += len(labels)
            
        model.eval()
        for i, (inputs, labels) in enumerate(validation_loader, 0):
            # move tensors to GPU if CUDA is available
            inputs = inputs.to(device) 
            labels = labels.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(inputs)
            _, pred = outputs.max(1)
            correct_validation += pred.eq(labels).sum().item()
            # calculate the batch loss
            loss = criterion(outputs, labels)
            # update average validation loss 
            validation_loss += loss.item()
            val_num += 1
            validation_img_num += len(labels)
            
        if epoch % 1 == 0:    # print every 200 mini-batches
            val_error = 1 - correct_validation / validation_img_num
            #print('[%d, %5d] train_loss: %.3f' % (epoch, max_epoch, train_loss / train_num))
            #print('[%d, %5d] validation_loss: %.3f' % (epoch, max_epoch, validation_loss / val_num))
            #print('%d epoch, training accuracy: %.4f' % (epoch, correct_train / train_img_num))
            #print('%d epoch, validation accuracy: %.4f' % (epoch, correct_validation / validation_img_num))


            if epoch == 0:
                min_val_error = val_error
             #   print('Current best.')

            if val_error < min_val_error:
                min_val_error = val_error
                config.best_epoch = epoch
                early_stop_timer = 0
             #   print('Current best.')
            else:
                early_stop_timer += 1
                if early_stop_timer >= config.early_stop :
                    print('Early Stop.\n Best epoch is', str(config.best_epoch))
                    return validation_accuracy[(config.best_epoch)]
                    break
            t_loss.append(train_loss / train_num)
            training_accuracy.append(correct_train / train_img_num)
            validation_accuracy.append(correct_validation / validation_img_num)
            running_loss = 0.0
            validation_loss = 0.0
            train_num = 0
            val_num = 0
            correct_train = 0
            correct_validation = 0
            total = 0
            #print('-----------------------------------------')
    #return correct_validation / validation_img_num


def valid(model, validation_loader, validation_dataset, criterion, config):
    test_loss = 0.0
    correct_test = 0
    test_num = 0
    cls = np.zeros(config.class_num)
    
    model.eval()
    for i, (inputs, labels) in enumerate(validation_loader, 0):
        # move tensors to GPU if CUDA is available
        inputs = inputs.to(device) 
        labels = labels.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(inputs)
        _, pred = outputs.max(1)
        correct_test += pred.eq(labels).sum().item()
        
        for j in range(config.class_num):
                cls[j] += (pred.eq(j) * pred.eq(labels)).sum().item()

    return correct_test / len(validation_dataset)*100

def Train_Net(config):
    global model
    if My_config.model == 0 :  # EfficientNet
        print("Model is EfficientNet")
        My_config.depth_mult = int(config["dim"][0])
        My_config.width_mult = int(config["dim"][1])
        My_config.image_size = int(224*config["dim"][2])
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=My_config.class_num , width_coefficient = My_config.width_mult, depth_coefficient = My_config.depth_mult)

    if My_config.model == 1 :  # MobileNet
        print("Model is MobileNet")
        My_config.layers = [0, 0, 0, 0, 0, 0]
        layer = int((config["dim"][0])*6-6)
        for i in range(layer):
            My_config.layers[3+i%3] += 1
        My_config.width_mult = int(config["dim"][1])
        My_config.image_size = int(224*config["dim"][2])
        model = MobileNet_scale(config = My_config)

    if My_config.model == 2 :  # ResNext
        print("Model is ResNet")
        add_layer = int(48*(config["dim"][0]-1)/3)
        My_config.layers = [3, 4, 6+add_layer, 3]
        My_config.width_per_group = int(64*config["dim"][1])
        My_config.image_size = int(224*config["dim"][2])

        model = MyResNeXt(config = My_config)
        model.fc = nn.Sequential(nn.Linear(2048,1024),nn.Linear(1024,512),nn.Linear(512,256),nn.LeakyReLU(),nn.Linear(256,128),nn.LeakyReLU(),nn.Linear(128,My_config.class_num))


    transform_train = trans.Compose([
        trans.Resize((My_config.image_size, My_config.image_size)),
        trans.ToTensor(),
        trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_validation = trans.Compose([
        trans.Resize((My_config.image_size, My_config.image_size)),
        trans.ToTensor(),
        trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = torchvision.datasets.ImageFolder(root = My_config.train_dataset_path ,transform=transform_train)   
    validation_dataset = torchvision.datasets.ImageFolder(root = My_config.validation_dataset_path ,transform=transform_validation)

    sampler = WRSampler(train_dataset, My_config)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=My_config.train_batchsize, sampler=sampler)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=My_config.eva_val_batchsize, shuffle=False)

    My_config.folder_names2code = train_dataset.class_to_idx
    max_epoch = My_config.max_epoch
    learning_rate = My_config.lr
    criterion = My_config.criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=[0.9, 0.999])
    
    model = model.to(device)
    
    train(model, criterion, optimizer, max_epoch, train_loader, validation_loader, My_config)
    acc = valid(model, validation_loader, validation_dataset, criterion, My_config)
    tune.report(accuracy=acc)  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', '-tp', type=str, default='/root/wei/DataScienceProject/PlantVillage/train', help='Note for trainning data set path')
    parser.add_argument('--valid_data_path', '-vp', type=str, default='/root/wei/DataScienceProject/PlantVillage/validation', help='note for validation data set path')
    parser.add_argument('--model', '-m', type=int, default=0, help='Choose mode -> 0:EfficinetNet, 1:MobileNetV2, 2:ResNext')
    parser.add_argument('--maxrange', '-r', type=int, default=2, help='Choose compute resource -> default:2, increase number if has more resource')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Choose batch size, default as 32')
    #parser.add_argument('--width', '-wi', type=int, default=2, help='Model maximun multiple width, default as 2')
    #parser.add_argument('--depth', '-de', type=int, default=2, help='Model maximum multiple depth, default as 2')
    #parser.add_argument('--resolution', '-re', type=int, default=2, help='Model maximum multiple resolution, default as 2')
    parser.add_argument('--sample_num', '-s', type=int, default=10, help='Hyperparameter sample number, default as 10')


    My_config = Config()
    My_config.train_dataset_path = parser.parse_args().train_data_path
    My_config.validation_dataset_path = parser.parse_args().valid_data_path
    My_config.model = parser.parse_args().model
    My_config.train_batchsize
    for class_name in os.listdir(My_config.train_dataset_path):
        My_config.FolderNames2English_names[str(My_config.num_class)] = class_name
        My_config.num_class+=1
    My_config.class_num = My_config.num_class

    gpus_per_trial = 1
    sample_num = parser.parse_args().sample_num
 
    dim = random_dim_by_CompoundMethod(parser.parse_args().maxrange)
    
    algo = OptunaSearch(sampler = TPESampler())
    scheduler = AsyncHyperBandScheduler()

    search_space = {
            "dim": tune.choice(dim[0:len(dim)])
    }

    analysis = run(Train_Net,
                        name="Model_Scaling",
                        config=search_space,
                        num_samples= 10,
                        metric="accuracy",
                        mode="max",
                        search_alg=algo,
                        scheduler=scheduler,
                        resources_per_trial={"cpu": 48, "gpu": gpus_per_trial})
