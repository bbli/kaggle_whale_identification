from Model import *
from DataSet import *
from utils import *

from time import time
# from tensorboardX import SummaryWriter
from ml_board import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from math import cos,pi
def cosine_drop(cos_period,explore_period,decay):
    factor = 1
    def f(episode):
        nonlocal factor
        global loss
        if episode!=0 and (episode %explore_period == 0):
            if loss>40:
                factor = factor*decay
                # print("Dropped Factor to: ",factor)
        modulus = episode % cos_period
        return factor*0.5*(1.1+cos(pi*modulus/cos_period))
    return f
################ **Loading DataSet** ##################
resizer = ReSizer(200,350)
image_scaler = ImageScalar(255)
channel_mover = AxisMover(-1,0)
tensor_converter = ToTensor()
batch_size = 16
transform = Compose([resizer,image_scaler,channel_mover,tensor_converter])

directory ='subset'
df = pd.read_csv(directory+'/train.csv')
df.index = df['Image']
img_names = getListOfImageNames(directory+'/data')

transform = Compose([resizer,image_scaler,channel_mover,tensor_converter])
random_dataset = RandomDataSet(df,img_names,directory,transform=transform)
random_loader = DataLoader(random_dataset,shuffle=True,batch_size=batch_size)
random_img_iterator = iter(random_loader)

dict_of_images = createDictOfImagesForEachLabel(df,img_names)
dict_of_dataloaders = createDictOfDataLoaders(dict_of_images,batch_size,directory,transform)
dict_of_dataiterators = createDictOfDataIterators(dict_of_dataloaders)
label_names = dict_of_dataiterators.keys()
################ **Setup and Hyperparameters** ##################
# @lp
# def main():
start_time = time()
w= SummaryWriter('commai','test')
# w = SummaryWriter("debug")
# use_cuda = True
use_cuda = False
device = torch.device("cuda" if  use_cuda else "cpu")
net = Net()
net.to(device)
net.train()

LR = 5e-3
cos_period = 80
drop_period = 300
batch_size = batch_size

optimizer = optim.SGD(net.parameters(),lr = LR,momentum=0.8)
scheduler = LambdaLR(optimizer,lr_lambda=cosine_drop(cos_period,drop_period,0.4))
criterion = nn.HingeEmbeddingLoss(margin = 5,reduction='none')


# w.add_text("Hyperparameters","learning_rate: {} batch size: {} cosine period: {} drop period: {}".format(LR,batch_size,cos_period,drop_period))
w.add_experiment_parameter("Learning Rate",LR)
w.add_experiment_parameter("Batch Size",batch_size)
w.add_experiment_parameter("Cosine Period",cos_period)
w.add_experiment_parameter("Drop Period",drop_period)
################ **Misc Variables** ##################
count = 0
# flag = False
# explore_count = 0
total_load_time = 0
total_train_time = 0

################ **Training Code** ##################
epochs = 10
batch_size = 16
for epoch in range(epochs):
    for label in list_of_labels:
        target = 1
        img_batch1 = RandomImagesFromLabel(label,batch_size)
        img_batch2 = RandomImagesFromLabel(label,batch_size)
        random_img_batch = RandomImages(batch_size)

        output1 = net(img_batch1)
        output2 = net(img_batch2)
        random_output = net(random_img_batch)

        for img in img_batch1:
            loss = loss + getHingeLoss(output1,output2)
            loss = loss + getHineLoss(output1,random_output)
    try:
        random_img_batch = next(random_img_iterator)
    else StopIteration:
        random_img_iterator = iter(train_loader)
        random_img_batch = next(random_img_iterator)



for epoch in range(epochs):
    print("Current epoch: ",epoch)
    load_time1 = time()
    for img,label in train_loader:
        ## Initial Log
        count += 1
        load_time2 = time()
        total_load_time += load_time2-load_time1 
        train_time1 = time()

        img,label = img.to(device).float(), label.to(device).float()
        output = net(img)

        ## Log
        max_output = output.detach().max().item()
        # print("Max output: ", max_output)
        w.add_scalar("Max Output",max_output)
        w.add_scalar("Std of Output",output.detach().std().item())

        loss = criterion(output,label)

        ## Log
        w.add_scalar("Loss",loss.item())
        print("Loss: ",loss.item())

        optimizer.zero_grad()
        loss.backward()

        ## Log
        avg_grad_last = getAverageGradientValue(net.fc_last)
        avg_grad_fc1 = getAverageGradientValue(net.fc1)
        avg_grad_conv1 = getAverageGradientValue(net.features[0].stacked_conv[0])
        w.add_scalar("Avg Gradient of fc last",avg_grad_last)
        w.add_scalar("Avg Gradient of fc1",avg_grad_fc1)
        w.add_scalar("Avg Gradient of conv1",avg_grad_conv1)
        w.add_scalar("Percentage of Dead Neurons Final Layer",net.freq_of_dead_neurons)
        # w.add_scalar("Bias Value before Final Layer",net.avg_bias_value)
        # print("Freq: ",net.freq_of_dead_neurons)
        # print("Output: ",output)
        # print("Learning Rate: ",optimizer.state_dict()['param_groups'][0]['lr'])
        w.add_scalar("Learning Rate",optimizer.state_dict()['param_groups'][0]['lr'])
        optimizer.step()
        scheduler.step()
        # print(optimizer.state_dict()['param_groups'][0]['lr'])

        ## Ending Log
        train_time2 = time()
        total_train_time += train_time2-train_time1
        load_time1 = time()
end_time = time()
w.close()
print("Total time: ",end_time-start_time)
print("Time to load dataset: ", total_load_time)
print("Time to train: ", total_train_time)
