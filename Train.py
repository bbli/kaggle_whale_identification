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
# drop_period = 300
batch_size = batch_size

optimizer = optim.SGD(net.parameters(),lr = LR,momentum=0.8)
scheduler = LambdaLR(optimizer,lr_lambda=cosine(cos_period))
criterion = nn.HingeEmbeddingLoss(margin = 5,reduction='none')


# w.add_text("Hyperparameters","learning_rate: {} batch size: {} cosine period: {} drop period: {}".format(LR,batch_size,cos_period,drop_period))
w.add_experiment_parameter("Learning Rate",LR)
w.add_experiment_parameter("Batch Size",batch_size)
w.add_experiment_parameter("Cosine Period",cos_period)
# w.add_experiment_parameter("Drop Period",drop_period)
################ **Misc Variables** ##################
count = 0
# flag = False
# explore_count = 0
total_load_time = 0
total_train_time = 0

################ **Training Code** ##################
epochs = 1
batch_size = 16
for epoch in range(epochs):
    # print(net.fc2.weight.grad)

    for label in label_names:
        ################ **Same Labels** ##################
        ## Generate next training images
        try:
            same_img_batch = next(dict_of_dataiterators[label])
        except StopIteration:
            dict_of_dataiterators[label] = iter(dict_of_dataloaders[label])
            same_img_batch = next(dict_of_dataiterators[label])

        ## No point in comparing with itself
        if len(same_img_batch) == 1:
            pass
        else:
            same_img_batch.to(device)
            outputs = net(same_img_batch)
            loss = getSameLabelLoss(outputs)
            w.add_scalar("Same Loss",loss.item())
            print("Same Loss: ",loss.item())
            BackpropAndUpdate(loss,optimizer,scheduler,w,net)
        ################ ** Mostly Different Labels** ##################
        try:
            random_img_batch, random_label_batch = next(random_img_iterator)
        except StopIteration:
            random_img_iterator = iter(random_loader)
            random_img_batch, random_label_batch = next(random_img_iterator)

        random_img_batch.to(device)
        output1 = net(same_img_batch)
        output2 = net(random_img_batch)
        targets = createTargets(label,random_label_batch)
        loss = getDifferentLabelLoss(output1,output2,targets,criterion)
        BackpropAndUpdate(loss,optimizer,scheduler,w,net)

        ##Log
        w.add_scalar("Different Loss",loss.item())
        print("Different Loss: ",loss.item())
        percentage_of_different_labels = getPercentageOfDifferentLabels(targets)
        w.add_scalar("Percentage of Different Labels",percentage_of_different_labels)

w.close()
