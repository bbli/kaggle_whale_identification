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
from line_profiler import LineProfiler
lp = LineProfiler()

def cosine_drop(cos_period,explore_period,decay):
    factor = 1
    def f(episode):
        nonlocal factor
        if episode!=0 and (episode %explore_period == 0) and factor>0.05:
            factor = factor*decay
            # print("Dropped Factor to: ",factor)
        modulus = episode % cos_period
        return factor*0.5*(1.1+cos(pi*modulus/cos_period))
    return f
# @lp
# def main():
################ **Loading DataSet** ##################
start = time()
resizer = ReSizer(200,350)
image_scaler = ImageScalar(255)
channel_mover = AxisMover(-1,0)
tensor_converter = ToTensor()
batch_size = 32
transform = Compose([resizer,image_scaler,channel_mover,tensor_converter])

directory ='subset'
df = pd.read_csv(directory+'/train.csv')
df.index = df['Image']
img_names = getListOfImageNames(directory+'/data')
train_img_names,val_img_names = getTrainValSplit(img_names)
percentage_new_whale = getPercentageOfNewWhales(val_img_names,df)
print("Percentage of new whales in val set: ",percentage_new_whale)

transform = Compose([resizer,image_scaler,channel_mover,tensor_converter])
random_dataset = RandomDataSet(df,train_img_names,directory,transform=transform)
random_loader = DataLoader(random_dataset,shuffle=True,batch_size=batch_size)
random_img_iterator = iter(random_loader)

dict_of_images = createDictOfImagesForEachLabel(df,train_img_names)
dict_of_dataloaders = createDictOfDataLoaders(dict_of_images,batch_size,directory,transform)
dict_of_dataiterators = createDictOfDataIterators(dict_of_dataloaders)
label_names = dict_of_dataiterators.keys()
################ **Setup and Hyperparameters** ##################
start_time = time()
w= SummaryWriter('whale','same_label')
# w = SummaryWriter("debug")
# use_cuda = True
use_cuda = False
device = torch.device("cuda" if  use_cuda else "cpu")
net = Net()
net.to(device)
net.train()

LR = 5e-3
cos_period = 120
drop_period = 450
batch_size = batch_size
epochs = 10

optimizer1 = optim.SGD(net.parameters(),lr = LR,momentum=0.8)
optimizer = optim.Adam(net.parameters(),lr= LR)
scheduler = LambdaLR(optimizer1,lr_lambda=cosine_drop(cos_period,drop_period,0.4))
criterion = nn.HingeEmbeddingLoss(margin = 5,reduction='none')


# w.add_text("Hyperparameters","learning_rate: {} batch size: {} cosine period: {} drop period: {}".format(LR,batch_size,cos_period,drop_period))
w.add_experiment_parameter("Learning Rate",LR)
w.add_experiment_parameter("Batch Size",batch_size)
# w.add_experiment_parameter("Cosine Period",cos_period)
# w.add_experiment_parameter("Epochs",epochs)
# w.add_experiment_parameter("Drop Period",drop_period)
################ **Misc Variables** ##################
count = 0
train_start = time()
# flag = False
# explore_count = 0
# total_load_time = 0
# total_train_time = 0

################ **Training Code** ##################
for epoch in range(epochs):
    # print(net.fc2.weight.grad)
    print("Current epoch: ",epoch)

    for label in label_names:
        count += 1
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
            w.add_scalar("Same Batch Count",same_img_batch.numpy().shape[0])
            same_img_batch = same_img_batch.to(device)
            outputs = net(same_img_batch)
            loss = getSameLabelLoss(outputs)
            BackpropAndUpdate(loss,optimizer,scheduler,w,net)

            ## Log
            w.add_scalar("Same Loss",loss.item())
            print("Same Loss: ",loss.item())
        ################ ** Mostly Different Labels** ##################
        try:
            random_img_batch, random_label_batch = next(random_img_iterator)
        except StopIteration:
            random_img_iterator = iter(random_loader)
            random_img_batch, random_label_batch = next(random_img_iterator)

        random_img_batch = random_img_batch.to(device)
        same_img_batch = same_img_batch.to(device)
        output1 = net(same_img_batch)
        output2 = net(random_img_batch)
        targets = createTargets(label,random_label_batch).to(device)
        loss = getDifferentLabelLoss(output1,output2,targets,criterion)
        BackpropAndUpdate(loss,optimizer,scheduler,w,net)

        ##Log
        w.add_scalar("Different Loss",loss.item())
        print("Different Loss: ",loss.item())
        percentage_of_different_labels = getPercentageOfDifferentLabels(targets)
        w.add_scalar("Percentage of Different Labels",percentage_of_different_labels)
    ################ **Evaluating after every epoch** ##################
    total_train_outputs,total_train_labels = getAllOutputsFromLoader(random_loader,net,device)

    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=6)
    neigh.fit(total_train_outputs)

    val_dataset = RandomDataSet(df,val_img_names,directory,transform)
    val_loader = DataLoader(val_dataset,shuffle=False,batch_size=64)
    total_val_outputs,total_val_labels = getAllOutputsFromLoader(val_loader,net,device)

    distances,indices = neigh.kneighbors(total_val_outputs)
    labels_prediction_matrix = convertIndicesToTrainLabels(indices,total_train_labels)

    score = map_per_set(total_val_labels,labels_prediction_matrix)
    w.add_scalar("Val Score",score)
train_end = time()        

################ **Evaluating** ##################
eval_start = time()
total_train_outputs,total_train_labels = getAllOutputsFromLoader(random_loader,net,device)

from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=10)
neigh.fit(total_train_outputs)

val_dataset = RandomDataSet(df,val_img_names,directory,transform)
val_loader = DataLoader(val_dataset,shuffle=False,batch_size=64)
total_val_outputs,total_val_labels = getAllOutputsFromLoader(val_loader,net,device)

distances,indices = neigh.kneighbors(total_val_outputs)
labels_prediction_matrix = convertIndicesToTrainLabels(indices,total_train_labels)

final_score = map_per_set(total_val_labels,labels_prediction_matrix)
w.add_experiment_parameter("Score",final_score)
w.add_thought("I feel like most of what I have been trying will need to be tried again later b/c they arn't addressing the problem at hand. In any case, am increasing epoch to 10 and trying Adam to try to solve this SAME LABEL Loss issue")
w.close()
end = time()
eval_end = time()
print("Time elapsed: ",end-start)
print("Train time: ",train_end -train_start)
print("Eval time: ",eval_end-eval_start)
