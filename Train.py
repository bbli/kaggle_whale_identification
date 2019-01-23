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
        if episode!=0 and (episode %explore_period == 0) and factor>0.2:
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
image_scaler = ImageScalar(127.5,1)
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
same_img_batch_iterator = generateSameImgBatch(dict_of_dataloaders,dict_of_dataiterators,label_names)
################ **Setup and Hyperparameters** ##################
start_time = time()
w= SummaryWriter('whale','same_label2')
w.add_thought("same label gradient updates are now with respect to multiple labels")
# w = SummaryWriter("debug")

# use_cuda = True
use_cuda = False
device = torch.device("cuda" if  use_cuda else "cpu")
net = Net()
net.to(device)
net.train()

LR = 6e-4
cos_period = 160
drop_period = 900
batch_size = batch_size
epochs = 5

## lower so we don't spike from new_whale and collapse everything to new whale
optimizer= optim.SGD(net.parameters(),lr = LR,momentum=0.8)
scheduler = LambdaLR(optimizer,lr_lambda=cosine_drop(cos_period,drop_period,0.4))

criterion = nn.HingeEmbeddingLoss(margin = 9,reduction='none')


# w.add_text("Hyperparameters","learning_rate: {} batch size: {} cosine period: {} drop period: {}".format(LR,batch_size,cos_period,drop_period))
w.add_experiment_parameter("Learning Rate",LR)
w.add_experiment_parameter("Batch Size",batch_size)
# w.add_experiment_parameter("Cosine Period",cos_period)
# w.add_experiment_parameter("Epochs",epochs)
# w.add_experiment_parameter("Drop Period",drop_period)
################ **Misc Variables** ##################
train_start = time()
# flag = False
# explore_count = 0
# total_load_time = 0
# total_train_time = 0

################ **Training Code** ##################
epoch=0
count = 0
while epoch <epochs:
    print("Current epoch: ",epoch)
    # print(net.fc2.weight.grad)
    net.train()

    img_count = 0
    label_count = 0
    one_img_labels = []
    while img_count < 1.5*batch_size:
        same_img_batch, same_label_batch = next(same_img_batch_iterator)

        ## Bookkeeping
        try:
            total_same_img_batch = torch.cat((total_same_img_batch,same_img_batch),0)
            total_same_label_batch = total_same_label_batch + same_label_batch
        except NameError:
            total_same_label_batch = same_label_batch
            total_same_img_batch = same_img_batch

        ## No point in comparing with itself
        if len(same_img_batch) == 1:
            one_img_labels.append(same_label_batch[0])
        else:
            img_count += len(same_img_batch) 
            label_count +=1

            outputs = same_img_batch.to(device)
            outputs = net(outputs)
            label_loss = getSameLabelLoss(outputs)

            ## Bookkeeping
            if label_count == 1:
                total_loss = label_loss
            else:
                total_loss = total_loss+label_loss

    total_loss = total_loss/label_count
    ## Log
    w.add_scalar("Same Loss",total_loss.item())
    print("Same Loss: ",total_loss.item())

    BackpropAndUpdate(w,net,total_loss,optimizer,scheduler)
    count += 1

    ################ ** Mostly Different Labels** ##################
    ## Restart a dataloader if exhausted
    try:
        random_img_batch, random_label_batch = next(random_img_iterator)
    except StopIteration:
        random_img_iterator = iter(random_loader)
        random_img_batch, random_label_batch = next(random_img_iterator)
        epoch +=1

    random_img_batch = random_img_batch.to(device)
    total_same_output = total_same_img_batch.to(device)
    total_same_output = net(total_same_output)
    random_output = net(random_img_batch)

    for idx,out in enumerate(total_same_output):
        targets = createTargets(total_same_label_batch[idx],random_label_batch).to(device)
        loss_per_output = getDifferentLabelLoss(out,random_output,targets,criterion)

        ## Making sure one sample labels don't get swallowed by new whale label
        if total_same_label_batch[idx] in one_img_labels:
            loss_per_output = loss_per_output*5

        ## Bookkeeping
        if idx == 0:
           total_loss2 = loss_per_output 
        else:
            total_loss2 = total_loss2 + loss_per_output
    total_loss2 = total_loss2/len(total_same_output)

    #Log
    w.add_scalar("Different Loss",total_loss2.item())
    print("Different Loss: ",total_loss2.item())

    if total_loss2 == 0:
        pass
    else:
        BackpropAndUpdate(w,net,total_loss2,optimizer,scheduler)
        count += 1

    ##Log
    # percentage_of_different_labels = getPercentageOfDifferentLabels(targets)
    # w.add_scalar("Percentage of Different Labels",percentage_of_different_labels)
    ################ **Evaluating after a certain period** ##################
    if count%250 == 0:
        net.eval()
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
net.eval()
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
w.close()
end = time()
eval_end = time()
print("Time elapsed: ",end-start)
print("Train time: ",train_end -train_start)
print("Eval time: ",eval_end-eval_start)
