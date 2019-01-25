from Model import *
from DataSet import *
from utils import *
import albumentations as aug

from time import time
# from tensorboardX import SummaryWriter
from ml_board import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.neighbors import NearestNeighbors
from math import cos,pi
from collections import Counter
from line_profiler import LineProfiler
lp = LineProfiler()
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

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
def getAllPairwiseFeatureLosses(total_same_output,total_same_label_batch,random_label_batch,random_output):
    global one_img_labels
    global device
    global feature_criterion
    for idx,out in enumerate(total_same_output):
        feature_targets = createFeatureTargets(total_same_label_batch[idx],random_label_batch).to(device)
        different_feature_loss = getDifferentLabelLoss(out,random_output,feature_targets,feature_criterion)

        ## Making sure one sample labels don't get swallowed by new whale label
        if total_same_label_batch[idx] in one_img_labels:
            different_feature_loss = different_feature_loss*5

        ## Bookkeeping
        if idx == 0:
           total_different_feature_loss = different_feature_loss 
        else:
            total_different_feature_loss = torch.cat((total_different_feature_loss,different_feature_loss),0)
    return total_different_feature_loss
def getAllPairwiseClassifierLosses(total_same_output,total_same_label_batch,random_label_batch,random_output):
    global device
    global classifier_criterion
    global sim_net
    for idx,out in enumerate(total_same_output):
        classifier_targets = createClassifierTargets(total_same_label_batch[idx],random_label_batch).to(device)
        different_classifier_loss = getDifferentClassifierLoss(out,random_output,classifier_targets,classifier_criterion,sim_net)

        ## Bookkeeping
        if idx == 0:
           total_different_classifier_loss = different_classifier_loss
        else:
            total_different_classifier_loss = torch.cat((total_different_classifier_loss,different_classifier_loss),0)
    return total_different_classifier_loss
########################
def convertNearestIndicesToProbabilities(indices,total_val_outputs,total_train_outputs):
    global sim_net
    global device
    total_val_outputs = torch.Tensor(total_val_outputs).float().to(device)
    total_train_outputs = torch.Tensor(total_train_outputs).float().to(device)

    total_preds = None
    for index_predictions,out in zip(indices,total_val_outputs):
        closest_outputs = getStackOfOutputs(index_predictions,total_train_outputs) 
        out = out.repeat(len(closest_outputs),1)
        preds = sim_net(out,closest_outputs)
        preds = preds[:,1]
        total_preds = accumulateTensor(total_preds,preds.view(1,-1))
    total_preds = total_preds.cpu().detach().numpy()
    return total_preds
def convertProbabilitiesToRanking(probs_matrix,labels_matrix):
    ranked_labels_matrix = []
    for index_probs,index_labels in zip(probs_matrix,labels_matrix):
        best_labels_dict = getBestLabels(index_probs,index_labels)
        sorted_pairs = sorted(best_labels_dict.items(),key=lambda x:x[1],reverse=True)
        ranked_labels = [x[0] for x in sorted_pairs]
        ranked_labels_matrix.append(ranked_labels)
    return ranked_labels_matrix

def getBestLabels(index_probs,index_labels):
    best_values_dict = {}
    for prob,label in zip(index_probs,index_labels):
        try:
            current_prob = best_values_dict[label]
            if prob>current_prob:
                best_values_dict[label] = prob
        except KeyError:
            best_values_dict[label] = prob
    return best_values_dict

# @lp
# def main():
################ **Loading DataSet** ##################
start = time()
directory ='subset'
df = pd.read_csv(directory+'/train.csv')
df.index = df['Image']
img_names = getListOfImageNames(directory+'/data')
train_img_names,val_img_names = getTrainValSplit(img_names)
percentage_new_whale = getPercentageOfNewWhales(val_img_names,df)
print("Percentage of new whales in val set: ",percentage_new_whale)

# resizer = ReSizer(200,350)
# image_scaler = ImageScalar(127.5,1)
channel_mover = AxisMover(-1,0)
tensor_converter = ToTensor()
batch_size = 16
# transform = Compose([resizer,image_scaler,channel_mover,tensor_converter])

## All transform that subclass BasicTransform will have default 0.5 probability of activiating
eval_aug_transform = aug.Compose([
        aug.Resize(200,350,p=1.0),
        aug.Normalize(p=1.0)
        ])
aug_transform = aug.Compose([
        aug.Resize(200,350,p=1.0),
        aug.VerticalFlip(),
        aug.HorizontalFlip(),
        aug.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.2,p=0.5),
        aug.HueSaturationValue(hue_shift_limit=0,sat_shift_limit=20,val_shift_limit=0,p=0.3),
        aug.GaussNoise(p=0.3),
        aug.Normalize(p=1.0)
        ])
post_transform = Compose([channel_mover,tensor_converter])


random_dataset = RandomDataSet(df,train_img_names,directory,aug_transform=aug_transform,post_transform=post_transform)
random_loader = DataLoader(random_dataset,shuffle=True,batch_size=batch_size)
random_img_iterator = iter(random_loader)

dict_of_images = createDictOfImagesForEachLabel(df,train_img_names)
dict_of_dataloaders = createDictOfDataLoaders(dict_of_images,batch_size,directory,aug_transform=aug_transform,post_transform=post_transform)
dict_of_dataiterators = createDictOfDataIterators(dict_of_dataloaders)
label_names = dict_of_dataiterators.keys()
same_img_batch_iterator = generateSameImgBatch(dict_of_dataloaders,dict_of_dataiterators,label_names)
################ **Setup and Hyperparameters** ##################
start_time = time()
w= SummaryWriter('whale','data_augment2')
w.add_thought("lots of code. added conversion between nearest neighbors probabilities and ranked labels for scoring")
# w = SummaryWriter("debug")

# use_cuda = True
use_cuda = False
device = torch.device("cuda" if  use_cuda else "cpu")
feature_net = FeatureNet()
feature_net.to(device)
feature_net.train()

sim_net = SimiliarityNet()
sim_net.to(device)
sim_net.train()

LR = 6e-4
cos_period = 160
drop_period = 600
batch_size = batch_size
max_epochs = 20

## lower so we don't spike from new_whale and collapse everything to new whale
feature_optimizer = optim.SGD(feature_net.parameters(),lr = LR,momentum=0.8)
feature_scheduler = LambdaLR(feature_optimizer ,lr_lambda=cosine_drop(cos_period,drop_period,0.4))
feature_criterion = nn.HingeEmbeddingLoss(margin = 9,reduction='none')

classifier_optimizer = optim.SGD(sim_net.parameters(),lr = LR,momentum=0.8)
classifier_scheduler = LambdaLR(classifier_optimizer ,lr_lambda=cosine_drop(cos_period,drop_period,0.4))
classifier_criterion = nn.CrossEntropyLoss(reduction='none')


# w.add_text("Hyperparameters","learning_rate: {} batch size: {} cosine period: {} drop period: {}".format(LR,batch_size,cos_period,drop_period))
w.add_experiment_parameter("Learning Rate",LR)
w.add_experiment_parameter("Batch Size",batch_size)
# w.add_experiment_parameter("Cosine Period",cos_period)
# w.add_experiment_parameter("Epochs",max_epochs)
# w.add_experiment_parameter("Drop Period",drop_period)
################ **Misc Variables** ##################
train_start = time()
val_score_list = []
# flag = False
# explore_count = 0
# total_load_time = 0
# total_train_time = 0

################ **Training Code** ##################
epoch=0
new_epoch = False
while epoch <max_epochs:
    # print("Current epoch: ",epoch)
    # print(feature_net.fc2.weight.grad)
    feature_net.train()
    classifier_optimizer.zero_grad()
    feature_optimizer.zero_grad()

    ################ **Same Labels** ##################
    img_count = 0
    one_img_labels = []
    total_same_label_batch = None
    total_same_img_batch = None
    total_same_feature_loss = None
    total_same_classifier_loss = None
    while img_count < 2.5*batch_size:
        same_img_batch, same_label_batch = next(same_img_batch_iterator)
        same_img_batch = same_img_batch.to(device)

        ## Bookkeeping
        total_same_img_batch = accumulateTensor(total_same_img_batch,same_img_batch)
        total_same_label_batch = accumulateLabels(total_same_label_batch,same_label_batch)

        ## No point in comparing with itself
        if len(same_img_batch) == 1:
            one_img_labels.append(same_label_batch[0])
        else:
            img_count += len(same_img_batch) 
            outputs = feature_net(same_img_batch)

            if epoch> 0.3*max_epochs:
                same_classifier_loss = getSameClassiferLoss(outputs,sim_net,classifier_criterion,device)
                total_same_classifier_loss = accumulateTensor(total_same_classifier_loss,same_classifier_loss)

            same_feature_loss = getSameFeatureLoss(outputs)
            total_same_feature_loss = accumulateTensor(total_same_feature_loss,same_feature_loss)
    # print("Same Classification Loss: ",total_same_classifier_loss.mean())
    ################ ** Mostly Different Labels** ##################
    ## Restart a dataloader if exhausted
    try:
        random_img_batch, random_label_batch = next(random_img_iterator)
        random_img_batch = random_img_batch.to(device) 
    except StopIteration:
        random_img_iterator = iter(random_loader)
        random_img_batch, random_label_batch = next(random_img_iterator)
        random_img_batch = random_img_batch.to(device)
        epoch +=1
        new_epoch = True
        print("Next epoch: ",epoch)

    total_same_output = feature_net(total_same_img_batch)
    random_output = feature_net(random_img_batch)
    ########################
    if epoch>0.3*max_epochs:
        total_different_classifier_loss = getAllPairwiseClassifierLosses(total_same_output,total_same_label_batch,random_label_batch,random_output)
    total_different_feature_loss = getAllPairwiseFeatureLosses(total_same_output,total_same_label_batch,random_label_batch,random_output)
    # print("Different Classification Loss: ",total_different_classifier_loss.mean())
    ################ **Backprop Time** ##################
    ## Equal weighting of same and different labels to encourage clustering
    total_different_feature_loss = total_different_feature_loss.mean()
    total_same_feature_loss = total_same_feature_loss.mean()
    total_feature_loss = total_different_feature_loss + total_same_feature_loss
    total_feature_loss.backward(retain_graph=True)

    if epoch>0.3*max_epochs:
        ## Unequal weighting of same and different labels
        total_classifier_loss = torch.cat((total_same_classifier_loss,total_different_classifier_loss))
        total_classifier_loss = total_classifier_loss.mean()
        total_classifier_loss.backward()

    ## Log
    w.add_scalar("Different Loss",total_different_feature_loss.item())
    print("Different Loss: ",total_different_feature_loss.item())
    w.add_scalar("Same Loss",total_same_feature_loss.item())
    print("Same Loss: ",total_same_feature_loss.item())
    if epoch>0.3*max_epochs:
        w.add_scalar("Classification Loss",total_classifier_loss.item())
        print("Classification Loss: ",total_classifier_loss.item())

    ################ **Updating** ##################
    if epoch>0.3*max_epochs:
        classifier_optimizer.step()
        classifier_scheduler.step()

    feature_optimizer.step()
    feature_scheduler.step()

    ##Log
    # percentage_of_different_labels = getPercentageOfDifferentLabels(targets)
    # w.add_scalar("Percentage of Different Labels",percentage_of_different_labels)
    ## Log
    # avg_grad_last = getAverageGradientValue(net.fc_last)
    # w.add_scalar("Avg Gradient of fc last",avg_grad_last)
    # avg_grad_fc1 = getAverageGradientValue(net.fc1)
    # w.add_scalar("Avg Gradient of fc1",avg_grad_fc1)
    avg_grad_conv1 = feature_net.getAverageGradientValue(feature_net.conv1)
    w.add_scalar("Avg Gradient of conv1",avg_grad_conv1)
    # w.add_scalar("Percentage of Dead Neurons Final Layer",net.freq_of_dead_neurons)
    # w.add_scalar("Bias Value before Final Layer",net.avg_bias_value)
    w.add_scalar("Feature Net LR",feature_optimizer.state_dict()['param_groups'][0]['lr'])
    if epoch>0.3*max_epochs:
        w.add_scalar("Classifier Net LR",classifier_optimizer.state_dict()['param_groups'][0]['lr'])

        del same_classifier_loss

        del total_same_classifier_loss
        del total_different_classifier_loss

        del total_classifier_loss

    del same_feature_loss
    del total_same_feature_loss
    del total_different_feature_loss

    del random_img_batch
    del same_img_batch
    del total_same_img_batch
    ################ **Evaluating after a certain period** ##################
    if new_epoch == True:
    # if True:
        feature_net.eval()
        sim_net.eval()
        
        total_train_outputs,total_train_labels = getAllOutputsFromLoader(random_loader,feature_net,device)

        neigh = NearestNeighbors(n_neighbors=20)
        neigh.fit(total_train_outputs)

        val_dataset = RandomDataSet(df,val_img_names,directory,aug_transform=eval_aug_transform,post_transform=post_transform)
        val_loader = DataLoader(val_dataset,shuffle=False,batch_size=64)
        total_val_outputs,total_val_labels = getAllOutputsFromLoader(val_loader,feature_net,device)

        distances,indices = neigh.kneighbors(total_val_outputs)
        labels_matrix = convertIndicesToTrainLabels(indices,total_train_labels)
        probs_matrix = convertNearestIndicesToProbabilities(indices,total_val_outputs,total_train_outputs)
        ranked_labels_matrix = convertProbabilitiesToRanking(probs_matrix,labels_matrix)

        score,_ = map_per_set(total_val_labels,ranked_labels_matrix)
        val_score_list.append(score)
        w.add_scalar("Val Score",float(score))
        new_epoch = False

        del total_train_outputs
        del total_train_labels
        del total_val_labels
        del total_val_outputs
train_end = time()        

################ **Evaluating** ##################
eval_start = time()
feature_net.eval()
total_train_outputs,total_train_labels = getAllOutputsFromLoader(random_loader,feature_net,device)

neigh = NearestNeighbors(n_neighbors=20)
neigh.fit(total_train_outputs)

val_dataset = RandomDataSet(df,val_img_names,directory,aug_transform=eval_aug_transform,post_transform=post_transform)
val_loader = DataLoader(val_dataset,shuffle=False,batch_size=64)
total_val_outputs,total_val_labels = getAllOutputsFromLoader(val_loader,feature_net,device)

distances,indices = neigh.kneighbors(total_val_outputs)
labels_matrix = convertIndicesToTrainLabels(indices,total_train_labels,total_train_outputs)

final_score,list_of_scores = map_per_set(total_val_labels,labels_matrix)
for score in list_of_scores:
    w.add_histogram("Score Frequency",score)
w.add_experiment_parameter("Score",final_score)
counter = Counter(list_of_scores)
w.close()
end = time()
eval_end = time()
print("Time elapsed: ",end-start)
print("Train time: ",train_end -train_start)
print("Eval time: ",eval_end-eval_start)
