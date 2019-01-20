import ipdb 
import cv2
# from itertools import permutation
# from numpy.random import permutation
from DataSet import *
from math import cos,pi
import numpy as np
import torch

def cyclic(period):
    def f(episode):
        modulus = episode % period
        return 1/(1+0.05*modulus)
    return f
def cosine(period):
    def f(episode):
        modulus = episode % period
        return 0.5*(1.1+cos(pi*modulus/period))
    return f

def getTrainValSplit(img_names):
    np.random.seed(10)
    local = np.array(img_names)
    np.random.shuffle(local)

    length = len(img_names)
    break_point = int(0.75*length)
    train_image_names,test_img_names = local[0:break_point], local[break_point:]
    return train_image_names, test_img_names

def getPercentageOfNewWhales(img_names,df):
    count = 0
    for img_name in img_names:
        Id = df.loc[img_name]["Id"]
        if Id == 'new_whale':
            count +=1
    return count/len(img_names)
########################
def createDictOfImagesForEachLabel(df,img_names):
    dict_of_images = {}
    for img_name in img_names:
        label = df.loc[img_name]['Id']
        try:
            dict_of_images[label].append(img_name)
        except KeyError:
            dict_of_images[label]  = []
            dict_of_images[label].append(img_name)

    return dict_of_images

def createDictOfDataLoaders(dict_of_images,batch_size,directory,transform):
    dict_of_dataloaders = {}
    for label in dict_of_images.keys():
        dataset = LabelDataSet(directory,label,dict_of_images,transform)
        dict_of_dataloaders[label] = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    return dict_of_dataloaders

def createDictOfDataIterators(dict_of_dataloaders):
    dict_of_dataiterators = {}
    for label in dict_of_dataloaders.keys():
        dict_of_dataiterators[label] = iter(dict_of_dataloaders[label])
    return dict_of_dataiterators
########################
def getSameLabelLoss(torch_tensor):
    for i,output in enumerate(torch_tensor):
        delta_vec = getDeltaVec(output,torch_tensor)
        delta_mag = getMagnitude(delta_vec)
        loss = torch.pow(delta_mag,2)
        if i == 0:
            total_loss = loss
        else:
            total_loss = torch.cat((total_loss,loss),0)
    return total_loss.mean()

    # delta_mag = torch.
def getDeltaVec(output,torch_tensor):
    '''
    Descriptions/Assumptions: Assumes torch tensor is a 2D tensor and output is a 1D. Does array broadcasting
    Arguments: 
    Returns: 
    '''
    return torch.abs(torch_tensor-output[np.newaxis,:])
def getMagnitude(tensor):
    tensor = torch.sqrt(torch.sum(torch.pow(tensor,2)+1e-7,dim=1))
    return tensor
########################
def createTargets(label,label_batch):
    targets_list = []
    for new_label in label_batch:
        if new_label == label:
            targets_list.append(1)
        else:
            targets_list.append(-1)
    targets_list = np.array(targets_list)
    return torch.from_numpy(targets_list).double()

def getDifferentLabelLoss(output1,output2,targets,criterion):
    for i,output in enumerate(output1):
        delta_vec = getDeltaVec(output,output2)
        delta_mag = getMagnitude(delta_vec)
        loss = criterion(delta_mag,targets)
        loss = torch.pow(loss,2)
        if i == 0:
            total_loss = loss
        else:
            total_loss = torch.cat((total_loss,loss),0)
        return total_loss.mean()

def BackpropAndUpdate(loss,optimizer,scheduler,w,net):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    ## Log
    # avg_grad_last = getAverageGradientValue(net.fc_last)
    # w.add_scalar("Avg Gradient of fc last",avg_grad_last)
    # avg_grad_fc1 = getAverageGradientValue(net.fc1)
    # w.add_scalar("Avg Gradient of fc1",avg_grad_fc1)
    avg_grad_conv1 = net.getAverageGradientValue(net.features[0].stacked_conv[0])
    w.add_scalar("Avg Gradient of conv1",avg_grad_conv1)
    # w.add_scalar("Percentage of Dead Neurons Final Layer",net.freq_of_dead_neurons)
    # w.add_scalar("Bias Value before Final Layer",net.avg_bias_value)
    w.add_scalar("Learning Rate",optimizer.state_dict()['param_groups'][0]['lr'])

def getPercentageOfDifferentLabels(targets):
    length = len(targets)
    count = 0
    for i in targets:
        if i == -1:
            count += 1
    return count/length
################ **Eval** ##################
def getAllOutputsFromLoader(dataloader,net,device):
    total_labels = []
    with torch.no_grad():
        for i,(img_batch,label_batch) in enumerate(dataloader):
            img_batch.to(device)
            preds = net(img_batch)
            total_labels = total_labels + list(label_batch) 
            if i == 0:
                total_preds = preds
            else:
                total_preds = torch.cat((total_preds,preds),0)
        total_preds = total_preds.numpy()
    return total_preds, total_labels

def convertIndicesToTrainLabels(indices,total_train_labels):
    labels_matrix = []
    for index_predictions in indices:
        labels_predictions = [total_train_labels[idx] for idx in index_predictions]
        labels_matrix.append(labels_predictions)
    return labels_matrix


################ **Metric Implementation from a Kaggle Kernel** ##################
## Note it just takes the top 5, and doesn't account for repeats/movement in rankings due to majority, etc...

def map_per_image(label, predictions):
    """Computes the precision score of one image.

    Parameters
    ----------
    label : string
            The true label of the image
    predictions : list
            A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """    
    try:
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0
def map_per_set(labels, predictions):
    """Computes the average over multiple images.

    Parameters
    ----------
    labels : list
             A list of the true labels. (Only one true label per images allowed!)
    predictions : list of list
             A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    return np.mean([map_per_image(l, p) for l,p in zip(labels, predictions)])

