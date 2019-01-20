import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from time import time
import numpy as np
import cv2
from shlex import split
import subprocess
import pandas as pd
# from line_profiler import LineProfiler
# lp = LineProfiler()
def checkTrainSetMean(dataset,center_scalar):
    mean =0
    ## final mean should be 0 since each pixel location has been normalized to 0 mean, and we are adding them all up as random variables
    # numbers are -0.003, 
    image_names = getListOfImageNames(dataset.directory)
    for name in image_names:
        img = cv2.imread(dataset.directory+'/'+name)
        a = np.mean(img-center_scalar.mean_values_array)
        mean += a 
    print("Mean pixel value-after transforms: {}".format(mean))

class CannyDetector():
    def __init__(self,lower,upper):
        self.lower = lower
        self.upper = upper
    def __call__(self,image):
        img = cv2.Canny(image,self.lower,self.upper)
        img = np.expand_dims(img,axis=2)
        return np.concatenate((img,image),axis=2)

class XCropper():
    def __init__(self,x_min,x_max):
        self.xmin = x_min
        self.xmax = x_max
    def __call__(self,image):
        img = image[self.xmin:self.xmax,...]
        return img

class MaxMagScalar():
    def __init__(self,factor):
        self.factor = factor
    def __call__(self,image):
        return image/(self.factor*self.max_mag)
    def fit(self,dataset):
        max_mag = 0
        for img,label in dataset:
            if type(img) == str:
                continue
            mag_array = np.zeros_like(img[:,:,0],dtype=np.float32) 
            for channel in range(img.shape[-1]):
                slice_array = img[:,:,channel]
                mag_array += slice_array**2
            temp_max_mag = np.sqrt(mag_array).max()
            if max_mag<temp_max_mag:
                max_mag = temp_max_mag
        self.max_mag = max_mag

class ToTensor():
    def __call__(self,image):
        return torch.tensor(image).float()

class DownSample():
    def __init__(self,factor):
        self.factor = factor
    def __call__(self,image):
        img = cv2.resize(image,(0,0),fx = 1/self.factor,fy = 1/self.factor,interpolation=cv2.INTER_AREA)
        return img
class ReSizer():
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __call__(self,image):
        return cv2.resize(image,(self.x,self.y))

class AxisMover():
    def __init__(self,src,des):
        self.src = src
        self.des = des
    def __call__(self,image):
        return np.moveaxis(image,-1,0)

class ImageScalar():
    def __init__(self,factor):
        self.factor = factor
    def __call__(self,image):
        return image/self.factor

class Standarize():
    def fit(self,dataset):
        self.mean_values_array= self.getMeanOverDataset(dataset)
    def __call__(self,image):
        return (image - self.mean_values_array)/255
    def getMeanOverDataset(self,dataset):
        '''
        Descriptions/Assumptions: assume all images are the same size 
        Arguments: 
        Returns: 
        '''
        count = 0
        start = time()
        image_names = getListOfImageNames(dataset.directory)

        img = cv2.imread(dataset.directory+'/'+image_names[0])
        mean_array = np.zeros(img.shape,dtype = np.float32)

        for name in image_names:
            img = cv2.imread(dataset.directory+'/'+name)
            ## since imread fails silently
            mean_array = mean_array + img
            count += 1
        end = time()
        print("Time elapsed: ",end-start) #about 80 seconds
        print("Number of Images: ",count)
        return mean_array/count

            
def getListOfImageNames(directory):
    '''
    Descriptions/Assumptions: directory is relative to where the script is
    Arguments: 
    Returns: 
    '''
    bashCommand1 = "ls "+directory
    p1 = subprocess.Popen(split(bashCommand1), stdout=subprocess.PIPE)
    # p2 = subprocess.Popen(split(bashCommand2),stdin=p1.stdout,stdout=subprocess.PIPE)
    output, error = p1.communicate()
    string = str(output)[2:-3]
    output = string.split('\\n')
    return output

################ **Main DataSet Class** ##################
class RandomDataSet(Dataset):
    def __init__(self,df,img_names,directory,transform=None,train=True):
        '''
        Descriptions/Assumptions: assumes labels is one level up, and that images have the following pattern "frame*.jpg"
        Arguments: 
        Returns: 
        '''
        self.directory = directory
        self.transform = transform
        self.train = train
        
        self.df = df
        self.img_names = img_names
        self.length = len(self.img_names)
    def __len__(self):
        return self.length
    # @lp
    def __getitem__(self,idx):
        ################ **Getting Pic** ##################
        img_name = self.img_names[idx]
        pic = cv2.imread(self.directory+'/data/'+img_name)
        # pic = cv2.cvtColor(pic1,cv2.COLOR_BGR2GRAY)

        if self.transform:
            pic = self.transform(pic)

        ################ **Getting Label and Returning** ##################
        if self.train:
            label = self.df['Id'][img_name]
            return pic,label
        else:
            return pic

class LabelDataSet(Dataset):
    def __init__(self,directory,label,dict_of_images,transform=None):
        self.directory = directory
        self.transform = transform

        self.img_names =  dict_of_images[label]
        self.length = len(self.img_names)

    def __len__(self):
        return self.length
    def __getitem__(self,idx):
        ################ **Getting Pic** ##################
        img_name = self.img_names[idx]
        pic = cv2.imread(self.directory+'/data/'+img_name)
        # pic = cv2.cvtColor(pic1,cv2.COLOR_BGR2GRAY)

        if self.transform:
            pic = self.transform(pic)

        return pic


if __name__ == '__main__':
    resizer = ReSizer(200,350)
    image_scaler = ImageScalar(255)
    channel_mover = AxisMover(-1,0)
    tensor_converter = ToTensor()

    directory_name ='subset'
    df = pd.read_csv(directory_name+'/train.csv')

    # preprocess_transform = Compose([resizer])
    preprocess_dataset = RandomDataSet(directory_name)

    train_transform = Compose([resizer,image_scaler,channel_mover,tensor_converter])
    train_dataset = RandomDataSet(directory_name,transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=10)
    img,label = preprocess_dataset[1]
