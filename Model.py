import torch.nn as nn
import torch.nn.functional as F
from DataSet import *
# from line_profiler import LineProfiler
# lp = LineProfiler()

################ **Methods for Inspecting Model** ##################
class FunctionalModule(nn.Module):
    def __init__(self):
        super().__init__()
    def _forward(x):
        raise NotImplementedError
    def forward(self,x):
        if self.show:
            print("size before convolve: {}".format(x.shape))
        x = self._forward(x)
        if self.show:
            print("size after convolve: {}".format(x.shape))
        return x
    @staticmethod
    def getAverageGradientValue(module):
        count = 0
        total = 0
        for param in module.parameters():
            gradient_array = param.grad
            count += getProductOfTuple(gradient_array)
            total += gradient_array.abs().sum().item()
        return total/count
    @staticmethod
    def getFrequencyOfDeadNeurons(torch_tensor):
        torch_matrix = torch_tensor.detach()
        negative_values = (torch_matrix==0).sum().item()
        total_number_of_values = getProductOfTuple(torch_matrix)
        return negative_values/total_number_of_values
    @staticmethod
    def getAverageBiasValue(module):
        bias_array = module.state_dict()['bias'].detach()
        return bias_array.abs().mean().item()

def getProductOfTuple(torch_tensor):
    out = 1
    for num in torch_tensor.shape:
        out = out*num
    return out

########################


class DoubleConvBlock(FunctionalModule):
    def __init__(self,in_channels,out_channels,kernel_size,show=False,padding=0):
        '''
        Descriptions/Assumptions: Only pads the first conv block
        Arguments: 
        Returns: 
        '''
        super().__init__()
        self.show = show
        self.stacked_conv = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),

                nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size,padding=padding),
                nn.BatchNorm2d(out_channels),
                # nn.ReLU(),
                )
    def _forward(self,x):
        x = self.stacked_conv(x)
        return x
class ConvBlock(FunctionalModule):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,pool=True,show=False):
        super().__init__()
        self.show = show
        if pool:
            self.conv = nn.Sequential(
                    nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                    )
        else:
            self.conv = nn.Sequential(
                    nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    )
    def _forward(self,x):
        x = self.conv(x)
        return x

class ResNetDouble(FunctionalModule):
    def __init__(self,in_channels,out_channels,kernel_size,padding,show=False):
        '''
        Descriptions/Assumptions: 
        Arguments: `padding` needs to be chosen so that output of convolution has same spatial dimensions
        Returns: 
        '''
        super().__init__()
        self.double_conv = DoubleConvBlock(in_channels,out_channels,kernel_size,padding=padding)
        self.show = show
    def _forward(self,x):
        y = self.double_conv(x)
        return F.relu(x+y)

class Net(FunctionalModule):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(3, 24, kernel_size=8,stride=2)
        self.features = nn.Sequential(
                self.conv1,
                nn.MaxPool2d(2),

                ConvBlock(24,32,kernel_size=3),
                nn.MaxPool2d(2),

                ResNetDouble(32,32,kernel_size=3,padding=1),

                ResNetDouble(32,32,kernel_size=3,padding=1),
                nn.MaxPool2d(2),

                ConvBlock(32,64,kernel_size=1),
                ResNetDouble(64,64,kernel_size=3,padding=1),

                ResNetDouble(64,64,kernel_size=3,padding=1),
                # nn.MaxPool2d(2),

                nn.AdaptiveAvgPool2d((2,2))
                # nn.Conv2d(128,1,kernel_size=1),
                # nn.ReLU(),
                )

        self.fc1 = nn.Linear(256,30)
        # self.fc1_batch = nn.BatchNorm1d(30)
        self.fc2 = nn.Linear(30,10)
        # self.fc_last = nn.Linear(100,1)

    # @property
    # def final_dead_neurons(self):
       # return self.freq_of_dead_neurons 
    # @property
    # def avg_bias_value(self):
       # return self.avg_bias_value 

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.shape[0],-1)
        
        x = F.relu(self.fc1(x))
        # x = self.fc1_batch(x)
        x = self.fc2(x)
        return x

def Distance(output1,output2):
    return torch.sum(torch.abs(output1-output2))
if __name__ == '__main__':
    net = Net()
    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=False)
    iterator = iter(train_loader)
    img1,label1 = next(iterator)
    img2,label2 = next(iterator)
    output1,output2 = net(img1,img2)
    delta = Distance(output1,output2)
    criterion = nn.HingeEmbeddingLoss(margin=2,)
