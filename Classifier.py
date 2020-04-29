import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F


import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from time import time
from torch.utils.data.sampler import SubsetRandomSampler

import random
from math import floor
import operator
from torch.nn import init
import copy
torch.random.manual_seed(1000)


criterion = torch.nn.CrossEntropyLoss()
criterion.cuda()


def validate(model,  criterion, val_loader, epoch):    
    model.eval()
    test_loss = 0
    correct = 0
    preds=torch.zeros([10000]) 
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            
            correct += pred.eq(target.view_as(pred)).sum().item()
        
    
    test_loss /= len(val_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, val_loader.sampler.__len__(),
        100. * correct / val_loader.sampler.__len__() ))
    
    return test_loss, 100. * correct / val_loader.sampler.__len__()


# Hyper-parameters
param = {
    'batch_size': 256,
    'test_batch_size': 256,
    'num_epochs':250,
    'delay': 251,
    'learning_rate': 0.1,
    'weight_decay': 3e-4,
}






import math
class DownsampleA(nn.Module):

  def __init__(self, nIn, nOut, stride):
    super(DownsampleA, self).__init__()
    assert stride == 2
    self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

  def forward(self, x):
    x = self.avg(x)
    return torch.cat((x, x.mul(0)), 1)

class ResNetBasicblock(nn.Module):
  expansion = 1
  """
  RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  """
  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(ResNetBasicblock, self).__init__()

    self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn_a = nn.BatchNorm2d(planes)
    
    self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn_b = nn.BatchNorm2d(planes)
    
    self.downsample = downsample

  def forward(self, x):
    residual = x

    basicblock = self.conv_a(x)
    basicblock = self.bn_a(basicblock)
    basicblock = F.relu(basicblock, inplace=True)

    basicblock = self.conv_b((basicblock))
    basicblock = self.bn_b(basicblock)

    if self.downsample is not None:
      residual = self.downsample(x)
    
    return F.relu(residual + basicblock, inplace=True)

class CifarResNet(nn.Module):
  """
  ResNet optimized for the Cifar dataset, as specified in
  https://arxiv.org/abs/1512.03385.pdf
  """
  def __init__(self, block, depth, num_classes):
    """ Constructor
    Args:
      depth: number of layers.
      num_classes: number of classes
      base_width: base width
    """
    super(CifarResNet, self).__init__()

    #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
    assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
    layer_blocks = (depth - 2) // 6
    print ('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))

    self.num_classes = num_classes
 
    self.conv_1_3x3 = nn.Conv2d(3,16, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn_1 = nn.BatchNorm2d(16)

    self.inplanes = 16
    self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
    self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
    self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
    self.avgpool = nn.AvgPool2d(8)
 
    self.classifier = nn.Linear(5184*block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        #m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        init.kaiming_normal(m.weight)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    
    x = self.conv_1_3x3(x)
    x = (F.relu(self.bn_1(x), inplace=True))
    x = self.stage_1(x)
    x = self.stage_2(x)
    x = self.stage_3(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    
    return self.classifier(x)
net = CifarResNet(ResNetBasicblock, 20, 2)



#net.load_state_dict(torch.load('Resnet20_8bit_d.pkl')) 
 


#net.load_state_dict(torch.load('net.pkl')) 
#net.eval()



from time import time


if torch.cuda.is_available():
    print('CUDA ensabled.')
    net.cuda()

net.train()

def lasso(var):
	return var.abs().sum()
# Train the model
criterion = nn.CrossEntropyLoss()
criterion=criterion.cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum =0.9,
    weight_decay=3e-5) 
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,140,200], gamma=0.1) 
net.eval()
def lasso(var):
	
	return var.abs().sum()


data=torch.load('flower.pt')
data=torch.FloatTensor(data)
data=data.reshape(33,3,300,300)
data1=torch.load('cact.pt')
data1=torch.FloatTensor(data1)
data1=data1.reshape(33,3,300,300)
#data=data.numpy()
from scipy.ndimage import gaussian_filter

data_tr=torch.zeros([44,3,300,300])
targ_tr=torch.zeros([44])
data_ts=torch.zeros([22,3,300,300])

data_tr[0:22,:,:,:]=data[0:22,:,:,:]
data_tr[22:44,:,:,:]=data1[0:22,:,:,:]
targ_tr[0:22]=0
targ_tr[22:44]=1
data_ts[0:11,:,:,:]=data[22:33,:,:,:]
data_ts[11:22,:,:,:]=data1[22:33,:,:,:]





data_tr=data_tr.cuda()
targ_tr=targ_tr.long().cuda()
data_ts=data_ts.cuda()

print(data.size())
beta=0.0005
gamma=0.0005



for epoch in range(40): 
    scheduler.step() 
    net.train() 
    print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs'])) 
    num_cor=0
    m=0
    n=22
    for i in range(2):
        loss=0
        datass=data_tr[m:n,:,:,:]
        targetss=targ_tr[m:n]
        n+=22
        m+=22
        loss = criterion(net(datass), targetss) #+loss_lasso
        print(loss.data) 
        
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()
    
        
    

       
torch.save(net.state_dict(), 'Resnet20_8_0.pkl')    
net.eval()
output = net(data_ts)
pred = output.max(1, keepdim=True)[1]
print(pred)
       
   

    
