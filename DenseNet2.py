#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ## 1. 기본 설정
# Parameter 설정 ( batch_size, learning_rate, layers)
# Data 가져오기
# 
# train_loader = torch.utils.data.DataLoader(
#     datasets.CIFAR10('../data',train=True,download=True,transform=transform_train),
#     batch_size=batch_size,shuffle=True
# )
# 
# test_loader = torch.utils.data.DataLoader(
#     datasets.CIFAR10('../data',train=False,transform=transform_test),
#     batch_size=batch_size,shuffle=True
# )

# In[3]:


batch_size=64
learning_rate = 0.1
layers = 100


# In[4]:


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    ])

train_set = datasets.CIFAR10('./data', train=True,
                                        download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True)


test_set = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False)


# ## 2. Block 만들기
# block 이어붙여 dense block 만들기
# Dense Block + Transition layer -> Dense Net

# In[5]:


class BasicBlock(nn.Module):
    def __init__(self,in_planes,out_planes,dropRate = 0.0):
        #input dimension을 정하고, output dimension을 정하고(growth_rate임), dropRate를 정함.
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace = True) # inplace 하면 input으로 들어온 것 자체를 수정하겠다는 뜻. 메모리 usage가 좀 좋아짐. 하지만 input을 없앰.
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride = 1, padding = 1, bias = False)
        self.droprate = dropRate
        
    def forward(self,x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate>0:
            out = F.dropout (out,p=self.droprate,training = self.training)
        return torch.cat([x,out],1)
        
class BottleneckBlock(nn.Module):
    def __init__(self,in_planes,out_planes,dropRate=0.0):
        #out_planes => growth_rate를 입력으로 받게 된다.
        super(BottleneckBlock,self).__init__()
        inter_planes = out_planes * 4 
        # bottleneck layer의 conv 1x1 filter channel 수는 4*growth_rate이다.
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv1 = nn.Conv2d(in_planes,inter_planes,kernel_size=1,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes,out_planes,kernel_size=3,stride=1,padding=0,bias=False)
        self.droprate = dropRate
        
    def forward(self,x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate>0:
            out = F.dropout(out,p=self.droprate,inplace=False,training = self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate>0:
            out = F.dropout(out,p=self.droprate,inplace=False,training = self.training)
        return torch.cat([x,out],1) # 입력으로 받은 x와 새로 만든 output을 합쳐서 내보낸다
        

DenseBlock
# In[6]:


class DenseBlock(nn.Module):
    def __init__(self,nb_layers,in_planes,growth_rate,block,dropRate=0.0):
        super(DenseBlock,self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    
    def _make_layer(self,block,in_planes,growth_rate,nb_layers,dropRate):
        layers=[]
        for i in range(nb_layers):
            layers.append(block(in_planes + i*growh_rate ,growth_rate,dropRate))            
        return nn.Sequential(*layers)
    
    def forward(self,x):
        return self.layer(x)

Transition Block
# In[7]:


class TransitionBlock(nn.Module):
    def __init__(self,in_planes,out_planes,dropRate=0.0):
        super(TransitionBlock,self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=1,padding=0,bias=False) 
        self.droprate = dropRate
        
    def forward(self,x):
        out = self.conv1(self.relu(self.bn1(x))) 
        if self.droprate>0:
            out = F.dropout(out,p=self.droprate,inplace=False,training=self.training)
        return F.avg_pool2d(out,2)

합쳐서 DenseNet 구현
# In[8]:


class DenseNet(nn.Module):
    def __init__(self,depth,num_classes,growth_rate=12,reduction=0.5,bottleneck=True,dropRate=0.0):
        super(DenseNet,self).__init__()
        num_of_blocks = 3
        in_planes = 16 # 2 * growth_rate
        n = (depth - num_of_blocks - 1)/num_of_blocks 
        # 총 depth에서 첫 conv , 2개의 transit , 마지막 linear 빼고 / num_of_blocks
        if reduction != 1 :
            in_planes = 2 * growth_rate
        if bottleneck == True:
            in_planes = 2 * growth_rate #논문에서 Bottleneck + Compression 할 경우 first layer은 2*growth_rate라고 했다.
            n = n/2 # conv 1x1 레이어가 추가되니까 !
            block = BottleneckBlock 
        else :
            block = BasicBlock
        
        n = int(n) #n = DenseBlock에서 block layer 개수를 의미한다.
        self.conv1 = nn.Conv2d(3,in_planes,kernel_size=3,stride=1,padding=1,bias=False) # input:RGB -> output:growthR*2
        
        
        #1st block
        # nb_layers,in_planes,growh_rate,block,dropRate
        self.block1 = DenseBlock(n,in_planes,growth_rate,block,dropRate)
        in_planes = int(in_planes+n*growh_rate) # 입력 + 레이어 만큼의 growth_rate
        
        # in_planes,out_planes,dropRate
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)),dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        
        
        #2nd block
        # nb_layers,in_planes,growh_rate,block,dropRate
        self.block2 = DenseBlock(n,in_planes,growth_rate,block,dropRate)
        in_planes = int(in_planes+n*growh_rate) # 입력 + 레이어 만큼의 growth_rate
        
        # in_planes,out_planes,dropRate
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)),dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        
        
        #3rd block
        # nb_layers,in_planes,growh_rate,block,dropRate
        self.block3 = DenseBlock(n,in_planes,growth_rate,block,dropRate)
        in_planes = int(in_planes+n*growth_rate) # 입력 + 레이어 만큼의 growh_rate
        
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace = True)
        
        self.fc = nn.Linear(in_planes,num_classes) # 마지막에 ave_pool 후에 1x1 size의 결과만 남음.
        
        self.in_planes = in_planes
        
        # module 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Conv layer들은 필터에서 나오는 분산 root(2/n)로 normalize 함
                # mean = 0 , 분산 = sqrt(2/n) // 이게 무슨 초기화 방법이었는지 기억이 안난다.
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d): # shifting param이랑 scaling param 초기화(?)
                m.weight.data.fill_(1) # 
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):# linear layer 초기화.
                m.bias.data.zero_()
        
    def forward(self,x):
        #x : 32*32
        out = self.conv1(x) # 32*32
        out = self.block1(out) # 32*32
        out = self.trans1(out) # 16*16
        out = self.block2(out) # 16*16
        out = self.trans2(out) # 8*8
        out = self.block3(out) # 8*8
        out = self.relu(self.bn1(out)) #8*8
        out = F.avg_pool2d(out,8) #1*1
        out = out.view(-1, self.in_planes) #channel수만 남기 때문에 Linear -> in_planes
        return self.fc(out)


# ## 3. Loss function & Optimizer
# Adam optimizer도 써보고 SGD도 써봤는데 SGD가 조금 더 성능이 좋은듯
# 

# In[9]:


# depth,num_classes <- cifar '10' ,growh_rate=12,reduction=0.5,bottleneck=True,dropRate=0.0

#model = torch.load('DenseNetModelSave.pt')
model = DenseNet(layers,10,growth_rate=12,dropRate = 0.0)

# get the number of model parameters 
print('Number of model parameters: {}'.format( sum([p.data.nelement() for p in model.parameters()]))) 

model = model.to(device) 
criterion = nn.CrossEntropyLoss().to(device)#해보자 한번
#optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate, momentum=0.9,nesterov=True,weight_decay=1e-4)


# ## 4. Train Loop

# In[26]:


def train(train_loader,model,criterion,optimizer,epoch):
    model.train()
    running_loss = 0.0
    for i, (input,target) in enumerate(train_loader):
        target = target.to(device)
        input = input.to(device)
        
        output = model(input)
        loss = criterion(output,target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if(i%20 == 0):
            # print("loss in epoch %d , step %d : %f" % (epoch, i,loss.data[0]))
            print("loss in epoch %d , step %d : %f" % (epoch, i,running_loss/20))
            running_loss = 0.0
            
    print('Finished Training')
            


# ## 5. Test Loop

# In[27]:


def test(test_loader,model,criterion,epoch):
    model.eval() # used where?
    correct = 0 
    
    for i, (input,target) in enumerate(test_loader):
        target = target.to(device) 
        input = input.to(device) 
        
        output = model(input) 
        loss = criterion(output,target) 
        
        pred = output.data.max(1, keepdim=True)[1] 
        correct += pred.eq(target.data.view_as(pred)).cpu().float().sum() 
        
    print("Accuracy in epoch %d : %f" % (epoch,100.0*correct/len(test_loader.dataset)))

Learning rate를 epoch에 따라 감소시켜줄 수 있게 함수 작성.
# In[28]:


def adjust_lr(optimizer, epoch, learning_rate):
    if epoch==15 : 
        learning_rate*=0.1 
        for param_group in optimizer.param_groups: 
            param_group['lr'] = learning_rate

epoch를 돌려서 epoch 마다 train & test
(test가 목적이므로 epoch를 1/10로 줄여 300,150을 -> 30, 15로 바꿔놓은 상태
# In[29]:


for epoch in range(0,30): 
    adjust_lr(optimizer,epoch,learning_rate) 
    train(train_loader,model,criterion,optimizer,epoch) 
    test(test_loader,model,criterion,epoch)

학습이 오래 걸리므로 모델 저장, 불러오기
# In[ ]:


# ... after training, save your model 
torch.save(model, 'DenseNetModelSave.pt')
# .. to load your previously training model: 
model = torch.load('DenseNetModelSave.pt')


# In[ ]:




