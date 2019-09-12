

import torch.nn.functional as F
from torch import nn
import torch

class DCT_nonlinear(nn.Module):    
    def __init__(self):
        super(DCT_nonlinear, self).__init__()
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=1, out_channels=16, kernel_size=(2, 2), stride=(2, 2),bias=False)
        self.deconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=(2, 2), stride=(2, 2),bias=False)
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2),bias=False)
        #self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2),bias=False)
        
        self.pad2d = nn.ZeroPad2d((1, 1, 1, 1))
        self.same = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1),bias=False)
    
        #self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=(2, 2),bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(2, 2), stride=(2, 2),bias=False)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(2, 2), stride=(2, 2),bias=False)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(2, 2), stride=(2, 2),bias=False)
    
        self.lr1 = nn.LeakyReLU(0.1)
        self.lr2 = nn.LeakyReLU(0.1)
        self.lr3 = nn.LeakyReLU(0.1)
        
        self.tanh = nn.Tanh()

        
        
        self.__init_parameters__()

    def __init_parameters__(self):
        # Initialize Parameters
        for m in self.modules():
            #print('m','m')
            if isinstance(m, nn.Conv2d):
                #torch.nn.init.xavier_uniform(m.weight)
                m.weight.data.fill_(0.01)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.fill_(0.01)
                #torch.nn.init.xavier_uniform(m.weight)
            
            
    def forward(self, x): 
        
        y = self.deconv1(x)
        #y = self.tanh(y)
        y = F.relu(y)
        
        y = self.deconv2(y)
        #y = self.tanh(y)
        y = F.relu(y)
        
        y = self.deconv3(y)
        #y = self.tanh(y)
        y = F.relu(y)
        
        #y = self.deconv4(y)
        #y = self.tanh(y)
        #y = F.relu(y)
        
        y = self.pad2d(y)
        y = self.same(y)
        #y = self.tanh(y)
        y = F.relu(y)
        

        #y = self.conv1(y)
        #y = self.tanh(y)
        #y = F.relu(y)
        
        y = self.conv2(y)
        #y = self.tanh(y)
        y = F.relu(y)
        #y = self.lr1(y)
        
        y = self.conv3(y)
        #y = self.tanh(y)
        y = F.relu(y)
        #y = self.lr1(y)
                
        y = self.conv4(y)
        y = self.tanh(y)
        #y = F.relu(y)
        #y = self.lr1(y)

        return y #*tensor_weights
        

        
class IDCT_nonlinear(nn.Module):
    def __init__(self):
        super(IDCT_nonlinear, self).__init__()
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=1, out_channels=16, kernel_size=(2, 2), stride=(2, 2),bias=False)
        self.deconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=(2, 2), stride=(2, 2),bias=False)
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2),bias=False)
        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2),bias=False)
        
        self.pad2d = nn.ZeroPad2d((1, 1, 1, 1))
        self.same = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1),bias=False)
    
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=(2, 2),bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(2, 2), stride=(2, 2),bias=False)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(2, 2), stride=(2, 2),bias=False)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(2, 2), stride=(2, 2),bias=False)
        
        self.tanh = nn.Tanh()
        self.__init_parameters__()

        
    def __init_parameters__(self):
        # Initialize Parameters
        for m in self.modules():
            #print('m','m')
            if isinstance(m, nn.Conv2d):
                #torch.nn.init.xavier_uniform(m.weight)
                m.weight.data.fill_(-0.01)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.fill_(0.01)
                #torch.nn.init.xavier_uniform(m.weight)
                       
                
    def forward(self, x): 
        #nn.Tanh()
        
        y = self.deconv1(x)
        #y = self.tanh(y)
        y = F.relu(y)
        
        y = self.deconv2(y)
        #y = self.tanh(y)
        y = F.relu(y)
        
        y = self.deconv3(y)
        #y = self.tanh(y)
        y = F.relu(y)
        
        y = self.deconv4(y)
        #y = self.tanh(y)
        y = F.relu(y)
        
        y = self.pad2d(y)
        y = self.same(y)
        #y = self.tanh(y)
        y = F.ReLu(y)

        y = self.conv1(y)
        #y = self.tanh(y)
        y = F.ReLu(y)
        #y = self.lr1(y)
        
        y = self.conv2(y)
        #y = self.tanh(y)
        y = F.ReLu(y)
        #y = self.lr2(y)
        
        y = self.conv3(y)
        #y = self.tanh(y)
        y = F.ReLu(y)
        #y = self.lr3(y)
        
        y = self.conv4(y)
        #y = self.tanh(y)
        y = F.ReLu(y)
        #y = self.lr3(y)
        
        return y
def kl_divergence(p, q):
    '''
    args:
        2 tensors `p` and `q`
    returns:
        kl divergence between the softmax of `p` and `q`
    '''

    p = F.softmax(p)
    q = F.softmax(q)

    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2
