import numpy as np
from scipy import fftpack
from torch.utils.data import Dataset
import os, glob
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch

def l1_norm (latente):
    l1_loss = torch.norm(latente, p=1)
    return l1_loss

def l1_norm_weights2(model):
    l1_regularization=0
    for param in model.parameters():
        l1_regularization = l1_regularization + torch.norm(param, p=1)
    return l1_regularization

def l2_norm (latente):
    euclidead_norm = torch.norm(latente, p=2)
    return euclidead_norm

def weights_norm(batch_size,block_size,latent, norm):
    tensor_weights = torch.zeros([batch_size,3,8,8], dtype=torch.float32)
    for bs in range (batch_size):
        for i in range(block_size):
            for j in range(block_size):        
                if i == 0 and j == 0:
                    tensor_weights[bs,:,i,j] = 1e-2
                elif j==0:
                    tensor_weights[bs,:,i,j] = tensor_weights[bs,0,i-1,7]*12/10
                else:   
                    tensor_weights[bs,:,i,j] = tensor_weights[bs,0,i,j-1]*12/10
    prod = latent * tensor_weights.cuda()
    weights_loss = torch.norm(prod, p=norm)

    return weights_loss,prod

def kl_divergence(p, q):
    
    p = torch.FloatTensor([p for _ in range(q.shape[0])]).unsqueeze(0)   
    funcs  = nn.Sigmoid()
    p  = funcs(p)
    q  = funcs(q)
    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    kl_div = s1 + s2 
    return kl_div

def loss_kl(dist_val,latent):
    latente = latente.view(-1)
    loss  = kl_divergence(dist_val, latente).clone()   
    return loss

def save(index, op, path_save,mydct, myidct,optimizer,scheduler):
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    torch.save(mydct.state_dict(),path_save+'/dctNN_{}_{}.pth'.format(op, index))
    torch.save(myidct.state_dict(),path_save+'/idctNN_{}_{}.pth'.format(op, index))
    torch.save(optimizer.state_dict(), path_save+'/optimizer_{}_{}.pth'.format(op, index))
    torch.save(scheduler.state_dict(), path_save+'/scheduler_{}_{}.pth'.format(op, index))
    
    
def load(op, num, path_load, mydct, myidct, optimizer, scheduler):

    mydct.load_state_dict(torch.load(path_load+'/dctNN_{}_{}.pth'.format(op, num)))
    myidct.load_state_dict(torch.load(path_load+'/idctNN_{}_{}.pth'.format(op, num)))
    optimizer.load_state_dict(torch.load(path_load+'/optimizer_{}_{}.pth'.format(op, num)))
    scheduler.load_state_dict(torch.load(path_load+'/scheduler_{}_{}.pth'.format(op, num)))
    
    return mydct,myidct, optimizer,scheduler

def compute_psnr(x, y):
    y = y.view(y.shape[0], -1)
    x = x.view(x.shape[0], -1)
    rmse = torch.sqrt(torch.mean((y - x) ** 2, dim=1))
    psnr = torch.mean(20. * torch.log10(1. / rmse))
    return psnr



def load_quantization_table(component):
    # Quantization Table for: Photoshop - (Save For Web 080)
    # (http://www.impulseadventure.com/photo/jpeg-quantization.html)
    if component == 'lum':
        q = np.array([[2, 2, 2, 2, 3, 4, 5, 6],
                      [2, 2, 2, 2, 3, 4, 5, 6],
                      [2, 2, 2, 2, 4, 5, 7, 9],
                      [2, 2, 2, 4, 5, 7, 9, 12],
                      [3, 3, 4, 5, 8, 10, 12, 12],
                      [4, 4, 5, 7, 10, 12, 12, 12],
                      [5, 5, 7, 9, 12, 12, 12, 12],
                      [6, 6, 9, 12, 12, 12, 12, 12]])
    elif component == 'chrom':
        q = np.array([[3, 3, 5, 9, 13, 15, 15, 15],
                      [3, 4, 6, 11, 14, 12, 12, 12],
                      [5, 6, 9, 14, 12, 12, 12, 12],
                      [9, 11, 14, 12, 12, 12, 12, 12],
                      [13, 14, 12, 12, 12, 12, 12, 12],
                      [15, 12, 12, 12, 12, 12, 12, 12],
                      [15, 12, 12, 12, 12, 12, 12, 12],
                      [15, 12, 12, 12, 12, 12, 12, 12]])
    else:
        raise ValueError((
            "component should be either 'lum' or 'chrom', "
            "but '{comp}' was found").format(comp=component))

    return q


def zigzag_points(rows, cols):
    # constants for directions
    UP, DOWN, RIGHT, LEFT, UP_RIGHT, DOWN_LEFT = range(6)

    # move the point in different directions
    def move(direction, point):
        return {
            UP: lambda point: (point[0] - 1, point[1]),
            DOWN: lambda point: (point[0] + 1, point[1]),
            LEFT: lambda point: (point[0], point[1] - 1),
            RIGHT: lambda point: (point[0], point[1] + 1),
            UP_RIGHT: lambda point: move(UP, move(RIGHT, point)),
            DOWN_LEFT: lambda point: move(DOWN, move(LEFT, point))
        }[direction](point)

    # return true if point is inside the block bounds
    def inbounds(point):
        return 0 <= point[0] < rows and 0 <= point[1] < cols

    # start in the top-left cell
    point = (0, 0)

    # True when moving up-right, False when moving down-left
    move_up = True

    for i in range(rows * cols):
        yield point
        if move_up:
            if inbounds(move(UP_RIGHT, point)):
                point = move(UP_RIGHT, point)
            else:
                move_up = False
                if inbounds(move(RIGHT, point)):
                    point = move(RIGHT, point)
                else:
                    point = move(DOWN, point)
        else:
            if inbounds(move(DOWN_LEFT, point)):
                point = move(DOWN_LEFT, point)
            else:
                move_up = True
                if inbounds(move(DOWN, point)):
                    point = move(DOWN, point)
                else:
                    point = move(RIGHT, point)


def bits_required(n):
    n = abs(n)
    result = 0
    while n > 0:
        n >>= 1
        result += 1
    return result


def binstr_flip(binstr):
    # check if binstr is a binary string
    if not set(binstr).issubset('01'):
        raise ValueError("binstr should have only '0's and '1's")
    return ''.join(map(lambda c: '0' if c == '1' else '1', binstr))


def uint_to_binstr(number, size):
    return bin(number)[2:][-size:].zfill(size)


def int_to_binstr(n):
    if n == 0:
        return ''

    binstr = bin(abs(n))[2:]

    # change every 0 to 1 and vice verse when n is negative
    return binstr if n > 0 else binstr_flip(binstr)


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def block_to_zigzag(block):
    return torch.FloatTensor([block[point] for point in zigzag_points(*block.shape)])


def dct_2d(image):
    return fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho')

def zigzag_to_block(zigzag):
    # assuming that the width and the height of the block are equal
    rows = cols = int(math.sqrt(len(zigzag)))

    if rows * cols != len(zigzag):
        raise ValueError("length of zigzag should be a perfect square")

    #block = np.empty((rows, cols), np.int32)
    block = torch.zeros([rows, cols], dtype=torch.float32).cuda()
    
    for i, point in enumerate(zigzag_points(rows, cols)):
        block[point] = zigzag[i]

    return block

def quantize(block, component):
    q = torch.from_numpy(load_quantization_table(component)).cuda().float()   
    #print(type(block),'Shape block', block.shape)
    r = torch.round(block / q)
    #r = (block / q).round()
    
    return r  #.astype(np.int32)


def dequantize(block, component):
    q = torch.from_numpy(load_quantization_table(component)).cuda().float()
    return block * q


def idct_2d(image):
    return fftpack.idct(fftpack.idct(image.T, norm='ortho').T, norm='ortho')


class BSDS500Crop128(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob('%s/*.*' % folder_path)) 
        self.transform = transform

    def __getitem__(self, index):
        path  = self.files[index % len(self.files)]
        img   = Image.open(path)
        ycbcr = img.convert('YCbCr')
        if self.transform is not None:
            ycbcr = self.transform(ycbcr)
        return ycbcr

    def __len__(self):
        return len(self.files)
    
def transform_data(size_p):       
    train_transform = transforms.Compose([
    transforms.RandomCrop((size_p, size_p)),
    transforms.ToTensor(),])    
    return train_transform


class ImageFolderYCbCr(Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self, root, transform=None):
        images = []

        for filename in os.listdir(root):
            for files in glob.glob('%s/*.*' % (root+'/'+filename)): 
                images.append(files)

        self.root = root
        self.imgs = images
        self.transform = transform

    def __getitem__(self, index):
        path  = self.imgs[index]
        img   = Image.open(path)
        ycbcr = img.convert('YCbCr')
        #ycbcr = np.array(ycbcr)
        if self.transform is not None:
            ycbcr = self.transform(ycbcr)
    
        return ycbcr

    def __len__(self):
        return len(self.imgs)
