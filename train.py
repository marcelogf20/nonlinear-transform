from torch import nn
import torch
from utils import *
from network import *


path_model = 'checkpoint/exp1/'
dataset_path = '/media/data/Datasets/samsung/database4/dir0'
path_save  = '/media/data/Datasets/samsung/modelos/nonlinear-transform/exp1'
beta = 5e-11
lr = 1e-4
batch_size = 4
block_size = 8

op='epoch'
load_op      = False
last_epoch   = 0
num_epochs   = 1
shuffle      = True
num_workers  = 6
patience     = 1
save_every   = 2.756e7
factor_decay = 0.5 


def main(dataset_path,lr,batch_size,path_save,path_model,op,load_op,last_epoch,num_epochs,shuffle,
    num_workers,patience,factor_decay,save_every,beta,block_size):

    mse_loss = nn.MSELoss().cuda()
    mydct = DCT_nonlinear().cuda()
    th  = Threshold().cuda()

    myidct = DCT_nonlinear().cuda()
    
    #mydct.to(torch.float)
    #myidct.to(torch.float)
        
    adam = torch.optim.Adam([{'params': mydct.parameters()},
                         {'params': myidct.parameters()},],lr=lr)
    
    optimizer = adam
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=factor_decay,
        patience=patience,
        verbose=True,
        )    
    
    if load_op:
        mydct,myidct,optimizer,scheduler2 = load(op,last_epoch, path_model, mydct, myidct,optimizer,scheduler)
        print('Model loaded')
    losses_distortion = []
    losses_sparse = []
    losses =[]

    train_transform = transform_data(block_size)
    dataset = BSDS500Crop128(dataset_path,train_transform)
   
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
    print(len(dataset),' ', len(dataloader))
    
    for ei in range(last_epoch+1, num_epochs+last_epoch+1):
        for bi, input in enumerate(dataloader):
            input = input.cuda()
            batch_size, channels, rows, cols = input.size()
            number_batch = 0
            block_index = 0

            latent = mydct(input)
            #latent = th(latent)
            output = myidct(latent)  
            
            distortion_loss  = mse_loss(input, output)
  
            sparsity_loss, prod = weights_norm(batch_size,block_size,latent,1)
            sparsity_loss  = sparsity_loss*beta

            loss = distortion_loss + sparsity_loss
            losses.append(loss.data.item())                 
            
            a = list(mydct.parameters())[0].clone()
            a2 = list(myidct.parameters())[0].clone()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            b = list(mydct.parameters())[0].clone()
            b2 = list(myidct.parameters())[0].clone()
            
            if torch.equal(a.data, b.data):
                print('equal mydct ')
            if torch.equal(a2.data, b2.data):
                print('equal myidct')
            if bi%save_every == 0:
                save(bi,'iter', path_save,mydct,myidct, optimizer,scheduler)
            if bi%400 == 0:
               print('prod',prod[0])
               print('latente[0]',latent[0])
               zeros = torch.sum(latent == 0).data.item()
               total = latent.numel()
               psrn = compute_psnr(input,output)
               print(distortion_loss.data.item(),sparsity_loss.data.item() )
               print('\n Época/Batch [{}]/[{}] loss distortion {:.4f}, loss sparsity {:.6f}, loss média {:.4f}'.format(ei,bi,distortion_loss.data.item(),
                                                                                                                  sparsity_loss.data.item(),loss.data.item()))
               print('Last batch: PSNR {:.3f}, taxa de zeros {:.3f}'.format(psrn.data.item(), 1e2*zeros/total))

        save(ei,'epoch', path_save,mydct,myidct, optimizer,scheduler)	

        scheduler.step(np.mean(losses))


main(dataset_path,lr,batch_size,path_save,path_model, op, load_op, last_epoch,num_epochs,
    shuffle,num_workers,patience,factor_decay,save_every, beta, block_size)
