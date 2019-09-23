from torch import nn
import torch
from utils import *
from network import *

dataset_path = 'imagens_teste2/'
output_file = 'mytest2.png'
lr = 1e-4
path_save = 'checkpoint/exp4/'
batch_size = 4
path_load = 'checkpoint/exp4/'
op='epoch'
num = 0
last_epoch = num
num_epochs = 300
shuffle = True
num_workers = 6
patience   = 200
save_every = 200
factor_decay = 0.5 
beta = 1e-3

def main(dataset_path,output_file,lr,batch_size,path_save,path_load,op,num,last_epoch,num_epochs,shuffle,
    num_workers,patience,factor_decay,save_every,size_patch,beta):
 
    mse_loss = nn.MSELoss()
    mydct = DCT_nonlinear()
    myidct = DCT_nonlinear()
    block_side = 8      

    mydct.to(torch.float)
    myidct.to(torch.float)
    
    
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
    
    if num:
        mydct,myidct,optimizer,scheduler2 = load(op,num, path_load, mydct, myidct,optimizer,scheduler)
        print('Model loaded')
    losses_distortion = []
    losses_sparse = []
    losses =[]

    train_transform = transform_data(size_patch);
    dataset = ImageFolderYCbCr(dataset_path,train_transform)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
        
    for ei in range(last_epoch+1, num_epochs+last_epoch+1):
        for bi, image_ycbcr in enumerate(dataloader):
            
            batch_size, channels, rows, cols = patches.size()
            number_batch = 0
            block_index = 0

            latent = mydct(image_ycbcr)
            blk_recons = myidct(latent)  
            
            distortion_loss  = mse_loss(yuv_batch, blk_recons)  
            sparsity_penalty = beta * l1_norm(latent)
            loss = distortion_loss + sparsity_penalty

            losses_distortion.append(loss_mse.data.tem())
            losses_sparse.append(sparsity_penalty.data.tem())
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
    
    
        scheduler.step(np.mean(losses))
        zeros = torch.sum(latente == 0).data.item()
        total = latente.numel()
        psrn = compute_psnr(yuv_batch, blk_recons)
        
        print('\n Época [{}] loss distortion {:.4f}, loss sparsity {:.4f}, loss média {.:4f}'.
              format(np.mean(losses_distortion), np.mean(losses_distortion), np.mean(losses)))
        print('Last batch: PSNR {:.3f}, taxa de zeros {:.3f}'.format(psrn.data.item(),  1e2*zeros/total))
                
        if ei % save_every == 0:
            save(ei,'epoch', path_save,mydct,myidct, optimizer,scheduler)



main(dataset_path, output_file,lr,batch_size,path_save,path_load,op,num,last_epoch,num_epochs,
     shuffle,num_workers,patience,factor_decay,save_every,beta)

