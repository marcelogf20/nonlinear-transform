from torch import nn
import torch
from utils import *



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

def main(dataset_path,output_file,lr,batch_size,path_save,path_load,op,num,last_epoch,num_epochs,shuffle,num_workers,
         patience,factor_decay,save_every):
 
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
    losses_mse = []
    losses_kl = []
    losses =[]
    
    dataset = BSDS500Crop128(dataset_path)
    dataloader = DataLoader(dataset,batch_size=1,shuffle=shuffle,num_workers=num_workers)

        
    for ei in range(last_epoch+1, num_epochs+last_epoch+1):
        #for input_file in glob.glob(os.path.join(dataset_path,'*.png')):
        for bi, image_ycbcr in enumerate(dataloader):
            
            image_ycbcr = np.transpose(image_ycbcr, (0,3,2,1))
            image_ycbcr = image_ycbcr[0,:,:,:]
            
            channels, rows, cols = image_ycbcr.shape
            #x = randint(0,rows-16-1)
            #y = randint(0,cols-16-1)
            x,y =5,4
            
            image_ycbcr =image_ycbcr[:,x:x+16,y:y+16]
            channels, rows, cols = image_ycbcr.shape

            blks_set = (image_ycbcr.float()/255).float()
            # block size: 8x8
            if rows % 8 == cols % 8 == 0:
                blocks_count = rows // 8 * cols // 8
            else:
                raise ValueError(("the width and height of the image "
                                  "should both be mutiples of 8"))        
            
            number_batch = 0
            block_index = 0
    
            for i in range(0, rows, 8):
                for j in range(0, cols, 8):

                    block_index += 1
                    yuv_patch = torch.zeros([1,1,8,8], dtype=torch.float32)
                                        
                    for k in range(1):
                        yuv_patch[0,k,:,:] = blks_set[k,i:i+8,j:j+8]     
                        
                    try:
                        yuv_batch =  torch.cat((yuv_batch, yuv_patch),0)                
                    except NameError:
                        yuv_batch = yuv_patch

            latente = mydct(yuv_batch)
            blk_recons = myidct(latente)  

            BETA = 0.1
            RHO = 0.01
            N_HIDDEN = 64
            rho = torch.zeros([1,1,8,8]).unsqueeze(0)
            rho[0,0,:,:] = RHO
                
                
            #[RHO for _ in range(N_HIDDEN)]).unsqueeze(0)
            #print(latente)
            #print(rho)
            
            loss_mse  = mse_loss(yuv_batch, blk_recons)             
            rho_hat = torch.sum(latente, dim=0, keepdim=True)
            sparsity_penalty = BETA * kl_divergence(rho, rho_hat)
            
            losses_mse.append(loss_mse)
            losses_kl.append(sparsity_penalty)
            
            loss = loss_mse + sparsity_penalty
            losses.append(loss.data.item())                 
            
            a = list(mydct.parameters())[0].clone()
            a2 = list(myidct.parameters())[0].clone()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            scheduler.step(loss.data.item())
 

            b = list(mydct.parameters())[0].clone()
            b2 = list(myidct.parameters())[0].clone()
            
            if torch.equal(a.data, b.data):
                print('equal mydct ')
            if torch.equal(a2.data, b2.data):
                print('equal myidct')
    
    
            if ei%20==0:
                zeros = torch.sum(latente == 0).data.item()
                total = latente.numel()
                print('\n Epoca',ei,'loss_mse',loss_mse.data.item(),'loss_kl',sparsity_penalty.data.item(),'porcentagem de zeros', 1e2*zeros/total)

                
            if ei%100==0:
                psrn = compute_psnr(yuv_batch, blk_recons)
                print('Epoch:',ei, 'loss média', np.mean(losses), 'Last PSNR ',psrn.data.item())
                print('\n Latente0',latente[0,0,:])
                print('\n Latente1',latente[1,0,:])
                print('\n Latente2',latente[2,0,:])
                save(ei,'epoch', path_save,mydct,myidct, optimizer,scheduler)
                losses=[]
            del yuv_batch            


main(dataset_path, output_file,lr,batch_size,path_save,path_load,op,num,last_epoch,num_epochs,
     shuffle,num_workers,patience,factor_decay,save_every)

