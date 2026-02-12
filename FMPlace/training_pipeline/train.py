from os.path import join, exists, isfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import h5py

import numpy as np

import kitti_dataset
import nclt_dataset 

import models.loss_functions as loss_functions
'''
class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        #self.margin = 0.3
        self.margin = 1.0  

    def forward(self, anchor, positive, negative):
        
        pos_dist = torch.sqrt((anchor - positive).pow(2).sum())
        neg_dist = torch.sqrt((anchor - negative).pow(2).sum(1))
        
        loss = F.relu(pos_dist-neg_dist + self.margin)
        return loss#.mean()
'''




def train_epoch(epoch, model, train_set, opt, device, optimizer, writer, pool_size):
    
    epoch_loss = 0

    n_batches = (len(train_set) + opt.batchSize - 1) // opt.batchSize

    #criterion = loss_functions.TripletLoss().to(device)

    # Investigate Batch Contrastive Loss - UPDATE: causing gradient explosion
    batch_criterion = loss_functions.BatchContrastiveLoss(temperature=0.07).to(device)
    
    model.eval()
    

    if epoch == 0 or epoch % 2 == 0:
        print('====> Building Cache for Hard Mining')
        train_set.mining=False
        train_set.cache = join(opt.cachePath, 'train_feat_cache.hdf5')
        with h5py.File(train_set.cache, mode='w') as h5: 

            h5feat = h5.create_dataset("features", 
                    [len(train_set), pool_size], 
                    dtype=np.float32)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, 
                batch_size=opt.batchSize, shuffle=False, 
                collate_fn=kitti_dataset.collate_fn)
            with torch.no_grad():
                for iteration, (query, positives, negatives, indices) in enumerate(training_data_loader, 1):
                    
                    query = query.to(device)
                    _, global_descs = model(query)
                    h5feat[indices, :] = global_descs.detach().cpu().numpy()
        train_set.mining=True
        train_set.refreshCache()
        
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, 
                batch_size=opt.batchSize, shuffle=True, 
                collate_fn=kitti_dataset.collate_fn)
    
    model.train()

    for iteration, (query, positives, negatives, indices) in enumerate(training_data_loader):

        B, C, H, W = query.shape
        input = torch.cat([query, positives, negatives])

        input = input.to(device)
        
        _, global_descs = model(input)

        # POST - Normalize embeddings (important for stable distance behaviour)
        

        global_descs_Q, global_descs_P, global_descs_N = torch.split(global_descs, [B, B, negatives.shape[0]])
        

        for o in optimizer:
            o.zero_grad()

        # Triplet Loss
        '''
        # no need to train the kps feature
        loss = 0
        num_negs = negatives.shape[0]//B
        for i in range(len(global_descs_Q)):
            max_loss = torch.max(criterion(global_descs_Q[i], global_descs_P[i], global_descs_N[num_negs*i:num_negs*(i+1)]))
            loss += max_loss
        
        #loss = criterion(global_descs_Q, global_descs_P, global_descs_N)
        '''
        # Average loss over the batch
        loss_contrastive = batch_criterion(global_descs_Q, global_descs_P)
        loss = loss_contrastive

        loss /= opt.batchSize
        loss.backward()
        
        for o in optimizer:
            o.step()
 

        batch_loss = loss.item()
        epoch_loss += batch_loss
        # debug: gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        if iteration % 50 == 0:
            print(f"Iteration {iteration}: gradient norm = {total_norm:.6f}, loss = {batch_loss:.6f}")
      
        if iteration % 50 == 0 or n_batches <= 10:
            print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, 
                n_batches, batch_loss), flush=True)
            writer.add_scalar('Train/Loss', batch_loss, 
                    ((epoch-1) * n_batches) + iteration)
            

    for o in optimizer:
        o.zero_grad()

    avg_loss = epoch_loss / n_batches

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), 
            flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)