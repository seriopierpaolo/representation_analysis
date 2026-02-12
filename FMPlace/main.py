

import argparse
from math import ceil
import random
import shutil
import json
from os.path import join, exists, isfile
from os import makedirs
import os
from datetime import datetime

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import h5py

#from sklearn.decomposition import PCA

from tensorboardX import SummaryWriter
import numpy as np

from tqdm import tqdm
import faiss

import kitti_dataset
import nclt_dataset 
import helilpr_dataset
import toyota_dataset

from models.model_selector import model_selector

from training_pipeline.train import train_epoch
import models.loss_functions as loss_functions

from utils.config_reader import get_config

import time

#os.environ['CUDA_VISIBLE_DEVICES'] = '4'

def infer(eval_set, return_local_feats=False):
    test_data_loader = DataLoader(
        dataset=eval_set,
        num_workers=opt.threads,
        batch_size=opt.cacheBatchSize,
        shuffle=False
    )

    model.eval()
    model.to(device)

    total_time = 0.0
    n_batches = 0

    with torch.no_grad():
        all_global_descs = []
        all_local_feats = []

        for _, (imgs, _) in enumerate(tqdm(test_data_loader, disable=True)):
            imgs = imgs.to(device)

            start_time = time.perf_counter()
            local_feat, global_desc = model(imgs)
            torch.cuda.synchronize() if device.type == 'cuda' else None  # ensure accurate timing on GPU
            end_time = time.perf_counter()

            total_time += (end_time - start_time)
            n_batches += 1

            all_global_descs.append(global_desc.detach().cpu().numpy())
            if return_local_feats:
                all_local_feats.append(local_feat.detach().cpu().numpy())

    avg_inference_time = total_time / n_batches if n_batches > 0 else 0
    print(f"Average inference time per batch: {avg_inference_time:.4f} seconds")

    if return_local_feats:
        return (
            np.concatenate(all_local_feats, axis=0),
            np.concatenate(all_global_descs, axis=0),
        )
    else:
        return np.concatenate(all_global_descs, axis=0)
    


'''
def getClusters(cluster_set):
    n_descriptors = 10000
    n_per_image = 25
    n_im = ceil(n_descriptors/n_per_image)

    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), n_im, replace=False))
    data_loader = DataLoader(dataset=cluster_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                sampler=sampler)

    if not exists(opt.cachePath):
        makedirs(opt.cachePath)

    initcache = join(opt.cachePath, 'desc_cen.hdf5')
    with h5py.File(initcache, mode='w') as h5: 
        with torch.no_grad():
            model.eval()
            print('====> Extracting Descriptors')
            all_feats = h5.create_dataset("descriptors", 
                        [n_descriptors, 128], 
                        dtype=np.float32)

            for iteration, (query, _, _, _) in enumerate(data_loader, 1):
                query = query.to(device)
                local_feat, _, _ = model(query)
                local_feat = local_feat.view(query.size(0), 128, -1).permute(0, 2, 1)
                
                batchix = (iteration-1)*opt.cacheBatchSize*n_per_image
                for ix in range(local_feat.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(local_feat.size(1), n_per_image, replace=False)
                    startix = batchix + ix*n_per_image
                    all_feats[startix:startix+n_per_image, :] = local_feat[ix, sample, :].detach().cpu().numpy()

                if iteration % 50 == 0 or len(data_loader) <= 10:
                    print("==> Batch ({}/{})".format(iteration, 
                        ceil(n_im/opt.cacheBatchSize)), flush=True)
        
        print('====> Clustering..')
        niter = 100
        kmeans = faiss.Kmeans(128, 64, niter=niter, verbose=False)
        kmeans.train(all_feats[...])

        print('====> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')
'''

def saveCheckpoint(state, is_best, model_out_path, filename='checkpoint.pth.tar'):
    filename = model_out_path+'/'+filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_out_path+'/'+'model_best.pth.tar')


if __name__ == "__main__":
    
    opt = get_config()
    device = torch.device("cuda:0")
    
    # Loading parameters
    #------------------------------------------------------
    
    USE_NCLT = opt.USE_NCLT 
    USE_HELILPR = opt.USE_HELILPR
    REPRESENTATION = opt.representation


    #overwriting representations
    REPRESENTATION = "fv_multi"
    USE_NCLT = True
    USE_HELILPR = True
    USE_TOYOTA = True

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    #------------------------------------------------------

    print('===> Building model')

    MODEL = "dino3vlad"
    MODEL = "dino3"

    #image_input_size = (128, 128)

    model, projection_dimension = model_selector(MODEL, device)
  

    model = model.to(device)

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {num_trainable/1e6:.2f}M / {num_total/1e6:.2f}M")


    #This code enables past weights loading, for testing
    """
    # initialize netvlad with pre-trained or cluster
    if opt.load_from != '':
        if opt.ckpt.lower() == 'latest':
            resume_ckpt = join(opt.load_from,  'checkpoint.pth.tar')
        elif opt.ckpt.lower() == 'best':
            resume_ckpt = join(opt.load_from, 'model_best.pth.tar')

        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)

            print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))
    else:
        initcache = join(opt.cachePath, 'desc_cen.hdf5')
        if not isfile(initcache):
            train_set = kitti_dataset.TrainingDataset()
            print('===> Calculating descriptors and clusters')
            getClusters(train_set)
        with h5py.File(initcache, mode='r') as h5: 
            clsts = h5.get("centroids")[...]
            traindescs = h5.get("descriptors")[...]
            model.pooling.init_params(clsts, traindescs) 
            model = model.cuda() 
    """


    mode = 'train'
    #if opt.mode.lower() == 'train':

    if mode.lower() == 'train':
        # preparing tensorboard
        writer = SummaryWriter(log_dir=join(opt.runsPath, datetime.now().strftime('%b%d_%H-%M-%S')))

        logdir = writer.file_writer.get_logdir()
        try:
            makedirs(logdir)
        except:
            pass

        with open(join(logdir, 'flags.json'), 'w') as f:
            f.write(json.dumps(
                {k:v for k,v in vars(opt).items()}
                ))
        print('===> Saving state to:', logdir)


        print('===> Loading dataset(s)')

        train_set = kitti_dataset.TrainingDataset(representation=REPRESENTATION) 

        # ADDING MORE DATA AUGMENTATION
        #train_set_aug = kitti_dataset.TrainingDatasetAugmented() 
        #comb_train_set = torch.utils.data.ConcatDataset([train_set, train_set_aug])
        #train_set = kitti_dataset.CombinedDataset(comb_train_set)

        val_set={}
        for seq in ['02', '05', '06', '08']:   
        # for seq in ['2012-02-04', '2012-03-17', '2012-06-15', '2012-09-28','2012-11-16','2013-02-23']:
            val_set[seq] = kitti_dataset.InferDataset(seq=seq, representation=REPRESENTATION)

        
        # Base DINO model (light fine-tune)
        optimizer_base = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.base_model.parameters()),
            lr=1e-5,          
            weight_decay=0.04 # mild regularization 
        )
        optimizers_list = [optimizer_base] 


        
        if (MODEL != "dino") & (MODEL !="dino3"):

            # NetVLAD head: new layer, needs faster learning
            optimizer_head = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.projection_model.parameters()),
                lr=1e-3,          # much higher, typical for small projection layers
                weight_decay=0.001
            )

            optimizers_list.append(optimizer_head)
        
            '''
            optimizer_mlp = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.reduction_head.parameters()),
                lr=1e-3,          # much higher, typical for small projection layers
                weight_decay=0.001
            )

            optimizers_list.append(optimizer_head)
            '''

        

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_trainable = sum(p.numel() for p in trainable_params)
        print(f"Trainable parameters: {len(trainable_params)} tensors, total elements: {total_trainable}")
        print("Optimizer parameter groups info:", optimizers_list)
        if total_trainable == 0:
            raise RuntimeError("No trainable parameters in model — check requires_grad flags")
            

        best_score = 0

        for epoch in range(opt.nEpochs):
            
            train_epoch(epoch, model, train_set, opt, device , optimizers_list, writer, projection_dimension)
            
            print('===> Testing Phase')

            
            print('===> Testing on KITTI sequences')
            dataset_name = "KITTI"
            recalls_kitti = []; f1s_kitti = []; aucs_kitti = []
            

            for seq in ['02', '05', '06']:
                print(f'----> Sequence {seq}')
                test_set = kitti_dataset.InferDataset(seq=seq, representation=REPRESENTATION)
                local_descs, global_descs = infer(test_set, return_local_feats=True)
                recall_at1, max_f1, auc = kitti_dataset.evaluateResults(
                    seq, global_descs=global_descs, dataset=test_set
                )

                recalls_kitti.append(recall_at1)
                f1s_kitti.append(max_f1)
                aucs_kitti.append(auc)

                #Saving on Tensorboard
                writer.add_scalars('val/' + dataset_name +'/Recall', {seq: recall_at1}, epoch)
                writer.add_scalars('val/' + dataset_name +'/AUC', {seq: auc}, epoch)
                writer.add_scalars('val/' + dataset_name +'/Max F1', {seq: max_f1}, epoch)

                print(f"[{seq}] | Recall: {recall_at1:.4f} | Max F1: {max_f1:.4f} | AUC: {auc:.4f}  | Epoch: {epoch}")
            
            mean_recall = np.mean(recalls_kitti)
            mean_auc = np.mean(aucs_kitti)
            mean_f1 = np.mean(f1s_kitti)

            writer.add_scalar('val/'+ dataset_name + '/Mean_Recall', mean_recall, epoch)
            writer.add_scalar('val/'+ dataset_name + '/Mean_AUC', mean_auc, epoch)
            writer.add_scalar('val/'+ dataset_name + '/Mean_F1', mean_f1, epoch)



            print(f"===> Mean KITTI Results | Recall: {mean_recall:.4f} | AUC: {mean_auc:.4f} | F1: {mean_f1:.4f}")

   



            '''
            if is_best:

                #TESTING ON OTHER DATASETS
                if USE_NCLT:  
                    print('===> Testing on NCLT sequences')
                    dataset_name = "NCLT"
                    
                    eval_seq =  ['2012-01-15', '2012-02-04', '2012-03-17', '2012-06-15']#, '2012-09-28', '2012-11-16', '2013-02-23']

                    eval_datasets = [], eval_global_descs = []
                    
                    recalls_nclt = [], f1s_nclt = [], aucs_nclt = []

                    for seq in eval_seq:   

                        test_set = nclt_dataset.InferDataset(seq=seq, representation=REPRESENTATION)   
                        global_descs = infer(test_set)
                        eval_global_descs.append(global_descs)
                        eval_datasets.append(test_set)
                        recall_nclt, f1_nclt, auc = nclt_dataset.evaluateResults(seq, eval_global_descs, eval_datasets)# (q_descs, db_descs, q_dataset, db_dataset)
                        recalls_nclt.append(recall_nclt)
                        
                        f1s_nclt.append(f1_nclt)
                        aucs_nclt.append(auc)
                    
                    for ii in range(1,len(recalls_nclt)):
                        print(f"[{eval_seq[ii]}] | Recall: {recalls_nclt[ii]:.4f} | Max F1: {f1s_nclt[ii]:.4f} | AUC: {aucs_nclt[ii]:.4f}  | Epoch: {epoch}")

                        #Saving on Tensorboard
                        writer.add_scalars('val/' + dataset_name +'/Recall', {eval_seq[ii]: recall_at1}, epoch)
                        writer.add_scalars('val/' + dataset_name +'AUC', {eval_seq[ii]: auc}, epoch)
                        writer.add_scalars('val/' + dataset_name +'Max F1', {eval_seq[ii]: max_f1}, epoch)
                



                USE_HELILPR = True
                if USE_HELILPR:
                    #print('====> Extracting Features of HeliLPR and calculating recalls')
                    print('===> Testing on HELILPR sequences')
                    dataset_name = "HELILPR"
                    eval_seq =  ['rb01', 'rb02', 'rb03'] 

                    recalls_helilpr = [], f1s_helilpr = [], aucs_helilpr = []

                    for seq in eval_seq:   

                        test_set = helilpr_dataset.InferDataset(seq=seq, representation=REPRESENTATION)   
                        global_descs = infer(test_set)

                        recall_helilpr, f1s_helilpr, aucs_helilpr = helilpr_dataset.evaluateResults(seq, global_descs, test_set)# (q_descs, db_descs, q_dataset, db_dataset)
                        
                        recalls_helilpr.append(recall_helilpr)
                        f1s_helilpr.append(f1s_helilpr)
                        aucs_helilpr.append(aucs_helilpr)

                        print('%s: %0.2f'%(seq, recall_helilpr))

                        #Saving on Tensorboard
                        writer.add_scalars('val/' + dataset_name +'/Recall', {seq: recall_at1}, epoch)
                        writer.add_scalars('val/' + dataset_name +'AUC', {seq: auc}, epoch)
                        writer.add_scalars('val/' + dataset_name +'Max F1', {seq: max_f1}, epoch)
                    







                USE_TOYOTA = True
                if USE_TOYOTA:
                    
                    print('===> Testing on Toyota sequences')
                    dataset_name = "HELILPR"
                
                    eval_seq =  ['00'] #'2012-06-15', '2012-09-28', '2012-11-16', '2013-02-23']
                    
                    recalls_toyota = [], f1s_toyota = [], aucs_toyota = []
                    for seq in eval_seq:   

                        print('Loading seq ', seq)
                        test_set = toyota_dataset.InferDataset(seq=seq, representation=REPRESENTATION)   
                        global_descs = infer(test_set)
                        print('Loaded seq ', seq)
                        recall_toyota,_,_ = toyota_dataset.evaluateResults(seq, global_descs, test_set)# (q_descs, db_descs, q_dataset, db_dataset)
                        recalls_toyota.append(recall_toyota)
                    
            '''




            is_best = mean_recall > best_score 
            if is_best:   
                best_score = mean_recall
            
            saveCheckpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'recalls': mean_recall,
                    'best_score': best_score,
                    'optimizer' : [o.state_dict() for o in optimizers_list],
            }, is_best, logdir)



        print('===> Best Recall: %0.2f'%(mean_recall*100))
        writer.close()


    #elif opt.mode.lower() == 'test':
    elif mode.lower() == 'test':
        
        print('===> Running evaluation step')

           
        recalls_kitti = []
        aucs_kitti = []
        f1s_kitti = []
        print('====> Extracting Features of KITTI and calculating recalls')
        eval_seq = ['02']  # add other sequences as needed

        for seq in eval_seq:
            print(f'----> Sequence {seq}')

            test_set = kitti_dataset.InferDataset(seq=seq)
            global_descs = infer(test_set)
            # evaluateResults signature: (seq, q_descs, local_feats_or_db, test_set, ...)
            recall_at1, max_f1, auc = kitti_dataset.evaluateResults(seq, global_descs, None, test_set)
            print(f'{seq} - Recall@1: {recall_at1:.2f}')
            print(f'{seq} - F1: {max_f1*100:.2f}')
            print(f'{seq} - AUC: {auc*100:.2f}')

            recalls_kitti.append(recall_at1)
            f1s_kitti.append(max_f1)
            aucs_kitti.append(auc)

        mean_recall = np.mean(recalls_kitti)
        print('\n################# Recall @ top 1 on KITTI ########################\n')
        for ii, s in enumerate(eval_seq):
            print(f'{s}: {recalls_kitti[ii]:.2f}')
        print(f'mean: {mean_recall:.2f}')


        print('mean: %0.2f'%(mean_recall))
        print('################# Global Loc Results on KITTI 08  ##################\n')

        #print('Success rate: %0.2f; Mean Trans. Err.: %0.2f; Mean Rot. Err.: %0.2f'%(success_rate*100, mean_trans_err, mean_rot_err))

        
        


        print('\n')
        


        USE_NCLT = True
        if USE_NCLT:
            print('====> Extracting Features of NCLT and calculating recalls')
            eval_seq =  ['2012-01-15', '2012-02-04', '2012-03-17'] #'2012-06-15', '2012-09-28', '2012-11-16', '2013-02-23']
            eval_datasets = []
            eval_global_descs = []
            recalls_nclt = []
            for seq in eval_seq:   

                test_set = nclt_dataset.InferDataset(seq=seq, representation=REPRESENTATION)   
                global_descs = infer(test_set)
                eval_global_descs.append(global_descs)
                eval_datasets.append(test_set)
                recall_nclt,max_f1,auc = nclt_dataset.evaluateResults(seq, eval_global_descs, eval_datasets)# (q_descs, db_descs, q_dataset, db_dataset)
                recalls_nclt.append(recall_nclt)
                
                print(f'{seq} - Recall@1: {recall_at1:.2f}')
                print(f'{seq} - F1: {max_f1*100:.2f}')
                print(f'{seq} - AUC: {auc*100:.2f}')

            print('\n################# Recall @ top 1 on NCLT ########################\n')
            mean_recall = np.mean(recalls_nclt[1:])


            for ii in range(1,len(eval_seq)):
                print('%s: %0.2f'%(eval_seq[ii], recalls_nclt[ii]))

        
        USE_HELILPR = True
        if USE_HELILPR:
            print('====> Extracting Features of HeliLPR and calculating recalls')
            eval_seq =  ['rb01', 'rb02', 'rb03'] 
            recalls_helilpr = []
            for seq in eval_seq:   

                print('Loading seq ', seq)
                test_set = helilpr_dataset.InferDataset(seq=seq, representation=REPRESENTATION)   
                global_descs = infer(test_set)
                print('Loaded seq ', seq)
                recall_helilpr,_,_ = helilpr_dataset.evaluateResults(seq, global_descs, test_set)# (q_descs, db_descs, q_dataset, db_dataset)
                recalls_helilpr.append(recall_helilpr)
            
                print(f'{seq} - Recall@1: {recall_at1:.2f}')
                print(f'{seq} - F1: {max_f1*100:.2f}')
                print(f'{seq} - AUC: {auc*100:.2f}')

            print('\n################# Recall @ top 1 on HeliLPR ########################\n')
            mean_recall = np.mean(recalls_helilpr)


            for ii in range(len(eval_seq)):
                print('%s: %0.2f'%(eval_seq[ii], recalls_helilpr[ii]))


        USE_TOYOTA = True
        if USE_TOYOTA:
            
            print('====> Extracting Features of Toyota and calculating recalls')
            eval_seq =  ['00'] #'2012-06-15', '2012-09-28', '2012-11-16', '2013-02-23']
            recalls_toyota = []
            for seq in eval_seq:   

                print('Loading seq ', seq)
                test_set = toyota_dataset.InferDataset(seq=seq, representation=REPRESENTATION)   
                global_descs = infer(test_set)
                print('Loaded seq ', seq)
                recall_toyota,_,_ = toyota_dataset.evaluateResults(seq, global_descs, test_set)# (q_descs, db_descs, q_dataset, db_dataset)
                recalls_toyota.append(recall_toyota)
            

            print('\n################# Recall @ top 1 on Toyota ########################\n')
            mean_recall = np.mean(recalls_toyota)

        

   

