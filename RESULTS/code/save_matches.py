


from math import ceil
import random
import numpy as np

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


from tqdm import tqdm
import faiss

import dataset_manager.kitti_dataset as kitti_dataset
import dataset_manager.nclt_dataset as nclt_dataset
import dataset_manager.helilpr_dataset as helilpr_dataset
import dataset_manager.toyota_dataset as toyota_dataset

from models.model_selector import model_selector



from utils.config_reader import get_config

import time

#os.environ['CUDA_VISIBLE_DEVICES'] = '4'

def load_weights(path, model):
    ckpt_path = os.path.join(path, 'model_best.pth.tar')
    if not os.path.isfile(ckpt_path):
        print(f"Warning: checkpoint not found at {ckpt_path}")
        return model
    print(f"Loading weights from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    return model



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
    






if __name__ == "__main__":
    
    opt = get_config()
    device = torch.device("cuda:1")
    
    # Loading parameters
    #------------------------------------------------------
    



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

    #MODEL = "dino3vlad"
    MODEL = "dino3"

    #image_input_size = (128, 128)

    model, projection_dimension = model_selector(MODEL, device)
  

    model = model.to(device)

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {num_trainable/1e6:.2f}M / {num_total/1e6:.2f}M")

    model = load_weights(f"./representation_analysis/RESULTS/Trainings/MSHead/"+REPRESENTATION+"/checkpoint/", model)

    #This code enables past weights loading, for testing
    mode = 'test'  # 'train' or 'test'
    if mode.lower() == 'test':
        
        print('===> Running evaluation step')

        USE_KITTI = False
        if USE_KITTI:
                
            recalls_kitti = []
            aucs_kitti = []
            f1s_kitti = []
            print('====> Extracting Features of KITTI and calculating recalls')
            eval_seq = ['02', '05', '06']  # add other sequences as needed

            for seq in eval_seq:
                print(f'----> Sequence {seq}')

                test_set = kitti_dataset.InferDataset(seq=seq, representation=REPRESENTATION)
                global_descs = infer(test_set)
                # evaluateResults signature: (seq, q_descs, local_feats_or_db, test_set, ...)
                recall_at1, max_f1, auc = kitti_dataset.evaluateResults(seq, global_descs, test_set)
                print(f'{seq} - Recall@1: {recall_at1*100:.2f}')
                print(f'{seq} - F1: {max_f1*100:.2f}')
                print(f'{seq} - AUC: {auc*100:.2f}')

                recalls_kitti.append(recall_at1)
                f1s_kitti.append(max_f1)
                aucs_kitti.append(auc)


            #print('Success rate: %0.2f; Mean Trans. Err.: %0.2f; Mean Rot. Err.: %0.2f'%(success_rate*100, mean_trans_err, mean_rot_err))

            
        


        print('\n')
        




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
                recall_toyota,max_f1_toyota,auc_toyota = toyota_dataset.evaluateResults(seq, global_descs, test_set)# (q_descs, db_descs, q_dataset, db_dataset)
                recalls_toyota.append(recall_toyota)
            
                print(f'{seq} - Recall@1: {recall_toyota*100:.2f}')
                print(f'{seq} - F1: {max_f1_toyota*100:.2f}')
                print(f'{seq} - AUC: {auc_toyota*100:.2f}')

            print('\n################# Recall @ top 1 on Toyota ########################\n')
            mean_recall = np.mean(recalls_toyota)

        

   

