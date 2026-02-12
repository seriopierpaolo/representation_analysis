import os
from os.path import join, exists
import numpy as np
import cv2
from imgaug import augmenters as iaa
import torch
import torch.utils.data as data

import h5py

from RANSAC import rigidRansac
import faiss

import utils.pointcloud_tools as pc_tools
import utils.image_tools as im_tools

import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend
import matplotlib.pyplot as plt
import time

kitti_seq_split_points = {"00":3000, "02":3400, "05":1000, "06":1000, '08':1000}
#representation = 'fv_multi'


#Use this boolean only with range and fv representations and if you want padding and resizing
PAD_RESIZE = False

ROTATE = True #Use only with cartesian and polar representation!



class InferDataset(data.Dataset):
    def __init__(self, seq, representation, dataset_path = '/dataset/kitti-dataset/lidar_representations/representation_analysis/',sample_inteval=1):
        super().__init__()

        self.sample_inteval = sample_inteval
        self.db_split_index = int(kitti_seq_split_points[seq]/sample_inteval)
        # bev path
        imgs_p = os.listdir(dataset_path+'sequences/'+seq+'/'+ representation +'/')
        imgs_p.sort()
        self.imgs_path = [dataset_path+'sequences/'+seq+ '/' + representation + '/' +imgs_p[i] for i in range(0,len(imgs_p), sample_inteval)]

        # gt_pose
        self.poses = np.loadtxt(dataset_path+'poses/'+seq+'.txt')[::sample_inteval]


    def __getitem__(self, index):
        
        img = cv2.imread(self.imgs_path[index], 0)

        #---------------------------------------------------------
        #ADDED PAD-RESIZE
        if PAD_RESIZE:
            img = im_tools.pad_resize(img, target_size=(128,128))
        #---------------------------------------------------------

        if 0:  #test rotation
            
            mat = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2 ), np.random.randint(0,360), 1)
            img = cv2.warpAffine(img, mat, img.shape[:2])

        img = (img.astype(np.float32))/256 
        img = img[np.newaxis, :, :].repeat(3,0)
        
        return  img, index

    def __len__(self):
        return len(self.imgs_path)
        


'''
# DA CAMBIARE
def evaluateResults(seq, global_descs, local_feats, dataset, match_results_save_path=None):

    if match_results_save_path is not None: 
        os.system('mkdir -p ' + match_results_save_path)
        all_errs = []
        #local_feats = local_feats.transpose(0,2,3,1)

    gt_thres = 5  # gt threshold

    #OPTIONAL STEP ADDED POST
    global_descs = global_descs / np.linalg.norm(global_descs, axis=1, keepdims=True)

    faiss_index = faiss.IndexFlatL2(global_descs.shape[1]) 
    faiss_index.add(global_descs[:dataset.db_split_index])

    _, predictions = faiss_index.search(global_descs[dataset.db_split_index+int(200/dataset.sample_inteval):], 1)  #top1
    
    
    eval_start_split_point = dataset.db_split_index+int(200/dataset.sample_inteval)
    all_positives = 0
    tp = 0
    for q_idx, pred in enumerate(predictions):

        query_idx = eval_start_split_point+q_idx
        gt_dis = (dataset.poses[query_idx] - dataset.poses[:dataset.db_split_index])**2
        positives = np.where(np.sum(gt_dis[:,[3,7,11]],axis=1) < gt_thres**2 )[0]
        if len(positives)>0:
            all_positives+=1
            if pred[0] in positives:
                tp += 1
    
      
    recall_top1 = tp / all_positives #tp/(tp+fp)

    

    if match_results_save_path is not None:
        all_errs = np.array(all_errs)
        success_loc = (all_errs[:,0]<2) & (all_errs[:,1]<5)
        success_rate = np.sum(success_loc)/all_positives
        mean_trans_err = np.mean(all_errs[success_loc,1])
        mean_rot_err = np.mean(all_errs[success_loc,0]) 
        return recall_top1, success_rate, mean_trans_err, mean_rot_err
    else:
        return recall_top1

'''
        
def collate_fn(batch):

    batch = list(filter (lambda x:x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query=np.array(query)
    positive=np.array(positive)
    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    
    negatives = torch.cat(negatives, 0)
    indices = list(indices)

    return query, positive, negatives, indices





def evaluateResults(seq, global_descs, dataset, gt_thres=5.0):
    # --- Normalize descriptors ---
    global_descs = global_descs / np.linalg.norm(global_descs, axis=1, keepdims=True)
    db_vecs = global_descs[:dataset.db_split_index]
    q_vecs  = global_descs[dataset.db_split_index:]

    # --- Build FAISS index ---
    faiss_index = faiss.IndexFlatL2(db_vecs.shape[1])
    faiss_index.add(db_vecs)

    # --- Measure FAISS search time ---
    start_time = time.perf_counter()
    dists, preds = faiss_index.search(q_vecs, 1)  # top-1 distances
    end_time = time.perf_counter()

    total_search_time = end_time - start_time
    avg_time_per_query = total_search_time / len(q_vecs)

    print(f"Total FAISS search time: {total_search_time:.4f} s")
    print(f"Average time per query: {avg_time_per_query*1000:.4f} ms")




    # --- Compute ground-truth matches ---
    tp_flags = []
    all_gt = 0

    

    for q_idx, (dist, pred) in enumerate(zip(dists[:, 0], preds[:, 0])):
        query_idx = dataset.db_split_index + q_idx

        gt_dis = (dataset.poses[query_idx] - dataset.poses[:dataset.db_split_index]) ** 2
        positives = np.where(np.sum(gt_dis[:, [3, 7, 11]], axis=1) < gt_thres ** 2)[0]
        if len(positives) == 0:
            continue
        all_gt += 1
        tp_flags.append((pred in positives, dist))

    # --- Compute Recall@1 ---
    tp = sum([f for f, _ in tp_flags])
    recall_at1 = tp / (all_gt + 1e-8)

    # --- Prepare arrays for precision-recall analysis ---
    distances = np.array([d for _, d in tp_flags])
    labels    = np.array([int(f) for f, _ in tp_flags])
    thresholds = np.linspace(distances.min(), distances.max(), 1000)

    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        preds_pos = distances < t
        tp = np.sum((preds_pos == 1) & (labels == 1))
        fp = np.sum((preds_pos == 1) & (labels == 0))
        fn = np.sum((preds_pos == 0) & (labels == 1))

        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    max_f1 = np.max(f1s)

    # --- Compute AUC (must sort by recall) ---
    sorted_idx = np.argsort(recalls)
    recalls_sorted = np.array(recalls)[sorted_idx]
    precisions_sorted = np.array(precisions)[sorted_idx]
    auc = np.trapz(precisions_sorted, recalls_sorted)

    # --- Plot the Precision-Recall Curve ---
    plt.figure(figsize=(7, 6))
    plt.plot(recalls_sorted, precisions_sorted, color='royalblue', linewidth=2, 
             label=f'PR Curve (AUC = {auc:.3f})')

    # Highlight max F1 point
    max_f1_idx = np.argmax(f1s)
    plt.scatter(recalls[max_f1_idx], precisions[max_f1_idx],
                color='crimson', s=80, edgecolor='white', zorder=5,
                label=f'Max F1 = {max_f1:.3f}')

    plt.title(f'Precision–Recall Curve — Seq {seq}', fontsize=13, pad=10)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.grid(alpha=0.3)
    plt.legend(frameon=True, loc='lower left', fontsize=10)
    plt.tight_layout()

    # --- Save the plot ---
    save_path = f'./representation_analysis/FMPlace/precision_recall_KITTI_seq_{seq}.png'
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Precision-Recall curve saved to: {save_path}")

    return recall_at1, max_f1, auc






class TrainingDataset(data.Dataset):
    def __init__(self, representation, dataset_path = '/dataset/kitti-dataset/lidar_representations/representation_analysis/',seq='00'):
        super().__init__()

        # bev path
        imgs_p = os.listdir(dataset_path+'sequences/'+seq+'/'+ representation +'/')
        imgs_p.sort()
        self.imgs_path = [dataset_path+'sequences/'+seq+'/'+ representation +'/'+i for i in imgs_p]

        # gt_pose, only first 3000 frames of KITTI for training
        self.poses = np.loadtxt(dataset_path+'poses/'+seq+'.txt')
        self.poses = self.poses[:3000]
        
        # neg, pos threshold
        self.pos_thres = 5
        self.neg_thres = 10 

        # compute pos and negs for each query
        self.num_neg = 10
        self.positives = []
        self.negatives = []

        for qi in range(len(self.poses)):
            q_pose = self.poses[qi]
            dises = np.sqrt(np.sum(((q_pose-self.poses)**2)[:,[3,7,11]],axis=1))            
            indexes = np.argsort(dises)

            remap_index = indexes[np.where(dises[indexes]<self.pos_thres)[0]]
            self.positives.append(remap_index)
            self.positives[-1] = self.positives[-1][1:] #exclude query itself

            negs = indexes[np.where(dises[indexes]>self.neg_thres)[0]]
            self.negatives.append(negs)

        self.mining = False
        self.cache = None # filepath of HDF5 containing feature vectors for images



    # refresh cache for hard mining
    def refreshCache(self):
        h5 = h5py.File(self.cache, mode='r')
        self.h5feat = np.array(h5.get("features"))

    def __getitem__(self, index):
        
        if self.mining:
            q_feat = self.h5feat[index]

            pos_feat = self.h5feat[self.positives[index]]
            dis_pos = np.sqrt(np.sum((q_feat.reshape(1,-1)-pos_feat)**2,axis=1))

            min_idx = np.where(dis_pos==np.max(dis_pos))[0][0] 
            pos_idx = np.random.choice(self.positives[index], 1)[0]#
            # pos_idx = self.positives[index][min_idx]

            neg_feat = self.h5feat[self.negatives[index].tolist()]
            dis_neg = np.sqrt(np.sum((q_feat.reshape(1,-1)-neg_feat)**2,axis=1))
            
            dis_loss = (-dis_neg) + 0.3
            dis_inc_index_tmp = dis_loss.argsort()[:-self.num_neg-1:-1]

            neg_idx = self.negatives[index][dis_inc_index_tmp[:self.num_neg]]

              
        else:
            pos_idx = self.positives[index][0]
        
            neg_idx = np.random.choice(np.arange(len(self.negatives[index])).astype(int), self.num_neg)
            neg_idx = self.negatives[index][neg_idx]
        

        
        query = cv2.imread(self.imgs_path[index])

        #---------------------------------------------------------
        #PAD-RESIZE ADDED
        if PAD_RESIZE:
            query = im_tools.pad_resize(query, target_size=(128,128))
        #---------------------------------------------------------

        # rot augmentation
        
        if ROTATE:
            mat = cv2.getRotationMatrix2D((query.shape[1]//2, query.shape[0]//2 ), np.random.randint(0,360), 1)
            query = cv2.warpAffine(query, mat, query.shape[:2])

        # Clip the pixel values to the valid range [0, 255]
        query = np.clip(query, 0, 255)
        
        query = query.transpose(2,0,1)

        #cv2.imwrite('./query.png', query[0])

        positive = cv2.imread(join(self.imgs_path[pos_idx]))#     

        #---------------------------------------------------------
        #PAD-RESIZE ADDED
        if PAD_RESIZE:
            positive = im_tools.pad_resize(positive, target_size=(128,128))
        #---------------------------------------------------------

        if ROTATE:
            mat = cv2.getRotationMatrix2D((positive.shape[1]//2, positive.shape[0]//2 ), np.random.randint(0,360), 1)
            positive = cv2.warpAffine(positive, mat, positive.shape[:2])
        
        positive = positive.transpose(2,0,1)
        
        #cv2.imwrite('./positive.png', positive[0])

    
        query = (query.astype(np.float32))/256
        positive = (positive.astype(np.float32)/256)

        negatives = []

        for neg_i in neg_idx:
        
            negative = cv2.imread(self.imgs_path[neg_i])

            #---------------------------------------------------------
            #PAD-RESIZE ADDED
            if PAD_RESIZE:
                negative = im_tools.pad_resize(negative, target_size=(128,128))
            #---------------------------------------------------------

            if ROTATE:
                mat = cv2.getRotationMatrix2D((negative.shape[1]//2, negative.shape[0]//2 ), np.random.randint(0,360), 1)
                negative = cv2.warpAffine(negative, mat, negative.shape[:2]) 
            
            negative = negative.transpose(2,0,1)
            negative = (negative)/256
            
            negatives.append(torch.from_numpy(negative.astype(np.float32)))

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, index

    def __len__(self):
        return len(self.poses)
        
    



class TrainingDatasetAugmented(data.Dataset):
    def __init__(self, dataset_path = 'datasets/KITTI/',seq='00'):
        super().__init__()

        # bev path
        imgs_p = os.listdir(dataset_path+seq+'/bev_imgs/')
        imgs_p.sort()
        self.imgs_path = [dataset_path+seq+'/bev_imgs/'+i for i in imgs_p]

        # gt_pose, only first 3000 frames of KITTI for training
        self.poses = np.loadtxt(dataset_path+'poses/'+seq+'.txt')
        self.poses = self.poses[:3000]
        
        # neg, pos threshold
        self.pos_thres = 10
        self.neg_thres = 25 # 

        # compute pos and negs for each query
        self.num_neg = 10
        self.positives = []
        self.negatives = []
        for qi in range(len(self.poses)):
            q_pose = self.poses[qi]
            dises = np.sqrt(np.sum(((q_pose-self.poses)**2)[:,[3,7,11]],axis=1))            
            indexes = np.argsort(dises)

            remap_index = indexes[np.where(dises[indexes]<self.pos_thres)[0]]
            self.positives.append(remap_index)
            self.positives[-1] = self.positives[-1][1:] #exclude query itself

            negs = indexes[np.where(dises[indexes]>self.neg_thres)[0]]
            self.negatives.append(negs)

        self.mining = False
        self.cache = None # filepath of HDF5 containing feature vectors for images



    # refresh cache for hard mining
    def refreshCache(self):
        h5 = h5py.File(self.cache, mode='r')
        self.h5feat = np.array(h5.get("features"))

    def __getitem__(self, index):
        
        if self.mining:
            q_feat = self.h5feat[index]

            pos_feat = self.h5feat[self.positives[index]]
            dis_pos = np.sqrt(np.sum((q_feat.reshape(1,-1)-pos_feat)**2,axis=1))

            min_idx = np.where(dis_pos==np.max(dis_pos))[0][0] 
            pos_idx = np.random.choice(self.positives[index], 1)[0]#
            # pos_idx = self.positives[index][min_idx]

            neg_feat = self.h5feat[self.negatives[index].tolist()]
            dis_neg = np.sqrt(np.sum((q_feat.reshape(1,-1)-neg_feat)**2,axis=1))
            
            dis_loss = (-dis_neg) + 0.3
            dis_inc_index_tmp = dis_loss.argsort()[:-self.num_neg-1:-1]

            neg_idx = self.negatives[index][dis_inc_index_tmp[:self.num_neg]]

              
        else:
            pos_idx = self.positives[index][0]
        
            neg_idx = np.random.choice(np.arange(len(self.negatives[index])).astype(int), self.num_neg)
            neg_idx = self.negatives[index][neg_idx]
        

        
        query = cv2.imread(self.imgs_path[index])
        # rot augmentation
        mat = cv2.getRotationMatrix2D((query.shape[1]//2, query.shape[0]//2 ), np.random.randint(0,360), 1)
        query = cv2.warpAffine(query, mat, query.shape[:2])

        #Adding Noise
        # Generate random noise with the same shape as the image
        noise = np.random.normal(0, 20, query.shape)

        # Add the noise to the image
        query = query + noise

        # Clip the pixel values to the valid range [0, 255]
        query = np.clip(query, 0, 255)
        
        query = query.transpose(2,0,1)

        #cv2.imwrite('./query.png', query[0])

        positive = cv2.imread(join(self.imgs_path[pos_idx]))#           
        mat = cv2.getRotationMatrix2D((positive.shape[1]//2, positive.shape[0]//2 ), np.random.randint(0,360), 1)
        positive = cv2.warpAffine(positive, mat, positive.shape[:2])
        positive = positive.transpose(2,0,1)
        
        #cv2.imwrite('./positive.png', positive[0])

    
        query = (query.astype(np.float32))/256
        positive = (positive.astype(np.float32)/256)

        negatives = []

        for neg_i in neg_idx:
        
            negative = cv2.imread(self.imgs_path[neg_i])
            mat = cv2.getRotationMatrix2D((negative.shape[1]//2, negative.shape[0]//2 ), np.random.randint(0,360), 1)
            negative = cv2.warpAffine(negative, mat, negative.shape[:2]) 
            negative = negative.transpose(2,0,1)
            negative = (negative)/256
            
            negatives.append(torch.from_numpy(negative.astype(np.float32)))

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, index

    def __len__(self):
        return len(self.poses)







class TrainingPointcloudDataset(data.Dataset):
    def __init__(self, dataset_path = 'datasets/KITTI/', seq='00'):
        super().__init__()

        #Pointclouds Path
        pc_p = os.listdir(dataset_path+seq+'velodyne')
        pc_p.sort()
        self.pc_path = [dataset_path+seq+'/velodyne/'+i for i in pc_p]

        #gt_pose, only first 3000 frames of KITTI 00 for training
        self.poses = np.loadtxt(dataset_path+seq+'poses/'+seq+'.txt')
        self.poses = self.poses[:3000]

        #Negative and Positive thresholds
        self.pos_thres = 10
        self.neg_thres = 25

        # Find Positives and Negatives for each query
        self.num_neg = 10
        self.positives = []
        self.negatives = []

        for qi in range(len(self.poses)):
            q_pose = self.pose[qi]
            dises = np.sqrt(np.sum(((q_pose-self.poses)**2)[:,[3,7,11]],axis=1))
            indexes = np.argsort(dises)
            remap_index = indexes[np.where(dises[indexes]<self.pos_thres)[0]]
            self.positives.append(remap_index)
            self.positives[-1] = self.positives[-1][1:] #exclude query itself
            
            negs = indexes[np.where(dises[indexes]>self.neg_thres)[0]]
            self.negatives.append(negs)

        self.mining = False
        self.cache = None # filepath of HDF5 containing feature vectors for images


    # refresh cache for hard mining
    def refreshCache(self):
        h5 = h5py.File(self.cache, mode='r')
        self.h5feat = np.array(h5.get("features"))

    def __getitem__(self, index):
        
        if self.mining:
            q_feat = self.h5feat[index]

            pos_feat = self.h5feat[self.positives[index]]
            dis_pos = np.sqrt(np.sum((q_feat.reshape(1,-1)-pos_feat)**2,axis=1))

            min_idx = np.where(dis_pos==np.max(dis_pos))[0][0] 
            pos_idx = np.random.choice(self.positives[index], 1)[0]#
            # pos_idx = self.positives[index][min_idx]

            neg_feat = self.h5feat[self.negatives[index].tolist()]
            dis_neg = np.sqrt(np.sum((q_feat.reshape(1,-1)-neg_feat)**2,axis=1))
            
            dis_loss = (-dis_neg) + 0.3
            dis_inc_index_tmp = dis_loss.argsort()[:-self.num_neg-1:-1]

            neg_idx = self.negatives[index][dis_inc_index_tmp[:self.num_neg]]

              
        else:
            # Qui prendo sempre il primo della lista dei positivi, non sarebbe meglio prenderne uno random? Forse è implementato nell'Hard Mining
            pos_idx = self.positives[index][0]
        
            neg_idx = np.random.choice(np.arange(len(self.negatives[index])).astype(int), self.num_neg)
            neg_idx = self.negatives[index][neg_idx]
        

        query = pc_tools.load_pointcloud(self.pc_path[index])
        query = pc_tools.filter_z_points(query, z_range=[-1.5,1.2])

        #Adding Noise
        # Generate random noise with the same shape as the pointcloud
        noise = np.random.normal(0, 0.1, query.shape)

        # Add the noise to the pointcloud
        query = query + noise

        
        positive = pc_tools.load_pointcloud(self.pc_path[pos_idx])
        positive = pc_tools.filter_z_points(positive, z_range=[-1.5,1.2])

        negatives = []

        for neg_i in neg_idx:
        
            negative = pc_tools.load_pointcloud(self.pc_path[pos_idx])
            negative = pc_tools.filter_z_points(negative, z_range=[-1.5,1.2])
            
            negatives.append(torch.from_numpy(negative.astype(np.float32)))

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, index

    def __len__(self):
        return len(self.poses)
    






