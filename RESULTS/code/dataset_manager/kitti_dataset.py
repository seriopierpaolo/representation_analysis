import os
from os.path import join, exists
import numpy as np
import cv2
from imgaug import augmenters as iaa
import torch
import torch.utils.data as data

import h5py


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

    return recall_at1, max_f1, auc













def evaluateResults_pr_only(seq, global_descs, dataset, gt_thres=5.0):
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
    '''
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
    '''
    return recall_at1, precisions_sorted, recalls_sorted





