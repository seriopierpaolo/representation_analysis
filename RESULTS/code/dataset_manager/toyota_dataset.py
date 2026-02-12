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

toyota_split_points = {"00":2500} # NOT USED

#representation = '256'


#Use this boolean only with range and fv representations and if you want padding and resizing
PAD_RESIZE = False

ROTATE = True #Use only with cartesian and polar representation!

import os
import cv2

SAVE_MATCHES = True
SAVE_TOP_K = 2
SAVE_BASE = f'./saved_matches/toyota/00/'
os.makedirs(SAVE_BASE, exist_ok=True)



class InferDataset(data.Dataset,):
    def __init__(self, 
                seq, 
                representation='cartesian_multi',
                dataset_path='/dataset/toyota-dataset/lidar_representations/representation_analysis/',
                poses_path='/dataset/toyota-dataset/lidar_representations/',
                sample_inteval=1):
        super().__init__()

        self.sample_inteval = sample_inteval
        self.db_split_index = int(toyota_split_points[seq]/sample_inteval)
        # bev path
        imgs_p = os.listdir(dataset_path + seq+'/'+ representation +'/')
        imgs_p.sort()
        self.imgs_path = [dataset_path + seq+ '/' + representation + '/' +imgs_p[i] for i in range(0,len(imgs_p), sample_inteval)]

        # gt_pose
        self.poses = np.loadtxt(poses_path  + '/poses.txt')[::sample_inteval]

        self.imgs_path = self.imgs_path[0:10000]
        self.poses = self.poses[0:10000]


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
        



def evaluateResults_save_match(seq, global_descs, dataset, gt_thres=5.0, window=500, start_frame=1000, offset=100):
    """
    Fully-online evaluation WITHOUT using dataset.db_split_index.
    """

    # normalize descriptors
    norms = np.linalg.norm(global_descs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    descs = global_descs / norms

    T = descs.shape[0]
    poses = dataset.poses
    tp_flags = []
    all_gt = 0
    total_search_time = 0.0
    num_searches = 0

    for t in range(start_frame, T):

        start_idx = max(0, t - window - offset)
        end_idx = t - offset

        if end_idx - start_idx < 1:
            continue

        db_slice = descs[start_idx:end_idx]
        q_vec = descs[t].reshape(1, -1)

        # FAISS search
        faiss_index = faiss.IndexFlatL2(db_slice.shape[1])
        faiss_index.add(db_slice)

        t0 = time.perf_counter()
        dists, preds = faiss_index.search(q_vec, SAVE_TOP_K)  # request top-K
        t1 = time.perf_counter()

        total_search_time += (t1 - t0)
        num_searches += 1

        ### --------------------------------------------------------------------
        ### SAVE MATCHES (QUERY + TOP-K RETRIEVED)
        ### --------------------------------------------------------------------
        if SAVE_MATCHES:
            save_dir = os.path.join(SAVE_BASE, f"{t:06d}")
            os.makedirs(save_dir, exist_ok=True)

            # save query image
            q_path = dataset.imgs_path[t]
            q_img = cv2.imread(q_path, cv2.IMREAD_COLOR)
            if q_img is not None:
                cv2.imwrite(os.path.join(save_dir, "query.png"), q_img)

            # save top-k matches
            for rank in range(SAVE_TOP_K):
                pred_local = int(preds[0, rank])
                pred_global = start_idx + pred_local

                match_path = dataset.imgs_path[pred_global]
                match_img = cv2.imread(match_path, cv2.IMREAD_COLOR)

                if match_img is not None:
                    cv2.imwrite(os.path.join(save_dir, f"match_{rank+1}.png"), match_img)
        ### --------------------------------------------------------------------

        # For recall computation we still use ONLY top-1
        pred_local = int(preds[0, 0])
        pred_global = start_idx + pred_local
        dist = float(dists[0, 0])

        # compute ground truth positives inside same window
        pose_q = poses[t]
        pose_db = poses[start_idx:end_idx]
        sq = (pose_db - pose_q) ** 2
        dists_pos = np.sum(sq[:, :3], axis=1)
        positives_local = np.where(dists_pos < (gt_thres ** 2))[0]

        if positives_local.size == 0:
            continue

        all_gt += 1
        positives_global = start_idx + positives_local
        tp_flags.append((int(pred_global in positives_global), dist))

    # no positives
    if len(tp_flags) == 0 or all_gt == 0:
        print("No loop-closure positives found in the chosen window/start settings.")
        return 0.0, 0.0, 0.0

    # timings
    avg_time_per_search = total_search_time / (num_searches + 1e-12)
    print(f"Total FAISS search time: {total_search_time:.4f} s")
    print(f"Average time per query: {avg_time_per_search * 1000:.4f} ms")

    # Recall@1
    tp = sum([f for f, _ in tp_flags])
    recall_at1 = tp / (all_gt + 1e-12)

    # PR Curve
    distances = np.array([d for _, d in tp_flags])
    labels = np.array([f for f, _ in tp_flags])

    d_min, d_max = distances.min(), distances.max()
    thresholds = np.array([d_min]) if d_min == d_max else np.linspace(d_min, d_max, 500)

    precisions, recalls, f1s = [], [], []
    for th in thresholds:
        preds_pos = distances < th
        tp_c = np.sum((preds_pos == 1) & (labels == 1))
        fp_c = np.sum((preds_pos == 1) & (labels == 0))
        fn_c = np.sum((preds_pos == 0) & (labels == 1))

        prec = tp_c / (tp_c + fp_c + 1e-12)
        rec = tp_c / (tp_c + fn_c + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)

    max_f1 = f1s.max()
    order = np.argsort(recalls)
    auc = np.trapz(precisions[order], recalls[order])

    # plot PR
    plt.figure(figsize=(7, 6))
    plt.plot(recalls[order], precisions[order], linewidth=2, label=f'AUC = {auc:.3f}')
    imax = np.argmax(f1s)
    plt.scatter(recalls[imax], precisions[imax], s=80, edgecolor='white',
                label=f'Max F1 = {max_f1:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall — Seq {seq}')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_path = f'./representation_analysis/FMPlace/precision_recall_TOYOTA_seq_{seq}.png'
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved PR curve to: {save_path}")

    return recall_at1, max_f1, auc






class InferDataset(data.Dataset,):
    def __init__(self, 
                seq, 
                representation='cartesian_multi',
                dataset_path='/dataset/toyota-dataset/lidar_representations/representation_analysis/',
                poses_path='/dataset/toyota-dataset/lidar_representations/',
                sample_inteval=1):
        super().__init__()

        self.sample_inteval = sample_inteval
        self.db_split_index = int(toyota_split_points[seq]/sample_inteval)
        # bev path
        imgs_p = os.listdir(dataset_path + seq+'/'+ representation +'/')
        imgs_p.sort()
        self.imgs_path = [dataset_path + seq+ '/' + representation + '/' +imgs_p[i] for i in range(0,len(imgs_p), sample_inteval)]

        # gt_pose
        self.poses = np.loadtxt(poses_path  + '/poses.txt')[::sample_inteval]

        self.imgs_path = self.imgs_path[0:10000]
        self.poses = self.poses[0:10000]


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
        



def evaluateResults(seq, global_descs, dataset, gt_thres=5.0, window=500, start_frame=1000, offset =100):
    """
    Fully-online evaluation WITHOUT using dataset.db_split_index.
    Treats the whole sequence as a timeline. For a query at frame t,
    only descriptors from frames [t-window, t-1] are searchable.

    Args:
        seq: sequence id (for plotting/file names)
        global_descs: (T, D) array of descriptors for every frame in chronological order
        dataset: object with .poses (T, >=3) giving global poses for every frame
        gt_thres: distance threshold (meters) to consider a positive loop
        window: number of previous frames allowed in the searchable buffer
        start_frame: first frame index to begin treating as query (useful to skip initial frames)
    Returns:
        recall_at1_percent, max_f1, auc
    """

    # normalize descriptors (avoid zero-division)
    norms = np.linalg.norm(global_descs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    descs = global_descs / norms

    T = descs.shape[0]
    poses = dataset.poses  
    tp_flags = []
    all_gt = 0
    total_search_time = 0.0
    num_searches = 0

    # iterate through timeline: each t is a potential query (skip start_frame)
    for t in range(start_frame, T):
        start_idx = max(0, t - window - offset)
        end_idx = t - offset  # exclusive; only past frames

        # need at least one candidate in history
        if end_idx - start_idx < 1:
            continue

        db_slice = descs[start_idx:end_idx]  # shape (M, D)
        q_vec = descs[t].reshape(1, -1)

        # build FAISS index for current window (IndexFlatL2 - simple & deterministic)
        faiss_index = faiss.IndexFlatL2(db_slice.shape[1])
        faiss_index.add(db_slice)

        # search top-1 in this window
        t0 = time.perf_counter()
        dists, preds = faiss_index.search(q_vec, 1)
        t1 = time.perf_counter()

        total_search_time += (t1 - t0)
        num_searches += 1


        pred_local = int(preds[0, 0])           # index inside db_slice
        pred_global = start_idx + pred_local    # convert to global frame index
        dist = float(dists[0, 0])

        # compute ground-truth positives *only inside the same window*
        pose_q = poses[t]
        pose_db = poses[start_idx:end_idx]

        sq = (pose_db - pose_q) ** 2
        dists_pos = np.sum(sq[:, :3], axis=1)  # squared Euclidean on xyz
        positives_local = np.where(dists_pos < (gt_thres ** 2))[0]

        if positives_local.size == 0:
            continue

        all_gt += 1
        positives_global = start_idx + positives_local
        tp_flags.append((int(pred_global in positives_global), dist))

    # if no evaluated queries, return zeros
    if len(tp_flags) == 0 or all_gt == 0:
        print("No loop-closure positives found in the chosen window/start settings.")
        return 0.0, 0.0, 0.0

    # timings
    avg_time_per_search = total_search_time / (num_searches + 1e-12)
    print(f"Total FAISS search time: {total_search_time:.4f} s")
    print(f"Average time per query: {avg_time_per_search * 1000:.4f} ms")

    # Recall@1
    tp = sum([f for f, _ in tp_flags])
    recall_at1 = tp / (all_gt + 1e-12)

    # Prepare arrays for PR analysis
    distances = np.array([d for _, d in tp_flags])
    labels    = np.array([f for f, _ in tp_flags])

    # safe thresholding (handle constant distances)
    d_min, d_max = distances.min(), distances.max()
    if d_min == d_max:
        thresholds = np.array([d_min])
    else:
        thresholds = np.linspace(d_min, d_max, 500)

    precisions, recalls, f1s = [], [], []
    for th in thresholds:
        preds_pos = distances < th
        tp_c = np.sum((preds_pos == 1) & (labels == 1))
        fp_c = np.sum((preds_pos == 1) & (labels == 0))
        fn_c = np.sum((preds_pos == 0) & (labels == 1))

        prec = tp_c / (tp_c + fp_c + 1e-12)
        rec  = tp_c / (tp_c + fn_c + 1e-12)
        f1   = 2 * prec * rec / (prec + rec + 1e-12)

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)

    max_f1 = f1s.max()
    # compute AUC (sort by recall)
    order = np.argsort(recalls)
    auc = np.trapz(precisions[order], recalls[order])

    # Plot PR curve
    plt.figure(figsize=(7, 6))
    plt.plot(recalls[order], precisions[order], linewidth=2, label=f'AUC = {auc:.3f}')
    imax = np.argmax(f1s)
    plt.scatter(recalls[imax], precisions[imax], s=80, edgecolor='white',
                label=f'Max F1 = {max_f1:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall — Seq {seq}')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_path = f'./representation_analysis/FMPlace/precision_recall_TOYOTA_seq_{seq}.png'
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved PR curve to: {save_path}")

    #return recall_at1, precisions[order], recalls[order]


    return recall_at1, max_f1, auc




def evaluateResults_pr_only(seq, global_descs, dataset, gt_thres=5.0, window=500, start_frame=1000, offset =100):
    """
    Fully-online evaluation WITHOUT using dataset.db_split_index.
    Treats the whole sequence as a timeline. For a query at frame t,
    only descriptors from frames [t-window, t-1] are searchable.

    Args:
        seq: sequence id (for plotting/file names)
        global_descs: (T, D) array of descriptors for every frame in chronological order
        dataset: object with .poses (T, >=3) giving global poses for every frame
        gt_thres: distance threshold (meters) to consider a positive loop
        window: number of previous frames allowed in the searchable buffer
        start_frame: first frame index to begin treating as query (useful to skip initial frames)
    Returns:
        recall_at1_percent, max_f1, auc
    """

    # normalize descriptors (avoid zero-division)
    norms = np.linalg.norm(global_descs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    descs = global_descs / norms

    T = descs.shape[0]
    poses = dataset.poses  
    tp_flags = []
    all_gt = 0
    total_search_time = 0.0
    num_searches = 0

    # iterate through timeline: each t is a potential query (skip start_frame)
    for t in range(start_frame, T):
        start_idx = max(0, t - window - offset)
        end_idx = t - offset  # exclusive; only past frames

        # need at least one candidate in history
        if end_idx - start_idx < 1:
            continue

        db_slice = descs[start_idx:end_idx]  # shape (M, D)
        q_vec = descs[t].reshape(1, -1)

        # build FAISS index for current window (IndexFlatL2 - simple & deterministic)
        faiss_index = faiss.IndexFlatL2(db_slice.shape[1])
        faiss_index.add(db_slice)

        # search top-1 in this window
        t0 = time.perf_counter()
        dists, preds = faiss_index.search(q_vec, 1)
        t1 = time.perf_counter()

        total_search_time += (t1 - t0)
        num_searches += 1

        pred_local = int(preds[0, 0])           # index inside db_slice
        pred_global = start_idx + pred_local    # convert to global frame index
        dist = float(dists[0, 0])

        # compute ground-truth positives *only inside the same window*
        pose_q = poses[t]
        pose_db = poses[start_idx:end_idx]

        sq = (pose_db - pose_q) ** 2
        dists_pos = np.sum(sq[:, :3], axis=1)  # squared Euclidean on xyz
        positives_local = np.where(dists_pos < (gt_thres ** 2))[0]

        if positives_local.size == 0:
            continue

        all_gt += 1
        positives_global = start_idx + positives_local
        tp_flags.append((int(pred_global in positives_global), dist))

    # if no evaluated queries, return zeros
    if len(tp_flags) == 0 or all_gt == 0:
        print("No loop-closure positives found in the chosen window/start settings.")
        return 0.0, 0.0, 0.0

    # timings
    avg_time_per_search = total_search_time / (num_searches + 1e-12)
    print(f"Total FAISS search time: {total_search_time:.4f} s")
    print(f"Average time per query: {avg_time_per_search * 1000:.4f} ms")

    # Recall@1
    tp = sum([f for f, _ in tp_flags])
    recall_at1 = tp / (all_gt + 1e-12)

    # Prepare arrays for PR analysis
    distances = np.array([d for _, d in tp_flags])
    labels    = np.array([f for f, _ in tp_flags])

    # safe thresholding (handle constant distances)
    d_min, d_max = distances.min(), distances.max()
    if d_min == d_max:
        thresholds = np.array([d_min])
    else:
        thresholds = np.linspace(d_min, d_max, 500)

    precisions, recalls, f1s = [], [], []
    for th in thresholds:
        preds_pos = distances < th
        tp_c = np.sum((preds_pos == 1) & (labels == 1))
        fp_c = np.sum((preds_pos == 1) & (labels == 0))
        fn_c = np.sum((preds_pos == 0) & (labels == 1))

        prec = tp_c / (tp_c + fp_c + 1e-12)
        rec  = tp_c / (tp_c + fn_c + 1e-12)
        f1   = 2 * prec * rec / (prec + rec + 1e-12)

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)

    max_f1 = f1s.max()
    # compute AUC (sort by recall)
    order = np.argsort(recalls)
    auc = np.trapz(precisions[order], recalls[order])

    # Plot PR curve
    plt.figure(figsize=(7, 6))
    plt.plot(recalls[order], precisions[order], linewidth=2, label=f'AUC = {auc:.3f}')
    imax = np.argmax(f1s)
    plt.scatter(recalls[imax], precisions[imax], s=80, edgecolor='white',
                label=f'Max F1 = {max_f1:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall — Seq {seq}')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_path = f'./representation_analysis/FMPlace/precision_recall_TOYOTA_seq_{seq}.png'
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved PR curve to: {save_path}")

    return recall_at1, precisions[order], recalls[order]
