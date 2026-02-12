# nclt_dataset_and_eval_optimized.py
import os
import numpy as np
import cv2
import torch
import torch.utils.data as data
import faiss
import matplotlib.pyplot as plt

import time

#TODO: abbassare numero di sweep per calcolo della precisione a 500



#REPRESENTATION = '256'
SPLIT_POINTS = {"2012-01-08": 4000, "2012-01-15": 4000, "2012-02-04": 4000}

class InferDataset(data.Dataset):
    def __init__(self, seq, representation, root='/dataset/nclt-dataset/lidar_representations/representation_analysis'):
        super().__init__()
        img_dir = os.path.join(root, seq, representation)

        img_p = os.listdir(img_dir)
        img_p.sort()
        sample_interval = 2
        self.imgs_path = [os.path.join(img_dir, img_p[f]) for f in range(0,len(img_p),sample_interval)]

        self.poses = np.loadtxt(os.path.join(root, 'poses', f'{seq}.txt'))[::sample_interval]

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs_path[idx], 0).astype(np.float32) / 256
        return np.repeat(img[None], 3, 0), idx

    def __len__(self): return len(self.imgs_path)





def evaluateResults(seq, global_descs, datasets, gt_thres=5.0):

    #os.makedirs(out_dir, exist_ok=True)

    # --- Normalize DB descriptors ---
    db_vecs = global_descs[0]
    db_vecs = db_vecs / np.linalg.norm(db_vecs, axis=1, keepdims=True)
    db_poses = datasets[0].poses

    # --- Build FAISS DB ---
    faiss_index = faiss.IndexFlatL2(db_vecs.shape[1])
    faiss_index.add(db_vecs)

    recall_at1 = 0
    max_f1_score = 0
    auc = 0



    for q_vecs, q_data in zip(global_descs[1:], datasets[1:]):

        # --- Normalize query descriptors ---
        q_vecs = q_vecs / np.linalg.norm(q_vecs, axis=1, keepdims=True)

        # --- FAISS search timing ---
        start_time = time.perf_counter()
        dists, preds = faiss_index.search(q_vecs, 1)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_time = total_time / len(q_vecs)

        print(f"[Seq {seq}] Search time {total_time:.4f}s "
              f"({avg_time*1000:.4f} ms/query)")

        # -------------------------------
        #   Ground-truth + TP collection
        # -------------------------------
        tp_flags = []
        all_gt = 0

        

        for q_idx, (dist, pred) in enumerate(zip(dists[:, 0], preds[:, 0])):
            # 2D distance using pose indices [4, 8]
            gt_dis = np.sum((q_data.poses[q_idx, [4, 8]] -
                             db_poses[:, [4, 8]]) ** 2, axis=1)
            positives = np.where(gt_dis < gt_thres ** 2)[0]

            if len(positives) == 0:
                continue

            all_gt += 1
            tp_flags.append((pred in positives, dist))

        if all_gt == 0:
            print("No GT matches found → skipping PR computation.")
            continue

        # Extract arrays
        labels = np.array([int(f) for f, _ in tp_flags])
        distances = np.array([d for _, d in tp_flags])

        # -------------------------------
        #   Recall@1
        # -------------------------------
        recall_at1 = labels.mean()

        # -------------------------------
        #   Precision–Recall computation
        #   (identical to second code)
        # -------------------------------
        thresholds = np.linspace(distances.min(), distances.max(), 1000)

        precisions, recalls, f1s = [], [], []

        for t in thresholds:
            preds_pos = distances < t

            tp = np.sum((preds_pos == 1) & (labels == 1))
            fp = np.sum((preds_pos == 1) & (labels == 0))
            fn = np.sum((preds_pos == 0) & (labels == 1))

            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        precisions = np.array(precisions)
        recalls = np.array(recalls)
        f1s = np.array(f1s)

        max_f1_idx = np.argmax(f1s)
        max_f1_score = f1s[max_f1_idx]

        # --- Sort by recall for AUC calculation ---
        sort_idx = np.argsort(recalls)
        recalls_sorted = recalls[sort_idx]
        precisions_sorted = precisions[sort_idx]
        auc = np.trapz(precisions_sorted, recalls_sorted)


    return recall_at1, max_f1_score, auc





def evaluateResults_pr_only(seq, global_descs, datasets, gt_thres=5.0):

    #os.makedirs(out_dir, exist_ok=True)

    # --- Normalize DB descriptors ---
    db_vecs = global_descs[0]
    db_vecs = db_vecs / np.linalg.norm(db_vecs, axis=1, keepdims=True)
    db_poses = datasets[0].poses

    # --- Build FAISS DB ---
    faiss_index = faiss.IndexFlatL2(db_vecs.shape[1])
    faiss_index.add(db_vecs)

    recall_at1 = 0
    max_f1_score = 0
    auc = 0



    for q_vecs, q_data in zip(global_descs[1:], datasets[1:]):

        # --- Normalize query descriptors ---
        q_vecs = q_vecs / np.linalg.norm(q_vecs, axis=1, keepdims=True)

        # --- FAISS search timing ---
        start_time = time.perf_counter()
        dists, preds = faiss_index.search(q_vecs, 1)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_time = total_time / len(q_vecs)

        print(f"[Seq {seq}] Search time {total_time:.4f}s "
              f"({avg_time*1000:.4f} ms/query)")

        # -------------------------------
        #   Ground-truth + TP collection
        # -------------------------------
        tp_flags = []
        all_gt = 0

        

        for q_idx, (dist, pred) in enumerate(zip(dists[:, 0], preds[:, 0])):
            # 2D distance using pose indices [4, 8]
            gt_dis = np.sum((q_data.poses[q_idx, [4, 8]] -
                             db_poses[:, [4, 8]]) ** 2, axis=1)
            positives = np.where(gt_dis < gt_thres ** 2)[0]

            if len(positives) == 0:
                continue

            all_gt += 1
            tp_flags.append((pred in positives, dist))

        if all_gt == 0:
            print("No GT matches found → skipping PR computation.")
            continue

        # Extract arrays
        labels = np.array([int(f) for f, _ in tp_flags])
        distances = np.array([d for _, d in tp_flags])

        # -------------------------------
        #   Recall@1
        # -------------------------------
        recall_at1 = labels.mean()

        # -------------------------------
        #   Precision–Recall computation
        #   (identical to second code)
        # -------------------------------
        thresholds = np.linspace(distances.min(), distances.max(), 1000)

        precisions, recalls, f1s = [], [], []

        for t in thresholds:
            preds_pos = distances < t

            tp = np.sum((preds_pos == 1) & (labels == 1))
            fp = np.sum((preds_pos == 1) & (labels == 0))
            fn = np.sum((preds_pos == 0) & (labels == 1))

            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        precisions = np.array(precisions)
        recalls = np.array(recalls)
        f1s = np.array(f1s)

        max_f1_idx = np.argmax(f1s)
        max_f1_score = f1s[max_f1_idx]

        # --- Sort by recall for AUC calculation ---
        sort_idx = np.argsort(recalls)
        recalls_sorted = recalls[sort_idx]
        precisions_sorted = precisions[sort_idx]
        


    return recall_at1, precisions_sorted, recalls_sorted

