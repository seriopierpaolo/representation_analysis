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



REPRESENTATION = '256'
SPLIT_POINTS = {"2012-01-08": 4000, "2012-01-15": 4000, "2012-02-04": 4000}

class InferDataset(data.Dataset):
    def __init__(self, seq, root='/dataset/nclt-dataset/lidar_representations/'):
        super().__init__()
        img_dir = os.path.join(root, 'bev', seq, REPRESENTATION)
        self.imgs_path = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
        self.poses = np.loadtxt(os.path.join(root, 'poses', f'{seq}.txt'))

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs_path[idx], 0).astype(np.float32) / 256
        return np.repeat(img[None], 3, 0), idx

    def __len__(self): return len(self.imgs_path)





def evaluateResults(seq, global_descs, datasets, gt_thres=5.0,
                    out_dir="./representation_analysis/FMPlace/"):

    os.makedirs(out_dir, exist_ok=True)

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

        # -------------------------------
        #   PR Curve (matching HELI style)
        # -------------------------------
        plt.figure(figsize=(7, 6))
        plt.plot(recalls_sorted, precisions_sorted,
                 color="royalblue", linewidth=2,
                 label=f"PR Curve (AUC = {auc:.3f})")

        # Mark max F1 point
        plt.scatter(recalls[max_f1_idx],
                    precisions[max_f1_idx],
                    color="crimson", s=80,
                    edgecolor="white", zorder=5,
                    label=f"Max F1 = {max_f1_score:.3f}")

        plt.title(f"Precision–Recall Curve — Seq {seq}", fontsize=13, pad=10)
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.grid(alpha=0.3)
        plt.legend(frameon=True, loc="lower left", fontsize=10)
        plt.tight_layout()

        save_path = os.path.join(
            out_dir, f"precision_recall_NCLT_seq_{seq}.png"
        )
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"[Seq {seq}] PR curve saved → {save_path}")

    return recall_at1, max_f1_score, auc



'''
def evaluateResults(seq, global_descs, datasets, gt_thres=5.0, out_dir="dimension_analysis/FMPlace/"):
    os.makedirs(out_dir, exist_ok=True)
    db_vecs = global_descs[0] / np.linalg.norm(global_descs[0], axis=1, keepdims=True)
    db_poses = datasets[0].poses

    faiss_index = faiss.IndexFlatL2(db_vecs.shape[1])
    faiss_index.add(db_vecs)
    recalls = []
    recall_at1 = 0
    f1s = [] # max f1 score for the last dataset
    auc = 0.0 # auc for the last dataset
    max_f1_score = 0.0 # Add a dedicated variable for the F1 score

    total_search_times = []  # store timing stats per query dataset

    for q_vecs, q_data in zip(global_descs[1:], datasets[1:]):
        q_vecs = q_vecs / np.linalg.norm(q_vecs, axis=1, keepdims=True)
        
        # --- Time FAISS search ---
        start_time = time.perf_counter()
        dists, preds = faiss_index.search(q_vecs, 1)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_time_per_query = total_time / len(q_vecs)
        total_search_times.append(avg_time_per_query)

        print(f"Search time for {len(q_vecs)} queries: {total_time:.4f}s "
              f"({avg_time_per_query*1000:.4f} ms/query)")

        tp_flags, all_gt = [], 0
        for q_idx, (dist, pred) in enumerate(zip(dists[:, 0], preds[:, 0])):
            gt_dis = np.sum((q_data.poses[q_idx, [4, 8]] - db_poses[:, [4, 8]]) ** 2, axis=1)
            positives = np.where(gt_dis < gt_thres ** 2)[0]
            if positives.size:
                all_gt += 1
                tp_flags.append((pred in positives, dist))

        if not all_gt:
            recalls.append(0.0)
            continue

        labels, distances = np.array([f for f, _ in tp_flags]), np.array([d for _, d in tp_flags])
        recall_at1 = labels.mean()
        thresholds = np.linspace(distances.min(), distances.max(), 500)

        precs, recs = [], []
        for t in thresholds:
            preds_pos = distances < t
            tp = np.sum(preds_pos & labels)
            fp = np.sum(preds_pos & ~labels)
            fn = np.sum(~preds_pos & labels)
            precs.append(tp / (tp + fp + 1e-8))
            recs.append(tp / (tp + fn + 1e-8))

        f1s = 2 * np.array(precs) * np.array(recs) / (np.array(precs) + np.array(recs) + 1e-8)
        auc = np.trapz(precs, recs)
        max_f1_idx = np.argmax(f1s)
        max_f1_score = f1s[max_f1_idx]
        
        plt.figure(figsize=(7, 6))
        plt.plot(recs, precs, 'b-', lw=2, label=f'AUC={auc:.3f}')
        plt.scatter(recs[max_f1_idx], precs[max_f1_idx], c='r', s=60, edgecolor='white', label=f'Max F1={f1s[max_f1_idx]:.3f}')
        plt.title(f'PR Curve — NCLT {seq}', fontsize=13)
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.xlim(0, 1); plt.ylim(0, 1.05); plt.grid(alpha=0.3); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'PR_curve_{seq}.png'), dpi=300)
        plt.close()

        print(f"Saved PR curve for {seq} → {out_dir}")

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
        save_path = f'./representation_analysis/FMPlace/precision_recall_NCLT_seq_{seq}.png'
        plt.savefig(save_path, dpi=300)
        plt.close()
        
   
    return recall_at1, max_f1_score, auc
'''


#IL CODICE SOPRA VALUTA PIU' SEQUENZE INSIEME, USANDO LA PRIMA COME QUERY E TUTTE LE ALTRE COME DATASET.
#IL CODICE SOTTO VALUTA UNA SOLA SEQUENZA ALLA VOLTA CON SPLIT FISSO DB/QUERY.



# nclt_dataset_and_eval.py
'''
import os
from os.path import join, exists
import numpy as np
import cv2
import torch
import torch.utils.data as data
import faiss
import matplotlib.pyplot as plt

representation = '256'
nclt_split_points = {"2012-01-08": 1500, "2012-01-15": 2000, "2012-02-04": 2500} 

class InferDataset(data.Dataset):
    """
    NCLT inference dataset that supports splitting a sequence into DB / Query
    using an explicit split_points dict (frames), or a split_ratio (fraction).
    The dataset exposes `db_split_index` to be used by evaluateResults.
    """

    def __init__(self, seq, dataset_path='/dataset/nclt-dataset/lidar_representations/', split_points=None, sample_interval=1):
        """
        - seq: sequence name (folder)
        - dataset_path: root path for NCLT repo (contains 'bev/<seq>/<representation>/')
        - split_points: optional dict mapping sequence name -> number_of_db_frames
        - split_ratio: fallback ratio to split sequence (db = int(len * split_ratio))
        - sample_interval: subsampling interval (like in your KITTI code)
        """
        super().__init__()
        self.representation = representation
        self.seq = seq
        self.dataset_path = dataset_path
        self.sample_interval = sample_interval

        self.db_split_index = int(nclt_split_points[seq]/sample_interval)

        imgs_dir = join(dataset_path, 'bev', seq, representation)
        imgs_p = sorted(os.listdir(imgs_dir))
        imgs_p = imgs_p[::sample_interval]

        poses_path = join(dataset_path, 'poses', f'{seq}.txt')
        self.poses = np.loadtxt(poses_path)[::sample_interval]

        # Align images and poses by truncating to the shortest length
        n_pose = len(self.poses)
        n_img = len(imgs_p)
        if n_img != n_pose:
            n = min(n_img, n_pose)
            imgs_p = imgs_p[:n]
            self.poses = self.poses[:n]
            print(f"NCLT sequence {seq}: truncated to {n} samples (poses={n_pose}, imgs={n_img}) to ensure 1:1 alignment.")

        # build final image paths AFTER trimming
        self.imgs_path = [join(imgs_dir, i) for i in imgs_p]

        # compute db split index
        if split_points and seq in split_points:
            self.db_split_index = int(split_points[seq] / max(1, sample_interval))
            # clamp
            self.db_split_index = min(self.db_split_index, len(self.imgs_path))


        # safety
        if self.db_split_index <= 0:
            raise ValueError("db_split_index computed as 0; pick larger split_ratio or check split_points.")
        if self.db_split_index >= len(self.imgs_path):
            # Make sure there's at least one query
            self.db_split_index = max(1, len(self.imgs_path) - 1)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index], 0)
        img = (img.astype(np.float32)) / 256.0
        img = img[np.newaxis, :, :].repeat(3, axis=0)
        return img, index


def evaluateResults(seq, global_descs, dataset,
                          gt_thres=5.0,
                          pose_dims=(4, 8, 12),
                          output_dir="./dimension_analysis/FMPlace/",
                          plot_prefix='nclt'):
    """
    Evaluate NCLT-like cross-sequence retrieval using a single DB (datasets_list[db_index]),
    and every other sequence as queries.

    - global_descs_list: list of numpy arrays, one per sequence (shape N_i x D)
    - datasets_list: list of dataset objects (each must have .poses and optionally .db_split_index)
    - db_index: index in lists that is the database sequence (default 0)
    - pose_dims: tuple/list of indices to select pose coordinates for distance computation (configurable)
    - returns: list of recall@1 for each query sequence (in same order as sequences excluding db)
    """

    assert len(global_descs) == len(dataset), "Descriptors and datasets length mismatch."

    #print("Dataset length:", len(dataset))
    #print("Poses shape:", dataset.poses.shape)
    #print("First 5 timestamps from poses:")
    #print(dataset.poses[:5, 0])
    #print("First 5 image timestamps or filenames:")
    #print([dataset.imgs[i][0] for i in range(5)])  # adjust if your dataset stores (img, timestamp)

    

    # normalize  vectors
    global_descs = global_descs / np.linalg.norm(global_descs, axis=1, keepdims=True)

    db_vecs = global_descs[:dataset.db_split_index]
    q_vecs  = global_descs[dataset.db_split_index:]
    #for i, (desc, ds) in enumerate(zip(global_descs_list, datasets_list)):
    #    print(f"[Seq {i}] descs: {desc.shape[0]}  poses: {len(ds.poses)}  db_split_index: {getattr(ds, 'db_split_index', 'N/A')}")



    # Build FAISS once
    #d = db_vecs
    faiss_index = faiss.IndexFlatL2(db_vecs.shape[1])
    faiss_index.add(db_vecs)
    dists, preds = faiss_index.search(q_vecs, 1)  # top-1 distances


    N = global_descs.shape[0]
    assert 0 < dataset.db_split_index < N, f"bad db_split_index: {dataset.db_split_index} / {N}"

    print(preds.max() < dataset.db_split_index)


    recalls = []


    # ensure output dir exists
    if not exists(output_dir):
        os.makedirs(output_dir)

    # Precompute db and query poses
    dataset_poses = dataset.poses

    # Select only the translation components: X=4, Y=8, Z=12
    pose_dims = (4, 8, 12)

    db_poses = dataset_poses[:dataset.db_split_index, pose_dims]
    q_poses  = dataset_poses[dataset.db_split_index:, pose_dims]





    # --- Compute ground-truth matches ---
    tp_flags = []
    all_gt = 0

    for q_idx, (dist, pred) in enumerate(zip(dists[:, 0], preds[:, 0])):
        gt_dis = np.sum((q_poses[q_idx] - db_poses) ** 2, axis=1)
        positives = np.where(gt_dis < gt_thres ** 2)[0]
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
    save_path = f'./dimension_analysis/FMPlace/precision_recall_curve_seq_{seq}.png'
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Precision-Recall curve saved to: {save_path}")

    

    return recall_at1 * 100, max_f1, auc

'''