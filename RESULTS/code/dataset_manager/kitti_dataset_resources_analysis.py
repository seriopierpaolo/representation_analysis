import os
from os.path import join, exists

import numpy as np
import cv2
import torch
import torch.utils.data as data

import faiss
import time

import utils.pointcloud_tools as pc_tools
import utils.image_tools as im_tools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


kitti_seq_split_points = {
    "00": 3000,
    "02": 3400,
    "05": 1000,
    "06": 1000,
    "08": 1000,
}

# Use this boolean only with range and front-view representations
# if you want padding and resizing.
PAD_RESIZE = False

# Use only with cartesian and polar representation.
ROTATE = True


# ============================================================
# Dataset
# ============================================================

class InferDataset(data.Dataset):
    def __init__(
        self,
        seq,
        representation,
        dataset_path="/dataset/kitti-dataset/lidar_representations/representation_analysis/",
        sample_inteval=1,
    ):
        super().__init__()

        self.seq = seq
        self.representation = representation
        self.dataset_path = dataset_path
        self.sample_inteval = sample_inteval

        self.db_split_index = int(kitti_seq_split_points[seq] / sample_inteval)

        imgs_dir = dataset_path + "sequences/" + seq + "/" + representation + "/"

        imgs_p = os.listdir(imgs_dir)
        imgs_p.sort()

        self.imgs_path = [
            dataset_path + "sequences/" + seq + "/" + representation + "/" + imgs_p[i]
            for i in range(0, len(imgs_p), sample_inteval)
        ]

        self.poses = np.loadtxt(dataset_path + "poses/" + seq + ".txt")[::sample_inteval]

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index], 0)

        if img is None:
            raise RuntimeError(f"Could not read image: {self.imgs_path[index]}")

        if PAD_RESIZE:
            img = im_tools.pad_resize(img, target_size=(128, 128))

        if 0:
            mat = cv2.getRotationMatrix2D(
                (img.shape[1] // 2, img.shape[0] // 2),
                np.random.randint(0, 360),
                1,
            )
            img = cv2.warpAffine(img, mat, img.shape[:2])

        img = img.astype(np.float32) / 256.0
        img = img[np.newaxis, :, :].repeat(3, 0)

        return img, index

    def __len__(self):
        return len(self.imgs_path)


# ============================================================
# Training collate function, retained for compatibility
# ============================================================

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))

    if len(batch) == 0:
        return None, None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = np.array(query)
    positive = np.array(positive)

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)

    negatives = torch.cat(negatives, 0)
    indices = list(indices)

    return query, positive, negatives, indices


# ============================================================
# Metric helpers
# ============================================================

def bytes_to_mb(x):
    if x is None:
        return None
    return float(x) / (1024.0 ** 2)


def summarize_values(values):
    values = [float(v) for v in values if v is not None]

    if len(values) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "sum": None,
        }

    arr = np.asarray(values, dtype=np.float64)

    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "sum": float(np.sum(arr)),
    }


# ============================================================
# Evaluation
# ============================================================

def evaluateResults(seq, global_descs, dataset, gt_thres=5.0, return_metrics=False):
    """
    Evaluates KITTI retrieval performance and profiles FAISS retrieval.

    Returns:
        recall_at1, max_f1, auc

    If return_metrics=True:
        recall_at1, max_f1, auc, summary_dict

    The summary dictionary contains aggregate-only timing and memory information.
    """

    evaluation_start_wall = time.perf_counter()

    # ------------------------------------------------------------
    # Descriptor normalization
    # ------------------------------------------------------------
    normalize_start = time.perf_counter()

    norms = np.linalg.norm(global_descs, axis=1, keepdims=True)
    global_descs = global_descs / (norms + 1e-12)

    normalize_time = time.perf_counter() - normalize_start

    db_vecs = global_descs[:dataset.db_split_index]
    q_vecs = global_descs[dataset.db_split_index:]

    # ------------------------------------------------------------
    # Build FAISS index
    # ------------------------------------------------------------
    index_build_start = time.perf_counter()

    faiss_index = faiss.IndexFlatL2(db_vecs.shape[1])
    faiss_index.add(db_vecs)

    index_build_time = time.perf_counter() - index_build_start

    # ------------------------------------------------------------
    # FAISS search
    # ------------------------------------------------------------
    search_start = time.perf_counter()

    dists, preds = faiss_index.search(q_vecs, 1)

    search_time = time.perf_counter() - search_start

    avg_time_per_query = search_time / len(q_vecs) if len(q_vecs) > 0 else 0.0

    print(f"Total FAISS search time: {search_time:.4f} s")
    print(f"Average FAISS time per query: {avg_time_per_query * 1000:.4f} ms")

    # ------------------------------------------------------------
    # Ground-truth matching
    # ------------------------------------------------------------
    gt_start = time.perf_counter()

    tp_flags = []
    all_gt = 0
    gt_positive_counts = []

    per_query_gt_time = []

    for q_idx, (dist, pred) in enumerate(zip(dists[:, 0], preds[:, 0])):
        q_gt_start = time.perf_counter()

        query_idx = dataset.db_split_index + q_idx

        gt_dis = (dataset.poses[query_idx] - dataset.poses[:dataset.db_split_index]) ** 2
        positives = np.where(np.sum(gt_dis[:, [3, 7, 11]], axis=1) < gt_thres ** 2)[0]

        per_query_gt_time.append(time.perf_counter() - q_gt_start)
        gt_positive_counts.append(len(positives))

        if len(positives) == 0:
            continue

        all_gt += 1
        tp_flags.append((pred in positives, dist))

    gt_time = time.perf_counter() - gt_start

    # ------------------------------------------------------------
    # Recall@1
    # ------------------------------------------------------------
    tp = sum([f for f, _ in tp_flags])
    recall_at1 = tp / (all_gt + 1e-8)

    # ------------------------------------------------------------
    # Precision-recall analysis
    # ------------------------------------------------------------
    pr_start = time.perf_counter()

    distances = np.array([d for _, d in tp_flags])
    labels = np.array([int(f) for f, _ in tp_flags])

    if len(distances) == 0:
        max_f1 = 0.0
        auc = 0.0
        thresholds = np.array([])
        precisions = []
        recalls = []
        f1s = []
    else:
        thresholds = np.linspace(distances.min(), distances.max(), 1000)

        precisions = []
        recalls = []
        f1s = []

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

        max_f1 = float(np.max(f1s))

        sorted_idx = np.argsort(recalls)
        recalls_sorted = np.array(recalls)[sorted_idx]
        precisions_sorted = np.array(precisions)[sorted_idx]

        auc = float(np.trapz(precisions_sorted, recalls_sorted))

    pr_time = time.perf_counter() - pr_start

    total_evaluation_time = time.perf_counter() - evaluation_start_wall

    summary = {
        "sequence": seq,

        "metrics": {
            "recall_at1": float(recall_at1),
            "max_f1": float(max_f1),
            "auc": float(auc),
            "recall_at1_percent": float(recall_at1 * 100.0),
            "max_f1_percent": float(max_f1 * 100.0),
            "auc_percent": float(auc * 100.0),
        },

        "dataset": {
            "num_descriptors_total": int(global_descs.shape[0]),
            "num_database_descriptors": int(db_vecs.shape[0]),
            "num_query_descriptors": int(q_vecs.shape[0]),
            "descriptor_dimension": int(global_descs.shape[1]),
            "descriptor_dtype": str(global_descs.dtype),
            "descriptor_memory_mb": bytes_to_mb(global_descs.nbytes),
            "database_descriptor_memory_mb": bytes_to_mb(db_vecs.nbytes),
            "query_descriptor_memory_mb": bytes_to_mb(q_vecs.nbytes),
            "db_split_index": int(dataset.db_split_index),
            "gt_threshold_m": float(gt_thres),
        },

        "faiss": {
            "index_type": "IndexFlatL2",
            "index_build_time_seconds": float(index_build_time),
            "index_build_time_ms": float(index_build_time * 1000.0),
            "total_search_time_seconds": float(search_time),
            "total_search_time_ms": float(search_time * 1000.0),
            "search_time_per_query_seconds": float(avg_time_per_query),
            "search_time_per_query_ms": float(avg_time_per_query * 1000.0),
        },

        "evaluation_timing": {
            "descriptor_normalization_time_seconds": float(normalize_time),
            "descriptor_normalization_time_ms": float(normalize_time * 1000.0),

            "ground_truth_matching_total_time_seconds": float(gt_time),
            "ground_truth_matching_total_time_ms": float(gt_time * 1000.0),

            "ground_truth_matching_per_query_time_seconds": summarize_values(
                per_query_gt_time
            ),
            "ground_truth_matching_per_query_time_ms": summarize_values(
                [v * 1000.0 for v in per_query_gt_time]
            ),

            "precision_recall_time_seconds": float(pr_time),
            "precision_recall_time_ms": float(pr_time * 1000.0),

            "total_evaluation_time_seconds": float(total_evaluation_time),
            "total_evaluation_time_ms": float(total_evaluation_time * 1000.0),
        },

        "ground_truth": {
            "num_queries_with_ground_truth": int(all_gt),
            "num_true_positive_top1": int(sum([f for f, _ in tp_flags])),
            "positive_count_per_query": summarize_values(gt_positive_counts),
        },
    }

    if return_metrics:
        return recall_at1, max_f1, auc, summary

    return recall_at1, max_f1, auc


def evaluateResults_pr_only(seq, global_descs, dataset, gt_thres=5.0):
    """
    Kept for compatibility with existing scripts.
    This function returns precision-recall arrays.
    """
    global_descs = global_descs / (
        np.linalg.norm(global_descs, axis=1, keepdims=True) + 1e-12
    )

    db_vecs = global_descs[:dataset.db_split_index]
    q_vecs = global_descs[dataset.db_split_index:]

    faiss_index = faiss.IndexFlatL2(db_vecs.shape[1])
    faiss_index.add(db_vecs)

    start_time = time.perf_counter()
    dists, preds = faiss_index.search(q_vecs, 1)
    end_time = time.perf_counter()

    total_search_time = end_time - start_time
    avg_time_per_query = total_search_time / len(q_vecs)

    print(f"Total FAISS search time: {total_search_time:.4f} s")
    print(f"Average FAISS time per query: {avg_time_per_query * 1000:.4f} ms")

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

    tp = sum([f for f, _ in tp_flags])
    recall_at1 = tp / (all_gt + 1e-8)

    distances = np.array([d for _, d in tp_flags])
    labels = np.array([int(f) for f, _ in tp_flags])

    if len(distances) == 0:
        return recall_at1, np.array([]), np.array([])

    thresholds = np.linspace(distances.min(), distances.max(), 1000)

    precisions = []
    recalls = []
    f1s = []

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

    sorted_idx = np.argsort(recalls)
    recalls_sorted = np.array(recalls)[sorted_idx]
    precisions_sorted = np.array(precisions)[sorted_idx]

    return recall_at1, precisions_sorted, recalls_sorted