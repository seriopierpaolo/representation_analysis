import os
import sys
import json
import time
import socket
import platform
import datetime
import traceback
import struct

import numpy as np
import faiss


# ============================================================
# Configuration
# ============================================================

# Folder containing raw HeliLPR point clouds.
#
# Expected structures supported by get_lidar_files():
#
#   HELILPR_RAW_LIDAR_ROOT/
#       rb01/
#           *.bin or *.npy
#       rb02/
#           *.bin or *.npy
#       rb03/
#           *.bin or *.npy
#
# or:
#
#   HELILPR_RAW_LIDAR_ROOT/
#       rb01/velodyne/
#       rb02/velodyne/
#       rb03/velodyne/
#
# or:
#
#   HELILPR_RAW_LIDAR_ROOT/
#       rb01/velodyne_sync/
#       rb02/velodyne_sync/
#       rb03/velodyne_sync/
#
HELILPR_RAW_LIDAR_ROOT = "/dataset/helilpr-dataset/vanilla"

# Existing representation-analysis root used only to load Velodyne_gt.txt.
HELILPR_REPRESENTATION_ROOT = (
    "/dataset/helilpr-dataset/lidar_representations/representation_analysis"
)

REPORT_DIR = "./representation_analysis/temp"
REPORT_PATH = os.path.join(REPORT_DIR, "scan_context_helilpr_resource_analysis.json")

HELILPR_EVAL_SEQUENCES = ["rb01", "rb02", "rb03"]

# Your original dictionary is kept for reference, but your InferDataset overrides
# db_split_index with int((len(imgs_path) * 3) / 4). This script follows the
# actual effective behavior: first 75% database, last 25% query.
HELILPR_SEQ_SPLIT_POINTS = {
    "rb01": 9000,
    "rb02": 7000,
    "rb03": 8000,
}

SAMPLE_INTERVAL = 1
GT_THRESHOLD_M = 5.0

# Scan Context descriptor parameters
NUM_RINGS = 20
NUM_SECTORS = 60
MAX_RADIUS = 80.0

# Height filtering. Adjust if your raw HeliLPR scans use a different frame.
MIN_Z = -5.0
MAX_Z = 5.0

# Coarse-to-fine Scan Context retrieval
USE_YAW_ALIGNMENT = True
NUM_CANDIDATES = 10

# Raw binary point dimension.
# Use 4 for x,y,z,intensity.
# Use 3 for x,y,z.
BIN_POINT_DIM = 4


# ============================================================
# Optional psutil for CPU/RAM profiling
# ============================================================

try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    psutil = None
    PSUTIL_AVAILABLE = False


# ============================================================
# Utility functions
# ============================================================

def bytes_to_mb(x):
    if x is None:
        return None
    return float(x) / (1024.0 ** 2)


def get_process_handle():
    if PSUTIL_AVAILABLE:
        return psutil.Process(os.getpid())
    return None


def get_rss_bytes(proc=None):
    if proc is not None:
        return int(proc.memory_info().rss)
    return None


def get_cpu_time_seconds(proc=None):
    if proc is not None:
        t = proc.cpu_times()
        return float(t.user + t.system)
    return float(time.process_time())


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
# HeliLPR loading
# ============================================================

def get_lidar_files(seq):
    """
    Returns sorted raw LiDAR files for one HeliLPR sequence.

    Edit this function if your HeliLPR raw point clouds use a different folder
    structure or extension.
    """

    candidate_dirs = [
        os.path.join(HELILPR_RAW_LIDAR_ROOT, seq, "Velodyne"),
        os.path.join(HELILPR_RAW_LIDAR_ROOT, seq),
        os.path.join(HELILPR_RAW_LIDAR_ROOT, seq, "velodyne_sync"),
        os.path.join(HELILPR_RAW_LIDAR_ROOT, "sequences", seq),
        os.path.join(HELILPR_RAW_LIDAR_ROOT, "sequences", seq, "velodyne"),
        os.path.join(HELILPR_RAW_LIDAR_ROOT, "sequences", seq, "velodyne_sync"),
    ]

    lidar_dir = None
    for d in candidate_dirs:
        if os.path.isdir(d):
            lidar_dir = d
            break

    if lidar_dir is None:
        raise RuntimeError(
            f"Could not find raw LiDAR directory for sequence {seq}. "
            f"Please edit HELILPR_RAW_LIDAR_ROOT or get_lidar_files()."
        )

    files = sorted([
        os.path.join(lidar_dir, f)
        for f in os.listdir(lidar_dir)
        if f.endswith(".bin") or f.endswith(".npy")
    ])

    if len(files) == 0:
        raise RuntimeError(f"No .bin or .npy LiDAR files found in {lidar_dir}")

    return files[::SAMPLE_INTERVAL]


def load_pointcloud(filename):
    points = []
    with open(filename, "rb") as file:
            while True:
                # 1. Read the exact size of one point (22 bytes)
                data = file.read(22) 
                
                # 2. Check if we read a full point; if not, break the loop
                if len(data) < 22:
                    break
                    
                # 3. UNPACK THE DATA INSIDE THE LOOP
                # Unpack x, y, z, intensity (16 bytes, 'ffff')
                x, y, z, intensity = struct.unpack('ffff', data[:16])
                
                # Unpack ring (2 bytes, 'H') and time (4 bytes, 'f')
                # Note: Since you only append [x, y, z, intensity], the ring and time 
                # variables are unpacked but not used in the final list.
                # ring = struct.unpack('H', data[16:18])[0]
                # time = struct.unpack('f', data[18:22])[0]
                
                # 4. Append the desired point data
                points.append([x, y, z, intensity])
    points = np.array(points, dtype=np.float32)
    return points




def load_poses(seq):
    """
    Loads HeliLPR poses.

    This follows your dataset code:
        dataset_path + seq + '/Velodyne_gt.txt'
    """
    pose_path = os.path.join(
        HELILPR_REPRESENTATION_ROOT,
        seq,
        "Velodyne_gt.txt",
    )

    poses = np.loadtxt(pose_path)[::SAMPLE_INTERVAL]
    return poses


# ============================================================
# Scan Context descriptor
# ============================================================

def make_scan_context(points):
    """
    Builds a Scan Context descriptor.

    Output:
        NUM_RINGS x NUM_SECTORS

    Each bin stores the maximum height value of points inside the corresponding
    radial-angular cell.
    """

    xyz = points[:, :3]

    valid_z = (xyz[:, 2] >= MIN_Z) & (xyz[:, 2] <= MAX_Z)
    xyz = xyz[valid_z]

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    r = np.sqrt(x ** 2 + y ** 2)
    valid_r = (r > 0.1) & (r <= MAX_RADIUS)

    x = x[valid_r]
    y = y[valid_r]
    z = z[valid_r]
    r = r[valid_r]

    theta = np.arctan2(y, x)
    theta = np.mod(theta, 2.0 * np.pi)

    ring_idx = np.floor(r / MAX_RADIUS * NUM_RINGS).astype(np.int32)
    sector_idx = np.floor(theta / (2.0 * np.pi) * NUM_SECTORS).astype(np.int32)

    ring_idx = np.clip(ring_idx, 0, NUM_RINGS - 1)
    sector_idx = np.clip(sector_idx, 0, NUM_SECTORS - 1)

    desc = np.zeros((NUM_RINGS, NUM_SECTORS), dtype=np.float32)

    # Shift z so negative heights are still represented.
    z_shifted = z - MIN_Z

    for ri, si, zi in zip(ring_idx, sector_idx, z_shifted):
        if zi > desc[ri, si]:
            desc[ri, si] = zi

    return desc


def make_ring_key(scan_context):
    """
    Ring key for coarse Scan Context retrieval.

    Shape:
        NUM_RINGS
    """
    return np.mean(scan_context, axis=1).astype(np.float32)


def scan_context_distance(sc1, sc2):
    """
    Yaw-invariant Scan Context distance.

    If USE_YAW_ALIGNMENT is True, the database descriptor is circularly shifted
    over sectors and the best alignment is used.
    """

    if not USE_YAW_ALIGNMENT:
        v1 = sc1.reshape(-1)
        v2 = sc2.reshape(-1)
        denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12
        return 1.0 - float(np.dot(v1, v2) / denom)

    best_dist = float("inf")

    for shift in range(NUM_SECTORS):
        shifted = np.roll(sc2, shift=shift, axis=1)

        sim_sum = 0.0
        valid_cols = 0

        for col in range(NUM_SECTORS):
            v1 = sc1[:, col]
            v2 = shifted[:, col]

            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)

            if n1 < 1e-12 or n2 < 1e-12:
                continue

            sim_sum += float(np.dot(v1, v2) / (n1 * n2 + 1e-12))
            valid_cols += 1

        if valid_cols == 0:
            dist = 1.0
        else:
            dist = 1.0 - sim_sum / valid_cols

        if dist < best_dist:
            best_dist = dist

    return best_dist


# ============================================================
# Descriptor generation
# ============================================================

def extract_scan_context_descriptors(seq):
    """
    Extracts Scan Context descriptors for one HeliLPR sequence.

    Returns:
        descriptors: N x NUM_RINGS x NUM_SECTORS
        ring_keys:   N x NUM_RINGS
        poses:       N x pose_dim
        summary:     aggregate resource dictionary
    """

    proc = get_process_handle()

    lidar_files = get_lidar_files(seq)
    poses = load_poses(seq)

    n = min(len(lidar_files), len(poses))
    lidar_files = lidar_files[:n]
    poses = poses[:n]

    descriptors = []
    ring_keys = []

    wall_times = []
    cpu_times = []
    cpu_utils = []
    ram_deltas = []
    num_points_values = []

    sequence_wall_start = time.perf_counter()
    sequence_cpu_start = get_cpu_time_seconds(proc)
    sequence_rss_start = get_rss_bytes(proc)

    for i, file_path in enumerate(lidar_files):
        rss_before = get_rss_bytes(proc)
        cpu_before = get_cpu_time_seconds(proc)
        wall_start = time.perf_counter()

        points = load_pointcloud(file_path)
        num_points_values.append(points.shape[0])

        sc = make_scan_context(points)
        rk = make_ring_key(sc)

        wall_time = time.perf_counter() - wall_start
        cpu_time = get_cpu_time_seconds(proc) - cpu_before
        rss_after = get_rss_bytes(proc)

        descriptors.append(sc)
        ring_keys.append(rk)

        wall_times.append(wall_time)
        cpu_times.append(cpu_time)

        if wall_time > 0:
            cpu_utils.append(100.0 * cpu_time / wall_time)

        if rss_before is not None and rss_after is not None:
            ram_deltas.append(bytes_to_mb(rss_after - rss_before))

        if (i + 1) % 1000 == 0:
            print(f"[Scan Context][HeliLPR {seq}] processed {i + 1}/{n}")

    descriptors = np.asarray(descriptors, dtype=np.float32)
    ring_keys = np.asarray(ring_keys, dtype=np.float32)

    sequence_wall_time = time.perf_counter() - sequence_wall_start
    sequence_cpu_time = get_cpu_time_seconds(proc) - sequence_cpu_start
    sequence_rss_end = get_rss_bytes(proc)

    generation_time_ms_values = [v * 1000.0 for v in wall_times]
    mean_generation_ms = (
        float(np.mean(generation_time_ms_values))
        if len(generation_time_ms_values) > 0
        else None
    )

    summary = {
        "sequence": seq,
        "num_scans": int(n),

        "descriptor_shape": list(descriptors.shape),
        "descriptor_dtype": str(descriptors.dtype),
        "descriptor_memory_mb": bytes_to_mb(descriptors.nbytes),

        "ring_key_shape": list(ring_keys.shape),
        "ring_key_dtype": str(ring_keys.dtype),
        "ring_key_memory_mb": bytes_to_mb(ring_keys.nbytes),

        "num_points_per_scan": summarize_values(num_points_values),

        "generation_time_per_scan_ms": summarize_values(generation_time_ms_values),
        "generation_cpu_time_per_scan_ms": summarize_values(
            [v * 1000.0 for v in cpu_times]
        ),
        "generation_cpu_utilization_percent": summarize_values(cpu_utils),
        "generation_ram_delta_mb": summarize_values(ram_deltas),

        "generation_fps": (
            float(1000.0 / mean_generation_ms)
            if mean_generation_ms is not None and mean_generation_ms > 0
            else None
        ),

        "sequence_level": {
            "total_generation_wall_time_seconds": float(sequence_wall_time),
            "total_generation_cpu_time_seconds": float(sequence_cpu_time),
            "estimated_cpu_utilization_percent": (
                float(100.0 * sequence_cpu_time / sequence_wall_time)
                if sequence_wall_time > 0
                else None
            ),
            "ram_rss_start_mb": bytes_to_mb(sequence_rss_start),
            "ram_rss_end_mb": bytes_to_mb(sequence_rss_end),
            "ram_rss_delta_mb": (
                bytes_to_mb(sequence_rss_end - sequence_rss_start)
                if sequence_rss_start is not None and sequence_rss_end is not None
                else None
            ),
        },
    }

    return descriptors, ring_keys, poses, summary


# ============================================================
# HeliLPR evaluation
# ============================================================

def evaluate_scan_context_helilpr(seq, descriptors, ring_keys, poses):
    """
    Evaluates Scan Context on one HeliLPR sequence.

    This follows your evaluation protocol:
    - database: first 3/4 of sequence
    - query: last 1/4 of sequence
    - pose distance indices: [1, 2, 3]
    - top-1 retrieval
    """

    db_split_index = int((len(descriptors) * 3) / 4)

    db_descs = descriptors[:db_split_index]
    q_descs = descriptors[db_split_index:]

    db_ring_keys = ring_keys[:db_split_index]
    q_ring_keys = ring_keys[db_split_index:]

    db_poses = poses[:db_split_index]
    q_poses = poses[db_split_index:]

    # ------------------------------------------------------------
    # Build ring-key FAISS index
    # ------------------------------------------------------------
    index_build_start = time.perf_counter()

    faiss_index = faiss.IndexFlatL2(db_ring_keys.shape[1])
    faiss_index.add(db_ring_keys.astype(np.float32))

    index_build_time = time.perf_counter() - index_build_start

    # ------------------------------------------------------------
    # Retrieval: ring-key candidates + yaw-aligned re-ranking
    # ------------------------------------------------------------
    k = min(NUM_CANDIDATES, db_ring_keys.shape[0])

    search_times = []
    rerank_times = []
    total_query_times = []

    pred_indices = []
    pred_distances = []

    retrieval_wall_start = time.perf_counter()

    for q_idx in range(q_ring_keys.shape[0]):
        query_start = time.perf_counter()

        search_start = time.perf_counter()
        _, candidate_indices = faiss_index.search(
            q_ring_keys[q_idx:q_idx + 1].astype(np.float32),
            k,
        )
        search_time = time.perf_counter() - search_start

        rerank_start = time.perf_counter()

        best_idx = None
        best_dist = float("inf")

        for candidate_idx in candidate_indices[0]:
            dist = scan_context_distance(q_descs[q_idx], db_descs[candidate_idx])

            if dist < best_dist:
                best_dist = dist
                best_idx = int(candidate_idx)

        rerank_time = time.perf_counter() - rerank_start
        total_query_time = time.perf_counter() - query_start

        search_times.append(search_time)
        rerank_times.append(rerank_time)
        total_query_times.append(total_query_time)

        pred_indices.append(best_idx)
        pred_distances.append(best_dist)

    retrieval_total_time = time.perf_counter() - retrieval_wall_start

    pred_indices = np.asarray(pred_indices, dtype=np.int32)
    pred_distances = np.asarray(pred_distances, dtype=np.float32)

    # ------------------------------------------------------------
    # Ground-truth matching, same as your HeliLPR evaluateResults()
    # ------------------------------------------------------------
    gt_start = time.perf_counter()

    tp_flags = []
    all_gt = 0
    positive_counts = []

    for q_idx, (dist, pred) in enumerate(zip(pred_distances, pred_indices)):
        query_idx = db_split_index + q_idx
        query_idx = min(query_idx, len(poses) - 1)

        gt_dis = (poses[query_idx] - poses[:db_split_index]) ** 2

        positives = np.where(
            np.sum(gt_dis[:, [1, 2, 3]], axis=1) < GT_THRESHOLD_M ** 2
        )[0]

        positive_counts.append(len(positives))

        if len(positives) == 0:
            continue

        all_gt += 1
        tp_flags.append((pred in positives, dist))

    gt_time = time.perf_counter() - gt_start

    # ------------------------------------------------------------
    # Recall@1, F1, AUC
    # ------------------------------------------------------------
    if len(tp_flags) == 0:
        recall_at1 = 0.0
        max_f1 = 0.0
        auc = 0.0
        pr_time = 0.0

    else:
        tp = sum([f for f, _ in tp_flags])
        recall_at1 = float(tp / (all_gt + 1e-8))

        labels = np.array([1 if f else 0 for f, _ in tp_flags], dtype=int)
        distances = np.array([d for _, d in tp_flags], dtype=float)

        pr_start = time.perf_counter()

        if len(distances) == 0:
            max_f1 = 0.0
            auc = 0.0
        else:
            thresholds = np.linspace(distances.min(), distances.max(), 1000)

            precisions = []
            recalls = []
            f1s = []

            total_pos = labels.sum()

            for t in thresholds:
                preds_pos = distances < t

                tp = np.sum((preds_pos == 1) & (labels == 1))
                fp = np.sum((preds_pos == 1) & (labels == 0))
                fn = total_pos - tp

                prec = tp / (tp + fp + 1e-12)
                rec = tp / (tp + fn + 1e-12)
                f1 = 2.0 * prec * rec / (prec + rec + 1e-12)

                precisions.append(prec)
                recalls.append(rec)
                f1s.append(f1)

            max_f1 = float(np.max(f1s))

            recalls = np.array(recalls)
            precisions = np.array(precisions)
            sort_idx = np.argsort(recalls)

            auc = float(np.trapz(precisions[sort_idx], recalls[sort_idx]))

        pr_time = time.perf_counter() - pr_start

    mean_total_query_ms = (
        float(np.mean([v * 1000.0 for v in total_query_times]))
        if len(total_query_times) > 0
        else None
    )

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
            "num_descriptors_total": int(descriptors.shape[0]),
            "num_database_descriptors": int(db_descs.shape[0]),
            "num_query_descriptors": int(q_descs.shape[0]),
            "descriptor_shape": [NUM_RINGS, NUM_SECTORS],
            "descriptor_dimension": int(NUM_RINGS * NUM_SECTORS),
            "descriptor_dtype": str(descriptors.dtype),
            "descriptor_memory_mb": bytes_to_mb(descriptors.nbytes),
            "database_descriptor_memory_mb": bytes_to_mb(db_descs.nbytes),
            "query_descriptor_memory_mb": bytes_to_mb(q_descs.nbytes),
            "db_split_index": int(db_split_index),
            "gt_threshold_m": float(GT_THRESHOLD_M),
        },

        "scan_context": {
            "num_rings": int(NUM_RINGS),
            "num_sectors": int(NUM_SECTORS),
            "max_radius_m": float(MAX_RADIUS),
            "min_z": float(MIN_Z),
            "max_z": float(MAX_Z),
            "use_yaw_alignment": bool(USE_YAW_ALIGNMENT),
            "num_candidates": int(NUM_CANDIDATES),
        },

        "retrieval_timing": {
            "ring_key_index_build_time_ms": float(index_build_time * 1000.0),

            "ring_key_search_time_per_query_ms": summarize_values(
                [v * 1000.0 for v in search_times]
            ),
            "rerank_time_per_query_ms": summarize_values(
                [v * 1000.0 for v in rerank_times]
            ),
            "total_retrieval_time_per_query_ms": summarize_values(
                [v * 1000.0 for v in total_query_times]
            ),

            "total_ring_key_search_time_ms": float(np.sum(search_times) * 1000.0),
            "total_rerank_time_ms": float(np.sum(rerank_times) * 1000.0),
            "total_retrieval_time_ms": float(np.sum(total_query_times) * 1000.0),
            "retrieval_total_wall_time_seconds": float(retrieval_total_time),
        },

        "evaluation_timing": {
            "ground_truth_matching_time_ms": float(gt_time * 1000.0),
            "precision_recall_time_ms": float(pr_time * 1000.0),
        },

        "ground_truth": {
            "num_queries_with_ground_truth": int(all_gt),
            "num_true_positive_top1": int(sum([f for f, _ in tp_flags])),
            "positive_count_per_query": summarize_values(positive_counts),
        },

        "retrieval_fps": (
            float(1000.0 / mean_total_query_ms)
            if mean_total_query_ms is not None and mean_total_query_ms > 0
            else None
        ),
    }

    return summary


# ============================================================
# Overall summary
# ============================================================

def build_overall_summary(generation_reports, evaluation_reports):
    generation_ms = []
    descriptor_mem = []

    recall_values = []
    f1_values = []
    auc_values = []
    retrieval_ms = []

    total_scans = 0
    total_queries = 0

    for report in generation_reports:
        generation_ms.append(report["generation_time_per_scan_ms"]["mean"])
        descriptor_mem.append(report["descriptor_memory_mb"])
        total_scans += report["num_scans"]

    for report in evaluation_reports:
        recall_values.append(report["metrics"]["recall_at1"])
        f1_values.append(report["metrics"]["max_f1"])
        auc_values.append(report["metrics"]["auc"])
        retrieval_ms.append(report["retrieval_timing"]["total_retrieval_time_per_query_ms"]["mean"])
        total_queries += report["dataset"]["num_query_descriptors"]

    return {
        "num_sequences": int(len(evaluation_reports)),
        "total_scans": int(total_scans),
        "total_queries": int(total_queries),

        "generation_time_per_scan_ms": summarize_values(generation_ms),
        "descriptor_memory_mb": summarize_values(descriptor_mem),
        "retrieval_time_per_query_ms": summarize_values(retrieval_ms),

        "retrieval_metrics": {
            "recall_at1": summarize_values(recall_values),
            "max_f1": summarize_values(f1_values),
            "auc": summarize_values(auc_values),
        },
    }


def get_environment_metadata():
    metadata = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "hostname": socket.gethostname(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "psutil_available": PSUTIL_AVAILABLE,

        "configuration": {
            "dataset": "HeliLPR",
            "raw_lidar_root": HELILPR_RAW_LIDAR_ROOT,
            "representation_root": HELILPR_REPRESENTATION_ROOT,
            "eval_sequences": HELILPR_EVAL_SEQUENCES,
            "sample_interval": SAMPLE_INTERVAL,
            "gt_threshold_m": GT_THRESHOLD_M,

            "num_rings": NUM_RINGS,
            "num_sectors": NUM_SECTORS,
            "max_radius_m": MAX_RADIUS,
            "min_z": MIN_Z,
            "max_z": MAX_Z,
            "use_yaw_alignment": USE_YAW_ALIGNMENT,
            "num_candidates": NUM_CANDIDATES,

            "bin_point_dim": BIN_POINT_DIM,
            "report_path": REPORT_PATH,
        },

        "metric_definitions": {
            "generation_time_per_scan_ms": (
                "Time to load a raw HeliLPR point cloud and compute its Scan Context descriptor."
            ),
            "ring_key_search_time_per_query_ms": (
                "FAISS search time using Scan Context ring keys."
            ),
            "rerank_time_per_query_ms": (
                "Yaw-aligned Scan Context candidate re-ranking time."
            ),
            "total_retrieval_time_per_query_ms": (
                "Ring-key search plus yaw-aligned re-ranking."
            ),
            "descriptor_memory_mb": (
                "Memory occupied by Scan Context descriptors."
            ),
            "recall_at1": (
                "Top-1 place-recognition recall over queries that have at least one ground-truth positive."
            ),
        },
    }

    if PSUTIL_AVAILABLE:
        vm = psutil.virtual_memory()
        metadata["system_memory"] = {
            "total_bytes": int(vm.total),
            "total_mb": bytes_to_mb(vm.total),
            "available_bytes": int(vm.available),
            "available_mb": bytes_to_mb(vm.available),
        }

    return metadata


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 80)
    print("Scan Context HeliLPR baseline")
    print(f"Sequences: {HELILPR_EVAL_SEQUENCES}")
    print(f"Sample interval: {SAMPLE_INTERVAL}")
    print(f"Descriptor: {NUM_RINGS} rings x {NUM_SECTORS} sectors")
    print(f"Yaw alignment: {USE_YAW_ALIGNMENT}")
    print(f"Candidates: {NUM_CANDIDATES}")
    print(f"Report path: {REPORT_PATH}")
    print("=" * 80)

    experiment_start = time.perf_counter()

    generation_reports = []
    evaluation_reports = []
    failed_sequences = []

    for seq in HELILPR_EVAL_SEQUENCES:
        try:
            print("=" * 80)
            print(f"Processing HeliLPR sequence {seq}")
            print("=" * 80)

            descriptors, ring_keys, poses, generation_summary = extract_scan_context_descriptors(seq)

            evaluation_summary = evaluate_scan_context_helilpr(
                seq=seq,
                descriptors=descriptors,
                ring_keys=ring_keys,
                poses=poses,
            )

            generation_reports.append(generation_summary)
            evaluation_reports.append(evaluation_summary)

            print(
                f"[Scan Context][HeliLPR {seq}] "
                f"Recall@1: {evaluation_summary['metrics']['recall_at1_percent']:.2f}% | "
                f"F1: {evaluation_summary['metrics']['max_f1_percent']:.2f}% | "
                f"AUC: {evaluation_summary['metrics']['auc_percent']:.2f}% | "
                f"Retrieval: {evaluation_summary['retrieval_timing']['total_retrieval_time_per_query_ms']['mean']:.3f} ms/query"
            )

        except Exception as exc:
            failed_sequences.append({
                "sequence": seq,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            })

            print(f"[FAILED][{seq}] {type(exc).__name__}: {exc}")

    experiment_time = time.perf_counter() - experiment_start

    report = {
        "report_type": "scan_context_helilpr_resource_analysis",
        "environment": get_environment_metadata(),
        "summary": {
            "total_experiment_wall_time_seconds": float(experiment_time),
            "num_sequences_requested": int(len(HELILPR_EVAL_SEQUENCES)),
            "num_sequences_successful": int(len(evaluation_reports)),
            "num_sequences_failed": int(len(failed_sequences)),
            "failed_sequences": failed_sequences,
            "overall_helilpr": (
                build_overall_summary(generation_reports, evaluation_reports)
                if len(evaluation_reports) > 0
                else None
            ),
        },
        "descriptor_generation": generation_reports,
        "sequence_evaluation": evaluation_reports,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 80)
    print("Scan Context HeliLPR baseline complete.")
    print(f"Report written to: {REPORT_PATH}")
    print("=" * 80)