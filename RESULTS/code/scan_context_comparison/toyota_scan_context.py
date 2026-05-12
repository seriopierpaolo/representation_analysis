import os
import sys
import json
import time
import socket
import platform
import datetime
import traceback

import numpy as np
import faiss


# ============================================================
# Configuration
# ============================================================

# Folder containing raw Toyota point clouds.
#
# Supported structures in get_lidar_files():
#
#   TOYOTA_RAW_LIDAR_ROOT/
#       00/
#           *.bin or *.npy
#
# or:
#
#   TOYOTA_RAW_LIDAR_ROOT/
#       00/velodyne/
#
# or:
#
#   TOYOTA_RAW_LIDAR_ROOT/
#       sequences/00/velodyne/
#
TOYOTA_RAW_LIDAR_ROOT = "/dataset/toyota-dataset/vanilla"

# Existing folder used by your Toyota code to load poses.txt.
TOYOTA_POSES_ROOT = "/dataset/toyota-dataset/lidar_representations"

REPORT_DIR = "./representation_analysis/temp"
REPORT_PATH = os.path.join(REPORT_DIR, "scan_context_toyota_resource_analysis.json")

TOYOTA_SEQUENCE = "00"

SAMPLE_INTERVAL = 1
MAX_FRAMES = 10000

# Same online evaluation settings as your Toyota code
GT_THRESHOLD_M = 5.0
WINDOW = 1000
START_FRAME = 1000
OFFSET = 100
PR_NUM_THRESHOLDS = 500

# Scan Context descriptor parameters
NUM_RINGS = 20
NUM_SECTORS = 60
MAX_RADIUS = 80.0

# Height filtering. Adjust if your Toyota raw scans use a different frame.
MIN_Z = -5.0
MAX_Z = 5.0

# Online retrieval
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
# Toyota loading
# ============================================================

def get_lidar_files(seq):
    """
    Returns sorted raw LiDAR files for Toyota sequence 00.

    Edit this function if your Toyota raw point clouds use a different folder
    layout or extension.
    """

    candidate_dirs = [
        os.path.join(TOYOTA_RAW_LIDAR_ROOT, seq, "pcs"),
        os.path.join(TOYOTA_RAW_LIDAR_ROOT, seq, "velodyne"),
        os.path.join(TOYOTA_RAW_LIDAR_ROOT, seq, "velodyne_sync"),
        os.path.join(TOYOTA_RAW_LIDAR_ROOT, "sequences", seq),
        os.path.join(TOYOTA_RAW_LIDAR_ROOT, "sequences", seq, "velodyne"),
        os.path.join(TOYOTA_RAW_LIDAR_ROOT, "sequences", seq, "velodyne_sync"),
    ]

    lidar_dir = None
    for d in candidate_dirs:
        if os.path.isdir(d):
            lidar_dir = d
            break

    if lidar_dir is None:
        raise RuntimeError(
            f"Could not find raw LiDAR directory for Toyota sequence {seq}. "
            f"Please edit TOYOTA_RAW_LIDAR_ROOT or get_lidar_files()."
        )

    files = sorted([
        os.path.join(lidar_dir, f)
        for f in os.listdir(lidar_dir)
        if f.endswith(".bin") or f.endswith(".npy")
    ])

    if len(files) == 0:
        raise RuntimeError(f"No .bin or .npy LiDAR files found in {lidar_dir}")

    files = files[::SAMPLE_INTERVAL]
    files = files[:MAX_FRAMES]

    return files


def load_pointcloud(bin_path: str) -> np.ndarray:
    """
    Reads a point cloud file (.bin) formatted as XYZI (4 channels of float32) 
    in the Point-Major (Interleaved) format.

    Args:
        bin_path: The full path to the .bin file (e.g., 'pc/000001.bin').

    Returns:
        A NumPy array (N x 4) where N is the number of points and the columns
        are [X, Y, Z, I] before transformation. Returns an empty array if the 
        file is not found.
    """
    if not os.path.exists(bin_path):
        print(f"Error: File not found at {bin_path}")
        return np.array([])

    # --- CRITICAL FIX: Assume Point-Major (Interleaved) Format ---
    # Based on the successful alternative load functions provided by the user, 
    # the data is stored as X1, Y1, Z1, I1, X2, Y2, Z2, I2, ...
    
    # Read the raw binary data and directly reshape to (N x 4)
    data = np.fromfile(bin_path, dtype=np.float32)
    
    # We must check if the total number of elements is divisible by 4.
    if data.size % 4 != 0:
        print(f"Error: File size {data.size} is not divisible by 4 (XYZI). Check file integrity.")
        return np.array([])
        
    # The resulting point_cloud is now correctly structured as [X, Y, Z, I]
    point_cloud = data.reshape(-1, 4)

    return point_cloud
    

def load_poses():
    """
    Loads Toyota poses.

    This follows your dataset code:
        poses_path + '/poses.txt'
    """
    pose_path = os.path.join(TOYOTA_POSES_ROOT, "poses.txt")
    poses = np.loadtxt(pose_path)[::SAMPLE_INTERVAL]
    poses = poses[:MAX_FRAMES]
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
    Extracts Scan Context descriptors for Toyota sequence 00.

    Returns:
        descriptors: N x NUM_RINGS x NUM_SECTORS
        ring_keys:   N x NUM_RINGS
        poses:       N x pose_dim
        summary:     aggregate resource dictionary
    """

    proc = get_process_handle()

    lidar_files = get_lidar_files(seq)
    poses = load_poses()

    n = min(len(lidar_files), len(poses), MAX_FRAMES)
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
            print(f"[Scan Context][Toyota {seq}] processed {i + 1}/{n}")

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
# Toyota online evaluation
# ============================================================

def evaluate_scan_context_toyota(seq, descriptors, ring_keys, poses):
    """
    Online loop-closure evaluation for Toyota.

    This follows your evaluateResults() logic:
    - no fixed db_split_index
    - for query t, database is [t - WINDOW - OFFSET, t - OFFSET)
    - query starts at START_FRAME
    - positives are computed inside the same window
    - pose distance uses pose coordinates [:3]
    """

    T = min(len(descriptors), len(poses))
    descriptors = descriptors[:T]
    ring_keys = ring_keys[:T]
    poses = poses[:T]

    tp_flags = []
    all_gt = 0

    ring_key_search_times = []
    rerank_times = []
    index_build_times = []
    total_query_times = []

    num_searches = 0
    skipped_no_history = 0
    skipped_no_positives = 0

    retrieval_wall_start = time.perf_counter()

    for t in range(START_FRAME, T):
        start_idx = max(0, t - WINDOW - OFFSET)
        end_idx = t - OFFSET

        if end_idx - start_idx < 1:
            skipped_no_history += 1
            continue

        query_start = time.perf_counter()

        db_ring_keys = ring_keys[start_idx:end_idx]
        db_descs = descriptors[start_idx:end_idx]

        q_ring_key = ring_keys[t:t + 1]
        q_desc = descriptors[t]

        # --------------------------------------------------------
        # Build FAISS index for the current online window
        # --------------------------------------------------------
        index_start = time.perf_counter()

        faiss_index = faiss.IndexFlatL2(db_ring_keys.shape[1])
        faiss_index.add(db_ring_keys.astype(np.float32))

        index_build_time = time.perf_counter() - index_start

        # --------------------------------------------------------
        # Search candidate shortlist
        # --------------------------------------------------------
        k = min(NUM_CANDIDATES, db_ring_keys.shape[0])

        search_start = time.perf_counter()
        _, candidate_indices = faiss_index.search(
            q_ring_key.astype(np.float32),
            k,
        )
        search_time = time.perf_counter() - search_start

        # --------------------------------------------------------
        # Yaw-aligned re-ranking
        # --------------------------------------------------------
        rerank_start = time.perf_counter()

        best_local = None
        best_dist = float("inf")

        for candidate_local in candidate_indices[0]:
            dist = scan_context_distance(q_desc, db_descs[int(candidate_local)])

            if dist < best_dist:
                best_dist = dist
                best_local = int(candidate_local)

        rerank_time = time.perf_counter() - rerank_start

        pred_global = start_idx + best_local
        dist = float(best_dist)

        # --------------------------------------------------------
        # Ground-truth positives inside the same online window
        # --------------------------------------------------------
        pose_q = poses[t]
        pose_db = poses[start_idx:end_idx]

        sq = (pose_db - pose_q) ** 2
        dists_pos = np.sum(sq[:, :3], axis=1)
        positives_local = np.where(dists_pos < (GT_THRESHOLD_M ** 2))[0]

        if positives_local.size == 0:
            skipped_no_positives += 1

            total_query_time = time.perf_counter() - query_start

            index_build_times.append(index_build_time)
            ring_key_search_times.append(search_time)
            rerank_times.append(rerank_time)
            total_query_times.append(total_query_time)
            num_searches += 1

            continue

        all_gt += 1
        positives_global = start_idx + positives_local

        tp_flags.append((int(pred_global in positives_global), dist))

        total_query_time = time.perf_counter() - query_start

        index_build_times.append(index_build_time)
        ring_key_search_times.append(search_time)
        rerank_times.append(rerank_time)
        total_query_times.append(total_query_time)
        num_searches += 1

        if (t + 1) % 1000 == 0:
            print(f"[Scan Context][Toyota] evaluated query frame {t + 1}/{T}")

    retrieval_wall_time = time.perf_counter() - retrieval_wall_start

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------
    if len(tp_flags) == 0 or all_gt == 0:
        recall_at1 = 0.0
        max_f1 = 0.0
        auc = 0.0
        pr_time = 0.0
    else:
        tp = sum([f for f, _ in tp_flags])
        recall_at1 = float(tp / (all_gt + 1e-12))

        distances = np.array([d for _, d in tp_flags], dtype=float)
        labels = np.array([f for f, _ in tp_flags], dtype=int)

        d_min = distances.min()
        d_max = distances.max()

        if d_min == d_max:
            thresholds = np.array([d_min])
        else:
            thresholds = np.linspace(d_min, d_max, PR_NUM_THRESHOLDS)

        precisions = []
        recalls = []
        f1s = []

        pr_start = time.perf_counter()

        for th in thresholds:
            preds_pos = distances < th

            tp_c = np.sum((preds_pos == 1) & (labels == 1))
            fp_c = np.sum((preds_pos == 1) & (labels == 0))
            fn_c = np.sum((preds_pos == 0) & (labels == 1))

            prec = tp_c / (tp_c + fp_c + 1e-12)
            rec = tp_c / (tp_c + fn_c + 1e-12)
            f1 = 2.0 * prec * rec / (prec + rec + 1e-12)

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        precisions = np.array(precisions)
        recalls = np.array(recalls)
        f1s = np.array(f1s)

        max_f1 = float(f1s.max())

        order = np.argsort(recalls)
        auc = float(np.trapz(precisions[order], recalls[order]))

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
            "num_descriptors_total": int(T),
            "num_queries_attempted": int(max(0, T - START_FRAME)),
            "num_searches": int(num_searches),
            "num_queries_with_ground_truth": int(all_gt),
            "descriptor_shape": [NUM_RINGS, NUM_SECTORS],
            "descriptor_dimension": int(NUM_RINGS * NUM_SECTORS),
            "descriptor_dtype": str(descriptors.dtype),
            "descriptor_memory_mb": bytes_to_mb(descriptors.nbytes),
            "ring_key_memory_mb": bytes_to_mb(ring_keys.nbytes),
            "gt_threshold_m": float(GT_THRESHOLD_M),
            "window": int(WINDOW),
            "start_frame": int(START_FRAME),
            "offset": int(OFFSET),
            "skipped_no_history": int(skipped_no_history),
            "skipped_no_positives": int(skipped_no_positives),
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
            "online_index_build_time_per_query_ms": summarize_values(
                [v * 1000.0 for v in index_build_times]
            ),
            "ring_key_search_time_per_query_ms": summarize_values(
                [v * 1000.0 for v in ring_key_search_times]
            ),
            "rerank_time_per_query_ms": summarize_values(
                [v * 1000.0 for v in rerank_times]
            ),
            "total_retrieval_time_per_query_ms": summarize_values(
                [v * 1000.0 for v in total_query_times]
            ),

            "total_online_index_build_time_ms": float(np.sum(index_build_times) * 1000.0),
            "total_ring_key_search_time_ms": float(np.sum(ring_key_search_times) * 1000.0),
            "total_rerank_time_ms": float(np.sum(rerank_times) * 1000.0),
            "total_retrieval_time_ms": float(np.sum(total_query_times) * 1000.0),
            "retrieval_total_wall_time_seconds": float(retrieval_wall_time),
        },

        "evaluation_timing": {
            "precision_recall_time_ms": float(pr_time * 1000.0),
        },

        "ground_truth": {
            "num_queries_with_ground_truth": int(all_gt),
            "num_true_positive_top1": int(sum([f for f, _ in tp_flags])),
        },

        "retrieval_fps": (
            float(1000.0 / mean_total_query_ms)
            if mean_total_query_ms is not None and mean_total_query_ms > 0
            else None
        ),
    }

    return summary


# ============================================================
# Environment and summary
# ============================================================

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
            "dataset": "Toyota",
            "sequence": TOYOTA_SEQUENCE,
            "raw_lidar_root": TOYOTA_RAW_LIDAR_ROOT,
            "poses_root": TOYOTA_POSES_ROOT,
            "sample_interval": SAMPLE_INTERVAL,
            "max_frames": MAX_FRAMES,
            "gt_threshold_m": GT_THRESHOLD_M,
            "window": WINDOW,
            "start_frame": START_FRAME,
            "offset": OFFSET,
            "pr_num_thresholds": PR_NUM_THRESHOLDS,

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
                "Time to load a raw Toyota point cloud and compute its Scan Context descriptor."
            ),
            "online_index_build_time_per_query_ms": (
                "Time to build a FAISS index for the current rolling database window."
            ),
            "ring_key_search_time_per_query_ms": (
                "FAISS search time using Scan Context ring keys in the rolling window."
            ),
            "rerank_time_per_query_ms": (
                "Yaw-aligned Scan Context candidate re-ranking time."
            ),
            "total_retrieval_time_per_query_ms": (
                "Online FAISS index construction plus ring-key search plus yaw-aligned re-ranking."
            ),
            "descriptor_memory_mb": (
                "Memory occupied by all Scan Context descriptors for the sequence."
            ),
            "recall_at1": (
                "Top-1 loop-closure recall over queries that have at least one ground-truth positive in the searchable window."
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
    print("Scan Context Toyota online baseline")
    print(f"Sequence: {TOYOTA_SEQUENCE}")
    print(f"Max frames: {MAX_FRAMES}")
    print(f"Window: {WINDOW}")
    print(f"Offset: {OFFSET}")
    print(f"Start frame: {START_FRAME}")
    print(f"Descriptor: {NUM_RINGS} rings x {NUM_SECTORS} sectors")
    print(f"Yaw alignment: {USE_YAW_ALIGNMENT}")
    print(f"Candidates: {NUM_CANDIDATES}")
    print(f"Report path: {REPORT_PATH}")
    print("=" * 80)

    experiment_start = time.perf_counter()

    failed = None

    try:
        descriptors, ring_keys, poses, generation_summary = extract_scan_context_descriptors(
            TOYOTA_SEQUENCE
        )

        evaluation_summary = evaluate_scan_context_toyota(
            seq=TOYOTA_SEQUENCE,
            descriptors=descriptors,
            ring_keys=ring_keys,
            poses=poses,
        )

    except Exception as exc:
        generation_summary = None
        evaluation_summary = None
        failed = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

        print(f"[FAILED] {type(exc).__name__}: {exc}")

    experiment_time = time.perf_counter() - experiment_start

    report = {
        "report_type": "scan_context_toyota_online_resource_analysis",
        "environment": get_environment_metadata(),
        "summary": {
            "total_experiment_wall_time_seconds": float(experiment_time),
            "successful": failed is None,
            "error": failed,
            "overall_toyota": {
                "generation_time_per_scan_ms": (
                    generation_summary["generation_time_per_scan_ms"]
                    if generation_summary is not None
                    else None
                ),
                "generation_fps": (
                    generation_summary["generation_fps"]
                    if generation_summary is not None
                    else None
                ),
                "descriptor_memory_mb": (
                    generation_summary["descriptor_memory_mb"]
                    if generation_summary is not None
                    else None
                ),
                "retrieval_time_per_query_ms": (
                    evaluation_summary["retrieval_timing"]["total_retrieval_time_per_query_ms"]
                    if evaluation_summary is not None
                    else None
                ),
                "retrieval_fps": (
                    evaluation_summary["retrieval_fps"]
                    if evaluation_summary is not None
                    else None
                ),
                "retrieval_metrics": (
                    evaluation_summary["metrics"]
                    if evaluation_summary is not None
                    else None
                ),
            } if failed is None else None,
        },
        "descriptor_generation": generation_summary,
        "sequence_evaluation": evaluation_summary,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 80)
    if failed is None:
        print("Scan Context Toyota online baseline complete.")
        print(
            f"Recall@1: {evaluation_summary['metrics']['recall_at1_percent']:.2f}% | "
            f"F1: {evaluation_summary['metrics']['max_f1_percent']:.2f}% | "
            f"AUC: {evaluation_summary['metrics']['auc_percent']:.2f}%"
        )
        print(
            f"Generation: {generation_summary['generation_time_per_scan_ms']['mean']:.3f} ms/scan | "
            f"Retrieval: {evaluation_summary['retrieval_timing']['total_retrieval_time_per_query_ms']['mean']:.3f} ms/query"
        )
    else:
        print("Scan Context Toyota online baseline failed.")
    print(f"Report written to: {REPORT_PATH}")
    print("=" * 80)