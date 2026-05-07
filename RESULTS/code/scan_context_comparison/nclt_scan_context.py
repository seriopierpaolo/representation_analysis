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

# Folder containing raw NCLT LiDAR point clouds.
#
# Expected structure, recommended:
#   NCLT_RAW_LIDAR_ROOT/
#       2012-01-08/
#           *.bin or *.npy
#       2012-01-15/
#           *.bin or *.npy
#       2012-02-04/
#           *.bin or *.npy
#
# If your raw NCLT folder has a different internal structure,
# edit get_lidar_files().
NCLT_RAW_LIDAR_ROOT = "/dataset/nclt-dataset/vanilla/pointcloud"

# Existing folder used by your NCLT representation/evaluation code.
# This is used only for poses.
NCLT_REPRESENTATION_ROOT = (
    "/dataset/nclt-dataset/lidar_representations/representation_analysis"
)

REPORT_DIR = "./representation_analysis/temp"
REPORT_PATH = os.path.join(REPORT_DIR, "scan_context_nclt_resource_analysis.json")

# Same protocol as your NCLT code
NCLT_EVAL_SEQUENCES = ["2012-01-15", "2012-02-04", "2012-06-15", "2013-02-23"]
DATABASE_SEQUENCE = "2012-01-15"
QUERY_SEQUENCES = ["2012-02-04", "2012-06-15", "2013-02-23"]

SAMPLE_INTERVAL = 2
GT_THRESHOLD_M = 5.0

# Scan Context parameters
NUM_RINGS = 20
NUM_SECTORS = 60
MAX_RADIUS = 80.0

# NCLT Velodyne has large vertical variation depending on preprocessing.
# These values are deliberately permissive.
MIN_Z = -5.0
MAX_Z = 5.0

# Coarse-to-fine retrieval
USE_YAW_ALIGNMENT = True
NUM_CANDIDATES = 10

# If your raw .bin files have x,y,z,intensity, set this to 4.
# If they have only x,y,z, set this to 3.
BIN_POINT_DIM = 4


# ============================================================
# Optional dependency for CPU/RAM profiling
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
# NCLT loading
# ============================================================

def get_lidar_files(seq):
    """
    Returns sorted raw LiDAR files for an NCLT sequence.

    Edit this function if your NCLT raw point clouds are stored in a different
    subfolder, e.g.:
        /dataset/nclt-dataset/2012-01-08/velodyne_sync/*.bin
    """

    candidate_dirs = [
        os.path.join(NCLT_RAW_LIDAR_ROOT, seq, "velodyne_sync"),
        os.path.join(NCLT_RAW_LIDAR_ROOT, seq, "velodyne"),
        os.path.join(NCLT_RAW_LIDAR_ROOT, "sequences", seq, "velodyne_sync"),
        os.path.join(NCLT_RAW_LIDAR_ROOT, "sequences", seq, "velodyne"),
        os.path.join(NCLT_RAW_LIDAR_ROOT, seq),
        os.path.join(NCLT_RAW_LIDAR_ROOT, "sequences", seq),
    ]

    lidar_dir = None
    for d in candidate_dirs:
        if os.path.isdir(d):
            lidar_dir = d
            break

    if lidar_dir is None:
        raise RuntimeError(
            f"Could not find raw LiDAR directory for sequence {seq}. "
            f"Please edit NCLT_RAW_LIDAR_ROOT or get_lidar_files()."
        )

    files = sorted([
        os.path.join(lidar_dir, f)
        for f in os.listdir(lidar_dir)
        if f.endswith(".bin") or f.endswith(".npy")
    ])

    if len(files) == 0:
        raise RuntimeError(f"No .bin or .npy LiDAR files found in {lidar_dir}")

    return files[::SAMPLE_INTERVAL]


def load_pointcloud(bin_path):
    """
    Loads NCLT point cloud data (X, Y, Z, I) from a .bin file.
    The file is expected to contain a sequence of (X, Y, Z, I, L)
    records where X, Y, Z are 16-bit unsigned integers (H), I is 
    8-bit unsigned integer (B), and L (label) is an 8-bit unsigned
    integer (B).
    
    The function returns a numpy array with columns [X, Y, Z, I].
    """
    
    # 1. Load the raw binary data as a 1D NumPy array using the 
    #    correct combined data type for the full record (X, Y, Z, I, L).
    #    '<H' for 16-bit unsigned int (X, Y, Z), 'B' for 8-bit unsigned int (I, L)
    
    # Define the structured dtype for one record:
    # 3 x (uint16) for X, Y, Z, 2 x (uint8) for I, L
    # Total size: 2 + 2 + 2 + 1 + 1 = 8 bytes per point
    nclt_dtype = np.dtype([
        ('x_s', '<H'),
        ('y_s', '<H'),
        ('z_s', '<H'),
        ('i', '<B'),
        ('l', '<B')
    ])
    
    # Load the entire file into the structured array
    data = np.fromfile(bin_path, dtype=nclt_dtype)
    
    # Define NCLT scaling and offset constants
    scaling = 0.005 # 5 mm
    offset = -100.0

    # 2. Apply the scaling and offset to X, Y, Z
    
    # X, Y, Z are calculated using single-precision floats (np.float32)
    x = data['x_s'].astype(np.float32) * scaling + offset
    y = data['y_s'].astype(np.float32) * scaling + offset
    z = data['z_s'].astype(np.float32) * scaling + offset

    R = [[1, 0, 0],[0, -1, 0],[0, 0, -1]]
    
    # Intensity (I) is kept as a float for consistency, but the calculation is simpler
    i = data['i'].astype(np.float32)

    # 3. Combine the calculated columns (X, Y, Z, I) into a single (N, 4) array
    
    pointcloud = np.stack([x, -y, -z, i], axis=-1)
    
    return pointcloud


def load_poses(seq):
    pose_path = os.path.join(NCLT_REPRESENTATION_ROOT, "poses", f"{seq}.txt")
    poses = np.loadtxt(pose_path)[::SAMPLE_INTERVAL]
    return poses


# ============================================================
# Scan Context descriptor
# ============================================================

def make_scan_context(points):
    """
    Builds a Scan Context descriptor.

    Output shape:
        NUM_RINGS x NUM_SECTORS

    Each bin stores the maximum height value of all points falling inside
    the corresponding radial-angular cell.
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

    # Shift height so negative z values do not suppress empty cells.
    z_shifted = z - MIN_Z

    for ri, si, zi in zip(ring_idx, sector_idx, z_shifted):
        if zi > desc[ri, si]:
            desc[ri, si] = zi

    return desc


def make_ring_key(scan_context):
    """
    Ring key for coarse search.

    Shape:
        NUM_RINGS
    """
    return np.mean(scan_context, axis=1).astype(np.float32)


def scan_context_distance(sc1, sc2):
    """
    Yaw-invariant Scan Context distance.

    The query descriptor is compared to circularly shifted database descriptors.
    The best sector alignment is used.
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
# Descriptor extraction
# ============================================================

def extract_scan_context_descriptors(seq):
    """
    Extracts Scan Context descriptors for one NCLT sequence.

    Returns:
        descriptors: N x NUM_RINGS x NUM_SECTORS
        ring_keys:   N x NUM_RINGS
        poses:       sampled NCLT poses
        summary:     aggregate profiling dictionary
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

        if (i + 1) % 500 == 0:
            print(f"[Scan Context][NCLT {seq}] processed {i + 1}/{n}")

    descriptors = np.asarray(descriptors, dtype=np.float32)
    ring_keys = np.asarray(ring_keys, dtype=np.float32)

    sequence_wall_time = time.perf_counter() - sequence_wall_start
    sequence_cpu_time = get_cpu_time_seconds(proc) - sequence_cpu_start
    sequence_rss_end = get_rss_bytes(proc)

    mean_generation_ms = np.mean([v * 1000.0 for v in wall_times]) if wall_times else None

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

        "generation_time_per_scan_ms": summarize_values(
            [v * 1000.0 for v in wall_times]
        ),
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
# NCLT evaluation
# ============================================================

def evaluate_scan_context_nclt(db_pack, query_packs):
    """
    Evaluates Scan Context using your NCLT protocol.

    db_pack:
        dictionary for database sequence

    query_packs:
        list of dictionaries for query sequences

    The first sequence, 2012-01-08, is used as database.
    Query sequences are compared against it.
    """

    db_seq = db_pack["sequence"]
    db_descs = db_pack["descriptors"]
    db_ring_keys = db_pack["ring_keys"]
    db_poses = db_pack["poses"]

    # Build FAISS ring-key database
    index_build_start = time.perf_counter()

    faiss_index = faiss.IndexFlatL2(db_ring_keys.shape[1])
    faiss_index.add(db_ring_keys.astype(np.float32))

    index_build_time = time.perf_counter() - index_build_start

    all_query_reports = []

    for q_pack in query_packs:
        q_seq = q_pack["sequence"]
        q_descs = q_pack["descriptors"]
        q_ring_keys = q_pack["ring_keys"]
        q_poses = q_pack["poses"]

        print(f"[Scan Context][NCLT] DB {db_seq} -> Query {q_seq}")

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

        # --------------------------------------------------------
        # Ground-truth and metrics, matching your NCLT logic
        # --------------------------------------------------------
        gt_start = time.perf_counter()

        tp_flags = []
        all_gt = 0
        positive_counts = []

        for q_idx, (dist, pred) in enumerate(zip(pred_distances, pred_indices)):
            gt_dis = np.sum(
                (q_poses[q_idx, [4, 8]] - db_poses[:, [4, 8]]) ** 2,
                axis=1,
            )

            positives = np.where(gt_dis < GT_THRESHOLD_M ** 2)[0]
            positive_counts.append(len(positives))

            if len(positives) == 0:
                continue

            all_gt += 1
            tp_flags.append((pred in positives, dist))

        gt_time = time.perf_counter() - gt_start

        if all_gt == 0:
            recall_at1 = 0.0
            max_f1 = 0.0
            auc = 0.0
        else:
            labels = np.array([int(f) for f, _ in tp_flags])
            distances = np.array([d for _, d in tp_flags])

            recall_at1 = float(labels.mean())

            thresholds = np.linspace(distances.min(), distances.max(), 1000)

            precisions = []
            recalls = []
            f1s = []

            pr_start = time.perf_counter()

            for t in thresholds:
                preds_pos = distances < t

                tp = np.sum((preds_pos == 1) & (labels == 1))
                fp = np.sum((preds_pos == 1) & (labels == 0))
                fn = np.sum((preds_pos == 0) & (labels == 1))

                prec = tp / (tp + fp + 1e-8)
                rec = tp / (tp + fn + 1e-8)
                f1 = 2.0 * prec * rec / (prec + rec + 1e-8)

                precisions.append(prec)
                recalls.append(rec)
                f1s.append(f1)

            max_f1 = float(np.max(f1s))

            sort_idx = np.argsort(recalls)
            recalls_sorted = np.array(recalls)[sort_idx]
            precisions_sorted = np.array(precisions)[sort_idx]

            auc = float(np.trapz(precisions_sorted, recalls_sorted))

            pr_time = time.perf_counter() - pr_start

        if all_gt == 0:
            pr_time = 0.0

        mean_total_query_ms = np.mean([v * 1000.0 for v in total_query_times])

        query_report = {
            "database_sequence": db_seq,
            "query_sequence": q_seq,

            "metrics": {
                "recall_at1": float(recall_at1),
                "max_f1": float(max_f1),
                "auc": float(auc),
                "recall_at1_percent": float(recall_at1 * 100.0),
                "max_f1_percent": float(max_f1 * 100.0),
                "auc_percent": float(auc * 100.0),
            },

            "dataset": {
                "num_database_descriptors": int(db_descs.shape[0]),
                "num_query_descriptors": int(q_descs.shape[0]),
                "descriptor_shape": [NUM_RINGS, NUM_SECTORS],
                "descriptor_dimension": int(NUM_RINGS * NUM_SECTORS),
                "descriptor_dtype": str(db_descs.dtype),
                "database_descriptor_memory_mb": bytes_to_mb(db_descs.nbytes),
                "query_descriptor_memory_mb": bytes_to_mb(q_descs.nbytes),
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
                if mean_total_query_ms > 0
                else None
            ),
        }

        print(
            f"[Scan Context][{db_seq}->{q_seq}] "
            f"Recall@1: {query_report['metrics']['recall_at1_percent']:.2f}% | "
            f"F1: {query_report['metrics']['max_f1_percent']:.2f}% | "
            f"AUC: {query_report['metrics']['auc_percent']:.2f}% | "
            f"Retrieval: {query_report['retrieval_timing']['total_retrieval_time_per_query_ms']['mean']:.3f} ms/query"
        )

        all_query_reports.append(query_report)

    return all_query_reports


# ============================================================
# Overall summary
# ============================================================

def build_overall_summary(sequence_generation_reports, query_reports):
    generation_ms = []
    descriptor_mem = []

    for report in sequence_generation_reports:
        generation_ms.append(report["generation_time_per_scan_ms"]["mean"])
        descriptor_mem.append(report["descriptor_memory_mb"])

    recall_values = []
    f1_values = []
    auc_values = []
    retrieval_ms = []

    total_queries = 0

    for report in query_reports:
        recall_values.append(report["metrics"]["recall_at1"])
        f1_values.append(report["metrics"]["max_f1"])
        auc_values.append(report["metrics"]["auc"])
        retrieval_ms.append(report["retrieval_timing"]["total_retrieval_time_per_query_ms"]["mean"])
        total_queries += report["dataset"]["num_query_descriptors"]

    return {
        "num_sequences": int(len(sequence_generation_reports)),
        "database_sequence": DATABASE_SEQUENCE,
        "query_sequences": QUERY_SEQUENCES,
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
            "dataset": "NCLT",
            "raw_lidar_root": NCLT_RAW_LIDAR_ROOT,
            "representation_root": NCLT_REPRESENTATION_ROOT,
            "database_sequence": DATABASE_SEQUENCE,
            "query_sequences": QUERY_SEQUENCES,
            "eval_sequences": NCLT_EVAL_SEQUENCES,
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
                "Time to load a raw NCLT point cloud and compute its Scan Context descriptor."
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
    print("Scan Context NCLT baseline")
    print(f"Database sequence: {DATABASE_SEQUENCE}")
    print(f"Query sequences: {QUERY_SEQUENCES}")
    print(f"Sample interval: {SAMPLE_INTERVAL}")
    print(f"Descriptor: {NUM_RINGS} rings x {NUM_SECTORS} sectors")
    print(f"Yaw alignment: {USE_YAW_ALIGNMENT}")
    print(f"Candidates: {NUM_CANDIDATES}")
    print(f"Report path: {REPORT_PATH}")
    print("=" * 80)

    experiment_start = time.perf_counter()

    sequence_packs = []
    sequence_generation_reports = []
    failed_sequences = []

    for seq in NCLT_EVAL_SEQUENCES:
        try:
            descriptors, ring_keys, poses, generation_summary = extract_scan_context_descriptors(seq)

            sequence_packs.append({
                "sequence": seq,
                "descriptors": descriptors,
                "ring_keys": ring_keys,
                "poses": poses,
            })

            sequence_generation_reports.append(generation_summary)

        except Exception as exc:
            failed_sequences.append({
                "sequence": seq,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            })

    if len(failed_sequences) > 0:
        print("Some sequences failed:")
        for item in failed_sequences:
            print(item["sequence"], item["error"]["message"])

    pack_by_seq = {p["sequence"]: p for p in sequence_packs}

    query_reports = []

    if DATABASE_SEQUENCE in pack_by_seq:
        db_pack = pack_by_seq[DATABASE_SEQUENCE]
        query_packs = [
            pack_by_seq[s]
            for s in QUERY_SEQUENCES
            if s in pack_by_seq
        ]

        query_reports = evaluate_scan_context_nclt(db_pack, query_packs)

    experiment_time = time.perf_counter() - experiment_start

    report = {
        "report_type": "scan_context_nclt_resource_analysis",
        "environment": get_environment_metadata(),
        "summary": {
            "total_experiment_wall_time_seconds": float(experiment_time),
            "num_sequences_requested": int(len(NCLT_EVAL_SEQUENCES)),
            "num_sequences_successful": int(len(sequence_generation_reports)),
            "num_sequences_failed": int(len(failed_sequences)),
            "failed_sequences": failed_sequences,
            "overall_nclt": build_overall_summary(
                sequence_generation_reports,
                query_reports,
            ) if len(query_reports) > 0 else None,
        },
        "descriptor_generation": sequence_generation_reports,
        "query_evaluation": query_reports,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 80)
    print("Scan Context NCLT baseline complete.")
    print(f"Report written to: {REPORT_PATH}")
    print("=" * 80)