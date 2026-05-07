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
# KITTI paths and configuration
# ============================================================

KITTI_RAW_SEQUENCES_FOLDER = "/dataset/kitti-dataset/vanilla_dataset/sequences"
KITTI_REPRESENTATION_ANALYSIS_FOLDER = (
    "/dataset/kitti-dataset/lidar_representations/representation_analysis/"
)

REPORT_DIR = "./representation_analysis/temp"
REPORT_PATH = os.path.join(REPORT_DIR, "scan_context_kitti_resource_analysis.json")

VELODYNE_FOLDER = "velodyne"

KITTI_EVAL_SEQUENCES = ["02", "05", "06"]

KITTI_SEQ_SPLIT_POINTS = {
    "00": 3000,
    "02": 3400,
    "05": 1000,
    "06": 1000,
    "08": 1000,
}

# Scan Context parameters
NUM_RINGS = 20
NUM_SECTORS = 60
MAX_RADIUS = 80.0

# Optional point filtering
MIN_Z = -3.0
MAX_Z = 5.0

# If True, descriptors are yaw-aligned by circular sector shift during comparison.
# This is more faithful to Scan Context than plain flattened L2.
USE_YAW_ALIGNMENT = True

# Candidate shortlist from ring-key retrieval.
# Larger values are more accurate but slower.
NUM_CANDIDATES = 10

GT_THRESHOLD_M = 5.0


# ============================================================
# Optional psutil for RAM/CPU profiling
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


def load_kitti_bin(bin_path):
    """
    Loads KITTI Velodyne .bin file.

    KITTI point format:
        x, y, z, reflectance
    """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points


def load_kitti_poses(seq):
    pose_path = os.path.join(
        KITTI_REPRESENTATION_ANALYSIS_FOLDER,
        "poses",
        f"{seq}.txt",
    )
    return np.loadtxt(pose_path)


def get_velodyne_files(seq):
    seq_path = os.path.join(
        KITTI_RAW_SEQUENCES_FOLDER,
        seq,
        VELODYNE_FOLDER,
    )

    files = sorted([
        f for f in os.listdir(seq_path)
        if f.endswith(".bin")
    ])

    return [os.path.join(seq_path, f) for f in files]


# ============================================================
# Scan Context descriptor
# ============================================================

def make_scan_context(points):
    """
    Builds a Scan Context descriptor.

    Descriptor shape:
        NUM_RINGS x NUM_SECTORS

    Each bin stores the maximum z-height of points falling inside that
    ring-sector bin. Empty bins remain zero.

    This is the standard Scan Context idea:
    radial bins x angular bins with max-height encoding.
    """

    xyz = points[:, :3]

    # Optional z filtering
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

    # Shift z so that negative heights do not suppress empty bins.
    # This keeps the max-height encoding stable while preserving relative height.
    z_shifted = z - MIN_Z

    for ri, si, zi in zip(ring_idx, sector_idx, z_shifted):
        if zi > desc[ri, si]:
            desc[ri, si] = zi

    return desc


def make_ring_key(scan_context):
    """
    Ring key used for coarse retrieval.

    Shape:
        NUM_RINGS

    We use the mean over sectors for each ring.
    """
    return np.mean(scan_context, axis=1).astype(np.float32)


def cosine_distance(a, b):
    """
    Cosine distance between two 1D vectors.
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return 1.0 - float(np.dot(a, b) / denom)


def scan_context_distance(sc1, sc2):
    """
    Computes yaw-invariant Scan Context distance by circularly shifting
    sectors and taking the best sector alignment.

    Lower is better.
    """

    if not USE_YAW_ALIGNMENT:
        return cosine_distance(sc1.reshape(-1), sc2.reshape(-1))

    best_dist = float("inf")

    for shift in range(NUM_SECTORS):
        shifted = np.roll(sc2, shift=shift, axis=1)

        # Column-wise similarity, following the usual Scan Context comparison idea.
        valid_cols = 0
        similarity_sum = 0.0

        for col in range(NUM_SECTORS):
            v1 = sc1[:, col]
            v2 = shifted[:, col]

            if np.linalg.norm(v1) < 1e-12 or np.linalg.norm(v2) < 1e-12:
                continue

            sim = np.dot(v1, v2) / ((np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-12)
            similarity_sum += sim
            valid_cols += 1

        if valid_cols == 0:
            dist = 1.0
        else:
            dist = 1.0 - similarity_sum / valid_cols

        if dist < best_dist:
            best_dist = dist

    return best_dist


# ============================================================
# Descriptor extraction and profiling
# ============================================================

def extract_scan_context_descriptors(seq):
    """
    Extracts Scan Context descriptors for one KITTI sequence and profiles
    descriptor-generation resource usage.

    Returns:
        descriptors: N x NUM_RINGS x NUM_SECTORS
        ring_keys:   N x NUM_RINGS
        summary:     aggregate profiling dict
    """

    proc = get_process_handle()
    bin_files = get_velodyne_files(seq)

    descriptors = []
    ring_keys = []

    wall_times = []
    cpu_times = []
    cpu_utils = []
    ram_delta_mb = []
    num_points_values = []

    sequence_wall_start = time.perf_counter()
    sequence_cpu_start = get_cpu_time_seconds(proc)
    sequence_rss_start = get_rss_bytes(proc)

    for i, bin_path in enumerate(bin_files):
        rss_before = get_rss_bytes(proc)
        cpu_before = get_cpu_time_seconds(proc)
        wall_start = time.perf_counter()

        points = load_kitti_bin(bin_path)
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
            ram_delta_mb.append(bytes_to_mb(rss_after - rss_before))

        if (i + 1) % 500 == 0:
            print(f"[Scan Context][seq {seq}] processed {i + 1}/{len(bin_files)} scans")

    descriptors = np.asarray(descriptors, dtype=np.float32)
    ring_keys = np.asarray(ring_keys, dtype=np.float32)

    sequence_wall_time = time.perf_counter() - sequence_wall_start
    sequence_cpu_time = get_cpu_time_seconds(proc) - sequence_cpu_start
    sequence_rss_end = get_rss_bytes(proc)

    summary = {
        "num_scans": int(len(bin_files)),

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
        "generation_ram_delta_mb": summarize_values(ram_delta_mb),

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

    mean_time_s = summary["generation_time_per_scan_ms"]["mean"]
    if mean_time_s is not None and mean_time_s > 0:
        summary["generation_fps"] = float(1000.0 / mean_time_s)
    else:
        summary["generation_fps"] = None

    return descriptors, ring_keys, summary


# ============================================================
# KITTI evaluation with Scan Context
# ============================================================

def evaluate_scan_context(seq, descriptors, ring_keys, poses):
    """
    Evaluates Scan Context using the same KITTI database/query split.

    Database:
        descriptors[:db_split_index]

    Queries:
        descriptors[db_split_index:]

    Retrieval:
        1. Use ring-key FAISS search for candidate shortlist.
        2. Re-rank candidates using yaw-aligned Scan Context distance.
    """

    db_split_index = KITTI_SEQ_SPLIT_POINTS[seq]

    db_descs = descriptors[:db_split_index]
    q_descs = descriptors[db_split_index:]

    db_ring_keys = ring_keys[:db_split_index]
    q_ring_keys = ring_keys[db_split_index:]

    # ------------------------------------------------------------
    # FAISS ring-key index
    # ------------------------------------------------------------
    index_build_start = time.perf_counter()

    faiss_index = faiss.IndexFlatL2(db_ring_keys.shape[1])
    faiss_index.add(db_ring_keys.astype(np.float32))

    index_build_time = time.perf_counter() - index_build_start

    # ------------------------------------------------------------
    # Candidate retrieval + SC re-ranking
    # ------------------------------------------------------------
    search_times = []
    rerank_times = []
    total_query_times = []

    pred_indices = []
    pred_distances = []

    k = min(NUM_CANDIDATES, db_ring_keys.shape[0])

    for q_idx in range(q_ring_keys.shape[0]):
        q_total_start = time.perf_counter()

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
        total_query_time = time.perf_counter() - q_total_start

        search_times.append(search_time)
        rerank_times.append(rerank_time)
        total_query_times.append(total_query_time)

        pred_indices.append(best_idx)
        pred_distances.append(best_dist)

    pred_indices = np.asarray(pred_indices, dtype=np.int32)
    pred_distances = np.asarray(pred_distances, dtype=np.float32)

    # ------------------------------------------------------------
    # Same Recall@1 / F1 / AUC protocol as your learned method
    # ------------------------------------------------------------
    tp_flags = []
    all_gt = 0
    gt_positive_counts = []

    gt_start = time.perf_counter()

    for q_idx, pred in enumerate(pred_indices):
        query_idx = db_split_index + q_idx

        gt_dis = (poses[query_idx] - poses[:db_split_index]) ** 2
        positives = np.where(
            np.sum(gt_dis[:, [3, 7, 11]], axis=1) < GT_THRESHOLD_M ** 2
        )[0]

        gt_positive_counts.append(len(positives))

        if len(positives) == 0:
            continue

        all_gt += 1
        tp_flags.append((pred in positives, pred_distances[q_idx]))

    gt_time = time.perf_counter() - gt_start

    tp = sum([f for f, _ in tp_flags])
    recall_at1 = tp / (all_gt + 1e-8)

    distances = np.array([d for _, d in tp_flags])
    labels = np.array([int(f) for f, _ in tp_flags])

    pr_start = time.perf_counter()

    if len(distances) == 0:
        max_f1 = 0.0
        auc = 0.0
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
            f1 = 2.0 * prec * rec / (prec + rec + 1e-8)

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        max_f1 = float(np.max(f1s))

        sorted_idx = np.argsort(recalls)
        recalls_sorted = np.array(recalls)[sorted_idx]
        precisions_sorted = np.array(precisions)[sorted_idx]

        auc = float(np.trapz(precisions_sorted, recalls_sorted))

    pr_time = time.perf_counter() - pr_start

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
            "descriptor_dimension": int(NUM_RINGS * NUM_SECTORS),
            "descriptor_shape": [NUM_RINGS, NUM_SECTORS],
            "descriptor_dtype": str(descriptors.dtype),
            "descriptor_memory_mb": bytes_to_mb(descriptors.nbytes),
            "db_split_index": int(db_split_index),
            "gt_threshold_m": float(GT_THRESHOLD_M),
        },

        "scan_context": {
            "num_rings": int(NUM_RINGS),
            "num_sectors": int(NUM_SECTORS),
            "max_radius_m": float(MAX_RADIUS),
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
        },

        "evaluation_timing": {
            "ground_truth_matching_time_ms": float(gt_time * 1000.0),
            "precision_recall_time_ms": float(pr_time * 1000.0),
        },

        "ground_truth": {
            "num_queries_with_ground_truth": int(all_gt),
            "num_true_positive_top1": int(sum([f for f, _ in tp_flags])),
            "positive_count_per_query": summarize_values(gt_positive_counts),
        },
    }

    mean_retrieval_ms = summary["retrieval_timing"]["total_retrieval_time_per_query_ms"]["mean"]

    if mean_retrieval_ms is not None and mean_retrieval_ms > 0:
        summary["retrieval_fps"] = float(1000.0 / mean_retrieval_ms)
    else:
        summary["retrieval_fps"] = None

    return summary


# ============================================================
# Overall summary
# ============================================================

def build_overall_summary(sequence_reports):
    recall_values = []
    f1_values = []
    auc_values = []

    generation_ms = []
    retrieval_ms = []
    descriptor_mem = []

    total_scans = 0
    total_queries = 0

    for report in sequence_reports:
        gen = report["descriptor_generation"]
        ev = report["evaluation"]

        total_scans += ev["dataset"]["num_descriptors_total"]
        total_queries += ev["dataset"]["num_query_descriptors"]

        recall_values.append(ev["metrics"]["recall_at1"])
        f1_values.append(ev["metrics"]["max_f1"])
        auc_values.append(ev["metrics"]["auc"])

        generation_ms.append(gen["generation_time_per_scan_ms"]["mean"])
        retrieval_ms.append(
            ev["retrieval_timing"]["total_retrieval_time_per_query_ms"]["mean"]
        )
        descriptor_mem.append(ev["dataset"]["descriptor_memory_mb"])

    overall = {
        "num_sequences": int(len(sequence_reports)),
        "total_scans": int(total_scans),
        "total_queries": int(total_queries),

        "generation_time_per_scan_ms": summarize_values(generation_ms),
        "retrieval_time_per_query_ms": summarize_values(retrieval_ms),
        "descriptor_memory_mb": summarize_values(descriptor_mem),

        "retrieval_metrics": {
            "recall_at1": summarize_values(recall_values),
            "max_f1": summarize_values(f1_values),
            "auc": summarize_values(auc_values),
        },
    }

    return overall


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
            "dataset": "KITTI",
            "eval_sequences": KITTI_EVAL_SEQUENCES,
            "raw_sequences_folder": KITTI_RAW_SEQUENCES_FOLDER,
            "representation_analysis_folder": KITTI_REPRESENTATION_ANALYSIS_FOLDER,
            "report_path": REPORT_PATH,

            "num_rings": NUM_RINGS,
            "num_sectors": NUM_SECTORS,
            "max_radius": MAX_RADIUS,
            "min_z": MIN_Z,
            "max_z": MAX_Z,
            "use_yaw_alignment": USE_YAW_ALIGNMENT,
            "num_candidates": NUM_CANDIDATES,
            "gt_threshold_m": GT_THRESHOLD_M,
        },

        "metric_definitions": {
            "generation_time_per_scan_ms": (
                "Time required to load one KITTI .bin scan and construct its Scan Context descriptor."
            ),
            "ring_key_search_time_per_query_ms": (
                "FAISS search time using the Scan Context ring key."
            ),
            "rerank_time_per_query_ms": (
                "Time required to compare candidate Scan Context descriptors using yaw-aligned distance."
            ),
            "total_retrieval_time_per_query_ms": (
                "Ring-key search plus Scan Context candidate re-ranking."
            ),
            "descriptor_memory_mb": (
                "Memory used by the Scan Context descriptor array."
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
    print("Scan Context KITTI baseline")
    print(f"Sequences: {KITTI_EVAL_SEQUENCES}")
    print(f"Descriptor: {NUM_RINGS} rings x {NUM_SECTORS} sectors")
    print(f"Yaw alignment: {USE_YAW_ALIGNMENT}")
    print(f"Candidate shortlist: {NUM_CANDIDATES}")
    print(f"Report path: {REPORT_PATH}")
    print("=" * 80)

    sequence_reports = []

    experiment_start = time.perf_counter()

    for seq in KITTI_EVAL_SEQUENCES:
        try:
            print("=" * 80)
            print(f"Processing KITTI sequence {seq}")
            print("=" * 80)

            poses = load_kitti_poses(seq)

            descriptors, ring_keys, generation_summary = extract_scan_context_descriptors(seq)

            evaluation_summary = evaluate_scan_context(
                seq=seq,
                descriptors=descriptors,
                ring_keys=ring_keys,
                poses=poses,
            )

            print(
                f"Seq {seq} | "
                f"Recall@1: {evaluation_summary['metrics']['recall_at1_percent']:.2f}% | "
                f"F1: {evaluation_summary['metrics']['max_f1_percent']:.2f}% | "
                f"AUC: {evaluation_summary['metrics']['auc_percent']:.2f}%"
            )

            sequence_reports.append({
                "sequence": seq,
                "descriptor_generation": generation_summary,
                "evaluation": evaluation_summary,
            })

        except Exception as exc:
            sequence_reports.append({
                "sequence": seq,
                "status": "failed",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            })

    experiment_time = time.perf_counter() - experiment_start

    successful_reports = [
        r for r in sequence_reports
        if r.get("status") != "failed"
    ]

    report = {
        "report_type": "scan_context_kitti_resource_analysis",
        "environment": get_environment_metadata(),
        "summary": {
            "total_experiment_wall_time_seconds": float(experiment_time),
            "num_sequences_requested": int(len(KITTI_EVAL_SEQUENCES)),
            "num_sequences_successful": int(len(successful_reports)),
            "num_sequences_failed": int(len(sequence_reports) - len(successful_reports)),
            "overall_kitti": (
                build_overall_summary(successful_reports)
                if len(successful_reports) > 0
                else None
            ),
        },
        "sequences": sequence_reports,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 80)
    print("Scan Context KITTI baseline complete.")
    print(f"Report written to: {REPORT_PATH}")
    print("=" * 80)