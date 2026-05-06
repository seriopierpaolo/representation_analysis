import os
import sys
import json
import time
import socket
import platform
import datetime
import traceback
import tracemalloc
from multiprocessing import Pool, cpu_count

import numpy as np

import lib.utils as utils
from lib.curvature import compute_curvature


# ============================================================
# Configuration
# ============================================================

POINTCLOUD_FOLDER = "/dataset/kitti-dataset/vanilla_dataset/sequences"
VELODYNE_FOLDER = "velodyne"

# Save all generated projection images locally in ./temp
TEMP_OUTPUT_FOLDER = os.path.abspath("./temp")

# JSON report path
REPORT_PATH = os.path.join(
    TEMP_OUTPUT_FOLDER,
    "resource_analysis_sequence_00_first_1000_summary.json"
)

# Only the first KITTI sequence
SEQUENCES = ["00"]

# Only the first 1000 frames
MAX_FRAMES_PER_SEQUENCE = 1000

# Number of workers.
#
# For reviewer-facing latency numbers, NUM_WORKERS = 1 is the cleanest option.
# For practical throughput, use more workers and report that separately.
NUM_WORKERS = min(8, cpu_count())

# Projection image sizes
BEV_IMAGE_SIZE = [256, 256]
POLAR_IMAGE_SIZE = [256, 256]
RANGE_IMAGE_SIZE = [96, 512]
FRONT_IMAGE_SIZE = [96, 256]

# LiDAR filtering
X_RANGE = [-40, 40]
Y_RANGE = [-40, 40]
Z_RANGE = [-1.5, 1.5]

# LiDAR model used by the projection functions
LIDAR_MODEL = "HDL64E"

# Curvature settings
CURVATURE_K_NEIGHBORS = 30
CURVATURE_NUM_PROCESSORS = 1


# ============================================================
# Optional dependencies
# ============================================================

try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    psutil = None
    PSUTIL_AVAILABLE = False


try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except Exception:
    pynvml = None
    PYNVML_AVAILABLE = False


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


def get_cpu_time_seconds(proc=None):
    """
    Returns process CPU time in seconds.

    This is user CPU time + system CPU time.
    """
    if proc is not None:
        t = proc.cpu_times()
        return float(t.user + t.system)

    return float(time.process_time())


def get_rss_bytes(proc=None):
    """
    Returns resident set size, RSS, for the current process.
    """
    if proc is not None:
        return int(proc.memory_info().rss)

    return None


def get_gpu_memory_snapshot():
    """
    Returns NVIDIA GPU VRAM usage for all visible GPUs.

    If pynvml is not available, or if no NVIDIA GPU is visible, returns None.

    Important:
    This projection script is expected to be CPU-side unless the functions in
    lib.utils internally use GPU operations. Therefore, GPU VRAM may be None
    or approximately unchanged.
    """
    if not PYNVML_AVAILABLE:
        return None

    try:
        gpu_info = []
        device_count = pynvml.nvmlDeviceGetCount()

        for idx in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)

            if isinstance(name, bytes):
                name = name.decode("utf-8")

            gpu_info.append({
                "gpu_index": idx,
                "gpu_name": name,
                "vram_used_bytes": int(mem.used),
                "vram_used_mb": bytes_to_mb(mem.used),
                "vram_total_bytes": int(mem.total),
                "vram_total_mb": bytes_to_mb(mem.total),
                "vram_free_bytes": int(mem.free),
                "vram_free_mb": bytes_to_mb(mem.free),
            })

        return gpu_info

    except Exception:
        return None


def summarize_gpu_delta(before, after):
    """
    Computes per-GPU VRAM delta.
    """
    if before is None or after is None:
        return None

    if len(before) != len(after):
        return None

    result = []

    for b, a in zip(before, after):
        delta = a["vram_used_bytes"] - b["vram_used_bytes"]

        result.append({
            "gpu_index": a["gpu_index"],
            "gpu_name": a["gpu_name"],
            "vram_used_before_bytes": b["vram_used_bytes"],
            "vram_used_after_bytes": a["vram_used_bytes"],
            "vram_delta_bytes": int(delta),
            "vram_used_before_mb": b["vram_used_mb"],
            "vram_used_after_mb": a["vram_used_mb"],
            "vram_delta_mb": bytes_to_mb(delta),
        })

    return result


def save_projection(img, frame_index, output_path):
    """
    Saves an image projection.

    If the image is single-channel H x W x 1, it is repeated to H x W x 3
    for compatibility with common image saving functions.
    """
    os.makedirs(output_path, exist_ok=True)

    if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    utils.save_img(img, frame_index, path=output_path)


# ============================================================
# Profiling
# ============================================================

def profile_block(block_name, fn):
    """
    Profiles one computation block.

    Metrics:
    - wall-clock time
    - process CPU time
    - estimated CPU utilization
    - RAM RSS before/after/delta
    - Python peak allocated memory via tracemalloc
    - optional GPU VRAM before/after/delta
    - output tensor shape, dtype and size
    """
    proc = get_process_handle()

    rss_before = get_rss_bytes(proc)
    cpu_before = get_cpu_time_seconds(proc)
    gpu_before = get_gpu_memory_snapshot()

    tracemalloc.start()
    wall_start = time.perf_counter()

    output = fn()

    wall_time = time.perf_counter() - wall_start
    current_py_mem, peak_py_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    cpu_after = get_cpu_time_seconds(proc)
    rss_after = get_rss_bytes(proc)
    gpu_after = get_gpu_memory_snapshot()

    cpu_time = cpu_after - cpu_before

    if wall_time > 0:
        estimated_cpu_utilization_percent = 100.0 * cpu_time / wall_time
    else:
        estimated_cpu_utilization_percent = None

    if isinstance(output, np.ndarray):
        output_info = {
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "nbytes": int(output.nbytes),
            "nbytes_mb": bytes_to_mb(output.nbytes),
        }
    else:
        output_info = {
            "type": str(type(output)),
        }

    metrics = {
        "name": block_name,

        "wall_time_seconds": float(wall_time),
        "cpu_process_time_seconds": float(cpu_time),
        "estimated_cpu_utilization_percent": (
            float(estimated_cpu_utilization_percent)
            if estimated_cpu_utilization_percent is not None
            else None
        ),

        "ram_rss_before_bytes": rss_before,
        "ram_rss_after_bytes": rss_after,
        "ram_rss_delta_bytes": (
            int(rss_after - rss_before)
            if rss_before is not None and rss_after is not None
            else None
        ),

        "ram_rss_before_mb": bytes_to_mb(rss_before),
        "ram_rss_after_mb": bytes_to_mb(rss_after),
        "ram_rss_delta_mb": (
            bytes_to_mb(rss_after - rss_before)
            if rss_before is not None and rss_after is not None
            else None
        ),

        "python_tracemalloc_current_bytes": int(current_py_mem),
        "python_tracemalloc_peak_bytes": int(peak_py_mem),
        "python_tracemalloc_current_mb": bytes_to_mb(current_py_mem),
        "python_tracemalloc_peak_mb": bytes_to_mb(peak_py_mem),

        "gpu_vram": summarize_gpu_delta(gpu_before, gpu_after),

        "output": output_info,
    }

    return output, metrics


# ============================================================
# Frame processing
# ============================================================

def process_frame(args):
    """
    Processes one LiDAR frame.

    Returns only the metrics needed for aggregate statistics.
    The final JSON will not include individual frame-level records.
    """
    frame_index, velodyne_file, seq = args

    result = {
        "status": "started",
        "sequence": seq,
        "frame_index": int(frame_index),
        "velodyne_file": velodyne_file,
        "error": None,
        "preprocessing": None,
        "projections": {},
    }

    try:
        proc = get_process_handle()

        velodyne_pointcloud_path = os.path.join(
            POINTCLOUD_FOLDER,
            seq,
            VELODYNE_FOLDER
        )

        file_path = os.path.join(velodyne_pointcloud_path, velodyne_file)

        print(f"[seq {seq}] Processing frame {frame_index + 1}: {velodyne_file}")

        # ------------------------------------------------------------
        # Preprocessing: load, filter, curvature
        # ------------------------------------------------------------
        preprocessing_rss_before = get_rss_bytes(proc)
        preprocessing_cpu_before = get_cpu_time_seconds(proc)
        preprocessing_wall_start = time.perf_counter()

        points = utils.load_pointcloud(file_path)
        raw_num_points = int(points.shape[0])

        points = utils.filter_box(
            points,
            x_range=X_RANGE,
            y_range=Y_RANGE,
            z_range=Z_RANGE
        )
        filtered_num_points = int(points.shape[0])

        curvatures = compute_curvature(
            "curvature",
            points,
            k_neighbors=CURVATURE_K_NEIGHBORS,
            n_processors=CURVATURE_NUM_PROCESSORS
        )

        points = np.hstack((points, curvatures[:, np.newaxis]))

        preprocessing_wall_time = time.perf_counter() - preprocessing_wall_start
        preprocessing_cpu_time = get_cpu_time_seconds(proc) - preprocessing_cpu_before
        preprocessing_rss_after = get_rss_bytes(proc)

        result["preprocessing"] = {
            "raw_num_points": raw_num_points,
            "filtered_num_points": filtered_num_points,
            "point_features_after_curvature": int(points.shape[1]),

            "wall_time_seconds": float(preprocessing_wall_time),
            "cpu_process_time_seconds": float(preprocessing_cpu_time),
            "estimated_cpu_utilization_percent": (
                float(100.0 * preprocessing_cpu_time / preprocessing_wall_time)
                if preprocessing_wall_time > 0
                else None
            ),

            "ram_rss_before_mb": bytes_to_mb(preprocessing_rss_before),
            "ram_rss_after_mb": bytes_to_mb(preprocessing_rss_after),
            "ram_rss_delta_mb": (
                bytes_to_mb(preprocessing_rss_after - preprocessing_rss_before)
                if preprocessing_rss_before is not None and preprocessing_rss_after is not None
                else None
            ),
        }

        # ------------------------------------------------------------
        # BEV projection
        # ------------------------------------------------------------
        bev_img, bev_metrics = profile_block(
            "bev",
            lambda: utils.cartesian_to_image(points, BEV_IMAGE_SIZE)
        )

        save_projection(
            bev_img,
            frame_index,
            os.path.join(TEMP_OUTPUT_FOLDER, seq, "bev")
        )

        result["projections"]["bev"] = bev_metrics

        # ------------------------------------------------------------
        # Polar projection
        # ------------------------------------------------------------
        def make_polar_projection():
            polar_points = utils.cartesian_to_polar(points)
            return utils.polar_to_image(polar_points, POLAR_IMAGE_SIZE)

        polar_img, polar_metrics = profile_block(
            "polar",
            make_polar_projection
        )

        save_projection(
            polar_img,
            frame_index,
            os.path.join(TEMP_OUTPUT_FOLDER, seq, "polar")
        )

        result["projections"]["polar"] = polar_metrics

        # ------------------------------------------------------------
        # Range projection
        # ------------------------------------------------------------
        range_img, range_metrics = profile_block(
            "range",
            lambda: utils.cartesian_to_range_image_lidar_specs(
                points,
                lidar=LIDAR_MODEL,
                image_size=RANGE_IMAGE_SIZE
            )
        )

        save_projection(
            range_img,
            frame_index,
            os.path.join(TEMP_OUTPUT_FOLDER, seq, "range")
        )

        result["projections"]["range"] = range_metrics

        # ------------------------------------------------------------
        # Front-view projection
        # ------------------------------------------------------------
        front_img, front_metrics = profile_block(
            "front",
            lambda: utils.cartesian_to_frontview(
                points,
                lidar=LIDAR_MODEL,
                image_size=FRONT_IMAGE_SIZE,
                fov_h=(-45, 45)
            )
        )

        save_projection(
            front_img,
            frame_index,
            os.path.join(TEMP_OUTPUT_FOLDER, seq, "front")
        )

        result["projections"]["front"] = front_metrics

        result["status"] = "finished"

    except Exception as exc:
        result["status"] = "failed"
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    return result


# ============================================================
# Aggregation
# ============================================================

def empty_metric_store():
    return []


def add_if_valid(store, value):
    if value is not None:
        store.append(float(value))


def summarize_values(values):
    """
    Computes paper-friendly summary statistics.

    The JSON intentionally stores only aggregate values, not per-frame values.
    """
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


def summarize_gpu_vram(gpu_entries):
    """
    Aggregates GPU VRAM deltas.

    gpu_entries is a list of gpu_vram lists returned by profile_block.
    """
    if len(gpu_entries) == 0:
        return None

    per_gpu = {}

    for entry in gpu_entries:
        if entry is None:
            continue

        for gpu in entry:
            idx = gpu["gpu_index"]

            if idx not in per_gpu:
                per_gpu[idx] = {
                    "gpu_index": idx,
                    "gpu_name": gpu["gpu_name"],
                    "vram_delta_mb_values": [],
                    "vram_used_before_mb_values": [],
                    "vram_used_after_mb_values": [],
                }

            add_if_valid(
                per_gpu[idx]["vram_delta_mb_values"],
                gpu.get("vram_delta_mb")
            )
            add_if_valid(
                per_gpu[idx]["vram_used_before_mb_values"],
                gpu.get("vram_used_before_mb")
            )
            add_if_valid(
                per_gpu[idx]["vram_used_after_mb_values"],
                gpu.get("vram_used_after_mb")
            )

    if len(per_gpu) == 0:
        return None

    output = []

    for idx, data in per_gpu.items():
        output.append({
            "gpu_index": data["gpu_index"],
            "gpu_name": data["gpu_name"],
            "vram_delta_mb": summarize_values(data["vram_delta_mb_values"]),
            "vram_used_before_mb": summarize_values(data["vram_used_before_mb_values"]),
            "vram_used_after_mb": summarize_values(data["vram_used_after_mb_values"]),
        })

    return output


def initialize_aggregate_store():
    projection_names = ["bev", "polar", "range", "front"]

    store = {
        "num_frames_successful": 0,
        "num_frames_failed": 0,
        "failed_frame_examples": [],

        "preprocessing": {
            "raw_num_points": [],
            "filtered_num_points": [],
            "wall_time_seconds": [],
            "cpu_process_time_seconds": [],
            "estimated_cpu_utilization_percent": [],
            "ram_rss_delta_mb": [],
        },

        "projections": {},

        "combined_four_projection_wall_time_seconds": [],
        "combined_four_projection_cpu_process_time_seconds": [],
    }

    for name in projection_names:
        store["projections"][name] = {
            "wall_time_seconds": [],
            "cpu_process_time_seconds": [],
            "estimated_cpu_utilization_percent": [],
            "ram_rss_delta_mb": [],
            "python_tracemalloc_peak_mb": [],
            "output_nbytes_mb": [],
            "gpu_vram_entries": [],
            "output_shape_examples": [],
        }

    return store


def update_aggregate_store(store, frame_result):
    projection_names = ["bev", "polar", "range", "front"]

    if frame_result.get("status") != "finished":
        store["num_frames_failed"] += 1

        if len(store["failed_frame_examples"]) < 10:
            store["failed_frame_examples"].append({
                "sequence": frame_result.get("sequence"),
                "frame_index": frame_result.get("frame_index"),
                "velodyne_file": frame_result.get("velodyne_file"),
                "error": frame_result.get("error"),
            })

        return

    store["num_frames_successful"] += 1

    pre = frame_result.get("preprocessing", {})

    add_if_valid(store["preprocessing"]["raw_num_points"], pre.get("raw_num_points"))
    add_if_valid(store["preprocessing"]["filtered_num_points"], pre.get("filtered_num_points"))
    add_if_valid(store["preprocessing"]["wall_time_seconds"], pre.get("wall_time_seconds"))
    add_if_valid(store["preprocessing"]["cpu_process_time_seconds"], pre.get("cpu_process_time_seconds"))
    add_if_valid(store["preprocessing"]["estimated_cpu_utilization_percent"], pre.get("estimated_cpu_utilization_percent"))
    add_if_valid(store["preprocessing"]["ram_rss_delta_mb"], pre.get("ram_rss_delta_mb"))

    combined_wall_time = 0.0
    combined_cpu_time = 0.0
    combined_valid = True

    for name in projection_names:
        proj = frame_result.get("projections", {}).get(name, {})
        proj_store = store["projections"][name]

        add_if_valid(proj_store["wall_time_seconds"], proj.get("wall_time_seconds"))
        add_if_valid(proj_store["cpu_process_time_seconds"], proj.get("cpu_process_time_seconds"))
        add_if_valid(proj_store["estimated_cpu_utilization_percent"], proj.get("estimated_cpu_utilization_percent"))
        add_if_valid(proj_store["ram_rss_delta_mb"], proj.get("ram_rss_delta_mb"))
        add_if_valid(proj_store["python_tracemalloc_peak_mb"], proj.get("python_tracemalloc_peak_mb"))

        output = proj.get("output", {})
        add_if_valid(proj_store["output_nbytes_mb"], output.get("nbytes_mb"))

        if len(proj_store["output_shape_examples"]) == 0 and "shape" in output:
            proj_store["output_shape_examples"].append({
                "shape": output.get("shape"),
                "dtype": output.get("dtype"),
            })

        if proj.get("gpu_vram") is not None:
            proj_store["gpu_vram_entries"].append(proj.get("gpu_vram"))

        wall = proj.get("wall_time_seconds")
        cpu = proj.get("cpu_process_time_seconds")

        if wall is None or cpu is None:
            combined_valid = False
        else:
            combined_wall_time += float(wall)
            combined_cpu_time += float(cpu)

    if combined_valid:
        store["combined_four_projection_wall_time_seconds"].append(combined_wall_time)
        store["combined_four_projection_cpu_process_time_seconds"].append(combined_cpu_time)


def finalize_summary(store, total_wall_time_seconds, total_tasks):
    projection_names = ["bev", "polar", "range", "front"]

    summary = {
        "num_frames_requested": int(total_tasks),
        "num_frames_successful": int(store["num_frames_successful"]),
        "num_frames_failed": int(store["num_frames_failed"]),
        "failed_frame_examples": store["failed_frame_examples"],

        "total_script_wall_time_seconds": float(total_wall_time_seconds),
        "num_workers": int(NUM_WORKERS),

        "preprocessing": {},
        "projections": {},

        "combined_four_projection_generation": {},
        "throughput": {},
    }

    # Preprocessing summary
    for metric, values in store["preprocessing"].items():
        summary["preprocessing"][metric] = summarize_values(values)

    # Per-projection summary
    for name in projection_names:
        proj_store = store["projections"][name]

        summary["projections"][name] = {
            "wall_time_seconds": summarize_values(proj_store["wall_time_seconds"]),
            "cpu_process_time_seconds": summarize_values(proj_store["cpu_process_time_seconds"]),
            "estimated_cpu_utilization_percent": summarize_values(
                proj_store["estimated_cpu_utilization_percent"]
            ),
            "ram_rss_delta_mb": summarize_values(proj_store["ram_rss_delta_mb"]),
            "python_tracemalloc_peak_mb": summarize_values(
                proj_store["python_tracemalloc_peak_mb"]
            ),
            "output_nbytes_mb": summarize_values(proj_store["output_nbytes_mb"]),
            "output_shape_examples": proj_store["output_shape_examples"],
            "gpu_vram": summarize_gpu_vram(proj_store["gpu_vram_entries"]),
        }

        wall_stats = summary["projections"][name]["wall_time_seconds"]
        mean_time = wall_stats["mean"]

        if mean_time is not None and mean_time > 0:
            summary["projections"][name]["mean_fps_single_worker_equivalent"] = float(
                1.0 / mean_time
            )
        else:
            summary["projections"][name]["mean_fps_single_worker_equivalent"] = None

    # Combined projection summary
    combined_wall_stats = summarize_values(
        store["combined_four_projection_wall_time_seconds"]
    )
    combined_cpu_stats = summarize_values(
        store["combined_four_projection_cpu_process_time_seconds"]
    )

    summary["combined_four_projection_generation"] = {
        "wall_time_seconds": combined_wall_stats,
        "cpu_process_time_seconds": combined_cpu_stats,
    }

    if combined_wall_stats["mean"] is not None and combined_wall_stats["mean"] > 0:
        summary["combined_four_projection_generation"]["mean_fps_single_worker_equivalent"] = float(
            1.0 / combined_wall_stats["mean"]
        )
    else:
        summary["combined_four_projection_generation"]["mean_fps_single_worker_equivalent"] = None

    if combined_wall_stats["p95"] is not None and combined_wall_stats["p95"] > 0:
        summary["combined_four_projection_generation"]["p95_fps_single_worker_equivalent"] = float(
            1.0 / combined_wall_stats["p95"]
        )
    else:
        summary["combined_four_projection_generation"]["p95_fps_single_worker_equivalent"] = None

    if combined_wall_stats["max"] is not None and combined_wall_stats["max"] > 0:
        summary["combined_four_projection_generation"]["worst_case_fps_single_worker_equivalent"] = float(
            1.0 / combined_wall_stats["max"]
        )
    else:
        summary["combined_four_projection_generation"]["worst_case_fps_single_worker_equivalent"] = None

    # End-to-end script throughput
    if total_wall_time_seconds > 0:
        summary["throughput"]["successful_frames_per_second_end_to_end"] = float(
            store["num_frames_successful"] / total_wall_time_seconds
        )
        summary["throughput"]["requested_frames_per_second_end_to_end"] = float(
            total_tasks / total_wall_time_seconds
        )
    else:
        summary["throughput"]["successful_frames_per_second_end_to_end"] = None
        summary["throughput"]["requested_frames_per_second_end_to_end"] = None

    return summary


# ============================================================
# Environment metadata
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
        "pynvml_available": PYNVML_AVAILABLE,
        "gpu_snapshot_at_start": get_gpu_memory_snapshot(),

        "configuration": {
            "pointcloud_folder": POINTCLOUD_FOLDER,
            "velodyne_folder": VELODYNE_FOLDER,
            "temp_output_folder": TEMP_OUTPUT_FOLDER,
            "report_path": REPORT_PATH,

            "sequences": SEQUENCES,
            "max_frames_per_sequence": MAX_FRAMES_PER_SEQUENCE,
            "num_workers": NUM_WORKERS,

            "bev_image_size": BEV_IMAGE_SIZE,
            "polar_image_size": POLAR_IMAGE_SIZE,
            "range_image_size": RANGE_IMAGE_SIZE,
            "front_image_size": FRONT_IMAGE_SIZE,

            "x_range": X_RANGE,
            "y_range": Y_RANGE,
            "z_range": Z_RANGE,

            "lidar_model": LIDAR_MODEL,

            "curvature_k_neighbors": CURVATURE_K_NEIGHBORS,
            "curvature_num_processors": CURVATURE_NUM_PROCESSORS,
        },

        "metric_definitions": {
            "wall_time_seconds": (
                "Elapsed real time measured with time.perf_counter(). "
                "This is the most relevant latency metric."
            ),
            "cpu_process_time_seconds": (
                "User + system CPU time consumed by the worker process."
            ),
            "estimated_cpu_utilization_percent": (
                "cpu_process_time_seconds / wall_time_seconds * 100. "
                "Values near 100 indicate approximately one fully used CPU core. "
                "Values above 100 can occur if native libraries use multiple CPU threads."
            ),
            "ram_rss_delta_mb": (
                "Change in resident set size, RSS, before and after the measured block. "
                "RSS is measured at the process level."
            ),
            "python_tracemalloc_peak_mb": (
                "Peak Python-level allocated memory during the measured block. "
                "This may not capture all native NumPy or external-library allocations."
            ),
            "gpu_vram": (
                "NVIDIA GPU VRAM before/after/delta measured with pynvml when available. "
                "If the projection code is CPU-only, GPU VRAM is expected to be unavailable "
                "or approximately unchanged."
            ),
            "mean_fps_single_worker_equivalent": (
                "Computed as 1 / mean wall-clock time."
            ),
            "p95_fps_single_worker_equivalent": (
                "Computed as 1 / p95 wall-clock time. This is more conservative than mean FPS."
            ),
            "worst_case_fps_single_worker_equivalent": (
                "Computed as 1 / max wall-clock time."
            ),
        },

        "reviewer_facing_notes": [
            "The JSON intentionally stores aggregate statistics only, not per-frame measurements.",
            "For real-time feasibility claims, report mean latency together with p95, p99, and maximum latency.",
            "For clean latency measurements, run this script with NUM_WORKERS = 1.",
            "For practical preprocessing throughput, run this script with the intended deployment number of workers.",
            "This script measures LiDAR projection generation. If the reviewer requests neural-network inference time and GPU VRAM, report the network forward-pass profiling separately.",
        ],
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
    os.makedirs(TEMP_OUTPUT_FOLDER, exist_ok=True)

    all_tasks = []

    for seq in SEQUENCES:
        velodyne_pointcloud_path = os.path.join(
            POINTCLOUD_FOLDER,
            seq,
            VELODYNE_FOLDER
        )

        velodyne_files = sorted([
            f for f in os.listdir(velodyne_pointcloud_path)
            if f.endswith(".bin")
        ])[:MAX_FRAMES_PER_SEQUENCE]

        for i, velodyne_file in enumerate(velodyne_files):
            all_tasks.append((i, velodyne_file, seq))

    print("=" * 80)
    print("LiDAR projection resource analysis")
    print(f"Sequences: {SEQUENCES}")
    print(f"Frames requested per sequence: {MAX_FRAMES_PER_SEQUENCE}")
    print(f"Total frames to process: {len(all_tasks)}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Image output folder: {TEMP_OUTPUT_FOLDER}")
    print(f"JSON report path: {REPORT_PATH}")
    print("=" * 80)

    aggregate_store = initialize_aggregate_store()

    script_start_wall = time.perf_counter()

    with Pool(processes=NUM_WORKERS) as pool:
        for frame_result in pool.imap_unordered(
            process_frame,
            all_tasks,
            chunksize=1
        ):
            update_aggregate_store(aggregate_store, frame_result)

            seq = frame_result.get("sequence")
            frame_index = frame_result.get("frame_index")
            status = frame_result.get("status")

            print(f"[seq {seq}] frame {frame_index}: {status}")

    total_script_wall_time = time.perf_counter() - script_start_wall

    report = {
        "report_type": "lidar_projection_resource_analysis_aggregate_only",
        "environment": get_environment_metadata(),
        "summary": finalize_summary(
            aggregate_store,
            total_script_wall_time,
            total_tasks=len(all_tasks)
        ),
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 80)
    print("Processing complete.")
    print(f"Generated images saved under: {TEMP_OUTPUT_FOLDER}")
    print(f"Aggregate JSON report written to: {REPORT_PATH}")
    print("=" * 80)